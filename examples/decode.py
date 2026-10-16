#!/usr/bin/env python3
"""Decode two event frames, reuse only completed outputs, and check tokens/KV against K.

This is a functional RESPECT Decode (#4) example, not a latency benchmark or the
full RESPECT (#5) Prefill/Flow pipeline. Run from a checkout with Alpamayo installed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET_REVISION = "b719eea7f0a63619ef51ec7f54178af0937ef050"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True, help="directory of trusted, locally built .pt inputs")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--clip-prefix", default="", help="select one event clip by its unique ID prefix")
    ap.add_argument("--frames", type=int, default=2, help="number of scored frames from that clip (default: 2)")
    ap.add_argument("--prepare-inputs-only", action="store_true", help="build missing inputs from the gated dataset, then exit (CPU)")
    ap.add_argument("--local-files-only", action="store_true", help="load the target model from the local cache only")
    ap.add_argument("--out", type=Path, default=Path("runs/decode_example.json"))
    args = ap.parse_args()
    if args.frames < 1:
        ap.error("--frames must be positive")

    from respect.eval.dataset import input_path, load_frames, static_sequences

    clips = load_frames(args.data / "event.jsonl")
    ids = sorted(cid for cid in clips if cid.startswith(args.clip_prefix))
    if not ids or (args.clip_prefix and len(ids) != 1):
        ap.error("--clip-prefix must match exactly one event clip (omit it to use the first)")
    cid = ids[0]
    frames = sorted((f for f in clips[cid].values() if f.scored), key=lambda f: f.frame_idx)[:args.frames]
    if len(frames) != args.frames:
        ap.error("the selected clip has fewer scored frames than --frames")
    paths = [input_path(args.inputs, cid, f.frame_idx) for f in frames]
    if args.prepare_inputs_only:
        import torch
        missing = [(f, p) for f, p in zip(frames, paths) if not p.exists()]
        if missing:
            import physical_ai_av
            from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
            avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=DATASET_REVISION)
            args.inputs.mkdir(parents=True, exist_ok=True)
            for f, p in missing:
                inp = load_physical_aiavdataset(clip_id=cid, t0_us=f.t0_us, avdi=avdi, maybe_stream=True)
                torch.save(inp, p)
        for p in paths:
            print(f"Input ready: {p}", flush=True)
        return
    for p in paths:
        if not p.is_file():
            ap.error(f"missing input: {p}; run with --prepare-inputs-only first, or supply existing inputs")

    import torch
    import triton
    import transformers
    from respect import CocHistory, Datastore, FixedKVBuffer, FrameState, RespectConfig, RespectDecoder, dedicated_kernels
    from respect.alpamayo.model import load_target, make_inputs, prefill, first_token, prompt_length
    from respect.candidates.fd_dflash import find_snapshot, load_fd_dflash
    from respect.decoding.sequential import clone_cache, sequential

    if not torch.cuda.is_available():
        ap.error("CUDA is required for Decode; --prepare-inputs-only does not require a GPU")
    cfg = RespectConfig()
    # Check the draft snapshot before allocating the target model.
    snapshot = find_snapshot()
    target = load_target(local_files_only=args.local_files_only)
    draft, mask_emb = load_fd_dflash(snapshot, torch.bfloat16, "cuda")
    history = CocHistory(target.eos)
    datastore = Datastore(static_sequences(args.data), gmax=cfg.datastore_key_len)
    last_coc = None
    records = []
    for f, p in zip(frames, paths):
        # Load only inputs you built yourself: torch.load below accepts pickled objects.
        inp = torch.load(p, map_location="cpu", weights_only=False)
        with torch.inference_mode():
            prepared, _ = make_inputs(target, inp)
            pf = prefill(target, prepared)  # fresh current-frame Prefill, default kernels
            pl = prompt_length(pf)
            fs = FrameState(prepared, pf, pl, first_token(pf, pl), cfg.max_new_tokens)
            buf = FixedKVBuffer(pl, cfg.buffer_capacities)
            buf.allocate(pf.past_key_values)
            buf.write_prefill(pf.past_key_values)
            decoder = RespectDecoder(target, buf, cfg, draft, mask_emb)
            candidates = history.candidates(cfg.num_prev_frames)
            with dedicated_kernels(target.model, pl):
                out = decoder.decode(fs, candidates, datastore, new_coc=last_coc)
                ref, cache = sequential(target.model, clone_cache(pf.past_key_values), fs.first_token,
                                        pl, target.eos, target.offset, target.size, cfg.max_new_tokens)
            length = pl + len(out.tokens)
            tokens_equal = out.tokens == ref
            kv_equal = cache.get_seq_length() == length and all(
                torch.equal(layer.keys, buf.base_k[i][:, :, :length, :]) and
                torch.equal(layer.values, buf.base_v[i][:, :, :length, :])
                for i, layer in enumerate(cache.layers))
        row = dict(clip_id=cid, frame_idx=f.frame_idx, t0_us=f.t0_us,
                   input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   previous_frames=len(history.entries), initial_candidates=len(candidates),
                   accepted_initial=out.accepted_initial, target_runs=out.target_runs,
                   tokens=out.tokens, reference_tokens=ref, tokens_equal=tokens_equal,
                   kv_equal=bool(kv_equal), eos_reached=out.tokens[-1] == target.eos)
        records.append(row)
        # Persist the check even if it fails; never reuse an unverified output.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(
            purpose="RESPECT Decode functional example; no performance measurement",
            reference="deterministic-kernel autoregressive decoding (K), same environment and Prefill",
            torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
            transformers=transformers.__version__, device=torch.cuda.get_device_name(),
            config=asdict(cfg), unique_clips=1, unique_frames=len(records),
            history_source="outputs generated in this example; initially empty",
            static_source="development CoCs only", frames=records), indent=2) + "\n")
        print(f"frame {f.frame_idx}: previous_frames={row['previous_frames']}, "
              f"tokens_equal={tokens_equal}, kv_equal={kv_equal}, eos={row['eos_reached']}", flush=True)
        if not (tokens_equal and kv_equal and row["eos_reached"]):
            raise RuntimeError(f"output check failed; see {args.out}")
        history.append(f.frame_idx, out.tokens)
        last_coc = tuple(out.tokens)  # inserted once at the next decode call
        del pf, fs, decoder, buf, cache, prepared, inp
    print(f"Checks passed. Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
