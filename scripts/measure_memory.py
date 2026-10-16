#!/usr/bin/env python3
"""Memory of one configuration in its own process (Table V of the paper).

A process loads only the modules its configuration uses and processes every scored
frame of the evaluation sets once, as one cycle of the real-time run:
  Prefill (with Vision Encoding) -> Decode -> Flow (one trajectory, seed 0).

Configurations
  S0  #0  default sequential decoding, default kernels. Prefill does not keep hidden states.
  K       sequential decoding with the dedicated kernels. Prefill does not keep hidden states.
  S4  #4  RESPECT: FD-DFlash, static (development CoCs) and drive datastore index, fixed KV
          buffer, dedicated kernels. Prefill keeps only what seeds FD-DFlash (the outputs of its
          target layers at the last context_len positions); the Prefill KV is released after it is
          written into the fixed buffer.
  S5  #5  RESPECT as a system: S4 with the fused Prefill (respect.system.fused, no autocast) and the Flow
          CUDA graphs (respect.system.flow_graph; graphs for CoC lengths 8-64 are captured in the warm-up,
          other lengths when they first occur), as scripts/run_realtime_fast.py.
Previous CoCs and the drive index come from the default sequential outputs of the clip (d = 1),
as in run_main.py. The FlashDrive baseline (#0') is measured by baselines/flashdrive/memory_fd.py.

The first frame is processed once as a warm-up. Then the CUDA peak statistics are reset at the
start of every stage, so the process peak after the warm-up is the maximum of the stage peaks.
Recorded per frame: allocated memory at the frame start and the peak allocated / reserved memory
of each stage. The system memory in use (MemTotal - MemAvailable) is sampled every 50 ms.

Output: <out>/<config>.jsonl (one row per frame) and <out>/<config>_summary.json
Usage: python scripts/measure_memory.py --config S4 --inputs INPUTS --out runs/memory [--limit-frames 3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.alpamayo.flow import trajectory  # noqa: E402
from respect.alpamayo.model import first_token, load_target, make_inputs, prefill, prompt_length  # noqa: E402
from respect.config import RespectConfig  # noqa: E402
from respect.decoding.sequential import sequential  # noqa: E402
from respect.eval.dataset import input_path, iterate, load_frames, static_sequences, view  # noqa: E402
from respect.kernels import patch as kernels  # noqa: E402

FLOW_SEED = 0
MIB = 2 ** 20


class SystemMemory:
    """Peak of MemTotal - MemAvailable, sampled every 50 ms."""

    def __init__(self) -> None:
        self.peak, self.on = 0.0, True
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    @staticmethod
    def used_mib() -> float:
        m = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            m[k] = int(v.split()[0])
        return (m["MemTotal"] - m["MemAvailable"]) / 1024

    def _run(self) -> None:
        while self.on:
            self.peak = max(self.peak, self.used_mib())
            time.sleep(0.05)

    def reset(self) -> None:
        self.peak = self.used_mib()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["S0", "K", "S4", "S5"])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--inputs", required=True, help="directory of <clip_id>_<frame:04d>.pt input files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="event,general")
    ap.add_argument("--limit-frames", type=int, default=0, help="stop each set after this many frames")
    ap.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    args = ap.parse_args()
    name = args.config
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sysmem = SystemMemory()
    sys_before = sysmem.used_mib()

    cfg = RespectConfig()
    target = load_target(args.model, local_files_only=True)
    model = target.model
    loaded = {"target": torch.cuda.memory_allocated() / MIB}
    static = dec = buf = flow = None
    if name in ("S4", "S5"):
        from respect.candidates.datastore import Datastore
        from respect.candidates.fd_dflash import find_snapshot, load_fd_dflash
        from respect.decoding.kv_buffer import FixedKVBuffer
        from respect.decoding.respect import FrameState, RespectDecoder
        par = next(model.parameters())
        draft, mask_emb = load_fd_dflash(find_snapshot(), par.dtype, par.device)
        loaded["fd_dflash"] = torch.cuda.memory_allocated() / MIB - loaded["target"]
        static = Datastore(static_sequences(Path(args.data)), gmax=cfg.datastore_key_len)
    seed = (list(draft.config.target_layer_ids), int(draft.context_len)) if name in ("S4", "S5") else None

    frames = []
    for set_name in args.sets.split(","):
        clips = load_frames(Path(args.data) / f"{set_name}.jsonl")
        k = 0
        for cid, _hist, scored in iterate(clips, set_name):
            for fr in scored:
                k += 1
                if args.limit_frames and k > args.limit_frames:
                    break
                frames.append((set_name, clips, cid, fr))

    def stage(rec, key, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        rec[key + "_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        rec[key + "_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / MIB, 1)
        rec[key + "_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / MIB, 1)
        return r

    def do_prefill(prepared):
        if name == "S5":
            from respect.system.fused import prefill_fast
            with prefill_fast():
                return prefill(target, prepared, autocast=False, hidden_states=False, seed_layers=seed)
        if seed:
            return prefill(target, prepared, hidden_states=False, seed_layers=seed)
        return prefill(target, prepared, hidden_states=False)

    @torch.no_grad()
    def cycle(set_name, clips, cid, fr) -> dict:
        nonlocal buf, dec, flow
        torch.cuda.synchronize()
        rec = dict(set=set_name, clip_id=cid, frame_idx=fr.frame_idx,
                   start_alloc_mib=round(torch.cuda.memory_allocated() / MIB, 1))
        inp = torch.load(input_path(Path(args.inputs), cid, fr.frame_idx), weights_only=False)
        prepared, raw = make_inputs(target, inp)
        pf = stage(rec, "prefill", lambda: do_prefill(prepared))
        PL = prompt_length(pf)
        ft = first_token(pf, PL)
        if name in ("S0", "K"):
            if name == "K":
                kernels.install(model, PL)
            tk, cache = stage(rec, "decode", lambda: sequential(model, pf.past_key_values, ft, PL, target.eos,
                                                                target.offset, target.size, fr.max_new_tokens))
            if name == "K":
                kernels.uninstall(model)
            del pf
            flow_cache = cache
        else:
            v = view(clips, cid, fr.frame_idx, cfg.num_prev_frames, target.eos)
            ds = static.with_drive(v.drive_base)

            def run4():
                nonlocal buf, dec, flow
                if buf is None:
                    buf = FixedKVBuffer(PL, cfg.buffer_capacities)
                    dec = RespectDecoder(target, buf, cfg, draft, mask_emb)
                assert buf.PL == PL, "the fixed buffer assumes one prompt length"
                buf.allocate(pf.past_key_values)
                buf.write_prefill(pf.past_key_values)
                if name == "S5" and flow is None:
                    from respect.system.flow_graph import FlowGraph
                    flow = FlowGraph(model, target.eos, PL, pf.past_key_values.layers, cfg.max_new_tokens)
                pf.past_key_values = None            # only the buffer holds the Prefill KV from here on
                fs = FrameState(prepared, pf, PL, ft, fr.max_new_tokens)
                kernels.install(model, PL)
                try:
                    return dec.decode(fs, v.candidates, ds, new_coc=v.newest).tokens
                finally:
                    kernels.uninstall(model)
            tk = stage(rec, "decode", run4)
            del pf
            flow_cache = buf.view(PL + len(tk))
        tk = [int(x) for x in tk]
        if name == "S5":
            stage(rec, "flow", lambda: flow(prepared, raw, flow_cache, tk, FLOW_SEED))
        else:
            stage(rec, "flow", lambda: trajectory(model, prepared, raw, flow_cache, tk, target.eos, FLOW_SEED))
        del flow_cache, prepared, raw, inp
        rec.update(prefill_len=PL, n_tokens=len(tk), match_reference=(tk == fr.reference_coc), tokens=tk)
        rec["peak_alloc_mib"] = max(rec[s + "_peak_alloc_mib"] for s in ("prefill", "decode", "flow"))
        rec["peak_reserved_mib"] = max(rec[s + "_peak_reserved_mib"] for s in ("prefill", "decode", "flow"))
        return rec

    t_all = time.perf_counter()
    cycle(*frames[0])                                       # warm-up, not recorded
    if name == "S5":                                        # second warm-up cycle and Flow graph capture
        cycle(*frames[0])
        _s, _c, cid0, fr0 = frames[0]
        prepared0, _r0 = make_inputs(target, torch.load(input_path(Path(args.inputs), cid0, fr0.frame_idx),
                                                         weights_only=False))
        with torch.no_grad():
            PL0 = prompt_length(do_prefill(prepared0))
            for n in range(8, 65):
                if PL0 + n not in flow.graphs:
                    toks = [0] * (n - 1) + [target.eos]
                    flow.capture(PL0 + n, *flow.expert_inputs(prepared0, toks, PL0 + n))
        del prepared0, _r0
    resident = torch.cuda.memory_allocated() / MIB
    sysmem.reset()
    fp = (out / f"{name}.jsonl").open("w")
    rows = []
    for i, f in enumerate(frames):
        r = cycle(*f)
        fp.write(json.dumps(r) + "\n")
        rows.append(r)
        if i % 50 == 0 or i == len(frames) - 1:
            fp.flush()
            print("[%s %d/%d] peak alloc %.1f MiB (prefill %.1f, decode %.1f, flow %.1f)  (%.1f min)" % (
                name, i + 1, len(frames), r["peak_alloc_mib"], r["prefill_peak_alloc_mib"],
                r["decode_peak_alloc_mib"], r["flow_peak_alloc_mib"], (time.perf_counter() - t_all) / 60), flush=True)
    fp.close()
    sysmem.on = False
    summary = dict(
        config=name, frames=len(rows), torch=torch.__version__, device=torch.cuda.get_device_name(0),
        argv=sys.argv, started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - (time.perf_counter() - t_all))),
        loaded_alloc_mib={k: round(v, 1) for k, v in loaded.items()},
        resident_alloc_after_warmup_mib=round(resident, 1),
        peak_alloc_mib=max(r["peak_alloc_mib"] for r in rows),
        peak_reserved_mib=max(r["peak_reserved_mib"] for r in rows),
        stage_peak_alloc_mib={s: max(r[s + "_peak_alloc_mib"] for r in rows) for s in ("prefill", "decode", "flow")},
        system_used_before_load_mib=round(sys_before, 1), system_used_peak_mib=round(sysmem.peak, 1),
        reference_match=sum(r["match_reference"] for r in rows),
        flow_graphs=(len(flow.graphs) if flow is not None else None),
        clocks=_sh("sudo -n /usr/bin/jetson_clocks --show")[-1500:], nvpmodel=_sh("nvpmodel -q"),
        minutes=round((time.perf_counter() - t_all) / 60, 1))
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("config", "frames", "peak_alloc_mib", "stage_peak_alloc_mib",
                                               "system_used_peak_mib", "reference_match", "minutes")}), flush=True)


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
