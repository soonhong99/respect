#!/usr/bin/env python3
"""Real-time execution with the output-preserving Prefill and Flow (Section VI-D).

Same schedule as scripts/run_realtime.py (inputs every 0.1 s on the wall clock, the
latest arrived frame is processed, 10 event clips x 5 start positions).

Run
  S4F  #4 run (RESPECT Decode) with the output-preserving system optimizations:
       Prefill  without autocast, with respect.system.fused.prefill_fast (fused RoPE,
                SwiGLU, RMSNorm and residual addition, one vision-attention call);
       Decode   RESPECT, identical to S4 of run_realtime.py;
       Flow     respect.system.flow_graph.FlowGraph (fixed KV buffer, fused kernels,
                one CUDA graph per diffusion step). Graphs for CoC lengths 8-64 are
                captured during warm-up; other lengths are captured when they first occur.

After all runs, outside timing, every processed frame is computed again with the
public implementation (BF16-autocast Prefill, sequential decoding with the
dedicated kernels, public ``trajectory()``), and the CoC tokens and the float32
trajectory bits are compared.

Inputs: scripts/build_realtime_inputs.py output.
Output: <out>/cycles.jsonl, <out>/verify.jsonl, <out>/contract.json
Usage: python scripts/run_realtime_fast.py --inputs REALTIME_INPUTS --out runs/realtime_fast [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.alpamayo.flow import trajectory  # noqa: E402
from respect.alpamayo.model import first_token, load_target, make_inputs, prefill, prompt_length  # noqa: E402
from respect.candidates.datastore import Datastore  # noqa: E402
from respect.candidates.fd_dflash import find_snapshot, load_fd_dflash  # noqa: E402
from respect.candidates.history import CocHistory, strip_eos  # noqa: E402
from respect.config import RespectConfig  # noqa: E402
from respect.decoding.kv_buffer import FixedKVBuffer  # noqa: E402
from respect.decoding.respect import FrameState, RespectDecoder  # noqa: E402
from respect.decoding.sequential import clone_cache, sequential  # noqa: E402
from respect.eval.dataset import load_frames, static_sequences  # noqa: E402
from respect.eval.realtime import STEP_S, restore  # noqa: E402
from respect.kernels import patch as kernels  # noqa: E402
from respect.system.flow_graph import FlowGraph  # noqa: E402
from respect.system.fused import prefill_fast  # noqa: E402

PHASES = [0, 6, 12, 18, 24]
FLOW_SEED = 0
PRECAPTURE = range(8, 65)          # CoC lengths whose Flow graphs are captured during warm-up


def sync() -> None:
    torch.cuda.synchronize()


def traj_sha(px: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(px, dtype=np.float32).tobytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--inputs", required=True, help="output directory of build_realtime_inputs.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="first clip, start position 0 only")
    ap.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = RespectConfig()
    target = load_target(args.model, local_files_only=True)
    model = target.model
    par = next(model.parameters())
    draft, mask_emb = load_fd_dflash(find_snapshot(), par.dtype, par.device)
    static = Datastore(static_sequences(Path(args.data)), gmax=cfg.datastore_key_len)
    caps = {(cid, i): f.max_new_tokens for cid, fr in load_frames(Path(args.data) / "event.jsonl").items()
            for i, f in fr.items()}
    inputs = Path(args.inputs)
    clips = sorted(p.name for p in inputs.iterdir() if (p / "meta.json").exists())
    if args.smoke:
        clips = clips[:1]
    phases = [0] if args.smoke else PHASES
    (out / "contract.json").write_text(json.dumps(dict(
        config=cfg.__dict__, run="S4F", phases=phases, clips=clips, step_s=STEP_S, flow_seed=FLOW_SEED,
        precapture_coc_lengths=[PRECAPTURE.start, PRECAPTURE.stop - 1], torch=torch.__version__,
        device=torch.cuda.get_device_name(0), argv=sys.argv, started=time.strftime("%Y-%m-%d %H:%M:%S"),
        clocks=_sh("sudo -n /usr/bin/jetson_clocks --show")[-1500:], nvpmodel=_sh("nvpmodel -q")), indent=2))

    state = dict(buf=None, dec=None, flow=None)

    def load_clip(cid):
        meta = json.loads((inputs / cid / "meta.json").read_text())
        small = {g: torch.load(inputs / cid / "frames" / f"{g:04d}.pt", weights_only=False)
                 for g in range(meta["g_lo"], meta["g_last"] + 1)}
        return meta, small

    @torch.no_grad()
    def cycle(cid, g, small, run, T0):
        r = dict(cfg="S4F", g=g)
        cap = caps.get((cid, g), cfg.max_new_tokens)
        ts = time.perf_counter()
        prepared, raw = make_inputs(target, restore(small, g))
        sync()
        t_in = time.perf_counter()
        assert not kernels.is_installed()
        with prefill_fast():
            pf = prefill(target, prepared, autocast=False)
        sync()
        t_pf = time.perf_counter()
        PL = prompt_length(pf)
        fs = FrameState(prepared, pf, PL, first_token(pf, PL), cap)
        if state["buf"] is None:
            state["buf"] = FixedKVBuffer(PL, cfg.buffer_capacities)
            state["dec"] = RespectDecoder(target, state["buf"], cfg, draft, mask_emb)
        buf, dec = state["buf"], state["dec"]
        assert buf.PL == PL, "the fixed buffer assumes one prompt length"
        buf.allocate(pf.past_key_values)
        buf.write_prefill(pf.past_key_values)
        kernels.install(model, PL)
        res = dec.decode(fs, run["history"].candidates(cfg.num_prev_frames), run["datastore"], new_coc=run["newest"])
        run["newest"] = None
        kernels.uninstall(model)
        sync()
        t_dec = time.perf_counter()
        tk = [int(x) for x in res.tokens]
        if state["flow"] is None:
            state["flow"] = FlowGraph(model, target.eos, PL, pf.past_key_values.layers, cfg.max_new_tokens)
        px = state["flow"](prepared, raw, buf.view(PL + len(tk)), tk, FLOW_SEED)
        sync()
        t_fl = time.perf_counter()
        run["history"].append(g, tk)
        run["newest"] = tuple(list(strip_eos(tk, target.eos)) + [target.eos])   # enters the drive index next cycle
        r.update(accepted=res.accepted_initial, rounds=res.follow_rounds, target_runs=res.target_runs,
                 ds_insert_ms=res.datastore_insert_ms, t_start=ts - T0, t_input=t_in - T0, t_prefill=t_pf - T0,
                 t_decode=t_dec - T0, t_pub=t_fl - T0, input_ms=(t_in - ts) * 1000, prefill_ms=(t_pf - t_in) * 1000,
                 decode_ms=(t_dec - t_pf) * 1000, flow_ms=(t_fl - t_dec) * 1000, cycle_ms=(t_fl - ts) * 1000,
                 prefill_len=PL, cap=cap, tokens=tk, n_tokens=len(tk), traj=np.round(px, 4).tolist(),
                 traj_sha=traj_sha(px))
        for k in ("input_ms", "prefill_ms", "decode_ms", "flow_ms", "cycle_ms",
                  "t_start", "t_input", "t_prefill", "t_decode", "t_pub"):
            r[k] = round(r[k], 4)
        return r

    def new_run():
        return dict(history=CocHistory(target.eos), datastore=static.with_drive([]), newest=None)

    def replay(cid, meta, small, phase):
        g0, g_end = meta["g_first_run"] + phase, meta["g_last"]
        run, rows, last = new_run(), [], None
        T0 = time.perf_counter()
        while True:
            g = min(g0 + int(math.floor((time.perf_counter() - T0) / STEP_S + 1e-9)), g_end)
            if last is not None and g <= last:
                if last >= g_end:
                    return rows
                time.sleep(max(0.0, T0 + (last + 1 - g0) * STEP_S - time.perf_counter()))
                continue
            r = cycle(cid, g, small, run, T0)
            r.update(clip_id=cid, phase=phase, g0=g0, g_end=g_end, e=meta["event_start_frame"],
                     k=(g - last) if last is not None else None, t_arrival=round((g - g0) * STEP_S, 4))
            rows.append(r)
            last = g

    # warm-up and Flow graph capture, not recorded
    meta0, small0 = load_clip(clips[0])
    tw = time.perf_counter()
    for rep in range(2):
        cycle(clips[0], meta0["g_first_run"] + rep, small0, new_run(), time.perf_counter())
    prepared0, _raw0 = make_inputs(target, restore(small0, meta0["g_first_run"]))
    with prefill_fast():
        pf0 = prefill(target, prepared0, autocast=False)
    PL0 = prompt_length(pf0)
    fl = state["flow"]
    for n in PRECAPTURE:
        if PL0 + n not in fl.graphs:
            toks = [0] * (n - 1) + [target.eos]
            fl.capture(PL0 + n, *fl.expert_inputs(prepared0, toks, PL0 + n))
    del pf0
    print("warm-up and %d Flow graphs %.1f s" % (len(fl.graphs), time.perf_counter() - tw), flush=True)

    fp = (out / "cycles.jsonl").open("a")
    t_all = time.perf_counter()
    for ci, cid in enumerate(clips):
        meta, small = (meta0, small0) if ci == 0 else load_clip(cid)
        for p in phases:
            rows = replay(cid, meta, small, p)
            for r in rows:
                fp.write(json.dumps(r) + "\n")
            fp.flush()
            print("[%s p%d S4F] %d cycles, cycle %.0f ms (Prefill %.0f, Decode %.0f, Flow %.0f), frames %d..%d (%.1f min)" % (
                cid[:8], p, len(rows), np.mean([r["cycle_ms"] for r in rows]), np.mean([r["prefill_ms"] for r in rows]),
                np.mean([r["decode_ms"] for r in rows]), np.mean([r["flow_ms"] for r in rows]), rows[0]["g"],
                rows[-1]["g"], (time.perf_counter() - t_all) / 60), flush=True)
    fp.close()
    print("Flow graphs captured during the runs: %d" % fl.captured_in_run, flush=True)

    # check against the public implementation, outside timing
    rows = [json.loads(line) for line in (out / "cycles.jsonl").read_text().splitlines()]
    todo = sorted({(r["clip_id"], r["g"]) for r in rows})
    bad_tok = bad_traj = 0
    cur, small = None, None
    with (out / "verify.jsonl").open("w") as fv:
        for cid, g in todo:
            if cid != cur:
                _meta, small = load_clip(cid)
                cur = cid
            prepared, raw = make_inputs(target, restore(small, g))
            pf = prefill(target, prepared)
            PL = prompt_length(pf)
            kernels.install(model, PL)
            ref, cache = sequential(model, clone_cache(pf.past_key_values), first_token(pf, PL), PL, target.eos,
                                    target.offset, target.size, caps.get((cid, g), cfg.max_new_tokens))
            kernels.uninstall(model)
            ref = [int(x) for x in ref]
            ref_sha = traj_sha(np.asarray(trajectory(model, prepared, raw, cache, ref, target.eos, FLOW_SEED)[0]).reshape(64, 3))
            got = [r for r in rows if r["clip_id"] == cid and r["g"] == g]
            ok_t = all(r["tokens"] == ref for r in got)
            ok_x = all(r["traj_sha"] == ref_sha for r in got)
            bad_tok += not ok_t
            bad_traj += not ok_x
            fv.write(json.dumps(dict(clip_id=cid, g=g, n_runs=len(got), tokens_ok=ok_t, traj_ok=ok_x,
                                     ref_tokens=ref, ref_traj_sha=ref_sha)) + "\n")
            del pf, cache
    print("verify: %d frames, token mismatches %d, trajectory mismatches %d" % (len(todo), bad_tok, bad_traj), flush=True)


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
