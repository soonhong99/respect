#!/usr/bin/env python3
"""Real-time execution on the board (Section VI-B): inputs arrive every 0.1 s on the
wall clock and the worker always processes the latest arrived frame.

Frame g of a clip arrives (g - g0) x 0.1 s after the run starts at frame g0. When a
cycle ends, the worker takes the most recent arrived frame and drops the others;
the run ends after the last frame of the clip has been processed. One cycle is
  input preparation (restore the image window, preprocess, tokenize)
  -> Prefill (with Vision Encoding) -> Decode -> Flow (one trajectory, seed 0),
with CUDA synchronized at every stage boundary.

Runs
  S0  Decode by the default sequential decoding (#0), default kernels
  S4  Decode by RESPECT (#4). Its Decode stage includes writing the Prefill KV into
      the fixed buffer, installing and removing the dedicated kernels and adding the
      previous completed CoC to the drive index. The initial candidates and the drive
      index use only the CoCs this run has already completed; the static index holds
      the development CoCs and is built once before running.
Every clip is run from g0 = E - 60 + p for p in 0, 6, 12, 18, 24 (E: event-start
frame), both configurations per start position, in alternating order.

After all runs, outside timing, every processed (configuration, clip, frame) is
decoded again sequentially (S4 with the dedicated kernels, S0 with the default
kernels) and the tokens are compared.

Inputs: scripts/build_realtime_inputs.py output.
Output: <out>/cycles.jsonl (one row per cycle), <out>/verify.jsonl, <out>/contract.json
Usage: python scripts/run_realtime.py --inputs REALTIME_INPUTS --out runs/realtime [--smoke]
"""
from __future__ import annotations

import argparse
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

PHASES = [0, 6, 12, 18, 24]
FLOW_SEED = 0


def sync() -> None:
    torch.cuda.synchronize()


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
    tc = time.perf_counter()
    static = Datastore(static_sequences(Path(args.data)), gmax=cfg.datastore_key_len)
    static_ms = (time.perf_counter() - tc) * 1000
    caps = {(cid, i): f.max_new_tokens for cid, fr in load_frames(Path(args.data) / "event.jsonl").items()
            for i, f in fr.items()}
    inputs = Path(args.inputs)
    clips = sorted(p.name for p in inputs.iterdir() if (p / "meta.json").exists())
    if args.smoke:
        clips = clips[:1]
    phases = [0] if args.smoke else PHASES
    (out / "contract.json").write_text(json.dumps(dict(
        config=cfg.__dict__, phases=phases, clips=clips, step_s=STEP_S, flow_seed=FLOW_SEED,
        static_datastore=dict(build_ms=round(static_ms, 1), entries=static.num_entries()),
        torch=torch.__version__, device=torch.cuda.get_device_name(0), argv=sys.argv,
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        clocks=_sh("sudo -n /usr/bin/jetson_clocks --show")[-1500:], nvpmodel=_sh("nvpmodel -q")), indent=2))

    state = dict(buf=None, dec=None)

    def load_clip(cid):
        meta = json.loads((inputs / cid / "meta.json").read_text())
        small = {g: torch.load(inputs / cid / "frames" / f"{g:04d}.pt", weights_only=False)
                 for g in range(meta["g_lo"], meta["g_last"] + 1)}
        return meta, small

    @torch.no_grad()
    def cycle(name, cid, g, small, run, T0):
        r = dict(cfg=name, g=g)
        cap = caps.get((cid, g), cfg.max_new_tokens)
        ts = time.perf_counter()
        prepared, raw = make_inputs(target, restore(small, g))
        sync()
        t_in = time.perf_counter()
        assert not kernels.is_installed()
        pf = prefill(target, prepared)
        sync()
        t_pf = time.perf_counter()
        PL = prompt_length(pf)
        fs = FrameState(prepared, pf, PL, first_token(pf, PL), cap)
        if name == "S0":
            tk, cache = sequential(model, pf.past_key_values, fs.first_token, PL, target.eos,
                                   target.offset, target.size, cap)
            tk = [int(x) for x in tk]
            sync()
            t_dec = time.perf_counter()
            r.update(target_runs=len(tk))
            flow_cache = cache
        else:
            if state["buf"] is None:
                state["buf"] = FixedKVBuffer(PL, cfg.buffer_capacities)
                state["dec"] = RespectDecoder(target, state["buf"], cfg, draft, mask_emb)
            buf, dec = state["buf"], state["dec"]
            assert buf.PL == PL, "the fixed buffer assumes one prompt length"
            buf.allocate(pf.past_key_values)
            buf.write_prefill(pf.past_key_values)
            kernels.install(model, PL)
            res = dec.decode(fs, run["history"].candidates(cfg.num_prev_frames), run["datastore"],
                             new_coc=run["newest"])
            run["newest"] = None
            kernels.uninstall(model)
            sync()
            t_dec = time.perf_counter()
            tk = res.tokens
            r.update(accepted=res.accepted_initial, n_prev=res.initial_candidates, front_nodes=res.initial_nodes,
                     rounds=res.follow_rounds, target_runs=res.target_runs, ds_insert_ms=res.datastore_insert_ms)
            flow_cache = buf.view(PL + len(tk))
        px = trajectory(model, prepared, raw, flow_cache, tk, target.eos, FLOW_SEED)[0]
        sync()
        t_fl = time.perf_counter()
        if name == "S4":
            run["history"].append(g, tk)
            run["newest"] = tuple(list(strip_eos(tk, target.eos)) + [target.eos])   # enters the drive index next cycle
        r.update(t_start=ts - T0, t_input=t_in - T0, t_prefill=t_pf - T0, t_decode=t_dec - T0, t_pub=t_fl - T0,
                 input_ms=(t_in - ts) * 1000, prefill_ms=(t_pf - t_in) * 1000, decode_ms=(t_dec - t_pf) * 1000,
                 flow_ms=(t_fl - t_dec) * 1000, cycle_ms=(t_fl - ts) * 1000, prefill_len=PL, cap=cap,
                 tokens=tk, n_tokens=len(tk), traj=np.round(np.asarray(px).reshape(64, 3), 4).tolist())
        for k in ("input_ms", "prefill_ms", "decode_ms", "flow_ms", "cycle_ms",
                  "t_start", "t_input", "t_prefill", "t_decode", "t_pub"):
            r[k] = round(r[k], 4)
        return r

    def new_run():
        return dict(history=CocHistory(target.eos), datastore=static.with_drive([]), newest=None)

    def replay(name, cid, meta, small, phase):
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
            r = cycle(name, cid, g, small, run, T0)
            r.update(clip_id=cid, phase=phase, g0=g0, g_end=g_end, e=meta["event_start_frame"],
                     k=(g - last) if last is not None else None, t_arrival=round((g - g0) * STEP_S, 4))
            rows.append(r)
            last = g

    meta0, small0 = load_clip(clips[0])
    tw = time.perf_counter()
    for name in ("S0", "S4", "S0", "S4"):              # warm-up, not recorded
        cycle(name, clips[0], meta0["g_first_run"], small0, new_run(), time.perf_counter())
    print("warm-up %.1f s, static index %.0f ms" % (time.perf_counter() - tw, static_ms), flush=True)

    fp = (out / "cycles.jsonl").open("a")
    t_all = time.perf_counter()
    for ci, cid in enumerate(clips):
        meta, small = (meta0, small0) if ci == 0 else load_clip(cid)
        for pi, p in enumerate(phases):
            for name in (("S0", "S4") if (ci + pi) % 2 == 0 else ("S4", "S0")):
                rows = replay(name, cid, meta, small, p)
                for r in rows:
                    fp.write(json.dumps(r) + "\n")
                fp.flush()
                print("[%s p%d %s] %d cycles, mean cycle %.0f ms (Decode %.0f), frames %d..%d  (%.1f min)" % (
                    cid[:8], p, name, len(rows), np.mean([r["cycle_ms"] for r in rows]),
                    np.mean([r["decode_ms"] for r in rows]), rows[0]["g"], rows[-1]["g"],
                    (time.perf_counter() - t_all) / 60), flush=True)
    fp.close()

    rows = [json.loads(line) for line in (out / "cycles.jsonl").read_text().splitlines()]
    todo = sorted({(r["clip_id"], r["g"], r["cfg"]) for r in rows})
    bad, cur, small = 0, None, None
    with (out / "verify.jsonl").open("w") as fv:
        for cid, g, name in todo:
            if cid != cur:
                _meta, small = load_clip(cid)
                cur = cid
            prepared, _raw = make_inputs(target, restore(small, g))
            pf = prefill(target, prepared)
            PL = prompt_length(pf)
            cap = caps.get((cid, g), cfg.max_new_tokens)
            if name == "S4":
                kernels.install(model, PL)
            tk, _c = sequential(model, clone_cache(pf.past_key_values), first_token(pf, PL), PL, target.eos,
                                target.offset, target.size, cap)
            if name == "S4":
                kernels.uninstall(model)
            tk = [int(x) for x in tk]
            got = [r["tokens"] for r in rows if (r["clip_id"], r["g"], r["cfg"]) == (cid, g, name)]
            ok = all(x == tk for x in got)
            bad += not ok
            fv.write(json.dumps(dict(cfg=name, clip_id=cid, g=g, n_runs=len(got), ok=ok, ref_tokens=tk)) + "\n")
            del prepared, pf, _c
    print("verify: %d (configuration, clip, frame), %d mismatches" % (len(todo), bad), flush=True)


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
