#!/usr/bin/env python3
"""Inputs of the real-time runs (Section VI-B): every frame of the evaluated event clips
from 63 frames before the event-start frame E to the end of the recording.

Each frame is loaded with the Alpamayo 1.5 loader and stored with only the newest
image of each camera (``respect.eval.realtime.shrink``); ``restore`` stacks frames
g-3..g back into the input of frame g. For every frame the restored input is
checked against the loader's input, and with ``--check-inputs`` the scored frames
are also checked against the input files of the main evaluation. A clip is written
only if every check passes.

The dataset revision is pinned to the snapshot used for the paper's inputs.

Output: <out>/<clip_id>/frames/<g:04d>.pt and <out>/<clip_id>/meta.json
Usage: python scripts/build_realtime_inputs.py --out REALTIME_INPUTS [--check-inputs INPUTS] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.eval.dataset import input_path, load_frames  # noqa: E402
from respect.eval.realtime import NUM_FRAMES, restore, same_value, shrink  # noqa: E402

STEP_US = 100_000
PRE = 60                                              # runs start 60 frames before E
REVISION = "b719eea7f0a63619ef51ec7f54178af0937ef050"  # dataset snapshot of the paper's inputs
_AVDI = None


def load_input(cid: str, g: int):
    global _AVDI
    import physical_ai_av
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    if _AVDI is None:
        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface(revision=REVISION)
    try:
        return g, load_physical_aiavdataset(clip_id=cid, t0_us=g * STEP_US, avdi=_AVDI, maybe_stream=True), None
    except Exception as e:                            # beyond the end of the recording
        return g, None, repr(e)[:300]


def main() -> None:
    import physical_ai_av
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-inputs", help="input files of the main evaluation, compared on scored frames")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--clips", help="comma-separated clip-id prefixes (default: every evaluated event clip)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clips = load_frames(Path(args.data) / "event.jsonl")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=REVISION)
    C = avdi.features.CAMERA
    cams = [C.CAMERA_CROSS_LEFT_120FOV, C.CAMERA_FRONT_WIDE_120FOV, C.CAMERA_CROSS_RIGHT_120FOV, C.CAMERA_FRONT_TELE_30FOV]
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))
    order = sorted(clips)
    if args.clips:
        order = [c for c in order if any(c.startswith(p) for p in args.clips.split(","))]
    for cid in order:
        cdir = out / cid
        if (cdir / "meta.json").exists():
            print(f"{cid[:8]} exists", flush=True)
            continue
        frames = clips[cid]
        e = int(next(iter(frames.values())).meta["event_start_frame"])
        scored = [i for i, f in frames.items() if f.scored]
        ends = []
        for f in cams:
            cam = avdi.get_clip_feature(cid, f, maybe_stream=True)
            ends.append(int(cam.timestamps[-1]))
            cam.close()
        g_lo, g_hi = e - PRE - (NUM_FRAMES - 1), min(ends) // STEP_US
        t0 = time.perf_counter()
        res, errs = {}, {}
        for g, inp, err in pool.map(load_input, [cid] * (g_hi - g_lo + 1), range(g_lo, g_hi + 1)):
            if err is None:
                res[g] = inp
            else:
                errs[g] = err
        last = g_lo
        while last + 1 in res:
            last += 1
        assert all(g > last for g in errs), f"{cid}: a frame inside the recording failed to load"
        small = {g: shrink(res[g]) for g in range(g_lo, last + 1)}
        bad_restore = [g for g in range(g_lo + NUM_FRAMES - 1, last + 1) if not same_value(restore(small, g), res[g])]
        bad_input = []
        if args.check_inputs:
            for g in scored:
                ref = torch.load(input_path(Path(args.check_inputs), cid, g), weights_only=False)
                if not same_value(ref, res[g]):
                    bad_input.append(g)
        ok = not bad_restore and not bad_input
        meta = dict(clip_id=cid, event_start_frame=e, g_lo=g_lo, g_first_run=e - PRE, g_last=last,
                    n_frames=len(small), bad_restore=bad_restore, bad_input=bad_input,
                    checked_inputs=bool(args.check_inputs), ok=ok, revision=REVISION)
        if ok:
            (cdir / "frames").mkdir(parents=True, exist_ok=True)
            for g, s in small.items():
                torch.save(s, cdir / "frames" / f"{g:04d}.pt")
            (cdir / "meta.json").write_text(json.dumps(meta, indent=2))
        print("%s frames %d..%d (%d), restore mismatches %d, input mismatches %d, %s, %.0f s" % (
            cid[:8], g_lo, last, len(small), len(bad_restore), len(bad_input), "OK" if ok else "FAIL",
            time.perf_counter() - t0), flush=True)
        del res, small
    pool.shutdown()


if __name__ == "__main__":
    main()
