#!/usr/bin/env python3
"""Trajectories of the public implementation (#0) on the evaluation frames, for trajectory figures.

For every scored frame: Prefill and default sequential decoding with the default PyTorch
kernels (#0, the public implementation's Decode), then Flow with seeds 0-5 on that cache,
exactly as the reference minADE6 of data/*.jsonl was obtained. Recorded per frame: the CoC
tokens and whether they equal reference_coc, the six trajectories, minADE6 / minFDE6 against
ego_future_xyz, the ground-truth future and the ego history. With --images, the latest
front-camera image (camera index 0) is saved as <out>/images/<clip>_<frame>.jpg (640x360).

The trajectories of RESPECT (#4, the s4_traj of run_main.py --flow) and of FlashDrive BF16
(baselines/flashdrive/quality_fd.py --fixed-seed 0) use the same seeds 0-5 on every frame,
so the three sets can be drawn on the same frame.

Output: <out>/traj_<set>.jsonl
Usage:
  python scripts/dump_trajectories.py --inputs INPUTS --out runs/traj [--sets event] [--images]
      [--limit-frames 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.alpamayo.flow import ground_truth, min_ade_fde, trajectory  # noqa: E402
from respect.alpamayo.model import first_token, load_target, make_inputs, prefill, prompt_length  # noqa: E402
from respect.decoding.sequential import clone_cache, sequential  # noqa: E402
from respect.eval.dataset import input_path, iterate, load_frames  # noqa: E402

SEEDS = [0, 1, 2, 3, 4, 5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="event,general")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--limit-frames", type=int, default=0, help="frames per set (0: all)")
    ap.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.images:
        (out / "images").mkdir(exist_ok=True)
    target = load_target(args.model, local_files_only=True)
    model = target.model
    t_start = time.perf_counter()
    for set_name in args.sets.split(","):
        clips = load_frames(Path(args.data) / f"{set_name}.jsonl")
        fpath = out / f"traj_{set_name}.jsonl"
        done = set()
        if fpath.exists():
            done = {(x["clip_id"], x["frame_idx"]) for x in map(json.loads, fpath.read_text().splitlines())}
        fp = fpath.open("a")
        n = 0
        tot = sum(1 for c in clips.values() for f in c.values() if f.scored)
        for cid, _hist, scored in iterate(clips, set_name):
            for fr in scored:
                n += 1
                if args.limit_frames and n > args.limit_frames:
                    break
                if (cid, fr.frame_idx) in done:
                    continue
                inp = torch.load(input_path(Path(args.inputs), cid, fr.frame_idx), weights_only=False)
                with torch.no_grad():
                    prepared, raw = make_inputs(target, inp)
                    pf = prefill(target, prepared)
                    PL = prompt_length(pf)
                    tk, cache = sequential(model, clone_cache(pf.past_key_values), first_token(pf, PL), PL,
                                           target.eos, target.offset, target.size, fr.max_new_tokens)
                    tk = [int(x) for x in tk]
                    traj = np.stack([np.asarray(trajectory(model, prepared, raw, cache, tk, target.eos, sd)[0])
                                     .reshape(64, 3) for sd in SEEDS])
                gt = ground_truth(inp)
                m = min_ade_fde(traj, gt)
                ref = fr.meta.get("reference_minADE6_m")
                row = dict(set=set_name, clip_id=cid, frame_idx=fr.frame_idx, tokens_S0=tk,
                           match_reference_coc=(tk == fr.reference_coc),
                           s0_minADE6_m=m["minADE_m"], s0_minFDE6_m=m["minFDE_m"], reference_minADE6_m=ref,
                           s0_traj=np.round(traj, 4).tolist(), gt_xyz=np.round(gt, 4).tolist(),
                           ego_history_xyz=np.round(np.asarray(inp["ego_history_xyz"]).reshape(-1, 3), 4).tolist(),
                           event=dict((k, fr.meta[k]) for k in ("event_start_frame", "frames_from_event",
                                                                 "event_type") if k in fr.meta))
                fp.write(json.dumps(row) + "\n")
                fp.flush()
                if args.images:
                    from PIL import Image
                    cams = [int(c) for c in inp["camera_indices"]]
                    img = inp["image_frames"][cams.index(0)][-1].permute(1, 2, 0).cpu().numpy()
                    Image.fromarray(img).resize((640, 360)).save(out / "images" / f"{cid[:8]}_{fr.frame_idx:04d}.jpg",
                                                                 quality=88)
                print("[%s %d/%d] %s f%d  CoC=ref %s  minADE6 %.4f (ref %s)  (%.1f min)" % (
                    set_name, n, tot, cid[:8], fr.frame_idx, row["match_reference_coc"], m["minADE_m"],
                    "-" if ref is None else "%.4f" % ref, (time.perf_counter() - t_start) / 60), flush=True)
                del inp, prepared, raw, pf, cache
        fp.close()
    print("done %.1f min" % ((time.perf_counter() - t_start) / 60), flush=True)


if __name__ == "__main__":
    main()
