#!/usr/bin/env python3
"""Input of the frame before the first scored frame of every evaluated clip (for offline_unified.py).

FlashDrive's streaming Prefill holds after the first window has been prefilled in full, and the
evaluation inputs hold only the scored frames.  This builds the window of frame (first scored - 1)
with the dataset loader used for the evaluation inputs (t0_us = frame x 100,000) and checks that
its last three image times equal the first three of the first scored frame's input.  CPU only;
needs the dataset (network access or a local Hugging Face cache) and the Alpamayo 1.5 package.

Usage: python baselines/flashdrive/fill_windows.py --inputs INPUTS --out FILL_WINDOWS
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
import physical_ai_av  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data"))
ap.add_argument("--inputs", required=True, help="evaluation inputs, to check the overlapping images")
ap.add_argument("--out", required=True)
args = ap.parse_args()
DATA, INPUTS = Path(args.data), Path(args.inputs)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
rows = []
for s in ("event", "general"):
    first = {}
    for line in (DATA / f"{s}.jsonl").read_text().splitlines():
        x = json.loads(line)
        if x["scored"]:
            first[x["clip_id"]] = min(first.get(x["clip_id"], 10 ** 9), x["frame_idx"])
    for cid in sorted(first):
        p = out / f"{cid}_{first[cid] - 1:04d}.pt"
        t = time.time()
        if not p.exists():
            d = load_physical_aiavdataset(cid, t0_us=(first[cid] - 1) * 100_000, num_frames=4, avdi=avdi)
            torch.save(d, p)
        d = torch.load(p, weights_only=False)
        r = torch.load(INPUTS / f"{cid}_{first[cid]:04d}.pt", weights_only=False)
        ok = bool(torch.equal(d["image_frames"][:, 1:], r["image_frames"][:, :3]))
        rows.append(dict(set=s, clip_id=cid, fill_frame=first[cid] - 1, overlap_equal=ok, s=round(time.time() - t, 1)))
        print(json.dumps(rows[-1]), flush=True)
(out / "check.json").write_text(json.dumps(rows, indent=1))
print("%d windows, overlap mismatches %d" % (len(rows), sum(not r["overlap_equal"] for r in rows)), flush=True)
