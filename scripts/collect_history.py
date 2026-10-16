#!/usr/bin/env python3
"""Earlier frames of the evaluation clips for the delay comparison at large d (Figure 12).

The delay comparison with stride d uses the CoCs of frames t-d, t-2d, ... of the same clip
(respect.eval.dataset.view, mode "stride").  data/general.jsonl keeps 30 frames before the
scored frames, so d = 40 has no previous frame there.  This script runs the default sequential
decoding of the public implementation (BF16, greedy, the same procedure as collect_dev.py) on the
frames t-d, t-2d, ... that are not in the data file, down to the first frame the dataset loader
accepts (MIN_FRAME), and writes them as unscored rows.

A few frames that are already in the data file are decoded again first; their CoC must equal
the stored reference CoC, which checks that the added rows come from the same procedure.

Output: <out> (one JSON row per frame, the data-file format).  run_main.py --history <out> merges them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_dev import STRIDE_US, load_input  # noqa: E402

MIN_FRAME = 18          # the loader needs 1.6 s of ego history before t0


def needed(rows: list[dict], delays: list[int]) -> list[tuple[str, int]]:
    have = defaultdict(set)
    for r in rows:
        have[r["clip_id"]].add(r["frame_idx"])
    need = set()
    for r in rows:
        if r["scored"]:
            for d in delays:
                for f in range(r["frame_idx"] - d, MIN_FRAME - 1, -d):
                    if f not in have[r["clip_id"]]:
                        need.add((r["clip_id"], f))
    return sorted(need)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--set", default="general")
    ap.add_argument("--delays", default="40")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks", type=int, default=3, help="stored frames decoded again as a check")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(x) for x in (Path(a.data) / f"{a.set}.jsonl").read_text().splitlines()]
    jobs = needed(rows, [int(x) for x in a.delays.split(",")])
    stored = [r for r in rows if not r["scored"]]
    checks = [(r["clip_id"], r["frame_idx"]) for r in stored[:: max(1, len(stored) // max(1, a.checks))][: a.checks]]
    ref = {(r["clip_id"], r["frame_idx"]): r["reference_coc"] for r in rows}
    out = Path(a.out)
    done = {(json.loads(x)["clip_id"], json.loads(x)["frame_idx"]) for x in out.read_text().splitlines()} \
        if out.exists() else set()
    todo = checks + [j for j in jobs if j not in done]
    if a.limit:
        todo = todo[: a.checks + a.limit]
    print(f"{a.set}: {len(jobs)} frames needed, {len(done)} done, {len(todo)} to decode ({len(checks)} checks)", flush=True)

    import torch
    from respect.alpamayo.model import first_token, load_target, make_inputs, prefill, prompt_length
    from respect.decoding.sequential import clone_cache, sequential
    target = load_target(local_files_only=True)
    pool = ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"))
    futs, t_start, fp, bad = {}, time.time(), out.open("a"), 0
    for k, (cid, f) in enumerate(todo):
        for i in range(k, min(k + 2 * a.workers, len(todo))):
            if i not in futs:
                futs[i] = pool.submit(load_input, todo[i][0], todo[i][1] * STRIDE_US)
        inp = futs.pop(k).result()
        with torch.no_grad():
            prepared, _ = make_inputs(target, inp)
            pf = prefill(target, prepared)
            PL = prompt_length(pf)
            tk, _cache = sequential(target.model, clone_cache(pf.past_key_values), first_token(pf, PL), PL,
                                    target.eos, target.offset, target.size, 128)
        tk = [int(x) for x in tk]
        if k < len(checks):
            same = tk == ref[(cid, f)]
            bad += not same
            print(f"[check] {cid[:8]} f{f} {'same as stored' if same else 'DIFFERENT from stored'}", flush=True)
            continue
        fp.write(json.dumps(dict(clip_id=cid, frame_idx=f, t0_us=f * STRIDE_US, scored=False, max_new_tokens=128,
                                 reference_coc=tk, added_for_delay=True)) + "\n")
        fp.flush()
        print(f"[{k + 1}/{len(todo)}] {cid[:8]} f{f} tokens {len(tk)} ({(time.time() - t_start) / 60:.1f} min)", flush=True)
    pool.shutdown()
    print(f"done, check mismatches {bad}", flush=True)


if __name__ == "__main__":
    main()
