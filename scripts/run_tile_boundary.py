#!/usr/bin/env python3
"""Forward time of one trie verification against the number of trie nodes (Figure 5).

A single chain candidate of length M gives a trie of exactly M nodes. The
attention kernel processes queries in tiles of 64 rows, so the time should step
up at 65 and 129 nodes. Each size is warmed up 3 times and timed 10 times.
The paper measures one development frame per event type: the event start frame
of the first clip of each type in data/dev_event.jsonl.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.alpamayo.model import load_target, make_inputs, prefill, prompt_length  # noqa: E402
from respect.candidates.trie import build_trie  # noqa: E402
from respect.decoding.kv_buffer import FixedKVBuffer  # noqa: E402
from respect.decoding.verify import verify_trie  # noqa: E402
from respect.eval.dataset import input_path, load_frames  # noqa: E402
from respect.kernels import dispatch  # noqa: E402
from respect.kernels.patch import dedicated_kernels  # noqa: E402

NODES = [8, 16, 32, 48, 56, 60, 63, 64, 65, 66, 68, 72, 80, 96, 128, 129, 160]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capacity", type=int, default=192)   # window of the evaluation runs
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    clips = load_frames(Path(args.data) / "dev_event.jsonl")
    picks = {}
    for cid, frames in clips.items():
        f0 = next(iter(frames.values()))
        picks.setdefault(f0.meta["event_type"], (cid, f0.meta["event_start_frame"]))
    target = load_target(local_files_only=True)
    results = [measure(target, clips, cid, fidx, et, args) for et, (cid, fidx) in sorted(picks.items())]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


def measure(target, clips, cid, fidx, event_type, args) -> dict:
    fr = clips[cid][fidx]
    inp = torch.load(input_path(Path(args.inputs), cid, fr.frame_idx), weights_only=False)
    prepared, _ = make_inputs(target, inp)
    pf = prefill(target, prepared)
    PL = prompt_length(pf)
    buf = FixedKVBuffer(PL, (args.capacity,))
    buf.allocate(pf.past_key_values)
    buf.write_prefill(pf.past_key_values)
    cache = buf.window(args.capacity)
    toks = fr.reference_coc
    rows = []
    with dedicated_kernels(target.model, PL):
        print(f"prompt {PL}, capacity {args.capacity}, query tile {dispatch.BM}")
        for n in NODES:
            tree = build_trie([tuple((toks * 40)[:n])])
            assert len(tree.nodes) == n
            for _ in range(3):
                verify_trie(target.model, prepared, cache, tree, PL, 0, args.capacity)
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                verify_trie(target.model, prepared, cache, tree, PL, 0, args.capacity)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            tiles = -(-n // dispatch.BM)
            rows.append(dict(nodes=n, tiles=tiles, median_ms=statistics.median(ts), min_ms=min(ts), max_ms=max(ts)))
            print(f"{n:5d} nodes  {tiles} tile(s)  median {rows[-1]['median_ms']:7.2f} ms")
    med = {r["nodes"]: r["median_ms"] for r in rows}
    print(f"{event_type}: 64 -> 65 nodes: {med[65] - med[64]:+.2f} ms,  128 -> 129 nodes: {med[129] - med[128]:+.2f} ms")
    return dict(event_type=event_type, clip_id=cid, frame_idx=fr.frame_idx, prompt_len=PL,
                capacity=args.capacity, reps=args.reps, rows=rows)


if __name__ == "__main__":
    main()
