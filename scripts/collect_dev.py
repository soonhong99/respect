#!/usr/bin/env python3
"""Build the development set: clip selection, reference CoCs and input files.

The development set supplies the hyperparameter selection (select_hparams.py) and
the static datastore index; it never overlaps the evaluated clips. Selection does
not look at model outputs.

  event    clips meeting the event-set conditions (validation split, OOD annotation,
           frames E-120 .. E+19 inside the clip) that are not evaluated, at most 5
           per event type drawn with seed 260929; 140 frames per clip, the 40 frames
           E-20 .. E+19 scored.
  general  the 30 clips that follow the 60 evaluated clips in the evaluation set's
           shuffled order (test split, chunks 1271 and 2680, seed 260908); 30
           history and 10 scored frames each, with the start frame drawn by the same
           random sequence.

Phases (run in order):
  select   write <out>/dev_selection.json (no GPU)
  collect  default sequential decoding (public implementation, BF16, greedy) of every
           frame -> <out>/dev_{event,general}.jsonl, and the inputs of scored frames
           -> <inputs>/<clip>_<frame:04d>.pt
Camera data of the event clips should be downloaded before ``collect``; frames are
loaded by CPU worker processes while the GPU decodes.
The repository's data/dev_*.jsonl are the output of this script in the paper's run.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STRIDE_US = 100_000
EV = dict(n_cand=100, n_before=20, n_after=20, lo=18, hi=194, cap=5, seed=260929)
GE = dict(seed=260908, chunks=[1271, 2680], n_scored=10, n_hist=30, lo=48, hi=185, n_test=60, n_dev=30)
_AVDI = None


def load_input(cid: str, t0_us: int):
    global _AVDI
    import physical_ai_av
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    if _AVDI is None:
        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface()
    return load_physical_aiavdataset(clip_id=cid, t0_us=t0_us, avdi=_AVDI, maybe_stream=True)


def select(data: Path, snapshot: Path) -> dict:
    import pandas as pd
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    ood = pd.read_parquet(snapshot / "reasoning" / "ood_reasoning.parquet")
    idx = pd.read_parquet(snapshot / "clip_index.parquet")
    test_ev = {json.loads(l)["clip_id"] for l in (data / "event.jsonl").read_text().splitlines()}
    test_ge = list(dict.fromkeys(json.loads(l)["clip_id"] for l in (data / "general.jsonl").read_text().splitlines()))
    loE, hiE = EV["lo"] + EV["n_cand"] + EV["n_before"], EV["hi"] - EV["n_after"] + 1
    pool = defaultdict(list)
    for cid, r in ood[ood["split"] == "val"].iterrows():
        ev = r["events"]
        if ev is None or (not isinstance(ev, (str, list)) and pd.isna(ev)):
            continue
        ev = json.loads(ev) if isinstance(ev, str) else list(ev)
        es = sorted(int((json.loads(e) if isinstance(e, str) else e)["event_start_frame"]) for e in ev)
        es = [E for E in es if loE <= E <= hiE]
        if es and str(cid) not in test_ev:
            E = es[0]
            pool[str(r["event_cluster"])].append(dict(
                clip_id=str(cid), event_start_frame=E, event_cluster=str(r["event_cluster"]),
                first_frame=E - EV["n_cand"] - EV["n_before"], last_frame=E + EV["n_after"] - 1,
                score_first=E - EV["n_before"], score_last=E + EV["n_after"] - 1))
    rng = random.Random(EV["seed"])
    event = []
    for cl in sorted(pool):
        items = sorted(pool[cl], key=lambda c: c["clip_id"])
        event += items if len(items) <= EV["cap"] else rng.sample(items, EV["cap"])
    event.sort(key=lambda c: (c["event_cluster"], c["clip_id"]))
    gp = idx[(idx["split"] == "test") & (idx["chunk"].isin(GE["chunks"])) & (idx["clip_is_valid"] == True)]  # noqa: E712
    order = sorted(gp.index.astype(str))
    rng = random.Random(GE["seed"])
    rng.shuffle(order)
    starts = {c: rng.randint(GE["lo"], GE["hi"]) for c in order}
    assert set(order[:GE["n_test"]]) == set(test_ge), "the evaluation set's order is not reproduced"
    general, skipped = [], []
    for cid in order[GE["n_test"]:]:
        if len(general) >= GE["n_dev"]:
            break
        s = starts[cid]
        try:
            for f in (s - GE["n_hist"], s + GE["n_scored"] - 1):
                load_physical_aiavdataset(clip_id=cid, t0_us=f * STRIDE_US, maybe_stream=True)
        except Exception:
            skipped.append(cid)
            continue
        general.append(dict(clip_id=cid, start=s, scored=list(range(s, s + GE["n_scored"])),
                            history=list(range(s - GE["n_hist"], s))))
    return dict(event=dict(clips=event), general=dict(clips=general, skipped=skipped),
                selection_uses_model_output=False)


def collect(sel: dict, out: Path, inputs: Path, workers: int) -> None:
    import torch
    from respect.alpamayo.model import load_target, make_inputs, prefill, first_token, prompt_length
    from respect.decoding.sequential import clone_cache, sequential
    target = load_target(local_files_only=True)
    jobs = [("event", c, f, c["score_first"] <= f <= c["score_last"])
            for c in sel["event"]["clips"] for f in range(c["first_frame"], c["last_frame"] + 1)]
    jobs += [("general", c, f, f in set(c["scored"])) for c in sel["general"]["clips"] for f in c["history"] + c["scored"]]
    inputs.mkdir(parents=True, exist_ok=True)
    files = {s: (out / f"dev_{s}.jsonl").open("a") for s in ("event", "general")}
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    futs, t_start = {}, time.time()
    for k, (s, c, f, scored) in enumerate(jobs):
        for i in range(k, min(k + 2 * workers, len(jobs))):
            if i not in futs:
                futs[i] = pool.submit(load_input, jobs[i][1]["clip_id"], jobs[i][2] * STRIDE_US)
        inp = futs.pop(k).result()
        with torch.no_grad():
            prepared, _ = make_inputs(target, inp)
            pf = prefill(target, prepared)
            PL = prompt_length(pf)
            tk, _cache = sequential(target.model, clone_cache(pf.past_key_values), first_token(pf, PL), PL,
                                    target.eos, target.offset, target.size, 128)
        row = dict(clip_id=c["clip_id"], frame_idx=f, t0_us=f * STRIDE_US, scored=scored, max_new_tokens=128,
                   reference_coc=[int(x) for x in tk])
        if s == "event":
            row.update(event_start_frame=c["event_start_frame"], frames_from_event=f - c["event_start_frame"],
                       event_type=c["event_cluster"])
        else:
            row.update(start_frame=c["start"])
        if scored:
            torch.save(inp, inputs / f"{c['clip_id']}_{f:04d}.pt")
        files[s].write(json.dumps(row) + "\n")
        files[s].flush()
        print(f"[{k + 1}/{len(jobs)}] {s} {c['clip_id'][:8]} f{f} tokens {len(tk)} ({(time.time() - t_start) / 60:.1f} min)",
              flush=True)
    pool.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["select", "collect"], required=True)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--snapshot", help="local snapshot of nvidia/PhysicalAI-Autonomous-Vehicles (select phase)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", help="directory for the scored frames' input files (collect phase)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.phase == "select":
        sel = select(Path(a.data), Path(a.snapshot))
        (out / "dev_selection.json").write_text(json.dumps(sel, indent=1))
        print(f"event {len(sel['event']['clips'])} clips, general {len(sel['general']['clips'])} clips")
    else:
        collect(json.loads((out / "dev_selection.json").read_text()), out, Path(a.inputs), a.workers)


if __name__ == "__main__":
    main()
