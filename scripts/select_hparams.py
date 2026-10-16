#!/usr/bin/env python3
"""Hyperparameter selection on the development set (Section IV and V-A of the paper).

The development frames (data/dev_{event,general}.jsonl) do not overlap the
evaluated frames. On every scored development frame, #4 runs with the dedicated
kernels in both orders, and every variant is checked against the dedicated-kernel
sequential run (tokens and the committed KV, bitwise).

  A  (phase sweep) N in 1..100 (17 values; general frames have 30 history frames, so
     N <= 30), the follow-up values at the defaults below, on every scored frame.
     The rule picks N.
  B  (phase followup) at the chosen N, on the frames that need follow-up verification
     at that N (the follow-up values do not enter the execution path of frames
     completed by initial verification): B1 FD-DFlash tree nodes x top-k x node
     budget, B2 datastore range x key length, with the defaults rerun in the same pass.
  (phase select) the rule picks the follow-up values; frames completed by
     initial verification keep their phase-A time at the chosen N.

Rule: score each value by the mean of the two sets' per-frame mean Decode time; values within
1% of the smallest score are not told apart by the mean (repeated measurements of one setting
differ by about 1%), and among them the smallest value is chosen (smaller N, key, tree, top-k and
budget; datastore range "same" before "all"), which keeps the verified candidates and trie small.
The p99 of every value is recorded alongside (Table II).
N is chosen among values present in both sets (<= 30).
Datastore during selection: the CoCs of the clip before the frame, and for range
"all" also every CoC of the other development clips (one index, no timed insertion).

Output: <out>/frames_{event,general}.jsonl (A), <out>/followup_{event,general}.jsonl (B),
<out>/selection.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.alpamayo.model import load_target  # noqa: E402
from respect.candidates.datastore import Datastore  # noqa: E402
from respect.config import RespectConfig  # noqa: E402
from respect.decoding.respect import RespectDecoder  # noqa: E402
from respect.eval.dataset import iterate, load_frames, view  # noqa: E402
from respect.eval.frames import Runner  # noqa: E402
from respect.kernels.patch import dedicated_kernels  # noqa: E402

DEF = dict(N=8, scope="all", gmax=2, tree=16, topk=4, budget=64)   # follow-up defaults; N is swept in phase A
NS_EVENT = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 50, 64, 80, 100]
NS_GENERAL = [n for n in NS_EVENT if n <= 30]
B1 = [dict(tree=t, topk=k, budget=b) for t in (7, 16, 32, 48) for k in (2, 4, 8) for b in (32, 64, 128)
      if not (t == 7 and k != 4) and t <= b]
B2 = [dict(scope=s, gmax=g) for s in ("same", "all") for g in (1, 2, 3)]
SETS = ("event", "general")
FOLLOW_KEYS = ("scope", "gmax", "tree", "topk", "budget")


def key(p: dict) -> str:
    return "|".join(f"{k}={p[k]}" for k in ("N", "scope", "gmax", "tree", "topk", "budget"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def sweep_variant(r: dict, n: int) -> dict:
    """Phase-A record of N = n with the follow-up values at the defaults."""
    return next(v for v in r["variants"] if v["params"]["N"] == n and
                all(v["params"][k] == DEF[k] for k in FOLLOW_KEYS))


def rule(scores: dict, cost=lambda c: c):
    """scores: value -> (per-set means, per-set p99s).  Within 1% of the best mean, the smallest value."""
    m = {c: float(np.mean(v[0])) for c, v in scores.items()}
    p = {c: float(np.mean(v[1])) for c, v in scores.items()}
    best = min(m.values())
    pick = min((c for c in scores if m[c] <= best * 1.01), key=cost)
    return pick, {str(c): [m[c], list(scores[c][0]), p[c]] for c in scores}


def stats(xs) -> tuple[float, float]:
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(np.percentile(xs, 99))


def choose_n(rows: dict) -> tuple[int, dict]:
    ns = sorted(set(NS_EVENT) & set(NS_GENERAL))
    sc = {}
    for n in ns:
        st = [stats([sweep_variant(r, n)["ms_mean2"] for r in rows[s]]) for s in SETS]
        sc[n] = ([x[0] for x in st], [x[1] for x in st])
    return rule(sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--inputs", required=True, help="input files of the development frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("all", "sweep", "followup", "select"), default="all")
    ap.add_argument("--limit-frames", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phases = ("sweep", "followup", "select") if args.phase == "all" else (args.phase,)
    clips = {s: load_frames(Path(args.data) / f"dev_{s}.jsonl") for s in SETS}
    if phases == ("select",):
        write_selection(out)
        return
    target = load_target(local_files_only=True)
    run = Runner(target)

    def corpus(set_name, cid, frame_idx, scope, gmax):
        same = [tuple(f.reference_coc) for i, f in clips[set_name][cid].items() if i < frame_idx]
        others = [tuple(f.reference_coc) for s in clips for c2 in clips[s] if c2 != cid
                  for f in clips[s][c2].values()] if scope == "all" else []
        return Datastore(same + others, gmax=gmax)

    def decoder(p):
        cfg = dataclasses.replace(RespectConfig(), num_prev_frames=p["N"], draft_tree_nodes=p["tree"],
                                  draft_topk=p["topk"], node_budget=p["budget"], datastore_key_len=p["gmax"])
        return RespectDecoder(target, run.buf, cfg, run.draft, run.mask_emb)

    def measure(x, set_name, cid, fr, variants, k_tokens, cache_k):
        cache_ds = {}
        recs = {key(p): dict(params=p) for p in variants}
        for pas, order in (("fwd", variants), ("rev", variants[::-1])):
            for p in order:
                ck = (p["scope"], p["gmax"])
                if ck not in cache_ds:
                    cache_ds[ck] = corpus(set_name, cid, fr.frame_idx, *ck)
                cands = view(clips[set_name], cid, fr.frame_idx, p["N"], target.eos).candidates
                dec = decoder(p)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                r = dec.decode(x.fs, cands, cache_ds[ck])
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                L = min(int(cache_k.get_seq_length()), x.fs.prompt_len + len(r.tokens))
                kv = all(torch.equal(cache_k.layers[i].keys[..., :L, :], run.buf.base_k[i][..., :L, :]) and
                         torch.equal(cache_k.layers[i].values[..., :L, :], run.buf.base_v[i][..., :L, :])
                         for i in range(len(cache_k.layers)))
                recs[key(p)][pas] = dict(ms=round(ms, 3), front_ms=round(r.initial_ms, 3), rounds=r.follow_rounds,
                                         front_nodes=r.initial_nodes, front_cands=r.initial_candidates,
                                         accepted=r.accepted_initial, n_tokens=len(r.tokens),
                                         round_acc=r.round_accepted, round_nodes=r.round_nodes,
                                         match_K=(r.tokens == k_tokens), kv_equal=kv)
        for v in recs.values():
            v["ms_mean2"] = round((v["fwd"]["ms"] + v["rev"]["ms"]) / 2, 3)
            v["match_K_both"] = bool(v["fwd"]["match_K"] and v["rev"]["match_K"])
        return list(recs.values())

    def pass_over(fname, plan, label):
        """Measure plan(set, cid, frame) -> variants (None: skip) on every scored frame."""
        t_start, warmed = time.perf_counter(), False
        for set_name in SETS:
            fpath = out / f"{fname}_{set_name}.jsonl"
            done = {(r["clip_id"], r["frame_idx"]) for r in jsonl(fpath)}
            fp = fpath.open("a")
            k_set = 0
            for cid, _hist, scored in iterate(clips[set_name], "dev"):
                for fr in scored:
                    variants = plan(set_name, cid, fr)
                    if variants is None:
                        continue
                    k_set += 1
                    if args.limit_frames and k_set > args.limit_frames:
                        break
                    if (cid, fr.frame_idx) in done:
                        continue
                    x = run.prepare(cid, fr, Path(args.inputs))
                    with dedicated_kernels(target.model, x.fs.prompt_len):
                        k_tokens, cache_k = run.sequential(x.fs)
                        if not warmed:   # first frame: every variant once, not recorded
                            measure(x, set_name, cid, fr, variants, k_tokens, cache_k)
                            warmed = True
                        recs = measure(x, set_name, cid, fr, variants, k_tokens, cache_k)
                        del cache_k
                    fp.write(json.dumps(dict(set=set_name, clip_id=cid, frame_idx=fr.frame_idx,
                                             prefill_len=x.fs.prompt_len, event_cluster=fr.meta.get("event_type"),
                                             tokens_K=k_tokens, variants=recs)) + "\n")
                    fp.flush()
                    bad = sum(not (v["match_K_both"] and v["fwd"]["kv_equal"]) for v in recs)
                    print("[%s %s %d] %s f%d  variants %d  mismatch %d  (%.1f min)" % (
                        label, set_name, k_set, cid[:8], fr.frame_idx, len(recs), bad,
                        (time.perf_counter() - t_start) / 60), flush=True)
            fp.close()

    if "sweep" in phases:
        ns = {"event": NS_EVENT, "general": NS_GENERAL}
        pass_over("frames", lambda s, cid, fr: [dict(DEF, N=n) for n in ns[s]], "A")

    if "followup" in phases:
        rows = {s: jsonl(out / f"frames_{s}.jsonl") for s in SETS}
        n_, _ = choose_n(rows)
        tail = {(r["clip_id"], r["frame_idx"]) for s in SETS for r in rows[s]
                if sweep_variant(r, n_)["fwd"]["rounds"] > 0}
        grid = [dict(DEF, N=n_)] + [dict(DEF, N=n_, **q) for q in B1 + B2]
        grid = list({key(p): p for p in grid}.values())
        print(f"followup: N = {n_}, {len(tail)} frames, {len(grid)} variants", flush=True)
        pass_over("followup", lambda s, cid, fr: grid if (cid, fr.frame_idx) in tail else None, "B")

    if "select" in phases:
        write_selection(out)


def write_selection(out: Path) -> None:
    rows = {s: jsonl(out / f"frames_{s}.jsonl") for s in SETS}
    follow = {s: jsonl(out / f"followup_{s}.jsonl") for s in SETS}
    chosen, tables = select(rows, follow)
    (out / "selection.json").write_text(json.dumps(dict(chosen=chosen, default=DEF, tables=tables,
                                                        n_frames={s: len(rows[s]) for s in SETS},
                                                        followup_frames={s: len(follow[s]) for s in SETS}), indent=1))
    print("chosen:", chosen, flush=True)


def select(rows: dict, follow: dict) -> tuple[dict, dict]:
    """N from phase A; follow-up values at the chosen N from phase B.  A frame completed by
    initial verification at the chosen N contributes its phase-A time to every follow-up value."""
    n_, tn = choose_n(rows)
    fb = {s: {(r["clip_id"], r["frame_idx"]): r for r in follow[s]} for s in SETS}

    def ms(s, r, q):
        f = fb[s].get((r["clip_id"], r["frame_idx"]))
        if f is None:
            return sweep_variant(r, n_)["ms_mean2"]
        want = dict(DEF, N=n_, **q)
        return next(v["ms_mean2"] for v in f["variants"] if all(v["params"][k] == want[k] for k in want))

    def pick(cands, to_q, cost=lambda c: c):
        sc = {}
        for c in cands:
            st = [stats([ms(s, r, to_q(c)) for r in rows[s]]) for s in SETS]
            sc[c] = ([x[0] for x in st], [x[1] for x in st])
        return rule(sc, cost)

    b1, t1 = pick([(q["tree"], q["topk"], q["budget"]) for q in B1], lambda c: dict(tree=c[0], topk=c[1], budget=c[2]))
    b2, t2 = pick([(q["scope"], q["gmax"]) for q in B2], lambda c: dict(scope=c[0], gmax=c[1]),
                  lambda c: (0 if c[0] == "same" else 1, c[1]))
    return (dict(N=n_, tree=b1[0], topk=b1[1], budget=b1[2], scope=b2[0], gmax=b2[1]),
            dict(N=tn, tree_topk_budget=t1, scope_gmax=t2))


if __name__ == "__main__":
    main()
