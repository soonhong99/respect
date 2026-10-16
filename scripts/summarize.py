#!/usr/bin/env python3
"""Aggregate raw results into summary.json: every number reported in the paper.

Input directory layout (results/paper holds the paper's own measurements):
  main/frames_{event,general}.jsonl   run_main.py --flow output (evaluation set)
  dev/frames_{event,general}.jsonl    select_hparams.py output (development set)
  dev/followup_{event,general}.jsonl, dev/selection.json
  dev/tile_boundary.json              run_tile_boundary.py output (development frames)
  flashdrive_unified/                  baselines/flashdrive/offline_unified.py output (#0')
  delay/, dev_ablation/, realtime/, realtime_flashdrive/, realtime_fast/   Section VI and V-E (scripts/section6.py)

Usage: python scripts/summarize.py results/paper [--out results/paper/summary.json]
Requires numpy only; no GPU.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.eval.dataset import load_frames  # noqa: E402

CONFIGS = ["S0", "S1", "S2b", "S2a", "S3", "K", "S1fK", "S2bK", "S2aK", "S4"]
BINS = [(-20, -16), (-15, -11), (-10, -6), (-5, -1), (0, 4), (5, 9), (10, 14), (15, 19)]
RNG = np.random.default_rng(20260930)
DS = [2, 3, 5, 10, 20]
MODES = ["contig", "stride"]
DEF = dict(N=8, scope="all", gmax=2, tree=16, topk=4, budget=64)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mean2(cfg, key):
    return (cfg["fwd"][key] + cfg["rev"][key]) / 2


def latency(a) -> dict:
    a = np.asarray(a, dtype=float)
    return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(a.max()))


def clip_bootstrap(rows, a, b, n=2000) -> list[float]:
    """95% interval of the paired mean difference a - b, resampling clips."""
    by = collections.defaultdict(list)
    for r, x, y in zip(rows, a, b):
        by[r["clip_id"]].append(x - y)
    clips = list(by)
    stats = [np.concatenate([by[clips[i]] for i in RNG.integers(0, len(clips), len(clips))]).mean()
             for _ in range(n)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def flashdrive(root: Path, data: Path, set_name: str) -> tuple[dict, dict]:
    """FlashDrive BF16 (#0') from baselines/flashdrive/offline_unified.py, keyed by (clip, frame).

    A frame's Decode time, verified blocks and energy are the mean of the two runs; the CoC is
    compared with the default sequential CoC (FlashDrive's tokens end at its stop token, the
    reference adds the trajectory-start token).
    """
    clips = load_frames(data / f"{set_name}.jsonl")
    runs = collections.defaultdict(list)
    for x in jsonl(root / "flashdrive_unified" / "fd.jsonl"):
        if x["set"] == set_name:
            runs[(x["clip_id"], x["frame_idx"])].append(x)
    rows = {k: dict(decode_ms=float(np.mean([x["decode_ms"] for x in v])),
                    blocks=float(np.mean([x["blocks"] for x in v])),
                    energy_j=float(np.mean([x["energy_j"] for x in v if "energy_j" in x])),
                    peak_delta_mib=max(x["peak_delta_mib"] for x in v), tokens=v[0]["tokens"],
                    same_tokens=all(x["tokens"] == v[0]["tokens"] for x in v), n_runs=len(v))
            for k, v in runs.items()}
    ms = np.array([x["decode_ms"] for x in rows.values()])
    match = sum(x["tokens"] == clips[c][f].reference_coc[:len(x["tokens"])] and
                len(clips[c][f].reference_coc) == len(x["tokens"]) + 1 for (c, f), x in rows.items())
    out = dict(n_frames=len(rows), runs_per_frame=min(x["n_runs"] for x in rows.values()), latency_ms=latency(ms),
               target_runs_mean=float(np.mean([x["blocks"] for x in rows.values()])),
               energy_j_mean=float(np.mean([x["energy_j"] for x in rows.values()])),
               peak_delta_mib_max=float(max(x["peak_delta_mib"] for x in rows.values())),
               coc_match_default_sequential=int(match), same_tokens_both_runs=int(sum(x["same_tokens"] for x in rows.values())))
    return out, rows


def same_environment_check(root: Path) -> dict | None:
    """#4 and the output-preserving Prefill/Flow check from the same run as #0' (PyTorch 2.11)."""
    p = root / "flashdrive_unified" / "respect.jsonl"
    if not p.exists():
        return None
    out = {}
    for s in ("event", "general"):
        rs = [r for r in jsonl(p) if r["set"] == s]
        pr = [r["preserve"] for r in rs]
        out[s] = dict(n=len(rs), s4_mean_ms=float(np.mean([r["s4_ms_mean2"] for r in rs])),
                      s4_match_K=int(sum(r["s4_match_K"] for r in rs)),
                      preserve={k: int(sum(x[k] for x in pr)) for k in ("kv_equal", "first_token_equal", "hidden_equal", "traj_equal")},
                      prefill_ms_median=float(np.median([x["prefill_ms"] for x in pr])),
                      prefill_fast_ms_median=float(np.median([x["prefill_fast_ms"] for x in pr])),
                      flow_public_ms_median=float(np.median([x["flow_public_ms"] for x in pr])),
                      flow_graph_ms_median=float(np.median([x["flow_graph_ms"] for x in pr])))
    return out


def summarize_set(root: Path, data: Path, key: str) -> dict:
    rows = jsonl(root / "main" / f"frames_{key}.jsonl")
    out = {"n_frames": len(rows), "configs": {}}
    ms = {c: np.array([r["configs"][c]["ms_mean2"] for r in rows]) for c in CONFIGS}
    for c in CONFIGS:
        cf = [r["configs"][c] for r in rows]
        out["configs"][c] = dict(
            latency_ms=latency(ms[c]),
            target_runs_mean=float(np.mean([mean2(x, "target_runs") for x in cf])),
            per_token_ms_pooled=float(ms[c].sum() / sum(x["n_tokens"] for x in cf)),
            energy_j_mean=float(np.mean([mean2(x, "energy_j") for x in cf])),
            power_module_w_mean=float(np.mean([mean2(x, "power_module_w") for x in cf])),
            peak_alloc_mib_max=float(max(max(x["fwd"]["peak_alloc_mib"], x["rev"]["peak_alloc_mib"]) for x in cf)),
            peak_delta_mib_max=float(max(max(x["fwd"]["peak_delta_mib"], x["rev"]["peak_delta_mib"]) for x in cf)),
            match_K=int(sum(x["match_K"] for x in cf)), match_S0=int(sum(x["match_S0"] for x in cf)),
            match_reference=int(sum(x["match_reference"] for x in cf)),
            same_fwd_rev=int(sum(x["same_fwd_rev"] for x in cf)), max_tokens=int(max(x["n_tokens"] for x in cf)))
        if "front_ms" in cf[0]["fwd"]:
            out["configs"][c]["initial_only_frames"] = int(sum(x["fwd"]["tail_tokens"] == 0 for x in cf))
    out["integrity"] = dict(kv_equal=int(sum(r["flow"]["kv_equal"] for r in rows)),
                            tokens_equal=int(sum(r["flow"]["tokens_equal"] for r in rows)))
    fn = np.array([r["configs"]["S4"]["fwd"]["front_nodes"] for r in rows])
    out["initial_trie"] = dict(nodes_mean=float(fn.mean()), frames_over_64=int((fn > 64).sum()))
    done = np.array([r["configs"]["S1fK"]["fwd"]["tail_tokens"] == 0 for r in rows])
    out["concentration"] = dict(done_frames=int(done.sum()), tail_frames=int((~done).sum()),
                                done_ms_sum=float(ms["S1fK"][done].sum()), tail_ms_sum=float(ms["S1fK"][~done].sum()))
    tail = ~done
    trows = [r for r, t in zip(rows, tail) if t]
    src = {c: dict(mean_ms=float(ms[c][tail].mean()), max_ms=float(ms[c][tail].max()),
                   tail_runs_mean=float(np.mean([mean2(r["configs"][c], "target_runs") - 1 for r in trows])))
           for c in ("S1fK", "S2bK", "S2aK", "S4")}
    ds_fast = ms["S2aK"][tail] < ms["S2bK"][tail]
    src["datastore_faster_frames"] = int(ds_fast.sum())
    src["dflash_faster_frames"] = int((~ds_fast).sum())
    for name, sel in (("on_datastore_faster", ds_fast), ("on_dflash_faster", ~ds_fast)):
        src[name] = {c: float(ms[c][tail][sel].mean()) for c in ("S2bK", "S2aK", "S4")}
    out["sources"] = src

    def agg(c, num, den, tail_only=False):
        n = d = 0.0
        for r in rows:
            for w in ("fwd", "rev"):
                q = r["configs"][c][w]
                if tail_only and q.get("rounds", 0) == 0:
                    continue
                n += q.get(num, 0)
                d += q.get(den, 0) if den else 1
        return n / d
    out["kernel_percall_ms"] = {"sequential": [agg("S0", "ms", "target_runs"), agg("K", "ms", "target_runs")],
                                "initial": [agg("S3", "front_ms", None), agg("S4", "front_ms", None)],
                                "follow_up": [agg("S3", "verify_ms", "rounds", True), agg("S4", "verify_ms", "rounds", True)]}
    out["follow_runs_on_follow_frames"] = {
        c: float(np.mean([mean2(r["configs"][c], "rounds") for r in rows if r["configs"][c]["fwd"]["rounds"] > 0]))
        for c in ("S3", "S4")}
    diff = np.array([not r["configs"]["S0"]["match_K"] for r in rows])
    ade4 = np.array([r["flow"]["s4_minADE6_m"] for r in rows])
    ade0 = np.array([r["reference_minADE6_m"] for r in rows])
    d = np.abs(ade4 - ade0)
    out["trajectory"] = dict(s4_minADE6_mean=float(ade4.mean()), s0_minADE6_mean=float(ade0.mean()),
                             s4_minFDE6_mean=float(np.mean([r["flow"]["s4_minFDE6_m"] for r in rows])),
                             s0_minFDE6_mean=float(np.mean([r["reference_minFDE6_m"] for r in rows])),
                             abs_diff_median=float(np.median(d)), share_under_1cm=float((d < 0.01).mean()),
                             s0_minADE6_median=float(np.median(ade0)),
                             seq_divergence=dict(frames=int(diff.sum()), s0_minADE6=float(ade0[diff].mean()),
                                                 k_minADE6=float(ade4[diff].mean())))
    out["ci95_mean_diff_ms"] = {"K-S4": clip_bootstrap(rows, ms["K"], ms["S4"]),
                                "S0-S1": clip_bootstrap(rows, ms["S0"], ms["S1"])}
    out["delay"] = delay(rows, ms["K"], ms["S4"])
    out["stages"] = stages(rows, ms)
    out["flashdrive"], fd_rows = flashdrive(root, data, key)
    if key == "event":
        rec = [dict(d=r["meta"]["frames_from_event"], seq=r["configs"]["S0"]["ms_mean2"],
                    res=r["configs"]["S4"]["ms_mean2"], fd=fd_rows[(r["clip_id"], r["frame_idx"])]["decode_ms"],
                    runs=mean2(r["configs"]["S4"], "target_runs")) for r in rows]
        tl = []
        for a, b in BINS:
            sel = [x for x in rec if a <= x["d"] <= b]
            row = dict(lo=a, hi=b, n=len(sel))
            for q in ("seq", "fd", "res"):
                v = np.array([x[q] for x in sel])
                row[q + "_mean"], row[q + "_max"] = float(v.mean()), float(v.max())
            row["res_max_runs"] = max(sel, key=lambda x: x["res"])["runs"]
            tl.append(row)
        out["event_timeline"] = tl
    return out


def adjacent_coc(data: Path, set_name: str) -> dict:
    """Section III-A, Fig. 3(a): how often consecutive frames repeat the reference CoC, and how many
    changed pairs differ from the first token."""
    pairs = same = first_differs = 0
    prefix = []
    for frames in load_frames(data / f"{set_name}.jsonl").values():
        idx = sorted(frames)
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                continue
            p, q = frames[a].reference_coc, frames[b].reference_coc
            pairs += 1
            same += p == q
            first_differs += p[0] != q[0]
            n = 0
            for u, v in zip(p, q):
                if u != v:
                    break
                n += 1
            prefix.append(n / len(q))
    return dict(pairs=pairs, identical=same, identical_share=same / pairs, first_token_differs=first_differs,
                first_token_differs_share_of_changed=first_differs / (pairs - same),
                prefix_share_mean=float(np.mean(prefix)))


def delay(rows, k, s4) -> dict:
    """Section V-D: #4 when the newest usable CoC is from frame t-d."""
    red = lambda a: dict(mean=float(1 - a.mean() / k.mean()), p99=float(1 - np.percentile(a, 99) / np.percentile(k, 99)),  # noqa: E731
                         max=float(1 - a.max() / k.max()))
    out = {"1": dict(latency_ms=latency(s4), reduction_vs_K=red(s4),
                     initial_only=int(sum(r["configs"]["S4"]["fwd"]["rounds"] == 0 for r in rows)), mismatch=0)}
    for d in DS:
        for m in MODES:
            q = f"{d}_{m}"
            a = np.array([r["lag"][q]["ms_mean2"] for r in rows])
            out[q] = dict(latency_ms=latency(a), reduction_vs_K=red(a),
                          initial_only=int(sum(r["lag"][q]["fwd"]["rounds"] == 0 for r in rows)),
                          mismatch=int(sum(not (r["lag"][q]["fwd"]["match_K"] and r["lag"][q]["rev"]["match_K"])
                                           for r in rows)),
                          candidates_mean=float(np.mean([r["lag"][q]["n_cand"] for r in rows])))
    return out


def stages(rows, ms) -> dict:
    """Stage times of the public implementation and the frame time they imply."""
    pre = np.array([r["prefill_ms"] for r in rows])
    flow = np.array([np.median(r["flow"]["flow_ms_per_seed"]) for r in rows])
    o1 = np.array([r["official"]["traj1"]["total_ms"] for r in rows])
    o6 = np.array([r["official"]["traj6"]["total_ms"] for r in rows if "traj6" in r["official"]])
    ins = np.array([mean2(r["configs"]["S4"], "ds_insert_ms") for r in rows])
    med = lambda a: float(np.median(a))  # noqa: E731
    return dict(prefill_incl_vision_ms=latency(pre), vision_ms=latency([r["vision_ms"][0] for r in rows]),
                flow_1traj_ms=latency(flow), official_1traj_total_ms=latency(o1),
                official_6traj_total_ms=latency(o6), official_6traj_frames=int(len(o6)),
                datastore_insert_ms=latency(ins),
                frame_time_median_ms=dict(sequential=med(pre + ms["S0"] + flow), dedicated_sequential=med(pre + ms["K"] + flow),
                                          respect=med(pre + ms["S4"] + flow)))


def nselect(rows: dict) -> list[dict]:
    """Supplementary N selection: per N (both development sets measured), the share of frames completed by initial
    verification alone and the mean, p99 and maximum Decode time, each the mean of the two sets."""
    by = {s: collections.defaultdict(list) for s in rows}
    for s, rs in rows.items():
        for r in rs:
            for v in r["variants"]:
                p = v["params"]
                if all(p[k] == DEF[k] for k in DEF if k != "N"):
                    by[s][p["N"]].append((v["fwd"]["rounds"] == 0, v["ms_mean2"]))
    out = []
    for n in sorted(set(by["event"]) & set(by["general"])):
        st = lambda f: float(np.mean([f(np.array([x[1] for x in by[s][n]])) for s in by]))  # noqa: E731
        out.append(dict(n=n, initial_complete_pct=float(np.mean([100 * np.mean([x[0] for x in by[s][n]]) for s in by])),
                        mean_ms=st(np.mean), p99_ms=st(lambda a: np.percentile(a, 99)), max_ms=st(np.max)))
    return out


def one_at_a_time(follow: dict, sel: dict) -> dict:
    """Supplementary continuation setting selection: the chosen setting with one continuation setting changed at a time, at the chosen N.
    Mean and p99 Decode time: the selection values (two sets averaged). Accepted tokens per continuation target
    run: pooled over the frames that need continuation verification at the chosen N, two sets averaged."""
    acc = {s: collections.defaultdict(lambda: [0, 0]) for s in follow}
    for s, rs in follow.items():
        for r in rs:
            for v in r["variants"]:
                p = v["params"]
                k = (p["scope"], p["gmax"], p["tree"], p["topk"], p["budget"])
                acc[s][k][0] += sum(v["fwd"]["round_acc"])
                acc[s][k][1] += v["fwd"]["rounds"]
    ch = sel["chosen"]
    tb, sg = sel["tables"]["tree_topk_budget"], sel["tables"]["scope_gmax"]

    def row(scope=ch["scope"], gmax=ch["gmax"], tree=ch["tree"], topk=ch["topk"], budget=ch["budget"]):
        k = (scope, gmax, tree, topk, budget)
        t = sg[str((scope, gmax))] if (tree, topk, budget) == (ch["tree"], ch["topk"], ch["budget"]) \
            else tb[str((tree, topk, budget))]
        return dict(mean_ms=t[0], p99_ms=t[2], accepted_per_run=float(np.mean([acc[s][k][0] / acc[s][k][1] for s in acc])))
    return dict(scope={sc: row(scope=sc) for sc in ("same", "all")}, key={g: row(gmax=g) for g in (1, 2, 3)},
                tree={t: row(tree=t) for t in (7, 16, 32, 48)}, topk={k: row(topk=k) for k in (2, 4, 8)},
                budget={b: row(budget=b) for b in (32, 64, 128)})


def dev(root: Path) -> dict | None:
    """Development-set selection (Section IV-B/C): N sweep, continuation grids at the chosen N, tile boundary."""
    if not (root / "dev" / "frames_event.jsonl").exists():
        return None
    rows = {s: jsonl(root / "dev" / f"frames_{s}.jsonl") for s in ("event", "general")}

    def var(r, **q):
        p = dict(DEF, **q)
        return next(v for v in r["variants"] if all(v["params"][k] == p[k] for k in p))
    ev = rows["event"]
    ns = sorted({v["params"]["N"] for r in ev for v in r["variants"]})
    nsweep = []
    for n in ns:
        vs = [var(r, N=n) for r in ev]
        m = lambda k: float(np.mean([(v["fwd"][k] + v["rev"][k]) / 2 for v in vs]))  # noqa: E731
        nsweep.append(dict(n=n, initial_ms=m("front_ms"), follow_ms=m("ms") - m("front_ms"), total_ms=m("ms"),
                           follow_frames=int(sum(v["fwd"]["rounds"] > 0 for v in vs)),
                           initial_nodes_mean=float(np.mean([v["fwd"]["front_nodes"] for v in vs]))))

    follow = {s: jsonl(root / "dev" / f"followup_{s}.jsonl") for s in ("event", "general")}
    sel = json.loads((root / "dev" / "selection.json").read_text())
    ch = sel["chosen"]

    def grid(q):
        out = {}
        for s, rs in follow.items():
            want = dict(DEF, N=ch["N"], **q)
            vs = [next(v for v in r["variants"] if all(v["params"][k] == want[k] for k in want)) for r in rs]
            out[s] = dict(frames=len(rs), mean_ms=float(np.mean([v["ms_mean2"] for v in vs])),
                          accepted_per_run=float(np.mean([a for v in vs for a in v["fwd"]["round_acc"]])),
                          nodes_per_run=float(np.mean([a for v in vs for a in v["fwd"]["round_nodes"]])))
        return out
    allrows = [r for d in (rows, follow) for rs in d.values() for r in rs]
    variants = sum(len(r["variants"]) for r in allrows)
    mismatch = sum(not (v["match_K_both"] and v["fwd"]["kv_equal"]) for r in allrows for v in r["variants"])
    tile = json.loads((root / "dev" / "tile_boundary.json").read_text())
    nodes = [r["nodes"] for r in tile[0]["rows"]]
    med = np.median([[r["median_ms"] for r in t["rows"]] for t in tile], axis=0)
    return dict(n_frames={s: len(v) for s, v in rows.items()},
                follow_frames={s: len(v) for s, v in follow.items()},
                variants=variants, mismatches=mismatch, chosen=sel["chosen"], tables=sel["tables"],
                nsweep=nsweep, nselect=nselect(rows), one_at_a_time=one_at_a_time(follow, sel),
                tree={t: grid(dict(tree=t)) for t in (7, 16, 32, 48)},
                topk={k: grid(dict(topk=k)) for k in (2, 4, 8)},
                budget={b: grid(dict(budget=b)) for b in (32, 64, 128)},
                datastore={f"{sc}_{g}": grid(dict(scope=sc, gmax=g)) for sc in ("same", "all") for g in (1, 2, 3)},
                tile_boundary=dict(frames=[(t["event_type"], t["clip_id"], t["frame_idx"]) for t in tile],
                                   nodes=nodes, median_of_frames_ms=med.tolist(),
                                   step_64_65_ms=[t["rows"][nodes.index(65)]["median_ms"] - t["rows"][nodes.index(64)]["median_ms"]
                                                  for t in tile]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out")
    args = ap.parse_args()
    root, data = Path(args.root), Path(args.data)
    s = {"sets": {k: summarize_set(root, data, k) for k in ("event", "general")},
         "adjacent_coc": {k: adjacent_coc(data, k) for k in ("event", "general")},
         "dev": dev(root)}
    s["flashdrive_same_environment"] = same_environment_check(root)
    if (root / "realtime" / "cycles.jsonl").exists():
        from section6 import section6
        s["section6"] = section6(root, data)
    out = Path(args.out) if args.out else root / "summary.json"
    out.write_text(json.dumps(s, indent=1))
    for k, v in s["sets"].items():
        c, fd = v["configs"], v["flashdrive"]
        k4, s4 = c["K"]["latency_ms"], c["S4"]["latency_ms"]
        print(f"[{k}] #4 mean {s4['mean']:.1f} ms, p99 {s4['p99']:.1f}, max {s4['max']:.1f} | vs deterministic-kernel "
              f"autoregressive: mean -{100 * (1 - s4['mean'] / k4['mean']):.1f}%, p99 -{100 * (1 - s4['p99'] / k4['p99']):.1f}%, "
              f"max -{100 * (1 - s4['max'] / k4['max']):.1f}% | #4 = reference in {c['S4']['match_K']}/{v['n_frames']} frames, "
              f"KV equal {v['integrity']['kv_equal']} | #0' mean {fd['latency_ms']['mean']:.1f} ms")
        dl = v["delay"]["20_stride"]
        print(f"      delay d=20 (stride): #4 mean {dl['latency_ms']['mean']:.1f} ms, "
              f"-{100 * dl['reduction_vs_K']['mean']:.1f}% vs deterministic-kernel autoregressive, mismatches {dl['mismatch']}")
    if "section6" in s:
        rt = s["section6"]["realtime"]
        for c in ("S0", "S4"):
            v = rt["configs"][c]
            print(f"[real-time {c}] cycle {v['cycle_s']['mean']:.2f} s, Decode {v['stage_mean_ms']['decode_ms']:.0f} ms, "
                  f"trajectory age {v['age_s']['mean']:.2f} s, expired {100 * v['expired_share']:.2f}%, "
                  f"event reaction {v['reaction_s']['mean']:.2f} s (max {v['reaction_s']['max']:.2f})")
        pr = rt["paired_S0_minus_S4"]["reaction_s"]
        print(f"      reaction S0 - S4 {pr['mean']:.2f} s, 95% [{pr['ci95'][0]:.2f}, {pr['ci95'][1]:.2f}], "
              f"verify mismatches {rt['verify']['mismatches']}/{rt['verify']['n']}")
        sl = s["section6"]["stage_link"]
        if sl:
            for c in ("S4", "FC8", "D8", "S4F"):
                if c in sl["configs"]:
                    v = sl["configs"][c]
                    print(f"[stage link {c}] cycle {v['cycle_s']:.2f} s, Prefill {v['stage_mean_ms']['prefill_ms']:.0f} ms, "
                          f"Decode {v['stage_mean_ms']['decode_ms']:.0f} ms, Flow {v['stage_mean_ms']['flow_ms']:.0f} ms, "
                          f"event reaction {v['reaction_mean_s']:.2f} s")
            for k in ("S4-FC8", "FC8-D8", "S4-S4F", "FC8-S4F"):
                if k in sl["pairs"]:
                    p = sl["pairs"][k]
                    print(f"      reaction {k} {p['reaction_diff_s']:.2f} s, 95% [{p['ci95'][0]:.2f}, {p['ci95'][1]:.2f}]")
            print(f"      same-CoC trajectory S4 vs E8: median {sl['trajectory_same_coc']['S4-E8']['median_m']:.2f} m; "
                  f"verify {sl['verify']}")
    if s["dev"]:
        print(f"[dev] chosen {s['dev']['chosen']}, variants {s['dev']['variants']}, mismatches {s['dev']['mismatches']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
