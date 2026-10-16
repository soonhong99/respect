"""Section VI and Section V-C numbers of the paper, computed from the raw results (no GPU).

Inputs under the results root:
  main/frames_{event,general}.jsonl        delay d <= 20 (``lag``), stage times
  delay/frames_{event,general}.jsonl       run_main.py --delay-only --delays 20,30,40
  dev_ablation/frames_{event,general}.jsonl run_main.py on the development set
  realtime/cycles.jsonl, verify.jsonl      run_realtime.py
  realtime/rss_leads.json                  extract_rss_leads.py
  realtime_flashdrive/cycles.jsonl, verify.jsonl   baselines/flashdrive/stage_link.py
  realtime_fast/cycles.jsonl, verify.jsonl         run_realtime_fast.py
"""
from __future__ import annotations

import collections
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.eval.dataset import load_frames  # noqa: E402
from respect.eval.realtime import HORIZON_S, rss_allowed_response, run_metrics  # noqa: E402

PHASES = [0, 6, 12, 18, 24]
B_MIN, B_MAX = 4.0, 8.0


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))]


def _pct_linear(xs, q):
    """Percentile with linear interpolation (numpy's default), as summarize.latency uses."""
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)


def _stats(xs):
    return dict(n=len(xs), mean=st.mean(xs), p95=_pct(xs, .95), p99=_pct(xs, .99), max=max(xs))


# ----------------------------------------------------------------- delay extension
def delay_extension(root: Path) -> dict:
    """Figure 12 points d = 30, 40 and the initial-verification acceptance share of every d."""
    out = {}
    for s in ("event", "general"):
        main = _jsonl(root / "main" / f"frames_{s}.jsonl")
        ext = {(r["clip_id"], r["frame_idx"]): r for r in _jsonl(root / "delay" / f"frames_{s}.jsonl")}
        k_ms = [r["configs"]["K"]["ms_mean2"] for r in main]
        k_mean = st.mean(k_ms)
        rows = {}
        for key in sorted({k for r in ext.values() for k in r["lag"]}, key=lambda x: (int(x.split("_")[0]), x)):
            rs = [r["lag"][key] for r in ext.values() if key in r["lag"]]
            ms = [x["ms_mean2"] for x in rs]
            rows[key] = dict(n=len(rs), mean_ms=st.mean(ms), reduction_vs_main_K=1 - st.mean(ms) / k_mean,
                             p99_ms=_pct_linear(ms, .99), max_ms=max(ms),
                             initial_only=sum(x["fwd"]["rounds"] == 0 for x in rs),
                             mismatch=sum(not (x["fwd"]["match_K"] and x["rev"]["match_K"]) for x in rs))
        for m in ("contig", "stride"):                 # d = 20 measured again in the extension session
            key = f"20_{m}"
            same = [(ext[(r["clip_id"], r["frame_idx"])]["lag"][key]["ms_mean2"], r["lag"][key]["ms_mean2"]) for r in main]
            rows[key]["session_change_vs_main"] = st.mean(a for a, _ in same) / st.mean(b for _, b in same) - 1

        def acc(x):
            return x["fwd"]["accepted"] / x["fwd"]["n_tokens"]
        share = {1: st.mean(acc(r["configs"]["S4"]) for r in main)}
        for d in (2, 3, 5, 10, 20):
            share[d] = st.mean(acc(r["lag"][f"{d}_stride"]) for r in main)
        for d in (30, 40):
            v = [acc(ext[(r["clip_id"], r["frame_idx"])]["lag"][f"{d}_stride"]) for r in main
                 if f"{d}_stride" in ext[(r["clip_id"], r["frame_idx"])]["lag"]]
            if v:
                share[d] = st.mean(v)
        out[s] = dict(K_mean_ms=k_mean, K_p99_ms=_pct_linear(k_ms, .99), K_max_ms=max(k_ms), rows=rows,
                      acceptance_share=share)
    return out


# ------------------------------------------------------------------- dev ablation
def dev_ablation(root: Path) -> dict | None:
    if not (root / "dev_ablation" / "frames_event.jsonl").exists():
        return None
    out = {}
    for s in ("event", "general"):
        rows = _jsonl(root / "dev_ablation" / f"frames_{s}.jsonl")
        cfgs = ["K", "S1fK", "S2bK", "S2aK", "S4"]
        mean = {c: st.mean(r["configs"][c]["ms_mean2"] for r in rows) for c in cfgs}
        out[s] = dict(n_frames=len(rows), mean_ms=mean, S4_reduction_vs_K=1 - mean["S4"] / mean["K"],
                      match_K={c: sum(r["configs"][c]["match_K"] for r in rows) for c in cfgs})
    return out


# ----------------------------------------------------------------- real-time runs
def _paired(per_run, metric, seed=260930, n=10000):
    """Mean of S0 - S4 over matched runs and its clip-level bootstrap 95% interval."""
    d = collections.defaultdict(list)
    for (cfg, cid, ph) in per_run:
        if cfg != "S0" or ("S4", cid, ph) not in per_run:
            continue
        a, b = metric(per_run[("S0", cid, ph)]), metric(per_run[("S4", cid, ph)])
        d[cid].append(a - b)
    vals = [x for v in d.values() for x in v]
    rng, cl, boots = random.Random(seed), list(d), []
    for _ in range(n):
        s = [x for c in (rng.choice(cl) for _ in cl) for x in d[c]]
        boots.append(st.mean(s))
    boots.sort()
    return dict(n=len(vals), mean=st.mean(vals), ci95=[boots[int(0.025 * n) - 1], boots[int(0.975 * n) - 1]],
                s4_shorter=sum(x > 0 for x in vals))


def realtime(root: Path) -> dict:
    rows = _jsonl(root / "realtime" / "cycles.jsonl")
    ver = _jsonl(root / "realtime" / "verify.jsonl")
    runs = collections.defaultdict(list)
    for r in rows:
        runs[(r["cfg"], r["clip_id"], r["phase"])].append(r)
    per_run = {k: run_metrics(v) for k, v in runs.items()}
    out = dict(cycles=len(rows), runs=len(runs), verify=dict(n=len(ver), mismatches=sum(not v["ok"] for v in ver)),
               stage_mean_ms=dict(input=st.mean(r["input_ms"] for r in rows), flow=st.mean(r["flow_ms"] for r in rows)),
               configs={})
    for cfg in ("S0", "S4"):
        R = [v for k, v in per_run.items() if k[0] == cfg]
        cyc = [r for r in rows if r["cfg"] == cfg]
        o = dict(cycle_s=_stats([x for v in R for x in v["cycles_s"]]),
                 stage_mean_ms={s: st.mean(x for v in R for x in v["stage_ms"][s])
                                for s in ("input_ms", "prefill_ms", "decode_ms", "flow_ms")},
                 decode_ms=_stats([x for v in R for x in v["stage_ms"]["decode_ms"]]),
                 age_s=_stats([x for v in R for x in v["ages"]]),
                 expired_share=sum(v["expired"] for v in R) / sum(v["n_ticks"] for v in R),
                 processed_share=st.mean(v["processed"] for v in R), covered_share=st.mean(v["covered"] for v in R),
                 reaction_s=_stats([v["reaction_s"] for v in R]), coc_s=_stats([v["coc_s"] for v in R]),
                 target_runs_mean=st.mean(r["target_runs"] for r in cyc))
        ks = sorted(x for v in R for x in v["k"])
        o["k"] = dict(mean=st.mean(ks), median=ks[int(.5 * (len(ks) - 1))], p10=ks[int(.1 * (len(ks) - 1))],
                      p90=ks[int(.9 * (len(ks) - 1))], max=ks[-1])
        if cfg == "S4":
            o["initial_complete_share"] = sum(r["accepted"] >= r["n_tokens"] for r in cyc) / len(cyc)
        out["configs"][cfg] = o
    out["reaction_to_cycle"] = {c: out["configs"][c]["reaction_s"]["mean"] / out["configs"][c]["cycle_s"]["mean"]
                                for c in ("S0", "S4")}
    out["paired_S0_minus_S4"] = dict(cycle_s=_paired(per_run, lambda v: st.mean(v["cycles_s"])),
                                     age_s=_paired(per_run, lambda v: st.mean(v["ages"])),
                                     reaction_s=_paired(per_run, lambda v: v["reaction_s"]))
    out["reaction_by_run"] = {f"{k[0]}|{k[1]}|{k[2]}": v["reaction_s"] for k, v in per_run.items()}
    return out


# ---------------------------------------------------------------------------- RSS
def rss(root: Path, rt: dict) -> dict:
    leads = json.loads((root / "realtime" / "rss_leads.json").read_text())
    out = {}
    for cid, x in leads.items():
        rho = {c: [rt["reaction_by_run"][f"{c}|{cid}|{p}"] for p in PHASES] for c in ("S0", "S4")}
        o = dict(event_type=x["event_type"], v_r=x["v_r"], lead=x["lead"], reaction_s=rho)
        L = x["lead"]
        if L and x["v_r"] >= 0.5:
            o["rho_star"] = {str(b): (L["gap_m"] - x["v_r"] ** 2 / (2 * b) + max(L["v_f"], 0) ** 2 / (2 * B_MAX)) / x["v_r"]
                             for b in (3.0, 4.0, 6.0)}
            o["rho_star_amax2"] = rss_allowed_response(x["v_r"], L["gap_m"], L["v_f"], B_MIN, B_MAX, a_max=2.0)
            rs = o["rho_star"]["4.0"]
            o["within"] = {c: sum(t <= rs for t in rho[c]) for c in rho}
            o["shortfall_m"] = {c: x["v_r"] * max(0.0, st.mean(rho[c]) - rs) for c in rho}
        out[cid] = o
    return out


# ------------------------------------------ Prefill and Flow acceleration (VI-D)
FD_CONFIGS = ["S4", "FC8", "FC10", "E8", "E10", "E10N", "D8", "D10N"]
FD_PAIRS = [("FC8", "D8"), ("FC10", "D10N"), ("S4", "D10N"), ("S4", "E10N"), ("FC8", "E8"), ("S4", "FC8"), ("D8", "FC8")]
FAST_PAIRS = [("S4", "S4F"), ("FC8", "S4F"), ("D8", "S4F"), ("D10N", "S4F"), ("E10N", "S4F")]


def _pair_diffs(met: dict, pairs: list, seed: int = 260930, n: int = 10000) -> dict:
    """Paired event-reaction differences a - b over matched (clip, start position) runs, with the
    clip-level bootstrap 95 % interval used for the #0 and #4 runs (_paired)."""
    out = {}
    for a, b in pairs:
        d = collections.defaultdict(list)
        cyc = []
        for (cfg, cid, ph) in met:
            if cfg == a and (b, cid, ph) in met:
                d[cid].append(met[(a, cid, ph)]["reaction_s"] - met[(b, cid, ph)]["reaction_s"])
                cyc.append(st.mean(met[(a, cid, ph)]["cycles_s"]) - st.mean(met[(b, cid, ph)]["cycles_s"]))
        vals = [x for v in d.values() for x in v]
        rng, cl, boots = random.Random(seed), list(d), []
        for _ in range(n):
            boots.append(st.mean([x for c in (rng.choice(cl) for _ in cl) for x in d[c]]))
        boots.sort()
        out[f"{a}-{b}"] = dict(n=len(vals), reaction_diff_s=st.mean(vals),
                               ci95=[boots[int(0.025 * n) - 1], boots[int(0.975 * n) - 1]],
                               second_shorter=sum(x > 0 for x in vals), cycle_diff_s=st.mean(cyc))
    return out


def stage_link(root: Path, rs: dict) -> dict | None:
    """FlashDrive stages linked with RESPECT Decode and the output-preserving Prefill and Flow, all
    measured in real time in one environment (baselines/flashdrive/stage_link.py, scripts/run_realtime_fast.py)."""
    if not (root / "realtime_flashdrive" / "cycles.jsonl").exists():
        return None
    import numpy as np
    rows = _jsonl(root / "realtime_flashdrive" / "cycles.jsonl")
    fast_dir = root / "realtime_fast"
    fast = _jsonl(fast_dir / "cycles.jsonl") if (fast_dir / "cycles.jsonl").exists() else []
    allrows = rows + fast
    runs = collections.defaultdict(list)
    for r in allrows:
        runs[(r["cfg"], r["clip_id"], r["phase"])].append(r)
    met = {k: run_metrics(v) for k, v in runs.items()}
    cfgs = FD_CONFIGS + (["S4F"] if fast else [])
    out = dict(configs={}, pairs={}, rss={}, trajectory_same_coc={}, decode_split={}, verify={})
    for c in cfgs:
        ks = [k for k in met if k[0] == c]
        cyc = [r for r in allrows if r["cfg"] == c]
        ages = [a for k in ks for a in met[k]["ages"]]
        react = [met[k]["reaction_s"] for k in ks]
        o = dict(runs=len(ks), cycles=len(cyc), cycle_s=st.mean(r["cycle_ms"] / 1000 for r in cyc),
                 stage_mean_ms={s: st.mean(r[s] for r in cyc) for s in ("input_ms", "prefill_ms", "decode_ms", "flow_ms")},
                 age_s=st.mean(ages), expired_share=sum(a > HORIZON_S for a in ages) / len(ages),
                 reaction_mean_s=st.mean(react), reaction_max_s=max(react),
                 k_median=float(np.median([r["k"] for r in cyc if r["k"] is not None])))
        if any("accepted" in r for r in cyc):
            o["initial_complete_share"] = sum(r.get("accepted") == r["n_tokens"] for r in cyc) / len(cyc)
        if any("kv_rope_ms" in r for r in cyc):
            o["kv_rope_ms"] = st.mean(r["kv_rope_ms"] for r in cyc if "kv_rope_ms" in r)
        out["configs"][c] = o
    out["pairs"].update(_pair_diffs(met, FD_PAIRS))
    if fast:
        out["pairs"].update(_pair_diffs(met, FAST_PAIRS))
    for cid, x in rs.items():
        if "rho_star" not in x or x["shortfall_m"]["S0"] <= 0:
            continue
        rho = x["rho_star"]["4.0"]
        mr = {c: st.mean(met[k]["reaction_s"] for k in met if k[0] == c and k[1] == cid) for c in cfgs}
        out["rss"][cid] = dict(rho_star=rho, reaction_s=mr, shortfall_m={c: x["v_r"] * max(0.0, v - rho) for c, v in mr.items()})
    by = collections.defaultdict(dict)                       # one row per (clip, start position, frame)
    for r in allrows:
        by[(r["clip_id"], r["phase"], r["g"])].setdefault(r["cfg"], r)
    for a, b in [("S4", "E10N"), ("S4", "D10N"), ("S4", "E8"), ("FC8", "D8")]:
        dev = [float(np.abs(np.asarray(v[a]["traj"]) - np.asarray(v[b]["traj"])).max())
               for v in by.values() if a in v and b in v and v[a]["tokens"] == v[b]["tokens"]]
        out["trajectory_same_coc"][f"{a}-{b}"] = dict(n=len(dev), median_m=float(np.median(dev)), max_m=max(dev))
    both = [v for v in by.values() if "FC8" in v and "D8" in v]
    for name, sel in (("initial_complete", [v for v in both if v["D8"]["accepted"] == v["D8"]["n_tokens"]]),
                      ("follow_up", [v for v in both if v["D8"]["accepted"] != v["D8"]["n_tokens"]])):
        out["decode_split"][name] = dict(n=len(sel), FC8_decode_ms=st.mean(v["FC8"]["decode_ms"] for v in sel),
                                         D8_decode_ms=st.mean(v["D8"]["decode_ms"] for v in sel))
    ver = _jsonl(root / "realtime_flashdrive" / "verify.jsonl")
    out["verify"]["flashdrive"] = {k: dict(n=sum(v["kind"] == k for v in ver), mismatches=sum(v["kind"] == k and not v["ok"] for v in ver))
                                   for k in ("R", "D", "FC")}
    if fast:
        vf = _jsonl(fast_dir / "verify.jsonl")
        out["verify"]["fast"] = dict(frames=len(vf), token_mismatches=sum(not v["tokens_ok"] for v in vf),
                                     trajectory_mismatches=sum(not v["traj_ok"] for v in vf))
    return out


# --------------------------------------------------------- supplementary checks
def n_check(data: Path) -> dict:
    """Share of development frames with a previous completed CoC equal to the current one."""
    out = {}
    for s in ("event", "general"):
        clips = load_frames(data / f"dev_{s}.jsonl")
        for d in (1, 20):
            rate = {}
            for n in (1, 2, 4, 8, 16):
                hit = tot = 0
                for fr in clips.values():
                    for i, f in fr.items():
                        if not f.scored:
                            continue
                        ent = [tuple(fr[g].reference_coc) for g in sorted(fr) if g <= i - d and (i - g) % d == 0][-n:]
                        tot += 1
                        hit += tuple(f.reference_coc) in set(ent)
                rate[n] = hit / tot
            out[f"{s}_d{d}"] = dict(rate=rate, n=tot)
    return out


def required_cycle(rt: dict, rs: dict) -> dict:
    """Cycle time whose mean reaction meets the RSS allowed response, and the cut it needs."""
    ratio = max(rt["reaction_to_cycle"].values())
    out = dict(reaction_to_cycle=ratio, events={})
    for cid, v in rs.items():
        if "rho_star" not in v or v["shortfall_m"]["S0"] <= 0:
            continue
        need = v["rho_star"]["4.0"] / ratio
        o = dict(rho_star=v["rho_star"]["4.0"], cycle_s=need)
        for c in ("S0", "S4"):
            sm = rt["configs"][c]["stage_mean_ms"]
            other = (sm["input_ms"] + sm["prefill_ms"] + sm["flow_ms"]) / 1000
            allowed = need - sm["decode_ms"] / 1000
            o[c] = dict(other_s=other, allowed_other_s=allowed, cut=(1 - allowed / other) if allowed > 0 else None)
        out["events"][cid] = o
    return out


def section6(root: Path, data: Path) -> dict:
    rt = realtime(root)
    rs = rss(root, rt)
    return dict(delay_extension=delay_extension(root), dev_ablation=dev_ablation(root), realtime=rt, rss=rs,
                stage_link=stage_link(root, rs), n_check=n_check(data),
                required_cycle=required_cycle(rt, rs), horizon_s=HORIZON_S)
