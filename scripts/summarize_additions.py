#!/usr/bin/env python3
"""Numbers added to the paper after summary.json: observed real-time delays, trajectory error of #0', N under delay,
signed stopping margins of the four RSS-critical events, and the three systems on the frame of Fig. 13.

Also reads realtime/cycles.jsonl and realtime_fast/cycles.jsonl (delay d of the real-time runs).

Inputs (results/paper unless noted):
  summary.json                              scripts/summarize.py output (K latency, d = 1 and d = 40 rows)
  main/frames_{event,general}.jsonl         #0 (reference) and #4 minADE6, CoC of #0 and of the dedicated-kernel run
  delay_observed/frames_{event,general}.jsonl
                                            run_main.py --delay-only --delays 16,20,23 --modes stride; the general set
                                            with --history data/general_history_d40.jsonl + data/general_history_d16_d23.jsonl
  flashdrive_quality/fd_quality.jsonl       baselines/flashdrive/quality_fd.py --fixed-seed 0 (#0', six trajectories)
  data/dev_{event,general}.jsonl            development set CoC of #0 (repository data/)
  frame_trajectories/bbbe775f_0160.json     six trajectories of #0, RESPECT (#4) and #0' and the recorded path (Fig. 13)

Usage: python scripts/summarize_additions.py results/paper [--out results/paper/summary_additions.json]
Requires numpy only; no GPU.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETS = ("event", "general")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def clip_bootstrap(rows, diff, rng, n=10000) -> list[float]:
    """95% interval of the mean difference over clip means, resampling clips (Section V-G interval)."""
    clips = sorted({r["clip_id"] for r in rows})
    g = np.array([np.mean([x for r, x in zip(rows, diff) if r["clip_id"] == c]) for c in clips])
    draws = rng.choice(len(clips), (n, len(clips)))
    return [float(v) for v in np.quantile(g[draws].mean(1), [0.025, 0.975])]


def observed_delay(res: Path, summary: dict) -> dict:
    """Section V-D: #4 at the median delays of the real-time runs (d = 23: #4 run, d = 16: RESPECT run)."""
    out = {}
    for s in SETS:
        k = summary["sets"][s]["configs"]["K"]["latency_ms"]
        rows = jsonl(res / "delay_observed" / f"frames_{s}.jsonl")
        for d in (16, 20, 23):
            q = f"{d}_stride"
            a = np.array([r["lag"][q]["ms_mean2"] for r in rows])
            out[f"{s}_{q}"] = dict(
                frames=len(rows), mean_ms=float(a.mean()), p99_ms=float(np.percentile(a, 99)),
                reduction_vs_K=dict(mean=float(1 - a.mean() / k["mean"]), p99=float(1 - np.percentile(a, 99) / k["p99"])),
                initial_only=int(sum(r["lag"][q]["fwd"]["rounds"] == 0 for r in rows)),
                mismatch=int(sum(not (r["lag"][q]["fwd"]["match_K"] and r["lag"][q]["rev"]["match_K"]) for r in rows)))
    return out


def delay_range(summary: dict) -> dict:
    """Section V-D: d = 1 -> 40, frames completed by initial verification and mean reduction (from summary.json)."""
    out = {}
    for s in SETS:
        d1 = summary["sets"][s]["delay"]["1"]
        d40 = summary["section6"]["delay_extension"][s]["rows"]["40_stride"]
        out[s] = dict(d1=dict(initial_only=d1["initial_only"], mean_reduction=d1["reduction_vs_K"]["mean"]),
                      d40=dict(frames=d40["n"], initial_only=d40["initial_only"], mean_reduction=d40["reduction_vs_main_K"]))
    return out


def realtime_delay(res: Path) -> dict:
    """Section V-D/E, Fig. 10 bands: delay d of the processed frames in the real-time runs. The last cycle of a run
    processes the last recorded frame (g = g_end), whose gap to the previous processed frame is cut by the end of the
    recording, so it is left out."""
    out = {}
    for name, sub, cfg in (("#0 run", "realtime", "S0"), ("#4 run", "realtime", "S4"), ("RESPECT run", "realtime_fast", "S4F")):
        ks = sorted(r["k"] for r in jsonl(res / sub / "cycles.jsonl")
                    if r["cfg"] == cfg and r["k"] is not None and r["g"] < r["g_end"])
        n = len(ks)
        out[name] = dict(cycles=n, median=ks[int(.5 * (n - 1))], p10=ks[int(.1 * (n - 1))], p90=ks[int(.9 * (n - 1))])
    return out


def trajectory_error(res: Path) -> dict:
    """Section V-G and Fig. 14: minADE6 of #0, RESPECT (measured with #4) and #0' on the same frames."""
    fd = {(r["clip_id"], r["frame_idx"]): r["fd_minADE6_m"] for r in jsonl(res / "flashdrive_quality" / "fd_quality.jsonl")}
    out, rng = {}, np.random.default_rng(261009)   # one generator, event set first
    for s in SETS:
        rows = jsonl(res / "main" / f"frames_{s}.jsonl")
        m0 = np.array([r["reference_minADE6_m"] for r in rows])
        m4 = np.array([r["flow"]["s4_minADE6_m"] for r in rows])
        mf = np.array([fd[(r["clip_id"], r["frame_idx"])] for r in rows])
        diff_coc = np.array([r["configs"]["S0"]["tokens"] != r["tokens_K"] for r in rows])
        d4, dfd = np.abs(m4 - m0), np.abs(mf - m0)
        out[s] = dict(
            frames=len(rows),
            mean_minADE6_m={"#0": float(m0.mean()), "#4": float(m4.mean()), "#0'": float(mf.mean())},
            fd_minus_4=dict(mean=float((mf - m4).mean()), ci95=clip_bootstrap(rows, mf - m4, rng)),
            diff_from_0={
                "#4": dict(median=float(np.median(d4)), max=float(d4.max()), over_0p05=int((d4 > 0.05).sum()),
                           over_0p05_all_in_coc_diff=bool(np.all(diff_coc[d4 > 0.05])),
                           same_coc=dict(frames=int((~diff_coc).sum()), median=float(np.median(d4[~diff_coc])),
                                         max=float(d4[~diff_coc].max()))),
                "#0'": dict(median=float(np.median(dfd)), max=float(dfd.max()), over_0p05=int((dfd > 0.05).sum()))},
            coc_diff_frames=dict(frames=int(diff_coc.sum()), mean_minADE6_m={"#0": float(m0[diff_coc].mean()),
                                                                          "#4": float(m4[diff_coc].mean())}))
    return out


def n_under_delay() -> dict:
    """Section V-D: share of development frames whose N most recent CoCs (frames t-d, t-2d, ...) contain the current CoC."""
    out = {}
    for s in SETS:
        by = collections.defaultdict(dict)
        for r in jsonl(ROOT / "data" / f"dev_{s}.jsonl"):
            by[r["clip_id"]][int(r["frame_idx"])] = r
        for d in (1, 16, 20, 23):
            rate = {}
            for n in (1, 2, 4, 8, 16):
                hit = tot = 0
                for fr in by.values():
                    for f, r in fr.items():
                        if not r["scored"]:
                            continue
                        ent = [tuple(fr[g]["reference_coc"]) for g in sorted(fr) if g <= f - d and (f - g) % d == 0][-n:]
                        tot += 1
                        hit += tuple(r["reference_coc"]) in set(ent)
                rate[n] = hit / tot
            out[f"{s}_d{d}"] = dict(frames=tot, rate=rate, n4_to_n8=rate[8] - rate[4])
    return out


def stopping_margin(summary: dict) -> dict:
    """Section V-F, Figs. 1 and 12: signed stopping margin v_r (rho* - rho) with rho the mean event response latency of
    each run (positive: the computed stop lies before the lead object's computed stop; b_min = 4 m/s^2)."""
    s6 = summary["section6"]
    out = {}
    for cid, r in s6["rss"].items():
        link = s6["stage_link"]["rss"].get(cid)
        if "rho_star" not in r or link is None:
            continue
        v, rho = r["v_r"], r["rho_star"]["4.0"]
        lat = {"#0": float(np.mean(r["reaction_s"]["S0"])), "#4": float(np.mean(r["reaction_s"]["S4"])),
               "FlashDrive BF16": link["reaction_s"]["FC8"], "RESPECT": link["reaction_s"]["S4F"]}
        out[cid[:8]] = dict(event_type=r["event_type"], lead=r["lead"]["cls"], ego_kmh=v * 3.6, rho_star_s=rho,
                            latency_s=lat, margin_m={k: v * (rho - x) for k, x in lat.items()})
    return out


def fig13_frame(res: Path) -> dict:
    """Section V-G and Fig. 13: RESPECT against #0 (largest coordinate and 2-D position difference over the six
    trajectories) and the distance between the mean endpoints of #0' and #0."""
    f = json.loads((res / "frame_trajectories" / "bbbe775f_0160.json").read_text())
    p0, p4, pf = (np.asarray(f[k])[:, :, :2] for k in ("public_implementation_0", "respect_4", "flashdrive_bf16_0prime"))
    return dict(clip_id=f["clip_id"], frame_idx=f["frame_idx"],
                respect_max_coordinate_diff_m=float(np.abs(p4 - p0).max()),
                respect_max_position_diff_m=float(np.linalg.norm(p4 - p0, axis=-1).max()),
                flashdrive_mean_endpoint_distance_m=float(np.linalg.norm(pf[:, -1].mean(0) - p0[:, -1].mean(0))),
                recorded_stop_forward_m=float(np.asarray(f["recorded_path"])[-1][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    summary = json.loads((a.results / "summary.json").read_text())
    out = dict(observed_delay=observed_delay(a.results, summary), realtime_delay=realtime_delay(a.results),
               delay_range=delay_range(summary),
               trajectory_error=trajectory_error(a.results), n_under_delay=n_under_delay(),
               stopping_margin=stopping_margin(summary), fig13_frame=fig13_frame(a.results))
    (a.out or a.results / "summary_additions.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
