#!/usr/bin/env python3
"""Draw the paper's measurement figures from summary.json (English labels).

Usage: python scripts/make_figures.py results/paper [--out figures]
See docs/manuscript_map.md for the mapping of all 14 paper figures, supplementary
figures, tables, and equations.
supp_* names identify the supplementary measurements. paper_method_figures.py
generates the conceptual/method illustrations as fig2_execution, fig5_method,
and fig6_kernel_order.
Fig. 13 renders the BEV comparison from public coordinates without the camera panel.
fig10 and fig14 also use the public additions/quality records; see REPRODUCE.md.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
C = {"S0": "#ADB5BD", "FD": "#D08A32", "S1": "#2D80B3", "S2b": "#8064B1", "S2a": "#B9A6D6", "S3": "#27907A",
     "S4": "#176C4E", "K": "#6C757D"}
NAMES = {"S0": "#0: baseline autoregressive", "FD": "#0\u2032: FlashDrive BF16", "S1": "#1: #0 + previous-CoC verification",
         "S2b": "#2: #1 + FD-DFlash", "S2a": "#2\u2032: #1 + datastore", "S3": "#3: #1 + FD-DFlash + datastore",
         "S4": "#4: #3 + det. kernels (RESPECT Decode)", "K": "Deterministic-kernel autoregressive"}
SHORT = {"S0": "#0 baseline", "FD": "#0\u2032 FlashDrive BF16", "S4": "#4 RESPECT Decode"}
SETS = {"event": "event-window set (400 frames)", "general": "nominal-driving set (600 frames)"}


def signed(v: float) -> str:
    return f"{v:+.0f}%".replace("-", "\u2212")


def top_legend(fig, axs, handles, labels, ncol, y=1.0, fontsize=6.5):
    """Legend centered over the plotting area rather than over the whole figure."""
    fig.tight_layout()
    x = (axs[0].get_position().x0 + axs[-1].get_position().x1) / 2
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(x, y), ncol=ncol, frameon=False,
               fontsize=fontsize)


def save(fig, out: Path, name: str, *, tight: bool = True) -> None:
    if tight and not fig.legends:
        fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_fig3_concentration(s, out):
    """Fig. 3: (a) consecutive frame pairs whose CoC repeats the previous one as a whole, is equal from the first token
    and then differs, or differs from the first token; (b) event-window set: frames completed by initial verification
    and their share of Decode time. Vertical stacked columns."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.6, 2.8), gridspec_kw=dict(wspace=0.55))

    def labels_summing_to_100(values):
        # Largest-remainder allocation at 0.1 percentage-point precision.
        # Preserve the raw bar heights; resolve equal remainders in segment order.
        # Thus 313/400 and 87/400 display as 78.3 and 21.7, summing to 100.0.
        scaled = [Decimal(str(value)) * 10 for value in values]
        units = [int(value) for value in scaled]
        order = sorted(range(len(values)), key=lambda i: (-(scaled[i] - units[i]), i))
        for i in order[:1000 - sum(units)]:
            units[i] += 1
        return [f"{value / 10:.1f}" for value in units]

    def cols(ax, rows, colors, names):
        for x, (lab, v) in enumerate(rows):
            bottom = 0
            for h, col, label in zip(v, colors, labels_summing_to_100(v)):
                ax.bar(x, h, bottom=bottom, color=col, width=0.62, edgecolor="white", linewidth=1.0)
                big = h >= 12
                ax.text(x + (0 if big else 0.35), bottom + h / 2, label, ha="center" if big else "left",
                        va="center", fontsize=6 if big else 5.5, color="white" if big else "#5b6770")
                bottom += h
        ax.set(xticks=range(len(rows)), xlim=(-0.55, len(rows) - 0.45), ylim=(0, 100))
        ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
        ax.tick_params(axis="x", length=0)
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], names, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=5.8, handlelength=0.9)
    adj = s["adjacent_coc"]
    rows = []
    for k, name in (("event", "Event\nwindow"), ("general", "Nominal\ndriving")):
        x = adj[k]
        full, none = x["identical"], x["first_token_differs"]
        rows.append((f"{name}\n{x['pairs']:,} pairs",
                     [100 * q / x["pairs"] for q in (full, x["pairs"] - full - none, none)]))
    cols(a, rows, ("#245a8a", "#8fb5d3", "#c5cbd1"),
         ["Full match", "Partial match from the first token", "Differs at the first token"])
    a.tick_params(axis="x", labelsize=5.8)
    a.set_ylabel("Share (%)")
    a.set_title("(a) Match with previous CoC", y=-0.3, fontsize=7)
    c = s["sets"]["event"]["concentration"]
    nf, tt = c["done_frames"] + c["tail_frames"], c["done_ms_sum"] + c["tail_ms_sum"]
    cols(b, [("Frames", [100 * c["done_frames"] / nf, 100 * c["tail_frames"] / nf]),
             ("Decode time", [100 * c["done_ms_sum"] / tt, 100 * c["tail_ms_sum"] / tt])],
         ("#245a8a", "#b87729"), ["Initial verification only", "Continuation needed"])
    b.set_title("(b) Event-window frames", y=-0.3, fontsize=7)
    save(fig, out, "fig3_concentration")


def draw_supp_n_select(s, out):
    """Original Fig. 4: N selection, including the 1% equivalence band."""
    with plt.rc_context({"font.size": 7.0}):
        rows = s["dev"]["nselect"]
        NS = [r["n"] for r in rows]
        CHOSEN = s["dev"]["chosen"]["N"]
        done = [r["initial_complete_pct"] for r in rows]
        mean, p99, mx = [[r[k] for r in rows] for k in ("mean_ms", "p99_ms", "max_ms")]
        rel = lambda ys: [100 * (y / ys[0] - 1) for y in ys]
        INK, MUTED, GRID, ACC, BAND = "#1E262C", "#5B6770", "#E6E9EC", "#1F5F8B", "#DCE8F2"
        fig, axs = plt.subplots(2, 1, figsize=(3.4, 2.45), sharex=True, gridspec_kw=dict(hspace=0.42, left=0.15, right=0.86, top=0.985, bottom=0.215, height_ratios=[0.8, 1.2]))
        i = NS.index(CHOSEN)
        tie = [n for n, m in zip(NS, mean) if m <= min(mean) * 1.01]
        a = axs[0]
        a.plot(NS, done, '-', color=ACC, lw=1.4, marker='o', ms=3.0, zorder=3)
        a.plot([CHOSEN], [done[i]], 'o', ms=6.4, mfc=ACC, mec='white', mew=1.1, zorder=5)
        a.text(CHOSEN * 1.07, done[i] - 0.6, f'{done[i]:.1f}%', color=ACC, fontsize=6.0, ha='left', va='top')
        a.set_ylabel('Completed (%)', labelpad=3)
        a.set_ylim(80, 91)
        b = axs[1]
        b.axhline(0, color=MUTED, lw=0.6, zorder=1)
        for ys, c, lab in ((rel(mx), '#B8C2CA', 'Max'), (rel(p99), '#B0303A', 'p99'), (rel(mean), ACC, 'Mean')):
            b.plot(NS, ys, '-', color=c, lw=1.4, marker='o', ms=3.0, zorder=3)
            b.plot([CHOSEN], [ys[i]], 'o', ms=6.4, mfc=c, mec='white', mew=1.1, zorder=5)
            xo, yo, ha, va = {'Mean': (1.08, 0.9, 'left', 'bottom'), 'p99': (1.0, 1.0, 'center', 'bottom'), 'Max': (0.93, -1.0, 'right', 'top')}[lab]
            if lab != 'Max':
                b.text(CHOSEN * xo, ys[i] + yo, f'{ys[i]:+.1f}%'.replace('-', '−'), color=c, fontsize=6.0, ha=ha, va=va, zorder=6)
            b.text(NS[-1] * 1.08, ys[-1], lab, color=c if lab != 'Max' else MUTED, va='center', ha='left', fontsize=7)
        b.set_ylabel('Decode time change\nfrom $N=1$ (%)', labelpad=3, linespacing=1.1)
        b.text(1.06, 9.5, '↑ longer', ha='left', va='center', fontsize=6.0, color=MUTED)
        b.text(1.06, -5.7, '↓ shorter', ha='left', va='center', fontsize=6.0, color=MUTED)
        for a_ in axs:
            a_.grid(axis='y', color=GRID, lw=0.5)
            a_.set_axisbelow(True)
            a_.axvspan(min(tie) / 1.12, max(tie) * 1.12, color=BAND, lw=0, zorder=0)
        axs[0].text(0.5, -0.13, '(a) Frames completed by initial verification', transform=axs[0].transAxes, ha='center', va='top', fontsize=7.4, color=INK)
        b.set_xscale('log', base=2)
        b.set_xticks([1, 2, 4, 8, 16, 24], ['1', '2', '4', '8', '16', '24'])
        b.minorticks_off()
        b.set_xlabel('Number of previous frames $N$', labelpad=1, color=MUTED)
        b.text(0.5, -0.37, '(b) Change in Decode time', transform=b.transAxes, ha='center', va='top', fontsize=7.4, color=INK)
        save(fig, out, "supp_n_select", tight=False)


def draw_supp_tile_boundary(s, out, root):
    """Original Fig. 5: median and range across the four measured frames."""
    with plt.rc_context({"font.size": 7.0}):
        from matplotlib.font_manager import FontProperties
        KR = FontProperties(family="DejaVu Sans")
        tb = s["dev"]["tile_boundary"]
        raw = json.loads((root / "dev" / "tile_boundary.json").read_text())
        NS = tb["nodes"]
        M = np.array([[r["median_ms"] for r in frame["rows"]] for frame in raw])
        med, lo, hi = np.median(M, 0), M.min(0), M.max(0)
        if not np.allclose(med, tb["median_of_frames_ms"]):
            raise ValueError("tile medians disagree with summary")
        C = '#426685'
        f, ax = plt.subplots(figsize=(3.45, 2.3))
        ax.fill_between(NS, lo, hi, color=C, alpha=0.15, lw=0)
        ax.plot(NS, med, 'o-', color=C, ms=2.6, lw=1)
        for z in (64.5, 128.5):
            ax.axvline(z, color='#aaa', lw=0.7, ls=':')
        for z, lab in ((64.5, 'query tiles 1→2'), (128.5, '2→3')):
            ax.text(z, 121, lab, ha='center', va='bottom', fontsize=6.2, color='#777', fontproperties=KR)
        i64, i65 = (NS.index(64), NS.index(65))
        R = '#C0392B'
        ax.plot([64, 65], [med[i64], med[i65]], ls='none', marker='o', ms=6, mfc='none', mec=R, mew=1.1, zorder=5)
        ax.plot([67.5, 67.5], [med[i64], med[i65]], color=R, lw=1.0)
        for y in (med[i64], med[i65]):
            ax.plot([66.3, 67.5], [y, y], color=R, lw=1.0)
        ax.text(70, (med[i64] + med[i65]) / 2, f'64→65 node\n+{med[i65] - med[i64]:.1f} ms', ha='left', va='center', fontsize=6.6, color=R, fontproperties=KR, linespacing=1.1)
        ax.set(xticks=[8, 64, 128, 160], ylim=(70, 122))
        ax.set_xlabel('Nodes in the continuation trie', fontproperties=KR, fontsize=7)
        ax.set_ylabel('Median forward time (ms)', fontproperties=KR, fontsize=7)
        ax.grid(alpha=0.12)
        f.tight_layout(pad=0.3)
        save(f, out, "supp_tile_boundary", tight=False)


def draw_fig7_configurations(s, out):
    """All configurations, one row each, stage groups in a shared centre column: (a) event-window set to the
    left, (b) nominal-driving set to the right, time growing to the right on a log axis. Each row marks the
    mean (circle), p99 and observed maximum (bars) with their values; the shaded columns give target runs
    per frame; the dashed line is #0's observed maximum (values above it in red)."""
    from matplotlib.lines import Line2D
    groups = [("Baselines", [("#0", "S0", "Baseline autoregressive", "#ADB5BD"),
                             ("#0′", "FD", "FlashDrive BF16", "#D08A32")]),
              ("Initial verification", [("#1", "S1", "#0 + previous-$N$-frame CoCs", "#2D80B3")]),
              ("Continuation verification", [("#2", "S2b", None, "#8064B1"), ("#2′", "S2a", None, "#B9A6D6"),
                                          ("#3", "S3", None, "#27907A")]),
              ("Deterministic kernels", [("", "K", "Deterministic-kernel autoregressive", "#6B7780"),
                                     ("#4", "S4", "#3 + det. kernels (RESPECT Decode)", "#176C4E")])]
    src = {"S2b": (1, 0), "S2a": (0, 1), "S3": (1, 1)}
    ink, muted, red = "#1E262C", "#5B6770", "#B0303A"
    bar = {"p99": dict(ms=7.5, mew=1.3), "max": dict(ms=10.5, mew=2.2)}
    rows, heads, y = [], [], 0.0
    for g, members in groups:
        heads.append((g, y)); y += 0.85
        for m in members:
            rows.append((m, y)); y += 1.0
        y += 0.25
    ymax = y

    def stats(k, code):
        v = s["sets"][k]
        d = v["flashdrive"] if code == "FD" else v["configs"][code]
        L = d["latency_ms"]
        return L["mean"], L["p99"], L["max"], d["target_runs_mean"]

    fig = plt.figure(figsize=(7.6, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.15, 3.6, 2.15], wspace=0.04, left=0.01, right=0.99, top=0.90,
                          bottom=0.14)
    axl, axc, axr = (fig.add_subplot(gs[i]) for i in range(3))
    for ax, k, title in ((axl, "event", "(a) Event-window set (400 frames)"), (axr, "general", "(b) Nominal-driving set (600 frames)")):
        ax.set_xscale("log"); ax.set_xlim(60, 13000); ax.set_ylim(ymax, -0.75)
        for sp in ("top", "left", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_yticks([]); ax.minorticks_off()
        ax.set_xticks([100, 300, 1000, 3000], ["100", "300", "1,000", "3,000"])
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="x", color="#EEF0F2", lw=0.6); ax.set_axisbelow(True)
        base = stats(k, "S0")[2]
        ax.axvline(base, color=red, lw=0.8, ls=(0, (4, 2)), zorder=1)
        for _, hy in heads:
            ax.axhline(hy - 0.12, color="#D5DADE", lw=0.6, zorder=0)
        for (rid, code, _, col), ry in rows:
            mean, p99, vmax, _ = stats(k, code)
            ax.plot([mean, vmax], [ry, ry], color=col, lw=3.0, alpha=0.35, solid_capstyle="round", zorder=2)
            ax.plot(mean, ry, marker="o", ms=5.0, mfc=col, mec="#2A333A", mew=0.6, ls="None", zorder=4)
            for v, nm in ((p99, "p99"), (vmax, "max")):
                ax.plot(v, ry, marker="|", color="#2A333A", ls="None", zorder=5, **bar[nm])
            ax.text(mean * 0.86, ry, f"{mean:,.0f}", ha="right", va="center", fontsize=5.9, color=ink)
            wide = vmax / p99 > 1.35
            ax.text(p99 * (1.0 if wide else 1.02), ry - 0.30, f"{p99:,.0f}", ha="center" if wide else "right",
                    va="bottom", fontsize=5.4, color=muted)
            over = code not in ("S0", "K") and vmax > base
            ax.text(vmax * 1.16, ry, f"{vmax:,.0f}", ha="left", va="center", fontsize=5.9, zorder=6,
                    color=red if over else ink, bbox=dict(fc="white", ec="none", pad=0.4))
            if over:
                ax.text(vmax * 1.16, ry + 0.36, f"(+{100 * (vmax / base - 1):.1f}%)", ha="left", va="center",
                        fontsize=5.2, zorder=6, color=red)
        ax.set_xlabel("Decode time (ms, log)", fontsize=7.0, labelpad=2)
        ax.text(0.5, -0.20, title, transform=ax.transAxes, ha="center", va="top", fontsize=8.2)
    axc.set_xlim(0, 1); axc.set_ylim(ymax, -0.75); axc.axis("off")
    rc, nc = 0.11, 0.5
    runx = (0.0, 1 - rc)
    for x0 in runx:
        axc.axvspan(x0, x0 + rc, color="#F1F3F5", lw=0, zorder=0)
        axc.text(x0 + rc / 2, -0.3, "Target runs\nper frame", ha="center", va="center", fontsize=6.0,
                 fontweight="bold", color=ink, linespacing=1.05)
    for g, hy in heads:
        axc.text(nc, hy + 0.1, g, ha="center", va="center", fontsize=7.2, fontweight="bold", color=ink)
        axc.plot([0.0, 1.0], [hy - 0.12, hy - 0.12], color="#D5DADE", lw=0.6)
    name_kw = dict(fontsize=6.4, color=ink, ha="left", va="center")
    fig.canvas.draw()
    inv = axc.transData.inverted()

    def edges(t):
        (x0, _), (x1, _) = inv.transform(t.get_window_extent())
        return x0, x1

    def width(txt):
        t = axc.text(0, 0, txt, **name_kw); w = edges(t)[1] - edges(t)[0]; t.remove(); return w
    terms = ("+ FD-DFlash", "+ datastore")
    xt = [0.10 + width("#1 ")]
    xt.append(xt[0] + width(terms[0] + " "))
    arts = []
    for (rid, code, name, col), ry in rows:
        for x0, k in zip(runx, ("event", "general")):
            axc.text(x0 + rc / 2, ry, f"{stats(k, code)[3]:.2f}", ha="center", va="center", fontsize=6.8, color=ink)
        mk, = axc.plot(0.0, ry, marker="s", ms=5.5, color=col, ls="None")
        a = [axc.text(0.025, ry, rid, ha="left", va="center", fontsize=7.0, color=ink, fontweight="bold")]
        if code in src:
            a.append(axc.text(0.10, ry, "#1", **name_kw))
            a += [axc.text(x, ry, t, **{**name_kw, "color": ink if use else "#CDD2D6"})
                  for use, x, t in zip(src[code], xt, terms)]
        else:
            a.append(axc.text(0.025 if not rid else 0.10, ry, name, **name_kw))
        arts.append((mk, a))
    fig.canvas.draw()
    right = max(edges(t)[1] for _, a in arts for t in a)
    shift = nc - (-0.012 + right) / 2
    for mk, a in arts:
        mk.set_xdata([mk.get_xdata()[0] + shift])
        for t in a:
            t.set_x(t.get_position()[0] + shift)
    hs = [Line2D([], [], color="#2A333A", marker="o", mfc="white", ls="None", ms=5, label="Mean")]
    hs += [Line2D([], [], color="#2A333A", marker="|", ls="None", label=n, **bar[k])
           for k, n in (("p99", "p99"), ("max", "Observed max"))]
    hs.append(Line2D([], [], color=red, lw=0.8, ls=(0, (4, 2)), label="Observed max of #0"))
    fig.legend(handles=hs, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4,
               handlelength=1.4, fontsize=6.9)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig7_configurations.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def draw_fig8_sources(root, out):
    """Mean and observed maximum on frames needing continuation (Fig. 8).

    The oracle takes the faster single-source time per frame before aggregation."""
    # Layout adapted from the paper figure renderer; values come from public results.
    with plt.rc_context({"font.size": 7.0, "axes.linewidth": 0.6, "xtick.major.size": 0, "ytick.major.width": 0.6, "ytick.major.size": 2.5, "hatch.linewidth": 0.7, "axes.labelcolor": "#1E262C", "xtick.color": "#1E262C", "ytick.color": "#5B6770"}):
        INK, MUTED, GRID, EDGE = "#1E262C", "#5B6770", "#E6E9EC", "#5B6770"
        BARS = [("S1fK", "#4 w/o both sources", "#C9CED3", "//////"), ("S2bK", "#4 w/o datastore", "#8064B1", None),
                ("S2aK", "#4 w/o FD-DFlash", "#B9A6D6", None), ("best", "Faster per frame (oracle)", "white", "//////"),
                ("S4", "#4 (both sources)", "#176C4E", None)]
        SETS = [("event", "Event window"), ("general", "Nominal driving")]

        vals = {}
        for s, _ in SETS:
            rows = [json.loads(x) for x in (root / "main" / f"frames_{s}.jsonl").read_text().splitlines()]
            t = [r for r in rows if r["configs"]["S1fK"]["fwd"]["tail_tokens"] != 0]
            g = {c: np.array([r["configs"][c]["ms_mean2"] for r in t]) for c in ("S1fK", "S2bK", "S2aK", "S4")}
            g["best"] = np.minimum(g["S2bK"], g["S2aK"])
            vals[s] = {c: (float(v.mean()), float(v.max())) for c, v in g.items()}

        fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.9), gridspec_kw=dict(hspace=0.42, left=0.15, right=0.99, top=0.86, bottom=0.08))
        W = 0.165
        for ax, k, ylim, step, tag in zip(axs, (0, 1), (1500, 4400), (500, 1000), ("(a) Mean", "(b) Observed maximum")):
            sp, gap = ylim * 0.075, ylim * 0.03
            for gi, (s, name) in enumerate(SETS):
                xs = [gi + (bi - 2) * W for bi in range(len(BARS))]
                vs = [vals[s][c][k] for c, *_ in BARS]
                for x, v, (c, _, col, hatch) in zip(xs, vs, BARS):
                    ax.bar(x, v, width=W * 0.9, color=col, ec=EDGE if hatch else "none", lw=0.6 if hatch else 0, hatch=hatch, zorder=3)
                # label heights from right to left: each above its own bar and above every label to its right
                ys = [0.0] * len(vs)
                for bi in range(len(vs) - 1, -1, -1):
                    y = vs[bi] + gap
                    if bi < len(vs) - 1:
                        y = max(y, ys[bi + 1] + sp, max(vs[bi + 1:]) + gap)
                    ys[bi] = y
                for bi, (x, v, y) in enumerate(zip(xs, vs, ys)):
                    c = BARS[bi][0]
                    color = INK if c == "S4" else MUTED
                    wt = "bold" if c == "S4" else "normal"
                    if bi == 0:
                        ax.text(x, v + gap * 0.5, f"{v:,.0f}", ha="center", va="bottom", fontsize=5.6, color=color, zorder=5)
                        continue
                    ax.plot([x, x, x + W * 0.45], [v, y, y], color=EDGE, lw=0.6, zorder=4)
                    ax.annotate("", xy=(x + W * 0.62, y), xytext=(x + W * 0.40, y), zorder=4,
                                arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.14", color=EDGE, lw=0.6, shrinkA=0, shrinkB=0))
                    ax.text(x + W * 0.66, y, f"{v:,.0f}", ha="left", va="center", fontsize=5.6, color=color, fontweight=wt, zorder=5)
            ax.set_xticks([0, 1], [n for _, n in SETS], fontsize=7.0)
            ax.set_xlim(-0.55, 1.62); ax.set_ylim(0, ylim)
            ax.set_yticks(range(0, ylim + 1 - (ylim % step), step), [f"{v:,}" for v in range(0, ylim + 1 - (ylim % step), step)])
            ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
            ax.set_ylabel("Decode time (ms)", labelpad=2)
            ax.text(0.5, -0.13, tag, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color=INK)
        h = [plt.Rectangle((0, 0), 1, 1, fc=col, ec=EDGE if hatch else "none", hatch=hatch, lw=0.6) for _, _, col, hatch in BARS]
        fig.legend(h, [n for _, n, _, _ in BARS], loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, frameon=False,
                   fontsize=6.0, handlelength=1.2, handleheight=0.8, columnspacing=0.8, handletextpad=0.4)
        save(fig, out, "fig8_sources")


def draw_supp_kernel_cost(s, out):
    """Per-call absolute times and changes with deterministic kernels (supplementary kernel cost)."""
    # Layout adapted from the paper figure renderer; values come from public results.
    with plt.rc_context({"font.size": 7.0, "axes.linewidth": 0.6, "xtick.major.size": 0, "ytick.major.width": 0.6, "ytick.major.size": 2.5, "hatch.linewidth": 0.7, "axes.labelcolor": "#1E262C", "xtick.color": "#1E262C", "ytick.color": "#5B6770"}):
        values = {k: v["kernel_percall_ms"] for k, v in s["sets"].items()}
        DEFAULT, DEDICATED, UP, DOWN, INK, MUTED = "#98A3AE", "#187B62", "#B43F35", "#1D669A", "#1E262C", "#5B6770"
        OPS = ["AR\nstep", "Initial\nverif.", "Contin.\nverif."]
        W = 0.42

        fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.25), sharey=True,
                                 gridspec_kw=dict(wspace=0.1, left=0.13, right=0.99, top=0.84, bottom=0.22))
        for ax, key, tag in zip(axes, ("event", "general"), ("(a) Event window", "(b) Nominal driving")):
            for i, (before, after) in enumerate(values[key][op] for op in ("sequential", "initial", "follow_up")):
                change = 100 * (after / before - 1)
                ax.bar(i - W / 2, before, width=W * 0.95, color=DEFAULT, lw=0, zorder=3)
                ax.bar(i + W / 2, after, width=W * 0.95, color=DEDICATED, lw=0, zorder=3)
                ax.text(i - W / 2, before - 3, f"{before:.1f}", ha="center", va="top", fontsize=5.2, color=INK)
                ax.text(i + W / 2, after - 3, f"{after:.1f}", ha="center", va="top", fontsize=5.2, color="white")
                col = UP if change > 0 else DOWN
                top = max(before, after) + 11
                xb, xa = i - W / 2, i + W / 2
                ax.plot([xb, xb, xa], [before + 2, top, top], color=col, lw=0.8, zorder=4, solid_capstyle="butt")
                ax.annotate("", xy=(xa, after + 1.5), xytext=(xa, top), zorder=4,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=col, lw=0.8,
                                            shrinkA=0, shrinkB=0))
                ax.text(i, top + 1.5, f"{change:+.0f}%".replace("-", "−"), ha="center", va="bottom",
                        fontsize=6.4, fontweight="bold", color=col)
            ax.set_xticks(range(3), OPS, fontsize=6.6)
            ax.set_xlim(-0.6, 2.6)
            ax.set_ylim(0, 160)
            ax.grid(axis="y", color="#E6E9EC", lw=0.5); ax.set_axisbelow(True)
            ax.text(0.5, -0.2, tag, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color=INK)
        axes[0].set_ylabel("Time per call (ms)", labelpad=2)
        axes[1].tick_params(axis="y", length=0)
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (DEFAULT, DEDICATED)]
        fig.legend(h, ["Default PyTorch kernels", "Deterministic kernels"], loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=2,
                   frameon=False, fontsize=6.6, handlelength=1.0, handleheight=0.8, columnspacing=1.2)
        save(fig, out, "supp_kernel_cost")


def draw_fig10_delay(s, add, out):
    """Section V-D: Decode-time reduction of #4 against deterministic-kernel autoregressive decoding under delay d
    (stride mode), mean (solid) and p99 (dashed). The shaded bands are the 10th-90th percentile delays of the
    real-time runs; the labels give the mean (above) and p99 (below) reduction at their medians, d = 16 and 23.
    The nominal-driving set at d = 16, 20, 23 uses the added earlier frames (delay_observed/); d = 40 of the
    nominal-driving set covers the 578 frames with a previous frame at t - 40."""
    ds = [1, 2, 3, 5, 10, 16, 20, 23, 40]
    col = {"event": "#B0303A", "general": "#1F5F8B"}
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    bands = add["realtime_delay"]
    for (name, c, xt, ha) in (("RESPECT run", "#DCEFE6", None, "right"), ("#4 run", "#E3E8EE", None, "left")):
        b = bands[name]
        ax.axvspan(b["p10"], b["p90"], color=c, lw=0, zorder=0)
        ax.text(b["p90"] + 0.3 if ha == "right" else b["p10"] - 0.3, 101, name, ha=ha, va="bottom", fontsize=5.8,
                color="#5B6770", clip_on=False)
    for k in ("event", "general"):
        marker = "o" if k == "event" else "s"
        v = s["sets"][k]
        kk = v["configs"]["K"]["latency_ms"]
        ext = s["section6"]["delay_extension"][k]["rows"]
        mean, p99 = [], []
        for d in ds:
            q = f"{d}_stride"
            if f"{k}_{q}" in add["observed_delay"] and (d in (16, 23) or k == "general"):
                r = add["observed_delay"][f"{k}_{q}"]; m, p = r["mean_ms"], r["p99_ms"]
            elif d <= 20:
                r = v["delay"]["1" if d == 1 else q]["latency_ms"]; m, p = r["mean"], r["p99"]
            else:
                r = ext[q]; m, p = r["mean_ms"], r["p99_ms"]
            mean.append(100 - 100 * m / kk["mean"]); p99.append(100 - 100 * p / kk["p99"])
        ax.plot(ds, mean, "-", color=col[k], lw=1.6, marker=marker, ms=3.4, zorder=4)
        ax.plot(ds, p99, ls=(0, (3, 1.6)), color=col[k], lw=1.3, marker=marker, ms=3.4, mfc="white", mew=1.0, zorder=3)
        for d in (16, 23):
            i = ds.index(d)
            ax.text(d, mean[i] + 3.5, f"{mean[i]:.0f}%", color=col[k], fontsize=6.0, ha="center", va="bottom", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
            ax.text(d, p99[i] - 3.5, f"{p99[i]:.0f}%", color=col[k], fontsize=6.0, ha="center", va="top", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
    ax.set_xscale("log"); ax.set_xlim(0.9, 45)
    ax.set_xticks([1, 2, 3, 5, 10, 16, 23, 40], ["1", "2", "3", "5", "10", "16", "23", "40"]); ax.minorticks_off()
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Decode-time reduction vs. det.-\nkernel autoregressive (%)", fontsize=7)
    ax.set_xlabel("Delay $d$ (frames)", fontsize=7)
    ax.grid(axis="y", color="#E6E9EC", lw=0.5); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=col["event"], lw=1.6, marker="o", ms=3.4),
         plt.Line2D([], [], color=col["general"], lw=1.6, marker="s", ms=3.4),
         plt.Line2D([], [], color="#5B6770", lw=1.6),
         plt.Line2D([], [], color="#5B6770", lw=1.3, ls=(0, (3, 1.6)))]
    ax.legend(h, ["Event window", "Nominal driving", "Mean", "p99"], loc="lower left", ncol=2, frameon=False,
              fontsize=6.2, handlelength=2.0, columnspacing=1.0)
    save(fig, out, "fig10_delay")


def draw_fig4_realtime_timeline(root, out):
    """Section III-C, Fig. 4: one clip and start position of the real-time runs (clip 61d3622b, start phase 24): the inputs
    arriving every 0.1 s and every processing cycle of the #0 and #4 runs as input preparation + Prefill, Decode and
    Flow; the arrow is the event response latency (arrival of frame E to the first trajectory from a frame at or
    after E)."""
    clip, phase = "61d3622b", 24
    x0, x1, show_until = -3.85, 6.1, 5.5
    ink, muted, light, c_pf, c_fl, c_ev = "#1E262C", "#5B6770", "#C9CFD4", "#8C969E", "#D5DADE", "#B0303A"
    c_dec = {"S0": "#6B7780", "S4": "#176C4E"}
    rows = [json.loads(line) for line in (root / "realtime" / "cycles.jsonl").read_text().splitlines() if clip in line]
    runs = {c: sorted([r for r in rows if r["phase"] == phase and r["cfg"] == c], key=lambda r: r["g"]) for c in ("S0", "S4")}
    e, g0 = runs["S0"][0]["e"], runs["S0"][0]["g0"]
    te = (e - g0) * 0.1
    first = {c: next(r for r in rr if r["g"] >= e) for c, rr in runs.items()}
    runs = {c: [r for r in rr if r["t_start"] - te <= show_until] for c, rr in runs.items()}
    y = {"in": 2.25, "S0": 1.15, "S4": 0.0}
    hgt = 0.34
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    fig.subplots_adjust(left=0.20, right=0.99, top=0.80, bottom=0.17)
    processed = {r["g"] for c in runs for r in runs[c]}
    for g in range(g0, g0 + int((x1 - (g0 - e) * 0.1) / 0.1) + 1):
        x = (g - g0) * 0.1 - te
        if x > x1:
            break
        hit = g in processed
        ax.plot([x, x], [y["in"] - 0.13, y["in"] + 0.13], color=ink if hit else light, lw=0.8 if hit else 0.5,
                solid_capstyle="butt", zorder=2)
    for c, rr in runs.items():
        for r in rr:
            st = r["t_start"] - te
            if st > x1:
                break
            for a, b, cl in ((r["t_start"], r["t_prefill"], c_pf), (r["t_prefill"], r["t_decode"], c_dec[c]),
                             (r["t_decode"], r["t_pub"], c_fl)):
                ax.barh(y[c], b - a, left=a - te, height=hgt, color=cl, edgecolor="white", linewidth=0.5, zorder=3)
            k = r["g"] - e
            ax.text(st + 0.06, y[c] + hgt / 2 + 0.06, "E" if k == 0 else f"E{k:+d}".replace("-", "\u2212"), ha="left",
                    va="bottom", fontsize=5.6, color=ink)
        t_pub = first[c]["t_pub"] - te
        ya = y[c] - hgt / 2 - 0.16
        ax.annotate("", xy=(t_pub, ya), xytext=(0, ya), arrowprops=dict(arrowstyle="<->", color=c_ev, lw=0.7,
                                                                        shrinkA=0, shrinkB=0))
        ax.text(t_pub / 2, ya - 0.05, f"Event response latency {t_pub:.2f} s", ha="center", va="top", fontsize=5.6,
                color=c_ev)
    ax.plot([0, 0], [-0.75, y["in"] + 0.42], color=c_ev, lw=0.7, ls=(0, (3, 2)), zorder=4, clip_on=False)
    ax.text(0, y["in"] + 0.46, "Event-start frame E arrives", ha="center", va="bottom", fontsize=5.8, color=c_ev)
    ax.text(x0 + 0.02, y["in"] + 0.18, "Inputs arriving every 0.1 s", ha="left", va="bottom", fontsize=5.8, color=muted)
    ax.set_yticks([y["in"], y["S0"], y["S4"]], ["Sensor input", "#0 run", "#4 run"], fontsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(x0, x1); ax.set_ylim(-0.75, 2.95)
    ax.set_xlabel("Time from the arrival of event-start frame E (s)", fontsize=6.6, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=6.6)
    hd = [plt.Rectangle((0, 0), 1, 1, color=c_pf), plt.Rectangle((0, 0), 1, 1, color=c_dec["S0"]),
          plt.Rectangle((0, 0), 1, 1, color=c_dec["S4"]), plt.Rectangle((0, 0), 1, 1, color=c_fl),
          plt.Line2D([], [], color=ink, lw=0.8), plt.Line2D([], [], color=light, lw=0.8)]
    fig.legend(hd, ["Input prep. + Prefill", "Decode (#0)", "Decode (#4)", "Flow", "Processed input", "Dropped input"],
               loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, frameon=False, fontsize=5.5,
               handlelength=0.9, handletextpad=0.3, columnspacing=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig4_realtime_timeline.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


class _Icons:
    """Car, truck and pedestrian icons drawn in inches so that they keep their shape on any axis scale."""

    def __init__(self, ax, w_in, h_in):
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        self.ax, self.sx, self.sy = ax, (x1 - x0) / w_in, (y1 - y0) / h_in

    def car(self, xf, y, color, ghost=False, length=0.30):
        from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon
        ax, sx, sy = self.ax, self.sx, self.sy
        L = length * sx; x0 = xf - L
        fc = "white" if ghost else color
        kw = dict(ec=color, lw=1.0, ls=(0, (2, 1.4)) if ghost else "-", zorder=7)
        ax.add_patch(FancyBboxPatch((x0, y - 0.035 * sy), L, 0.06 * sy, boxstyle="round,pad=0,rounding_size=0.012",
                                    mutation_aspect=sy / sx, fc=fc, **kw))
        ax.add_patch(Polygon([(x0 + 0.20 * L, y + 0.025 * sy), (x0 + 0.34 * L, y + 0.075 * sy),
                              (x0 + 0.70 * L, y + 0.075 * sy), (x0 + 0.84 * L, y + 0.025 * sy)], fc=fc, **kw))
        if not ghost:
            ax.add_patch(Polygon([(x0 + 0.29 * L, y + 0.03 * sy), (x0 + 0.38 * L, y + 0.063 * sy),
                                  (x0 + 0.66 * L, y + 0.063 * sy), (x0 + 0.76 * L, y + 0.03 * sy)],
                                 fc="white", ec="none", alpha=0.85, zorder=8))
        for wx in (0.22, 0.78):
            ax.add_patch(Ellipse((x0 + wx * L, y - 0.035 * sy), 0.06 * sx, 0.06 * sy,
                                 fc="white" if ghost else _INK, ec=color if ghost else _INK, lw=0.9, zorder=8,
                                 ls=(0, (2, 1.4)) if ghost else "-"))

    def faint(self, xf, y, color, length=0.30, alpha=0.28):
        from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon
        ax, sx, sy = self.ax, self.sx, self.sy
        L = length * sx; x0 = xf - L
        ax.add_patch(FancyBboxPatch((x0, y - 0.035 * sy), L, 0.06 * sy, boxstyle="round,pad=0,rounding_size=0.012",
                                    mutation_aspect=sy / sx, fc=color, ec="none", alpha=alpha, zorder=6))
        ax.add_patch(Polygon([(x0 + 0.20 * L, y + 0.025 * sy), (x0 + 0.34 * L, y + 0.075 * sy),
                              (x0 + 0.70 * L, y + 0.075 * sy), (x0 + 0.84 * L, y + 0.025 * sy)],
                             fc=color, ec="none", alpha=alpha, zorder=6))
        for wx in (0.22, 0.78):
            ax.add_patch(Ellipse((x0 + wx * L, y - 0.035 * sy), 0.06 * sx, 0.06 * sy, fc=_INK, ec="none",
                                 alpha=alpha * 0.8, zorder=6))

    def truck(self, xf, y, color, length=0.42):
        from matplotlib.patches import Ellipse, Polygon, Rectangle
        ax, sx, sy = self.ax, self.sx, self.sy
        L = length * sx; x0 = xf - L
        ax.add_patch(Rectangle((x0, y - 0.035 * sy), 0.72 * L, 0.13 * sy, fc=color, ec="none", zorder=7))
        ax.add_patch(Polygon([(x0 + 0.75 * L, y - 0.035 * sy), (x0 + 0.75 * L, y + 0.075 * sy),
                              (x0 + 0.90 * L, y + 0.075 * sy), (xf, y + 0.02 * sy), (xf, y - 0.035 * sy)],
                             fc=color, ec="none", zorder=7))
        for wx in (0.12, 0.26, 0.85):
            ax.add_patch(Ellipse((x0 + wx * L, y - 0.035 * sy), 0.06 * sx, 0.06 * sy, fc=_INK, ec="none", zorder=8))

    def person(self, x, y):
        from matplotlib.patches import Ellipse
        ax, sx, sy = self.ax, self.sx, self.sy
        ax.add_patch(Ellipse((x, y + 0.085 * sy), 0.05 * sx, 0.05 * sy, fc=_INK, ec="none", zorder=9))
        ax.plot([x, x], [y + 0.055 * sy, y - 0.02 * sy], color=_INK, lw=1.8, zorder=9, solid_capstyle="round")
        ax.plot([x - 0.035 * sx, x, x + 0.035 * sx], [y - 0.075 * sy, y - 0.02 * sy, y - 0.075 * sy], color=_INK, lw=1.4,
                zorder=9)
        ax.plot([x - 0.04 * sx, x + 0.04 * sx], [y + 0.03 * sy, y + 0.03 * sy], color=_INK, lw=1.3, zorder=9)

    def cross(self, x, y):
        for dx in (-1, 1):
            self.ax.plot([x - dx * 0.05 * self.sx, x + dx * 0.05 * self.sx], [y - 0.06 * self.sy, y + 0.06 * self.sy],
                         color="#C62828", lw=2.2, zorder=10, solid_capstyle="round")


_INK = "#1E262C"


def draw_fig12_stopping_margin(s, out):
    """Section V-F: the four events beyond the RSS allowed response time drawn as road lanes. In each lane the
    ego car keeps its speed until the new trajectory (solid, v_r x reaction latency), then brakes at 4 m/s²
    (hatched); the dashed outline is the stop position and the vertical line where the lead object ends up
    when it brakes at 8 m/s², so the overshoot equals the RSS shortfall v_r (rho - rho*). Reaction latency is
    the mean of the five start positions."""
    from matplotlib.patches import Rectangle
    s6 = s["section6"]
    rss, link = s6["rss"], s6["stage_link"]["rss"]
    muted, road, red, green = "#5B6770", "#EEF0F2", "#C62828", "#1F7A5A"
    cfg = [("S0", "#0 run (baseline autoregressive)", "#B0303A"), ("S4", "#4 run (RESPECT Decode)", "#1F5F8B"),
           ("FC8", "FlashDrive BF16 run", "#6B4E9B"), ("S4F", "RESPECT run (#5)", "#1F7A5A")]
    events = [("0b929729", "Construction zone", "Pedestrian", "person"), ("bbbe775f", "Pedestrian crowd", "Stopped car", "car"),
              ("b1968a87", "Unusual vehicle behavior", "Car", "car"), ("25c50d83", "Unusual vehicle behavior", "Truck", "truck")]
    full = {k[:8]: k for k in rss}

    def react(cid, c):
        return float(np.mean(rss[cid]["reaction_s"][c])) if c in ("S0", "S4") else float(link[cid]["reaction_s"][c])
    pw, ph = 0.485 * 7.1, 0.37 * 5.55

    def panel(ax, cid, title, obj, kind, tag):
        x = rss[cid]; v = x["v_r"]; gap = x["lead"]["gap_m"]; vf = max(x["lead"]["v_f"], 0.0)
        line = gap + vf * vf / 16.0; brake = v * v / 8.0; rs = x["rho_star"]["4.0"]
        rows = [(c, name, col, react(cid, c)) for c, name, col in cfg]
        xmax = max(v * r + brake for *_, r in rows) * 1.03
        xmin = -0.33 * xmax / (pw - 0.33)
        ax.set_xlim(xmin, xmax); ax.set_ylim(-0.45, 4.6); ax.axis("off")
        ic = _Icons(ax, pw, ph)
        ax.add_patch(Rectangle((xmin, -0.45), xmax - xmin, 4.15, fc=road, ec="none", zorder=0))
        ax.plot([line, line], [-0.45, 3.7], color=_INK, lw=1.1, ls=(0, (3, 2)), zorder=2)
        yi = 4.02
        if kind == "person":
            ic.person(line, yi); xt = line + 0.08 * ic.sx
        elif kind == "truck":
            ic.truck(line + 0.42 * ic.sx, yi, "#8A949C"); xt = line + 0.48 * ic.sx
        else:
            ic.car(line + 0.30 * ic.sx, yi, "#8A949C"); xt = line + 0.38 * ic.sx
        head = f"{obj} {gap:.0f} m ahead" + (f", {vf * 3.6:.0f} km/h" if vf > 0.5 else "")
        sub = f"Allowable response time {rs:.1f} s"
        ha = "left"
        if (line - xmin) / (xmax - xmin) > 0.42:
            xt, ha = line - (0.06 if kind == "person" else 0.05) * ic.sx, "right"
        ax.text(xt, yi + 0.13, head, fontsize=6.4, fontweight="bold", color=_INK, va="bottom", ha=ha)
        ax.text(xt, yi + 0.10, sub, fontsize=5.9, color=muted, va="top", ha=ha)
        for i, (c, name, col, r) in enumerate(rows):
            y = 2.95 - i * 0.95
            d = v * r; stop = d + brake
            ax.add_patch(Rectangle((0, y - 0.05), d, 0.10, fc=col, ec="none", zorder=3))
            ax.add_patch(Rectangle((d, y - 0.05), brake, 0.10, fc="white", ec=col, hatch="//////", lw=0, zorder=3))
            for t in range(1, int(r) + 1):
                ic.faint(v * t, y, col)
            ic.car(0, y, col)
            ic.car(stop, y, col, ghost=True)
            ax.text(xmin + 0.02 * ic.sx, y + 0.30, name, fontsize=6.2, fontweight="bold", color=col, ha="left",
                    va="bottom")
            margin = line - stop
            if margin < 0:
                ic.cross(line, y)
            mc = red if margin < 0 else green
            msg = f"Stopping margin {margin:+.0f} m".replace("-", "\u2212")
            ax.text(d, y - 0.13, f"{r:.1f} s", fontsize=5.8, color=col, ha="center", va="top", zorder=11)
            ax.text(xmax, y + 0.30, msg, fontsize=6.0, fontweight="bold", color=mc, ha="right", va="bottom", zorder=11,
                    bbox=dict(fc=road, ec="none", pad=0.6))
        ax.text(0.5, -0.03, f"({tag}) {title}, ego {v * 3.6:.0f} km/h", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.6, fontweight="bold", color=_INK)

    with plt.rc_context({"hatch.linewidth": 0.8}):
        fig = plt.figure(figsize=(7.1, 5.55))
        lg = fig.add_axes([0.0, 0.915, 1.0, 0.08]); lg.axis("off"); lg.set_xlim(0, 100); lg.set_ylim(0, 1)
        li = _Icons(lg, 7.1, 0.08 * 5.55)
        y1, y2 = 0.74, 0.22
        li.car(5.0, y1 - 0.02, muted)
        lg.text(5.8, y1, "Position at event start", fontsize=6.6, va="center")
        lg.add_patch(Rectangle((24, y1 - 0.06), 5, 0.12, fc=muted, ec="none"))
        lg.text(29.6, y1, "Event-start speed during the event response latency (value at the line end)", fontsize=6.6, va="center")
        li.car(84.0, y1 - 0.02, muted, ghost=True)
        lg.text(84.8, y1, "Stop position", fontsize=6.6, va="center")
        lg.plot([1, 1], [y2 - 0.17, y2 + 0.17], color=_INK, lw=1.1, ls=(0, (2, 1.5)))
        lg.text(2.0, y2, "Lead object (if moving, its stop point at 8 m/s²)", fontsize=6.6, va="center")
        lg.add_patch(Rectangle((38.5, y2 - 0.06), 5, 0.12, fc="white", ec=muted, hatch="//////", lw=0.6))
        lg.text(44.1, y2, "Braking at 4 m/s²", fontsize=6.6, va="center")
        li.cross(58.5, y2)
        lg.text(59.8, y2, "Stopping margin < 0", fontsize=6.6, va="center")
        li.faint(78.0, y2 - 0.02, muted)
        lg.text(78.8, y2, "Position every 1 s", fontsize=6.6, va="center")
        pos = [(0.0, 0.50), (0.51, 0.50), (0.0, 0.035), (0.51, 0.035)]
        for (cid, title, obj, kind), (px, py), tag in zip(events, pos, "abcd"):
            ax = fig.add_axes([px + 0.005, py + 0.045, 0.485, 0.37])
            panel(ax, full[cid], title, obj, kind, tag)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig12_stopping_margin.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def draw_fig9_event_timeline(s, out):
    """Original Fig. 10: five-frame windows and event-onset reduction arrows."""
    with plt.rc_context({"font.size": 7.0}):
        from matplotlib.font_manager import FontProperties
        from matplotlib.lines import Line2D
        KR = FontProperties(family="DejaVu Sans")
        tl = s["sets"]["event"]["event_timeline"]
        bins = [(r["lo"], r["hi"]) for r in tl]
        M = {k+"_ms": np.array([r[k+"_mean"] for r in tl]) for k in ("seq", "fd", "res")}
        X = {k+"_ms": np.array([r[k+"_max"] for r in tl]) for k in ("seq", "fd", "res")}
        MUTED, GRID, C_SEQ, C_FD, C_RES = ('#6b6b66', '#e8e8e5', '#85909b', '#b8620c', '#118a5a')
        kw = dict(mec='white', mew=0.6)
        xs = np.arange(len(bins))
        sg = lambda z: ('0' if z == 0 else f'{z:+d}').replace('-', '−')
        lab = [f'{sg(a)}\n~{sg(b)}' for a, b in bins]
        kr = KR
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 4.3), sharex=True)
        EV = 4
        for ax, D, title, ylim, yt in ((ax1, M, '(a) Window mean', (0, 1700), [0, 400, 800, 1200, 1600]), (ax2, X, '(b) Window max', (0, 4300), [0, 1000, 2000, 3000, 4000])):
            ax.axvspan(EV - 0.5, EV + 0.5, color='#f7e6eb', lw=0, zorder=0)
            ax.grid(axis='y', color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.plot(xs, D['seq_ms'], color=C_SEQ, lw=1.6, marker='o', ms=3.6, zorder=3, **kw)
            ax.plot(xs, D['fd_ms'], color=C_FD, lw=1.5, marker='^', ms=4.0, zorder=3, **kw)
            ax.plot(xs, D['res_ms'], color=C_RES, lw=2.0, marker='s', ms=3.6, zorder=4, **kw)
            ax.set_ylim(*ylim)
            ax.set_yticks(yt)
            ax.set_ylabel('Decode time (ms)', fontproperties=kr, fontsize=6.8)
            ax.tick_params(labelsize=6, colors=MUTED)
            ax.set_xlim(-0.45, len(bins) - 0.55)
            hi, lo = (D['fd_ms'][EV], D['res_ms'][EV])
            ax.annotate('', (EV, lo), (EV, hi), zorder=6, arrowprops=dict(arrowstyle='<|-|>', color='#7a2e46', lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=6))
            ax.text(EV + 0.22, (hi + lo) / 2, f'{(lo / hi - 1) * 100:.1f}%'.replace('-', '−'), ha='left', va='center', fontsize=6.6, color='#7a2e46', fontweight='bold', zorder=7)
        ax1.text(EV, 1690, 'Event-onset window', ha='center', va='top', fontsize=6.2, color='#8a3b53', fontproperties=kr)
        hd = [Line2D([], [], color=C_SEQ, lw=1.6, marker='o', ms=3.6, mec='white', mew=0.6, label='#0 Baseline'), Line2D([], [], color=C_FD, lw=1.5, marker='^', ms=4.0, mec='white', mew=0.6, label='#0′ FlashDrive BF16'), Line2D([], [], color=C_RES, lw=2.0, marker='s', ms=3.6, mec='white', mew=0.6, label='#4 (RESPECT Decode)')]
        lg = f.legend(handles=hd, loc='upper center', ncol=3, frameon=False, fontsize=6.3, handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0.55, 1.0))
        for t in lg.get_texts():
            t.set_fontproperties(kr)
            t.set_fontsize(6.3)
        for ax in (ax1, ax2):
            ax.set_xticks(xs)
            ax.set_xticklabels(lab, fontsize=5.8, linespacing=0.9)
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_xlabel('Frames from the event-start frame (5-frame windows)', fontproperties=kr, fontsize=6.6, labelpad=2)
        f.subplots_adjust(left=0.15, right=0.98, top=0.93, bottom=0.175, hspace=0.62)
        for ax, title in ((ax1, '(a) Window mean'), (ax2, '(b) Window max')):
            box = ax.get_position()
            f.text((box.x0 + box.x1) / 2, box.y0 - 0.125, title, ha='center', va='top', fontsize=7.4, fontproperties=kr)
        save(f, out, "fig9_event_timeline")


def draw_fig14_trajectory_error(root, out):
    """Section V-G: minADE6 of RESPECT (measured with #4, which gives the same CoC) and of #0' against #0 on the same
    frames; a point on the diagonal (y = x) has the same error as #0, a point in the shaded region a larger one."""
    fd = {(r["clip_id"], r["frame_idx"]): r["fd_minADE6_m"]
          for r in map(json.loads, (root / "flashdrive_quality" / "fd_quality.jsonl").read_text().splitlines())}
    lim = 9.0
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.05))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.86, bottom=0.29, wspace=0.30)
    for ax, (k, title) in zip(axs, (("event", "(a) Event window\n(400 frames)"), ("general", "(b) Nominal driving\n(600 frames)"))):
        rows = [json.loads(line) for line in (root / "main" / f"frames_{k}.jsonl").read_text().splitlines()]
        b = np.array([r["reference_minADE6_m"] for r in rows])
        s4 = np.array([r["flow"]["s4_minADE6_m"] for r in rows])
        f = np.array([fd[(r["clip_id"], r["frame_idx"])] for r in rows])
        ax.fill_between([0, lim], [0, lim], [lim, lim], color="#F6E6E4", lw=0, zorder=0)
        ax.plot([0, lim], [0, lim], color="#5B6770", lw=0.7, zorder=1)
        ax.text(8.0, 8.55, "$y=x$", rotation=45, ha="center", va="center", fontsize=5.4, color="#5B6770")
        ax.scatter(b, f, marker="^", s=6, color="#C9772A", alpha=0.6, lw=0, zorder=2)
        ax.scatter(b, s4, s=4.5, color="#187B62", lw=0, zorder=3)
        ax.set(xlim=(0, lim), ylim=(0, lim), xticks=[0, 3, 6, 9], yticks=[0, 3, 6, 9], aspect="equal")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("minADE6 of #0 (m)", fontsize=6.0, labelpad=1)
        ax.set_title(title, y=-0.56, fontsize=6.8)
    axs[0].set_ylabel("minADE6 of each config. (m)", fontsize=6.0, labelpad=1)
    fig.legend([plt.Line2D([], [], marker="o", ls="None", color="#187B62", ms=3.2),
                plt.Line2D([], [], marker="^", ls="None", color="#C9772A", ms=3.8)],
               ["RESPECT", "#0\u2032 (FlashDrive BF16)"], loc="upper center", ncol=2, frameon=False, fontsize=6.0,
               handletextpad=0.2, columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig14_trajectory_error.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((root / "summary.json").read_text())
    draw_fig3_concentration(s, out)
    if s.get("dev"):
        draw_supp_n_select(s, out)
        draw_supp_tile_boundary(s, out, root)
    draw_fig7_configurations(s, out)
    draw_fig8_sources(root, out)
    draw_supp_kernel_cost(s, out)
    draw_fig9_event_timeline(s, out)
    add = json.loads((root / "summary_additions.json").read_text())
    draw_fig10_delay(s, add, out)
    draw_fig4_realtime_timeline(root, out)
    if "section6" in s:
        draw_fig12_stopping_margin(s, out)
    from paper_system_figures import draw_fig11_realtime, draw_fig13_trajectories
    draw_fig11_realtime(s, out)
    draw_fig13_trajectories(root, out)
    draw_fig14_trajectory_error(root, out)
    from paper_method_figures import draw_method_figures
    draw_method_figures(s, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
