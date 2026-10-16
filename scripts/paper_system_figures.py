"""CPU-only renderers for paper Figs. 11 and 13 from published results.

Called by make_figures.py. Fig. 13 renders the BEV comparison only; the camera
panel requires gated dataset access and sensor images are not redistributed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def realtime_rows(summary):
    """Keep the paper's two environments explicit; do not pool their runs."""
    slow = summary["section6"]["realtime"]["configs"]
    fast = summary["section6"]["stage_link"]["configs"]
    rows = []
    for name, code, source, same_weights in (
        ("Public implementation (#0)", "S0", slow, True),
        ("RESPECT Decode (#4)", "S4", slow, True),
        ("RESPECT (#5)", "S4F", fast, True),
        ("FlashDrive BF16", "FC8", fast, False),
        ("FlashDrive + RESPECT Decode", "D8", fast, False),
    ):
        run = source[code]
        stage = run["stage_mean_ms"]
        stages = [stage["input_ms"] + stage["prefill_ms"], stage["decode_ms"], stage["flow_ms"]]
        if source is slow:
            mean, maximum = run["reaction_s"]["mean"], run["reaction_s"]["max"]
        else:
            mean, maximum = run["reaction_mean_s"], run["reaction_max_s"]
        rows.append(dict(name=name, code=code, stages_s=np.asarray(stages) / 1000,
                         reaction_mean_s=mean, reaction_max_s=maximum,
                         same_weights=same_weights))
    return rows


def trajectory_metrics(frame):
    public = np.asarray(frame["public_implementation_0"], dtype=float)
    respect = np.asarray(frame["respect_4"], dtype=float)
    flashdrive = np.asarray(frame["flashdrive_bf16_0prime"], dtype=float)
    if any(x.shape != (6, 64, 3) for x in (public, respect, flashdrive)):
        raise ValueError("Expected six trajectories with 64 xyz positions per system")
    return dict(respect_max_coordinate_diff_m=float(np.abs(respect[:, :, :2] - public[:, :, :2]).max()),
                flashdrive_mean_endpoint_distance_m=float(np.linalg.norm(
                    flashdrive[:, -1, :2].mean(0) - public[:, -1, :2].mean(0))))


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"{name}.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def draw_fig11_realtime(summary, out):
    """Stage means and event response across 10 clips x 5 start positions."""
    rows = realtime_rows(summary)
    colors = ["#B7C3CC", "#C0392B", "#5B7FA6"]
    ys = [0, 1, 2, 3.6, 4.6]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.9),
                               gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.12))
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.17, top=0.84)
    for row, y in zip(rows, ys):
        if row["code"] == "S4F":
            for ax in (a, b):
                ax.axhspan(y - .45, y + .45, color="#E6F2EC", zorder=0)
        left = 0.
        for value, color in zip(row["stages_s"], colors):
            a.barh(y, value, left=left, height=.62, color=color, edgecolor="white", lw=.6, zorder=2)
            left += value
        a.text(left + .06, y, f"{left:.2f}", va="center", fontsize=8, fontweight="bold")
        a.text(-.04, y, row["name"], ha="right", va="center", transform=a.get_yaxis_transform(),
               fontsize=8, fontweight="bold" if row["code"] == "S4F" else "normal")
        color = "#1F7A5A" if row["same_weights"] else "#8A5A9E"
        mean, maximum = row["reaction_mean_s"], row["reaction_max_s"]
        b.plot([mean, maximum], [y, y], color=color, alpha=.6, lw=1.4)
        b.plot(maximum, y, marker="|", ms=13, mew=1.4, color=color, alpha=.6)
        b.plot(mean, y, "o", ms=7, color=color, mec="white", mew=.8)
        b.text(mean - .32, y, f"{mean:.2f}", ha="right", va="center", fontsize=8, fontweight="bold")
        b.text(maximum + .22, y, f"{maximum:.1f}", va="center", fontsize=7.5, color="#5B6770")
    for y, name, color in ((-.72, "Same model weights", "#1F7A5A"),
                            (2.9, "Changed model weights", "#8A5A9E")):
        a.text(-.04, y, name, ha="right", va="center", transform=a.get_yaxis_transform(),
               fontsize=8, fontweight="bold", color=color)
    for ax in (a, b):
        ax.set(ylim=(5.25, -1.05), yticks=[])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#E3E7EA", lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    a.set(xlim=(0, 3.95), xticks=[0, 1, 2, 3], xlabel="(a) Stage times per cycle (s)")
    b.set(xlim=(0, 9.6), xticks=[0, 2, 4, 6, 8], xlabel="(b) Event response latency (s)")
    handles = [Patch(color=c, label=n) for c, n in zip(colors, ("Input + Prefill", "Decode", "Flow"))]
    handles += [Line2D([], [], marker="o", ls="none", color="#5B6770", label="Mean"),
                Line2D([], [], color="#5B6770", label="Observed max.")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.62, 1), ncol=5,
               frameon=False, fontsize=8, handlelength=1.4, columnspacing=1)
    _save(fig, out, "fig11_realtime_breakdown")


def draw_fig13_trajectories(root, out):
    """BEV of paper Fig. 13: all six paths/system; no gated camera pixels."""
    frame = json.loads((Path(root) / "frame_trajectories/bbbe775f_0160.json").read_text())
    metrics = trajectory_metrics(frame)
    p0 = np.asarray(frame["public_implementation_0"])
    pr = np.asarray(frame["respect_4"])
    pf = np.asarray(frame["flashdrive_bf16_0prime"])
    gt = np.asarray(frame["recorded_path"])
    e0, ef = p0[:, -1, :2].mean(0), pf[:, -1, :2].mean(0)
    c0, cr, cf, ink = "#B4BCC2", "#1F7A5A", "#B8601A", "#1E262C"
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    fig.subplots_adjust(left=.12, right=.61, top=.88, bottom=.14)
    for k in range(6):
        for paths, color, width, style in ((p0, c0, 3., "-"), (pr, cr, 1.25, "--"), (pf, cf, 1.1, "-.")):
            ax.plot(-paths[k, :, 1], paths[k, :, 0], color=color, lw=width, ls=style)
    ax.plot(-gt[:, 1], gt[:, 0], color=ink, lw=1.5, zorder=5)
    ax.plot(-gt[-1, 1], gt[-1, 0], "s", color=ink, mec="white", ms=5, zorder=6)
    ax.plot(-e0[1], e0[0], "o", ms=9, mfc="white", mec="#5B6770", zorder=6)
    ax.plot(-pr[:, -1, 1].mean(), pr[:, -1, 0].mean(), "o", ms=4.5, color=cr, zorder=7)
    ax.plot(-ef[1], ef[0], "^", ms=7, color=cf, mec="white", zorder=6)
    # Final-Word display: bend toward the label, away from the trajectory lines.
    gap_arrow = ax.annotate("", xy=(-ef[1], ef[0]), xytext=(-e0[1], e0[0]), zorder=8,
                arrowprops=dict(arrowstyle="<|-|>,head_length=0.4,head_width=0.2", color=ink, lw=1,
                                connectionstyle="arc3,rad=0.45", shrinkA=6, shrinkB=6))
    gap_arrow.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
    ax.plot(0, 0, marker="^", ms=10, color=ink, zorder=6)
    ax.text(1.2, -.7, "Ego", fontsize=8, va="center")
    ax.set(xlim=(-26, 5), ylim=(-4, 35), yticks=[0, 10, 20, 30], aspect="equal",
           xlabel="Left (m)", ylabel="Forward (m)")
    ax.set_xticks([-20, -10, 0], ["20", "10", "0"])
    ax.grid(color="#E3E7EA", lw=.6)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pedestrian-crowd event: clip bbbe775f, frame 160", fontsize=10, y=.99)
    handles = [Line2D([], [], color=c0, lw=3, marker="o", mfc="white", mec="#5B6770"),
               Line2D([], [], color=cr, lw=1.25, ls="--", marker="o", ms=4),
               Line2D([], [], color=cf, lw=1.1, ls="-.", marker="^"),
               Line2D([], [], color=ink, lw=1.5, marker="s", ms=4)]
    fig.legend(handles, ["Public implementation (#0)\n(mean endpoint at 6.4 s)",
                         "RESPECT (measured with #4)\n(mean endpoint)",
                         "FlashDrive BF16 (#0′)\n(mean endpoint)", "Recorded path\n(square: stop)"],
               loc="upper left", bbox_to_anchor=(.68, .88), frameon=False, fontsize=8,
               handlelength=2.7, labelspacing=1)
    ax.annotate(f"RESPECT ≈ public implementation\nmax. coordinate difference\n{metrics['respect_max_coordinate_diff_m']:.4f} m",
                xy=(-e0[1] - .8, e0[0] - .8), xytext=(-12, 4), color=cr, fontsize=7,
                ha="center", va="center", zorder=8, bbox=dict(fc="white", ec="none", pad=.5),
                arrowprops=dict(arrowstyle="-", color=cr, lw=.7, shrinkA=1, shrinkB=2))
    ax.text(-11, 28, f"FlashDrive BF16:\nmean endpoint {metrics['flashdrive_mean_endpoint_distance_m']:.2f} m\nfrom public implementation",
            fontsize=8, fontweight="bold", color=cf, ha="left", va="center", zorder=8,
            bbox=dict(fc="white", ec="none", pad=.5), linespacing=1.1)
    fig.text(.70, .25, "Six trajectories per system\nPrediction horizon: 6.4 s\nSame six Flow noise samples",
             fontsize=8, color="#5B6770", va="top", linespacing=1.35)
    _save(fig, out, "fig13_trajectories")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="published result directory, e.g. results/paper")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.root / "summary.json").read_text())
    draw_fig11_realtime(summary, args.out)
    draw_fig13_trajectories(args.root, args.out)
    print(f"Figures 11 and 13 written to {args.out}")


if __name__ == "__main__":
    main()
