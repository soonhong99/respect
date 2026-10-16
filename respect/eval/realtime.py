"""Real-time execution: inputs arrive every 0.1 s and only the latest one is processed.

A frame's input holds the four most recent images of each camera (0.1 s apart).
``restore`` rebuilds it from per-frame files that keep only the newest image of
each camera, so a clip is stored once instead of four times.

Latest-frame policy: when a cycle finishes, the worker processes the most recent
frame that has arrived and drops the others. Queued frames would make the input
of every later cycle older, and batching them would add Prefill work for inputs
whose trajectories are superseded by the newest one.

The helpers below compute the quantities reported in Section VI from the cycle
records of ``scripts/run_realtime.py``:
  trajectory age   time since the arrival of the input of the trajectory in use
  expired share    share of 0.1 s ticks whose trajectory age exceeds the 6.4 s horizon
  event reaction   publication time of the first trajectory built from a frame at or
                   after the event-start frame E, minus the arrival time of E
"""
from __future__ import annotations

import math
from typing import Any

STEP_S = 0.1          # input arrival interval
HORIZON_S = 6.4       # trajectory length
NUM_FRAMES = 4        # images per camera in one input


def shrink(inp: dict[str, Any]) -> dict[str, Any]:
    """Keep only the newest image of each camera (stored form of one frame)."""
    s = dict(inp)
    s["image_frames"] = inp["image_frames"][:, -1:].clone()
    s["absolute_timestamps"] = inp["absolute_timestamps"][:, -1:].clone()
    s.pop("relative_timestamps", None)
    return s


def restore(small: dict[int, dict[str, Any]], g: int) -> dict[str, Any]:
    """Input of frame ``g``: the newest images of frames g-3..g stacked per camera."""
    import torch
    base = dict(small[g])
    frames = [small[g - NUM_FRAMES + 1 + i] for i in range(NUM_FRAMES)]
    base["image_frames"] = torch.cat([f["image_frames"] for f in frames], dim=1)
    base["absolute_timestamps"] = torch.cat([f["absolute_timestamps"] for f in frames], dim=1)
    at = base["absolute_timestamps"]
    base["relative_timestamps"] = (at - at.min()).float() * 1e-6
    return base


def same_value(a: Any, b: Any) -> bool:
    """Value-level equality of nested inputs (tensors, arrays, dicts, lists, scalars)."""
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def run_metrics(cycles: list[dict]) -> dict:
    """Metrics of one real-time run (one configuration, clip and start position)."""
    rs = sorted(cycles, key=lambda r: r["t_pub"])
    g0, g_end, e = rs[0]["g0"], rs[0]["g_end"], rs[0]["e"]
    t_end = max((g_end - g0) * STEP_S, rs[-1]["t_pub"])
    ages, j = [], 0
    for tick in range(int(math.ceil(rs[0]["t_pub"] / STEP_S - 1e-9)), int(round(t_end / STEP_S)) + 1):
        tau = tick * STEP_S
        while j + 1 < len(rs) and rs[j + 1]["t_pub"] <= tau + 1e-9:
            j += 1
        ages.append(tau - rs[j]["t_arrival"])
    processed = {r["g"] for r in rs}
    covered = set()
    for g in processed:
        covered.update(range(g - NUM_FRAMES + 1, g + 1))
    n_arr = g_end - g0 + 1
    first_after = next(r for r in rs if r["g"] >= e)
    t_e = (e - g0) * STEP_S
    return dict(cycles_s=[r["cycle_ms"] / 1000 for r in rs], ages=ages,
                expired=sum(a > HORIZON_S for a in ages), n_ticks=len(ages),
                processed=len(processed) / n_arr,
                covered=sum(1 for g in range(g0, g_end + 1) if g in covered) / n_arr,
                reaction_s=first_after["t_pub"] - t_e, coc_s=first_after["t_decode"] - t_e,
                k=[r["k"] for r in rs if r["k"] is not None],
                stage_ms={s: [r[s] for r in rs] for s in ("input_ms", "prefill_ms", "decode_ms", "flow_ms")})


def latest_frame_schedule(g0: int, g_end: int, service_s) -> list[tuple[int, float, float]]:
    """Cycles (frame, start, publication) of the latest-frame policy for a service-time function.

    ``service_s(g, k)`` returns the cycle time in seconds for frame ``g`` processed
    ``k`` frames after the previous one (``k`` is None for the first cycle).
    """
    t, last, out = 0.0, None, []
    while True:
        g = min(g0 + int(math.floor(t / STEP_S + 1e-9)), g_end)
        if last is not None and g <= last:
            if last >= g_end:
                return out
            t = (last + 1 - g0) * STEP_S
            continue
        s = service_s(g, (g - last) if last is not None else None)
        out.append((g, t, t + s))
        t += s
        last = g


def rss_min_distance(v_r: float, v_f: float, rho: float, b_min: float = 4.0, b_max: float = 8.0,
                     a_max: float = 0.0) -> float:
    """RSS longitudinal safe distance for response time ``rho`` (Shalev-Shwartz et al., 2017)."""
    v = v_r + a_max * rho
    return v_r * rho + 0.5 * a_max * rho ** 2 + v ** 2 / (2 * b_min) - max(v_f, 0.0) ** 2 / (2 * b_max)


def rss_allowed_response(v_r: float, gap: float, v_f: float, b_min: float = 4.0, b_max: float = 8.0,
                         a_max: float = 0.0) -> float:
    """Largest response time whose RSS safe distance fits in ``gap`` (NaN if none)."""
    if rss_min_distance(v_r, v_f, 0.0, b_min, b_max, a_max) > gap:
        return float("nan")
    lo, hi = 0.0, 120.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if rss_min_distance(v_r, v_f, mid, b_min, b_max, a_max) > gap:
            hi = mid
        else:
            lo = mid
    return lo

