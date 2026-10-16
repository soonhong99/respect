#!/usr/bin/env python3
"""Lead object of every evaluated event clip at the event-start frame E (Section VI-C).

From the dataset's ego motion, vehicle dimensions and offline obstacle labels
(``obstacle.offline``, rig frame; moved to the world frame with the ego pose at
the label's reference time), at frame E:
  v_r   ego speed, from the ego positions at frames E-1 and E+1
  lead  the nearest object ahead whose centre lies inside the ego lane
        (lateral offset below half the sum of the ego and object widths)
  gap   longitudinal distance from the ego front to the object centre minus half
        the object length
  v_f   the object's speed along the ego heading, from its labels at E-1 and E+1
The label of a track at a frame is the one nearest to that frame's time.

Output: results/paper/realtime/rss_leads.json (one entry per clip)
Usage: python scripts/extract_rss_leads.py --out results/paper/realtime/rss_leads.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.eval.dataset import load_frames  # noqa: E402

REVISION = "b719eea7f0a63619ef51ec7f54178af0937ef050"
STEP_US = 100_000


def main() -> None:
    import physical_ai_av
    from scipy.spatial.transform import Rotation
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=REVISION)
    clips = load_frames(Path(args.data) / "event.jsonl")
    res = {}
    for cid in sorted(clips):
        e = int(next(iter(clips[cid].values())).meta["event_start_frame"])
        ego = avdi.get_clip_feature(cid, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
        dims = avdi.get_clip_feature(cid, avdi.features.CALIBRATION.VEHICLE_DIMENSIONS, maybe_stream=True)
        ob = avdi.get_clip_feature(cid, avdi.features.LABELS.OBSTACLE_OFFLINE, maybe_stream=True)["obstacle.offline"]
        assert set(ob.reference_frame) == {"rig"}
        ts = np.array([(e - 1) * STEP_US, e * STEP_US, (e + 1) * STEP_US], dtype=np.int64)
        pose = ego(ts)
        xyz = pose.pose.translation
        yaw = pose.pose.rotation.as_euler("zyx")[1][0]
        per = defaultdict(dict)                     # frame -> track -> (dt, x, y, length, width, class)
        tr = ob.reference_frame_timestamp_us.values.astype(np.int64)
        near = (tr >= ts[0] - STEP_US) & (tr <= ts[-1] + STEP_US)
        sub = ob[near]
        e_pose = ego(sub.reference_frame_timestamp_us.values.astype(np.int64)).pose
        world = e_pose.rotation.apply(np.stack([sub.center_x, sub.center_y, sub.center_z], 1)) + e_pose.translation
        for i, (t, trk) in enumerate(zip(sub.timestamp_us.values, sub.track_id.values)):
            f = int(round(t / STEP_US))
            if f not in (e - 1, e, e + 1):
                continue
            dt = abs(int(t) - f * STEP_US)
            cur = per[f].get(int(trk))
            if cur is None or dt < cur[0]:
                per[f][int(trk)] = (dt, float(world[i, 0]), float(world[i, 1]), float(sub.size_x.values[i]),
                                    float(sub.size_y.values[i]), str(sub.label_class.values[i]))
        v_r = float(np.linalg.norm(xyz[2, :2] - xyz[0, :2]) / (2 * STEP_US / 1e6))
        c, s = math.cos(yaw), math.sin(yaw)
        front = float(dims.rear_axle_to_bbox_center) + float(dims.length) / 2
        lead = None
        for trk, (_, x, y, sx, sy, cls) in per[e].items():
            dx, dy = x - xyz[1, 0], y - xyz[1, 1]
            lon, lat = c * dx + s * dy, -s * dx + c * dy
            if lon <= front or abs(lat) > (float(dims.width) + sy) / 2:
                continue
            gap = lon - front - sx / 2
            a, b = per[e - 1].get(trk), per[e + 1].get(trk)
            v_f = (c * (b[1] - a[1]) + s * (b[2] - a[2])) / (2 * STEP_US / 1e6) if (a and b) else 0.0
            if lead is None or gap < lead["gap_m"]:
                lead = dict(track=trk, cls=cls, gap_m=gap, v_f=v_f)
        res[cid] = dict(event_start_frame=e, event_type=next(iter(clips[cid].values())).meta.get("event_type"),
                        v_r=v_r, lead=lead)
        print(cid[:8], "v_r %.2f" % v_r, lead, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
