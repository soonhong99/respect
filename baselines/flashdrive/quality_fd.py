#!/usr/bin/env python3
"""minADE6 of FlashDrive BF16 (#0') on the evaluation frames (Section V-I of the paper).

The process loads FlashDrive as #0' of offline_unified.py (W4A8 off; streaming, DFlash, expert
fusion, torch.compile on; temperature 0) and processes every evaluation clip once in the same
order: the window of the frame before the first scored frame (fill_windows.py) starts the stream,
then the scored frames are processed in order (d = 1). The only change from #0' is the number of
trajectory samples, 6 instead of 1: FlashDrive generates one CoC per window and samples six
trajectories from it with its own Flow, as the six Flow seeds of #0 and #4 (one CoC, six samples).
The diffusion noise is seeded with torch.manual_seed(frame_idx) before every call, or with the same seed on
every frame when --fixed-seed is given (as #0 and #4, whose six Flow seeds are the same on every frame).

Recorded per scored frame: CoC tokens, the six trajectories (float32), minADE6 / minFDE6 against
ego_future_xyz (respect.alpamayo.flow.min_ade_fde, x-y plane, 64 waypoints) and the call time.
The CoC tokens are compared afterwards with flashdrive_unified/fd.jsonl (one sample) to check that
the sample count does not change FlashDrive's CoC.

Output: <out>/fd_quality.jsonl and <out>/fd_quality_summary.json
Usage:
  python baselines/flashdrive/quality_fd.py --inputs INPUTS --fill FILL_WINDOWS \
      --flashdrive /path/to/flashdrive --out runs/fd_quality [--limit-frames 2 --limit-clips 1]
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from respect.alpamayo.flow import ground_truth, min_ade_fde  # noqa: E402
from respect.eval.dataset import input_path, iterate, load_frames  # noqa: E402

MAX_NEW, DIFF_STEPS, CACHE_STEPS = 128, 8, [3, 4, 5, 6]        # FlashDrive's setting, as #0'
K = 6

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(REPO / "data"))
ap.add_argument("--inputs", required=True)
ap.add_argument("--fill", required=True, help="output directory of baselines/flashdrive/fill_windows.py")
ap.add_argument("--flashdrive", required=True, help="FlashDrive checkout (added to sys.path, not installed)")
ap.add_argument("--extra-path", default=None, help="optional directory appended to sys.path for FlashDrive's dependencies")
ap.add_argument("--fd-model", default="z-lab/Alpamayo-1.5-10B")
ap.add_argument("--out", required=True)
ap.add_argument("--limit-frames", type=int, default=0, help="scored frames per clip (0: all)")
ap.add_argument("--limit-clips", type=int, default=0, help="clips per set (0: all)")
ap.add_argument("--fixed-seed", type=int, default=None, help="seed of every call (default: the frame index)")
args = ap.parse_args()
DATA, INPUTS, FILL, FD_PATH = Path(args.data), Path(args.inputs), Path(args.fill), args.fd_model
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


SETS = {s: load_frames(DATA / f"{s}.jsonl") for s in ("event", "general")}
sys.path.insert(0, str(Path(args.flashdrive)))
if args.extra_path:
    sys.path.append(args.extra_path)
from flashdrive import FlashDriveMixin, _EXTENDED_VISION_RANGES, resolve_model_class  # noqa: E402
from flashdrive.streaming import convert_to_streaming_window  # noqa: E402

mc = resolve_model_class(FD_PATH)
mc._supports_sdpa = True
fcfg = mc.config_class.from_pretrained(FD_PATH)
fcfg.attn_implementation = fcfg._attn_implementation = "sdpa"
fdm = mc.from_pretrained(FD_PATH, config=fcfg, dtype=torch.bfloat16).to("cuda").eval()
fdm.__class__ = type("FD", (FlashDriveMixin, mc), {})
fdm.setup_rollout()
fdm.setup_torch_compile("max-autotune")
fdm.setup_streaming(extended_vision_ranges=_EXTENDED_VISION_RANGES[mc])
fdm.setup_expert_fusion()
fdm.setup_dflash(f"{FD_PATH}-DFlash")
with torch.no_grad():                         # as in offline_unified.py: the added mask-token row of lm_head is zero
    fdm.vlm.lm_head.weight[fdm._dflash_mask_token_id].zero_()
helper = importlib.import_module(f"{mc.__module__.split('.')[0]}.helper")
processor = helper.get_processor(fdm.tokenizer)
VS = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
VE = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
STOP = int(fdm._dflash_stop_token_id)
EGO = ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")


def fd_window(inp, streaming):
    frames = inp["image_frames"].flatten(0, 1)
    msgs = helper.create_message(frames, camera_indices=inp["camera_indices"])
    tok = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                        return_dict=True, return_tensors="pt")
    mi = {"tokenized_data": tok, **{k: inp[k] for k in EGO}}
    return convert_to_streaming_window(mi, VS, VE) if streaming else mi


def fd_call(mi, seed):
    torch.manual_seed(seed if args.fixed_seed is None else args.fixed_seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        data = helper.to_device(mi, "cuda")
        pred = fdm.sample_trajectories_streaming(data=data, num_traj_samples=K, return_extra=True,
                                                 max_new_tokens=MAX_NEW, temperature=0.0, top_p=1.0,
                                                 diffusion_kwargs={"inference_step": DIFF_STEPS,
                                                                   "cache_steps": CACHE_STEPS,
                                                                   "int_method": "euler_with_cache"})
    torch.cuda.synchronize()
    return pred, round((time.perf_counter() - t0) * 1000, 3)


S = {}
_orig_loop = fdm._dflash_decode_loop


def _loop(*a, **k):                           # as offline_unified.py: the generated tokens from the first token
    out = _orig_loop(*a, **k)
    S["gen"] = out[0][0, k["num_input_tokens"]:int(out[1]) + 2].tolist()
    return out


fdm._dflash_decode_loop = _loop


def tokens_of():
    gen = S["gen"]
    return gen[:gen.index(STOP) + 1] if STOP in gen else gen


def load_inp(cid, g):
    p = INPUTS / f"{cid}_{g:04d}.pt"
    return torch.load(p if p.exists() else input_path(INPUTS, cid, g), weights_only=False)


def run_clip(s, cid, fp, nframes=0):
    sc = [f for f in SETS[s][cid].values() if f.scored]
    if nframes:
        sc = sc[:nframes]
    fill = torch.load(FILL / f"{cid}_{sc[0].frame_idx - 1:04d}.pt", weights_only=False)
    fdm._past_key_values = None              # new stream: the first window takes the full-Prefill path
    fdm.setup_streaming(extended_vision_ranges=_EXTENDED_VISION_RANGES[mc])
    mods = (fdm.vlm.model.visual, fdm.vlm.model.language_model)
    saved = [{k: getattr(m, k) for k in list(vars(m)) if k.startswith("_cached_")} for m in mods]
    for m in mods:
        m.reset_shape_caches()
    pred, _ = fd_call(fd_window(fill, streaming=False), sc[0].frame_idx - 1)
    assert pred[0] is None, "the fill window did not take the first-window path"
    for m, sv in zip(mods, saved):
        for k, v in sv.items():
            setattr(m, k, v)
    rows = []
    for f in sc:
        inp = load_inp(cid, f.frame_idx)
        pred, ms = fd_call(fd_window(inp, streaming=True), f.frame_idx)
        traj = pred[0].detach().float().cpu().numpy().reshape(-1, 64, 3)
        assert traj.shape[0] == K, traj.shape
        m = min_ade_fde(traj, ground_truth(inp))
        row = dict(set=s, clip_id=cid, frame_idx=f.frame_idx, call_ms=ms, tokens=tokens_of(),
                   fd_minADE6_m=m["minADE_m"], fd_minFDE6_m=m["minFDE_m"], traj=np.round(traj, 4).tolist())
        rows.append(row)
        if fp is not None:
            fp.write(json.dumps(row) + "\n")
    if fp is not None:
        fp.flush()
    return rows


PLAN = []
for s in ("event", "general"):
    cids = [c for c, _h, _sc in iterate(SETS[s], s)]
    PLAN += [(s, c) for c in (cids[:args.limit_clips] if args.limit_clips else cids)]

t_all = time.perf_counter()
run_clip(*PLAN[0], None, nframes=3)                       # warm-up and compilation, not recorded
print("FD warm-up %.1f min" % ((time.perf_counter() - t_all) / 60), flush=True)
rows = []
with (OUT / "fd_quality.jsonl").open("w") as fp:
    for i, (s, cid) in enumerate(PLAN):
        rr = run_clip(s, cid, fp, nframes=args.limit_frames)
        rows += rr
        print("[FDQ %d/%d %s %s] %d frames, mean minADE6 %.3f m  (%.1f min)" % (
            i + 1, len(PLAN), s, cid[:8], len(rr), np.mean([r["fd_minADE6_m"] for r in rr]),
            (time.perf_counter() - t_all) / 60), flush=True)
summary = dict(config="FD_K6", fixed_seed=args.fixed_seed, frames=len(rows), torch=torch.__version__, device=torch.cuda.get_device_name(0),
               argv=sys.argv, flashdrive_git=sh(f"git -C {args.flashdrive} rev-parse HEAD"),
               mean_minADE6_m={s: float(np.mean([r["fd_minADE6_m"] for r in rows if r["set"] == s]))
                               for s in ("event", "general") if any(r["set"] == s for r in rows)},
               clocks=sh("sudo -n /usr/bin/jetson_clocks --show")[-1500:], nvpmodel=sh("nvpmodel -q"),
               minutes=round((time.perf_counter() - t_all) / 60, 1))
(OUT / "fd_quality_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("frames", "mean_minADE6_m", "minutes")}), flush=True)
