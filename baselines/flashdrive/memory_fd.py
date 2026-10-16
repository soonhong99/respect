#!/usr/bin/env python3
"""Memory of FlashDrive BF16 (#0') in its own process (Table V of the paper).

The process loads only FlashDrive (as in offline_unified.py: W4A8 off; streaming, DFlash, expert
fusion, torch.compile on; temperature 0) and processes every evaluation clip once: the window of
the frame before the first scored frame (fill_windows.py) starts the stream, then the scored frames
are processed in order. One call of FlashDrive's inference function runs Prefill, Decode and Flow
(one trajectory).

The first three frames of the first clip are processed as a warm-up (compilation). Then the CUDA
peak statistics are reset at the start of every call and at the end of FlashDrive's Prefill and
Decode functions, so the process peak after the warm-up is the maximum of the stage peaks.
Recorded per frame (and per clip-start window): allocated memory at the start and the peak
allocated / reserved memory of Prefill, Decode and Flow. The system memory in use
(MemTotal - MemAvailable) is sampled every 50 ms.

Output: <out>/FD.jsonl and <out>/FD_summary.json
Usage:
  python baselines/flashdrive/memory_fd.py --inputs INPUTS --fill FILL_WINDOWS \
      --flashdrive /path/to/flashdrive --out runs/memory [--limit-frames 3]
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from measure_memory import SystemMemory  # noqa: E402
from respect.eval.dataset import input_path, iterate, load_frames  # noqa: E402

MAX_NEW, DIFF_STEPS, CACHE_STEPS = 128, 8, [3, 4, 5, 6]        # FlashDrive's setting
MIB = 2 ** 20

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
args = ap.parse_args()
DATA, INPUTS, FILL, FD_PATH = Path(args.data), Path(args.inputs), Path(args.fill), args.fd_model
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
sysmem = SystemMemory()
sys_before = sysmem.used_mib()


def sync():
    torch.cuda.synchronize()


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
loaded_mib = torch.cuda.memory_allocated() / MIB
helper = importlib.import_module(f"{mc.__module__.split('.')[0]}.helper")
processor = helper.get_processor(fdm.tokenizer)
VS = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
VE = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
EGO = ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")

S = {}


def _peak(key):
    S[key + "_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / MIB, 1)
    S[key + "_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / MIB, 1)
    torch.cuda.reset_peak_memory_stats()


def _wrap(name, key):
    f = getattr(fdm, name)

    def g(*a, **k):
        out = f(*a, **k)
        sync()
        _peak(key)
        return out
    setattr(fdm, name, g)


_wrap("_dflash_prefill", "prefill")
_wrap("_dflash_generate", "decode")


def fd_window(inp, streaming):
    frames = inp["image_frames"].flatten(0, 1)
    msgs = helper.create_message(frames, camera_indices=inp["camera_indices"])
    tok = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                        return_dict=True, return_tensors="pt")
    mi = {"tokenized_data": tok, **{k: inp[k] for k in EGO}}
    return convert_to_streaming_window(mi, VS, VE) if streaming else mi


def fd_call(mi):
    S.clear()
    sync()
    S["start_alloc_mib"] = round(torch.cuda.memory_allocated() / MIB, 1)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        data = helper.to_device(mi, "cuda")
        pred = fdm.sample_trajectories_streaming(data=data, num_traj_samples=1, return_extra=True,
                                                 max_new_tokens=MAX_NEW, temperature=0.0, top_p=1.0,
                                                 diffusion_kwargs={"inference_step": DIFF_STEPS,
                                                                   "cache_steps": CACHE_STEPS,
                                                                   "int_method": "euler_with_cache"})
    sync()
    _peak("flow")
    keys = [k for k in ("prefill", "decode", "flow") if k + "_peak_alloc_mib" in S]
    S["peak_alloc_mib"] = max(S[k + "_peak_alloc_mib"] for k in keys)
    S["peak_reserved_mib"] = max(S[k + "_peak_reserved_mib"] for k in keys)
    return pred[0]


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
    pred = fd_call(fd_window(fill, streaming=False))
    assert pred is None, "the fill window did not take the first-window path"
    rows = [dict(set=s, clip_id=cid, frame_idx=sc[0].frame_idx - 1, window="clip_start", **S)]
    for m, sv in zip(mods, saved):
        for k, v in sv.items():
            setattr(m, k, v)
    for f in sc:
        pred = fd_call(fd_window(load_inp(cid, f.frame_idx), streaming=True))
        assert pred is not None
        rows.append(dict(set=s, clip_id=cid, frame_idx=f.frame_idx, window="scored", **S))
    if fp is not None:
        for r in rows:
            fp.write(json.dumps(r) + "\n")
        fp.flush()
    return rows


PLAN = []
for s in ("event", "general"):
    cids = [c for c, _h, _sc in iterate(SETS[s], s)]
    PLAN += [(s, c) for c in (cids[:args.limit_clips] if args.limit_clips else cids)]

t_all = time.perf_counter()
run_clip(*PLAN[0], None, nframes=3)                       # warm-up and compilation, not recorded
resident = torch.cuda.memory_allocated() / MIB
print("FD warm-up %.1f min" % ((time.perf_counter() - t_all) / 60), flush=True)
sysmem.reset()
rows = []
with (OUT / "FD.jsonl").open("w") as fp:
    for i, (s, cid) in enumerate(PLAN):
        rr = run_clip(s, cid, fp, nframes=args.limit_frames)
        rows += rr
        print("[FD %d/%d %s %s] %d calls, peak alloc %.1f MiB  (%.1f min)" % (
            i + 1, len(PLAN), s, cid[:8], len(rr), max(r["peak_alloc_mib"] for r in rr),
            (time.perf_counter() - t_all) / 60), flush=True)
sysmem.on = False
scored = [r for r in rows if r["window"] == "scored"]
summary = dict(
    config="FD", frames=len(scored), clip_start_windows=len(rows) - len(scored), torch=torch.__version__,
    device=torch.cuda.get_device_name(0), argv=sys.argv, flashdrive_git=sh(f"git -C {args.flashdrive} rev-parse HEAD"),
    loaded_alloc_mib=round(loaded_mib, 1), resident_alloc_after_warmup_mib=round(resident, 1),
    peak_alloc_mib=max(r["peak_alloc_mib"] for r in rows),
    peak_alloc_scored_mib=max(r["peak_alloc_mib"] for r in scored),
    peak_reserved_mib=max(r["peak_reserved_mib"] for r in rows),
    stage_peak_alloc_mib={k: max(r[k + "_peak_alloc_mib"] for r in rows if k + "_peak_alloc_mib" in r)
                          for k in ("prefill", "decode", "flow")},
    system_used_before_load_mib=round(sys_before, 1), system_used_peak_mib=round(sysmem.peak, 1),
    clocks=sh("sudo -n /usr/bin/jetson_clocks --show")[-1500:], nvpmodel=sh("nvpmodel -q"),
    minutes=round((time.perf_counter() - t_all) / 60, 1))
(OUT / "FD_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("frames", "peak_alloc_mib", "peak_alloc_scored_mib", "stage_peak_alloc_mib",
                                           "system_used_peak_mib", "minutes")}), flush=True)
