#!/usr/bin/env python3
"""FlashDrive BF16 (#0') on the evaluation inputs with the Decode boundary of #4 (Section V).

One process loads the public model and FlashDrive (as in stage_link.py) and runs:
A. #0': FlashDrive as published (W4A8 off; streaming, DFlash, expert fusion, torch.compile on;
   temperature 0).  Every clip is a new stream: the window of the frame before the first scored
   frame (fill_windows.py) fills the streaming cache, then the scored frames are processed in order,
   so the previous window's KV reuse holds (d = 1).  Every clip is processed twice (forward and
   reverse clip order).  Decode is timed from the end of _dflash_prefill to the end of
   _dflash_generate (CUDA synchronized): first-token choice, draft/verify loop and the
   trajectory-start forward.  Recorded: Decode time, verified blocks, tokens up to the stop token,
   Decode-region memory increase, mean module power and energy.
B. #4 in the same environment: as S4 of run_main.py (evaluation inputs, public Prefill, previous
   CoCs and drive index from the default sequential outputs of the clip, static index of the
   development CoCs), one warm-up and two timed runs per frame with the same boundary, checked
   against sequential decoding with the dedicated kernels.
C. Output-preserving Prefill and Flow (respect.system) on the same frames: Prefill KV, first token
   and hidden states of every layer, and the Flow trajectory (float32 bits) against the public
   implementation.
The public model's Prefill KV is compared before and after FlashDrive is imported.

Output: <out>/fd.jsonl, <out>/respect.jsonl, <out>/contract.json (smoke_* with --smoke)
Usage:
  python baselines/flashdrive/offline_unified.py --inputs INPUTS --fill FILL_WINDOWS \
      --flashdrive /path/to/flashdrive --out runs/flashdrive_unified [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
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
from respect.alpamayo.flow import trajectory  # noqa: E402
from respect.alpamayo.model import first_token, load_target, make_inputs, prefill, prompt_length  # noqa: E402
from respect.alpamayo.power import PowerSampler  # noqa: E402
from respect.candidates.datastore import Datastore  # noqa: E402
from respect.candidates.fd_dflash import find_snapshot, load_fd_dflash  # noqa: E402
from respect.config import RespectConfig  # noqa: E402
from respect.decoding.kv_buffer import FixedKVBuffer  # noqa: E402
from respect.decoding.respect import FrameState, RespectDecoder  # noqa: E402
from respect.decoding.sequential import clone_cache, sequential  # noqa: E402
from respect.eval.dataset import input_path, iterate, load_frames, static_sequences, view  # noqa: E402
from respect.kernels import patch as kernels  # noqa: E402
from respect.system.flow_graph import FlowGraph  # noqa: E402
from respect.system.fused import prefill_fast  # noqa: E402

MAX_NEW, DIFF_STEPS, CACHE_STEPS = 128, 8, [3, 4, 5, 6]        # FlashDrive's setting
FLOW_SEED = 0

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(REPO / "data"))
ap.add_argument("--inputs", required=True, help="evaluation inputs (<clip_id>_<frame:04d>.pt), as for run_main.py")
ap.add_argument("--fill", required=True, help="output directory of baselines/flashdrive/fill_windows.py")
ap.add_argument("--flashdrive", required=True, help="FlashDrive checkout (added to sys.path, not installed)")
ap.add_argument("--extra-path", default=None, help="optional directory appended to sys.path for FlashDrive's dependencies")
ap.add_argument("--fd-model", default="z-lab/Alpamayo-1.5-10B")
ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true", help="first event and first general clip, four scored frames each")
args = ap.parse_args()
DATA, INPUTS, FILL, FD_SRC, FD_PATH = Path(args.data), Path(args.inputs), Path(args.fill), Path(args.flashdrive), args.fd_model
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
PFX = "smoke_" if args.smoke else ""
NSMOKE = 4


def sync():
    torch.cuda.synchronize()


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ---------------- public implementation and RESPECT (loaded before FlashDrive is imported) ----------------
cfg = RespectConfig()
target = load_target("nvidia/Alpamayo-1.5-10B", local_files_only=True)
model = target.model
par = next(model.parameters())
draft, mask_emb = load_fd_dflash(find_snapshot(), par.dtype, par.device)
static = Datastore(static_sequences(DATA), gmax=cfg.datastore_key_len)
SETS = {s: load_frames(DATA / f"{s}.jsonl") for s in ("event", "general")}



def kv_digest(pf):
    h = hashlib.sha256()
    for L in pf.past_key_values.layers:
        h.update(L.keys.float().cpu().numpy().tobytes())
        h.update(L.values.float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# Importing FlashDrive must not change the already loaded public model: compare one Prefill KV before and after
_c0 = next(c for c, _h, _s in iterate(SETS["event"], "event"))
_f0 = [f for f in SETS["event"][_c0].values() if f.scored][0].frame_idx
_p0, _r0 = make_inputs(target, torch.load(INPUTS / f"{_c0}_{_f0:04d}.pt", weights_only=False))
KV_BEFORE = kv_digest(prefill(target, _p0))

# ---------------- FlashDrive ----------------
sys.path.insert(0, str(FD_SRC))
if args.extra_path:
    sys.path.append(args.extra_path)
import flashdrive  # noqa: E402
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
with torch.no_grad():                         # as in stage_link.py: the added mask-token row of lm_head is set to zero
    fdm.vlm.lm_head.weight[fdm._dflash_mask_token_id].zero_()
KV_AFTER = kv_digest(prefill(target, _p0))
print("public Prefill KV before/after FlashDrive import:", KV_BEFORE, KV_AFTER, KV_BEFORE == KV_AFTER, flush=True)
assert KV_BEFORE == KV_AFTER, "FlashDrive import changed the public implementation's Prefill"
del _p0, _r0
helper = importlib.import_module(f"{mc.__module__.split('.')[0]}.helper")
processor = helper.get_processor(fdm.tokenizer)
VS = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
VE = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
STOP = int(fdm._dflash_stop_token_id)
EGO = ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")

S = {}


def _wrap(name, fn_after):
    f = getattr(fdm, name)

    def g(*a, **k):
        out = f(*a, **k)
        sync()
        fn_after(out)
        return out
    setattr(fdm, name, g)


def _after_prefill(_out):
    S["t_pf"] = time.perf_counter()
    S["m0"] = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def _after_generate(out):
    S["t_gen"] = time.perf_counter()
    S["peak"] = torch.cuda.max_memory_allocated()
    S["out"] = out


_wrap("_dflash_prefill", _after_prefill)
_wrap("_dflash_generate", _after_generate)
_orig_block = fdm._dflash_block_step


def _block(*a, **k):
    S["blocks"] = S.get("blocks", 0) + 1
    return _orig_block(*a, **k)


fdm._dflash_block_step = _block
_orig_loop = fdm._dflash_decode_loop


def _loop(*a, **k):
    out = _orig_loop(*a, **k)
    S["gen"] = out[0][0, k["num_input_tokens"]:int(out[1]) + 2].tolist()     # from the first token; cut at the stop token below
    return out


fdm._dflash_decode_loop = _loop


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
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        data = helper.to_device(mi, "cuda")
        pred = fdm.sample_trajectories_streaming(data=data, num_traj_samples=1, return_extra=True,
                                                 max_new_tokens=MAX_NEW, temperature=0.0, top_p=1.0,
                                                 diffusion_kwargs={"inference_step": DIFF_STEPS,
                                                                   "cache_steps": CACHE_STEPS,
                                                                   "int_method": "euler_with_cache"})
    sync()
    return pred[0], t0


def scored_of(s, cid):
    return [f for f in SETS[s][cid].values() if f.scored]


def plan():
    out = []
    for s in ("event", "general"):
        cids = [c for c, _h, _sc in iterate(SETS[s], s)]
        if args.smoke:
            cids = cids[:1]
        out += [(s, c) for c in cids]
    return out


PLAN = plan()
(OUT / f"{PFX}contract.json").write_text(json.dumps(dict(
    purpose="#0' on the evaluation inputs, #4 in the same environment, output-preserving Prefill and Flow check", clips=PLAN, smoke=args.smoke,
    fd=dict(model=FD_PATH, max_new_tokens=MAX_NEW, diffusion_steps=DIFF_STEPS, cache_steps=CACHE_STEPS, temperature=0.0,
            w4a8=False, lm_head_mask_row_zero=True, git=sh(f"git -C {FD_SRC} rev-parse HEAD")),
    respect_config=cfg.__dict__, respect_git=sh(f"git -C {REPO} rev-parse HEAD"), hashes={"script": sha(__file__)},
    public_prefill_kv_before_after_fd_import=[KV_BEFORE, KV_AFTER], fd_stop_token=STOP, target_eos=int(target.eos),
    torch=torch.__version__, device=torch.cuda.get_device_name(0), argv=sys.argv, started=time.strftime("%Y-%m-%d %H:%M:%S"),
    clocks=sh("sudo -n /usr/bin/jetson_clocks --show")[-1500:], nvpmodel=sh("nvpmodel -q")), indent=2, ensure_ascii=False))

sampler = PowerSampler()
sampler.start()
time.sleep(0.5)


def load_inp(cid, g):
    return torch.load(INPUTS / f"{cid}_{g:04d}.pt", weights_only=False) if (INPUTS / f"{cid}_{g:04d}.pt").exists() \
        else torch.load(input_path(INPUTS, cid, g), weights_only=False)


# ---------------- A. #0' ----------------
def run_fd_clip(s, cid, rep, fp, frames=None):
    sc = scored_of(s, cid)
    if args.smoke or frames is not None:
        sc = sc[:frames or NSMOKE]
    fill = torch.load(FILL / f"{cid}_{sc[0].frame_idx - 1:04d}.pt", weights_only=False)
    fdm._past_key_values = None              # new stream: the first window takes the full-Prefill path
    fdm.setup_streaming(extended_vision_ranges=_EXTENDED_VISION_RANGES[mc])
    # FlashDrive clears its shape-specific caches only after the 16-image first window and rebuilds them for the
    # 4-image windows.  Clear them before a new stream's first window, then restore the previous stream's 4-image
    # cache objects so that the compiled graphs keep reading the same objects.
    mods = (fdm.vlm.model.visual, fdm.vlm.model.language_model)
    saved = [{k: getattr(m, k) for k in list(vars(m)) if k.startswith("_cached_")} for m in mods]
    for m in mods:
        m.reset_shape_caches()
    pred, _ = fd_call(fd_window(fill, streaming=False))
    assert pred is None, "the fill window did not take the first-window path"
    for m, sv in zip(mods, saved):
        for k, v in sv.items():
            setattr(m, k, v)
    rows = []
    for f in sc:
        pred, t0 = fd_call(fd_window(load_inp(cid, f.frame_idx), streaming=True))
        assert pred is not None
        gen = S["gen"]
        gen = gen[:gen.index(STOP) + 1] if STOP in gen else gen
        dec_ms = (S["t_gen"] - S["t_pf"]) * 1000
        pw = sampler.window(S["t_pf"], S["t_gen"]) or {}
        r = dict(set=s, clip_id=cid, frame_idx=f.frame_idx, rep=rep, decode_ms=round(dec_ms, 3),
                 prefill_ms=round((S["t_pf"] - t0) * 1000, 3), blocks=S.get("blocks", 0), tokens=gen,
                 n_tokens=len(gen), peak_alloc_mib=round(S["peak"] / 2 ** 20, 1),
                 peak_delta_mib=round((S["peak"] - S["m0"]) / 2 ** 20, 1))
        if pw:
            r.update(power_module_w=pw["module_w"], power_gpu_w=pw["gpu_w"], energy_j=round(pw["module_w"] * dec_ms / 1000, 4))
        fp.write(json.dumps(r) + "\n")
        rows.append(r)
    fp.flush()
    return rows


t_all = time.perf_counter()
s0, c0 = PLAN[0]
run_fd_clip(s0, c0, -1, open("/dev/null", "w"), frames=3)            # warm-up and compilation, not recorded
print("FD warm-up %.1f min" % ((time.perf_counter() - t_all) / 60), flush=True)
with (OUT / f"{PFX}fd.jsonl").open("a") as fp:
    for rep, order in ((0, PLAN), (1, PLAN[::-1])):
        if args.smoke and rep == 1:
            break
        for s, cid in order:
            rows = run_fd_clip(s, cid, rep, fp)
            print("[FD r%d %s %s] %d frames, Decode %.0f ms, blocks %.2f  (%.1f min)" % (
                rep, s, cid[:8], len(rows), np.mean([r["decode_ms"] for r in rows]), np.mean([r["blocks"] for r in rows]),
                (time.perf_counter() - t_all) / 60), flush=True)

# ---------------- B, C. #4 in this environment and the output-preserving check ----------------
buf = dec = None
flows = {}


def eq_cache(a, b):
    return all(torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values) for x, y in zip(a.layers, b.layers))


def eq_hidden(a, b):
    ha, hb = a.hidden_states[0], b.hidden_states[0]
    return len(ha) == len(hb) and all(torch.equal(x, y) for x, y in zip(ha, hb))


def decode_s4(fs, v):
    ds = static.with_drive(v.drive_base)                   # outside timing
    sync()
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    res = dec.decode(fs, v.candidates, ds, new_coc=v.newest)
    sync()
    t1 = time.perf_counter()
    pw = sampler.window(t0, t1) or {}
    ms = (t1 - t0) * 1000
    r = dict(ms=round(ms, 3), target_runs=res.target_runs, accepted=res.accepted_initial,
             peak_alloc_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
             peak_delta_mib=round((torch.cuda.max_memory_allocated() - m0) / 2 ** 20, 1))
    if pw:
        r.update(power_module_w=pw["module_w"], power_gpu_w=pw["gpu_w"], energy_j=round(pw["module_w"] * ms / 1000, 4))
    return [int(x) for x in res.tokens], r


nb = 0
with (OUT / f"{PFX}respect.jsonl").open("a") as fp:
    for s, cid in PLAN:
        sc = scored_of(s, cid)[:NSMOKE] if args.smoke else scored_of(s, cid)
        for f in sc:
            inp = load_inp(cid, f.frame_idx)
            prepared, raw = make_inputs(target, inp)
            v = view(SETS[s], cid, f.frame_idx, cfg.num_prev_frames, target.eos)
            sync()
            ta = time.perf_counter()
            pf = prefill(target, prepared)
            sync()
            tb = time.perf_counter()
            with prefill_fast():
                pff = prefill(target, prepared, autocast=False)
            sync()
            tc = time.perf_counter()
            PL = prompt_length(pf)
            pre = dict(kv_equal=eq_cache(pf.past_key_values, pff.past_key_values),
                       first_token_equal=bool(torch.equal(pf.sequences, pff.sequences)),
                       hidden_equal=eq_hidden(pf, pff),
                       prefill_ms=round((tb - ta) * 1000, 3), prefill_fast_ms=round((tc - tb) * 1000, 3))
            del pff
            if buf is None:
                buf = FixedKVBuffer(PL, cfg.buffer_capacities)
                dec = RespectDecoder(target, buf, cfg, draft, mask_emb)
            assert buf.PL == PL
            buf.allocate(pf.past_key_values)
            buf.write_prefill(pf.past_key_values)
            fs = FrameState(prepared, pf, PL, first_token(pf, PL), f.max_new_tokens)
            kernels.install(model, PL)
            decode_s4(fs, v)                                     # per-frame warm-up, as in run_main.py
            runs = [decode_s4(fs, v) for _ in range(2)]
            k, kc = sequential(model, clone_cache(pf.past_key_values), fs.first_token, PL, target.eos,
                               target.offset, target.size, fs.max_tokens)
            kernels.uninstall(model)
            k = [int(x) for x in k]
            if PL not in flows:
                flows[PL] = FlowGraph(model, target.eos, PL, pf.past_key_values.layers, cfg.max_new_tokens)
            sync()
            td = time.perf_counter()
            pg = flows[PL](prepared, raw, kc, k, FLOW_SEED, in_run=False)
            sync()
            te = time.perf_counter()
            pe = np.asarray(trajectory(model, prepared, raw, clone_cache(kc), k, target.eos, FLOW_SEED)[0]).reshape(64, 3)
            sync()
            tf = time.perf_counter()
            row = dict(set=s, clip_id=cid, frame_idx=f.frame_idx, prefill_len=PL, tokens_K=k,
                       s4=[r for _, r in runs], s4_ms_mean2=round((runs[0][1]["ms"] + runs[1][1]["ms"]) / 2, 3),
                       s4_match_K=all(t == k for t, _ in runs), s4_tokens=runs[0][0],
                       match_reference=(k == f.reference_coc),
                       preserve=dict(**pre, traj_equal=bool(np.array_equal(pe.astype(np.float32), pg.astype(np.float32))),
                                     traj_maxabs_m=float(np.abs(pe - pg).max()),
                                     flow_graph_ms=round((te - td) * 1000, 3), flow_public_ms=round((tf - te) * 1000, 3)))
            fp.write(json.dumps(row) + "\n")
            fp.flush()
            nb += 1
            p = row["preserve"]
            print("[#4 %s %s f%d] %.0f ms match K %s | KV %s first %s hidden %s traj %s  (%.1f min)" % (
                s, cid[:8], f.frame_idx, row["s4_ms_mean2"], row["s4_match_K"], p["kv_equal"], p["first_token_equal"],
                p["hidden_equal"], p["traj_equal"], (time.perf_counter() - t_all) / 60), flush=True)
            del pf, kc, prepared, raw, inp
sampler.stop()
print("done", flush=True)
