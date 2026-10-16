#!/usr/bin/env python3
"""FlashDrive's stages linked with RESPECT Decode in one process, run in real time (Section VI-D).

Same schedule as scripts/run_realtime.py (inputs every 0.1 s on the wall clock, the latest
arrived frame is processed, 10 event clips x 5 start positions).  Runs:
  S4    public Prefill + RESPECT Decode + public Flow (the #4 run in this environment)
  FC8   FlashDrive BF16 as published: full Prefill + DFlash Decode + FlashDrive Flow
        (8 steps, velocity reused in steps 3-6)
  FC10  FC8 with a 10-step Euler Flow (as in the public implementation, no reuse)
  E8    public Prefill + RESPECT Decode + FlashDrive Flow (8 steps)
  E10   E8 with a 10-step Euler Flow
  E10N  E10 with the public checkpoint's Flow weights (expert, action_in_proj, action_out_proj)
  D8    FlashDrive full Prefill + RESPECT Decode + FlashDrive Flow; differs from FC8 only in Decode
  D10N  FlashDrive full Prefill + RESPECT Decode + Flow with the public weights, 10 steps
The paper reports S4, FC8 and D8, and the trajectory difference between S4 and E8 at equal CoC.

Linking
  FlashDrive's streaming Prefill reuses the previous window's KV, which holds only when
  consecutive processed frames are 0.1 s apart; here every cycle prefills all 16 images through
  FlashDrive's compiled path (_encode, _dflash_prefill, torch.compile max-autotune).
  FlashDrive caches keys before RoPE and the public implementation after RoPE, so keys are
  rotated with the same position table when KV moves between the two (float32, then BF16).
  The rotation for D runs counts as Decode time; moving the CoC KV into FlashDrive's cache for
  E and D runs counts as Flow time.  Flow noise is torch.randn after seed 0, as in the public
  implementation.  W4A8 is off; DFlash, expert fusion and torch.compile are on.  The lm_head row
  that FlashDrive adds for its mask token is set to zero (see below).

The public model is loaded before FlashDrive is imported (FlashDrive replaces the Qwen3-VL patch
embedding class on import).  FlashDrive is added to sys.path, not installed.

Output: <out>/cycles.jsonl, <out>/verify.jsonl, <out>/contract.json (smoke_* with --smoke)
Usage:
  python baselines/flashdrive/stage_link.py --inputs REALTIME_INPUTS --flashdrive /path/to/flashdrive \
      --out runs/realtime_flashdrive [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from respect.alpamayo.flow import seed_everything, trajectory  # noqa: E402
from respect.alpamayo.model import first_token, load_target, make_inputs, prefill, prompt_length  # noqa: E402
from respect.candidates.datastore import Datastore  # noqa: E402
from respect.candidates.fd_dflash import find_snapshot, load_fd_dflash  # noqa: E402
from respect.candidates.history import CocHistory, strip_eos  # noqa: E402
from respect.config import RespectConfig  # noqa: E402
from respect.decoding.kv_buffer import FixedKVBuffer  # noqa: E402
from respect.decoding.respect import FrameState, RespectDecoder  # noqa: E402
from respect.decoding.sequential import clone_cache, sequential  # noqa: E402
from respect.eval.dataset import load_frames, static_sequences  # noqa: E402
from respect.eval.realtime import STEP_S, restore  # noqa: E402
from respect.kernels import patch as kernels  # noqa: E402

PHASES = [0, 6, 12, 18, 24]
CONFIGS = ["S4", "FC8", "FC10", "E8", "E10", "E10N", "D8", "D10N"]
FLOW_SEED = 0
FD_MAX_NEW = 128
FAST_STEPS, FAST_CACHE = 8, [3, 4, 5, 6]          # FlashDrive's setting (bench_flashdrive.py)
FULL_STEPS = 10                                    # Euler steps of the public FlowMatching

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data"))
ap.add_argument("--inputs", required=True, help="output directory of scripts/build_realtime_inputs.py")
ap.add_argument("--out", required=True)
ap.add_argument("--flashdrive", required=True, help="FlashDrive checkout (added to sys.path, not installed)")
ap.add_argument("--extra-path", default=None,
                help="optional directory appended to sys.path for FlashDrive's pure-Python dependencies")
ap.add_argument("--fd-model", default="z-lab/Alpamayo-1.5-10B")
ap.add_argument("--smoke", action="store_true", help="path checks on the first clip, then start position 0 only")
args = ap.parse_args()
INPUTS, DATA, FD_SRC, FD_PATH = Path(args.inputs), Path(args.data), Path(args.flashdrive), args.fd_model
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
PFX = "smoke_" if args.smoke else ""


def sync() -> None:
    torch.cuda.synchronize()


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ---------------- public implementation + RESPECT (loaded before FlashDrive is imported) ----------------
cfg = RespectConfig()
target = load_target("nvidia/Alpamayo-1.5-10B", local_files_only=True)
model = target.model
par = next(model.parameters())
draft, mask_emb = load_fd_dflash(find_snapshot(), par.dtype, par.device)
tc = time.perf_counter()
static = Datastore(static_sequences(DATA), gmax=cfg.datastore_key_len)
static_ms = (time.perf_counter() - tc) * 1000
caps = {(cid, i): f.max_new_tokens for cid, fr in load_frames(DATA / "event.jsonl").items() for i, f in fr.items()}

# ---------------- FlashDrive ----------------
sys.path.insert(0, str(FD_SRC))
if args.extra_path:
    sys.path.append(args.extra_path)
import flashdrive  # noqa: E402
from flashdrive import FlashDriveMixin, _EXTENDED_VISION_RANGES, resolve_model_class  # noqa: E402
from flashdrive._backbone import StaticCache  # noqa: E402
from flashdrive._backbone import apply_rotary_pos_emb_single  # noqa: E402,F401
from flashdrive.diffusion import euler_with_cache  # noqa: E402
from alpamayo1_5.models.token_utils import replace_padding_after_eos  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half  # noqa: E402

mc = resolve_model_class(FD_PATH)
mc._supports_sdpa = True
fcfg = mc.config_class.from_pretrained(FD_PATH)
fcfg.attn_implementation = fcfg._attn_implementation = "sdpa"
fdm = mc.from_pretrained(FD_PATH, config=fcfg, dtype=torch.bfloat16).to("cuda").eval()


def weight_check() -> dict:
    """Whether same-named tensors of the two checkpoints are bitwise equal (before FlashDrive's setup)."""
    a = dict(model.named_parameters())
    n_same = n_diff = n_shape = 0
    diff_names = []
    for name, p in fdm.named_parameters():
        q = a.get(name)
        if q is None:
            continue
        if q.shape != p.shape:
            if q.numel() == p.numel() and torch.equal(q.reshape(-1), p.reshape(-1)):
                n_same += 1
            else:
                n_shape += 1
                diff_names.append(name)
            continue
        if torch.equal(q, p):
            n_same += 1
        else:
            n_diff += 1
            diff_names.append(name)
    return dict(same=n_same, differ=n_diff, shape_differ=n_shape, only_in_fd=len(dict(fdm.named_parameters())) - n_same - n_diff - n_shape,
                diff_names=diff_names[:20])


WEIGHTS = weight_check()
print("weights:", {k: v for k, v in WEIGHTS.items() if k != "diff_names"}, WEIGHTS["diff_names"][:5], flush=True)
fdm.__class__ = type("FD", (FlashDriveMixin, mc), {})
fdm.setup_rollout()
fdm.setup_torch_compile("max-autotune")
fdm.setup_streaming(extended_vision_ranges=_EXTENDED_VISION_RANGES[mc])
fdm.setup_expert_fusion()
fdm.setup_dflash(f"{FD_PATH}-DFlash")
# setup_dflash adds one vocabulary entry for the mask token and fills only the new input-embedding row from
# the checkpoint; the new lm_head row keeps its random initial value, so the target can pick the mask token.
# The row is set to zero so that only real vocabulary tokens are chosen.
_lm = fdm.vlm.lm_head.weight
assert _lm.data_ptr() != fdm.vlm.get_input_embeddings().weight.data_ptr(), "lm_head is tied to the input embedding"
assert _lm.shape[0] == fdm._dflash_mask_token_id + 1
with torch.no_grad():
    _lm[fdm._dflash_mask_token_id].zero_()
N_ACT = fdm.num_action_tokens


def _flow_params(src, fused: bool) -> list[torch.Tensor]:
    """Flow weights in FlashDrive's fused layout (fused=True: src is the FlashDrive model)."""
    out = []
    for layer in src.expert.layers:
        at, ml = layer.self_attn, layer.mlp
        if fused:
            out += [at.qkv_proj.weight, ml.gate_up_proj.weight]
        else:
            out += [torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight]), torch.cat([ml.gate_proj.weight, ml.up_proj.weight])]
        names = [n for n, _ in at.named_parameters() if not n.startswith(("q_proj", "k_proj", "v_proj", "qkv_proj"))]
        names_m = [n for n, _ in ml.named_parameters() if not n.startswith(("gate_proj", "up_proj", "gate_up_proj"))]
        out += [dict(at.named_parameters())[n] for n in sorted(names)] + [dict(ml.named_parameters())[n] for n in sorted(names_m)]
        out += [p for n, p in sorted(layer.named_parameters()) if not n.startswith(("self_attn.", "mlp."))]
    for mod in (src.expert.norm, src.action_in_proj, src.action_out_proj):
        out += [p for _, p in sorted(mod.named_parameters())]
    return out


FD_FLOW = [p.detach().clone() for p in _flow_params(fdm, True)]          # z-lab values
NV_FLOW = [p.detach().clone() for p in _flow_params(model, False)]       # nvidia values, fused layout
assert [t.shape for t in FD_FLOW] == [t.shape for t in NV_FLOW]
_flow_now = ["zlab"]


def set_flow_weights(which: str) -> None:
    """Swap FlashDrive's Flow weights in place (the compiled graphs read the same addresses)."""
    if _flow_now[0] == which:
        return
    with torch.no_grad():
        for dst, src in zip(_flow_params(fdm, True), NV_FLOW if which == "nvidia" else FD_FLOW):
            dst.copy_(src)
    sync()
    _flow_now[0] = which
ACT_DIMS = tuple(fdm.action_space.get_action_space_dims())


# ---------------- FlashDrive stages ----------------
def fd_alloc(PL: int) -> None:
    if fdm._past_key_values is not None:
        assert fdm.prefill_seq_length == PL, "the FlashDrive cache assumes one prompt length"
        return
    fdm.prefill_seq_length = PL
    fdm.max_cache_len = PL + FD_MAX_NEW + fdm.dflash_block_size + N_ACT
    fdm._past_key_values = StaticCache(config=fdm.vlm.config, max_cache_len=fdm.max_cache_len, max_batch_size=1)


def fd_positions(ids: torch.Tensor, thw: torch.Tensor) -> torch.Tensor:
    """Position table as in _first_prefill: get_rope_index, extended by one per slot up to max_cache_len."""
    pos, deltas = fdm.vlm.model.get_rope_index(ids, thw, None, None)
    fdm._last_rope_deltas = deltas                       # D runs reuse it as the public model's rope_deltas
    pad = fdm.max_cache_len - ids.shape[1]
    if pad > 0:
        pos = torch.cat([pos, pos[:, :, -1:] + torch.arange(1, pad + 1, device=ids.device)], dim=-1)
    return pos


@torch.inference_mode()
def fd_prefill(prepared: dict) -> tuple:
    """FlashDrive full Prefill of all 16 images (no streaming reuse): compiled _encode and _dflash_prefill."""
    ids, pv, thw = prepared["input_ids"], prepared["pixel_values"], prepared["image_grid_thw"]
    PL = ids.shape[1]
    fd_alloc(PL)
    image_embeds, deepstack = fdm._encode(pv, thw)
    pos = fd_positions(ids, thw)
    fdm.streaming_position_ids = pos
    embeds, image_mask = fdm._embed_tokens_with_images(ids, image_embeds)
    logits, target_hidden = fdm._dflash_prefill(
        inputs_embeds=embeds, position_ids=pos, cache_position=torch.arange(PL, device=ids.device),
        visual_pos_masks=image_mask[..., 0], deepstack_image_embeds=deepstack, streaming_attention_mask=None)
    return logits, target_hidden, pos


@torch.inference_mode()
def fd_decode(ids: torch.Tensor, logits: torch.Tensor, target_hidden: torch.Tensor, pos: torch.Tensor) -> list[int]:
    """The part of _dflash_generate after Prefill (temperature 0).  Returns the CoC tokens + the trajectory-start token."""
    from flashdrive.dflash import sample_tokens
    device = ids.device
    PL = fdm.prefill_seq_length
    max_length = PL + FD_MAX_NEW
    out_ids = torch.full((1, max_length + fdm.dflash_block_size), fdm._dflash_mask_token_id, dtype=torch.long, device=device)
    out_ids[:, :PL] = fdm.tokenizer.pad_token_id
    traj_tok = torch.tensor([[fdm.traj_start_token_id]], device=device)
    if not fdm.has_compiled_step("dflash_traj_forward"):
        logits, target_hidden = logits.clone(), target_hidden.clone()
        fdm._dflash_traj_forward(traj_tok, pos, torch.tensor([PL], device=device))
    first = sample_tokens(logits.unsqueeze(1), fdm._dflash_traj_mask, 0.0, 1.0)
    out_ids[:, PL:PL + 1] = first
    out_ids, cur_len, traj_cached = fdm._dflash_decode_loop(
        output_ids=out_ids, num_input_tokens=PL, position_ids=pos, target_hidden=target_hidden,
        max_new_tokens=FD_MAX_NEW, temperature=0.0, top_p=1.0)
    if not traj_cached:
        fdm._dflash_traj_forward(traj_tok, pos, torch.tensor([cur_len], device=device, dtype=torch.long))
    gen = out_ids[0, PL:].tolist()
    return [int(x) for x in gen] + [int(fdm.traj_start_token_id)]


@torch.inference_mode()
def fd_action(action_start: int, raw: dict, steps: int) -> np.ndarray:
    """FlashDrive Flow.  steps=8: FlashDrive's setting (velocity reused in steps 3-6); steps=10: Euler as in the public FlowMatching."""
    device = torch.device("cuda")
    idx = torch.arange(fdm.max_cache_len, device=device).unsqueeze(0)
    keep = (idx < action_start) | ((idx >= action_start) & (idx < action_start + N_ACT))
    attn = torch.where(keep[:, None, None, :], 0.0, torch.finfo(torch.bfloat16).min).to(torch.bfloat16)   # query dtype
    cache_position = torch.arange(action_start, action_start + N_ACT, device=device)
    seed_everything(FLOW_SEED)
    noise = torch.randn(1, *ACT_DIMS, device=device)        # the same noise as the public _euler
    if steps == FAST_STEPS:
        noise = noise.to(torch.bfloat16)
    ekw = {"is_causal": False} if fdm.config.expert_non_causal_attention else {}

    def action_fn(b: dict) -> torch.Tensor:
        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            emb = fdm.action_in_proj(x, t)
            if emb.dim() == 2:
                emb = emb.view(x.shape[0], N_ACT, -1)
            h = fdm.expert(inputs_embeds=emb, position_ids=b["position_ids"], past_key_values=fdm._past_key_values,
                           attention_mask=b["attention_mask"], cache_position=b["cache_position"], use_cache=True,
                           **ekw).last_hidden_state[:, -N_ACT:]
            return fdm.action_out_proj(h).view(-1, *ACT_DIMS)

        if steps == FAST_STEPS:
            return euler_with_cache(fdm.diffusion, noise=b["noise"], batch_size=1, step_fn=step_fn, device=device,
                                    cache_steps=FAST_CACHE, inference_step=FAST_STEPS)
        x = b["noise"]
        ts = torch.linspace(0.0, 1.0, FULL_STEPS + 1, device=device)
        nd = len(ACT_DIMS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(FULL_STEPS):
                dt = (ts[i + 1] - ts[i]).view(1, *[1] * nd).expand(1, *[1] * nd)
                t0 = ts[i].view(1, *[1] * nd).expand(1, *[1] * nd)
                x = x + dt * step_fn(x, t0)
        return x

    act = fdm._compiled_step(f"action_{steps}", {"position_ids": fdm.streaming_position_ids, "attention_mask": attn,
                                                  "cache_position": cache_position, "noise": noise}, action_fn)
    hx = raw["ego_history_xyz"][:, -1]
    hr = raw["ego_history_rot"][:, -1]
    pred_xyz, _ = fdm.action_space.action_to_traj(act, hx, hr)            # sample dtype kept, as in FlashDrive
    return pred_xyz.detach().cpu().numpy().reshape(64, 3)


@torch.inference_mode()
def to_fd_cache(buf: FixedKVBuffer, length: int, pos: torch.Tensor, start: int = 0) -> None:
    """Copy the first ``length`` slots of the RESPECT buffer into the FlashDrive cache, undoing RoPE on keys.

    The public implementation stores keys after RoPE (cos, sin in BF16); FlashDrive stores keys before RoPE
    and applies the same position table at every attention.  k = k_r cos - rotate_half(k_r) sin in float32.
    """
    dummy = torch.zeros(1, 1, 1, device=pos.device, dtype=torch.bfloat16)
    cos, sin = fdm.vlm.model.language_model.rotary_emb(dummy, pos)       # [1, max_cache_len, D], BF16
    cos = cos[:, start:length].float().unsqueeze(1)
    sin = sin[:, start:length].float().unsqueeze(1)
    for bk, bv, layer in zip(buf.base_k, buf.base_v, fdm._past_key_values.layers):
        kr = bk[:, :, start:length, :].float()
        layer.keys[:1, :, start:length, :].copy_((kr * cos - rotate_half(kr) * sin).to(layer.keys.dtype))
        layer.values[:1, :, start:length, :].copy_(bv[:, :, start:length, :])


@torch.inference_mode()
def from_fd_prefill(buf: FixedKVBuffer, PL: int, pos: torch.Tensor) -> None:
    """Copy FlashDrive's Prefill KV (keys before RoPE) into the RESPECT buffer: k_r = k cos + rotate_half(k) sin."""
    dummy = torch.zeros(1, 1, 1, device=pos.device, dtype=torch.bfloat16)
    cos, sin = fdm.vlm.model.language_model.rotary_emb(dummy, pos)
    cos = cos[:, :PL].float().unsqueeze(1)
    sin = sin[:, :PL].float().unsqueeze(1)
    for bk, bv, layer in zip(buf.base_k, buf.base_v, fdm._past_key_values.layers):
        k = layer.keys[:1, :, :PL, :].float()
        bk[:, :, :PL, :].copy_((k * cos + rotate_half(k) * sin).to(bk.dtype))
        bv[:, :, :PL, :].copy_(layer.values[:1, :, :PL, :])


class FDPrefill:
    """What RESPECT reads from a Prefill result (first token, FD-DFlash context), filled from FlashDrive's Prefill."""

    def __init__(self, PL: int, first: int, context: torch.Tensor) -> None:
        self.past_key_values = None
        self.sequences = torch.full((1, PL + 1), first, dtype=torch.long, device=context.device)
        layers = sorted(draft.config.target_layer_ids)              # FlashDrive concatenates them in layer order
        chunks = context.chunk(len(layers), dim=-1)
        step = [None] * (max(layers) + 2)
        for li, c in zip(layers, chunks):
            step[li + 1] = c
        self.hidden_states = (tuple(step),)


@torch.no_grad()
def d_reference(prepared: dict, cap: int) -> list[int]:
    """Reference of the D runs: sequential decoding with the dedicated kernels on the same FlashDrive Prefill KV."""
    from flashdrive.dflash import sample_tokens
    logits, th, pos = fd_prefill(prepared)
    PL = fdm.prefill_seq_length
    first = int(sample_tokens(logits.clone().unsqueeze(1), fdm._dflash_traj_mask, 0.0, 1.0)[0, 0])
    from_fd_prefill(state["buf"], PL, pos)
    model.vlm.model.rope_deltas = fdm._last_rope_deltas
    kernels.install(model, PL)
    ref, _c = sequential(model, clone_cache(state["buf"].view(PL)), first, PL, target.eos, target.offset, target.size, cap)
    kernels.uninstall(model)
    return [int(x) for x in ref]


# ---------------- one cycle ----------------
state = dict(buf=None, dec=None)


def respect_decode(fs: FrameState, run: dict, kv_written: bool = False) -> tuple[list[int], dict]:
    if state["buf"] is None:
        state["buf"] = FixedKVBuffer(fs.prompt_len, cfg.buffer_capacities)
        state["dec"] = RespectDecoder(target, state["buf"], cfg, draft, mask_emb)
    buf, dec = state["buf"], state["dec"]
    assert buf.PL == fs.prompt_len
    if not kv_written:
        buf.allocate(fs.prefill.past_key_values)
        buf.write_prefill(fs.prefill.past_key_values)
    else:
        assert buf.base_k, "D runs need the buffer allocated (by the S4 warm-up)"
    kernels.install(model, fs.prompt_len)
    res = dec.decode(fs, run["history"].candidates(cfg.num_prev_frames), run["datastore"], new_coc=run["newest"])
    run["newest"] = None
    kernels.uninstall(model)
    info = dict(accepted=res.accepted_initial, rounds=res.follow_rounds, target_runs=res.target_runs,
                ds_insert_ms=res.datastore_insert_ms)
    return [int(x) for x in res.tokens], info


@torch.no_grad()
def cycle(name: str, cid: str, g: int, small: dict, run: dict, T0: float) -> dict:
    r = dict(cfg=name, g=g)
    cap = caps.get((cid, g), cfg.max_new_tokens)
    ts = time.perf_counter()
    prepared, raw = make_inputs(target, restore(small, g))
    sync()
    t_in = time.perf_counter()
    xfer_ms = None
    if name.startswith("D"):
        logits, th, pos = fd_prefill(prepared)
        logits, th = logits.clone(), th.clone()
        sync()
        t_pf = time.perf_counter()
        PL = fdm.prefill_seq_length
        from flashdrive.dflash import sample_tokens
        first = int(sample_tokens(logits.unsqueeze(1), fdm._dflash_traj_mask, 0.0, 1.0)[0, 0])
        tr = time.perf_counter()
        from_fd_prefill(state["buf"], PL, pos)
        model.vlm.model.rope_deltas = fdm._last_rope_deltas
        sync()
        r["kv_rope_ms"] = round((time.perf_counter() - tr) * 1000, 3)
        fs = FrameState(prepared, FDPrefill(PL, first, th), PL, first, cap)
        tk, info = respect_decode(fs, run, kv_written=True)
        r.update(info)
        sync()
        t_dec = time.perf_counter()
        tx = time.perf_counter()
        to_fd_cache(state["buf"], PL + len(tk), pos, start=PL)
        sync()
        xfer_ms = (time.perf_counter() - tx) * 1000
        set_flow_weights("nvidia" if name == "D10N" else "zlab")
        px = fd_action(PL + len(tk), raw, FAST_STEPS if name == "D8" else FULL_STEPS)
        run["history"].append(g, tk)
        run["newest"] = tuple(list(strip_eos(tk, target.eos)) + [target.eos])
    elif name.startswith("FC"):
        logits, th, pos = fd_prefill(prepared)
        sync()
        t_pf = time.perf_counter()
        tk = fd_decode(prepared["input_ids"], logits, th, pos)
        sync()
        t_dec = time.perf_counter()
        PL = fdm.prefill_seq_length
        set_flow_weights("zlab")
        px = fd_action(PL + len(tk), raw, FAST_STEPS if name == "FC8" else FULL_STEPS)
    else:
        assert not kernels.is_installed()
        pf = prefill(target, prepared)
        sync()
        t_pf = time.perf_counter()
        PL = prompt_length(pf)
        fs = FrameState(prepared, pf, PL, first_token(pf, PL), cap)
        tk, info = respect_decode(fs, run)
        r.update(info)
        sync()
        t_dec = time.perf_counter()
        if name == "S4":
            px = trajectory(model, prepared, raw, state["buf"].view(PL + len(tk)), tk, target.eos, FLOW_SEED)[0]
            px = np.asarray(px).reshape(64, 3)
        else:
            fd_alloc(PL)
            pos = fd_positions(prepared["input_ids"], prepared["image_grid_thw"])
            fdm.streaming_position_ids = pos
            tx = time.perf_counter()
            to_fd_cache(state["buf"], PL + len(tk), pos)
            sync()
            xfer_ms = (time.perf_counter() - tx) * 1000
            set_flow_weights("nvidia" if name == "E10N" else "zlab")
            px = fd_action(PL + len(tk), raw, FAST_STEPS if name == "E8" else FULL_STEPS)
        run["history"].append(g, tk)
        run["newest"] = tuple(list(strip_eos(tk, target.eos)) + [target.eos])
    sync()
    t_fl = time.perf_counter()
    r.update(t_start=ts - T0, t_input=t_in - T0, t_prefill=t_pf - T0, t_decode=t_dec - T0, t_pub=t_fl - T0,
             input_ms=(t_in - ts) * 1000, prefill_ms=(t_pf - t_in) * 1000, decode_ms=(t_dec - t_pf) * 1000,
             flow_ms=(t_fl - t_dec) * 1000, cycle_ms=(t_fl - ts) * 1000, prefill_len=int(PL), cap=cap,
             tokens=tk, n_tokens=len(tk), traj=np.round(px, 4).tolist())
    if xfer_ms is not None:
        r["kv_transfer_ms"] = round(xfer_ms, 3)
    for k in ("input_ms", "prefill_ms", "decode_ms", "flow_ms", "cycle_ms", "t_start", "t_input", "t_prefill", "t_decode", "t_pub"):
        r[k] = round(r[k], 4)
    return r


def new_run() -> dict:
    return dict(history=CocHistory(target.eos), datastore=static.with_drive([]), newest=None)


def load_clip(cid: str):
    meta = json.loads((INPUTS / cid / "meta.json").read_text())
    small = {g: torch.load(INPUTS / cid / "frames" / f"{g:04d}.pt", weights_only=False)
             for g in range(meta["g_lo"], meta["g_last"] + 1)}
    return meta, small


def replay(name: str, cid: str, meta: dict, small: dict, phase: int) -> list[dict]:
    g0, g_end = meta["g_first_run"] + phase, meta["g_last"]
    run, rows, last = new_run(), [], None
    T0 = time.perf_counter()
    while True:
        g = min(g0 + int(math.floor((time.perf_counter() - T0) / STEP_S + 1e-9)), g_end)
        if last is not None and g <= last:
            if last >= g_end:
                return rows
            time.sleep(max(0.0, T0 + (last + 1 - g0) * STEP_S - time.perf_counter()))
            continue
        r = cycle(name, cid, g, small, run, T0)
        r.update(clip_id=cid, phase=phase, g0=g0, g_end=g_end, e=meta["event_start_frame"],
                 k=(g - last) if last is not None else None, t_arrival=round((g - g0) * STEP_S, 4))
        rows.append(r)
        last = g


# ---------------- contract ----------------
clips = sorted(p.name for p in INPUTS.iterdir() if (p / "meta.json").exists())
assert len(clips) == 10, clips
run_clips = clips[:1] if args.smoke else clips
phases = [0] if args.smoke else PHASES
(OUT / f"{PFX}contract.json").write_text(json.dumps(dict(
    purpose="FlashDrive stages linked with RESPECT Decode, real time", configs=CONFIGS, phases=phases, clips=run_clips,
    step_s=STEP_S, flow_seed=FLOW_SEED, fd_model=FD_PATH, fd_max_new_tokens=FD_MAX_NEW,
    fast_flow=dict(steps=FAST_STEPS, cache_steps=FAST_CACHE), full_flow=dict(steps=FULL_STEPS, method="euler"),
    respect_config=cfg.__dict__, static_datastore_build_ms=round(static_ms, 1), weights=WEIGHTS,
    hashes={"script": sha(__file__), "flashdrive/_base.py": sha(FD_SRC / "flashdrive/_base.py"),
            "flashdrive/dflash.py": sha(FD_SRC / "flashdrive/dflash.py"), "flashdrive/_backbone.py": sha(FD_SRC / "flashdrive/_backbone.py")},
    flashdrive_git=sh(f"git -C {FD_SRC} rev-parse HEAD"), respect_git=sh(f"git -C {Path(__file__).resolve().parents[2]} rev-parse HEAD"),
    torch=torch.__version__, device=torch.cuda.get_device_name(0), argv=sys.argv, started=time.strftime("%Y-%m-%d %H:%M:%S"),
    clocks=sh("sudo -n /usr/bin/jetson_clocks --show")[-1500:], nvpmodel=sh("nvpmodel -q")), indent=2, ensure_ascii=False))

# ---------------- warm-up (not recorded, includes compilation) ----------------
meta0, small0 = load_clip(clips[0])
tw = time.perf_counter()
for rep in range(2):
    for name in CONFIGS:
        t1 = time.perf_counter()
        cycle(name, clips[0], meta0["g_first_run"] + rep, small0, new_run(), time.perf_counter())
        print("warm-up %s rep%d %.1f s" % (name, rep, time.perf_counter() - t1), flush=True)
print("warm-up total %.1f min" % ((time.perf_counter() - tw) / 60), flush=True)

# ---------------- smoke checks (not recorded) ----------------
if args.smoke:
    torch.cuda.reset_peak_memory_stats()
    meta = meta0
    PLs = set()
    for g in range(meta["g_first_run"], meta["g_last"] + 1, 10):
        p, _r = make_inputs(target, restore(small0, g))
        PLs.add(int(p["input_ids"].shape[1]))
    fck = open(OUT / "smoke_check.jsonl", "w")
    e = meta["event_start_frame"]
    for g in (meta["g_first_run"], e - 20, e, e + 10):
        prepared, raw = make_inputs(target, restore(small0, g))
        rec = dict(clip_id=clips[0], g=g)
        # sequential decoding with the dedicated kernels (reference CoC) and the public Flow
        pf = prefill(target, prepared)
        PL = prompt_length(pf)
        cap = caps.get((clips[0], g), cfg.max_new_tokens)
        kernels.install(model, PL)
        tk_k, cache_k = sequential(model, clone_cache(pf.past_key_values), first_token(pf, PL), PL, target.eos,
                                   target.offset, target.size, cap)
        kernels.uninstall(model)
        tk_k = [int(x) for x in tk_k]
        px_eager = np.asarray(trajectory(model, prepared, raw, cache_k, tk_k, target.eos, FLOW_SEED)[0]).reshape(64, 3)
        # RESPECT with an empty history (no initial candidates): same CoC?
        fs = FrameState(prepared, pf, PL, first_token(pf, PL), cap)
        tk_r, _info = respect_decode(fs, new_run())
        rec["respect_eq_k"] = tk_r == tk_k
        # E: RESPECT KV -> FlashDrive cache -> FlashDrive Flow
        fd_alloc(PL)
        pos = fd_positions(prepared["input_ids"], prepared["image_grid_thw"])
        fdm.streaming_position_ids = pos
        set_flow_weights("zlab")
        to_fd_cache(state["buf"], PL + len(tk_r), pos)
        unroped = [layer.keys[:1, :, :PL, :].clone() for layer in fdm._past_key_values.layers]
        px_e8 = fd_action(PL + len(tk_r), raw, FAST_STEPS)
        to_fd_cache(state["buf"], PL + len(tk_r), pos)
        px_e10 = fd_action(PL + len(tk_r), raw, FULL_STEPS)
        set_flow_weights("nvidia")
        to_fd_cache(state["buf"], PL + len(tk_r), pos)
        px_e10n = fd_action(PL + len(tk_r), raw, FULL_STEPS)
        set_flow_weights("zlab")
        # FC: FlashDrive's own Prefill and Decode
        logits, th, pos2 = fd_prefill(prepared)
        fd_keys = [layer.keys[:1, :, :PL, :].float() for layer in fdm._past_key_values.layers]
        kd = [((u.float() - f).abs().max().item(), ((u.float() - f).norm() / f.norm()).item()) for u, f in zip(unroped, fd_keys)]
        tk_fd = fd_decode(prepared["input_ids"], logits, th, pos2)
        px_fc8 = fd_action(PL + len(tk_fd), raw, FAST_STEPS)
        rec.update(fd_input_ids_eq=bool(torch.equal(prepared["input_ids"], fdm.fuse_traj_tokens(
            raw["tokenized_data"]["input_ids"],
            {"ego_history_xyz": raw["ego_history_xyz"], "ego_history_rot": raw["ego_history_rot"]}))),
                   fd_coc_eq_k=strip_eos(tk_fd, target.eos) == strip_eos(tk_k, target.eos),
                   n_tokens_k=len(tk_k), n_tokens_fd=len(tk_fd),
                   unrope_key_maxabs=max(a for a, _ in kd), unrope_key_rel=max(b for _, b in kd),
                   traj_dev_m=dict(E8=float(np.abs(px_e8 - px_eager).max()), E10=float(np.abs(px_e10 - px_eager).max()),
                                   E10N=float(np.abs(px_e10n - px_eager).max()),
                                   FC8=float(np.abs(px_fc8 - px_eager).max())),
                   traj_end_dev_m=dict(E8=float(np.linalg.norm(px_e8[-1, :2] - px_eager[-1, :2])),
                                       E10=float(np.linalg.norm(px_e10[-1, :2] - px_eager[-1, :2])),
                                       E10N=float(np.linalg.norm(px_e10n[-1, :2] - px_eager[-1, :2]))))
        # D: FlashDrive Prefill + RESPECT Decode (empty history); reference on the same KV
        rd = cycle("D8", clips[0], g, small0, new_run(), time.perf_counter())
        ref_d = d_reference(prepared, cap)
        rec.update(d_eq_ref=rd["tokens"] == ref_d, d_eq_fd=strip_eos(rd["tokens"], target.eos) == strip_eos(tk_fd, target.eos),
                   d_eq_k=rd["tokens"] == tk_k, d_kv_rope_ms=rd.get("kv_rope_ms"),
                   traj_dev_d8_vs_fc8_m=float(np.abs(np.asarray(rd["traj"]) - px_fc8).max()))
        if rec["fd_coc_eq_k"]:
            # same CoC: trajectories from FlashDrive's KV and from the RESPECT KV (only Prefill numerics differ)
            rec["traj_dev_fc8_vs_e8_m"] = float(np.abs(px_fc8 - px_e8).max())
        print("check", json.dumps(rec), flush=True)
        fck.write(json.dumps(rec) + "\n")
    fck.write(json.dumps(dict(prompt_lengths=sorted(PLs), peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))) + "\n")
    fck.close()
    print("prompt lengths", sorted(PLs), flush=True)

# ---------------- real-time runs ----------------
fp = (OUT / f"{PFX}cycles.jsonl").open("a")
t_all = time.perf_counter()
for ci, cid in enumerate(run_clips):
    meta, small = (meta0, small0) if ci == 0 else load_clip(cid)
    for pi, p in enumerate(phases):
        k = (ci * len(phases) + pi) % len(CONFIGS)
        for name in CONFIGS[k:] + CONFIGS[:k]:
            try:
                rows = replay(name, cid, meta, small, p)
            except Exception:
                if kernels.is_installed():
                    kernels.uninstall(model)
                (OUT / f"{PFX}errors.log").open("a").write("=== %s p%d %s ===\n%s\n" % (cid, p, name, traceback.format_exc()))
                print("  failed %s p%d %s" % (cid[:8], p, name), flush=True)
                continue
            for r in rows:
                fp.write(json.dumps(r) + "\n")
            fp.flush()
            print("[%s p%d %s] %d cycles, cycle %.0f ms (prefill %.0f, decode %.0f, flow %.0f), frames %d..%d (%.1f min)" % (
                cid[:8], p, name, len(rows), np.mean([r["cycle_ms"] for r in rows]), np.mean([r["prefill_ms"] for r in rows]),
                np.mean([r["decode_ms"] for r in rows]), np.mean([r["flow_ms"] for r in rows]), rows[0]["g"], rows[-1]["g"],
                (time.perf_counter() - t_all) / 60), flush=True)
    if ci > 0:
        del small
fp.close()
print("peak alloc %.2f GiB" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)

# ---------------- checks (not recorded, full run only) ----------------
# Runs with RESPECT Decode are compared with sequential decoding with the dedicated kernels on the same Prefill
# KV; FlashDrive runs (FC8, FC10) with the CoC of the same frame decoded again by FlashDrive (repeatability).
if not args.smoke:
    rows = [json.loads(x) for x in (OUT / "cycles.jsonl").read_text().splitlines()]
    kind_of = lambda c: "FC" if c.startswith("FC") else ("D" if c.startswith("D") else "R")
    todo = sorted({(r["clip_id"], r["g"], kind_of(r["cfg"])) for r in rows})
    bad, cur, small = 0, None, None
    with (OUT / "verify.jsonl").open("w") as fv:
        for cid, g, kind in todo:
            if cid != cur:
                _m, small = load_clip(cid)
                cur = cid
            prepared, raw = make_inputs(target, restore(small, g))
            if kind == "FC":
                logits, th, pos = fd_prefill(prepared)
                ref = fd_decode(prepared["input_ids"], logits, th, pos)
            elif kind == "D":
                ref = d_reference(prepared, caps.get((cid, g), cfg.max_new_tokens))
            else:
                pf = prefill(target, prepared)
                PL = prompt_length(pf)
                kernels.install(model, PL)
                ref, _c = sequential(model, clone_cache(pf.past_key_values), first_token(pf, PL), PL, target.eos,
                                     target.offset, target.size, caps.get((cid, g), cfg.max_new_tokens))
                kernels.uninstall(model)
                ref = [int(x) for x in ref]
            got = [r["tokens"] for r in rows if r["clip_id"] == cid and r["g"] == g and kind_of(r["cfg"]) == kind]
            ok = all(x == ref for x in got)
            bad += not ok
            fv.write(json.dumps(dict(kind=kind, clip_id=cid, g=g, n_runs=len(got), ok=ok, ref_tokens=ref)) + "\n")
    print("verify: %d (clip, frame, kind), %d mismatches" % (len(todo), bad), flush=True)
