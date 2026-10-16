"""Trajectory generation from a committed CoC and its KV cache, and trajectory metrics.

``trajectories`` reruns only the second half of Alpamayo 1.5's
``sample_trajectories_from_data_with_vlm_rollout``: the Flow expert reads the
KV cache of the prompt and of every committed CoC token, including EOS, and
samples a trajectory. The diffusion noise is reset to a fixed seed right before
sampling, so two runs with identical KV produce identical trajectories.
"""
from __future__ import annotations

import contextlib
import random
from typing import Any, Iterator

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def fixed_diffusion_noise(model: Any, seed: int) -> Iterator[None]:
    original_sample = model.diffusion.sample

    def wrapped_sample(*args: Any, **kwargs: Any) -> Any:
        seed_everything(seed)
        return original_sample(*args, **kwargs)

    model.diffusion.sample = wrapped_sample
    try:
        yield
    finally:
        model.diffusion.sample = original_sample


def trajectory(model: Any, prepared: dict[str, Any], raw: dict[str, Any], cache: Any,
               tokens: list[int], eos: int, seed: int, num_traj_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sample trajectories conditioned on the committed CoC ``tokens`` and ``cache``."""
    import einops

    device = torch.device("cuda")
    ego_history_xyz = raw["ego_history_xyz"]
    ego_history_rot = raw["ego_history_rot"]
    gen = torch.tensor([tokens], device=device, dtype=torch.long)
    full = torch.cat([prepared["input_ids"], gen], dim=1)
    offset = model._find_eos_offset(sequences=full, eos_token_id=eos, device=device, warn=False)
    rope_deltas = model.vlm.model.rope_deltas
    kv_len = cache.get_seq_length()
    n_diff = model.action_space.get_action_space_dims()[0]
    b_star = num_traj_samples
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=kv_len,
        n_diffusion_tokens=n_diff, b_star=b_star, device=device, prefix_mask=None)
    forward_kwargs: dict[str, Any] = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        emb = model.action_in_proj(x, t)
        if emb.dim() == 2:
            emb = emb.view(b, n_diff, -1)
        out = model.expert(inputs_embeds=emb, position_ids=position_ids, past_key_values=cache,
                           attention_mask=attention_mask, use_cache=True, **forward_kwargs)
        cache.crop(kv_len)
        last = out.last_hidden_state[:, -n_diff:]
        return model.action_out_proj(last).view(-1, *model.action_space.get_action_space_dims())

    with fixed_diffusion_noise(model, seed), torch.autocast("cuda", dtype=torch.bfloat16):
        action = model.diffusion.sample(batch_size=b_star, step_fn=step_fn, device=device, return_all_steps=False)
    hist_xyz = einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=b_star)
    hist_rot = einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=b_star)
    pred_xyz, pred_rot = model.action_space.action_to_traj(action, hist_xyz, hist_rot)
    pred_xyz = einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=1, nj=b_star)
    pred_rot = einops.rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...", ns=1, nj=b_star)
    return pred_xyz.detach().cpu().numpy(), pred_rot.detach().cpu().numpy()


def ground_truth(inp: dict[str, Any]) -> np.ndarray:
    value = inp.get("ego_future_xyz")
    if value is None:
        raise RuntimeError("input has no ego_future_xyz ground truth")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    gt = np.asarray(value, dtype=np.float32).reshape(-1, 3)
    if gt.shape[0] < 64:
        raise RuntimeError(f"expected at least 64 future waypoints, got {gt.shape}")
    return gt[:64]


def min_ade_fde(predictions: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """minADE and minFDE over K sampled trajectories, on the x-y plane.

    ``predictions`` has shape [K, 64, 3]; ADE averages the position error over the
    64 future waypoints (6.4 s).
    """
    if predictions.ndim != 3 or predictions.shape[1:] != (64, 3):
        raise RuntimeError(f"expected predictions [K,64,3], got {predictions.shape}")
    err = np.linalg.norm(predictions[:, :, :2] - gt[None, :, :2], axis=2)
    ade = err.mean(axis=1)
    fde = err[:, -1]
    return {"minADE_m": float(ade.min()), "minFDE_m": float(fde.min())}
