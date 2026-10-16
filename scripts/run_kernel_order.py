#!/usr/bin/env python3
"""Kernel-order check of Section IV-G: same ancestor values, different slots.

The inputs that the layer-0 attention receives at CoC position 25 of one frame
are captured during a sequential run with the dedicated kernels: one query and
26 node keys/values. The same 26 values are then placed in consecutive slots
(0-25) and in scattered slots (0-17 and 200-207) and passed to

  blocked   the development kernel that sums node slots tile by tile
  logical   RESPECT's kernel that accumulates nodes one at a time in logical order

and the two layouts are compared. Five random layouts are checked as well.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from respect.alpamayo.model import load_target, make_inputs, prefill, prompt_length  # noqa: E402
from respect.decoding.sequential import clone_cache, forward_tokens  # noqa: E402
from respect.eval.dataset import input_path, load_frames  # noqa: E402
from respect.kernels import dispatch  # noqa: E402
from respect.kernels.attention import fused_attn_hyb  # noqa: E402
from respect.kernels.patch import dedicated_kernels  # noqa: E402
from respect.kernels.reference_blocked import fused_attn  # noqa: E402

NPAD = 256


def capture(target, inp, tokens, n_node, n_layers):
    """Layer-0 attention inputs at the step whose node segment holds ``n_node`` keys."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import transformers.integrations.sdpa_attention as SA

    model = target.model
    prepared, _ = make_inputs(target, inp)
    pf = prefill(target, prepared)
    PL = prompt_length(pf)
    got, count = {}, {"n": 0}

    def recorder(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        layer = count["n"] % n_layers
        count["n"] += 1
        if layer == 0 and key.shape[2] - PL == n_node and not got:
            got.update(query=query[0].clone(), kp=key[0][:, :PL, :].clone(), vp=value[0][:, :PL, :].clone(),
                       kn=key[0][:, PL:, :].clone(), vn=value[0][:, PL:, :].clone(),
                       scale=float(scaling if scaling is not None else query.shape[-1] ** -0.5))
        return dispatch.unified_forward(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kw)

    with dedicated_kernels(model, PL), torch.no_grad():
        SA.sdpa_attention_forward = recorder
        ALL_ATTENTION_FUNCTIONS["sdpa"] = recorder
        try:
            cache, pos = clone_cache(pf.past_key_values), PL
            for tok in tokens[:n_node]:
                cache, _ = forward_tokens(model, torch.tensor([[tok]], device="cuda"), cache, pos)
                pos += 1
        finally:
            SA.sdpa_attention_forward = dispatch.unified_forward
            ALL_ATTENTION_FUNCTIONS["sdpa"] = dispatch.unified_forward
    return got


def layout(g, slots):
    K, V, q = g["kn"], g["vn"], g["query"]
    HK, _, D = K.shape
    kn = torch.zeros(HK, NPAD, D, device="cuda", dtype=K.dtype)
    vn = torch.zeros(HK, NPAD, D, device="cuda", dtype=V.dtype)
    msk = torch.zeros(64, NPAD, device="cuda", dtype=torch.bool)
    for i, c in enumerate(slots):
        kn[:, c, :] = K[:, i, :]
        vn[:, c, :] = V[:, i, :]
        msk[0, c] = True
    msk[1:, 0] = True
    qq = torch.zeros(q.shape[0], 64, D, device="cuda", dtype=q.dtype)
    qq[:, 0, :] = q[:, 0, :]
    return qq, g["kp"].contiguous(), g["vp"].contiguous(), kn, vn, msk


def run(kernel, g, slots):
    args = layout(g, slots)
    if kernel == "blocked":
        return fused_attn(*args, g["scale"], BM=64, BN=128, num_warps=4, num_stages=1)[:, 0, :]
    return fused_attn_hyb(*args, g["scale"], BM=64, BN=128, BNB=16, num_warps=4, num_stages=1)[:, 0, :]


def diff(a, b):
    return float((a.float() - b.float()).abs().max()), int((a != b).sum()), a.numel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--clip", default="796cef90")
    ap.add_argument("--frame", type=int, default=161)
    ap.add_argument("--n-node", type=int, default=26)
    args = ap.parse_args()

    clips = load_frames(Path(args.data) / "event.jsonl")
    cid = next(c for c in clips if c.startswith(args.clip))
    fr = clips[cid][args.frame]
    target = load_target(local_files_only=True)
    n_layers = len(target.model.vlm.model.language_model.layers)
    inp = torch.load(input_path(Path(args.inputs), cid, fr.frame_idx), weights_only=False)
    g = capture(target, inp, fr.reference_coc, args.n_node, n_layers)
    g = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in g.items()}
    cont = list(range(args.n_node))
    scat = list(range(18)) + list(range(200, 200 + args.n_node - 18))
    print(f"frame {cid[:8]} f{args.frame}, layer 0, {args.n_node} node keys")
    print(f"{'kernel':<10}{'layouts':<34}{'max |diff|':>12}{'differing':>16}")
    for kernel in ("blocked", "logical"):
        a = run(kernel, g, cont)
        m, n, tot = diff(a, run(kernel, g, scat))
        print(f"{kernel:<10}{'consecutive vs 0-17 + 200-207':<34}{m:>12.4e}{n:>10} / {tot}")
        random.seed(0)
        worst = 0.0
        for _ in range(5):
            worst = max(worst, diff(a, run(kernel, g, sorted(random.sample(range(NPAD), args.n_node))))[0])
        print(f"{kernel:<10}{'consecutive vs 5 random layouts':<34}{worst:>12.4e}")


if __name__ == "__main__":
    main()
