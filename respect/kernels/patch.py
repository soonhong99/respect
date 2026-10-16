"""Install the dedicated kernels into the target language model.

A transformer reduces over three axes only: matrix products reduce over the
input dimension, attention over key positions, and normalization over the
feature dimension. All three are replaced so that every reduction runs in a
fixed order regardless of how many rows (tokens, trie nodes) a call carries:
the seven linear layers of every decoder layer and the vocabulary projection,
the four RMS normalizations of every layer and the final one, and SDPA.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch
import torch.nn as nn

from respect.kernels import dispatch
from respect.kernels.gemm import linear
from respect.kernels.rmsnorm import rmsnorm

GEMM_CFG = dict(BM=32, BN=128, BK=128, num_warps=4, num_stages=3)
LM_OUT_DTYPE = torch.float32  # vocabulary scores are kept in fp32


class FixedLinear(nn.Module):
    def __init__(self, base: nn.Module, out_dtype: str | None = None) -> None:
        super().__init__()
        self.base = base
        self.out_dtype = out_dtype

    def forward(self, x):
        od = LM_OUT_DTYPE if self.out_dtype == "lm" else None
        out = linear(x, self.base.weight, out_dtype=od, **GEMM_CFG)
        if getattr(self.base, "bias", None) is not None:
            out = out + self.base.bias
        return out


class FixedNorm(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x):
        return rmsnorm(x, self.base.weight, self.base.variance_epsilon)


LINEARS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
           ("self_attn", "o_proj"),
           ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
NORMS = [("self_attn", "q_norm"), ("self_attn", "k_norm"),
         (None, "input_layernorm"), (None, "post_attention_layernorm")]
_saved: list = []


def _wrap(parent, name, cls, lm=False):
    base = getattr(parent, name, None)
    if base is None or isinstance(base, cls):
        return False
    _saved.append((parent, name, base))
    setattr(parent, name, cls(base, "lm") if (lm and cls is FixedLinear) else cls(base))
    return True


def is_installed() -> bool:
    return bool(_saved)


def install(model: Any, prompt_len: int) -> int:
    lm = model.vlm.model.language_model if hasattr(model.vlm.model, "language_model") else model.vlm.model
    for layer in lm.layers:
        for pn, cn in LINEARS:
            p = getattr(layer, pn, None)
            if p is not None:
                _wrap(p, cn, FixedLinear)
        for pn, cn in NORMS:
            p = layer if pn is None else getattr(layer, pn, None)
            if p is not None:
                _wrap(p, cn, FixedNorm)
    for nm in ("norm", "final_layernorm"):
        if hasattr(lm, nm) and hasattr(getattr(lm, nm), "variance_epsilon"):
            _wrap(lm, nm, FixedNorm)
    if getattr(model.vlm, "lm_head", None) is not None:
        _wrap(model.vlm, "lm_head", FixedLinear, lm=True)
    dispatch.install(prompt_len)
    return len(_saved)


def uninstall(model: Any) -> None:
    dispatch.uninstall()
    while _saved:
        parent, name, base = _saved.pop()
        setattr(parent, name, base)


@contextlib.contextmanager
def dedicated_kernels(model: Any, prompt_len: int) -> Iterator[None]:
    """Run the enclosed block with the dedicated kernels installed.

    Install after Prefill: Prefill itself runs on the default kernels.
    """
    install(model, prompt_len)
    try:
        yield
    finally:
        uninstall(model)
