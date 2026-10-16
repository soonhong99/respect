"""Fixed-capacity KV buffer shared by initial and follow-up verification.

The buffer holds the Prefill KV followed by ``max(capacities)`` free slots, and
is allocated once and reused across frames (the same preallocation idea as
vLLM). A window view exposes the first ``prompt_len + capacity`` slots to a
forward pass, and new keys are written in place through ``cache_position``.
After each verification the KV entries of the accepted path are moved forward
so that the next forward reads the same values at the same slots as a
sequential run.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import cache_utils
from transformers.cache_utils import DynamicCache, DynamicLayer


class WindowLayer(cache_utils.StaticLayer):
    """One layer of the shared buffer, exposing only its first ``window`` slots."""

    def __init__(self, base_k: torch.Tensor, base_v: torch.Tensor, window: int) -> None:
        super().__init__(max_cache_len=window)
        self._base_k, self._base_v, self._window = base_k, base_v, window
        self.keys = base_k[:, :, :window, :]
        self.values = base_v[:, :, :window, :]
        self.max_batch_size, self.num_heads, _, self.head_dim = base_k.shape
        self.dtype, self.device = base_k.dtype, base_k.device
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        position = cache_kwargs["cache_position"]
        self.keys.index_copy_(2, position, key_states)
        self.values.index_copy_(2, position, value_states)
        return self.keys, self.values


class WindowCache(cache_utils.Cache):
    def __init__(self, layers: list[WindowLayer]) -> None:
        super().__init__(layers=layers)


class FixedKVBuffer:
    def __init__(self, prompt_len: int, capacities: tuple[int, ...]) -> None:
        self.PL = prompt_len
        self.caps = sorted(capacities)
        self.base_k: list[torch.Tensor] = []
        self.base_v: list[torch.Tensor] = []
        self.caches: dict[int, WindowCache] = {}

    def allocate(self, template: Any) -> None:
        """Allocate once, shaped after a Prefill cache. Later calls are no-ops."""
        if self.base_k:
            return
        C = self.PL + self.caps[-1]
        for layer in template.layers:
            k = layer.keys
            shape = (k.shape[0], k.shape[1], C, k.shape[3])
            self.base_k.append(torch.zeros(shape, dtype=k.dtype, device=k.device))
            self.base_v.append(torch.zeros(shape, dtype=k.dtype, device=k.device))
        for cap in self.caps:
            self.caches[cap] = WindowCache(
                [WindowLayer(bk, bv, self.PL + cap) for bk, bv in zip(self.base_k, self.base_v)])

    def write_prefill(self, prefill_cache: Any) -> None:
        """Copy the Prefill KV into the head of the buffer (done once per frame)."""
        PL = self.PL
        for i, layer in enumerate(prefill_cache.layers):
            self.base_k[i][:, :, :PL, :].copy_(layer.keys[..., :PL, :])
            self.base_v[i][:, :, :PL, :].copy_(layer.values[..., :PL, :])

    def window(self, capacity: int) -> WindowCache:
        return self.caches[capacity]

    def fit(self, n: int) -> int:
        """Smallest capacity that holds ``n`` slots."""
        return next(c for c in self.caps if c >= n)

    def view(self, length: int) -> DynamicCache:
        """A cache that sees the first ``length`` slots without copying."""
        c = DynamicCache()
        c.layers = []
        for bk, bv in zip(self.base_k, self.base_v):
            layer = DynamicLayer()
            layer.keys = bk[:, :, :length, :]
            layer.values = bv[:, :, :length, :]
            layer.dtype, layer.device, layer.is_initialized = bk.dtype, bk.device, True
            c.layers.append(layer)
        return c


def compact(cache: Any, prompt_len: int, base: int, slots: list[int]) -> None:
    """Move the KV of the accepted path to consecutive slots starting at ``prompt_len + base``."""
    if not slots:
        return
    dev = cache.layers[0].keys.device
    src = torch.tensor([prompt_len + base + s for s in slots], device=dev)
    dst = torch.arange(prompt_len + base, prompt_len + base + len(slots), device=dev)
    if torch.equal(src, dst):
        return
    for layer in cache.layers:
        layer.keys.index_copy_(2, dst, layer.keys.index_select(2, src))
        layer.values.index_copy_(2, dst, layer.values.index_select(2, src))
