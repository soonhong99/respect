"""One target run that verifies every node of a trie."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from respect.candidates.trie import Trie, trie_ancestry


@torch.no_grad()
def verify_trie(model: Any, prepared: dict, cache: Any, tree: Trie,
                prompt_len: int, base: int, capacity: int) -> Any:
    """Verify all trie nodes in one forward pass on the fixed-capacity cache.

    Node ``i`` is written to slot ``prompt_len + base + i`` (its physical slot)
    while its rotary position follows its depth in the trie (its logical
    position). The mask opens the prompt and the committed tokens to every
    node, and opens each node only to its ancestors.
    """
    dev = prepared["input_ids"].device
    q = len(tree.nodes)
    ids = torch.tensor([[n.token for n in tree.nodes]], device=dev, dtype=torch.long)
    start = prompt_len + base
    cache_position = torch.arange(start, start + q, device=dev)

    rope = getattr(model.vlm.model, "rope_deltas", None)
    if rope is None:
        raise RuntimeError("rope_deltas is missing; verify only after Prefill")
    delta = torch.tensor([[start]], device=dev, dtype=torch.long) + rope.to(device=dev, dtype=torch.long)
    depth = torch.tensor([n.depth for n in tree.nodes], device=dev, dtype=torch.long)
    logical = (depth.view(1, 1, q) + delta.view(1, 1, 1)).expand(3, -1, -1)

    dt = torch.bfloat16
    neg = torch.finfo(dt).min
    m = torch.full((1, 1, q, prompt_len + capacity), neg, device=dev, dtype=dt)
    m[..., :start] = 0.0
    anc = torch.from_numpy(np.asarray(trie_ancestry(tree))).to(dev)
    m[0, 0, :, start:start + q] = torch.where(
        anc, torch.zeros((), device=dev, dtype=dt), torch.full((), neg, device=dev, dtype=dt))
    with torch.autocast("cuda", dtype=dt):
        return model.vlm(input_ids=ids, pixel_values=None, past_key_values=cache,
                         cache_position=cache_position, position_ids=logical,
                         attention_mask=m, use_cache=True, return_dict=True,
                         output_hidden_states=True)
