"""Token-by-token greedy decoding: the reference that RESPECT must reproduce."""
from __future__ import annotations

import copy
from typing import Any

import torch

from respect.decoding.acceptance import greedy_token


def clone_cache(cache: Any) -> Any:
    """Deep copy of a KV cache so that a run never mutates the shared Prefill state."""
    return copy.deepcopy(cache)


def forward_tokens(model: Any, token_ids: torch.Tensor, cache: Any, start_position: int) -> tuple[Any, torch.Tensor]:
    positions = torch.arange(start_position, start_position + token_ids.shape[1],
                             device="cuda", dtype=torch.long)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(input_ids=token_ids, pixel_values=None, past_key_values=cache,
                        cache_position=positions, use_cache=True)
    return out.past_key_values, out.logits.float()


@torch.no_grad()
def sequential(model: Any, cache: Any, token: int, prompt_len: int, eos: int,
               offset: int, size: int, max_tokens: int) -> tuple[list[int], Any]:
    """Generate from ``token`` until EOS or ``max_tokens``, one target run per token.

    ``token`` is the next token to feed (already chosen) and ``cache`` holds the
    KV of everything before it. Returns the generated tokens including ``token``.
    """
    out, pos = [], prompt_len
    while len(out) < max_tokens:
        out.append(token)
        cache, logits = forward_tokens(model, torch.tensor([[token]], device="cuda", dtype=torch.long), cache, pos)
        pos += 1
        if token == eos:
            break
        token = int(greedy_token(logits[:, -1, :], offset, size).item())
    return out, cache
