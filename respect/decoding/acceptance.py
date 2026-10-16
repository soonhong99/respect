"""Greedy token choice and the acceptance walk over a verified trie."""
from __future__ import annotations

from typing import Callable

import torch

from respect.candidates.trie import Trie


def mask_action_vocab(logits: torch.Tensor, offset: int, size: int) -> torch.Tensor:
    """Exclude the trajectory-token range; a CoC contains text tokens only.

    This is the same constraint Alpamayo 1.5 applies during CoC generation.
    """
    masked = logits.float().clone()
    masked[:, offset:offset + size] = float("-inf")
    return masked


def greedy_token(logits: torch.Tensor, offset: int, size: int) -> torch.Tensor:
    """Token choice of the sequential reference, applied row by row."""
    return mask_action_vocab(logits, offset, size).argmax(dim=-1)


def walk(tree: Trie, first: int, cap: int, eos: int,
         pick: Callable[[list[int], int], int]) -> tuple[list[int], list[int], int]:
    """Follow the trie from ``first`` while candidates match the target's choice.

    ``pick(accepted, node)`` returns the target's token after ``node``. The walk
    accepts candidate tokens while they match and stops at the first mismatch or
    when a branch ends; the target's token at that point is the correction token.

    Returns (node path, accepted tokens, correction token).
    """
    path, acc = [], []
    ch, rec = tree.roots.get(first), first
    while ch is not None and len(acc) < cap:
        nd = tree.nodes[ch]
        path.append(ch)
        acc.append(nd.token)
        if nd.token == eos:
            break
        rec = pick(acc, ch)
        ch = nd.children.get(rec)
    return path, acc, rec
