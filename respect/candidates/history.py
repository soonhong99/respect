"""Initial candidates: the CoCs completed in the previous N frames."""
from __future__ import annotations

from typing import Iterable


def strip_eos(tokens: Iterable[int], eos: int) -> tuple[int, ...]:
    out = [int(t) for t in tokens]
    while out and out[-1] == eos:
        out.pop()
    return tuple(out)


def unique_recent(entries: list[tuple[int, tuple[int, ...]]], n: int) -> list[tuple[int, tuple[int, ...]]]:
    """Keep the newest occurrence of each distinct CoC among the last ``n`` frames.

    ``entries`` is a list of (frame index, CoC without EOS) in frame order.
    """
    recent = entries[-n:]
    rev, seen = [], set()
    for f, s in reversed(recent):
        if s and s not in seen:
            rev.append((f, s))
            seen.add(s)
    return list(reversed(rev))


class CocHistory:
    """Rolling record of the CoCs produced by previous frames of one drive."""

    def __init__(self, eos: int) -> None:
        self.eos = eos
        self.entries: list[tuple[int, tuple[int, ...]]] = []

    def append(self, frame_idx: int, tokens: Iterable[int]) -> None:
        self.entries.append((frame_idx, strip_eos(tokens, self.eos)))

    def candidates(self, n: int) -> list[tuple[int, ...]]:
        """Distinct CoCs of the last ``n`` frames, each terminated by EOS."""
        return [tuple(list(s) + [self.eos]) for _, s in unique_recent(self.entries, n)]


def first_token_filter(candidates: list[tuple[int, ...]], first: int) -> list[tuple[int, ...]]:
    """Keep only sentences whose first token equals the token fixed by Prefill.

    A sentence with a different first token can never be accepted and would only
    enlarge the initial trie. When no sentence qualifies, the trie is the single
    first-token node.
    """
    return [s for s in candidates if s and s[0] == first] or [(first,)]
