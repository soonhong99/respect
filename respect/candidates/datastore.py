"""Datastore searched by the last one or two committed tokens.

The datastore has two indexes. The static index holds CoCs of drives that are
not evaluated (the development set of the paper) and is built once before
running, like loading weights. The drive index holds the CoCs completed earlier
in the current drive; a new CoC is added when it becomes available. A lookup
sums the continuation counts of both indexes for the longest matching key.

Following REST, keys of length ``gmax`` are tried first and shortened on a miss.
The continuation runs to the end of the stored sentence; the caller trims it to
the node budget. Ties in frequency keep insertion order (static index first).
"""
from __future__ import annotations

import collections
from typing import Iterable


def _index(seqs: Iterable[tuple[int, ...]], gmax: int) -> dict[int, dict]:
    idx = {g: collections.defaultdict(collections.Counter) for g in range(1, gmax + 1)}
    for s in seqs:
        _add(idx, s, gmax)
    return idx


def _add(idx: dict[int, dict], s: tuple[int, ...], gmax: int) -> None:
    for g in range(1, gmax + 1):
        d = idx[g]
        for i in range(len(s) - g):
            d[tuple(s[i:i + g])][tuple(s[i + g:])] += 1


class Datastore:
    def __init__(self, static_seqs: Iterable[tuple[int, ...]] = (), gmax: int = 2,
                 drive_seqs: Iterable[tuple[int, ...]] = ()) -> None:
        self.gmax = gmax
        self.static = _index(static_seqs, gmax)
        self.drive = _index(drive_seqs, gmax)

    @classmethod
    def empty(cls, gmax: int = 2) -> "Datastore":
        return cls((), gmax=gmax)

    def with_drive(self, drive_seqs: Iterable[tuple[int, ...]]) -> "Datastore":
        """Same static index, fresh drive index built from ``drive_seqs`` in order."""
        ds = Datastore.__new__(Datastore)
        ds.gmax, ds.static, ds.drive = self.gmax, self.static, _index(drive_seqs, self.gmax)
        return ds

    def without_static(self) -> "Datastore":
        """Drive index only (the ``same clip`` datastore range compared in the paper)."""
        ds = Datastore.__new__(Datastore)
        ds.gmax, ds.static, ds.drive = self.gmax, _index((), self.gmax), self.drive
        return ds

    def add(self, seq: tuple[int, ...]) -> None:
        """Add one completed CoC to the drive index."""
        _add(self.drive, tuple(seq), self.gmax)

    def lookup(self, prefix: list[int], cmax: int = 64) -> list[tuple]:
        """Continuations after the longest matching key, most frequent first."""
        for g in range(min(self.gmax, len(prefix)), 0, -1):
            key = tuple(prefix[-g:])
            a, b = self.static[g].get(key), self.drive[g].get(key)
            if a or b:
                c = collections.Counter(a) if a else collections.Counter()
                if b:
                    c.update(b)
                return [s for s, _ in c.most_common(cmax)]
        return []

    def num_entries(self) -> int:
        return sum(len(v) for v in self.static[self.gmax].values())
