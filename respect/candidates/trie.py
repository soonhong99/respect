"""Prefix trie over candidate token sequences.

Nodes are appended in insertion order, so a parent always precedes its
children. The verification kernels rely on this order: a node's ancestor set is
complete before the node itself is visited.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class Node:
    token: int
    parent: int | None
    depth: int
    children: OrderedDict[int, int] = field(default_factory=OrderedDict)


@dataclass
class Trie:
    nodes: list[Node]
    roots: OrderedDict[int, int]
    seqs: list[tuple[int, ...]]


def build_trie(seqs: Iterable[tuple[int, ...]]) -> Trie:
    """Merge sequences that share a prefix into one trie. Duplicates are dropped."""
    nodes: list[Node] = []
    roots: OrderedDict[int, int] = OrderedDict()
    unique, seen = [], set()
    for s in seqs:
        if s and s not in seen:
            unique.append(s)
            seen.add(s)
    for s in unique:
        parent = None
        for depth, tok in enumerate(s):
            children = roots if parent is None else nodes[parent].children
            if tok not in children:
                children[tok] = len(nodes)
                nodes.append(Node(tok, parent, depth))
            parent = children[tok]
    return Trie(nodes, roots, unique)


def trie_ancestry(tree: Trie) -> np.ndarray:
    """Boolean matrix whose entry (i, j) is True when node i may attend to node j.

    A node attends to itself and to its ancestors only, so sibling branches never
    see each other.
    """
    q = len(tree.nodes)
    allowed = np.zeros((q, q), dtype=np.bool_)
    for i, node in enumerate(tree.nodes):
        allowed[i, i] = True
        if node.parent is not None:
            # Parents precede children, so the parent's row is already complete.
            allowed[i] |= allowed[node.parent]
    return allowed
