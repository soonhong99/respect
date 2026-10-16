import numpy as np

from respect.candidates.datastore import Datastore
from respect.candidates.history import CocHistory, first_token_filter
from respect.candidates.trie import build_trie, trie_ancestry
from respect.decoding.acceptance import walk

EOS = 99


def test_shared_prefix_is_merged():
    t = build_trie([(1, 2, 3, EOS), (1, 2, 4, EOS), (1, 2, 3, EOS)])
    assert [n.token for n in t.nodes] == [1, 2, 3, EOS, 4, EOS]
    assert len(t.seqs) == 2


def test_parent_precedes_child_and_ancestry():
    t = build_trie([(1, 2, 3), (1, 5)])
    for i, n in enumerate(t.nodes):
        assert n.parent is None or n.parent < i
    a = trie_ancestry(t)
    # node "5" (index 3) sees itself and the root only, not the sibling branch
    assert np.flatnonzero(a[3]).tolist() == [0, 3]


def test_history_dedup_keeps_newest_and_adds_eos():
    h = CocHistory(EOS)
    for f, s in enumerate([(1, 2), (3, 4), (1, 2), (5,)]):
        h.append(f, list(s) + [EOS])
    assert h.candidates(3) == [(3, 4, EOS), (1, 2, EOS), (5, EOS)]


def test_first_token_filter():
    assert first_token_filter([(1, 2), (3, 4)], 3) == [(3, 4)]
    assert first_token_filter([(1, 2)], 7) == [(7,)]


def test_walk_accepts_until_mismatch():
    t = build_trie([(1, 2, 3, EOS)])
    target = {0: 2, 1: 9}                      # target agrees after node 0, disagrees after node 1
    path, acc, rec = walk(t, 1, 128, EOS, lambda a, ch: target[ch])
    assert acc == [1, 2] and rec == 9 and path == [0, 1]


def test_walk_stops_at_eos():
    t = build_trie([(1, EOS)])
    path, acc, rec = walk(t, 1, 128, EOS, lambda a, ch: EOS)
    assert acc == [1, EOS]


def test_datastore_longest_key_first():
    ds = Datastore([(1, 2, 3, 4), (9, 2, 5)])
    assert ds.lookup([1, 2]) == [(3, 4)]         # 2-token key (1, 2)
    assert ds.lookup([7, 2]) == [(3, 4), (5,)]   # falls back to 1-token key (2,)
    assert ds.lookup([42]) == []


def test_datastore_static_and_drive_counts_add():
    ds = Datastore([(1, 2, 3), (1, 2, 4)]).with_drive([(1, 2, 4)])
    assert ds.lookup([1, 2]) == [(4,), (3,)]            # 4 seen twice (static + drive), 3 once
    ds.add((1, 2, 3))
    ds.add((1, 2, 3))
    assert ds.lookup([1, 2]) == [(3,), (4,)]
    assert ds.without_static().lookup([1, 2]) == [(3,), (4,)]     # drive only: 3 twice, 4 once


def test_datastore_with_drive_leaves_static_untouched():
    base = Datastore([(5, 6, 7)])
    d1 = base.with_drive([(5, 6, 8)])
    d1.add((5, 6, 8))
    assert base.lookup([5, 6]) == [(7,)]
    assert base.with_drive([]).lookup([5, 6]) == [(7,)]
