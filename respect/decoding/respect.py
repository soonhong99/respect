"""RESPECT decoding: initial verification of previous CoCs, then follow-up verification.

Initial verification merges the distinct CoCs of the previous N frames whose
first token equals the token fixed by Prefill into one trie and verifies it with
one target run. If EOS is accepted the CoC is complete. Otherwise follow-up
verification repeats: FD-DFlash and datastore candidates rooted at the
correction token are merged into one trie of at most ``node_budget`` nodes and
verified, until EOS is accepted.

All verification runs on one fixed-capacity KV buffer. After every run only the
KV of the accepted path is kept, in logical order, so the next run reads the
same values at the same slots as a sequential run. With the dedicated kernels
installed (``respect.kernels.dedicated_kernels``) the output tokens and KV are
bitwise identical to ``sequential`` under the same kernels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from respect.candidates.datastore import Datastore
from respect.candidates.fd_dflash import FDDFlash
from respect.candidates.history import first_token_filter
from respect.candidates.trie import build_trie
from respect.config import RespectConfig
from respect.decoding.acceptance import greedy_token, walk
from respect.decoding.kv_buffer import FixedKVBuffer, compact
from respect.decoding.sequential import sequential
from respect.decoding.verify import verify_trie


@dataclass
class FrameState:
    """Per-frame inputs shared by every decoding variant."""
    prepared: dict
    prefill: Any            # output of respect.alpamayo.model.prefill
    prompt_len: int
    first_token: int
    max_tokens: int


@dataclass
class DecodeResult:
    tokens: list[int]
    target_runs: int
    accepted_initial: int
    initial_nodes: int
    initial_candidates: int
    follow_rounds: int = 0
    round_nodes: list[int] = field(default_factory=list)
    round_accepted: list[int] = field(default_factory=list)
    initial_ms: float = 0.0
    follow_ms: float = 0.0
    draft_ms: float = 0.0
    lookup_ms: float = 0.0
    datastore_insert_ms: float = 0.0
    verify_ms: float = 0.0


class RespectDecoder:
    def __init__(self, target: Any, buffer: FixedKVBuffer, cfg: RespectConfig = RespectConfig(),
                 draft: Any = None, mask_emb: torch.Tensor | None = None) -> None:
        self.t = target
        self.model = target.model
        self.buf = buffer
        self.cfg = cfg
        self.draft, self.mask_emb = draft, mask_emb

    # ------------------------------------------------------------------ initial
    def _initial(self, fs: FrameState, candidates: list[tuple[int, ...]]):
        cands = first_token_filter(candidates, fs.first_token)
        tree = build_trie(cands)
        q = len(tree.nodes)
        cap = self.buf.fit(q)
        cache = self.buf.window(cap)
        out = verify_trie(self.model, fs.prepared, cache, tree, fs.prompt_len, 0, cap)
        am = greedy_token(out.logits[0], self.t.offset, self.t.size).tolist()
        path, acc, rec = walk(tree, fs.first_token, fs.max_tokens, self.t.eos, lambda a, ch: am[ch])
        compact(cache, fs.prompt_len, 0, path)
        return acc, rec, q, len(cands), out.hidden_states, path

    # ---------------------------------------------------------------- follow-up
    @torch.no_grad()
    def _follow(self, fs: FrameState, committed: list[int], root: int,
                datastore: Datastore, drafter: FDDFlash | None, res: DecodeResult,
                draft_nodes: int) -> list[int]:
        # 192 slots for budgets up to 64; a larger budget needs max_tokens + budget slots
        capacity = self.buf.fit(max(self.cfg.follow_capacity, fs.max_tokens + self.cfg.node_budget))
        cache = self.buf.window(capacity)
        tokens, cur = list(committed), root
        cap_, PL, eos = fs.max_tokens, fs.prompt_len, self.t.eos
        while True:
            base = len(tokens)
            if base >= cap_ or base >= capacity:
                return tokens
            room = min(cap_, capacity) - base
            t0 = time.perf_counter()
            dseqs = drafter.tree(cur, draft_nodes, self.cfg.draft_topk) if drafter is not None else []
            t1 = time.perf_counter()
            segs = datastore.lookup(tokens + [cur])
            order = dseqs + [tuple([cur] + [int(x) for x in g]) for g in segs]
            cands: list[tuple[int, ...]] = []
            for seq in order:
                seq = seq[:room]
                if not seq or seq in cands:
                    continue
                if len(build_trie(cands + [seq]).nodes) > self.cfg.node_budget:
                    continue
                cands.append(seq)
            if not cands:
                cands = [(cur,)[:room] or (cur,)]
            tree = build_trie(cands)
            t2 = time.perf_counter()
            out = verify_trie(self.model, fs.prepared, cache, tree, PL, base, capacity)
            am = greedy_token(out.logits[0], self.t.offset, self.t.size).tolist()
            path, accepted, recovery = walk(tree, cur, cap_ - base, eos, lambda a, ch: am[ch])
            compact(cache, PL, base, path)
            if drafter is not None:
                drafter.push(out.hidden_states, keep=path)
            t3 = time.perf_counter()
            res.draft_ms += (t1 - t0) * 1000
            res.lookup_ms += (t2 - t1) * 1000
            res.verify_ms += (t3 - t2) * 1000
            res.follow_rounds += 1
            res.round_nodes.append(len(tree.nodes))
            res.round_accepted.append(len(accepted))
            tokens.extend(accepted)
            if (accepted and accepted[-1] == eos) or len(tokens) >= cap_:
                return tokens
            cur = recovery

    # ------------------------------------------------------------------- public
    def decode(self, fs: FrameState, candidates: list[tuple[int, ...]],
               datastore: Datastore | None = None, use_draft: bool = True,
               follow: str = "verify", draft_nodes: int | None = None,
               new_coc: tuple[int, ...] | None = None) -> DecodeResult:
        """Decode one frame.

        ``candidates`` are the previous CoCs (``CocHistory.candidates(N)``).
        ``follow="verify"`` runs follow-up verification with the enabled sources;
        with neither source it verifies one token per run. ``follow="sequential"``
        generates the rest token by token (configuration #1 of the paper).
        The Prefill KV must already be in the buffer (``FixedKVBuffer.write_prefill``).
        ``new_coc`` is the CoC that became available since the last frame; it is
        added to the datastore's drive index inside the timed region.
        """
        datastore = datastore if datastore is not None else Datastore.empty(self.cfg.datastore_key_len)
        draft_nodes = self.cfg.draft_tree_nodes if draft_nodes is None else draft_nodes
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        insert_ms = 0.0
        if new_coc is not None:
            ti = time.perf_counter()
            datastore.add(new_coc)
            insert_ms = (time.perf_counter() - ti) * 1000
        acc, rec, nodes, ncand, hs, path = self._initial(fs, candidates)
        torch.cuda.synchronize()
        tm = time.perf_counter()
        res = DecodeResult(tokens=[], target_runs=1, accepted_initial=len(acc),
                           initial_nodes=nodes, initial_candidates=ncand, datastore_insert_ms=insert_ms)
        eos = self.t.eos
        if (acc and acc[-1] == eos) or len(acc) == fs.max_tokens:
            tokens = acc
        elif follow == "sequential":
            PL = fs.prompt_len
            tail, _ = sequential(self.model, self.buf.view(PL + len(acc)), rec, PL + len(acc), eos,
                                 self.t.offset, self.t.size, fs.max_tokens - len(acc))
            tokens = acc + [int(x) for x in tail]
            res.target_runs = 1 + len(tail)
        else:
            drafter = None
            if use_draft and self.draft is not None:
                drafter = FDDFlash(self.model, self.draft, self.mask_emb, self.t.offset, self.t.size)
                drafter.reset()
                drafter.seed_from_prefill(fs.prefill)
                if path:
                    drafter.push(hs, keep=path)
            tokens = self._follow(fs, acc, rec, datastore, drafter, res, draft_nodes)
            res.target_runs = 1 + res.follow_rounds
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        res.tokens = [int(x) for x in tokens]
        res.initial_ms = (tm - t0) * 1000
        res.follow_ms = (t1 - tm) * 1000
        return res
