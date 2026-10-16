"""FD-DFlash candidates: the DFlash draft model FlashDrive trained for Alpamayo 1.5.

The drafter reads the hidden states of selected target layers (the target
features) over the last ``context_len`` committed positions, plus the embedding
of the correction token followed by mask embeddings. It predicts a token
distribution for each of the next ``block_size - 1`` positions in one pass.
Candidates form a tree: the top-1 chain, extended with lower-ranked branches in
order of cumulative log-probability until the node budget is reached.
"""
from __future__ import annotations

import glob
import heapq
from pathlib import Path
from typing import Any

import torch

from respect.decoding.acceptance import mask_action_vocab
from respect.third_party.fd_dflash_model import DFlashDraftModel


def load_fd_dflash(path: str, dtype: torch.dtype, device: torch.device) -> tuple[Any, torch.Tensor]:
    """Load the draft network and its mask embedding from a local snapshot directory."""
    draft = DFlashDraftModel.from_pretrained(path, dtype=dtype).to(device).eval()
    mask_emb = torch.load(Path(path) / "mask_embedding.pt", map_location=device, weights_only=True).to(dtype)
    return draft, mask_emb


def find_snapshot(repo: str = "z-lab/Alpamayo-1.5-10B-DFlash") -> str:
    """Local Hugging Face snapshot of the FD-DFlash weights."""
    name = "models--" + repo.replace("/", "--")
    hits = glob.glob(str(Path.home() / ".cache/huggingface/hub" / name / "snapshots/*/"))
    if not hits:
        raise FileNotFoundError(f"{repo} is not in the local Hugging Face cache; run `hf download {repo}`")
    return hits[0]


class FDDFlash:
    """Per-frame drafter state: a rolling window of target features."""

    def __init__(self, model: Any, draft: Any, mask_emb: torch.Tensor, offset: int, size: int) -> None:
        self.model, self.draft = model, draft
        self.mask_emb = mask_emb
        self.off, self.size = offset, size
        self.BS, self.CL = draft.block_size, draft.context_len
        self.layers = list(draft.config.target_layer_ids)
        self.emb = model.vlm.model.language_model.embed_tokens
        self.ctx: torch.Tensor | None = None

    def reset(self) -> None:
        self.ctx = None

    def seed_from_prefill(self, prefill: Any) -> bool:
        """Fill the target-feature window from the last Prefill positions."""
        hs = getattr(prefill, "hidden_states", None)
        if not hs:
            return False
        step = hs[0]
        if len(step) <= max(self.layers) + 1:
            return False
        cap = torch.cat([step[i + 1][:, -self.CL:, :] for i in self.layers], dim=-1)
        self.ctx = cap[:, -self.CL:, :].to(self.mask_emb.dtype)
        return self.ctx.shape[1] >= self.CL

    def push(self, hidden_states: tuple, keep: list[int] | None = None) -> None:
        """Append the target features of a verification forward.

        With ``keep``, only the listed positions (the accepted path) are appended,
        in order.
        """
        cap = torch.cat([hidden_states[i + 1] for i in self.layers], dim=-1)
        if keep is not None:
            if not keep:
                return
            ix = torch.tensor(keep, device=cap.device)
            cap = cap.index_select(1, ix)
        self.ctx = cap if self.ctx is None else torch.cat([self.ctx, cap], dim=1)
        self.ctx = self.ctx[:, -self.CL:, :]

    @torch.no_grad()
    def logits(self, cur_tok: int) -> torch.Tensor | None:
        if self.ctx is None or self.ctx.shape[1] < self.CL:
            return None
        noise = self.mask_emb.view(1, 1, -1).expand(1, self.BS, -1).clone()
        noise[:, 0, :] = self.emb(torch.tensor([[cur_tok]], device=noise.device, dtype=torch.long))[:, 0, :]
        pos = torch.arange(self.CL + self.BS, device=noise.device).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.draft(target_hidden=self.ctx, noise_embedding=noise, position_ids=pos, is_causal=False)
            lg = self.model.vlm.lm_head(h[:, 1:, :]).float()
        return lg[0]

    def tree(self, cur: int, budget: int, topk: int) -> list[tuple[int, ...]]:
        """Candidate sequences rooted at ``cur``: the top-1 chain first, then branches.

        ``budget`` counts tree nodes excluding the root. With ``budget`` equal to the
        number of predicted positions, only the top-1 chain is returned.
        """
        lg = self.logits(cur)
        if lg is None:
            return []
        masked = mask_action_vocab(lg, self.off, self.size)
        top1 = [int(x) for x in masked.argmax(-1).tolist()]
        lp = torch.log_softmax(masked, -1)
        val, idx = lp.topk(topk, -1)
        val, idx = val.tolist(), idx.tolist()
        P = len(val)
        chosen, score = set(), {}
        for d in range(P):
            path = tuple(top1[:d + 1])
            chosen.add(path)
            score[path] = sum(val[j][0] for j in range(d + 1))
        heap: list = []

        def push_children(path: tuple) -> None:
            d = len(path)
            if d >= P:
                return
            bs = score[path] if path else 0.0
            for r in range(topk):
                child = path + (idx[d][r],)
                if child not in chosen:
                    heapq.heappush(heap, (-(bs + val[d][r]), child))

        push_children(())
        for p in list(chosen):
            push_children(p)
        while heap and len(chosen) < budget:
            ns, child = heapq.heappop(heap)
            if child in chosen or (len(child) > 1 and child[:-1] not in chosen):
                continue
            chosen.add(child)
            score[child] = -ns
            push_children(child)
        leaves = [p for p in chosen if not any(q[:len(p)] == p and len(q) > len(p) for q in chosen)]
        leaves.sort(key=lambda p: -score[p])
        return [tuple([cur] + top1)] + [tuple([cur] + list(p)) for p in leaves if list(p) != top1]
