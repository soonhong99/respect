"""Per-frame setup shared by the evaluation scripts."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from respect.alpamayo.model import Target, first_token, make_inputs, prefill, prompt_length
from respect.candidates.fd_dflash import find_snapshot, load_fd_dflash
from respect.config import RespectConfig
from respect.decoding.kv_buffer import FixedKVBuffer
from respect.decoding.respect import FrameState, RespectDecoder
from respect.decoding.sequential import clone_cache, sequential
from respect.eval.dataset import Frame, input_path
from respect.kernels import patch as kernels


@dataclass
class FrameInputs:
    fs: FrameState
    prepared: dict
    raw: dict
    inp: dict
    prefill_copy_ms: float


class Runner:
    """Model, drafter and the fixed KV buffer, reused across frames."""

    def __init__(self, target: Target, cfg: RespectConfig = RespectConfig()) -> None:
        self.t, self.cfg = target, cfg
        par = next(target.model.parameters())
        self.draft, self.mask_emb = load_fd_dflash(find_snapshot(), par.dtype, par.device)
        self.buf: FixedKVBuffer | None = None
        self.dec: RespectDecoder | None = None

    def prepare(self, cid: str, fr: Frame, input_dir: Path) -> FrameInputs:
        inp = torch.load(input_path(input_dir, cid, fr.frame_idx), weights_only=False)
        prepared, raw = make_inputs(self.t, inp)
        assert not kernels.is_installed(), "Prefill runs on the default kernels"
        pf = prefill(self.t, prepared)
        PL = prompt_length(pf)
        if self.buf is None:
            self.buf = FixedKVBuffer(PL, self.cfg.buffer_capacities)
            self.dec = RespectDecoder(self.t, self.buf, self.cfg, self.draft, self.mask_emb)
        assert self.buf.PL == PL, "the fixed buffer assumes one prompt length"
        self.buf.allocate(pf.past_key_values)
        torch.cuda.synchronize()
        tq = time.perf_counter()
        self.buf.write_prefill(pf.past_key_values)
        torch.cuda.synchronize()
        copy_ms = (time.perf_counter() - tq) * 1000
        fs = FrameState(prepared, pf, PL, first_token(pf, PL), fr.max_new_tokens)
        return FrameInputs(fs, prepared, raw, inp, copy_ms)

    def sequential(self, fs: FrameState) -> tuple[list[int], Any]:
        tk, cache = sequential(self.t.model, clone_cache(fs.prefill.past_key_values), fs.first_token,
                               fs.prompt_len, self.t.eos, self.t.offset, self.t.size, fs.max_tokens)
        return [int(x) for x in tk], cache


def timed(fn):
    """Run ``fn`` between CUDA synchronizations; return (result, ms)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000
