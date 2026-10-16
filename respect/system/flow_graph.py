"""Flow with a fixed KV buffer and one CUDA graph per diffusion step (Section VI-D).

Computes the same trajectory as ``respect.alpamayo.flow.trajectory`` (one sample).
The public implementation appends the diffusion tokens' KV to the BF16 cache with
``torch.cat``; the expert's keys are float32 and its values BF16, so the
concatenated keys are promoted to float32 while the values stay BF16.  The buffer
uses the same dtypes so that attention reads the same values.

The buffer holds the longest KV once and is shared; one graph is captured per KV
length (prompt + CoC), each attending to the first ``kv_len + diffusion tokens``
slots.  ``capture`` can pre-capture the common lengths before a timed run.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from respect.alpamayo.flow import seed_everything
from respect.decoding.kv_buffer import WindowCache, WindowLayer
from respect.system.fused import flow_fast

NUM_STEPS = 10          # Euler steps of the public FlowMatching


class FlowGraph:
    def __init__(self, model: Any, eos: int, prompt_len: int, layers_like: list, max_new_tokens: int = 128) -> None:
        self.m, self.eos = model, eos
        self.nd = model.action_space.get_action_space_dims()[0]
        self.x_dims = tuple(model.diffusion.x_dims)
        self.ekw = {"is_causal": False} if model.config.expert_non_causal_attention else {}
        cap = prompt_len + max_new_tokens + self.nd
        self.bk = [torch.zeros(L.keys.shape[0], L.keys.shape[1], cap, L.keys.shape[3], device="cuda",
                               dtype=torch.float32) for L in layers_like]
        self.bv = [torch.zeros(L.values.shape[0], L.values.shape[1], cap, L.values.shape[3], device="cuda",
                               dtype=L.values.dtype) for L in layers_like]
        self.sx = torch.zeros(1, *self.x_dims, device="cuda")
        self.st = torch.zeros(1, *[1] * len(self.x_dims), device="cuda")
        self.graphs: dict[int, tuple] = {}
        self.captured_in_run = 0

    def expert_inputs(self, prepared: dict, tokens: list[int], kv_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        m, dev = self.m, torch.device("cuda")
        gen = torch.tensor([tokens], device=dev, dtype=torch.long)
        offset = m._find_eos_offset(sequences=torch.cat([prepared["input_ids"], gen], 1), eos_token_id=self.eos,
                                    device=dev, warn=False)
        return m._build_expert_pos_ids_and_attn_mask(offset=offset, rope_deltas=m.vlm.model.rope_deltas,
                                                     kv_cache_seq_len=kv_len, n_diffusion_tokens=self.nd, b_star=1,
                                                     device=dev, prefix_mask=None)

    def capture(self, kv_len: int, pos: torch.Tensor, mask: torch.Tensor) -> None:
        m, nd = self.m, self.nd
        wc = WindowCache([WindowLayer(a, b, kv_len + nd) for a, b in zip(self.bk, self.bv)])
        cpos = torch.arange(kv_len, kv_len + nd, device="cuda")
        hold = dict(pos=pos.clone(), mask=mask.clone())

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = m.action_in_proj(self.sx, self.st)
                if emb.dim() == 2:
                    emb = emb.view(1, nd, -1)
                out = m.expert(inputs_embeds=emb, position_ids=hold["pos"], past_key_values=wc,
                               attention_mask=hold["mask"], cache_position=cpos, use_cache=True, **self.ekw)
                return m.action_out_proj(out.last_hidden_state[:, -nd:]).view(-1, *self.x_dims)

        with flow_fast():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    step()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = step()
        hold["keep"] = (cpos, wc)          # tensors read by the graph live as long as the graph
        self.graphs[kv_len] = (g, hold, out)

    @torch.no_grad()
    def __call__(self, prepared: dict, raw: dict, cache: Any, tokens: list[int], seed: int = 0,
                 in_run: bool = True) -> np.ndarray:
        kv_len = cache.get_seq_length()
        pos, mask = self.expert_inputs(prepared, tokens, kv_len)
        for a, b, L in zip(self.bk, self.bv, cache.layers):
            a[:, :, :kv_len].copy_(L.keys)
            b[:, :, :kv_len].copy_(L.values)
        if kv_len not in self.graphs:
            self.capture(kv_len, pos, mask)
            self.captured_in_run += int(in_run)
        g, hold, out = self.graphs[kv_len]
        hold["pos"].copy_(pos)
        hold["mask"].copy_(mask)
        seed_everything(seed)
        x = torch.randn(1, *self.x_dims, device="cuda")       # the first RNG call of the public _euler
        ts = torch.linspace(0.0, 1.0, NUM_STEPS + 1, device="cuda")
        n = len(self.x_dims)
        for i in range(NUM_STEPS):
            dt = (ts[i + 1] - ts[i]).view(1, *[1] * n).expand(1, *[1] * n)
            t0 = ts[i].view(1, *[1] * n).expand(1, *[1] * n)
            self.sx.copy_(x)
            self.st.copy_(t0)
            g.replay()
            x = x + dt * out
        hx, hr = raw["ego_history_xyz"][:, -1], raw["ego_history_rot"][:, -1]
        return self.m.action_space.action_to_traj(x, hx, hr)[0].detach().cpu().numpy().reshape(64, 3)
