"""Output-preserving fused kernels for Prefill and Flow (Section VI-D).

Every element-wise operation is computed at the same rounding points as PyTorch:
with BF16 inputs every operation is rounded to BF16 (round-to-nearest-even), with
float32 inputs it stays in float32, and the output dtype follows PyTorch's type
promotion.  Multiplications and additions are not fused into FMA.  For RMS
normalization only the squares are computed in the kernel and the mean is left to
PyTorch, so the reduction order is unchanged.  The Prefill KV cache and the Flow
trajectory are therefore bitwise identical to the public implementation.

``prefill_fast()``  patches used during Prefill: RoPE, SwiGLU, RMSNorm, the residual
                    addition fused with the post-attention RMSNorm squares, and one
                    vision-attention call for all images.
``flow_fast()``     the same kernels for the Flow expert (no vision patches).
"""
from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl
import transformers.models.qwen3_vl.modeling_qwen3_vl as q3

try:
    from triton.language.extra.cuda import libdevice
except ImportError:
    from triton.language.extra import libdevice


# ------------------------------------------------------------------ kernels
@triton.jit
def _rne(x, ROUND: tl.constexpr):
    """If ROUND, round a float32 value to the nearest-even BF16 value (kept in float32).

    Done with integer bit operations so that the compiler cannot drop the intermediate rounding.
    """
    if ROUND:
        u = x.to(tl.uint32, bitcast=True)
        u = (u + (((u >> 16) & 1) + 0x7FFF)) & 0xFFFF0000
        x = u.to(tl.float32, bitcast=True)
    return x


@triton.jit
def _rope_k(X, C, S, Y, H, L, D, sxb, sxh, sxl, sxd, scb, scl, ROUND: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    l = pid % L
    h = (pid // L) % H
    b = pid // (L * H)
    half = D // 2
    d = tl.arange(0, BD)
    m = d < D
    base = X + b * sxb + h * sxh + l * sxl
    x = tl.load(base + d * sxd, mask=m, other=0.0).to(tl.float32)
    dr = tl.where(d < half, d + half, d - half)
    xr = tl.load(base + dr * sxd, mask=m, other=0.0).to(tl.float32)
    xr = tl.where(d < half, -xr, xr)
    c = tl.load(C + b * scb + l * scl + d, mask=m, other=0.0).to(tl.float32)
    s = tl.load(S + b * scb + l * scl + d, mask=m, other=0.0).to(tl.float32)
    a = _rne(libdevice.mul_rn(x, c), ROUND)
    r = _rne(libdevice.mul_rn(xr, s), ROUND)
    y = _rne(libdevice.add_rn(a, r), ROUND)
    tl.store(Y + pid * D + d, y.to(Y.dtype.element_ty), mask=m)


def rope(x, cos, sin):
    """q*cos + rotate_half(q)*sin.  x [B,H,L,D], cos/sin [B,L,D]."""
    out_dtype = torch.promote_types(x.dtype, cos.dtype)
    B, H, L, D = x.shape
    y = torch.empty((B, H, L, D), device=x.device, dtype=out_dtype)
    cos, sin = cos.contiguous(), sin.contiguous()
    _rope_k[(B * H * L,)](x, cos, sin, y, H, L, D, *x.stride(), cos.stride(0), cos.stride(1),
                          ROUND=out_dtype == torch.bfloat16, BD=triton.next_power_of_2(D), num_warps=1)
    return y


@triton.jit
def _rope_v_k(X, C, S, Y, NH, D, sxl, sxh, sxd, scl, BD: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % NH
    l = pid // NH
    half = D // 2
    d = tl.arange(0, BD)
    m = d < D
    base = X + l * sxl + h * sxh
    x = tl.load(base + d * sxd, mask=m, other=0.0).to(tl.float32)
    dr = tl.where(d < half, d + half, d - half)
    xr = tl.load(base + dr * sxd, mask=m, other=0.0).to(tl.float32)
    xr = tl.where(d < half, -xr, xr)
    c = tl.load(C + l * scl + d, mask=m, other=0.0).to(tl.float32)
    s = tl.load(S + l * scl + d, mask=m, other=0.0).to(tl.float32)
    y = libdevice.add_rn(libdevice.mul_rn(x, c), libdevice.mul_rn(xr, s))
    tl.store(Y + pid * D + d, y.to(Y.dtype.element_ty), mask=m)


def rope_vision(x, cos, sin):
    """One tensor of ``apply_rotary_pos_emb_vision``: computed in float32, returned in the input dtype."""
    Lq, NH, D = x.shape
    y = torch.empty((Lq, NH, D), device=x.device, dtype=x.dtype)
    cos, sin = cos.float().contiguous(), sin.float().contiguous()
    _rope_v_k[(Lq * NH,)](x, cos, sin, y, NH, D, *x.stride(), cos.stride(0), BD=triton.next_power_of_2(D), num_warps=1)
    return y


@triton.jit
def _silu_mul_k(G, U, Y, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    g = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + off, mask=m, other=0.0).to(tl.float32)
    a = _rne(libdevice.div_rn(g, libdevice.add_rn(1.0, libdevice.exp(-g))), True)
    tl.store(Y + off, _rne(libdevice.mul_rn(a, u), True).to(tl.bfloat16), mask=m)


def silu_mul(g, u):
    if g.dtype != torch.bfloat16 or u.dtype != torch.bfloat16:
        return torch.nn.functional.silu(g) * u
    g, u = g.contiguous(), u.contiguous()
    y = torch.empty_like(g)
    n = g.numel()
    _silu_mul_k[(triton.cdiv(n, 4096),)](g, u, y, n, BLOCK=4096, num_warps=8)
    return y


@triton.jit
def _sq_k(X, Q, N, BN: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, BN)
    m = off < N
    x = tl.load(X + row * N + off, mask=m, other=0.0).to(tl.float32)
    tl.store(Q + row * N + off, libdevice.mul_rn(x, x), mask=m)


@triton.jit
def _add_sq_k(R, H, Xo, Q, N, ROUND: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, BN)
    m = off < N
    r = tl.load(R + row * N + off, mask=m, other=0.0).to(tl.float32)
    h = tl.load(H + row * N + off, mask=m, other=0.0).to(tl.float32)
    x = _rne(libdevice.add_rn(r, h), ROUND)
    tl.store(Xo + row * N + off, x.to(Xo.dtype.element_ty), mask=m)
    tl.store(Q + row * N + off, libdevice.mul_rn(x, x), mask=m)


@triton.jit
def _norm_apply_k(X, V, W, Y, N, eps, ROUND: tl.constexpr, BN: tl.constexpr):
    """y = w * to_input_dtype(x * rsqrt(var + eps)); ROUND when the input (and output) is BF16."""
    row = tl.program_id(0)
    off = tl.arange(0, BN)
    m = off < N
    x = tl.load(X + row * N + off, mask=m, other=0.0).to(tl.float32)
    v = tl.load(V + row)
    r = libdevice.rsqrt(libdevice.add_rn(v, eps))
    xh = _rne(libdevice.mul_rn(x, r), ROUND)
    w = tl.load(W + off, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * N + off, _rne(libdevice.mul_rn(w, xh), ROUND).to(Y.dtype.element_ty), mask=m)


def _nw(BN):
    return 8 if BN >= 2048 else (4 if BN >= 512 else 2)


def norm_from(xs, sq, w, eps):
    """xs [M, N] contiguous, sq = xs.float() ** 2.  The mean is taken by PyTorch, the rest by the kernel."""
    M, N = xs.shape
    var = sq.mean(-1).contiguous()
    out_dtype = torch.promote_types(w.dtype, xs.dtype)
    y = torch.empty((M, N), device=xs.device, dtype=out_dtype)
    BN = triton.next_power_of_2(N)
    _norm_apply_k[(M,)](xs, var, w.contiguous(), y, N, eps, ROUND=xs.dtype == torch.bfloat16, BN=BN, num_warps=_nw(BN))
    return y


def rmsnorm(x, w, eps):
    xs = x.reshape(-1, x.shape[-1]).contiguous()
    M, N = xs.shape
    sq = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BN = triton.next_power_of_2(N)
    _sq_k[(M,)](xs, sq, N, BN=BN, num_warps=_nw(BN))
    return norm_from(xs, sq, w, eps).reshape(x.shape)


def add_sq(r, h):
    """x = r + h with PyTorch type promotion and rounding, and x.float() ** 2."""
    out_dtype = torch.promote_types(r.dtype, h.dtype)
    rs, hs = r.reshape(-1, r.shape[-1]).contiguous(), h.reshape(-1, h.shape[-1]).contiguous()
    M, N = rs.shape
    x = torch.empty((M, N), device=r.device, dtype=out_dtype)
    sq = torch.empty((M, N), device=r.device, dtype=torch.float32)
    BN = triton.next_power_of_2(N)
    _add_sq_k[(M,)](rs, hs, x, sq, N, ROUND=out_dtype == torch.bfloat16, BN=BN, num_warps=_nw(BN))
    return x, sq


# ------------------------------------------------------------------ patches
ORIG = dict(rope=q3.apply_rotary_pos_emb, rope_v=q3.apply_rotary_pos_emb_vision, mlp=q3.Qwen3VLTextMLP.forward,
            norm=q3.Qwen3VLTextRMSNorm.forward, layer=q3.Qwen3VLTextDecoderLayer.forward,
            vattn=q3.Qwen3VLVisionAttention.forward)


# During ``generate`` transformers wraps the forward of layer instances to record hidden states, so a
# class-level patch can stay on an instance after it is removed from the class.  Every patched
# function therefore checks the switch and falls back to the original function when it is off.
_ON = dict(on=False, vision=False)


def _p_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if not _ON["on"]:
        return ORIG["rope"](q, k, cos, sin, position_ids, unsqueeze_dim)
    return rope(q, cos, sin), rope(k, cos, sin)


def _p_rope_v(q, k, cos, sin):
    if not (_ON["on"] and _ON["vision"]):
        return ORIG["rope_v"](q, k, cos, sin)
    return rope_vision(q, cos, sin), rope_vision(k, cos, sin)


def _p_mlp(self, x):
    if not _ON["on"]:
        return ORIG["mlp"](self, x)
    return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))


def _p_norm(self, x):
    if not _ON["on"]:
        return ORIG["norm"](self, x)
    return rmsnorm(x, self.weight, self.variance_epsilon)


def _p_layer(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, past_key_values=None,
             use_cache=False, cache_position=None, **kwargs):
    if not _ON["on"]:
        return ORIG["layer"](self, hidden_states, position_embeddings, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, **kwargs)
    residual = hidden_states
    h = self.input_layernorm(hidden_states)
    h, _ = self.self_attn(hidden_states=h, attention_mask=attention_mask, position_ids=position_ids,
                          past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position,
                          position_embeddings=position_embeddings, **kwargs)
    x, sq = add_sq(residual, h)                      # residual + attention output, and the squares for the post-attention norm
    pal = self.post_attention_layernorm
    hn = norm_from(x, sq, pal.weight, pal.variance_epsilon).reshape(*residual.shape[:-1], -1)
    x = x.reshape(*residual.shape[:-1], -1)
    return x + self.mlp(hn)


def _p_vattn(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
    """One SDPA call for all images instead of one per image (only when all images have the same size)."""
    n = cu_seqlens.numel() - 1
    seq = hidden_states.shape[0]
    if seq % n or not (_ON["on"] and _ON["vision"]):
        return ORIG["vattn"](self, hidden_states, cu_seqlens, rotary_pos_emb=rotary_pos_emb,
                             position_embeddings=position_embeddings, **kwargs)
    qq, kk, vv = self.qkv(hidden_states).reshape(seq, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    qq, kk = q3.apply_rotary_pos_emb_vision(qq, kk, cos, sin)
    c = seq // n
    qq = qq.reshape(n, c, self.num_heads, -1).transpose(1, 2)
    kk = kk.reshape(n, c, self.num_heads, -1).transpose(1, 2)
    vv = vv.reshape(n, c, self.num_heads, -1).transpose(1, 2)
    o = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, scale=self.scaling)
    return self.proj(o.transpose(1, 2).reshape(seq, -1))


def _install(vision: bool):
    _ON["on"], _ON["vision"] = True, vision
    q3.apply_rotary_pos_emb = _p_rope
    q3.Qwen3VLTextMLP.forward = _p_mlp
    q3.Qwen3VLTextRMSNorm.forward = _p_norm
    q3.Qwen3VLTextDecoderLayer.forward = _p_layer
    if vision:
        q3.apply_rotary_pos_emb_vision = _p_rope_v
        q3.Qwen3VLVisionAttention.forward = _p_vattn


def _uninstall():
    _ON["on"], _ON["vision"] = False, False
    q3.apply_rotary_pos_emb, q3.apply_rotary_pos_emb_vision = ORIG["rope"], ORIG["rope_v"]
    q3.Qwen3VLTextMLP.forward, q3.Qwen3VLTextRMSNorm.forward = ORIG["mlp"], ORIG["norm"]
    q3.Qwen3VLTextDecoderLayer.forward, q3.Qwen3VLVisionAttention.forward = ORIG["layer"], ORIG["vattn"]


@contextlib.contextmanager
def prefill_fast():
    _install(vision=True)
    try:
        yield
    finally:
        _uninstall()


@contextlib.contextmanager
def flow_fast():
    _install(vision=False)
    try:
        yield
    finally:
        _uninstall()
