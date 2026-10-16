"""Attention for trie verification with a slot-order-independent sum over nodes.

The prompt segment is processed in fixed-size tiles with no mask. The node
segment computes the scores of a block of nodes with one matrix product, then
accumulates the nodes one at a time in logical order. Masked slots get a score
of -inf and contribute exactly zero, so the result for a node depends only on
the values of its ancestors and their order, not on where siblings sit in the
buffer. All accumulation is fp32; the output is cast once at the end.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_hyb(
    Q, KP, VP, KN, VN, MSK, OUT,
    sqh, sqm, sqd, skph, skpn, skpd, svph, svpn, svpd,
    sknh, sknn, sknd, svnh, svnn, svnd, smm, smn, soh, som, sod,
    M, NP, sm_scale,
    GQA: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    NNC: tl.constexpr, BNB: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_h = tl.program_id(1)
    kvh = pid_h // GQA
    offs_m = pid_m*BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    offs_b = tl.arange(0, BNB)
    m_valid = offs_m < M
    q = tl.load(Q + pid_h*sqh + offs_m[:, None]*sqm + offs_d[None, :]*sqd,
                mask=m_valid[:, None], other=0.0)
    m_i = tl.full((BM,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    acc = tl.zeros((BM, BD), dtype=tl.float32)

    for start in range(0, NP, BN):
        offs_n = start + tl.arange(0, BN)
        n_valid = offs_n < NP
        k = tl.load(KP + kvh*skph + offs_n[:, None]*skpn + offs_d[None, :]*skpd,
                    mask=n_valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(n_valid[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        v = tl.load(VP + kvh*svph + offs_n[:, None]*svpn + offs_d[None, :]*svpd,
                    mask=n_valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    for start in range(0, NNC, BNB):
        offs_n = start + offs_b
        k = tl.load(KN + kvh*sknh + offs_n[:, None]*sknn + offs_d[None, :]*sknd)
        qk = tl.dot(q, tl.trans(k)) * sm_scale            # [BM, BNB]
        allow = tl.load(MSK + offs_m[:, None]*smm + offs_n[None, :]*smn,
                        mask=m_valid[:, None], other=0)
        qk = tl.where(allow != 0, qk, float("-inf"))
        for j in tl.static_range(BNB):
            qkj = tl.sum(tl.where(offs_b[None, :] == j, qk, 0.0), axis=1)   # exactly one entry survives
            m_new = tl.maximum(m_i, qkj)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qkj - m_new)
            v = tl.load(VN + kvh*svnh + (start+j)*svnn + offs_d*svnd).to(tl.float32)
            acc = acc * alpha[:, None] + p[:, None] * v[None, :]
            l_i = l_i * alpha + p
            m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(OUT + pid_h*soh + offs_m[:, None]*som + offs_d[None, :]*sod,
             acc.to(OUT.dtype.element_ty), mask=m_valid[:, None])


def fused_attn_hyb(q, kp, vp, kn, vn, msk, sm_scale=None,
                   BM=64, BN=128, BNB=16, num_warps=4, num_stages=1):
    H, M, D = q.shape
    HK, NP, _ = kp.shape
    NN = kn.shape[1]
    assert NN % BNB == 0
    if sm_scale is None: sm_scale = D ** -0.5
    out = torch.empty_like(q)
    grid = (triton.cdiv(M, BM), H)
    _attn_hyb[grid](
        q, kp, vp, kn, vn, msk, out,
        q.stride(0), q.stride(1), q.stride(2),
        kp.stride(0), kp.stride(1), kp.stride(2),
        vp.stride(0), vp.stride(1), vp.stride(2),
        kn.stride(0), kn.stride(1), kn.stride(2),
        vn.stride(0), vn.stride(1), vn.stride(2),
        msk.stride(0), msk.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        M, NP, sm_scale,
        GQA=H // HK, BM=BM, BN=BN, BD=D, NNC=NN, BNB=BNB,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
