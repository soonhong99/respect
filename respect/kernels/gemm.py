"""Linear layer whose result does not depend on the number of rows.

Every output element is accumulated over the K axis in the same fixed order, so a
row gives the same bits whether it is computed alone or together with other rows.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M, N, K,
          sam, sak, sbn, sbk, scm, scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    om = pm*BM + tl.arange(0, BM)
    on = pn*BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):  # fixed order
        ok = k0 + tl.arange(0, BK)
        a = tl.load(A + om[:, None]*sam + ok[None, :]*sak,
                    mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        b = tl.load(B + on[:, None]*sbn + ok[None, :]*sbk,
                    mask=(on[:, None] < N) & (ok[None, :] < K), other=0.0)
        acc += tl.dot(a, tl.trans(b))
    tl.store(C + om[:, None]*scm + on[None, :]*scn, acc.to(C.dtype.element_ty),
             mask=(om[:, None] < M) & (on[None, :] < N))


def linear(x, w, BM=32, BN=128, BK=64, num_warps=4, num_stages=2, out_dtype=None):
    """x (M, K) @ w (N, K)^T -> (M, N). Bias is handled by the caller."""
    xs = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = xs.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=out_dtype or x.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm[grid](xs, w, out, M, N, K,
                xs.stride(0), xs.stride(1), w.stride(0), w.stride(1),
                out.stride(0), out.stride(1),
                BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages)
    return out.reshape(*x.shape[:-1], N)
