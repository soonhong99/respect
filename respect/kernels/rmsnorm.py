"""RMS normalization whose result does not depend on the number of rows.

One program handles one whole row, so the feature-axis sum is always taken in
the same order. The rounding points follow the PyTorch implementation: upcast to
fp32, mean of squares, rsqrt, downcast to the input dtype, multiply by weight.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(X, W, Y, N, eps, BN: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, BN)
    m = off < N
    x = tl.load(X + row*N + off, mask=m, other=0.0).to(tl.float32)
    var = tl.sum(x*x, axis=0) / N  # fixed order
    xh = x * tl.rsqrt(var + eps)
    xh = xh.to(Y.dtype.element_ty).to(tl.float32)  # downcast where PyTorch does
    w = tl.load(W + off, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*N + off, (w*xh).to(Y.dtype.element_ty), mask=m)


def rmsnorm(x, w, eps):
    xs = x.reshape(-1, x.shape[-1]).contiguous()
    M, N = xs.shape
    out = torch.empty_like(xs)
    BN = triton.next_power_of_2(N)
    nw = 8 if BN >= 2048 else (4 if BN >= 512 else 2)
    _rmsnorm[(M,)](xs, w.contiguous(), out, N, eps, BN=BN, num_warps=nw)
    return out.reshape(x.shape)
