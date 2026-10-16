"""Bitwise checks of the dedicated kernels (CUDA + Triton required)."""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_linear_is_row_count_invariant():
    from respect.kernels.gemm import linear
    torch.manual_seed(0)
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
    kw = dict(BM=32, BN=128, BK=128, num_warps=4, num_stages=3)
    full = linear(x, w, **kw)
    for m in (1, 7, 33):
        assert torch.equal(linear(x[:m], w, **kw), full[:m])


@cuda
def test_rmsnorm_is_row_count_invariant():
    from respect.kernels.rmsnorm import rmsnorm
    torch.manual_seed(0)
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    full = rmsnorm(x, w, 1e-6)
    assert torch.equal(rmsnorm(x[:1], w, 1e-6), full[:1])


@cuda
def test_attention_is_slot_layout_invariant():
    """Same 26 node values in consecutive slots vs. scattered slots (0-17 and 200-207)."""
    from respect.kernels.attention import fused_attn_hyb
    torch.manual_seed(0)
    H, HK, D, NP, n = 32, 8, 128, 300, 26
    q = torch.randn(H, 64, D, device="cuda", dtype=torch.bfloat16)
    kp = torch.randn(HK, NP, D, device="cuda", dtype=torch.bfloat16)
    vp = torch.randn(HK, NP, D, device="cuda", dtype=torch.bfloat16)
    kv_k = torch.randn(HK, n, D, device="cuda", dtype=torch.bfloat16)
    kv_v = torch.randn(HK, n, D, device="cuda", dtype=torch.bfloat16)
    NN = 224
    slots_a = list(range(n))
    slots_b = list(range(18)) + list(range(200, 208))
    outs = []
    for slots in (slots_a, slots_b):
        kn = torch.zeros(HK, NN, D, device="cuda", dtype=torch.bfloat16)
        vn = torch.zeros_like(kn)
        msk = torch.zeros(64, NN, device="cuda", dtype=torch.bool)
        for j, s in enumerate(slots):
            kn[:, s] = kv_k[:, j]
            vn[:, s] = kv_v[:, j]
        msk[:, slots] = True
        outs.append(fused_attn_hyb(q, kp, vp, kn, vn, msk, BM=64, BN=128, BNB=16, num_warps=4, num_stages=1))
    assert torch.equal(outs[0], outs[1])
