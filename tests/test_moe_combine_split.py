"""The fused combine must never run split-k (regression test).

``exl3_moe_gemm`` with ``topk_weights`` folds the MoE combine into the down
projection's epilogue: its output is one row per TOKEN. Its split-k path did
not know that -- the split kernel is launched without the weights and sorted
ids, and the split epilogue writes one row per ROUTED row -- so with split-k on
(the default) a small grid got unweighted rows in the wrong places and wrote
past the end of the output. Fixed in gemm.cu by never splitting a fused combine.

The shapes are chosen so that the grid, (H/128) x expert blocks, is smaller
than the SM count on both the RTX 5090 (170) and GB10 (48): without the fix the
split path is taken on either.
"""
import pytest
import torch

from cuda_exl3 import ops as _ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, I, BITS, CB, BM = 12, 1024, 768, 4, 2, 16


@pytest.fixture
def gemm():
    _ops._try_native()
    return torch.ops.cuda_exl3_C


@pytest.mark.parametrize("M,top_k", [(1, 4), (2, 2), (1, 1)])
def test_fused_combine_with_split_k_enabled_matches_the_reference(gemm, M, top_k):
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    nblk = M * top_k                          # one routed pair per expert block
    rows = nblk * BM
    assert (H // 128) * nblk < sms, "precondition: a grid small enough to split"
    assert int(gemm.exl3_get_moe_acc_cap()) >= rows * H, "split-k must be allowed"

    torch.manual_seed(0)
    tr = torch.randint(-32768, 32767, (E, I // 16, H // 16, 16 * BITS),
                       dtype=torch.int16, device="cuda")
    svh = (torch.randn((E, H), device="cuda") * 0.05).half()
    a = torch.randn((1, rows, I), dtype=torch.half, device="cuda") * 0.05
    eids = torch.randperm(E, device="cuda")[:nblk].int()
    nr = torch.tensor([rows], dtype=torch.int32, device="cuda")
    sids = torch.full((rows,), M * top_k, dtype=torch.int32, device="cuda")
    pos = torch.arange(nblk, device="cuda") * BM
    sids[pos] = torch.arange(nblk, dtype=torch.int32, device="cuda")
    w = torch.rand((M, top_k), device="cuda") + 0.5

    # per routed row, unfused and unsplit: the reference the combine must equal
    prev = int(gemm.exl3_get_moe_acc_cap())
    try:
        gemm.exl3_set_moe_acc_cap(0)
        rows_out = gemm.exl3_moe_gemm(a, tr, svh, svh, eids, nr, [H], CB, BM,
                                      torch.bfloat16, sids, None, M, top_k).float()
    finally:
        gemm.exl3_set_moe_acc_cap(prev)
    ref = torch.zeros(M, H, device="cuda")
    for p in range(M * top_k):
        ref[p // top_k] += w.view(-1)[p] * rows_out[pos[p]]

    fused = gemm.exl3_moe_gemm(a, tr, svh, svh, eids, nr, [H], CB, BM,
                               torch.bfloat16, sids, w, M, top_k).float()
    assert fused.shape == (M, H)
    assert ref.norm() > 0
    assert (fused - ref).norm() / ref.norm() < 1e-2
