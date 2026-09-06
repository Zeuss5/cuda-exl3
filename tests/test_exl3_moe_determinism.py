"""CUDA_EXL3_DETERMINISTIC has to cover the MoE stage, not just the dense one.

The down projection's epilogue scales each routed row by its routing weight and
accumulates it into the token's row with atomics, so a token's top-k sum lands
in whatever order the blocks retire. That is the same trade split-k makes, and
for a long time the flag disabled only split-k -- so on a MoE model, which is
the only kind this plugin is used for, "bit-exact everywhere" was not true.

Under the flag moe.py sends the down projection through exl3_moe_combine
instead, which walks k in order per token. These keep both halves of that
honest: the two paths must agree in value, and the combine path must be
bit-exact when repeated.
"""
import pytest
import torch

from cuda_exl3 import ops as _ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, I, BITS, CB = 32, 2048, 768, 4, 1


@pytest.fixture
def gemm():
    _ops._try_native()
    return torch.ops.cuda_exl3_C


def _routed(rows, block_m, top_k, dev="cuda"):
    """Operands for the down projection: (rows, I) in, (M, H) out per token."""
    torch.manual_seed(0)
    nblk = rows // block_m
    M = rows // top_k
    trellis = torch.randint(-32768, 32767, (E, I // 16, H // 16, 16 * BITS),
                            dtype=torch.int16, device=dev)
    svh = (torch.randn((E, H), device=dev) * 0.05).half()
    a = torch.randn((1, rows, I), dtype=torch.half, device=dev) * 0.05
    eids = torch.randint(0, E, (nblk,), dtype=torch.int32, device=dev)
    nr = torch.tensor([rows], dtype=torch.int32, device=dev)
    # Every routed slot is live and distinct, so each token really does sum
    # top_k rows and the orders the two paths use are genuinely different.
    sorted_ids = torch.randperm(rows, dtype=torch.int32, device=dev)
    w = torch.rand((M, top_k), dtype=torch.float32, device=dev) + 0.5
    return trellis, svh, a, eids, nr, sorted_ids, w, M


@pytest.mark.parametrize("rows,block_m,top_k", [(256, 16, 4), (512, 32, 8)])
def test_combine_path_matches_the_fused_epilogue(gemm, rows, block_m, top_k):
    trellis, svh, a, eids, nr, sids, w, M = _routed(rows, block_m, top_k)

    fused = gemm.exl3_moe_gemm(a, trellis, svh, svh, eids, nr, [H], CB, block_m,
                               torch.bfloat16, sids, w, M, top_k)
    rows_out = gemm.exl3_moe_gemm(a, trellis, svh, svh, eids, nr, [H], CB,
                                  block_m, torch.bfloat16, sids, None, M, top_k)
    combined = gemm.exl3_moe_combine(rows_out, sids, w, M, eids, block_m)

    assert fused.shape == combined.shape == (M, H)
    f, c = fused.float(), combined.float()
    assert f.norm() > 0, "fixture produced an all-zero output"
    # Not 1e-3, and the reason is the point of the next test: the fused
    # epilogue accumulates through bf16 atomics, so it rounds once per routed
    # row, while the combine sums in fp32 and rounds once per token. The two
    # therefore differ by roughly sqrt(top_k) bf16 ulps, not by fp32 noise.
    assert (f - c).norm() / f.norm() < 1e-2


@pytest.mark.parametrize("rows,block_m,top_k", [(256, 16, 4), (512, 32, 8)])
def test_combine_path_is_bit_exact_when_repeated(gemm, rows, block_m, top_k):
    trellis, svh, a, eids, nr, sids, w, M = _routed(rows, block_m, top_k)

    def once():
        rows_out = gemm.exl3_moe_gemm(a, trellis, svh, svh, eids, nr, [H], CB,
                                      block_m, torch.bfloat16, sids, None, M,
                                      top_k)
        return gemm.exl3_moe_combine(rows_out, sids, w, M, eids, block_m)

    first = once()
    assert first.float().norm() > 0, "fixture produced an all-zero output"
    for _ in range(8):
        assert torch.equal(once(), first)


@pytest.mark.parametrize("top_k", [1, 4, 8, 16])
def test_combine_path_is_the_more_accurate_one(gemm, top_k):
    """The flag buys accuracy as well as reproducibility, and should keep doing so.

    Measured here against an fp64 reference: the combine sits at 1.7e-3
    regardless of top_k -- that is just the bf16 rounding of the stored result
    -- while the fused epilogue starts at 2.4e-3 and grows with top_k as the
    partial sums round, reaching 4.3e-3 at GLM-5.3-Flash's top_k of 8.
    """
    rows, block_m = 512, 32
    trellis, svh, a, eids, nr, sids, w, M = _routed(rows, block_m, top_k)

    fused = gemm.exl3_moe_gemm(a, trellis, svh, svh, eids, nr, [H], CB, block_m,
                               torch.bfloat16, sids, w, M, top_k)
    rows_out = gemm.exl3_moe_gemm(a, trellis, svh, svh, eids, nr, [H], CB,
                                  block_m, torch.bfloat16, sids, None, M, top_k)
    combined = gemm.exl3_moe_combine(rows_out, sids, w, M, eids, block_m)

    inv = torch.empty(M * top_k, dtype=torch.long, device="cuda")
    inv[sids.long()] = torch.arange(rows, device="cuda")
    ref = (rows_out.double()[inv].view(M, top_k, H)
           * w.double().unsqueeze(-1)).sum(1)

    err_fused = ((fused.double() - ref).norm() / ref.norm()).item()
    err_comb = ((combined.double() - ref).norm() / ref.norm()).item()
    assert err_comb < 2e-3, err_comb
    assert err_comb <= err_fused, (err_comb, err_fused)
    if top_k >= 8:
        assert err_fused > 1.5 * err_comb, (err_fused, err_comb)
