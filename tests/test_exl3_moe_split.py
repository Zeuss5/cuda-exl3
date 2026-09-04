"""MoE split-k is on by default, gated on wave occupancy: a grid that already
fills the machine is never split, because past one full wave the accumulator
traffic and epilogue launch cost more than the occupancy they buy. These keep it
honest: the split result must match the unsplit one, and the accumulator must be
left zeroed so a second call does not add to stale partials.
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


def _operands(rows, block_m, dev="cuda"):
    torch.manual_seed(0)
    nblk = rows // block_m
    trellis = torch.randint(-32768, 32767, (E, H // 16, 2 * I // 16, 16 * BITS),
                            dtype=torch.int16, device=dev)
    svh = (torch.randn((E, 2 * I), device=dev) * 0.05).half()
    a = torch.randn((2, rows, H), dtype=torch.half, device=dev) * 0.05
    eids = torch.randint(0, E, (nblk,), dtype=torch.int32, device=dev)
    eids[nblk // 2] = -1                       # a block belonging to no expert
    nr = torch.tensor([rows], dtype=torch.int32, device=dev)
    return trellis, svh, a, eids, nr


@pytest.mark.parametrize("rows,block_m", [(256, 16), (512, 32)])
def test_split_matches_unsplit(gemm, rows, block_m):
    trellis, svh, a, eids, nr = _operands(rows, block_m)
    run = lambda: gemm.exl3_moe_gemm(a, trellis, svh, svh, eids, nr, [I, I], CB,
                                     block_m, torch.bfloat16)
    prev = int(gemm.exl3_get_moe_acc_cap())
    try:
        gemm.exl3_set_moe_acc_cap(0)
        unsplit = run()
        gemm.exl3_set_moe_acc_cap(1 << 24)
        split = run()
    finally:
        gemm.exl3_set_moe_acc_cap(prev)

    live = (eids >= 0).repeat_interleave(block_m)
    u, s = unsplit[live].float(), split[live].float()
    # split-k reduces through fp32 atomics, so the order differs; the values
    # must not.
    assert (u - s).norm() / u.norm() < 1e-3


@pytest.mark.parametrize("rows,block_m", [(256, 16), (512, 32)])
def test_split_leaves_accumulator_zeroed(gemm, rows, block_m):
    """Two split calls in a row must agree: the epilogue re-zeroes what it read."""
    trellis, svh, a, eids, nr = _operands(rows, block_m)
    prev = int(gemm.exl3_get_moe_acc_cap())
    try:
        gemm.exl3_set_moe_acc_cap(1 << 24)
        first = gemm.exl3_moe_gemm(a, trellis, svh, svh, eids, nr, [I, I], CB,
                                   block_m, torch.bfloat16)
        second = gemm.exl3_moe_gemm(a, trellis, svh, svh, eids, nr, [I, I], CB,
                                    block_m, torch.bfloat16)
    finally:
        gemm.exl3_set_moe_acc_cap(prev)
    live = (eids >= 0).repeat_interleave(block_m)
    f, s = first[live].float(), second[live].float()
    assert (f - s).norm() / f.norm() < 1e-3


def test_default_is_on(gemm):
    """Split-k is enabled; pick_split's wave gate decides per shape."""
    assert int(gemm.exl3_get_moe_acc_cap()) > 0


@pytest.mark.parametrize("rows,block_m", [(256, 16), (512, 32)])
@pytest.mark.parametrize("cap", [0, 1 << 24], ids=["unsplit", "split"])
def test_unowned_blocks_read_as_zero(gemm, rows, block_m, cap):
    """Blocks marked -1 must come back zeroed, not merely unread.

    Without expert parallel a -1 block is only padding and nothing looks at it,
    which is why `test_split_matches_unsplit` above masks those rows out. Under
    expert parallel the same marker covers every block routed to an expert
    another rank owns, and those rows are real (token, expert) pairs:
    exl3_moe_combine gathers them unconditionally and the result goes into an
    all-reduce. The gemm allocates with at::empty, so "skip the block" used to
    mean "return whatever the allocator last put there", and multiplying by a
    zero routing weight does not launder it -- NaN * 0 is NaN.

    The poison below is what a warm server does for free: run once, fill the
    result with NaN, drop it, and the caching allocator hands the same block
    back to the next same-shaped at::empty.
    """
    trellis, svh, a, eids, nr = _operands(rows, block_m)
    run = lambda: gemm.exl3_moe_gemm(a, trellis, svh, svh, eids, nr, [I, I], CB,
                                     block_m, torch.bfloat16)
    prev = int(gemm.exl3_get_moe_acc_cap())
    try:
        gemm.exl3_set_moe_acc_cap(cap)
        poison = run()
        poison.fill_(float("nan"))
        del poison
        out = run()
    finally:
        gemm.exl3_set_moe_acc_cap(prev)

    dead = (eids < 0).repeat_interleave(block_m)
    assert dead.any(), "fixture must mark at least one block unowned"
    assert torch.isfinite(out[dead]).all(), "unowned rows carry NaN into the all-reduce"
    assert int(torch.count_nonzero(out[dead])) == 0, "unowned rows are not zeroed"
