"""The MoE down-projection's SwiGLU is folded into its input transform.

These check the fused kernel against the two-kernel path it replaced: the same
result, no writes into blocks that belong to no expert, and -- because the
product is now formed in fp32 rather than the GEMM's output dtype -- accuracy
that is no worse.
"""
import pytest
import torch

from cuda_exl3 import ops as _ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPES = [(768, 8, 32, 5), (1536, 4, 16, 7), (2048, 3, 64, 3)]


def _run(dt, I, E, block_m, nblk):
    _ops._try_native()
    ops = torch.ops.cuda_exl3_C
    dev = "cuda"
    rows = nblk * block_m
    torch.manual_seed(0)
    inter = torch.randn((rows, 2 * I), dtype=dt, device=dev) * 2.0
    suh = torch.randn((E, 1, I), dtype=torch.half, device=dev)
    expert_ids = torch.randint(0, E, (nblk,), dtype=torch.int32, device=dev)
    # moe_align_block_size marks blocks belonging to no expert with -1.
    expert_ids[nblk // 2] = -1
    n_rows = torch.tensor([rows], dtype=torch.int32, device=dev)
    empty = torch.empty(0, dtype=torch.int32, device=dev)

    def had(act):
        out = torch.zeros((1, rows, I), dtype=torch.half, device=dev)
        ops.exl3_moe_had_in(act, out, suh, empty, expert_ids, n_rows, block_m, 1, rows)
        return out

    old = had((torch.nn.functional.silu(inter[:, :I]) * inter[:, I:]).contiguous())
    f = inter.float()
    exact = had((torch.nn.functional.silu(f[:, :I]) * f[:, I:]).half().contiguous())

    got = torch.zeros((1, rows, I), dtype=torch.half, device=dev)
    ops.exl3_moe_glu_had_in(inter, got, suh, expert_ids, n_rows, block_m)

    live = (expert_ids >= 0).repeat_interleave(block_m)
    return old[0], got[0], exact[0], live


@pytest.mark.parametrize("dt", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_unfused_path(dt, shape):
    old, got, _, live = _run(dt, *shape)
    o, g = old[live].float(), got[live].float()
    # bf16 in: the two differ by where the SwiGLU is evaluated, so allow the
    # reference's own bf16 rounding. test_accuracy_no_worse pins the direction.
    tol = 5e-3 if dt is torch.bfloat16 else 1e-3
    assert (o - g).norm() / o.norm() < tol


@pytest.mark.parametrize("dt", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_no_worse(dt, shape):
    old, got, exact, live = _run(dt, *shape)
    e = exact[live].float()
    err_old = (old[live].float() - e).norm() / e.norm()
    err_new = (got[live].float() - e).norm() / e.norm()
    assert err_new <= err_old * 1.05


@pytest.mark.parametrize("shape", SHAPES)
def test_leaves_expertless_blocks_untouched(shape):
    _, got, _, live = _run(torch.bfloat16, *shape)
    assert got[~live].abs().max().item() == 0
