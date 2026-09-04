"""Zero-padding a routed expert's intermediate must be exact, not approximate.

A trellis cannot be sliced by three, so TP=3 either goes through expert parallel
-- which costs 1.3-1.5x on whichever rank draws the most experts -- or the
intermediate is padded up to a width that does slice. 2048 -> 2304 is the
smallest pad that gives 3 x 768 on a 128-column Hadamard boundary.

The cheap way to build that sidecar is to not re-encode anything: widen w13's
output, set the padded columns' `svh` to zero, and leave whatever codes happen
to sit behind w2's padded input rows. That is only safe if the zeros are exact.
Trellis dequantisation of a "zero" block would not be, which is the whole reason
the scales carry it instead of the codes.

Guards the recipe end to end:
  (a) svh = 0 on a padded w13 output block => intermediate exactly 0 there,
  (b) it survives silu(gate) * up,
  (c) the final output does not depend on w2's padded-row codes at all.
"""
import pytest
import torch

from cuda_exl3 import ops as _ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, I, BITS, CB, BM = 8, 2048, 768, 4, 1, 16   # I = 768 = one TP=3 rank of 2304
PAD, ROWS = 128, 64


@pytest.fixture
def gemm():
    _ops._try_native()
    return torch.ops.cuda_exl3_C


def _prefix(gemm):
    """Everything up to and including the down projection's input.

    Built once and shared, so the two down projections below differ in exactly
    one thing: the codes behind w2's padded input rows. Rebuilding the prefix
    instead would also vary allocator state, and the outputs here are small
    enough that a stray uninitialised lane shows up as a one-ULP difference,
    which is indistinguishable from the effect being measured.
    """
    dev = "cuda"
    torch.manual_seed(0)
    tr13 = torch.randint(-32768, 32767, (E, H // 16, 2 * I // 16, 16 * BITS),
                         dtype=torch.int16, device=dev)
    sv13 = (torch.randn((E, 2 * I), device=dev) * 0.05).half()
    sv13[:, I - PAD:I] = 0                      # padded gate columns
    sv13[:, 2 * I - PAD:2 * I] = 0              # padded up columns
    su13 = (torch.randn((E, 2, H), device=dev) * 0.05).half()

    eids = torch.randint(0, E, (ROWS // BM,), dtype=torch.int32, device=dev)
    nr = torch.tensor([ROWS], dtype=torch.int32, device=dev)
    x = torch.randn((ROWS, H), dtype=torch.bfloat16, device=dev) * 0.05
    sid = torch.arange(ROWS, dtype=torch.int32, device=dev)

    a13 = torch.empty((2, ROWS, H), dtype=torch.half, device=dev)
    gemm.exl3_moe_had_in(x, a13, su13, sid, eids, nr, BM, 1, ROWS)
    inter = gemm.exl3_moe_gemm(a13, tr13, su13, sv13, eids, nr, [I, I], CB, BM,
                               torch.bfloat16)

    su2 = (torch.randn((E, 1, I), device=dev) * 0.05).half()
    a2 = torch.empty((1, ROWS, I), dtype=torch.half, device=dev)
    gemm.exl3_moe_glu_had_in(inter, a2, su2, eids, nr, BM)
    sv2 = (torch.randn((E, H), device=dev) * 0.05).half()
    tr2 = torch.randint(-32768, 32767, (E, I // 16, H // 16, 16 * BITS),
                        dtype=torch.int16, device=dev)
    return inter, a2, su2, sv2, tr2, eids, nr


def _down(gemm, a2, su2, sv2, tr2, eids, nr):
    return gemm.exl3_moe_gemm(a2, tr2, su2, sv2, eids, nr, [H], CB, BM, torch.bfloat16)


def test_padded_columns_are_exactly_zero(gemm):
    inter, a2, *_ = _prefix(gemm)
    assert (inter[:, I - PAD:I] == 0).all(), "padded gate columns are not exactly zero"
    assert (inter[:, 2 * I - PAD:2 * I] == 0).all(), "padded up columns are not exactly zero"
    # silu(0) * 0 = 0, so the pad stays dead through the activation.
    assert (a2[0, :, I - PAD:] == 0).all(), "pad survived silu(gate)*up as nonzero"


def test_output_ignores_w2_padded_row_codes(gemm):
    _, a2, su2, sv2, tr2, eids, nr = _prefix(gemm)
    plain = _down(gemm, a2, su2, sv2, tr2, eids, nr)
    scrambled_tr2 = tr2.clone()
    torch.manual_seed(99)
    scrambled_tr2[:, (I - PAD) // 16:, :, :] = torch.randint(
        -32768, 32767, (E, PAD // 16, H // 16, 16 * BITS),
        dtype=torch.int16, device="cuda")
    scrambled = _down(gemm, a2, su2, sv2, scrambled_tr2, eids, nr)
    assert torch.equal(plain, scrambled), (
        "the down projection read its padded input rows -- the sidecar would then "
        "depend on codes nobody chose")

    # The assertion above says a change does nothing, so prove the change is
    # real: the same scramble one k-group lower, on rows the pad does not cover,
    # must move the output.
    live_tr2 = tr2.clone()
    torch.manual_seed(99)
    live_tr2[:, (I - PAD) // 16 - 1:(I - PAD) // 16, :, :] = torch.randint(
        -32768, 32767, (E, 1, H // 16, 16 * BITS), dtype=torch.int16, device="cuda")
    assert not torch.equal(plain, _down(gemm, a2, su2, sv2, live_tr2, eids, nr)), (
        "scrambling live rows changed nothing either -- the test is vacuous")
