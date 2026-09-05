"""Loading an EXL3 tensor into a parameter whose dim vLLM has padded.

EXL3 weights cannot be zero-extended -- a trellis is not a dense tensor -- but a
padded *output* dim needs no extension: `svh` scales elementwise after the output
Hadamard, so zeroing it on the pad makes those columns exactly zero whatever the
trellis holds. The condition is that the pad may not share a 128-column Hadamard
block with real output, because the block mixes across its columns before svh is
applied.

A padded *input* dim is the mirror case: the producing layer's pad columns are
already exact zeros, so the padded input positions carry zeros and whatever `suh`
holds there cannot reach the output -- but the slice must not run off the end of
the checkpoint tensor while getting there.
"""
import pytest
import torch

from cuda_exl3.parameter import Exl3SuhParameter

HAD_BLOCK = 128


def _suh(shard_size, tp_rank, tp_size, monkeypatch):
    # The base parameter asks vLLM for its rank at construction; there is no
    # distributed group in a unit test.
    import vllm.model_executor.parameter as vp
    monkeypatch.setattr(vp, "get_tensor_model_parallel_rank", lambda: tp_rank)
    if hasattr(vp, "get_tensor_model_parallel_world_size"):
        monkeypatch.setattr(vp, "get_tensor_model_parallel_world_size", lambda: tp_size)
    return Exl3SuhParameter(
        data=torch.full((1, shard_size), float("nan")),
        weight_loader=lambda *a, **k: None,
    )


def test_row_parallel_slice_within_the_checkpoint(monkeypatch):
    """Unpadded: every rank's slice exists and is copied verbatim."""
    full = torch.arange(768, dtype=torch.half)
    for rank in range(3):
        p = _suh(256, rank, 3, monkeypatch)
        p.load_row_parallel_weight(full)
        assert torch.equal(p.data[0], full[rank * 256:(rank + 1) * 256])


def test_row_parallel_slice_past_the_end_is_zero_filled(monkeypatch):
    """Padded input dim: the last rank's slice runs past the checkpoint.

    Copy what exists, zero the rest. The alternative -- narrowing by the padded
    width -- reads off the end of the tensor, which is an out-of-bounds read
    rather than a wrong number, so it fails loudly on some builds and silently
    on others.
    """
    real = 640                      # checkpoint width
    shard = 256                     # padded per-rank width, 3 x 256 = 768 > 640
    p = _suh(shard, 2, 3, monkeypatch)
    p.load_row_parallel_weight(torch.arange(real, dtype=torch.half))
    avail = real - 2 * shard        # 128 real entries on the last rank
    assert avail == 128
    assert torch.equal(p.data[0, :avail],
                       torch.arange(2 * shard, real, dtype=torch.half))
    assert torch.all(p.data[0, avail:] == 0), "pad must be zero, not NaN"
    assert torch.isfinite(p.data[0]).all()


def test_row_parallel_rank_entirely_in_the_pad(monkeypatch):
    """A rank whose whole slice is pad gets zeros, not a negative-length narrow."""
    p = _suh(256, 2, 3, monkeypatch)
    p.load_row_parallel_weight(torch.arange(400, dtype=torch.half))
    assert torch.all(p.data[0] == 0)
    assert torch.isfinite(p.data[0]).all()


@pytest.mark.parametrize("n_real,padded,ok", [
    (154880, 155136, True),    # 154880 = 1210 x 128: pad is whole blocks
    (154880, 154944, False),   # +64: pad straddles the last real block
    (51456, 51712, True),      # one rank's share, 51456 = 402 x 128
    (51400, 51712, False),     # real does not end on a block boundary
])
def test_the_alignment_rule(n_real, padded, ok):
    """The gate the loader applies, stated as arithmetic.

    Real output must end on a 128 boundary; then and only then does the pad
    occupy whole Hadamard blocks and zeroing svh leave the real columns alone.
    """
    # Both ends: real must end on a boundary so the pad starts on one, and the
    # padded total must itself be whole blocks or the layer is malformed for a
    # kernel whose output tile is 128 wide.
    accepted = (padded >= n_real
                and n_real % HAD_BLOCK == 0
                and padded % HAD_BLOCK == 0)
    assert accepted == ok
