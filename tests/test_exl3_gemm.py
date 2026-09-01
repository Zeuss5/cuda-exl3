"""Correctness tests for the EXL3 kernels.

Run against a real EXL3 checkpoint:

    VLLM_EXL3_TEST_MODEL=/path/to/model pytest tests/ -v

Without a checkpoint the tests are skipped: EXL3 trellis data cannot be
meaningfully synthesized, since the codebook and bit packing are part of the
on-disk format.
"""

import json
import os

import pytest
import torch

MODEL = os.environ.get(
    "VLLM_EXL3_TEST_MODEL", "/home/shadeform/vllm/models/Qwen3.8-27B-EXL3-5.5bpw"
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL), reason=f"no EXL3 checkpoint at {MODEL}"
)


@pytest.fixture(scope="module")
def tensors():
    from safetensors import safe_open

    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    cache = {}

    def get(name):
        f = idx[name]
        if f not in cache:
            cache[f] = safe_open(f"{MODEL}/{f}", framework="pt", device="cuda:0")
        return cache[f].get_tensor(name)

    return get


LAYERS = [
    "model.language_model.layers.3.self_attn.q_proj",   # k=5120  n=12288
    "model.language_model.layers.3.mlp.up_proj",        # k=5120  n=17408
    "model.language_model.layers.3.mlp.down_proj",      # k=17408 n=5120
]


def _load(get, name):
    return get(f"{name}.trellis"), get(f"{name}.suh").half(), get(f"{name}.svh").half()


def _reference(trellis, suh, svh, x):
    """Dense fp16 weight in the original basis, via exllamav3's reconstruct."""
    from exllamav3.ext import exllamav3_ext as ext

    bits = trellis.shape[2] // 16
    w = torch.empty(
        (trellis.shape[0] * 16, trellis.shape[1] * 16), dtype=torch.half, device="cuda"
    )
    ext.reconstruct_had_slice(w, trellis, suh, svh, bits, False, True, 0)
    return x.float() @ w.float()


@pytest.mark.parametrize("name", LAYERS)
@pytest.mark.parametrize("m", [1, 7, 16, 31, 32, 64, 128, 129, 256, 512, 1024])
def test_gemm_matches_dense_reference(tensors, name, m):
    """Covers every BM tier and both the fused and split-k epilogues."""
    from vllm_exl3 import ops

    trellis, suh, svh = _load(tensors, name)
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16

    torch.manual_seed(m)
    x = torch.randn((m, k), dtype=torch.half, device="cuda") / (k ** 0.5)
    out = ops.exl3_linear(x, trellis, suh.view(1, -1), svh, [n], cb=2)

    want = _reference(trellis, suh, svh, x)
    rel = (out.float() - want).abs().mean().item() / want.abs().mean().item()
    # fp16 accumulation over k=5120..17408; the dense reference accumulates in fp32
    assert rel < 5e-3, f"{name} m={m}: relative error {rel:.3e}"


@pytest.mark.parametrize("name", LAYERS)
def test_multi_shard_matches_single(tensors, name):
    """Splitting a tensor into two declared shards must not change the result.

    Fused layers rely on this: one trellis spanning every shard, each shard
    addressed by offset with its own suh row, all in one launch.
    """
    from vllm_exl3 import ops

    trellis, suh, svh = _load(tensors, name)
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    m = 64

    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.half, device="cuda") / (k ** 0.5)

    whole = ops.exl3_linear(x, trellis, suh.view(1, -1), svh, [n], cb=2)

    # Same suh for both shards, so the maths is identical -- only the launch
    # geometry and the shard-map lookup differ.
    half_n = (n // 2) // 128 * 128
    two = ops.exl3_linear(x, trellis, suh.view(1, -1).repeat(2, 1), svh,
                          [half_n, n - half_n], cb=2)

    # Not bit-exact: split-k is chosen from the column count, so the two calls
    # reduce partial sums in a different order (fp32 atomics are unordered).
    torch.testing.assert_close(whole, two, rtol=5e-3, atol=1e-4)


@pytest.mark.parametrize("name", LAYERS)
def test_bf16_activations(tensors, name):
    """bf16 in -> bf16 out, converted inside the kernels."""
    from vllm_exl3 import ops

    trellis, suh, svh = _load(tensors, name)
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    m = 64
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") / (k ** 0.5)
    out = ops.exl3_linear(x, trellis, suh.view(1, -1), svh, [n], cb=2)
    assert out.dtype == torch.bfloat16

    want = _reference(trellis, suh, svh, x.half())
    rel = (out.float() - want).abs().mean().item() / want.abs().mean().item()
    assert rel < 1e-2, f"{name}: bf16 relative error {rel:.3e}"


def test_deterministic_mode_is_bit_exact(tensors, monkeypatch):
    """With split-k disabled every call must be bit-identical."""
    from vllm_exl3 import ops

    monkeypatch.setattr(ops, "DETERMINISTIC", True)
    trellis, suh, svh = _load(tensors, LAYERS[2])
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    m = 16
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.half, device="cuda") / (k ** 0.5)
    a = ops.exl3_linear(x, trellis, suh.view(1, -1), svh, [n], cb=2)
    b = ops.exl3_linear(x, trellis, suh.view(1, -1), svh, [n], cb=2)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_accumulator_is_left_zeroed(tensors):
    """Split-k relies on the epilogue re-zeroing the accumulator.

    If it did not, the second call would add to stale partial sums.
    """
    from vllm_exl3 import ops

    trellis, suh, svh = _load(tensors, LAYERS[2])  # down_proj: narrow n -> splits
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    m = 16

    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.half, device="cuda") / (k ** 0.5)
    a = ops.exl3_linear(x, trellis, suh.view(1, -1), svh, [n], cb=2)
    b = ops.exl3_linear(x, trellis, suh.view(1, -1), svh, [n], cb=2)
    # Repeated calls must agree: the epilogue re-zeroes the accumulator, so the
    # second call does not add to stale partial sums.
    torch.testing.assert_close(a, b, rtol=5e-3, atol=1e-4)
