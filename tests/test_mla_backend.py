"""The vLLM-facing half of the sparse-MLA backend.

The kernel itself is covered by test_mla_decode. What is left to check is the
glue: that the top-k list vLLM hands us -- per-request logical token ids with
-1 holes -- lands on the right cache rows after the block table is applied.
"""
import pytest
import torch

from vllm_exl3 import ops as _ops

_ops._try_native()
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch.ops, "vllm_exl3_C"),
    reason="needs the compiled kernel on a GPU",
)


class _Meta:
    """The fields forward_mqa reads out of FlashInferMLASparseMetadata."""

    def __init__(self, req_id_per_token, block_table, block_size):
        self.req_id_per_token = req_id_per_token
        self.block_table = block_table
        self.block_size = block_size


def _impl(topk_buf, kv_lora_rank, scale, fp8=False):
    from vllm_exl3.attention import Exl3MLASparseImpl

    o = object.__new__(Exl3MLASparseImpl)
    o.topk_indices_buffer = topk_buf
    o.kv_lora_rank = kv_lora_rank
    o.scale = scale
    o._fp8_cache = fp8
    o._no_seqlens = torch.empty(0, dtype=torch.int32, device="cuda")
    return o


class _Layer:
    def __init__(self, k_scale):
        self._k_scale_float = k_scale


@pytest.mark.parametrize("head_size", [512, 576])
@pytest.mark.parametrize("holes", [False, True])
def test_forward_mqa_matches_a_gather_reference(head_size, holes):
    torch.manual_seed(0)
    dev = "cuda"
    heads, rank, topk, block_size = 16, 512, 128, 64
    reqs, blocks_per_req = 3, 6                    # 384 slots per request
    ctx = blocks_per_req * block_size
    num_blocks = reqs * blocks_per_req + 2

    cache = torch.randn(num_blocks, block_size, head_size, device=dev,
                        dtype=torch.bfloat16) * 0.05
    # Deliberately non-contiguous request layout, so a wrong block table shows.
    block_table = torch.randperm(num_blocks, device=dev)[: reqs * blocks_per_req]
    block_table = block_table.view(reqs, blocks_per_req).int()

    req_id_per_token = torch.arange(reqs, device=dev, dtype=torch.int32)
    q = torch.randn(reqs, heads, head_size, device=dev, dtype=torch.bfloat16) * 0.05
    logical = torch.stack(
        [torch.randperm(ctx, device=dev)[:topk] for _ in range(reqs)]
    ).int()
    if holes:
        logical[:, ::5] = -1

    scale = 1.0 / head_size**0.5
    impl = _impl(logical, rank, scale)
    out, lse = impl.forward_mqa(q, cache, _Meta(req_id_per_token, block_table,
                                                block_size), None)
    assert lse is None

    flat = cache.view(-1, head_size)
    ref = torch.zeros(reqs, heads, rank, device=dev, dtype=torch.float32)
    for r in range(reqs):
        ids = logical[r]
        ids = ids[ids >= 0].long()
        slots = block_table[r][ids // block_size].long() * block_size + ids % block_size
        k = flat[slots].float()
        p = ((q[r].float() @ k.T) * scale).softmax(-1)
        ref[r] = p @ k[:, :rank]

    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 0.02, err


@pytest.mark.parametrize("head_size", [512, 576])
def test_forward_mqa_with_an_fp8_cache(head_size):
    """The scale is folded into the softmax scale and the output, never applied
    per element, so the result must match a dequantised reference."""
    torch.manual_seed(0)
    dev = "cuda"
    heads, rank, topk, block_size = 16, 512, 256, 64
    reqs, blocks_per_req = 2, 8
    ctx = blocks_per_req * block_size
    num_blocks = reqs * blocks_per_req + 2

    ref_cache = torch.randn(num_blocks, block_size, head_size, device=dev) * 0.05
    k_scale = ref_cache.abs().max().item() / 448.0
    cache = (ref_cache / k_scale).to(torch.float8_e4m3fn)
    deq = cache.to(torch.float32) * k_scale

    block_table = torch.randperm(num_blocks, device=dev)[: reqs * blocks_per_req]
    block_table = block_table.view(reqs, blocks_per_req).int()
    req_id_per_token = torch.arange(reqs, device=dev, dtype=torch.int32)
    q = torch.randn(reqs, heads, head_size, device=dev, dtype=torch.bfloat16) * 0.05
    logical = torch.stack(
        [torch.randperm(ctx, device=dev)[:topk] for _ in range(reqs)]
    ).int()

    scale = 1.0 / head_size**0.5
    impl = _impl(logical, rank, scale, fp8=True)
    out, _ = impl.forward_mqa(q, cache,
                              _Meta(req_id_per_token, block_table, block_size),
                              _Layer(k_scale))

    flat = deq.view(-1, head_size)
    ref = torch.zeros(reqs, heads, rank, device=dev, dtype=torch.float32)
    for r in range(reqs):
        ids = logical[r].long()
        slots = block_table[r][ids // block_size].long() * block_size + ids % block_size
        k = flat[slots]
        ref[r] = ((q[r].float() @ k.T) * scale).softmax(-1) @ k[:, :rank]

    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 0.02, err


def test_rejects_an_unreadable_cache_layout():
    from vllm_exl3.attention import Exl3MLASparseImpl

    with pytest.raises(NotImplementedError, match="bfloat16 or e4m3"):
        Exl3MLASparseImpl(
            num_heads=16, head_size=512, scale=1.0, num_kv_heads=1,
            alibi_slopes=None, sliding_window=None, kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None, attn_type="decoder",
            kv_sharing_target_layer_name=None,
        )


@pytest.fixture(scope="module")
def _vllm_world():
    """A one-rank vLLM world, so the impl can be built the way vLLM builds it."""
    import os

    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29593")
    cfg = VllmConfig()
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method="env://", local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        yield
    if dist.is_initialized():
        dist.destroy_process_group()


def test_constructs_the_way_vllm_constructs_it(_vllm_world):
    """GLM-5.3-Flash's exact attention dimensions, through the real base class."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    from vllm_exl3.attention import Exl3MLASparseImpl

    with set_current_vllm_config(VllmConfig()):
        kv_b_proj = ColumnParallelLinear(
            512, 16 * (256 + 256), bias=False, return_bias=False
        )
        impl = Exl3MLASparseImpl(
            num_heads=16, head_size=512, scale=0.04, num_kv_heads=1,
            alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
            logits_soft_cap=None, attn_type="decoder",
            kv_sharing_target_layer_name=None,
            topk_indices_buffer=torch.zeros(8, 2048, dtype=torch.int32,
                                            device="cuda"),
            q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=256,
            qk_rope_head_dim=0, qk_head_dim=256, v_head_dim=256,
            kv_b_proj=kv_b_proj,
        )
    assert impl.is_sparse
    assert not impl.can_return_lse_for_decode
    assert impl.kv_lora_rank == 512


def test_plugin_binds_the_custom_backend_slot():
    import vllm_exl3
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    vllm_exl3.register()
    assert (
        AttentionBackendEnum.CUSTOM.get_path()
        == "vllm_exl3.attention.Exl3MLASparseBackend"
    )
