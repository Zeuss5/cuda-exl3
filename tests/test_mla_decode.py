"""Fused sparse-MLA decode kernel.

Checks it against a plain torch reference across batch, head count, topk and
split chunk -- including a chunk that does not divide topk, a short sequence,
and holes in the selection (-1 rows the indexer left empty).
"""
import pytest
import torch

from vllm_exl3 import ops as _ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
D, DV = 576, 512


def _reference(q, kv, sel, seqlens, scale):
    B, H, _ = q.shape
    out = torch.zeros(B, H, DV, device=q.device, dtype=torch.float32)
    for b in range(B):
        rows = sel[b, : int(seqlens[b])].long()
        rows = rows[rows >= 0]
        if rows.numel() == 0:
            continue
        k = kv[rows].float()
        p = torch.softmax((q[b].float() @ k.t()) * scale, dim=-1)
        out[b] = p @ k[:, :DV]
    return out


def _run(B, H, topk, chunk, holes=False, rows=4096, seed=0):
    _ops._try_native()
    dev = "cuda"
    torch.manual_seed(seed)
    kv = torch.randn(rows, D, device=dev, dtype=torch.bfloat16) * 0.05
    q = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16) * 0.05
    sel = torch.stack([torch.randperm(rows, device=dev)[:topk] for _ in range(B)]).int()
    if holes:
        sel[:, ::7] = -1
    sl = torch.full((B,), topk, device=dev, dtype=torch.int32)
    scale = 1.0 / (D ** 0.5)
    got = torch.ops.vllm_exl3_C.mla_decode(q, kv, sel, sl, scale, DV, chunk, 0)
    ref = _reference(q, kv, sel, sl, scale)
    return ((ref - got.float()).norm() / ref.norm()).item()


@pytest.mark.parametrize("B,H,topk,chunk", [
    (1, 16, 2048, 64), (1, 16, 2048, 16), (4, 16, 2048, 32),
    (1, 8, 512, 32), (2, 16, 1024, 128), (1, 16, 300, 64), (1, 16, 2048, 24),
])
def test_matches_reference(B, H, topk, chunk):
    assert _run(B, H, topk, chunk) < 5e-3


def test_selection_holes():
    """-1 marks a slot the indexer left empty; it must contribute nothing."""
    assert _run(2, 16, 1024, 64, holes=True) < 5e-3


def test_chunk_choice_does_not_change_the_answer():
    """The split factor is a scheduling knob, so results must not depend on it."""
    errs = [_run(1, 16, 2048, c, seed=3) for c in (16, 32, 64, 128)]
    assert max(errs) < 5e-3
    assert max(errs) - min(errs) < 1e-3
