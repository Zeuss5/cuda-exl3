"""Sparse-MLA decode: this plugin's kernel against b12x, under a cold cache.

Two things make a sparse-MLA benchmark flatter than it should be:

* A cache small enough to sit in L2. GLM's latent is 1 KB per token, so an
  8k-row cache is 8 MB against a 128 MB L2 -- every read hits.
* One selection replayed. Even against a large cache, replaying the same
  top-k list makes the *selected* rows L2-resident after the first pass.

So: a 1 GiB cache, and a fresh selection for every call inside the graph.

The roofline is not the 1.79 TB/s on the spec sheet. A stream copy on this card
reaches 1461 GB/s, and this access pattern is a scatter of 1 KB rows, not a
stream; `torch.index_select` over the same rows is the honest ceiling, measured
here rather than assumed.
"""
import itertools
import os

import torch

from vllm_exl3 import ops as _ops

_ops._try_native()

DEV = "cuda"
DV = 512
TOPK = 2048
HEADS = 16              # 64 heads at TP=4
ROWS = 1 << 20          # 1 GiB of latent cache at head_dim 512
INNER = 20              # distinct selections per graph


def graph_us(fs, reps=30):
    for f in fs[:3]:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fs:
            f()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
    beg.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return beg.elapsed_time(end) / reps / len(fs) * 1000


def gather_ceiling(kv, batch, sels):
    """Read-only-ish reference: gather the same rows, nothing else."""
    out = torch.empty(batch * TOPK, kv.shape[1], device=DEV, dtype=kv.dtype)
    fs = [
        (lambda s=s: torch.index_select(kv, 0, s.view(-1).long(), out=out))
        for s in sels
    ]
    return graph_us(fs)


def bench_ours(kv, q, sels, seqlens):
    d = kv.shape[1]
    best = None
    for chunk, wide in itertools.product(
            (16, 32, 48, 64, 96, 128, 192, 256), (1, 2, 3, 4)):
        us = graph_us([
            (lambda s=s, c=chunk, w=wide: torch.ops.vllm_exl3_C.mla_decode(
                q, kv, s, seqlens, 1.0 / d**0.5, DV, c, w, 1.0))
            for s in sels
        ])
        if best is None or us < best[0]:
            best = (us, chunk, wide)
    return best


def bench_b12x(batch, d, sels):
    import b12x.attention.sparse_mla as sm

    page = 64
    caps = sm.Caps(
        device=torch.device(DEV), num_q_heads=HEADS, max_q_rows=batch,
        max_width=TOPK, dtype=torch.bfloat16, kv_dtype=torch.bfloat16,
        head_dim=d, v_head_dim=DV, mode="decode", max_batch=batch,
        max_kv_rows=ROWS, max_page_table_width=TOPK // page + 2,
        max_chunks_per_row=64,
    )
    plan = sm.plan(caps)
    kv = torch.randn(ROWS // page, page, d, device=DEV, dtype=torch.bfloat16) * 0.05
    q = torch.randn(batch, HEADS, d, device=DEV, dtype=torch.bfloat16) * 0.05
    sl = torch.full((batch,), TOPK, device=DEV, dtype=torch.int32)
    scratch = {
        sp.name: torch.empty(sp.shape, dtype=sp.dtype, device=sp.device or DEV)
        for sp in plan.scratch_specs()
    }
    best = None
    for splits in (None, 4, 8, 16, 32):
        try:
            fs = []
            for s in sels:
                binding = plan.bind(scratch=scratch, q=q, selected_indices=s,
                                    cache_seqlens_int32=sl,
                                    nsa_cache_seqlens_int32=sl)
                fs.append(lambda b=binding, sp=splits: sm.run_decode(
                    kv_cache=kv, binding=b, sm_scale=1.0 / d**0.5,
                    v_head_dim=DV, forced_num_splits=sp))
            us = graph_us(fs)
        except Exception as e:
            if os.environ.get("MLA_VERBOSE"):
                print("  b12x", splits, type(e).__name__, str(e)[:120])
            continue
        if best is None or us < best[0]:
            best = (us, splits)
    del kv
    torch.cuda.empty_cache()
    return best


def main():
    torch.manual_seed(0)
    print(f"cache {ROWS} rows, topk {TOPK}, heads {HEADS}, "
          f"{INNER} distinct selections per graph\n")
    for d in (576, 512):
        f32 = torch.randn(ROWS, d, device=DEV) * 0.05
        kv = f32.to(torch.bfloat16)
        kv8 = (f32 / (f32.abs().max().item() / 448.0)).to(torch.float8_e4m3fn)
        del f32
        torch.cuda.empty_cache()
        print(f"head_dim {d} ({d * ROWS * 2 / 2**30:.1f} GiB bf16 cache)")
        print(f"{'batch':>6s} {'gather':>9s} {'ours':>9s} {'ours fp8':>9s} "
              f"{'b12x':>9s} {'fp8 GB/s':>9s}")
        for batch in (1, 4, 16, 32):
            q = torch.randn(batch, HEADS, d, device=DEV, dtype=torch.bfloat16) * 0.05
            sels = [torch.randint(0, ROWS, (batch, TOPK), device=DEV,
                                  dtype=torch.int32) for _ in range(INNER)]
            sl = torch.full((batch,), TOPK, device=DEV, dtype=torch.int32)
            ceil_us = gather_ceiling(kv, batch, sels)
            ours = bench_ours(kv, q, sels, sl)
            ours8 = bench_ours(kv8, q, sels, sl)
            theirs = bench_b12x(batch, d, sels)
            gb = batch * TOPK * d / ours8[0] / 1e3
            print(f"{batch:>6d} {ceil_us:>8.1f}u {ours[0]:>8.1f}u {ours8[0]:>8.1f}u "
                  f"{('%.1fu' % theirs[0]) if theirs else '     n/a':>9s} "
                  f"{gb:>9.0f}")
        del kv, kv8
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
