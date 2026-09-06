"""What the sparse-MLA kernel costs at prefill shapes, and against which ceiling.

#5 ranks MLA prefill sixth at 8.2% of a chunk and marks it "not measured -- the
trace does not carry the selected-key count". That is the whole difficulty: the
kernel gathers `topk` latent rows per query row, and whether the traffic is
`rows x topk` or the far smaller union of those selections depends on how much
consecutive rows overlap and how much of the overlap survives in cache.

So run the same shape twice with the only difference being overlap, exactly as
arm C of the MoE expert-reread bench isolated block re-reads:

  independent  every query row draws its own selection            no reuse
  drifting     row i is row i-1 with a few entries replaced       maximal reuse

If the two cost the same, the kernel is gather-bound at `rows x topk x D` and
there is no reuse to win. If drifting is faster, the union is the real ceiling
and the gap is the lever.

Run: python bench/bench_mla_prefill.py [head_dim]
"""
import sys
import time

import torch

from cuda_exl3 import ops as _ops

_ops._try_native()

dev = "cuda"
torch.manual_seed(0)
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
D = int(_args[0]) if _args else 576
DV = 512
H = 16                      # 64 heads at TP=4
TOPK = 2048                 # GLM index_topk
CTX = 262144                # 288 MB of latent at D=576: not L2-resident anywhere
DRIFT = 2                   # a near-static selection: the cache-friendly bound
# Measured by #5 from the DSA indexer's top-k on a ~70K prefill, 7168 query rows
# pooled: 2049 selected keys per row (at the index_topk ceiling), adjacent-row
# overlap 0.926 min-normalised, Jaccard 0.862. So ~152 of 2048 keys turn over
# per row -- two orders of magnitude more than DRIFT above, which is what makes
# the production arm worth running separately rather than inferring.
PROD_TURNOVER = 0.074


def ruler():
    """Achievable read bandwidth in this same binary, per #5's spec."""
    a = torch.empty(1 << 31, dtype=torch.bfloat16, device=dev)
    a.normal_()
    for _ in range(3):
        a.sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        a.sum()
    torch.cuda.synchronize()
    gbs = a.numel() * 2 * 10 / (time.perf_counter() - t0) / 1e9
    del a
    torch.cuda.empty_cache()
    return gbs


def timeit(f, reps=8):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        f()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000          # us


def selections(rows, kind, ctx=None):
    ctx = ctx or CTX
    if kind == "independent":
        return torch.randint(0, ctx, (rows, TOPK), device=dev, dtype=torch.int32)
    if kind == "production":
        # Row i keeps a random (1 - turnover) of row i-1 and redraws the rest
        # from the context, which reproduces both the measured adjacent overlap
        # and the way the union random-walks up to the context length.
        sel = torch.empty((rows, TOPK), device=dev, dtype=torch.int32)
        cur = torch.randperm(ctx, device=dev)[:TOPK].int()
        sel[0] = cur
        for i in range(1, rows):
            m = torch.rand(TOPK, device=dev) < PROD_TURNOVER
            cur = cur.clone()
            cur[m] = torch.randint(0, ctx, (int(m.sum()),), device=dev,
                                   dtype=torch.int32)
            sel[i] = cur
        return sel
    # Start from one selection and drift it: row i differs from row i-1 in
    # DRIFT positions, so the union over a chunk is topk + rows*DRIFT.
    base = torch.randperm(ctx, device=dev)[:TOPK].int()
    sel = base.repeat(rows, 1)
    if DRIFT:
        pos = torch.randint(0, TOPK, (rows, DRIFT), device=dev)
        new = torch.randint(0, ctx, (rows, DRIFT), device=dev, dtype=torch.int32)
        # cumulative: each row keeps every earlier row's replacements
        for r in range(1, rows):
            sel[r:, pos[r]] = new[r]
    return sel


def main():
    gbs = ruler()
    kv = torch.randn(CTX, D, device=dev, dtype=torch.bfloat16) * 0.05
    print(f"# head_dim={D} heads={H} topk={TOPK} ctx={CTX} "
          f"({CTX * D * 2 / 1e6:.0f} MB latent)")
    print(f"# achievable read bandwidth (bf16 sum, 4 GiB): {gbs:.0f} GB/s")
    print(f"{'rows':>6s} {'selection':>12s} {'us':>9s} {'per-row GB/s':>13s} "
          f"{'%ruler':>7s} {'union GB/s':>11s} {'%ruler':>7s}")
    summary = []
    for rows in (256, 512, 1024, 2048):
        got = {}
        q = torch.randn(rows, H, D, device=dev, dtype=torch.bfloat16) * 0.05
        sl = torch.full((rows,), TOPK, device=dev, dtype=torch.int32)
        for kind in ("independent", "drifting"):
            sel = selections(rows, kind)
            f = lambda: torch.ops.cuda_exl3_C.mla_decode(
                q, kv, sel, sl, 1.0 / (D ** 0.5), DV, 64, 1, 1.0)
            us = timeit(f)
            per_row = rows * TOPK * D * 2
            union = min(int(sel.unique().numel()), CTX) * D * 2
            print(f"{rows:>6d} {kind:>12s} {us:>9.1f} "
                  f"{per_row / us / 1e3:>13.0f} {per_row / us / 1e3 / gbs * 100:>6.0f}% "
                  f"{union / us / 1e3:>11.0f} {union / us / 1e3 / gbs * 100:>6.0f}%")
            got[kind] = us
            del sel

        # The drifting arm touches a few MB, so it is entirely cache-resident
        # and its time is the kernel's compute/issue floor with the traffic
        # taken away. The independent arm has to move rows x topk x D from HBM.
        # Comparing both against the ruler says which one binds, and how much
        # of the two the kernel manages to overlap.
        hbm_us = per_row / gbs / 1e3
        floor = max(got["drifting"], hbm_us)
        summary.append((rows, got["drifting"], hbm_us, got["independent"],
                        got["independent"] / floor))

    print()
    print("Decomposition -- 'compute' is the resident arm, 'hbm' is per-row "
          "traffic at the ruler:")
    print(f"{'rows':>6s} {'compute us':>11s} {'hbm us':>9s} {'actual us':>10s} "
          f"{'vs the larger':>14s}")
    for rows, comp, hbm, act, ratio in summary:
        print(f"{rows:>6d} {comp:>11.0f} {hbm:>9.0f} {act:>10.0f} {ratio:>13.2f}x")


if __name__ == "__main__" and "--ctx" not in sys.argv:
    main()


def ctx_sweep(gbs):
    """Where production sits between the two arms, as a function of context.

    The per-chunk working set is the union of the rows' selections, which for
    the production turnover random-walks upward and is capped by the context
    length. So whether the chunk is cache-resident is decided by the context,
    not by the selection pattern -- and the two parts have very different L2.
    """
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    rows = 1792                                    # #5's steady chunk
    print()
    print(f"# chunk of {rows} rows, turnover {PROD_TURNOVER:.3f}/row "
          f"(adjacent overlap {1 - PROD_TURNOVER:.3f}), L2 = {l2 / 2**20:.0f} MiB")
    print(f"{'ctx':>8s} {'latent MB':>10s} {'arm':>12s} {'us':>9s} "
          f"{'working set':>12s} {'vs L2':>7s}")
    for ctx in (32768, 71680, 262144):
        kv = torch.randn(ctx, D, device=dev, dtype=torch.bfloat16) * 0.05
        q = torch.randn(rows, H, D, device=dev, dtype=torch.bfloat16) * 0.05
        sl = torch.full((rows,), TOPK, device=dev, dtype=torch.int32)
        for kind in ("drifting", "production", "independent"):
            sel = selections(rows, kind, ctx)
            f = lambda: torch.ops.cuda_exl3_C.mla_decode(
                q, kv, sel, sl, 1.0 / (D ** 0.5), DV, 64, 1, 1.0)
            us = timeit(f)
            ws = int(sel.unique().numel()) * D * 2
            print(f"{ctx:>8d} {ctx * D * 2 / 1e6:>10.0f} {kind:>12s} {us:>9.1f} "
                  f"{ws / 2**20:>10.0f} MiB {ws / l2:>6.2f}x")
            del sel
        del kv, q
        torch.cuda.empty_cache()


if __name__ == "__main__" and "--ctx" in sys.argv:
    ctx_sweep(ruler())
