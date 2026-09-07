"""What the MoE intermediate round-trip costs, and what fusing it away is worth.

MiaAI-Lab's exl3_fat_moe folds the gate/up output Hadamard, svh, clamp, SwiGLU,
down_suh and the down projection's input Hadamard into the gate/up GEMM's
epilogue, so the (rows, 2I) intermediate never exists. We materialise it and run
exl3_moe_glu_had_in over it. Our BN_ is already 128 -- "must equal HAD_N: a
block owns whole Hadamard blocks" -- so the same fusion is geometrically
available; this sizes it before anyone builds it.
"""
import time, torch
from cuda_exl3 import ops as _ops
_ops._try_native()
g = torch.ops.cuda_exl3_C

dev = "cuda"
E, H, I, BITS, CB = 288, 4096, 2048, 4, 1
TOPK, BLOCK_M = 8, 64
torch.manual_seed(0)

def ruler():
    a = torch.empty(1 << 31, dtype=torch.bfloat16, device=dev); a.normal_()
    for _ in range(3): a.sum()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): a.sum()
    torch.cuda.synchronize()
    gbs = a.numel() * 2 * 10 / (time.perf_counter() - t0) / 1e9
    del a; torch.cuda.empty_cache(); return gbs

def timeit(f, reps=10):
    for _ in range(3): f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(reps): f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000

gbs = ruler()
EL = 96                                     # local experts at EP over 3 ranks
w13 = torch.randint(-32768, 32767, (EL, H // 16, 2 * I // 16, 16 * BITS),
                    dtype=torch.int16, device=dev)
suh13 = (torch.randn((EL, H), device=dev) * 0.05).half()
svh13 = (torch.randn((EL, 2 * I), device=dev) * 0.05).half()
suh2 = (torch.randn((EL, 1, I), device=dev) * 0.05).half()
print(f"# ruler {gbs:.0f} GB/s   E_local={EL} H={H} I={I} block_m={BLOCK_M}")
print(f"{'M':>6s} {'rows':>7s} {'gemm us':>9s} {'glu us':>8s} {'total':>8s} "
      f"{'saved us':>9s} {'saving':>7s}")
for M in (512, 2048):
    rows = M * TOPK
    nblk = rows // BLOCK_M
    a13 = torch.randn((2, rows, H), dtype=torch.half, device=dev) * 0.05
    eids = torch.randint(0, EL, (nblk,), dtype=torch.int32, device=dev)
    nr = torch.tensor([rows], dtype=torch.int32, device=dev)
    sids = torch.randperm(rows, dtype=torch.int32, device=dev)

    gemm = lambda: g.exl3_moe_gemm(a13, w13, suh13, svh13, eids, nr, [I, I], CB,
                                   BLOCK_M, torch.half, sids, None, M, TOPK)
    inter = gemm()
    a2 = torch.empty((1, rows, I), dtype=torch.half, device=dev)
    glu = lambda: g.exl3_moe_glu_had_in(inter, a2, suh2, eids, nr, BLOCK_M)

    t_gemm, t_glu = timeit(gemm), timeit(glu)
    # Fusing removes: the gemm's extra I columns of output write, and the whole
    # glu kernel (reads 2I, writes I). Four units of rows x I x 2 bytes.
    unit = rows * I * 2
    saved = t_glu + unit / (gbs * 1e3)
    print(f"{M:>6d} {rows:>7d} {t_gemm:>9.1f} {t_glu:>8.1f} "
          f"{t_gemm + t_glu:>8.1f} {saved:>9.1f} "
          f"{saved / (t_gemm + t_glu) * 100:>6.1f}%")
    del a13, inter, a2
    torch.cuda.empty_cache()
