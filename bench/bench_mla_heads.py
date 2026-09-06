"""What a head group costs, and why TP=3 pays for 32 heads to use 22.

Found by #5 on a 48-SM part: 22 heads cost 13-16% more *per head* than 16 in
the compute-only arm, with the sign flipping in the bandwidth-bound arm -- so
the penalty lives in the compute path.

The mechanism is in the launcher: hpb is 16, or 8 when `wide >= 3`, and
`hgroups = ceil(H / hpb)`. Every head group independently gathers the same topk
rows. 64 heads split three ways is 22 per rank, which needs two groups where
TP=4's 16 needs one -- so the gather runs twice for 1.375x the heads.

Note the 8th kernel argument is `wide`, a shape selector, not hpb itself:
sweeping it over 8 and 16 selects hpb=8 both times and hides the effect.
"""
import itertools, torch
from cuda_exl3 import ops as _ops
_ops._try_native()

dev, D, DV, TOPK = "cuda", 576, 512, 2048
CTX = 262144
torch.manual_seed(0)
kv = torch.randn(CTX, D, device=dev, dtype=torch.bfloat16) * 0.05

def timeit(f, reps=8):
    for _ in range(3): f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(reps): f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000

rows = 1792
sel = torch.randint(0, CTX, (rows, TOPK), device=dev, dtype=torch.int32)
# production-like: 7.4% turnover per row
cur = torch.randperm(CTX, device=dev)[:TOPK].int()
sel[0] = cur
for i in range(1, rows):
    m = torch.rand(TOPK, device=dev) < 0.074
    cur = cur.clone(); cur[m] = torch.randint(0, CTX, (int(m.sum()),), device=dev, dtype=torch.int32)
    sel[i] = cur
sl = torch.full((rows,), TOPK, device=dev, dtype=torch.int32)

print(f"{'H':>3s} {'wide':>5s} {'hpb':>4s} {'hgroups':>8s} {'best us':>9s} {'chunk':>6s} {'us/head':>8s}")
base = {}
for H in (16, 22, 32):
    q = torch.randn(rows, H, D, device=dev, dtype=torch.bfloat16) * 0.05
    for hpb in (1, 2, 3, 4):
        best = None
        for chunk in (32, 64, 96, 128, 256):
            f = lambda c=chunk, h=hpb: torch.ops.cuda_exl3_C.mla_decode(
                q, kv, sel, sl, 1.0 / (D ** 0.5), DV, c, h, 1.0)
            try: us = timeit(f)
            except Exception: continue
            if best is None or us < best[0]: best = (us, chunk)
        if best is None: continue
        _hpb = min(H, 8 if (hpb >= 3 and H >= 16) else 16)
        hg = (H + _hpb - 1) // _hpb
        print(f"{H:>3d} {hpb:>5d} {_hpb:>4d} {hg:>8d} {best[0]:>9.1f} {best[1]:>6d} {best[0]/H:>8.1f}")
        base.setdefault(H, best[0]); base[H] = min(base[H], best[0])
print()
print(f"per-head cost, best over hpb:  H=16 {base[16]/16:.1f} us   "
      f"H=22 {base[22]/22:.1f} us ({base[22]/22/(base[16]/16):.3f}x)   "
      f"H=32 {base[32]/32:.1f} us ({base[32]/32/(base[16]/16):.3f}x)")
print(f"H=22 total vs H=16 total: {base[22]/base[16]:.3f}x for 1.375x the heads")
