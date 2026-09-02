"""Our fused sparse-MLA decode vs the roofline, sweeping the split chunk."""
import torch
from vllm_exl3 import ops as _ops
_ops._try_native()
dev = "cuda"; torch.manual_seed(0)
D, DV, HBM = 576, 512, 1.79e12
SMS = torch.cuda.get_device_properties(0).multi_processor_count

def graph_time(f, reps=50):
    for _ in range(5): f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    torch.cuda.synchronize()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    a, b_ = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): g.replay()
    b_.record(); torch.cuda.synchronize()
    return a.elapsed_time(b_) / reps * 1000

H, topk, rows = 16, 2048, 8192
kv = torch.randn(rows, D, device=dev, dtype=torch.bfloat16) * 0.05
print(f"SMs={SMS}  H={H} topk={topk} (GLM at TP=4)")
print(f"{'batch':>6s} {'chunk':>6s} {'splits':>7s} {'blocks':>7s} {'us':>8s} {'GB/s':>8s} {'%roof':>7s}")
for B in (1, 4, 16):
    q = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16) * 0.05
    sel = torch.stack([torch.randperm(rows, device=dev)[:topk] for _ in range(B)]).int()
    sl = torch.full((B,), topk, device=dev, dtype=torch.int32)
    kvb = B * topk * D * 2
    sol = kvb / HBM * 1e6
    best = None
    for chunk in (8, 11, 16, 24, 32, 64, 128):
        f = lambda c=chunk: torch.ops.vllm_exl3_C.mla_decode(q, kv, sel, sl, 1.0/(D**0.5), DV, c)
        try: us = graph_time(f)
        except Exception as e:
            print(f"{B:>6d} {chunk:>6d} chunk failed: {str(e)[:50]}"); continue
        splits = (topk + chunk - 1)//chunk
        print(f"{B:>6d} {chunk:>6d} {splits:>7d} {splits*B:>7d} {us:>8.1f} {kvb/us/1e3:>8.0f} {sol/us*100:>6.1f}%")
        if best is None or us < best[0]: best = (us, chunk)
    print(f"       -> best chunk={best[1]} at {best[0]:.1f} us ({sol/best[0]*100:.1f}% of roofline, "
          f"roofline {sol:.1f} us)  [b12x: {'20.5' if B==1 else '21.0' if B==4 else '30.6'} us]\n")
