"""What TP all-reduce costs on a box with no P2P, and what the link could give.

Both GB10 and the RTX PRO 6000 report NS for every peer pair, so vLLM disables
its CustomAllreduce and every TP collective goes through NCCL's host-staged
transport. #5 measured that at 28% of the physical pair capacity on GB10; this
is the same question here, plus the ruler a replacement would have to beat.

Run: torchrun --nproc-per-node=4 bench/bench_allreduce.py
"""

import os
import time

import torch
import torch.distributed as dist


def timeit(fn, iters, graph=False):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        run = g.replay
    else:
        run = fn
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6      # us


def pcie_ruler(dev):
    """One GPU's own PCIe bandwidth, the ceiling any host-staged path pays."""
    n = 64 << 20
    h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(n, dtype=torch.uint8, device=dev)
    out = {}
    for name, src, dst in (("h2d", h, d), ("d2h", d, h)):
        for _ in range(3):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        out[name] = n * 20 / (time.perf_counter() - t0) / 1e9
    return out


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl")

    if rank == 0:
        r = pcie_ruler(dev)
        print(f"# world={world}  pcie h2d {r['h2d']:.1f} GB/s   d2h {r['d2h']:.1f} GB/s")
        print(f"{'bytes':>10s} {'eager us':>9s} {'graph us':>9s} "
              f"{'algBW':>8s} {'busBW':>8s} {'GLM-5.3 shape':>22s}")
    dist.barrier()

    # hidden 4096, bf16: one all-reduce per layer carries M x 4096 x 2 bytes.
    shapes = [(m, f"M={m} tokens") for m in (1, 8, 16, 64, 256, 1024, 2048, 8192)]
    for m, label in shapes:
        x = torch.randn(m, 4096, dtype=torch.bfloat16, device=dev)
        nbytes = x.numel() * x.element_size()
        iters = 200 if nbytes < (4 << 20) else 50

        def fn():
            dist.all_reduce(x)

        eager = timeit(fn, iters)
        graph = timeit(fn, iters, graph=True)
        if rank == 0:
            alg = nbytes / (graph * 1e-6) / 1e9
            bus = alg * 2 * (world - 1) / world
            print(f"{nbytes:>10d} {eager:>9.1f} {graph:>9.1f} "
                  f"{alg:>7.1f}G {bus:>7.1f}G {label:>22s}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
