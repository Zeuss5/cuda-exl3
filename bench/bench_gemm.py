"""EXL3 GEMM correctness + roofline benchmark.

Reports achieved TFLOPS and GB/s against the machine's measured speed of light.
An EXL3 matmul reads k*n*bits/8 bytes of trellis and does 2*m*k*n flops, so its
arithmetic intensity is 16*m/bits flop/byte: memory bound for small batches,
compute bound for large ones. Speed of light is the max of the two limits.
"""

import argparse
import json
import time

import torch
from safetensors import safe_open

# Measured on RTX PRO 6000 Blackwell (see bench/roofline.py); override with --peak*
PEAK_TFLOPS = 400.0
PEAK_GBS = 1520.0

MODEL = "/home/shadeform/vllm/models/Qwen3.8-27B-EXL3-5.5bpw"


def synth_tensor(k, n, bits):
    """Random trellis of the right shape, for machines with no checkpoint.

    Trellis decode is data-independent, so timings are the real thing; the
    decoded values are noise, which only matters for the relerr column -- and
    that column still means something, because both kernels decode the same
    noise and must agree.
    """
    t = torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16),
                      dtype=torch.int16, device="cuda")
    suh = (torch.randn(k, device="cuda") * 0.1).half()
    svh = (torch.randn(n, device="cuda") * 0.1).half()
    return t, suh, svh


def load_tensor(model, name):
    idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
    cache = {}

    def get(k):
        f = idx[k]
        if f not in cache:
            cache[f] = safe_open(f"{model}/{f}", framework="pt", device="cuda:0")
        return cache[f].get_tensor(k)

    return get(f"{name}.trellis"), get(f"{name}.suh").half(), get(f"{name}.svh").half()


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def sol_us(m, k, n, bits):
    """Speed-of-light time in microseconds."""
    wbytes = k * n * bits / 8 + m * k * 2 + m * n * 2
    mem = wbytes / (PEAK_GBS * 1e9)
    flops = 2.0 * m * k * n
    comp = flops / (PEAK_TFLOPS * 1e12)
    return max(mem, comp) * 1e6, wbytes, flops


def main():
    global PEAK_TFLOPS, PEAK_GBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="*", default=[
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.mlp.up_proj",
        "model.language_model.layers.3.mlp.down_proj",
    ])
    ap.add_argument("--m", nargs="*", type=int,
                    default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--synthetic", nargs="*", metavar="K,N,BITS",
                    help="benchmark these shapes with random trellis instead of a "
                         "checkpoint, e.g. --synthetic 4096,2048,4 512,4096,4")
    ap.add_argument("--peak-tflops", type=float, default=PEAK_TFLOPS,
                    help="this machine's dense bf16 peak; the default is an RTX "
                         "PRO 6000's and %%SoL is meaningless elsewhere")
    ap.add_argument("--peak-gbs", type=float, default=PEAK_GBS,
                    help="this machine's measured HBM bandwidth")
    ap.add_argument("--bank", type=int, default=1, metavar="N",
                    help="cycle N distinct weight tensors so the trellis does not "
                         "sit in L2. On a small-L2 device (GB10 has 24 MB) a "
                         "single tensor is resident and %%SoL reads over 100%%")
    args = ap.parse_args()
    PEAK_TFLOPS, PEAK_GBS = args.peak_tflops, args.peak_gbs
    print(f"speed-of-light reference: {PEAK_TFLOPS:.0f} TFLOPS / {PEAK_GBS:.0f} GB/s"
          + ("  (defaults -- pass --peak-tflops/--peak-gbs for this machine)"
             if (PEAK_TFLOPS, PEAK_GBS) == (400.0, 1520.0) else ""))

    from cuda_exl3 import ops
    try:
        from exllamav3.ext import exllamav3_ext as ext
    except ImportError:
        ext = None
        print("exllamav3 not installed: skipping the reference column. Its kernels "
              "are the comparison baseline, not a dependency of this plugin.")

    if args.synthetic is not None:
        shapes = args.synthetic or ["4096,2048,4", "2048,4096,4", "4096,4096,4"]
        work = [(f"synthetic {sh}", tuple(int(v) for v in sh.split(","))) for sh in shapes]
    else:
        work = [(n, None) for n in args.layers]

    for name, shape in work:
        if shape is not None:
            t, suh, svh = synth_tensor(*shape)
        else:
            t, suh, svh = load_tensor(args.model, name)
        bits = t.shape[2] // 16
        bank = [t] + ([synth_tensor(t.shape[0] * 16, t.shape[1] * 16, bits)[0]
                       for _ in range(args.bank - 1)] if args.bank > 1 else [])
        k, n = t.shape[0] * 16, t.shape[1] * 16
        wmb = t.numel() * 2 / 1e6
        label = name if shape is not None else name.split("layers.")[1]
        print(f"\n=== {label}  k={k} n={n} bits={bits} "
              f"trellis={wmb:.1f}MB ===")
        print(f"{'m':>6} {'ours us':>9} {'exl3 us':>9} {'speedup':>8} "
              f"{'TFLOPS':>8} {'GB/s':>8} {'SoL us':>8} {'%SoL':>6} {'relerr':>9}")

        for m in args.m:
            x = torch.randn((m, k), dtype=torch.half, device="cuda") / (k ** 0.5)
            xh = torch.empty_like(x)
            c_ref = torch.empty((m, n), dtype=torch.half, device="cuda")
            suh2 = suh.view(1, -1)

            c_ours = ops.exl3_linear(x, t, suh2, svh, [n], 2)
            if ext is not None:
                ext.exl3_gemm(x, t, c_ref, suh, xh, svh, -1, False, True, 0)
                torch.cuda.synchronize()
                denom = c_ref.float().abs().mean().item()
                err = (c_ours.float() - c_ref.float()).abs().mean().item() / max(denom, 1e-9)
            else:
                # No reference kernel: check ours against itself in m=32 chunks,
                # which still catches a batch-dependent bug.
                ref = torch.cat([ops.exl3_linear(x[i:i + 32], t, suh2, svh, [n], 2)
                                 for i in range(0, m, 32)])
                denom = ref.float().abs().mean().item()
                err = (c_ours.float() - ref.float()).abs().mean().item() / max(denom, 1e-9)

            # Cycle distinct weights so the trellis is not simply L2-resident.
            t_ours = bench(lambda i=[0]: (
                ops.exl3_linear(x, bank[(i.__setitem__(0, i[0] + 1) or i[0]) % len(bank)],
                                suh2, svh, [n], 2)))
            t_ref = (bench(lambda: ext.exl3_gemm(x, t, c_ref, suh, xh, svh, -1,
                                                 False, True, 0))
                     if ext is not None else float("nan"))

            s, wbytes, flops = sol_us(m, k, n, bits)
            us = t_ours * 1e6
            ref_us = f"{t_ref * 1e6:9.1f}" if ext is not None else f"{'-':>9}"
            spd = f"{t_ref / t_ours:7.2f}x" if ext is not None else f"{'-':>8}"
            print(f"{m:6d} {us:9.1f} {ref_us} {spd} "
                  f"{flops/t_ours/1e12:8.1f} {wbytes/t_ours/1e9:8.0f} {s:8.1f} "
                  f"{100*s/us:5.0f}% {err:9.2e}")


if __name__ == "__main__":
    main()
