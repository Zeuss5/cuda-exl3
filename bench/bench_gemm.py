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
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="*", default=[
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.mlp.up_proj",
        "model.language_model.layers.3.mlp.down_proj",
    ])
    ap.add_argument("--m", nargs="*", type=int,
                    default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    from cuda_exl3 import ops
    from exllamav3.ext import exllamav3_ext as ext

    for name in args.layers:
        t, suh, svh = load_tensor(args.model, name)
        bits = t.shape[2] // 16
        k, n = t.shape[0] * 16, t.shape[1] * 16
        wmb = t.numel() * 2 / 1e6
        print(f"\n=== {name.split('layers.')[1]}  k={k} n={n} bits={bits} "
              f"trellis={wmb:.1f}MB ===")
        print(f"{'m':>6} {'ours us':>9} {'exl3 us':>9} {'speedup':>8} "
              f"{'TFLOPS':>8} {'GB/s':>8} {'SoL us':>8} {'%SoL':>6} {'relerr':>9}")

        for m in args.m:
            x = torch.randn((m, k), dtype=torch.half, device="cuda") / (k ** 0.5)
            xh = torch.empty_like(x)
            c_ref = torch.empty((m, n), dtype=torch.half, device="cuda")
            suh2 = suh.view(1, -1)

            ext.exl3_gemm(x, t, c_ref, suh, xh, svh, -1, False, True, 0)
            c_ours = ops.exl3_linear(x, t, suh2, svh, [n], 2)
            torch.cuda.synchronize()

            denom = c_ref.float().abs().mean().item()
            err = (c_ours.float() - c_ref.float()).abs().mean().item() / max(denom, 1e-9)

            t_ours = bench(lambda: ops.exl3_linear(x, t, suh2, svh, [n], 2))
            t_ref = bench(lambda: ext.exl3_gemm(x, t, c_ref, suh, xh, svh, -1, False, True, 0))

            s, wbytes, flops = sol_us(m, k, n, bits)
            us = t_ours * 1e6
            print(f"{m:6d} {us:9.1f} {t_ref*1e6:9.1f} {t_ref/t_ours:7.2f}x "
                  f"{flops/t_ours/1e12:8.1f} {wbytes/t_ours/1e9:8.0f} {s:8.1f} "
                  f"{100*s/us:5.0f}% {err:9.2e}")


if __name__ == "__main__":
    main()
