"""This plugin's EXL3 GEMM against the 2x-DGX-Spark kit's `exl3_fat_gemm`.

Same GPU, same trellis tensor, same shapes -- the comparison neither project
could make from its own numbers, because 73.5 TFLOP/s on GB10 and 320 TFLOPS on
sm_120 say nothing about each other.

Their kernel is not vendored here. Point --fat-src at a checkout of
https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark (Apache-2.0; the
kernel itself descends from the MIT-licensed MiaAI-Lab kit) and --exl3-src at
https://github.com/turboderp-org/exllamav3, whose headers it includes.

It only accepts K4 MCG trellis tensors, so the weights must be a 4-bit mcg
checkpoint -- GLM-5.3-Flash-tr3-4bpw is one, and its expert shapes are the ones
that matter here.
"""
import argparse
import os
import time

import torch
from safetensors import safe_open


def build_fat(fat_src, exl3_src, workdir):
    """Compile their kernel where its `../util.h` includes resolve."""
    from torch.utils.cpp_extension import load

    ext = os.path.join(exl3_src, "exllamav3", "exllamav3_ext")
    os.makedirs(os.path.join(workdir, "quant"), exist_ok=True)
    for src, dst in (
        (f"{fat_src}/overlay/exl3_fat_gemm.cu", "quant/exl3_fat_gemm.cu"),
        (f"{fat_src}/overlay/exl3_fat_gemm.cuh", "quant/exl3_fat_gemm.cuh"),
    ):
        with open(src, "rb") as f, open(os.path.join(workdir, dst), "wb") as g:
            g.write(f.read())
    for rel in ("util.h", "util.cuh", "ptx.cuh", "compat.cuh",
                "quant/exl3_dq.cuh", "quant/hadamard_inner.cuh", "quant/codebook.cuh"):
        link = os.path.join(workdir, rel)
        if not os.path.exists(link):
            os.symlink(os.path.join(ext, rel), link)
    binding = os.path.join(workdir, "binding.cpp")
    with open(binding, "w") as f:
        f.write('#include <torch/extension.h>\n'
                '#include "quant/exl3_fat_gemm.cuh"\n'
                'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
                '    m.def("exl3_fat_gemm", &exl3_fat_gemm, "fat EXL3 GEMM");\n}\n')
    return load(name="spark_fat_gemm",
                sources=[binding, os.path.join(workdir, "quant/exl3_fat_gemm.cu")],
                extra_include_paths=[workdir],
                extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


def timeit(fn, iters=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/shadeform/vllm/models/GLM-5.3-Flash-tr3-4bpw")
    ap.add_argument("--shard", default="model-00001-of-00120.safetensors")
    ap.add_argument("--prefix", default="model.language_model.layers.10.mlp.experts.0.")
    ap.add_argument("--fat-src", required=True)
    ap.add_argument("--exl3-src", required=True)
    ap.add_argument("--workdir", default="/tmp/spark_fat_bench")
    ap.add_argument("--m", default="128,512,1024,2048,3584,4096")
    a = ap.parse_args()

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.makedirs(a.workdir, exist_ok=True)
    fat = build_fat(a.fat_src, a.exl3_src, a.workdir)

    from exllamav3.ext import exllamav3_ext as ext

    from cuda_exl3 import ops

    path = os.path.join(a.model, a.shard)

    def get(name):
        with safe_open(path, "pt", device="cuda") as f:
            return f.get_tensor(name)

    for proj in ("gate_proj", "down_proj"):
        tr = get(a.prefix + proj + ".trellis").contiguous()
        svh = get(a.prefix + proj + ".svh").contiguous()
        suh = get(a.prefix + proj + ".suh").contiguous()
        k, n = tr.shape[0] * 16, tr.shape[1] * 16
        print(f"\n=== {proj}  k={k} n={n} ===")
        print(f"{'m':>6} {'ours':>9} {'fat':>9} {'exl3':>9} | {'ours TF':>8} "
              f"{'fat TF':>8} {'exl3 TF':>8} | {'ours/fat':>9} {'agree':>9}")
        for m in [int(v) for v in a.m.split(",")]:
            x = torch.randn(m, k, device="cuda", dtype=torch.half) * 0.1
            flop = 2.0 * m * k * n
            out_ours = ops.exl3_linear(x, tr, suh.unsqueeze(0), svh, [n], 1)
            t_ours = timeit(lambda: ops.exl3_linear(x, tr, suh.unsqueeze(0), svh, [n], 1))

            xh = torch.empty_like(x)
            out_fat = torch.empty(m, n, device="cuda", dtype=torch.float)

            def run_fat():
                ext.had_r_128(x, xh, suh, None, 1.0)      # their exl3.py does this
                fat.exl3_fat_gemm(xh, tr, out_fat, svh, 4, True, False)

            run_fat()
            t_fat = timeit(run_fat)

            c_ref = torch.empty(m, n, device="cuda", dtype=torch.half)
            xh2 = torch.empty_like(x)

            def run_ref():
                ext.exl3_gemm(x, tr, c_ref, suh, xh2, svh, -1, False, True, 0)

            run_ref()
            t_ref = timeit(run_ref)

            rel = ((out_fat.half() - out_ours.float().half()).abs().max()
                   / out_ours.abs().max().clamp(min=1e-6)).item()
            print(f"{m:>6} {t_ours:>8.1f}u {t_fat:>8.1f}u {t_ref:>8.1f}u | "
                  f"{flop / t_ours / 1e6:>8.1f} {flop / t_fat / 1e6:>8.1f} "
                  f"{flop / t_ref / 1e6:>8.1f} | {t_fat / t_ours:>8.2f}x {rel:>9.1e}")


if __name__ == "__main__":
    main()
