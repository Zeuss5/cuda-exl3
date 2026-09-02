import os
import sys
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(THIS_DIR, "vllm_exl3", "csrc")


def _default_arch_list() -> str:
    """Pick the arch list to build for.

    Honour TORCH_CUDA_ARCH_LIST if the caller set it. Otherwise build only for
    the GPUs actually present, which keeps the (already long) nvcc time down
    during development. Falls back to the set of archs EXL3 kernels support at
    all if no device is visible at build time (e.g. a CI wheel builder).
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    try:
        import torch

        if torch.cuda.is_available():
            caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
            return ";".join(f"{maj}.{min}" for maj, min in sorted(caps))
    except Exception:
        pass
    # mma.m16n8k16 + cp.async are the floor -> sm_80 and up.
    return "8.0;8.6;8.9;9.0;12.0"


os.environ["TORCH_CUDA_ARCH_LIST"] = _default_arch_list()
print(f"[vllm-exl3] building for TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

sources = [
    os.path.join("vllm_exl3", "csrc", f)
    for f in [
        "bindings.cpp",
        "hadamard.cu",
        "gemm.cu",
        "mla_decode.cu",
    ]
]

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    # 177: declared but never referenced (template instantiation fan-out)
    "-Xcudafe", "--diag_suppress=177",
]
if os.environ.get("VLLM_EXL3_DEBUG"):
    nvcc_flags += ["-g", "-G", "-DVLLM_EXL3_DEBUG"]

missing = [s for s in sources if not os.path.exists(s)]
if missing:
    # Lets the pure-Python plugin be installed before the kernels are written;
    # at runtime it falls back to the exllamav3 backend (see vllm_exl3/ops.py).
    print(f"[vllm-exl3] WARNING: skipping CUDA extension, missing sources: {missing}")
    ext_modules = []
    cmdclass = {}
else:
    ext_modules = [
        CUDAExtension(
            name="vllm_exl3._C",
            sources=sources,
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
