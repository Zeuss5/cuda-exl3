// EXL3 input transform: A_had[g] = had128(x * suh[g]).
//
// ExLlamaV3 folds this into the front of its gemm and pays for it with a
// cooperative launch (the grid-wide sync between transform and matmul).
// Splitting it out costs one extra pass over the activations -- a few
// microseconds, since x is small and L2-resident -- and in exchange the gemm
// becomes an ordinary kernel: no cooperative launch, no grid size tied to the SM
// count, and nothing special to do under CUDA graph capture.
//
// A fused layer (qkv_proj, gate_up_proj) needs one transform per shard, because
// each shard was quantized with its own input scales. All of them are produced
// in a single launch here, loading x once and writing G outputs, rather than one
// launch and one re-read of x per shard.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_common.cuh"
#include "exl3_had.cuh"

namespace vllm_exl3 {

template <typename IN_T>
__global__ void exl3_had_in_kernel(const IN_T* __restrict__ x, half* __restrict__ a_had,
                                   const half* __restrict__ suh, int m, int k, int groups)
{
    int blocks_per_row = k / 128;
    long long total = (long long) m * blocks_per_row;
    int warps_per_block = blockDim.x / 32;
    long long w = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (w >= total) return;

    int row = (int) (w / blocks_per_row);
    int blk = (int) (w % blocks_per_row);
    long long off = (long long) row * k + blk * 128;
    int lane = threadIdx.x & 31;

    for (int g = 0; g < groups; ++g)
        had128_warp_in<IN_T>(x + off, a_had + (long long) g * m * k + off,
                             suh + (long long) g * k + blk * 128, lane);
}

void exl3_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(out.scalar_type() == at::kHalf, "exl3_had_in: out must be float16");
    TORCH_CHECK(suh.scalar_type() == at::kHalf, "exl3_had_in: suh must be float16");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "exl3_had_in: needs contiguous tensors");

    int k = x.size(-1);
    int m = x.numel() / k;
    int groups = suh.dim() == 2 ? (int) suh.size(0) : 1;
    TORCH_CHECK(k % 128 == 0, "exl3_had_in: k must be a multiple of 128, got ", k);
    TORCH_CHECK(suh.numel() == (long long) groups * k, "exl3_had_in: suh size mismatch");
    TORCH_CHECK(out.numel() >= (long long) groups * m * k, "exl3_had_in: out too small");

    long long total_warps = (long long) m * (k / 128);
    const int threads = 256;
    long long blocks = (total_warps + threads / 32 - 1) / (threads / 32);
    auto stream = at::cuda::getCurrentCUDAStream();
    half* o = (half*) out.data_ptr();
    const half* s = (const half*) suh.data_ptr();

    if (x.scalar_type() == at::kHalf)
        exl3_had_in_kernel<half><<<(unsigned) blocks, threads, 0, stream>>>(
            (const half*) x.data_ptr(), o, s, m, k, groups);
    else if (x.scalar_type() == at::kBFloat16)
        exl3_had_in_kernel<__nv_bfloat16><<<(unsigned) blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*) x.data_ptr(), o, s, m, k, groups);
    else
        TORCH_CHECK(false, "exl3_had_in: x must be float16 or bfloat16, got ", x.scalar_type());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm_exl3
