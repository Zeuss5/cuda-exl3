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

// ---------------------------------------------------------------------------
// MoE variant: gather the routed rows and apply that row's expert's scales.
//
// Rows arrive already sorted by expert and padded so each expert's run is a
// whole multiple of block_m (vLLM's moe_align_block_size), which is what lets
// the GEMM treat the expert as uniform over a row block. Padding rows are
// zeroed here so they contribute nothing downstream.
// ---------------------------------------------------------------------------

template <typename IN_T>
__global__ void exl3_moe_had_in_kernel(const IN_T* __restrict__ x,
                                       half* __restrict__ a_had,
                                       const half* __restrict__ suh,
                                       const int* __restrict__ sorted_ids,
                                       const int* __restrict__ expert_ids,
                                       const int* __restrict__ n_rows,
                                       int rows, int k, int groups, int block_m,
                                       int top_k, int m_valid)
{
    int blocks_per_row = k / 128;
    long long total = (long long) rows * blocks_per_row;
    int warps_per_block = blockDim.x / 32;
    long long w = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (w >= total) return;

    int row = (int) (w / blocks_per_row);
    // Surplus rows from the worst-case padding are never read by the gemm
    // (it retires those blocks on the same count), so skip them entirely.
    if (n_rows && row >= *n_rows) return;
    int blk = (int) (w % blocks_per_row);
    int lane = threadIdx.x & 31;
    long long dst = (long long) row * k + blk * 128;

    int idx = sorted_ids[row];
    if (idx >= m_valid)
    {
        // Padding row: emit zeros so the matmul result is zero too.
        for (int g = 0; g < groups; ++g)
            ((half4*) (a_had + (long long) g * rows * k + dst))[lane] =
                half4{__float2half2_rn(0.f), __float2half2_rn(0.f)};
        return;
    }

    int token = idx / top_k;
    int e = expert_ids[row / block_m];
    if (e < 0) return;                  // block belongs to no expert
    const IN_T* src = x + (long long) token * k + blk * 128;

    for (int g = 0; g < groups; ++g)
        had128_warp_in<IN_T>(src, a_had + (long long) g * rows * k + dst,
                             suh + ((long long) e * groups + g) * k + blk * 128, lane);
}

void exl3_moe_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh,
                     const at::Tensor& sorted_ids, const at::Tensor& expert_ids,
                     const at::Tensor& n_rows, int64_t block_m, int64_t top_k,
                     int64_t m_valid)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(out.scalar_type() == at::kHalf, "exl3_moe_had_in: out must be float16");
    TORCH_CHECK(suh.dim() == 3, "exl3_moe_had_in: suh must be (experts, groups, k)");
    TORCH_CHECK(sorted_ids.scalar_type() == at::kInt && expert_ids.scalar_type() == at::kInt,
                "exl3_moe_had_in: sorted_ids/expert_ids must be int32");

    int k = (int) x.size(-1);
    int groups = (int) suh.size(1);
    int rows = (int) sorted_ids.numel();
    TORCH_CHECK(k % 128 == 0, "exl3_moe_had_in: k must be a multiple of 128");
    TORCH_CHECK((int) suh.size(2) == k, "exl3_moe_had_in: suh k mismatch");
    TORCH_CHECK(out.numel() >= (long long) groups * rows * k, "exl3_moe_had_in: out too small");
    TORCH_CHECK((long long) expert_ids.numel() * block_m >= rows,
                "exl3_moe_had_in: expert_ids covers ", expert_ids.numel() * block_m,
                " rows but sorted_ids has ", rows);

    long long total_warps = (long long) rows * (k / 128);
    const int threads = 256;
    long long blocks = (total_warps + threads / 32 - 1) / (threads / 32);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int* nr = n_rows.numel() ? n_rows.data_ptr<int>() : nullptr;

    if (x.scalar_type() == at::kHalf)
        exl3_moe_had_in_kernel<half><<<(unsigned) blocks, threads, 0, stream>>>(
            (const half*) x.data_ptr(), (half*) out.data_ptr(), (const half*) suh.data_ptr(),
            sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(), nr, rows, k, groups,
            (int) block_m, (int) top_k, (int) m_valid);
    else if (x.scalar_type() == at::kBFloat16)
        exl3_moe_had_in_kernel<__nv_bfloat16><<<(unsigned) blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*) x.data_ptr(), (half*) out.data_ptr(),
            (const half*) suh.data_ptr(), sorted_ids.data_ptr<int>(),
            expert_ids.data_ptr<int>(), nr, rows, k, groups, (int) block_m, (int) top_k,
            (int) m_valid);
    else
        TORCH_CHECK(false, "exl3_moe_had_in: x must be float16 or bfloat16");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm_exl3
