// Shared device-side primitives for the EXL3 kernels.
//
// The tensor-core fragment types, the PTX wrappers and the codebook/trellis
// decoders below follow ExLlamaV3 (https://github.com/turboderp-org/exllamav3,
// MIT licence, (c) turboderp). The EXL3 bit-packing and the procedural codebook
// are part of the on-disk format, so these have to match it exactly.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Fragment types for mma.m16n8k16
// ---------------------------------------------------------------------------

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
    __device__ const T& operator[](int i) const { return elems[i]; }
};

using FragA = Vec<half2, 4>;   // 16x16 A fragment
using FragB = Vec<half2, 2>;   // 16x8  B fragment
using FragC = Vec<float, 4>;   // 16x8  fp32 accumulator
using FragC_h = Vec<half2, 2>; // 16x8  fp16 accumulator (half the registers)

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

// ---------------------------------------------------------------------------
// PTX wrappers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k16(const FragA& a, const FragB& b, FragC& c)
{
    const uint32_t* A = reinterpret_cast<const uint32_t*>(&a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(&b);
    float* C = reinterpret_cast<float*>(&c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

// fp16-accumulate variant. Note there is no bf16 accumulator for mma: bf16
// inputs always accumulate to fp32. fp16 in / fp16 out is the only form that
// actually halves the accumulator register cost, which is what buys a larger M
// tile (and so more MMA work per dequantized weight).
__device__ __forceinline__ void mma_m16n8k16_h(const FragA& a, const FragB& b, FragC_h& c)
{
    const uint32_t* A = reinterpret_cast<const uint32_t*>(&a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(&b);
    uint32_t* C = reinterpret_cast<uint32_t*>(&c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(C[0]), "+r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* glob_ptr)
{
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem),
                 "l"(glob_ptr), "n"(16));
}

__device__ __forceinline__ void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;\n" ::);
}

// Load four 8x8 b16 tiles straight into mma A-fragment layout.
__device__ __forceinline__ void ldsm4(FragA& frag_a, const void* smem_ptr)
{
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(smem));
}

// ---------------------------------------------------------------------------
// Bitfield extraction helpers used by the trellis decoder
// ---------------------------------------------------------------------------

static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    uint64_t result64;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(result64) : "l"(value), "r"(offset), "r"(length));
    return static_cast<uint32_t>(result64);
}

#define FSHF_IMM(dst, lo, hi, imm) \
    asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) \
    asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

// Predicated 16-byte async copy: when `pred` is false the destination is
// zero-filled instead (cp.async's src-size operand), which is what out-of-range
// rows of an A tile need.
__device__ __forceinline__ void cp_async16_pred(void* smem_ptr, const void* glob_ptr, bool pred)
{
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int src_size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem),
                 "l"(glob_ptr), "n"(16), "r"(src_size));
}
