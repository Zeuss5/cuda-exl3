// Fused sparse-MLA decode for SM120 (consumer Blackwell).
//
// GLM-5.3-Flash routes each query to `topk` selected latent rows. In MLA the
// key and the value are the *same* cached row -- K uses all `D` dims, V the
// first `DV` -- so a block that owns all query heads for a chunk of keys reads
// each row once and reuses it across every head. That reuse is the whole point:
// parallelising over heads instead would re-read the cache H times and put the
// kernel H x above its own roofline.
//
// The work at decode is small (topk x D bytes, ~1.3 us of HBM at topk=2048), so
// occupancy, not bandwidth, is what a naive kernel loses to. Keys are therefore
// split across blocks flash-decoding style, each block keeping an online softmax
// and a partial output, and a second pass merges the partials by log-sum-exp.
//
// Only mma-free math is used here: the per-key work is a dot product and an
// axpy, both of which map to plain FMA and warp shuffles, so nothing depends on
// wgmma/TMA and the kernel is portable to any SM80+ part.

// Where the time actually goes (ncu, batch 1, topk 2048, 16 heads at TP=4):
//
//     duration 21.95 us   DRAM 6.9%   L1/TEX 59.8%   compute 23.8%
//     warp cycles per issued instruction 10.08 -- warps stall ~90% of the time
//     stalls/issue: mio_throttle 1.93, short_scoreboard 1.66,
//                   long_scoreboard 1.52, wait 1.49, not_selected 1.10
//     achieved occupancy 33.2% against a theoretical 66.7%
//
// So this is bound by the shared-memory/LSU pipe, not by DRAM (6.9%) and not by
// math (23.8%). The cost is 18 separate 2-byte shared loads per key per lane:
// each wastes half of the 128 B/cycle the pipe delivers and occupies a whole
// MIO slot.
//
// Three fixes for that were tried and each made it slower; they are recorded so
// the next pass does not repeat them:
//
//   * Hoisting the k row into registers once per key and reusing it across both
//     heads and both projections (68 shared reads -> 18). 28.7 -> 30.8 us. It
//     removes redundant reads but lengthens the live range enough to cost more
//     than it saves.
//   * 16 warps per block instead of 8 (one head per warp), which drops the
//     kernel from 109 to 64 registers and doubles warps per SM. 28.7 -> 32.8 us:
//     at batch 1 the grid is only ~64 blocks on 188 SMs, so occupancy per SM was
//     never the binding constraint.
//   * Vectorising the shared reads by giving each lane a contiguous 16-dim slice
//     (18 scalar loads -> 3 int4/bf16x2 loads). 28.7 -> 35.6 us, because
//     contiguous ownership makes lane L read shared word L*8, so lanes 0, 4,
//     8 ... collide on bank 0 -- an 8-way conflict. The strided layout this
//     kernel uses is conflict-free. Vectorising here needs an XOR swizzle of the
//     16-byte chunks, the same trick gemm.cu applies to its A tiles.
//
// The structural answer is to stop doing QK as an FMA chain plus a warp-wide
// shuffle reduction per key and make it an mma: a tile of keys is a matrix, so
// S = Q @ K^T maps onto m16n8k16 with m = 16 heads exactly, reading K from
// shared through ldmatrix (vectorised and conflict-free by construction), and
// O += P @ V is a second mma. That also folds the reduce kernel's launch into
// the same pass. It is a rewrite rather than a tweak, which is why it is not in
// this version.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>
#include <map>
#include <algorithm>

#include "exl3_common.cuh"

// bf16 -> fp32 is a 16-bit shift, so a packed pair unpacks with two ALU ops and
// no round trip through memory. Taking the address of a loaded int4 to read it
// as bf16 would have spilled it to local.
__device__ __forceinline__ void bf16x2_f32(uint32_t u, float& a, float& b)
{
    a = __uint_as_float(u << 16);
    b = __uint_as_float(u & 0xffff0000u);
}

// The other direction, round-to-nearest-even, packed two at a time. Done with
// bit math rather than __float2bfloat16 so nothing has its address taken: an
// int4 built through a pointer spills to local memory.
// fp8 e4m3 pair -> bf16 pair. Both conversions are exact: e4m3 carries three
// mantissa bits against bf16's seven, and its 448 maximum is far inside range.
__device__ __forceinline__ uint32_t e4m3x2_bf16(uint32_t packed)
{
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t) packed));
    float lo, hi;
    asm("cvt.f32.f16 %0, %1;" : "=f"(lo) : "h"((uint16_t) (h2 & 0xffffu)));
    asm("cvt.f32.f16 %0, %1;" : "=f"(hi) : "h"((uint16_t) (h2 >> 16)));
    const uint32_t x = __float_as_uint(lo), y = __float_as_uint(hi);
    return ((x + 0x7fffu + ((x >> 16) & 1u)) >> 16)
         | ((y + 0x7fffu + ((y >> 16) & 1u)) & 0xffff0000u);
}

__device__ __forceinline__ uint32_t f32x2_bf16(float a, float b)
{
    const uint32_t x = __float_as_uint(a), y = __float_as_uint(b);
    const uint32_t rx = (x + 0x7fffu + ((x >> 16) & 1u)) >> 16;
    const uint32_t ry = (y + 0x7fffu + ((y >> 16) & 1u));
    return rx | (ry & 0xffff0000u);
}

namespace vllm_exl3 {

#ifndef MLA_MAX_HEADS
#define MLA_MAX_HEADS 32
#endif

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// One block owns (batch b, key-split s) and every query head.
//   grid  = (splits, batch)
//   block = NWARPS * 32, each warp taking HEADS_PER_WARP heads
// One block owns (batch b, key-split s) and every query head.
//   grid  = (splits, batch)
//   block = NWARPS * 32, each warp taking HPW heads for the whole chunk
//
// HPW is a template parameter, not a runtime bound: the accumulator has to live
// in registers, and a runtime-indexed local array spills every byte of it to
// local memory (v1 did exactly that and ran 3.4x slower than the kernel it was
// meant to beat). Keys are staged a tile at a time through cp.async so several
// latent rows are in flight instead of one.
// Both matmuls run on tensor cores. A tile of keys is a matrix, so
// S = Q @ K^T is an mma with m = 16 heads exactly, and `row.col` wants B stored
// [n][k] = [key][dim], which is already the cache layout. O += P @ V is a second
// mma with the roles rotated -- m = dim, n = head, k = key -- so V arrives as an
// A fragment through the transposing ldmatrix, and P, produced [head][key], is
// the B fragment as-is.
//
// Softmax runs once per tile rather than once per key. That is what makes the
// mma possible at all (the accumulator can only be rescaled between mmas) and it
// also cuts the rescale traffic by a factor of TILE.
//
// The shared row stride is padded by 8 halves: at the natural 576 the stride is
// 1152 B = 288 words, a multiple of 32, so every group of lanes reading down a
// column would land on the same four banks -- an 8-way conflict. At 584 the rows
// step by 4 banks and every ldmatrix covers all 32 exactly once.
template <int D, int DV, int NWARPS, int TILE, typename KT>
__global__ __launch_bounds__(NWARPS * 32) void mla_decode_partial_kernel(
    const __nv_bfloat16* __restrict__ q,     // (B, H, D)
    const KT* __restrict__ kv,               // (rows, D) latent cache
    const int* __restrict__ sel,             // (B, topk) row ids, -1 = empty
    const int* __restrict__ seqlens,         // (B) valid entries of sel
    // (B, H, splits, DV): splits contiguous, so the merge walks them 1 KB apart
    // instead of H * DV * 2 = 16 KB apart, which is a different page every time.
    __nv_bfloat16* __restrict__ part_o,
    float* __restrict__ part_m,              // (B, splits, H)
    float* __restrict__ part_l,              // (B, splits, H)
    int H, int topk, int splits, int chunk, float scale, int hpb)
{
    constexpr int KS = D + 8;                // padded shared row stride
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int MMA_M = 16;                // heads per QK tile
    constexpr int NBLK = TILE / 8;           // QK n-blocks of 8 keys
    constexpr int NBLK_E = NBLK < NWARPS ? NBLK : NWARPS;
    constexpr int KSL = NWARPS / NBLK_E;     // leftover warps split D instead
    constexpr int KSTEPS = (D / 16) / KSL;
    constexpr int MTILES = DV / 16;          // PV m-tiles (dims)
    constexpr int MTPW = MTILES / NWARPS;    // m-tiles per warp
    constexpr int NT = MMA_M / 8;            // PV n-tiles (heads)
    constexpr int PS = TILE + 8;             // padded P row stride
    // An fp8 cache is read into registers, widened, and written to shared as
    // bf16, so there is nothing for cp.async to do and only one buffer is
    // needed; the load latency is hidden by prefetching the next tile into
    // registers instead. The dequant scale never appears here: it folds into
    // the softmax scale for QK and into the merged output for PV.
    constexpr bool KV8 = !std::is_same<KT, __nv_bfloat16>::value;
    constexpr int NBUF = KV8 ? 1 : 2;
    constexpr int EPL = KV8 ? 16 : 8;        // cache elements per 16-byte load
    constexpr int NPF = (TILE * (D / EPL) + NWARPS * 32 - 1) / (NWARPS * 32);
    static_assert(NTHREADS >= MMA_M * TILE, "softmax pass needs one thread per score");
    static_assert(TILE % 16 == 0, "PV mma consumes 16 keys at a time");

    // Q needs 16 x KS halves and a K tile needs TILE x KS. At TILE = 16 those
    // are the same size, so Q is staged into the second K buffer, read out into
    // register fragments, and the buffer is handed back to the pipeline. Double
    // buffering therefore costs no shared memory at all.
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* k_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);      // NBUF x TILE x KS
    __nv_bfloat16* q_s = k_s + (NBUF - 1) * TILE * KS;                    // aliases a buffer
    __nv_bfloat16* p_s = k_s + NBUF * TILE * KS;                          // 16 x PS
    float* s_s = reinterpret_cast<float*>(p_s + MMA_M * PS);              // KSL x 16 x TILE
    float* m_s = s_s + KSL * MMA_M * TILE;                                // 16
    float* l_s = m_s + MMA_M;                                             // 16
    float* c_s = l_s + MMA_M;                                             // 16
    int* sel_s = reinterpret_cast<int*>(c_s + MMA_M);                     // chunk

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int s = blockIdx.x;
    const int hg = blockIdx.y;
    const int b = blockIdx.z;
    const int h0 = hg * hpb;

    const int lo = s * chunk;
    const int valid = seqlens ? seqlens[b] : topk;
    const int hi = min(lo + chunk, valid);

    // Q for this block's heads, staged once, 16 bytes at a time -- one scalar
    // half per iteration meant 36 dependent global loads before the first tile.
    constexpr int QV = D / 8;
    for (int i = tid; i < MMA_M * QV; i += NTHREADS) {
        const int r = i / QV, c = (i % QV) * 8;
        const int h = h0 + r;
        int4 v = int4{0, 0, 0, 0};
        if (r < hpb && h < H) v = *reinterpret_cast<const int4*>(q + ((size_t) b * H + h) * D + c);
        *reinterpret_cast<int4*>(q_s + r * KS + c) = v;
    }
    if (tid < MMA_M) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
    for (int i = tid; i < hi - lo; i += NTHREADS) sel_s[i] = sel[(size_t) b * topk + lo + i];

    // acc[t][n] holds C[dim][head] for m-tile (warp + t*NWARPS), n-tile n.
    FragC acc[MTPW][NT];
#pragma unroll
    for (int t = 0; t < MTPW; ++t)
#pragma unroll
        for (int nn = 0; nn < NT; ++nn)
#pragma unroll
            for (int f = 0; f < 4; ++f) acc[t][nn][f] = 0.f;

    __syncthreads();

    // Q's A fragments never change, so read them once and free the buffer.
    const int nb_w = warp % NBLK_E, ksl = warp / NBLK_E;
    FragA q_f[KSTEPS];
#pragma unroll
    for (int ki = 0; ki < KSTEPS; ++ki)
        ldsm4(q_f[ki], q_s + ((lane & 7) + 8 * ((lane >> 3) & 1)) * KS
                            + (ksl * KSTEPS + ki) * 16 + (lane >> 4) * 8);
    __syncthreads();

    const int ntiles = (hi - lo + TILE - 1) / TILE;

    // The selection list is a gather: reading it inside the staging loop would put
    // a dependent global load in front of every cp.async, which pins the kernel on
    // long_scoreboard. It was hoisted to shared above. Always commit a group, even
    // past the last tile, so wait_group(1) names the tile about to be read.
    auto stage = [&](int t) {
        if (t < ntiles) {
            __nv_bfloat16* dst_b = k_s + (t & 1) * (TILE * KS);
            const int n = min(TILE, hi - lo - t * TILE);
            for (int i = tid; i < TILE * (D / 8); i += NTHREADS) {
                const int r = i / (D / 8), c = (i % (D / 8)) * 8;
                const int row = (r < n) ? sel_s[t * TILE + r] : -1;
                void* dst = dst_b + r * KS + c;
                if (row >= 0) {
                    const __nv_bfloat16* src =
                        reinterpret_cast<const __nv_bfloat16*>(kv) + (size_t) row * D + c;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                 :: "r"((uint32_t) __cvta_generic_to_shared(dst)), "l"(src));
                } else {
                    *reinterpret_cast<int4*>(dst) = int4{0, 0, 0, 0};
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    // fp8 path: 16 cache bytes per thread per slot, widened on the way to shared.
    uint4 pf[NPF];
    auto fetch = [&](int t) {
#pragma unroll
        for (int u = 0; u < NPF; ++u) {
            const int i = tid + u * NTHREADS;
            if (i >= TILE * (D / EPL)) break;
            const int r = i / (D / EPL), c = (i % (D / EPL)) * EPL;
            const int row = (t < ntiles && r < min(TILE, hi - lo - t * TILE))
                          ? sel_s[t * TILE + r] : -1;
            pf[u] = (row >= 0) ? *reinterpret_cast<const uint4*>(kv + (size_t) row * D + c)
                               : uint4{0, 0, 0, 0};
        }
    };
    auto widen = [&]() {
#pragma unroll
        for (int u = 0; u < NPF; ++u) {
            const int i = tid + u * NTHREADS;
            if (i >= TILE * (D / EPL)) break;
            const int r = i / (D / EPL), c = (i % (D / EPL)) * EPL;
            const uint32_t* w = &pf[u].x;
            uint32_t out[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                out[2 * e] = e4m3x2_bf16(w[e]);
                out[2 * e + 1] = e4m3x2_bf16(w[e] >> 16);
            }
            __nv_bfloat16* dst = k_s + r * KS + c;
            *reinterpret_cast<int4*>(dst) =
                make_int4(out[0], out[1], out[2], out[3]);
            *reinterpret_cast<int4*>(dst + 8) =
                make_int4(out[4], out[5], out[6], out[7]);
        }
    };

    if constexpr (KV8) fetch(0); else stage(0);
    for (int t = 0; t < ntiles; ++t) {
        const int base = lo + t * TILE;
        const int n = min(TILE, hi - base);
        const __nv_bfloat16* k_cur = k_s + (KV8 ? 0 : (t & 1) * (TILE * KS));
        if constexpr (KV8) {
            widen();
            __syncthreads();
            fetch(t + 1);
        } else {
            stage(t + 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();
        }

        // S[16 heads][TILE keys] = Q @ K^T
        {
            for (int nb = nb_w; nb < NBLK; nb += NBLK_E) {
                FragC c{};
#pragma unroll
                for (int f = 0; f < 4; ++f) c[f] = 0.f;
#pragma unroll
                for (int ki = 0; ki < KSTEPS; ++ki) {
                    const int k0 = (ksl * KSTEPS + ki) * 16;
                    FragB fb;
                    const __nv_bfloat16* krow = k_cur + (nb * 8 + (lane >> 2)) * KS
                                              + k0 + (lane & 3) * 2;
                    reinterpret_cast<uint32_t*>(&fb)[0] =
                        *reinterpret_cast<const uint32_t*>(krow);
                    reinterpret_cast<uint32_t*>(&fb)[1] =
                        *reinterpret_cast<const uint32_t*>(krow + 8);
                    mma_m16n8k16_bf16(q_f[ki], fb, c);
                }
                const int cr = lane >> 2, cc = (lane & 3) * 2;
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    float* sp = s_s + ksl * (MMA_M * TILE) + nb * 8 + cc + f;
                    sp[(cr + 0) * TILE] = c[f];
                    sp[(cr + 8) * TILE] = c[2 + f];
                }
            }
        }
        __syncthreads();

        // Tile softmax: one thread per (head, key), reduced across the TILE
        // lanes that share a head.
        for (int i = tid; i < MMA_M * TILE; i += NTHREADS) {
            const int h = i / TILE, j = i % TILE;
            float raw = s_s[h * TILE + j];
#pragma unroll
            for (int u = 1; u < KSL; ++u) raw += s_s[u * (MMA_M * TILE) + h * TILE + j];
            float sc = (j < n && sel_s[t * TILE + j] >= 0) ? raw * scale : -INFINITY;

            float mt = sc;
#pragma unroll
            for (int off = TILE >> 1; off; off >>= 1)
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, off));
            const float m_old = m_s[h];
            const float m_new = fmaxf(m_old, mt);
            const bool empty = !isfinite(m_new);
            const float corr = empty ? 1.f : __expf(m_old - m_new);
            const float p = empty ? 0.f : __expf(sc - m_new);
            p_s[h * PS + j] = __float2bfloat16(p);

            float ps = p;
#pragma unroll
            for (int off = TILE >> 1; off; off >>= 1)
                ps += __shfl_xor_sync(0xffffffff, ps, off);
            __syncwarp();
            if (j == 0) { m_s[h] = m_new; l_s[h] = l_s[h] * corr + ps; c_s[h] = corr; }
        }
        __syncthreads();

        // O = O * corr + P @ V, with V transposed on the way out of shared.
#pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            const int hb = nn * 8 + (lane & 3) * 2;
            const float c0 = c_s[hb], c1 = c_s[hb + 1];
#pragma unroll
            for (int mt = 0; mt < MTPW; ++mt) {
                acc[mt][nn][0] *= c0; acc[mt][nn][1] *= c1;
                acc[mt][nn][2] *= c0; acc[mt][nn][3] *= c1;
            }
        }
        for (int kb = 0; kb < TILE; kb += 16) {
            FragB fb[NT];
#pragma unroll
            for (int nn = 0; nn < NT; ++nn) {
                const __nv_bfloat16* pr = p_s + (nn * 8 + (lane >> 2)) * PS
                                        + kb + (lane & 3) * 2;
                reinterpret_cast<uint32_t*>(&fb[nn])[0] =
                    *reinterpret_cast<const uint32_t*>(pr);
                reinterpret_cast<uint32_t*>(&fb[nn])[1] =
                    *reinterpret_cast<const uint32_t*>(pr + 8);
            }
#pragma unroll
            for (int mt = 0; mt < MTPW; ++mt) {
                const int mbase = (warp + mt * NWARPS) * 16;
                FragA fa;
                ldsm4_trans(fa, k_cur + (kb + 8 * (lane >> 4) + (lane & 7)) * KS
                                    + mbase + 8 * ((lane >> 3) & 1));
#pragma unroll
                for (int nn = 0; nn < NT; ++nn) mma_m16n8k16_bf16(fa, fb[nn], acc[mt][nn]);
            }
        }
        __syncthreads();
    }

    if (tid < MMA_M && tid < hpb && h0 + tid < H) {
        const size_t o = ((size_t) b * H + h0 + tid) * splits + s;
        part_m[o] = m_s[tid]; part_l[o] = l_s[tid];
    }

    // The accumulator layout puts consecutive lanes on different heads at the
    // same dim, so writing it straight out is 32 scattered dwords per lane. Bounce
    // it through the (now dead) K tile instead and write coalesced rows. The 260
    // stride keeps the staging conflict-free: at 256 every head lands on one bank.
    constexpr int OSH = 256, OSS = OSH + 4;
    float* osh = reinterpret_cast<float*>(k_s);
    // The merge reads these back, but this kernel pushes many times their volume
    // of cache through L2 first and evicts them. Ask L2 to keep them.
    uint64_t keep;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(keep));
#pragma unroll
    for (int pass = 0; pass < DV / OSH; ++pass) {
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MTPW; ++mt) {
            const int mi = warp + mt * NWARPS;
            if (mi / (OSH / 16) != pass) continue;
            const int mbase = mi * 16 - pass * OSH;
#pragma unroll
            for (int nn = 0; nn < NT; ++nn) {
                const int hb = nn * 8 + (lane & 3) * 2, d0 = mbase + (lane >> 2);
#pragma unroll
                for (int f = 0; f < 4; ++f)
                    osh[(hb + (f & 1)) * OSS + d0 + 8 * (f >> 1)] = acc[mt][nn][f];
            }
        }
        __syncthreads();
        // Eight dims per thread in one 16-byte store, instead of one bf16 each.
        for (int i = tid; i < MMA_M * (OSH / 8); i += NTHREADS) {
            const int hl = i / (OSH / 8), d = (i % (OSH / 8)) * 8;
            if (hl >= hpb || h0 + hl >= H) continue;
            const float* src = osh + hl * OSS + d;
            __nv_bfloat16* dst = part_o
                + (((size_t) b * H + h0 + hl) * splits + s) * DV + pass * OSH + d;
            asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1,%2,%3,%4}, %5;\n"
                         :: "l"(dst), "r"(f32x2_bf16(src[0], src[1])),
                            "r"(f32x2_bf16(src[2], src[3])),
                            "r"(f32x2_bf16(src[4], src[5])),
                            "r"(f32x2_bf16(src[6], src[7])), "l"(keep)
                         : "memory");
        }
    }
}

// Merge the per-split partials. Standard log-sum-exp combine: rescale each
// split's output by exp(m_s - m*) and divide by the summed weight.
//
// The merge is small -- a megabyte at batch 1 -- but it was costing as much as
// the attention itself, for three reasons: every thread rescanned the whole
// per-split max/sum before doing any output work, exp(m_s - m*) was recomputed
// once per output dim instead of once per split, and 64-thread blocks left the
// machine mostly idle. Now the weights are computed once into shared, and the
// split axis is spread over SG thread groups so there are SG independent loads
// in flight per output element.
// DIMS output dims per block, VEC of them per thread, and SG groups splitting
// the split axis so there are SG independent loads in flight per dim.
template <int DV, int DIMS, int VEC, int SG>
__global__ __launch_bounds__(DIMS / VEC* SG) void mla_decode_reduce_kernel(
    const __nv_bfloat16* __restrict__ part_o,   // (B, H, splits, DV)
    const float* __restrict__ part_m,           // (B, H, splits)
    const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out,
    int H, int splits, float out_scale)
{
    constexpr int TPG = DIMS / VEC;          // threads per split group
    static_assert(TPG <= 32, "a split group must fit in one warp");
    constexpr int NT = TPG * SG;
    constexpr int NW = NT / 32;
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int d0 = blockIdx.z * DIMS;
    const int tid = threadIdx.x;
    const int g = tid / TPG, t = tid % TPG;

    extern __shared__ __align__(16) char red_raw[];
    float* w = reinterpret_cast<float*>(red_raw);       // splits
    float* red = w + splits;                            // NW x DIMS
    __shared__ float xchg[NW], xchl[NW];
    __shared__ float m_all, inv;

    // (max, weight) is an associative pair under the log-sum-exp combine, so one
    // reduction produces both. The old code ran two -- and read the per-split max
    // from global twice, then finished each reduction on a single thread.
    constexpr int RMAX = 8;
    const size_t mbase = ((size_t) b * H + h) * splits;
    float m_l = -INFINITY, l_l = 0.f;
#pragma unroll 1
    for (int r = 0; r < RMAX; ++r) {
        const int sp = tid + r * NT;
        if (sp >= splits) break;
        const float mm = part_m[mbase + sp], ll = part_l[mbase + sp];
        const float M = fmaxf(m_l, mm);
        l_l = isfinite(M) ? l_l * __expf(m_l - M) + ll * __expf(mm - M) : 0.f;
        m_l = M;
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffff, m_l, off);
        const float l2 = __shfl_xor_sync(0xffffffff, l_l, off);
        const float M = fmaxf(m_l, m2);
        l_l = isfinite(M) ? l_l * __expf(m_l - M) + l2 * __expf(m2 - M) : 0.f;
        m_l = M;
    }
    if ((tid & 31) == 0) { xchg[tid >> 5] = m_l; xchl[tid >> 5] = l_l; }
    __syncthreads();
    if (tid < 32) {
        float mv = (tid < NW) ? xchg[tid] : -INFINITY;
        float lv = (tid < NW) ? xchl[tid] : 0.f;
#pragma unroll
        for (int off = 16; off; off >>= 1) {
            const float m2 = __shfl_xor_sync(0xffffffff, mv, off);
            const float l2 = __shfl_xor_sync(0xffffffff, lv, off);
            const float M = fmaxf(mv, m2);
            lv = isfinite(M) ? lv * __expf(mv - M) + l2 * __expf(m2 - M) : 0.f;
            mv = M;
        }
        if (tid == 0) { m_all = mv; inv = lv > 0.f ? 1.f / lv : 0.f; }
    }
    __syncthreads();

    if (!isfinite(m_all)) {                       // no keys landed here
        for (int d = tid; d < DIMS; d += NT)
            out[((size_t) b * H + h) * DV + d0 + d] = __float2bfloat16(0.f);
        return;
    }
    for (int sp = tid; sp < splits; sp += NT)
        w[sp] = __expf(part_m[mbase + sp] - m_all);
    __syncthreads();

    float acc[VEC];
#pragma unroll
    for (int e = 0; e < VEC; ++e) acc[e] = 0.f;
    for (int sp = g; sp < splits; sp += SG) {
        const uint4 r = *reinterpret_cast<const uint4*>(
            part_o + (((size_t) b * H + h) * splits + sp) * DV + d0 + t * VEC);
        const float ws = w[sp];
        const uint32_t* u = reinterpret_cast<const uint32_t*>(&r);
#pragma unroll
        for (int e = 0; e < VEC; e += 2) {
            float lo, hi;
            bf16x2_f32(u[e >> 1], lo, hi);
            acc[e] = fmaf(lo, ws, acc[e]);
            acc[e + 1] = fmaf(hi, ws, acc[e + 1]);
        }
    }
    // Lanes of one warp span 32/TPG groups; fold those with shuffles, then the
    // NW warps through shared.
#pragma unroll
    for (int off = TPG; off < 32; off <<= 1)
#pragma unroll
        for (int e = 0; e < VEC; ++e) acc[e] += __shfl_xor_sync(0xffffffff, acc[e], off);
    if ((tid & 31) < TPG) {
#pragma unroll
        for (int e = 0; e < VEC; ++e) red[(tid >> 5) * DIMS + t * VEC + e] = acc[e];
    }
    __syncthreads();
    for (int d = tid; d < DIMS; d += NT) {
        float v = red[d];
#pragma unroll
        for (int i = 1; i < NW; ++i) v += red[i * DIMS + d];
        out[((size_t) b * H + h) * DV + d0 + d] =
            __float2bfloat16(v * inv * out_scale);
    }
}

}  // namespace vllm_exl3

namespace vllm_exl3 {

// Which (chunk, block shape) wins is shape-dependent and not guessable:
// splitting keys buys blocks but grows the partial buffer the reduce must read,
// while splitting heads buys blocks for free in the reduce but re-reads K per
// group. At batch 1 the answer is a 4-head block; by batch 4 it is a 16-head
// one. So time the candidates once per shape and remember the winner.
namespace {

std::map<uint64_t, std::pair<int, int>>& mla_tune_cache() {
    static std::map<uint64_t, std::pair<int, int>> c;
    return c;
}

uint64_t mla_tune_key(int B, int H, int topk) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t v : {(uint64_t) B, (uint64_t) H, (uint64_t) topk}) {
        h ^= v; h *= 1099511628211ull;
    }
    return h;
}

bool mla_tuning_enabled() {
    static const bool on = [] {
        const char* e = getenv("VLLM_EXL3_MLA_TUNE");
        return !(e && *e == '0');
    }();
    return on;
}

}  // namespace

at::Tensor mla_decode(const at::Tensor& q, const at::Tensor& kv,
                      const at::Tensor& sel, const at::Tensor& seqlens,
                      double scale, int64_t v_head_dim, int64_t split_chunk,
                      int64_t heads_per_block, double kv_scale)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    TORCH_CHECK(q.dim() == 3, "mla_decode: q must be (batch, heads, head_dim)");
    TORCH_CHECK(kv.dim() == 2, "mla_decode: kv must be (rows, head_dim)");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "mla_decode: q must be bfloat16");
    const bool kv8 = kv.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16 || kv8,
                "mla_decode: kv must be bfloat16 or float8_e4m3fn");
    // An fp8 cache stores k / kv_scale. Both matmuls are linear in it, so the
    // scale never has to touch an element: it folds into the softmax scale for
    // Q@K^T and into the merged output for P@V.
    const double sscale = kv8 ? scale * kv_scale : scale;
    const double oscale = kv8 ? kv_scale : 1.0;
    TORCH_CHECK(sel.scalar_type() == at::kInt, "mla_decode: sel must be int32");

    const int B = (int) q.size(0), H = (int) q.size(1), D = (int) q.size(2);
    const int topk = (int) sel.size(1);
    const int DV = (int) v_head_dim;
    // 576 is the DeepSeek latent (512 nope + 64 rope); 512 is GLM-5.3-Flash,
    // which has qk_rope_head_dim = 0.
    TORCH_CHECK((D == 576 || D == 512) && DV == 512,
                "mla_decode: built for head_dim 576 or 512 with v_head_dim 512, got ",
                D, "/", DV);

    int chunk = (int) split_chunk;
    int wide = (int) heads_per_block;  // 1: 8 warps x 16 keys, 2: 16 warps x 32
    if (chunk <= 0 || wide <= 0) {
        const uint64_t key = mla_tune_key(B, H, topk);
        auto& cache = mla_tune_cache();
        auto it = cache.find(key);
        if (it == cache.end()) {
            std::pair<int, int> best{32, 1};
            if (mla_tuning_enabled() &&
                at::cuda::currentStreamCaptureStatusMayInitCtx() ==
                    at::cuda::CaptureStatus::None) {
                float best_ms = 1e30f;
                cudaEvent_t beg, end;
                cudaEventCreate(&beg); cudaEventCreate(&end);
                for (int c : {16, 32, 48, 64, 96, 128, 192, 256, 384}) {
                    for (int wd : {1, 2, 3, 4}) {
                        mla_decode(q, kv, sel, seqlens, scale, v_head_dim, c, wd, kv_scale);
                        cudaEventRecord(beg);
                        for (int r = 0; r < 3; ++r)
                            mla_decode(q, kv, sel, seqlens, scale, v_head_dim, c, wd, kv_scale);
                        cudaEventRecord(end);
                        cudaEventSynchronize(end);
                        float ms = 0.f; cudaEventElapsedTime(&ms, beg, end);
                        if (ms < best_ms) { best_ms = ms; best = {c, wd}; }
                    }
                }
                cudaEventDestroy(beg); cudaEventDestroy(end);
            }
            it = cache.emplace(key, best).first;
        }
        if (chunk <= 0) chunk = it->second.first;
        if (wide <= 0) wide = it->second.second;
    }
    // Splitting keys needs an LSE merge, so more splits means a bigger partial
    // buffer and a costlier reduce. Splitting *heads* costs nothing -- their
    // outputs are disjoint -- at the price of re-reading K once per head group.
    // The mma tile is 16 heads wide, so that is the only useful group size.
    // Codes 3 and 4 halve the head group. That wastes half of every mma, but at
    // batch 1 the kernel is nowhere near bandwidth-bound and the extra blocks
    // cost nothing but a second read of the cache -- the partial write, which is
    // the expensive part, is per head and so unchanged.
    const int hpb = std::min(H, (wide >= 3 && H >= 16) ? 8 : 16);
    const int hgroups = (H + hpb - 1) / hpb;

    const int splits = (topk + chunk - 1) / chunk;

    auto dev = q.device();
    // Allocated per call rather than kept in a global workspace. A workspace
    // that grows reallocates, and a reallocation between two CUDA graph captures
    // leaves the earlier graph pointing at freed memory, which faults during
    // capture of the later one. Under capture these land in the graph's private
    // pool and stay valid; outside it the caching allocator makes them cheap.
    const auto fopt = at::TensorOptions().dtype(at::kFloat).device(dev);
    auto po = at::empty({(int64_t) B * splits * H * DV},
                        at::TensorOptions().dtype(at::kBFloat16).device(dev));
    auto pm = at::empty({(int64_t) B * splits * H}, fopt);
    auto pl = at::empty({(int64_t) B * splits * H}, fopt);
    auto out = at::empty({B, H, DV}, q.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    // Two block shapes. The wide one halves the block count for the same thread
    // count, which halves both the Q re-read and the partial write: those are
    // per block, not per key, and at low batch they outweigh the cache traffic.
#define MLA_LAUNCH(D_, NW_, TILE_, KT_)                                             \
    do {                                                                       \
        constexpr int KSL_ = (NW_) / ((TILE_) / 8);                            \
        constexpr int NBUF_ = std::is_same<KT_, __nv_bfloat16>::value ? 2 : 1; \
        const size_t smem = NBUF_ * (TILE_) * ((D_) + 8)                       \
                              * sizeof(__nv_bfloat16)                          \
                          + 16 * ((TILE_) + 8) * sizeof(__nv_bfloat16)         \
                          + (size_t) KSL_ * 16 * (TILE_) * sizeof(float)       \
                          + 3 * 16 * sizeof(float)                             \
                          + (size_t) chunk * sizeof(int);                      \
        auto kern = mla_decode_partial_kernel<D_, 512, NW_, TILE_, KT_>;       \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,\
                             101376);                                          \
        kern<<<grid, (NW_) * 32, smem, stream>>>(                              \
            (const __nv_bfloat16*) q.data_ptr(),                               \
            (const KT_*) kv.data_ptr(), sel.data_ptr<int>(),                   \
            seqlens.numel() ? seqlens.data_ptr<int>() : nullptr,               \
            (__nv_bfloat16*) po.data_ptr(), pm.data_ptr<float>(),              \
            pl.data_ptr<float>(), H, topk, splits, chunk, (float) sscale,      \
            hpb);                                                              \
    } while (0)

    dim3 grid(splits, hgroups, B);
#define MLA_PICK(D_, KT_)                                                      \
    do {                                                                       \
        if (wide == 2 || wide == 4) { MLA_LAUNCH(D_, 16, 32, KT_); }           \
        else { MLA_LAUNCH(D_, 8, 16, KT_); }                                   \
    } while (0)

    if (D == 576) {
        if (kv8) { MLA_PICK(576, __nv_fp8_e4m3); } else { MLA_PICK(576, __nv_bfloat16); }
    } else {
        if (kv8) { MLA_PICK(512, __nv_fp8_e4m3); } else { MLA_PICK(512, __nv_bfloat16); }
    }
#undef MLA_PICK
#undef MLA_LAUNCH

    // The merge splits its 256 threads between output dims and the split axis.
    // Spreading over more splits than exist just idles the extra groups, so the
    // shape follows the split count: at 11 splits, 32 groups left two thirds of
    // the block doing nothing and still paid for the cross-group reduce.
#define MLA_REDUCE(SG_)                                                        \
    do {                                                                       \
        constexpr int VEC_ = 8;                                                \
        constexpr int DIMS_ = VEC_ * 256 / (SG_) < 256                         \
                            ? VEC_ * 256 / (SG_) : 256;                        \
        constexpr int NT_ = DIMS_ / VEC_ * (SG_);                              \
        const size_t rsm = ((size_t) splits + NT_ / 32 * DIMS_) * sizeof(float);\
        mla_decode_reduce_kernel<512, DIMS_, VEC_, SG_>                        \
            <<<dim3(H, B, 512 / DIMS_), NT_, rsm, stream>>>(                   \
                (const __nv_bfloat16*) po.data_ptr(), pm.data_ptr<float>(),    \
                pl.data_ptr<float>(), (__nv_bfloat16*) out.data_ptr(), H,      \
                splits, (float) oscale);                                       \
    } while (0)

    if (splits >= 24) { MLA_REDUCE(32); }
    else if (splits >= 12) { MLA_REDUCE(16); }
    else if (splits >= 6) { MLA_REDUCE(8); }
    else { MLA_REDUCE(4); }
#undef MLA_REDUCE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace vllm_exl3
