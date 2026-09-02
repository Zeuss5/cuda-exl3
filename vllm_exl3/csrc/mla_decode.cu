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
template <int D, int DV, int NWARPS, int TILE>
__global__ __launch_bounds__(NWARPS * 32) void mla_decode_partial_kernel(
    const __nv_bfloat16* __restrict__ q,     // (B, H, D)
    const __nv_bfloat16* __restrict__ kv,    // (rows, D) latent cache
    const int* __restrict__ sel,             // (B, topk) row ids, -1 = empty
    const int* __restrict__ seqlens,         // (B) valid entries of sel
    float* __restrict__ part_o,              // (B, splits, H, DV)
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
    static_assert(NTHREADS >= MMA_M * TILE, "softmax pass needs one thread per score");
    static_assert(TILE % 16 == 0, "PV mma consumes 16 keys at a time");

    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);      // 16 x KS
    __nv_bfloat16* k_s = q_s + MMA_M * KS;                                // TILE x KS
    __nv_bfloat16* p_s = k_s + TILE * KS;                                 // 16 x PS
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

    // Q for this block's heads, staged once.
    for (int i = tid; i < MMA_M * D; i += NTHREADS) {
        const int r = i / D, c = i % D;
        const int h = h0 + r;
        q_s[r * KS + c] = (r < hpb && h < H) ? q[((size_t) b * H + h) * D + c]
                                             : __float2bfloat16(0.f);
    }
    if (tid < MMA_M) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }

    // acc[t][n] holds C[dim][head] for m-tile (warp + t*NWARPS), n-tile n.
    FragC acc[MTPW][NT];
#pragma unroll
    for (int t = 0; t < MTPW; ++t)
#pragma unroll
        for (int nn = 0; nn < NT; ++nn)
#pragma unroll
            for (int f = 0; f < 4; ++f) acc[t][nn][f] = 0.f;

    // The selection list is a gather: reading it inside the staging loop puts a
    // dependent global load in front of every cp.async, which pins the kernel on
    // long_scoreboard. It is at most `chunk` ints, so hoist it.
    for (int i = tid; i < hi - lo; i += NTHREADS) sel_s[i] = sel[(size_t) b * topk + lo + i];
    __syncthreads();

    const int ntiles = (hi - lo + TILE - 1) / TILE;

    for (int t = 0; t < ntiles; ++t) {
        const int base = lo + t * TILE;
        const int n = min(TILE, hi - base);

        for (int i = tid; i < TILE * (D / 8); i += NTHREADS) {
            const int r = i / (D / 8), c = (i % (D / 8)) * 8;
            const int row = (r < n) ? sel_s[t * TILE + r] : -1;
            void* dst = k_s + r * KS + c;
            if (row >= 0) {
                const __nv_bfloat16* src = kv + (size_t) row * D + c;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"((uint32_t) __cvta_generic_to_shared(dst)), "l"(src));
            } else {
                *reinterpret_cast<int4*>(dst) = int4{0, 0, 0, 0};
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        // S[16 heads][TILE keys] = Q @ K^T
        {
            const int nb_w = warp % NBLK_E, ksl = warp / NBLK_E;
            for (int nb = nb_w; nb < NBLK; nb += NBLK_E) {
                FragC c{};
#pragma unroll
                for (int f = 0; f < 4; ++f) c[f] = 0.f;
                for (int ki = 0; ki < KSTEPS; ++ki) {
                    const int k0 = (ksl * KSTEPS + ki) * 16;
                    FragA fa;
                    ldsm4(fa, q_s + ((lane & 7) + 8 * ((lane >> 3) & 1)) * KS
                                  + k0 + (lane >> 4) * 8);
                    FragB fb;
                    const __nv_bfloat16* krow = k_s + (nb * 8 + (lane >> 2)) * KS
                                              + k0 + (lane & 3) * 2;
                    reinterpret_cast<uint32_t*>(&fb)[0] =
                        *reinterpret_cast<const uint32_t*>(krow);
                    reinterpret_cast<uint32_t*>(&fb)[1] =
                        *reinterpret_cast<const uint32_t*>(krow + 8);
                    mma_m16n8k16_bf16(fa, fb, c);
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
                ldsm4_trans(fa, k_s + (kb + 8 * (lane >> 4) + (lane & 7)) * KS
                                    + mbase + 8 * ((lane >> 3) & 1));
#pragma unroll
                for (int nn = 0; nn < NT; ++nn) mma_m16n8k16_bf16(fa, fb[nn], acc[mt][nn]);
            }
        }
        __syncthreads();
    }

    if (tid < MMA_M && tid < hpb && h0 + tid < H) {
        const size_t o = ((size_t) b * splits + s) * H + h0 + tid;
        part_m[o] = m_s[tid]; part_l[o] = l_s[tid];
    }
#pragma unroll
    for (int mt = 0; mt < MTPW; ++mt) {
        const int mbase = (warp + mt * NWARPS) * 16;
#pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            const int hb = nn * 8 + (lane & 3) * 2;
            const int d0 = mbase + (lane >> 2);
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int hl = hb + (f & 1), dd = d0 + 8 * (f >> 1);
                if (hl >= hpb || h0 + hl >= H) continue;
                part_o[(((size_t) b * splits + s) * H + h0 + hl) * DV + dd] = acc[mt][nn][f];
            }
        }
    }
}

// Merge the per-split partials. Standard log-sum-exp combine: rescale each
// split's output by exp(m_s - m*) and divide by the summed weight.
// One block per (batch, head, slice of the output dims). Sixteen blocks -- one
// per head -- left 188 SMs almost idle and cost 6.8 us of a 28.7 us call, which
// is a quarter of the kernel spent merging a few kilobytes.
template <int DV, int DIMS_PER_BLOCK>
__global__ void mla_decode_reduce_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out,
    int H, int splits)
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int d0 = blockIdx.z * DIMS_PER_BLOCK;
    const int lane = threadIdx.x;

    float m_all = -INFINITY;
    for (int s = 0; s < splits; ++s)
        m_all = fmaxf(m_all, part_m[((size_t) b * splits + s) * H + h]);
    if (!isfinite(m_all)) {                       // no keys landed here
        for (int d = d0 + lane; d < d0 + DIMS_PER_BLOCK; d += blockDim.x)
            out[((size_t) b * H + h) * DV + d] = __float2bfloat16(0.f);
        return;
    }

    float l_all = 0.f;
    for (int s = 0; s < splits; ++s) {
        const size_t base = ((size_t) b * splits + s) * H + h;
        l_all += part_l[base] * __expf(part_m[base] - m_all);
    }
    const float inv = l_all > 0.f ? 1.f / l_all : 0.f;

    for (int d = d0 + lane; d < d0 + DIMS_PER_BLOCK; d += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < splits; ++s) {
            const size_t base = ((size_t) b * splits + s) * H + h;
            v += part_o[base * DV + d] * __expf(part_m[base] - m_all);
        }
        out[((size_t) b * H + h) * DV + d] = __float2bfloat16(v * inv);
    }
}

}  // namespace vllm_exl3

namespace {
using namespace vllm_exl3;

// Workspace for the split partials, grown in place and never freed, so that
// nothing reallocates once CUDA graphs are being captured.
at::Tensor g_po, g_pm, g_pl;

at::Tensor& grow(at::Tensor& t, int64_t n, at::ScalarType dt, const at::Device& dev) {
    if (!t.defined() || t.numel() < n)
        t = at::empty({n}, at::TensorOptions().dtype(dt).device(dev));
    return t;
}
}  // namespace

namespace vllm_exl3 {

// Which (chunk, heads_per_block) wins is shape-dependent and not guessable:
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
                      int64_t heads_per_block)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    TORCH_CHECK(q.dim() == 3, "mla_decode: q must be (batch, heads, head_dim)");
    TORCH_CHECK(kv.dim() == 2, "mla_decode: kv must be (rows, head_dim)");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16,
                "mla_decode: q and kv must be bfloat16");
    TORCH_CHECK(sel.scalar_type() == at::kInt, "mla_decode: sel must be int32");

    const int B = (int) q.size(0), H = (int) q.size(1), D = (int) q.size(2);
    const int topk = (int) sel.size(1);
    const int DV = (int) v_head_dim;
    TORCH_CHECK(D == 576 && DV == 512,
                "mla_decode: built for head_dim 576 / v_head_dim 512, got ", D, "/", DV);

    constexpr int TILE = 16;
    int chunk = (int) split_chunk;
    int hpb_in = (int) heads_per_block;
    if (chunk <= 0 || hpb_in <= 0) {
        const uint64_t key = mla_tune_key(B, H, topk);
        auto& cache = mla_tune_cache();
        auto it = cache.find(key);
        if (it == cache.end()) {
            std::pair<int, int> best{32, std::min(H, 16)};
            if (mla_tuning_enabled() &&
                at::cuda::currentStreamCaptureStatusMayInitCtx() ==
                    at::cuda::CaptureStatus::None) {
                float best_ms = 1e30f;
                cudaEvent_t beg, end;
                cudaEventCreate(&beg); cudaEventCreate(&end);
                for (int c : {16, 32, 64, 128, 256}) {
                    const int hb = std::min(H, 16);
                    {
                        mla_decode(q, kv, sel, seqlens, scale, v_head_dim, c, hb);
                        cudaEventRecord(beg);
                        for (int r = 0; r < 3; ++r)
                            mla_decode(q, kv, sel, seqlens, scale, v_head_dim, c, hb);
                        cudaEventRecord(end);
                        cudaEventSynchronize(end);
                        float ms = 0.f; cudaEventElapsedTime(&ms, beg, end);
                        if (ms < best_ms) { best_ms = ms; best = {c, hb}; }
                    }
                }
                cudaEventDestroy(beg); cudaEventDestroy(end);
            }
            it = cache.emplace(key, best).first;
        }
        if (chunk <= 0) chunk = it->second.first;
        if (hpb_in <= 0) hpb_in = it->second.second;
    }
    // Splitting keys needs an LSE merge, so more splits means a bigger partial
    // buffer and a costlier reduce. Splitting *heads* costs nothing -- their
    // outputs are disjoint -- at the price of re-reading K once per head group.
    // The mma tile is 16 heads wide, so that is the only useful group size.
    const int hpb = std::min(H, 16);
    const int hgroups = (H + hpb - 1) / hpb;
    (void) hpb_in;

    const int splits = (topk + chunk - 1) / chunk;

    auto dev = q.device();
    auto& po = grow(g_po, (int64_t) B * splits * H * DV, at::kFloat, dev);
    auto& pm = grow(g_pm, (int64_t) B * splits * H, at::kFloat, dev);
    auto& pl = grow(g_pl, (int64_t) B * splits * H, at::kFloat, dev);
    auto out = at::empty({B, H, DV}, q.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int NWARPS = 8;
    constexpr int KSL = NWARPS / (TILE / 8);
    const size_t smem = (16 + TILE) * (576 + 8) * sizeof(__nv_bfloat16)
                      + 16 * (TILE + 8) * sizeof(__nv_bfloat16)
                      + (size_t) KSL * 16 * TILE * sizeof(float)
                      + 3 * 16 * sizeof(float)
                      + (size_t) chunk * sizeof(int);

    dim3 grid(splits, hgroups, B);
    auto kern = mla_decode_partial_kernel<576, 512, NWARPS, TILE>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
    kern<<<grid, NWARPS * 32, smem, stream>>>(
        (const __nv_bfloat16*) q.data_ptr(),
        (const __nv_bfloat16*) kv.data_ptr(), sel.data_ptr<int>(),
        seqlens.numel() ? seqlens.data_ptr<int>() : nullptr,
        po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
        H, topk, splits, chunk, (float) scale, hpb);

    constexpr int RED_DIMS = 64;
    mla_decode_reduce_kernel<512, RED_DIMS>
        <<<dim3(H, B, 512 / RED_DIMS), 64, 0, stream>>>(
        po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
        (__nv_bfloat16*) out.data_ptr(), H, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace vllm_exl3
