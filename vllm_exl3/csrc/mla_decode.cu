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
template <int D, int DV, int NWARPS, int HPW, int TILE>
__global__ __launch_bounds__(NWARPS * 32) void mla_decode_partial_kernel(
    const __nv_bfloat16* __restrict__ q,     // (B, H, D)
    const __nv_bfloat16* __restrict__ kv,    // (rows, D) latent cache
    const int* __restrict__ sel,             // (B, topk) row ids, -1 = empty
    const int* __restrict__ seqlens,         // (B) valid entries of sel
    float* __restrict__ part_o,              // (B, splits, H, DV)
    float* __restrict__ part_m,              // (B, splits, H)
    float* __restrict__ part_l,              // (B, splits, H)
    int H, int topk, int splits, int chunk, float scale)
{
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* k_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // 2 x TILE x D
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int s = blockIdx.x;
    const int b = blockIdx.y;
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int VPT = DV / 32;             // v dims each lane owns

    const int lo = s * chunk;
    const int valid = seqlens ? seqlens[b] : topk;
    const int hi = min(lo + chunk, valid);

    constexpr int QPT = D / 32;              // q dims each lane owns
    float acc[HPW][VPT], q_r[HPW][QPT];
    float m_run[HPW], l_run[HPW];
#pragma unroll
    for (int i = 0; i < HPW; ++i) {
        m_run[i] = -INFINITY;
        l_run[i] = 0.f;
#pragma unroll
        for (int d = 0; d < VPT; ++d) acc[i][d] = 0.f;
        // q is loop-invariant: hold this lane's slice in registers rather than
        // re-reading it from global for every key (v2 did, and paid ~2x for it).
        const int h = warp * HPW + i;
        const __nv_bfloat16* qh = q + ((size_t) b * H + (h < H ? h : 0)) * D;
#pragma unroll
        for (int d = 0; d < QPT; ++d) q_r[i][d] = __bfloat162float(qh[d * 32 + lane]);
    }

    // Stage one tile of latent rows into shared; rows that do not exist are
    // left alone and masked out by `n` below.
    auto stage = [&](int buf, int base) {
        const int n = max(0, min(TILE, hi - base));
        __nv_bfloat16* dst = k_s + buf * TILE * D;
        for (int i = tid; i < n * (D / 8); i += NTHREADS) {
            const int r = i / (D / 8), c = (i % (D / 8)) * 8;
            const int row = sel[(size_t) b * topk + base + r];
            void* d_ = dst + r * D + c;
            if (row >= 0) {
                const __nv_bfloat16* src = kv + (size_t) row * D + c;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"((uint32_t) __cvta_generic_to_shared(d_)), "l"(src));
            } else {
                *reinterpret_cast<int4*>(d_) = int4{0, 0, 0, 0};
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    stage(0, lo);
    int buf = 0;
    for (int base = lo; base < hi; base += TILE) {
        const int n = min(TILE, hi - base);
        // Always commit, even past the end (stage clamps to zero rows). The
        // group count then stays predictable and wait_group(1) really does wait
        // for the tile we are about to read.
        stage(buf ^ 1, base + TILE);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(1));
        __syncthreads();

        const __nv_bfloat16* tile = k_s + buf * TILE * D;

        // Score the whole tile before consuming it. Done key-by-key, each
        // iteration is a dependent chain -- 18 FMAs, a five-deep shuffle
        // reduction, two exps -- and the next key cannot start. Batching the
        // partial dots first gives TILE*HPW independent chains to interleave.
        float part[TILE][HPW];
#pragma unroll
        for (int j = 0; j < TILE; ++j) {
            const __nv_bfloat16* k = tile + j * D;
#pragma unroll
            for (int i = 0; i < HPW; ++i) {
                float dot = 0.f;
#pragma unroll
                for (int d = 0; d < QPT; ++d)
                    dot += q_r[i][d] * __bfloat162float(k[d * 32 + lane]);
                part[j][i] = dot;
            }
        }
#pragma unroll
        for (int j = 0; j < TILE; ++j)
#pragma unroll
            for (int i = 0; i < HPW; ++i) part[j][i] = warp_sum(part[j][i]);

        for (int j = 0; j < n; ++j) {
            if (sel[(size_t) b * topk + base + j] < 0) continue;   // empty slot
            const __nv_bfloat16* k = tile + j * D;
#pragma unroll
            for (int i = 0; i < HPW; ++i) {
                if (warp * HPW + i >= H) break;
                const float sc = part[j][i] * scale;
                const float m_new = fmaxf(m_run[i], sc);
                const float corr = __expf(m_run[i] - m_new);
                const float p = __expf(sc - m_new);
                l_run[i] = l_run[i] * corr + p;
                m_run[i] = m_new;
#pragma unroll
                for (int d = 0; d < VPT; ++d)
                    acc[i][d] = acc[i][d] * corr + p * __bfloat162float(k[d * 32 + lane]);
            }
        }
        __syncthreads();
        buf ^= 1;
    }

#pragma unroll
    for (int i = 0; i < HPW; ++i) {
        const int h = warp * HPW + i;
        if (h >= H) break;
        const size_t base = (((size_t) b * splits + s) * H + h);
        if (lane == 0) { part_m[base] = m_run[i]; part_l[base] = l_run[i]; }
#pragma unroll
        for (int d = 0; d < VPT; ++d) part_o[base * DV + d * 32 + lane] = acc[i][d];
    }
}

// Merge the per-split partials. Standard log-sum-exp combine: rescale each
// split's output by exp(m_s - m*) and divide by the summed weight.
template <int DV>
__global__ void mla_decode_reduce_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out,
    int H, int splits)
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;

    float m_all = -INFINITY;
    for (int s = 0; s < splits; ++s)
        m_all = fmaxf(m_all, part_m[((size_t) b * splits + s) * H + h]);
    if (!isfinite(m_all)) {                       // no keys landed here
        for (int d = lane; d < DV; d += blockDim.x)
            out[((size_t) b * H + h) * DV + d] = __float2bfloat16(0.f);
        return;
    }

    float l_all = 0.f;
    for (int s = 0; s < splits; ++s) {
        const size_t base = ((size_t) b * splits + s) * H + h;
        l_all += part_l[base] * __expf(part_m[base] - m_all);
    }
    const float inv = l_all > 0.f ? 1.f / l_all : 0.f;

    for (int d = lane; d < DV; d += blockDim.x) {
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

at::Tensor mla_decode(const at::Tensor& q, const at::Tensor& kv,
                      const at::Tensor& sel, const at::Tensor& seqlens,
                      double scale, int64_t v_head_dim, int64_t split_chunk)
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

    constexpr int NWARPS = 8;
    constexpr int TILE = 8;
    const int hpw = (H + NWARPS - 1) / NWARPS;
    TORCH_CHECK(hpw <= 4, "mla_decode: at most ", NWARPS * 4, " heads");

    int chunk = (int) split_chunk;
    if (chunk <= 0) chunk = 64;
    const int splits = (topk + chunk - 1) / chunk;

    auto dev = q.device();
    auto& po = grow(g_po, (int64_t) B * splits * H * DV, at::kFloat, dev);
    auto& pm = grow(g_pm, (int64_t) B * splits * H, at::kFloat, dev);
    auto& pl = grow(g_pl, (int64_t) B * splits * H, at::kFloat, dev);
    auto out = at::empty({B, H, DV}, q.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    const size_t smem = 2 * TILE * 576 * sizeof(__nv_bfloat16);

    dim3 grid(splits, B);
#define MLA_LAUNCH(HPW_)                                                       \
    mla_decode_partial_kernel<576, 512, NWARPS, HPW_, TILE>                    \
        <<<grid, NWARPS * 32, smem, stream>>>(                                 \
            (const __nv_bfloat16*) q.data_ptr(),                               \
            (const __nv_bfloat16*) kv.data_ptr(), sel.data_ptr<int>(),         \
            seqlens.numel() ? seqlens.data_ptr<int>() : nullptr,               \
            po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),  \
            H, topk, splits, chunk, (float) scale)

    if (hpw <= 1) { MLA_LAUNCH(1); }
    else if (hpw == 2) { MLA_LAUNCH(2); }
    else if (hpw == 3) { MLA_LAUNCH(3); }
    else { MLA_LAUNCH(4); }
#undef MLA_LAUNCH

    mla_decode_reduce_kernel<512><<<dim3(H, B), 256, 0, stream>>>(
        po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
        (__nv_bfloat16*) out.data_ptr(), H, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace vllm_exl3
