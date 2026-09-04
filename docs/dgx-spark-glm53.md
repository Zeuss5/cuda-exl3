# Running GLM-5.3-Flash on DGX Sparks

**Status: the kernels are now measured on GB10; the end-to-end serving path is
not.** This plugin is developed on 4x RTX PRO 6000 Blackwell (sm_120), and a
tester brought it up on a DGX Spark and reported the kernel numbers in §1a. I do
not have a Spark, so anything below about serving GLM end to end on one is still
reasoning rather than a measurement, and is marked as such.

**Does GB10 need a separate kernel path? No.** The same source builds for
sm_121 and reaches 90-96% of that machine's bandwidth on the memory-bound paths.
What it does need is per-device *tuning*, which `bench/calibrate.py` produces,
and there is one genuine gap at large M on dense layers -- both in §1a.

The best existing source on this deployment is
[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark),
a production kit for exactly this model on exactly this hardware, with its own
GB10 kernel work. Read it first. This document is about what changes if you put
*this* plugin underneath it, and it borrows that kit's measurements throughout
with attribution.

## 1. The hardware, and why these kernels already fit it

GB10 (sm_121) is consumer-Blackwell-class silicon, and the constraints that
matter are the ones this plugin was written against from the start:

| Constraint | GB10 | This plugin |
|---|---|---|
| `tcgen05` / TMEM | **absent** — warp-level `mma.sync` is the only tensor-core path ([CUTLASS #2947](https://github.com/NVIDIA/cutlass/issues/2947), NVIDIA staff) | uses `mma.m16n8k16`, `cp.async` and `ldmatrix` only. No wgmma, no TMA, no 2-SM MMA |
| Shared memory | 101,376 B/CTA ([forum](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960)) — same as sm_120; SGLang-class 147 KB MoE configs fail | largest block is ~43 KB. The MLA kernel asks the driver for the device's own `MaxSharedMemoryPerBlockOptin` rather than assuming a number |
| Memory bandwidth | **241 GB/s** measured by our tester (218 reported elsewhere) | see §4 -- this is the whole story on a Spark |
| Toolchain | CUDA 13.x system `ptxas`; older bundled ptxas has no sm_121 | builds with CUDA 13; `setup.py` adds `12.1` to the arch list after asking `nvcc --list-gpu-arch` whether it knows `compute_121` |
| Usable memory | ~121 GiB per node | the 4bpw checkpoint is 164 GB, so see §2 |

Numbers in the GB10 column are from the Reederey87 kit's `docs/11-gb10-kernel-program.md`
and the sources it cites.

Nothing here needs a code change. The kernels were kept to the sm_80-era
instruction set deliberately, which is the same intake rule that kit adopted.

## 1a. Measured on GB10 (sm_121)

Reported by a tester on a DGX Spark. Builds clean in 86 s with
`TORCH_CUDA_ARCH_LIST=12.1`; the 29 model-free tests pass.

| path | result |
|---|---|
| **MoE expert kernel**, full 288-expert bank | **3.3x vs bf16** at M=1, **3.7-4.2x** at M=8-1024, 2.8x at M=3584 |
| MoE bandwidth attainment | 81% at M=1, **90-91% at M=8-128** (of 241 GB/s measured) |
| **Sparse-MLA** | **86-91%** of bandwidth at B>=16, **95-96%** at B=64-512 |
| fp8 KV cache | **+1.5-1.9x** (it is +1.3-1.8x on sm_120, so it matters more here, as expected on a sixth of the bandwidth) |
| TP=3, head counts 1-64 | uniform 2.6e-3 relative error, no head-count anomaly |
| **Dense non-MoE layers, M>=128** | **0.6-0.8x of cuBLAS bf16** |

Two things to take from that. The memory-bound paths -- which is what this
plugin is for -- are close to the machine, and **241 GB/s** is the bandwidth to
plan against rather than the 218 quoted elsewhere.

And the last row is the one real gap. On sm_120 the dense EXL3 GEMM beats cuBLAS
bf16 comfortably; on GB10 at M>=128 it loses. The trellis decode is ALU work
that does not shrink with the weights, and GB10 has proportionally less of it to
spare, so above some M the 4-bit read stops paying for the decode. **The fix is
a different path, not a different tiling:** reconstruct the weight to bf16 once
and call cuBLAS, which is what ExLlamaV3 does at large M. It is not built here,
and it should not be guessed at without a GB10 to measure on. Until then, on a
Spark the quantisation is buying memory on dense layers rather than speed, while
the MoE path -- where each weight is read once per expert and reused across
hundreds of rows -- wins at 3.3-4.2x.

The autotuner ports cleanly: it times candidates with `cudaEvent` on whatever
device it is on and reads `MaxSharedMemoryPerBlockOptin` rather than assuming
sm_120's. At B>=16 nothing beat its pick. At B=1 a manual `chunk=96` beat it by
7-14%, which was a real bug -- a batch-1 decode is ~7 us and it was timing three
reps, so 21 us, the same order as the timer measuring it. Short shapes now get
20 reps.

## 1b. Calibrate before benchmarking

Three constants are tuned rather than derived, and their defaults were measured
on 188 SMs, 128 MB of L2 and 1.46 TB/s. The GEMM's split-k thresholds and the
MoE `block_m` ladder are the two that do not self-adjust:

```
python bench/calibrate.py
```

It sweeps them on the device it finds, needs no checkpoint (synthetic shapes;
trellis decode is data-independent so the timings are real), and prints an
export block if anything beats the defaults by more than 2%. `bench_gemm.py
--synthetic k,n,bits` does the same for a single shape.

## 2. Sizing: two nodes, not one

`brandonmusic/GLM-5.3-Flash-tr3-4bpw` is 164 GB of safetensors against ~121 GiB
of usable unified memory per Spark. One node cannot hold it. The reference
deployment is **two nodes, TP=2 over a direct 200Gb QSFP link, ~82 GiB resident
per node**, which leaves room for KV and activations.

Four nodes would give headroom, but TP=4 across two links is a different
topology problem and nobody has published it; TP=2 is the proven shape.

## 3. Building

The plugin builds for sm_121 from the same source as sm_120:

```bash
TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=8 pip install . --no-build-isolation
```

Use `12.1a` instead if something else in your image wants the arch-accelerated
variant; these kernels do not need it, since they use no arch-specific
instruction. With no `TORCH_CUDA_ARCH_LIST` set and no GPU visible, `setup.py`
falls back to `8.0;8.6;8.9;9.0;12.0` plus `12.1` when nvcc reports
`compute_121`.

**You need a base image that carries the model definition.**
`Glm5NextForConditionalGeneration` is in no *released or nightly* vLLM —
`glm5_next.py` 404s on `main` and the architecture is absent from the registry —
but there is an **official day-0 preview image**, and it is multi-arch:

```
vllm/vllm-openai:glm53-flash              # amd64 + arm64
vllm/vllm-openai:glm53-flash-arm64-cu130  # what the 2x Spark kit pins
vllm/vllm-openai:glm53-flash-x86_64-cu130
```

So aarch64 is not the obstacle it would otherwise be. Pin by digest; the 2x
Spark kit pins `glm53-flash-arm64-cu130@sha256:905c0293…`.

`docker/Dockerfile.sparse-mla` in this repo layers the plugin onto any of them:

```bash
docker build -f docker/Dockerfile.sparse-mla \
  --build-arg BASE=vllm/vllm-openai:glm53-flash-arm64-cu130 \
  -t glm53-exl3-spark .
```

## 3a. Two things upstream of this plugin that stop it booting

Neither is in these kernels; both were hit bringing GLM-5.3-Flash up on 2x GB10
(issue #2), and both cost a day if you meet them cold.

**vLLM's sparse-attention indexer picks a top-k kernel GB10 cannot run.** With
`select_k = index_topk / index_kpool = 2048 / 4 = 512`,
`sparse_attn_indexer_kpool.py` calls `torch.ops._C.persistent_topk`, which fails
at startup:

```
persistent_topk would oversubscribe and the FilteredTopK fallback requires
>=128KB smem per block (have 101376). total_ctas=85 > num_sms*occupancy=48
(TopK=512, vec_size=4, ctas_per_group=85, smem=49152)
```

48 SMs is the whole problem: the CTA count the kernel wants does not fit the
part, and the fallback wants more shared memory than sm_121 offers. Skip the
persistent path and let it fall through to `top_k_per_row_decode`, in both
`sparse_attn_indexer_kpool.py` and `sparse_attn_indexer.py`. This is vLLM's
top-k machinery, not ours -- we only consume the indices it produces -- and its
GB10 fallback does not currently exist.

**Built-in MTP needs `--block-size 256`.** Enabling `mtp` (formerly
`glm5_next_mtp`) with fp8 KV otherwise dies in DeepGEMM `attention.hpp:320`: on
arch 12 with an fp8 cache `block_kv` must be exactly 64, and with
`index_kpool = 4` that means a 256-token page. Without MTP the default block
size is fine. Note this is a different constraint from the one our own backend
reports through `get_supported_kernel_block_sizes()` ([64, 256]) -- they agree
at 256, which is why that is the value to use.

Measured cost of the MTP arm on 2 nodes: the KV pool drops 22% (2.55M -> 1.99M
tokens at `--gpu-memory-utilization 0.85`) and single-stream goes 14.6 -> 30.3
tok/s at 70% acceptance, 3.3 tokens/step.

## 4. What actually limits a Spark, and what this plugin does about it

On GB10 the wall is **weight streaming, not MMA**. The Reederey87 kit measures
weight streaming at roughly 63% of every prefill step, and computes that even a
*perfect* MoE kernel would be capped at about +4–5% end-to-end prefill by
Amdahl. That is the single most important thing to internalise before tuning
anything: on a Spark, **bytes are the lever, not TFLOPS**.

Which sets the priorities for this plugin on that machine:

* **4-bit weights are the point.** That is what EXL3 buys you, and on a
  218 GB/s machine it is worth far more than it is on a 1.4 TB/s one.
* **Use the fp8 KV cache.** `--kv-cache-dtype fp8` halves cache bytes. On the
  RTX PRO 6000 it is worth 1.3–1.8x on the attention kernel in isolation and
  almost nothing end to end, because that machine is not short of bandwidth. A
  Spark has 6.7x less of it, so the same change should matter considerably more
  there. The scale never touches an element -- it folds into the softmax scale
  and the merged output -- so it costs no arithmetic.
* **The autotuner adapts to the device.** Its choices are keyed on
  `(rows, heads, topk, head_dim, cache dtype)` and measured on the machine it is
  running on, which matters here: a Spark has far fewer SMs than the 188 these
  defaults were tuned against, so the split counts that fill one machine are not
  the ones that fill the other. Nothing is hard-coded for sm_120.

Serving flags, once the image exists:

```
--attention-backend CUSTOM \
--kv-cache-dtype fp8 \
--tensor-parallel-size 2
```

`CUSTOM` is how vLLM lets an out-of-tree package supply an attention backend;
this plugin binds that slot to its sparse-MLA implementation. See the main
README for what that backend does and does not cover (decode is ours; prefill
runs through it too, because vLLM has no MLA prefill backend for this model).

## 5. How the kernels compare — measured

The Reederey87 kit replaced ExLlamaV3's large-M expert path with its own
`overlay/exl3_fat_gemm.cu`. So did this plugin, independently. Their kernel
compiles for sm_120 **unchanged** — same intake constraints, same instruction
set — so the comparison neither project could make from its own numbers is
actually available: same GPU, same trellis tensor, same shapes.

`bench/bench_vs_spark_fat_gemm.py` runs it. Their kernel is not vendored here;
point it at a checkout. Both kernels take the identical K4 MCG trellis straight
from `GLM-5.3-Flash-tr3-4bpw`, and their outputs agree to **~6e-4 relative**,
which cross-validates both implementations.

RTX PRO 6000 Blackwell (sm_120), GLM expert shapes, layer 10 expert 0:

**gate_proj, k=4096 n=2048**

| m | ours | fat | ours TFLOPS | fat TFLOPS | ours/fat |
|---|---|---|---|---|---|
| 128 | 24.9 us | 117.8 | 86.3 | 18.2 | **4.73x** |
| 512 | 57.6 | 126.2 | 149.1 | 68.1 | **2.19x** |
| 1024 | 95.1 | 136.9 | 180.6 | 125.5 | 1.44x |
| 2048 | 152.5 | 224.4 | 225.3 | 153.1 | 1.47x |
| 3584 | 245.1 | 353.1 | 245.3 | 170.3 | 1.44x |
| 4096 | 296.6 | 364.2 | 231.7 | 188.7 | 1.23x |

**down_proj, k=2048 n=4096**

| m | ours | fat | ours TFLOPS | fat TFLOPS | ours/fat |
|---|---|---|---|---|---|
| 128 | 24.7 us | 65.6 | 87.0 | 32.7 | **2.66x** |
| 1024 | 82.1 | 113.0 | 209.3 | 152.0 | 1.38x |
| 3584 | 213.0 | 263.8 | 282.3 | 227.9 | 1.24x |
| 4096 | 251.1 | 309.5 | 273.7 | 222.0 | 1.23x |

So on identical silicon **this plugin's kernel is ahead everywhere, by 1.23x at
the largest batches and 2.7-4.7x at small M**, and by 1.44x at M=3584, the batch
size their kit tunes for. That replaces the "both at about 80% of their own
machine" reading in an earlier draft of this document, which was an inference
from two numbers measured on different hardware and turned out to be the wrong
conclusion.

**What this does not establish.** Their tile configuration -- 128x16x128, three
cp.async stages -- was chosen for GB10: far fewer SMs and 218 GB/s. Running it
on 188 SMs at 1.46 TB/s tests their *tile choice on our machine*, not their
kernel on its own. A loss here is not proof that this plugin wins on a Spark,
and the honest experiment is still the one in §4: build both for sm_121 and run
this same benchmark there. It is now a script rather than a plan.

One number is suggestive, though. Their kernel reaches 170-228 TFLOPS here
against the 73.5 TFLOP/s they measure on GB10 -- a factor of 2.3-3.1, well short
of the ~4.5x compute ratio between the two parts. That is consistent with their
own finding that GB10 is bandwidth-bound rather than MMA-bound, and it means
neither kernel's sm_120 result should be extrapolated to a Spark by scaling.

A note on the third column the benchmark prints: stock ExLlamaV3's `exl3_gemm`
measures 19-27 TFLOPS at these batch sizes, but that is not the baseline the fat
kernel replaced and it is not a fair use of it. `exl3_gemm` is a small-M decode
kernel; at large M the ExLlamaV3 stack reconstructs the weight and calls cuBLAS,
which is what both fat kernels are actually competing with.

## 6. Reference points from the 2x Spark kit

For calibration, from
[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark)
(their stack, not this one; 2 nodes, TP=2, DFlash2 speculative decoding):

| | |
|---|---|
| Context window | 1,000,000 tokens |
| Prose decode | ~28–31 tok/s at the 1M window |
| Structured decode | ~67 tok/s at speculative acceptance 1.0 |
| Cold prefill | ~1000–1070 tok/s solo at 240k |

For contrast, this plugin on 4x RTX PRO 6000 at TP=4 measures ~105 tok/s
single-stream decode and 5.8k tok/s prefill at 8k context without speculation.
The machines are not comparable; the gap is mostly the 6.7x bandwidth
difference and the node count, not the kernels.

Their kit's hardest-won findings are about prefix caching and scheduler
fairness on the hybrid KDA+MLA architecture, not kernels, and none of that is
addressed here. If you are deploying rather than benchmarking, that is the
document you want.
