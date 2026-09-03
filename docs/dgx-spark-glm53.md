# Running GLM-5.3-Flash on DGX Sparks

**Status: untested by the author.** This plugin is developed and measured on
4x RTX PRO 6000 Blackwell (sm_120). I do not have a DGX Spark. Everything below
that is hardware-specific about GB10 is cited to a source, and everything that is
a measurement of this plugin is labelled with the machine it was measured on.
Treat it as a bring-up guide with the reasoning shown, not a validated recipe.

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
| Memory bandwidth | **218 GB/s** measured LPDDR5X unified | see §4 -- this is the whole story on a Spark |
| Toolchain | CUDA 13.x system `ptxas`; older bundled ptxas has no sm_121 | builds with CUDA 13; `setup.py` adds `12.1` to the arch list after asking `nvcc --list-gpu-arch` whether it knows `compute_121` |
| Usable memory | ~121 GiB per node | the 4bpw checkpoint is 164 GB, so see §2 |

Numbers in the GB10 column are from the Reederey87 kit's `docs/11-gb10-kernel-program.md`
and the sources it cites.

Nothing here needs a code change. The kernels were kept to the sm_80-era
instruction set deliberately, which is the same intake rule that kit adopted.

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

**The blocker is not this plugin, it is the model definition.**
`Glm5NextForConditionalGeneration` is in no released or nightly vLLM — I
checked: `glm5_next.py` 404s on `main` and the architecture is absent from the
registry. It exists in community images as an out-of-tree `vllm.models.glm5next`
package. Those images are **x86_64**, and a Spark is aarch64, so you cannot
simply reuse the one this repo's own Docker guide points at. You need a vLLM
build for arm64 that carries the model definition — which is what the
Reederey87 kit's `Dockerfile` produces, and the reason it builds its own image.

Once you have such an image, `docker/Dockerfile.sparse-mla` in this repo layers
the plugin onto it:

```bash
docker build -f docker/Dockerfile.sparse-mla --build-arg BASE=<your arm64 image> \
  -t glm53-exl3-spark .
```

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

## 5. How the kernels compare — honestly

The Reederey87 kit replaced ExLlamaV3's kernels with its own `exl3_fat_gemm.cu`
for the MoE path. So did this plugin, independently. Both report against their
own machine:

| | their GB10 fat kernel | this plugin, sm_120 |
|---|---|---|
| Measured | **73.5 TFLOP/s** (up from 52, +41%) | **317–327 TFLOPS** at m ≥ 512 |
| Their machine's ceiling | ~92 TFLOP/s for that op | ~400 TFLOPS |
| Fraction of ceiling | ~80% | ~79–82% |
| Versus ExLlamaV3's own kernel | not stated in those terms | **3.9–5.0x** at m ≥ 512 |

**Both kernels sit at about 80% of what their respective machine can do.** That
is the honest read, and it means there is no basis for claiming this plugin
would be faster on a Spark — the two numbers are measured on machines a factor
of four apart in compute and nearly seven apart in bandwidth, and neither
project has run the other's kernel on the other's hardware.

What *is* fair to say is narrower and more useful:

1. Neither kernel is leaving much on the table against its own hardware, so a
   swap is unlikely to be transformative either way. That kit reached the same
   conclusion when it evaluated b12x's `trellis3_t256` EXL3 kernel and parked it:
   the bar was to beat its own 73.5 TFLOP/s, and it computed the end-to-end
   ceiling for a perfect replacement at +4–5%.
2. The discriminating experiment is cheap and nobody has run it: build this
   plugin for sm_121 and microbenchmark `exl3_linear` / `exl3_moe_gemm` on the
   rank-sliced GLM shapes (hidden 4096, moe_intermediate 2048; under TP=2 that
   is gate/up 4096→1024 and down 1024→4096) against the incumbent, in a stopped
   window. `bench/bench_gemm.py` does exactly this and prints achieved TFLOPS
   against a measured speed-of-light, so it transfers unchanged.
3. The attention side is a genuinely different question and is *not* covered by
   either kit's GEMM work. This plugin's sparse-MLA decode kernel sustains about
   92% of its machine's stream-copy bandwidth at batch 32 and beats b12x by
   2.5x at batch 1. On a bandwidth-starved machine, a bandwidth-efficient
   attention kernel plus an fp8 cache is the part most likely to transfer.

If you run any of this, the numbers would be welcome — particularly (2), which
would settle a question two projects have now both left open.

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
