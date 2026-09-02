# vllm-exl3

EXL3 ([ExLlamaV3](https://github.com/turboderp-org/exllamav3) trellis quantization)
support for vLLM, with custom CUDA kernels tuned for continuous batching.

```bash
pip install -e .            # needs torch + nvcc; builds vllm_exl3._C
vllm serve TelperionAI/Qwen3.8-27B-EXL3-5.5bpw
```

vLLM picks the plugin up through the `vllm.general_plugins` entry point, so
`quant_method: "exl3"` checkpoints just load. CUDA graphs and `torch.compile`
work; nothing to configure.

## Why custom kernels

ExLlamaV3's GEMM fixes `TILESIZE_M` at 16 and loops over the batch in 16-row
chunks, re-reading the entire trellis for each chunk. That is close to optimal
for single-stream decode, and it is what the format was designed around. But it
makes cost linear in batch size: on an RTX PRO 6000 Blackwell the kernel sits at
~1.65 TB/s -- exactly HBM speed -- from m=24 upward, because it is re-reading
weights, not computing. A 512-row batch reads a 47 MB tensor 32 times.

Continuous batching lives precisely in that range. So this kernel tiles M: a
block owns `BM` rows x `BN` columns and the *whole* k extent, reads its slice of
the trellis **once**, and amortizes the dequant across all `BM` rows. Owning the
full k extent also means the block holds a complete output row segment, so the
output Hadamard and `svh` scaling fold into the epilogue -- no second pass, and
no `k x n` fp16 scratch buffer.

## One op per layer

`torch.ops.vllm_exl3_C.exl3_linear` does a whole quantized linear in one call:

* **Activation transform, fused.** `had(x * suh)` for every shard in a single
  launch, reading `x` once. bf16 -> fp16 conversion happens in the same load, so
  there are no separate `.to()` kernels -- those were ~10% of decode GPU time.
* **All shards in one GEMM.** Fused layers (`qkv_proj`, `gate_up_proj`,
  Qwen3.5's `in_proj_qkvz`) are one launch. Only the activation slice differs per
  shard; trellis, `svh` and the output are addressed by absolute column, so a
  block just looks up which shard its column range belongs to.
* **Epilogue emits the caller's dtype** directly, bf16 or fp16.
* **Registered through `torch.library`**, not raw pybind, so Dynamo can trace it.
  A bare pybind function is opaque to Dynamo and blocks vLLM's CUDA graph
  capture outright. Being functional (it allocates and returns its output) also
  makes the fake/meta implementation trivial, and lets inductor fuse around it --
  it ends up merged with the surrounding RMSNorm.

Two further departures from upstream:

* **The input transform is a separate kernel.** ExLlamaV3 fuses it into the front
  of its GEMM and pays with a cooperative launch (a grid-wide sync between
  transform and matmul). Splitting it costs one extra pass over the activations
  -- a few microseconds, since `x` is small and L2-resident -- and in exchange
  the GEMM is an ordinary kernel: no cooperative launch, no grid size tied to the
  SM count, nothing special under graph capture.

* **Split-k for narrow layers and small batches.** `down_proj` is only 40 blocks
  wide at `BN=128`, leaving most of a 188-SM GPU idle. Splitting k adds blocks
  without multiplying how often `A` is re-read (shrinking `BN` would). Partials
  go to an fp32 accumulator via coalesced atomics; the epilogue Hadamard consumes
  them and re-zeroes the buffer, so no per-call memset is needed. The split
  factor is capped so accumulator traffic stays under ~30% of the weight bytes it
  is trying to stream faster.

## Measured

RTX PRO 6000 Blackwell (188 SMs), measured roofline **1520 GB/s** HBM read and
**400 TFLOPS** fp16 dense. An EXL3 matmul reads `k*n*bits/8` bytes and does
`2*m*k*n` flops, so arithmetic intensity is `16m/bits` flop/byte: memory bound
below m~99 at 6 bits, compute bound above. Speed of light is the max of the two
limits. `bench/bench_gemm.py` reports achieved TFLOPS, GB/s and %SoL per shape.

Qwen3.5-27B 5.5bpw, 6-bit tensors, speedup vs `exllamav3_ext.exl3_gemm`:

| layer | m=16 | m=32 | m=64 | m=128 | m=256 | m=512 | m=2048 | m=8192 |
|---|---|---|---|---|---|---|---|---|
| `q_proj` 5120x12288 | 1.11x | 1.86x | 2.43x | 2.65x | 2.59x | 3.05x | 4.28x | 4.50x |
| `up_proj` 5120x17408 | 1.08x | 2.01x | 2.43x | 2.30x | 3.17x | 3.88x | 4.02x | 3.94x |
| `down_proj` 17408x5120 | 1.42x | 2.71x | 3.37x | 3.66x | 4.03x | 4.26x | 4.58x | 4.98x |

Faster at every batch size measured, and **101-128% of the roofline above** at
m=16-32. Over 100% is not an error: that roofline charges every weight byte to
HBM, and this GPU has a **128 MiB L2** -- a whole `q_proj` (47 MB) or `down_proj`
(67 MB) trellis fits in it, so part of the traffic never reaches memory. The
effective read rate at m=16 is ~1.9 TB/s against 1.52 TB/s of HBM. At m>=2048 the
compute bound binds instead, at 71-83% (330 TFLOPS peak).

### End to end (online serving)

`vllm serve` + `vllm bench serve`, 8k in / 1k out, bf16, no MTP, prefix caching
**disabled** and a distinct seed per run (with caching on and a fixed seed the
runs replay each other's prompts and "TTFT" becomes a cache lookup). Compared
against the same checkpoint on exllamav3 via TabbyAPI, same GPU.

Prefill, 8000-token prompt:

| engine | prefill tok/s |
|---|---|
| exllamav3 / TabbyAPI, 1 GPU | ~3,050 (engine-reported) |
| vllm-exl3, TP=1 | **4,816** (1.58x) |
| vllm-exl3, TP=4 | **7,674** (2.52x) |

Output token throughput, and per-token decode latency (TPOT):

| conc | exllamav3 1 GPU | vllm-exl3 TP=1 | vllm-exl3 TP=4 |
|---|---|---|---|
| 1 | 25.3 tok/s (TPOT 17.9 ms) | 54.4 (16.8 ms) | 92.7 (9.8 ms) |
| 4 | 38.7 (30.3 ms) | 150.7 (21.9 ms) | 249.7 (13.5 ms) |
| 16 | did not complete | 307.3 (37.0 ms) | 504.1 (23.4 ms) |
| 64 | did not complete | 389.4 (108.1 ms) | 698.3 (59.6 ms) |

Two honest caveats. At concurrency 1 the two are close on decode latency (17.9 vs
16.8 ms/token) -- both are bandwidth-bound reading the same 16.5 GB of weights,
and this kernel is already at ~89% of that limit, so there is little left to win
there. And exllamav3's "did not complete" rows are requests timing out in the
benchmark client, not crashes: with 8k prompts its prefill is effectively
serialized across requests at ~3k tok/s, so 16 arrivals queue ~43 s before decode
starts. That is a scheduling difference (vLLM interleaves chunked prefill with
decode), not a statement about exllamav3's kernels.

Offline (short prompts, batch of identical requests), for reference:

**Known gap:** at m<=16 upstream is still marginally faster on two of three
shapes -- its dedicated GEMV path runs at ~90% of speed of light and is hard to
beat. That is single-stream decode; anything with real batching is past it.

## What profiling found

Profiling a batch-32 decode run (`bench/`-adjacent scripts in the repo history)
turned up three things, in order of size:

1. **Only 2.14s of a 7.39s run was GPU work.** Everything else was launch
   overhead, because the pybind entry point blocked CUDA graph capture. Fixing
   the registration and enabling graphs took the run to 2.34s.
2. **~10% of GPU time was elementwise dtype conversion** -- two extra passes over
   activations per linear. Folding the conversion into the Hadamard load/store
   and the epilogue removed it, along with ~44k kernel launches.
3. **50.2M shared-memory bank conflicts in the GEMM** (`ncu`). `ldmatrix` reads 8
   rows at one column offset; with a 64-byte A row those land on only 2 distinct
   bank groups, and no XOR swizzle can fix it because there are only 4 columns to
   permute. Padding the row stride to 80 bytes spreads them across 8 banks and
   cut conflicts 8.4x, to 6.0M.

A fourth came from the roofline rather than a counter: at small batch the kernel
is *dequant*-bound, not memory-bound, and with two warp rows each trellis tile
was being decoded twice. A 16-wide warp tile (one warp row) decodes each tile
exactly once and lifted m=16-32 from ~66% to 84-98% of speed of light.

## What is supported

**Bitrates and codebooks: all of them.** Every EXL3 bitrate (1-8 bits, which may
vary per tensor within one checkpoint) and all three procedural codebooks --
`3inst`, `mcg`, `mul1` -- are instantiated. The codebook multiplier is a constant
of the codebook id rather than per-tensor data, so the id is all the kernel
needs. Verified end-to-end on both a 5.5bpw checkpoint (bits {5,6,7}) and a
4.0bpw one (bits {4,6}).

**Models: dense architectures that vLLM already supports.** The plugin only
provides the EXL3 runtime -- it does not implement any model. Layers whose
tensors are not EXL3 in the checkpoint (norms, embeddings, an unquantized vision
tower) fall through to vLLM's normal paths untouched, so a multimodal model works
as long as its language model is the quantized part.

**Mixture-of-experts works.** Routed experts run as a grouped GEMM: vLLM's
`moe_align_block_size` sorts the (token, expert) pairs and pads each expert's run
to a whole row block, so every block belongs to one expert and the kernel just
offsets the trellis by a per-block expert id. Verified on Qwen3.5-35B-A3B (256
experts, top-8, 4-bit, `mcg` codebook), eager and under CUDA graphs.

The EXL3-specific part is that `suh` is per expert *and* per shard, and it lives
inside the input Hadamard -- so the activation transform has to happen after
routing. `exl3_moe_had_in` gathers each routed row and transforms it with its own
expert's scales in one pass.

One caveat:

* **Module-name resolution is heuristic.** vLLM's module paths do not always
  match the checkpoint's (Qwen3.5 is `model.language_model.layers.N` on disk but
  `language_model.model.layers.N` in vLLM), so names are matched exactly first
  and then with the `model`/`language_model` wrapper segments dropped. A model
  that renames layers more aggressively than that would need a mapping entry.
  `VLLM_EXL3_DEBUG_NAMES=1` logs what resolved and what did not.

## VRAM

Exactly what ExLlamaV3 stores: `trellis` (int16), `suh`/`svh` (fp16), bias. No
dequantized weights are ever materialized, and the activation and split-k
workspaces are process-wide rather than per layer. Qwen3.5-27B at 5.5bpw loads to
**18.89 GiB** of parameters against a 19.95 GiB checkpoint.

## Notes on the format

`suh` is one value per input channel, but it differs between the shards of a
fused layer: q/k/v (and gate/up) are quantized as separate tensors with
separately chosen input scales. Only the sign pattern is shared; the magnitudes
genuinely differ, and `suh` sits *inside* the Hadamard, so the shards cannot be
folded into a single vector. `trellis` and `svh` do concatenate along the output
dim (every shard of a fused group shares a bitrate), so a layer keeps one trellis
addressed by offset, plus one `suh` row per shard.

Slicing that trellis in Python would produce a non-contiguous view whose row
stride the kernel cannot recover -- hence the shard-map approach.

## Determinism

Split-k reduces with fp32 atomics, so results are reproducible in value but not
bit-exact run to run. `VLLM_EXL3_DETERMINISTIC=1` disables it: bit-exact
everywhere, slower for small batches and narrow layers.

## Environment variables

| variable | meaning |
|---|---|
| `VLLM_EXL3_BACKEND` | `native` (default), or `exllamav3` to run upstream's kernels as an oracle |
| `VLLM_EXL3_DETERMINISTIC` | `1` disables split-k for bit-exact output |
| `VLLM_EXL3_DEBUG_NAMES` | log which modules resolve to EXL3 vs unquantized |

## Tests

```bash
VLLM_EXL3_TEST_MODEL=/path/to/exl3-model pytest tests/ -v
```

Covers every BM tier, both epilogues, multi-shard equivalence, bf16 activations
and the split-k accumulator invariant, against a dense fp16 reference
reconstructed from the same trellis.

## MTP (speculative decoding)

```python
speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 1}
```

The checkpoint's MTP head is EXL3-quantized at 4 bits, but it is **missing from
`quantization_config.json`** -- so `Exl3Config` recovers any such module by
reading the safetensors headers (shape/dtype only, no tensor data). Offline
tok/s, no-MTP -> MTP: c=1 60.8 -> 93.6, c=4 216 -> 391, c=16 787 -> 1241,
c=64 1745 -> 1969. Biggest gains at low concurrency, where the GPU is idle
waiting on weights.

## Tensor parallel

Works unchanged: `tensor_parallel_size=4` shards the trellis, `suh` and `svh`
through the stock vLLM parameter path. Splitting the input dim stays exact
because the EXL3 input transform is a *block-diagonal* Hadamard over 128-element
blocks and every split dim here is 128-divisible; `svh` and the output Hadamard
are linear, so they commute with the all-reduce. Weights land at ~7 GiB/GPU.

### MoE throughput

Qwen3.5-35B-A3B (256 experts, top-8, 4-bit, `mcg`), one GPU, greedy decode:

| concurrency | tok/s | ms/token |
|---|---|---|
| 1 | 204 | 4.91 |
| 4 | 661 | 6.05 |
| 8 | 1244 | 6.43 |
| 16 | 2252 | 7.11 |
| 32 | 3160 | 10.13 |
| 64 | 4611 | 13.88 |

At c=8 the decode is GPU-bound and splits roughly: **38% EXL3 GEMMs** (of which
the expert GEMMs proper are 16%), 8% gated-delta-net, 8% assorted elementwise,
8% an unquantized bf16 GEMM (the router), 6% vLLM's routing/alignment kernels,
5% the dense activation transform, 4% the split-k epilogue, 4% the MoE activation
transform, and 1.5% the combine.

Folding SwiGLU into the down-projection's input transform took a chunk out of
that elementwise slice. Online, 8k in / 1k out, prefix caching off, one GPU:

| | prefill (tok/s) | c=1 | c=8 | c=32 | c=64 |
|---|---|---|---|---|---|
| before | 21,825 | 191.5 | 705.8 | 1141.5 | 1268.5 |
| after | 23,791 | 200.8 | 742.9 | 1211.6 | 1313.4 |
| | +9.0% | +4.9% | +5.3% | +6.1% | +3.5% |

(Decode columns are output tok/s. These are far below the short-prompt table
above because every request carries 8k of context.)

## Versus NVFP4

`unsloth/Qwen3.8-27B-NVFP4` is the same base model, so it is a direct read on
where a trellis format stands against native FP4 tensor cores. One RTX PRO 6000
Blackwell each, 8k in / 1k out, prefix caching off, `--gpu-memory-utilization
0.9`, **FP8 KV cache on both sides**, NVFP4 on its fastest backend
(`--kernel-config '{"linear_backend":"flashinfer_b12x"}'`).

Two things have to be matched or the comparison is meaningless, and both default
in NVFP4's favour:

* Unsloth's checkpoint ships a `kv_cache_scheme`, so vLLM silently gives it an
  **FP8 KV cache** while EXL3 runs BF16 -- half the KV traffic per decode step.
* `linear_backend` `auto`/`cutlass` is not the fast path on SM120. `flashinfer_b12x`
  is worth ~18% on decode TPOT at c=1. (Plain `b12x` cannot load this checkpoint:
  its FP8 kernel wants static per-tensor activation scales, the checkpoint has
  dynamic per-token.)

End-to-end output tok/s -- every request pays an 8k prefill, so this folds
prefill and decode together:

| tok/s | NVFP4 (21.83 GiB) | EXL3 5.5bpw (19.98 GiB) | EXL3 4.0bpw (15.73 GiB) |
|---|---|---|---|
| prefill | **11,862** | 5,182 | 5,435 |
| c=1 | 56.1 | 56.6 | **67.6** |
| c=8 | **317.3** | 253.7 | 283.9 |
| c=32 | **645.9** | 408.2 | 438.0 |
| c=64 | **772.8** | 448.3 | 471.6 |

Decode alone (concurrency / mean TPOT), with prefill taken out:

| decode tok/s | NVFP4 | EXL3 5.5bpw | EXL3 4.0bpw |
|---|---|---|---|
| c=1 | 58.2 | 62.0 | **75.0** |
| c=8 | 358.6 | 320.9 | **365.5** |
| c=32 | **836.4** | 622.0 | 675.4 |
| c=64 | **1054.0** | 706.7 | 744.2 |

The crossover is the whole story, and it is exactly what the kernel analysis
predicts. At c=1-8 decode is weight-bandwidth-bound, the EXL3 GEMM is already at
the HBM roofline, and the format with fewer bits wins -- 4.0bpw is **22% faster
than NVFP4 at c=1** on a checkpoint 28% smaller. From c=32 up the batch is large
enough to be compute-bound, and there the trellis costs ~74 instructions per 8
weights to dequantize into bf16 mma while NVFP4 feeds FP4 tensor cores directly:
NVFP4 wins by 24% at c=32 and 42% at c=64. Prefill is the same effect at its
extreme -- **2.3x**.

### MoE: Qwen3.6-35B-A3B

`nvidia/Qwen3.6-35B-A3B-NVFP4` vs `UnstableLlama/Qwen3.6-35B-A3B-exl3-4.00bpw`,
same harness. Neither checkpoint ships a `kv_cache_scheme`, so both ran BF16 KV
with nothing to match by hand. On SM120 the only NVFP4 MoE backends that load at
all are `auto` and `flashinfer_b12x` -- `flashinfer_cutedsl` and `cutlass` both
reject the scheme on this device.

| tok/s | NVFP4 b12x | NVFP4 auto | EXL3 4.00bpw |
|---|---|---|---|
| prefill | **34,316** | 30,038 | 24,394 |
| c=1 | 211.7 | **213.9** | 204.6 |
| c=8 | **799.1** | 777.5 | 746.5 |
| c=32 | **1390.4** | 1375.9 | 1185.3 |
| c=64 | **1674.7** | 1632.9 | 1424.0 |

Decode alone (concurrency / mean TPOT):

| decode tok/s | NVFP4 b12x | NVFP4 auto | EXL3 4.00bpw |
|---|---|---|---|
| c=1 | 221.2 | **225.2** | 218.3 |
| c=8 | **870.5** | 860.2 | 852.9 |
| c=32 | 1607.2 | **1636.8** | 1451.9 |
| c=64 | 2006.9 | **2017.0** | 1824.9 |

The gap is far smaller than on the dense model: EXL3 is within **3% at c=1** and
**2% at c=8** on decode, and 11-13% behind at c=32-64, against 2.3x on dense
prefill and 42% on dense decode at c=64. MoE decode streams one expert slice per
row-block and is weight-bound almost everywhere, which is the regime the trellis
is competitive in -- there is much less compute-bound headroom for FP4 tensor
cores to exploit. Prefill still favours NVFP4, but by 1.41x rather than 2.3x.

EXL3 also leaves considerably more room for context: 65.44 GiB of KV against
51.59 GiB at the same `--gpu-memory-utilization 0.9`, i.e. 2.56M vs 2.02M tokens
(**+27%**), from an 18.07 GiB checkpoint against 21.85 GiB. For MoE, `auto` is
already fine for decode; `flashinfer_b12x` is worth 14% on prefill.

So EXL3 is the better choice for local, low-concurrency serving on this hardware,
and NVFP4 for prefill-heavy or high-concurrency serving. Closing the compute-bound
gap needs a codebook that decodes into fp8/nvfp4 rather than bf16 -- a format
change, not a kernel change. Note these are speed numbers at different bit
budgets; no accuracy comparison was run.

## MoE notes

Two things worth knowing if you touch this path:

* vLLM's MoE weight loader decides `is_fused = loaded_weight.dim() == 3`, i.e. it
  treats any 3-D tensor as *all experts stacked*. An EXL3 trellis is genuinely
  3-D **per expert**, so that heuristic slices it apart on the wrong axis, and
  there is no hook before the decision. This module therefore installs its own
  `load_weights`.
* `moe_align_block_size` marks blocks belonging to no expert with `-1`, and those
  appear inside the live row range, not only in the tail. Both MoE kernels skip
  them explicitly; without that the trellis pointer goes negative and the launch
  faults.

Padding each expert's run up to a whole block wastes rows when the batch is small
and the expert count large: with 256 experts and 8 routed, live rows are 20-32x
the useful rows below 32 tokens. That cost is **sublinear but not free** --
measured by varying the block size on Qwen3.5-35B-A3B:

| block | c=1 | c=8 | c=32 |
|---|---|---|---|
| 16 | 186 tok/s | 1173 | 2684 |
| 32 | 172 (0.92x) | 1031 (0.88x) | 2294 (0.85x) |
| 64 | 158 (0.85x) | 848 (0.72x) | 1716 (0.64x) |
| 128 | 123 (0.66x) | 610 (0.52x) | 1161 (0.43x) |

Quadrupling the block costs ~2x, not ~4x, because weight traffic is unchanged
(the same experts are read either way) -- the surplus is MMA work on a
memory-bound kernel. But it is real, so the block size drops to 16 when there are
few tokens per expert. **16 is the floor**: `mma.m16n8k16` computes 16 rows at a
time, so no finer alignment is expressible.

It costs no VRAM. The padded buffers are transient and reused by the caching
allocator: peak activation is 1.04 GiB for the MoE model against 1.01 GiB for the
dense one, and weights land at 17.79 GiB against a 19.58 GiB checkpoint.

## Not yet done

Multi-node. Bit-exact determinism under split-k (see above).

## Attribution

The trellis bit packing, the procedural codebook and the mma fragment layout are
part of the EXL3 on-disk format and follow ExLlamaV3 (MIT, (c) turboderp); those
headers are vendored with attribution in `vllm_exl3/csrc/`. The GEMM tiling,
split-k, fused epilogue, shard map and vLLM integration are new.

## Where the time goes

Profiled with the torch profiler and `ncu` (`bench/profile_workload.py`), 8k
context on one GPU:

* **Prefill is 82% this GEMM**, running at ~292 TFLOPS -- about 71% of the fp16
  compute peak. That is roughly what cuBLAS achieves on the same shapes, while
  reading 2.9x fewer bytes.
* **Decode at 8k context** (c=16) splits ~60% GEMM, ~25% attention, ~8% GDN. The
  GEMM there is at 84-98% of the memory-bandwidth limit.

`ncu` at prefill sizes: tensor pipe 77.8% active, occupancy 2 blocks/SM
(register-bound at 99 registers), and the dominant stall is `math_pipe_throttle`
-- the trellis dequant competes with MMA for issue slots. This is structural:
Marlin's int4 dequant is 4 instructions for 8 weights (two `lop3`, an `hsub2`
and an `hfma2`), while EXL3's procedural trellis codebook needs ~52 for 8 --
**13x more ALU work per weight**. Marlin can be MMA-bound; this kernel cannot.

The one thing worth stealing from Marlin *was* its tile shape. It uses BK=64 so
an A row is exactly 128 B = 32 banks, which makes an 8-way XOR swizzle
conflict-free with no padding (`transform_a`, `... ^ (row % 8)`). This kernel
originally used BK=32, where a 64 B row leaves only 4 columns to permute and no
XOR can fix it -- hence the stride padding, which cost a pipeline stage. Adopting
BK=64 across all tiers was worth **10-26%** (tensor pipe 70.4% -> 77.8%) and
raised the small-batch shapes to 85-112% of the memory bound.

That large L2 also mis-calibrated the split-k heuristic. Its cost is an extra
read-modify-write of the fp32 accumulator, which the original heuristic charged
against HBM weight bytes -- but at m<=256 that accumulator is only a few MB and
stays in L2, so it is far cheaper than modelled. Discounting it when it fits
(`VLLM_EXL3_L2_GAIN`, default 2) unblocked split-k exactly where the kernel is
block-starved, and was worth up to **1.5x** in the m=16..256 range (`up_proj`
m=32: 59.4 -> 39.2 us). It is a discount, not a bypass -- treating the traffic as
free over-splits and regresses (`down_proj` m=128 went 101 -> 121 us).

Seven things were tried and rejected against measurement:

* **Hoisting the dequant's index arithmetic** out of the inner loop. `dq4`
  recomputes bit indices per call including a `% 48` (non-power-of-two, so a
  magic-multiply sequence), and those depend only on the lane. Precomputing them
  changed `smsp__inst_executed` by exactly zero instructions -- ptxas was already
  hoisting all of it. The ~74 instructions per 8 weights are real dequant work,
  not addressing overhead.

* A wider warp tile (WARP_N=32) at BM=128, re-evaluated after the bank-conflict
  fix changed the ldsm/dequant balance. Consistently ~3% slower; reverted.
* The cheaper 8-wide decode path for 6-bit. It does not apply: at the lane*8
  alignment the 58-bit window crosses three 32-bit words, so two loads cannot
  cover it. ExLlamaV3's `dq4` x2 for 6 bits is correct.

* **fp16 accumulation** (`VLLM_EXL3_FP16_ACC=1`, opt-in, off by default). Halving
  the accumulator registers affords BM=256 and so twice the MMA work per
  dequantized weight -- the obvious lever against the issue pressure above. It
  does not pay off: BM=256 needs 145 registers and 69 KB of shared, which drops
  occupancy to 1 block/SM and cancels the gain (8192-row `q_proj` 3518 -> 3599 us;
  512-row `down_proj` regressed 403 -> 694 us). Relative error rose 8-16x at the
  same time (3.5e-4 -> 2.6e-3, and 4.9e-3 on `down_proj`, where k=17408 gives the
  most accumulation steps). Left in as a flag so the result does not get
  re-derived. Note there is no bf16 accumulator to try instead -- bf16 mma inputs
  always accumulate to fp32.

* **Fusing the split-k epilogue into the GEMM.** With split-k on, the Hadamard
  epilogue is a second kernel launch, and at low concurrency turning split-k off
  entirely is roughly a wash (MoE decode, tok/s: c=1 205.2 vs 202.8, c=8 1240.5
  vs 1245.6, c=32 3244.7 vs 3182.8) -- the launch was eating the parallelism the
  split buys. The obvious fix is a counter-based epilogue: each block bumps a
  per-tile atomic after its partials land, and the last split to arrive owns the
  Hadamard. Implemented (`__threadfence()`, `atomicAdd` on a tile counter,
  partials re-read through L2 with `__ldcg`) and clearly worse: `q_proj` m=32
  33.6 -> 46.4 us, m=64 49.3 -> 69.8, `up_proj` m=16 36.3 -> 51.4. The fence on
  every split block plus the scattered, poorly-parallelised finalisation cost
  more than the launch they save. A kernel boundary is simply a cheaper
  grid-wide barrier than one built by hand.

* **Split-k for the MoE expert gemm** (`VLLM_EXL3_MOE_ACC_MAX_ELEMS`, off by
  default). The expert gemm is block-starved at low concurrency -- c=1 with
  top_k=8 gives 8 padded row-blocks, so w13 launches ~96 blocks against 188 SMs
  -- and splitting k is a real kernel win (rows=512, block_m=32: 33.0 -> 27.5 us,
  -17%). It does not survive end to end: splitting adds an epilogue launch per
  gemm per layer, ~96 extra kernels per decode step, which costs more than the
  parallelism buys (8k in / 1k out, tok/s: c=8 742.9 -> 712.8, c=32
  1211.6 -> 1196.2, c=1 and c=64 a wash). This is the same wall the fused
  epilogue hit from the other side. Kept behind the knob, and covered by tests
  so it cannot rot, because the trade-off is hardware-dependent.

* **Fusing the input transform into the gemm prologue.** At decode `exl3_had_in`
  is ~2.3 us to move ~100 KB -- about 0.06 us of actual traffic -- and it is 4.7%
  of decode GPU time across ~160 launches per step. Since the gemm re-reads A
  once per n-block either way, building A in the prologue looked free: same
  traffic, minus a launch and a global round trip. Implemented with BK=128, so a
  pipeline stage is exactly one Hadamard block, writing straight into the swizzled
  shared layout (`had128_warp_in_swz`). Correct to 8.4e-5 across 54 shape/dtype
  combinations, and much slower: `q_proj` m=16 24.8 -> 33.0 us, m=32 33.0 -> 63.7,
  `up_proj` m=32 41.2 -> 83.5. The traffic argument was right and irrelevant. What
  it missed is that the A copy is a `cp.async` that overlaps with the previous
  stage's mma, while the fused version is a synchronous chain of warp shuffles
  sitting on the pipeline's critical path -- and it runs once per n-block (96x for
  `q_proj`) instead of once per layer. A separate kernel does the transform once
  and lets `cp.async` hide the read; that is worth more than the launch it costs.

At large batch the kernel now runs at **81% of peak MMA throughput**, issuing 6.5
non-MMA instructions per MMA -- 827M ALU+FMA against 178M tensor instructions at
m=4096. That ratio is set by the format: a procedural trellis codebook costs
~74 instructions per 8 weights where an int4 LUT costs 4. Issue slots are only
~33% occupied, so the limit is the math pipes competing with the tensor pipe,
not instruction bandwidth.

One thing tuning *did* still find: the best block-M is shape-dependent, not just
batch-dependent. At m=128, `up_proj` (n=17408) is 16% faster with BM=64 while
`q_proj` (n=12288) prefers BM=128 -- no static rule captures both. The kernel now
times the BM tiers once per distinct shape and caches the winner (skipped during
CUDA graph capture, which cannot tolerate the syncs). That is worth ~7-10% in the
m=128..256 range.

The same argument then applies to the split factor, which `pick_split` picks from
a cost model. Block size and split interact -- a bigger tile means fewer blocks,
so more splitting is needed to fill the machine -- yet the split used to be
computed once from the *heuristic* BM and then paired with whatever BM the tuner
chose. Recomputing it per candidate, and letting the tuner also try one step
either side of the model's answer (`base/2`, `base`, `base*2`), is worth a
further 5-14% in the same band, with no shape regressing:

| shape | m=32 | m=128 | m=256 | m=512 |
|---|---|---|---|---|
| `q_proj` | 35.0 -> 33.0 | 95.1 -> 88.2 | 166.1 -> 152.2 | -- |
| `up_proj` | 41.1 -> 39.3 | 113.5 -> 106.7 | 184.7 -> 168.8 | -- |
| `down_proj` | 39.1 -> 37.1 | 114.9 -> 98.4 | 198.1 -> 179.2 | 371.3 -> 339.4 |

The cost model is still what graph capture falls back on, so the tuner's answer
has to be cached during eager warm-up to reach captured decode.

Beating it further needs a different structure, not tuning. The honest options
are: decode the trellis into shared once per block and stream several M tiles
past it (needs the accumulators for those tiles to live somewhere, which is the
same register wall fp16 accumulation hit); or low-precision tensor cores, which
as measured above would need a codebook designed to decode into fp8/nvfp4 -- a
format change, not a kernel change.
