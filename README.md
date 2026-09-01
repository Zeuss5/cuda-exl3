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

Three things were tried and rejected against measurement:

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

So the kernel is at the practical ceiling of this design. Beating it further
means a different structure -- e.g. decoding the trellis into shared memory once
per block and feeding several M tiles from it, or the low-precision tensor cores
(fp8 / nvfp4), which would need the dequant to target those formats rather than
fp16.
