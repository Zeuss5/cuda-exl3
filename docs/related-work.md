# Related work, and what is worth taking from it

Two other projects run EXL3 GLM-5.3-Flash on DGX Sparks. Both are MIT licensed
and share lineage: `vcruz305/vllm-exl3` credits Mia's AI Lab for its fat GEMM
and `exl3.py` derivations, and both credit turboderp's ExLlamaV3 for the format
and the reference kernels.

* `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` -- a deployment recipe.
* `vcruz305/vllm-exl3` -- an out-of-tree vLLM plugin, the same shape as this one.

Nothing here is copied from either; this file records what they do, what is
worth adopting, and what we already had. Their headline speedups are measured
against reconstruct-then-GEMM baselines (dequantising routed experts to dense
before the matmul), which is not what this plugin does either, so those figures
do not compare against it.

## Convergent, and reassuring

Both run one fused MoE launch per layer, shard gate/up column-wise and down
row-wise, all-reduce once, and never reconstruct experts to dense. `vllm-exl3`
also folds inline routing and an atomic token scatter into the down projection,
which is the same design as our fused combine (5814c7f), arrived at
independently on both sides.

## The idea worth pursuing: a cooperative decode kernel

`vcruz305/vllm-exl3`'s `p2b_moe.cu` runs the whole MoE decode as **one
cooperative launch** -- `cudaLaunchCooperativeKernel`, persistent CTAs pulling
from a work queue, `grid.sync()` between four phases:

1. input Hadamard for gate and up,
2. batched GEMV for gate and up,
3. SwiGLU plus the down projection's input Hadamard,
4. batched GEMV for down, output Hadamard, and a routing-weighted `atomicAdd`
   into the token's row.

Reported at 497 -> 287.8 us per layer.

Two things make this interesting for us specifically, and they are the two
conclusions our own fusion work arrived at the hard way:

**It is GEMV, not GEMM.** In MoE decode the rows per expert is about one, and
stays about one until `batch x top_k` greatly exceeds the expert count -- for
GLM's 288 experts at top-8 that is M >= 576 before our 16-row `mma` tile is even
full. So the tensor cores are mostly idle and the tile's padded rows are pure
traffic: measured here as 33.6 MB per layer of `a13` against 314.6 MB of
weights, about 10%. A GEMV streams weights and does FMA, with no tile to pad.

**One launch removes the inter-kernel stalls.** Measured here, the non-GEMM
kernels cost 1.17 ms per step and CUDA graph capture does not recover it -- it
is real GPU time, not launch overhead. A grid sync between phases is cheaper
than a kernel boundary.

It also answers the thing that killed our fused input transform (a8270f9,
reverted in 76598b2): fusing the transform into the GEMM re-ran it once per
column tile, 16 times over. A phase-separated cooperative kernel computes it
once, syncs, and every later phase reads it.

Risks to settle before building it: cooperative launches interact badly with
stream capture in some drivers and vLLM captures decode into a CUDA graph; the
grid must be resident, which caps occupancy; and it only pays where rows per
expert is small, so it is an added decode path, not a replacement.

## Recipe-level items, none of which are kernel work

* **Right-size the sparse-indexer prefill workspace.** vLLM defaults it to
  `max_model_len * 40` entries, about 5 GB locked at 1M context. Computing the
  per-step legal maximum instead reportedly recovers ~26% of the KV pool. This
  is the largest single practical win on the list and it is configuration.
* **`fp8_ds_mla` packed KV** at 656 B/token/layer, against our plain fp8.
* **Serialise prefill against decode** so a peer's prefill does not evict the
  decode working set. Consistent with the L2 behaviour measured in the MLA
  autotuner notes.
* **DFlash2 draft attention must be bidirectional inside the sliding window.**
  Pinning a causal-in-block backend collapses acceptance at later positions
  (0.31 against 0.959 on structured prompts).
* **Acceptance falls off hard on prose**: ~0.98 -> 0.83 per position on
  structured text, ~0.75 -> 0.06 on prose. Deep drafts are a code-and-math win,
  not a general one, which explains single-stream gaps better than kernel speed
  does.

## Where this plugin is ahead

Both zero-pad GLM's 512-wide latent into the 576-wide GLM_NSA geometry to
satisfy an existing sparse kernel. Ours runs `head_dim = 512` natively and is
1.6-2.4x faster than b12x, which refuses 512 outright.
