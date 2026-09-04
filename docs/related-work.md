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

### Measured before building it

**Cuda-graph capture: fine.** A `grid.sync()` kernel stream-captures and replays
correctly here, so the design is usable where it would help. That was the risk
that could have killed it outright.

**The cooperative packaging is a loss on this card, and should not be copied.**
Three phases over 42 layers, a grid-synced kernel against three ordinary
launches:

  elements     3 kernels    1 cooperative    per boundary
    16384       0.324 ms       0.512 ms        -1.5 us
   262144       0.303          0.518           -1.7 us
  4194304       1.034          1.354           -2.5 us

A grid barrier costs 1.5-2.5 us *more* than a kernel boundary. Launches are
cheap, especially inside a graph, while a grid barrier serialises on the slowest
of ~1500 resident blocks and pins the grid to `resident x SMs`.

This also explains a measurement that had puzzled us: the non-GEMM kernels cost
1.17 ms per step and cuda graphs did not recover it. That was never boundary
overhead -- it is the work of writing and reading the intermediates, which a
cooperative kernel does not avoid either, since its phases communicate through
global memory (work items are per expert and group, so phase 3 needs all of
phase 2).

**And it is not right on GB10 either, once the graph is accounted for.** The
block-count argument does hold: measured on 48 SMs (#1), a cooperative grid is
288 blocks against ~1128 here, and *uncaptured* the cooperative arm wins there by
up to 1.4 us per boundary -- the opposite sign to this card.

But both of those measurements were taken outside a CUDA graph, and vLLM captures
decode. Capture removes 59% of the separate-launch arm's overhead and almost none
of the cooperative arm's, and the sign flips back: +0.23 to +0.32 us per boundary
on 48 SMs in a graph, equal at bandwidth-bound sizes.

So the honest rule is not about SM count at all. **Inside a CUDA graph a kernel
boundary is cheap enough that a grid barrier is never worth paying for**, on
either part. The 1.73x is presumably real uncaptured; it does not survive
capture, which is the only way decode runs.

**So the transferable idea is the GEMV formulation, not the fusion.** Taken in
f4987cf, in the cheapest form that captures it: rather than restructuring into a
GEMV, just stop fetching the padding rows, since cp.async zero-fills a row it is
not given. Worth +2.0 to +2.2% of decode throughput and -1.2 to -2.6% of TTFT end
to end. Decode has
about one row per expert against our 16-row `mma` tile, and the padding is
33.6 MB per layer of pure traffic against 314.6 MB of weights. Removing it is
worth roughly 9%, and the access pattern will support it: a gather of 50
scattered experts runs at 1438 GB/s against a 1451 GB/s contiguous copy, so the
memory system delivers 98.5% of stream for this pattern and our own 1326 GB/s is
kernel-side.

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
