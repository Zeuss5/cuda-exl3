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
not given. Worth +2.0 to +2.2% of decode throughput and -1.2 to -2.6% of TTFT
here, and +1.6 to +4.5% on GB10, where the MoE is a larger share of a step.

### Do not build the GEMV rewrite

Measured after f4987cf, TP=4 serving shapes, the gate/up gemm against a plain
gather of the same experts' weights:

    M    live experts   w13 MB   gemm    achieved   gather ceiling   of ceiling
    8              60    125.8   84.7us  1523 GB/s      1422 GB/s        107%
    16            114    239.1  183.2us  1339           1451              92%
    64            252    528.5  424.8us  1287           1460              88%
    256           287    601.9  477.0us  1360           1460              93%

The decode path is at the memory system's limit -- over it at M=8, where 125.8 MB
of weights fits the 128 MB L2 and the column-block re-reads hit cache. A GEMV
would be competing for 7-12%, part of which is the A and output traffic charged
against it rather than inefficiency, in exchange for reverse-engineering the mma
fragment layout inside the trellis tile. Not worth it on this part.

That also means MoE decode is finished as a kernel target here. What is left in a
decode step is attention and the collectives; what is left in prefill is the
all-reduce, at 38% of the budget, which is NCCL over host memory because these
cards have no peer-to-peer. Decode has
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

## What is left in the MoE stage, and its ceiling

`exl3_moe_had_in` was the last sub-roofline kernel in the stage (#5): 37-57% of
achievable bandwidth on GB10, 57% here. Removing a 64-bit divide from its index
math (a47da6e) took it to 63%, worth 10-18%.

The rest is not bandwidth. Splitting the kernel against the same gather-and-write
traffic with no transform at all:

    M     live rows   had_in   gather only   transform   had_in GB/s   floor GB/s
    64          512   20.4us        12.3us       8.1us           412          684
    256        2048   40.1us        17.8us      22.3us           836         1886
    1024       8192  127.9us        86.9us      41.0us          1049         1545

The 128-point Hadamard is 32-56% of the kernel, so it is about half memory and
half ALU, and the transform is five rounds of `__shfl_xor` over 32 lanes holding
four elements each -- already close to minimal for a 128-point butterfly in that
layout.

That bounds the opportunity rather than opening one. A *free* transform would
take M=256 from 40.1 to 17.8 us, and this kernel is 4% of a prefill chunk, so the
ceiling on any further work here is about 2% of prefill and it is not reachable.
Recorded so the next person does not rediscover the 57% figure and assume it is
traffic to be removed.

## The all-reduce, and why a custom one does not pay here

`#5` ranks NCCL all-reduce first among prefill targets on GB10: 16.5% of the
chunk at 13.9 GB/s of bus bandwidth against 50 GB/s of physical pair capacity.
tpurtell's recipe ships a matching answer -- a B12x PCIe one-shot all-reduce
behind `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE`, default **384 KB**, used only
inside a captured graph and falling through to PyNCCL above the cutoff. The
kernels are closed, but the shape of the adapter is itself the finding: a small
cutoff and a graph-only gate is what you build when the win is latency on small
messages, not bandwidth on large ones.

Measured here on 4x RTX PRO 6000 (`bench/bench_allreduce.py`), where
`nvidia-smi topo -p2p r` reports NS for every pair, so vLLM's own
`CustomAllreduce` is off and every collective is host-staged:

    bytes        eager us   graph us   algBW    busBW
    8192  (M=1)      13.2       13.6    0.6G     0.9G
    65536 (M=8)      20.6       20.8    3.2G     4.7G
    524288           61.1       61.3    8.6G    12.9G
    67108864       3171.3     3170.2   21.2G    31.8G

Neither `NCCL_ALGO` nor `NCCL_PROTO` improves on the default at any of these
sizes -- the auto-selected protocol is already the best of Ring/Tree x LL/LL128,
and forcing LL costs 1.9x at 512 KB.

Then the two rulers that decide whether a replacement could do better:

* PCIe copy engine, one GPU: **56.5 GB/s** each way.
* Kernel-driven load/store to mapped host memory (`uint4`, best of 16-1024
  blocks): 15.5 GB/s at 64 KB, 31.5 at 256 KB, 46.2 at 1 MB, 51.8 at 16 MB --
  a fixed floor of about **3.5 us** per touch.
* **GPU-to-GPU flag visibility through pinned host memory: 6.74 us one way,
  13.49 us round trip** (single-thread spin on a `volatile` word, 2000 trips).

That last number closes the item. A one-shot all-reduce at the decode shape
(M=8, 64 KB) has to write its slot, make the write visible to three peers, and
read theirs: 3.6 us + 6.74 us + 8.3 us is already **~15 us against NCCL's 20.8**,
and that is the floor of an implementation that does not exist yet. At 8 KB
NCCL's entire call is 13.6 us -- roughly twice the one-way visibility latency of
the fabric, which is what any algorithm requiring every rank to see every other
rank's contribution must pay.

So on a no-P2P PCIe box the all-reduce is *latency*-bound at decode sizes and at
the wire for large ones (31.8 GB/s of bus over a 56.5 GB/s link, with every byte
crossing PCIe twice). NCCL is within about 1.4x of the floor at the size that
matters and there is no 2-3x sitting there. The lever is fewer or larger
all-reduces, or overlapping them with compute -- not a faster collective kernel.

Credit: the cutoff-and-graph-gate design that prompted this measurement is
tpurtell's, in `patches/b12x_pcie_all_reduce.py` of
`tpurtell/glm-5.3-flash-ext3-2x-rtx` (Apache-2.0).

## MLA at prefill: the ceiling is the kernel's own work, not the gather

`#5` ranks MLA prefill sixth at 8.2% of a chunk and marks it **not measured** --
"the trace does not carry the selected-key count". That is the whole difficulty:
the kernel gathers `topk` latent rows per query row, and whether the traffic is
`rows x topk` or the much smaller union of those selections depends on how much
consecutive rows overlap.

`bench/bench_mla_prefill.py` settles it the way arm C settled the MoE question --
same shape twice, the only difference being overlap. `head_dim 576`, 16 heads
(64 at TP=4), `topk 2048`, 262 144 rows of latent (302 MB, not L2-resident on
either part), ruler in the same binary at **1522 GB/s**:

    rows    selection        us  per-row GB/s  %ruler
     256  independent     498.5          1212     80%
     256     drifting     316.9          1906    125%
    2048  independent    3991.6          1211     80%
    2048     drifting    2748.9          1758    116%

The drifting arm reads **125% of the ruler** on the per-row model, which is only
possible if rows are being served from cache -- so the kernel does exploit
overlap, and it is 1.45x faster when the overlap is there.

The useful part is the decomposition. The drifting arm touches a few MB, so it
is entirely resident and its time is the kernel's compute-and-issue floor with
the traffic taken away; the independent arm has to move `rows x topk x D`:

    rows  compute us    hbm us  actual us  vs the larger
     256         317       397        498          1.26x
     512         657       794        970          1.22x
    1024        1302      1588       1920          1.21x
    2048        2749      3175       3992          1.26x

Two things follow, and the second matters more.

**The kernel runs at 1.21-1.26x the larger of its two floors**, consistently
across a 8x range of rows. That is imperfect overlap of traffic against work,
and closing it is worth 21-26% of MLA prefill, i.e. about 1.7-2% of a prefill
chunk on `#5`'s breakdown.

**The compute floor is 87% of the traffic floor** (2749 vs 3175 us). So at
prefill this kernel is not gather-bound the way it is at decode -- the two are
nearly balanced, and a production selection pattern that overlaps heavily pushes
it over into being bound by its own work. DSA top-k moves about one row in 2048
per step, so production is nearer the drifting arm than the independent one,
which would make MLA prefill **compute-bound**. That is the opposite of the
assumption a traffic-only reading of the trace would produce.

What is not measured: where production actually sits between the two arms. The
selected-key count `#5` says the trace does not carry is exactly the datum that
would place it. Until then the honest statement is that the ceiling is bounded
by 2749 us and the floor by 3175 us at 2048 rows, and the kernel is at 3992.
