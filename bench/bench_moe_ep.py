"""MoE cost under expert parallel vs tensor parallel, and the block_m landscape.

Both arrangements do the same global work and differ only in how it is cut:

  TP rank: every expert, every routed pair, 1/intermediate_tp of the columns.
  EP rank: 1/tp of the experts, the pairs routed to them, the full column width.

So the per-rank figures below are directly comparable, and because this runs on
one GPU it excludes collectives and inter-rank imbalance by construction -- which
is the point. If EP/TP is around 1.0 here but the served throughput is not, the
cost is between the ranks, not inside this kernel.

One caveat on the EP rows: which experts a rank owns is a draw. At small M the
spread across ranks is large (with top-8 over 4 ranks the mean rank owns 2 but
the step waits for the one that drew 3 or 4), so --seed changes the answer and a
single row is a sample, not the mean. --ranks sweeps the draw instead.

  python bench/bench_moe_ep.py --experts 288 --tp 3 --m 1,16,64,2048
  python bench/bench_moe_ep.py --tp 3 --mode blockm --m 8,64,512,2048
"""
import argparse, statistics, sys, torch

try:
    import cuda_exl3.ops as _ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
except ImportError as e:                                    # pragma: no cover
    sys.exit(f"needs cuda_exl3 and vllm: {e}")

BITS_DEFAULT, CB_DEFAULT = 4, 1


def _tiers(per_expert):
    if per_expert < 16: return 16
    if per_expert < 48: return 32
    return 64 if per_expert < 96 else 128


def _time(fn, reps):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(b.elapsed_time(e))
    return statistics.median(ts)


def build(a, M, ep, rank, block_m=None):
    """One rank's operands. Returns (step_fn, block_m, live_blocks, blocks)."""
    g = torch.ops.cuda_exl3_C
    dev = "cuda"
    e_local = a.experts // a.tp if ep else a.experts
    iw      = a.inter if ep else a.inter // a.tp
    topk_ids = torch.randint(0, a.experts, (M, a.topk), dtype=torch.int32, device=dev)

    emap = None
    if ep:
        emap = torch.full((a.experts,), -1, dtype=torch.int32, device=dev)
        lo = rank * e_local
        emap[lo:lo + e_local] = torch.arange(e_local, dtype=torch.int32, device=dev)

    # The block size wants the global count: only E_local/E_global of the pairs
    # are this rank's, so the local count overstates occupancy by the EP factor.
    bm = block_m or _tiers(M * a.topk / (a.experts if ep else e_local))
    sid, eid, nr = moe_align_block_size(topk_ids, bm, a.experts if ep else e_local,
                                        expert_map=emap, pad_sorted_ids=True)
    eid = eid.int(); nr = nr.int()
    rows = min(eid.numel() * bm, sid.numel())
    eid = eid[: rows // bm]

    H, I_, B, C = a.hidden, iw, a.bits, a.cb
    t13 = torch.randint(-32768, 32767, (e_local, H // 16, 2 * I_ // 16, 16 * B),
                        dtype=torch.int16, device=dev)
    s13 = (torch.randn((e_local, 2 * I_), device=dev) * 0.05).half()
    a13 = torch.randn((2, rows, H), dtype=torch.half, device=dev) * 0.05
    t2  = torch.randint(-32768, 32767, (e_local, I_ // 16, H // 16, 16 * B),
                        dtype=torch.int16, device=dev)
    s2  = (torch.randn((e_local, H), device=dev) * 0.05).half()
    a2  = torch.randn((1, rows, I_), dtype=torch.half, device=dev) * 0.05

    def step():
        g.exl3_moe_gemm(a13, t13, s13, s13, eid, nr, [I_, I_], C, bm, torch.bfloat16)
        g.exl3_moe_gemm(a2,  t2,  s2,  s2,  eid, nr, [H],      C, bm, torch.bfloat16)

    return step, bm, int((eid >= 0).sum()), eid.numel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=288)
    p.add_argument("--tp", type=int, default=4, help="ranks; EP owns experts/tp each")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=3072, help="per-expert, unsharded")
    p.add_argument("--layers", type=int, default=92, help="only scales the ms/step column")
    p.add_argument("--m", default="1,16,64,2048")
    p.add_argument("--bits", type=int, default=BITS_DEFAULT)
    p.add_argument("--cb", type=int, default=CB_DEFAULT)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ranks", type=int, default=1,
                   help="EP: time this many different owning ranks and report the spread")
    p.add_argument("--mode", choices=["compare", "blockm"], default="compare")
    a = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs CUDA")
    _ops._try_native()
    ms_list = [int(x) for x in a.m.split(",")]

    if a.mode == "blockm":
        print(f"block_m landscape, {a.experts} experts / {a.experts // a.tp} local, "
              f"top-{a.topk}  (ms per layer, both projections)")
        print(f"{'M':>6} {'arr':>4} " + " ".join(f"{'bm='+str(b):>9}" for b in (16, 32, 64, 128)) + "   best")
        for M in ms_list:
            for ep in (False, True):
                row = {}
                for bm in (16, 32, 64, 128):
                    torch.manual_seed(a.seed)
                    try:
                        step, *_ = build(a, M, ep, 0, block_m=bm)
                        row[bm] = _time(step, a.reps)
                    except Exception:
                        row[bm] = float("nan")
                best = min((v, k) for k, v in row.items() if v == v)[1]
                print(f"{M:6d} {'EP' if ep else 'TP':>4} "
                      + " ".join(f"{row[b]:9.4f}" for b in (16, 32, 64, 128))
                      + f"   bm={best}")
        return

    print(f"{a.experts} experts / {a.experts // a.tp} local at TP={a.tp}, top-{a.topk}, "
          f"hidden {a.hidden}, inter {a.inter}, {a.bits}-bit")
    # The TP arrangement only exists when the intermediate slices 128-aligned.
    # It usually does not at TP=3, which is the whole reason EP is on the table:
    # 2048/3 is not an integer, let alone a multiple of 128. Say so and carry on
    # with the EP rows rather than dying on the reference.
    sliceable = a.inter % (128 * a.tp) == 0
    print(f"{'M':>6} {'arr':>4} {'bm':>4} {'live/blk':>11} {'ms/layer':>9} "
          f"{'ms/step':>9}   EP/TP")
    if not sliceable:
        print(f"  (no TP row: intermediate {a.inter} does not slice {a.tp} ways on a "
              f"128 boundary -- {a.inter / a.tp:.2f} per rank. Pad to "
              f"{((a.inter + 128 * a.tp - 1) // (128 * a.tp)) * 128 * a.tp} to get one.)")
    for M in ms_list:
        tp_ms = None
        if sliceable:
            torch.manual_seed(a.seed)
            step, bm, live, blk = build(a, M, False, 0)
            tp_ms = _time(step, a.reps)
            print(f"{M:6d} {'TP':>4} {bm:4d} {live:5d}/{blk:<5d} {tp_ms:9.4f} "
                  f"{tp_ms * a.layers:9.2f}")
        eps = []
        for r in range(a.ranks):
            torch.manual_seed(a.seed)
            step, bm, live, blk = build(a, M, True, r)
            ms = _time(step, a.reps)
            eps.append((ms, live, blk, bm))
        for ms, live, blk, bm in eps:
            rel = f"   {ms / tp_ms:.2f}x" if tp_ms else ""
            print(f"{'':>6} {'EP' if sliceable else 'EP*':>4} {bm:4d} {live:5d}/{blk:<5d} "
                  f"{ms:9.4f} {ms * a.layers:9.2f}{rel}")
        if a.ranks > 1:
            lo, hi = min(e[0] for e in eps), max(e[0] for e in eps)
            print(f"{'':>6} {'':>4} {'':>4} {'spread over ranks':>11} "
                  f"{lo:9.4f}..{hi:.4f}   {hi / lo:.2f}x imbalance")


if __name__ == "__main__":
    main()
