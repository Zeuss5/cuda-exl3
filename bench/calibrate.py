"""Calibrate the device-dependent constants on whatever GPU you are on.

Three things in this plugin are tuned rather than derived, and their defaults
were measured on an RTX PRO 6000 Blackwell (188 SMs, 128 MB L2, 1.46 TB/s). A
GB10 is a different machine -- ~5x fewer SMs, a much smaller L2, a sixth of the
bandwidth -- so the defaults are not obviously right there. The MLA kernel is
unaffected: it autotunes against `cudaEvent` at runtime on whatever device it
finds. These do not.

    python bench/calibrate.py                 # both sweeps, prints exports
    python bench/calibrate.py --only splitk

Needs no checkpoint: shapes are synthetic, and trellis decode is
data-independent so the timings are real.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

CHILD = "--_child"


def device_info():
    import torch
    p = torch.cuda.get_device_properties(0)
    return dict(name=p.name, cc=f"{p.major}.{p.minor}", sms=p.multi_processor_count,
                l2_mb=round(p.L2_cache_size / 2**20, 1))


def child_splitk(shapes, ms):
    import time, torch
    from cuda_exl3 import ops
    total = 0.0
    for k, n, bits in shapes:
        t = torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16),
                          dtype=torch.int16, device="cuda")
        suh = (torch.randn(1, k, device="cuda") * 0.1).half()
        svh = (torch.randn(n, device="cuda") * 0.1).half()
        for m in ms:
            x = (torch.randn(m, k, device="cuda") * 0.1).half()
            f = lambda: ops.exl3_linear(x, t, suh, svh, [n], 1)
            for _ in range(10):
                f()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                f()
            torch.cuda.synchronize()
            total += (time.perf_counter() - t0) / 30 * 1e6
    print(json.dumps({"us": total}))


def child_blockm(bm):
    import time, torch
    from cuda_exl3 import ops  # noqa: F401
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    ops_ = torch.ops.cuda_exl3_C
    M, H, I, E, TOPK = 4096, 4096, 512, 288, 8
    w = torch.randint(-32768, 32767, (E, H // 16, (2 * I) // 16, 64),
                      dtype=torch.int16, device="cuda")
    suh = (torch.randn(E, 2, H, device="cuda") * .1).half()
    svh = (torch.randn(E, 2 * I, device="cuda") * .1).half()
    ids = torch.randint(0, E, (M, TOPK), device="cuda", dtype=torch.int32)
    si, ei, nr = moe_align_block_size(ids, bm, E, pad_sorted_ids=True)
    si, ei, nr = si.int(), ei.int(), nr.int()
    rows = min(ei.numel() * bm, si.numel())
    e = ei[: rows // bm]
    a = torch.empty((2, rows, H), dtype=torch.half, device="cuda")
    f = lambda: ops_.exl3_moe_gemm(a, w, suh, svh, e, nr, [I, I], 1, bm, torch.half)
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        f()
    torch.cuda.synchronize()
    print(json.dumps({"us": (time.perf_counter() - t0) / 20 * 1e6}))


def run_child(env, args):
    e = dict(os.environ); e.update({k: str(v) for k, v in env.items()})
    r = subprocess.run([sys.executable, __file__, CHILD] + args,
                       capture_output=True, text=True, env=e)
    for line in r.stdout.splitlines()[::-1]:
        if line.startswith("{"):
            return json.loads(line)["us"]
    return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["splitk", "blockm"])
    ap.add_argument(CHILD, dest="child", nargs="*")
    a, rest = ap.parse_known_args()

    if a.child is not None:
        if a.child and a.child[0] == "blockm":
            return child_blockm(int(a.child[1]))
        return child_splitk([(4096, 2048, 4), (2048, 4096, 4), (4096, 4096, 4)],
                            [16, 64, 256, 1024])

    info = device_info()
    print(f"device: {info['name']}  cc {info['cc']}  {info['sms']} SMs  "
          f"{info['l2_mb']} MB L2")
    print("defaults were measured on 188 SMs / 128 MB L2 / 1.46 TB/s\n")
    out = {}

    if a.only in (None, "splitk"):
        base = run_child({}, [])
        print(f"split-k sweep (baseline {base:.0f} us for the whole set)")
        best, best_env = base, None
        for tgt, bud, gain in itertools.product((1.5, 3.0, 6.0), (0.15, 0.30, 0.60),
                                                (1.0, 2.0, 4.0)):
            env = {"CUDA_EXL3_SPLIT_TARGET": tgt, "CUDA_EXL3_SPLIT_BUDGET": bud,
                   "CUDA_EXL3_L2_GAIN": gain}
            us = run_child(env, [])
            if us < best:
                best, best_env = us, env
        if best_env and best < base * 0.98:
            print(f"  best {best:.0f} us ({100*(base-best)/base:.1f}% better)")
            out.update(best_env)
        else:
            print("  defaults are within 2% of the best combination -- keep them")

    if a.only in (None, "blockm"):
        print("\nMoE block_m ladder (4096 tokens, top-8 of 288 experts)")
        res = {bm: run_child({}, ["blockm", str(bm)]) for bm in (16, 32, 64, 128)}
        for bm, us in res.items():
            print(f"  block_m {bm:>3}: {us:>9.0f} us")
        best_bm = min(res, key=res.get)
        print(f"  best at this token count: {best_bm}")
        print("  (the shipped heuristic scales block_m with tokens per expert;"
              " override only if this disagrees badly)")

    if out:
        print("\nexport " + " ".join(f"{k}={v}" for k, v in out.items()))
    else:
        print("\nnothing to override on this device.")


if __name__ == "__main__":
    main()
