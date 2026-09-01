import os, sys, time, collections
os.environ.setdefault("VLLM_EXL3_BACKEND", "native")
M = "/home/shadeform/vllm/models/Qwen3.8-27B-EXL3-5.5bpw"

def main():
    import torch
    from torch.profiler import profile, ProfilerActivity
    from vllm import LLM, SamplingParams
    mode = sys.argv[1]                      # prefill | decode
    conc = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    llm = LLM(model=M, dtype="bfloat16", max_model_len=10240,
              max_num_batched_tokens=8192, max_num_seqs=128,
              gpu_memory_utilization=0.85, enforce_eager=False,
              trust_remote_code=True, limit_mm_per_prompt={"image":0,"video":0},
              enable_prefix_caching=False)

    tok = llm.get_tokenizer()
    import random
    random.seed(0)
    def prompt(i):
        ids = [random.randint(1000, 100000) for _ in range(8000)]
        return tok.decode(ids)

    prompts = [prompt(i) for i in range(conc)]
    ntok = 1 if mode == "prefill" else 128
    sp = SamplingParams(temperature=0.0, max_tokens=ntok, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)      # warm
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    n = sum(len(o.outputs[0].token_ids) for o in out)
    print(f"\n### mode={mode} conc={conc} wall={wall:.2f}s out={n}")
    if mode == "prefill":
        print(f"### prefill throughput {conc*8000/wall:.0f} tok/s")
    else:
        print(f"### decode {n/wall:.1f} tok/s")

    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in prof.key_averages():
        if e.device_time_total > 0:
            agg[e.key][0] += e.device_time_total/1e3; agg[e.key][1] += e.count
    tot = sum(v[0] for v in agg.values())
    print(f"### total GPU {tot:.1f} ms\n{'ms':>9} {'%':>6} {'calls':>7}  kernel")
    for k,(ms,c) in sorted(agg.items(), key=lambda kv:-kv[1][0])[:16]:
        print(f"{ms:9.1f} {100*ms/tot:6.1f} {c:7d}  {k[:88]}")

if __name__ == "__main__":
    main()
