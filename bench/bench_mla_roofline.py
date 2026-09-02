"""How close is b12x sparse-MLA decode to the GPU bound at GLM-5.3-Flash's shape?

MLA shares one latent KV row across all query heads, so a decode step reads
topk x head_dim of cache regardless of head count. That read is the roofline:
anything above it is the kernel's own overhead. Also sweeps the two knobs
run_decode already exposes (backend, forced_num_splits) -- autotuning the
existing kernel before considering a new one.
"""
import itertools, time, torch
import b12x.attention.sparse_mla as sm

dev = torch.device("cuda")
HBM = 1.79e12
torch.manual_seed(0)

HEAD_DIM, V_HEAD_DIM, PAGE = 576, 512, 64
TOPK = 2048                      # GLM index_topk
HEADS_TP4 = 64 // 4              # 64 heads, TP=4

def build(batch, heads, seq_len, dtype=torch.bfloat16):
    caps = sm.Caps(device=dev, num_q_heads=heads, max_q_rows=batch,
                   max_width=TOPK, dtype=dtype, kv_dtype=dtype,
                   head_dim=HEAD_DIM, v_head_dim=V_HEAD_DIM, mode="decode",
                   max_batch=batch, max_kv_rows=seq_len * batch,
                   max_page_table_width=TOPK // PAGE + 2,
                   max_chunks_per_row=64)
    plan = sm.plan(caps)
    npages = (seq_len + PAGE - 1) // PAGE * batch + 4
    kv = torch.randn(npages, PAGE, HEAD_DIM, device=dev, dtype=dtype) * 0.05
    q = torch.randn(batch, heads, HEAD_DIM, device=dev, dtype=dtype) * 0.05
    width = TOPK // PAGE
    pt = torch.randint(0, npages, (batch, width), device=dev, dtype=torch.int32)
    sl = torch.full((batch,), min(seq_len, TOPK), device=dev, dtype=torch.int32)
    # the indexer's output: which KV rows this query attends to
    sel = torch.randint(0, min(seq_len, TOPK), (batch, TOPK), device=dev,
                        dtype=torch.int32).sort(dim=-1).values
    scratch = {sp.name: torch.empty(sp.shape, dtype=sp.dtype,
                                    device=sp.device or dev)
               for sp in plan.scratch_specs()}
    binding = plan.bind(scratch=scratch, q=q, selected_indices=sel,
                        cache_seqlens_int32=sl, nsa_cache_seqlens_int32=sl)
    return binding, q, kv, pt, sl

def tm(f, reps=50, inner=20):
    """GPU time only.

    The planned API does real host work per call (validation, scratch
    resolution, Triton dispatch) which swamps a decode kernel measured
    eagerly -- 480 us flat across batch 1..16. Production runs this inside a
    CUDA graph, so capture one and replay it.
    """
    for _ in range(5): f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s_)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(inner): f()
    torch.cuda.synchronize()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
    beg.record()
    for _ in range(reps): g.replay()
    end.record(); torch.cuda.synchronize()
    return beg.elapsed_time(end) / reps / inner * 1000

print(f"supported on this device: {sm.is_supported(dev)}")
print(f"\nGLM sparse-MLA decode, head_dim={HEAD_DIM} topk={TOPK} heads={HEADS_TP4} (TP=4)")
print(f"{'batch':>6s} {'backend':>10s} {'splits':>7s} {'us':>8s} {'GB/s':>8s} {'%roofline':>10s}")
for batch in (1, 4, 16):
    binding, q, kv, pt, sl = build(batch, HEADS_TP4, 4096)
    kv_bytes = batch * TOPK * HEAD_DIM * 2      # one latent row per selected key
    sol = kv_bytes / HBM * 1e6
    best = None
    for backend, splits in itertools.product([None], [None, 1, 2, 4, 8, 16]):
        try:
            # the binding already owns q, selected_indices and the seqlens
            f = lambda: sm.run_decode(kv_cache=kv, binding=binding,
                                      sm_scale=1.0 / (HEAD_DIM ** 0.5),
                                      v_head_dim=V_HEAD_DIM,
                                      backend=backend, forced_num_splits=splits)
            f()
            us = tm(f)
        except Exception as e:
            continue
        tag = (backend or "default", "auto" if splits is None else splits)
        print(f"{batch:>6d} {str(tag[0]):>10s} {str(tag[1]):>7s} {us:>8.1f} "
              f"{kv_bytes/us/1e3:>8.0f} {sol/us*100:>9.1f}%")
        if best is None or us < best[0]: best = (us, tag)
    if best:
        print(f"       -> best {best[1]} at {best[0]:.1f} us "
              f"({sol/best[0]*100:.1f}% of roofline, roofline {sol:.1f} us)\n")
