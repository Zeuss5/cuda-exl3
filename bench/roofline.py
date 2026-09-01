import torch, time
torch.cuda.init()
dev=0; props=torch.cuda.get_device_properties(dev)
print(f"{props.name}  SMs={props.multi_processor_count}  cc={props.major}.{props.minor}")
print(f"  total mem {props.total_memory/2**30:.1f} GiB")
def bench(fn,it,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it

# --- HBM bandwidth: big streaming copy ---
N=1<<30  # 1 GiB of half = 512M elems
a=torch.empty(N//2, dtype=torch.half, device=dev)
b=torch.empty_like(a)
a.normal_()
t=bench(lambda: b.copy_(a), 30)
bw = 2*a.numel()*2/t/1e9   # read+write
print(f"\nHBM copy bandwidth : {bw:8.1f} GB/s   ({t*1e3:.3f} ms for {a.numel()*2/2**30:.2f} GiB r+w)")
# read-only bandwidth via sum
t=bench(lambda: a.sum(), 30)
print(f"HBM read bandwidth : {a.numel()*2/t/1e9:8.1f} GB/s")

# --- fp16 tensor core peak ---
for (M,K,N_) in [(8192,8192,8192),(16384,16384,16384)]:
    x=torch.randn(M,K,dtype=torch.half,device=dev); w=torch.randn(K,N_,dtype=torch.half,device=dev)
    y=torch.empty(M,N_,dtype=torch.half,device=dev)
    t=bench(lambda: torch.mm(x,w,out=y), 20, 5)
    print(f"fp16 GEMM {M}x{K}x{N_}: {2*M*K*N_/t/1e12:8.1f} TFLOPS")
    del x,w,y; torch.cuda.empty_cache()
