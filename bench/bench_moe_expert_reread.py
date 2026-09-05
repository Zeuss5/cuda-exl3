"""Does the MoE trellis get re-read once per block, or stay resident across an
expert's blocks?

Arm C of the bench specified by @NNNtrance in issue #5. If time scales with
blocks the trellis is re-read and an expert-stationary schedule is worth 14-27%
of the large-M weight traffic; if it scales with rows the weights already stay
resident and there is nothing to win.

96 distinct local experts held fixed; 1..4 blocks per expert at block_m=16.
Weight bytes are 96 x 8.389 MB if resident, 96*N x 8.389 MB if re-read.
"""
import statistics, torch, cuda_exl3.ops as _ops
_ops._try_native(); g=torch.ops.cuda_exl3_C
dev,EL,H,I,BITS,CB,BM=("cuda",96,4096,2048,4,1,16)
props=torch.cuda.get_device_properties(0)

# ruler, same binary and run
buf=torch.empty(4*1024**3//2, dtype=torch.bfloat16, device=dev)
def t(fn,r=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(r):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)
ms=t(lambda: buf.sum())
ruler=buf.numel()*2/(ms/1e3)/1e9
del buf; torch.cuda.empty_cache()
print(f"{props.name}, {props.multi_processor_count} SMs, L2 {props.L2_cache_size/2**20:.0f} MiB")
print(f"ruler: bf16 sum over 4 GiB = {ruler:.0f} GB/s\n")

torch.manual_seed(0)
tr=torch.randint(-32768,32767,(EL,H//16,2*I//16,16*BITS),dtype=torch.int16,device=dev)
sv=(torch.randn((EL,2*I),device=dev)*0.05).half()
su=(torch.randn((EL,2,H),device=dev)*0.05).half()
w_bytes=tr[0].numel()*2
print(f"per expert w13 = {w_bytes/1e6:.3f} MB   (fits L2: {w_bytes < props.L2_cache_size})")
print(f"{'N':>2} {'blocks':>7} {'rows':>7} {'us':>9} {'GB/s per-expert':>16} {'GB/s per-block':>15} {'vs ruler':>9}")
base=None
for N in (1,2,3,4):
    blocks=EL*N
    eids=torch.arange(EL,device=dev,dtype=torch.int32).repeat_interleave(N).contiguous()
    rows=blocks*BM
    nr=torch.tensor([rows],dtype=torch.int32,device=dev)
    a13=torch.randn((2,rows,H),dtype=torch.half,device=dev)*0.05
    us=t(lambda: g.exl3_moe_gemm(a13,tr,su,sv,eids,nr,[I,I],CB,BM,torch.bfloat16))*1000
    lo=EL*w_bytes; hi=blocks*w_bytes
    print(f"{N:2d} {blocks:7d} {rows:7d} {us:9.1f} {lo/(us/1e6)/1e9:16.0f} {hi/(us/1e6)/1e9:15.0f}"
          f" {hi/(us/1e6)/1e9/ruler*100:8.0f}%")
    if base is None: base=us
    else: print(f"   {'':>7} {'':>7} time x{us/base:.2f} vs blocks x{N:.0f} "
                f"-> {'RE-READ per block' if us/base > N*0.75 else 'RESIDENT across blocks'}")
