import torch, itertools
from vllm_exl3 import ops as _ops
_ops._try_native()
dev="cuda"; torch.manual_seed(0); D,DV,HBM=576,512,1.79e12
SMS=torch.cuda.get_device_properties(0).multi_processor_count
def gt(f,reps=50):
    for _ in range(5): f()
    torch.cuda.synchronize()
    st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    torch.cuda.synchronize()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(reps): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)/reps*1000
H,topk,rows=16,2048,8192
kv=torch.randn(rows,D,device=dev,dtype=torch.bfloat16)*0.05
print(f"{'B':>3s} {'chunk':>6s} {'hpb':>4s} {'blocks':>7s} {'us':>7s} {'%roof':>6s}")
for B in (1,4,16):
    q=torch.randn(B,H,D,device=dev,dtype=torch.bfloat16)*0.05
    sel=torch.stack([torch.randperm(rows,device=dev)[:topk] for _ in range(B)]).int()
    sl=torch.full((B,),topk,device=dev,dtype=torch.int32)
    sol=B*topk*D*2/HBM*1e6; best=None
    for chunk,hpb in itertools.product((16,24,32,64,128),(16,8,4,2)):
        f=lambda c=chunk,h=hpb: torch.ops.vllm_exl3_C.mla_decode(q,kv,sel,sl,1.0/(D**0.5),DV,c,h)
        try: us=gt(f)
        except Exception: continue
        blocks=((topk+chunk-1)//chunk)*(H//hpb)*B
        if best is None or us<best[0]: best=(us,chunk,hpb,blocks)
    print(f"{B:>3d} {best[1]:>6d} {best[2]:>4d} {best[3]:>7d} {best[0]:>7.1f} {sol/best[0]*100:>5.1f}%"
          f"   [b12x {'20.5' if B==1 else '21.0' if B==4 else '30.6'}]")
