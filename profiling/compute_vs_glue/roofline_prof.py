#!/usr/bin/env python3
# Per-layer TRT profiler -> compute (matmul/conv) vs non-compute (norm/elementwise/layout) breakdown.
# Usage: roofline_prof.py <engine> <stage_name> [reps]
import sys, numpy as np, tensorrt as trt, torch, collections

eng_path=sys.argv[1]; stage=sys.argv[2] if len(sys.argv)>2 else "stage"; REPS=int(sys.argv[3]) if len(sys.argv)>3 else 50
L=trt.Logger(trt.Logger.ERROR)
eng=trt.Runtime(L).deserialize_cuda_engine(open(eng_path,"rb").read())
ctx=eng.create_execution_context()
ten={}
for i in range(eng.num_io_tensors):
    nm=eng.get_tensor_name(i); sh=tuple(ctx.get_tensor_shape(nm))
    sh=tuple(1 if d<0 else d for d in sh)
    if any(d<0 for d in eng.get_tensor_shape(nm)): ctx.set_input_shape(nm, sh)
    dt=trt.nptype(eng.get_tensor_dtype(nm))
    t=torch.zeros(sh, dtype=getattr(torch,np.dtype(dt).name), device="cuda")
    ten[nm]=t; ctx.set_tensor_address(nm, t.data_ptr())

class Prof(trt.IProfiler):
    def __init__(s): trt.IProfiler.__init__(s); s.d=collections.OrderedDict()
    def report_layer_time(s,n,ms): s.d[n]=s.d.get(n,0.0)+ms

st=torch.cuda.Stream()
for _ in range(10): ctx.execute_async_v3(st.cuda_stream)
torch.cuda.synchronize()
prof=Prof(); ctx.profiler=prof
for _ in range(REPS):
    ctx.execute_async_v3(st.cuda_stream); torch.cuda.synchronize()

def cls(n):
    s=n.lower()
    if any(k in s for k in ["gemm","matmul","conv","einsum","fc_","_mm","dense","myelin","foreign","linear","bmm","qkv"]): return "COMPUTE(matmul/conv)"
    if "layernorm" in s or "rmsnorm" in s or ("norm" in s and "reformat" not in s): return "LayerNorm"
    if "softmax" in s: return "Softmax"
    if any(k in s for k in ["gelu","silu","sigmoid","swish","add","mul","sub","elementwise","pointwise","pwn","bias","act","residual"]): return "elementwise/act"
    if any(k in s for k in ["reshape","transpose","shuffle","concat","slice","copy","gather","permute","reformat","cast","pad","constant"]): return "layout/copy"
    return "other"
cat=collections.Counter();
for n,ms in prof.d.items(): cat[cls(n)]+=ms
tot=sum(cat.values()) or 1e-9
print(f"=== {stage} ({eng_path.split('/')[-1]})  total={tot/REPS:.3f} ms/inf  {len(prof.d)} layers ===")
for k,v in cat.most_common(): print(f"  {k:22s} {100*v/tot:5.1f}%  {v/REPS:.3f} ms")
comp=cat["COMPUTE(matmul/conv)"]
print(f"  >>> COMPUTE {100*comp/tot:.1f}%  |  NON-COMPUTE {100*(tot-comp)/tot:.1f}%")
# top 8 layers for sanity (detect myelin-blob domination)
print("  top layers:")
for n,ms in sorted(prof.d.items(),key=lambda x:-x[1])[:8]: print(f"    {ms/REPS:7.3f} ms  {cls(n):22s} {n[:70]}")
