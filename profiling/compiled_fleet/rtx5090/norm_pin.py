#!/usr/bin/env python3
"""Trap follow-up: can plain --fp16 + fp32-pinned RMSNorm layers match stronglyTyped's
accuracy on the GR00T Qwen3 LLM, and is it any faster?"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import sys, time, hashlib
import numpy as np, tensorrt as trt, torch, onnx
sys.path.insert(0,_W+'/trtwork5090')
from trt_groot import SHAPES, ONNX, load_inputs
L=trt.Logger(trt.Logger.ERROR)
m=onnx.load(ONNX["llm"], load_external_data=False)
norm=set(n.name for n in m.graph.node if n.op_type in ("Pow","ReduceMean","Sqrt","Sub","Div","Cast"))
print("candidate norm-ish onnx nodes:",len(norm))
def build_pinned():
    b=trt.Builder(L); net=b.create_network(0); p=trt.OnnxParser(net,L)
    assert p.parse_from_file(ONNX["llm"])
    cfg=b.create_builder_config()
    prof=b.create_optimization_profile()
    for nm,shp in SHAPES["llm"].items(): prof.set_shape(nm,shp,shp,shp)
    cfg.add_optimization_profile(prof); cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,16<<30)
    cfg.set_flag(trt.BuilderFlag.FP16); cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    n=0
    for i in range(net.num_layers):
        l=net.get_layer(i)
        if l.name in norm or any(k in l.name for k in ("Pow","ReduceMean","Sqrt","norm","Norm")):
            try: l.precision=trt.float32; n+=1
            except Exception: pass
    print("pinned layers:",n,"of",net.num_layers)
    plan=b.build_serialized_network(net,cfg)
    return bytes(plan) if plan else None
blob=build_pinned()
if blob is None: print("BUILD FAILED"); sys.exit(1)
eng=trt.Runtime(L).deserialize_cuda_engine(blob); ctx=eng.create_execution_context()
insp=eng.create_engine_inspector(); insp.execution_context=ctx
li=insp.get_engine_information(trt.LayerInformationFormat.JSON)
host=load_inputs("llm",eng,ctx); dev={}; outs={}
for i in range(eng.num_io_tensors):
    nm=eng.get_tensor_name(i); dt=trt.nptype(eng.get_tensor_dtype(nm))
    t=torch.empty(tuple(ctx.get_tensor_shape(nm)),dtype=getattr(torch,np.dtype(dt).name),device="cuda")
    if eng.get_tensor_mode(nm)==trt.TensorIOMode.INPUT:
        t.copy_(torch.from_numpy(np.ascontiguousarray(host[nm])).cuda())
    else: outs[nm]=t
    dev[nm]=t; ctx.set_tensor_address(nm,int(t.data_ptr()))
st=torch.cuda.Stream()
with torch.cuda.stream(st):
    for _ in range(250): ctx.execute_async_v3(st.cuda_stream)
torch.cuda.synchronize()
evs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(50)]
with torch.cuda.stream(st):
    for a,b in evs: a.record(st); ctx.execute_async_v3(st.cuda_stream); b.record(st)
torch.cuda.synchronize()
ms=sorted(a.elapsed_time(b) for a,b in evs)[25]
got=list(outs.values())[0].detach().float().cpu().numpy()
ref=np.load(_W+"/trtwork/ref_llm.npy").astype(np.float32).reshape(got.shape)
a=got.ravel().astype(np.float64); b=ref.ravel().astype(np.float64)
cos=a@b/(np.linalg.norm(a)*np.linalg.norm(b)); rel=np.abs(a-b).mean()/np.abs(b).mean()
print(f"fp16+pinnedNorm: md5={hashlib.md5(blob).hexdigest()} median={ms:.4f} ms  nFloat={li.count(chr(34)+'Float'+chr(34))}  cosine={cos:.6f}  rel_mae={rel:.6f}")
print("   compare: stronglyTyped 2.0437 ms cos 0.999823 | plain fp16 2.047 ms cos 0.854230")
