#!/usr/bin/env python3
"""Does the ViT->LLM->4xDiT chain cost more than the sum of its engines?
Latency question only; values are not chained (shapes are correct). No accuracy claim."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import sys, time, json
import numpy as np, tensorrt as trt, torch
sys.path.insert(0,_W+'/trtwork5090')
from trt_groot import build, load_inputs
L=trt.Logger(trt.Logger.ERROR)
ctxs={}; keep={}
for stage in ["vit","llm","dit"]:
    blob,_=build(stage, True, 5, L)
    eng=trt.Runtime(L).deserialize_cuda_engine(blob); ctx=eng.create_execution_context()
    host=load_inputs(stage,eng,ctx); dev={}
    for i in range(eng.num_io_tensors):
        nm=eng.get_tensor_name(i); dt=trt.nptype(eng.get_tensor_dtype(nm))
        t=torch.empty(tuple(ctx.get_tensor_shape(nm)),dtype=getattr(torch,np.dtype(dt).name),device="cuda")
        if eng.get_tensor_mode(nm)==trt.TensorIOMode.INPUT:
            t.copy_(torch.from_numpy(np.ascontiguousarray(host[nm])).cuda())
        dev[nm]=t; ctx.set_tensor_address(nm,int(t.data_ptr()))
    ctxs[stage]=ctx; keep[stage]=(eng,dev)
st=torch.cuda.Stream()
order=[("vit",1),("llm",1),("dit",4)]
with torch.cuda.stream(st):
    for _ in range(30):
        for k,n in order:
            for _ in range(n): ctxs[k].execute_async_v3(st.cuda_stream)
torch.cuda.synchronize()
alone={}
for k,n in order:
    evs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(50)]
    with torch.cuda.stream(st):
        for a,b in evs:
            a.record(st)
            for _ in range(n): ctxs[k].execute_async_v3(st.cuda_stream)
            b.record(st)
    torch.cuda.synchronize()
    alone[k]=sorted(a.elapsed_time(b) for a,b in evs)[25]
evs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(50)]
with torch.cuda.stream(st):
    for a,b in evs:
        a.record(st)
        for k,n in order:
            for _ in range(n): ctxs[k].execute_async_v3(st.cuda_stream)
        b.record(st)
torch.cuda.synchronize()
seq=sorted(a.elapsed_time(b) for a,b in evs)[25]
sop=sum(alone.values())
out=dict(per_engine_ms={k:round(v,4) for k,v in alone.items()},
         sum_of_parts_ms=round(sop,4), sequential_chain_ms=round(seq,4),
         handoff_cost_ms=round(seq-sop,4), handoff_pct=round(100*(seq-sop)/seq,2),
         note="latency/handoff only; values not chained between engines; shapes correct")
print(json.dumps(out,indent=1))
json.dump(out,open(_W+"/trtwork5090/r_5090_chaincost.json","w"),indent=1)
