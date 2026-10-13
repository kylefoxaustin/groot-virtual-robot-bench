#!/usr/bin/env python3
"""Is '4 x DiT_step' a fair accounting? Time 1 step vs 4 back-to-back steps in one stream."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import sys, time
import numpy as np, tensorrt as trt, torch
sys.path.insert(0,_W+'/trtwork5090')
from trt_groot import build, load_inputs
L=trt.Logger(trt.Logger.ERROR)
blob,_=build("dit", True, 5, L)
eng=trt.Runtime(L).deserialize_cuda_engine(blob); ctx=eng.create_execution_context()
host=load_inputs("dit",eng,ctx); dev={}
for i in range(eng.num_io_tensors):
    nm=eng.get_tensor_name(i); dt=trt.nptype(eng.get_tensor_dtype(nm))
    t=torch.empty(tuple(ctx.get_tensor_shape(nm)),dtype=getattr(torch,np.dtype(dt).name),device="cuda")
    if eng.get_tensor_mode(nm)==trt.TensorIOMode.INPUT:
        t.copy_(torch.from_numpy(np.ascontiguousarray(host[nm])).cuda())
    dev[nm]=t; ctx.set_tensor_address(nm,int(t.data_ptr()))
st=torch.cuda.Stream()
with torch.cuda.stream(st):
    for _ in range(300): ctx.execute_async_v3(st.cuda_stream)
torch.cuda.synchronize()
for n in (1,4):
    evs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(50)]
    with torch.cuda.stream(st):
        for a,b in evs:
            a.record(st)
            for _ in range(n): ctx.execute_async_v3(st.cuda_stream)
            b.record(st)
    torch.cuda.synchronize()
    ms=sorted(a.elapsed_time(b) for a,b in evs)[25]
    print(f"{n} step(s): median {ms:.4f} ms   per-step {ms/n:.4f} ms")
    if n==1: one=ms
    else: print(f"  -> 4x_single = {4*one:.4f} ms ; measured_4 = {ms:.4f} ms ; delta {100*(ms-4*one)/(4*one):+.2f}%")
