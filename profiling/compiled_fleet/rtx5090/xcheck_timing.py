#!/usr/bin/env python3
"""Independent check on the timing METRIC itself: cudaEvent-bracketed (the method trt_groot.py uses)
vs wall-clock-with-sync around each engine call (the chain_cost.py method). They should agree."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import sys, time
import numpy as np, tensorrt as trt, torch
sys.path.insert(0,_W+'/trtwork5090')
from trt_groot import build, load_inputs
L=trt.Logger(trt.Logger.ERROR)
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
    st=torch.cuda.Stream()
    with torch.cuda.stream(st):
        for _ in range(200): ctx.execute_async_v3(st.cuda_stream)
    torch.cuda.synchronize()
    evs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(50)]
    with torch.cuda.stream(st):
        for a,b in evs: a.record(st); ctx.execute_async_v3(st.cuda_stream); b.record(st)
    torch.cuda.synchronize()
    ev_ms=sorted(a.elapsed_time(b) for a,b in evs)[25]
    wall=[]
    with torch.cuda.stream(st):
        for _ in range(50):
            torch.cuda.synchronize(); t0=time.perf_counter()
            ctx.execute_async_v3(st.cuda_stream); torch.cuda.synchronize()
            wall.append((time.perf_counter()-t0)*1000)
    w_ms=sorted(wall)[25]
    print(f"{stage}: cudaEvent={ev_ms:.4f} ms   wallclock+sync={w_ms:.4f} ms   delta={100*(w_ms-ev_ms)/ev_ms:+.2f}% (wall includes launch+sync overhead)")
    del ctx,eng,dev; torch.cuda.empty_cache()
