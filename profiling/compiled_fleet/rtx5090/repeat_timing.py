#!/usr/bin/env python3
"""Same engine, N separate timing passes -> true run-to-run spread (vs build-to-build)."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import sys, time, json
import numpy as np, tensorrt as trt, torch
sys.path.insert(0,_W+'/trtwork5090')
from trt_groot import build, load_inputs
L=trt.Logger(trt.Logger.ERROR); out={}
for stage in ["vit","llm","dit"]:
    blob,_=build(stage, True, 5, L)
    eng=trt.Runtime(L).deserialize_cuda_engine(blob); ctx=eng.create_execution_context()
    host=load_inputs(stage,eng,ctx); dev={}
    for i in range(eng.num_io_tensors):
        nm=eng.get_tensor_name(i); dt=trt.nptype(eng.get_tensor_dtype(nm))
        shp=tuple(ctx.get_tensor_shape(nm))
        t=torch.empty(shp,dtype=getattr(torch,np.dtype(dt).name),device="cuda")
        if eng.get_tensor_mode(nm)==trt.TensorIOMode.INPUT:
            t.copy_(torch.from_numpy(np.ascontiguousarray(host[nm])).cuda())
        dev[nm]=t; ctx.set_tensor_address(nm,int(t.data_ptr()))
    st=torch.cuda.Stream(); meds=[]; clocks=[]
    for rep in range(6):
        torch.cuda.synchronize(); t0=time.perf_counter()
        with torch.cuda.stream(st):
            while (time.perf_counter()-t0)*1000<500: 
                ctx.execute_async_v3(st.cuda_stream); torch.cuda.synchronize()
        evs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(50)]
        with torch.cuda.stream(st):
            for a,b in evs: a.record(st); ctx.execute_async_v3(st.cuda_stream); b.record(st)
        torch.cuda.synchronize()
        ms=sorted(a.elapsed_time(b) for a,b in evs); meds.append(ms[25])
        clocks.append(torch.cuda.clock_rate() if hasattr(torch.cuda,'clock_rate') else 0)
    m=np.array(meds)
    out[stage]=dict(medians=[round(x,4) for x in meds], mean=round(float(m.mean()),4),
                    run_to_run_spread_pct=round(float(100*(m.max()-m.min())/m.mean()),2))
    print(stage, out[stage], flush=True)
    del ctx,eng,dev; torch.cuda.empty_cache()
json.dump(out,open(_W+"/trtwork5090/r_5090_run2run.json","w"),indent=1)
