#!/usr/bin/env python3
"""Steady-state board power while the compiled GR00T chain runs flat out on the 5090."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import sys, time, json, subprocess, threading
import numpy as np, tensorrt as trt, torch
sys.path.insert(0,_W+'/trtwork5090')
from trt_groot import build, load_inputs
L=trt.Logger(trt.Logger.ERROR); ctxs={}; keep={}
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
idle=[float(x) for x in subprocess.check_output(
    "for i in $(seq 12); do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits; sleep 0.25; done",
    shell=True).split()]
samples=[]; stop=threading.Event()
def sampler():
    while not stop.is_set():
        try: samples.append(float(subprocess.check_output(
            ["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"]).strip()))
        except Exception: pass
        time.sleep(0.2)
st=torch.cuda.Stream(); order=[("vit",1),("llm",1),("dit",4)]
with torch.cuda.stream(st):
    for _ in range(50):
        for k,n in order:
            for _ in range(n): ctxs[k].execute_async_v3(st.cuda_stream)
torch.cuda.synchronize()
th=threading.Thread(target=sampler); th.start()
t0=time.perf_counter(); iters=0
with torch.cuda.stream(st):
    while time.perf_counter()-t0 < 20:
        for _ in range(20):
            for k,n in order:
                for _ in range(n): ctxs[k].execute_async_v3(st.cuda_stream)
            iters+=1
        torch.cuda.synchronize()
torch.cuda.synchronize(); el=time.perf_counter()-t0
stop.set(); th.join()
s=np.array(samples); i=np.array(idle)
chain_ms = el*1000/iters
out=dict(idle_power_W=round(float(i.mean()),1), load_power_W_mean=round(float(s.mean()),1),
         load_power_W_max=round(float(s.max()),1), n_power_samples=len(s),
         measured_chain_ms=round(chain_ms,4), chains_per_s=round(iters/el,2),
         energy_per_inference_J=round(float(s.mean())*chain_ms/1000,4),
         net_energy_per_inference_J=round((float(s.mean())-float(i.mean()))*chain_ms/1000,4))
print(json.dumps(out,indent=1))
json.dump(out,open(_W+"/trtwork5090/r_5090_power.json","w"),indent=1)
