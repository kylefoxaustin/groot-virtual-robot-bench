#!/usr/bin/env python3
# Multi-stream d2d copy to push actmon ABOVE a single copy's saturation (bracket the DiT stage).
# Exact GB/s = 2*bytes/wall aggregated over K streams. Samples actmon. Run as root.
import sys, time, threading, json, numpy as np
try:
    from cuda import cudart
except ImportError:
    from cuda.bindings import runtime as cudart
A="/sys/kernel/debug/bpmp/debug/actmon/mc_all_avg_activity"
def rd(): return int(open(A).read())
def CK(r):
    e=r[0] if isinstance(r,(tuple,list)) else r
    if int(e)!=0: raise RuntimeError("cuda "+str(e))
    if isinstance(r,(tuple,list)) and len(r)==2: return r[1]
NB=256*1024*1024
def run(K,dur=6.0):
    bufs=[(CK(cudart.cudaMalloc(NB)),CK(cudart.cudaMalloc(NB))) for _ in range(K)]
    streams=[int(CK(cudart.cudaStreamCreate())) for _ in range(K)]
    D2D=cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    for _ in range(8):
        for (s_,d_),st in zip(bufs,streams): cudart.cudaMemcpyAsync(d_,s_,NB,D2D,st)
        for st in streams: cudart.cudaStreamSynchronize(st)
    samples=[]; stop=[False]
    def smp():
        time.sleep(1.0)
        while not stop[0]: samples.append(rd()); time.sleep(0.2)
    th=threading.Thread(target=smp); th.start()
    t0=time.perf_counter(); nbytes=0
    while time.perf_counter()-t0<dur:
        for (s_,d_),st in zip(bufs,streams): cudart.cudaMemcpyAsync(d_,s_,NB,D2D,st)
        for st in streams: cudart.cudaStreamSynchronize(st)
        nbytes+=2*NB*K
    wall=time.perf_counter()-t0; stop[0]=True; th.join()
    for (s_,d_) in bufs: cudart.cudaFree(s_); cudart.cudaFree(d_)
    return int(np.mean(samples)), round(nbytes/wall/1e9,1)
R={"board":sys.argv[1],"multistream":[]}
for K in (1,2,3,4):
    a,g=run(K); R["multistream"].append({"streams":K,"actmon":a,"gbps":g}); print("K=%d  actmon %d  %.1f GB/s"%(K,a,g))
open("/tmp/gpu_ms_%s.json"%sys.argv[1].lower(),"w").write(json.dumps(R,indent=1)); print("DONE")
