#!/usr/bin/env python3
# GPU DRAM-bandwidth calibration: timed device-to-device cudaMemcpy reports EXACT achieved
# GB/s (2*bytes/time; a pure copy = N read + N write, no write-allocate ambiguity), while
# sampling BPMP actmon. Duty-cycling sweeps average BW from low to saturation -> gives the
# actmon<->GB/s line ACROSS the workload's actmon range (which a CPU burn cannot reach).
# Run as root (actmon needs it). arg1 = board name.
import sys, time, threading, json, numpy as np
try:
    from cuda import cudart
except ImportError:
    from cuda.bindings import runtime as cudart
A="/sys/kernel/debug/bpmp/debug/actmon/mc_all_avg_activity"
def rd(): return int(open(A).read())
def CK(r):
    e=r[0] if isinstance(r,(tuple,list)) else r
    if int(e)!=0: raise RuntimeError("cuda error "+str(e))
    if isinstance(r,(tuple,list)) and len(r)==2: return r[1]
NB=512*1024*1024
src=CK(cudart.cudaMalloc(NB)); dst=CK(cudart.cudaMalloc(NB))
stream=CK(cudart.cudaStreamCreate())
D2D=cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
def copy():
    CK(cudart.cudaMemcpyAsync(dst,src,NB,D2D,stream)); CK(cudart.cudaStreamSynchronize(stream))
for _ in range(12): copy()                       # warmup
def measure(duty,dur=6.0):
    samples=[]; stop=[False]
    def smp():
        time.sleep(1.0)                          # let it reach steady state before sampling
        while not stop[0]: samples.append(rd()); time.sleep(0.2)
    th=threading.Thread(target=smp); th.start()
    t0=time.perf_counter(); nbytes=0
    while time.perf_counter()-t0<dur:
        c0=time.perf_counter(); copy(); cw=time.perf_counter()-c0
        nbytes+=2*NB
        if duty<1.0: time.sleep(cw*(1.0/duty-1.0))
    wall=time.perf_counter()-t0; stop[0]=True; th.join()
    return int(np.mean(samples)), round(nbytes/wall/1e9,1)
idle=int(np.mean([rd() for _ in range(6) if not time.sleep(0.2)]))
R={"board":sys.argv[1],"method":"GPU d2d cudaMemcpy, GB/s=2*bytes/wall (exact), actmon mean","idle_floor":idle,"gpu_cal":[]}
for d in (1.0,0.6,0.35,0.2):
    a,g=measure(d); R["gpu_cal"].append({"duty":d,"actmon":a,"gbps":g}); print("duty %.2f  actmon %d  %.1f GB/s"%(d,a,g))
open("/tmp/gpu_bw_%s.json"%sys.argv[1].lower(),"w").write(json.dumps(R,indent=1))
print("IDLE",idle); print("DONE")
