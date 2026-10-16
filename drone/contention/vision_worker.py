#!/usr/bin/env python3
# Inspection-burst worker: the 50 MP inspection AI stand-in (segmentation over 190 tiles of 512 x 512 = one 50 MP frame).
# Runs a TensorRT engine on ZERO inputs (the timing does not depend on the pixel values) and logs its own [start, end],
# measured after the engine load and 5 warm-up inferences; that interval is the burst window the nav rate is binned over.
#   args: ENGINE SPEC OUT_JSON     SPEC "n:190" = 190 inferences back to back;  "t:6.0" = back to back for 6.0 s
# Needs tensorrt + cuda-python (cuda.cudart or cuda.bindings.runtime) + numpy.
import sys, time, json, numpy as np, tensorrt as trt
try:
    from cuda import cudart
except ImportError:
    from cuda.bindings import runtime as cudart
eng, spec, out = sys.argv[1], sys.argv[2], sys.argv[3]
LOG=trt.Logger(trt.Logger.ERROR); H2D=cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
def CK(r):
    e=r[0] if isinstance(r,(tuple,list)) else r
    if int(e)!=0: raise RuntimeError("cuda "+str(e))
    if isinstance(r,(tuple,list)) and len(r)==2: return r[1]
rt=trt.Runtime(LOG); e=rt.deserialize_cuda_engine(open(eng,"rb").read()); ctx=e.create_execution_context()
for i in range(e.num_io_tensors):
    n=e.get_tensor_name(i); shp=e.get_tensor_shape(n); dt=trt.nptype(e.get_tensor_dtype(n))
    vol=int(np.prod([max(1,d) for d in shp])); nb=vol*np.dtype(dt).itemsize
    d=CK(cudart.cudaMalloc(nb))
    if e.get_tensor_mode(n)==trt.TensorIOMode.INPUT:
        z=np.zeros(vol,dtype=dt); CK(cudart.cudaMemcpy(d,z.ctypes.data,nb,H2D))
    ctx.set_tensor_address(n,int(d))
stream=int(CK(cudart.cudaStreamCreate()))
def inf(): ctx.execute_async_v3(stream); CK(cudart.cudaStreamSynchronize(stream))
for _ in range(5): inf()
kind,val=spec.split(":"); t0=time.time(); lat=[]
if kind=="n":
    for _ in range(int(val)):
        a=time.time(); inf(); lat.append((time.time()-a)*1e3)
else:
    while time.time()-t0<float(val):
        a=time.time(); inf(); lat.append((time.time()-a)*1e3)
wall=time.time()-t0
json.dump({"engine":eng,"spec":spec,"n":len(lat),"wall_s":round(wall,3),
           "per_inf_ms_mean":round(float(np.mean(lat)),3),"start":t0,"end":time.time()},open(out,"w"))
print("vision_worker done: %d inf, %.2fs, %.2f ms/inf"%(len(lat),wall,np.mean(lat)))
