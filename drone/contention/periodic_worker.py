#!/usr/bin/env python3
"""Periodic co-tenant worker: a TensorRT engine fired at a fixed cadence, with a host-to-device copy of a host-resident
random input every frame (as a sensor ingest would), 10 warm-up inferences, per-fire logging and a 'started' file so the
orchestrator can bin windows. Used for the radar fusion stand-in (hz:20, or back to back) and for route planning
(seg_1024 at an ASSUMED duty, e.g. hz:0.333333 = one frame every 3 s). Needs tensorrt + cuda-python + numpy.
  args: engine spec out [started_file]
  spec "hz:20,t:12.0" = fire at 20 Hz for 12 s ; "flat:400" = 400 back-to-back inferences
"""
import sys, time, json, numpy as np, tensorrt as trt
try:
    from cuda import cudart
except ImportError:
    from cuda.bindings import runtime as cudart
eng, spec, out = sys.argv[1], sys.argv[2], sys.argv[3]; started = sys.argv[4] if len(sys.argv) > 4 else None
LOG = trt.Logger(trt.Logger.ERROR); H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
def CK(r):
    e = r[0] if isinstance(r, (tuple, list)) else r
    if int(e) != 0: raise RuntimeError("cuda " + str(e))
    if isinstance(r, (tuple, list)) and len(r) == 2: return r[1]
rt = trt.Runtime(LOG); e = rt.deserialize_cuda_engine(open(eng, "rb").read()); ctx = e.create_execution_context()
in_dev = in_host = in_nb = None
for i in range(e.num_io_tensors):
    n = e.get_tensor_name(i); shp = e.get_tensor_shape(n); dt = trt.nptype(e.get_tensor_dtype(n))
    vol = int(np.prod([max(1, d) for d in shp])); nb = vol * np.dtype(dt).itemsize
    d = CK(cudart.cudaMalloc(nb))
    if e.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
        in_dev, in_nb = int(d), nb
        in_host = np.random.default_rng(0).standard_normal(vol).astype(dt)
    ctx.set_tensor_address(n, int(d))
stream = int(CK(cudart.cudaStreamCreate()))
def inf():
    CK(cudart.cudaMemcpyAsync(in_dev, in_host.ctypes.data, in_nb, H2D, stream))
    ctx.execute_async_v3(stream); CK(cudart.cudaStreamSynchronize(stream))
for _ in range(10): inf()
lat = []; fire = []; t0 = time.time()
if started: open(started, "w").write(str(t0))
parts = dict(p.split(":") for p in spec.split(","))
if "hz" in parts:
    hz = float(parts["hz"]); dur = float(parts["t"]); period = 1.0 / hz; nxt = t0
    while time.time() - t0 < dur:
        a = time.time(); inf(); lat.append((time.time() - a) * 1e3); fire.append(a)
        nxt += period; s = nxt - time.time()
        if s > 0: time.sleep(min(s, max(0.0, t0 + dur - time.time())))
else:
    for _ in range(int(parts["flat"])):
        a = time.time(); inf(); lat.append((time.time() - a) * 1e3); fire.append(a)
end = time.time(); wall = end - t0
json.dump({"engine": eng, "spec": spec, "n": len(lat), "wall_s": round(wall, 3),
           "per_inf_ms_mean": round(float(np.mean(lat)), 4), "per_inf_ms_p95": round(float(np.percentile(lat, 95)), 4),
           "per_inf_ms_max": round(float(np.max(lat)), 4),
           "achieved_hz": round(len(lat) / wall, 3), "duty_pct": round(100 * sum(lat) / 1e3 / wall, 2), "ingest": "H2D per frame",
           "start": t0, "end": end, "fire_rel_s": [round(f - t0, 5) for f in fire], "lat_ms": [round(x, 4) for x in lat]}, open(out, "w"))
print("periodic_worker %s: %d inf, %.2fs, %.4f ms/inf, %.2f Hz, duty %.2f%%" % (spec, len(lat), wall, np.mean(lat), len(lat) / wall, 100 * sum(lat) / 1e3 / wall))
