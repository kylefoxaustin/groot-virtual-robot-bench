"""Planner-only gate: run ONE planner engine (an upstream-built .trt, whose 64-char hash prefix is stripped, or a plain
engine such as trtexec --onnx planner_sonic.onnx --noTF32 --saveEngine=...) on the planner reference set and print the per-call max |dqpos|
vs the onnxruntime fp32 reference. Python TensorRT runtime + ctypes cudart (no torch). The script behind the TF32 diagnosis; this copy also accepts a plain engine.
usage: python3 planner_gate.py <planner .trt> <planner refset.npz> <out.npz>"""
import ctypes, sys, time, json, numpy as np, tensorrt as trt
PREF = sys.argv[2]; OUT = sys.argv[3]
cu = ctypes.CDLL('libcudart.so')
def ck(r):
    assert r == 0, f'cuda err {r}'
def malloc(n):
    p = ctypes.c_void_p(); ck(cu.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n))); return p
H2D, D2H = 1, 2
def load(path):
    b = open(path, 'rb').read(); h = b[:64]
    if not all(c in b'0123456789abcdef' for c in h): h = b''   # plain engine (no upstream hash prefix)
    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(b[len(h):])
    assert eng is not None, path
    return eng, h.decode()
NP = {trt.DataType.FLOAT: np.float32, trt.DataType.INT32: np.int32, trt.DataType.INT64: np.int64, trt.DataType.BOOL: np.bool_}
class Runner:
    def __init__(s, path):
        s.eng, s.hash = load(path); s.ctx = s.eng.create_execution_context(); s.names = [s.eng.get_tensor_name(i) for i in range(s.eng.num_io_tensors)]
        s.buf = {}; s.meta = {}
        for n in s.names:
            sh = tuple(s.ctx.get_tensor_shape(n)); dt = NP[s.eng.get_tensor_dtype(n)]
            s.meta[n] = (sh, dt, s.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
            nb = int(np.prod(sh)) * np.dtype(dt).itemsize; s.buf[n] = malloc(max(nb, 8)); s.ctx.set_tensor_address(n, s.buf[n].value)
        s.stream = ctypes.c_void_p(); ck(cu.cudaStreamCreate(ctypes.byref(s.stream)))
    def run(s, feeds):
        for n, (sh, dt, isin) in s.meta.items():
            if isin:
                x = np.ascontiguousarray(np.asarray(feeds[n]).astype(dt).reshape(sh))
                ck(cu.cudaMemcpy(ctypes.c_void_p(s.buf[n].value), x.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(x.nbytes), H2D))
        assert s.ctx.execute_async_v3(s.stream.value); ck(cu.cudaStreamSynchronize(s.stream))
        out = {}
        for n, (sh, dt, isin) in s.meta.items():
            if not isin:
                y = np.empty(sh, dt); ck(cu.cudaMemcpy(y.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(s.buf[n].value), ctypes.c_size_t(y.nbytes), D2H)); out[n] = y
        return out
p = np.load(PREF); allowed = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]); pl = Runner(sys.argv[1]); Q, N = [], []
for i in range(len(p['mode'])):
    f = {'context_mujoco_qpos': p['context_mujoco_qpos'][i], 'target_vel': [p['target_vel'][i]], 'mode': [p['mode'][i]],
         'movement_direction': p['movement_direction'][i], 'facing_direction': p['facing_direction'][i], 'random_seed': [p['seed'][i]],
         'has_specific_target': [[0]], 'specific_target_positions': np.zeros((1, 4, 3)), 'specific_target_headings': np.zeros((1, 4)),
         'allowed_pred_num_tokens': allowed, 'height': [p['height'][i]]}
    o = pl.run(f); Q.append(o['mujoco_qpos'].reshape(64, 36)); N.append(int(np.asarray(o['num_pred_frames']).reshape(-1)[0]))
Q = np.stack(Q); N = np.array(N); mx = [np.abs(Q[i, :N[i]] - p['ort_qpos'][i, :N[i]]).max() for i in range(len(N))]
print(sys.argv[1], 'trt', trt.__version__, 'npf agree vs ort', int((N == p['ort_npf']).sum()), 'per-call max |dq| vs ORT p50/p90/max', np.percentile(mx, 50), np.percentile(mx, 90), max(mx))
np.savez(OUT, planner_qpos=Q, planner_npf=N)
