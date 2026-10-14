"""A2 correctness gate: run the EXACT .trt engines the upstream deploy built on this board (strip the 64-char hash
prefix upstream writes) on the 5090 reference set. Python TensorRT runtime + ctypes cudart (no torch).
Outputs npz of board outputs; scoring happens on the host (score_gate.py).
usage: python3 engine_gate.py <gear_sonic_deploy dir> <encdec refset.npz> <planner refset.npz> <out.npz>   (refsets: make_refsets.py)"""
import ctypes, sys, time, json, numpy as np, tensorrt as trt
D = sys.argv[1]; REF = sys.argv[2]; PREF = sys.argv[3]; OUT = sys.argv[4]
cu = ctypes.CDLL('libcudart.so')
def ck(r):
    assert r == 0, f'cuda err {r}'
def malloc(n):
    p = ctypes.c_void_p(); ck(cu.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n))); return p
H2D, D2H = 1, 2
def load(path):
    b = open(path, 'rb').read(); h = b[:64]; eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(b[64:])
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
r = np.load(REF); p = np.load(PREF); res = {}
info = {'trt_version': trt.__version__}
enc = Runner(f'{D}/policy/release/encoder_model_encoder.trt'); dec = Runner(f'{D}/policy/release/policy_model_decoder.trt')
pl = Runner(f'{D}/planner/target_vel/V2/planner_planner_sonic.trt')
info['io'] = {k: {n: [list(m[0]), np.dtype(m[1]).name, m[2]] for n, m in R.meta.items()} for k, R in (('enc', enc), ('dec', dec), ('planner', pl))}
t = time.time(); res['enc_out'] = np.stack([enc.run({'obs_dict': x})['encoded_tokens'].reshape(-1) for x in r['enc_in']]); info['enc_s'] = time.time() - t
t = time.time(); res['dec_out'] = np.stack([dec.run({'obs_dict': x})['action'].reshape(-1) for x in r['dec_in']]); info['dec_s'] = time.time() - t
# planted-fault input: features 64..993 from a different tick (wrong observation), tokens kept
di = r['dec_in'].copy(); di[:, 64:] = np.roll(r['dec_in'][:, 64:], 500, axis=0)
res['dec_out_planted_wrongobs'] = np.stack([dec.run({'obs_dict': x})['action'].reshape(-1) for x in di])
allowed = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
Q, N = [], []
for i in range(len(p['mode'])):
    f = {'context_mujoco_qpos': p['context_mujoco_qpos'][i], 'target_vel': [p['target_vel'][i]], 'mode': [p['mode'][i]],
         'movement_direction': p['movement_direction'][i], 'facing_direction': p['facing_direction'][i], 'random_seed': [p['seed'][i]],
         'has_specific_target': [[0]], 'specific_target_positions': np.zeros((1, 4, 3)), 'specific_target_headings': np.zeros((1, 4)),
         'allowed_pred_num_tokens': allowed, 'height': [p['height'][i]]}
    o = pl.run(f); Q.append(o['mujoco_qpos'].reshape(64, 36)); N.append(int(np.asarray(o['num_pred_frames']).reshape(-1)[0]))
res['planner_qpos'] = np.stack(Q); res['planner_npf'] = np.array(N)
info['hash_prefix'] = {'enc': enc.hash, 'dec': dec.hash, 'planner': pl.hash}
np.savez(OUT, **res); json.dump(info, open(OUT.replace('.npz', '.json'), 'w'), indent=1); print(json.dumps(info)[:2000])
