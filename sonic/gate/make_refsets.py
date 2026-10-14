"""Build the two reference sets the engine-correctness gate (A2) compares a board against, from RTX 5090 probe output.
  encdec: the exact encoder/decoder tensors TensorRT consumed and produced, every tick of one closed-loop run
          (enc_in [N,1762], enc_out [N,64], dec_in [N,994], dec_out [N,29], float32)  <- enc_in.f32 / enc_out.f32 / dec_in.f32 / dec_out.f32
  planner: every planner call's inputs and the 5090 TensorRT outputs (from planner_io.bin, written by the probe's PlannerIO), plus an
          onnxruntime fp32 run of the same ONNX on the same inputs (ort_qpos / ort_npf), the reference the gate scores against
Our published gate used run v1_r1 for encdec (sha256 07c62321... of the npz) and a separate planner-logging run for the planner set.
A re-implementation of the original builder; on our run directories it reproduces every array of both reference sets exactly.
usage: python3 make_refsets.py encdec <run_dir> <out.npz>
       python3 make_refsets.py planner <run_dir> <planner_sonic.onnx> <out.npz>"""
import sys, numpy as np
mode, d = sys.argv[1], sys.argv[2]
if mode == 'encdec':
    f = lambda n, w: np.fromfile(f'{d}/{n}.f32', np.float32).reshape(-1, w)
    r = dict(enc_in=f('enc_in', 1762), enc_out=f('enc_out', 64), dec_in=f('dec_in', 994), dec_out=f('dec_out', 29))
    assert len({len(v) for v in r.values()}) == 1 and all(np.isfinite(v).all() for v in r.values())
    assert np.array_equal(r['dec_in'][:, :64], r['enc_out']), 'decoder input must start with the encoder tokens'
    assert np.allclose(r['enc_out'] * 16, np.round(r['enc_out'] * 16), atol=1e-5), 'encoder tokens must sit on the 1/16 FSQ grid'
    np.savez(sys.argv[3], **r); print('encdec ticks', len(r['enc_in']))
else:
    import onnxruntime as ort
    dt = np.dtype([('t', '<i8'), ('kind', '<i4'), ('mode', '<i4'), ('seed', '<i4'), ('npf', '<i4'), ('tv', '<f4'), ('h', '<f4'), ('mv', '<f4', 3),
                   ('face', '<f4', 3), ('ctx', '<f4', (4, 36)), ('qpos', '<f4', (64, 36))])
    a = np.fromfile(f'{d}/planner_io.bin', dtype=dt)
    s = ort.InferenceSession(sys.argv[3], providers=['CPUExecutionProvider'])
    allowed = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64); Q, N = [], []
    for r in a:
        f = {'context_mujoco_qpos': r['ctx'][None].astype(np.float32), 'target_vel': np.array([r['tv']], np.float32), 'mode': np.array([r['mode']], np.int64),
             'movement_direction': r['mv'][None].astype(np.float32), 'facing_direction': r['face'][None].astype(np.float32),
             'random_seed': np.array([r['seed']], np.int64), 'has_specific_target': np.zeros((1, 1), np.int64),
             'specific_target_positions': np.zeros((1, 4, 3), np.float32), 'specific_target_headings': np.zeros((1, 4), np.float32),
             'allowed_pred_num_tokens': allowed, 'height': np.array([r['h']], np.float32)}
        q, n = s.run(None, f); Q.append(q[0]); N.append(int(np.asarray(n).reshape(-1)[0]))
    np.savez(sys.argv[4], kind=a['kind'], mode=a['mode'], seed=a['seed'], target_vel=a['tv'], height=a['h'], movement_direction=a['mv'],
             facing_direction=a['face'], context_mujoco_qpos=a['ctx'], trt_qpos=a['qpos'], trt_npf=a['npf'], ort_qpos=np.array(Q), ort_npf=np.array(N))
    print('planner calls', len(a), 'npf agree TRT vs ORT', int((np.array(N) == a['npf']).sum()))
