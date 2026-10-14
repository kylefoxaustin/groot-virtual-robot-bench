"""Harness check for the planner gate: the feed construction used by engine_gate.py / planner_gate.py, run through host
onnxruntime fp32, must reproduce the reference set's logged ort_qpos (it did: npf 86/86, per-call max |dq| p90 1.0e-5).
usage: python3 planner_ort_check.py <planner refset.npz> <gear_sonic_deploy>/planner/target_vel/V2/planner_sonic.onnx"""
import numpy as np, onnxruntime as ort, sys
p = np.load(sys.argv[1])
s = ort.InferenceSession(sys.argv[2], providers=['CPUExecutionProvider'])
allowed = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64); mx = []; nok = 0
for i in range(len(p['mode'])):
    f = {'context_mujoco_qpos': p['context_mujoco_qpos'][i][None].astype(np.float32), 'target_vel': np.array([p['target_vel'][i]], np.float32),
         'mode': np.array([p['mode'][i]], np.int64), 'movement_direction': p['movement_direction'][i][None].astype(np.float32),
         'facing_direction': p['facing_direction'][i][None].astype(np.float32), 'random_seed': np.array([p['seed'][i]], np.int64),
         'has_specific_target': np.zeros((1, 1), np.int64), 'specific_target_positions': np.zeros((1, 4, 3), np.float32),
         'specific_target_headings': np.zeros((1, 4), np.float32), 'allowed_pred_num_tokens': allowed, 'height': np.array([p['height'][i]], np.float32)}
    q, n = s.run(None, f); n = int(n.reshape(-1)[0]); nok += n == p['ort_npf'][i]
    mx.append(np.abs(q[0, :n] - p['ort_qpos'][i, :n]).max())
print('ort', ort.__version__, 'npf agree', nok, 'per-call max |dq| p50/p90/max', np.percentile(mx, 50), np.percentile(mx, 90), max(mx))
