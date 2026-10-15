# Write the two files the board runtimes read next to the planner bins (run in the planner work dir after build_blocks.py):
#   blocks/manifest.txt     "block X" / "in <tensor>" / "out <tensor>" per NPU block, in S R P D order (deploy shim + sonic_rt)
#   blocks/planner_calls.f32 every logged planner call of the reference set, 154 float32 per call:
#                            context_mujoco_qpos[144] mode target_vel movement_direction[3] facing_direction[3] seed height (sonic_rt)
# usage: python3 make_manifest.py [<planner refset.npz from sonic/gate/make_refsets.py>]   (calls file only with the refset)
import json, sys, numpy as np
B = json.load(open('blocks/blocks.json'))
with open('blocks/manifest.txt', 'w') as f:
    for b in 'SRPD':
        f.write(f'block {b}\n'); [f.write(f'in {x}\n') for x in B[b]['ins']]; [f.write(f'out {x}\n') for x in B[b]['outs']]
if len(sys.argv) < 2: print('manifest written; no refset given, no planner_calls.f32'); sys.exit(0)
r = np.load(sys.argv[1])
N = len(r['mode']); C = np.zeros((N, 154), np.float32)
C[:, :144] = r['context_mujoco_qpos'].reshape(N, 144); C[:, 144] = r['mode']; C[:, 145] = r['target_vel']
C[:, 146:149] = r['movement_direction']; C[:, 149:152] = r['facing_direction']; C[:, 152] = r['seed']; C[:, 153] = r['height']
C.tofile('blocks/planner_calls.f32'); print('calls', N, 'seeds', sorted(set(r['seed'].tolist())))
