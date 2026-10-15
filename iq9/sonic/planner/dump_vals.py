# Run the full upstream planner once on a fixed walk input with EVERY intermediate tensor exposed, and pickle the values
# (vals_walk.pkl) and shapes/dtypes (shapes.pkl). build_blocks.py, fix_*.py and fp16sim.py read them for shapes and for
# constant-folding tensors that do not depend on the inputs.
# usage (in the planner work dir, with planner_sonic.onnx linked there): python3 dump_vals.py
import onnx, onnxruntime as ort, numpy as np, sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from inputs import make
m = onnx.load('planner_sonic.onnx'); g = m.graph
outs = []; seen = set(o.name for o in g.output)
for n in g.node:
    if n.op_type == 'Constant': continue
    for o in n.output:
        if o and o not in seen: outs.append(o); seen.add(o)
del g.output[:]
for o in ['mujoco_qpos', 'num_pred_frames'] + outs:
    g.output.append(onnx.helper.make_empty_tensor_value_info(o))
onnx.save(m, 'planner_allout.onnx', save_as_external_data=True, location='planner_allout.data')
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s = ort.InferenceSession('planner_allout.onnx', so, providers=['CPUExecutionProvider'])
names = [o.name for o in s.get_outputs()]
res = {}
for tag, inp in [('walk', make(2, -1.0)), ('idle', make(0, -1.0, (0, 0, 0))), ('run', make(3, -1.0)), ('seed7', make(2, -1.0, seed=7))]:
    r = s.run(None, inp); print(tag, 'npf', r[1])
    res[tag] = {n: (v.shape, v.dtype) for n, v in zip(names, r)}
    if tag == 'walk': pickle.dump(dict(zip(names, r)), open('vals_walk.pkl', 'wb'))
pickle.dump(res, open('shapes.pkl', 'wb'))
