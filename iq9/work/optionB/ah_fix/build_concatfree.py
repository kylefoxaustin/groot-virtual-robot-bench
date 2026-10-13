#!/usr/bin/env python3
"""action_encoder CONCAT-FREE splice — avoids a Concat->MatMul pattern that HTP context-gen
can mishandle in some QNN versions (e.g. by dropping the 2nd concat input, time_emb). Decomposition:
   add_72 = cat_1 @ W2[eid] + b2 = add_21 @ W2a[eid] + time_emb @ W2b[eid] + b2
The middle term (time_proj) is fixed per timestep -> precompute on host, feed as an
ADDITIVE input. Graph keeps W1/W3 category-selection; only the W2 table is sliced to its
add_21 half (rows :1536) and a `time_proj` input is added via an Add."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import os, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

AH=_gvrb_os.environ.get("GROOT_ONNX", _W+"/onnx")+"/action_head"; WORK=_W+"/optionB/ah_fix"
IO=f"{WORK}/io"; EID=24; TIMESTEPS=[0,250,500,750]
OUT=f"{WORK}/onnx/action_encoder_cf.onnx"

m=onnx.load(f"{AH}/action_encoder.onnx", load_external_data=True); g=m.graph
I={i.name:numpy_helper.to_array(i) for i in g.initializer}
W2=I["val_26"]  # (32,3072,1536) fp16
W2b=W2[:,1536:,:].astype(np.float32)  # (32,1536,1536) -> time_emb half

# precompute time_proj[t] = time_emb[t] @ W2b[eid]
for t in TIMESTEPS:
    te=np.fromfile(f"{IO}/time_emb_t{t}.raw",np.float32).reshape(1,16,1536)
    tp=(te @ W2b[EID]).astype(np.float32)  # (1,16,1536)
    tp.tofile(f"{IO}/time_proj_t{t}.raw")
    print(f"time_proj t={t}: |mean|={np.abs(tp).mean():.3f} absmax={np.abs(tp).max():.2f}")

# slice W2 table to add_21 half (rows :1536) -> new (32,1536,1536)
W2a=np.ascontiguousarray(W2[:,:1536,:])
for i,ini in enumerate(g.initializer):
    if ini.name=="val_26":
        g.initializer[i].CopyFrom(numpy_helper.from_array(W2a, name="val_26")); break

# drop sinusoidal chain + the concat
drop={"node_unsqueeze","node_expand","node__to_copy","node_unsqueeze_2","node_mul_17",
      "node_sin","node_cos","node_cat","node__to_copy_1","node_cat_1"}
keep=[n for n in g.node if n.name not in drop]
del g.node[:]; g.node.extend(keep)

# bmm_1 now consumes add_21 directly (not cat_1)
for n in g.node:
    if n.name=="node_bmm_1": n.input[0]="add_21"
# insert Add(bmm_1, time_proj) -> bmm_1_te ; add_72 consumes bmm_1_te
add_te=helper.make_node("Add", ["bmm_1","time_proj"], ["bmm_1_te"], name="node_add_time")
for n in g.node:
    if n.name=="node_add_72": n.input[0]="bmm_1_te"
# insert add_te right before add_72
nodes=list(g.node); idx=[i for i,n in enumerate(nodes) if n.name=="node_add_72"][0]
nodes.insert(idx, add_te); del g.node[:]; g.node.extend(nodes)

# inputs: drop timesteps_tensor, add time_proj, pin batch=1
new_in=[i for i in g.input if i.name!="timesteps_tensor"]
del g.input[:]; g.input.extend(new_in)
g.input.append(helper.make_tensor_value_info("time_proj", TensorProto.FLOAT16, [1,16,1536]))
def pin(vi):
    d=vi.type.tensor_type.shape.dim[0]; d.ClearField("dim_param"); d.dim_value=1
for i in g.input:
    if i.name in ("actions","embodiment_id"): pin(i)
for o in g.output: pin(o)
# prune unused initializers
used=set();  [used.update(n.input) for n in g.node]
keep_i=[ini for ini in g.initializer if ini.name in used]; del g.initializer[:]; g.initializer.extend(keep_i)
onnx.checker.check_model(m, full_check=False)
onnx.save(m, OUT, save_as_external_data=True, location="action_encoder_cf.onnx.data", all_tensors_to_one_file=True)
print("SAVED", OUT, "inputs:", [(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input], "#nodes",len(g.node))

# verify vs the (correct) spliced-concat onnx
import onnxruntime as ort
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sc=ort.InferenceSession(f"{WORK}/onnx/action_encoder_te.onnx",so,providers=["CPUExecutionProvider"])
scf=ort.InferenceSession(OUT,so,providers=["CPUExecutionProvider"])
act=np.fromfile(f"{IO}/../../io/init_actions.raw",np.float32).reshape(1,16,32)
act=np.fromfile(_W+"/optionB/io/init_actions.raw",np.float32).reshape(1,16,32)
eid=np.array([EID],np.int64)
for t in TIMESTEPS:
    te=np.fromfile(f"{IO}/time_emb_t{t}.raw",np.float32).reshape(1,16,1536)
    tp=np.fromfile(f"{IO}/time_proj_t{t}.raw",np.float32).reshape(1,16,1536)
    r_concat=sc.run(["output"],{"actions":act.astype(np.float16),"time_emb":te.astype(np.float16),"embodiment_id":eid})[0].astype(np.float32)
    r_cf=scf.run(["output"],{"actions":act.astype(np.float16),"time_proj":tp.astype(np.float16),"embodiment_id":eid})[0].astype(np.float32)
    rel=np.mean(np.abs(r_cf-r_concat))/np.mean(np.abs(r_concat))
    np.save(f"{WORK}/ref/aenc_cf_t{t}.npy", r_cf)
    print(f"  t={t}: concat-free vs concat rel_mae={rel:.2e}  cf|mean|={np.abs(r_cf).mean():.3f}")
print("DONE_CF")
