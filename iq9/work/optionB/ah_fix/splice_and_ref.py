#!/usr/bin/env python3
"""GR00T action-head fp16 fix — STEP A: splice action_encoder (host-precomputed time_emb),
extract time_emb for the 4 denoise timesteps, and build host fp32-accurate per-graph refs.

- action_encoder_te.onnx: replaces the on-graph sinusoidal timestep embedding (the tensor
  `_to_copy_1` that feeds node_cat_1) with a graph INPUT `time_emb` (1,16,1536). Removes the
  timestep->sin/cos->cat chain + the timesteps_tensor input. a_emb path (W1) untouched.
- time_emb_t{0,250,500,750}.raw: the EXACT model tau embedding (run the original onnx's
  sinusoidal subgraph in ORT-CPU/fp32), saved float32 (QNN I/O boundary is float32).
- ref/*.npy: host ORT (fp16 onnx, CPU upcasts->fp32) per-graph outputs for the real captured
  inputs, to gate the board bins (rel_mae/cos).
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import os, sys, numpy as np, onnx
from onnx import helper, TensorProto
import onnxruntime as ort

AH   = _gvrb_os.environ.get("GROOT_ONNX", _W+"/onnx")+"/action_head"
IO   = _W+"/optionB/io"
WORK = _W+"/optionB/ah_fix"
OUT_ONNX = f"{WORK}/onnx/action_encoder_te.onnx"
os.makedirs(f"{WORK}/onnx", exist_ok=True); os.makedirs(f"{WORK}/io", exist_ok=True); os.makedirs(f"{WORK}/ref", exist_ok=True)

EID = 24  # captured embodiment_id (gr1)
TIMESTEPS = [0, 250, 500, 750]

def sess(p):
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

# ---- load real captured inputs ----
state = np.fromfile(f"{IO}/state.raw", np.float32).reshape(1,1,64)
init_actions = np.fromfile(f"{IO}/init_actions.raw", np.float32).reshape(1,16,32)
eid64 = np.array([EID], np.int64)

# ============================================================
# 1) time_emb extraction: add _to_copy_1 as an output of the ORIGINAL onnx, run per timestep
# ============================================================
m = onnx.load(f"{AH}/action_encoder.onnx", load_external_data=True)
g = m.graph
# add _to_copy_1 as extra output
g.output.append(helper.make_tensor_value_info("_to_copy_1", TensorProto.FLOAT16, [1,16,1536]))
onnx.save(m, f"{WORK}/onnx/_aenc_probe.onnx", save_as_external_data=True,
          location="_aenc_probe.onnx.data", all_tensors_to_one_file=True)
sp = sess(f"{WORK}/onnx/_aenc_probe.onnx")
time_embs = {}
for t in TIMESTEPS:
    outs = sp.run(["output","_to_copy_1"], {
        "actions": init_actions.astype(np.float16),
        "timesteps_tensor": np.array([t], np.int64),
        "embodiment_id": eid64})
    te = outs[1].astype(np.float32)          # (1,16,1536) tau embedding
    time_embs[t] = te
    te.tofile(f"{WORK}/io/time_emb_t{t}.raw")
    # also the FULL-onnx action_encoder output (fp32 ref) for this (actions=init_actions,t)
    np.save(f"{WORK}/ref/aenc_out_t{t}.npy", outs[0].astype(np.float32))
    print(f"time_emb t={t}: shape {te.shape} |mean| {np.abs(te).mean():.5f} range[{te.min():.3f},{te.max():.3f}]  aenc_out|mean|={np.abs(outs[0]).mean():.4f}")
# sanity: t0 vs t750 time_emb MUST differ (host is correct)
d = np.abs(time_embs[0]-time_embs[750]).max()
print(f"host time_emb t0-vs-t750 maxdiff = {d:.4f}  (MUST be >0 -> host sinusoidal is correct)")

# ============================================================
# 2) build action_encoder_te.onnx  (splice: _to_copy_1 -> graph input time_emb)
# ============================================================
m2 = onnx.load(f"{AH}/action_encoder.onnx", load_external_data=True)
g2 = m2.graph
# nodes to drop: the timestep/sinusoidal chain (everything producing _to_copy_1)
drop = {"node_unsqueeze","node_expand","node__to_copy","node_unsqueeze_2","node_mul_17",
        "node_sin","node_cos","node_cat","node__to_copy_1"}
keep = [n for n in g2.node if n.name not in drop]
del g2.node[:]
g2.node.extend(keep)
# remove timesteps_tensor input, add time_emb input (fp16, static batch=1)
new_inputs = [i for i in g2.input if i.name != "timesteps_tensor"]
del g2.input[:]
g2.input.extend(new_inputs)
g2.input.append(helper.make_tensor_value_info("time_emb", TensorProto.FLOAT16, [1,16,1536]))
# node_cat_1 now consumes [add_21, time_emb]
for n in g2.node:
    if n.name == "node_cat_1":
        n.input[:] = ["add_21", "time_emb"]
# pin batch dim = 1 on actions/embodiment_id/output
def pin(vi):
    d0 = vi.type.tensor_type.shape.dim[0]
    d0.ClearField("dim_param"); d0.dim_value = 1
for i in g2.input:
    if i.name in ("actions","embodiment_id"): pin(i)
for o in g2.output: pin(o)
# drop now-unused initializers (val_5? etc. keep all that remain referenced)
used = set()
for n in g2.node:
    used.update(n.input)
keep_inits = [ini for ini in g2.initializer if ini.name in used]
del g2.initializer[:]
g2.initializer.extend(keep_inits)
onnx.checker.check_model(m2, full_check=False)
onnx.save(m2, OUT_ONNX, save_as_external_data=True, location="action_encoder_te.onnx.data", all_tensors_to_one_file=True)
print(f"\nSPLICED -> {OUT_ONNX}")
print("  inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g2.input])
print("  #nodes:", len(g2.node))

# verify spliced onnx reproduces the full onnx's action_encoder output when fed the host time_emb
st = sess(OUT_ONNX)
for t in TIMESTEPS:
    o = st.run(["output"], {"actions": init_actions.astype(np.float16),
                            "time_emb": time_embs[t].astype(np.float16),
                            "embodiment_id": eid64})[0].astype(np.float32)
    ref = np.load(f"{WORK}/ref/aenc_out_t{t}.npy")
    rel = np.mean(np.abs(o-ref))/(np.mean(np.abs(ref))+1e-12)
    print(f"  spliced vs full aenc t={t}: rel_mae={rel:.2e} (must be ~0 -> splice is exact)")

# ============================================================
# 3) host fp32 refs for state_encoder + action_decoder (real inputs)
# ============================================================
ss = sess(f"{AH}/state_encoder.onnx")
so = ss.run(["output"], {"state": state.astype(np.float16), "embodiment_id": eid64})[0].astype(np.float32)
np.save(f"{WORK}/ref/state_out.npy", so)
print(f"\nstate_encoder host ref: {so.shape} |mean|={np.abs(so).mean():.4f} absmax={np.abs(so).max():.3f}")
np.save(f"{WORK}/ref/_state_in.npy", state); np.save(f"{WORK}/ref/_init_actions.npy", init_actions)
print("DONE_SPLICE_AND_REF")
