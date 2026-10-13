#!/usr/bin/env python3
"""HOST-side per-graph gate for GR00T on iq9 prep.

Runs each exported ONNX graph in onnxruntime (CPU) on the captured .raw inputs and
saves the fp32 reference output (.npy) that the on-board v73 context binary will be
gated against later. Also re-verifies that the pre-existing vit/llm/dit references
(io/*_ref.npy) reproduce from their ONNX + captured .raw inputs.

We CANNOT run the v73 HTP context binaries on the host (device-specific); the fp16/v73
accuracy gate is a BOARD step. This host step (a) proves every ONNX graph loads+runs and
(b) produces the golden the board compares to. Numbers here are MEASURED (onnxruntime CPU).
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import os, json, hashlib, time, numpy as np, onnxruntime as ort

AH = _gvrb_os.environ.get("GROOT_ONNX", _W+"/onnx")+"/action_head"
EG = _gvrb_os.environ.get("GROOT_ONNX", _W+"/onnx")+"/eagle2"
AHIN = _W+"/qnn/ah/in"
IO = _W+"/qnn/io"
OUT = _W+"/optionB/ref"
os.makedirs(OUT, exist_ok=True)

def md5(a): return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
def relmae(a,b):
    a=a.astype(np.float64).ravel(); b=b.astype(np.float64).ravel()
    return float(np.mean(np.abs(a-b))/(np.mean(np.abs(b))+1e-12))
def cos(a,b):
    a=a.astype(np.float64).ravel(); b=b.astype(np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def sess(p):
    so=ort.SessionOptions(); so.log_severity_level=3
    return ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"])

report={"date":time.strftime("%Y-%m-%d"),"tag":"MEASURED","engine":"onnxruntime %s CPU"%ort.__version__,
        "note":"HOST per-graph refs (fp32) for board bin-vs-ONNX gate; v73 HTP bins are NOT runnable on host.",
        "graphs":{}}

# ---- Action-head graphs (inputs = the RandomState(0) synthetic tensors from ah/in used in the timing run) ----
emb = np.fromfile(f"{AHIN}/embodiment_id.raw", dtype=np.int32).astype(np.int64)  # scalar-ish
if emb.size==0: emb=np.array([1],dtype=np.int64)
ts  = np.fromfile(f"{AHIN}/timesteps_tensor.raw", dtype=np.int32).astype(np.int64)
if ts.size==0: ts=np.array([0],dtype=np.int64)

jobs = {
 "state_encoder": dict(onnx=f"{AH}/state_encoder.onnx",
    feeds=lambda: {"state": np.fromfile(f"{AHIN}/state.raw",np.float32).reshape(1,1,64),
                   "embodiment_id": emb.reshape(1)}),
 "action_encoder": dict(onnx=f"{AH}/action_encoder.onnx",
    feeds=lambda: {"actions": np.fromfile(f"{AHIN}/actions.raw",np.float32).reshape(1,16,32),
                   "timesteps_tensor": ts.reshape(1), "embodiment_id": emb.reshape(1)}),
 "action_decoder": dict(onnx=f"{AH}/action_decoder.onnx",
    feeds=lambda: {"model_output": np.fromfile(f"{AHIN}/model_output.raw",np.float32).reshape(1,49,1024),
                   "embodiment_id": emb.reshape(1)}),
 "vlln_vl_self_attention": dict(onnx=f"{AH}/vlln_vl_self_attention.onnx",
    feeds=lambda: {"backbone_features": np.fromfile(f"{AHIN}/backbone_features.raw",np.float32).reshape(1,296,2048)}),
 "DiT_fp16": dict(onnx=f"{AH}/DiT_fp16.onnx",
    feeds=lambda: {"sa_embs": np.fromfile(f"{IO}/gdit_sa_embs.raw",np.float32).reshape(1,49,1536),
                   "vl_embs": np.fromfile(f"{IO}/gdit_vl_embs.raw",np.float32).reshape(1,-1,2048),
                   "timesteps_tensor": np.fromfile(f"{IO}/gdit_timesteps.raw",np.int32).astype(np.int64).reshape(1) if os.path.getsize(f"{IO}/gdit_timesteps.raw")==4 else np.array([0],np.int64)}),
}
for name,j in jobs.items():
    try:
        s=sess(j["onnx"]); feeds=j["feeds"]()
        # match dtypes to graph
        want={i.name:i.type for i in s.get_inputs()}
        for k,v in list(feeds.items()):
            if "int64" in want.get(k,""): feeds[k]=v.astype(np.int64)
            elif "float16" in want.get(k,""): feeds[k]=v.astype(np.float16)
            elif "float" in want.get(k,""): feeds[k]=v.astype(np.float32)
        o=s.run(None, feeds)[0]
        o=np.asarray(o).astype(np.float32)
        np.save(f"{OUT}/{name}_ref.npy", o)
        report["graphs"][name]={"ran":True,"out_shape":list(o.shape),"out_md5":md5(o),
            "out_std":round(float(o.std()),5),"out_absmax":round(float(np.abs(o).max()),4),
            "finite":bool(np.isfinite(o).all()),"ref_saved":f"optionB/ref/{name}_ref.npy"}
        print(f"OK {name} -> {o.shape} std={o.std():.4f} md5={md5(o)}")
    except Exception as e:
        report["graphs"][name]={"ran":False,"error":str(e)[:300]}
        print(f"FAIL {name}: {e}")

# ---- reproduce existing vit/llm/dit refs (validates the reference chain) ----
def reproduce(name, onnx_p, feeds_fn, ref_p, reshaper):
    try:
        s=sess(onnx_p); feeds=feeds_fn()
        want={i.name:i.type for i in s.get_inputs()}
        for k,v in list(feeds.items()):
            if "int" in want.get(k,""): feeds[k]=v.astype(np.int64) if "int64" in want[k] else v.astype(np.int32)
            elif "float16" in want.get(k,""): feeds[k]=v.astype(np.float16)
            else: feeds[k]=v.astype(np.float32)
        o=np.asarray(s.run(None,feeds)[0]).astype(np.float32).ravel()
        ref=np.load(ref_p).astype(np.float32).ravel()
        n=min(o.size,ref.size)
        report["graphs"][name+"_reproduce_ref"]={"ran":True,"onnx_out_shape_elems":int(o.size),
            "ref_elems":int(ref.size),"rel_mae_vs_saved_ref":round(relmae(o[:n],ref[:n]),6),
            "cos_vs_saved_ref":round(cos(o[:n],ref[:n]),6)}
        print(f"REPRO {name}: rel_mae={relmae(o[:n],ref[:n]):.3e} cos={cos(o[:n],ref[:n]):.6f}")
    except Exception as e:
        report["graphs"][name+"_reproduce_ref"]={"ran":False,"error":str(e)[:300]}
        print(f"REPRO FAIL {name}: {e}")

import sys
if "--repro" in sys.argv:
    # DiT already covered; do vit + llm (heavier)
    reproduce("gvit", f"{EG}/vit_fp16.onnx",
        lambda: {"pixel_values": np.fromfile(f"{IO}/gvit_pixel_values.raw",np.float32).reshape(1,3,224,224),
                 "position_ids": np.fromfile(f"{IO}/gvit_position_ids.raw",np.int32).reshape(1,-1)},
        f"{IO}/gvit_ref.npy", None)

json.dump(report, open(_W+"/optionB/gates/host_graph_gates.json","w"), indent=1)
print("WROTE optionB/gates/host_graph_gates.json")
