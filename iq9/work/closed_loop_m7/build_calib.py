#!/usr/bin/env python3
"""M7 calibration builder: run the fp16 GR00T ONNX pipeline (SAME graphs the board
bins are built from) over the REAL captured in-loop frames, dumping per-graph INPUT
tensors + input_list files for qairt-quantizer int8 calibration.

Body graphs (vit/llm/vlln/state): one calib sample per frame.
Head graphs (aenc/dit/adec): one sample per (frame, denoise-step) for the first
HEAD_FRAMES frames -> covers the flow-matching action-token distribution across t.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import os, sys, glob, json
import numpy as np
import onnxruntime as ort

M4B = _W+"/closed_loop_m4/build"
FTO = f"{M4B}/onnx"
CAL = _W+"/closed_loop_m7/calib_raws"
OUT = _W+"/closed_loop_m7/int8_build/calib"
TPDIR = _W+"/optionB/ah_fix/io"
NOISE = np.load(_gvrb_os.environ.get("GVRB_NOISE_BANK", _W+"/noise_bank.npy"))   # (20,50,1,16,32)
IMAGE_TOKEN = 151669
HEAD_FRAMES = 20
os.environ["GROOT_GLUE"] = f"{M4B}/glue_constants"
sys.path.insert(0, _W+"/optionB/src")
from host_glue_groot import GrootGlue

PROV = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
def sess(p): return ort.InferenceSession(p, so, providers=PROV)
print("loading sessions...", flush=True)
s_vit  = sess(f"{FTO}/eagle2/vit_fp16.onnx")
s_llm  = sess(_gvrb_os.environ.get("GROOT_ONNX", _W+"/onnx")+"/eagle2/llm_fp16.onnx")
s_vlln = sess(f"{FTO}/action_head/vlln_vl_self_attention.onnx")
s_state= sess(f"{FTO}/action_head_baked/state_encoder_baked.onnx")
s_aenc = sess(f"{FTO}/action_head_baked/action_encoder_cf_baked.onnx")
s_adec = sess(f"{FTO}/action_head_baked/action_decoder_baked.onnx")
s_dit  = sess(f"{FTO}/action_head/DiT_fp16.onnx")
g = GrootGlue()
TP = {td: np.fromfile(f"{TPDIR}/time_proj_t{td}.raw", np.float32).reshape(1,16,1536)
      for td in (0,250,500,750)}

GRAPHS = ["groot_vit","groot_llm","vlln_vl_self_attention_47","state_encoder_47",
          "action_encoder_47","groot_dit","action_decoder_47"]
for gg in GRAPHS:
    os.makedirs(f"{OUT}/{gg}", exist_ok=True)
lists = {gg: [] for gg in GRAPHS}

def dump(gr, idx, arrs):
    """arrs: dict name-> (np, dtype). returns the input_list line."""
    parts = []
    for name,(a,dt) in arrs.items():
        p = f"{OUT}/{gr}/{name}_{idx:03d}.raw"
        np.ascontiguousarray(a, dt).tofile(p)
        parts.append(f"{name}:={p}")
    lists[gr].append(" ".join(parts))

frames = sorted(glob.glob(f"{CAL}/frame_*"))
print(f"{len(frames)} frames", flush=True)
for fi, fd in enumerate(frames):
    meta = json.load(open(f"{fd}/meta.json"))
    ep, call = meta["ep"], meta["call"]
    pixel = np.fromfile(f"{fd}/pixel_values.raw", np.float32).reshape(1,3,224,224)
    position_ids = np.fromfile(f"{fd}/position_ids.raw", np.float32).astype(np.int64).reshape(1,256)
    input_ids = np.fromfile(f"{fd}/input_ids.raw", np.int64).reshape(1,-1)
    attn = np.fromfile(f"{fd}/attention_mask.raw", np.float32).astype(np.int64).reshape(1,-1)
    state = np.fromfile(f"{fd}/state.raw", np.float32).reshape(1,1,64)
    init = np.ascontiguousarray(NOISE[ep, call], np.float32)  # (1,16,32)

    # --- body ---
    dump("groot_vit", fi, {"pixel_values": (pixel, np.float32), "position_ids": (position_ids, np.int32)})
    vit = s_vit.run(None, {"pixel_values": pixel.astype(np.float16), "position_ids": position_ids})[0].astype(np.float32)
    proj = g.mlp1_project(vit.reshape(1,256,1152))
    ie = g.embed_and_scatter(input_ids, proj, IMAGE_TOKEN)          # (1,N,2048)
    dump("groot_llm", fi, {"inputs_embeds": (ie, np.float32), "attention_mask": (attn, np.int32)})
    backbone = s_llm.run(None, {"inputs_embeds": ie.astype(np.float16), "attention_mask": attn})[0].astype(np.float32)
    dump("vlln_vl_self_attention_47", fi, {"backbone_features": (backbone, np.float32)})
    vl = s_vlln.run(None, {"backbone_features": backbone.astype(np.float16)})[0].astype(np.float32)
    dump("state_encoder_47", fi, {"state": (state, np.float32)})
    sf = s_state.run(None, {"state": state.astype(np.float16)})[0].astype(np.float32)

    # --- head denoise loop ---
    if fi < HEAD_FRAMES:
        actions = init.copy()
        for s in range(g.num_steps):
            td = g.t_discretized(s)                # 0,250,500,750
            tp = TP[td]
            idx = fi*g.num_steps + s
            dump("action_encoder_47", idx, {"actions": (actions, np.float32), "time_proj": (tp, np.float32)})
            af = s_aenc.run(None, {"actions": actions.astype(np.float16), "time_proj": tp.astype(np.float16)})[0].astype(np.float32)
            af2 = g.add_action_pos_embed(af.reshape(1,16,1536))
            sa = g.assemble_sa_embs(sf.reshape(1,1,1536), af2)      # (1,49,1536)
            ts = np.array([td], np.int64)
            dump("groot_dit", idx, {"sa_embs": (sa, np.float32), "vl_embs": (vl, np.float32),
                                    "timesteps_tensor": (ts, np.int32)})
            mo = s_dit.run(None, {"sa_embs": sa.astype(np.float16), "vl_embs": vl.astype(np.float16),
                                  "timesteps_tensor": ts})[0].astype(np.float32)
            dump("action_decoder_47", idx, {"model_output": (mo, np.float32)})
            pred = s_adec.run(None, {"model_output": mo.astype(np.float16)})[0].astype(np.float32)
            actions = g.euler(actions, pred.reshape(1,49,32))
    if fi % 5 == 0:
        print(f"frame {fi}/{len(frames)} done", flush=True)

for gg in GRAPHS:
    with open(f"{OUT}/{gg}/input_list.txt", "w") as f:
        f.write("\n".join(lists[gg]) + "\n")
    print(f"{gg}: {len(lists[gg])} calib samples", flush=True)
print("DONE_BUILD_CALIB")
