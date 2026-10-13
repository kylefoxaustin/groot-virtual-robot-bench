#!/usr/bin/env python3
"""M4 host reference = the board FUNCTION on host (FT fp16 ONNX + FT numpy glue) run on the
SAME real board-input raws the base bins were gated with (optionB/io/*.raw — checkpoint-
independent observation + seeded noise). Produces per-graph host refs + host_e2e action for
the pre-loop board gate. Also a zeroed-pixel null e2e for the null baseline.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import os, sys, json, numpy as np
import onnxruntime as ort

BUILD = _W+"/closed_loop_m4/build"
FTO   = f"{BUILD}/onnx"
IO    = _W+"/optionB/io"          # reuse base board-input raws (obs + noise)
REF   = f"{BUILD}/ref_ft"; os.makedirs(REF, exist_ok=True)
os.environ["GROOT_GLUE"] = f"{BUILD}/glue_constants"   # FT glue
sys.path.insert(0, _W+"/optionB/src")
from host_glue_groot import GrootGlue

IMAGE_TOKEN = 151669
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
def sess(p): return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
s_vit  = sess(f"{FTO}/eagle2/vit_fp16.onnx")
s_vlln = sess(f"{FTO}/action_head/vlln_vl_self_attention.onnx")
s_state= sess(f"{FTO}/action_head/state_encoder.onnx")
s_aenc = sess(f"{FTO}/action_head/action_encoder.onnx")
s_adec = sess(f"{FTO}/action_head/action_decoder.onnx")
s_dit  = sess(f"{FTO}/action_head/DiT_fp16.onnx")
# LLM: reuse the base bin => host ref uses base LLM onnx (weights identical)
s_llm  = sess(_gvrb_os.environ.get("GROOT_ONNX", _W+"/onnx")+"/eagle2/llm_fp16.onnx")
g = GrootGlue()

pixel = np.fromfile(f"{IO}/pixel_values.raw", np.float32).reshape(1,3,224,224)
position_ids = np.fromfile(f"{IO}/position_ids.raw", np.float32).astype(np.int64).reshape(1,256)
input_ids = np.fromfile(f"{IO}/input_ids.raw", np.int64).reshape(1,-1)
attention_mask = np.fromfile(f"{IO}/attention_mask.raw", np.float32).astype(np.int64).reshape(1,-1)
state = np.fromfile(f"{IO}/state.raw", np.float32).reshape(1,1,64)
eid = np.array([24], np.int64)
init_actions = np.fromfile(f"{IO}/init_actions.raw", np.float32).reshape(1,16,32)
N = input_ids.shape[1]
print(f"inputs: N={N} pixel{pixel.shape} state{state.shape} init|std|={init_actions.std():.4f}")

def run_e2e(pix, save_pergraph=False):
    vit = s_vit.run(None, {"pixel_values": pix.astype(np.float16),
                           "position_ids": position_ids})[0].astype(np.float32)
    proj = g.mlp1_project(vit.reshape(1,256,1152))
    ie = g.embed_and_scatter(input_ids, proj, IMAGE_TOKEN)
    backbone = s_llm.run(None, {"inputs_embeds": ie.astype(np.float16),
                                "attention_mask": attention_mask})[0].astype(np.float32)
    vl = s_vlln.run(None, {"backbone_features": backbone.astype(np.float16)})[0].astype(np.float32)
    sf = s_state.run(None, {"state": state.astype(np.float16), "embodiment_id": eid})[0].astype(np.float32)
    if save_pergraph:
        np.save(f"{REF}/vit.npy", vit); np.save(f"{REF}/vlln.npy", vl); np.save(f"{REF}/state.npy", sf)
    actions = init_actions.copy()
    for s in range(g.num_steps):
        td = np.array([g.t_discretized(s)], np.int64)
        af = s_aenc.run(None, {"actions": actions.astype(np.float16), "timesteps_tensor": td,
                               "embodiment_id": eid})[0].astype(np.float32)
        af2 = g.add_action_pos_embed(af.reshape(1,16,1536))
        sa = g.assemble_sa_embs(sf.reshape(1,1,1536), af2)
        mo = s_dit.run(None, {"sa_embs": sa.astype(np.float16), "vl_embs": vl.astype(np.float16),
                              "timesteps_tensor": td})[0].astype(np.float32)
        pred = s_adec.run(None, {"model_output": mo.astype(np.float16), "embodiment_id": eid})[0].astype(np.float32)
        if save_pergraph and s == g.num_steps-1:   # save last-step per-graph (matches board's surviving Result_0)
            np.save(f"{REF}/aenc.npy", af); np.save(f"{REF}/dit.npy", mo); np.save(f"{REF}/adec.npy", pred)
        actions = g.euler(actions, pred.reshape(1,49,32))
    return actions

host_e2e = run_e2e(pixel, save_pergraph=True)
np.save(f"{REF}/host_e2e_ft.npy", host_e2e)
null_e2e = run_e2e(np.zeros_like(pixel))
np.save(f"{REF}/host_e2e_ft_null.npy", null_e2e)
rel_null = float(np.mean(np.abs(host_e2e-null_e2e))/(np.mean(np.abs(host_e2e))+1e-12))
print(f"host_e2e |std|={host_e2e.std():.4f} ; host vs zeroed-pixel null rel_mae={rel_null:.4f} (sensitivity)")
print("saved refs:", sorted(os.listdir(REF)))
print("DONE_M4_HOST_REF")
