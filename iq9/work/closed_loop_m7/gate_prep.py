#!/usr/bin/env python3
"""Produce a HELD-OUT per-graph reference input set (all FLOAT32 raws, as the board's
resident_orch / qnn-net-run float path reads them) for the int8-vs-fp16 per-graph gate.
Uses frame_044 (NOT in the head calibration subset) run through the fp16 ONNX pipeline.
Also saves the fp16 host reference OUTPUT of each graph for a 3-way sanity (optional)."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import os, sys, json, numpy as np, onnxruntime as ort
M4B=_W+"/closed_loop_m4/build"; FTO=f"{M4B}/onnx"
CAL=_W+"/closed_loop_m7/calib_raws"
OUT=_W+"/closed_loop_m7/int8_build/gate_in"; os.makedirs(OUT,exist_ok=True)
TPDIR=_W+"/optionB/ah_fix/io"; NOISE=np.load(_gvrb_os.environ.get("GVRB_NOISE_BANK", _W+"/noise_bank.npy"))
IMAGE_TOKEN=151669; os.environ["GROOT_GLUE"]=f"{M4B}/glue_constants"
sys.path.insert(0,_W+"/optionB/src"); from host_glue_groot import GrootGlue
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
P=[("CUDAExecutionProvider",{"device_id":0}),"CPUExecutionProvider"]
def sess(p): return ort.InferenceSession(p,so,providers=P)
s_vit=sess(f"{FTO}/eagle2/vit_fp16.onnx"); s_llm=sess(_gvrb_os.environ.get("GROOT_ONNX", _W+"/onnx")+"/eagle2/llm_fp16.onnx")
s_vlln=sess(f"{FTO}/action_head/vlln_vl_self_attention.onnx")
s_state=sess(f"{FTO}/action_head_baked/state_encoder_baked.onnx")
s_aenc=sess(f"{FTO}/action_head_baked/action_encoder_cf_baked.onnx")
s_adec=sess(f"{FTO}/action_head_baked/action_decoder_baked.onnx")
s_dit=sess(f"{FTO}/action_head/DiT_fp16.onnx"); g=GrootGlue()
def f32(a,p): np.ascontiguousarray(a,np.float32).tofile(f"{OUT}/{p}")
fd=f"{CAL}/frame_044"; meta=json.load(open(f"{fd}/meta.json")); ep,call=meta["ep"],meta["call"]
pixel=np.fromfile(f"{fd}/pixel_values.raw",np.float32).reshape(1,3,224,224)
input_ids=np.fromfile(f"{fd}/input_ids.raw",np.int64).reshape(1,-1)
state=np.fromfile(f"{fd}/state.raw",np.float32).reshape(1,1,64)
position_ids=np.arange(256,dtype=np.int64).reshape(1,256); attn=np.ones((1,296),np.int64)
init=np.ascontiguousarray(NOISE[ep,call],np.float32)
# body (ALL inputs written float32; net-run float-path casts int-target tensors)
f32(pixel,"vit__pixel_values.raw"); f32(position_ids,"vit__position_ids.raw")
vit=s_vit.run(None,{"pixel_values":pixel.astype(np.float16),"position_ids":position_ids})[0].astype(np.float32)
proj=g.mlp1_project(vit.reshape(1,256,1152)); ie=g.embed_and_scatter(input_ids,proj,IMAGE_TOKEN)
f32(ie,"llm__inputs_embeds.raw"); f32(attn,"llm__attention_mask.raw")
backbone=s_llm.run(None,{"inputs_embeds":ie.astype(np.float16),"attention_mask":attn})[0].astype(np.float32)
f32(backbone,"vlln__backbone_features.raw")
vl=s_vlln.run(None,{"backbone_features":backbone.astype(np.float16)})[0].astype(np.float32)
f32(state,"state__state.raw")
sf=s_state.run(None,{"state":state.astype(np.float16)})[0].astype(np.float32)
# head at denoise step 1 (td=250) — a representative mid-trajectory sample
actions=init.copy()
for s in range(2):
    td=g.t_discretized(s); tp=np.fromfile(f"{TPDIR}/time_proj_t{td}.raw",np.float32).reshape(1,16,1536)
    if s==1:
        f32(actions,"aenc__actions.raw"); f32(tp,"aenc__time_proj.raw")
    af=s_aenc.run(None,{"actions":actions.astype(np.float16),"time_proj":tp.astype(np.float16)})[0].astype(np.float32)
    af2=g.add_action_pos_embed(af.reshape(1,16,1536)); sa=g.assemble_sa_embs(sf.reshape(1,1,1536),af2)
    if s==1:
        f32(sa,"dit__sa_embs.raw"); f32(vl,"dit__vl_embs.raw"); f32(np.array([td],np.int64),"dit__timesteps_tensor.raw")
    mo=s_dit.run(None,{"sa_embs":sa.astype(np.float16),"vl_embs":vl.astype(np.float16),"timesteps_tensor":np.array([td],np.int64)})[0].astype(np.float32)
    if s==1:
        f32(mo,"adec__model_output.raw")
    pred=s_adec.run(None,{"model_output":mo.astype(np.float16)})[0].astype(np.float32)
    actions=g.euler(actions,pred.reshape(1,49,32))
print("GATE_IN_DONE", sorted(os.listdir(OUT)))
