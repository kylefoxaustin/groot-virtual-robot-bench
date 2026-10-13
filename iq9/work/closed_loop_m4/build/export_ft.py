#!/usr/bin/env python3
"""M4 export: export the 6 CHANGED GR00T stage graphs from the FINE-TUNED checkpoint
(gr00t-n1.5-robocasa-tabletop-posttrain) to fp16 ONNX, matching the base export's shapes.

CHANGED graphs (from base-vs-ft weight diff): ViT, vlln_vl_self_attention, state_encoder,
action_encoder, DiT, action_decoder. LLM is byte-identical (reuse base bin) so SKIPPED here.

Reuses NVIDIA's export_onnx.export_eagle2_vit / export_action_head verbatim (fp16 route).
Forces attention_mask seq_len = 296 and state = (1,1,64) so the exported ONNX bake the SAME
concrete shapes as the base bins (drop-in for the same board loop + GrootPrep).
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import os, sys, time
import numpy as np
import torch

N15 = _gvrb_os.environ.get("GROOT_REPO", _W+"/Isaac-GR00T")
sys.path.insert(0, N15)
sys.path.insert(0, os.path.join(N15, "deployment_scripts"))

OUT = _W+"/closed_loop_m4/build/onnx"
FT = _gvrb_os.environ.get("GROOT_FT_MODEL", _W+"/models/gr00t-n1.5-robocasa-tabletop-posttrain")
DATA_CONFIG = "fourier_gr1_arms_waist"
EMB = "gr1"
DENOISE = 4
N_TOKENS = 296  # match base (capture_config.json N_tokens=296)

os.makedirs(os.path.join(OUT, "eagle2"), exist_ok=True)
os.makedirs(os.path.join(OUT, "action_head"), exist_ok=True)

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
import export_onnx as EO

print(f"[M4-EXPORT] loading FT policy {FT}", flush=True)
t0 = time.time()
dc = load_data_config(DATA_CONFIG)
policy = Gr00tPolicy(model_path=FT, modality_config=dc.modality_config(),
                     modality_transform=dc.transform(), embodiment_tag=EMB,
                     denoising_steps=DENOISE, device="cuda")
print(f"[M4-EXPORT] loaded {time.time()-t0:.1f}s", flush=True)

# shape-only stand-ins (export functions only read .shape[1]/.shape[2])
attention_mask = torch.ones((1, N_TOKENS), dtype=torch.int64).cuda()
state = torch.zeros((1, 1, policy.model.action_head.config.max_state_dim), dtype=torch.float16).cuda()
print(f"[M4-EXPORT] attn seq={attention_mask.shape[1]} state={tuple(state.shape)}", flush=True)

print("[M4-EXPORT] === ViT (fp16) ===", flush=True)
t0 = time.time()
EO.export_eagle2_vit(
    policy.model.backbone.eagle_model.vision_model.vision_model,
    OUT, vit_dtype="fp16", modality_configs=None, embodiment_tag=EMB,
    policy=policy, denoising_steps=DENOISE, data_config=DATA_CONFIG, model_path=FT)
print(f"[M4-EXPORT] ViT done {time.time()-t0:.1f}s", flush=True)

print("[M4-EXPORT] === action_head (vlln, state_enc, action_enc, DiT, action_dec) fp16 ===", flush=True)
t0 = time.time()
EO.export_action_head(
    policy, OUT, state, attention_mask, dit_dtype="fp16",
    modality_configs=None, embodiment_tag=EMB, denoising_steps=DENOISE,
    data_config=DATA_CONFIG, model_path=FT)
print(f"[M4-EXPORT] action_head done {time.time()-t0:.1f}s", flush=True)

# also dump the FT glue constants (mlp1_0 weight/bias, future_tokens, position_embedding)
# straight from the loaded policy — these CHANGED vs base (embed_tokens is UNCHANGED -> reuse base)
gdir = _W+"/closed_loop_m4/build/glue_constants"
os.makedirs(gdir, exist_ok=True)
ah = policy.model.action_head
eagle = policy.model.backbone.eagle_model
mlp1 = eagle.mlp1
# mlp1 is Sequential([Linear]) per capture_config
lin = mlp1[0] if hasattr(mlp1, "__getitem__") else mlp1
np.save(f"{gdir}/mlp1_0_weight.npy", lin.weight.detach().float().cpu().numpy())
np.save(f"{gdir}/mlp1_0_bias.npy", lin.bias.detach().float().cpu().numpy())
np.save(f"{gdir}/ah_future_tokens_weight.npy", ah.future_tokens.weight.detach().float().cpu().numpy())
np.save(f"{gdir}/ah_position_embedding_weight.npy", ah.position_embedding.weight.detach().float().cpu().numpy())
print("[M4-EXPORT] FT glue constants dumped:", os.listdir(gdir), flush=True)
print("[M4-EXPORT] DONE", flush=True)
