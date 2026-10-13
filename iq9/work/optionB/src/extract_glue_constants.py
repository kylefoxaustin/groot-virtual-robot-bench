#!/usr/bin/env python3
"""Extract the torch-less HOST-GLUE constant tensors GR00T needs on the board.

The board has NO torch. The reference loop (trt_model_forward.py / action_head_utils.py)
runs several ops as host torch that must become numpy-with-constants on the board:
  - eagle_model.mlp1        (ViT-embeds -> LLM-dim projection, after pixel_shuffle)
  - embed_tokens.weight     (input_ids -> input_embeds lookup, then image-token scatter)
  - action_head.future_tokens.weight        (concatenated into sa_embs each denoise step)
  - action_head.position_embedding.weight   (add_pos_embed on action_features)
Extracted straight from the safetensors (no torch/cuda import needed) to .npy (fp32).
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json, numpy as np, os, torch
from safetensors import safe_open

CKPT=_gvrb_os.environ.get("GROOT_MODEL", _W+"/models/GR00T-N1.5-3B")
OUT=_W+"/optionB/glue_constants"
os.makedirs(OUT, exist_ok=True)
idx=json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
WANT=["backbone.eagle_model.mlp1.0.weight","backbone.eagle_model.mlp1.0.bias",
      "backbone.eagle_model.language_model.model.embed_tokens.weight",
      "action_head.future_tokens.weight","action_head.position_embedding.weight"]
handles={}
manifest={}
for key in WANT:
    shard=idx.get(key)
    if shard is None:
        print("MISSING",key); manifest[key]={"missing":True}; continue
    if shard not in handles:
        handles[shard]=safe_open(f"{CKPT}/{shard}","numpy")
    t=handles[shard].get_tensor(key)
    arr=np.asarray(t).astype(np.float32)
    fn=key.replace("backbone.eagle_model.","").replace("language_model.model.","").replace("action_head.","ah_").replace(".","_")+".npy"
    np.save(f"{OUT}/{fn}", arr)
    manifest[key]={"file":fn,"shape":list(arr.shape),"dtype_src":str(t.dtype),
                   "std":round(float(arr.std()),5),"absmax":round(float(np.abs(arr).max()),4)}
    print(f"{key:70s} {str(list(arr.shape)):22s} -> {fn}")
json.dump(manifest, open(f"{OUT}/manifest.json","w"), indent=1)
print("WROTE", OUT+"/manifest.json")
