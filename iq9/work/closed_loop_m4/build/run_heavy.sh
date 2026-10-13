#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
set -e
cd ${GVRB_WORK}/closed_loop_m4/build
echo "[heavy] START $(date +%T)"
bash m4_build.sh vlln_vl_self_attention_47 onnx/action_head/vlln_vl_self_attention.onnx -s backbone_features 1,296,2048
echo "[heavy] vlln done $(date +%T)"
bash m4_build.sh groot_vit onnx/eagle2/vit_fp16.onnx -s pixel_values 1,3,224,224 -s position_ids 1,256
echo "[heavy] vit done $(date +%T)"
bash m4_build.sh groot_dit onnx/action_head/DiT_fp16.onnx -s sa_embs 1,49,1536 -s vl_embs 1,296,2048 -s timesteps_tensor 1
echo "[heavy] dit done $(date +%T)"
echo "ALL_HEAVY_DONE"
