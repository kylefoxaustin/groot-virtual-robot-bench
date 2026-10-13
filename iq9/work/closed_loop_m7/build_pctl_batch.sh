#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
set -euo pipefail
cd ${GVRB_WORK}/closed_loop_m7
echo "@@ $(date +%T) vit"; bash build_int8_pctl.sh groot_vit ${GVRB_WORK}/closed_loop_m4/build/dlc/groot_vit.dlc "pixel_values position_ids vit_embeds" 2>&1 | tail -5
echo "@@ $(date +%T) vlln"; bash build_int8_pctl.sh vlln_vl_self_attention_47 ${GVRB_WORK}/closed_loop_m4/build/dlc/vlln_vl_self_attention_47.dlc "backbone_features output" 2>&1 | tail -5
echo "@@ $(date +%T) llm"; bash build_int8_pctl.sh groot_llm ${GVRB_WORK}/qnn/groot_llm.dlc "inputs_embeds attention_mask embeddings" 2>&1 | tail -5
echo "ALL_PCTL_DONE $(date +%T)"
