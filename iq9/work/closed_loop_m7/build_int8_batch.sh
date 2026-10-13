#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
set -euo pipefail
cd ${GVRB_WORK}/closed_loop_m7
build() { echo "@@@@@ $(date +%T) $1"; bash build_int8.sh "$@" 2>&1 | tail -8; }
build action_encoder_47 ${GVRB_WORK}/optionB/ah_fix/dlc/action_encoder_47.dlc "actions time_proj output"
build vlln_vl_self_attention_47 ${GVRB_WORK}/closed_loop_m4/build/dlc/vlln_vl_self_attention_47.dlc "backbone_features output"
build groot_vit ${GVRB_WORK}/closed_loop_m4/build/dlc/groot_vit.dlc "pixel_values position_ids vit_embeds"
build groot_dit ${GVRB_WORK}/closed_loop_m4/build/dlc/groot_dit.dlc "sa_embs vl_embs timesteps_tensor output"
build groot_llm ${GVRB_WORK}/qnn/groot_llm.dlc "inputs_embeds attention_mask embeddings"
echo "ALL_INT8_DONE $(date +%T)"
