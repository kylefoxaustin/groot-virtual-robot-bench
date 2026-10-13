#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
set -euo pipefail
cd ${GVRB_WORK}/closed_loop_m7
b(){ echo "@@ $(date +%T) $1"; bash build_int8_pctl.sh "$@" 2>&1 | tail -4; }
b state_encoder_47 ${GVRB_WORK}/optionB/ah_fix/dlc/state_encoder_47.dlc "state output"
b action_encoder_47 ${GVRB_WORK}/optionB/ah_fix/dlc/action_encoder_47.dlc "actions time_proj output"
b groot_dit ${GVRB_WORK}/closed_loop_m4/build/dlc/groot_dit.dlc "sa_embs vl_embs timesteps_tensor output"
b action_decoder_47 ${GVRB_WORK}/optionB/ah_fix/dlc/action_decoder_47.dlc "model_output output"
echo "HEAD_PCTL_DONE $(date +%T)"
