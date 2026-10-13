#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
: "${IQ9_WORK:=/opt/gvrb}"
: "${IQ9_HOST:=iq9}"
set -euo pipefail
cd ${GVRB_WORK}/closed_loop_m7
PY=${GROOT_PY:-${GVRB_WORK}/venv-groot/bin/python}
echo "=== waiting for campaign_final (a16head+fp16) to finish ==="
until grep -q "CAMPAIGN_FINAL_DONE" logs/campaign_final.log 2>/dev/null; do sleep 20; done
echo "=== campaign done; building per-row W8A16 head (sequential) ==="
b(){ echo "@@ $(date +%T) $1"; bash build_a16_perrow.sh "$@" 2>&1 | tail -3; }
b state_encoder_47  ${GVRB_WORK}/optionB/ah_fix/dlc/state_encoder_47.dlc  "state output"
b action_encoder_47 ${GVRB_WORK}/optionB/ah_fix/dlc/action_encoder_47.dlc "actions time_proj output"
b action_decoder_47 ${GVRB_WORK}/optionB/ah_fix/dlc/action_decoder_47.dlc "model_output output"
b groot_dit         ${GVRB_WORK}/closed_loop_m4/build/dlc/groot_dit.dlc   "sa_embs vl_embs timesteps_tensor output"
echo "=== deploy per-row head bins ==="
for g in state_encoder_47 action_encoder_47 action_decoder_47 groot_dit; do
  scp -q int8_build/ctx/${g}_a16pr.bin "${IQ9_HOST}:${IQ9_WORK}/optionB/ctx_int8/"; echo "pushed ${g}_a16pr.bin"
done
echo "=== run prhead (fp16 body + per-row W8A16 head) 20 seeds ==="
$PY m7_run.py --arm prhead --n_episodes 20 --base_seed 0 --port 5597 --work runs/work/prhead --out_dir runs/prhead > logs/run_prhead.log 2>&1
echo "PRHEAD_DONE $(date +%T) -> $(grep -o '"n_success"[^,]*' runs/prhead/rollout_summary.json 2>/dev/null || true)"
