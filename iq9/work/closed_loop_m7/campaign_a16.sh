#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
set -euo pipefail
# W8A16 closed-loop arms: 20 seeds each, sequential (one board/GPU).
# Per-episode files written incrementally by m4_rollout, so a mid-run kill loses only the
# in-flight episode. resident_orch is closed cleanly (SIGINT) by m7_run — never SIGKILL.
cd ${GVRB_WORK}/closed_loop_m7
mkdir -p logs runs/work
PY=${GROOT_PY:-${GVRB_WORK}/venv-groot/bin/python}
run(){ arm=$1; n=$2; port=$3;
  echo "@@@@@ $(date +%T) START $arm ($n seeds)";
  rc=0; $PY m7_run.py --arm $arm --n_episodes $n --base_seed 0 --port $port \
    --work runs/work/$arm --out_dir runs/$arm > logs/run_$arm.log 2>&1 || rc=$?;
  echo "@@@@@ $(date +%T) DONE $arm rc=$rc -> $(grep -o '"n_success"[^,]*' runs/$arm/rollout_summary.json 2>/dev/null || true)";
  [ "$rc" -eq 0 ] || { echo "arm $arm failed; see logs/run_$arm.log" >&2; exit "$rc"; }
  sleep 5; }
run a16head 20 5591
run a16all  20 5592
echo "CAMPAIGN_A16_DONE $(date +%T)"
