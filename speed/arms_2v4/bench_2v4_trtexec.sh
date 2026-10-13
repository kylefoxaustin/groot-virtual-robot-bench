#!/usr/bin/env bash
# 2-arm vs 4-arm action-head timing with trtexec only (for a board without torch in the host env).
# Published Thor numbers used this: 3 fresh builds per (stage, dim), GPU Compute Time mean.
# usage: GROOT_ONNX=... bash bench_2v4_trtexec.sh [N_BUILDS=3] > thor_2v4.txt
set -euo pipefail
: "${GROOT_ONNX:?source setup/env.sh first}"
NB=${1:-3}; AH="$GROOT_ONNX/action_head"; W=$(mktemp -d); trap 'rm -rf "${W:?}"' EXIT
for g in action_encoder action_encoder_4arm action_decoder action_decoder_4arm; do
  [ -f "$AH/$g.onnx" ] || { echo "missing $AH/$g.onnx -- run speed/export_onnx.sh, then speed/arms_2v4/widen_action_head_onnx.py \$GROOT_ONNX" >&2; exit 1; }
done
shapes() {
  case "$1" in
    action_encoder)      echo "actions:1x16x32,timesteps_tensor:1,embodiment_id:1" ;;
    action_encoder_4arm) echo "actions:1x16x64,timesteps_tensor:1,embodiment_id:1" ;;
    action_decoder|action_decoder_4arm) echo "model_output:1x49x1024,embodiment_id:1" ;;
  esac
}
for g in action_encoder action_encoder_4arm action_decoder action_decoder_4arm; do
  for b in $(seq 1 "$NB"); do
    s=$(shapes "$g")
    rc=0
    trtexec --onnx="$AH/$g.onnx" --fp16 --minShapes="$s" --optShapes="$s" --maxShapes="$s" \
         --iterations=2000 --avgRuns=2000 --noDataTransfers --useSpinWait --saveEngine="$W/$g.$b.engine" > "$W/$g.$b.log" 2>&1 || rc=$?
    if [ $rc -ne 0 ]; then
      echo "ERROR: trtexec exited $rc for $g build $b; last lines of its log:" >&2; tail -5 "$W/$g.$b.log" >&2; exit $rc
    fi
    ms=$(awk '/GPU Compute Time:/ {for(i=1;i<=NF;i++) if($i=="mean") {print $(i+2); exit}}' "$W/$g.$b.log")
    [ -n "$ms" ] || { echo "ERROR: no 'GPU Compute Time' line in the trtexec log for $g build $b" >&2; exit 1; }
    echo "$g build$b gpu_compute_mean_ms=$ms"
  done
done
