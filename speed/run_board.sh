#!/usr/bin/env bash
# Speed benchmarks on one board: build TWO independent sets of the 7 fp16 TensorRT engines, run the
# eager / hybrid / full-TensorRT end-to-end A/B on each, fold both into one record.
#   bash speed/run_board.sh 5090|thor|orin ["clock note"]
# Thor: run inside the container --  bash setup/thor_container.sh 'bash speed/run_board.sh thor "jetson_clocks MAXN"'
# Lock clocks first if you can (5090: sudo nvidia-smi -lgc/-lmc; Jetson: sudo jetson_clocks) and say so in the note.
# Needs: the fp16 ONNX in $GROOT_ONNX (speed/export_onnx.sh) and the input step $GROOT_STEP_NPZ (speed/make_step0.py).
# env: GROOT_PY (python with torch + tensorrt), ALLOW_DATASET_INPUT=1 to read PickNPlace ds[0] directly instead of the npz (x86, needs decord).
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
B="${1:?label (free text used in file names, e.g. 5090, thor, orin, mygpu)}"; NOTE="${2:-not recorded}"; PY="${GROOT_PY:-python3}"
OUT="$GVRB_WORK/speed/$B"; mkdir -p "$OUT"
for f in eagle2/vit_fp16.onnx eagle2/llm_fp16.onnx action_head/vlln_vl_self_attention.onnx action_head/state_encoder.onnx \
         action_head/action_encoder.onnx action_head/DiT_fp16.onnx action_head/action_decoder.onnx; do
  [ -f "$GROOT_ONNX/$f" ] || { echo "missing $GROOT_ONNX/$f -- run \`bash speed/export_onnx.sh\` and \`\$GROOT_PY speed/make_step0.py \$GROOT_STEP_NPZ\` first (docs/SPEED.md)" >&2; exit 1; }
done
if [ ! -f "$GROOT_STEP_NPZ" ]; then
  if [ "${ALLOW_DATASET_INPUT:-0}" = 1 ]; then   # x86 only: read PickNPlace ds[0] directly (needs decord)
    echo "note: $GROOT_STEP_NPZ missing, ALLOW_DATASET_INPUT=1 -> the harness reads PickNPlace ds[0] directly"; unset GROOT_STEP_NPZ
  else
    echo "missing $GROOT_STEP_NPZ -- run \`\$GROOT_PY speed/make_step0.py \$GROOT_STEP_NPZ\` first (docs/SPEED.md)" >&2; exit 1
  fi
fi
for b in A B; do
  [ -f "$OUT/eng$b/action_decoder.engine" ] || "$PY" "$GVRB_ROOT/speed/build_engines.py" "$OUT/eng$b" 296
  "$PY" "$GVRB_ROOT/speed/e2e_eager_hybrid_fulltrt.py" "$OUT/eng$b" "$OUT/res_${B}_$b.json"
done
"$PY" "$GVRB_ROOT/speed/make_e2e_record.py" "$B" "$OUT/res_${B}_A.json" "$OUT/res_${B}_B.json" "$OUT/${B}_fulltrt_e2e.json" "$NOTE"
REF="$GVRB_ROOT/results/speed/groot_${B}_fulltrt_e2e_2026-10-12.json"; [ -f "$REF" ] || REF="results/speed/groot_{5090,thor,orin}_fulltrt_e2e_2026-10-12.json"
echo "record -> $OUT/${B}_fulltrt_e2e.json   (compare with $REF)"
