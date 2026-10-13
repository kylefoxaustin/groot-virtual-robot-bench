#!/usr/bin/env bash
# Export the 7 fp16 ONNX graphs of GR00T N1.5 (NVIDIA's deployment_scripts/export_onnx.py, patched by
# setup/fetch_upstream.sh) into $GROOT_ONNX/{eagle2,action_head}/. Run ONCE on the x86 host; copy
# $GROOT_ONNX to the Jetsons so every board builds its engines from byte-identical ONNX.
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
mkdir -p "$GROOT_ONNX"
cd "$GROOT_REPO"   # the script defaults its dataset to ./demo_data/robot_sim.PickNPlace
"$GROOT_PY" deployment_scripts/export_onnx.py --model-path "$GROOT_MODEL" --onnx-model-path "$GROOT_ONNX" \
  --data-config fourier_gr1_arms_only --embodiment-tag gr1 \
  --llm-dtype fp16 --vit-dtype fp16 --dit-dtype fp16 --denoising-steps 4
ls -la "$GROOT_ONNX"/eagle2 "$GROOT_ONNX"/action_head
echo "md5 of the published export (yours may differ byte-wise under another torch version):"
cat "$GVRB_ROOT/results/speed/onnx_manifest.json"
