#!/bin/bash
# GR00T stage engines on Thor AGX — byte-identical recipe to the Orin run.
# Inputs: the stage-input .raw files written by profiling/compiled_fleet/dump_stage_inputs.py into
# $GVRB_STAGE_IO (default $GVRB_WORK/trtwork/io), and the fp16 ONNX in $GVRB_WORK/onnx (speed/export_onnx.sh).
set -euo pipefail
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ON=${GROOT_ONNX:-${GVRB_WORK}/onnx}
IO=${GVRB_STAGE_IO:-${GVRB_WORK}/trtwork/io}
OUT=${GVRB_WORK}/trtout
mkdir -p "$OUT"
for f in vit_pixel_values vit_position_ids llm_inputs_embeds llm_attention_mask dit_sa_embs dit_vl_embs dit_timesteps_tensor; do
  [ -f "$IO/$f.raw" ] || { echo "missing $IO/$f.raw -- run profiling/compiled_fleet/dump_stage_inputs.py first (profiling/README.md)" >&2; exit 1; }
done

shapes_vit="--minShapes=pixel_values:1x3x224x224,position_ids:1x256 --optShapes=pixel_values:1x3x224x224,position_ids:1x256 --maxShapes=pixel_values:1x3x224x224,position_ids:1x256"
inputs_vit="--loadInputs=pixel_values:$IO/vit_pixel_values.raw,position_ids:$IO/vit_position_ids.raw"
onnx_vit="$ON/eagle2/vit_fp16.onnx"

shapes_llm="--minShapes=inputs_embeds:1x296x2048,attention_mask:1x296 --optShapes=inputs_embeds:1x296x2048,attention_mask:1x296 --maxShapes=inputs_embeds:1x296x2048,attention_mask:1x296"
inputs_llm="--loadInputs=inputs_embeds:$IO/llm_inputs_embeds.raw,attention_mask:$IO/llm_attention_mask.raw"
onnx_llm="$ON/eagle2/llm_fp16.onnx"

shapes_dit="--minShapes=sa_embs:1x49x1536,vl_embs:1x296x2048,timesteps_tensor:1 --optShapes=sa_embs:1x49x1536,vl_embs:1x296x2048,timesteps_tensor:1 --maxShapes=sa_embs:1x49x1536,vl_embs:1x296x2048,timesteps_tensor:1"
inputs_dit="--loadInputs=sa_embs:$IO/dit_sa_embs.raw,vl_embs:$IO/dit_vl_embs.raw,timesteps_tensor:$IO/dit_timesteps_tensor.raw"
onnx_dit="$ON/action_head/DiT_fp16.onnx"

FAILED=0
run_one () {   # stage variant extraflags
  s=$1; v=$2; shift 2
  tag=${s}_${v}
  eval onnx=\$onnx_$s; eval shp=\"\$shapes_$s\"; eval inp=\"\$inputs_$s\"
  echo "########## $tag ##########"
  rc=0
  # shellcheck disable=SC2086  # $shp / $inp hold several flags each
  $TRTEXEC --onnx="$onnx" $shp \
    --saveEngine="$OUT/$tag.engine" --exportLayerInfo="$OUT/$tag.layers.json" \
    --profilingVerbosity=detailed --exportOutput="$OUT/$tag.out.json" \
    --exportTimes="$OUT/$tag.times.json" \
    --iterations=50 --warmUp=500 --avgRuns=50 \
    $inp "$@" > "$OUT/$tag.log" 2>&1 || rc=$?
  echo "rc=$rc  $(grep -m1 'GPU Compute Time' "$OUT/$tag.log" || true)"
  grep -m1 -E '&&&& (PASSED|FAILED)' "$OUT/$tag.log" || true
  if [ $rc -ne 0 ]; then
    echo "ERROR: trtexec exited $rc for $tag (log: $OUT/$tag.log)" >&2
    grep -iE '\[E\]|error' "$OUT/$tag.log" | head -10 || true
    FAILED=$((FAILED+1))
  fi
}

for s in vit llm dit; do
  run_one $s st5  --stronglyTyped --builderOptimizationLevel=5
  run_one $s fp16 --fp16
  run_one $s st   --stronglyTyped
done
if [ $FAILED -ne 0 ]; then echo "ERROR: $FAILED trtexec run(s) failed" >&2; exit 1; fi
echo ALL_DONE
