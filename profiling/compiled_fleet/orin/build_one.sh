#!/bin/bash
# Build + time + export the output of one GR00T stage engine on Orin with trtexec.
# usage: bash build_one.sh <tag> [extra trtexec args]     tag starts with vit|llm|dit, e.g. vit_st5 --stronglyTyped --builderOptimizationLevel=5
# Inputs: the stage-input .raw files written by profiling/compiled_fleet/dump_stage_inputs.py into
# $GVRB_STAGE_IO (default $GVRB_WORK/trtwork/io), and the fp16 ONNX in $GVRB_WORK/onnx (speed/export_onnx.sh).
set -euo pipefail
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
TAG=${1:?usage: build_one.sh <tag> [extra trtexec args]}; shift
TRTEXEC=/usr/src/tensorrt/bin/trtexec
O=${GVRB_WORK}/trt
IO=${GVRB_STAGE_IO:-${GVRB_WORK}/trtwork/io}
mkdir -p "$O"
case "$TAG" in
  vit*) ONNX=${GROOT_ONNX:-${GVRB_WORK}/onnx}/eagle2/vit_fp16.onnx
        SH="pixel_values:1x3x224x224,position_ids:1x256"
        LI="pixel_values:$IO/vit_pixel_values.raw,position_ids:$IO/vit_position_ids.raw";;
  llm*) ONNX=${GROOT_ONNX:-${GVRB_WORK}/onnx}/eagle2/llm_fp16.onnx
        SH="inputs_embeds:1x296x2048,attention_mask:1x296"
        LI="inputs_embeds:$IO/llm_inputs_embeds.raw,attention_mask:$IO/llm_attention_mask.raw";;
  dit*) ONNX=${GROOT_ONNX:-${GVRB_WORK}/onnx}/action_head/DiT_fp16.onnx
        SH="sa_embs:1x49x1536,vl_embs:1x296x2048,timesteps_tensor:1"
        LI="sa_embs:$IO/dit_sa_embs.raw,vl_embs:$IO/dit_vl_embs.raw,timesteps_tensor:$IO/dit_timesteps_tensor.raw";;
  *) echo "unknown tag (must start with vit|llm|dit)" >&2; exit 2;;
esac
for f in ${LI//,/ }; do
  f=${f#*:}; [ -f "$f" ] || { echo "missing $f -- run profiling/compiled_fleet/dump_stage_inputs.py first (profiling/README.md)" >&2; exit 1; }
done
rc=0
$TRTEXEC --onnx="$ONNX" \
  --minShapes="$SH" --optShapes="$SH" --maxShapes="$SH" \
  --saveEngine="$O/$TAG.engine" \
  --exportLayerInfo="$O/$TAG.layers.json" --profilingVerbosity=detailed \
  --exportOutput="$O/$TAG.out.json" --exportTimes="$O/$TAG.times.json" \
  --iterations=50 --warmUp=500 --avgRuns=50 \
  --loadInputs="$LI" "$@" 2>&1 | tee "$O/$TAG.log" || rc=$?
if [ $rc -ne 0 ]; then echo "ERROR: trtexec exited $rc for $TAG (log: $O/$TAG.log)" >&2; exit $rc; fi
