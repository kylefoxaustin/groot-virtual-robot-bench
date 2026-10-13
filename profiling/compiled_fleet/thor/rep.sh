#!/bin/bash
# Repeat build of the canonical --stronglyTyped --builderOptimizationLevel=5 ("st5") engines (build-to-build variance).
# Inputs: see thor_groot.sh ($GVRB_STAGE_IO from dump_stage_inputs.py, fp16 ONNX in $GVRB_WORK/onnx).
set -euo pipefail
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
IO=${GVRB_STAGE_IO:-${GVRB_WORK}/trtwork/io}; OUT=${GVRB_WORK}/trtout
mkdir -p "$OUT"
for f in vit_pixel_values vit_position_ids llm_inputs_embeds llm_attention_mask dit_sa_embs dit_vl_embs dit_timesteps_tensor; do
  [ -f "$IO/$f.raw" ] || { echo "missing $IO/$f.raw -- run profiling/compiled_fleet/dump_stage_inputs.py first (profiling/README.md)" >&2; exit 1; }
done
run () {   # tag, trtexec args...
  tag=$1; shift; rc=0
  "$TRTEXEC" "$@" > "$OUT/$tag.log" 2>&1 || rc=$?
  if [ $rc -ne 0 ]; then echo "ERROR: trtexec exited $rc for $tag (log: $OUT/$tag.log)" >&2; exit $rc; fi
}
run vit_st5c --onnx=${GROOT_ONNX:-${GVRB_WORK}/onnx}/eagle2/vit_fp16.onnx --minShapes=pixel_values:1x3x224x224,position_ids:1x256 --optShapes=pixel_values:1x3x224x224,position_ids:1x256 --maxShapes=pixel_values:1x3x224x224,position_ids:1x256 --saveEngine=$OUT/vit_st5c.engine --exportOutput=$OUT/vit_st5c.out.json --exportLayerInfo=$OUT/vit_st5c.layers.json --profilingVerbosity=detailed --iterations=50 --warmUp=500 --avgRuns=50 --loadInputs=pixel_values:$IO/vit_pixel_values.raw,position_ids:$IO/vit_position_ids.raw --stronglyTyped --builderOptimizationLevel=5
run llm_st5c --onnx=${GROOT_ONNX:-${GVRB_WORK}/onnx}/eagle2/llm_fp16.onnx --minShapes=inputs_embeds:1x296x2048,attention_mask:1x296 --optShapes=inputs_embeds:1x296x2048,attention_mask:1x296 --maxShapes=inputs_embeds:1x296x2048,attention_mask:1x296 --saveEngine=$OUT/llm_st5c.engine --exportOutput=$OUT/llm_st5c.out.json --exportLayerInfo=$OUT/llm_st5c.layers.json --profilingVerbosity=detailed --iterations=50 --warmUp=500 --avgRuns=50 --loadInputs=inputs_embeds:$IO/llm_inputs_embeds.raw,attention_mask:$IO/llm_attention_mask.raw --stronglyTyped --builderOptimizationLevel=5
run dit_st5c --onnx=${GROOT_ONNX:-${GVRB_WORK}/onnx}/action_head/DiT_fp16.onnx --minShapes=sa_embs:1x49x1536,vl_embs:1x296x2048,timesteps_tensor:1 --optShapes=sa_embs:1x49x1536,vl_embs:1x296x2048,timesteps_tensor:1 --maxShapes=sa_embs:1x49x1536,vl_embs:1x296x2048,timesteps_tensor:1 --saveEngine=$OUT/dit_st5c.engine --exportOutput=$OUT/dit_st5c.out.json --exportLayerInfo=$OUT/dit_st5c.layers.json --profilingVerbosity=detailed --iterations=50 --warmUp=500 --avgRuns=50 --loadInputs=sa_embs:$IO/dit_sa_embs.raw,vl_embs:$IO/dit_vl_embs.raw,timesteps_tensor:$IO/dit_timesteps_tensor.raw --stronglyTyped --builderOptimizationLevel=5
grep -h -m1 "GPU Compute Time" "$OUT/vit_st5c.log" "$OUT/llm_st5c.log" "$OUT/dit_st5c.log" || echo "no GPU Compute Time line in the logs (see $OUT/*.log)"
