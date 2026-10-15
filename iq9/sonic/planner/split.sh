#!/bin/bash
# (x86 host) Upstream planner_sonic.onnx -> 4 NPU blocks (S, R, P, D) + 5 host stages (H0..H4) -> rewrites -> CPU checks ->
# v73 fp16 context bins. Work dir: $IQ9_SONIC_WORK/planner (every script here runs with that as its current directory).
# The planner's data-dependent ops (ArgMax, TopK, OneHot, CumSum, ScatterND, the seed hash) stay on the CPU in the host stages;
# the transformer/conv bulk goes to the NPU blocks. Needs python3 with onnx + onnxruntime + numpy, and the QAIRT SDK.
#   split.sh [<planner refset.npz>]   with the refset (sonic/gate/make_refsets.py planner ...) it also writes manifest + calls
set -e
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}; W=$IQ9_SONIC_WORK/planner; mkdir -p $W/blocks; cd $W
ln -sf $SONIC_DEPLOY/planner/target_vel/V2/planner_sonic.onnx planner_sonic.onnx
$PY $HERE/dump_vals.py                 # every intermediate on a walk input -> vals_walk.pkl, shapes.pkl
$PY $HERE/build_blocks.py              # blocks/planner_{S,R,P,D}.onnx + blocks.json
$PY $HERE/build_host.py                # blocks/host_H{0..4}.onnx + host.json
$PY $HERE/fix_resize.py R D            # exact rewrite (not the fix; kept so the blocks match ours)
$PY $HERE/fix_dilated.py R D           # the fix: dilated Conv1d -> Pad + 3 Slices + Concat + k=1 Conv
$PY $HERE/test_chain.py                # host stages + fp32 blocks must equal the full planner (maxdiff ~0, npf equal)
for b in S R P D; do bash $HERE/build_ctx.sh $b; done
$PY $HERE/make_manifest.py ${1:-}         # blocks/manifest.txt (+ blocks/planner_calls.f32 with the refset)
