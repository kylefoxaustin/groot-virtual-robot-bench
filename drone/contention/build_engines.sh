#!/usr/bin/env bash
# Build the stand-in TensorRT engines (fp16) from $DRONE_STANDINS/*.onnx (drone/contention/make_standins.py), with the
# flags our published runs used:
#   x86 (RTX 5090) : build_trt.py (TensorRT Python API, FP16 flag, 4 GiB workspace) for all three
#   thor           : trtexec  seg_512 --fp16 --builderOptimizationLevel=2 ; seg_1024 --fp16 ; radar_fusion --fp16
#   orin           : trtexec  seg_512 and seg_1024 --fp16 --builderOptimizationLevel=2 ; radar_fusion --fp16
#   bash drone/contention/build_engines.sh x86|thor|orin
# env: WORKER_PY (python with TensorRT 10.x; default $DRONE_WORK/venv-workers from setup_workers_x86.sh), TRTEXEC (default
#      /usr/src/tensorrt/bin/trtexec). TensorRT 11 (the GR00T venv on x86) cannot build these: it dropped the FP16 builder flag.
# We used TensorRT 10.16.1 (x86, system python), 10.16.2 (Thor, JetPack) and 10.3.0 (Orin, JetPack) for these engines.
set -euo pipefail
cd "$(dirname "$0")/../.."; source setup/env.sh; source drone/env.sh
B="${1:?x86|thor|orin}"; D="$DRONE_STANDINS"; T="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
for f in seg_512 seg_1024 radar_fusion; do [ -f "$D/$f.onnx" ] || { echo "missing $D/$f.onnx -- run drone/contention/make_standins.py first" >&2; exit 1; }; done
if [ "$B" = x86 ]; then
  PY="${WORKER_PY:-$DRONE_WORK/venv-workers/bin/python}"
  [ -x "${PY%% *}" ] || { echo "no $PY -- run: bash drone/contention/setup_workers_x86.sh" >&2; exit 1; }
  for f in seg_512 seg_1024 radar_fusion; do "$PY" drone/contention/build_trt.py "$D/$f.onnx" "$D/$f.engine"; done
else
  L512="--builderOptimizationLevel=2"; L1024=""; [ "$B" = orin ] && L1024="--builderOptimizationLevel=2"
  "$T" --onnx="$D/seg_512.onnx" --saveEngine="$D/seg_512.engine" --fp16 $L512 > "$D/build_seg_512.log" 2>&1
  "$T" --onnx="$D/seg_1024.onnx" --saveEngine="$D/seg_1024.engine" --fp16 $L1024 > "$D/build_seg_1024.log" 2>&1
  "$T" --onnx="$D/radar_fusion.onnx" --saveEngine="$D/radar_fusion.engine" --fp16 > "$D/build_radar.log" 2>&1
  grep -h -oE "&&&& (PASSED|FAILED)[^ ]*" "$D"/build_*.log
fi
ls -la "$D"/*.engine
