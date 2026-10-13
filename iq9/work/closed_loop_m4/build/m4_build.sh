#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# M4 build: ONNX -> fp16 v73 DLC -> context bin, EXACT base recipe (build_one.sh):
#   fp16_relaxed_precision:1, vtcm_mb:8, O:3, dsp_arch v73, soc_id 60, fusion=default(ON).
# Explicit input shapes (-s) so the ctx bins bake the SAME concrete shapes as the base bins
# (batch=1, seq N=296). DLC basename == target baked graph name.
# Usage: m4_build.sh <graph_name> <onnx_path> [-s NAME DIMS]...
set -e
G=$1; ONNX=$2; shift 2; SHAPES="$@"
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK
export PYTHONPATH=$SDK/lib/python:$PYTHONPATH
export LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/closed_loop_m4/build
mkdir -p $D/dlc $D/ctx $D/cfg $D/logs
echo "=== [$G] ONNX->DLC ==="
$BIN/qairt-converter --input_network "$ONNX" --output_path $D/dlc/$G.dlc $SHAPES \
  > $D/logs/$G.conv.log 2>&1 && echo "  conv OK" || { echo "  CONV FAIL"; tail -5 $D/logs/$G.conv.log; exit 1; }
cat > $D/cfg/cfg_$G.json <<EOF
{ "graphs":[{"graph_names":["$G"],"fp16_relaxed_precision":1,"vtcm_mb":8,"O":3}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
EOF
cat > $D/cfg/ext_$G.json <<EOF
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg/cfg_$G.json" } }
EOF
echo "=== [$G] ctx-bin ==="
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/$G.dlc \
  --config_file $D/cfg/ext_$G.json --output_dir $D/ctx --binary_file ${G}_fp16 \
  > $D/logs/$G.ctx.log 2>&1 && echo "  ctx OK" || { echo "  CTX FAIL"; tail -8 $D/logs/$G.ctx.log; exit 1; }
ls -la $D/ctx/${G}_fp16.bin && echo "OK_BUILD_$G" || echo "FAIL_BUILD_$G"
