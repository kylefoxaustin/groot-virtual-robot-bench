#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# build_one.sh <onnx_path> <graph_name> [shape_args...]
# ONNX -> fp16 v73 DLC -> context binary, fp16_relaxed_precision route.
set -e
ONNX=$1; G=$2; shift 2; SHAPES="$@"
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK
export PYTHONPATH=$SDK/lib/python:$PYTHONPATH
export LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/optionB/ah_fix
mkdir -p $D/dlc $D/ctx $D/cfg
echo "=== ONNX->DLC ($G) ==="
$BIN/qairt-converter --input_network $ONNX --output_path $D/dlc/$G.dlc $SHAPES 2>&1 | grep -iE "SUCCESS|ERROR|Total MACs|graph name|graph_name" | head -6
cat > $D/cfg/cfg_$G.json <<EOF
{ "graphs":[{"graph_names":["$G"],"fp16_relaxed_precision":1,"vtcm_mb":8,"O":3}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
EOF
cat > $D/cfg/ext_$G.json <<EOF
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg/cfg_$G.json" } }
EOF
echo "=== ctx-bin ($G) ==="
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/$G.dlc \
  --config_file $D/cfg/ext_$G.json --output_dir $D/ctx --binary_file ${G}_fp16 2>&1 | grep -iE "error|fail|success|finish|not found|ignor|vtcm" | head -8
ls -la $D/ctx/${G}_fp16.bin 2>/dev/null && echo "OK_BUILD_$G" || echo "FAIL_BUILD_$G"
