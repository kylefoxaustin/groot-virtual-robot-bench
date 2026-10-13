#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# convert_llm.sh <tag> <onnx_path>   -> fp32 DLC at m8/dlc/<tag>/groot_llm.dlc  (graph name groot_llm)
set -e
TAG=$1; ONNX=$2
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK PYTHONPATH=$SDK/lib/python:$PYTHONPATH LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/closed_loop_m8; mkdir -p $D/dlc/$TAG $D/logs
echo "=== convert $TAG from $ONNX $(date +%T) ==="
$BIN/qairt-converter --input_network "$ONNX" --output_path $D/dlc/$TAG/groot_llm.dlc \
  -s inputs_embeds 1,296,2048 -s attention_mask 1,296 \
  > $D/logs/conv_${TAG}.log 2>&1 && echo "CONV_OK_$TAG" || { echo "CONV_FAIL_$TAG"; tail -8 $D/logs/conv_${TAG}.log; exit 1; }
ls -la $D/dlc/$TAG/groot_llm.dlc
