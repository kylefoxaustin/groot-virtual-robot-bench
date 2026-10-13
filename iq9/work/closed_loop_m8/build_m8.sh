#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# build_m8.sh <tag> <act_bw>   quantize dlc/<tag>/groot_llm.dlc -> W8A{bw} per-row, ctx bin
# Identical recipe to closed_loop_m7/build_a16_perrow.sh except act_bitwidth is a parameter.
set -e
TAG=$1; ABW=$2
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK PYTHONPATH=$SDK/lib/python:$PYTHONPATH LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/closed_loop_m8
mkdir -p $D/dlc/$TAG $D/ctx $D/cfg $D/logs
CALIB=${GVRB_WORK}/closed_loop_m7/int8_build/calib/groot_llm/input_list.txt
DLC=$D/dlc/$TAG/groot_llm.dlc
NAME=${TAG}_a${ABW}
echo "=== quant $NAME (W8A${ABW} per-row, calib=$(wc -l < $CALIB)) $(date +%T) ==="
$BIN/qairt-quantizer --input_dlc "$DLC" --input_list "$CALIB" \
  --act_bitwidth $ABW --weights_bitwidth 8 --bias_bitwidth 8 \
  --use_per_channel_quantization --use_per_row_quantization --enable_per_row_quantized_bias \
  --act_quantizer_calibration percentile --percentile_calibration_value 99.99 \
  --preserve_io_datatype inputs_embeds attention_mask embeddings \
  --output_dlc $D/dlc/$TAG/groot_llm_${NAME}.dlc > $D/logs/${NAME}.quant.log 2>&1 && echo "  quant OK" \
  || { echo "  QUANT FAIL"; tail -18 $D/logs/${NAME}.quant.log; exit 1; }
cat > $D/cfg/cfg_${NAME}.json <<EOJ
{ "graphs":[{"graph_names":["groot_llm"],"vtcm_mb":8,"O":3}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
EOJ
cat > $D/cfg/ext_${NAME}.json <<EOJ
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg/cfg_${NAME}.json" } }
EOJ
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/$TAG/groot_llm_${NAME}.dlc \
  --config_file $D/cfg/ext_${NAME}.json --output_dir $D/ctx --binary_file groot_llm_${NAME} \
  > $D/logs/${NAME}.ctx.log 2>&1 && echo "  ctx OK" || { echo "  CTX FAIL"; tail -12 $D/logs/${NAME}.ctx.log; exit 1; }
ls -la $D/ctx/groot_llm_${NAME}.bin && echo "OK_BUILD_$NAME"
