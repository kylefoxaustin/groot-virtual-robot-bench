#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# W16A16 + per-channel + per-ROW weights: a pipeline-soundness check for the quantization
# toolchain. If ViT/LLM pass cos>0.99 here, lower-precision gaps can be read as precision loss.
# If they stay below 0.95, int16 fixed-point may not cover the transformer's activation range.
set -e
G=$1; FDLC=$2; PRES=$3
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601; BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK PYTHONPATH=$SDK/lib/python:$PYTHONPATH LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/closed_loop_m7/int8_build; CALIB=$D/calib/$G/input_list.txt
mkdir -p $D/dlc $D/ctx $D/cfg $D/logs
echo "=== [$G] W16A16 + per-channel + per-ROW ==="
$BIN/qairt-quantizer --input_dlc "$FDLC" --input_list "$CALIB" \
  --act_bitwidth 16 --weights_bitwidth 16 --bias_bitwidth 8 \
  --use_per_channel_quantization --use_per_row_quantization --enable_per_row_quantized_bias \
  --act_quantizer_calibration percentile --percentile_calibration_value 99.99 \
  --preserve_io_datatype $PRES \
  --output_dlc $D/dlc/${G}_w16a16pr.dlc > $D/logs/${G}.w16a16pr.quant.log 2>&1 && echo "  quant OK" || { echo QFAIL; tail -15 $D/logs/${G}.w16a16pr.quant.log; exit 1; }
cat > $D/cfg/cfg_${G}_w16pr.json <<E2
{ "graphs":[{"graph_names":["$G"],"vtcm_mb":8,"O":3}], "devices":[{"dsp_arch":"v73","soc_id":60}] }
E2
cat > $D/cfg/ext_${G}_w16pr.json <<E2
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so", "config_file_path":"$D/cfg/cfg_${G}_w16pr.json" } }
E2
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/${G}_w16a16pr.dlc \
  --config_file $D/cfg/ext_${G}_w16pr.json --output_dir $D/ctx --binary_file ${G}_w16a16pr \
  > $D/logs/${G}.w16a16pr.ctx.log 2>&1 && echo "  ctx OK" || { echo CTXFAIL; tail -12 $D/logs/${G}.w16a16pr.ctx.log; exit 1; }
ls -la $D/ctx/${G}_w16a16pr.bin && echo "OK_W16A16PR_$G"
