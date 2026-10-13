#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# build_a16_perrow.sh <graph> <float_dlc> "<preserve_io>"
# W8A16 with per-CHANNEL (conv) AND per-ROW (MatMul/FullyConnected) int8 weights.
# --use_per_channel_quantization alone applies only to convolution ops, so transformer
# MatMul/FC weights stay PER-TENSOR (one scale for a whole [2048,2048] matrix). --use_per_row_
# quantization gives each output row/channel its own int8 scale (common practice for LLM weights).
set -e
G=$1; FDLC=$2; PRES=$3
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK PYTHONPATH=$SDK/lib/python:$PYTHONPATH LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/closed_loop_m7/int8_build; CALIB=$D/calib/$G/input_list.txt
mkdir -p $D/dlc $D/ctx $D/cfg $D/logs
echo "=== [$G] W8A16 + per-channel + per-ROW weights (calib=$(wc -l < $CALIB)) ==="
$BIN/qairt-quantizer --input_dlc "$FDLC" --input_list "$CALIB" \
  --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 8 \
  --use_per_channel_quantization --use_per_row_quantization --enable_per_row_quantized_bias \
  --act_quantizer_calibration percentile --percentile_calibration_value 99.99 \
  --preserve_io_datatype $PRES \
  --output_dlc $D/dlc/${G}_a16pr.dlc > $D/logs/${G}.a16pr.quant.log 2>&1 && echo "  quant OK" \
  || { echo "  QUANT FAIL"; tail -18 $D/logs/${G}.a16pr.quant.log; exit 1; }
# confirm per-row engaged: a FullyConnected weight should now show axis-quant (per-row)
grep -c "use_per_row_quantization=True" <($BIN/qairt-dlc-info -i $D/dlc/${G}_a16pr.dlc 2>/dev/null | head -5) >/dev/null && echo "  per-row flag recorded"
cat > $D/cfg/cfg_${G}_a16pr.json <<EOF
{ "graphs":[{"graph_names":["$G"],"vtcm_mb":8,"O":3}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
EOF
cat > $D/cfg/ext_${G}_a16pr.json <<EOF
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg/cfg_${G}_a16pr.json" } }
EOF
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/${G}_a16pr.dlc \
  --config_file $D/cfg/ext_${G}_a16pr.json --output_dir $D/ctx --binary_file ${G}_a16pr \
  > $D/logs/${G}.a16pr.ctx.log 2>&1 && echo "  ctx OK" || { echo "  CTX FAIL"; tail -12 $D/logs/${G}.a16pr.ctx.log; exit 1; }
ls -la $D/ctx/${G}_a16pr.bin && echo "OK_A16PR_$G"
