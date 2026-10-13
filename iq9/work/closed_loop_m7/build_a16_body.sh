#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# build_a16_body.sh <graph_name> <float_dlc> "<preserve_io names...>"
# W8A16: int8 per-channel WEIGHTS + int16 ACTIVATIONS — a common recipe for transformers.
# Discriminating build vs the W8A8 body: if this clears cosine>0.95 while W8A8 does not,
# the W8A8 loss comes from 8-bit activation precision, not from the build.
set -e
G=$1; FDLC=$2; PRES=$3
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK PYTHONPATH=$SDK/lib/python:$PYTHONPATH LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/closed_loop_m7/int8_build
CALIB=$D/calib/$G/input_list.txt
mkdir -p $D/dlc $D/ctx $D/cfg $D/logs
echo "=== [$G] quantize W8A16 (calib=$(wc -l < $CALIB) samples, preserve: $PRES) ==="
$BIN/qairt-quantizer --input_dlc "$FDLC" --input_list "$CALIB" \
  --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 8 --use_per_channel_quantization \
  --act_quantizer_calibration percentile --percentile_calibration_value 99.99 \
  --preserve_io_datatype $PRES \
  --output_dlc $D/dlc/${G}_a16.dlc > $D/logs/${G}.a16.quant.log 2>&1 && echo "  quant OK" \
  || { echo "  QUANT FAIL"; tail -15 $D/logs/${G}.a16.quant.log; exit 1; }
NINT16=$($BIN/qairt-dlc-info -i $D/dlc/${G}_a16.dlc 2>/dev/null | grep -oiE "uFxp_16|sFxp_16" | sort | uniq -c || true)
echo "  16-bit act op count: $NINT16"
# reuse the same htp ctx cfg (vtcm 8, O3, v73)
cat > $D/cfg/cfg_${G}_a16.json <<EOF
{ "graphs":[{"graph_names":["$G"],"vtcm_mb":8,"O":3}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
EOF
cat > $D/cfg/ext_${G}_a16.json <<EOF
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg/cfg_${G}_a16.json" } }
EOF
echo "=== [$G] ctx-bin (a16) ==="
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/${G}_a16.dlc \
  --config_file $D/cfg/ext_${G}_a16.json --output_dir $D/ctx --binary_file ${G}_a16 \
  > $D/logs/${G}.a16.ctx.log 2>&1 && echo "  ctx OK" || { echo "  CTX FAIL"; tail -12 $D/logs/${G}.a16.ctx.log; exit 1; }
ls -la $D/ctx/${G}_a16.bin && echo "OK_A16_$G"
