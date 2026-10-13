#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# build_w16a16.sh <graph_name> <float_dlc> "<preserve_io names...>"
# W16A16 discriminator: int16 WEIGHTS + int16 ACTIVATIONS. If this reaches cos>0.95 vs
# fp16 while W8A16 does not, 8-bit weights are the limiting factor (not the converter and not
# activation precision). If W16A16 also fails the gate, suspect the quantization path for this
# graph before drawing conclusions about int8.
set -e
G=$1; FDLC=$2; PRES=$3
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK PYTHONPATH=$SDK/lib/python:$PYTHONPATH LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/closed_loop_m7/int8_build
CALIB=$D/calib/$G/input_list.txt
mkdir -p $D/dlc $D/ctx $D/cfg $D/logs
echo "=== [$G] quantize W16A16 (calib=$(wc -l < $CALIB) samples) ==="
$BIN/qairt-quantizer --input_dlc "$FDLC" --input_list "$CALIB" \
  --act_bitwidth 16 --weights_bitwidth 16 --bias_bitwidth 8 --use_per_channel_quantization \
  --act_quantizer_calibration percentile --percentile_calibration_value 99.99 \
  --preserve_io_datatype $PRES \
  --output_dlc $D/dlc/${G}_w16a16.dlc > $D/logs/${G}.w16a16.quant.log 2>&1 && echo "  quant OK" \
  || { echo "  QUANT FAIL"; tail -15 $D/logs/${G}.w16a16.quant.log; exit 1; }
cat > $D/cfg/cfg_${G}_w16a16.json <<EOF
{ "graphs":[{"graph_names":["$G"],"vtcm_mb":8,"O":3}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
EOF
cat > $D/cfg/ext_${G}_w16a16.json <<EOF
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg/cfg_${G}_w16a16.json" } }
EOF
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/${G}_w16a16.dlc \
  --config_file $D/cfg/ext_${G}_w16a16.json --output_dir $D/ctx --binary_file ${G}_w16a16 \
  > $D/logs/${G}.w16a16.ctx.log 2>&1 && echo "  ctx OK" || { echo "  CTX FAIL"; tail -12 $D/logs/${G}.w16a16.ctx.log; exit 1; }
ls -la $D/ctx/${G}_w16a16.bin && echo "OK_W16A16_$G"
