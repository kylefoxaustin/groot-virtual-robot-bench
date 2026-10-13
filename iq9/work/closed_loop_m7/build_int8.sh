#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# build_int8.sh <graph_name> <float_dlc> "<preserve_io names...>"
# W8A8 int8 (per-channel weights) with REAL calibration + FLOAT IO preserved (so the
# resident_orch float-memcpy graph chaining consumes it unchanged). Then HTP v73 ctx bin.
set -e
G=$1; FDLC=$2; PRES=$3
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK PYTHONPATH=$SDK/lib/python:$PYTHONPATH LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/closed_loop_m7/int8_build
CALIB=$D/calib/$G/input_list.txt
mkdir -p $D/dlc $D/ctx $D/cfg $D/logs
echo "=== [$G] quantize W8A8 (calib=$(wc -l < $CALIB) samples, preserve: $PRES) ==="
$BIN/qairt-quantizer --input_dlc "$FDLC" --input_list "$CALIB" \
  --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8 --use_per_channel_quantization \
  --preserve_io_datatype $PRES \
  --output_dlc $D/dlc/${G}_int8.dlc > $D/logs/${G}.quant.log 2>&1 && echo "  quant OK" \
  || { echo "  QUANT FAIL"; tail -15 $D/logs/${G}.quant.log; exit 1; }
# sanity: IO must be Float_32, internal must have uFxp_8 ops
IOQ=$($BIN/qairt-dlc-info -i $D/dlc/${G}_int8.dlc 2>/dev/null | awk '/Input Name/{f=1} /Output Name/{f=2} f&&/^\| [a-z]/{print}' | grep -c "Float_32" || true)
NINT=$($BIN/qairt-dlc-info -i $D/dlc/${G}_int8.dlc 2>/dev/null | grep -c "uFxp_8" || true)
echo "  float-IO rows=$IOQ  internal uFxp_8 ops=$NINT"
cat > $D/cfg/cfg_$G.json <<EOF
{ "graphs":[{"graph_names":["$G"],"vtcm_mb":8,"O":3}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
EOF
cat > $D/cfg/ext_$G.json <<EOF
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg/cfg_$G.json" } }
EOF
echo "=== [$G] ctx-bin ==="
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/${G}_int8.dlc \
  --config_file $D/cfg/ext_$G.json --output_dir $D/ctx --binary_file ${G}_int8 \
  > $D/logs/${G}.ctx.log 2>&1 && echo "  ctx OK" || { echo "  CTX FAIL"; tail -10 $D/logs/${G}.ctx.log; exit 1; }
ls -la $D/ctx/${G}_int8.bin && echo "OK_INT8_$G"
