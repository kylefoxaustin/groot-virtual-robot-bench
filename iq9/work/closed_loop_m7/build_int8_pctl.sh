#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# percentile-calibrated W8A8 (outlier-robust) for a body graph. args: G FDLC "PRES"
set -e
G=$1; FDLC=$2; PRES=$3
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601; BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export QNN_SDK_ROOT=$SDK QAIRT_SDK_ROOT=$SDK PYTHONPATH=$SDK/lib/python:$PYTHONPATH LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/closed_loop_m7/int8_build; CALIB=$D/calib/$G/input_list.txt
mkdir -p $D/dlc $D/ctx $D/cfg $D/logs
$BIN/qairt-quantizer --input_dlc "$FDLC" --input_list "$CALIB" \
  --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8 --use_per_channel_quantization \
  --act_quantizer_calibration percentile --percentile_calibration_value 99.9 \
  --preserve_io_datatype $PRES \
  --output_dlc $D/dlc/${G}_int8p.dlc > $D/logs/${G}.quantp.log 2>&1 && echo "  quant(pctl) OK" || { echo QFAIL; tail -8 $D/logs/${G}.quantp.log; exit 1; }
NINT=$($BIN/qairt-dlc-info -i $D/dlc/${G}_int8p.dlc 2>/dev/null | grep -c "uFxp_8" || true); echo "  uFxp_8 ops=$NINT"
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/${G}_int8p.dlc \
  --config_file $D/cfg/ext_$G.json --output_dir $D/ctx --binary_file ${G}_int8p \
  > $D/logs/${G}.ctxp.log 2>&1 && echo "  ctx OK" || { echo CTXFAIL; tail -8 $D/logs/${G}.ctxp.log; exit 1; }
ls -la $D/ctx/${G}_int8p.bin && echo "OK_PCTL_$G"
