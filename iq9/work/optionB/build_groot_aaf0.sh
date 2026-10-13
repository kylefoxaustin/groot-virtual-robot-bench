#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# build_groot_aaf0.sh <dlc_path> <graph_name>
# Rebuild GR00T fp16 v73 ctx bin with advanced_activation_fusion:false.
# ONLY change vs build_one.sh is the added fusion flag; reuses the existing DLC.
set -e
DLC=$1; G=$2
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=${GVRB_WORK}/optionB
OUT=$D/ctx_aaf0; mkdir -p $OUT $D/cfg_aaf0
cat > $D/cfg_aaf0/cfg_${G}.json <<EOF
{ "graphs":[{"graph_names":["$G"],"fp16_relaxed_precision":1,"vtcm_mb":8,"O":3,"advanced_activation_fusion":false}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
EOF
cat > $D/cfg_aaf0/ext_${G}.json <<EOF
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg_aaf0/cfg_${G}.json" } }
EOF
echo "=== build_aaf0 $G (dlc=$DLC) ==="
for attempt in 1 2 3; do
  $BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $DLC \
    --config_file $D/cfg_aaf0/ext_${G}.json --output_dir $OUT --binary_file ${G}_aaf0_fp16 2>&1 \
    | grep -iE "error|fail|success|finish|mismatch" | head -6 && break
  echo "retry $attempt"; sleep 2
done
ls -la $OUT/${G}_aaf0_fp16.bin 2>/dev/null && echo "OK_BUILD_$G" || echo "FAIL_BUILD_$G"
