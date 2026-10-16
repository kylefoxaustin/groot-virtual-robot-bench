#!/bin/bash
# (x86 host) Rebuild GR00T's vlln + vl_self_attention graph for the drone prompt length: 289 tokens.
# vl_self_attention is bidirectional and has no attention mask, so right-padding the 289-token drone sequence to the
# GR00T port's 296 would change every valid token's output. The LLM is causal, so it keeps its 296-token build: the
# orchestrator's BFRAME feeds this graph only the 289 valid LLM rows.
# Same ONNX (dynamic sequence dim, from speed/export_onnx.sh) and the same recipe as the GR00T port's vlln: qairt-converter
# with NO precision flag (never --float_bitwidth 16), fp16 applied at context generation by the HTP config, vtcm 8 MB, O3, v73.
# The graph name must stay vlln_vl_self_attention_47: BFRAME looks it up by that name, and the HTP config's graph_names
# must match it or every graph option is silently ignored.
# Two arms: fix (default fusion) and aaf0 (advanced_activation_fusion: false). Both bins must differ (md5 printed).
#   bash drone/iq9/port/build_vlln289.sh [OUT_DIR]     (default $DRONE_IQ9_WORK/ctx289)
set -e
OUT=${1:-$DRONE_IQ9_WORK/ctx289}
SDK=$QAIRT_SDK_ROOT/${QAIRT_VER:-2.47.0.260601}
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export LD_LIBRARY_PATH=$LIB:${LD_LIBRARY_PATH:-} PYTHONPATH=$SDK/lib/python:${PYTHONPATH:-}
ONNX=${VLLN_ONNX:-$GROOT_ONNX/action_head/vlln_vl_self_attention.onnx}
[ -f "$ONNX" ] || { echo "missing $ONNX -- run speed/export_onnx.sh first" >&2; exit 1; }
G=vlln_vl_self_attention_47
mkdir -p $OUT; cd $OUT
[ -f $G.dlc ] || $BIN/qairt-converter --input_network $ONNX -s backbone_features 1,289,2048 \
    --output_path $OUT/$G.dlc > $OUT/convert.log 2>&1
grep -E "CONVERSION_SUCCESS" $OUT/convert.log || { echo "conversion failed, see $OUT/convert.log" >&2; exit 1; }
for ARM in fix aaf0; do
  EXTRA=""; [ $ARM = aaf0 ] && EXTRA=',"advanced_activation_fusion":false'
  cat > cfg_$ARM.json <<EOF
{ "graphs":[{"graph_names":["$G"],"vtcm_mb":8,"O":3$EXTRA}], "devices":[{"dsp_arch":"v73","soc_id":60}] }
EOF
  cat > ext_$ARM.json <<EOF
{ "backend_extensions": { "shared_library_path":"$LIB/libQnnHtpNetRunExtensions.so", "config_file_path":"$OUT/cfg_$ARM.json" } }
EOF
  for attempt in 1 2 3; do
    $BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $OUT/$G.dlc \
      --config_file $OUT/ext_$ARM.json --output_dir $OUT --binary_file vlln289_${ARM}_fp16 > ctx_$ARM.log 2>&1 && break
    echo "retry $attempt ($ARM)"; sleep 2
  done
  ls -la $OUT/vlln289_${ARM}_fp16.bin
done
md5sum $OUT/*.bin
[ "$(md5sum < $OUT/vlln289_fix_fp16.bin)" != "$(md5sum < $OUT/vlln289_aaf0_fp16.bin)" ] || echo "WARNING: fix and aaf0 bins are identical: check graph_names in cfg_aaf0.json (the option was not applied)"
