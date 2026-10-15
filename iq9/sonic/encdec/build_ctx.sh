#!/bin/bash
# build_ctx.sh <graph> [aaf0]   (x86 host) ONNX -> DLC -> Hexagon v73 context binary, fp16_relaxed_precision, vtcm 8 MB, O3.
#   <graph> = sonic_enc_core | sonic_dec | sonic_enc_full (make_graphs.py output in $IQ9_SONIC_WORK/onnx/<graph>.onnx)
#   aaf0    = advanced_activation_fusion false. Use it: with the default fusion, the fp16 SiLU MLPs of the encoder/decoder
#             came out wrong on v73 (compare a default build with the gate if you want to see it).
# The converter gets NO precision flag (the DLC stays fp32; fp16 is chosen at context-binary time). graph_names must equal
# the DLC basename or the HTP config block is ignored silently.
# Outputs: $IQ9_SONIC_WORK/ctx/<graph>_<aaf0_fp16|fp16>.bin (+ .info.json)
set -e
G=$1; VAR=${2:-}
SDK=$QAIRT_SDK_ROOT/$QAIRT_VER
source $SDK/bin/envsetup.sh >/dev/null 2>&1
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=$IQ9_SONIC_WORK; mkdir -p $D/dlc $D/cfg $D/ctx
[ -f $D/dlc/$G.dlc ] || $BIN/qairt-converter --input_network $D/onnx/$G.onnx --output_path $D/dlc/$G.dlc > $D/dlc/$G.convert.log 2>&1
grep -iE "error|warn" $D/dlc/$G.convert.log | head -5 || true
EXTRA=""; TAG=fp16
if [ "$VAR" = "aaf0" ]; then EXTRA=',"advanced_activation_fusion":false'; TAG=aaf0_fp16; fi
cat > $D/cfg/cfg_${G}_$TAG.json <<J
{ "graphs":[{"graph_names":["$G"],"fp16_relaxed_precision":1,"vtcm_mb":8,"O":3$EXTRA}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
J
cat > $D/cfg/ext_${G}_$TAG.json <<J
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg/cfg_${G}_$TAG.json" } }
J
for a in 1 2; do
  $BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/$G.dlc \
    --config_file $D/cfg/ext_${G}_$TAG.json --output_dir $D/ctx --binary_file ${G}_$TAG > $D/ctx/${G}_$TAG.gen.log 2>&1 && break
  echo "retry $a"; sleep 2
done
$BIN/qnn-context-binary-utility --context_binary $D/ctx/${G}_$TAG.bin --json_file $D/ctx/${G}_$TAG.info.json >/dev/null 2>&1
python3 - $D/ctx/${G}_$TAG <<'PY'
import json,sys,os
b=sys.argv[1]; d=json.load(open(b+".info.json")); g=d["info"]["graphs"][0]["info"]; gi=g["graphBlobInfo"]["info"]
print(f"{os.path.basename(b)} graphName={g.get('graphName')} vtcmSize={gi.get('vtcmSize')} spillFill={gi.get('spillFillBufferSize')} bytes={os.path.getsize(b+'.bin')}")
for t in g.get("graphInputs",[])+g.get("graphOutputs",[]):
    ti=t["info"]; print("   io", ti.get("name"), ti.get("dataType"), ti.get("dimensions"))
PY
