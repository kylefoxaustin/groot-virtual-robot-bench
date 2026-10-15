#!/bin/bash
# build_ctx.sh <block S|R|P|D>   (x86 host) planner block ONNX -> DLC -> Hexagon v73 context binary: fp16_relaxed_precision,
# advanced_activation_fusion false, vtcm 8 MB, O3, graph_names == DLC basename (otherwise the config is ignored silently).
# Input: $IQ9_SONIC_WORK/planner/blocks/planner_<B>.onnx (split.sh). Output: $IQ9_SONIC_WORK/planner/qnn/ctx/planner_<B>_aaf0_fp16.bin
set -e
B=$1; G=planner_$B
SDK=$QAIRT_SDK_ROOT/$QAIRT_VER
source $SDK/bin/envsetup.sh >/dev/null 2>&1
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
D=$IQ9_SONIC_WORK/planner/qnn; O=$IQ9_SONIC_WORK/planner/blocks; mkdir -p $D/dlc $D/cfg $D/ctx
rm -f $D/dlc/$G.dlc   # always reconvert: the block ONNX changes when the rewrites run
$BIN/qairt-converter --input_network $O/$G.onnx --output_path $D/dlc/$G.dlc > $D/dlc/$G.convert.log 2>&1
grep -iE "error" $D/dlc/$G.convert.log | head -5 || true
TAG=aaf0_fp16
cat > $D/cfg/cfg_${G}_$TAG.json <<J
{ "graphs":[{"graph_names":["$G"],"fp16_relaxed_precision":1,"vtcm_mb":8,"O":3,"advanced_activation_fusion":false}],
  "devices":[{"dsp_arch":"v73","soc_id":60}] }
J
cat > $D/cfg/ext_${G}_$TAG.json <<J
{ "backend_extensions": { "shared_library_path":"${LIB}/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$D/cfg/cfg_${G}_$TAG.json" } }
J
$BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $D/dlc/$G.dlc \
    --config_file $D/cfg/ext_${G}_$TAG.json --output_dir $D/ctx --binary_file ${G}_$TAG > $D/ctx/${G}_$TAG.gen.log 2>&1 || { tail -20 $D/ctx/${G}_$TAG.gen.log; exit 1; }
$BIN/qnn-context-binary-utility --context_binary $D/ctx/${G}_$TAG.bin --json_file $D/ctx/${G}_$TAG.info.json >/dev/null 2>&1
python3 - $D/ctx/${G}_$TAG <<'PY'
import json,sys,os
b=sys.argv[1]; d=json.load(open(b+".info.json")); g=d["info"]["graphs"][0]["info"]; gi=g["graphBlobInfo"]["info"]
print(f"{os.path.basename(b)} graphName={g.get('graphName')} vtcmSize={gi.get('vtcmSize')} spillFill={gi.get('spillFillBufferSize')} bytes={os.path.getsize(b+'.bin')}")
for t in g.get("graphInputs",[])+g.get("graphOutputs",[]):
    ti=t["info"]; print("   io", ti.get("name"), ti.get("dataType"), ti.get("dimensions"))
PY
