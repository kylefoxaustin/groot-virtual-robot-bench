#!/bin/bash
# (x86 host) Compile the seeded contention stand-ins to Hexagon v73 fp16 context binaries for the IQ-9075's second NSP.
# Inputs ($DRONE_STANDINS): seg_512.onnx, seg_1024.onnx, radar_fusion.onnx (drone/contention/make_standins.py),
# seg_512_dilfree.onnx (make_dilfree.py) and seg1024/seg_1024_dilfree.onnx (make_seg1024_check.py).
# The working fp16 route: qairt-converter with NO precision flag (never --float_bitwidth 16, which can corrupt MatMul/Gemm);
# fp16 is applied at context generation by the HTP backend config. graph_names must equal the DLC basename, or every graph
# option is silently ignored. soc_model 77 = the IQ-9075 (v73).
# Arms: fix (default fusion) and aaf0 (advanced_activation_fusion: false); a warning is printed if the two bins of a graph are identical.
# Graphs: seg512df, radar512, seg1024df (what the plans use), and the canonical dilated seg512 / seg1024 (fix only; for the
# comparison that motivates the rewrite -- they can be very slow on the NPU, see README).
#   bash drone/iq9/contention/build_standins_qnn.sh [OUT_DIR] [GRAPH ...]     (default $DRONE_IQ9_WORK/standins_ctx, all graphs)
set -u
OUT=${1:-$DRONE_IQ9_WORK/standins_ctx}; shift || true
GRAPHS=${*:-"seg512df radar512 seg1024df seg512 seg1024"}
A=$DRONE_STANDINS
SDK=$QAIRT_SDK_ROOT/${QAIRT_VER:-2.47.0.260601}
BIN=$SDK/bin/x86_64-linux-clang; LIB=$SDK/lib/x86_64-linux-clang
export LD_LIBRARY_PATH=$LIB:${LD_LIBRARY_PATH:-} PYTHONPATH=$SDK/lib/python:${PYTHONPATH:-}
mkdir -p $OUT; cd $OUT
src() {   # graph -> "onnx inputspec"
  case $1 in
    seg512)    echo "$A/seg_512.onnx img 1,3,512,512" ;;
    seg512df)  echo "$A/seg_512_dilfree.onnx img 1,3,512,512" ;;
    radar512)  echo "$A/radar_fusion.onnx pc 1,5,512" ;;
    seg1024)   echo "$A/seg_1024.onnx img 1,3,1024,1024" ;;
    seg1024df) echo "$A/seg1024/seg_1024_dilfree.onnx img 1,3,1024,1024" ;;
    *) echo "unknown graph $1" >&2; exit 1 ;;
  esac
}
for G in $GRAPHS; do
  set -- $(src $G); ONNX=$1; IN="$2 $3"
  [ -f $ONNX ] || { echo "missing $ONNX" >&2; continue; }
  [ -f $G.dlc ] || $BIN/qairt-converter --input_network $ONNX -s $IN --output_path $OUT/$G.dlc > $OUT/convert_$G.log 2>&1
  grep -m1 -E "CONVERSION_SUCCESS|ERROR" $OUT/convert_$G.log || tail -3 $OUT/convert_$G.log
  [ -f $G.dlc ] || { echo "conversion of $G failed, see $OUT/convert_$G.log" >&2; continue; }
  ARMS="fix aaf0"; case $G in seg512|seg1024) ARMS=fix ;; esac
  for ARM in $ARMS; do
    EXTRA=""; [ $ARM = aaf0 ] && EXTRA=',"advanced_activation_fusion":false'
    echo "{ \"graphs\":[{\"graph_names\":[\"$G\"],\"vtcm_mb\":8,\"O\":3$EXTRA}], \"devices\":[{\"dsp_arch\":\"v73\",\"soc_model\":77}] }" > cfg_${G}_$ARM.json
    echo "{ \"backend_extensions\": { \"shared_library_path\":\"$LIB/libQnnHtpNetRunExtensions.so\", \"config_file_path\":\"$OUT/cfg_${G}_$ARM.json\" } }" > ext_${G}_$ARM.json
    for attempt in 1 2 3; do
      $BIN/qnn-context-binary-generator --backend $LIB/libQnnHtp.so --dlc_path $OUT/$G.dlc \
        --config_file $OUT/ext_${G}_$ARM.json --output_dir $OUT --binary_file ${G}_${ARM}_fp16 > ctx_${G}_$ARM.log 2>&1 && break
      echo "retry $attempt ($G $ARM)"; sleep 2
    done
    ls -la $OUT/${G}_${ARM}_fp16.bin 2>&1
  done
  if [ -f ${G}_fix_fp16.bin ] && [ -f ${G}_aaf0_fp16.bin ] && cmp -s ${G}_fix_fp16.bin ${G}_aaf0_fp16.bin; then
    echo "WARNING: $G fix and aaf0 bins are identical: check graph_names in cfg_${G}_aaf0.json (or the option changes nothing for this graph)"
  fi
done
md5sum $OUT/*.bin
echo BUILD_DONE
