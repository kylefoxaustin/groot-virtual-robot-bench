#!/bin/bash
# Cross-build upstream g1_deploy_onnx_ref for the IQ-9075 (aarch64 Yocto) with QNN in place of TensorRT/CUDA.
# What changes vs upstream: three headers are replaced by QNN shims with the same class names and public API
# (shim/encoder.hpp, shim/control_policy.hpp, shim/localmotion_kplanner_tensorrt.hpp, plus shim/qnn_shim_rt.hpp), and
# shim/override/ stubs out <cuda_runtime.h> and <TRTInference/InferenceEngine.h>. g1_deploy_onnx_ref.cpp itself gets only this
# repo's two patches: sonic/harness/sonic_probe.patch (timing probe) and sonic/harness/latefix.patch (late-start settings,
# inert unless SONIC_CTRL_FIFO / SONIC_MAIN_SLEEP are set). Upstream includes its headers by relative path, so the build
# compiles a COPY of the deploy source with the shims dropped into its include/ ($IQ9_SONIC_WORK/deploy_src).
# Prereqs: source setup/env.sh sonic/env.sh iq9/sonic/env.sh; $SONIC_REPO at $SONIC_COMMIT after `git lfs pull`;
# deploy/get_sampleapp.sh; deploy/build_sysroot.sh; onnxruntime 1.20.1 aarch64 at $IQ9_SONIC_ORT; Eigen 3 headers on the host.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); GVRB=$(cd "$HERE/../../.." && pwd)
SDK=$QAIRT_SDK_ROOT/$QAIRT_VER; Q=$IQ9_SONIC_SAMPLEAPP; ORT=$IQ9_SONIC_ORT; R=$IQ9_SONIC_SYSROOT
UP=$SONIC_DEPLOY; U=$UP/thirdparty/unitree_sdk2
D=$IQ9_SONIC_WORK/deploy_src; OUT=$IQ9_SONIC_WORK/build
file $U/lib/aarch64/libunitree_sdk2.a | grep -q "ar archive" || { echo "libunitree_sdk2.a is not an archive: run git lfs pull in \$SONIC_REPO"; exit 1; }
rm -rf $D; mkdir -p $D $OUT/obj; cp -a $UP/src/g1/g1_deploy_onnx_ref/. $D/
if ! grep -q sonic_probe.hpp $D/src/g1_deploy_onnx_ref.cpp; then
  cp $GVRB/sonic/harness/sonic_probe.hpp $D/src/ && patch -d $D -p5 < $GVRB/sonic/harness/sonic_probe.patch
fi
grep -q sonic_rtctl.hpp $D/src/g1_deploy_onnx_ref.cpp || patch -d $D -p5 < $GVRB/sonic/harness/latefix.patch
cp $HERE/shim/encoder.hpp $HERE/shim/control_policy.hpp $HERE/shim/localmotion_kplanner_tensorrt.hpp $HERE/shim/qnn_shim_rt.hpp $D/include/
QINC="-I$Q -I$Q/Log -I$Q/Utils -I$Q/WrapperUtils -I$Q/PAL/include -I$SDK/include/QNN"
INC="-I$HERE/shim/override -I$D/include -I$UP/src -I$U/include -I$U/thirdparty/include -I$U/thirdparty/include/ddscxx -I$R/include -I/usr/include/eigen3 -I$ORT/include $QINC"
FL="-std=c++20 -O3 -ffast-math -DNDEBUG -DHAS_ROS2=0 -DQNN_API= -fPIC -Wno-write-strings -Wno-deprecated-declarations"
QSRCS="$Q/Utils/DynamicLoadUtil.cpp $Q/Utils/IOTensor.cpp $Q/Utils/DataUtil.cpp $Q/Utils/QnnSampleAppUtils.cpp $Q/Utils/QnnDlcUtils.cpp $Q/WrapperUtils/QnnWrapperUtils.cpp $Q/Log/Logger.cpp $Q/Log/LogUtils.cpp $Q/PAL/src/linux/Directory.cpp $Q/PAL/src/linux/DynamicLoading.cpp $Q/PAL/src/linux/FileOp.cpp $Q/PAL/src/linux/Path.cpp $Q/PAL/src/common/GetOpt.cpp $Q/PAL/src/common/StringOp.cpp"
SRCS="$D/src/g1_deploy_onnx_ref.cpp $D/src/cnpy.cpp $D/src/file_sink.cpp $D/src/fk.cpp $D/src/motor_gain_scaling.cpp $D/src/state_logger.cpp $UP/src/audio_thread/audio_thread.cpp"
pids=()
for s in $SRCS; do $IQ9_CXX $FL $INC -c $s -o $OUT/obj/$(basename $s).o & pids+=($!); done
for s in $QSRCS; do $IQ9_CXX -std=gnu++17 -O2 -fPIC -Wno-write-strings -DQNN_API= $QINC -c $s -o $OUT/obj/q_$(basename $s).o & pids+=($!); done
for p in ${pids[@]}; do wait $p; done
$IQ9_CXX -o $OUT/g1_deploy_onnx_ref_qnn $OUT/obj/*.o $U/lib/aarch64/libunitree_sdk2.a -L$U/thirdparty/lib/aarch64 -lddscxx -lddsc \
  -L$R/lib -lzmq -l:libz.so.1 -L$ORT/lib -lonnxruntime -Wl,-rpath,$IQ9_SONIC_BOARD_DIR/lib -lpthread -ldl
file $OUT/g1_deploy_onnx_ref_qnn
