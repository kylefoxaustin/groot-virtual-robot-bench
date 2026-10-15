#!/bin/bash
# Cross-compile sonic_rt (x86 host -> aarch64 IQ-9075). Needs $IQ9_SONIC_SAMPLEAPP (deploy/get_sampleapp.sh) and $IQ9_SONIC_ORT.
# The rpath points at $IQ9_SONIC_BOARD_DIR/lib, where stage_board.sh puts libonnxruntime.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); SDK=$QAIRT_SDK_ROOT/$QAIRT_VER; Q=$IQ9_SONIC_SAMPLEAPP; ORT=$IQ9_SONIC_ORT
OUT=$IQ9_SONIC_WORK/build; mkdir -p $OUT
INC="-I$HERE -I$Q -I$Q/Log -I$Q/Utils -I$Q/WrapperUtils -I$Q/PAL/include -I$SDK/include/QNN -I$ORT/include"
SRCS="$Q/Utils/DynamicLoadUtil.cpp $Q/Utils/IOTensor.cpp $Q/Utils/DataUtil.cpp \
  $Q/Utils/QnnSampleAppUtils.cpp $Q/Utils/QnnDlcUtils.cpp $Q/WrapperUtils/QnnWrapperUtils.cpp \
  $Q/Log/Logger.cpp $Q/Log/LogUtils.cpp $Q/PAL/src/linux/Directory.cpp $Q/PAL/src/linux/DynamicLoading.cpp \
  $Q/PAL/src/linux/FileOp.cpp $Q/PAL/src/linux/Path.cpp $Q/PAL/src/common/GetOpt.cpp $Q/PAL/src/common/StringOp.cpp"
$IQ9_CXX -std=gnu++17 -fPIC -O2 -Wno-write-strings -fvisibility=hidden -DQNN_API= $INC $HERE/sonic_rt.cpp $SRCS \
  -L$ORT/lib -lonnxruntime -Wl,-rpath,$IQ9_SONIC_BOARD_DIR/lib -ldl -lpthread -o $OUT/sonic_rt
file $OUT/sonic_rt
