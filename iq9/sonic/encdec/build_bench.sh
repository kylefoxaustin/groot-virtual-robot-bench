#!/bin/bash
# Cross-compile sonic_bench (x86 host -> aarch64 IQ-9075). Needs $IQ9_SONIC_SAMPLEAPP (deploy/get_sampleapp.sh).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); SDK=$QAIRT_SDK_ROOT/$QAIRT_VER; Q=$IQ9_SONIC_SAMPLEAPP
OUT=$IQ9_SONIC_WORK/build; mkdir -p $OUT
INC="-I$Q -I$Q/Log -I$Q/Utils -I$Q/WrapperUtils -I$Q/PAL/include -I$SDK/include/QNN"
SRCS="$HERE/sonic_bench.cpp $Q/Utils/DynamicLoadUtil.cpp $Q/Utils/IOTensor.cpp $Q/Utils/DataUtil.cpp \
  $Q/Utils/QnnSampleAppUtils.cpp $Q/Utils/QnnDlcUtils.cpp $Q/WrapperUtils/QnnWrapperUtils.cpp \
  $Q/Log/Logger.cpp $Q/Log/LogUtils.cpp $Q/PAL/src/linux/Directory.cpp $Q/PAL/src/linux/DynamicLoading.cpp \
  $Q/PAL/src/linux/FileOp.cpp $Q/PAL/src/linux/Path.cpp $Q/PAL/src/common/GetOpt.cpp $Q/PAL/src/common/StringOp.cpp"
$IQ9_CXX -std=gnu++17 -fPIC -O2 -Wno-write-strings -fvisibility=hidden -DQNN_API= $INC $SRCS -ldl -lpthread -o $OUT/sonic_bench
file $OUT/sonic_bench
