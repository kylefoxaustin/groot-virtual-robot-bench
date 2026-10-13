#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${QAIRT_SDK_ROOT:=/opt/qcom/qairt}"
: "${GVRB_WORK:=$HOME/gvrb}"
# Cross-compile GR00T resident_orch for the iq9 board (aarch64, Yocto/glibc 2.35).
set -e
SDK=${QAIRT_SDK_ROOT}/2.47.0.260601
HERE=${GVRB_WORK}/optionB/resident
SRC=${GVRB_WORK}/optionB/src   # resident_orch.cpp + Utils/ Log/ WrapperUtils/ PAL/ SampleApp.hpp QnnTypeMacros.hpp copied from the SDK's examples/QNN/SampleApp/SampleApp/src
CXX=aarch64-linux-gnu-g++-11

INC="-I$SRC -I$SRC/Log -I$SRC/Utils -I$SRC/WrapperUtils -I$SRC/PAL/include -I$SDK/include/QNN"
CXXFLAGS="-std=gnu++14 -fPIC -O2 -Wno-write-strings -fvisibility=hidden -DQNN_API= $INC"

OUT=$HERE/build
mkdir -p $OUT
cd $OUT

SRCS="$SRC/resident_orch.cpp \
  $SRC/Utils/DynamicLoadUtil.cpp $SRC/Utils/IOTensor.cpp $SRC/Utils/DataUtil.cpp \
  $SRC/Utils/QnnSampleAppUtils.cpp $SRC/Utils/QnnDlcUtils.cpp \
  $SRC/WrapperUtils/QnnWrapperUtils.cpp \
  $SRC/Log/Logger.cpp $SRC/Log/LogUtils.cpp \
  $SRC/PAL/src/linux/Directory.cpp $SRC/PAL/src/linux/DynamicLoading.cpp \
  $SRC/PAL/src/linux/FileOp.cpp $SRC/PAL/src/linux/Path.cpp \
  $SRC/PAL/src/common/GetOpt.cpp $SRC/PAL/src/common/StringOp.cpp"

echo "Compiling GR00T resident_orch (aarch64)..."
$CXX $CXXFLAGS $SRCS -ldl -o $OUT/resident_orch
echo "OK -> $OUT/resident_orch"
file $OUT/resident_orch
