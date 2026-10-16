#!/bin/bash
# (x86 host) Build the drone's resident orchestrator: the GR00T port's iq9/work/optionB/src/resident_orch.cpp plus
#   port/resident_orch_drone.patch        BFRAME (the drone backbone in one call) and --device N (strict NSP pin)
#   contention/resident_orch_execf.patch  EXECT / EXECF / EXECNOP / EXECONLY (contention study; applied on top)
# Two binaries, both cross-compiled for aarch64 (Yocto, glibc 2.35):
#   $DRONE_IQ9_WORK/orch/build/resident_orch_drone     = source + drone patch
#   $DRONE_IQ9_WORK/orch/build/resident_orch_drone_g   = source + drone patch + execf patch
# The SDK's SampleApp helper sources (Qualcomm-licensed, not in this repo) are copied from YOUR QAIRT install into the
# work dir; the checkout is never modified.
#   bash drone/iq9/port/build_resident_orch.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd); GVRB=$(cd "$HERE/../../.." && pwd)
SDK=$QAIRT_SDK_ROOT/${QAIRT_VER:-2.47.0.260601}
SAMPLE=$SDK/examples/QNN/SampleApp/SampleApp/src
CXX=${IQ9_CXX:-aarch64-linux-gnu-g++-11}
W=$DRONE_IQ9_WORK/orch; SRC=$W/src; OUT=$W/build
[ -d "$SAMPLE" ] || { echo "missing $SAMPLE (QAIRT_SDK_ROOT / QAIRT_VER)" >&2; exit 1; }
rm -rf $SRC; mkdir -p $SRC $OUT
cp -r $SAMPLE/Utils $SAMPLE/Log $SAMPLE/WrapperUtils $SAMPLE/PAL $SAMPLE/SampleApp.hpp $SAMPLE/QnnTypeMacros.hpp $SRC/
INC="-I$SRC -I$SRC/Log -I$SRC/Utils -I$SRC/WrapperUtils -I$SRC/PAL/include -I$SDK/include/QNN"
CXXFLAGS="-std=gnu++14 -fPIC -O2 -Wno-write-strings -fvisibility=hidden -DQNN_API= $INC"
HELPERS="$SRC/Utils/DynamicLoadUtil.cpp $SRC/Utils/IOTensor.cpp $SRC/Utils/DataUtil.cpp \
  $SRC/Utils/QnnSampleAppUtils.cpp $SRC/Utils/QnnDlcUtils.cpp $SRC/WrapperUtils/QnnWrapperUtils.cpp \
  $SRC/Log/Logger.cpp $SRC/Log/LogUtils.cpp \
  $SRC/PAL/src/linux/Directory.cpp $SRC/PAL/src/linux/DynamicLoading.cpp $SRC/PAL/src/linux/FileOp.cpp $SRC/PAL/src/linux/Path.cpp \
  $SRC/PAL/src/common/GetOpt.cpp $SRC/PAL/src/common/StringOp.cpp"
cp $GVRB/iq9/work/optionB/src/resident_orch.cpp $SRC/resident_orch.cpp
patch --forward -s $SRC/resident_orch.cpp < $HERE/resident_orch_drone.patch
echo "compiling resident_orch_drone ..."
$CXX $CXXFLAGS $SRC/resident_orch.cpp $HELPERS -ldl -o $OUT/resident_orch_drone
patch --forward -s $SRC/resident_orch.cpp < $HERE/../contention/resident_orch_execf.patch
echo "compiling resident_orch_drone_g ..."
$CXX $CXXFLAGS $SRC/resident_orch.cpp $HELPERS -ldl -o $OUT/resident_orch_drone_g
file $OUT/resident_orch_drone $OUT/resident_orch_drone_g
