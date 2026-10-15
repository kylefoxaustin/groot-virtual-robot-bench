#!/bin/bash
# Copy the QNN SampleApp helper sources (Qualcomm-licensed, NOT in this repo) from YOUR QAIRT SDK into $IQ9_SONIC_SAMPLEAPP.
# Every IQ-9075 SONIC binary (sonic_bench, sonic_rt, the deploy) compiles them for context loading and tensor setup.
set -e
SRC=$QAIRT_SDK_ROOT/$QAIRT_VER/examples/QNN/SampleApp/SampleApp/src
mkdir -p $IQ9_SONIC_SAMPLEAPP
cp -r $SRC/Utils $SRC/Log $SRC/WrapperUtils $SRC/PAL $SRC/SampleApp.hpp $SRC/QnnTypeMacros.hpp $IQ9_SONIC_SAMPLEAPP/
ls $IQ9_SONIC_SAMPLEAPP
