#!/bin/bash
# Native build of upstream gear_sonic_deploy (+ the measurement probe) on a Jetson with the board's own TensorRT.
# Prereqs: source sonic/env.sh; $SONIC_REPO checked out at $SONIC_COMMIT with sonic/harness/sonic_probe.patch applied and
# sonic_probe.hpp copied next to g1_deploy_onnx_ref.cpp (sonic/README.md, "Run it yourself"); onnxruntime 1.16.3 aarch64 at $SONIC_ORT.
set -e
D=$SONIC_DEPLOY; T=$SONIC_WORK/trtroot
mkdir -p $T; ln -sfn /usr/include/aarch64-linux-gnu $T/include; ln -sfn /usr/lib/aarch64-linux-gnu $T/lib
export TensorRT_ROOT=$T onnxruntime_ROOT=$SONIC_ORT CUDA_HOME=/usr/local/cuda CUDAToolkit_ROOT=/usr/local/cuda HAS_ROS2=0
export PATH=/usr/local/cuda/bin:$PATH
cd $D && mkdir -p build && cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > $SONIC_WORK/cmake_configure.log 2>&1
cmake --build . -j$(nproc) > $SONIC_WORK/cmake_build.log 2>&1
ls -la $D/target/release/
