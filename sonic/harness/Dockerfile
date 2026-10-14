# Minimal build/run env for gear_sonic_deploy (g1_deploy_onnx_ref) on x86_64 + an NVIDIA GPU (our RTX 5090 runs).
# Mirrors upstream scripts/install_deps.sh package list (apt branch), WITHOUT ROS2.
# TensorRT 10.13.0.35 (the x86 version upstream requires) is bind-mounted at /opt/TensorRT at run time ($SONIC_TRT).
# build: docker build -t sonic-deploy:trt1013 sonic/harness
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang cmake git pkg-config patchelf zlib1g-dev curl wget ca-certificates \
    libyaml-cpp-dev libeigen3-dev libmsgpack-dev libzmq3-dev nlohmann-json3-dev libgtest-dev \
    iproute2 && rm -rf /var/lib/apt/lists/*
# cppzmq headers if the distro package lacks them
RUN test -f /usr/include/zmq.hpp || (curl -sL https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq.hpp -o /usr/include/zmq.hpp && curl -sL https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq_addon.hpp -o /usr/include/zmq_addon.hpp)
RUN curl -sL https://github.com/casey/just/releases/download/1.43.0/just-1.43.0-x86_64-unknown-linux-musl.tar.gz | tar xz -C /usr/local/bin just
RUN curl -sL https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz | tar xz -C /opt && mv /opt/onnxruntime-linux-x64-1.16.3 /opt/onnxruntime
ENV CUDA_HOME=/usr/local/cuda CUDAToolkit_ROOT=/usr/local/cuda TensorRT_ROOT=/opt/TensorRT \
    onnxruntime_DIR=/opt/onnxruntime/lib/cmake/onnxruntime \
    LD_LIBRARY_PATH=/opt/TensorRT/lib:/opt/onnxruntime/lib:/usr/local/cuda/lib64
WORKDIR /workspace
