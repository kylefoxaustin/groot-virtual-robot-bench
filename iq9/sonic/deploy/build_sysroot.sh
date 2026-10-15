#!/bin/bash
# Headers and libraries the upstream deploy needs for aarch64 that a cross toolchain does not have ($IQ9_SONIC_SYSROOT):
#  * header-only: msgpack (C++), nlohmann/json, cppzmq (zmq.hpp, zmq_addon.hpp) + zmq.h -- copied from the x86 host's
#    /usr/include (Ubuntu 22.04: apt install libmsgpack-dev nlohmann-json3-dev libzmq3-dev; cppzmq from its v4.10.0 release
#    if your libzmq3-dev has no zmq.hpp). They are architecture-independent.
#  * libzmq 4.3.4, cross-built here (no libsodium/CURVE).
#  * libz: copied from the board (the Yocto image has it), so the deploy links the board's own zlib.
# Prereqs: aarch64-linux-gnu-gcc-11/g++-11, cmake, curl, ssh/scp to $IQ9_HOST.
set -e
R=$IQ9_SONIC_SYSROOT; mkdir -p $R/include $R/lib $IQ9_SONIC_WORK/src
for p in /usr/include/msgpack.hpp /usr/include/msgpack /usr/include/nlohmann /usr/include/zmq.hpp /usr/include/zmq_addon.hpp /usr/include/zmq.h /usr/include/zmq_utils.h; do
  if [ -e $p ]; then cp -r $p $R/include/; else echo "missing on this host: $p"; fi
done
[ -f $R/include/zmq.hpp ] || { curl -sL https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq.hpp -o $R/include/zmq.hpp
                               curl -sL https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq_addon.hpp -o $R/include/zmq_addon.hpp; }
cd $IQ9_SONIC_WORK/src
[ -d libzmq-4.3.4 ] || (curl -sL https://github.com/zeromq/libzmq/releases/download/v4.3.4/zeromq-4.3.4.tar.gz | tar xz && mv zeromq-4.3.4 libzmq-4.3.4)
mkdir -p libzmq-4.3.4/build-a64 && cd libzmq-4.3.4/build-a64
cmake .. -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER=$IQ9_CC -DCMAKE_CXX_COMPILER=$IQ9_CXX \
  -DCMAKE_BUILD_TYPE=Release -DWITH_LIBSODIUM=OFF -DENABLE_CURVE=OFF -DBUILD_TESTS=OFF -DZMQ_BUILD_TESTS=OFF -DWITH_DOCS=OFF \
  -DBUILD_STATIC=OFF -DCMAKE_INSTALL_PREFIX=$R > cmake.log 2>&1
make -j$(nproc) > make.log 2>&1 && make install > install.log 2>&1
# the board's zlib (find its real file name first: ssh $IQ9_HOST 'ls /usr/lib/libz.so*')
Z=$(ssh $IQ9_HOST 'readlink -f /usr/lib/libz.so.1')
scp -q $IQ9_HOST:$Z $R/lib/ && ln -sf $(basename $Z) $R/lib/libz.so.1 && ln -sf libz.so.1 $R/lib/libz.so
ls $R/include $R/lib
