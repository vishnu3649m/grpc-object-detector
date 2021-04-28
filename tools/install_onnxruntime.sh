#!/usr/bin/env bash

# A script to build and install libonnxruntime.

git clone https://github.com/microsoft/onnxruntime.git --recursive /tmp/onnxruntime

# Build & install nsync which is required by onnxruntime
cd /tmp/onnxruntime/cmake/external/nsync
mkdir build
cd build
cmake -DCMAKE_CXX_STANDARD=17 -DNSYNC_ENABLE_TESTS=OFF ..
make -j `nproc`
make install

# Build and install libonnxruntime
cd /tmp/onnxruntime/cmake
mkdir build
cd build
cmake -DCMAKE_CXX_STANDARD=17 \
      -Donnxruntime_RUN_ONNX_TESTS=OFF \
      -Donnxruntime_BUILD_WINML_TESTS=OFF \
      -Donnxruntime_GENERATE_TEST_REPORTS=OFF \
      -Donnxruntime_BUILD_SHARED_LIB=ON \
      -Donnxruntime_PREFER_SYSTEM_LIB=ON \
      -Donnxruntime_BUILD_UNIT_TESTS=OFF \
      -Donnxruntime_USE_OPENMP=ON ..
make -j `nproc`
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
ldconfig
cd ~
rm -r /tmp/onnxruntime
