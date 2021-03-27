#!/usr/bin/env bash

command -v cmake &> /dev/null
if [ $? -gt 0 ]; then
  echo "cmake: missing. Please install cmake and re-run this script."
  exit 1
fi

command -v make &> /dev/null
if [ $? -gt 0 ]; then
  echo "make: missing. Please install make and re-run this script."
  exit 1
fi

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install
