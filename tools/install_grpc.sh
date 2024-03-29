#!/usr/bin/env bash

# check if the necessary packages are installed
CMDS="git pkg-config cmake"
PKGS="build-essential autoconf libtool git pkg-config"
missing=""

for i in $CMDS; do
  command -v $i &> /dev/null
  if [ $? -gt 0 ]; then
    missing="${missing} ${i}"
  fi
done

for i in $PKGS; do
  dpkg-query -l $i &> /dev/null
  if [ $? -gt 0 ]; then
    missing="${missing} ${i}"
  fi
done

if [ -n "$missing" ]; then
  echo "The following required packages are missing:"
  echo $missing | tr ' ' '\n'
  echo "Please install them and re-run this script."
  exit 1
fi

if pkg-config protobuf && pkg-config grpc && pkg-config gpr && pkg-config grpc++; then
  echo 'gRPC and C++ plugin are already installed and registered with pkg-config'
else
  git clone --recurse-submodules https://github.com/grpc/grpc.git /tmp/grpc
  cd /tmp/grpc
  git checkout tags/v1.37.0
  mkdir build
  cd build
  cmake -DCMAKE_CXX_STANDARD=17 -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DABSL_ENABLE_INSTALL=ON ..
  make -j `nproc`
  make install
  cd ~
  rm -r /tmp/grpc
fi
