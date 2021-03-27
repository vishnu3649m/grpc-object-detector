gRPC Video Analyzer
-------------------
[![Build Status](https://travis-ci.com/vishnu3649m/grpc-video-analyzer.svg?branch=main)](https://travis-ci.com/vishnu3649m/grpc-video-analyzer)
![Generic badge](https://img.shields.io/badge/os-linux-lightgrey)

A video analysis and image detection service via gRPC.

## Introduction

The main motivation for this project is to explore gRPC and to see how well-suited 
it is for providing video analytics as a service. It runs deep learning-based object 
detectors & trackers on your videos/images to detect, track & count objects you want. 

Feel free to fork this repo and adapt this for your own video analysis services.

This server provides the following services:
- Detection Service for Images
    - `GetDetectableObjects`: Returns the list of objects supported by the server
    - `DetectImage`: Detects objects of interest in the provided image
- Detection Service for Videos _(in progress)_
    - `GetDetectableObjects`: Returns the list of objects supported by the server
    - `DetectObjectsPerFrame`: Detects objects of interest in every frame of the provided video file (or network-accessible stream)
    - `CountObjects`: Detect and track unique objects of interest within the provided video file (or network-accessible stream)

Refer to the [protos](protos) directory for exact description of all gRPC services.

This server uses OpenCV for all image and video I/O. 

## Building gRPC Video Analyzer
gRPC Video Analyzer is currently developed for 64-bit Linux only.

#### Dependencies 
gRPC Video Analyzer was written using C++17. A compiler that supports C++17 is needed.
Only GCC has been tested. The CMake build system is used to build the project
and the minimum required version is 3.10.

gRPC Video Analyzer depends on the following libraries that you would need to install:
- OpenCV: v3.2 or greater (Hardware acceleration and other optimizations provided by specific backends is dependent on your specific installation)
- gRPC: v1.37.0 or greater (Refer [here](https://github.com/grpc/grpc/blob/master/BUILDING.md) for instructions on how to build gRPC)
- Pthreads

The following third-party libraries are included as submodules within this repo:
- [Abseil](https://abseil.io/) (for awesome utilities)
- [Loguru](https://github.com/emilk/loguru) (for logging)
- [CLI11](https://github.com/CLIUtils/CLI11) (for command-line parsing and handling)
- [ONNX Runtime](https://www.onnxruntime.ai/) (for performing deep-learning inference)

#### Building and Installing
```
git clone --recursive https://github.com/vishnu3649m/grpc-video-analyzer.git
cd grpc-video-analyzer
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j $(nproc) && make install
```

#### Running
```
grpc_va_server start
```
