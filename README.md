gRPC Object Detector
--------------------
[![ci](https://img.shields.io/circleci/build/gh/vishnu3649m/grpc-object-detector?logo=circleci&style=flat-square)](https://app.circleci.com/pipelines/github/vishnu3649m/grpc-object-detector)
[![codecov](https://img.shields.io/codecov/c/gh/vishnu3649m/grpc-object-detector?style=flat-square&logo=codecov)](https://app.codecov.io/gh/vishnu3649m/grpc-object-detector)
[![LinuxOS](https://img.shields.io/badge/os-linux-lightgrey?style=flat-square)]()

An object detection service for images via gRPC.

## Introduction

The main motivation for this project is to explore gRPC and to see how well-suited 
it is for providing object detection as a service. It runs deep learning-based object 
detector(s) on your images to detect count objects you want.

Feel free to fork this repo and adapt this for your own object detection services.

This server provides the following services:
- Detection Service for Images
    - `GetDetectableObjects`: Returns the list of objects detectable by the server
    - `DetectImage`: Detects objects of interest in the provided image

Refer to the [protos](protos) directory for exact description of all gRPC services.

## Building gRPC Object Detector
gRPC Object Detector is currently developed for 64-bit Linux only.

#### Dependencies 
gRPC Object Detector was written using C++17. A compiler that supports C++17 is needed.
Only GCC has been tested. The CMake build system is used to build the project
and the minimum required version is 3.13.

gRPC Object Detector depends on the following libraries that you would need to install:
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
git clone --recursive https://github.com/vishnu3649m/grpc-object-detector.git
cd grpc-object-detector
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j $(nproc) && make install
```

#### Running
```
grpc-objdet-server start
```
