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
- Running object detection on images
    - `GetDetectableObjects`: Returns the list of objects the server can detect
    - `DetectImage`: Returns a list of detections for an image

Refer to the [protos](protos) directory for more usage details of the gRPC services.

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

#### Running the server

> NOTE:</br>The object detectors need the files within the `config` folder to 
> initialize. It is important to run the server from this directory so detectors
> can find the config files upon startup.

```
gRPC Object Detector
Usage: grpc-objdet-server [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -V,--version                Prints version info and exits
  -d,--detector-type TEXT REQUIRED
                              The type of detector to serve
```

For example, to serve a YOLOv4 model trained on the COCO dataset, run:
```shell
grpc-objdet-server -d onnx_yolov4_coco
```

