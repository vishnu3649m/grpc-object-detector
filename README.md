gRPC Video Analyzer
-------------------
[![Build Status](https://travis-ci.com/vishnu-muthiah/grpc-video-analyzer.svg?branch=main)](https://travis-ci.com/vishnu-muthiah/grpc-video-analyzer)
![Generic badge](https://img.shields.io/badge/os-linux-lightgrey)

A gRPC-based server that runs deep learning-based object detectors & trackers on your
videos/images to detect, count or track objects you want.

## Introduction

The main motivation for this project is to explore gRPC and to see how well-suited 
it is for serving video analysis capabilities. Feel free to fork this repo and adapt
this for your own video analysis applications.

This server provides the following services:
- Detection Service for Images
    - `GetDetectableObjects`: Returns the list of objects supported by the server
    - `DetectImage`: Detects objects of interest in the provided image
- Detection Service for Videos _(in progress)_
    - `GetDetectableObjects`: Returns the list of objects supported by the server
    - `DetectObjectsPerFrame`: Detects objects of interest in every frame of the provided video file (or network-accessible stream)
    - `CountObjects`: Detect and track unique objects of interest within the provided video file (or network-accessible stream)

Refer to the [protos](protos) directory for exact description of all gRPC services.

## How to build & run

#### Dependencies 
- OpenCV: v3.2 or greater
- Pthreads

#### Building and Installing
```
git clone --recursive https://github.com/vishnu-muthiah/grpc-video-analyzer.git
cd grpc-video-analyzer
mkdir build && cd build
cmake -G "CodeBlocks - Unix Makefiles" ..
make -j $(nproc) && make install
```

#### Running
```
grpc_va_server start
```
