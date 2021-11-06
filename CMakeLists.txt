cmake_minimum_required(VERSION 3.18)
project(grpc-obj-det)

set(CMAKE_CXX_STANDARD 17)

# check for and build dependencies
find_library(ONNXRUNTIME_LIB NAMES onnxruntime)
if (ONNXRUNTIME_LIB)
    message(STATUS "Found libonnxruntime")
else()
    # For building ONNX Runtime from submodule if not found
    set(onnxruntime_RUN_ONNX_TESTS OFF)
    set(onnxruntime_BUILD_WINML_TESTS OFF)
    set(onnxruntime_GENERATE_TEST_REPORTS OFF)
    set(onnxruntime_BUILD_SHARED_LIB ON)
    set(onnxruntime_PREFER_SYSTEM_LIB ON)
    set(onnxruntime_BUILD_UNIT_TESTS OFF)
    set(onnxruntime_USE_OPENMP ON)
    set(NSYNC_ENABLE_TESTS OFF)

    add_subdirectory(libs/onnxruntime/cmake/external/nsync)
    add_subdirectory(libs/onnxruntime/cmake)
endif()
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(absl REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv)
add_subdirectory(libs/googletest)
add_subdirectory(protos/)
add_library(loguru STATIC libs/loguru/loguru.cpp libs/loguru/loguru.hpp)

# Core obj detection + server lib
add_library(objdet-server-core STATIC
        src/grpc_obj_det/DetectorInterface.h
        src/grpc_obj_det/FaceEyesDetector.h src/grpc_obj_det/FaceEyesDetector.cc
        src/grpc_obj_det/ImageDetectionService.cc src/grpc_obj_det/ImageDetectionService.h
        src/grpc_obj_det/OnnxYoloV4Detector.cc src/grpc_obj_det/OnnxYoloV4Detector.h
        src/grpc_obj_det/onnx_yolov4_detector_utils.cc src/grpc_obj_det/DetectorFactory.h)
target_include_directories(objdet-server-core PUBLIC
        protos/build
        libs/loguru
        libs/onnxruntime/include
        libs/xtl/include
        libs/xtensor/include)
target_link_libraries(objdet-server-core PUBLIC
        objdet-grpc-proto
        loguru
        pthread
        dl
        ${OPENCV_LIBRARIES}
        absl::strings
        absl::str_format
        onnxruntime)

add_executable(grpc-objdet-server src/main.cc)
target_include_directories(grpc-objdet-server PUBLIC libs/CLI11/include)
target_link_libraries(grpc-objdet-server PUBLIC objdet-server-core)

add_executable(test-objdet-server
        tests/test_detector_interface.cc
        tests/test_img_det_service.cc
        tests/test_onnx_yolov4_detector.cc)
target_link_libraries(test-objdet-server PUBLIC
        objdet-server-core
        gtest
        gtest_main)
target_include_directories(test-objdet-server PUBLIC
        src
        protos/build
        libs/googletest/googletest/include)
target_compile_options(test-objdet-server PUBLIC -O0 -g --coverage)
target_link_options(test-objdet-server PUBLIC --coverage)
add_test(NAME objdet_server_tests COMMAND test-objdet-server)

install(TARGETS grpc-objdet-server DESTINATION /usr/local/bin)
