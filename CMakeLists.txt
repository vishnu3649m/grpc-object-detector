cmake_minimum_required(VERSION 3.13)
project(grpc-va-server)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime build options
set(onnxruntime_RUN_ONNX_TESTS OFF)
set(onnxruntime_BUILD_WINML_TESTS OFF)
set(onnxruntime_GENERATE_TEST_REPORTS OFF)
set(onnxruntime_BUILD_SHARED_LIB ON)
set(onnxruntime_PREFER_SYSTEM_LIB ON)
set(onnxruntime_BUILD_UNIT_TESTS OFF)
set(onnxruntime_USE_OPENMP ON)
set(NSYNC_ENABLE_TESTS OFF)

# check for and build dependencies
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(absl REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv)
add_subdirectory(libs/onnxruntime/cmake/external/nsync)
add_subdirectory(libs/onnxruntime/cmake)
add_subdirectory(libs/googletest)
add_subdirectory(protos/)
add_library(loguru STATIC libs/loguru/loguru.cpp libs/loguru/loguru.hpp)

# Core VA server lib
add_library(va_server_core STATIC
        src/VideoAnalyzer/DetectorInterface.h
        src/VideoAnalyzer/FaceEyesDetector.h src/VideoAnalyzer/FaceEyesDetector.cc
        src/VideoAnalyzer/ImageDetectionService.cc src/VideoAnalyzer/ImageDetectionService.h
        src/VideoAnalyzer/OnnxYoloDetector.cc src/VideoAnalyzer/OnnxYoloDetector.h src/VideoAnalyzer/onnx_yolo_detector_utils.cc)
target_include_directories(va_server_core PUBLIC
        protos/build
        libs/loguru
        libs/CLI11/include
        libs/onnxruntime/include
        libs/xtl/include
        libs/xtensor/include)
link_directories(va_server_core PUBLIC libs/onnxruntime/build/Linux/Release)
target_link_libraries(va_server_core PUBLIC
        va_grpc_proto
        loguru
        pthread
        dl
        ${OPENCV_LIBRARIES}
        absl::strings
        absl::str_format
        onnxruntime)

add_executable(grpc-va-server src/main.cc)
target_link_libraries(grpc-va-server PUBLIC loguru va_server_core)

add_executable(test-va-server
        tests/test_detector_interface.cc
        tests/test_img_det_service.cc
        tests/test_onnx_yolov4_detector.cc)
target_link_libraries(test-va-server PUBLIC
        gtest_main
        va_server_core
        absl::strings
        absl::str_format)
target_include_directories(test-va-server PUBLIC src libs/googletest/googletest/include)
add_test(NAME va_server_tests COMMAND test-va-server)

install(TARGETS grpc-va-server DESTINATION /usr/local/bin)
