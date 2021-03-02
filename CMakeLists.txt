cmake_minimum_required(VERSION 3.15)
project(grpc_va_server)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(OPENCV REQUIRED opencv)

add_subdirectory(libs/googletest)
add_subdirectory(libs/abseil-cpp)
add_library(loguru STATIC libs/loguru/loguru.cpp libs/loguru/loguru.hpp)

# Core VA server lib
add_library(va_server_core STATIC
        src/VideoAnalyzer/DetectorInterface.h
        src/VideoAnalyzer/FaceEyesDetector.h src/VideoAnalyzer/FaceEyesDetector.cpp)
target_include_directories(va_server_core PUBLIC src libs/loguru libs/CLI11/include libs/abseil-cpp/)
target_link_libraries(va_server_core PUBLIC loguru pthread dl ${OPENCV_LIBRARIES} absl::strings absl::str_format)

add_executable(grpc_va_server src/main.cpp)
target_link_libraries(grpc_va_server PUBLIC loguru va_server_core pthread dl ${OPENCV_LIBRARIES})

add_executable(test_va_server tests/test_detector_interface.cpp)
target_link_libraries(test_va_server PUBLIC gtest_main va_server_core absl::strings absl::str_format)
target_include_directories(test_va_server PUBLIC src libs/googletest/googletest/include libs/abseil-cpp)
add_test(NAME va_server_tests COMMAND test_va_server)

install(TARGETS grpc_va_server)
