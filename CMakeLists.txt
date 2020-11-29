cmake_minimum_required(VERSION 3.15)
project(grpc_va_server)

set(CMAKE_CXX_STANDARD 17)

add_library(loguru STATIC libs/loguru/loguru.cpp libs/loguru/loguru.hpp)

add_executable(grpc_va_server src/main.cpp)

target_include_directories(grpc_va_server PUBLIC libs/)
target_link_libraries(grpc_va_server PRIVATE loguru PUBLIC pthread dl)

install(TARGETS grpc_va_server)
