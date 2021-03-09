# adapted from https://github.com/grpc/grpc/blob/master/examples/cpp/cmake/common.cmake
# and https://github.com/grpc/grpc/blob/master/examples/cpp/route_guide/CMakeLists.txt

# assumes gRPC and its dependencies are installed and findable by cmake

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

# find protobuf installation
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
find_program(_PROTOBUF_PROTOC protoc)

# find grpc installation
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_GRPC_GRPCPP gRPC::grpc++)
set(_REFLECTION gRPC::grpc++_reflection)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

message(STATUS "Configuring Makefile for protos in ${CMAKE_CURRENT_SOURCE_DIR}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(image_detection_proto_srcs "${CMAKE_CURRENT_SOURCE_DIR}/build/image_detection.pb.cc")
set(image_detection_proto_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/build/image_detection.pb.h")
set(image_detection_grpc_srcs "${CMAKE_CURRENT_SOURCE_DIR}/build/image_detection.grpc.pb.cc")
set(image_detection_grpc_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/build/image_detection.grpc.pb.h")
add_custom_command(
        OUTPUT "${image_detection_proto_srcs}" "${image_detection_proto_hdrs}" "${image_detection_grpc_srcs}" "${image_detection_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_SOURCE_DIR}/build"
        --cpp_out "${CMAKE_CURRENT_SOURCE_DIR}/build"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "image_detection.proto"
        DEPENDS "image_detection.proto")

add_library(va_grpc_proto STATIC
        ${image_detection_proto_srcs}
        ${image_detection_proto_hdrs}
        ${image_detection_grpc_srcs}
        ${image_detection_grpc_hdrs})
target_link_libraries(va_grpc_proto PUBLIC ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})
