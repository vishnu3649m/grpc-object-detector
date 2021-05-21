#include <iostream>

#include <loguru.hpp>
#include <CLI/CLI.hpp>
#include <absl/strings/str_format.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include "grpc_obj_det/ImageDetectionService.h"

using namespace std;

#define GRPC_OBJ_DET_VERSION_MAJOR 0
#define GRPC_OBJ_DET_VERSION_MINOR 1
#define GRPC_OBJ_DET_VERSION_PATCH 0

int main(int argc, char **argv) {
  loguru::init(argc, argv);

  bool version = false;
  string detector_type;
  int port_num = 8081;

  CLI::App app{"gRPC Object Detector"};

  app.add_flag("--version,-V", version, "Prints version info and exits");
  app.add_option("--detector-type,-d",
                 detector_type,
                 "The type of detector to serve")->required();
  app.add_option("--port,-p",
                 port_num,
                 "Port for server to listen at",
                 true);

  CLI11_PARSE(app, argc, argv);

  if (version) {
    string version_msg = absl::StrFormat("gRPC Object Detector %d.%d.%d",
                                         GRPC_OBJ_DET_VERSION_MAJOR,
                                         GRPC_OBJ_DET_VERSION_MINOR,
                                         GRPC_OBJ_DET_VERSION_PATCH);
    cout << version_msg << endl;
    return 0;
  }

  LOG_F(INFO, "Starting gRPC server...");
  std::string address = absl::StrFormat("0.0.0.0:%d", port_num);

  try {
    ObjDet::Grpc::ImageDetectionService service{detector_type};

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    LOG_F(INFO, "gRPC Object Detection Server listening on %s", address.c_str());
    server->Wait();
  } catch (ObjDet::Grpc::ImageDetectionServiceInitError &error) {
    LOG_F(ERROR, "ImageDetection Service could not start because: %s. Exiting!",
          error.what());
    return 1;
  }

  return 0;
}
