#include <iostream>

#include <loguru.hpp>
#include <CLI/CLI.hpp>
#include <absl/strings/str_format.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include "VideoAnalyzer/ImageDetectionService.h"

using namespace std;

#define VIDEO_ANALYZER_VERSION_MAJOR 0
#define VIDEO_ANALYZER_VERSION_MINOR 1
#define VIDEO_ANALYZER_VERSION_PATCH 0

int main(int argc, char **argv) {
    CLI::App app{"gRPC Video Analyzer"};

    bool version = false;
    string action;

    app.add_flag("--version,-V", version, "Prints version info and exits");
    app.add_option("action", action, "Server action to perform")->check(CLI::IsMember({"start"}));

    CLI11_PARSE(app, argc, argv);

    if (version) {
        string version_msg = absl::StrFormat("gRPC Video Analyzer %d.%d.%d",
                                             VIDEO_ANALYZER_VERSION_MAJOR,
                                             VIDEO_ANALYZER_VERSION_MINOR,
                                             VIDEO_ANALYZER_VERSION_PATCH);
        cout << version_msg << endl;
        return 0;
    }

    loguru::init(argc, argv);

    LOG_F(INFO, "Starting gRPC server...");
    std::string address = "127.0.0.1:8081";
    ImageDetectionService service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    LOG_F(INFO, "gRPC Video Analyzer Server listening on %s", address.c_str());
    server->Wait();

    return 0;
}
