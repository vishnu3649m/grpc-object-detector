/**
 * Tests a running server using an independent client.
 */

#include <fstream>
#include <iostream>
#include <thread>
#include <string>

#include <gtest/gtest.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/create_channel.h>

#include "VideoAnalyzer/ImageDetectionService.h"

class ImageDetectionClient {
    std::unique_ptr<VA::Grpc::ImageDetection::Stub> stub_;

public:
    explicit ImageDetectionClient(const std::shared_ptr<grpc::ChannelInterface>& channel)
        : stub_{VA::Grpc::ImageDetection::NewStub(channel)} {

    }

    std::pair<grpc::StatusCode, std::vector<std::string>>
    GetDetectableObjects(const std::vector<std::string> &desired_objects = std::vector<std::string>{}) {
        grpc::ClientContext context;
        VA::Grpc::DetectableObjectsResponse response;
        VA::Grpc::DetectableObjectsRequest request;
        std::vector<std::string> available_objects;

        for (const auto &object : desired_objects)
            request.add_object_of_interest(object);
        grpc::Status status = stub_->GetDetectableObjects(
                &context, request, &response);

        if (status.ok()) {
            for (const auto &object : response.available_object())
                available_objects.push_back(object);
        }

        return {status.error_code(), available_objects};
    }

    std::pair<grpc::StatusCode, std::vector<VA::Grpc::Detection>>
    DetectImage(const std::vector<std::string> &objects, const std::string &img_file_path) {
        grpc::ClientContext context;
        VA::Grpc::ImageDetectionRequest request;
        VA::Grpc::ImageDetectionResponse response;
        std::pair<grpc::StatusCode, std::vector<VA::Grpc::Detection>> response_info;

        for (const auto &object : objects)
            request.add_object_to_detect(object);

        std::streampos img_size;
        char *img_buffer;
        std::ifstream img_file(img_file_path, std::ios::in | std::ios::binary | std::ios::ate);
        if (img_file.is_open()) {
            img_size = img_file.tellg();
            img_buffer = new char[img_size];
            img_file.seekg(0, std::ios::beg);
            img_file.read(img_buffer, img_size);
            img_file.close();
            request.set_image(img_buffer, img_size);
            std::cout << "Read in image of size " << img_size << std::endl;
            delete [] img_buffer;
        } else {
            std::cout << "Could not open image file!\n";
        }

        grpc::Status status = stub_->DetectImage(&context, request, &response);

        if (status.ok()) {
            for (const auto &det : response.detections())
                response_info.second.push_back(det);
        }

        response_info.first = status.error_code();
        return response_info;
    }
};

void grpc_server_task(std::unique_ptr<grpc::Server> &server) {
    std::string address = "127.0.0.1:8081";
    ImageDetectionService service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server_(builder.BuildAndStart());
    server = std::move(server_);
    if (server != nullptr) {
        std::cout << "gRPC server under test running at: " << address << std::endl;
        server->Wait();
    }
}

class GrpcServerUnderTest : public ::testing::Test {
    std::unique_ptr<grpc::Server> server{nullptr};
    std::thread t;

protected:
    GrpcServerUnderTest() {
        if (server == nullptr) {
            std::cout << "Starting gRPC server under test...\n";
            std::thread t_(grpc_server_task, std::ref(server));
            t = std::move(t_);
        }
    }

    ~GrpcServerUnderTest() override {
        if (server != nullptr) {
            std::cout << "Shutting down gRPC server under test...\n";
            server->Shutdown();
        }
        t.join();
        server.reset(nullptr);
        std::cout << "gRPC server under test was shut down\n";
    }
};

TEST_F(GrpcServerUnderTest, DetectableObjectsRequestReturnsAllSupportedObjects) {
    ImageDetectionClient client(grpc::CreateChannel("localhost:8081",
                                                    grpc::InsecureChannelCredentials()));
    auto response = client.GetDetectableObjects();
    EXPECT_EQ(response.first, grpc::StatusCode::OK);
    ASSERT_EQ(response.second.size(), 2);
    EXPECT_NE(std::find(response.second.begin(), response.second.end(), "face"),
              response.second.end());
    EXPECT_NE(std::find(response.second.begin(), response.second.end(), "eye"),
              response.second.end());
}

TEST_F(GrpcServerUnderTest, DetectableObjectsRequestReturnsNoUnsupportedObjects) {
    ImageDetectionClient client(grpc::CreateChannel("localhost:8081",
                                                    grpc::InsecureChannelCredentials()));

    std::vector<std::string> objects_to_request = {"spacecraft", "satellite"};
    auto response = client.GetDetectableObjects(objects_to_request);
    EXPECT_EQ(response.first, grpc::StatusCode::OK);
    EXPECT_EQ(response.second.size(), 0);
}

TEST_F(GrpcServerUnderTest, DetectableObjectsRequestReturnsOnlySupportedObjectsWhenExplicitlyRequested) {
    ImageDetectionClient client(grpc::CreateChannel("localhost:8081",
                                                    grpc::InsecureChannelCredentials()));

    std::vector<std::string> objects_to_request = {"face"};
    auto response = client.GetDetectableObjects(objects_to_request);
    EXPECT_EQ(response.first, grpc::StatusCode::OK);
    EXPECT_EQ(response.second.size(), 1);
    EXPECT_EQ(response.second[0], "face");

    objects_to_request = {"eye"};
    response = client.GetDetectableObjects(objects_to_request);
    EXPECT_EQ(response.first, grpc::StatusCode::OK);
    EXPECT_EQ(response.second.size(), 1);
    EXPECT_EQ(response.second[0], "eye");
}

TEST_F(GrpcServerUnderTest, DetectableObjectsRequestTreatedCaseInsensitiveWhenCheckingSupportedObjects) {
    ImageDetectionClient client(grpc::CreateChannel("localhost:8081",
                                                    grpc::InsecureChannelCredentials()));

    std::vector<std::string> objects_to_request = {"Face"};
    auto response = client.GetDetectableObjects(objects_to_request);
    EXPECT_EQ(response.first, grpc::StatusCode::OK);
    EXPECT_EQ(response.second.size(), 1);
    EXPECT_EQ(response.second[0], "face");

    objects_to_request = {"eYe"};
    response = client.GetDetectableObjects(objects_to_request);
    EXPECT_EQ(response.first, grpc::StatusCode::OK);
    EXPECT_EQ(response.second.size(), 1);
    EXPECT_EQ(response.second[0], "eye");
}

TEST_F(GrpcServerUnderTest, DetectImageRequestReturnsDetectionsForSupportedObjects) {
    ImageDetectionClient client(grpc::CreateChannel("localhost:8081",
                                                    grpc::InsecureChannelCredentials()));

    std::vector<std::string> objects_to_detect = {"face", "eye"};
    auto response = client.DetectImage(objects_to_detect, "tests/data/faces.jpg");
    EXPECT_EQ(response.first, grpc::StatusCode::OK);
    EXPECT_GT(response.second.size(), 0);
}

TEST_F(GrpcServerUnderTest, DetectImageRequestReturnsInvalidArgumentForEmptyObjectsRequest) {
    ImageDetectionClient client(grpc::CreateChannel("localhost:8081",
                                                    grpc::InsecureChannelCredentials()));

    std::vector<std::string> objects_to_detect;
    auto response = client.DetectImage(objects_to_detect, "tests/data/faces.jpg");
    EXPECT_EQ(response.first, grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.second.size(), 0);
}
