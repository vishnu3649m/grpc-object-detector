/**
 * Tests a running server using an independent client.
 */

#include <fstream>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <grpcpp/server.h>

#include "grpc_obj_det/ImageDetectionService.h"

class ImageDetectionGetDetectableObjectsTest : public ::testing::Test {
 protected:
  ImageDetectionService service;
  grpc::ServerContext context;
  ObjDet::Grpc::DetectableObjectsRequest request;
  ObjDet::Grpc::DetectableObjectsResponse response;

  grpc::Status req_() {
    return service.GetDetectableObjects(&context, &request, &response);
  }
};

class ImageDetectionDetectImageTest : public ::testing::Test {
 protected:
  ImageDetectionService service;
  grpc::ServerContext context;
  ObjDet::Grpc::ImageDetectionRequest request;
  ObjDet::Grpc::ImageDetectionResponse response;

  /* Directly populates the ImageDetectionRequest's image field. */
  void read_image(const std::string &filepath) {
    std::streampos img_size;
    char *img_buffer;
    std::ifstream img_file(filepath,
                           std::ios::in | std::ios::binary | std::ios::ate);
    if (img_file.is_open()) {
      img_size = img_file.tellg();
      img_buffer = new char[img_size];
      img_file.seekg(0, std::ios::beg);
      img_file.read(img_buffer, img_size);
      img_file.close();
      request.set_image(img_buffer, img_size);
      std::cout << "Request populated with image " << filepath << " of size "
                << img_size << std::endl;
      delete[] img_buffer;
    } else {
      std::cout << "Could not open image file!\n";
    }
  }

  grpc::Status req_() {
    return service.DetectImage(&context, &request, &response);
  }
};

TEST_F(ImageDetectionGetDetectableObjectsTest,
       EmptyRequestReturnsAllSupportedObjects) {
  grpc::Status status = req_();

  ASSERT_TRUE(status.ok());

  auto &responded_objects = response.available_object();
  ASSERT_EQ(responded_objects.size(), 2);
  EXPECT_NE(std::find(responded_objects.begin(),
                      responded_objects.end(),
                      "face"),
            responded_objects.end());
  EXPECT_NE(std::find(responded_objects.begin(),
                      responded_objects.end(),
                      "eye"),
            responded_objects.end());
}

TEST_F(ImageDetectionGetDetectableObjectsTest, ReturnsNoUnsupportedObjects) {
  request.add_object_of_interest("spacecraft");
  request.add_object_of_interest("satellite");

  grpc::Status status = req_();

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(response.available_object().size(), 0);
}

TEST_F(ImageDetectionGetDetectableObjectsTest,
       ReturnsOnlySupportedObjectsWhenExplicitlyRequested1) {
  request.add_object_of_interest("face");

  grpc::Status status = req_();

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(response.available_object().size(), 1);
  EXPECT_EQ(response.available_object().at(0), "face");
}

TEST_F(ImageDetectionGetDetectableObjectsTest,
       ReturnsOnlySupportedObjectsWhenExplicitlyRequested2) {
  request.add_object_of_interest("eye");

  grpc::Status status = req_();

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(response.available_object().size(), 1);
  EXPECT_EQ(response.available_object().at(0), "eye");
}

TEST_F(ImageDetectionGetDetectableObjectsTest,
       IsCaseInsensitiveWhenCheckingSupportedObjects1) {
  request.add_object_of_interest("Face");

  grpc::Status status = req_();

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(response.available_object().size(), 1);
  EXPECT_EQ(response.available_object().at(0), "face");
}

TEST_F(ImageDetectionGetDetectableObjectsTest,
       IsCaseInsensitiveWhenCheckingSupportedObjects2) {
  request.add_object_of_interest("eYe");

  grpc::Status status = req_();

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(response.available_object().size(), 1);
  EXPECT_EQ(response.available_object().at(0), "eye");
}

TEST_F(ImageDetectionDetectImageTest, ReturnsDetectionsForSupportedObjects) {
  request.add_object_to_detect("face");
  request.add_object_to_detect("eye");
  read_image("tests/data/faces.jpg");

  grpc::Status status = req_();

  ASSERT_TRUE(status.ok());
  EXPECT_GT(response.detections().size(), 0);
}

TEST_F(ImageDetectionDetectImageTest,
       ReturnsInvalidArgumentForEmptyObjectsRequest) {
  read_image("tests/data/faces.jpg");

  grpc::Status status = req_();

  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(response.detections().size(), 0);
}
