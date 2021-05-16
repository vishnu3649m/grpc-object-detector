/**
 * Tests a running server using an independent client.
 */

#include <fstream>
#include <iostream>

#include <absl/strings/str_split.h>
#include <gtest/gtest.h>
#include <grpcpp/server.h>

#include "grpc_obj_det/ImageDetectionService.h"

TEST(ImageDetectionServiceTest, RaisesExceptionWhenUnknownDetectorIsSpecified) {
  ASSERT_THROW(ObjDet::Grpc::ImageDetectionService("some_unknown_detector"),
               ObjDet::Grpc::ImageDetectionServiceInitError);
}

class ImageDetectionDetectImageTest : public ::testing::Test {
 protected:
  ObjDet::Grpc::ImageDetectionService service{"cascade_face_detector"};
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

TEST(ImageDetectionGetDetectableObjectsTest,
       ReturnsAllObjectsFromCascadeFaceDetector) {
  ObjDet::Grpc::ImageDetectionService service{"cascade_face_detector"};
  grpc::ServerContext context;
  ObjDet::Grpc::DetectableObjectsRequest request;
  ObjDet::Grpc::DetectableObjectsResponse response;

  grpc::Status status = service.GetDetectableObjects(&context, &request, &response);

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

TEST(ImageDetectionGetDetectableObjectsTest,
     ReturnsAllObjectsFromDetectorTrainedOnCocoDataset) {
  std::string coco_labels = "person\n"
                            "bicycle\n"
                            "car\n"
                            "motorbike\n"
                            "aeroplane\n"
                            "bus\n"
                            "train\n"
                            "truck\n"
                            "boat\n"
                            "traffic light\n"
                            "fire hydrant\n"
                            "stop sign\n"
                            "parking meter\n"
                            "bench\n"
                            "bird\n"
                            "cat\n"
                            "dog\n"
                            "horse\n"
                            "sheep\n"
                            "cow\n"
                            "elephant\n"
                            "bear\n"
                            "zebra\n"
                            "giraffe\n"
                            "backpack\n"
                            "umbrella\n"
                            "handbag\n"
                            "tie\n"
                            "suitcase\n"
                            "frisbee\n"
                            "skis\n"
                            "snowboard\n"
                            "sports ball\n"
                            "kite\n"
                            "baseball bat\n"
                            "baseball glove\n"
                            "skateboard\n"
                            "surfboard\n"
                            "tennis racket\n"
                            "bottle\n"
                            "wine glass\n"
                            "cup\n"
                            "fork\n"
                            "knife\n"
                            "spoon\n"
                            "bowl\n"
                            "banana\n"
                            "apple\n"
                            "sandwich\n"
                            "orange\n"
                            "broccoli\n"
                            "carrot\n"
                            "hot dog\n"
                            "pizza\n"
                            "donut\n"
                            "cake\n"
                            "chair\n"
                            "sofa\n"
                            "potted plant\n"
                            "bed\n"
                            "dining table\n"
                            "toilet\n"
                            "tvmonitor\n"
                            "laptop\n"
                            "mouse\n"
                            "remote\n"
                            "keyboard\n"
                            "cell phone\n"
                            "microwave\n"
                            "oven\n"
                            "toaster\n"
                            "sink\n"
                            "refrigerator\n"
                            "book\n"
                            "clock\n"
                            "vase\n"
                            "scissors\n"
                            "teddy bear\n"
                            "hair drier\n"
                            "toothbrush";

  std::unordered_set<std::string> expected_labels {absl::StrSplit(coco_labels, '\n')};

  ObjDet::Grpc::ImageDetectionService service{"onnx_yolov4_coco"};
  grpc::ServerContext context;
  ObjDet::Grpc::DetectableObjectsRequest request;
  ObjDet::Grpc::DetectableObjectsResponse response;

  grpc::Status status = service.GetDetectableObjects(&context, &request, &response);

  ASSERT_TRUE(status.ok());

  std::unordered_set<std::string> returned_labels(response.available_object().begin(), response.available_object().end());
  ASSERT_EQ(returned_labels, expected_labels);
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
