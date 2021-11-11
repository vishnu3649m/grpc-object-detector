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

TEST(ImageDetectionListAvailableDetectorsTest, ReturnsRegisteredCascadeFaceDetector) {
    ObjDet::Grpc::ImageDetectionService service{"cascade_face_detector"};

    grpc::ServerContext context;
    ObjDet::Grpc::AvailableDetectorsRequest request;
    ObjDet::Grpc::AvailableDetectorsResponse response;

    auto expected_labels = std::unordered_set<std::string>{"face", "eye"};

    grpc::Status status = service.ListAvailableDetectors(&context, &request, &response);
    ASSERT_TRUE(status.ok());

    auto &registered_detectors = response.detectors();
    ASSERT_EQ(registered_detectors.size(), 1);

    auto &detector = registered_detectors[0];
    ASSERT_EQ(detector.name(), "cascade_face_detector");
    ASSERT_EQ(detector.model(), "HaarCascade");

    std::unordered_set<std::string> received_labels(detector.detected_objects().begin(), detector.detected_objects().end());
    ASSERT_EQ(received_labels, expected_labels);
}

TEST(ImageDetectionListAvailableDetectorsTest, ReturnsRegisteredOnnxYoloV4CocoDetector) {
    ObjDet::Grpc::ImageDetectionService service{"onnx_yolov4_coco"};

    grpc::ServerContext context;
    ObjDet::Grpc::AvailableDetectorsRequest request;
    ObjDet::Grpc::AvailableDetectorsResponse response;

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

    std::unordered_set<std::string> expected_labels{absl::StrSplit(coco_labels, '\n')};

    grpc::Status status = service.ListAvailableDetectors(&context, &request, &response);
    ASSERT_TRUE(status.ok());

    auto &registered_detectors = response.detectors();
    ASSERT_EQ(registered_detectors.size(), 1);

    auto &detector = registered_detectors[0];
    ASSERT_EQ(detector.name(), "onnx_yolov4_coco");
    ASSERT_EQ(detector.model(), "YoloV4");

    std::unordered_set<std::string> received_labels(detector.detected_objects().begin(), detector.detected_objects().end());
    ASSERT_EQ(received_labels, expected_labels);
}

TEST(ImageDetectionDetectImageTest, ServiceWithOneRegisteredDetectorReturnsDetections) {
  ObjDet::Grpc::ImageDetectionService service("cascade_face_detector");
  grpc::ServerContext context;
  ObjDet::Grpc::ImageDetectionRequest request;
  ObjDet::Grpc::ImageDetectionResponse response;

  request.set_detector_name("cascade_face_detector");

  std::streampos img_size;
  char *img_buffer;
  std::ifstream img_file("tests/data/faces.jpg",
                         std::ios::in | std::ios::binary | std::ios::ate);
  if (img_file.is_open()) {
    img_size = img_file.tellg();
    img_buffer = new char[img_size];
    img_file.seekg(0, std::ios::beg);
    img_file.read(img_buffer, img_size);
    img_file.close();
    request.set_image(img_buffer, img_size);
    delete[] img_buffer;
  } else {
    std::cout << "Could not open image file!\n";
  }

  grpc::Status status = service.DetectImage(&context, &request, &response);

  ASSERT_TRUE(status.ok());
  EXPECT_GT(response.detections().size(), 0);
}


TEST(ImageDetectionDetectImageTest, ServiceWithMultipleRegisteredDetectorsRunsSpecifiedDetector) {
  ObjDet::Grpc::ImageDetectionService service{"cascade_face_detector", "onnx_yolov4_coco"};
  grpc::ServerContext context;
  ObjDet::Grpc::ImageDetectionRequest request;
  ObjDet::Grpc::ImageDetectionResponse response;

  request.set_detector_name("onnx_yolov4_coco");

  std::streampos img_size;
  char *img_buffer;
  std::ifstream img_file("tests/data/horses.jpg",
                         std::ios::in | std::ios::binary | std::ios::ate);
  if (img_file.is_open()) {
    img_size = img_file.tellg();
    img_buffer = new char[img_size];
    img_file.seekg(0, std::ios::beg);
    img_file.read(img_buffer, img_size);
    img_file.close();
    request.set_image(img_buffer, img_size);
    delete[] img_buffer;
  } else {
    std::cout << "Could not open image file!\n";
  }

  grpc::Status status = service.DetectImage(&context, &request, &response);

  ASSERT_TRUE(status.ok());

  auto& detections = response.detections();
  ASSERT_EQ(detections.size(), 5);
  for (const auto& det : detections) {
     ASSERT_EQ(det.object_name(), "horse");
  }
}
