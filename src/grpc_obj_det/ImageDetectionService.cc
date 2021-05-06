//
// Created by Vishnu on 3/3/21.
//

#include <opencv2/opencv.hpp>

#include "FaceEyesDetector.h"
#include "ImageDetectionService.h"

grpc::Status ImageDetectionService::GetDetectableObjects(::grpc::ServerContext *context,
                                                         const ::ObjDet::Grpc::DetectableObjectsRequest *request,
                                                         ::ObjDet::Grpc::DetectableObjectsResponse *response) {
  std::unordered_set<std::string> valid_objects{"face", "eye"};

  if (request->object_of_interest_size()) {
    for (auto obj : request->object_of_interest()) {
      std::transform(obj.begin(), obj.end(), obj.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
      if (valid_objects.count(obj))
        response->add_available_object(obj);
    }
  } else {
    for (const auto &obj : valid_objects)
      response->add_available_object(obj);
  }

  return grpc::Status::OK;
}

grpc::Status ImageDetectionService::DetectImage(::grpc::ServerContext *context,
                                                const ::ObjDet::Grpc::ImageDetectionRequest *request,
                                                ::ObjDet::Grpc::ImageDetectionResponse *response) {
  std::vector<char> img_bytes(request->image().begin(), request->image().end());
  cv::Mat img = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
  std::unordered_set<std::string> valid_objects = {"face", "eye"};
  std::unordered_set<std::string> requested_objects;

  if (img.empty())
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Valid image could not be parsed from the provided bytes.");

  for (auto obj : request->object_to_detect()) {
    std::transform(obj.begin(), obj.end(), obj.begin(), [](unsigned char c) {
      return std::tolower(c);
    });
    if (valid_objects.count(obj))
      requested_objects.insert(obj);
  }
  if (requested_objects.empty())
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "At least 1 object detectable by this server needs to be provided. "
                        "Refer to GetDetectableObjects RPC for supported objects.");

  cv::Size size = img.size();
  ObjDet::FaceEyesDetector face_detector(
      "config/cascade_face_detector/haarcascade_frontalface_alt.xml",
      "config/cascade_face_detector/haarcascade_eye_tree_eyeglasses.xml");
  face_detector.initialize();
  if (!face_detector.is_initialized())
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Server could not initialize necessary resources to process this request.");

  auto detections = face_detector.detect(img);
  for (const auto &det : detections) {
    auto *detection_msg = response->add_detections();
    detection_msg->set_object_name(face_detector.class_id_to_label(det.class_id));
    detection_msg->set_confidence(det.confidence);
    detection_msg->set_top_left_x(int(det.box.left * size.width));
    detection_msg->set_top_left_y(int(det.box.top * size.height));
    detection_msg->set_width(int(det.box.width * size.width));
    detection_msg->set_height(int(det.box.height * size.height));
  }

  return grpc::Status::OK;
}
