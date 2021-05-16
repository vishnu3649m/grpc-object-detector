
#include <absl/strings/str_format.h>
#include <opencv2/opencv.hpp>

#include "FaceEyesDetector.h"
#include "ImageDetectionService.h"


ObjDet::Grpc::ImageDetectionService::ImageDetectionService(const std::string &detector_type) {
  detector = ObjDet::DetectorFactory::get_detector(detector_type);
  if (detector == nullptr)
    throw ImageDetectionServiceInitError(absl::StrFormat(
        "Could not instantiate detector of type: %s",
        detector_type.c_str()));
  detector->initialize();
}

grpc::Status ObjDet::Grpc::ImageDetectionService::GetDetectableObjects(::grpc::ServerContext *context,
                                                                       const ::ObjDet::Grpc::DetectableObjectsRequest *request,
                                                                       ::ObjDet::Grpc::DetectableObjectsResponse *response) {
  for (const auto &obj : detector->available_objects_lookup())
    response->add_available_object(obj);

  return grpc::Status::OK;
}

grpc::Status ObjDet::Grpc::ImageDetectionService::DetectImage(::grpc::ServerContext *context,
                                                              const ::ObjDet::Grpc::ImageDetectionRequest *request,
                                                              ::ObjDet::Grpc::ImageDetectionResponse *response) {
  std::vector<char> img_bytes(request->image().begin(), request->image().end());
  cv::Mat img = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
  std::unordered_set<std::string> valid_objects = detector->available_objects_lookup();
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

  auto detections = detector->detect(img);
  for (const auto &det : detections) {
    auto *detection_msg = response->add_detections();
    detection_msg->set_object_name(detector->class_id_to_label(det.class_id));
    detection_msg->set_confidence(det.confidence);
    detection_msg->set_top_left_x(int(det.box.left * size.width));
    detection_msg->set_top_left_y(int(det.box.top * size.height));
    detection_msg->set_width(int(det.box.width * size.width));
    detection_msg->set_height(int(det.box.height * size.height));
  }

  return grpc::Status::OK;
}
