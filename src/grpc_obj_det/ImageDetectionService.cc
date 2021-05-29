
#include <chrono>

#include <absl/strings/str_format.h>
#include <loguru.hpp>
#include <opencv2/opencv.hpp>

#include "ImageDetectionService.h"


ObjDet::Grpc::ImageDetectionService::ImageDetectionService(const std::string &detector_type) {
  LOG_F(INFO, "Instantiating detector: %s", detector_type.c_str());
  detector = ObjDet::DetectorFactory::get_detector(detector_type);
  if (detector == nullptr)
    throw ImageDetectionServiceInitError(absl::StrFormat(
        "Could not instantiate detector of type: %s",
        detector_type.c_str()));
  detector->initialize();
  if (detector->is_initialized())
    LOG_F(INFO, "Successfully instantiated detector: %s", detector_type.c_str());
  else
    throw ImageDetectionServiceInitError(absl::StrFormat(
        "Could not instantiate detector of type: %s",
        detector_type.c_str()));
}

grpc::Status ObjDet::Grpc::ImageDetectionService::GetDetectableObjects(::grpc::ServerContext *context,
                                                                       const ::ObjDet::Grpc::DetectableObjectsRequest *request,
                                                                       ::ObjDet::Grpc::DetectableObjectsResponse *response) {
  LOG_F(INFO, "GetDetectableObjects request received");
  for (const auto &obj : detector->available_objects_lookup())
    response->add_available_object(obj);

  LOG_F(INFO, "Responding: OK");
  return grpc::Status::OK;
}

grpc::Status ObjDet::Grpc::ImageDetectionService::DetectImage(::grpc::ServerContext *context,
                                                              const ::ObjDet::Grpc::ImageDetectionRequest *request,
                                                              ::ObjDet::Grpc::ImageDetectionResponse *response) {
  LOG_F(INFO, "DetectImage request received");
  auto t_start = std::chrono::high_resolution_clock::now();

  std::vector<char> img_bytes(request->image().begin(), request->image().end());
  cv::Mat img = cv::imdecode(img_bytes, cv::IMREAD_COLOR);

  if (img.empty()) {
    LOG_F(INFO, "Responding: INVALID ARGUMENT - Valid image could not be "
                "parsed from the provided bytes.");
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Valid image could not be parsed from the provided bytes.");
  }

  process_image(img, *response);

  auto t_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = t_end - t_start;

  LOG_F(INFO, "Responding: OK - Took %f ms; Detections: %d",
        elapsed.count(), response->detections().size());
  return grpc::Status::OK;
}

grpc::Status ObjDet::Grpc::ImageDetectionService::DetectMultipleImages(
    grpc::ServerContext *context,
    grpc::ServerReaderWriter<ObjDet::Grpc::ImageDetectionResponse,
                             ObjDet::Grpc::ImageDetectionRequest> *stream) {

  ObjDet::Grpc::ImageDetectionRequest image_request;
  while(stream->Read(&image_request)) {
    std::unique_lock<std::mutex> lock(this->stream_mutex);
    ObjDet::Grpc::ImageDetectionResponse response;

    std::vector<char> img_bytes(image_request.image().begin(), image_request.image().end());
    cv::Mat img = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
    if (img.empty()) {
      LOG_F(WARNING, "Valid image could not be parsed from the provided bytes. "
                     "Skipping this image and sending empty list of detections");
    } else {
      process_image(img, response);
    }

    stream->Write(response);
  }

  return grpc::Status::OK;
}

void ObjDet::Grpc::ImageDetectionService::process_image(const cv::Mat &img,
                                                        ObjDet::Grpc::ImageDetectionResponse &resp) {
  cv::Size size = img.size();
  auto detections = detector->detect(img);
  for (const auto &det : detections) {
    auto *detection_msg = resp.add_detections();
    detection_msg->set_object_name(detector->class_id_to_label(det.class_id));
    detection_msg->set_confidence(det.confidence);
    detection_msg->set_top_left_x(int(det.box.left * size.width));
    detection_msg->set_top_left_y(int(det.box.top * size.height));
    detection_msg->set_width(int(det.box.width * size.width));
    detection_msg->set_height(int(det.box.height * size.height));
  }
}
