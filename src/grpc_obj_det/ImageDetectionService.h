/**
 * Concrete defs for C++ implementation of the specified gRPC services.
 */

#ifndef GRPC_OBJ_DET_IMAGEDETECTIONSERVICE_H
#define GRPC_OBJ_DET_IMAGEDETECTIONSERVICE_H

#include <grpc/grpc.h>

#include "image_detection.grpc.pb.h"
#include "DetectorFactory.h"

namespace ObjDet::Grpc {

class ImageDetectionServiceInitError : public std::runtime_error {
 public:
  explicit ImageDetectionServiceInitError(const std::string& arg)
      : std::runtime_error(arg) {
  }
};

class ImageDetectionService final : public ::ObjDet::Grpc::ImageDetection::Service {
 public:
  explicit ImageDetectionService(const std::string &detector_type);
  ImageDetectionService(std::initializer_list<std::string> det_types);
  explicit ImageDetectionService(std::vector<std::string> det_types);

  grpc::Status ListAvailableDetectors(::grpc::ServerContext* context,
                                      const ::ObjDet::Grpc::AvailableDetectorsRequest* request,
                                      ::ObjDet::Grpc::AvailableDetectorsResponse* response) override;

  grpc::Status DetectImage(::grpc::ServerContext *context,
                           const ::ObjDet::Grpc::ImageDetectionRequest *request,
                           ::ObjDet::Grpc::ImageDetectionResponse *response) override;

  grpc::Status DetectMultipleImages(grpc::ServerContext *context,
                                    grpc::ServerReaderWriter<ObjDet::Grpc::ImageDetectionResponse,
                                                             ObjDet::Grpc::ImageDetectionRequest> *stream) override;

 private:
  std::unordered_map<std::string, std::unique_ptr<ObjDet::DetectorInterface>> detectors;
  std::mutex stream_mutex;
};

}


#endif //GRPC_OBJ_DET_IMAGEDETECTIONSERVICE_H
