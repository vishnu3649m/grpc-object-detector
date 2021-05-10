/**
 * Concrete defs for C++ implementation of the specified gRPC services.
 */

#ifndef GRPC_OBJ_DET_IMAGEDETECTIONSERVICE_H
#define GRPC_OBJ_DET_IMAGEDETECTIONSERVICE_H

#include <grpc/grpc.h>

#include "image_detection.grpc.pb.h"
#include "DetectorFactory.h"

class ImageDetectionService final : public ::ObjDet::Grpc::ImageDetection::Service {
 public:
  explicit ImageDetectionService(const std::string &detector_type);

  grpc::Status GetDetectableObjects(::grpc::ServerContext *context,
                                    const ::ObjDet::Grpc::DetectableObjectsRequest *request,
                                    ::ObjDet::Grpc::DetectableObjectsResponse *response) override;

  grpc::Status DetectImage(::grpc::ServerContext *context,
                           const ::ObjDet::Grpc::ImageDetectionRequest *request,
                           ::ObjDet::Grpc::ImageDetectionResponse *response) override;

 private:
  std::unique_ptr<ObjDet::DetectorInterface> detector;
};


#endif //GRPC_OBJ_DET_IMAGEDETECTIONSERVICE_H