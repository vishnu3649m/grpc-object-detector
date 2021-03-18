//
// Created by Vishnu on 3/3/21.
//

#ifndef GRPC_VA_SERVER_IMAGEDETECTIONSERVICE_H
#define GRPC_VA_SERVER_IMAGEDETECTIONSERVICE_H

#include <grpc/grpc.h>

#include "image_detection.grpc.pb.h"

class ImageDetectionService final : public ::VA::Grpc::ImageDetection::Service {
public:
    grpc::Status GetDetectableObjects(::grpc::ServerContext* context,
                                      const ::VA::Grpc::DetectableObjectsRequest* request,
                                      ::VA::Grpc::DetectableObjectsResponse* response) override;
    grpc::Status DetectImage(::grpc::ServerContext* context,
                             const ::VA::Grpc::ImageDetectionRequest* request,
                             ::VA::Grpc::ImageDetectionResponse* response) override;
};


#endif //GRPC_VA_SERVER_IMAGEDETECTIONSERVICE_H
