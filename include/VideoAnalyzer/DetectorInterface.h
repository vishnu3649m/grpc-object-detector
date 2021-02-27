//
// Created by Vishnu on 18/2/21.
//

#ifndef GRPC_VA_SERVER_DETECTORINTERFACE_H
#define GRPC_VA_SERVER_DETECTORINTERFACE_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace VA {

struct RectTLWH {
    float top;
    float left;
    float width;
    float height;
};

struct Detection {
    int class_id;
    RectTLWH box;
    float confidence;
};

class DetectorInterface {
public:
    virtual void initialize() = 0;
    virtual std::vector<Detection> detect(const cv::Mat & img) = 0;
    virtual std::string class_id_to_label(int class_id) = 0;
    virtual ~DetectorInterface() = default;
};

}

#endif //GRPC_VA_SERVER_DETECTORINTERFACE_H
