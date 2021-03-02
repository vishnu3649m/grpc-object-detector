//
// Created by Vishnu on 18/2/21.
//

#ifndef GRPC_VA_SERVER_FACEEYESDETECTOR_H
#define GRPC_VA_SERVER_FACEEYESDETECTOR_H

#include <string>
#include <utility>
#include <opencv2/objdetect.hpp>

#include "DetectorInterface.h"

namespace VA {

class FaceEyesDetector : public DetectorInterface {
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;
    std::string face_config_file;
    std::string eyes_config_file;
    std::vector<std::string> class_label_map;
    bool init = false;

public:
    FaceEyesDetector(std::string face_cascade_file,
                     std::string eyes_cascade_file) :
        face_config_file{std::move(face_cascade_file)},
        eyes_config_file{std::move(eyes_cascade_file)}
        {}
    ~FaceEyesDetector() override = default;

    void initialize() override;
    std::vector<Detection> detect(const cv::Mat & img) override;
    std::string class_id_to_label(int class_id) const override;
    bool is_initialized() const override;
};

}

#endif //GRPC_VA_SERVER_FACEEYESDETECTOR_H
