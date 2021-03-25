/**
 * \brief Yolo Detector implemented using the ONNX Runtime
 */

#ifndef GRPC_VA_SERVER_SRC_VIDEOANALYZER_ONNXYOLODETECTOR_H_
#define GRPC_VA_SERVER_SRC_VIDEOANALYZER_ONNXYOLODETECTOR_H_

#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

#include "DetectorInterface.h"

namespace VA {

class OnnxYoloDetector : public DetectorInterface {
  std::string onnx_model_path;
  std::string anchors_file_path;
  bool init = false;

  Ort::Experimental::Session *session = nullptr;
  int64_t input_w = 0;
  int64_t input_h = 0;

 public:
  explicit OnnxYoloDetector(std::string _onnx_model_path,
                            std::string _anchors_file_path) :
      onnx_model_path{std::move(_onnx_model_path)},
      anchors_file_path{std::move(_anchors_file_path)} { }

  ~OnnxYoloDetector() override {
    delete session;
  };

  void initialize() override;

  std::vector<Detection> detect(const cv::Mat &img) override;

  std::string class_id_to_label(int class_id) const override;

  bool is_initialized() const override;
};

}

#endif //GRPC_VA_SERVER_SRC_VIDEOANALYZER_ONNXYOLODETECTOR_H_
