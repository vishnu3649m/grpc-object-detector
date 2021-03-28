/**
 * \brief Yolo Detector implemented using the ONNX Runtime
 */

#ifndef GRPC_VA_SERVER_SRC_VIDEOANALYZER_ONNXYOLODETECTOR_H_
#define GRPC_VA_SERVER_SRC_VIDEOANALYZER_ONNXYOLODETECTOR_H_

#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

#include "DetectorInterface.h"

namespace VA {

/**
 * A YOLOv4 detector served using the ONNX runtime. To initialize and run this
 * detector, the following configuration files are needed:
 * * ONNX model file containing the model architecture and weights.
 * * Text file containing the anchor dimensions used when training the model.
 * * Text file containing class labels.
 */
class OnnxYoloDetector : public DetectorInterface {
  std::string onnx_model_path;
  std::string anchors_file_path;
  std::string class_labels_path;
  bool init = false;

  Ort::Experimental::Session *session = nullptr;
  std::vector<std::vector<int64_t>> anchors;
  std::vector<std::string> class_labels;
  int64_t input_w = 0;
  int64_t input_h = 0;

 public:
  OnnxYoloDetector(std::string _onnx_model_path,
                   std::string _anchors_file_path,
                   std::string _class_labels_path) :
      onnx_model_path{std::move(_onnx_model_path)},
      anchors_file_path{std::move(_anchors_file_path)},
      class_labels_path{std::move(_class_labels_path)} { }

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
