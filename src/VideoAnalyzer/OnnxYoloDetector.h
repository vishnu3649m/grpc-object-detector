/**
 * \brief Yolo Detector implemented using the ONNX Runtime
 */

#ifndef GRPC_VA_SERVER_SRC_VIDEOANALYZER_ONNXYOLODETECTOR_H_
#define GRPC_VA_SERVER_SRC_VIDEOANALYZER_ONNXYOLODETECTOR_H_

#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>
#include <xtensor/xarray.hpp>

#include "DetectorInterface.h"

inline std::string print_shape(const std::vector<int64_t> &v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << " x ";
  ss << v[v.size() - 1];
  return ss.str();
}

/**
 * Reads in anchors from a provided text file.
 *
 * Each YOLO model is trained to predict object bounding boxes around a set of
 * anchors. After the image is passed through the neural network, the output
 * tensor(s) need to be parsed and transformed using the same set of anchors.
 * Each output layer should have the same number of anchors. The file should
 * contain (# output layers x # anchors) anchors presented as (w, h) pairs of
 * integer values in a single line.
 *
 * For example, for a YOLO model with 3 output layers & 3 anchors per layer:
 * ```
 * 12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401
 * ```
 *
 * @param filepath  Path of file containing anchor dimensions.
 * @param n_anchors Number of anchors per output layer.
 * @param anchors   Object ref where the parsed anchors should be stored.
 * @return          `true` if expected number of valid anchors were successfully
 *                  parsed and stored in `anchors`; `false` otherwise. If
 *                  `false`, none of the elements within `anchors` should be
 *                  treated as valid.
 */
bool read_anchors(const std::string &filepath,
                  int64_t n_anchors,
                  std::vector<int64_t> &anchors);

/**
 * Reads in human-readable labels of each class detected by the YOLO model.
 *
 * The YOLO model predicts the presence of object classes based on 0-indexing.
 * This function retrieves the human friendly label for each class index (or ID)
 * from a file. The file is expected to contain one label per line, starting
 * with the label for class ID 0, and there should be `n_classes` lines.
 *
 * @param filepath      Path to text file containing class labels.
 * @param n_classes     Number of classes that the model detects.
 * @param cls_labels    Destination vector to store labels in. **The vector
 *                      would be cleared before adding labels!**
 * @return              `true` if the expected number of class labels were read
 *                      successfully; `false` otherwise.
 */
bool read_labels(const std::string &filepath,
                 int64_t n_classes,
                 std::vector<std::string> &cls_labels);

/**
 * Preprocess an image to according to the YOLO model's input requirements.
 *
 * @param img           Image to preprocess
 * @param target_size   Model's input resolution
 * @return              A stream of floating point values representing each
 *                      pixel values as (R, G & B) in row-major order.
 */
std::vector<float> preprocess_image(const cv::Mat &img,
                                    int64_t target_size);

/**
 * Gets all predictions (bounding boxes, objectness probabilities and class
 * probabilities) from all 3 YOLO output layers.
 *
 * @param outputs       Outputs from ONNX session inference
 * @param anchors       Anchors for this YOLO model
 * @param input_size    Input resolution of this YOLO model
 * @return              A tensor of shape n_batch x (total grid * anchors boxes) x (5 + n_classes)
 */
xt::xarray<float> get_all_predictions(const std::vector<Ort::Value> &outputs,
                                      const std::vector<int64_t> &anchors,
                                      int64_t input_size);

/**
 * Filters away predicted boxes that are invalid or have a detection confidence
 * score less than the desired threshold.
 *
 * @param preds         xt::xarray containing all predictions for an input.
 *                      Should be of shape n_batch x n_preds x (5 + n_classes).
 * @param img_h         Height of original image
 * @param img_w         Width of original image
 * @param input_size    Model input resolution
 * @param threshold     Detection confidence threshold
 * @return              A vector of size n_batch containing a vector of filtered
 *                      detections for each image in the batch.
 */
std::vector<std::vector<VA::Detection>> filter_predictions(xt::xarray<float> &preds,
                                                           float img_h,
                                                           float img_w,
                                                           float input_size,
                                                           float threshold);

/**
 * Performs non-maximal suppression on detections to remove overlapping boxes
 * with lower confidence scores.
 *
 * @param detections    Detections to perform NMS on
 * @param iou_threshold Intersection threshold above which to discard boxes
 * @return              Vector of detections after NMS
 */
std::vector<VA::Detection> nms(const std::vector<VA::Detection> &detections,
                               float iou_threshold);

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
  std::vector<int64_t> anchors;
  std::vector<std::string> class_labels;
  int64_t input_size = 0;

 public:
  OnnxYoloDetector(std::string _onnx_model_path,
                   std::string _anchors_file_path,
                   std::string _class_labels_path) :
      onnx_model_path{std::move(_onnx_model_path)},
      anchors_file_path{std::move(_anchors_file_path)},
      class_labels_path{std::move(_class_labels_path)} {
  }

  ~OnnxYoloDetector() override {
    delete session;
  };

  OnnxYoloDetector(const OnnxYoloDetector &other) = delete;

  OnnxYoloDetector &operator=(const OnnxYoloDetector &other) {
    if (this != &other) {
      delete this->session;
      onnx_model_path = other.onnx_model_path;
      anchors_file_path = other.anchors_file_path;
      class_labels_path = other.class_labels_path;
    }
    return *this;
  }

  void initialize() override;

  std::vector<Detection> detect(const cv::Mat &img) override;

  std::string class_id_to_label(int class_id) const override;

  bool is_initialized() const override;
};

}

#endif //GRPC_VA_SERVER_SRC_VIDEOANALYZER_ONNXYOLODETECTOR_H_
