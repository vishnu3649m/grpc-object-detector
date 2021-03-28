#include <loguru.hpp>
#include <absl/strings/str_split.h>
#include <absl/strings/numbers.h>

#include "OnnxYoloDetector.h"

static inline std::string print_shape(const std::vector<int64_t> &v) {
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
static bool read_anchors(const std::string &filepath,
                         int64_t n_anchors,
                         std::vector<std::vector<int64_t>> &anchors) {
  int64_t expected_tokens = n_anchors * 3 * 2;
  std::ifstream file(filepath);

  if (!file.is_open()) {
    LOG_F(ERROR,
          "Could not read file containing anchors: %s",
          filepath.c_str());
    return false;
  }

  std::string line;
  std::getline(file, line);
  file.close();

  std::vector<std::string> tokens = absl::StrSplit(line, ',');
  if (tokens.size() == expected_tokens) {
    for (int i = 0; i < expected_tokens; i += 2) {
      int64_t w, h;
      if (!absl::SimpleAtoi(tokens[i], &w)) {
        LOG_F(ERROR, "Error parsing token %s into an integer",
              tokens[0].c_str());
        return false;
      }
      if (!absl::SimpleAtoi(tokens[i + 1], &h)) {
        LOG_F(ERROR, "Error parsing token %s into an integer",
              tokens[1].c_str());
        return false;
      }
      anchors.push_back({w, h});
    }

    if (anchors.size() == n_anchors * 3) {
      LOG_F(INFO, "Read in anchors successfully");
      return true;
    } else {
      LOG_F(INFO, "Supposed to read %zu anchors but only read %zu",
            n_anchors * 3, anchors.size());
      return false;
    }
  } else {
    LOG_F(ERROR, "Anchors file does not contain the proper dimensions of "
                 "anchors. There should be `n_anchors * 3` anchors provided"
                 " with each anchor represented as `W,H`.");
    return false;
  }
}

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
static bool read_labels(const std::string &filepath,
                        int64_t n_classes,
                        std::vector<std::string> &cls_labels) {
  std::ifstream file(filepath);

  if (!file.is_open()) {
    LOG_F(ERROR, "Could not read class labels file: %s", filepath.c_str());
    return false;
  }

  // ensure that the destination vector is empty because the index of the vector
  // would serve as the key (class ID).
  cls_labels.clear();

  std::string line;
  while (std::getline(file, line)) {
    cls_labels.push_back(line);
    if (cls_labels.size() == n_classes) break;
  }
  file.close();

  if (cls_labels.size() == n_classes) {
    LOG_F(INFO, "Read in labels for %ld classes successfully", n_classes);
    return true;
  } else {
    LOG_F(ERROR, "Expected %ld labels but only read in %zu labels!",
          n_classes, cls_labels.size());
    return false;
  }
}

void VA::OnnxYoloDetector::initialize() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov4");
  Ort::SessionOptions options;
  session = new Ort::Experimental::Session(env, onnx_model_path, options);

  /**
   * Check that the provided model meets the required conditions. Then extract
   * relevant model properties such as input size, num classes, etc.
   */
  std::vector<std::string> input_names = session->GetInputNames();
  std::vector<std::vector<int64_t> > input_shapes = session->GetInputShapes();

  LOG_F(INFO,
        "ONNX Yolo Model Input Node Name/Shape (%zu):",
        input_names.size());
  for (size_t i = 0; i < input_names.size(); i++)
    LOG_F(INFO,
          "\t%s : %s",
          input_names[i].data(),
          print_shape(input_shapes[i]).data());

  if (input_shapes.size() != 1) {
    LOG_F(ERROR,
          "This detector requires the YOLO model to have only 1 input layer. "
          "But the ONNX model supplied contains %zu input layers.",
          input_shapes.size());
    return;
  }

  std::vector<std::string> output_names = session->GetOutputNames();
  std::vector<std::vector<int64_t> > output_shapes = session->GetOutputShapes();
  LOG_F(INFO,
        "ONNX Yolo Model Output Node Name/Shape (%zu):",
        output_names.size());
  for (size_t i = 0; i < output_names.size(); i++)
    LOG_F(INFO,
          "\t%s : %s",
          output_names[i].data(),
          print_shape(output_shapes[i]).data());

  if (output_shapes.empty() || output_shapes[0].size() != 5) {
    LOG_F(ERROR,
          "This detector requires the YOLO model to have at least 1 output "
          "layer of shape: -1 x -1 x -1 x n_anchors x detections.");
    return;
  }

  this->input_w = input_shapes[0][1];
  this->input_h = input_shapes[0][2];
  int64_t n_anchors = output_shapes[0][3];
  int64_t n_classes = output_shapes[0][4] - 5;

  /**
   * Read in anchors for the model
   */
  if (read_anchors(anchors_file_path, n_anchors, this->anchors)) {
    std::stringstream ss;
    for (const auto &a : this->anchors)
      ss << a[0] << "," << a[1] << "  ";
    LOG_F(INFO, "Anchors read in: %s", ss.str().c_str());
  } else {
    LOG_F(WARNING,
          "Not initializing detector due to errors in reading anchors");
    return;
  }

  /**
   * Read in class labels
   */
   if (!read_labels(class_labels_path, n_classes, this->class_labels)) {
     LOG_F(WARNING,
           "Not initializing detector due to errors in reading class labels");
     return;
   }

   init = true;
}

std::vector<VA::Detection> VA::OnnxYoloDetector::detect(const cv::Mat &img) {
  return std::vector<VA::Detection>();
}

std::string VA::OnnxYoloDetector::class_id_to_label(int class_id) const {
  return std::string("");
}

bool VA::OnnxYoloDetector::is_initialized() const {
  return init;
}
