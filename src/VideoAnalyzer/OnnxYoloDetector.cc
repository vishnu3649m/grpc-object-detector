#include <loguru.hpp>
#include <absl/strings/str_split.h>
#include <absl/strings/numbers.h>

#include "OnnxYoloDetector.h"

static inline std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << " x ";
  ss << v[v.size() - 1];
  return ss.str();
}

static bool read_anchors(const std::string &filepath, int64_t n_anchors, std::vector<std::vector<int64_t>> &anchors) {
  int64_t expected_tokens = n_anchors * 3 * 2;
  std::ifstream file(filepath);

  if (!file.is_open()) {
    LOG_F(ERROR, "Could not read file containing anchors: %s", filepath.c_str());
    return false;
  }

  std::string line;
  std::getline(file, line);
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

void VA::OnnxYoloDetector::initialize() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov4");
  Ort::SessionOptions options;
  session = new Ort::Experimental::Session(env, onnx_model_path, options);

  std::vector<std::string> input_names = session->GetInputNames();
  std::vector<std::vector<int64_t> > input_shapes = session->GetInputShapes();

  LOG_F(INFO, "ONNX Yolo Model Input Node Name/Shape (%zu):", input_names.size());
  for (size_t i = 0; i < input_names.size(); i++)
    LOG_F(INFO, "\t%s : %s", input_names[i].data(), print_shape(input_shapes[i]).data());

  if (input_shapes.size() != 1) {
    LOG_F(ERROR,
          "This detector requires the YOLO model to have only 1 input layer. "
          "But the ONNX model supplied contains %zu input layers.",
          input_shapes.size());
    return;
  }

  input_w = input_shapes[0][1];
  input_h = input_shapes[0][2];

  std::vector<std::string> output_names = session->GetOutputNames();
  std::vector<std::vector<int64_t> > output_shapes = session->GetOutputShapes();
  LOG_F(INFO, "ONNX Yolo Model Output Node Name/Shape (%zu):", output_names.size());
  for (size_t i = 0; i < output_names.size(); i++)
    LOG_F(INFO, "\t%s : %s", output_names[i].data(), print_shape(output_shapes[i]).data());

  if (output_shapes.empty() || output_shapes[0].size() != 5) {
    LOG_F(ERROR,
          "This detector requires the YOLO model to have at least 1 output layer of shape: "
          "-1 x -1 x -1 x n_anchors x detections.");
    return;
  }

  int64_t n_anchors = output_shapes[0][3];
  std::vector<std::vector<int64_t>> anchors;

  if (read_anchors(anchors_file_path, n_anchors, anchors)) {
    std::stringstream ss;
    for (const auto &a : anchors)
      ss << a[0] << "," << a[1] << "  ";
    LOG_F(INFO, "Anchors read in: %s", ss.str().c_str());
    init = true;
  } else {
    LOG_F(WARNING, "Not initializing detector due to errors in reading anchors");
  }
}

std::vector<VA::Detection> VA::OnnxYoloDetector::detect(const cv::Mat &img) {
  return std::vector<VA::Detection>();
}

std::string VA::OnnxYoloDetector::class_id_to_label(int class_id) const {
  return std::string("");
}

bool VA::OnnxYoloDetector::is_initialized() const {
  return true;
}
