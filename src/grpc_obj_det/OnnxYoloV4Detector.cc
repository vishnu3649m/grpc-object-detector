#include <unordered_set>

#include <loguru.hpp>

#include "OnnxYoloV4Detector.h"


void ObjDet::OnnxYoloV4Detector::initialize() {
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

  if (input_shapes[0][1] != input_shapes[0][2] || input_shapes[0][1] % 32 != 0) {
    LOG_F(ERROR,
          "This detector requires the model input resolution to have the same "
          "width and height. The input resolution should be divisible by 32");
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

  this->input_size = input_shapes[0][1];
  int64_t n_anchors = output_shapes[0][3];
  int64_t n_classes = output_shapes[0][4] - 5;

  /**
   * Read in anchors for the model
   */
  if (read_anchors(anchors_file_path, n_anchors, this->anchors)) {
    std::stringstream ss;
    for (const auto &a : this->anchors)
      ss << a << ",";
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

std::vector<ObjDet::Detection> ObjDet::OnnxYoloV4Detector::detect(const cv::Mat &img) {
  std::vector<ObjDet::Detection> dets;
  int img_w = img.size().width;
  int img_h = img.size().height;

  LOG_F(INFO, "Image size: %d x %d x %d", img_w, img_h, img.channels());

  std::vector<float> input_tensor_raw = preprocess_image(img, this->input_size);

  LOG_F(INFO, "");

  std::vector<Ort::Value> input_tensors;
  std::vector<int64_t> input_tensor_shape = {1, input_size, input_size, 3};
  input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
      input_tensor_raw.data(),
      input_tensor_raw.size(),
      input_tensor_shape));
  auto input_tensor_info = input_tensors[0].GetTensorTypeAndShapeInfo();
  if (!input_tensors[0].IsTensor() ||
      !(input_tensor_info.GetShape() == input_tensor_shape)) {
    LOG_F(WARNING, "Image could not be transformed into model's expected input "
                   "tensor. Skipping this image...");
    return dets;
  }

  auto output_tensors = session->Run(session->GetInputNames(),
                                     input_tensors,
                                     session->GetOutputNames());

  auto parsed_preds = get_all_predictions(output_tensors,
                                          this->anchors,
                                          this->input_size);

  LOG_F(INFO, "");

  auto img_preds = filter_predictions(parsed_preds,
                                      float(img_h),
                                      float(img_w),
                                      float(this->input_size),
                                      0.25f);

  LOG_F(INFO, "");

  for (auto &preds : img_preds)
    dets = nms(preds, 0.213f);

  std::for_each(dets.begin(), dets.end(), [img_w, img_h](ObjDet::Detection &det){
    det.box.left /= float(img_w);
    det.box.top /= float(img_h);
    det.box.width /= float(img_w);
    det.box.height /= float(img_h);
  });

  return dets;
}

std::string ObjDet::OnnxYoloV4Detector::class_id_to_label(int class_id) const {
  if (class_id < 0 || class_id >= class_labels.size())
    return "";

  return class_labels[class_id];
}

bool ObjDet::OnnxYoloV4Detector::is_initialized() const {
  return init;
}

std::unordered_set<std::string> ObjDet::OnnxYoloV4Detector::available_objects_lookup() const {
  return std::unordered_set<std::string>(class_labels.begin(),
                                         class_labels.end());
}
