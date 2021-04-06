#include <unordered_set>

#include <loguru.hpp>
#include <absl/strings/str_split.h>
#include <absl/strings/numbers.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xsort.hpp>

#include "OnnxYoloDetector.h"

static inline std::string print_shape(const std::vector<int64_t> &v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << " x ";
  ss << v[v.size() - 1];
  return ss.str();
}

template<class A>
static inline std::string print_shape(const A &t) {
  std::stringstream ss("");
  ss << xt::adapt(t.shape());
  return ss.str();
}

template<class A>
static inline std::string print_tensor(const A &t) {
  std::stringstream ss("");
  ss << t;
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
                         std::vector<int64_t> &anchors) {
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
      anchors.push_back(w);
      anchors.push_back(h);
    }

    if (anchors.size() == n_anchors * 3 * 2) {
      LOG_F(INFO, "Read in anchors successfully");
      return true;
    } else {
      LOG_F(INFO, "Supposed to read %zu anchors but only read %zu",
            n_anchors * 3 * 2, anchors.size());
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

static std::vector<float> preprocess_image(const cv::Mat &img,
                                           int64_t target_size) {
  auto img_h = img.size().height;
  auto img_w = img.size().width;

  float scale = std::min((float) target_size / (float) img_h,
                         (float) target_size / (float) img_w);
  int scaled_h = int(scale * img_h);
  int scaled_w = int(scale * img_w);

  cv::Mat padded_input(cv::Size(target_size, target_size),
                       CV_8UC3,
                       cv::Scalar(128, 128, 128));
  int height_offset = (target_size - scaled_h) / 2;
  int width_offset = (target_size - scaled_w) / 2;
  cv::Mat img_region(padded_input,
                     cv::Range(height_offset, height_offset + scaled_h),
                     cv::Range(width_offset, width_offset + scaled_w));
  cv::resize(img, img_region, cv::Size(scaled_w, scaled_h));

  std::vector<float> input_tensor_data;
  for (int64_t h = 0; h < target_size; ++h) {
    for (int64_t w = 0; w < target_size; ++w) {
      auto pixel = padded_input.at<cv::Vec3b>(cv::Point(w, h));
      input_tensor_data.push_back(float(pixel[2]) / 255.0f);
      input_tensor_data.push_back(float(pixel[1]) / 255.0f);
      input_tensor_data.push_back(float(pixel[0]) / 255.0f);
    }
  }

  LOG_F(INFO, "Elements in input tensor: %zu", input_tensor_data.size());
  int offset = (200 * 416 + 250) * 3;
  LOG_F(INFO,
        "Pixel value at [200][250]: %f, %f, %f",
        input_tensor_data[offset],
        input_tensor_data[offset + 1],
        input_tensor_data[offset + 2]);

  return input_tensor_data;
}

/**
 * Post-process prediction boxes of a YOLOv4 output layer.
 *
 * This function extracts the actual bounding box predictions (based on the
 * anchors) & scales them to the original input resolution. Then it combines
 * all predictions boxes from the 2-D grid into a single axis.
 *
 * Output tensor from the YOLO model is expected to be of shape:
 *      n_batch x grid_size x grid_size x n_anchors x (5 + n_classes)
 * After all predicted boxes are scaled and combined, the tensor shape will be:
 *      n_batch x (grid_size * grid_size * n_anchors) x (5 + n_classes)
 *
 * @note  The output tensor passed to this function will be modified in place.
 *
 * @param tensor    YOLO output layer
 * @param anchors   Anchors used when training this layer
 * @param stride    Input size / grid size
 * @param xyscale   xyscale specified during model training
 * @return
 */
static void postprocess_bboxes(xt::xarray<float> &tensor,
                               const std::vector<int64_t> &anchors,
                               int64_t stride,
                               float xyscale) {
  int64_t batch_size = tensor.shape()[0];
  int64_t grid_size = tensor.shape()[1];
  int64_t num_anchors = tensor.shape()[3];
  int64_t prediction_size = tensor.shape()[4];

  auto dxdy = xt::view(tensor,
                       xt::all(),
                       xt::all(),
                       xt::all(),
                       xt::all(),
                       xt::range(0, 2));
  auto dwdh = xt::view(tensor,
                       xt::all(),
                       xt::all(),
                       xt::all(),
                       xt::all(),
                       xt::range(2, 4));

  auto[xy_grid_1, xy_grid_0] = xt::meshgrid(xt::arange<float>(grid_size),
                                            xt::arange<float>(grid_size));
  auto xy_grid = xt::expand_dims(xt::expand_dims(xt::stack(std::make_tuple(xy_grid_0, xy_grid_1),
                                                           2),
                                                 2),
                                 0);

  auto expit_xy = 1 / (1 + xt::exp(-1 * dxdy));
  auto scaled_xy = ((expit_xy * xyscale) - 0.5 * (xyscale - 1) + xy_grid) * stride;

  auto anchors_tensor = xt::cast<float>(xt::adapt(anchors, {3, 2}));
  auto scaled_wh = xt::exp(dwdh) * anchors_tensor;

  auto scaled_bbox = xt::round(xt::concatenate(std::make_tuple(scaled_xy,
                                                               scaled_wh),
                                               4));
  auto bbox = xt::view(tensor,
                       xt::all(),
                       xt::all(),
                       xt::all(),
                       xt::all(),
                       xt::range(0, 4));
  bbox = xt::eval(scaled_bbox);

  tensor.reshape({batch_size,
                  grid_size * grid_size * num_anchors,
                  prediction_size});
}

/**
 * Gets all predictions (bounding boxes, objectness probabilities and class
 * probabilities) from all 3 YOLO output layers.
 *
 * @param outputs       Outputs from ONNX session inference
 * @param anchors       Anchors for this YOLO model
 * @param input_size    Input resolution of this YOLO model
 * @return              A tensor of shape n_batch x (total grid * anchors boxes) x (5 + n_classes)
 */
static auto get_all_predictions(const std::vector<Ort::Value> &outputs,
                                const std::vector<int64_t> &anchors,
                                int64_t input_size) {
  std::vector<xt::xarray<float>> pred_boxes;
  const auto anchors_iter = anchors.cbegin();
  int anchors_idx = 0;
  for (auto &output : outputs) {
    auto output_tensor_info = output.GetTensorTypeAndShapeInfo();
    int64_t grid_size = output_tensor_info.GetShape()[1];
    int64_t stride = input_size / grid_size;
    const auto *data = output.GetTensorData<float>();
    auto tensor = xt::xarray<float>(xt::adapt(data,
                                              output_tensor_info.GetElementCount(),
                                              xt::no_ownership(),
                                              output_tensor_info.GetShape()));

    postprocess_bboxes(tensor,
                       std::vector<int64_t>(anchors_iter + anchors_idx,
                                            anchors_iter + anchors_idx + 6),
                       stride,
                       1.2);
    anchors_idx += 6;

    pred_boxes.push_back(tensor);
  }

  auto predictions = xt::concatenate(std::make_tuple(pred_boxes[0],
                                                     pred_boxes[1],
                                                     pred_boxes[2]),
                                     1);

  LOG_F(INFO, "Predictions: %s", print_shape(predictions).c_str());
  std::vector<int> idxes{450, 2000, 5000, 7329, 9999};
  for (int idx : idxes)
    LOG_F(INFO,
          "%f, %f, %f",
          predictions.at(0, idx, 0),
          predictions.at(0, idx, 4),
          predictions.at(0, idx, 5));

  return predictions;
}

static inline VA::Detection pred_to_detection(float xmin, float ymin,
                                              float xmax, float ymax,
                                              float score, int cls_id) {
  return VA::Detection{cls_id, VA::RectTLWH(xmin, ymin, xmax, ymax), score};
}

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
static std::vector<std::vector<VA::Detection>> filter_predictions(xt::xarray<float> &preds,
                                                                  float img_h,
                                                                  float img_w,
                                                                  float input_size,
                                                                  float threshold) {
  size_t n_classes = preds.shape()[2] - 5;
  auto pred_xy = xt::view(preds, xt::all(), xt::all(), xt::range(0, 2));
  auto pred_wh = xt::view(preds, xt::all(), xt::all(), xt::range(2, 4));

  /* Convert box in (mid-x, mid-y, width, height) within model input to
   * (xmin, ymin, xmax, ymax) with respect to the original image dimensions. */
  float resize_ratio = std::min(input_size / img_w,
                                input_size / img_h);
  float dw = (input_size - resize_ratio * img_w) / 2;
  float dh = (input_size - resize_ratio * img_h) / 2;

  auto min_coord = xt::eval(pred_xy - pred_wh * 0.5);
  auto max_coord = xt::eval(pred_xy + pred_wh * 0.5);
  auto xmin = xt::fmax((xt::view(min_coord,
                                 xt::all(),
                                 xt::all(),
                                 xt::range(0, 1)) - dw) / resize_ratio,
                       0.0f);
  auto ymin = xt::fmax((xt::view(min_coord,
                                 xt::all(),
                                 xt::all(),
                                 xt::range(1, 2)) - dh) / resize_ratio,
                       0.0f);
  auto xmax = xt::fmin((xt::view(max_coord,
                                 xt::all(),
                                 xt::all(),
                                 xt::range(0, 1)) - dw) / resize_ratio,
                       img_w - 1);
  auto ymax = xt::fmin((xt::view(max_coord,
                                 xt::all(),
                                 xt::all(),
                                 xt::range(1, 2)) - dh) / resize_ratio,
                       img_h - 1);

  auto pred_box = xt::view(preds, xt::all(), xt::all(), xt::range(0, 4));
  pred_box = xt::eval(xt::round(xt::concatenate(std::make_tuple(xmin,
                                                                ymin,
                                                                xmax,
                                                                ymax),
                                                2)));

  /** Mask out boxes that have invalid dimensions. */
  auto invalid = xmin > xmax || ymin > ymax;
  auto invalid_mask = xt::masked_view(pred_box, invalid);
  invalid_mask = 0.0f;

  auto bbox_scale = xt::sqrt(
      xt::prod(xt::view(preds, xt::all(), xt::all(), xt::range(2, 4)) -
               xt::view(preds, xt::all(), xt::all(), xt::range(0,2)), 2));
  auto scale_mask = bbox_scale > 0.0f && xt::isfinite(bbox_scale);

  /** Mask out boxes that have low scores. */
  auto pred_conf = xt::view(preds, xt::all(), xt::all(), 4);
  auto pred_prob = xt::view(preds,
                            xt::all(),
                            xt::all(),
                            xt::range(5, 5 + n_classes));
  auto pred_class = xt::argmax(pred_prob, 2);
  auto pred_score = xt::eval(pred_conf * xt::amax(pred_prob, 2));
  auto score_mask = pred_score > threshold;

  /** Convert predictions in tensor to VA::Detection objects. */
  auto mask = scale_mask && score_mask;
  std::vector<std::vector<VA::Detection>> filtered_preds;
  int64_t n_batch = preds.shape()[0];
  for (int64_t i = 0; i < n_batch; ++i) {
    auto selected = xt::flatten_indices(xt::argwhere(xt::view(mask,
                                                              i,
                                                              xt::all())));
    std::vector<VA::Detection> dets;
    for (const auto &p : selected) {
      dets.push_back(pred_to_detection(preds.at(i, p, 0),
                                       preds.at(i, p, 1),
                                       preds.at(i, p, 2),
                                       preds.at(i, p, 3),
                                       pred_score.at(i, p),
                                       int(pred_class.at(i, p))));
    }
    filtered_preds.push_back(dets);
  }

  int _i = 0;
  for (const auto &img_preds : filtered_preds) {
    LOG_F(INFO, "Preds for img %d:", _i++);
    for (int _j = 0; _j < img_preds.size(); _j += 4) {
      const auto &pred = img_preds[_j];
      LOG_F(INFO,
            "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d",
            pred.box.left,
            pred.box.top,
            pred.box.width,
            pred.box.height,
            pred.confidence,
            pred.class_id);
    }
  }

  return filtered_preds;
}

static inline float calculate_iou(const VA::RectTLWH &box1,
                                  const VA::RectTLWH &box2) {
  float box1_area = box1.width * box1.height;
  float box2_area = box2.width * box2.height;

  float left = std::max(box1.left, box2.left);
  float top = std::max(box1.top, box2.top);
  float right = std::min(box1.left + box1.width,
                         box2.left + box2.width);
  float bottom = std::min(box1.top + box1.height,
                          box2.top + box2.height);
  float inter_area = std::max(right - left, 0.0f) * std::max(bottom - top,
                                                             0.0f);
  float union_area = box1_area + box2_area - inter_area;
  return inter_area / union_area;
}

static inline float iou_with_existing(const VA::Detection &candidate,
                                      const std::vector<VA::Detection> &existing) {
  float highest_iou = 0.0f;
  for (const auto &e : existing)
    if (float iou = calculate_iou(e.box, candidate.box); iou > highest_iou)
      highest_iou = iou;
  return highest_iou;
}

/**
 * Performs non-maximal suppression on detections to remove overlapping boxes
 * with lower confidence scores.
 *
 * @param detections    Detections to perform NMS on
 * @param iou_threshold Intersection threshold above which to discard boxes
 * @return              Vector of detections after NMS
 */
static std::vector<VA::Detection> nms(const std::vector<VA::Detection> &detections,
                                      float iou_threshold) {
  std::vector<VA::Detection> post_nms_detections;
  std::unordered_map<int, std::vector<VA::Detection>> classes_in_img;

  for (auto &det : detections)
    if (classes_in_img.find(det.class_id) != classes_in_img.end())
      classes_in_img[det.class_id].push_back(det);
    else
      classes_in_img[det.class_id] = std::vector<VA::Detection>{det};

  auto comparator = [](const VA::Detection &a, const VA::Detection &b) {
    return a.confidence < b.confidence;
  };

  for (auto &c : classes_in_img) {
    auto &dets = c.second;
    std::make_heap(dets.begin(), dets.end(), comparator);
    std::vector<VA::Detection> best_dets;

    while (!dets.empty()) {
      VA::Detection candidate = dets[0];
      std::pop_heap(dets.begin(), dets.end(), comparator);
      dets.pop_back();
      if (iou_with_existing(candidate, best_dets) < iou_threshold)
        best_dets.push_back(candidate);
    }
    post_nms_detections.insert(post_nms_detections.end(),
                               best_dets.begin(),
                               best_dets.end());
  }

  return post_nms_detections;
}

std::vector<VA::Detection> VA::OnnxYoloDetector::detect(const cv::Mat &img) {
  std::vector<VA::Detection> dets;
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

  auto parsed_preds = xt::eval(get_all_predictions(output_tensors,
                                                   this->anchors,
                                                   this->input_size));

  LOG_F(INFO, "");

  auto img_preds = filter_predictions(parsed_preds,
                                      float(img_h),
                                      float(img_w),
                                      float(this->input_size),
                                      0.25f);

  LOG_F(INFO, "");

  for (auto &preds : img_preds)
    dets = nms(preds, 0.213f);

  std::for_each(dets.begin(), dets.end(), [img_w, img_h](VA::Detection &det){
    det.box.left /= float(img_w);
    det.box.top /= float(img_h);
    det.box.width /= float(img_w);
    det.box.height /= float(img_h);
  });

  return dets;
}

std::string VA::OnnxYoloDetector::class_id_to_label(int class_id) const {
  return std::string("");
}

bool VA::OnnxYoloDetector::is_initialized() const {
  return init;
}
