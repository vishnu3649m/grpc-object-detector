
#include "FaceEyesDetector.h"

void ObjDet::FaceEyesDetector::initialize() {
  class_label_map = {"face", "eye"};
  face_cascade.load(cv::String(face_config_file));
  eyes_cascade.load(cv::String(eyes_config_file));
  init = true;
}

std::vector<ObjDet::Detection> ObjDet::FaceEyesDetector::detect(const cv::Mat &img) {
  cv::Size size = img.size();
  std::vector<cv::Rect> face_boxes;
  cv::Mat greyscale_img;
  std::vector<ObjDet::Detection> dets;

  cv::cvtColor(img, greyscale_img, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(greyscale_img, greyscale_img);
  face_cascade.detectMultiScale(greyscale_img,
                                face_boxes,
                                1.1,
                                2,
                                0u | cv::CASCADE_SCALE_IMAGE,
                                cv::Size(30, 30));

  for (const cv::Rect &box : face_boxes) {
    dets.push_back({0,
                       RectTLWH(box, size.width, size.height),
                       0.0f});

    cv::Mat face_roi = greyscale_img(box);
    std::vector<cv::Rect> eyes;
    eyes_cascade.detectMultiScale(face_roi,
                                  eyes,
                                  1.1,
                                  2,
                                  0u | cv::CASCADE_SCALE_IMAGE,
                                  cv::Size(30, 30));

    for (const cv::Rect &eyes_box : eyes)
      dets.push_back({1,
                         RectTLWH(eyes_box, box.width, box.height),
                         0.0f});
  }

  return dets;
}

std::string ObjDet::FaceEyesDetector::class_id_to_label(int class_id) const {
  if (class_id < 0 || class_id > 1)
    return "";
  return class_label_map[class_id];
}

bool ObjDet::FaceEyesDetector::is_initialized() const {
  return init;
}

std::unordered_set<std::string> ObjDet::FaceEyesDetector::available_objects_lookup() const {
  return std::unordered_set<std::string>(class_label_map.begin(),
                                         class_label_map.end());
}

std::pair<std::string, std::string> ObjDet::FaceEyesDetector::describe() const {
  return {this->name, this->model};
}
