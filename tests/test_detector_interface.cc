

#include <random>

#include <gtest/gtest.h>
#include <absl/strings/str_format.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

#include "VideoAnalyzer/DetectorInterface.h"
#include "VideoAnalyzer/FaceEyesDetector.h"
#include "VideoAnalyzer/OnnxYoloDetector.h"

using namespace std;

class DummyConcreteDetector : public VA::DetectorInterface {
 public:
  DummyConcreteDetector() = default;

  ~DummyConcreteDetector() override = default;

  void initialize() override {
    default_random_engine generator;
    uniform_int_distribution distribution(0, 9);

    class_id = distribution(generator);
    init = true;
  }

  bool is_initialized() const override {
    return init;
  }

  vector<VA::Detection> detect(const cv::Mat &img) override {
    cv::Size size = img.size();
    cv::Mat grayscale_img;
    cv::Mat binary_img;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    vector<VA::Detection> detections;

    cv::cvtColor(img, grayscale_img, cv::COLOR_BGR2GRAY);
    cv::threshold(grayscale_img, binary_img, 127, 255, cv::THRESH_BINARY);
    cv::findContours(binary_img,
                     contours,
                     hierarchy,
                     cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);

    for (auto &contour_ : contours) {
      cv::Rect box = cv::boundingRect(contour_);
      detections.push_back(
          VA::Detection{class_id,
              VA::RectTLWH(box, size.width, size.height),
              0.2});
    }

    return detections;
  }

  string class_id_to_label(int class_id_) const override {
    return absl::StrFormat("class_%d", class_id_);
  }

 private:
  int class_id = 0;
  bool init = false;
};

pair<VA::DetectorInterface *,
     string> create_detector_under_test(const string &type) {
  if (type == "face_eyes_detector") {
    return {new VA::FaceEyesDetector(
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml"),
            "tests/data/faces.jpg"};
  } else if (type == "yolov4") {
    return {new VA::OnnxYoloDetector("config/yolov4.onnx",
                                     "config/yolov4_anchors.txt",
                                     "config/coco_labels.txt"),
            "tests/data/faces.jpg"};
  } else {
    return {new DummyConcreteDetector(), "tests/data/1.jpg"};
  }
}

class DetectorInterfaceTest : public testing::TestWithParam<string> {
 public:
  DetectorInterfaceTest() {
    auto[det_, test_file] = create_detector_under_test(GetParam());
    detector = det_;
    detector->initialize();
    detections = detector->detect(cv::imread(test_file));
  }

  ~DetectorInterfaceTest() override {
    delete detector;
  }

 protected:
  VA::DetectorInterface *detector;
  vector<VA::Detection> detections;
};

TEST_P(DetectorInterfaceTest, CanInitializeSuccessfully) {
  ASSERT_TRUE(detector->is_initialized());
}

TEST_P(DetectorInterfaceTest, ProducesDetections) {
  ASSERT_TRUE(detector->is_initialized());
  ASSERT_GT(detections.size(), 0);
}

TEST_P(DetectorInterfaceTest, ProducesNormalizedBoundingBoxes) {
  ASSERT_TRUE(detector->is_initialized());
  ASSERT_GT(detections.size(), 0);

  for (auto &det : detections) {
    EXPECT_GE(det.box.left, 0.0f);
    EXPECT_LE(det.box.left, 1.0f);

    EXPECT_GE(det.box.top, 0.0f);
    EXPECT_LE(det.box.top, 1.0f);

    EXPECT_GE(det.box.width, 0.0f);
    EXPECT_LE(det.box.width, 1.0f);

    EXPECT_GE(det.box.height, 0.0f);
    EXPECT_LE(det.box.height, 1.0f);
  }
}

TEST_P(DetectorInterfaceTest, ProducesConfidenceScoresForEachDetection) {
  ASSERT_TRUE(detector->is_initialized());
  ASSERT_GT(detections.size(), 0);

  for (auto &det : detections) {
    EXPECT_GE(det.confidence, 0.0f);
    EXPECT_LE(det.confidence, 1.0f);
  }
}

TEST_P(DetectorInterfaceTest,
       MapsAllDetectedObjectClassIdsToValidHumanReadableStrings) {
  ASSERT_TRUE(detector->is_initialized());
  ASSERT_GT(detections.size(), 0);

  for (auto &det : detections) {
    string label = detector->class_id_to_label(det.class_id);

    EXPECT_NE(label, "");
  }
}

INSTANTIATE_TEST_SUITE_P(FactoryCreatedDetector,
                         DetectorInterfaceTest,
                         testing::Values("face_eyes_detector",
                                         "dummy",
                                         "yolov4"));
