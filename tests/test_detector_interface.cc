

#include <random>

#include <gtest/gtest.h>
#include <absl/strings/str_format.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

#include "grpc_obj_det/DetectorFactory.h"

using namespace std;

TEST(DetectorFactoryTest, ReturnsNullPtrWhenUnknownDetectorRequested) {
  ASSERT_EQ(ObjDet::DetectorFactory::get_detector("some_unknown_detector"),
            nullptr);
}

class DetectorInterfaceTest : public testing::TestWithParam<string> {
 public:
  DetectorInterfaceTest() {
    string test_file = "tests/data/faces.jpg";
    detector = ObjDet::DetectorFactory::get_detector(GetParam());
    detector->initialize();
    detections = detector->detect(cv::imread(test_file));
  }

  ~DetectorInterfaceTest() override {

  }

 protected:
  unique_ptr<ObjDet::DetectorInterface> detector;
  vector<ObjDet::Detection> detections;
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

TEST_P(DetectorInterfaceTest, ProvidesValidClassLabelLookup) {
  ASSERT_TRUE(detector->is_initialized());
  ASSERT_GT(detections.size(), 0);
  auto lookup = detector->available_objects_lookup();

  for (auto &det : detections)
    ASSERT_NE(lookup.find(detector->class_id_to_label(det.class_id)),
              lookup.end());
}

INSTANTIATE_TEST_SUITE_P(FactoryCreatedDetector,
                         DetectorInterfaceTest,
                         testing::Values("cascade_face_detector",
                                         "onnx_yolov4_coco"));
