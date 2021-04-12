#include <gtest/gtest.h>

#include "VideoAnalyzer/OnnxYoloDetector.h"

using namespace std;

#define TEST_ABS_ERROR 0.0001

TEST(OnnxYoloV4DetectorTest, CanInitWhenProvidedValidConfigFiles) {
  auto detector = VA::OnnxYoloDetector("config/yolov4.onnx",
                                       "config/yolov4_anchors.txt",
                                       "config/coco_labels.txt");
  detector.initialize();
  ASSERT_TRUE(detector.is_initialized());
}

TEST(OnnxYoloV4DetectorTest, DoesNotInitWhenProvidedInvalidOnnxFile) {
  auto detector = VA::OnnxYoloDetector("tests/data/squeezenet1.0-7.onnx",
                                       "config/yolov4_anchors.txt",
                                       "config/coco_labels.txt");
  detector.initialize();
  ASSERT_FALSE(detector.is_initialized());
}

TEST(OnnxYoloV4DetectorTest, DoesNotInitWhenProvidedInvalidAnchorsFile) {
  auto detector = VA::OnnxYoloDetector("config/yolov4.onnx",
                                       "tests/data/invalid_yolov4_coco_anchors_1.txt",
                                       "config/coco_labels.txt");
  detector.initialize();
  ASSERT_FALSE(detector.is_initialized());

  detector = VA::OnnxYoloDetector("config/yolov4.onnx",
                                  "tests/data/invalid_yolov4_coco_anchors_2.txt",
                                  "config/coco_labels.txt");
  detector.initialize();
  ASSERT_FALSE(detector.is_initialized());
}

TEST(OnnxYoloV4DetectorTest, DoesNotInitWhenProvidedInvalidClassLabelsFile) {
  auto detector = VA::OnnxYoloDetector("config/yolov4.onnx",
                                       "config/yolov4_anchors.txt",
                                       "tests/data/invalid_coco_labels.txt");
  detector.initialize();
  ASSERT_FALSE(detector.is_initialized());
}

TEST(OnnxYoloV4DetectorTest, PreprocessesImageCorrectly) {
  cv::Mat img = cv::imread("tests/data/kite.jpg");
  int64_t input_size = 416;
  int64_t expected_tensor_size = 1 * input_size * input_size * 3;

  ASSERT_FALSE(img.empty());

  vector<float> input_tensor_data = preprocess_image(img, input_size);
  ASSERT_EQ(input_tensor_data.size(), expected_tensor_size);

  /* Pixels to test (represented as x, y pairs) as well as the expected
   * RGB values. This is for the kite.jpg image file. */
  vector<pair<pair<int, int>, vector<float>>> pixels_under_test = {
      {{10, 50}, {0.50196, 0.50196, 0.50196}},
      {{250, 200}, {0.18039, 0.40392, 0.37647}},
      {{200, 250}, {0.75686, 0.72549, 0.65882}},
      {{410, 350}, {0.50196, 0.50196, 0.50196}}
  };

  for (auto &test : pixels_under_test) {
    auto &pixel = test.first;
    auto &expected = test.second;
    int64_t offset = (pixel.second * input_size + pixel.first) * 3;
    EXPECT_NEAR(input_tensor_data[offset], expected[0], TEST_ABS_ERROR);
    EXPECT_NEAR(input_tensor_data[offset + 1], expected[1], TEST_ABS_ERROR);
    EXPECT_NEAR(input_tensor_data[offset + 2], expected[2], TEST_ABS_ERROR);
  }
}

TEST(OnnxYoloV4DetectorTest, CanDetectObjectsInImage) {
  auto detector = VA::OnnxYoloDetector("config/yolov4.onnx",
                                       "config/yolov4_anchors.txt",
                                       "config/coco_labels.txt");
  detector.initialize();
  ASSERT_TRUE(detector.is_initialized());

  auto detections = detector.detect(cv::imread("tests/data/kite.jpg"));
  // hash table keyed by class_id to store counts of each object class
  unordered_map<int, int> object_counts;
  for (const auto &det : detections)
    if (object_counts.find(det.class_id) != object_counts.end())
      object_counts[det.class_id]++;
    else
      object_counts[det.class_id] = 1;

  ASSERT_EQ(detections.size(), 14);
  ASSERT_NE(object_counts.find(0), object_counts.end());
  EXPECT_EQ(object_counts[0], 7);
  ASSERT_NE(object_counts.find(33), object_counts.end());
  EXPECT_EQ(object_counts[33], 7);
}
