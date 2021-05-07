#include <gtest/gtest.h>

#include <utility>

#include "grpc_obj_det/OnnxYoloV4Detector.h"

using namespace std;

#define TEST_ABS_ERROR 0.0001

TEST(OnnxYoloV4DetectorTest, CanInitWhenProvidedValidConfigFiles) {
  auto detector = ObjDet::OnnxYoloV4Detector("config/onnx_yolov4/yolov4.onnx",
                                             "config/onnx_yolov4/yolov4_anchors.txt",
                                             "config/onnx_yolov4/coco_labels.txt");
  detector.initialize();
  ASSERT_TRUE(detector.is_initialized());
}

TEST(OnnxYoloV4DetectorTest, DoesNotInitWhenProvidedInvalidOnnxFile) {
  auto detector = ObjDet::OnnxYoloV4Detector("tests/data/squeezenet1.0-7.onnx",
                                             "config/onnx_yolov4/yolov4_anchors.txt",
                                             "config/onnx_yolov4/coco_labels.txt");
  detector.initialize();
  ASSERT_FALSE(detector.is_initialized());
}

TEST(OnnxYoloV4DetectorTest, DoesNotInitWhenProvidedInvalidAnchorsFile) {
  auto detector = ObjDet::OnnxYoloV4Detector("config/onnx_yolov4/yolov4.onnx",
                                             "tests/data/invalid_yolov4_coco_anchors_1.txt",
                                             "config/onnx_yolov4/coco_labels.txt");
  detector.initialize();
  ASSERT_FALSE(detector.is_initialized());

  detector = ObjDet::OnnxYoloV4Detector("config/onnx_yolov4/yolov4.onnx",
                                        "tests/data/invalid_yolov4_coco_anchors_2.txt",
                                        "config/onnx_yolov4/coco_labels.txt");
  detector.initialize();
  ASSERT_FALSE(detector.is_initialized());
}

TEST(OnnxYoloV4DetectorTest, DoesNotInitWhenProvidedInvalidClassLabelsFile) {
  auto detector = ObjDet::OnnxYoloV4Detector("config/onnx_yolov4/yolov4.onnx",
                                             "config/onnx_yolov4/yolov4_anchors.txt",
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

struct ImageTestCase {
  string filename;
  unordered_map<string, int> expected_detections;

  ImageTestCase(string &&_filename,
                initializer_list<pair<string, int>> expected_list) {
    filename = move(_filename);
    for (auto &expected : expected_list)
      expected_detections.insert(expected);
  }
};

class OnnxYoloV4ObjectDetectionTest :
    public testing::TestWithParam<ImageTestCase> {

};

TEST_P(OnnxYoloV4ObjectDetectionTest, CanDetectObjectsInImage) {
  auto detector = ObjDet::OnnxYoloV4Detector(
      "config/onnx_yolov4/yolov4.onnx",
      "config/onnx_yolov4/yolov4_anchors.txt",
      "config/onnx_yolov4/coco_labels.txt"
  );
  detector.initialize();
  ASSERT_TRUE(detector.is_initialized());

  auto image_test_case = GetParam();

  auto detections = detector.detect(cv::imread(image_test_case.filename));

  // hash table keyed by class label to store detection counts of each class
  unordered_map<string, int> object_counts;
  for (const auto &det : detections) {
    string label = detector.class_id_to_label(det.class_id);
    if (object_counts.find(label) != object_counts.end())
      object_counts[label]++;
    else
      object_counts[label] = 1;
  }

  EXPECT_EQ(object_counts, image_test_case.expected_detections);
}

INSTANTIATE_TEST_SUITE_P(YoloV4ExpectedDetections,
                         OnnxYoloV4ObjectDetectionTest,
                         testing::Values(
                             ImageTestCase("tests/data/kite.jpg",
                                           {{"person", 7}, {"kite", 7}}),
                             ImageTestCase("tests/data/dog.jpg",
                                           {{"dog", 1}, {"bicycle", 1}, {"truck", 1}}),
                             ImageTestCase("tests/data/horses.jpg",
                                           {{"horse", 5}}),
                             ImageTestCase("tests/data/house.jpg",
                                           {{"person", 1}, {"refrigerator", 1}, {"vase", 1}, {"chair", 3}, {"potted plant", 1}, {"tvmonitor", 2}}),
                             ImageTestCase("tests/data/vegetables.jpg",
                                           {{"broccoli", 2}, {"carrot", 5}, {"dining table", 1}})
                         ));
