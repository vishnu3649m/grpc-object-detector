#include <gtest/gtest.h>

#include "VideoAnalyzer/OnnxYoloDetector.h"

using namespace std;

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
