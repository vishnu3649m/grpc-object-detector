

#include <random>

#include <gtest/gtest.h>
#include <absl/strings/str_format.h>
#include <opencv2/opencv.hpp>

#include "VideoAnalyzer/DetectorInterface.h"

using namespace std;

class DummyConcreteDetector : public VA::DetectorInterface {
public:
    DummyConcreteDetector() = default;
    ~DummyConcreteDetector() override = default;

    void initialize() override {
        default_random_engine generator;
        uniform_int_distribution distribution(0, 9);

        class_id = distribution(generator);
    }

    vector<VA::Detection> detect(const cv::Mat & img) override {
        cv::Size size = img.size();
        cv::Mat grayscale_img;
        cv::Mat binary_img;
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        vector<VA::Detection> detections;

        cv::cvtColor(img, grayscale_img, cv::COLOR_BGR2GRAY);
        cv::threshold(grayscale_img, binary_img, 127, 255, cv::THRESH_BINARY);
        cv::findContours(binary_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        for (auto & contour_ : contours) {
            cv::Rect box = cv::boundingRect(contour_);
            detections.push_back(
                    VA::Detection{class_id,
                                  VA::RectTLWH(box, size.width, size.height),
                                  0.2});
        }

        return detections;
    }

    string class_id_to_label(int class_id_) override {
        return absl::StrFormat("class_%d", class_id_);
    }

private:
    int class_id = 0;
};


class DetectorInterfaceTest : public ::testing::Test {
protected:
    VA::DetectorInterface *detector;
    vector<VA::Detection> detections;

    DetectorInterfaceTest() {
        detector = new DummyConcreteDetector();
        detections = detector->detect(cv::imread("tests/data/1.jpg"));
    }

    ~DetectorInterfaceTest() override {
        delete detector;
    }
};


TEST_F(DetectorInterfaceTest, ProducesDetections) {
    ASSERT_GT(detections.size(), 0);
}

TEST_F(DetectorInterfaceTest, ProducesNormalizedBoundingBoxes) {
    ASSERT_GT(detections.size(), 0);

    for (auto & det : detections) {
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

TEST_F(DetectorInterfaceTest, ProducesConfidenceScoresForEachDetection) {
    ASSERT_GT(detections.size(), 0);

    for (auto & det : detections) {
        EXPECT_GE(det.confidence, 0.0f);
        EXPECT_LE(det.confidence, 1.0f);
    }
}

TEST_F(DetectorInterfaceTest, MapsAllDetectedObjectClassIdsToHumanReadableString) {
    ASSERT_GT(detections.size(), 0);

    for (auto & det : detections) {
        int cls_id = det.class_id;
        EXPECT_GE(cls_id, 0);
        EXPECT_LE(cls_id, 9);
        EXPECT_EQ(detector->class_id_to_label(cls_id),
                  absl::StrFormat("class_%d", cls_id));
    }
}
