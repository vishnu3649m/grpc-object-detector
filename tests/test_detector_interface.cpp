

#include <gtest/gtest.h>
#include "VideoAnalyzer/DetectorInterface.h"

TEST(DetectionResults, ContainsNormalizedBoundingBox) {
    VA::Detection det = {0, {0.1, 0.2, 0.3, 0.4}, 0.8};

    EXPECT_FLOAT_EQ(det.box.top, 0.1);
    EXPECT_FLOAT_EQ(det.box.left, 0.2);
    EXPECT_FLOAT_EQ(det.box.width, 0.3);
    EXPECT_FLOAT_EQ(det.box.height, 0.4);
}
