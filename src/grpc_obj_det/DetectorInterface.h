/**
 * Core interfaces for developers to implement object detectors compatible with
 * the gRPC server.
 */

#ifndef GRPC_OBJ_DET_DETECTORINTERFACE_H
#define GRPC_OBJ_DET_DETECTORINTERFACE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_set>

namespace ObjDet {

struct RectTLWH {
  RectTLWH(const cv::Rect &box, int w, int h) :
      top{(float) box.y / (float) h},
      left{(float) box.x / (float) w},
      width{(float) box.width / (float) w},
      height{(float) box.height / (float) h} {
  }

  RectTLWH(float xmin, float ymin, float xmax, float ymax) :
      top{ymin},
      left{xmin},
      width{xmax - xmin},
      height{ymax - ymin} {
  }

  float top;
  float left;
  float width;
  float height;
};

struct Detection {
  int class_id;
  RectTLWH box;
  float confidence;
};

class DetectorInterface {
 public:
  virtual void initialize() = 0;

  virtual std::vector<Detection> detect(const cv::Mat &img) = 0;

  virtual std::unordered_set<std::string> available_objects_lookup() const = 0;

  virtual std::string class_id_to_label(int class_id) const = 0;

  virtual bool is_initialized() const = 0;

  virtual ~DetectorInterface() = default;
};

}

#endif //GRPC_OBJ_DET_DETECTORINTERFACE_H
