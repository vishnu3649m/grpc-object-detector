/**
 * A factory for instantiating object detection modules on request.
 */

#ifndef GRPC_OBJ_DET_DETECTORFACTORY_H_
#define GRPC_OBJ_DET_DETECTORFACTORY_H_

#include <string>

#include "FaceEyesDetector.h"
#include "OnnxYoloV4Detector.h"

namespace ObjDet {

class DetectorFactory {
 public:
  static std::unique_ptr<ObjDet::DetectorInterface> get_detector(const std::string& name) {
    if (name == "cascade_face_detector") {
      return std::unique_ptr<ObjDet::DetectorInterface>(
          new ObjDet::FaceEyesDetector(name,
                                       "config/cascade_face_detector/haarcascade_frontalface_alt.xml",
                                       "config/cascade_face_detector/haarcascade_eye_tree_eyeglasses.xml"));
    } else if (name == "onnx_yolov4_coco") {
      return std::unique_ptr<ObjDet::DetectorInterface>(
          new ObjDet::OnnxYoloV4Detector(name,
                                         "config/onnx_yolov4/yolov4.onnx",
                                         "config/onnx_yolov4/yolov4_anchors.txt",
                                         "config/onnx_yolov4/coco_labels.txt"));
    } else {
      return std::unique_ptr<ObjDet::DetectorInterface>(nullptr);
    }
  }
};

}

#endif //GRPC_OBJ_DET_DETECTORFACTORY_H_
