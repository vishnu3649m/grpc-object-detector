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

/**
 * To represent a bounding box using the left-most x value, top-most y value and
 * its width and height. All 4 values should be normalised against the width/height
 * of the image (between 0 and 1).
 */
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

/**
 * To represent a single detection within an image.
 */
struct Detection {
  int class_id;
  RectTLWH box;
  float confidence;
};

/**
 * Interface for defining and implementing object detectors servable by the server.
 */
class DetectorInterface {
 public:
  /**
   * Initializes the detector and readies it for detections.
   *
   * Steps to take to initialize the model, such as parsing the provided config
   * files and allocating memory for the model, are specific to the model and the
   * runtime used to serve the model.
   *
   * This method should manage internal state such that the method `is_initialized()`
   * should return true if initialization is successful and false otherwise. If
   * initialization is not successful, the instance would not be used.
   */
  virtual void initialize() = 0;

  /**
   * Describes the detector and model.
   *
   * @return A pair containing name registered for this detector instance and
   * the model type/architecture.
   */
  virtual std::pair<std::string, std::string> describe() const = 0;

  /**
   * Runs object detector on an image and returns a vector of detections.
   *
   * @param img A valid image decoded by OpenCV.
   * @return Detections from the object detection model.
   */
  virtual std::vector<Detection> detect(const cv::Mat &img) = 0;

  /**
   * Returns a constant-time lookup of object class labels that this detector
   * can detect.
   *
   * This allows the server to lookup quickly if this detector can detect an
   * object class or not. This is necessary when 1 server has to serve multiple
   * object detectors and users request for a particular object to the detected.
   *
   * @return A set of object class labels.
   */
  virtual std::unordered_set<std::string> available_objects_lookup() const = 0;

  /**
   * Maps an object class ID to a human-readable object label string.
   *
   * This allows the server to map the id found within Detection to a string when
   * responding to a request.
   *
   * @param class_id The ID of the class to retrieve the label for.
   * @return A string containing the class label.
   */
  virtual std::string class_id_to_label(int class_id) const = 0;

  /**
   * Checks if the detector was successfully initialized.
   *
   * @return true if initialized; false otherwise.
   */
  virtual bool is_initialized() const = 0;

  virtual ~DetectorInterface() = default;
};

}

#endif //GRPC_OBJ_DET_DETECTORINTERFACE_H
