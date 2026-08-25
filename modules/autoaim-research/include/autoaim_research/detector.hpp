#pragma once

#include <list>
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>

#include "tasks/auto_aim/armor.hpp"

namespace autoaim_research {

struct DetectorConfig {
  std::string model_path;
  float score_threshold = 0.5F;
  float nms_threshold = 0.45F;
};

// Adapted from TongjiSuperPower/sp_vision_25 YOLOV5. The network output and
// four-corner parser are unchanged (1x25200x22); ONNX Runtime replaces the
// upstream OpenVINO loader so the protected ONNX can be selected by path.
class TongjiYoloDetector {
 public:
  explicit TongjiYoloDetector(DetectorConfig config);

  std::list<auto_aim::Armor> detect(const cv::Mat& bgr_image);
  cv::Size lastOutputShape() const noexcept { return last_output_shape_; }

 private:
  std::list<auto_aim::Armor> parse(cv::Mat output, double scale) const;
  static double sigmoid(double value);

  DetectorConfig config_;
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  cv::Size last_output_shape_{};
};

}  // namespace autoaim_research
