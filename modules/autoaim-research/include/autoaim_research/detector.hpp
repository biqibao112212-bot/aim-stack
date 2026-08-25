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
  std::string engine_path;
  std::string inference_backend = "tensorrt_fp16";
};

struct DetectorTiming {
  double host_prepare_ms = 0.0;
  double gpu_preprocess_ms = 0.0;
  double inference_ms = 0.0;
  double output_copy_ms = 0.0;
  double postprocess_ms = 0.0;
  double total_ms = 0.0;
};

// Adapted from TongjiSuperPower/sp_vision_25 YOLOV5. The network output and
// four-corner parser are unchanged (1x25200x22). Direct TensorRT is the
// production research backend; ONNX Runtime CPU remains available for A/B.
class TongjiYoloDetector {
 public:
  explicit TongjiYoloDetector(DetectorConfig config);
  ~TongjiYoloDetector();

  TongjiYoloDetector(const TongjiYoloDetector&) = delete;
  TongjiYoloDetector& operator=(const TongjiYoloDetector&) = delete;

  std::list<auto_aim::Armor> detect(const cv::Mat& bgr_image);
  cv::Size lastOutputShape() const noexcept { return last_output_shape_; }
  const DetectorTiming& lastTiming() const noexcept { return last_timing_; }
  const std::string& backendName() const noexcept { return backend_name_; }

 private:
  struct TensorRtState;

  std::list<auto_aim::Armor> detectOnnxRuntime(const cv::Mat& bgr_image);
  std::list<auto_aim::Armor> detectTensorRt(const cv::Mat& bgr_image);
  std::list<auto_aim::Armor> parse(cv::Mat output, double scale) const;
  static double sigmoid(double value);

  DetectorConfig config_;
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<TensorRtState> tensorrt_;
  std::string input_name_;
  std::string output_name_;
  std::string backend_name_;
  cv::Size last_output_shape_{};
  DetectorTiming last_timing_{};
};

}  // namespace autoaim_research
