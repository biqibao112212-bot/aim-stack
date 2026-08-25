#include <exception>
#include <iostream>

#include <opencv2/core.hpp>

#include "autoaim_research/detector.hpp"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: autoaim_research_model_smoke ASSET [BACKEND]\n";
    return 2;
  }
  try {
    autoaim_research::DetectorConfig config;
    config.score_threshold = 0.5F;
    config.nms_threshold = 0.45F;
    config.inference_backend = argc == 3 ? argv[2] : "onnxruntime_cpu";
    if (config.inference_backend == "tensorrt_fp16") {
      config.engine_path = argv[1];
    } else {
      config.model_path = argv[1];
    }
    autoaim_research::TongjiYoloDetector detector(std::move(config));
    const cv::Mat black(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto detections = detector.detect(black);
    const cv::Size shape = detector.lastOutputShape();
    if (shape.width != 22 || shape.height != 25200) {
      std::cerr << "unexpected model output: " << shape.height << "x"
                << shape.width << "\n";
      return 3;
    }
    std::cout << "model_smoke_ok output=" << shape.height << "x" << shape.width
              << " detections_on_black=" << detections.size()
              << " backend=" << detector.backendName()
              << " inference_ms=" << detector.lastTiming().inference_ms
              << " total_ms=" << detector.lastTiming().total_ms << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
