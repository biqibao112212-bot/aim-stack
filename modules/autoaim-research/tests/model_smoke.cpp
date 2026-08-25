#include <exception>
#include <iostream>

#include <opencv2/core.hpp>

#include "autoaim_research/detector.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: autoaim_research_model_smoke MODEL.onnx\n";
    return 2;
  }
  try {
    autoaim_research::TongjiYoloDetector detector({argv[1], 0.5F, 0.45F});
    const cv::Mat black(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto detections = detector.detect(black);
    const cv::Size shape = detector.lastOutputShape();
    if (shape.width != 22 || shape.height != 25200) {
      std::cerr << "unexpected model output: " << shape.height << "x"
                << shape.width << "\n";
      return 3;
    }
    std::cout << "model_smoke_ok output=" << shape.height << "x" << shape.width
              << " detections_on_black=" << detections.size() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
