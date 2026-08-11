#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace aim_sim_bridge {

// Environment-gated evidence capture for simulation-only corner repair studies.
// It does not modify the image delivered to the production detector.
class Stage3ImageCapture {
 public:
  Stage3ImageCapture() {
    const char* root_value = std::getenv("AIM_SIM_STAGE3_IMAGE_DIR");
    if (root_value == nullptr || root_value[0] == '\0') return;

    root_ = std::filesystem::path(root_value);
    stride_ = parseBoundedInteger("AIM_SIM_STAGE3_IMAGE_STRIDE", 4, 1, 10000);
    jpeg_quality_ = parseBoundedInteger(
        "AIM_SIM_STAGE3_IMAGE_JPEG_QUALITY", 95, 1, 100);
    std::error_code error;
    std::filesystem::create_directories(root_ / "images", error);
    if (error) {
      error_ = "image directory creation failed: " + error.message();
      return;
    }
    index_.open(root_ / "image_index.jsonl", std::ios::out | std::ios::trunc);
    if (!index_) {
      error_ = "image index open failed";
      return;
    }
    enabled_ = true;
  }

  bool enabled() const { return enabled_; }
  const std::string& error() const { return error_; }

  bool submit(const cv::Mat& bgr_image, std::uint64_t producer_epoch,
              std::uint64_t image_seq, std::uint64_t timestamp_ns) {
    if (!enabled_) return error_.empty();
    const std::uint64_t ordinal = observed_++;
    if (ordinal % static_cast<std::uint64_t>(stride_) != 0) return true;
    if (bgr_image.empty()) {
      return fail("selected stage3 image is empty");
    }

    const std::string basename = std::to_string(producer_epoch) + "-" +
        std::to_string(image_seq) + "-" + std::to_string(timestamp_ns) + ".jpg";
    const std::filesystem::path relative = std::filesystem::path("images") / basename;
    const std::filesystem::path destination = root_ / relative;
    const std::vector<int> parameters{
        cv::IMWRITE_JPEG_QUALITY, jpeg_quality_,
    };
    try {
      if (!cv::imwrite(destination.string(), bgr_image, parameters)) {
        return fail("image write returned false: " + destination.string());
      }
    } catch (const cv::Exception& exception) {
      return fail("image write failed: " + std::string(exception.what()));
    }

    index_ << "{\"schema_version\":\"stage3-image-index-v1\""
           << ",\"producer_epoch\":" << producer_epoch
           << ",\"frame_seq\":" << image_seq
           << ",\"timestamp_ns\":" << timestamp_ns
           << ",\"relative_path\":\"images/" << basename << "\""
           << ",\"width\":" << bgr_image.cols
           << ",\"height\":" << bgr_image.rows
           << ",\"channels\":" << bgr_image.channels()
           << ",\"jpeg_quality\":" << jpeg_quality_
           << ",\"capture_stride\":" << stride_ << "}\n";
    index_.flush();
    if (!index_) return fail("image index write failed");
    return true;
  }

 private:
  static int parseBoundedInteger(const char* name, int fallback,
                                 int minimum, int maximum) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    try {
      const long parsed = std::stol(value);
      return static_cast<int>(std::clamp<long>(parsed, minimum, maximum));
    } catch (...) {
      return fallback;
    }
  }

  bool fail(std::string message) {
    error_ = std::move(message);
    enabled_ = false;
    return false;
  }

  bool enabled_ = false;
  std::filesystem::path root_;
  std::ofstream index_;
  std::uint64_t observed_ = 0;
  int stride_ = 4;
  int jpeg_quality_ = 95;
  std::string error_;
};

}  // namespace aim_sim_bridge
