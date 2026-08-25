#include "autoaim_research/detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace autoaim_research {

TongjiYoloDetector::TongjiYoloDetector(DetectorConfig config)
    : config_(std::move(config)),
      environment_(ORT_LOGGING_LEVEL_WARNING, "autoaim_research") {
  if (config_.model_path.empty() ||
      !std::filesystem::is_regular_file(config_.model_path)) {
    throw std::runtime_error("YOLO model does not exist: " + config_.model_path);
  }
  if (!(config_.score_threshold > 0.0F && config_.score_threshold < 1.0F)) {
    throw std::runtime_error("score_threshold must be in (0, 1)");
  }
  if (!(config_.nms_threshold > 0.0F && config_.nms_threshold < 1.0F)) {
    throw std::runtime_error("nms_threshold must be in (0, 1)");
  }
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = std::make_unique<Ort::Session>(
      environment_, config_.model_path.c_str(), session_options_);
  if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
    throw std::runtime_error("YOLO model must expose exactly one input and output");
  }
  Ort::AllocatorWithDefaultOptions allocator;
  const auto input_name = session_->GetInputNameAllocated(0, allocator);
  const auto output_name = session_->GetOutputNameAllocated(0, allocator);
  input_name_ = input_name.get();
  output_name_ = output_name.get();

  const auto input_shape =
      session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape != std::vector<std::int64_t>({1, 3, 640, 640})) {
    throw std::runtime_error("YOLO input must be [1,3,640,640]");
  }
}

std::list<auto_aim::Armor> TongjiYoloDetector::detect(const cv::Mat& bgr_image) {
  if (bgr_image.empty() || bgr_image.type() != CV_8UC3) {
    throw std::runtime_error("detector requires a non-empty CV_8UC3 image");
  }

  const double scale = std::min(640.0 / static_cast<double>(bgr_image.rows),
                                640.0 / static_cast<double>(bgr_image.cols));
  const int resized_height = static_cast<int>(bgr_image.rows * scale);
  const int resized_width = static_cast<int>(bgr_image.cols * scale);
  cv::Mat letterbox(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::resize(bgr_image, letterbox(cv::Rect(0, 0, resized_width, resized_height)),
             {resized_width, resized_height});

  cv::Mat blob = cv::dnn::blobFromImage(letterbox, 1.0 / 255.0, {640, 640},
                                        cv::Scalar(), true, false, CV_32F);
  if (!blob.isContinuous()) {
    throw std::runtime_error("YOLO input tensor is not contiguous");
  }

  constexpr std::array<std::int64_t, 4> input_shape{1, 3, 640, 640};
  const Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(
      memory, blob.ptr<float>(), blob.total(), input_shape.data(),
      input_shape.size());
  const char* input_names[]{input_name_.c_str()};
  const char* output_names[]{output_name_.c_str()};
  Ort::RunOptions run_options;
  auto tensors = session_->Run(run_options, input_names, &input, 1,
                               output_names, 1);
  if (tensors.size() != 1 || !tensors.front().IsTensor()) {
    throw std::runtime_error("YOLO did not return one tensor");
  }

  const auto output_shape =
      tensors.front().GetTensorTypeAndShapeInfo().GetShape();
  if (output_shape.size() != 3 || output_shape[0] != 1 ||
      !((output_shape[1] == 25200 && output_shape[2] == 22) ||
        (output_shape[1] == 22 && output_shape[2] == 25200))) {
    throw std::runtime_error("YOLO output must be [1,25200,22] or [1,22,25200]");
  }
  float* output_data = tensors.front().GetTensorMutableData<float>();

  cv::Mat output;
  if (output_shape[2] == 22) {
    output = cv::Mat(static_cast<int>(output_shape[1]), 22, CV_32F,
                     output_data);
  } else {
    const cv::Mat transposed(22, static_cast<int>(output_shape[2]), CV_32F,
                             output_data);
    output = transposed.t();
  }
  last_output_shape_ = {output.cols, output.rows};
  return parse(output, scale);
}

std::list<auto_aim::Armor> TongjiYoloDetector::parse(cv::Mat output,
                                                     double scale) const {
  std::vector<int> color_ids;
  std::vector<int> number_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> corner_sets;
  const float logit_threshold = static_cast<float>(
      std::log(config_.score_threshold / (1.0F - config_.score_threshold)));

  for (int row_index = 0; row_index < output.rows; ++row_index) {
    const float* row = output.ptr<float>(row_index);
    if (!std::isfinite(row[8]) || row[8] < logit_threshold) continue;
    const float score = static_cast<float>(sigmoid(row[8]));
    if (score < config_.score_threshold) continue;

    int color_id = 0;
    for (int index = 1; index < 4; ++index) {
      if (row[9 + index] > row[9 + color_id]) color_id = index;
    }
    int number_id = 0;
    for (int index = 1; index < 9; ++index) {
      if (row[13 + index] > row[13 + number_id]) number_id = index;
    }

    // Tongji YOLOV5 public output order -> TL, TR, BR, BL.
    std::vector<cv::Point2f> corners{
        {row[0] / static_cast<float>(scale), row[1] / static_cast<float>(scale)},
        {row[6] / static_cast<float>(scale), row[7] / static_cast<float>(scale)},
        {row[4] / static_cast<float>(scale), row[5] / static_cast<float>(scale)},
        {row[2] / static_cast<float>(scale), row[3] / static_cast<float>(scale)}};
    const auto finite = [](const cv::Point2f& point) {
      return std::isfinite(point.x) && std::isfinite(point.y);
    };
    if (!std::all_of(corners.begin(), corners.end(), finite)) continue;

    float min_x = corners.front().x;
    float max_x = corners.front().x;
    float min_y = corners.front().y;
    float max_y = corners.front().y;
    for (const auto& point : corners) {
      min_x = std::min(min_x, point.x);
      max_x = std::max(max_x, point.x);
      min_y = std::min(min_y, point.y);
      max_y = std::max(max_y, point.y);
    }
    if (max_x - min_x < 1.0F || max_y - min_y < 1.0F) continue;

    color_ids.push_back(color_id);
    number_ids.push_back(number_id);
    confidences.push_back(score);
    boxes.emplace_back(static_cast<int>(std::floor(min_x)),
                       static_cast<int>(std::floor(min_y)),
                       static_cast<int>(std::ceil(max_x - min_x)),
                       static_cast<int>(std::ceil(max_y - min_y)));
    corner_sets.push_back(std::move(corners));
  }

  std::vector<int> kept;
  cv::dnn::NMSBoxes(boxes, confidences, config_.score_threshold,
                    config_.nms_threshold, kept);
  std::list<auto_aim::Armor> armors;
  for (const int index : kept) {
    auto_aim::Armor armor(color_ids[index], number_ids[index], confidences[index],
                          boxes[index], corner_sets[index]);
    armor.priority = auto_aim::ArmorPriority::third;
    armor.duplicated = false;
    if (armor.name != auto_aim::ArmorName::not_armor) {
      armors.push_back(std::move(armor));
    }
  }
  return armors;
}

double TongjiYoloDetector::sigmoid(double value) {
  if (value >= 0.0) return 1.0 / (1.0 + std::exp(-value));
  const double exponential = std::exp(value);
  return exponential / (1.0 + exponential);
}

}  // namespace autoaim_research
