#include "autoaim_research/detector.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "autoaim_research/cuda_preprocess.hpp"

namespace autoaim_research {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMilliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void requireCuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(status));
}

std::size_t tensorElements(const nvinfer1::Dims& dimensions) {
  std::size_t elements = 1;
  for (int index = 0; index < dimensions.nbDims; ++index) {
    if (dimensions.d[index] <= 0) {
      throw std::runtime_error("TensorRT engine has a dynamic or invalid shape");
    }
    elements *= static_cast<std::size_t>(dimensions.d[index]);
  }
  return elements;
}

std::size_t elementBytes(nvinfer1::DataType type) {
  if (type == nvinfer1::DataType::kFLOAT) return sizeof(float);
  if (type == nvinfer1::DataType::kHALF) return sizeof(std::uint16_t);
  throw std::runtime_error("TensorRT detector supports only FP32/FP16 I/O");
}

class TensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "TensorRT: " << message << '\n';
    }
  }
};

}  // namespace

struct TongjiYoloDetector::TensorRtState {
  explicit TensorRtState(const std::string& engine_path) {
    requireCuda(cudaSetDevice(0), "cudaSetDevice");
    std::ifstream input(engine_path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot open TensorRT engine: " + engine_path);
    }
    const std::vector<char> bytes{std::istreambuf_iterator<char>(input), {}};
    if (bytes.empty()) throw std::runtime_error("TensorRT engine is empty");

    runtime = nvinfer1::createInferRuntime(logger);
    if (runtime == nullptr) throw std::runtime_error("createInferRuntime failed");
    engine = runtime->deserializeCudaEngine(bytes.data(), bytes.size());
    if (engine == nullptr) {
      throw std::runtime_error("TensorRT engine deserialization failed");
    }
    if (engine->getNbIOTensors() != 2) {
      throw std::runtime_error("TensorRT engine must expose one input and output");
    }
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
      const char* name = engine->getIOTensorName(index);
      if (name == nullptr) throw std::runtime_error("TensorRT tensor has no name");
      if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        input_name = name;
      } else {
        output_name = name;
      }
    }
    if (input_name.empty() || output_name.empty()) {
      throw std::runtime_error("TensorRT I/O roles are invalid");
    }
    input_dims = engine->getTensorShape(input_name.c_str());
    output_dims = engine->getTensorShape(output_name.c_str());
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 ||
        input_dims.d[1] != 3 || input_dims.d[2] != 640 ||
        input_dims.d[3] != 640) {
      throw std::runtime_error("TensorRT input must be [1,3,640,640]");
    }
    if (output_dims.nbDims != 3 || output_dims.d[0] != 1 ||
        !((output_dims.d[1] == 25200 && output_dims.d[2] == 22) ||
          (output_dims.d[1] == 22 && output_dims.d[2] == 25200))) {
      throw std::runtime_error(
          "TensorRT output must be [1,25200,22] or [1,22,25200]");
    }
    input_type = engine->getTensorDataType(input_name.c_str());
    output_type = engine->getTensorDataType(output_name.c_str());
    input_bytes = tensorElements(input_dims) * elementBytes(input_type);
    output_bytes = tensorElements(output_dims) * elementBytes(output_type);

    context = engine->createExecutionContext();
    if (context == nullptr) {
      throw std::runtime_error("TensorRT execution context creation failed");
    }
    requireCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    requireCuda(cudaEventCreate(&gpu_start), "cudaEventCreate(start)");
    requireCuda(cudaEventCreate(&preprocess_done),
                "cudaEventCreate(preprocess)");
    requireCuda(cudaEventCreate(&inference_done),
                "cudaEventCreate(inference)");
    requireCuda(cudaEventCreate(&copy_done), "cudaEventCreate(copy)");
    requireCuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc(input)");
    requireCuda(cudaMalloc(&device_output, output_bytes), "cudaMalloc(output)");
    requireCuda(cudaMallocHost(&host_output, output_bytes),
                "cudaMallocHost(output)");
    if (!context->setTensorAddress(input_name.c_str(), device_input) ||
        !context->setTensorAddress(output_name.c_str(), device_output)) {
      throw std::runtime_error("TensorRT setTensorAddress failed");
    }

    requireCuda(cudaMemsetAsync(device_input, 0, input_bytes, stream),
                "cudaMemsetAsync(warmup)");
    if (!context->enqueueV3(stream)) {
      throw std::runtime_error("TensorRT warmup enqueueV3 failed");
    }
    requireCuda(cudaStreamSynchronize(stream), "TensorRT warmup sync");
  }

  ~TensorRtState() {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    if (host_raw != nullptr) cudaFreeHost(host_raw);
    if (host_output != nullptr) cudaFreeHost(host_output);
    if (device_raw != nullptr) cudaFree(device_raw);
    if (device_input != nullptr) cudaFree(device_input);
    if (device_output != nullptr) cudaFree(device_output);
    if (gpu_start != nullptr) cudaEventDestroy(gpu_start);
    if (preprocess_done != nullptr) cudaEventDestroy(preprocess_done);
    if (inference_done != nullptr) cudaEventDestroy(inference_done);
    if (copy_done != nullptr) cudaEventDestroy(copy_done);
    if (stream != nullptr) cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
  }

  void ensureRawCapacity(std::size_t bytes) {
    if (bytes <= raw_capacity) return;
    if (host_raw != nullptr) requireCuda(cudaFreeHost(host_raw), "cudaFreeHost(raw)");
    if (device_raw != nullptr) requireCuda(cudaFree(device_raw), "cudaFree(raw)");
    host_raw = nullptr;
    device_raw = nullptr;
    requireCuda(cudaMallocHost(&host_raw, bytes), "cudaMallocHost(raw)");
    requireCuda(cudaMalloc(&device_raw, bytes), "cudaMalloc(raw)");
    raw_capacity = bytes;
  }

  TensorRtLogger logger;
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t gpu_start = nullptr;
  cudaEvent_t preprocess_done = nullptr;
  cudaEvent_t inference_done = nullptr;
  cudaEvent_t copy_done = nullptr;
  void* host_raw = nullptr;
  void* host_output = nullptr;
  void* device_raw = nullptr;
  void* device_input = nullptr;
  void* device_output = nullptr;
  std::size_t raw_capacity = 0;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  nvinfer1::DataType input_type{};
  nvinfer1::DataType output_type{};
  nvinfer1::Dims input_dims{};
  nvinfer1::Dims output_dims{};
  std::string input_name;
  std::string output_name;
};

TongjiYoloDetector::TongjiYoloDetector(DetectorConfig config)
    : config_(std::move(config)),
      environment_(ORT_LOGGING_LEVEL_WARNING, "autoaim_research") {
  if (!(config_.score_threshold > 0.0F && config_.score_threshold < 1.0F)) {
    throw std::runtime_error("score_threshold must be in (0, 1)");
  }
  if (!(config_.nms_threshold > 0.0F && config_.nms_threshold < 1.0F)) {
    throw std::runtime_error("nms_threshold must be in (0, 1)");
  }
  if (config_.inference_backend == "tensorrt_fp16") {
    if (config_.engine_path.empty() ||
        !std::filesystem::is_regular_file(config_.engine_path)) {
      throw std::runtime_error("TensorRT engine does not exist: " +
                               config_.engine_path);
    }
    tensorrt_ = std::make_unique<TensorRtState>(config_.engine_path);
    backend_name_ = "tensorrt_fp16";
    return;
  }
  if (config_.inference_backend != "onnxruntime_cpu") {
    throw std::runtime_error("unknown inference_backend: " +
                             config_.inference_backend);
  }
  if (config_.model_path.empty() ||
      !std::filesystem::is_regular_file(config_.model_path)) {
    throw std::runtime_error("YOLO model does not exist: " + config_.model_path);
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
  backend_name_ = "onnxruntime_cpu";
}

TongjiYoloDetector::~TongjiYoloDetector() = default;

std::list<auto_aim::Armor> TongjiYoloDetector::detect(const cv::Mat& bgr_image) {
  if (bgr_image.empty() || bgr_image.type() != CV_8UC3) {
    throw std::runtime_error("detector requires a non-empty CV_8UC3 image");
  }
  if (tensorrt_ != nullptr) return detectTensorRt(bgr_image);
  return detectOnnxRuntime(bgr_image);
}

std::list<auto_aim::Armor> TongjiYoloDetector::detectOnnxRuntime(
    const cv::Mat& bgr_image) {
  const auto total_begin = Clock::now();

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
  const auto preprocess_end = Clock::now();

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
  const auto inference_end = Clock::now();
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
  auto armors = parse(output, scale);
  const auto total_end = Clock::now();
  last_timing_ = {};
  last_timing_.host_prepare_ms =
      elapsedMilliseconds(total_begin, preprocess_end);
  last_timing_.inference_ms =
      elapsedMilliseconds(preprocess_end, inference_end);
  last_timing_.postprocess_ms =
      elapsedMilliseconds(inference_end, total_end);
  last_timing_.total_ms = elapsedMilliseconds(total_begin, total_end);
  return armors;
}

std::list<auto_aim::Armor> TongjiYoloDetector::detectTensorRt(
    const cv::Mat& bgr_image) {
  const auto total_begin = Clock::now();
  auto& state = *tensorrt_;
  const std::size_t row_bytes = static_cast<std::size_t>(bgr_image.cols) * 3;
  const std::size_t image_bytes = row_bytes * bgr_image.rows;
  state.ensureRawCapacity(image_bytes);
  if (bgr_image.isContinuous() && bgr_image.step == row_bytes) {
    std::memcpy(state.host_raw, bgr_image.data, image_bytes);
  } else {
    auto* destination = static_cast<std::uint8_t*>(state.host_raw);
    for (int row = 0; row < bgr_image.rows; ++row) {
      std::memcpy(destination + static_cast<std::size_t>(row) * row_bytes,
                  bgr_image.ptr(row), row_bytes);
    }
  }
  const double scale = std::min(640.0 / static_cast<double>(bgr_image.rows),
                                640.0 / static_cast<double>(bgr_image.cols));
  const int resized_height = static_cast<int>(bgr_image.rows * scale);
  const int resized_width = static_cast<int>(bgr_image.cols * scale);
  const auto host_prepare_end = Clock::now();

  requireCuda(cudaEventRecord(state.gpu_start, state.stream),
              "cudaEventRecord(start)");
  requireCuda(cudaMemcpyAsync(state.device_raw, state.host_raw, image_bytes,
                              cudaMemcpyHostToDevice, state.stream),
              "cudaMemcpyAsync(raw H2D)");
  requireCuda(launchLetterboxBgrToRgbChw(
                  static_cast<const std::uint8_t*>(state.device_raw),
                  bgr_image.cols, bgr_image.rows, row_bytes,
                  state.device_input,
                  state.input_type == nvinfer1::DataType::kHALF,
                  resized_width, resized_height, static_cast<float>(scale),
                  state.stream),
              "launchLetterboxBgrToRgbChw");
  requireCuda(cudaEventRecord(state.preprocess_done, state.stream),
              "cudaEventRecord(preprocess)");
  if (!state.context->enqueueV3(state.stream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }
  requireCuda(cudaEventRecord(state.inference_done, state.stream),
              "cudaEventRecord(inference)");
  requireCuda(cudaMemcpyAsync(state.host_output, state.device_output,
                              state.output_bytes, cudaMemcpyDeviceToHost,
                              state.stream),
              "cudaMemcpyAsync(output D2H)");
  requireCuda(cudaEventRecord(state.copy_done, state.stream),
              "cudaEventRecord(copy)");
  requireCuda(cudaEventSynchronize(state.copy_done),
              "cudaEventSynchronize(copy)");

  float gpu_preprocess_ms = 0.0F;
  float inference_ms = 0.0F;
  float output_copy_ms = 0.0F;
  requireCuda(cudaEventElapsedTime(&gpu_preprocess_ms, state.gpu_start,
                                   state.preprocess_done),
              "cudaEventElapsedTime(preprocess)");
  requireCuda(cudaEventElapsedTime(&inference_ms, state.preprocess_done,
                                   state.inference_done),
              "cudaEventElapsedTime(inference)");
  requireCuda(cudaEventElapsedTime(&output_copy_ms, state.inference_done,
                                   state.copy_done),
              "cudaEventElapsedTime(copy)");

  cv::Mat output;
  const int dimension_one = state.output_dims.d[1];
  const int dimension_two = state.output_dims.d[2];
  if (state.output_type == nvinfer1::DataType::kHALF) {
    cv::Mat half;
    if (dimension_two == 22) {
      half = cv::Mat(dimension_one, dimension_two, CV_16F,
                     state.host_output);
    } else {
      const cv::Mat transposed(dimension_one, dimension_two, CV_16F,
                               state.host_output);
      half = transposed.t();
    }
    half.convertTo(output, CV_32F);
  } else if (dimension_two == 22) {
    output = cv::Mat(dimension_one, dimension_two, CV_32F,
                     state.host_output);
  } else {
    const cv::Mat transposed(dimension_one, dimension_two, CV_32F,
                             state.host_output);
    output = transposed.t();
  }
  last_output_shape_ = {output.cols, output.rows};
  auto armors = parse(output, scale);
  const auto total_end = Clock::now();
  last_timing_ = {};
  last_timing_.host_prepare_ms =
      elapsedMilliseconds(total_begin, host_prepare_end);
  last_timing_.gpu_preprocess_ms = gpu_preprocess_ms;
  last_timing_.inference_ms = inference_ms;
  last_timing_.output_copy_ms = output_copy_ms;
  last_timing_.postprocess_ms = elapsedMilliseconds(
      host_prepare_end, total_end) - gpu_preprocess_ms - inference_ms -
      output_copy_ms;
  last_timing_.postprocess_ms = std::max(0.0, last_timing_.postprocess_ms);
  last_timing_.total_ms = elapsedMilliseconds(total_begin, total_end);
  return armors;
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
