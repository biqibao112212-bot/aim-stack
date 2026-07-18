/**
 * @file yolo11_buff.cpp
 * @brief YOLO11 能量机关关键点检测实现（TensorRT 推理）
 */

#include "yolo11_buff.hpp"
#include "yolo11_preprocess_kernel.hpp"

#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <new>

namespace
{
std::filesystem::path resolve_engine_path(const std::string & config, const YAML::Node & yaml)
{
    const std::string configured_path =
        yaml["yolo11_model_path"] ? yaml["yolo11_model_path"].as<std::string>() : "assets/best.engine";

    std::filesystem::path raw_path(configured_path);
    if (raw_path.is_absolute()) {
        return raw_path;
    }

    const std::filesystem::path config_path(config);
    std::vector<std::filesystem::path> candidates = {
        raw_path,
        config_path.parent_path() / raw_path,
        config_path.parent_path().parent_path() / raw_path
    };

    for (const auto & candidate : candidates) {
        if (!candidate.empty() && std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return candidates.front();
}

float clamp_threshold(float value, float min_value, float max_value)
{
    return std::max(min_value, std::min(value, max_value));
}

template <typename T>
bool read_optional_nested_scalar(
    const YAML::Node & parent, const char * child_key, T & value)
{
    if (parent && parent[child_key]) {
        value = parent[child_key].as<T>();
        return true;
    }
    return false;
}

bool debug_save_low_conf_enabled()
{
    static const bool enabled = []() {
        const char * env = std::getenv("BUFF_SAVE_LOW_CONF");
        if (env == nullptr) {
            return false;
        }
        return std::string(env) != "0";
    }();
    return enabled;
}

bool draw_all_candidates_enabled()
{
    static const bool enabled = []() {
        const char * env = std::getenv("BUFF_SHOW_ALL_CANDIDATES");
        if (env == nullptr) {
            return false;
        }
        return std::string(env) != "0";
    }();
    return enabled;
}

bool profile_enabled()
{
    static const bool enabled = []() {
        const char * env = std::getenv("BUFF_PROFILE");
        if (env == nullptr) {
            return false;
        }
        return std::string(env) != "0";
    }();
    return enabled;
}

bool gpu_preprocess_enabled()
{
    static const bool enabled = []() {
        const char * env = std::getenv("BUFF_GPU_PREPROCESS");
        if (env == nullptr) {
            return true;
        }
        return std::string(env) != "0";
    }();
    return enabled;
}

bool is_pending_label_for_schema(int label, int num_classes)
{
    if (num_classes >= 6) {
        return label == 1 || label == 4;
    }
    return label == 0 || label == 2;
}

bool is_context_label_for_schema(int label, int num_classes)
{
    if (num_classes >= 6) {
        return label == 0 || label == 2 || label == 3 || label == 5;
    }
    return label == 1 || label == 3;
}

std::string buff_label_name_for_schema(int label, int num_classes)
{
    if (num_classes >= 6) {
        switch (label) {
            case 0: return "R-Unlit";
            case 1: return "R-Pending";
            case 2: return "R-Activated";
            case 3: return "B-Unlit";
            case 4: return "B-Pending";
            case 5: return "B-Activated";
            default: return "Unknown";
        }
    }
    switch (label) {
        case 0: return "R-Target";
        case 1: return "R-Hit";
        case 2: return "B-Target";
        case 3: return "B-Hit";
        default: return "Unknown";
    }
}

double ticks_to_ms(int64 start, int64 end)
{
    return (end - start) * 1000.0 / static_cast<double>(cv::getTickFrequency());
}

double ticks_to_ms(int64 start)
{
    return ticks_to_ms(start, cv::getTickCount());
}
}

namespace auto_buff
{

// ==================== 辅助宏与结构 ====================

// 检查 CUDA 运行时错误
#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            tools::logger()->error("Cuda failure: {}", ret); \
            abort(); \
        } \
    } while (0)

// 自定义删除器，用于 unique_ptr 管理 TensorRT 对象
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj) delete obj; // 较新版本的 TensorRT C++ API 使用 delete，旧版本可能需要 obj->destroy()
    }
};

// ==================== TRTLogger 实现 ====================

void TRTLogger::log(Severity severity, const char * msg) noexcept
{
    // 根据 severity 级别输出日志
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            tools::logger()->error("[TRT] {}", msg);
            break;
        case Severity::kERROR:
            tools::logger()->error("[TRT] {}", msg);
            break;
        case Severity::kWARNING:
            tools::logger()->warn("[TRT] {}", msg);
            break;
        case Severity::kINFO:
            // tools::logger()->info("[TRT] {}", msg); // 减少日志刷屏
            break;
        case Severity::kVERBOSE:
            // tools::logger()->debug("[TRT] {}", msg);
            break;
    }
}

// ==================== 构造函数 ====================

YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
: YOLO11_BUFF(config, Options{})
{
}

YOLO11_BUFF::YOLO11_BUFF(const std::string & config, const Options & options)
: stream_(nullptr), device_input_(nullptr), device_output_(nullptr), host_output_(nullptr), host_output_half_(nullptr)
{
    profile_enabled_ = options.enable_profiling || profile_enabled();
    output_host_memory_mode_ = options.output_host_memory_mode;
    gpu_preprocess_enabled_ = gpu_preprocess_enabled();

    // 1. 解析 YAML，并从配置里加载模型路径与阈值
  auto yaml = YAML::LoadFile(config);
    const std::filesystem::path engine_path = resolve_engine_path(config, yaml);
    const auto buff_detector_yaml = yaml["buff_detector"];
    const auto detector_yaml = yaml["detector"];

    if (
        !read_optional_nested_scalar(buff_detector_yaml, "confidence_threshold", conf_threshold_) &&
        !read_optional_nested_scalar(detector_yaml, "confidence_threshold", conf_threshold_) &&
        yaml["min_confidence"]) {
        conf_threshold_ = yaml["min_confidence"].as<float>();
    }

    if (
        !read_optional_nested_scalar(buff_detector_yaml, "nms_iou_threshold", nms_iou_threshold_) &&
        !read_optional_nested_scalar(detector_yaml, "nms_iou_threshold", nms_iou_threshold_) &&
        yaml["nms_iou_threshold"]) {
        nms_iou_threshold_ = yaml["nms_iou_threshold"].as<float>();
    }

    conf_threshold_ = clamp_threshold(conf_threshold_, 0.05f, 0.99f);
    nms_iou_threshold_ = clamp_threshold(nms_iou_threshold_, 0.05f, 0.95f);

    tools::logger()->info(
        "YOLO11 buff config -> engine: {}, conf_threshold: {:.2f}, nms_iou: {:.2f}, preprocess: {}",
        engine_path.string(),
        conf_threshold_,
        nms_iou_threshold_,
        gpu_preprocess_enabled_ ? "gpu" : "cpu");

  // 2. 调用 loadEngine() 加载 TensorRT 引擎
  if (!loadEngine(engine_path.string())) {
    tools::logger()->error("Failed to load engine: {}", engine_path.string());
    exit(-1);
  }

  // 3. 调用 allocateBuffers() 分配 GPU/CPU 缓冲区
  if (!allocateBuffers()) {
    tools::logger()->error("Failed to allocate buffers");
    exit(-1);
  }

  // 4. 创建 CUDA 流
  cudaError_t ret = cudaStreamCreate(&stream_);
  if (ret != cudaSuccess) {
    tools::logger()->error("Failed to create CUDA stream: {}", cudaGetErrorString(ret));
    exit(-1);
  }

  if (profile_enabled_ && !createProfileEvents()) {
    tools::logger()->warn("Failed to create CUDA timing events; GPU profile split disabled");
  }

  const char * warmup_env = std::getenv("BUFF_TRT_WARMUP");
  if (warmup_env == nullptr || std::string(warmup_env) != "0") {
    cudaMemsetAsync(device_input_, 0, input_size_, stream_);
    context_->setTensorAddress(input_tensor_name_.c_str(), device_input_);
    context_->setTensorAddress(output_tensor_name_.c_str(), device_output_);
    if (context_->enqueueV3(stream_)) {
      cudaStreamSynchronize(stream_);
    } else {
      tools::logger()->warn("TensorRT warmup enqueue failed");
    }
  }

  // 5. (可选) 调用 printEngineInfo() 打印引擎信息
  printEngineInfo();
}

// ==================== 析构函数 ====================

YOLO11_BUFF::~YOLO11_BUFF()
{
    printProfile();

    // 1. 释放缓冲区
    releaseBuffers();

    destroyProfileEvents();

    // 2. 销毁 CUDA 流
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    // 注意：unique_ptr 会自动释放 runtime_, engine_, context_
}

// ==================== 加载 TensorRT 引擎 ====================

bool YOLO11_BUFF::loadEngine(const std::string & engine_path)
{
  // 1. 以二进制模式读取 .engine 文件到内存
  std::ifstream file(engine_path, std::ios::binary);
  if (!file.good()) {
    tools::logger()->error("Engine file not found: {}", engine_path);
    return false;
  }

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  // 初始化 TensorRT 插件 (如果模型包含插件层)
  // initLibNvInferPlugins(&logger_, "");

  // 2. 创建 TensorRT Runtime
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) return false;

  // 3. 反序列化引擎
  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
  if (!engine_) return false;

  // 4. 创建执行上下文
  context_.reset(engine_->createExecutionContext());
  if (!context_) return false;

  // 5. 获取输入输出的维度信息，设置 output_rows_, output_cols_ 等
  int nbIOTensors = engine_->getNbIOTensors();
  for (int i = 0; i < nbIOTensors; ++i) {
    const char* name = engine_->getIOTensorName(i);
    nvinfer1::Dims dims = engine_->getTensorShape(name);
        const nvinfer1::DataType dtype = engine_->getTensorDataType(name);

    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_tensor_name_ = name;
            input_dtype_ = dtype;
      // 输入固定为 [1, 3, H, W]
      input_c_ = dims.d[1];
      input_h_ = dims.d[2];
      input_w_ = dims.d[3];
    } else {
            output_tensor_name_ = name;
            output_dtype_ = dtype;
      // 输出为 [1, C, anchors]，C=18 表示 5*(x,y)，C=23 表示 5*(x,y,v)
      output_rows_ = dims.d[1];
      output_cols_ = dims.d[2];
    }
  }

    if (input_tensor_name_.empty() || output_tensor_name_.empty()) {
        tools::logger()->error("Failed to resolve input/output tensor names from engine");
        return false;
    }

    bool layout_resolved = false;
    for (int candidate_classes : {6, 4}) {
        for (int candidate_kpt_dim : {2, 3}) {
            if (output_rows_ == 4 + candidate_classes + NUM_POINTS * candidate_kpt_dim) {
                num_classes_ = candidate_classes;
                keypoint_dim_ = candidate_kpt_dim;
                keypoint_offset_ = 4 + num_classes_;
                layout_resolved = true;
                break;
            }
        }
        if (layout_resolved) {
            break;
        }
    }

    if (!layout_resolved) {
        for (int candidate_kpt_dim : {2, 3}) {
            const int candidate_classes = output_rows_ - 4 - NUM_POINTS * candidate_kpt_dim;
            if (candidate_classes > 0 && candidate_classes <= 32) {
                num_classes_ = candidate_classes;
                keypoint_dim_ = candidate_kpt_dim;
                keypoint_offset_ = 4 + num_classes_;
                layout_resolved = true;
                break;
            }
        }
    }

    if (!layout_resolved || (keypoint_dim_ != 2 && keypoint_dim_ != 3)) {
        tools::logger()->error(
            "Unsupported YOLO pose output rows: {}, expected 4 + classes + {} keypoints",
            output_rows_,
            NUM_POINTS);
        return false;
    }

    tools::logger()->info(
        "Engine IO dtype - input: {}, output: {}, output_rows: {}, classes: {}, keypoint_dim: {}",
        static_cast<int>(input_dtype_),
        static_cast<int>(output_dtype_),
        output_rows_,
        num_classes_,
        keypoint_dim_);

  return true;
}
// ==================== 分配缓冲区 ====================

bool YOLO11_BUFF::allocateOutputTransferBuffer()
{
    output_host_memory_pinned_ = false;
    output_host_memory_fallback_used_ = false;

    const bool inject_failure =
        output_host_memory_mode_ == OutputHostMemoryMode::InjectPinnedAllocationFailure;
    const bool try_pinned =
        output_host_memory_mode_ == OutputHostMemoryMode::PinnedPreferred;

    if (try_pinned && output_size_ <= kMaxPinnedOutputBytes) {
        void * pinned_buffer = nullptr;
        const cudaError_t ret =
            cudaHostAlloc(&pinned_buffer, output_size_, cudaHostAllocDefault);
        if (ret == cudaSuccess && pinned_buffer != nullptr) {
            if (output_dtype_ == nvinfer1::DataType::kHALF) {
                host_output_half_ = static_cast<__half *>(pinned_buffer);
            } else {
                host_output_ = static_cast<float *>(pinned_buffer);
            }
            output_host_memory_pinned_ = true;
            tools::logger()->info(
                "YOLO11 buff D2H host buffer: pinned, bytes={}", output_size_);
            return true;
        }
        output_host_memory_fallback_used_ = true;
        tools::logger()->warn(
            "Pinned YOLO output allocation failed ({} bytes): {}; using pageable fallback",
            output_size_,
            cudaGetErrorString(ret));
        (void)cudaGetLastError();
    } else if (try_pinned) {
        output_host_memory_fallback_used_ = true;
        tools::logger()->warn(
            "YOLO output buffer {} bytes exceeds pinned limit {} bytes; using pageable fallback",
            output_size_,
            kMaxPinnedOutputBytes);
    } else if (inject_failure) {
        output_host_memory_fallback_used_ = true;
        tools::logger()->info(
            "YOLO pinned output allocation failure injected; using pageable fallback");
    }

    if (output_dtype_ == nvinfer1::DataType::kHALF) {
        host_output_half_ = new (std::nothrow) __half[output_elem_count_];
        if (host_output_half_ == nullptr) {
            return false;
        }
    } else {
        host_output_ = new (std::nothrow) float[output_elem_count_];
        if (host_output_ == nullptr) {
            return false;
        }
    }

    tools::logger()->info(
        "YOLO11 buff D2H host buffer: pageable, bytes={}, fallback={}",
        output_size_,
        output_host_memory_fallback_used_);
    return true;
}

void YOLO11_BUFF::releaseOutputTransferBuffer()
{
    void * transfer_buffer = output_dtype_ == nvinfer1::DataType::kHALF
        ? static_cast<void *>(host_output_half_)
        : static_cast<void *>(host_output_);
    if (transfer_buffer == nullptr) {
        output_host_memory_pinned_ = false;
        return;
    }

    if (output_host_memory_pinned_) {
        const cudaError_t ret = cudaFreeHost(transfer_buffer);
        if (ret != cudaSuccess) {
            tools::logger()->warn(
                "Failed to release pinned YOLO output buffer: {}", cudaGetErrorString(ret));
        }
    } else if (output_dtype_ == nvinfer1::DataType::kHALF) {
        delete[] host_output_half_;
    } else {
        delete[] host_output_;
    }

    if (output_dtype_ == nvinfer1::DataType::kHALF) {
        host_output_half_ = nullptr;
    } else {
        host_output_ = nullptr;
    }
    output_host_memory_pinned_ = false;
}

bool YOLO11_BUFF::allocateBuffers()
{
    // 1. 计算输入张量大小 (NCHW: 1 * C * H * W)
    input_elem_count_ = static_cast<size_t>(input_c_) * input_h_ * input_w_;
    const size_t input_elem_size =
            (input_dtype_ == nvinfer1::DataType::kHALF) ? sizeof(__half) : sizeof(float);
    input_size_ = input_elem_count_ * input_elem_size;

    // 2. 计算输出张量大小
    output_elem_count_ = static_cast<size_t>(output_rows_) * output_cols_;
    const size_t output_elem_size =
            (output_dtype_ == nvinfer1::DataType::kHALF) ? sizeof(__half) : sizeof(float);
    output_size_ = output_elem_count_ * output_elem_size;

  // 3. 分配 GPU 输入缓冲区
  if (cudaMalloc(&device_input_, input_size_) != cudaSuccess) return false;

  // 4. 分配 GPU 输出缓冲区
  if (cudaMalloc(&device_output_, output_size_) != cudaSuccess) return false;

    if (gpu_preprocess_enabled_) {
        constexpr size_t kDefaultRawImageCapacity = 3000u * 2000u * 3u;
        if (cudaMalloc(&device_raw_image_, kDefaultRawImageCapacity) == cudaSuccess) {
            device_raw_image_capacity_ = kDefaultRawImageCapacity;
        } else {
            tools::logger()->warn("Failed to preallocate raw image GPU buffer; CPU preprocess fallback enabled");
            device_raw_image_ = nullptr;
            device_raw_image_capacity_ = 0;
            gpu_preprocess_enabled_ = false;
        }
    }

    // 5. 分配 CPU 输出缓冲区（后处理统一使用 float）
    if (output_dtype_ == nvinfer1::DataType::kHALF) {
        host_output_ = new (std::nothrow) float[output_elem_count_];
        if (host_output_ == nullptr) {
            return false;
        }
    }
    if (!allocateOutputTransferBuffer()) {
        if (output_dtype_ == nvinfer1::DataType::kHALF) {
            delete[] host_output_;
            host_output_ = nullptr;
        }
        return false;
    }

    if (input_dtype_ == nvinfer1::DataType::kHALF) {
        host_input_half_cache_.resize(input_elem_count_);
    } else {
        host_input_float_cache_.resize(input_elem_count_);
    }

  return true;
}

// ==================== 释放缓冲区 ====================

void YOLO11_BUFF::releaseBuffers()
{
    // 1. 释放 GPU 输入缓冲区
    if (device_input_) {
        cudaFree(device_input_);
        device_input_ = nullptr;
    }

    // 2. 释放 GPU 输出缓冲区
    if (device_output_) {
        cudaFree(device_output_);
        device_output_ = nullptr;
    }

    if (device_raw_image_) {
        cudaFree(device_raw_image_);
        device_raw_image_ = nullptr;
        device_raw_image_capacity_ = 0;
    }

    // 3. 释放 CPU 输出缓冲区
    releaseOutputTransferBuffer();
    if (output_dtype_ == nvinfer1::DataType::kHALF && host_output_) {
        delete[] host_output_;
        host_output_ = nullptr;
    }
}

// ==================== 图像预处理 ====================

bool YOLO11_BUFF::createProfileEvents()
{
    destroyProfileEvents();
    const bool ok =
        cudaEventCreate(&h2d_start_event_) == cudaSuccess &&
        cudaEventCreate(&h2d_end_event_) == cudaSuccess &&
        cudaEventCreate(&preprocess_kernel_end_event_) == cudaSuccess &&
        cudaEventCreate(&trt_end_event_) == cudaSuccess &&
        cudaEventCreate(&d2h_end_event_) == cudaSuccess;
    if (!ok) {
        destroyProfileEvents();
        return false;
    }
    profile_events_ready_ = true;
    return true;
}

void YOLO11_BUFF::destroyProfileEvents()
{
    if (h2d_start_event_) cudaEventDestroy(h2d_start_event_);
    if (h2d_end_event_) cudaEventDestroy(h2d_end_event_);
    if (preprocess_kernel_end_event_) cudaEventDestroy(preprocess_kernel_end_event_);
    if (trt_end_event_) cudaEventDestroy(trt_end_event_);
    if (d2h_end_event_) cudaEventDestroy(d2h_end_event_);
    h2d_start_event_ = nullptr;
    h2d_end_event_ = nullptr;
    preprocess_kernel_end_event_ = nullptr;
    trt_end_event_ = nullptr;
    d2h_end_event_ = nullptr;
    profile_events_ready_ = false;
    gpu_timing_valid_ = false;
}

void YOLO11_BUFF::beginFrameProfile()
{
    gpu_timing_valid_ = false;
    if (!profile_enabled_) {
        return;
    }
    last_frame_profile_ = FrameProfile{};
    last_frame_profile_.enabled = true;
    last_frame_profile_.output_host_pinned = output_host_memory_pinned_;
    last_frame_profile_.d2h_bytes = output_size_;
}

void YOLO11_BUFF::finalizeGpuProfile()
{
    if (!gpu_timing_valid_) {
        last_frame_profile_.gpu_events_valid = false;
        return;
    }

    float h2d_ms = 0.0f;
    float preprocess_kernel_ms = 0.0f;
    float trt_ms = 0.0f;
    float d2h_ms = 0.0f;
    const bool ok =
        cudaEventElapsedTime(&h2d_ms, h2d_start_event_, h2d_end_event_) == cudaSuccess &&
        cudaEventElapsedTime(
            &preprocess_kernel_ms, h2d_end_event_, preprocess_kernel_end_event_) == cudaSuccess &&
        cudaEventElapsedTime(
            &trt_ms, preprocess_kernel_end_event_, trt_end_event_) == cudaSuccess &&
        cudaEventElapsedTime(&d2h_ms, trt_end_event_, d2h_end_event_) == cudaSuccess;
    if (!ok) {
        gpu_timing_valid_ = false;
        last_frame_profile_.gpu_events_valid = false;
        return;
    }

    last_frame_profile_.gpu_events_valid = true;
    if (last_frame_profile_.gpu_preprocess_used) {
        last_frame_profile_.raw_h2d_gpu_ms = h2d_ms;
        last_frame_profile_.preprocess_kernel_gpu_ms = preprocess_kernel_ms;
    } else {
        last_frame_profile_.model_input_h2d_gpu_ms = h2d_ms;
    }
    last_frame_profile_.trt_gpu_ms = trt_ms;
    last_frame_profile_.d2h_gpu_ms = d2h_ms;
}

float YOLO11_BUFF::preprocess(const cv::Mat & input_image)
{
  if (gpu_preprocess_enabled_) {
    if (profile_enabled_) last_frame_profile_.gpu_preprocess_used = true;
    const float scale = preprocessGpu(input_image);
    if (scale > 0.0f) {
      return scale;
    }
    if (profile_enabled_) last_frame_profile_.gpu_preprocess_fallback = true;
  }
  return preprocessCpu(input_image);
}

float YOLO11_BUFF::preprocessCpu(const cv::Mat & input_image)
{
  const int64 cpu_preprocess_start = profile_enabled_ ? cv::getTickCount() : 0;
  if (profile_enabled_) {
    last_frame_profile_.gpu_preprocess_used = false;
    last_frame_profile_.h2d_bytes = input_size_;
  }
  // 1. Letterbox 变换：保持宽高比缩放，填充黑边
  float r = std::min((float)input_h_ / input_image.rows, (float)input_w_ / input_image.cols);
  int padw = std::round(input_image.cols * r);
  int padh = std::round(input_image.rows * r);

  // 计算居中填充的偏移量
  float dw = (input_w_ - padw) / 2.0f;
  float dh = (input_h_ - padh) / 2.0f;

  int top = int(std::round(dh - 0.1f));
  int left = int(std::round(dw - 0.1f));

  // 2. 直接在 640x640 画布 ROI 内 resize，避免额外 copyMakeBorder。
  letterbox_cache_.create(input_h_, input_w_, CV_8UC3);
  letterbox_cache_.setTo(cv::Scalar(114, 114, 114));
  cv::Mat roi = letterbox_cache_(cv::Rect(left, top, padw, padh));
  if ((int)input_image.cols != padw || (int)input_image.rows != padh) {
    cv::resize(input_image, roi, cv::Size(padw, padh));
  } else {
    input_image.copyTo(roi);
  }

  // 3. Ultralytics YOLO 预处理：BGR->RGB、归一化到 [0, 1]、转 NCHW。
  const int area = input_h_ * input_w_;
  if (input_dtype_ == nvinfer1::DataType::kHALF) {
      __half * dst_r = host_input_half_cache_.data();
      __half * dst_g = dst_r + area;
      __half * dst_b = dst_g + area;
      for (int y = 0; y < input_h_; ++y) {
          const uchar * row = letterbox_cache_.ptr<uchar>(y);
          const int row_offset = y * input_w_;
          for (int x = 0; x < input_w_; ++x) {
              const uchar * px = row + x * 3;
              const int idx = row_offset + x;
              dst_r[idx] = __float2half(static_cast<float>(px[2]) / 255.0f);
              dst_g[idx] = __float2half(static_cast<float>(px[1]) / 255.0f);
              dst_b[idx] = __float2half(static_cast<float>(px[0]) / 255.0f);
          }
      }
  } else {
      float * dst_r = host_input_float_cache_.data();
      float * dst_g = dst_r + area;
      float * dst_b = dst_g + area;
      for (int y = 0; y < input_h_; ++y) {
          const uchar * row = letterbox_cache_.ptr<uchar>(y);
          const int row_offset = y * input_w_;
          for (int x = 0; x < input_w_; ++x) {
              const uchar * px = row + x * 3;
              const int idx = row_offset + x;
              dst_r[idx] = static_cast<float>(px[2]) / 255.0f;
              dst_g[idx] = static_cast<float>(px[1]) / 255.0f;
              dst_b[idx] = static_cast<float>(px[0]) / 255.0f;
          }
      }
  }

  // 4. 将预处理后的数据拷贝到 GPU
    if (profile_enabled_) {
        last_frame_profile_.cpu_preprocess_ms = ticks_to_ms(cpu_preprocess_start);
    }
    gpu_timing_valid_ = profile_enabled_ && profile_events_ready_;
    if (gpu_timing_valid_ && cudaEventRecord(h2d_start_event_, stream_) != cudaSuccess) {
        gpu_timing_valid_ = false;
    }

    const int64 h2d_api_start = profile_enabled_ ? cv::getTickCount() : 0;
    cudaError_t h2d_status = cudaSuccess;
    if (input_dtype_ == nvinfer1::DataType::kHALF) {
        h2d_status = cudaMemcpyAsync(
            device_input_,
            host_input_half_cache_.data(),
            input_size_,
            cudaMemcpyHostToDevice,
            stream_);
    } else {
        h2d_status = cudaMemcpyAsync(
            device_input_,
            host_input_float_cache_.data(),
            input_size_,
            cudaMemcpyHostToDevice,
            stream_);
    }
    if (profile_enabled_) {
        last_frame_profile_.h2d_api_ms = ticks_to_ms(h2d_api_start);
        last_frame_profile_.h2d_status = static_cast<int>(h2d_status);
    }
    if (h2d_status != cudaSuccess) {
        gpu_timing_valid_ = false;
        tools::logger()->warn(
            "CPU preprocess H2D copy failed: {}", cudaGetErrorString(h2d_status));
    } else if (gpu_timing_valid_) {
        const bool events_ok =
            cudaEventRecord(h2d_end_event_, stream_) == cudaSuccess &&
            cudaEventRecord(preprocess_kernel_end_event_, stream_) == cudaSuccess;
        if (!events_ok) {
            gpu_timing_valid_ = false;
        }
    }

  return r;  // 返回缩放比例
}

bool YOLO11_BUFF::ensureDeviceRawCapacity(size_t required_bytes)
{
    if (required_bytes <= device_raw_image_capacity_) {
        return true;
    }

    if (device_raw_image_) {
        cudaFree(device_raw_image_);
        device_raw_image_ = nullptr;
        device_raw_image_capacity_ = 0;
    }

    const cudaError_t ret = cudaMalloc(&device_raw_image_, required_bytes);
    if (ret != cudaSuccess) {
        tools::logger()->error(
          "Failed to allocate raw image GPU buffer ({} bytes): {}",
          required_bytes,
          cudaGetErrorString(ret));
        return false;
    }
    device_raw_image_capacity_ = required_bytes;
    return true;
}

float YOLO11_BUFF::preprocessGpu(const cv::Mat & input_image)
{
  if (input_image.empty() || input_image.type() != CV_8UC3) {
    return -1.0f;
  }

  const float r = std::min((float)input_h_ / input_image.rows, (float)input_w_ / input_image.cols);
  const int resized_w = std::round(input_image.cols * r);
  const int resized_h = std::round(input_image.rows * r);
  const float dw = (input_w_ - resized_w) / 2.0f;
  const float dh = (input_h_ - resized_h) / 2.0f;
  const int left = int(std::round(dw - 0.1f));
  const int top = int(std::round(dh - 0.1f));

  const size_t row_bytes = static_cast<size_t>(input_image.cols) * input_image.elemSize();
  const size_t required_bytes = row_bytes * static_cast<size_t>(input_image.rows);
  if (profile_enabled_) {
    last_frame_profile_.gpu_preprocess_used = true;
    last_frame_profile_.h2d_bytes = required_bytes;
  }
  if (!ensureDeviceRawCapacity(required_bytes)) {
    return -1.0f;
  }

  gpu_timing_valid_ = profile_enabled_ && profile_events_ready_;

  if (gpu_timing_valid_ && cudaEventRecord(h2d_start_event_, stream_) != cudaSuccess) {
    gpu_timing_valid_ = false;
  }

  const int64 h2d_api_start = profile_enabled_ ? cv::getTickCount() : 0;
  cudaError_t ret = cudaMemcpy2DAsync(
    device_raw_image_,
    row_bytes,
    input_image.data,
    input_image.step,
    row_bytes,
    input_image.rows,
    cudaMemcpyHostToDevice,
    stream_);
  if (profile_enabled_) {
    last_frame_profile_.h2d_api_ms = ticks_to_ms(h2d_api_start);
    last_frame_profile_.h2d_status = static_cast<int>(ret);
  }
  if (ret != cudaSuccess) {
    tools::logger()->warn("GPU preprocess H2D copy failed: {}", cudaGetErrorString(ret));
    gpu_timing_valid_ = false;
    return -1.0f;
  }
  if (gpu_timing_valid_ && cudaEventRecord(h2d_end_event_, stream_) != cudaSuccess) {
    gpu_timing_valid_ = false;
  }

  const int64 preprocess_launch_start = profile_enabled_ ? cv::getTickCount() : 0;
  if (input_dtype_ == nvinfer1::DataType::kHALF) {
    launch_yolo11_preprocess_half(
      static_cast<const uint8_t *>(device_raw_image_),
      input_image.cols,
      input_image.rows,
      row_bytes,
      static_cast<__half *>(device_input_),
      input_w_,
      input_h_,
      resized_w,
      resized_h,
      left,
      top,
      stream_);
  } else {
    launch_yolo11_preprocess_float(
      static_cast<const uint8_t *>(device_raw_image_),
      input_image.cols,
      input_image.rows,
      row_bytes,
      static_cast<float *>(device_input_),
      input_w_,
      input_h_,
      resized_w,
      resized_h,
      left,
      top,
      stream_);
  }
  if (profile_enabled_) {
    last_frame_profile_.preprocess_launch_api_ms = ticks_to_ms(preprocess_launch_start);
  }

  ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    tools::logger()->warn("GPU preprocess kernel launch failed: {}", cudaGetErrorString(ret));
    gpu_timing_valid_ = false;
    return -1.0f;
  }

  if (
    gpu_timing_valid_ &&
    cudaEventRecord(preprocess_kernel_end_event_, stream_) != cudaSuccess) {
    gpu_timing_valid_ = false;
  }

  return r;
}

// ==================== 执行推理 ====================

bool YOLO11_BUFF::inference()
{
  // 为执行上下文设置张量地址
    context_->setTensorAddress(input_tensor_name_.c_str(), device_input_);
    context_->setTensorAddress(output_tensor_name_.c_str(), device_output_);

  const int64 enqueue_api_start = profile_enabled_ ? cv::getTickCount() : 0;

  // 2. 执行异步推理
  const bool status = context_->enqueueV3(stream_);
  if (profile_enabled_) {
    last_frame_profile_.enqueue_api_ms = ticks_to_ms(enqueue_api_start);
  }
  if (!status) {
    gpu_timing_valid_ = false;
    return false;
  }
  if (gpu_timing_valid_ && cudaEventRecord(trt_end_event_, stream_) != cudaSuccess) {
    gpu_timing_valid_ = false;
  }

  // 3. 将输出从 GPU 拷贝到 CPU
    const int64 d2h_api_start = profile_enabled_ ? cv::getTickCount() : 0;
    const cudaError_t d2h_status = output_dtype_ == nvinfer1::DataType::kHALF
        ? cudaMemcpyAsync(
            host_output_half_, device_output_, output_size_, cudaMemcpyDeviceToHost, stream_)
        : cudaMemcpyAsync(
            host_output_, device_output_, output_size_, cudaMemcpyDeviceToHost, stream_);
    if (profile_enabled_) {
        last_frame_profile_.d2h_api_ms = ticks_to_ms(d2h_api_start);
        last_frame_profile_.d2h_status = static_cast<int>(d2h_status);
    }
    if (d2h_status != cudaSuccess) {
        gpu_timing_valid_ = false;
        tools::logger()->warn(
            "YOLO output D2H copy failed: {}", cudaGetErrorString(d2h_status));
        return false;
    }
    if (gpu_timing_valid_ && cudaEventRecord(d2h_end_event_, stream_) != cudaSuccess) {
        gpu_timing_valid_ = false;
    }

  // 4. 同步 CUDA 流，确保数据拷贝完成
  const int64 sync_wait_start = profile_enabled_ ? cv::getTickCount() : 0;
  const cudaError_t sync_status = cudaStreamSynchronize(stream_);
  if (profile_enabled_) {
      last_frame_profile_.sync_wait_ms = ticks_to_ms(sync_wait_start);
      last_frame_profile_.sync_status = static_cast<int>(sync_status);
  }
  if (sync_status != cudaSuccess) {
      gpu_timing_valid_ = false;
      tools::logger()->warn(
          "YOLO CUDA stream synchronization failed: {}", cudaGetErrorString(sync_status));
      return false;
  }

  if (profile_enabled_) finalizeGpuProfile();

  const int64 output_convert_start = profile_enabled_ ? cv::getTickCount() : 0;
  if (output_dtype_ == nvinfer1::DataType::kHALF) {
      for (size_t i = 0; i < output_elem_count_; ++i) {
          host_output_[i] = __half2float(host_output_half_[i]);
      }
  }
  if (profile_enabled_) {
      last_frame_profile_.output_convert_ms = ticks_to_ms(output_convert_start);
  }

  return true;
}

// ==================== 后处理（带 NMS）====================

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::postprocess(float scale_factor)
{
  std::vector<Object> results;
  const int64 decode_start = profile_enabled_ ? cv::getTickCount() : 0;

  // 1. 获取数据指针
  // 输出维度: [1, 18, 8400] 或 [1, 23, 8400]
  // 内存布局: [Channels, Anchors] -> Flattened
  // 访问方式: raw_data[channel_idx * stride + anchor_idx]
  const float* raw_data = host_output_;
  const int stride = output_cols_; // 8400

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> class_ids;
  std::vector<std::vector<cv::Point2f>> keypoints_list;

  // 2. 遍历所有 Anchor (8400)
  for (int i = 0; i < output_cols_; ++i) {
      // --- 寻找最大置信度的类别 (Channel 4-7) ---
      // 4:红色待击打, 5:红色已击打, 6:蓝色待击打, 7:蓝色已击打
      float max_score = -1.0f;
      int best_class_id = -1;

      // 遍历 4 个类别通道，找出最大值
      for (int c = 0; c < num_classes_; ++c) {
          float score = raw_data[(4 + c) * stride + i];
          if (score > max_score) {
              max_score = score;
              best_class_id = c;
          }
      }

      // 使用配置里的置信度阈值进行过滤
      if (max_score > conf_threshold_) {
          // --- 解析边界框 (Channel 0-3: x, y, w, h) ---
          // 注意：这里的坐标是相对于 640x640 输入图的
          float cx = raw_data[0 * stride + i];
          float cy = raw_data[1 * stride + i];
          float w  = raw_data[2 * stride + i];
          float h  = raw_data[3 * stride + i];

          // 转为左上角坐标 (cx, cy) -> (left, top)
          int left = static_cast<int>(cx - 0.5f * w);
          int top  = static_cast<int>(cy - 0.5f * h);
          int width = static_cast<int>(w);
          int height = static_cast<int>(h);

          boxes.emplace_back(left, top, width, height);
          confidences.push_back(max_score);
          class_ids.push_back(best_class_id);

          // --- 解析关键点 ---
          // 旧模型: 5 points * (x,y)；新模型: 5 points * (x,y,visible)。
          std::vector<cv::Point2f> kpts;
          kpts.reserve(NUM_POINTS);
          for (int k = 0; k < NUM_POINTS; ++k) {
              const int base = keypoint_offset_ + k * keypoint_dim_;
              float kx = raw_data[base * stride + i];
              float ky = raw_data[(base + 1) * stride + i];
              kpts.emplace_back(kx, ky);
          }
          keypoints_list.push_back(kpts);
      }
  }

  // 3. NMS 非极大值抑制，去除重叠框
  if (profile_enabled_) {
      last_frame_profile_.decode_ms = ticks_to_ms(decode_start);
      last_frame_profile_.decoded_anchors = boxes.size();
  }
  std::vector<int> nms_indices;
  const int64 nms_start = profile_enabled_ ? cv::getTickCount() : 0;
  cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_iou_threshold_, nms_indices);
  if (profile_enabled_) {
      last_frame_profile_.nms_ms = ticks_to_ms(nms_start);
      last_frame_profile_.nms_kept = nms_indices.size();
  }

  // 4. 根据 NMS 结果构建最终输出
  const int64 result_build_start = profile_enabled_ ? cv::getTickCount() : 0;
  for (int idx : nms_indices) {
      Object obj;
      obj.rect = boxes[idx];
      obj.prob = confidences[idx];
      obj.label = class_ids[idx];
      obj.kpt = keypoints_list[idx];
      results.push_back(obj);
  }

  // 5. 按置信度降序排序（最高置信度在前）
  std::sort(results.begin(), results.end(),
      [](const Object& a, const Object& b) { return a.prob > b.prob; });
  if (profile_enabled_) {
      last_frame_profile_.result_build_ms = ticks_to_ms(result_build_start);
  }

  return results;
}

// ==================== 获取单个最佳候选框（小符） ====================

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(
  cv::Mat & image,
  bool draw_results)
{
  const int64 start = cv::getTickCount();
  beginFrameProfile();

  // 检查图像是否为空
  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::vector<Object>();
  }

  // 1. 预处理图像
  // preprocess 返回 Letterbox 的缩放比例 r = min(640/h, 640/w)
  float r = preprocess(image);
  const int64 preprocess_end = cv::getTickCount();

  // 2. 执行推理
  if (!inference()) {
      tools::logger()->error("Inference failed!");
      return std::vector<Object>();
  }
  const int64 inference_end = cv::getTickCount();

  // 3. 后处理，获取所有检测结果
  // 此时 objects 中的坐标是在 640x640 画布上的
  std::vector<Object> all_objects = postprocess(1.0f);
  const int64 postprocess_end = cv::getTickCount();

  // --- 坐标还原 (Undo Letterbox) ---
  // 计算 padding 偏移量
  // preprocess 逻辑中：new_shape = image_shape * r, 居中填充
  int unpad_w = std::round(image.cols * r);
  int unpad_h = std::round(image.rows * r);
  float dw = (input_w_ - unpad_w) / 2.0f;
  float dh = (input_h_ - unpad_h) / 2.0f;

  for (auto &obj : all_objects) {
      // 还原 Rect: (x_net - dw) / r
      obj.rect.x = (obj.rect.x - dw) / r;
      obj.rect.y = (obj.rect.y - dh) / r;
      obj.rect.width /= r;
      obj.rect.height /= r;

      // 边界保护：防止坐标超出原图范围
      obj.rect.x = std::max(0.0f, obj.rect.x);
      obj.rect.y = std::max(0.0f, obj.rect.y);
      obj.rect.width = std::min(obj.rect.width, (float)image.cols - obj.rect.x);
      obj.rect.height = std::min(obj.rect.height, (float)image.rows - obj.rect.y);

      // 还原 Keypoints
      for (auto &pt : obj.kpt) {
          pt.x = (pt.x - dw) / r;
          pt.y = (pt.y - dh) / r;
      }
  }
  const int64 restore_end = cv::getTickCount();

  // 4. 只保留置信度最高的一个结果
  // 0:红色待击打, 1:红色已击打, 2:蓝色待击打, 3:蓝色已击打
  std::vector<Object> target_objects;

  for (const auto& obj : all_objects) {
      // 保留所有“待击打”状态的扇叶。
      if (is_pending_label_for_schema(obj.label, num_classes_)) {
          target_objects.push_back(obj);
      }
  }

  // 5. 选取最佳目标
  if (!target_objects.empty()) {
      // 只保留第一个（置信度最高的）
      target_objects.resize(1);

      // 调试保存 (低置信度时保存图片)
      const float debug_upper = std::min(0.99f, conf_threshold_ + 0.1f);
      if (
          debug_save_low_conf_enabled() &&
          target_objects[0].prob < debug_upper &&
          target_objects[0].prob > conf_threshold_) {
          save("debug_low_conf_" + std::to_string(start), image);
      }
  }
  const int64 filter_end = cv::getTickCount();

  // 6. 在图像上绘制结果
  // 默认只画最终参与决策的目标；如需看全部候选，请设置 BUFF_SHOW_ALL_CANDIDATES=1。
  float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  if (draw_results) {
      const auto & objects_to_draw = draw_all_candidates_enabled() ? all_objects : target_objects;
      drawResults(image, objects_to_draw, 1.0f / t);
  }
  const int64 draw_end = cv::getTickCount();

  if (profile_enabled_) {
      last_frame_profile_.total_ms = ticks_to_ms(start, draw_end);
      last_frame_profile_.restore_ms = ticks_to_ms(postprocess_end, restore_end);
      last_frame_profile_.filter_ms = ticks_to_ms(restore_end, filter_end);
      last_frame_profile_.draw_ms = ticks_to_ms(filter_end, draw_end);
      last_frame_profile_.returned_candidates = target_objects.size();
      last_frame_profile_.completed = true;
      recordProfile();
  }

  // 返回筛选后的结果（如果没有待击打目标，这里返回的就是空向量）
  return target_objects;
}

// ==================== 获取置信度最高的两个候选框（用于大符预测等） ====================

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_twocandidatebox(
  cv::Mat & image,
  bool draw_results)
{
  const int64 start = cv::getTickCount();
  beginFrameProfile();

  // 检查图像是否为空
  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::vector<Object>();
  }

  // 1. 预处理图像
  float r = preprocess(image);
  const int64 preprocess_end = cv::getTickCount();

  // 2. 执行推理
  if (!inference()) {
      tools::logger()->error("Inference failed!");
      return std::vector<Object>();
  }
  const int64 inference_end = cv::getTickCount();

  // 3. 后处理，获取所有检测结果 (已按置信度降序排序)
  std::vector<Object> all_objects = postprocess(1.0f);
  const int64 postprocess_end = cv::getTickCount();

  // --- 坐标还原 (Undo Letterbox) ---
  int unpad_w = std::round(image.cols * r);
  int unpad_h = std::round(image.rows * r);
  float dw = (input_w_ - unpad_w) / 2.0f;
  float dh = (input_h_ - unpad_h) / 2.0f;

  for (auto &obj : all_objects) {
      // 还原 Rect
      obj.rect.x = (obj.rect.x - dw) / r;
      obj.rect.y = (obj.rect.y - dh) / r;
      obj.rect.width /= r;
      obj.rect.height /= r;

      // 边界保护
      obj.rect.x = std::max(0.0f, obj.rect.x);
      obj.rect.y = std::max(0.0f, obj.rect.y);
      obj.rect.width = std::min(obj.rect.width, (float)image.cols - obj.rect.x);
      obj.rect.height = std::min(obj.rect.height, (float)image.rows - obj.rect.y);

      // 还原 Keypoints
      for (auto &pt : obj.kpt) {
          pt.x = (pt.x - dw) / r;
          pt.y = (pt.y - dh) / r;
      }
  }
  const int64 restore_end = cv::getTickCount();

  // 4. 筛选逻辑：大符主要依赖“待击打”目标。
  // 2026 规则中大符每组随机点亮 2 块；击中第一块后该灯臂改变灯效，
  // 实际画面里通常会直接熄灭、检测不到。这里返回待击打目标，并兼容性附带少量
  // 已击打目标；上层以“上一锁定目标消失”为主要切换依据。
  // 0:红色待击打, 1:红色已击打, 2:蓝色待击打, 3:蓝色已击打
  std::vector<Object> target_objects;
  std::vector<Object> hit_objects;

  for (const auto& obj : all_objects) {
      if (is_pending_label_for_schema(obj.label, num_classes_)) {
          target_objects.push_back(obj);
      } else if (is_context_label_for_schema(obj.label, num_classes_)) {
          hit_objects.push_back(obj);
      }
  }

  // 5. 选取前两个待击打目标，并附带前两个已击打目标作为上下文。
  std::vector<Object> selected_objects;
  if (!target_objects.empty()) {
      if (target_objects.size() > 2) {
          target_objects.resize(2);
      }
      selected_objects.insert(selected_objects.end(), target_objects.begin(), target_objects.end());

      // 调试保存 (检查第一候选的置信度)
      const float debug_upper = std::min(0.99f, conf_threshold_ + 0.1f);
      if (
          debug_save_low_conf_enabled() &&
          target_objects[0].prob < debug_upper &&
          target_objects[0].prob > conf_threshold_) {
          save("debug_low_conf_two_" + std::to_string(start), image);
      }
  }
  if (hit_objects.size() > 2) {
      hit_objects.resize(2);
  }
  selected_objects.insert(selected_objects.end(), hit_objects.begin(), hit_objects.end());
  const int64 filter_end = cv::getTickCount();

  // 6. 在图像上绘制结果
  // 默认只画最终参与决策的目标；如需看全部候选，请设置 BUFF_SHOW_ALL_CANDIDATES=1。
  float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  if (draw_results) {
      const auto & objects_to_draw = draw_all_candidates_enabled() ? all_objects : selected_objects;
      drawResults(image, objects_to_draw, 1.0f / t);
  }
  const int64 draw_end = cv::getTickCount();

  if (profile_enabled_) {
      last_frame_profile_.total_ms = ticks_to_ms(start, draw_end);
      last_frame_profile_.restore_ms = ticks_to_ms(postprocess_end, restore_end);
      last_frame_profile_.filter_ms = ticks_to_ms(restore_end, filter_end);
      last_frame_profile_.draw_ms = ticks_to_ms(filter_end, draw_end);
      last_frame_profile_.returned_candidates = selected_objects.size();
      last_frame_profile_.completed = true;
      recordProfile();
  }

  // 返回筛选后的结果：待击打目标在前，已击打目标仅作为可选上下文。
  return selected_objects;
}



// ==================== 绘制检测结果 ====================

void YOLO11_BUFF::drawResults(cv::Mat & image, const std::vector<Object> & objects, float fps)
{
  for (const auto& obj : objects) {
      // 根据 label 设置颜色和文字
      // 0:红待, 1:红已, 2:蓝待, 3:蓝已
      cv::Scalar color;
      std::string type_str;

      type_str = buff_label_name_for_schema(obj.label, num_classes_);
      if (is_pending_label_for_schema(obj.label, num_classes_)) {
          color = cv::Scalar(0, 255, 0);
      } else if (is_context_label_for_schema(obj.label, num_classes_)) {
          color = cv::Scalar(0, 0, 255);
      } else {
          color = cv::Scalar(255, 255, 255);
      }

      if (false) switch(obj.label) {
          case 0: color = cv::Scalar(0, 255, 0); type_str = "R-Target"; break; // 绿
          case 1: color = cv::Scalar(0, 0, 255); type_str = "R-Hit"; break;    // 红
          case 2: color = cv::Scalar(0, 255, 0); type_str = "B-Target"; break; // 绿
          case 3: color = cv::Scalar(0, 0, 255); type_str = "B-Hit"; break;    // 红
          default: color = cv::Scalar(255, 255, 255); type_str = "Unknown"; break;
      }

      // 1. 绘制边界框
      cv::rectangle(image, obj.rect, color, 2);

      // 2. 绘制标签（类别 + 置信度）
      std::string label = cv::format("%s: %.2f", type_str.c_str(), obj.prob);
      cv::putText(image, label, cv::Point(obj.rect.x, obj.rect.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

      // 3. 绘制关键点和扇叶四边形，方便直接观察网络输出
      for (size_t i = 0; i < obj.kpt.size(); ++i) {
          const bool is_r_point = (i == 4);
          const cv::Scalar kpt_color = is_r_point ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
          const int radius = is_r_point ? 5 : 4;
          cv::circle(image, obj.kpt[i], radius, kpt_color, -1);

          if (!is_r_point) {
              cv::putText(
                  image,
                  std::to_string(i),
                  obj.kpt[i] + cv::Point2f(4.0f, -4.0f),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.45,
                  kpt_color,
                  1);
          }
      }

      if (obj.kpt.size() >= 4) {
          std::vector<cv::Point> blade_polygon;
          blade_polygon.reserve(4);
          for (int i = 0; i < 4; ++i) {
              blade_polygon.emplace_back(cv::Point(
                  static_cast<int>(std::round(obj.kpt[i].x)),
                  static_cast<int>(std::round(obj.kpt[i].y))));
          }
          const std::vector<std::vector<cv::Point>> polygons = {blade_polygon};
          cv::polylines(image, polygons, true, color, 2);
      }

      if (obj.kpt.size() >= 5) {
          const cv::Point2f blade_center = (obj.kpt[0] + obj.kpt[1] + obj.kpt[2] + obj.kpt[3]) / 4.0f;
          cv::circle(image, blade_center, 4, cv::Scalar(255, 255, 255), -1);
          cv::line(image, obj.kpt[4], blade_center, cv::Scalar(255, 255, 255), 1);
      }
  }

  // 4. 绘制 FPS
  cv::putText(image, cv::format("YOLO FPS: %.2f", fps), cv::Point(20, std::max(40, image.rows - 20)),
              cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2);
}


// ==================== 打印引擎信息 ====================

void YOLO11_BUFF::printEngineInfo()
{
  tools::logger()->info("=== TensorRT Engine Bindings ===");
  int numBindings = engine_->getNbIOTensors();
  for (int i = 0; i < numBindings; ++i) {
      const char* name = engine_->getIOTensorName(i);
      bool isInput = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
      nvinfer1::Dims dims = engine_->getTensorShape(name);
      nvinfer1::DataType dtype = engine_->getTensorDataType(name);

      std::string dimStr = "[";
      for (int j = 0; j < dims.nbDims; ++j) {
          dimStr += std::to_string(dims.d[j]);
          if (j < dims.nbDims - 1) dimStr += " x ";
      }
      dimStr += "]";

      tools::logger()->info("Binding {}: '{}', IO: {}, Dims: {}, DataType: {}",
          i, name, (isInput ? "Input" : "Output"), dimStr, static_cast<int>(dtype));
  }
      tools::logger()->info(
          "=== Buff Output: [1, {}, {}] (classes={}, keypoints_start={}, keypoint_dim={}) ===",
          output_rows_,
          output_cols_,
          num_classes_,
          keypoint_offset_,
          keypoint_dim_);
}

// ==================== 保存图像 ====================

void YOLO11_BUFF::save(const std::string & name, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) {
    std::filesystem::create_directories(saveDir);
  }
  const std::filesystem::path savePath = saveDir / (name + ".jpg");
  cv::imwrite(savePath.string(), image);
}

void YOLO11_BUFF::recordProfile()
{
  if (!profile_enabled_) {
    return;
  }

  const auto update = [](double value, double & sum, double & max_value) {
      sum += value;
      max_value = std::max(max_value, value);
  };
  profile_.frames++;
  update(last_frame_profile_.total_ms, profile_.sum.total_ms, profile_.max.total_ms);
  update(
      last_frame_profile_.cpu_preprocess_ms,
      profile_.sum.cpu_preprocess_ms,
      profile_.max.cpu_preprocess_ms);
  update(last_frame_profile_.h2d_api_ms, profile_.sum.h2d_api_ms, profile_.max.h2d_api_ms);
  update(
      last_frame_profile_.preprocess_launch_api_ms,
      profile_.sum.preprocess_launch_api_ms,
      profile_.max.preprocess_launch_api_ms);
  update(
      last_frame_profile_.enqueue_api_ms,
      profile_.sum.enqueue_api_ms,
      profile_.max.enqueue_api_ms);
  update(last_frame_profile_.d2h_api_ms, profile_.sum.d2h_api_ms, profile_.max.d2h_api_ms);
  update(
      last_frame_profile_.sync_wait_ms,
      profile_.sum.sync_wait_ms,
      profile_.max.sync_wait_ms);
  update(
      last_frame_profile_.raw_h2d_gpu_ms,
      profile_.sum.raw_h2d_gpu_ms,
      profile_.max.raw_h2d_gpu_ms);
  update(
      last_frame_profile_.model_input_h2d_gpu_ms,
      profile_.sum.model_input_h2d_gpu_ms,
      profile_.max.model_input_h2d_gpu_ms);
  update(
      last_frame_profile_.preprocess_kernel_gpu_ms,
      profile_.sum.preprocess_kernel_gpu_ms,
      profile_.max.preprocess_kernel_gpu_ms);
  update(last_frame_profile_.trt_gpu_ms, profile_.sum.trt_gpu_ms, profile_.max.trt_gpu_ms);
  update(last_frame_profile_.d2h_gpu_ms, profile_.sum.d2h_gpu_ms, profile_.max.d2h_gpu_ms);
  update(
      last_frame_profile_.output_convert_ms,
      profile_.sum.output_convert_ms,
      profile_.max.output_convert_ms);
  update(last_frame_profile_.decode_ms, profile_.sum.decode_ms, profile_.max.decode_ms);
  update(last_frame_profile_.nms_ms, profile_.sum.nms_ms, profile_.max.nms_ms);
  update(
      last_frame_profile_.result_build_ms,
      profile_.sum.result_build_ms,
      profile_.max.result_build_ms);
  update(last_frame_profile_.restore_ms, profile_.sum.restore_ms, profile_.max.restore_ms);
  update(last_frame_profile_.filter_ms, profile_.sum.filter_ms, profile_.max.filter_ms);
  update(last_frame_profile_.draw_ms, profile_.sum.draw_ms, profile_.max.draw_ms);
}

void YOLO11_BUFF::printProfile() const
{
  if (!profile_enabled_ || profile_.frames == 0) {
    return;
  }

  const double frames = static_cast<double>(profile_.frames);
  tools::logger()->info(
    "[YOLO11_BUFF_PROFILE] frames={} pinned={} avg_ms(total={:.3f}, raw_h2d_gpu={:.3f}, "
    "model_h2d_gpu={:.3f}, preprocess_kernel_gpu={:.3f}, trt_gpu={:.3f}, d2h_gpu={:.3f}, "
    "sync_wait_cpu={:.3f}, decode_cpu={:.3f}, nms_cpu={:.3f}, result_build_cpu={:.3f}, "
    "restore_cpu={:.3f}, filter_cpu={:.3f}, draw_cpu={:.3f}) max_ms(total={:.3f}, "
    "raw_h2d_gpu={:.3f}, model_h2d_gpu={:.3f}, preprocess_kernel_gpu={:.3f}, trt_gpu={:.3f}, "
    "d2h_gpu={:.3f}, sync_wait_cpu={:.3f}, decode_cpu={:.3f}, nms_cpu={:.3f}, "
    "result_build_cpu={:.3f}, restore_cpu={:.3f}, filter_cpu={:.3f}, draw_cpu={:.3f})",
    profile_.frames,
    output_host_memory_pinned_,
    profile_.sum.total_ms / frames,
    profile_.sum.raw_h2d_gpu_ms / frames,
    profile_.sum.model_input_h2d_gpu_ms / frames,
    profile_.sum.preprocess_kernel_gpu_ms / frames,
    profile_.sum.trt_gpu_ms / frames,
    profile_.sum.d2h_gpu_ms / frames,
    profile_.sum.sync_wait_ms / frames,
    profile_.sum.decode_ms / frames,
    profile_.sum.nms_ms / frames,
    profile_.sum.result_build_ms / frames,
    profile_.sum.restore_ms / frames,
    profile_.sum.filter_ms / frames,
    profile_.sum.draw_ms / frames,
    profile_.max.total_ms,
    profile_.max.raw_h2d_gpu_ms,
    profile_.max.model_input_h2d_gpu_ms,
    profile_.max.preprocess_kernel_gpu_ms,
    profile_.max.trt_gpu_ms,
    profile_.max.d2h_gpu_ms,
    profile_.max.sync_wait_ms,
    profile_.max.decode_ms,
    profile_.max.nms_ms,
    profile_.max.result_build_ms,
    profile_.max.restore_ms,
    profile_.max.filter_ms,
    profile_.max.draw_ms);
}
}
