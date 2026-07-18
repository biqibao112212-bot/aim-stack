/**
 * @file yolo11_buff.hpp
 * @brief YOLO11 能量机关关键点检测模块（TensorRT 推理）
 */

#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// TensorRT 头文件
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "tools/logger.hpp"

namespace auto_buff
{

const std::vector<std::string> class_names = {"buff", "r"};

/**
 * @class TRTLogger
 * @brief TensorRT 日志类，用于接收 TensorRT 内部日志输出
 */
class TRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

/**
 * @class YOLO11_BUFF
 * @brief YOLO11 能量机关检测器，使用 TensorRT 进行推理
 */
class YOLO11_BUFF
{
public:
  enum class OutputHostMemoryMode
  {
    PinnedPreferred,
    PageableOnly,
    InjectPinnedAllocationFailure
  };

  struct Options
  {
    OutputHostMemoryMode output_host_memory_mode = OutputHostMemoryMode::PinnedPreferred;
    bool enable_profiling = false;
  };

  struct FrameProfile
  {
    bool enabled = false;
    bool completed = false;
    bool gpu_events_valid = false;
    bool gpu_preprocess_used = false;
    bool gpu_preprocess_fallback = false;
    bool output_host_pinned = false;
    int h2d_status = 0;
    int d2h_status = 0;
    int sync_status = 0;
    size_t h2d_bytes = 0;
    size_t d2h_bytes = 0;
    size_t decoded_anchors = 0;
    size_t nms_kept = 0;
    size_t returned_candidates = 0;
    double total_ms = 0.0;
    double cpu_preprocess_ms = 0.0;
    double h2d_api_ms = 0.0;
    double preprocess_launch_api_ms = 0.0;
    double enqueue_api_ms = 0.0;
    double d2h_api_ms = 0.0;
    double sync_wait_ms = 0.0;
    double raw_h2d_gpu_ms = 0.0;
    double model_input_h2d_gpu_ms = 0.0;
    double preprocess_kernel_gpu_ms = 0.0;
    double trt_gpu_ms = 0.0;
    double d2h_gpu_ms = 0.0;
    double output_convert_ms = 0.0;
    double decode_ms = 0.0;
    double nms_ms = 0.0;
    double result_build_ms = 0.0;
    double restore_ms = 0.0;
    double filter_ms = 0.0;
    double draw_ms = 0.0;
  };

  /**
   * @struct Object
   * @brief 检测结果结构体
   */
  struct Object
  {
    cv::Rect_<float> rect;          // 目标框
    int label;                       // 类别标签
    float prob;                      // 置信度
    std::vector<cv::Point2f> kpt;   // 关键点
  };

  /**
   * @brief 构造函数：加载配置并初始化 TensorRT 引擎
   * @param config YAML 配置文件路径
   */
  explicit YOLO11_BUFF(const std::string & config);
  YOLO11_BUFF(const std::string & config, const Options & options);

  /**
   * @brief 析构函数：释放 TensorRT 资源和 CUDA 内存
   */
  ~YOLO11_BUFF();

  // 禁用拷贝构造和赋值（因为持有 GPU 资源）
  YOLO11_BUFF(const YOLO11_BUFF &) = delete;
  YOLO11_BUFF & operator=(const YOLO11_BUFF &) = delete;

  /**
   * @brief 检测单个目标（取置信度最高的框）
   * @param image 输入图像（会被绘制调试信息）
   * @return 检测结果列表（最多一个元素）
   */
  std::vector<Object> get_onecandidatebox(cv::Mat & image, bool draw_results = true);

  /**
   * @brief 检测两个个目标（取置信度最高的两个框）
   * @param image 输入图像（会被绘制调试信息）
   * @return 检测结果列表（最多两个个元素）
   */
  std::vector<Object> get_twocandidatebox(cv::Mat & image, bool draw_results = true);

  int num_classes() const { return num_classes_; }
  bool output_host_memory_pinned() const noexcept { return output_host_memory_pinned_; }
  bool output_host_memory_fallback_used() const noexcept
  {
    return output_host_memory_fallback_used_;
  }
  size_t output_transfer_bytes() const noexcept { return output_size_; }
  FrameProfile last_frame_profile() const noexcept { return last_frame_profile_; }

private:
  // ==================== TensorRT 核心对象 ====================
  TRTLogger logger_;                                        // TensorRT 日志器
  std::unique_ptr<nvinfer1::IRuntime> runtime_;             // TensorRT 运行时
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;           // TensorRT 推理引擎
  std::unique_ptr<nvinfer1::IExecutionContext> context_;    // 推理执行上下文

  // ==================== CUDA 资源 ====================
  cudaStream_t stream_;             // CUDA 流，用于异步操作
  void * device_input_;             // GPU 输入缓冲区
  void * device_output_;            // GPU 输出缓冲区
  void * device_raw_image_ = nullptr;  // GPU 原始 BGR 图像缓存（GPU 预处理输入）
  size_t device_raw_image_capacity_ = 0;
  float * host_output_;             // CPU 输出缓冲区（用于后处理）
  __half * host_output_half_;       // 当输出为 FP16 时的临时 CPU 缓冲区

  std::string input_tensor_name_;   // 输入张量名
  std::string output_tensor_name_;  // 输出张量名
  nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kFLOAT;
  std::vector<float> host_input_float_cache_;
  std::vector<__half> host_input_half_cache_;  // 输入为 FP16 时的 host 临时缓存
  cv::Mat letterbox_cache_;
  OutputHostMemoryMode output_host_memory_mode_ = OutputHostMemoryMode::PinnedPreferred;
  bool output_host_memory_pinned_ = false;
  bool output_host_memory_fallback_used_ = false;
  bool gpu_preprocess_enabled_ = true;
  bool gpu_timing_valid_ = false;
  bool profile_events_ready_ = false;
  cudaEvent_t h2d_start_event_ = nullptr;
  cudaEvent_t h2d_end_event_ = nullptr;
  cudaEvent_t preprocess_kernel_end_event_ = nullptr;
  cudaEvent_t trt_end_event_ = nullptr;
  cudaEvent_t d2h_end_event_ = nullptr;
  FrameProfile last_frame_profile_;

  // ==================== 模型参数 ====================
  int input_h_ = 640;               // 输入图像高度
  int input_w_ = 640;               // 输入图像宽度
  int input_c_ = 3;                 // 输入通道数
  int output_rows_=18;                 // 输出行数
  int output_cols_=8400;                 // 输出列数 (如 8400)
  int num_classes_ = 4;                // 类别数：旧模型4类，新三态模型6类
  int keypoint_dim_ = 2;               // 每个关键点维度：旧模型为 x,y；新模型为 x,y,visible
  int keypoint_offset_ = 8;            // 4 box + num_classes
  size_t input_size_;               // 输入张量字节大小
  size_t output_size_;              // 输出张量字节大小
  size_t input_elem_count_ = 0;     // 输入元素数量（N*C*H*W）
  size_t output_elem_count_ = 0;    // 输出元素数量

  const int NUM_POINTS = 5;         // 关键点数量
  float conf_threshold_ = 0.5f;     // 置信度阈值（优先从配置文件读取）
  float nms_iou_threshold_ = 0.4f;  // NMS IoU 阈值

  struct ProfileStats
  {
    size_t frames = 0;
    FrameProfile sum;
    FrameProfile max;
  };

  bool profile_enabled_ = false;
  ProfileStats profile_;

  // ==================== 私有方法 ====================

  /** @brief 加载 TensorRT 引擎文件（.engine） */
  bool loadEngine(const std::string & engine_path);

  /** @brief 分配 GPU 和 CPU 缓冲区 */
  bool allocateBuffers();

  bool allocateOutputTransferBuffer();
  void releaseOutputTransferBuffer();

  /** @brief 释放所有缓冲区 */
  void releaseBuffers();

  bool createProfileEvents();
  void destroyProfileEvents();
  void beginFrameProfile();
  void finalizeGpuProfile();

  /**
   * @brief 图像预处理：letterbox 变换 + 归一化 + BGR2RGB + NHWC2NCHW
   * @param input_image 输入图像
   * @return 缩放因子（用于后处理还原坐标）
   */
  float preprocess(const cv::Mat & input_image);
  float preprocessCpu(const cv::Mat & input_image);
  float preprocessGpu(const cv::Mat & input_image);
  bool ensureDeviceRawCapacity(size_t required_bytes);

  /** @brief 执行 TensorRT 推理 */
  bool inference();

  /**
   * @brief 后处理：解析输出张量 + NMS 去重
   * @param scale_factor 预处理时的缩放因子
   * @return 检测结果列表（按置信度降序排序）
   *
   * 处理流程：
   * 1. 收集所有置信度 > conf_threshold_ 的检测框
   * 2. 使用 NMS 去除 IoU > nms_iou_threshold_ 的重叠框
   * 3. 按置信度降序排序
   */
  std::vector<Object> postprocess(float scale_factor);

  /** @brief 在图像上绘制检测结果（边界框、关键点、FPS） */
  void drawResults(cv::Mat & image, const std::vector<Object> & objects, float fps);

  /** @brief 打印引擎信息（输入输出绑定信息） */
  void printEngineInfo();

  /** @brief 保存图像到文件 */
  void save(const std::string & name, const cv::Mat & image);

  void recordProfile();
  void printProfile() const;

  static constexpr size_t kMaxPinnedOutputBytes = 8u * 1024u * 1024u;
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__YOLO11_BUFF_HPP
