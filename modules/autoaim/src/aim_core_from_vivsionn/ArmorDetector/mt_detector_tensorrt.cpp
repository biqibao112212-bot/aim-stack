#include "mt_detector_tensorrt.hpp"
#include "preprocess_kernel.hpp" 
#include "runtime_paths.h"

#include <cuda_fp16.h> 
#include <cstdint>     
#include <yaml-cpp/yaml.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <npp.h>
#include <fstream>
#include <vector>
#include <iostream> 
#include <cmath>
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <sstream>
#include <string_view>

#define TRT_LOG(msg) std::cout << "[TRT INFO] " << msg << std::endl
#define TRT_ERR(msg) std::cerr << "[TRT ERROR] " << msg << std::endl

class TrtLogger : public nvinfer1::ILogger
{
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TRT WARN] " << msg << std::endl;
    }
  }
} gLogger;

namespace rm {

namespace {

constexpr int kArmorColorBlue = 0;
constexpr int kArmorColorRed = 1;
constexpr int kArmorColorGray = 2;
constexpr int kArmorColorPurple = 3;
constexpr int kArmorColorUnknown = -1;

int decodeArmorColorIdx(int color_idx)
{
  if (color_idx >= kArmorColorBlue && color_idx <= kArmorColorPurple) {
    return color_idx;
  }
  return kArmorColorUnknown;
}

bool isNeutralArmorColor(int color)
{
  return color == kArmorColorGray || color == kArmorColorPurple;
}

bool isEnemyOrNeutralArmorColor(int armor_color, int target_enemy_color)
{
  return armor_color == target_enemy_color || isNeutralArmorColor(armor_color);
}

bool detectorStageProfilingEnabled()
{
  const char* value = std::getenv("AIM_SIM_PROFILE_DETECTOR_STAGES");
  if (value == nullptr) return false;
  const std::string text(value);
  return text == "1" || text == "true" || text == "TRUE" || text == "yes" ||
         text == "YES" || text == "on" || text == "ON";
}

int detectorActiveSlots()
{
  const char* value = std::getenv("AIM_SIM_DETECTOR_ACTIVE_SLOTS");
  if (value == nullptr) return K_PIPELINE_DEPTH;

  const std::string_view text(value);
  int parsed = 0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), parsed, 10);
  if (text.empty() || result.ec != std::errc{} ||
      result.ptr != text.data() + text.size() || parsed < 1 ||
      parsed > K_PIPELINE_DEPTH) {
    throw std::runtime_error(
        "AIM_SIM_DETECTOR_ACTIVE_SLOTS must be an integer in [1, 3]");
  }
  return parsed;
}

std::uint64_t elapsedNs(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end)
{
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

} // namespace

static size_t get_binding_size_bytes(
  const nvinfer1::ICudaEngine* engine, const char* name, nvinfer1::DataType dtype)
{
  auto dims = engine->getTensorShape(name);
  size_t size = 1;
  for (int j = 0; j < dims.nbDims; ++j) {
    size *= dims.d[j];
  }
  int element_size = 4; 
  if (dtype == nvinfer1::DataType::kHALF) element_size = 2;
  else if (dtype == nvinfer1::DataType::kINT8) element_size = 1;
  return size * element_size;
}

static std::string dims_to_string(const nvinfer1::Dims& d) {
    std::string s = "(";
    for (int i = 0; i < d.nbDims; ++i) {
        s += std::to_string(d.d[i]);
        if (i < d.nbDims - 1) s += ", ";
    }
    s += ")";
    return s;
}

static std::string type_to_string(nvinfer1::DataType t) {
    if (t == nvinfer1::DataType::kFLOAT) return "FLOAT (FP32)";
    if (t == nvinfer1::DataType::kHALF) return "HALF (FP16)";
    if (t == nvinfer1::DataType::kINT8) return "INT8";
    return "UNKNOWN";
}

MultiThreadDetectorTRT::MultiThreadDetectorTRT(const std::string& config_path, bool debug)
:   debug_(debug),
    running_(true),
    active_slots_(detectorActiveSlots()),
    profile_stages_(detectorStageProfilingEnabled()),
    queue_raw_(K_PIPELINE_DEPTH),
    free_slots_(K_PIPELINE_DEPTH),
    pending_slots_(K_PIPELINE_DEPTH)
{
    TRT_LOG("Initializing Detector...");

    std::vector<std::string> model_candidates;
    if (!config_path.empty()) {
        model_candidates.push_back(
            rm::runtime_paths::resolveExistingPath(config_path).string());
    }
    model_candidates.push_back(
        rm::runtime_paths::repoPath("src/ArmorDetector/model/0526.engine").string());

    std::string model_path;
    for (const auto& candidate : model_candidates) {
        std::ifstream file(candidate, std::ios::binary);
        if (file.good()) {
            model_path = candidate;
            break;
        }
    }
    if (model_path.empty()) {
        std::ostringstream oss;
        oss << "Failed to find TRT engine. Checked:";
        for (const auto& candidate : model_candidates) {
            oss << "\n  - " << candidate;
        }
        throw std::runtime_error(oss.str());
    }
    
    min_confidence_ = 0.5;
    score_threshold_ = 0.5;
    nms_threshold_ = 0.45;
    
    roi_ = cv::Rect(0, 0, 1440, 1080);
    offset_ = cv::Point2f(roi_.x, roi_.y);
    use_roi_ = false;

    cudaSetDevice(0);
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Failed to load TRT engine: " + model_path);
    std::vector<char> trtModelStream(std::istreambuf_iterator<char>(file), {});
    file.close();

    initLibNvInferPlugins(&gLogger, "");
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size());
    if(!engine_) throw std::runtime_error("Deserialize Engine Failed");

    input_name_ = engine_->getIOTensorName(0);
    output_name_ = engine_->getIOTensorName(1);
    
    input_dims_ = engine_->getTensorShape(input_name_.c_str());
    output_dims_ = engine_->getTensorShape(output_name_.c_str());
    
    auto input_type = engine_->getTensorDataType(input_name_.c_str());
    auto output_type = engine_->getTensorDataType(output_name_.c_str());

    // 鎵撳嵃璋冭瘯淇℃伅锛屽府鍔╁畾浣嶉棶棰?
    std::cout << "=== Model Info ===" << std::endl;
    std::cout << "Input:  " << input_name_ << " | Shape: " << dims_to_string(input_dims_) << " | Type: " << type_to_string(input_type) << std::endl;
    std::cout << "Output: " << output_name_ << " | Shape: " << dims_to_string(output_dims_) << " | Type: " << type_to_string(output_type) << std::endl;
    std::cout << "==================" << std::endl;

    input_size_ = get_binding_size_bytes(engine_, input_name_.c_str(), input_type);
    output_size_ = get_binding_size_bytes(engine_, output_name_.c_str(), output_type);

    streams_.resize(K_PIPELINE_DEPTH);
    contexts_.resize(K_PIPELINE_DEPTH);
    device_buffers_.resize(K_PIPELINE_DEPTH);
    host_buffers_.resize(K_PIPELINE_DEPTH);
    events_.resize(K_PIPELINE_DEPTH);
    job_metadata_.resize(K_PIPELINE_DEPTH);

    // 鏄惧瓨鍒嗛厤
    const size_t max_raw_img_size = 3000 * 2000 * 3; 
    const size_t letterbox_img_size = 640 * 640 * 3;

    for (int i = 0; i < K_PIPELINE_DEPTH; ++i) {
        cudaStreamCreate(&streams_[i]);
        cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming);
        contexts_[i] = engine_->createExecutionContext();

        cudaMalloc(&device_buffers_[i].d_raw_img, max_raw_img_size);
        cudaMalloc(&device_buffers_[i].d_letterbox_img, letterbox_img_size);
        
        // 瀹夊叏鑰冭檻锛氬鏋?input_size 璁＄畻鍑烘潵寰堝皬锛堟瘮濡?FP16锛夛紝浣嗛澶勭悊 kernel 鍐欎簡 FP32锛屼細瀵艰嚧婧㈠嚭
        // 寤鸿鍒嗛厤绋嶅ぇ涓€鐐逛互闃蹭竾涓€锛屼絾杩欓噷鍏堟寜鏍囧噯鍒嗛厤
        cudaMalloc(&device_buffers_[i].d_tensor_in, input_size_);
        cudaMalloc(&device_buffers_[i].d_tensor_out, output_size_);

        cudaHostAlloc(&host_buffers_[i].h_pinned_input, max_raw_img_size, cudaHostAllocDefault);
        cudaHostAlloc(&host_buffers_[i].h_pinned_output, output_size_, cudaHostAllocDefault);

        if (i < active_slots_) free_slots_.push(i);
    }

    if (profile_stages_) {
        timing_events_.resize(K_PIPELINE_DEPTH);
        try {
            for (auto& timing : timing_events_) {
                cudaEvent_t* handles[] = {
                    &timing.gpu_start, &timing.h2d_done,
                    &timing.preprocess_done, &timing.trt_done,
                    &timing.d2h_done};
                for (cudaEvent_t* handle : handles) {
                    const cudaError_t status = cudaEventCreate(handle);
                    if (status != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("CUDA detector timing event creation failed: ") +
                            cudaGetErrorString(status));
                    }
                    ++profile_timing_event_count_;
                }
            }
        } catch (...) {
            for (auto& timing : timing_events_) {
                cudaEvent_t* handles[] = {
                    &timing.gpu_start, &timing.h2d_done,
                    &timing.preprocess_done, &timing.trt_done,
                    &timing.d2h_done};
                for (cudaEvent_t* handle : handles) {
                    if (*handle != nullptr) {
                        cudaEventDestroy(*handle);
                        *handle = nullptr;
                    }
                }
            }
            profile_timing_event_count_ = 0;
            throw;
        }
    }

    launcher_thread_ = std::thread(&MultiThreadDetectorTRT::launcher_loop, this);
}

MultiThreadDetectorTRT::~MultiThreadDetectorTRT()
{
    stop();
    for (auto& s : streams_) if(s) cudaStreamSynchronize(s);
    release_trt_objects();
}

void MultiThreadDetectorTRT::stop()
{
    running_ = false;
    queue_raw_.stop();
    free_slots_.stop();
    pending_slots_.stop();
    if (launcher_thread_.joinable()) launcher_thread_.join();
}

void MultiThreadDetectorTRT::release_trt_objects()
{
    for (auto& timing : timing_events_) {
        cudaEvent_t* handles[] = {
            &timing.gpu_start, &timing.h2d_done,
            &timing.preprocess_done, &timing.trt_done, &timing.d2h_done};
        for (cudaEvent_t* handle : handles) {
            if (*handle != nullptr) {
                cudaEventDestroy(*handle);
                *handle = nullptr;
            }
        }
    }
    profile_timing_event_count_ = 0;
    for (int i = 0; i < streams_.size(); ++i) {
        if (contexts_[i]) { delete contexts_[i]; contexts_[i] = nullptr; }
        if (events_[i])   { cudaEventDestroy(events_[i]); events_[i] = nullptr; }
        if (streams_[i])  { cudaStreamDestroy(streams_[i]); streams_[i] = nullptr; }
    }
    for (auto& buf : device_buffers_) {
        cudaFree(buf.d_raw_img); 
        cudaFree(buf.d_letterbox_img);
        cudaFree(buf.d_tensor_in); 
        cudaFree(buf.d_tensor_out);
    }
    for (auto& buf : host_buffers_) {
        cudaFreeHost(buf.h_pinned_input); 
        cudaFreeHost(buf.h_pinned_output);
    }
    if (engine_)  { delete engine_; engine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
}

bool MultiThreadDetectorTRT::push(rm::Frame frame, std::chrono::steady_clock::time_point t)
{
    if (frame.srcImg.empty()) return false;
    
    cv::Rect current_roi = roi_;
    cv::Point2f current_offset = {0, 0};
    bool current_use_roi = use_roi_;

    if (current_use_roi) {
        if (current_roi.width == -1) current_roi.width = frame.srcImg.cols;
        if (current_roi.height == -1) current_roi.height = frame.srcImg.rows;
        current_offset = offset_;
    }

    RawJob job;
    job.timestamp = t;
    job.current_offset = current_offset;
    job.current_use_roi = current_use_roi;
    job.roi = current_roi;

    // Move the only full-frame image buffer across threads. We keep a shallow
    // alias in `context_frame.srcImg` for legacy consumers until the downstream
    // metadata-only refactor lands.
    job.img = std::move(frame.srcImg);
    frame.debugImg.release();
    frame.yoloImg.release();
    frame.srcImg = job.img;
    job.context_frame = std::move(frame);

    return queue_raw_.push(std::move(job));
}

void MultiThreadDetectorTRT::launcher_loop()
{
    cudaSetDevice(0); 
    static uint64_t job_seq = 0;

    while (running_.load())
    {
        RawJob job = queue_raw_.pop();
        if (!running_.load()) break; 
        ++job_seq;
        const bool profile_sampled =
            profile_stages_ &&
            (job_seq % K_DETECTOR_PROFILE_SAMPLE_STRIDE == 0);
        const auto profile_raw_dequeued = profile_sampled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        
        int slot_idx = free_slots_.pop();
        if (!running_.load()) break; 
        const auto profile_slot_acquired = profile_sampled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        
        auto& d_bufs = device_buffers_[slot_idx];
        auto& h_bufs = host_buffers_[slot_idx];
        auto context = contexts_[slot_idx];
        cudaStream_t stream = streams_[slot_idx];
        npp_stream_ctx_.hStream = stream;

        cv::Rect& roi = job.roi;
        if (!job.current_use_roi) roi = cv::Rect(0, 0, job.img.cols, job.img.rows);

        if (job.img.empty()) {
            std::cerr << "[TRT WARN] empty input image in launcher_loop, seq=" << job_seq << std::endl;
            free_slots_.push(slot_idx);
            continue;
        }
        
        // 绠€鍗曠殑 ROI 杈圭晫淇濇姢
        roi = roi & cv::Rect(0, 0, job.img.cols, job.img.rows);
        if (roi.area() <= 0) {
             std::cerr << "[TRT WARN] invalid ROI in launcher_loop, seq=" << job_seq
                       << " roi=" << roi.width << "x" << roi.height
                       << " img=" << job.img.cols << "x" << job.img.rows << std::endl;
             free_slots_.push(slot_idx);
             continue;
        }

        auto scale = std::min(640.0 / roi.width, 640.0 / roi.height);
        auto h = static_cast<int>(roi.height * scale);
        auto w = static_cast<int>(roi.width * scale);
        
        job_metadata_[slot_idx].timestamp = job.timestamp;
        job_metadata_[slot_idx].scale = scale;
        job_metadata_[slot_idx].offset = job.current_offset;
        job_metadata_[slot_idx].use_roi = job.current_use_roi;
        job_metadata_[slot_idx].context_frame = std::move(job.context_frame);
        job_metadata_[slot_idx].profile_sampled = profile_sampled;
        if (profile_sampled) {
            job_metadata_[slot_idx].profile_raw_dequeued = profile_raw_dequeued;
            job_metadata_[slot_idx].profile_slot_acquired = profile_slot_acquired;
        }
        if (debug_) {
            job_metadata_[slot_idx].original_img = job.img;
        } else {
            job_metadata_[slot_idx].original_img.release();
        }

        cv::Mat h_pinned_mat(roi.height, roi.width, CV_8UC3, h_bufs.h_pinned_input);
        job.img(roi).copyTo(h_pinned_mat);

        if (profile_sampled) {
            cudaEventRecord(timing_events_[slot_idx].gpu_start, stream);
        }
        cudaMemcpyAsync(d_bufs.d_raw_img, h_bufs.h_pinned_input,
            h_pinned_mat.total() * h_pinned_mat.elemSize(), cudaMemcpyHostToDevice, stream);
        if (profile_sampled) {
            cudaEventRecord(timing_events_[slot_idx].h2d_done, stream);
        }
            
        cudaMemsetAsync(d_bufs.d_letterbox_img, 0, 640 * 640 * 3, stream);
        
        NppiSize src_sz = {roi.width, roi.height}, dst_sz = {640, 640};
        NppiRect src_rc = {0, 0, roi.width, roi.height};
        NppiRect dst_rc = {0, 0, w, h}; 

        nppiResize_8u_C3R_Ctx((const Npp8u*)d_bufs.d_raw_img, h_pinned_mat.step, src_sz, src_rc,
            (Npp8u*)d_bufs.d_letterbox_img, 640 * 3, dst_sz, dst_rc, NPPI_INTER_LINEAR, npp_stream_ctx_);

        // 杩欓噷鐨勫己鍒惰浆鎹㈤渶瑕佹牸澶栧皬蹇冿紝濡傛灉 kernel 鍜?engine 绫诲瀷涓嶄竴鑷达紝鍙兘鍑洪敊
        launch_preprocess_kernel((const uint8_t*)d_bufs.d_letterbox_img, (uint16_t*)d_bufs.d_tensor_in, 640, 640, stream);
        if (profile_sampled) {
            cudaEventRecord(timing_events_[slot_idx].preprocess_done, stream);
        }
        
        context->setTensorAddress(input_name_.c_str(), d_bufs.d_tensor_in);
        context->setTensorAddress(output_name_.c_str(), d_bufs.d_tensor_out);
        if(!context->enqueueV3(stream)) {
            TRT_ERR("enqueueV3 failed!");
        }
        if (profile_sampled) {
            cudaEventRecord(timing_events_[slot_idx].trt_done, stream);
        }
        
        cudaMemcpyAsync(h_bufs.h_pinned_output, d_bufs.d_tensor_out, output_size_, cudaMemcpyDeviceToHost, stream);
        if (profile_sampled) {
            cudaEventRecord(timing_events_[slot_idx].d2h_done, stream);
        }
        cudaEventRecord(events_[slot_idx], stream);
        if (profile_sampled) {
            job_metadata_[slot_idx].profile_gpu_queued = std::chrono::steady_clock::now();
        }
        pending_slots_.push(slot_idx);
    }
}

// 鎵撳紑 mt_detector_tensorrt.cpp锛屾浛鎹㈠師鏉ョ殑 pop_impl 鍑芥暟

template <typename T>
T MultiThreadDetectorTRT::pop_impl()
{
    int slot_idx = 0;
    if (!pending_slots_.wait_pop(slot_idx)) return T{};
    auto& metadata = job_metadata_[slot_idx];
    const bool profile_sampled = profile_stages_ && metadata.profile_sampled;
    const auto profile_pending_dequeued = profile_sampled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    cudaEventSynchronize(events_[slot_idx]);
    const auto profile_event_wait_done = profile_sampled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    
    void* h_output_buffer = host_buffers_[slot_idx].h_pinned_output;

    int dim1 = output_dims_.d[1];
    int dim2 = output_dims_.d[2];
    auto dtype = engine_->getTensorDataType(output_name_.c_str());

    cv::Mat output_tensor;

    // 澶勭悊 FP16 鍜?FP32锛屼互鍙婂姩鎬佸舰鐘跺垽鏂?
    if (dtype == nvinfer1::DataType::kHALF) {
        if (dim1 > dim2) {
            cv::Mat raw_output_half(dim1, dim2, CV_16F, h_output_buffer);
            raw_output_half.convertTo(output_tensor, CV_32F);
        } else {
            cv::Mat temp(dim1, dim2, CV_16F, h_output_buffer);
            cv::Mat transposed = temp.t(); 
            transposed.convertTo(output_tensor, CV_32F);
        }
    } else { // 榛樿涓?FLOAT
        if (dim1 > dim2) {
            output_tensor = cv::Mat(dim1, dim2, CV_32F, h_output_buffer);
        } else {
            cv::Mat temp(dim1, dim2, CV_32F, h_output_buffer);
            output_tensor = temp.t(); 
        }
    }

    // 璋冪敤 postprocess 鑾峰彇璇嗗埆鍒扮殑瑁呯敳鏉?
    const auto profile_fp_convert_done = profile_sampled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    std::vector<rm::ArmorForDetect> armors;
    if (!output_tensor.empty()) {
        armors = postprocess(metadata.scale, output_tensor, metadata.original_img, metadata.offset, metadata.use_roi);
    }
    const auto profile_postprocess_done = profile_sampled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};

    if (attack_all_colors_.load(std::memory_order_relaxed))
    {
        int self_team = metadata.context_frame.fb.self_team;
        int target_enemy_color =
            (self_team == rm::FeedBackData::SELF_BLUE) ? kArmorColorRed : kArmorColorBlue;
        
        for (auto& armor : armors) {
            armor.color = target_enemy_color;
            armor.left_light.color = target_enemy_color;
            armor.right_light.color = target_enemy_color;
        }
    }

    const uint8_t self_team = metadata.context_frame.fb.self_team;
    int target_enemy_color = -1;
    if (self_team == rm::FeedBackData::SELF_BLUE) {
        target_enemy_color = kArmorColorRed;
    } else if (self_team == rm::FeedBackData::SELF_RED) {
        target_enemy_color = kArmorColorBlue;
    }

    if (target_enemy_color >= 0) {
        armors.erase(
            std::remove_if(
                armors.begin(), armors.end(),
                [target_enemy_color](const rm::ArmorForDetect& armor) {
                    return !isEnemyOrNeutralArmorColor(armor.color, target_enemy_color);
                }),
            armors.end());
    }

    const auto timestamp = metadata.timestamp;
    rm::Frame context_frame = std::move(metadata.context_frame);
    cv::Mat original_img = std::move(metadata.original_img);

    if (profile_sampled) {
        const auto profile_completion_done = std::chrono::steady_clock::now();
        profile_raw_queue_wait_ns_.fetch_add(
            elapsedNs(metadata.timestamp, metadata.profile_raw_dequeued),
            std::memory_order_relaxed);
        profile_slot_wait_ns_.fetch_add(
            elapsedNs(metadata.profile_raw_dequeued, metadata.profile_slot_acquired),
            std::memory_order_relaxed);
        profile_launcher_host_ns_.fetch_add(
            elapsedNs(metadata.profile_slot_acquired, metadata.profile_gpu_queued),
            std::memory_order_relaxed);
        profile_pending_order_wait_ns_.fetch_add(
            elapsedNs(metadata.profile_gpu_queued, profile_pending_dequeued),
            std::memory_order_relaxed);
        profile_event_wait_ns_.fetch_add(
            elapsedNs(profile_pending_dequeued, profile_event_wait_done),
            std::memory_order_relaxed);
        profile_fp_convert_ns_.fetch_add(
            elapsedNs(profile_event_wait_done, profile_fp_convert_done),
            std::memory_order_relaxed);
        profile_postprocess_nms_ns_.fetch_add(
            elapsedNs(profile_fp_convert_done, profile_postprocess_done),
            std::memory_order_relaxed);
        profile_completion_bookkeeping_ns_.fetch_add(
            elapsedNs(profile_postprocess_done, profile_completion_done),
            std::memory_order_relaxed);
        profile_wall_ns_.fetch_add(
            elapsedNs(metadata.timestamp, profile_completion_done),
            std::memory_order_relaxed);

        const auto& timing = timing_events_[slot_idx];
        float h2d_ms = 0.0F;
        float preprocess_ms = 0.0F;
        float trt_ms = 0.0F;
        float d2h_ms = 0.0F;
        float stream_ms = 0.0F;
        const cudaError_t gpu_statuses[] = {
            cudaEventElapsedTime(&h2d_ms, timing.gpu_start, timing.h2d_done),
            cudaEventElapsedTime(
                &preprocess_ms, timing.h2d_done, timing.preprocess_done),
            cudaEventElapsedTime(
                &trt_ms, timing.preprocess_done, timing.trt_done),
            cudaEventElapsedTime(&d2h_ms, timing.trt_done, timing.d2h_done),
            cudaEventElapsedTime(&stream_ms, timing.gpu_start, timing.d2h_done)};
        const bool gpu_ok = std::all_of(
            std::begin(gpu_statuses), std::end(gpu_statuses),
            [](cudaError_t status) { return status == cudaSuccess; });
        if (gpu_ok) {
            const auto msToNs = [](float ms) {
                return static_cast<std::uint64_t>(std::max(0.0F, ms) * 1.0e6F);
            };
            profile_gpu_h2d_ns_.fetch_add(msToNs(h2d_ms), std::memory_order_relaxed);
            profile_gpu_preprocess_ns_.fetch_add(
                msToNs(preprocess_ms), std::memory_order_relaxed);
            profile_gpu_trt_ns_.fetch_add(msToNs(trt_ms), std::memory_order_relaxed);
            profile_gpu_d2h_ns_.fetch_add(msToNs(d2h_ms), std::memory_order_relaxed);
            profile_gpu_stream_ns_.fetch_add(
                msToNs(stream_ms), std::memory_order_relaxed);
        } else {
            profile_errors_.fetch_add(1, std::memory_order_relaxed);
        }
        profile_completed_.fetch_add(1, std::memory_order_relaxed);
    }

    free_slots_.push(slot_idx);

    if constexpr (std::is_same_v<T, std::tuple<std::vector<rm::ArmorForDetect>, std::chrono::steady_clock::time_point, rm::Frame>>)
        return {std::move(armors), timestamp, std::move(context_frame)};
    else
        return {std::move(original_img), std::move(armors), timestamp, std::move(context_frame)};
}

DetectorTimingSnapshot MultiThreadDetectorTRT::timingSnapshot() const
{
    DetectorTimingSnapshot out;
    out.active_slots = static_cast<std::uint64_t>(active_slots_);
    out.enabled = profile_stages_;
    out.timing_event_count = profile_timing_event_count_;
    out.sample_stride = K_DETECTOR_PROFILE_SAMPLE_STRIDE;
    if (!profile_stages_) return out;
    out.completed = profile_completed_.load(std::memory_order_relaxed);
    out.errors = profile_errors_.load(std::memory_order_relaxed);
    out.raw_queue_wait_ns = profile_raw_queue_wait_ns_.load(std::memory_order_relaxed);
    out.slot_wait_ns = profile_slot_wait_ns_.load(std::memory_order_relaxed);
    out.launcher_host_ns = profile_launcher_host_ns_.load(std::memory_order_relaxed);
    out.pending_order_wait_ns =
        profile_pending_order_wait_ns_.load(std::memory_order_relaxed);
    out.event_wait_ns = profile_event_wait_ns_.load(std::memory_order_relaxed);
    out.fp_convert_ns = profile_fp_convert_ns_.load(std::memory_order_relaxed);
    out.postprocess_nms_ns =
        profile_postprocess_nms_ns_.load(std::memory_order_relaxed);
    out.completion_bookkeeping_ns =
        profile_completion_bookkeeping_ns_.load(std::memory_order_relaxed);
    out.wall_ns = profile_wall_ns_.load(std::memory_order_relaxed);
    out.gpu_h2d_ns = profile_gpu_h2d_ns_.load(std::memory_order_relaxed);
    out.gpu_preprocess_ns =
        profile_gpu_preprocess_ns_.load(std::memory_order_relaxed);
    out.gpu_trt_ns = profile_gpu_trt_ns_.load(std::memory_order_relaxed);
    out.gpu_d2h_ns = profile_gpu_d2h_ns_.load(std::memory_order_relaxed);
    out.gpu_stream_ns = profile_gpu_stream_ns_.load(std::memory_order_relaxed);
    return out;
}

std::tuple<std::vector<rm::ArmorForDetect>, std::chrono::steady_clock::time_point, rm::Frame>
MultiThreadDetectorTRT::pop() {
    return pop_impl<std::tuple<std::vector<rm::ArmorForDetect>, std::chrono::steady_clock::time_point, rm::Frame>>();
}

std::tuple<cv::Mat, std::vector<rm::ArmorForDetect>, std::chrono::steady_clock::time_point, rm::Frame>
MultiThreadDetectorTRT::debug_pop() {
    return pop_impl<std::tuple<cv::Mat, std::vector<rm::ArmorForDetect>, std::chrono::steady_clock::time_point, rm::Frame>>();
}

std::vector<rm::ArmorForDetect> MultiThreadDetectorTRT::postprocess(
  double scale, cv::Mat& output, const cv::Mat& bgr_img,
  const cv::Point2f& offset, bool use_roi)
{
    std::vector<rm::ArmorForDetect> armors;
    
    // [宕╂簝淇鍏抽敭鐐筣 妫€鏌ヨ緭鍑洪€氶亾鏁?
    // 浠ｇ爜涓闂簡 row[21]锛屽鏋?output.cols < 22锛屽繀宕?
    if (output.cols < 22) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "[CRITICAL ERROR] Model output channels (" << output.cols 
                      << ") is less than required (22). Check your model!" << std::endl;
            warned = true;
        }
        return armors; // 鐩存帴杩斿洖锛岄伩鍏嶅穿婧?
    }

    std::vector<int> class_ids; 
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> kps_list;

    for (int r = 0; r < output.rows; r++) {
        float* row = output.ptr<float>(r);
        
        // 璁块棶瓒婄晫楂樺嵄鍖猴細纭繚 output.cols 瓒冲澶?
        // sigmoid(x) < 0.5 iff x < 0 for finite x. Preserve the original
        // sigmoid path for accepted rows, non-finite values, and any future
        // threshold other than the exact production value.
        const float objectness_logit = row[8];
        if (score_threshold_ == 0.5f && std::isfinite(objectness_logit) &&
            objectness_logit < 0.0f) {
            continue;
        }
        float score = (float)sigmoid((double)objectness_logit);
        if (score < score_threshold_) continue;

        float max_color_score = -1.0f;
        int color_idx = -1;
        for(int c = 9; c <= 12; ++c) {
            if(row[c] > max_color_score) {
                max_color_score = row[c];
                color_idx = c - 9;
            }
        }

        float max_num_score = -1.0f;
        int num_idx = -1;
        for(int c = 13; c <= 21; ++c) {
            if(row[c] > max_num_score) {
                max_num_score = row[c];
                num_idx = c - 13;
            }
        }

        std::vector<cv::Point2f> kps;
        float px[] = {row[0], row[6], row[4], row[2]};
        float py[] = {row[1], row[7], row[5], row[3]};

        float min_x = 1e5, min_y = 1e5, max_x = -1e5, max_y = -1e5;
        for(int k=0; k<4; ++k) {
            float x = px[k] / scale;
            float y = py[k] / scale;
            if(use_roi) { x += offset.x; y += offset.y; }
            kps.push_back({x, y});

            if(x < min_x) min_x = x; if(x > max_x) max_x = x;
            if(y < min_y) min_y = y; if(y > max_y) max_y = y;
        }

        float width = max_x - min_x;
        float height = max_y - min_y;
        if(width < 1 || height < 1) continue;

        const int col_id = decodeArmorColorIdx(color_idx);
        if (col_id == kArmorColorUnknown) continue;

        int num_id = 1;
        if (num_idx == 0) num_id = 7; 
        else if (num_idx >= 1 && num_idx <= 5) num_id = num_idx;
        else if (num_idx == 6) num_id = 6;
        else if (num_idx == 7 || num_idx == 8) num_id = 8; 

        boxes.push_back(cv::Rect(min_x, min_y, width, height));
        confidences.push_back(score);
        class_ids.push_back(num_id | (col_id << 8)); 
        kps_list.push_back(kps);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

    for (int idx : indices) {
        auto& kps = kps_list[idx];
        
        std::sort(kps.begin(), kps.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });
        if (kps[0].y > kps[1].y) std::swap(kps[0], kps[1]); 
        if (kps[2].y > kps[3].y) std::swap(kps[2], kps[3]); 

        rm::Light leftLight, rightLight;
        leftLight.top = kps[0];    leftLight.bottom = kps[1];
        rightLight.top = kps[2];   rightLight.bottom = kps[3];
        
        leftLight.center = (leftLight.top + leftLight.bottom) / 2.0;
        leftLight.length = cv::norm(leftLight.top - leftLight.bottom);
        leftLight.tilt_angle = std::atan2(std::abs(leftLight.top.x - leftLight.bottom.x), std::abs(leftLight.top.y - leftLight.bottom.y)) * 180 / CV_PI;

        rightLight.center = (rightLight.top + rightLight.bottom) / 2.0;
        rightLight.length = cv::norm(rightLight.top - rightLight.bottom);
        rightLight.tilt_angle = std::atan2(std::abs(rightLight.top.x - rightLight.bottom.x), std::abs(rightLight.top.y - rightLight.bottom.y)) * 180 / CV_PI;

        rm::ArmorForDetect armor(leftLight, rightLight);
        armor.confidence = confidences[idx];
        
        int combined = class_ids[idx];
        armor.number = combined & 0xFF; 
        int color_code = (combined >> 8) & 0xFF; 
        
        armor.classfication_result = std::to_string(armor.number);
        armor.color = color_code;
        armor.left_light.color = color_code;
        armor.right_light.color = color_code;
        
        armor.vertex.clear();
        armor.vertex.push_back(leftLight.bottom); 
        armor.vertex.push_back(leftLight.top);
        armor.vertex.push_back(rightLight.top); 
        armor.vertex.push_back(rightLight.bottom);

        if (check_armor(armor)) armors.push_back(armor);
    }
    return armors;
}

void MultiThreadDetectorTRT::sort_keypoints(std::vector<cv::Point2f>& points) {}
bool MultiThreadDetectorTRT::check_armor(const rm::ArmorForDetect& armor) const {
    return armor.confidence >= min_confidence_;
}
double MultiThreadDetectorTRT::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

} // namespace rm
