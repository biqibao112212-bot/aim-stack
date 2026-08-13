#ifndef RM_MT_DETECTOR_HPP
#define RM_MT_DETECTOR_HPP

#include <vector>
#include <chrono>
#include <cstdint>
#include <tuple>
#include <list>
#include <thread>
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "GeneralDeclaration/generalDeclaration.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <npp.h>

// 纭繚寮曠敤浜嗘纭殑澶存枃浠?
#include "thread_safe_queue.hpp" 

namespace rm { 

// 娴佹按绾挎繁搴︿繚鎸佷负 3 鍗冲彲锛屽お灏忔棤娉曟帺鐩栦紶杈撳欢杩燂紝澶ぇ娴垂鏄惧瓨
constexpr int K_PIPELINE_DEPTH = 3; 
constexpr std::uint64_t K_DETECTOR_PROFILE_SAMPLE_STRIDE = 16;
static_assert(K_DETECTOR_PROFILE_SAMPLE_STRIDE % K_PIPELINE_DEPTH != 0,
              "detector profile samples must rotate across pipeline slots");

struct RawJob {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    cv::Point2f current_offset;
    bool current_use_roi;
    cv::Rect roi;
    
    rm::Frame context_frame;
};

struct JobMetadata {
    std::chrono::steady_clock::time_point timestamp;
    cv::Mat original_img; 
    cv::Point2f offset;
    bool use_roi;
    double scale;
    
    rm::Frame context_frame;
    bool profile_sampled = false;
    std::chrono::steady_clock::time_point profile_raw_dequeued;
    std::chrono::steady_clock::time_point profile_slot_acquired;
    std::chrono::steady_clock::time_point profile_gpu_queued;
};

struct DetectorTimingSnapshot {
    std::uint64_t active_slots = K_PIPELINE_DEPTH;
    bool enabled = false;
    std::uint64_t timing_event_count = 0;
    std::uint64_t sample_stride = 0;
    std::uint64_t completed = 0;
    std::uint64_t errors = 0;
    std::uint64_t raw_queue_wait_ns = 0;
    std::uint64_t slot_wait_ns = 0;
    std::uint64_t launcher_host_ns = 0;
    std::uint64_t pending_order_wait_ns = 0;
    std::uint64_t event_wait_ns = 0;
    std::uint64_t fp_convert_ns = 0;
    std::uint64_t postprocess_nms_ns = 0;
    std::uint64_t completion_bookkeeping_ns = 0;
    std::uint64_t wall_ns = 0;
    std::uint64_t gpu_h2d_ns = 0;
    std::uint64_t gpu_preprocess_ns = 0;
    std::uint64_t gpu_trt_ns = 0;
    std::uint64_t gpu_d2h_ns = 0;
    std::uint64_t gpu_stream_ns = 0;
};

struct DeviceBuffers {
    void* d_raw_img;        
    void* d_letterbox_img;  
    void* d_tensor_in;      
    void* d_tensor_out;     
};

struct HostBuffers {
    void* h_pinned_input;   
    void* h_pinned_output;
};

class MultiThreadDetectorTRT
{
public:
    MultiThreadDetectorTRT(const std::string& config_path, bool debug = false);
    ~MultiThreadDetectorTRT();

    void setAttackAllColors(bool enabled)
    {
        attack_all_colors_.store(enabled, std::memory_order_relaxed);
    }

    bool push(rm::Frame frame, std::chrono::steady_clock::time_point t);
    void stop();
    bool running() const { return running_.load(); }

    std::tuple<std::vector<rm::ArmorForDetect>, std::chrono::steady_clock::time_point, rm::Frame> pop();
    
    std::tuple<cv::Mat, std::vector<rm::ArmorForDetect>, std::chrono::steady_clock::time_point, rm::Frame> debug_pop();
    DetectorTimingSnapshot timingSnapshot() const;

private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    NppStreamContext npp_stream_ctx_{0};
    std::atomic<bool> attack_all_colors_{false};

    std::vector<cudaStream_t> streams_;
    std::vector<nvinfer1::IExecutionContext*> contexts_;
    std::vector<DeviceBuffers> device_buffers_;
    std::vector<HostBuffers> host_buffers_;
    std::vector<cudaEvent_t> events_;
    struct SlotTimingEvents {
        cudaEvent_t gpu_start = nullptr;
        cudaEvent_t h2d_done = nullptr;
        cudaEvent_t preprocess_done = nullptr;
        cudaEvent_t trt_done = nullptr;
        cudaEvent_t d2h_done = nullptr;
    };
    std::vector<SlotTimingEvents> timing_events_;
    std::vector<JobMetadata> job_metadata_;

    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    size_t input_size_;
    size_t output_size_;
    bool input_fp16_ = false;
    std::string input_name_;
    std::string output_name_;

    std::thread launcher_thread_;
    std::atomic<bool> running_;
    int active_slots_ = K_PIPELINE_DEPTH;
    bool profile_stages_ = false;
    std::uint64_t profile_timing_event_count_ = 0;
    std::atomic<std::uint64_t> profile_completed_{0};
    std::atomic<std::uint64_t> profile_errors_{0};
    std::atomic<std::uint64_t> profile_raw_queue_wait_ns_{0};
    std::atomic<std::uint64_t> profile_slot_wait_ns_{0};
    std::atomic<std::uint64_t> profile_launcher_host_ns_{0};
    std::atomic<std::uint64_t> profile_pending_order_wait_ns_{0};
    std::atomic<std::uint64_t> profile_event_wait_ns_{0};
    std::atomic<std::uint64_t> profile_fp_convert_ns_{0};
    std::atomic<std::uint64_t> profile_postprocess_nms_ns_{0};
    std::atomic<std::uint64_t> profile_completion_bookkeeping_ns_{0};
    std::atomic<std::uint64_t> profile_wall_ns_{0};
    std::atomic<std::uint64_t> profile_gpu_h2d_ns_{0};
    std::atomic<std::uint64_t> profile_gpu_preprocess_ns_{0};
    std::atomic<std::uint64_t> profile_gpu_trt_ns_{0};
    std::atomic<std::uint64_t> profile_gpu_d2h_ns_{0};
    std::atomic<std::uint64_t> profile_gpu_stream_ns_{0};

    // [淇敼鐐筣 鍏抽敭锛佸皢绗簩涓ā鏉垮弬鏁拌涓?true (PopWhenFull)
    // 杩欐牱褰?push 婊℃椂锛屼細鑷姩 pop 鎺夋渶鏃х殑锛屼繚璇侀槦鍒楁案杩滄槸鏈€鏂扮殑
    tools::ThreadSafeQueue<RawJob, true> queue_raw_;
    
    tools::ThreadSafeQueue<int> free_slots_;
    tools::ThreadSafeQueue<int> pending_slots_;

    bool debug_ = false;
    double min_confidence_ = 0.5;
    double score_threshold_ = 0.5;
    double nms_threshold_ = 0.45;
    cv::Rect roi_;
    cv::Point2f offset_;
    bool use_roi_ = false;
    
    void launcher_loop();
    
    template <typename T>
    T pop_impl();
    
    void release_trt_objects();

    std::vector<rm::ArmorForDetect> postprocess(
      double scale, cv::Mat& output, const cv::Mat& bgr_img,
      const cv::Point2f& offset, bool use_roi);
      
    bool check_armor(const rm::ArmorForDetect& armor) const;
    double sigmoid(double x);
    void sort_keypoints(std::vector<cv::Point2f>& points);
};

} // namespace rm

#endif // RM_MT_DETECTOR_HPP
