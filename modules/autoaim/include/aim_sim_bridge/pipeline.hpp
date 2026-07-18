#pragma once

#include "aim_sim_bridge/aim_types.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <iterator>
#include <memory>
#include <cstdint>
#include <mutex>
#include <string>

namespace aim_sim_bridge
{

struct AimPipelineCounters
{
    std::uint64_t submitted = 0;
    std::uint64_t submission_rate_limited = 0;
    std::uint64_t max_detector_submit_hz = 0;
    std::uint64_t detector_active_slots = 0;
    std::uint64_t overwritten = 0;
    std::uint64_t detector_completed = 0;
    std::uint64_t solve_completed = 0;
    std::uint64_t tracker_aim_completed = 0;
    std::uint64_t final_completed = 0;
    std::uint64_t delivered_completed = 0;
    std::uint64_t submission_cpu_ns = 0;
    std::uint64_t detector_latency_ns = 0;
    std::uint64_t solve_cpu_ns = 0;
    std::uint64_t tracker_aim_cpu_ns = 0;
    std::uint64_t finalize_cpu_ns = 0;
    std::uint64_t pipeline_latency_ns = 0;
    bool detector_profile_enabled = false;
    std::uint64_t detector_profile_timing_event_count = 0;
    std::uint64_t detector_profile_sample_stride = 0;
    std::uint64_t detector_profile_completed = 0;
    std::uint64_t detector_profile_errors = 0;
    std::uint64_t detector_raw_queue_wait_ns = 0;
    std::uint64_t detector_slot_wait_ns = 0;
    std::uint64_t detector_launcher_host_ns = 0;
    std::uint64_t detector_pending_order_wait_ns = 0;
    std::uint64_t detector_event_wait_ns = 0;
    std::uint64_t detector_fp_convert_ns = 0;
    std::uint64_t detector_postprocess_nms_ns = 0;
    std::uint64_t detector_completion_bookkeeping_ns = 0;
    std::uint64_t detector_profile_wall_ns = 0;
    std::uint64_t detector_gpu_h2d_ns = 0;
    std::uint64_t detector_gpu_preprocess_ns = 0;
    std::uint64_t detector_gpu_trt_ns = 0;
    std::uint64_t detector_gpu_d2h_ns = 0;
    std::uint64_t detector_gpu_stream_ns = 0;
};

namespace detail
{
inline constexpr std::size_t kCompletionQueueCapacity = 8;

class DetectorSubmissionRateGate
{
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    explicit DetectorSubmissionRateGate(std::uint64_t max_hz = 0)
        : max_hz_(max_hz),
          period_(max_hz == 0
                      ? Clock::duration::zero()
                      : std::chrono::duration_cast<Clock::duration>(
                            std::chrono::duration<double>(1.0 /
                                                          static_cast<double>(max_hz))))
    {
    }

    bool admit(TimePoint now)
    {
        if (max_hz_ == 0) return true;
        // A frequency above the clock's representable resolution rounds to a
        // zero period, for which rate limiting cannot be expressed.
        if (period_ <= Clock::duration::zero()) return true;
        if (!initialized_) {
            initialized_ = true;
            const auto remaining = TimePoint::max() - now;
            next_due_ = remaining < period_ ? TimePoint::max() : now + period_;
            return true;
        }
        if (next_due_ == TimePoint::max()) return false;
        if (now < next_due_) return false;

        // Preserve the long-term phase across jitter and long gaps. Skipped
        // deadlines are discarded: this call admits exactly one current input
        // and advances the deadline to the first period strictly after now.
        const auto lateness = now - next_due_;
        const auto remaining = TimePoint::max() - next_due_;
        const auto elapsed_periods = lateness / period_;
        const auto max_periods = remaining / period_;
        if (elapsed_periods >= max_periods) {
            next_due_ = TimePoint::max();
        } else {
            next_due_ += period_ * (elapsed_periods + 1);
        }
        return true;
    }

    std::uint64_t maxHz() const { return max_hz_; }

private:
    std::uint64_t max_hz_ = 0;
    Clock::duration period_{};
    bool initialized_ = false;
    TimePoint next_due_{};
};

// Called immediately after appending the new profile when a latest-drop raw
// queue reports overflow. Launched/waiting profiles form an arbitrary prefix;
// the tail is exactly [old raw suffix of raw_capacity, new profile].
template <typename Container>
bool eraseOldestOverwrittenRawSuffixItem(
    Container& submitted, std::size_t raw_capacity)
{
    if (raw_capacity == 0 || submitted.size() <= raw_capacity) return false;
    submitted.erase(std::prev(submitted.end(), raw_capacity + 1));
    return true;
}

template <typename Container, typename Predicate>
bool takeFirstMatchingAndErasePrefix(
    Container& submitted, Predicate matches,
    typename Container::value_type& matched)
{
    const auto it = std::find_if(submitted.begin(), submitted.end(), matches);
    if (it == submitted.end()) return false;
    matched = *it;
    submitted.erase(submitted.begin(), std::next(it));
    return true;
}

class AimPipelineStageTelemetry
{
public:
    void recordSubmission(std::uint64_t elapsed_ns, bool overwritten)
    {
        submission_cpu_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
        submitted_.fetch_add(1, std::memory_order_relaxed);
        if (overwritten) overwritten_.fetch_add(1, std::memory_order_relaxed);
    }

    void recordSubmissionRateLimited()
    {
        submission_rate_limited_.fetch_add(1, std::memory_order_relaxed);
    }

    void recordDetectorCompletion(std::uint64_t elapsed_ns)
    {
        detector_latency_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
        detector_completed_.fetch_add(1, std::memory_order_relaxed);
    }

    void recordSolveCompletion(std::uint64_t elapsed_ns)
    {
        solve_cpu_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
        solve_completed_.fetch_add(1, std::memory_order_relaxed);
    }

    void recordTrackerAimCompletion(std::uint64_t elapsed_ns)
    {
        tracker_aim_cpu_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
        tracker_aim_completed_.fetch_add(1, std::memory_order_relaxed);
    }

    void recordFinalCompletion(
        std::uint64_t finalize_ns, std::uint64_t pipeline_latency_ns)
    {
        finalize_cpu_ns_.fetch_add(finalize_ns, std::memory_order_relaxed);
        pipeline_latency_ns_.fetch_add(pipeline_latency_ns, std::memory_order_relaxed);
        final_completed_.fetch_add(1, std::memory_order_relaxed);
    }

    void recordDeliveredCompletion()
    {
        delivered_completed_.fetch_add(1, std::memory_order_relaxed);
    }

    AimPipelineCounters snapshot() const
    {
        AimPipelineCounters out;
        out.submitted = submitted_.load(std::memory_order_relaxed);
        out.submission_rate_limited =
            submission_rate_limited_.load(std::memory_order_relaxed);
        out.overwritten = overwritten_.load(std::memory_order_relaxed);
        out.detector_completed = detector_completed_.load(std::memory_order_relaxed);
        out.solve_completed = solve_completed_.load(std::memory_order_relaxed);
        out.tracker_aim_completed =
            tracker_aim_completed_.load(std::memory_order_relaxed);
        out.final_completed = final_completed_.load(std::memory_order_relaxed);
        out.delivered_completed = delivered_completed_.load(std::memory_order_relaxed);
        out.submission_cpu_ns = submission_cpu_ns_.load(std::memory_order_relaxed);
        out.detector_latency_ns = detector_latency_ns_.load(std::memory_order_relaxed);
        out.solve_cpu_ns = solve_cpu_ns_.load(std::memory_order_relaxed);
        out.tracker_aim_cpu_ns = tracker_aim_cpu_ns_.load(std::memory_order_relaxed);
        out.finalize_cpu_ns = finalize_cpu_ns_.load(std::memory_order_relaxed);
        out.pipeline_latency_ns = pipeline_latency_ns_.load(std::memory_order_relaxed);
        return out;
    }

private:
    std::atomic<std::uint64_t> submitted_{0};
    std::atomic<std::uint64_t> submission_rate_limited_{0};
    std::atomic<std::uint64_t> overwritten_{0};
    std::atomic<std::uint64_t> detector_completed_{0};
    std::atomic<std::uint64_t> solve_completed_{0};
    std::atomic<std::uint64_t> tracker_aim_completed_{0};
    std::atomic<std::uint64_t> final_completed_{0};
    std::atomic<std::uint64_t> delivered_completed_{0};
    std::atomic<std::uint64_t> submission_cpu_ns_{0};
    std::atomic<std::uint64_t> detector_latency_ns_{0};
    std::atomic<std::uint64_t> solve_cpu_ns_{0};
    std::atomic<std::uint64_t> tracker_aim_cpu_ns_{0};
    std::atomic<std::uint64_t> finalize_cpu_ns_{0};
    std::atomic<std::uint64_t> pipeline_latency_ns_{0};
};

template <typename T, std::size_t Capacity>
class BoundedCompletionQueue
{
public:
    static_assert(Capacity > 0, "completion queue capacity must be positive");

    // Preserve the freshest bounded suffix. The consumer still observes that
    // suffix in production order, so command identities never regress.
    bool publish(T value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool overwritten = values_.size() == Capacity;
        if (overwritten) values_.pop_front();
        values_.push_back(std::move(value));
        return overwritten;
    }

    bool tryTake(T& value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (values_.empty()) return false;
        value = std::move(values_.front());
        values_.pop_front();
        return true;
    }

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return values_.size();
    }

private:
    mutable std::mutex mutex_;
    std::deque<T> values_;
};
}  // namespace detail

class IAimPipeline
{
public:
    virtual ~IAimPipeline() = default;

    virtual AimCommand process(const SimFrame& frame) = 0;
    virtual std::string backendName() const = 0;
    virtual AimPipelineCounters counters() const { return {}; }
};

std::unique_ptr<IAimPipeline> createAimPipeline(const AimBridgeConfig& config);

}  // namespace aim_sim_bridge
