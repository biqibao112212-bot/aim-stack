#ifndef AUTO_BUFF_RUNE_PIPELINE_HPP
#define AUTO_BUFF_RUNE_PIPELINE_HPP

#include "buff_aimer.hpp"
#include "buff_detector.hpp"
#include "buff_solver.hpp"
#include "buff_tracker.hpp"
#include "generalDeclaration.h"
#include "latest_result_mailbox.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace auto_buff
{

// Strict startup selector parser used by the pipeline and CPU-only focused
// tests. Missing/empty/"0" preserves the legacy handoff; exact "1" selects
// inline ordered commit. Every other value is a construction error.
bool buffOrderedCommitEnabledFromValue(const char* value);
bool buffObservationSupersetEnabledFromValue(const char* value);
int buffProposalWorkersFromValue(const char* value, bool observation_profile);

struct BuffObservationSupersetCost
{
    bool enabled = false;
    bool proposal_ready = false;
    bool used_legacy_fallback = false;
    std::uint64_t proposal_total_ns = 0;
    std::uint64_t candidate_scan_ns = 0;
    std::uint64_t r_preprocess_ns = 0;
    std::uint64_t r_template_ns = 0;
    std::uint64_t r_contour_ns = 0;
    std::uint64_t pnp_extract_ns = 0;
    std::uint64_t pnp_reduce_ns = 0;
    std::uint64_t worker_context_setup_ns = 0;
    std::uint64_t fallback_solve_ns = 0;
    std::uint64_t ordered_commit_ns = 0;
    std::uint64_t union_roi_pixels = 0;
    std::uint64_t template_result_pixels = 0;
    std::uint64_t scratch_allocations = 0;
    std::uint64_t scratch_reuses = 0;
    std::uint64_t response_cells_scanned = 0;
    std::uint64_t support_rejected_cells = 0;
    std::uint64_t distance_tested_cells = 0;
    std::uint64_t contour_copy_bytes_avoided = 0;
    std::uint32_t retained_candidate_count = 0;
    std::uint32_t target_hypothesis_count = 0;
    std::uint32_t hit_context_count = 0;
    std::uint32_t anchor_count = 0;
    std::uint32_t scale_count = 0;
    std::uint32_t contour_count = 0;
    std::uint32_t pnp_proposal_count = 0;
    std::uint32_t cap_events = 0;
    std::uint32_t fallback_events = 0;
    std::uint32_t fallback_reason = 0;
    bool coverage_complete = false;
};

struct BuffRunePipelineOptions
{
    // Keep the historical behavior for every existing consumer.  Headless
    // consumers must opt out explicitly at construction time.
    bool emit_debug_artifacts = true;
    // Opt-in only for a bounded benchmark/replay.  Production consumers keep
    // this disabled so the pipeline does not retain an unbounded timing trace.
    bool collect_completion_samples = false;
};

struct BuffSolveCostSample
{
    std::uint64_t outer_total_ns = 0;
    std::uint64_t packet_setup_ns = 0;
    std::uint64_t solve_mutex_wait_ns = 0;
    std::uint64_t set_image_size_ns = 0;
    std::uint64_t set_pose_ns = 0;
    std::uint64_t detector_total_ns = 0;
    std::uint64_t state_snapshot_ns = 0;
    std::uint64_t accounted_ns = 0;
    std::uint64_t unaccounted_ns = 0;
    BuffObservationSupersetCost observation_superset;
    BuffDetectorCostSample detector;
};

struct BuffRuneCompletionSample
{
    std::uint64_t completion_sequence = 0;
    double yolo_ms = 0.0;
    double solve_ms = 0.0;
    double essential_track_aim_ms = 0.0;
    double debug_artifact_ms = 0.0;
    BuffSolveCostSample solve_cost;
};

struct BuffRunePipelineCounters
{
    bool ordered_commit_inline = false;
    bool track_thread_started = false;
    bool observation_superset_enabled = false;
    bool observation_superset_ready = false;
    std::uint64_t pushed_frames = 0;
    std::uint64_t input_queue_overwrites = 0;
    std::uint64_t yolo_completed = 0;
    std::uint64_t yolo_queue_overwrites = 0;
    std::uint64_t solve_completed = 0;
    std::uint64_t detection_queue_overwrites = 0;
    std::uint64_t ordered_commit_failures = 0;
    std::uint64_t observation_proposal_attempts = 0;
    std::uint64_t observation_proposal_fallbacks = 0;
    std::uint64_t observation_proposal_candidates = 0;
    std::uint64_t observation_pnp_proposals = 0;
    std::uint64_t observation_union_roi_pixels = 0;
    std::uint64_t observation_template_result_pixels = 0;
    std::uint64_t observation_cap_events = 0;
    std::uint64_t observation_identity_gaps = 0;
    std::uint64_t observation_identity_failures = 0;
    std::uint32_t proposal_worker_count = 0;
    std::uint64_t proposal_submitted = 0;
    std::uint64_t proposal_completed = 0;
    std::uint64_t proposal_committed = 0;
    std::uint32_t proposal_active_workers = 0;
    std::uint32_t proposal_max_active_workers = 0;
    std::uint32_t proposal_input_occupancy = 0;
    std::uint32_t proposal_input_max_occupancy = 0;
    std::uint32_t proposal_reorder_occupancy = 0;
    std::uint32_t proposal_reorder_max_occupancy = 0;
    std::uint32_t proposal_inflight = 0;
    std::uint32_t proposal_max_inflight = 0;
    std::uint64_t proposal_terminal_gaps = 0;
    std::uint64_t proposal_terminal_failures = 0;
    std::uint64_t proposal_cancelled = 0;
    std::uint64_t proposal_stale = 0;
    std::uint64_t proposal_head_wait_ns = 0;
    std::array<std::uint64_t, 4> proposal_worker_completed{};
    std::array<std::uint64_t, 4> proposal_worker_total_ns{};
    std::uint64_t essential_completed = 0;
    std::uint64_t published_results = 0;
    std::uint64_t popped_results = 0;
};

bool buffProposalCapacityAccepted(
    std::uint32_t worker_count, double aggregate_service_hz) noexcept;
bool buffProposalDrainAccepted(const BuffRunePipelineCounters& counters) noexcept;

struct BuffShotGateSnapshot
{
    bool requested = false;
    bool allowed = false;
    bool pending_detected = false;
    bool r_center_ok = false;
    bool pnp_ok = false;
    bool tracker_ok = false;
    bool gimbal_ok = false;
    bool stable_ok = false;
    int stable_frames = 0;
    int reason_code = 0;
    double yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
    double pitch_error_deg = std::numeric_limits<double>::quiet_NaN();
    double pnp_reproj_error_px = std::numeric_limits<double>::quiet_NaN();
    double pnp_model_center_error_px = std::numeric_limits<double>::quiet_NaN();
};

struct BuffRuneResult
{
    rm::Frame frame;
    std::optional<PowerRune> rune;
    rm::DebugHudSnapshot overlay;
    rm::ControlData control;
    std::chrono::steady_clock::time_point frame_timestamp{};
    std::chrono::steady_clock::time_point result_timestamp{};
    std::uint64_t completion_timestamp_ns = 0;
    std::uint64_t completion_sequence = 0;
    BuffRunePipelineCounters completion_counters;
    BuffSolveCostSample solve_cost;
    bool has_control = false;
    bool debug_artifacts_emitted = false;
    bool switch_deferred = false;
    bool target_switched = false;
    int selected_target_index = -1;
    int fps = 0;
    double infer_ms = 0.0;
    double yolo_ms = 0.0;
    double solve_ms = 0.0;
    double essential_track_aim_ms = 0.0;
    double debug_artifact_ms = 0.0;
    double track_aim_ms = 0.0;
    double pipeline_delay_ms = 0.0;
    double predict_ms = 0.0;
    double base_predict_ms = 0.0;
    double fly_time_ms = 0.0;
    BuffTracker::DebugSnapshot tracker_debug;
    BuffShotGateSnapshot shot_gate;
};

struct BuffExactValidSourceIdentity
{
    std::uint64_t producer_epoch = 0;
    std::uint64_t image_sequence = 0;
};

struct BuffExactValidCaptureDiagnostics
{
    bool armed = false;
    bool complete = false;
    bool order_valid = true;
    bool ring_truncated = false;
    std::size_t max_frames = 0;
    std::size_t max_bytes = 0;
    std::size_t retained_frames = 0;
    std::size_t retained_bytes = 0;
    std::uint64_t observed_completions = 0;
    std::uint64_t accepted_completions = 0;
    std::uint64_t evicted_frames = 0;
    std::uint64_t evicted_bytes = 0;
    std::uint64_t invalid_identity_rejects = 0;
    std::uint64_t duplicate_identity_rejects = 0;
    std::uint64_t regression_identity_rejects = 0;
    std::uint64_t completion_order_rejects = 0;
    std::uint64_t unretainable_frames = 0;
    std::uint64_t unretainable_bytes = 0;
    bool has_first_retained_identity = false;
    bool has_last_retained_identity = false;
    BuffExactValidSourceIdentity first_retained_identity;
    BuffExactValidSourceIdentity last_retained_identity;
    std::uint64_t first_retained_completion_sequence = 0;
    std::uint64_t last_retained_completion_sequence = 0;
    bool has_first_evicted_identity = false;
    bool has_last_evicted_identity = false;
    BuffExactValidSourceIdentity first_evicted_identity;
    BuffExactValidSourceIdentity last_evicted_identity;
    std::uint64_t first_evicted_completion_sequence = 0;
    std::uint64_t last_evicted_completion_sequence = 0;
};

// Testable implementation of the default-off exact-valid capture hook.  The
// pipeline constructs this object only when AIM_BUFF_EXACT_VALID_CAPTURE_DIR is
// non-empty.  Captured cv::Mat storage is retained by refcount; no image clone
// is performed by this class.
class BuffExactValidSequenceCapture
{
public:
    static constexpr std::size_t kHardMaxFrames = 512;
    static constexpr std::size_t kHardMaxBytes = 1536ULL * 1024ULL * 1024ULL;

    explicit BuffExactValidSequenceCapture(
        std::string capture_dir,
        std::size_t max_frames = kHardMaxFrames,
        std::size_t max_bytes = kHardMaxBytes);

    void reset();
    void observeCompletion(const BuffRuneResult& result);
    BuffExactValidCaptureDiagnostics diagnostics() const;
    std::vector<BuffExactValidSourceIdentity> retainedIdentities() const;

private:
    struct ExpectedOutcome
    {
        bool rune_present = false;
        int rune_type = -1;
        std::size_t solved_blades = 0;
        bool target_solved = false;
        bool has_control = false;
        bool current_target_control = false;
        bool switch_deferred = false;
        bool target_switched = false;
        int selected_target_index = -1;
        int control_aiming_state = 0;
        int control_shot_mode = 0;
        int control_shot_buff_mode = 0;
        double control_gimbal_yaw_deg = 0.0;
        double control_gimbal_pitch_deg = 0.0;
        double control_yaw_error_deg = 0.0;
        cv::Point2f r_center{};
        Eigen::Vector3d rune_xyz = Eigen::Vector3d::Zero();
        Eigen::Vector3d rune_ypd = Eigen::Vector3d::Zero();
        Eigen::Vector3d rune_ypr = Eigen::Vector3d::Zero();
        Eigen::Vector3d blade_xyz = Eigen::Vector3d::Zero();
        Eigen::Vector3d blade_ypd = Eigen::Vector3d::Zero();
        int target_pnp_method = -1;
        std::array<int, 4> target_pnp_order{{-1, -1, -1, -1}};
        double target_pnp_reproj_error_px = std::numeric_limits<double>::quiet_NaN();
        double target_pnp_score = std::numeric_limits<double>::quiet_NaN();
        double target_pnp_model_center_error_px = std::numeric_limits<double>::quiet_NaN();
    };

    struct CapturedCompletion
    {
        rm::Frame frame;
        std::uint64_t completion_sequence = 0;
        std::uint64_t completion_timestamp_ns = 0;
        std::size_t image_storage_bytes = 0;
        ExpectedOutcome expected;
    };

    bool writeSequenceManifest(const BuffRuneResult& trigger);
    void evictOldest();
    void refreshRetainedBounds();

    std::string capture_dir_;
    std::size_t max_frames_ = kHardMaxFrames;
    std::size_t max_bytes_ = kHardMaxBytes;
    std::deque<CapturedCompletion> retained_;
    BuffExactValidCaptureDiagnostics diagnostics_;
    bool has_last_observed_identity_ = false;
    BuffExactValidSourceIdentity last_observed_identity_;
    std::uint64_t last_observed_completion_sequence_ = 0;
};

struct BuffYoloPacket
{
    rm::Frame frame;
    std::vector<YOLO11_BUFF::Object> candidates;
    std::chrono::steady_clock::time_point frame_timestamp{};
    double bullet_speed = 0.0;
    PowerRune_type mode = SMALL;
    uint64_t generation = 0;
    std::uint64_t proposal_sequence = 0;
    double yolo_ms = 0.0;
    struct ObservationSupersetProposal
    {
        std::uint64_t producer_epoch = 0;
        std::uint64_t image_sequence = 0;
        cv::Rect union_roi;
        std::vector<YOLO11_BUFF::Object> candidates;
        BuffCanonicalObservation observation;
        SolverFrameContext solver_frame;
        ExhaustivePnpProposal pnp;
        BuffObservationSupersetCost cost;
    };
    std::optional<ObservationSupersetProposal> observation_proposal;
};

struct BuffDetectionPacket
{
    rm::Frame frame;
    std::optional<PowerRune> rune;
    std::vector<YOLO11_BUFF::Object> candidates;
    std::chrono::steady_clock::time_point frame_timestamp{};
    double bullet_speed = 0.0;
    PowerRune_type mode = SMALL;
    uint64_t generation = 0;
    double infer_ms = 0.0;
    double yolo_ms = 0.0;
    double solve_ms = 0.0;
    bool switch_deferred = false;
    bool target_switched = false;
    int selected_target_index = -1;
    BuffSolveCostSample solve_cost;
};

class BuffRunePipeline
{
public:
    explicit BuffRunePipeline(
        const std::string& config_path,
        BuffRunePipelineOptions options = BuffRunePipelineOptions{});
    ~BuffRunePipeline();

    BuffRunePipeline(const BuffRunePipeline&) = delete;
    BuffRunePipeline& operator=(const BuffRunePipeline&) = delete;

    void push(rm::Frame frame);
    void reset();
    bool tryPopLatest(BuffRuneResult* result);
    BuffRunePipelineCounters counters() const noexcept;
    std::vector<BuffRuneCompletionSample> completionSamples() const;
    bool refreshControlWithFeedback(
        BuffRuneResult* result,
        const rm::FeedBackData& live_feedback,
        std::chrono::steady_clock::time_point timestamp);

private:
    struct DebugPredictionOverlay
    {
        double source_time_ms = 0.0;
        double target_time_ms = 0.0;
        Eigen::Vector3d center_world = Eigen::Vector3d::Zero();
        double yaw = 0.0;
        double roll = 0.0;
    };

    struct HistoricalPredictionSeed
    {
        double time_ms = 0.0;
        double relative_time_s = std::numeric_limits<double>::quiet_NaN();
        Eigen::VectorXd state;
        double target_roll_offset = std::numeric_limits<double>::quiet_NaN();
        int voter_direction = 0;
        int history_size = 0;
        int reinit_reason = 0;
        bool switch_deferred = false;
        bool target_switched = false;
    };

    struct HistoricalPredictionMatch
    {
        double source_time_ms = 0.0;
        double eval_dt_s = 0.0;
        double target_roll_offset = std::numeric_limits<double>::quiet_NaN();
        Eigen::VectorXd predicted_state;
    };

    void inferLoop();
    void solveLoop();
    void proposalWorkerLoop(std::size_t worker_index);
    void proposalCommitLoop();
    bool commitOrderedPacket(BuffYoloPacket packet);
    void trackAimLoop();
    BuffYoloPacket runYolo(rm::Frame frame, uint64_t generation);
    void buildObservationSupersetScaffold(
        BuffYoloPacket* packet, BuffCanonicalWorkerScratch* scratch = nullptr);
    void completeObservationProposal(
        BuffYoloPacket* packet, Solver* scratch_solver);
    void recordObservationIdentity(const BuffYoloPacket& packet);
    BuffDetectionPacket solveRune(BuffYoloPacket packet);
    BuffRuneResult buildResult(BuffDetectionPacket packet);
    std::chrono::steady_clock::time_point timestampForFrame(const rm::Frame& frame);
    Eigen::Quaterniond gimbalQuaternion(const rm::Frame& frame) const;
    BuffShotGateSnapshot evaluateShotGate(
        const rm::Frame& frame,
        const AimCommand& command,
        const std::optional<PowerRune>& rune,
        const BuffTracker::DebugSnapshot& tracker_debug,
        bool switch_deferred,
        bool target_switched,
        bool update_stability);
    rm::ControlData makeControl(
        const rm::Frame& frame, const AimCommand& command, bool shot_allowed);
    uint8_t buffModeFlag(const rm::Frame& frame) const;
    void updateFps();
    void enqueueHistoricalPredictionSeed(
        double current_time_ms,
        const BuffTracker::DebugSnapshot& tracker_debug);
    std::optional<HistoricalPredictionMatch> buildHistoricalPredictionMatch(
        double current_time_ms) const;
    void enqueueDebugPredictionOverlay(double current_time_ms);
    void drawPastPredictionOverlay(cv::Mat& image, double current_time_ms);
    void maybeCaptureExactValid(const BuffRuneResult& result);
    void recordCompletionSample(const BuffRuneResult& result);

    std::string config_path_;
    bool emit_debug_artifacts_ = true;
    bool collect_completion_samples_ = false;
    bool observation_superset_enabled_ = false;
    bool ordered_commit_inline_ = false;
    int proposal_worker_count_ = 1;
    Buff_Detector detector_;
    Solver solver_;
    Solver visual_solver_;
    BuffTracker tracker_;
    Aimer aimer_;

    std::atomic<bool> stop_{false};
    std::thread infer_thread_;
    std::thread solve_thread_;
    std::thread track_thread_;
    std::vector<std::thread> proposal_worker_threads_;

    struct ProposalTerminal
    {
        std::uint64_t sequence = 0;
        std::uint64_t generation = 0;
        std::size_t worker_index = 0;
        BuffYoloPacket packet;
        bool success = false;
        std::string error;
    };
    static constexpr std::size_t kProposalMaxWorkers = 4;
    mutable std::mutex proposal_mutex_;
    std::condition_variable proposal_input_cv_;
    std::condition_variable proposal_reorder_cv_;
    std::deque<BuffYoloPacket> proposal_input_;
    std::map<std::uint64_t, ProposalTerminal> proposal_reorder_;
    std::array<bool, 4> proposal_worker_has_terminal_{};
    bool proposal_accepting_ = true;
    std::size_t proposal_inflight_ = 0;
    std::uint64_t next_proposal_sequence_ = 1;
    std::uint64_t next_commit_sequence_ = 1;

    std::mutex input_mutex_;
    std::condition_variable input_cv_;
    std::optional<rm::Frame> latest_input_;
    std::mutex push_identity_mutex_;
    std::uint64_t last_pushed_epoch_ = 0;

    std::mutex detection_mutex_;
    std::condition_variable detection_cv_;
    std::optional<BuffDetectionPacket> latest_detection_;

    std::mutex yolo_mutex_;
    std::condition_variable yolo_cv_;
    std::optional<BuffYoloPacket> latest_yolo_;

    LatestResultMailbox<BuffRuneResult> output_mailbox_;
    std::mutex infer_mutex_;
    std::mutex solve_mutex_;
    std::mutex state_mutex_;
    std::mutex ordered_commit_mutex_;
    std::mutex observation_identity_mutex_;
    std::mutex time_mutex_;
    mutable std::mutex completion_samples_mutex_;
    std::vector<BuffRuneCompletionSample> completion_samples_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<std::uint64_t> pushed_frames_{0};
    std::atomic<std::uint64_t> input_queue_overwrites_{0};
    std::atomic<std::uint64_t> yolo_completed_{0};
    std::atomic<std::uint64_t> yolo_queue_overwrites_{0};
    std::atomic<std::uint64_t> solve_completed_{0};
    std::atomic<std::uint64_t> detection_queue_overwrites_{0};
    std::atomic<std::uint64_t> ordered_commit_failures_{0};
    std::atomic<std::uint64_t> observation_proposal_attempts_{0};
    std::atomic<std::uint64_t> observation_proposal_fallbacks_{0};
    std::atomic<std::uint64_t> observation_proposal_candidates_{0};
    std::atomic<std::uint64_t> observation_pnp_proposals_{0};
    std::atomic<std::uint64_t> observation_union_roi_pixels_{0};
    std::atomic<std::uint64_t> observation_template_result_pixels_{0};
    std::atomic<std::uint64_t> observation_cap_events_{0};
    std::atomic<std::uint64_t> observation_identity_gaps_{0};
    std::atomic<std::uint64_t> observation_identity_failures_{0};
    std::atomic<std::uint64_t> observation_ready_consumptions_{0};
    std::atomic<std::uint64_t> proposal_submitted_{0};
    std::atomic<std::uint64_t> proposal_completed_{0};
    std::atomic<std::uint64_t> proposal_committed_{0};
    std::atomic<std::uint32_t> proposal_active_workers_{0};
    std::atomic<std::uint32_t> proposal_max_active_workers_{0};
    std::atomic<std::uint32_t> proposal_input_max_occupancy_{0};
    std::atomic<std::uint32_t> proposal_reorder_max_occupancy_{0};
    std::atomic<std::uint32_t> proposal_max_inflight_{0};
    std::atomic<std::uint64_t> proposal_terminal_gaps_{0};
    std::atomic<std::uint64_t> proposal_terminal_failures_{0};
    std::atomic<std::uint64_t> proposal_cancelled_{0};
    std::atomic<std::uint64_t> proposal_stale_{0};
    std::atomic<std::uint64_t> proposal_head_wait_ns_{0};
    std::array<std::atomic<std::uint64_t>, 4> proposal_worker_completed_{};
    std::array<std::atomic<std::uint64_t>, 4> proposal_worker_total_ns_{};
    std::atomic<std::uint64_t> essential_completed_{0};
    std::atomic<std::uint64_t> published_results_{0};
    std::atomic<std::uint64_t> popped_results_{0};

    std::uint64_t last_observation_epoch_ = 0;
    std::uint64_t last_observation_sequence_ = 0;

    bool time_base_ready_ = false;
    double time_base_ms_ = 0.0;
    std::chrono::steady_clock::time_point time_base_tp_{};

    int fps_ = 0;
    int fps_count_ = 0;
    std::chrono::steady_clock::time_point fps_start_tp_{};
    bool draw_yolo_results_ = false;
    bool draw_r_binary_mask_ = true;
    std::unique_ptr<BuffExactValidSequenceCapture> exact_valid_capture_;
    bool shot_gate_enabled_ = true;
    int shot_gate_min_stable_frames_ = 3;
    double shot_gate_max_pnp_reproj_error_px_ = 28.0;
    double shot_gate_max_model_center_error_px_ = 8.0;
    double shot_gate_max_yaw_error_deg_ = 1.5;
    double shot_gate_max_pitch_error_deg_ = 1.5;
    int shot_gate_stable_frames_ = 0;
    std::deque<HistoricalPredictionSeed> historical_prediction_seeds_;
    std::deque<DebugPredictionOverlay> debug_prediction_overlays_;
    std::optional<rm::ControlData> last_valid_control_;
};

}  // namespace auto_buff

#endif  // AUTO_BUFF_RUNE_PIPELINE_HPP
