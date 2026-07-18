#ifndef AUTO_BUFF__BUFF_DETECTOR_HPP
#define AUTO_BUFF__BUFF_DETECTOR_HPP

#include <yaml-cpp/yaml.h>
#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include "buff_solver.hpp"
#include "buff_type.hpp"
#include "yolo11_buff.hpp"

namespace auto_buff
{

const int LOSE_MAX = 20;

enum class BuffDetectorEarlyExit : std::uint8_t
{
  None = 0,
  RawResultsEmpty,
  NoRuneCandidate,
  TargetSwitchDeferred,
  TargetSelectionFailed,
  NoFanblades,
  PnpUnsolved,
};

struct BuffRSearchCostSample
{
  std::uint64_t total_ns = 0;
  std::uint64_t prior_ns = 0;
  std::uint64_t roi_setup_ns = 0;
  std::uint64_t preprocess_ns = 0;
  std::uint64_t circle_mask_ns = 0;
  std::uint64_t template_scale_search_ns = 0;
  std::uint64_t find_contours_ns = 0;
  std::uint64_t contour_filter_score_ns = 0;
  std::uint64_t selection_state_commit_ns = 0;
  std::uint64_t debug_materialization_ns = 0;
  std::uint64_t template_result_pixels = 0;
  std::uint64_t debug_copied_bytes = 0;
  std::uint64_t debug_copied_elements = 0;

  std::uint32_t motion_comparisons = 0;
  std::uint32_t motion_matches = 0;
  std::uint32_t geometry_blade_pairs = 0;
  std::uint32_t geometry_center_hypotheses = 0;
  std::uint32_t prior_candidates = 0;
  std::uint32_t roi_width = 0;
  std::uint32_t roi_height = 0;
  std::uint32_t roi_pixels = 0;
  std::uint32_t min_template_size = 0;
  std::uint32_t max_template_size = 0;
  std::uint32_t template_scale_count = 0;
  std::uint32_t template_builds = 0;
  std::uint32_t template_cache_hits = 0;
  std::uint32_t match_template_calls = 0;
  std::uint32_t template_distance_gate_passes = 0;
  std::uint32_t winning_template_size = 0;
  std::uint32_t contours_total = 0;
  std::uint32_t contours_area_pass = 0;
  std::uint32_t contours_aspect_pass = 0;
  std::uint32_t contours_circularity_pass = 0;
  std::uint32_t contours_radius_pass = 0;
  std::uint32_t contours_accepted = 0;

  // 0 prior, 1 contour, 2 template, 3 held center.
  std::uint8_t selected_source = 0;
  double radius = 0.0;
  double winning_template_score = 0.0;
};

struct BuffDetectorCostSample
{
  std::uint64_t total_ns = 0;
  std::uint64_t dispatch_reset_ns = 0;
  std::uint64_t result_classification_ns = 0;
  std::uint64_t target_selection_ns = 0;
  std::uint64_t fanblade_build_ns = 0;
  std::uint64_t r_center_ns = 0;
  std::uint64_t pnp_solve_ns = 0;
  std::uint64_t reject_state_commit_ns = 0;

  std::uint32_t mode = 0;
  std::uint32_t raw_candidates = 0;
  std::uint32_t target_candidates = 0;
  std::uint32_t hit_candidates = 0;
  std::uint32_t target_candidates_examined = 0;
  std::uint32_t constructed_fanblades = 0;
  bool locked_target_matched = false;
  bool target_switch_deferred = false;
  bool target_switched = false;
  BuffDetectorEarlyExit early_exit = BuffDetectorEarlyExit::None;
  BuffRSearchCostSample r_search;
  BuffPnpCostSample pnp;
};

constexpr std::uint32_t kBuffObservationMaxTargets = 2;
constexpr std::uint32_t kBuffObservationMaxHitContext = 2;
constexpr std::uint32_t kBuffObservationMaxAnchors = 6;
constexpr std::uint64_t kBuffObservationMaxTemplateResultPixels = 250000;
constexpr std::uint32_t kBuffObservationMaxContours = 4096;

enum class BuffObservationFallbackReason : std::uint8_t
{
  None = 0,
  InvalidInput,
  CandidateCap,
  AnchorCap,
  ResponsePixelCap,
  ContourCap,
  MissingCoverage,
  HoldAmbiguity,
  PnpCap,
  PnpRMismatch,
  NumericalFailure,
};

struct BuffTargetHypothesis
{
  std::uint32_t hypothesis_index = 0;
  std::uint32_t source_candidate_index = 0;
  YOLO11_BUFF::Object target;
  std::vector<FanBlade> fanblades;
  cv::Point2f yolo_r{};
  double blade_radius = 0.0;
};

struct BuffRAnchor
{
  std::uint32_t hypothesis_index = 0;
  // 0: YOLO R, 1: raw pair-geometry center.
  std::uint8_t source = 0;
  cv::Point2f center{};
  double radius = 0.0;
  cv::Rect support_roi;
};

struct BuffRScaleResponse
{
  int template_size = 0;
  cv::Point global_origin{};
  cv::Mat scores;
};

struct BuffRContourObservation
{
  cv::Point2f center_global{};
  double area = 0.0;
  double aspect_ratio = 0.0;
  double circularity = 0.0;
};

struct BuffCanonicalRChoice
{
  std::uint32_t hypothesis_index = 0;
  cv::Point2f canonical_prior{};
  cv::Point2f r_center{};
  // 0 prior, 1 contour, 2 template.
  std::uint8_t source = 0;
  bool template_valid = false;
  double template_score = -1.0;
  int template_size = 0;
  RSearchDebug debug;
};

struct BuffCanonicalObservation
{
  cv::Size image_size;
  std::uint32_t raw_candidate_count = 0;
  std::vector<BuffTargetHypothesis> targets;
  std::vector<YOLO11_BUFF::Object> hit_context;
  cv::Rect union_roi;
  std::vector<BuffRAnchor> anchors;
  std::vector<BuffRScaleResponse> scale_responses;
  std::vector<BuffRContourObservation> contours;
  std::vector<BuffCanonicalRChoice> r_choices;
  std::uint64_t union_roi_pixels = 0;
  std::uint64_t template_result_pixels = 0;
  std::uint64_t preprocess_ns = 0;
  std::uint64_t template_ns = 0;
  std::uint64_t contour_ns = 0;
  std::uint64_t scratch_allocations = 0;
  std::uint64_t scratch_reuses = 0;
  std::uint64_t response_cells_scanned = 0;
  std::uint64_t support_rejected_cells = 0;
  std::uint64_t distance_tested_cells = 0;
  std::uint64_t contour_copy_bytes_avoided = 0;
  std::uint32_t cap_events = 0;
  bool ready = false;
  bool requires_legacy_fallback = false;
  BuffObservationFallbackReason fallback_reason = BuffObservationFallbackReason::None;
};

struct BuffCanonicalWorkerScratch
{
  std::vector<cv::Mat> channels;
  cv::Mat gray;
  cv::Mat gamma;
  cv::Mat threshold;
  cv::Mat binary;
  cv::Mat kernel;
  cv::Mat mask;
  cv::Mat masked;
  cv::Mat support_lookup;
  std::vector<int> scales;
  std::uint64_t allocations = 0;
  std::uint64_t reuses = 0;
};

// 追踪状态枚举
enum TrackStatus {
  LOSE,
  TEM_LOSE,
  TRACK
};

class Buff_Detector
{
public:
  Buff_Detector(const std::string & config);

  void set_mode(PowerRune_type mode) { mode_ = mode; }
  void reset();

	  // 核心检测入口
	  std::optional<PowerRune> detect(
	    cv::Mat & bgr_img, const Solver & solver, bool draw_yolo_results = true);
	  std::vector<YOLO11_BUFF::Object> detect_candidates(
	    PowerRune_type mode, cv::Mat & bgr_img, bool draw_yolo_results = true);
		  std::optional<PowerRune> solve_candidates(
		    PowerRune_type mode,
		    cv::Mat & bgr_img,
		    const Solver & solver,
		    const std::vector<YOLO11_BUFF::Object> & results,
		    std::chrono::steady_clock::time_point timestamp,
		    BuffDetectorCostSample* cost = nullptr);
	  bool last_switch_deferred() const { return last_switch_deferred_; }
	  bool last_target_switched() const { return last_target_switched_; }
	  int last_selected_target_index() const { return last_selected_target_index_; }
	  std::optional<RSearchDebug> last_r_search_debug() const { return last_r_search_debug_; }

    // Default-off observation-superset numerical profile. This method is pure
    // with respect to detector history: it reads immutable configuration and
    // same-frame pixels/candidates only and returns move-owned artifacts.
    BuffCanonicalObservation build_canonical_observation(
      PowerRune_type mode,
      const cv::Mat & bgr_img,
      const std::vector<YOLO11_BUFF::Object> & results,
      BuffCanonicalWorkerScratch * scratch = nullptr) const;
    // Ordered-state preflight for the fixed-R PnP table. A template miss can
    // consume the legacy hold center without new pixel work, but only when
    // that center is exactly the R used by the extracted PnP proposal.
    bool canonical_observation_commit_supported(
      const BuffCanonicalObservation & observation) const;
    std::optional<PowerRune> solve_canonical_observation(
      PowerRune_type mode,
      const BuffCanonicalObservation & observation,
      Solver & solver,
      const SolverFrameContext & solver_frame,
      const ExhaustivePnpProposal & pnp_proposal,
      std::chrono::steady_clock::time_point timestamp,
      BuffDetectorCostSample * cost = nullptr);

	private:
	  std::optional<PowerRune> solve_big_buff(
	    cv::Mat & bgr_img,
	    const Solver & solver,
	    const std::vector<YOLO11_BUFF::Object> & results,
	    std::chrono::steady_clock::time_point timestamp,
	    BuffDetectorCostSample* cost);
	  std::optional<PowerRune> solve_small_buff(
	    cv::Mat & bgr_img,
	    const Solver & solver,
	    const std::vector<YOLO11_BUFF::Object> & results,
	    BuffDetectorCostSample* cost);
	  int select_big_buff_target_index(
	    const std::vector<YOLO11_BUFF::Object> & targets,
		    const std::vector<YOLO11_BUFF::Object> & hit_context,
		    const cv::Size & image_size,
		    std::chrono::steady_clock::time_point timestamp,
		    bool * switch_deferred,
		    bool * target_switched,
		    BuffDetectorCostSample* cost);

  void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img, int class_id) const;
  cv::Point2f get_r_prior(
    const std::vector<FanBlade> & fanblades,
    cv::Point2f yolo_r_center,
    const cv::Size & image_size,
    BuffRSearchCostSample* cost) const;

  // 增加 yolo_r_center 参数，直接接收 YOLO 给出的预测点
  cv::Point2f get_r_center(
    std::vector<FanBlade> & fanblades,
    cv::Mat & bgr_img,
    int class_id,
    cv::Point2f yolo_r_center,
    BuffRSearchCostSample* cost);

  void handle_lose(
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now());

  TrackStatus status_;
  int lose_;
  YOLO11_BUFF MODE_;
  PowerRune_type mode_ = SMALL;

  double r_search_radius_scale_ = 1.15;
  double r_search_radius_min_ = 18.0;
  double r_search_radius_max_ = 140.0;
  double r_min_area_ = 12.0;
  double r_max_area_ = 1200.0;
  double r_max_aspect_ratio_ = 1.8;
  double r_min_circularity_ = 0.45;
  double r_max_accept_ratio_ = 0.85;
  double r_geometry_gate_scale_ = 4.0;
  double r_geometry_gate_min_ = 40.0;
  double r_geometry_gate_max_ = 260.0;
  double r_yolo_gate_scale_ = 2.2;
  double r_yolo_gate_min_ = 25.0;
  double r_yolo_gate_max_ = 180.0;
  int r_binary_threshold_ = 150;
  double target_switch_missing_timeout_s_ = 0.2;

	  std::optional<PowerRune> last_powerrune_ = std::nullopt;
	  std::optional<RSearchDebug> last_r_search_debug_ = std::nullopt;
	  std::optional<std::chrono::steady_clock::time_point> locked_target_missing_since_ = std::nullopt;
	  std::optional<double> locked_target_angle_ = std::nullopt;
	  std::optional<std::chrono::steady_clock::time_point> locked_target_update_time_ = std::nullopt;
	  int r_template_hold_miss_count_ = 0;
	  bool last_switch_deferred_ = false;
	  bool last_target_switched_ = false;
	  int last_selected_target_index_ = -1;
	};

} // namespace auto_buff

#endif // AUTO_BUFF__BUFF_DETECTOR_HPP
