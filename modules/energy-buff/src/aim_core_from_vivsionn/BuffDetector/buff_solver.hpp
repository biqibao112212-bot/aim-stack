#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "buff_runtime_config.hpp"
#include "params.h"
#include "tools/math_tools.hpp"

namespace auto_buff
{

// Fixed-size, completion-local instrumentation.  Every field is accumulated
// by the ordered solve thread; no logging or reporting lock is taken here.
struct BuffPnpCostSample
{
  std::uint64_t total_ns = 0;
  std::uint64_t set_image_size_ns = 0;
  std::uint64_t prevalidation_object_points_ns = 0;
  std::uint64_t fanblade_point_prep_ns = 0;
  std::uint64_t fast_order_build_ns = 0;
  std::uint64_t fast_solvepnp_ns = 0;
  std::uint64_t fast_reprojection_score_ns = 0;
  std::uint64_t fallback_order_build_ns = 0;
  std::uint64_t fallback_solvepnp_ns = 0;
  std::uint64_t fallback_reprojection_score_ns = 0;
  std::uint64_t solution_reject_gates_ns = 0;
  std::uint64_t debug_materialization_ns = 0;
  std::uint64_t camera_to_world_ns = 0;
  std::uint64_t primary_commit_ns = 0;

  std::uint32_t set_image_size_calls = 0;
  std::uint32_t params_reload_checks = 0;
  std::uint32_t params_reloads = 0;
  std::uint32_t intrinsics_changes = 0;
  std::uint32_t image_size_changes = 0;
  std::uint32_t total_blades = 0;
  std::uint32_t eligible_blades = 0;
  std::uint32_t solved_blades = 0;
  std::uint32_t fast_unique_orders = 0;
  std::uint32_t fast_pnp_calls = 0;
  std::uint32_t fast_solve_successes = 0;
  std::uint32_t fast_valid_tvec = 0;
  std::uint32_t fast_reprojection_gate_passes = 0;
  std::uint32_t fallback_invocations = 0;
  std::uint32_t fallback_fast_failed = 0;
  std::uint32_t fallback_fast_error_gate = 0;
  std::uint32_t fallback_pnp_calls = 0;
  std::uint32_t fallback_solve_successes = 0;
  std::uint32_t fallback_valid_tvec = 0;
  std::uint32_t reprojection_calls = 0;
  std::uint32_t project_points_calls = 0;
  std::uint32_t candidate_score_evaluations = 0;
  std::uint32_t invalid_tvec_rejects = 0;
  std::uint32_t reprojection_gate_rejects = 0;
  std::uint32_t invalid_world_rejects = 0;

  std::int32_t selected_blade = -1;
  std::int32_t selected_method = -1;
  std::int32_t selected_order_index = -1;
  // 0 none, 1 prior fast path, 2 exhaustive fallback.
  std::int32_t selected_source = 0;
  bool prior_pose_valid = false;
  bool solved = false;
  double final_reprojection_error_px = 0.0;
  double final_model_center_error_px = 0.0;
};

using BuffPointOrder = std::array<int, 4>;

// Immutable per-frame calibration/pose snapshot.  It is created by the
// ordered calibration owner and can then be copied to proposal workers without
// allowing those workers to reload Params or observe later pose changes.
struct SolverFrameContext {
  std::uint64_t calibration_version = 0;
  cv::Mat camera_matrix;
  cv::Mat distort_coeffs;
  Eigen::Matrix3d R_camera2gimbal = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_gimbal2world = Eigen::Matrix3d::Identity();
};

struct SolverPriorSnapshot {
  bool valid = false;
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  BuffPointOrder order{0, 1, 2, 3};
};

// Solver-owned adapter used by the detector's canonical-R proposal.  Keeping
// this type independent of detector internals prevents a circular dependency.
struct SolverPnpHypothesis {
  std::uint32_t hypothesis_index = 0;
  std::uint64_t source_pixel_signature = 0;
  cv::Point2f r_center{};
  std::vector<FanBlade> fanblades;
  const std::vector<FanBlade> *fanblades_view = nullptr;

  const std::vector<FanBlade> & blades() const {
    return fanblades_view != nullptr ? *fanblades_view : fanblades;
  }
};

struct RawPnpSolution {
  std::uint32_t hypothesis_index = 0;
  std::uint32_t blade_index = 0;
  std::uint8_t order_index = 0;
  int method = -1;
  bool solve_returned = false;
  bool valid_tvec = false;
  BuffPointOrder order{0, 1, 2, 3};
  std::array<cv::Point2f, 4> raw_image_points{};
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  double reprojection_error_px = std::numeric_limits<double>::max();
};

enum class ExhaustivePnpStatus : std::uint8_t {
  Ready,
  CapacityExceeded,
  InvalidContext,
  InvalidInput,
};

struct ExhaustivePnpHypothesisMetadata {
  std::uint32_t hypothesis_index = 0;
  std::uint64_t source_pixel_signature = 0;
  cv::Point2f r_center{};
  std::uint32_t fanblade_count = 0;
};

struct ExhaustivePnpProposal {
  ExhaustivePnpStatus status = ExhaustivePnpStatus::InvalidContext;
  std::uint64_t calibration_version = 0;
  std::vector<ExhaustivePnpHypothesisMetadata> hypotheses;
  std::vector<RawPnpSolution> solutions;
  std::uint32_t expected_solution_count = 0;
};

struct PnpReductionResult {
  bool applicable = false;
  bool solved = false;
  bool used_fast_prior = false;
  std::uint32_t selected_blade = 0;
  RawPnpSolution selected;
  double score = std::numeric_limits<double>::max();
  SolverPriorSnapshot prior_after;
};

class Solver {
public:
  // 3D 模型点定义 (单位: 米)
  // 顺序必须与 YOLO 关键点输出顺序一致：
  // 0:上, 1:左, 2:下, 3:右, 4:扇叶中心, 5:R标
  const std::vector<cv::Point3f> OBJECT_POINTS = {
      cv::Point3f(0, 0, 827e-3),       // 0: 上
      cv::Point3f(0, -127e-3, 700e-3), // 1: 左
      cv::Point3f(0, 0, 573e-3),       // 2: 下
      cv::Point3f(0, 127e-3, 700e-3),  // 3: 右
      cv::Point3f(0, 0, 700e-3),       // 4: 扇叶中心
      cv::Point3f(0, 0, 0)             // 5: R 标中心 (原点)
  };

  explicit Solver(const std::string &config_path);

  // Reset only cross-frame PnP/debug priors. Calibration and immutable model
  // geometry remain loaded. BuffRunePipeline::reset owns the synchronization.
  void reset();

  /**
   * @brief 更新云台到世界坐标系的旋转矩阵 (由电控 IMU 数据驱动)
   */
  void set_R_gimbal2world(const Eigen::Quaterniond &q);

  /**
   * @brief 更新当前图像尺寸，用于在 CAMERA_UPSIDE_DOWN 翻图后同步镜像主点。
   */
  void set_image_size(const cv::Size &image_size,
                      BuffPnpCostSample *cost = nullptr) const;

  SolverFrameContext makeFrameContext(const cv::Size &image_size,
                                      BuffPnpCostSample *cost = nullptr) const;
  SolverPriorSnapshot priorSnapshot() const;

  // Pure fixed-R extraction: exactly eight legacy point orders by IPPE then
  // ITERATIVE, with useExtrinsicGuess=false and an entry retained for every
  // attempted solve (including failures).  No shared prior is read or written.
  ExhaustivePnpProposal
  buildExhaustiveProposal(const std::vector<SolverPnpHypothesis> &hypotheses,
                          const SolverFrameContext &frame,
                          BuffPnpCostSample *cost = nullptr) const;

  // Ordered reduction.  The current prior is supplied explicitly and applied
  // only when the caller invokes applyPrior(result.prior_after).
  PnpReductionResult reduceProposal(PowerRune *rune,
                                    std::uint32_t selected_hypothesis,
                                    const ExhaustivePnpProposal &proposal,
                                    const SolverFrameContext &frame,
                                    const SolverPriorSnapshot &current_prior,
                                    BuffPnpCostSample *cost = nullptr) const;
  PnpReductionResult solveFromProposal(PowerRune &rune,
                                       std::uint32_t selected_hypothesis,
                                       const ExhaustivePnpProposal &proposal,
                                       const SolverFrameContext &frame,
                                       BuffPnpCostSample *cost = nullptr);
  void applyPrior(const SolverPriorSnapshot &prior);

  /**
   * @brief PnP 解算核心函数
   * @param rune 输入/输出参数，解算结果将填充到 rune 的 xyz/ypr 成员中
   */
  void solve(PowerRune & rune, BuffPnpCostSample* cost = nullptr) const;

  /**
   * @brief 根据当前的 rvec/tvec 将 3D 点投影回像素坐标 (用于调试绘图)
   */
  cv::Point2f point_3d_to_pixel(const cv::Point3f& pt_3d) const;

  /**
   * @brief 获取理论上的 R 标中心像素点 (用于辅助 Detector 修正圆心)
   */
  cv::Point2f get_projected_r_center() const;

  const std::vector<cv::Point3f>& get_object_points() const { return OBJECT_POINTS; }

  // 【新增】为了通过测试代码，加回这个接口
  Eigen::Matrix3d R_gimbal2world() const { return R_gimbal2world_; }

  // 【新增】加回重投影功能 (用于可视化调试)
  std::vector<cv::Point2f> reproject_buff(const Eigen::Vector3d & xyz_in_world, double yaw, double roll) const;

private:
  void refresh_intrinsics_from_params(
    bool force = false, BuffPnpCostSample* cost = nullptr) const;

  // 相机参数
  std::string config_path_;
  mutable cv::Mat base_camera_matrix_;
  mutable cv::Mat camera_matrix_;
  mutable cv::Mat distort_coeffs_;
  mutable bool camera_upside_down_ = false;
  mutable cv::Size last_image_size_;
  mutable std::unique_ptr<Params> params_;
  mutable std::uint64_t calibration_version_ = 0;

  // 坐标系变换矩阵
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();

  // PnP 解算结果缓存 (用于重投影)
  mutable cv::Vec3d rvec_;
  mutable cv::Vec3d tvec_;
  mutable bool last_pose_valid_ = false;
  mutable cv::Vec3d last_rvec_;
  mutable cv::Vec3d last_tvec_;
  mutable std::array<int, 4> last_point_order_ = {0, 1, 2, 3};
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__SOLVER_HPP
