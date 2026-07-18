/**
 * @file buff_solver.cpp
 * @brief 能量机关解算器实现
 */

#include "buff_solver.hpp"
#include "params.h"
#include "tools/math_tools.hpp"
#include "tools/logger.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{
using PointPermutation = std::array<int, 4>;
using CostClock = std::chrono::steady_clock;

std::uint64_t cost_ns(const CostClock::time_point & begin)
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(CostClock::now() - begin).count());
}

bool is_finite_point(const cv::Point2f & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool is_valid_tvec(const cv::Vec3d & tvec)
{
  return std::isfinite(tvec[0]) && std::isfinite(tvec[1]) && std::isfinite(tvec[2]) &&
         tvec[2] > 1e-4;
}

double nan_value()
{
  return std::numeric_limits<double>::quiet_NaN();
}

float nan_float()
{
  return std::numeric_limits<float>::quiet_NaN();
}

bool exact_point_equal(const cv::Point2f & lhs, const cv::Point2f & rhs)
{
  return std::memcmp(&lhs.x, &rhs.x, sizeof(lhs.x)) == 0 &&
         std::memcmp(&lhs.y, &rhs.y, sizeof(lhs.y)) == 0;
}

bool exact_raw_points_equal(
  const std::array<cv::Point2f, 4> & lhs,
  const std::vector<cv::Point2f> & rhs)
{
  if (rhs.size() < lhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!exact_point_equal(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

bool mat_nearly_equal(const cv::Mat & lhs, const cv::Mat & rhs, double eps = 1e-9)
{
  if (lhs.empty() || rhs.empty()) {
    return lhs.empty() && rhs.empty();
  }
  if (lhs.size() != rhs.size() || lhs.type() != rhs.type()) {
    return false;
  }
  return cv::norm(lhs, rhs, cv::NORM_INF) <= eps;
}

std::pair<double, double> radial_tangential_error(
  const cv::Point2f & predicted,
  const cv::Point2f & actual,
  const cv::Point2f & r_center)
{
  if (!is_finite_point(predicted) || !is_finite_point(actual) || !is_finite_point(r_center)) {
    return {nan_value(), nan_value()};
  }
  const cv::Point2f radial = actual - r_center;
  const double radius = cv::norm(radial);
  if (!std::isfinite(radius) || radius < 1e-3) {
    return {nan_value(), nan_value()};
  }
  const cv::Point2f radial_unit(
    static_cast<float>(radial.x / radius),
    static_cast<float>(radial.y / radius));
  const cv::Point2f tangent_unit(-radial_unit.y, radial_unit.x);
  const cv::Point2f error = predicted - actual;
  return {
    static_cast<double>(error.x * radial_unit.x + error.y * radial_unit.y),
    static_cast<double>(error.x * tangent_unit.x + error.y * tangent_unit.y)};
}

void reset_pnp_debug(auto_buff::FanBlade & blade)
{
  blade.pnp_observed_points.clear();
  blade.pnp_input_reprojected_points.clear();
  blade.pnp_reprojected_points.clear();
  blade.pnp_point_errors_px.clear();
  blade.pnp_model_center = cv::Point2f(nan_float(), nan_float());
  blade.pnp_reproj_error_px = nan_value();
  blade.pnp_score = nan_value();
  blade.pnp_model_center_error_px = nan_value();
  blade.pnp_model_center_radial_error_px = nan_value();
  blade.pnp_model_center_tangent_error_px = nan_value();
  blade.pnp_method = -1;
  blade.pnp_order = {0, 1, 2, 3};
}

void reset_solution(auto_buff::PowerRune & rune)
{
  rune.xyz_in_world.setZero();
  rune.ypd_in_world.setZero();
  rune.ypr_in_world.setZero();
  rune.blade_xyz_in_world.setZero();
  rune.blade_ypd_in_world.setZero();
}

void reset_blade_solution(auto_buff::FanBlade & blade)
{
  blade.solved = false;
  blade.rune_xyz_in_world.setZero();
  blade.rune_ypd_in_world.setZero();
  blade.ypr_in_world.setZero();
  blade.blade_xyz_in_world.setZero();
  blade.blade_ypd_in_world.setZero();
  reset_pnp_debug(blade);
}

double reprojection_error(
  const std::vector<cv::Point3f> & object_points,
  const std::vector<cv::Point2f> & image_points,
  const cv::Mat & camera_matrix,
  const cv::Mat & distort_coeffs,
  const cv::Vec3d & rvec,
  const cv::Vec3d & tvec,
  auto_buff::BuffPnpCostSample * cost,
  std::vector<cv::Point2f> * projected_scratch = nullptr)
{
  if (cost != nullptr) {
    ++cost->reprojection_calls;
    ++cost->project_points_calls;
  }
  thread_local std::vector<cv::Point2f> thread_projected_points;
  std::vector<cv::Point2f> & projected_points = projected_scratch != nullptr
    ? *projected_scratch : thread_projected_points;
  projected_points.clear();
  if (projected_points.capacity() < image_points.size()) {
    projected_points.reserve(image_points.size());
  }
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, distort_coeffs, projected_points);
  if (projected_points.size() != image_points.size()) {
    return std::numeric_limits<double>::max();
  }

  double error_sum = 0.0;
  for (size_t i = 0; i < image_points.size(); ++i) {
    error_sum += cv::norm(projected_points[i] - image_points[i]);
  }
  return error_sum / static_cast<double>(image_points.size());
}

double rotation_distance(const cv::Vec3d & rvec_a, const cv::Vec3d & rvec_b)
{
  cv::Mat R_a;
  cv::Mat R_b;
  cv::Rodrigues(rvec_a, R_a);
  cv::Rodrigues(rvec_b, R_b);
  const cv::Mat R_delta = R_a * R_b.t();
  const double trace =
    R_delta.at<double>(0, 0) + R_delta.at<double>(1, 1) + R_delta.at<double>(2, 2);
  const double cos_theta = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(cos_theta);
}

double mean_blade_radius(const auto_buff::FanBlade & blade)
{
  const size_t point_count = std::min<size_t>(4, blade.points.size());
  if (point_count == 0) {
    return 0.0;
  }

  double radius_sum = 0.0;
  for (size_t i = 0; i < point_count; ++i) {
    radius_sum += cv::norm(blade.points[i] - blade.center);
  }
  return radius_sum / static_cast<double>(point_count);
}

const std::vector<PointPermutation> & point_permutations()
{
  static const std::vector<PointPermutation> permutations = {
    {0, 1, 2, 3},
    {1, 2, 3, 0},
    {2, 3, 0, 1},
    {3, 0, 1, 2},
    {0, 3, 2, 1},
    {3, 2, 1, 0},
    {2, 1, 0, 3},
    {1, 0, 3, 2},
  };
  return permutations;
}

bool solve_with_best_permutation(
  const std::vector<cv::Point3f> & object_points,
  const std::vector<cv::Point2f> & raw_image_points,
  const cv::Point2f & r_center,
  const cv::Mat & camera_matrix,
  const cv::Mat & distort_coeffs,
  bool last_pose_valid,
  const cv::Vec3d & last_rvec,
  const cv::Vec3d & last_tvec,
  const PointPermutation & last_order,
  cv::Vec3d & best_rvec,
  cv::Vec3d & best_tvec,
  PointPermutation & best_order,
  double & best_error,
  double & best_score,
  int & best_method,
  auto_buff::BuffPnpCostSample * cost)
{
  best_error = std::numeric_limits<double>::max();
  best_score = std::numeric_limits<double>::max();
  best_order = {0, 1, 2, 3};
  best_method = -1;
  bool found = false;

  for (const auto & order : point_permutations()) {
    const auto order_begin = CostClock::now();
    std::vector<cv::Point2f> image_points;
    image_points.reserve(5);
    for (int idx : order) {
      image_points.push_back(raw_image_points[idx]);
    }
    image_points.push_back(r_center);
    if (cost != nullptr) {
      cost->fallback_order_build_ns += cost_ns(order_begin);
    }

    for (int method : {cv::SOLVEPNP_IPPE, cv::SOLVEPNP_ITERATIVE}) {
      cv::Vec3d rvec;
      cv::Vec3d tvec;
      const auto solve_begin = CostClock::now();
      if (cost != nullptr) {
        ++cost->fallback_pnp_calls;
      }
      const bool solved = cv::solvePnP(
        object_points, image_points, camera_matrix, distort_coeffs, rvec, tvec, false, method);
      if (cost != nullptr) {
        cost->fallback_solvepnp_ns += cost_ns(solve_begin);
        if (solved) {
          ++cost->fallback_solve_successes;
        }
      }

      if (!solved || !is_valid_tvec(tvec)) {
        if (cost != nullptr && solved) {
          ++cost->invalid_tvec_rejects;
        }
        continue;
      }
      if (cost != nullptr) {
        ++cost->fallback_valid_tvec;
      }

      const auto score_begin = CostClock::now();
      const double error =
        reprojection_error(
          object_points, image_points, camera_matrix, distort_coeffs, rvec, tvec, cost);
      double score = error;
      if (last_pose_valid) {
        score += 35.0 * cv::norm(tvec - last_tvec);
        score += 6.0 * rotation_distance(rvec, last_rvec);
        if (order != last_order) {
          score += 1.5;
        }
      }
      if (cost != nullptr) {
        ++cost->candidate_score_evaluations;
        cost->fallback_reprojection_score_ns += cost_ns(score_begin);
      }

      if (score < best_score) {
        best_score = score;
        best_error = error;
        best_rvec = rvec;
        best_tvec = tvec;
        best_order = order;
        best_method = method;
        found = true;
      }
    }
  }

  return found;
}

bool solve_with_candidate_orders(
  const std::vector<cv::Point3f> & object_points,
  const std::vector<cv::Point2f> & raw_image_points,
  const cv::Point2f & r_center,
  const cv::Mat & camera_matrix,
  const cv::Mat & distort_coeffs,
  bool last_pose_valid,
  const cv::Vec3d & last_rvec,
  const cv::Vec3d & last_tvec,
  const PointPermutation & last_order,
  const std::vector<PointPermutation> & candidate_orders,
  const std::vector<int> & candidate_methods,
  bool use_last_pose_as_guess,
  cv::Vec3d & best_rvec,
  cv::Vec3d & best_tvec,
  PointPermutation & best_order,
  double & best_error,
  double & best_score,
  int & best_method,
  auto_buff::BuffPnpCostSample * cost)
{
  best_error = std::numeric_limits<double>::max();
  best_score = std::numeric_limits<double>::max();
  best_order = {0, 1, 2, 3};
  best_method = -1;
  bool found = false;

  if (cost != nullptr) {
    cost->fast_unique_orders += static_cast<std::uint32_t>(candidate_orders.size());
  }

  for (const auto & order : candidate_orders) {
    const auto order_begin = CostClock::now();
    std::vector<cv::Point2f> image_points;
    image_points.reserve(5);
    for (int idx : order) {
      image_points.push_back(raw_image_points[idx]);
    }
    image_points.push_back(r_center);
    if (cost != nullptr) {
      cost->fast_order_build_ns += cost_ns(order_begin);
    }

    for (int method : candidate_methods) {
      cv::Vec3d rvec;
      cv::Vec3d tvec;
      bool use_guess = false;
      if (use_last_pose_as_guess && last_pose_valid && method == cv::SOLVEPNP_ITERATIVE) {
        rvec = last_rvec;
        tvec = last_tvec;
        use_guess = true;
      }

      const auto solve_begin = CostClock::now();
      if (cost != nullptr) {
        ++cost->fast_pnp_calls;
      }
      const bool solved = cv::solvePnP(
        object_points, image_points, camera_matrix, distort_coeffs, rvec, tvec, use_guess, method);
      if (cost != nullptr) {
        cost->fast_solvepnp_ns += cost_ns(solve_begin);
        if (solved) {
          ++cost->fast_solve_successes;
        }
      }

      if (!solved || !is_valid_tvec(tvec)) {
        if (cost != nullptr && solved) {
          ++cost->invalid_tvec_rejects;
        }
        continue;
      }
      if (cost != nullptr) {
        ++cost->fast_valid_tvec;
      }

      const auto score_begin = CostClock::now();
      const double error =
        reprojection_error(
          object_points, image_points, camera_matrix, distort_coeffs, rvec, tvec, cost);
      double score = error;
      if (last_pose_valid) {
        score += 35.0 * cv::norm(tvec - last_tvec);
        score += 6.0 * rotation_distance(rvec, last_rvec);
        if (order != last_order) {
          score += 1.5;
        }
      }
      if (cost != nullptr) {
        ++cost->candidate_score_evaluations;
        cost->fast_reprojection_score_ns += cost_ns(score_begin);
      }

      if (score < best_score) {
        best_score = score;
        best_error = error;
        best_rvec = rvec;
        best_tvec = tvec;
        best_order = order;
        best_method = method;
        found = true;
      }
    }
  }

  return found;
}

std::vector<PointPermutation> fast_candidate_orders(
  bool prior_pose_valid,
  const PointPermutation & prior_order)
{
  std::vector<PointPermutation> orders;
  if (prior_pose_valid) {
    orders.push_back(prior_order);
  }
  const PointPermutation raw_order = {0, 1, 2, 3};
  if (std::find(orders.begin(), orders.end(), raw_order) == orders.end()) {
    orders.push_back(raw_order);
  }
  return orders;
}

void store_pnp_debug(
  auto_buff::FanBlade & blade,
  const std::vector<cv::Point3f> & object_points,
  const std::vector<cv::Point3f> & full_object_points,
  const std::vector<cv::Point2f> & raw_image_points,
  const cv::Point2f & r_center,
  const cv::Mat & camera_matrix,
  const cv::Mat & distort_coeffs,
  const cv::Vec3d & rvec,
  const cv::Vec3d & tvec,
  const PointPermutation & order,
  double reproj_error,
  double score,
  int method,
  auto_buff::BuffPnpCostSample * cost)
{
  reset_pnp_debug(blade);
  blade.pnp_reproj_error_px = reproj_error;
  blade.pnp_score = score;
  blade.pnp_method = method;
  blade.pnp_order = order;

  blade.pnp_observed_points.reserve(5);
  for (int idx : order) {
    if (idx >= 0 && static_cast<size_t>(idx) < raw_image_points.size()) {
      blade.pnp_observed_points.push_back(raw_image_points[idx]);
    } else {
      blade.pnp_observed_points.emplace_back(nan_float(), nan_float());
    }
  }
  blade.pnp_observed_points.push_back(r_center);

  if (cost != nullptr) {
    cost->project_points_calls += 2;
  }
  cv::projectPoints(
    object_points, rvec, tvec, camera_matrix, distort_coeffs, blade.pnp_input_reprojected_points);
  cv::projectPoints(
    full_object_points, rvec, tvec, camera_matrix, distort_coeffs, blade.pnp_reprojected_points);

  const size_t point_count =
    std::min(blade.pnp_observed_points.size(), blade.pnp_input_reprojected_points.size());
  blade.pnp_point_errors_px.reserve(point_count);
  for (size_t i = 0; i < point_count; ++i) {
    if (is_finite_point(blade.pnp_observed_points[i]) &&
        is_finite_point(blade.pnp_input_reprojected_points[i])) {
      blade.pnp_point_errors_px.push_back(
        cv::norm(blade.pnp_observed_points[i] - blade.pnp_input_reprojected_points[i]));
    } else {
      blade.pnp_point_errors_px.push_back(nan_value());
    }
  }

  if (blade.pnp_reprojected_points.size() > 4) {
    blade.pnp_model_center = blade.pnp_reprojected_points[4];
    if (is_finite_point(blade.pnp_model_center) && is_finite_point(blade.center)) {
      blade.pnp_model_center_error_px = cv::norm(blade.pnp_model_center - blade.center);
      const auto radial_tangent =
        radial_tangential_error(blade.pnp_model_center, blade.center, r_center);
      blade.pnp_model_center_radial_error_px = radial_tangent.first;
      blade.pnp_model_center_tangent_error_px = radial_tangent.second;
    }
  }
}

int point_permutation_index(const PointPermutation & order)
{
  const auto & permutations = point_permutations();
  const auto iter = std::find(permutations.begin(), permutations.end(), order);
  return iter == permutations.end()
    ? -1
    : static_cast<int>(std::distance(permutations.begin(), iter));
}
}

namespace auto_buff
{

// ==================== 构造函数 ====================
Solver::Solver(const std::string & config_path)
  : config_path_(config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  // 加载外参 (相机 -> 云台)
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  // 加载外参 (云台 -> IMU Body, 通常是单位矩阵)
  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());

  // 内参跟随主参数文件，避免 buff_config.yaml 里复制的标定和主链路不一致。
  params_ = std::make_unique<Params>();
  refresh_intrinsics_from_params(true);
}

void Solver::reset()
{
  rvec_ = cv::Vec3d{};
  tvec_ = cv::Vec3d{};
  last_pose_valid_ = false;
  last_rvec_ = cv::Vec3d{};
  last_tvec_ = cv::Vec3d{};
  last_point_order_ = {0, 1, 2, 3};
}

void Solver::refresh_intrinsics_from_params(bool force, BuffPnpCostSample* cost) const
{
  if (!params_) {
    return;
  }
  if (!force) {
    if (cost != nullptr) {
      ++cost->params_reload_checks;
    }
    if (!params_->reload()) {
      return;
    }
    if (cost != nullptr) {
      ++cost->params_reloads;
    }
  }
  if (params_->CAMERA_MATRIX.empty() || params_->RADIAL_DISTORTION.empty()) {
    throw std::runtime_error("Buff solver requires CAMERA_MATRIX and RADIAL_DISTORTION in param.yaml");
  }

  cv::Mat camera_matrix;
  cv::Mat distort_coeffs;
  params_->CAMERA_MATRIX.convertTo(camera_matrix, CV_64F);
  params_->RADIAL_DISTORTION.convertTo(distort_coeffs, CV_64F);
  base_camera_matrix_ = camera_matrix.clone();
  distort_coeffs_ = distort_coeffs.clone();
  const BuffRuntimeConfig buff_config = load_buff_runtime_config(config_path_);
  camera_matrix.at<double>(0, 2) += buff_config.correction_cx;
  camera_matrix.at<double>(1, 2) += buff_config.correction_cy;
  const bool intrinsics_changed =
    force ||
    !mat_nearly_equal(base_camera_matrix_, camera_matrix) ||
    !mat_nearly_equal(distort_coeffs_, distort_coeffs) ||
    camera_upside_down_ != params_->CAMERA_UPSIDE_DOWN;
  if (!intrinsics_changed) {
    return;
  }
  if (cost != nullptr) {
    ++cost->intrinsics_changes;
  }

  ++calibration_version_;

  base_camera_matrix_ = camera_matrix.clone();
  distort_coeffs_ = distort_coeffs.clone();
  camera_matrix_ = base_camera_matrix_.clone();
  camera_upside_down_ = params_->CAMERA_UPSIDE_DOWN;
  last_image_size_ = cv::Size();
  tools::logger()->info(
    "Buff solver intrinsics from param.yaml/buff_config.yaml -> fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}, buff_cx={:.2f}, buff_cy={:.2f}, upside_down={}",
    base_camera_matrix_.at<double>(0, 0),
    base_camera_matrix_.at<double>(1, 1),
    base_camera_matrix_.at<double>(0, 2),
    base_camera_matrix_.at<double>(1, 2),
    buff_config.correction_cx,
    buff_config.correction_cy,
    camera_upside_down_);
}

void Solver::set_image_size(const cv::Size& image_size, BuffPnpCostSample* cost) const
{
  const auto begin = CostClock::now();
  if (cost != nullptr) {
    ++cost->set_image_size_calls;
  }
  refresh_intrinsics_from_params(false, cost);
  if (base_camera_matrix_.empty()) {
    if (cost != nullptr) {
      cost->set_image_size_ns += cost_ns(begin);
    }
    return;
  }
  camera_matrix_ = base_camera_matrix_.clone();
  if (
    camera_upside_down_ &&
    image_size.width > 0 &&
    image_size.height > 0) {
    camera_matrix_.at<double>(0, 2) =
      static_cast<double>(image_size.width - 1) - base_camera_matrix_.at<double>(0, 2);
    camera_matrix_.at<double>(1, 2) =
      static_cast<double>(image_size.height - 1) - base_camera_matrix_.at<double>(1, 2);
  }
  if (image_size != last_image_size_) {
    if (cost != nullptr) {
      ++cost->image_size_changes;
    }
    last_image_size_ = image_size;
    ++calibration_version_;
    tools::logger()->info(
      "Buff solver active intrinsics -> image={}x{}, fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
      image_size.width,
      image_size.height,
      camera_matrix_.at<double>(0, 0),
      camera_matrix_.at<double>(1, 1),
      camera_matrix_.at<double>(0, 2),
      camera_matrix_.at<double>(1, 2));
  }
  if (cost != nullptr) {
    cost->set_image_size_ns += cost_ns(begin);
  }
}

SolverFrameContext Solver::makeFrameContext(const cv::Size &image_size,
                                            BuffPnpCostSample *cost) const {
  set_image_size(image_size, cost);
  SolverFrameContext frame;
  frame.calibration_version = calibration_version_;
  frame.camera_matrix = camera_matrix_.clone();
  frame.distort_coeffs = distort_coeffs_.clone();
  frame.R_camera2gimbal = R_camera2gimbal_;
  frame.t_camera2gimbal = t_camera2gimbal_;
  frame.R_gimbal2world = R_gimbal2world_;
  return frame;
}

SolverPriorSnapshot Solver::priorSnapshot() const {
  SolverPriorSnapshot prior;
  prior.valid = last_pose_valid_;
  prior.rvec = last_rvec_;
  prior.tvec = last_tvec_;
  prior.order = last_point_order_;
  return prior;
}

void Solver::applyPrior(const SolverPriorSnapshot &prior) {
  last_pose_valid_ = prior.valid;
  last_rvec_ = prior.rvec;
  last_tvec_ = prior.tvec;
  last_point_order_ = prior.order;
  if (prior.valid) {
    rvec_ = prior.rvec;
    tvec_ = prior.tvec;
  }
}

// ==================== 更新姿态 ====================
void Solver::set_R_gimbal2world(const Eigen::Quaterniond &q) {
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  // 转换链：Gimbal -> IMU Body -> World
  R_gimbal2world_ =
      R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

// ==================== PnP 解算 ====================
ExhaustivePnpProposal Solver::buildExhaustiveProposal(
    const std::vector<SolverPnpHypothesis> &hypotheses,
    const SolverFrameContext &frame, BuffPnpCostSample *cost) const {
  constexpr std::uint32_t kMaxRawPnpSolutions = 64;
  ExhaustivePnpProposal proposal;
  proposal.calibration_version = frame.calibration_version;
  proposal.hypotheses.reserve(hypotheses.size());
  if (frame.camera_matrix.empty() || frame.distort_coeffs.empty() ||
      frame.camera_matrix.type() != CV_64F ||
      frame.distort_coeffs.type() != CV_64F ||
      !cv::checkRange(frame.camera_matrix) ||
      !cv::checkRange(frame.distort_coeffs)) {
    proposal.status = ExhaustivePnpStatus::InvalidContext;
    return proposal;
  }

  std::uint64_t expected = 0;
  for (std::size_t hypothesis_position = 0;
       hypothesis_position < hypotheses.size(); ++hypothesis_position) {
    const auto &hypothesis = hypotheses[hypothesis_position];
    if (!is_finite_point(hypothesis.r_center)) {
      proposal.status = ExhaustivePnpStatus::InvalidInput;
      proposal.solutions.clear();
      return proposal;
    }
    for (std::size_t prior = 0; prior < hypothesis_position; ++prior) {
      if (hypotheses[prior].hypothesis_index == hypothesis.hypothesis_index) {
        proposal.status = ExhaustivePnpStatus::InvalidInput;
        proposal.solutions.clear();
        return proposal;
      }
    }
    proposal.hypotheses.push_back(ExhaustivePnpHypothesisMetadata{
        hypothesis.hypothesis_index, hypothesis.source_pixel_signature,
        hypothesis.r_center,
        static_cast<std::uint32_t>(hypothesis.blades().size())});
    for (const auto &blade : hypothesis.blades()) {
      if (blade.points.size() >= 4) {
        for (std::size_t point = 0; point < 4; ++point) {
          if (!is_finite_point(blade.points[point])) {
            proposal.status = ExhaustivePnpStatus::InvalidInput;
            proposal.solutions.clear();
            return proposal;
          }
        }
        expected += point_permutations().size() * 2U;
      }
    }
  }
  proposal.expected_solution_count =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          expected, std::numeric_limits<std::uint32_t>::max()));
  if (expected > kMaxRawPnpSolutions) {
    proposal.status = ExhaustivePnpStatus::CapacityExceeded;
    return proposal;
  }

  std::vector<cv::Point3f> object_points;
  object_points.reserve(5);
  object_points.insert(object_points.end(), OBJECT_POINTS.begin(), OBJECT_POINTS.begin() + 4);
  object_points.push_back(OBJECT_POINTS[5]);
  std::vector<cv::Point2f> image_points(5);
  proposal.solutions.reserve(static_cast<std::size_t>(expected));
  for (const auto &hypothesis : hypotheses) {
    for (std::size_t blade_index = 0; blade_index < hypothesis.blades().size();
         ++blade_index) {
      const auto &blade = hypothesis.blades()[blade_index];
      if (blade.points.size() < 4) {
        continue;
      }
      const std::array<cv::Point2f, 4> raw_points{
          blade.points[0], blade.points[1], blade.points[2], blade.points[3]};
      const auto &orders = point_permutations();
      for (std::size_t order_index = 0; order_index < orders.size();
           ++order_index) {
        const auto order_begin = CostClock::now();
        const auto &order = orders[order_index];
        std::size_t image_index = 0;
        for (int idx : order) {
          image_points[image_index++] = raw_points[static_cast<std::size_t>(idx)];
        }
        image_points[4] = hypothesis.r_center;
        if (cost != nullptr) {
          cost->fallback_order_build_ns += cost_ns(order_begin);
        }
        for (int method : {cv::SOLVEPNP_IPPE, cv::SOLVEPNP_ITERATIVE}) {
          RawPnpSolution raw;
          raw.hypothesis_index = hypothesis.hypothesis_index;
          raw.blade_index = static_cast<std::uint32_t>(blade_index);
          raw.order_index = static_cast<std::uint8_t>(order_index);
          raw.method = method;
          raw.order = order;
          raw.raw_image_points = raw_points;
          const auto solve_begin = CostClock::now();
          if (cost != nullptr) {
            ++cost->fallback_pnp_calls;
          }
          raw.solve_returned = cv::solvePnP(
              object_points, image_points, frame.camera_matrix,
              frame.distort_coeffs, raw.rvec, raw.tvec, false, method);
          if (cost != nullptr) {
            cost->fallback_solvepnp_ns += cost_ns(solve_begin);
            if (raw.solve_returned) {
              ++cost->fallback_solve_successes;
            }
          }
          raw.valid_tvec = raw.solve_returned && is_valid_tvec(raw.tvec);
          if (raw.valid_tvec) {
            if (cost != nullptr) {
              ++cost->fallback_valid_tvec;
            }
            const auto score_begin = CostClock::now();
            raw.reprojection_error_px = reprojection_error(
                object_points, image_points, frame.camera_matrix,
                frame.distort_coeffs, raw.rvec, raw.tvec, cost);
            if (cost != nullptr) {
              cost->fallback_reprojection_score_ns += cost_ns(score_begin);
            }
          } else if (cost != nullptr && raw.solve_returned) {
            ++cost->invalid_tvec_rejects;
          }
          proposal.solutions.push_back(std::move(raw));
        }
      }
    }
  }
  proposal.status = ExhaustivePnpStatus::Ready;
  return proposal;
}

PnpReductionResult Solver::reduceProposal(
    PowerRune *rune, std::uint32_t selected_hypothesis,
    const ExhaustivePnpProposal &proposal, const SolverFrameContext &frame,
    const SolverPriorSnapshot &current_prior, BuffPnpCostSample *cost) const {
  const auto reduce_begin = CostClock::now();
  PnpReductionResult result;
  result.prior_after = current_prior;
  if (rune == nullptr || proposal.status != ExhaustivePnpStatus::Ready ||
      proposal.calibration_version != frame.calibration_version ||
      frame.camera_matrix.empty() || frame.distort_coeffs.empty()) {
    return result;
  }
  const auto metadata = std::find_if(
      proposal.hypotheses.begin(), proposal.hypotheses.end(),
      [selected_hypothesis](const ExhaustivePnpHypothesisMetadata &value) {
        return value.hypothesis_index == selected_hypothesis;
      });
  if (metadata == proposal.hypotheses.end() ||
      !exact_point_equal(metadata->r_center, rune->r_center) ||
      metadata->fanblade_count != rune->fanblades.size()) {
    return result;
  }

  // Prove complete fixed-R coverage before touching output state.  Missing,
  // reordered, or differently sourced inputs request whole-frame legacy solve.
  for (std::size_t blade_index = 0; blade_index < rune->fanblades.size();
       ++blade_index) {
    const auto &blade = rune->fanblades[blade_index];
    if (blade.points.size() < 4) {
      continue;
    }
    const std::vector<cv::Point2f> points(blade.points.begin(),
                                          blade.points.begin() + 4);
    std::size_t entry = 0;
    for (const auto &raw : proposal.solutions) {
      if (raw.hypothesis_index != selected_hypothesis ||
          raw.blade_index != blade_index) {
        continue;
      }
      const std::size_t order_index = entry / 2;
      const int method =
          entry % 2 == 0 ? cv::SOLVEPNP_IPPE : cv::SOLVEPNP_ITERATIVE;
      if (order_index >= point_permutations().size() ||
          raw.order_index != order_index ||
          raw.order != point_permutations()[order_index] ||
          raw.method != method ||
          !exact_raw_points_equal(raw.raw_image_points, points)) {
        return result;
      }
      ++entry;
    }
    if (entry != point_permutations().size() * 2U) {
      return result;
    }
  }

  result.applicable = true;
  if (cost != nullptr) {
    cost->total_blades += static_cast<std::uint32_t>(rune->fanblades.size());
    cost->prior_pose_valid = current_prior.valid;
  }
  reset_solution(*rune);
  for (auto &blade : rune->fanblades) {
    reset_blade_solution(blade);
  }
  std::vector<cv::Point3f> object_points;
  object_points.insert(object_points.end(), OBJECT_POINTS.begin(),
                       OBJECT_POINTS.begin() + 4);
  object_points.push_back(OBJECT_POINTS[5]);

  int primary_index = -1;
  RawPnpSolution primary_raw;
  double primary_score = std::numeric_limits<double>::max();
  int primary_source = 0;
  for (std::size_t blade_index = 0; blade_index < rune->fanblades.size();
       ++blade_index) {
    auto &blade = rune->fanblades[blade_index];
    if (blade.points.size() < 4) {
      continue;
    }
    if (cost != nullptr) {
      ++cost->eligible_blades;
    }
    const std::vector<cv::Point2f> points(blade.points.begin(),
                                          blade.points.begin() + 4);
    const double reprojection_gate =
        std::clamp(mean_blade_radius(blade) * 0.35, 8.0, 28.0);
    RawPnpSolution selected;
    double selected_score = std::numeric_limits<double>::max();
    bool solved = false;
    int selected_source = 0;

    if (current_prior.valid) {
      const auto order_begin = CostClock::now();
      const auto orders = fast_candidate_orders(true, current_prior.order);
      if (cost != nullptr) {
        cost->fast_order_build_ns += cost_ns(order_begin);
      }
      cv::Vec3d rvec;
      cv::Vec3d tvec;
      BuffPointOrder order{0, 1, 2, 3};
      double error = std::numeric_limits<double>::max();
      int method = -1;
      static const std::vector<int> methods{cv::SOLVEPNP_ITERATIVE};
      solved = solve_with_candidate_orders(
          object_points, points, rune->r_center, frame.camera_matrix,
          frame.distort_coeffs, true, current_prior.rvec, current_prior.tvec,
          current_prior.order, orders, methods, true, rvec, tvec, order, error,
          selected_score, method, cost);
      if (solved) {
        selected.hypothesis_index = selected_hypothesis;
        selected.blade_index = static_cast<std::uint32_t>(blade_index);
        selected.order = order;
        selected.order_index =
            static_cast<std::uint8_t>(point_permutation_index(order));
        selected.method = method;
        selected.solve_returned = true;
        selected.valid_tvec = is_valid_tvec(tvec);
        std::copy_n(points.begin(), 4, selected.raw_image_points.begin());
        selected.rvec = rvec;
        selected.tvec = tvec;
        selected.reprojection_error_px = error;
      }
      if (solved && error <= reprojection_gate) {
        selected_source = 1;
        if (cost != nullptr) {
          ++cost->fast_reprojection_gate_passes;
        }
      }
    }

    if (!solved || selected.reprojection_error_px > reprojection_gate) {
      if (cost != nullptr) {
        ++cost->fallback_invocations;
        if (current_prior.valid && !solved) {
          ++cost->fallback_fast_failed;
        }
        if (current_prior.valid && solved &&
            selected.reprojection_error_px > reprojection_gate) {
          ++cost->fallback_fast_error_gate;
        }
      }
      solved = false;
      selected_score = std::numeric_limits<double>::max();
      for (const auto &raw : proposal.solutions) {
        if (raw.hypothesis_index != selected_hypothesis ||
            raw.blade_index != blade_index || !raw.solve_returned ||
            !raw.valid_tvec) {
          continue;
        }
        const auto score_begin = CostClock::now();
        double score = raw.reprojection_error_px;
        if (current_prior.valid) {
          score += 35.0 * cv::norm(raw.tvec - current_prior.tvec);
          score += 6.0 * rotation_distance(raw.rvec, current_prior.rvec);
          if (raw.order != current_prior.order) {
            score += 1.5;
          }
        }
        if (cost != nullptr) {
          ++cost->candidate_score_evaluations;
          cost->fallback_reprojection_score_ns += cost_ns(score_begin);
        }
        if (score < selected_score) {
          selected = raw;
          selected_score = score;
          solved = true;
        }
      }
      selected_source = solved ? 2 : 0;
    }

    const auto reject_begin = CostClock::now();
    if (!solved || !selected.valid_tvec) {
      if (cost != nullptr) {
        cost->solution_reject_gates_ns += cost_ns(reject_begin);
      }
      continue;
    }
    if (selected.reprojection_error_px > reprojection_gate) {
      if (cost != nullptr) {
        ++cost->reprojection_gate_rejects;
        cost->solution_reject_gates_ns += cost_ns(reject_begin);
      }
      continue;
    }
    if (cost != nullptr) {
      cost->solution_reject_gates_ns += cost_ns(reject_begin);
    }

    const auto debug_begin = CostClock::now();
    store_pnp_debug(blade, object_points, OBJECT_POINTS, points, rune->r_center,
                    frame.camera_matrix, frame.distort_coeffs, selected.rvec,
                    selected.tvec, selected.order,
                    selected.reprojection_error_px, selected_score,
                    selected.method, cost);
    if (cost != nullptr) {
      cost->debug_materialization_ns += cost_ns(debug_begin);
    }

    const auto world_begin = CostClock::now();
    Eigen::Vector3d t_buff2camera;
    cv::cv2eigen(selected.tvec, t_buff2camera);
    cv::Mat rotation_matrix;
    cv::Rodrigues(selected.rvec, rotation_matrix);
    Eigen::Matrix3d R_buff2camera;
    cv::cv2eigen(rotation_matrix, R_buff2camera);
    const Eigen::Vector3d xyz_in_gimbal =
        frame.R_camera2gimbal * t_buff2camera + frame.t_camera2gimbal;
    blade.rune_xyz_in_world = frame.R_gimbal2world * xyz_in_gimbal;
    blade.rune_ypd_in_world = tools::xyz2ypd(blade.rune_xyz_in_world);
    const Eigen::Vector3d blade_in_buff(0, 0, 700e-3);
    const Eigen::Vector3d blade_in_camera =
        R_buff2camera * blade_in_buff + t_buff2camera;
    const Eigen::Vector3d blade_in_gimbal =
        frame.R_camera2gimbal * blade_in_camera + frame.t_camera2gimbal;
    blade.blade_xyz_in_world = frame.R_gimbal2world * blade_in_gimbal;
    blade.blade_ypd_in_world = tools::xyz2ypd(blade.blade_xyz_in_world);
    blade.ypr_in_world = tools::eulers(
        frame.R_gimbal2world * frame.R_camera2gimbal * R_buff2camera, 2, 1, 0);
    if (!blade.rune_xyz_in_world.array().isFinite().all() ||
        !blade.ypr_in_world.array().isFinite().all() ||
        !blade.blade_ypd_in_world.array().isFinite().all()) {
      reset_blade_solution(blade);
      if (cost != nullptr) {
        ++cost->invalid_world_rejects;
        cost->camera_to_world_ns += cost_ns(world_begin);
      }
      continue;
    }
    if (cost != nullptr) {
      cost->camera_to_world_ns += cost_ns(world_begin);
      ++cost->solved_blades;
    }
    blade.solved = true;
    if (primary_index < 0) {
      primary_index = static_cast<int>(blade_index);
      primary_raw = selected;
      primary_score = selected_score;
      primary_source = selected_source;
    }
  }

  if (primary_index < 0) {
    if (cost != nullptr) {
      cost->total_ns += cost_ns(reduce_begin);
    }
    return result;
  }
  const auto commit_begin = CostClock::now();
  const int original_primary_index = primary_index;
  if (primary_index != 0) {
    std::swap(rune->fanblades[0],
              rune->fanblades[static_cast<std::size_t>(primary_index)]);
  }
  const auto &primary_blade = rune->fanblades[0];
  rune->xyz_in_world = primary_blade.rune_xyz_in_world;
  rune->ypd_in_world = primary_blade.rune_ypd_in_world;
  rune->ypr_in_world = primary_blade.ypr_in_world;
  rune->blade_xyz_in_world = primary_blade.blade_xyz_in_world;
  rune->blade_ypd_in_world = primary_blade.blade_ypd_in_world;
  result.solved = true;
  result.used_fast_prior = primary_source == 1;
  result.selected_blade = static_cast<std::uint32_t>(original_primary_index);
  result.selected = primary_raw;
  result.score = primary_score;
  result.prior_after = SolverPriorSnapshot{true, primary_raw.rvec,
                                           primary_raw.tvec, primary_raw.order};
  if (cost != nullptr) {
    cost->solved = true;
    cost->selected_blade = original_primary_index;
    cost->selected_method = primary_raw.method;
    cost->selected_order_index = point_permutation_index(primary_raw.order);
    cost->selected_source = primary_source;
    cost->final_reprojection_error_px = primary_raw.reprojection_error_px;
    cost->final_model_center_error_px = primary_blade.pnp_model_center_error_px;
    cost->primary_commit_ns += cost_ns(commit_begin);
    cost->total_ns += cost_ns(reduce_begin);
  }
  return result;
}

PnpReductionResult
Solver::solveFromProposal(PowerRune &rune, std::uint32_t selected_hypothesis,
                          const ExhaustivePnpProposal &proposal,
                          const SolverFrameContext &frame,
                          BuffPnpCostSample *cost) {
  const SolverPriorSnapshot current_prior = priorSnapshot();
  PnpReductionResult result = reduceProposal(
      &rune, selected_hypothesis, proposal, frame, current_prior, cost);
  if (result.applicable && result.solved) {
    applyPrior(result.prior_after);
  }
  return result;
}

void Solver::solve(PowerRune & rune, BuffPnpCostSample* cost) const
{
  const auto solve_begin = CostClock::now();
  const auto finish = [&]() {
    if (cost != nullptr) {
      cost->total_ns += cost_ns(solve_begin);
    }
  };
  const auto prevalidation_begin = CostClock::now();
  if (cost != nullptr) {
    cost->total_blades += static_cast<std::uint32_t>(rune.fanblades.size());
    cost->prior_pose_valid = last_pose_valid_;
  }
  // 1. 检查输入有效性
  if (rune.fanblades.empty()) {
    reset_solution(rune);
    if (cost != nullptr) {
      cost->prevalidation_object_points_ns += cost_ns(prevalidation_begin);
    }
    finish();
    return;
  }

  if (!is_finite_point(rune.r_center)) {
    tools::logger()->debug("[Solver] Invalid r_center, skip solvePnP");
    reset_solution(rune);
    if (cost != nullptr) {
      cost->prevalidation_object_points_ns += cost_ns(prevalidation_begin);
    }
    finish();
    return;
  }

  for (auto & blade : rune.fanblades) {
    reset_blade_solution(blade);
  }

  // 2. 准备对应的 3D 点
  // 取 OBJECT_POINTS 的前 4 个 (扇叶角点) + 第 6 个 (R 标中心, index 5)
  std::vector<cv::Point3f> object_points_subset;
  object_points_subset.insert(object_points_subset.end(), OBJECT_POINTS.begin(), OBJECT_POINTS.begin() + 4);
  object_points_subset.push_back(OBJECT_POINTS[5]);

  const bool prior_pose_valid = last_pose_valid_;
  const cv::Vec3d prior_rvec = last_rvec_;
  const cv::Vec3d prior_tvec = last_tvec_;
  const PointPermutation prior_order = last_point_order_;
  if (cost != nullptr) {
    cost->prevalidation_object_points_ns += cost_ns(prevalidation_begin);
  }

  int primary_idx = -1;
  cv::Vec3d primary_rvec;
  cv::Vec3d primary_tvec;
  PointPermutation primary_order = {0, 1, 2, 3};
  int primary_method = -1;
  int primary_source = 0;
  double primary_error = 0.0;

  for (size_t blade_idx = 0; blade_idx < rune.fanblades.size(); ++blade_idx) {
    const auto point_prep_begin = CostClock::now();
    auto & blade = rune.fanblades[blade_idx];
    if (blade.points.size() < 4) {
      if (cost != nullptr) {
        cost->fanblade_point_prep_ns += cost_ns(point_prep_begin);
      }
      continue;
    }
    if (cost != nullptr) {
      ++cost->eligible_blades;
    }

    std::vector<cv::Point2f> raw_image_points(blade.points.begin(), blade.points.begin() + 4);
    if (cost != nullptr) {
      cost->fanblade_point_prep_ns += cost_ns(point_prep_begin);
    }
    if (raw_image_points.size() + 1 != object_points_subset.size()) {
      tools::logger()->error("[Solver] Points size mismatch!");
      continue;
    }

    cv::Vec3d blade_rvec;
	    cv::Vec3d blade_tvec;
	    PointPermutation best_order = {0, 1, 2, 3};
	    double best_error = std::numeric_limits<double>::max();
	    double best_score = std::numeric_limits<double>::max();
	    int best_method = -1;
	    const double blade_radius = mean_blade_radius(blade);
	    const double reproj_gate = std::clamp(blade_radius * 0.35, 8.0, 28.0);
	    bool solved = false;
	    int selected_source = 0;

	    if (prior_pose_valid) {
	      static const std::vector<int> fast_methods = {cv::SOLVEPNP_ITERATIVE};
	      const auto order_begin = CostClock::now();
	      const auto orders = fast_candidate_orders(prior_pose_valid, prior_order);
	      if (cost != nullptr) {
	        cost->fast_order_build_ns += cost_ns(order_begin);
	      }
	      solved = solve_with_candidate_orders(
	        object_points_subset,
	        raw_image_points,
	        rune.r_center,
	        camera_matrix_,
	        distort_coeffs_,
	        prior_pose_valid,
	        prior_rvec,
	        prior_tvec,
	        prior_order,
	        orders,
	        fast_methods,
	        true,
	        blade_rvec,
	        blade_tvec,
	        best_order,
	        best_error,
	        best_score,
	        best_method,
	        cost);
	      if (solved && best_error <= reproj_gate) {
	        selected_source = 1;
	        if (cost != nullptr) {
	          ++cost->fast_reprojection_gate_passes;
	        }
	      }
	    }

	    if (!solved || best_error > reproj_gate) {
	      if (cost != nullptr) {
	        ++cost->fallback_invocations;
	        if (prior_pose_valid && !solved) {
	          ++cost->fallback_fast_failed;
	        }
	        if (prior_pose_valid && solved && best_error > reproj_gate) {
	          ++cost->fallback_fast_error_gate;
	        }
	      }
	      solved = solve_with_best_permutation(
	        object_points_subset,
	        raw_image_points,
	        rune.r_center,
	        camera_matrix_,
	        distort_coeffs_,
	        prior_pose_valid,
	        prior_rvec,
	        prior_tvec,
	        prior_order,
	        blade_rvec,
	        blade_tvec,
	        best_order,
	        best_error,
	        best_score,
	        best_method,
	        cost);
	      selected_source = solved ? 2 : 0;
	    }

	    const auto reject_begin = CostClock::now();
	    if (!solved || !is_valid_tvec(blade_tvec)) {
	      tools::logger()->debug("[Solver] solvePnP failed or produced invalid tvec");
	      if (cost != nullptr) {
	        cost->solution_reject_gates_ns += cost_ns(reject_begin);
	      }
	      continue;
	    }

	    if (best_error > reproj_gate) {
	      tools::logger()->debug(
	        "[Solver] Reject unstable solvePnP blade={}, reproj_error={:.2f}, gate={:.2f}, score={:.2f}",
        blade_idx, best_error, reproj_gate, best_score);
      if (cost != nullptr) {
        ++cost->reprojection_gate_rejects;
        cost->solution_reject_gates_ns += cost_ns(reject_begin);
      }
      continue;
    }
    if (cost != nullptr) {
      cost->solution_reject_gates_ns += cost_ns(reject_begin);
    }

    const auto debug_begin = CostClock::now();
    store_pnp_debug(
      blade,
      object_points_subset,
      OBJECT_POINTS,
      raw_image_points,
      rune.r_center,
      camera_matrix_,
      distort_coeffs_,
      blade_rvec,
      blade_tvec,
      best_order,
      best_error,
      best_score,
      best_method,
      cost);
    if (cost != nullptr) {
      cost->debug_materialization_ns += cost_ns(debug_begin);
    }

    if (best_order != PointPermutation{0, 1, 2, 3}) {
      tools::logger()->debug(
        "[Solver] Use permuted point order [{}, {}, {}, {}], blade={}, method={}, reproj_error={:.2f}, score={:.2f}",
        best_order[0], best_order[1], best_order[2], best_order[3], blade_idx, best_method,
        best_error, best_score);
    }

    const auto world_begin = CostClock::now();
    Eigen::Vector3d t_buff2camera;
    cv::cv2eigen(blade_tvec, t_buff2camera);

    cv::Mat rmat;
    cv::Rodrigues(blade_rvec, rmat);
    Eigen::Matrix3d R_buff2camera;
    cv::cv2eigen(rmat, R_buff2camera);

    Eigen::Vector3d xyz_in_camera = t_buff2camera;
    Eigen::Vector3d xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
    blade.rune_xyz_in_world = R_gimbal2world_ * xyz_in_gimbal;
    blade.rune_ypd_in_world = tools::xyz2ypd(blade.rune_xyz_in_world);

    Eigen::Vector3d blade_xyz_in_buff(0, 0, 700e-3);
    Eigen::Vector3d blade_xyz_in_camera = R_buff2camera * blade_xyz_in_buff + t_buff2camera;
    Eigen::Vector3d blade_xyz_in_gimbal = R_camera2gimbal_ * blade_xyz_in_camera + t_camera2gimbal_;
    blade.blade_xyz_in_world = R_gimbal2world_ * blade_xyz_in_gimbal;
    blade.blade_ypd_in_world = tools::xyz2ypd(blade.blade_xyz_in_world);

    Eigen::Matrix3d R_buff2gimbal = R_camera2gimbal_ * R_buff2camera;
    Eigen::Matrix3d R_buff2world = R_gimbal2world_ * R_buff2gimbal;
    blade.ypr_in_world = tools::eulers(R_buff2world, 2, 1, 0);

    if (
      !blade.rune_xyz_in_world.array().isFinite().all() ||
      !blade.ypr_in_world.array().isFinite().all() ||
      !blade.blade_ypd_in_world.array().isFinite().all()) {
      reset_blade_solution(blade);
      if (cost != nullptr) {
        ++cost->invalid_world_rejects;
        cost->camera_to_world_ns += cost_ns(world_begin);
      }
      continue;
    }
    if (cost != nullptr) {
      cost->camera_to_world_ns += cost_ns(world_begin);
      ++cost->solved_blades;
    }

    blade.solved = true;
    if (primary_idx < 0 || blade_idx == 0) {
      primary_idx = static_cast<int>(blade_idx);
      primary_rvec = blade_rvec;
      primary_tvec = blade_tvec;
      primary_order = best_order;
      primary_method = best_method;
      primary_source = selected_source;
      primary_error = best_error;
    }
  }

  if (primary_idx < 0) {
    tools::logger()->debug("[Solver] No valid fanblade solve result");
    reset_solution(rune);
    finish();
    return;
  }

  const auto commit_begin = CostClock::now();
  const int selected_blade = primary_idx;
  if (primary_idx != 0) {
    std::swap(rune.fanblades[0], rune.fanblades[primary_idx]);
  }

  const auto & primary_blade = rune.fanblades[0];
  rune.xyz_in_world = primary_blade.rune_xyz_in_world;
  rune.ypd_in_world = primary_blade.rune_ypd_in_world;
  rune.ypr_in_world = primary_blade.ypr_in_world;
  rune.blade_xyz_in_world = primary_blade.blade_xyz_in_world;
  rune.blade_ypd_in_world = primary_blade.blade_ypd_in_world;

  last_pose_valid_ = true;
  rvec_ = primary_rvec;
  tvec_ = primary_tvec;
  last_rvec_ = primary_rvec;
  last_tvec_ = primary_tvec;
  last_point_order_ = primary_order;
  if (cost != nullptr) {
    cost->solved = true;
    cost->selected_blade = selected_blade;
    cost->selected_method = primary_method;
    cost->selected_order_index = point_permutation_index(primary_order);
    cost->selected_source = primary_source;
    cost->final_reprojection_error_px = primary_error;
    cost->final_model_center_error_px = primary_blade.pnp_model_center_error_px;
    cost->primary_commit_ns += cost_ns(commit_begin);
  }
  finish();
}

// ==================== 辅助投影函数 ====================
cv::Point2f Solver::point_3d_to_pixel(const cv::Point3f& pt_3d) const
{
  std::vector<cv::Point3f> pts3d = { pt_3d };
  std::vector<cv::Point2f> pts2d;
  // 使用当前的 rvec_, tvec_ 进行投影
  cv::projectPoints(pts3d, rvec_, tvec_, camera_matrix_, distort_coeffs_, pts2d);
  return pts2d.empty() ? cv::Point2f(0,0) : pts2d[0];
}

cv::Point2f Solver::get_projected_r_center() const
{
    return point_3d_to_pixel(cv::Point3f(0, 0, 0));
}

// ==================== 重投影 (实现测试代码需求) ====================
std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, double yaw, double roll) const
{
  // 1. 构建 Buff 在世界坐标系下的位姿
  // R 标坐标系定义：Z轴垂直符面，X轴指向某个方向。Roll 决定扇叶转动。
  Eigen::Matrix3d R_buff2world = tools::rotation_matrix(Eigen::Vector3d(yaw, 0.0, roll));
  Eigen::Vector3d t_buff2world = xyz_in_world;

  // 2. 逆向变换链：World -> Gimbal -> Camera -> Buff (求 PnP 的逆过程)
  // 公式推导：
  // T_c2w = T_g2w * T_c2g
  // T_b2c = T_c2w^-1 * T_b2w
  // R_b2c = R_c2w^T * R_b2w
  // t_b2c = R_c2w^T * (t_b2w - t_c2w)

  Eigen::Matrix3d R_cam2world = R_gimbal2world_ * R_camera2gimbal_;
  // 注意：这里假设 Gimbal 原点在 World 系下的平移为 0 (或者忽略底盘移动，只看相对)
  // 如果 xyz_in_world 是相对于 Gimbal 中心的，则 World 原点就是 Gimbal 中心。
  // 在打符测试中通常也是这么假设的。
  Eigen::Vector3d t_cam2world = R_gimbal2world_ * t_camera2gimbal_;

  Eigen::Matrix3d R_buff2camera = R_cam2world.transpose() * R_buff2world;
  Eigen::Vector3d t_buff2camera = R_cam2world.transpose() * (t_buff2world - t_cam2world);

  // 3. 转为 OpenCV 格式
  cv::Vec3d rvec;
  cv::Mat R_cv;
  cv::eigen2cv(R_buff2camera, R_cv);
  cv::Rodrigues(R_cv, rvec);
  cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  // 4. 投影所有 3D 模型点
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);

  return image_points;
}

} // namespace auto_buff
