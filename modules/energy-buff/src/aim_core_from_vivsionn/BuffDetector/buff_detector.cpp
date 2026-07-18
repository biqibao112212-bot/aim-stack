/**
 * @file buff_detector.cpp
 * @brief 能量机关检测器实现
 */

#include "buff_detector.hpp"
#include "tools/logger.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_map>

namespace {
  using CostClock = std::chrono::steady_clock;

  std::uint64_t cost_ns(CostClock::time_point begin, CostClock::time_point end)
  {
    return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
  }

  constexpr int kRTemplateHoldMaxMisses = 5;
  constexpr double kRBinaryAlpha = 0.2;
  constexpr float kRBinaryGamma = 2.0f;

    cv::Mat buildGammaTable(float gamma) {
        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
        }
        return lookUpTable;
    }

    cv::Point2f clamp_point(const cv::Point2f & point, const cv::Size & image_size)
    {
      if (image_size.width <= 0 || image_size.height <= 0) {
        return point;
      }
      return {
        std::clamp(point.x, 0.0f, static_cast<float>(image_size.width - 1)),
        std::clamp(point.y, 0.0f, static_cast<float>(image_size.height - 1))
      };
    }

    bool is_finite_point(const cv::Point2f & point)
    {
      return std::isfinite(point.x) && std::isfinite(point.y);
    }

    bool has_finite_keypoints(
      const auto_buff::YOLO11_BUFF::Object & object,
      size_t required_count)
    {
      if (object.kpt.size() < required_count) {
        return false;
      }
      for (size_t i = 0; i < required_count; ++i) {
        if (!is_finite_point(object.kpt[i])) {
          return false;
        }
      }
      return true;
    }

    bool has_blade_keypoints(const auto_buff::YOLO11_BUFF::Object & object)
    {
      return has_finite_keypoints(object, 4);
    }

    bool has_rune_keypoints(const auto_buff::YOLO11_BUFF::Object & object)
    {
      return has_finite_keypoints(object, 5);
    }

    cv::Mat make_r_template(int size)
    {
      size = std::max(16, size);
      cv::Mat templ = cv::Mat::zeros(size, size, CV_8U);
      const double scale = static_cast<double>(size) / 48.0;
      const int thickness = std::max(2, static_cast<int>(std::lround(6.0 * scale)));
      const int x_left = static_cast<int>(std::lround(14.0 * scale));
      const int x_mid = static_cast<int>(std::lround(28.0 * scale));
      const int x_right = static_cast<int>(std::lround(36.0 * scale));
      const int y_top = static_cast<int>(std::lround(8.0 * scale));
      const int y_mid = static_cast<int>(std::lround(23.0 * scale));
      const int y_bottom = static_cast<int>(std::lround(40.0 * scale));

      cv::line(templ, cv::Point(x_left, y_top), cv::Point(x_left, y_bottom), cv::Scalar(255), thickness, cv::LINE_AA);
      cv::line(templ, cv::Point(x_left, y_top), cv::Point(x_mid, y_top), cv::Scalar(255), thickness, cv::LINE_AA);
      cv::ellipse(
        templ,
        cv::Point(x_mid, static_cast<int>(std::lround(15.0 * scale))),
        cv::Size(
          std::max(3, static_cast<int>(std::lround(10.0 * scale))),
          std::max(3, static_cast<int>(std::lround(8.0 * scale)))),
        0.0,
        -80.0,
        95.0,
        cv::Scalar(255),
        thickness,
        cv::LINE_AA);
      cv::line(templ, cv::Point(x_left, y_mid), cv::Point(x_mid, y_mid), cv::Scalar(255), thickness, cv::LINE_AA);
      cv::line(templ, cv::Point(x_mid, y_mid), cv::Point(x_right, y_bottom), cv::Scalar(255), thickness, cv::LINE_AA);
      cv::threshold(templ, templ, 64, 255, cv::THRESH_BINARY);
      return templ;
    }

    struct RTemplateMatch
    {
      cv::Point2f center;
      cv::Rect rect;
      double score = -1.0;
      int hits = 0;
      bool valid = false;
    };

    RTemplateMatch match_r_template(
      const cv::Mat & masked_img,
      const cv::Point2f & r_prior_roi,
      double radius,
      auto_buff::BuffRSearchCostSample * cost)
    {
      RTemplateMatch best;
      if (masked_img.empty() || masked_img.channels() != 1 || radius <= 1.0) {
        return best;
      }

      const int min_size = std::max(14, static_cast<int>(std::lround(radius * 0.45)));
      const int max_size = std::min(
        std::min(masked_img.cols, masked_img.rows),
        std::max(min_size, static_cast<int>(std::lround(radius * 1.25))));
      if (max_size < min_size) {
        return best;
      }
      if (cost != nullptr) {
        cost->min_template_size = static_cast<std::uint32_t>(min_size);
        cost->max_template_size = static_cast<std::uint32_t>(max_size);
      }

      // The ordered solve thread reuses a small, immutable template set.
      // Thread-local ownership avoids cross-pipeline locking while preserving
      // the exact generated pixels and original scale order.
      thread_local std::unordered_map<int, cv::Mat> template_cache;
      thread_local const bool template_cache_enabled = [] {
        const char * disable = std::getenv("AIM_BUFF_DISABLE_R_TEMPLATE_CACHE");
        return disable == nullptr || std::string(disable) != "1";
      }();

      for (int size = min_size; size <= max_size; size += std::max(2, size / 8)) {
        if (masked_img.cols < size || masked_img.rows < size) continue;

        if (cost != nullptr) {
          ++cost->template_scale_count;
          ++cost->match_template_calls;
          cost->template_result_pixels += static_cast<std::uint64_t>(
            masked_img.cols - size + 1) *
            static_cast<std::uint64_t>(masked_img.rows - size + 1);
        }

        cv::Mat uncached_template;
        const cv::Mat * templ = nullptr;
        if (!template_cache_enabled) {
          uncached_template = make_r_template(size);
          templ = &uncached_template;
          if (cost != nullptr) {
            ++cost->template_builds;
          }
        } else {
          auto template_iter = template_cache.find(size);
          if (template_iter == template_cache.end()) {
            template_iter = template_cache.emplace(size, make_r_template(size)).first;
            if (cost != nullptr) {
              ++cost->template_builds;
            }
          } else if (cost != nullptr) {
            ++cost->template_cache_hits;
          }
          templ = &template_iter->second;
        }
        cv::Mat result;
        cv::matchTemplate(masked_img, *templ, result, cv::TM_CCOEFF_NORMED);

        double min_val = 0.0;
        double max_val = 0.0;
        cv::Point min_loc;
        cv::Point max_loc;
        cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

        const cv::Point2f center(
          static_cast<float>(max_loc.x + size * 0.5),
          static_cast<float>(max_loc.y + size * 0.5));
        const double dist = cv::norm(center - r_prior_roi);
        if (dist > radius) continue;
        if (cost != nullptr) {
          ++cost->template_distance_gate_passes;
        }

        const double distance_penalty = 0.25 * (dist / std::max(radius, 1.0));
        const double score = max_val - distance_penalty;
        if (!best.valid || score > best.score) {
          best.valid = true;
          best.center = center;
          best.rect = cv::Rect(max_loc.x, max_loc.y, size, size);
          best.score = score;
          if (cost != nullptr) {
            cost->winning_template_size = static_cast<std::uint32_t>(size);
            cost->winning_template_score = score;
          }
        }
      }

      if (best.valid) {
        best.hits = 1;
      }
      return best;
    }

    double contour_circularity(const std::vector<cv::Point> & contour)
    {
      const double perimeter = cv::arcLength(contour, true);
      if (perimeter < 1e-3) {
        return 0.0;
      }
      const double area = std::abs(cv::contourArea(contour));
      return 4.0 * CV_PI * area / (perimeter * perimeter);
    }

    double mean_blade_radius(const auto_buff::FanBlade & fanblade)
    {
      const size_t point_count = std::min<size_t>(4, fanblade.points.size());
      if (point_count == 0) {
        return 0.0;
      }

      double radius_sum = 0.0;
      for (size_t i = 0; i < point_count; ++i) {
        radius_sum += cv::norm(fanblade.points[i] - fanblade.center);
      }
      return radius_sum / static_cast<double>(point_count);
    }

    cv::Point2f weighted_average(
      const std::vector<std::pair<cv::Point2f, double>> & points,
      const cv::Point2f & fallback)
    {
      double weight_sum = 0.0;
      cv::Point2f sum(0.0f, 0.0f);
      for (const auto & [point, weight] : points) {
        if (!is_finite_point(point) || weight <= 0.0) {
          continue;
        }
        sum.x += static_cast<float>(point.x * weight);
        sum.y += static_cast<float>(point.y * weight);
        weight_sum += weight;
      }
      if (weight_sum <= 1e-6) {
        return fallback;
      }
      return {sum.x / static_cast<float>(weight_sum), sum.y / static_cast<float>(weight_sum)};
    }

    bool point_inside_image(const cv::Point2f & point, const cv::Size & image_size)
    {
      return is_finite_point(point) &&
             point.x >= 0.0f &&
             point.y >= 0.0f &&
             point.x < static_cast<float>(image_size.width) &&
             point.y < static_cast<float>(image_size.height);
    }

    std::optional<cv::Point2f> pair_geometry_r_center(
      const std::vector<auto_buff::FanBlade> & fanblades,
      const cv::Point2f & reference,
      const cv::Size & image_size,
      double max_reference_dist,
      auto_buff::BuffRSearchCostSample * cost)
    {
      if (fanblades.size() < 2 || !is_finite_point(reference)) {
        return std::nullopt;
      }

      constexpr double kPi = 3.14159265358979323846;
      const std::array<double, 2> candidate_angles = {2.0 * kPi / 5.0, 4.0 * kPi / 5.0};
      double best_score = std::numeric_limits<double>::max();
      cv::Point2f best_center;
      bool found = false;

      for (size_t i = 0; i < fanblades.size(); ++i) {
        const cv::Point2f p1 = fanblades[i].center;
        if (!is_finite_point(p1)) {
          continue;
        }
        for (size_t j = i + 1; j < fanblades.size(); ++j) {
          if (cost != nullptr) {
            ++cost->geometry_blade_pairs;
          }
          const cv::Point2f p2 = fanblades[j].center;
          if (!is_finite_point(p2)) {
            continue;
          }

          const cv::Point2f chord = p2 - p1;
          const double chord_len = cv::norm(chord);
          if (chord_len < 5.0) {
            continue;
          }

          const cv::Point2f mid = (p1 + p2) * 0.5f;
          const cv::Point2f normal(
            static_cast<float>(-chord.y / chord_len),
            static_cast<float>(chord.x / chord_len));

          for (double theta : candidate_angles) {
            const double half_tan = std::tan(theta * 0.5);
            if (std::abs(half_tan) < 1e-6) {
              continue;
            }
            const float h = static_cast<float>(chord_len / (2.0 * half_tan));

            for (float sign : {-1.0f, 1.0f}) {
              if (cost != nullptr) {
                ++cost->geometry_center_hypotheses;
              }
              const cv::Point2f center = mid + normal * (sign * h);
              if (!point_inside_image(center, image_size)) {
                continue;
              }

              const double ref_dist = cv::norm(center - reference);
              if (ref_dist > max_reference_dist) {
                continue;
              }

              const double radius_mean = 0.5 * (cv::norm(center - p1) + cv::norm(center - p2));
              const double radius_balance =
                std::abs(cv::norm(center - p1) - cv::norm(center - p2)) / std::max(radius_mean, 1.0);
              const double score = ref_dist + 15.0 * radius_balance;

              if (score < best_score) {
                best_score = score;
                best_center = center;
                found = true;
              }
            }
          }
        }
      }

      if (!found) {
        return std::nullopt;
      }
      return best_center;
    }

    std::vector<cv::Point2f> raw_pair_geometry_r_centers(
      const std::vector<auto_buff::FanBlade> & fanblades,
      const cv::Size & image_size)
    {
      std::vector<cv::Point2f> centers;
      if (fanblades.size() < 2) return centers;
      constexpr double kPi = 3.14159265358979323846;
      const std::array<double, 2> candidate_angles = {
        2.0 * kPi / 5.0, 4.0 * kPi / 5.0};
      for (size_t i = 0; i < fanblades.size(); ++i) {
        const cv::Point2f p1 = fanblades[i].center;
        if (!is_finite_point(p1)) continue;
        for (size_t j = i + 1; j < fanblades.size(); ++j) {
          const cv::Point2f p2 = fanblades[j].center;
          if (!is_finite_point(p2)) continue;
          const cv::Point2f chord = p2 - p1;
          const double chord_len = cv::norm(chord);
          if (chord_len < 5.0) continue;
          const cv::Point2f mid = (p1 + p2) * 0.5f;
          const cv::Point2f normal(
            static_cast<float>(-chord.y / chord_len),
            static_cast<float>(chord.x / chord_len));
          for (double theta : candidate_angles) {
            const double half_tan = std::tan(theta * 0.5);
            if (std::abs(half_tan) < 1e-6) continue;
            const float h = static_cast<float>(chord_len / (2.0 * half_tan));
            for (float sign : {-1.0f, 1.0f}) {
              const cv::Point2f center = mid + normal * (sign * h);
              if (!point_inside_image(center, image_size)) continue;
              const auto duplicate = std::find_if(
                centers.begin(), centers.end(), [&](const cv::Point2f & existing) {
                  return existing.x == center.x && existing.y == center.y;
                });
              if (duplicate == centers.end()) centers.push_back(center);
            }
          }
        }
      }
      return centers;
    }

    bool is_target_label(int label, int num_classes)
    {
      if (num_classes >= 6) {
        return label == 1 || label == 4;
      }
      return label == 0 || label == 2;
    }

    bool is_hit_label(int label, int num_classes)
    {
      if (num_classes >= 6) {
        return label == 0 || label == 2 || label == 3 || label == 5;
      }
      return label == 1 || label == 3;
    }

    bool is_red_label(int label, int num_classes)
    {
      if (num_classes >= 6) {
        return label >= 0 && label <= 2;
      }
      return label <= 1;
    }

    cv::Point2f blade_center_from_object(const auto_buff::YOLO11_BUFF::Object & object)
    {
      cv::Point2f center(0, 0);
      const size_t point_count = std::min<size_t>(4, object.kpt.size());
      if (point_count == 0) {
        return center;
      }

      for (size_t i = 0; i < point_count; ++i) {
        center += object.kpt[i];
      }
      center *= 1.0f / static_cast<float>(point_count);
      return center;
    }

    double wrapped_angle_diff(double a, double b)
    {
      return std::atan2(std::sin(a - b), std::cos(a - b));
    }

    double blade_angle_around_r(const cv::Point2f & blade_center, const cv::Point2f & r_center)
    {
      return std::atan2(
        static_cast<double>(blade_center.y - r_center.y),
        static_cast<double>(blade_center.x - r_center.x));
    }

    template <typename T>
    void read_optional_scalar(const YAML::Node & node, const char * key, T & value)
    {
      if (node && node[key]) {
        value = node[key].as<T>();
      }
    }
}

namespace auto_buff
{

// ==================== 构造函数 ====================

Buff_Detector::Buff_Detector(const std::string & config)
    : status_(LOSE), lose_(0), MODE_(config)
{
  const auto yaml = YAML::LoadFile(config);
  const auto detector_yaml = yaml["buff_detector"];

  read_optional_scalar(detector_yaml, "r_search_radius_scale", r_search_radius_scale_);
  read_optional_scalar(detector_yaml, "r_search_radius_min", r_search_radius_min_);
  read_optional_scalar(detector_yaml, "r_search_radius_max", r_search_radius_max_);
  read_optional_scalar(detector_yaml, "r_min_area", r_min_area_);
  read_optional_scalar(detector_yaml, "r_max_area", r_max_area_);
  read_optional_scalar(detector_yaml, "r_max_aspect_ratio", r_max_aspect_ratio_);
  read_optional_scalar(detector_yaml, "r_min_circularity", r_min_circularity_);
  read_optional_scalar(detector_yaml, "r_max_accept_ratio", r_max_accept_ratio_);
  read_optional_scalar(detector_yaml, "r_geometry_gate_scale", r_geometry_gate_scale_);
  read_optional_scalar(detector_yaml, "r_geometry_gate_min", r_geometry_gate_min_);
  read_optional_scalar(detector_yaml, "r_geometry_gate_max", r_geometry_gate_max_);
  read_optional_scalar(detector_yaml, "r_yolo_gate_scale", r_yolo_gate_scale_);
  read_optional_scalar(detector_yaml, "r_yolo_gate_min", r_yolo_gate_min_);
  read_optional_scalar(detector_yaml, "r_yolo_gate_max", r_yolo_gate_max_);
  read_optional_scalar(detector_yaml, "r_binary_threshold", r_binary_threshold_);
  read_optional_scalar(
    detector_yaml, "target_switch_missing_timeout_s", target_switch_missing_timeout_s_);

  r_search_radius_scale_ = std::max(0.1, r_search_radius_scale_);
  r_search_radius_min_ = std::max(1.0, r_search_radius_min_);
  r_search_radius_max_ = std::max(r_search_radius_min_, r_search_radius_max_);
  r_min_area_ = std::max(1.0, r_min_area_);
  r_max_area_ = std::max(r_min_area_, r_max_area_);
  r_max_aspect_ratio_ = std::max(1.0, r_max_aspect_ratio_);
  r_min_circularity_ = std::clamp(r_min_circularity_, 0.0, 1.0);
  r_max_accept_ratio_ = std::clamp(r_max_accept_ratio_, 0.1, 2.0);
  r_geometry_gate_scale_ = std::max(0.1, r_geometry_gate_scale_);
  r_geometry_gate_min_ = std::max(1.0, r_geometry_gate_min_);
  r_geometry_gate_max_ = std::max(r_geometry_gate_min_, r_geometry_gate_max_);
  r_yolo_gate_scale_ = std::max(0.1, r_yolo_gate_scale_);
  r_yolo_gate_min_ = std::max(1.0, r_yolo_gate_min_);
  r_yolo_gate_max_ = std::max(r_yolo_gate_min_, r_yolo_gate_max_);
  r_binary_threshold_ = std::clamp(r_binary_threshold_, 0, 255);
  if (!std::isfinite(target_switch_missing_timeout_s_) ||
      target_switch_missing_timeout_s_ < 0.0) {
    target_switch_missing_timeout_s_ = 0.2;
  }

  tools::logger()->info(
    "Buff detector config -> r_radius(scale={:.2f}, min={:.1f}, max={:.1f}) "
    "r_filter(area={:.1f}-{:.1f}, aspect<={:.2f}, circularity>={:.2f}, accept<={:.2f}) "
    "r_prior(geom_gate={:.2f}/{:.1f}-{:.1f}, yolo_gate={:.2f}/{:.1f}-{:.1f}) "
    "r_binary_threshold={} "
    "r_hold(template_misses={}) "
    "target_switch_missing_timeout_s={:.3f}",
    r_search_radius_scale_,
    r_search_radius_min_,
    r_search_radius_max_,
    r_min_area_,
    r_max_area_,
    r_max_aspect_ratio_,
    r_min_circularity_,
    r_max_accept_ratio_,
    r_geometry_gate_scale_,
    r_geometry_gate_min_,
    r_geometry_gate_max_,
    r_yolo_gate_scale_,
    r_yolo_gate_min_,
    r_yolo_gate_max_,
    r_binary_threshold_,
    kRTemplateHoldMaxMisses,
    target_switch_missing_timeout_s_);
}

void Buff_Detector::reset()
{
  status_ = LOSE;
  lose_ = 0;
  last_powerrune_ = std::nullopt;
  last_r_search_debug_ = std::nullopt;
  locked_target_missing_since_.reset();
  locked_target_angle_.reset();
  locked_target_update_time_.reset();
  r_template_hold_miss_count_ = 0;
  last_switch_deferred_ = false;
  last_target_switched_ = false;
  last_selected_target_index_ = -1;
}

int Buff_Detector::select_big_buff_target_index(
  const std::vector<YOLO11_BUFF::Object> & targets,
  const std::vector<YOLO11_BUFF::Object> & hit_context,
  const cv::Size & image_size,
  std::chrono::steady_clock::time_point timestamp,
  bool * switch_deferred,
  bool * target_switched,
  BuffDetectorCostSample * cost)
{
  if (cost != nullptr) {
    cost->target_candidates_examined += static_cast<std::uint32_t>(targets.size());
  }
  if (switch_deferred != nullptr) {
    *switch_deferred = false;
  }
  if (target_switched != nullptr) {
    *target_switched = false;
  }
  if (targets.empty()) {
    return -1;
  }
  if (!last_powerrune_.has_value() || last_powerrune_->fanblades.empty()) {
    locked_target_missing_since_.reset();
    locked_target_angle_.reset();
    locked_target_update_time_.reset();
    return 0;
  }

  const auto & last_target = last_powerrune_->target();
  const cv::Point2f last_center = last_target.center;
  const cv::Point2f last_r_center = last_powerrune_->r_center;
  const double last_blade_radius = std::max(mean_blade_radius(last_target), 1.0);
  const double hit_switch_gate = std::clamp(last_blade_radius * 1.2, 24.0, 120.0);
  const double diagonal = std::max(
    1.0,
    std::hypot(static_cast<double>(image_size.width), static_cast<double>(image_size.height)));
  if (!locked_target_angle_.has_value() &&
      is_finite_point(last_center) &&
      is_finite_point(last_r_center)) {
    locked_target_angle_ = blade_angle_around_r(last_center, last_r_center);
  }
  const double lock_dt_s = locked_target_update_time_.has_value()
    ? std::max(0.0, std::chrono::duration<double>(timestamp - *locked_target_update_time_).count())
    : 0.0;
  const double angle_lock_gate = std::clamp(0.20 + 3.0 * lock_dt_s, 0.35, 0.85);

  int matched_last_target_idx = -1;
  double matched_last_target_dist = std::numeric_limits<double>::max();
  double matched_last_target_angle_error = std::numeric_limits<double>::max();
  for (size_t i = 0; i < targets.size(); ++i) {
    const auto & target = targets[i];
    if (target.kpt.size() < 4) continue;
    const cv::Point2f center = blade_center_from_object(target);
    const cv::Point2f current_r =
      target.kpt.size() >= 5 && is_finite_point(target.kpt[4]) ? target.kpt[4] : last_r_center;
    const double dist = cv::norm(center - last_center);
    const double angle_error = locked_target_angle_.has_value() && is_finite_point(current_r)
      ? std::abs(wrapped_angle_diff(blade_angle_around_r(center, current_r), *locked_target_angle_))
      : std::numeric_limits<double>::max();
    const bool center_matched = dist < hit_switch_gate;
    const bool angle_matched = angle_error < angle_lock_gate;
    const bool lock_matched = locked_target_angle_.has_value() ? angle_matched : center_matched;
    if (lock_matched &&
        (angle_error < matched_last_target_angle_error ||
         (std::abs(angle_error - matched_last_target_angle_error) < 1e-6 &&
          dist < matched_last_target_dist))) {
      matched_last_target_dist = dist;
      matched_last_target_angle_error = angle_error;
      matched_last_target_idx = static_cast<int>(i);
    }
  }

  if (matched_last_target_idx >= 0) {
    if (cost != nullptr) {
      cost->locked_target_matched = true;
    }
    locked_target_missing_since_.reset();
    return matched_last_target_idx;
  } else {
    if (!locked_target_missing_since_.has_value()) {
      locked_target_missing_since_ = timestamp;
    }
    const double missing_elapsed_s = std::max(
      0.0,
      std::chrono::duration<double>(timestamp - *locked_target_missing_since_).count());
    if (missing_elapsed_s < target_switch_missing_timeout_s_) {
      if (switch_deferred != nullptr) {
        *switch_deferred = true;
      }
      if (cost != nullptr) {
        cost->target_switch_deferred = true;
      }
      tools::logger()->debug(
        "[BuffDetector] Hold target switch, locked target missing {:.3f}/{:.3f}s",
        missing_elapsed_s,
        target_switch_missing_timeout_s_);
      return -1;
	    }
	  }

  if (target_switched != nullptr) {
    *target_switched = true;
  }
  if (cost != nullptr) {
    cost->target_switched = true;
  }

	  cv::Point2f consumed_center = last_center;
  for (const auto & hit_object : hit_context) {
    if (hit_object.kpt.size() < 4) continue;
    const cv::Point2f center = blade_center_from_object(hit_object);
    const double dist = cv::norm(center - last_center);
    if (dist < hit_switch_gate) {
      consumed_center = center;
      break;
    }
  }

  int best_idx = 0;
  double best_score = -std::numeric_limits<double>::max();
  for (size_t i = 0; i < targets.size(); ++i) {
    if (targets[i].kpt.size() < 4) continue;
    const cv::Point2f center = blade_center_from_object(targets[i]);
    const double conf = std::clamp(static_cast<double>(targets[i].prob), 0.0, 1.0);

    double score = conf;
    // 只有上一锁定目标连续不可见超时后才会走到这里，此时优先切到离旧目标更远的待击打装甲。
    score += 0.7 * std::clamp(cv::norm(center - consumed_center) / diagonal, 0.0, 1.0);

    if (score > best_score) {
      best_score = score;
      best_idx = static_cast<int>(i);
    }
  }

  return best_idx;
}

// ==================== 图像预处理 ====================

void Buff_Detector::handle_img(
  const cv::Mat & bgr_img, cv::Mat & dilated_img, int class_id) const
{
  std::vector<cv::Mat> channels;
  cv::split(bgr_img, channels); // 分离通道: 0->B, 1->G, 2->R

  cv::Mat gray_img;

  if (is_red_label(class_id, MODE_.num_classes())) {
      // 红队: R - alpha * B
      cv::addWeighted(channels[2], 1.0, channels[0], -kRBinaryAlpha, 0, gray_img);
  } else {
      // 蓝队: B - alpha * R
      cv::addWeighted(channels[0], 1.0, channels[2], -kRBinaryAlpha, 0, gray_img);
  }

  // --------- Gamma 矫正 (压暗光晕，进一步断开连接) ---
  cv::Mat gamma_img;
  static const cv::Mat gamma_lut = buildGammaTable(kRBinaryGamma);
  cv::LUT(gray_img, gamma_lut, gamma_img);

  // 二值化：高于阈值变 255，低于变 0
  cv::Mat binary_img;
  cv::threshold(gamma_img, binary_img, r_binary_threshold_, 255, cv::THRESH_BINARY);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  // 实际上是做腐蚀操作减少黏连
  cv::erode(binary_img, dilated_img, kernel);
}

// ==================== 计算 R 标中心 ====================

cv::Point2f Buff_Detector::get_r_prior(
  const std::vector<FanBlade> & fanblades,
  cv::Point2f yolo_r_center,
  const cv::Size & image_size,
  BuffRSearchCostSample * cost) const
{
  if (fanblades.empty()) {
    return yolo_r_center;
  }

  if (!is_finite_point(yolo_r_center)) {
    yolo_r_center = fanblades.front().center;
  }
  yolo_r_center = clamp_point(yolo_r_center, image_size);

  const double blade_scale = std::max(mean_blade_radius(fanblades.front()), 1.0);
  std::optional<cv::Point2f> motion_prior = std::nullopt;

  if (last_powerrune_.has_value() && !last_powerrune_->fanblades.empty() && lose_ <= 2) {
    std::vector<std::pair<cv::Point2f, double>> motion_candidates;
    std::vector<bool> used_last(last_powerrune_->fanblades.size(), false);
    const double match_gate = std::clamp(blade_scale * 3.0, 35.0, 240.0);

    for (size_t curr_idx = 0; curr_idx < fanblades.size(); ++curr_idx) {
      const auto & curr = fanblades[curr_idx];
      if (!is_finite_point(curr.center)) {
        continue;
      }

      int best_last_idx = -1;
      double best_dist = std::numeric_limits<double>::max();
      for (size_t last_idx = 0; last_idx < last_powerrune_->fanblades.size(); ++last_idx) {
        if (used_last[last_idx]) {
          continue;
        }
        const auto & last = last_powerrune_->fanblades[last_idx];
        if (!is_finite_point(last.center)) {
          continue;
        }
        if (cost != nullptr) {
          ++cost->motion_comparisons;
        }
        const double dist = cv::norm(curr.center - last.center);
        if (dist < best_dist) {
          best_dist = dist;
          best_last_idx = static_cast<int>(last_idx);
        }
      }

      if (best_last_idx < 0 || best_dist > match_gate) {
        continue;
      }

      used_last[best_last_idx] = true;
      if (cost != nullptr) {
        ++cost->motion_matches;
      }
      const auto & matched_last = last_powerrune_->fanblades[best_last_idx];
      const cv::Point2f candidate = last_powerrune_->r_center + (curr.center - matched_last.center);
      const double weight = (curr_idx == 0) ? 1.0 : 0.75;
      motion_candidates.emplace_back(candidate, weight);
    }

    if (!motion_candidates.empty()) {
      motion_prior = weighted_average(motion_candidates, yolo_r_center);
    }
  }

  const cv::Point2f reference = motion_prior.value_or(yolo_r_center);
  const double geometry_gate = std::clamp(
    blade_scale * r_geometry_gate_scale_,
    r_geometry_gate_min_,
    r_geometry_gate_max_);
  const auto geometry_prior =
    pair_geometry_r_center(fanblades, reference, image_size, geometry_gate, cost);

  const double yolo_gate = std::clamp(
    blade_scale * r_yolo_gate_scale_,
    r_yolo_gate_min_,
    r_yolo_gate_max_);
  const bool yolo_agrees_with_motion =
    motion_prior.has_value() && cv::norm(yolo_r_center - *motion_prior) <= yolo_gate;
  const bool yolo_agrees_with_geometry =
    geometry_prior.has_value() && cv::norm(yolo_r_center - *geometry_prior) <= yolo_gate;
  const bool yolo_is_trusted =
    (!motion_prior.has_value() && !geometry_prior.has_value()) ||
    yolo_agrees_with_motion ||
    yolo_agrees_with_geometry;

  std::vector<std::pair<cv::Point2f, double>> prior_candidates;
  if (geometry_prior.has_value()) {
    prior_candidates.emplace_back(*geometry_prior, 0.55);
  }
  if (motion_prior.has_value()) {
    prior_candidates.emplace_back(*motion_prior, geometry_prior.has_value() ? 0.35 : 0.72);
  }
  if (yolo_is_trusted) {
    prior_candidates.emplace_back(yolo_r_center, prior_candidates.empty() ? 1.0 : 0.10);
  }
  if (cost != nullptr) {
    cost->prior_candidates = static_cast<std::uint32_t>(prior_candidates.size());
  }

  cv::Point2f fused_prior = weighted_average(prior_candidates, yolo_r_center);
  fused_prior = clamp_point(fused_prior, image_size);

  tools::logger()->debug(
    "[get_r_prior] yolo=({:.1f},{:.1f}) motion={} geometry={} trusted_yolo={} fused=({:.1f},{:.1f})",
    yolo_r_center.x,
    yolo_r_center.y,
    motion_prior.has_value() ? "yes" : "no",
    geometry_prior.has_value() ? "yes" : "no",
    yolo_is_trusted,
    fused_prior.x,
    fused_prior.y);

  return fused_prior;
}

cv::Point2f Buff_Detector::get_r_center(
  std::vector<FanBlade> & fanblades,
  cv::Mat & bgr_img,
  int class_id,
  cv::Point2f yolo_r_center,
  BuffRSearchCostSample * cost)
{
  const auto total_begin = CostClock::now();
  last_r_search_debug_ = std::nullopt;

  const auto held_r_center = [&]() -> std::optional<cv::Point2f> {
    if (r_template_hold_miss_count_ >= kRTemplateHoldMaxMisses ||
        !last_powerrune_.has_value() ||
        !is_finite_point(last_powerrune_->r_center)) {
      return std::nullopt;
    }
    return clamp_point(last_powerrune_->r_center, bgr_img.size());
  };

  /// 错误处理：无扇叶
  if (fanblades.empty()) {
    tools::logger()->debug("[Buff_Detector] 无法计算 r_center!");
    if (cost != nullptr) {
      cost->total_ns += cost_ns(total_begin, CostClock::now());
    }
    return {0, 0};
  }

  /// Step 1: 用 YOLO 点 + 多扇叶几何/时序关系构造更稳的 R 先验
  cv::Point2f safe_yolo_r_center = yolo_r_center;
  if (!is_finite_point(safe_yolo_r_center)) {
    safe_yolo_r_center = fanblades.front().center;
  }

  safe_yolo_r_center = clamp_point(safe_yolo_r_center, bgr_img.size());
  const auto prior_begin = CostClock::now();
  cv::Point2f r_prior = clamp_point(
    get_r_prior(fanblades, safe_yolo_r_center, bgr_img.size(), cost), bgr_img.size());
  if (cost != nullptr) {
    cost->prior_ns += cost_ns(prior_begin, CostClock::now());
  }

  tools::logger()->debug(
    "[get_r_center] yolo=({}, {}), prior=({}, {})",
    safe_yolo_r_center.x, safe_yolo_r_center.y, r_prior.x, r_prior.y);

  // 用扇叶几何尺度自适应估算搜索半径，避免依赖固定点序。
  const auto roi_begin = CostClock::now();
  const double blade_radius = mean_blade_radius(fanblades.front());
  const double radius = std::clamp(
    blade_radius * r_search_radius_scale_,
    r_search_radius_min_,
    r_search_radius_max_);
  if (cost != nullptr) {
    cost->radius = radius;
  }

  /// Step 2: 图像处理 + 掩膜筛选 R 标区域。只处理 R 先验附近的 ROI，避免整帧阈值/轮廓开销。
  const int roi_margin = static_cast<int>(std::ceil(radius + 8.0));
  const int roi_x0 = std::max(0, static_cast<int>(std::floor(r_prior.x)) - roi_margin);
  const int roi_y0 = std::max(0, static_cast<int>(std::floor(r_prior.y)) - roi_margin);
  const int roi_x1 = std::min(
    bgr_img.cols, static_cast<int>(std::ceil(r_prior.x)) + roi_margin + 1);
  const int roi_y1 = std::min(
    bgr_img.rows, static_cast<int>(std::ceil(r_prior.y)) + roi_margin + 1);
  if (roi_x1 <= roi_x0 || roi_y1 <= roi_y0) {
    if (cost != nullptr) {
      cost->roi_setup_ns += cost_ns(roi_begin, CostClock::now());
    }
    const auto hold_center = held_r_center();
    const cv::Point2f best_r_center = hold_center.value_or(r_prior);
    RSearchDebug debug;
    debug.yolo_center = safe_yolo_r_center;
    debug.prior_center = r_prior;
    debug.raw_center = best_r_center;
    debug.radius = radius;
    debug.used_hold_center = hold_center.has_value();
    last_r_search_debug_ = debug;
    if (hold_center.has_value()) {
      ++r_template_hold_miss_count_;
    }
    if (cost != nullptr) {
      cost->selected_source = hold_center.has_value() ? 3 : 0;
      cost->total_ns += cost_ns(total_begin, CostClock::now());
    }
    return best_r_center;
  }

  const cv::Rect search_roi(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0);
  const cv::Point2f roi_offset(
    static_cast<float>(search_roi.x), static_cast<float>(search_roi.y));
  const cv::Point2f r_prior_roi = r_prior - roi_offset;
  if (cost != nullptr) {
    cost->roi_width = static_cast<std::uint32_t>(search_roi.width);
    cost->roi_height = static_cast<std::uint32_t>(search_roi.height);
    cost->roi_pixels = static_cast<std::uint32_t>(search_roi.area());
    cost->roi_setup_ns += cost_ns(roi_begin, CostClock::now());
  }

  cv::Mat dilated_img;
  const auto preprocess_begin = CostClock::now();
  handle_img(bgr_img(search_roi), dilated_img, class_id);
  if (cost != nullptr) {
    cost->preprocess_ns += cost_ns(preprocess_begin, CostClock::now());
  }

  const auto mask_begin = CostClock::now();
  cv::Mat mask = cv::Mat::zeros(dilated_img.size(), CV_8U);
  cv::circle(mask, r_prior_roi, static_cast<int>(std::round(radius)), cv::Scalar(255), -1);

  cv::Mat masked_img;
  cv::bitwise_and(dilated_img, mask, masked_img);
  if (cost != nullptr) {
    cost->circle_mask_ns += cost_ns(mask_begin, CostClock::now());
  }

  const auto template_begin = CostClock::now();
  const RTemplateMatch template_match =
    match_r_template(masked_img, r_prior_roi, radius, cost);
  if (cost != nullptr) {
    cost->template_scale_search_ns += cost_ns(template_begin, CostClock::now());
  }

  /// Step 3: 轮廓检测 + 最小外接矩形筛选
  const auto find_contours_begin = CostClock::now();
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(masked_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  if (cost != nullptr) {
    cost->find_contours_ns += cost_ns(find_contours_begin, CostClock::now());
    cost->contours_total = static_cast<std::uint32_t>(contours.size());
  }

  cv::Point2f best_r_center = r_prior;
  double best_score = std::numeric_limits<double>::max();
  double best_dist = std::numeric_limits<double>::max();
  std::vector<std::vector<cv::Point>> accepted_contours;
  std::vector<cv::Point2f> accepted_centers;

  const auto contour_filter_begin = CostClock::now();
  for (const auto & contour : contours) {
    const double area = std::abs(cv::contourArea(contour));
    if (area < r_min_area_ || area > r_max_area_) {
      continue;
    }
    if (cost != nullptr) {
      ++cost->contours_area_pass;
    }

    const auto rect = cv::minAreaRect(contour);
    const double rect_w = rect.size.width;
    const double rect_h = rect.size.height;
    if (rect_w < 1e-3 || rect_h < 1e-3) {
      continue;
    }

    const double ratio = std::max(rect_w, rect_h) / std::min(rect_w, rect_h);
    if (ratio > r_max_aspect_ratio_) {
      continue;
    }
    if (cost != nullptr) {
      ++cost->contours_aspect_pass;
    }

    const double circularity = contour_circularity(contour);
    if (circularity < r_min_circularity_) {
      continue;
    }
    if (cost != nullptr) {
      ++cost->contours_circularity_pass;
    }

    const cv::Point2f contour_center = rect.center + roi_offset;
    const double dist = cv::norm(contour_center - r_prior);
    if (dist > radius) {
      continue;
    }
    if (cost != nullptr) {
      ++cost->contours_radius_pass;
      ++cost->contours_accepted;
    }

    accepted_contours.push_back(contour);
    accepted_centers.push_back(contour_center);

    // 候选越接近先验、越接近圆形，分数越低。
    const double score =
      1.6 * (dist / std::max(radius, 1.0)) +
      1.2 * std::abs(ratio - 1.0) +
      0.8 * (1.0 - circularity);

    if (score < best_score) {
      best_score = score;
      best_dist = dist;
      best_r_center = contour_center;
    }
  }
  if (cost != nullptr) {
    cost->contour_filter_score_ns += cost_ns(contour_filter_begin, CostClock::now());
  }

  const auto selection_begin = CostClock::now();
  bool used_contour_center = best_dist <= radius * r_max_accept_ratio_;
  if (!used_contour_center) {
    best_r_center = r_prior;
  }
  bool used_template = false;
  bool used_hold_center = false;
  cv::Rect template_rect_global;
  if (template_match.valid && template_match.score >= 0.15) {
    best_r_center = template_match.center + roi_offset;
    used_template = true;
    r_template_hold_miss_count_ = 0;
    template_rect_global = template_match.rect + search_roi.tl();
  } else {
    const auto hold_center = held_r_center();
    if (hold_center.has_value()) {
      best_r_center = *hold_center;
      used_hold_center = true;
      used_contour_center = false;
      ++r_template_hold_miss_count_;
    }
  }
  if (cost != nullptr) {
    cost->selected_source = used_template ? 2 : (used_hold_center ? 3 : (used_contour_center ? 1 : 0));
    cost->selection_state_commit_ns += cost_ns(selection_begin, CostClock::now());
  }

  const cv::Point2f raw_r_center = best_r_center;
  const auto debug_materialization_begin = CostClock::now();
  RSearchDebug debug;
  debug.yolo_center = safe_yolo_r_center;
  debug.prior_center = r_prior;
  debug.raw_center = raw_r_center;
  debug.roi_rect = search_roi;
  debug.accepted_contour_points = accepted_contours;
  debug.accepted_centers = accepted_centers;
  debug.masked_roi = masked_img.clone();
  debug.template_rect = template_rect_global;
  debug.template_score = template_match.valid ? template_match.score : 0.0;
  debug.template_hits = template_match.hits;
  debug.radius = radius;
  debug.total_contours = static_cast<int>(contours.size());
  debug.accepted_count = static_cast<int>(accepted_contours.size());
  debug.used_template = used_template;
  debug.used_hold_center = used_hold_center;
  debug.used_contour_center = used_contour_center;
  last_r_search_debug_ = debug;
  if (cost != nullptr) {
    std::uint64_t contour_points = 0;
    for (const auto & contour : accepted_contours) {
      contour_points += static_cast<std::uint64_t>(contour.size());
    }
    cost->debug_copied_elements +=
      contour_points + static_cast<std::uint64_t>(accepted_centers.size());
    cost->debug_copied_bytes +=
      static_cast<std::uint64_t>(masked_img.total() * masked_img.elemSize()) +
      contour_points * sizeof(cv::Point) +
      static_cast<std::uint64_t>(accepted_centers.size()) * sizeof(cv::Point2f);
    cost->debug_materialization_ns +=
      cost_ns(debug_materialization_begin, CostClock::now());
  }

  tools::logger()->debug(
    "[get_r_center] raw=({}, {}), final=({}, {}), source={}, accepted_contours={}, radius={:.1f}",
    raw_r_center.x,
    raw_r_center.y,
    best_r_center.x,
    best_r_center.y,
    used_template ? "template" : (used_hold_center ? "hold" : (used_contour_center ? "contour" : "prior")),
    accepted_contours.size(),
    radius);

  if (cost != nullptr) {
    cost->total_ns += cost_ns(total_begin, CostClock::now());
  }
  return best_r_center;
}

// ==================== 丢失处理 ====================

void Buff_Detector::handle_lose(std::chrono::steady_clock::time_point timestamp)
{
  lose_++;
  if (lose_ >= LOSE_MAX) {
    if (
      mode_ == BIG &&
      last_powerrune_.has_value() &&
      locked_target_missing_since_.has_value()) {
      const double missing_elapsed_s = std::max(
        0.0,
        std::chrono::duration<double>(timestamp - *locked_target_missing_since_).count());
      if (missing_elapsed_s < target_switch_missing_timeout_s_) {
        status_ = TEM_LOSE;
        lose_ = LOSE_MAX - 1;
        return;
      }
    }
    status_ = LOSE;
    last_powerrune_ = std::nullopt;
    locked_target_missing_since_.reset();
    locked_target_angle_.reset();
    locked_target_update_time_.reset();
    r_template_hold_miss_count_ = 0;
  } else {
    status_ = TEM_LOSE;
  }
}

// ==================== 识别函数 ====================
// 核心调度函数
std::optional<PowerRune> Buff_Detector::detect(
  cv::Mat & bgr_img,
  const Solver & solver,
  bool draw_yolo_results)
{
  const std::vector<YOLO11_BUFF::Object> results =
    detect_candidates(mode_, bgr_img, draw_yolo_results);
  return solve_candidates(mode_, bgr_img, solver, results, std::chrono::steady_clock::now());
}

std::vector<YOLO11_BUFF::Object> Buff_Detector::detect_candidates(
  PowerRune_type mode,
  cv::Mat & bgr_img,
  bool draw_yolo_results)
{
  if (mode == SMALL) {
    return MODE_.get_onecandidatebox(bgr_img, draw_yolo_results);
  }
  return MODE_.get_twocandidatebox(bgr_img, draw_yolo_results);
}

BuffCanonicalObservation Buff_Detector::build_canonical_observation(
  PowerRune_type mode,
  const cv::Mat & bgr_img,
  const std::vector<YOLO11_BUFF::Object> & results,
  BuffCanonicalWorkerScratch * scratch) const
{
  BuffCanonicalObservation out;
  out.image_size = bgr_img.size();
  out.raw_candidate_count = static_cast<std::uint32_t>(results.size());
  const auto fail = [&](BuffObservationFallbackReason reason, bool cap = false) {
    out.ready = false;
    out.requires_legacy_fallback = true;
    out.fallback_reason = reason;
    if (cap) ++out.cap_events;
  };
  if (bgr_img.empty() || bgr_img.type() != CV_8UC3) {
    fail(BuffObservationFallbackReason::InvalidInput);
    return out;
  }

  std::vector<std::pair<std::uint32_t, YOLO11_BUFF::Object>> targets;
  for (std::uint32_t index = 0; index < results.size(); ++index) {
    const auto & result = results[index];
    if (is_target_label(result.label, MODE_.num_classes()) && has_rune_keypoints(result)) {
      targets.emplace_back(index, result);
    } else if (is_hit_label(result.label, MODE_.num_classes()) && has_blade_keypoints(result)) {
      out.hit_context.push_back(result);
    }
  }
  if (targets.size() > kBuffObservationMaxTargets ||
      out.hit_context.size() > kBuffObservationMaxHitContext) {
    fail(BuffObservationFallbackReason::CandidateCap, true);
    return out;
  }
  if (targets.empty()) {
    // Empty observation is complete and state free. The ordered reducer owns
    // the corresponding loss transition.
    out.ready = true;
    return out;
  }
  if (mode == SMALL && targets.size() > 1) {
    fail(BuffObservationFallbackReason::CandidateCap, true);
    return out;
  }

  const cv::Rect image_bounds(0, 0, bgr_img.cols, bgr_img.rows);
  for (std::uint32_t hypothesis_index = 0;
       hypothesis_index < targets.size(); ++hypothesis_index) {
    BuffTargetHypothesis hypothesis;
    hypothesis.hypothesis_index = hypothesis_index;
    hypothesis.source_candidate_index = targets[hypothesis_index].first;
    hypothesis.target = targets[hypothesis_index].second;
    hypothesis.yolo_r = hypothesis.target.kpt[4];

    auto append_blade = [&](const YOLO11_BUFF::Object & object) {
      if (!has_blade_keypoints(object)) return;
      const cv::Point2f center = blade_center_from_object(object);
      hypothesis.fanblades.emplace_back(FanBlade(
        std::vector<cv::Point2f>(object.kpt.begin(), object.kpt.begin() + 4),
        center, _target));
    };
    append_blade(hypothesis.target);
    if (mode == BIG) {
      for (std::uint32_t other = 0; other < targets.size(); ++other) {
        if (other != hypothesis_index) append_blade(targets[other].second);
      }
    }
    if (hypothesis.fanblades.empty()) {
      fail(BuffObservationFallbackReason::InvalidInput);
      return out;
    }
    hypothesis.blade_radius = mean_blade_radius(hypothesis.fanblades.front());
    out.targets.push_back(std::move(hypothesis));
  }

  bool has_union = false;
  cv::Rect union_roi;
  auto add_anchor = [&](std::uint32_t hypothesis_index, std::uint8_t source,
                        cv::Point2f center, double radius) -> bool {
    if (!point_inside_image(center, bgr_img.size())) return true;
    const auto duplicate = std::find_if(
      out.anchors.begin(), out.anchors.end(), [&](const BuffRAnchor & anchor) {
        return anchor.center.x == center.x && anchor.center.y == center.y;
      });
    if (duplicate != out.anchors.end()) return true;
    if (out.anchors.size() >= kBuffObservationMaxAnchors) return false;
    const int margin = static_cast<int>(std::ceil(radius + 8.0));
    const cv::Rect support(
      static_cast<int>(std::floor(center.x)) - margin,
      static_cast<int>(std::floor(center.y)) - margin,
      margin * 2 + 1, margin * 2 + 1);
    BuffRAnchor anchor;
    anchor.hypothesis_index = hypothesis_index;
    anchor.source = source;
    anchor.center = center;
    anchor.radius = radius;
    anchor.support_roi = support & image_bounds;
    if (anchor.support_roi.empty()) return true;
    union_roi = has_union ? (union_roi | anchor.support_roi) : anchor.support_roi;
    has_union = true;
    out.anchors.push_back(std::move(anchor));
    return true;
  };

  for (const auto & hypothesis : out.targets) {
    const double radius = std::clamp(
      hypothesis.blade_radius * r_search_radius_scale_,
      r_search_radius_min_, r_search_radius_max_);
    if (!add_anchor(hypothesis.hypothesis_index, 0, hypothesis.yolo_r, radius)) {
      fail(BuffObservationFallbackReason::AnchorCap, true);
      return out;
    }
    for (const auto & center :
         raw_pair_geometry_r_centers(hypothesis.fanblades, bgr_img.size())) {
      if (!add_anchor(hypothesis.hypothesis_index, 1, center, radius)) {
        fail(BuffObservationFallbackReason::AnchorCap, true);
        return out;
      }
    }
  }
  if (!has_union || out.anchors.empty()) {
    fail(BuffObservationFallbackReason::MissingCoverage);
    return out;
  }
  out.union_roi = union_roi;
  out.union_roi_pixels = static_cast<std::uint64_t>(union_roi.area());

  std::vector<int> local_scales;
  std::vector<int> & scales = scratch != nullptr ? scratch->scales : local_scales;
  scales.clear();
  if (scratch != nullptr && scales.capacity() < 16) scales.reserve(16);
  for (const auto & hypothesis : out.targets) {
    const double radius = std::clamp(
      hypothesis.blade_radius * r_search_radius_scale_,
      r_search_radius_min_, r_search_radius_max_);
    const int min_size = std::max(14, static_cast<int>(std::lround(radius * 0.45)));
    const int max_size = std::min(
      std::min(union_roi.width, union_roi.height),
      std::max(min_size, static_cast<int>(std::lround(radius * 1.25))));
    for (int size = min_size; size <= max_size; size += std::max(2, size / 8)) {
      if (union_roi.width < size || union_roi.height < size) continue;
      if (std::find(scales.begin(), scales.end(), size) == scales.end()) scales.push_back(size);
    }
  }
  std::sort(scales.begin(), scales.end());
  for (const int size : scales) {
    out.template_result_pixels +=
      static_cast<std::uint64_t>(union_roi.width - size + 1) *
      static_cast<std::uint64_t>(union_roi.height - size + 1);
  }
  if (out.template_result_pixels > kBuffObservationMaxTemplateResultPixels) {
    fail(BuffObservationFallbackReason::ResponsePixelCap, true);
    return out;
  }

  const auto preprocess_begin = CostClock::now();
  cv::Mat local_binary;
  cv::Mat local_mask;
  cv::Mat local_masked;
  cv::Mat & binary = scratch != nullptr ? scratch->binary : local_binary;
  cv::Mat & mask = scratch != nullptr ? scratch->mask : local_mask;
  cv::Mat & masked = scratch != nullptr ? scratch->masked : local_masked;
  auto account_mat = [&](const cv::Mat & mat, const cv::Size & size, int type) {
    if (scratch == nullptr) return;
    if (!mat.empty() && mat.size() == size && mat.type() == type) {
      ++scratch->reuses;
    } else {
      ++scratch->allocations;
    }
  };
  // All retained target labels in one mechanism share the same team/color.
  if (scratch == nullptr) {
    handle_img(bgr_img(union_roi), binary, out.targets.front().target.label);
  } else {
    const cv::Mat roi = bgr_img(union_roi);
    if (scratch->channels.size() != 3) scratch->channels.resize(3);
    for (auto & channel : scratch->channels) account_mat(channel, roi.size(), CV_8U);
    cv::split(roi, scratch->channels);
    account_mat(scratch->gray, roi.size(), CV_8U);
    if (is_red_label(out.targets.front().target.label, MODE_.num_classes())) {
      cv::addWeighted(scratch->channels[2], 1.0, scratch->channels[0],
                      -kRBinaryAlpha, 0, scratch->gray);
    } else {
      cv::addWeighted(scratch->channels[0], 1.0, scratch->channels[2],
                      -kRBinaryAlpha, 0, scratch->gray);
    }
    static const cv::Mat gamma_lut = buildGammaTable(kRBinaryGamma);
    account_mat(scratch->gamma, roi.size(), CV_8U);
    cv::LUT(scratch->gray, gamma_lut, scratch->gamma);
    account_mat(scratch->threshold, roi.size(), CV_8U);
    cv::threshold(scratch->gamma, scratch->threshold, r_binary_threshold_, 255,
                  cv::THRESH_BINARY);
    if (scratch->kernel.empty()) {
      scratch->kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
      ++scratch->allocations;
    } else {
      ++scratch->reuses;
    }
    account_mat(binary, roi.size(), CV_8U);
    cv::erode(scratch->threshold, binary, scratch->kernel);
  }
  account_mat(mask, binary.size(), CV_8U);
  mask.create(binary.size(), CV_8U);
  mask.setTo(cv::Scalar(0));
  for (const auto & anchor : out.anchors) {
    cv::circle(
      mask, anchor.center - cv::Point2f(
        static_cast<float>(union_roi.x), static_cast<float>(union_roi.y)),
      static_cast<int>(std::round(anchor.radius)), cv::Scalar(255), -1);
  }
  account_mat(masked, binary.size(), CV_8U);
  cv::bitwise_and(binary, mask, masked);
  out.preprocess_ns = cost_ns(preprocess_begin, CostClock::now());

  const auto template_begin = CostClock::now();
  thread_local std::unordered_map<int, cv::Mat> canonical_template_cache;
  for (const int size : scales) {
    auto template_iter = canonical_template_cache.find(size);
    if (template_iter == canonical_template_cache.end()) {
      template_iter = canonical_template_cache.emplace(size, make_r_template(size)).first;
    }
    BuffRScaleResponse response;
    response.template_size = size;
    response.global_origin = union_roi.tl();
    cv::matchTemplate(
      masked, template_iter->second, response.scores, cv::TM_CCOEFF_NORMED);
    out.scale_responses.push_back(std::move(response));
  }
  out.template_ns = cost_ns(template_begin, CostClock::now());

  const auto contour_begin = CostClock::now();
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(masked, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  if (contours.size() > kBuffObservationMaxContours) {
    fail(BuffObservationFallbackReason::ContourCap, true);
    return out;
  }
  out.contours.reserve(contours.size());
  for (const auto & local : contours) {
    BuffRContourObservation observation;
    out.contour_copy_bytes_avoided +=
      static_cast<std::uint64_t>(local.size()) * sizeof(cv::Point);
    observation.area = std::abs(cv::contourArea(local));
    const auto rect = cv::minAreaRect(local);
    observation.center_global = rect.center + cv::Point2f(
      static_cast<float>(union_roi.x), static_cast<float>(union_roi.y));
    const double rw = rect.size.width;
    const double rh = rect.size.height;
    observation.aspect_ratio = rw >= 1e-3 && rh >= 1e-3
      ? std::max(rw, rh) / std::min(rw, rh)
      : std::numeric_limits<double>::infinity();
    observation.circularity = contour_circularity(local);
    out.contours.push_back(std::move(observation));
  }
  out.contour_ns = cost_ns(contour_begin, CostClock::now());

  for (const auto & hypothesis : out.targets) {
    const double radius = std::clamp(
      hypothesis.blade_radius * r_search_radius_scale_,
      r_search_radius_min_, r_search_radius_max_);
    const double geometry_gate = std::clamp(
      hypothesis.blade_radius * r_geometry_gate_scale_,
      r_geometry_gate_min_, r_geometry_gate_max_);
    const auto geometry = pair_geometry_r_center(
      hypothesis.fanblades, hypothesis.yolo_r, bgr_img.size(), geometry_gate, nullptr);
    const double yolo_gate = std::clamp(
      hypothesis.blade_radius * r_yolo_gate_scale_,
      r_yolo_gate_min_, r_yolo_gate_max_);
    std::vector<std::pair<cv::Point2f, double>> priors;
    if (geometry.has_value()) priors.emplace_back(*geometry, 0.55);
    if (!geometry.has_value() || cv::norm(hypothesis.yolo_r - *geometry) <= yolo_gate) {
      priors.emplace_back(hypothesis.yolo_r, priors.empty() ? 1.0 : 0.10);
    }
    const cv::Point2f prior = clamp_point(
      weighted_average(priors, hypothesis.yolo_r), bgr_img.size());

    BuffCanonicalRChoice choice;
    choice.hypothesis_index = hypothesis.hypothesis_index;
    choice.canonical_prior = prior;
    choice.r_center = prior;
    bool template_found = false;
    double template_score = -std::numeric_limits<double>::infinity();
    cv::Rect template_rect;
    for (const auto & response : out.scale_responses) {
      cv::Mat local_support;
      cv::Mat & support = scratch != nullptr
        ? scratch->support_lookup
        : local_support;
      account_mat(support, response.scores.size(), CV_8U);
      support.create(response.scores.size(), CV_8U);
      support.setTo(cv::Scalar(0));
      for (const auto & anchor : out.anchors) {
        if (anchor.hypothesis_index != hypothesis.hypothesis_index) continue;
        const int x0 = std::max(0, anchor.support_roi.x - response.global_origin.x);
        const int y0 = std::max(0, anchor.support_roi.y - response.global_origin.y);
        const int x1 = std::min(
          response.scores.cols - 1,
          anchor.support_roi.x + anchor.support_roi.width -
            response.template_size - response.global_origin.x);
        const int y1 = std::min(
          response.scores.rows - 1,
          anchor.support_roi.y + anchor.support_roi.height -
            response.template_size - response.global_origin.y);
        if (x0 <= x1 && y0 <= y1) {
          support(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)).setTo(cv::Scalar(255));
        }
      }
      for (int y = 0; y < response.scores.rows; ++y) {
        const float * row = response.scores.ptr<float>(y);
        const std::uint8_t * support_row = support.ptr<std::uint8_t>(y);
        for (int x = 0; x < response.scores.cols; ++x) {
          ++out.response_cells_scanned;
          if (support_row[x] == 0) {
            ++out.support_rejected_cells;
            continue;
          }
          const cv::Point top_left = response.global_origin + cv::Point(x, y);
          const cv::Point2f center(
            static_cast<float>(top_left.x) + response.template_size * 0.5f,
            static_cast<float>(top_left.y) + response.template_size * 0.5f);
          ++out.distance_tested_cells;
          const double dist = cv::norm(center - prior);
          if (dist > radius) continue;
          const double score = static_cast<double>(row[x]) -
            0.25 * dist / std::max(radius, 1.0);
          if (!template_found || score > template_score) {
            template_found = true;
            template_score = score;
            choice.r_center = center;
            choice.template_size = response.template_size;
            template_rect = cv::Rect(
              top_left.x, top_left.y,
              response.template_size, response.template_size);
          }
        }
      }
    }

    bool contour_found = false;
    double contour_score = std::numeric_limits<double>::infinity();
    cv::Point2f contour_center = prior;
    for (const auto & contour : out.contours) {
      if (contour.area < r_min_area_ || contour.area > r_max_area_ ||
          contour.aspect_ratio > r_max_aspect_ratio_ ||
          contour.circularity < r_min_circularity_) continue;
      const double dist = cv::norm(contour.center_global - prior);
      if (dist > radius) continue;
      const double score =
        1.6 * dist / std::max(radius, 1.0) +
        1.2 * std::abs(contour.aspect_ratio - 1.0) +
        0.8 * (1.0 - contour.circularity);
      if (score < contour_score) {
        contour_score = score;
        contour_center = contour.center_global;
        contour_found = true;
      }
    }

    choice.template_score = template_found ? template_score : 0.0;
    if (template_found && template_score >= 0.15) {
      choice.source = 2;
      choice.template_valid = true;
      choice.debug.template_rect = template_rect;
      choice.debug.template_score = template_score;
      choice.debug.template_hits = 1;
      choice.debug.used_template = true;
    } else {
      if (const char * trace = std::getenv("AIM_BUFF_CANONICAL_TRACE");
          trace != nullptr && std::string(trace) == "1") {
        tools::logger()->info(
          "Canonical R unsupported: hypothesis={} found={} best_score={:.6f} prior=({:.3f},{:.3f}) responses={} pixels={}",
          hypothesis.hypothesis_index, template_found, template_score,
          prior.x, prior.y, out.scale_responses.size(), out.template_result_pixels);
      }
      // Preserve the state-free contour/prior decision. The ordered reducer
      // may replace it with the legacy held R without any additional pixel
      // search; fixed-R compatibility is checked before PnP extraction.
      if (contour_found &&
          cv::norm(contour_center - prior) <= radius * r_max_accept_ratio_) {
        choice.source = 1;
        choice.r_center = contour_center;
        choice.debug.used_contour_center = true;
      }
    }
    choice.debug.yolo_center = hypothesis.yolo_r;
    choice.debug.prior_center = prior;
    choice.debug.raw_center = choice.r_center;
    choice.debug.radius = radius;
    out.r_choices.push_back(std::move(choice));
  }

  out.ready = out.r_choices.size() == out.targets.size();
  if (scratch != nullptr) {
    out.scratch_allocations = scratch->allocations;
    out.scratch_reuses = scratch->reuses;
    scratch->allocations = 0;
    scratch->reuses = 0;
  }
  out.requires_legacy_fallback = !out.ready;
  if (!out.ready && out.fallback_reason == BuffObservationFallbackReason::None) {
    out.fallback_reason = BuffObservationFallbackReason::MissingCoverage;
  }
  return out;
}

bool Buff_Detector::canonical_observation_commit_supported(
  const BuffCanonicalObservation & observation) const
{
  if (!observation.ready || observation.requires_legacy_fallback) return false;
  if (r_template_hold_miss_count_ >= kRTemplateHoldMaxMisses ||
      !last_powerrune_.has_value() ||
      !is_finite_point(last_powerrune_->r_center)) {
    return true;
  }
  const cv::Point2f held = clamp_point(last_powerrune_->r_center, observation.image_size);
  return std::all_of(
    observation.r_choices.begin(), observation.r_choices.end(),
    [&held](const BuffCanonicalRChoice & choice) {
      return choice.template_valid ||
             (choice.r_center.x == held.x && choice.r_center.y == held.y);
    });
}

std::optional<PowerRune> Buff_Detector::solve_canonical_observation(
  PowerRune_type mode,
  const BuffCanonicalObservation & observation,
  Solver & solver,
  const SolverFrameContext & solver_frame,
  const ExhaustivePnpProposal & pnp_proposal,
  std::chrono::steady_clock::time_point timestamp,
  BuffDetectorCostSample * cost)
{
  const auto total_begin = CostClock::now();
  if (cost != nullptr) {
    *cost = BuffDetectorCostSample{};
    cost->mode = static_cast<std::uint32_t>(mode);
    cost->raw_candidates = observation.raw_candidate_count;
    cost->target_candidates = static_cast<std::uint32_t>(observation.targets.size());
    cost->hit_candidates = static_cast<std::uint32_t>(observation.hit_context.size());
  }
  last_switch_deferred_ = false;
  last_target_switched_ = false;
  last_selected_target_index_ = -1;
  mode_ = mode;

  if (!observation.ready || observation.requires_legacy_fallback) {
    throw std::logic_error("canonical observation committed without complete coverage");
  }
  if (observation.targets.empty()) {
    handle_lose(timestamp);
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::NoRuneCandidate;
      cost->total_ns += cost_ns(total_begin, CostClock::now());
    }
    return std::nullopt;
  }

  std::uint32_t selected_hypothesis = 0;
  bool switch_deferred = false;
  bool target_switched = false;
  if (mode == BIG) {
    std::vector<YOLO11_BUFF::Object> targets;
    targets.reserve(observation.targets.size());
    for (const auto & hypothesis : observation.targets) targets.push_back(hypothesis.target);
    const int selected = select_big_buff_target_index(
      targets, observation.hit_context,
      observation.image_size,
      timestamp, &switch_deferred, &target_switched, cost);
    if (switch_deferred) {
      last_switch_deferred_ = true;
      status_ = TEM_LOSE;
      lose_ = std::min(lose_ + 1, LOSE_MAX - 1);
      if (cost != nullptr) {
        cost->early_exit = BuffDetectorEarlyExit::TargetSwitchDeferred;
        cost->total_ns += cost_ns(total_begin, CostClock::now());
      }
      return std::nullopt;
    }
    if (selected < 0 || selected >= static_cast<int>(observation.targets.size())) {
      handle_lose(timestamp);
      if (cost != nullptr) {
        cost->early_exit = BuffDetectorEarlyExit::TargetSelectionFailed;
        cost->total_ns += cost_ns(total_begin, CostClock::now());
      }
      return std::nullopt;
    }
    selected_hypothesis = static_cast<std::uint32_t>(selected);
    last_selected_target_index_ = selected;
    last_target_switched_ = target_switched;
  }

  const auto hypothesis_iter = std::find_if(
    observation.targets.begin(), observation.targets.end(),
    [selected_hypothesis](const BuffTargetHypothesis & value) {
      return value.hypothesis_index == selected_hypothesis;
    });
  const auto r_iter = std::find_if(
    observation.r_choices.begin(), observation.r_choices.end(),
    [selected_hypothesis](const BuffCanonicalRChoice & value) {
      return value.hypothesis_index == selected_hypothesis;
    });
  if (hypothesis_iter == observation.targets.end() || r_iter == observation.r_choices.end() ||
      !is_finite_point(r_iter->r_center)) {
    throw std::logic_error("canonical observation selected uncovered target/R");
  }

  PowerRune rune;
  rune.type = mode;
  rune.fanblades = hypothesis_iter->fanblades;
  rune.r_center = r_iter->r_center;
  rune.r_search_debug = r_iter->debug;
  bool used_hold_center = false;
  if (r_iter->template_valid) {
    r_template_hold_miss_count_ = 0;
  } else if (r_template_hold_miss_count_ < kRTemplateHoldMaxMisses &&
             last_powerrune_.has_value() &&
             is_finite_point(last_powerrune_->r_center)) {
    const cv::Point2f held = clamp_point(last_powerrune_->r_center, observation.image_size);
    if (held.x != rune.r_center.x || held.y != rune.r_center.y) {
      throw std::logic_error("canonical held R does not match fixed-R PnP proposal");
    }
    rune.r_center = held;
    rune.r_search_debug->raw_center = held;
    rune.r_search_debug->used_template = false;
    rune.r_search_debug->used_contour_center = false;
    rune.r_search_debug->used_hold_center = true;
    used_hold_center = true;
    ++r_template_hold_miss_count_;
  }
  rune.target_switched = target_switched;
  rune.switch_deferred = false;
  if (cost != nullptr) {
    cost->constructed_fanblades = static_cast<std::uint32_t>(rune.fanblades.size());
    cost->r_search.selected_source = used_hold_center ? 3 : r_iter->source;
    cost->r_search.winning_template_score = r_iter->template_score;
    cost->r_search.winning_template_size =
      static_cast<std::uint32_t>(std::max(0, r_iter->template_size));
    cost->r_search.template_result_pixels = observation.template_result_pixels;
    cost->r_search.contours_total = static_cast<std::uint32_t>(observation.contours.size());
  }

  const auto pnp_begin = CostClock::now();
  const PnpReductionResult reduction = solver.solveFromProposal(
    rune, selected_hypothesis, pnp_proposal, solver_frame,
    cost != nullptr ? &cost->pnp : nullptr);
  if (cost != nullptr) cost->pnp_solve_ns += cost_ns(pnp_begin, CostClock::now());
  if (!reduction.applicable) {
    throw std::logic_error("canonical observation PnP proposal is not applicable");
  }
  if (!reduction.solved || rune.is_unsolve()) {
    handle_lose(timestamp);
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::PnpUnsolved;
      cost->total_ns += cost_ns(total_begin, CostClock::now());
    }
    return std::nullopt;
  }

  status_ = TRACK;
  lose_ = 0;
  locked_target_missing_since_.reset();
  last_r_search_debug_ = rune.r_search_debug;
  if (mode == BIG && !rune.fanblades.empty() &&
      is_finite_point(rune.target().center) && is_finite_point(rune.r_center)) {
    locked_target_angle_ = blade_angle_around_r(rune.target().center, rune.r_center);
    locked_target_update_time_ = timestamp;
  } else {
    locked_target_angle_.reset();
    locked_target_update_time_.reset();
  }
  last_powerrune_ = rune;
  if (cost != nullptr) cost->total_ns += cost_ns(total_begin, CostClock::now());
  return last_powerrune_;
}

std::optional<PowerRune> Buff_Detector::solve_candidates(
  PowerRune_type mode,
  cv::Mat & bgr_img,
  const Solver & solver,
  const std::vector<YOLO11_BUFF::Object> & results,
  std::chrono::steady_clock::time_point timestamp,
  BuffDetectorCostSample * cost)
{
  const auto total_begin = CostClock::now();
  if (cost != nullptr) {
    *cost = BuffDetectorCostSample{};
    cost->mode = static_cast<std::uint32_t>(mode);
    cost->raw_candidates = static_cast<std::uint32_t>(results.size());
  }
  const auto dispatch_begin = CostClock::now();
  last_switch_deferred_ = false;
  last_target_switched_ = false;
  last_selected_target_index_ = -1;
  mode_ = mode;
  if (cost != nullptr) {
    cost->dispatch_reset_ns += cost_ns(dispatch_begin, CostClock::now());
  }
  std::optional<PowerRune> result;
  if (mode == SMALL) {
    result = solve_small_buff(bgr_img, solver, results, cost);
  } else {
    result = solve_big_buff(bgr_img, solver, results, timestamp, cost);
  }
  if (cost != nullptr) {
    cost->total_ns += cost_ns(total_begin, CostClock::now());
  }
  return result;
}

// 大符逻辑
std::optional<PowerRune> Buff_Detector::solve_big_buff(
  cv::Mat & bgr_img,
  const Solver & solver,
  const std::vector<YOLO11_BUFF::Object> & results,
  std::chrono::steady_clock::time_point timestamp,
  BuffDetectorCostSample * cost)
{
  const auto reject_begin = CostClock::now();
  auto mark_switch_deferred_if_recent = [&]() {
    if (!last_powerrune_.has_value() || !locked_target_missing_since_.has_value()) {
      return;
    }
    const double missing_elapsed_s = std::max(
      0.0,
      std::chrono::duration<double>(timestamp - *locked_target_missing_since_).count());
    if (missing_elapsed_s < target_switch_missing_timeout_s_) {
      last_switch_deferred_ = true;
    }
  };

  if (results.empty()) {
    if (last_powerrune_.has_value() && !locked_target_missing_since_.has_value()) {
      locked_target_missing_since_ = timestamp;
    }
    mark_switch_deferred_if_recent();
    handle_lose(timestamp);
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::RawResultsEmpty;
      cost->reject_state_commit_ns += cost_ns(reject_begin, CostClock::now());
    }
    return std::nullopt;
  }

  const auto classification_begin = CostClock::now();
  std::vector<YOLO11_BUFF::Object> target_results;
  std::vector<YOLO11_BUFF::Object> hit_context;
  for (const auto & result : results) {
    if (is_target_label(result.label, MODE_.num_classes()) && has_rune_keypoints(result)) {
      target_results.push_back(result);
    } else if (is_hit_label(result.label, MODE_.num_classes()) && has_blade_keypoints(result)) {
      hit_context.push_back(result);
    }
  }
  if (cost != nullptr) {
    cost->target_candidates = static_cast<std::uint32_t>(target_results.size());
    cost->hit_candidates = static_cast<std::uint32_t>(hit_context.size());
    cost->result_classification_ns += cost_ns(classification_begin, CostClock::now());
  }

  if (target_results.empty()) {
    if (last_powerrune_.has_value() && !locked_target_missing_since_.has_value()) {
      locked_target_missing_since_ = timestamp;
    }
    mark_switch_deferred_if_recent();
    handle_lose(timestamp);
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::NoRuneCandidate;
      cost->reject_state_commit_ns += cost_ns(classification_begin, CostClock::now());
    }
    return std::nullopt;
  }

  // 2. 目标锁定策略：
  // - 第一击阶段在两块随机点亮装甲中保持连续锁定。
  // - 只有上一锁定灯臂连续消失达到配置阈值，才切到另一块仍待击打装甲。
  bool switch_deferred = false;
  bool target_switched = false;
  const auto target_selection_begin = CostClock::now();
  const int best_idx =
    select_big_buff_target_index(
      target_results, hit_context, bgr_img.size(), timestamp,
      &switch_deferred, &target_switched, cost);
  if (cost != nullptr) {
    cost->target_selection_ns += cost_ns(target_selection_begin, CostClock::now());
    cost->target_switch_deferred = switch_deferred;
    cost->target_switched = target_switched;
  }
  if (switch_deferred) {
    last_switch_deferred_ = true;
    status_ = TEM_LOSE;
    lose_ = std::min(lose_ + 1, LOSE_MAX - 1);
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::TargetSwitchDeferred;
    }
    return std::nullopt;
  }
  if (best_idx < 0 || best_idx >= static_cast<int>(target_results.size())) {
    handle_lose(timestamp);
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::TargetSelectionFailed;
    }
    return std::nullopt;
  }

  const auto fanblade_build_begin = CostClock::now();
  const YOLO11_BUFF::Object final_result = target_results[best_idx];
  last_target_switched_ = target_switched;
  last_selected_target_index_ = best_idx;

  // 3. 将两块待击打目标都转换为 FanBlade，第一块仍为当前瞄准目标。
  std::vector<FanBlade> fanblades;

  auto append_fanblade = [&](const YOLO11_BUFF::Object & object) {
    if (!has_blade_keypoints(object)) return;

    cv::Point2f blade_center(0, 0);
    for (int i = 0; i < 4; ++i) {
      blade_center += object.kpt[i];
    }
    blade_center /= 4.0f;

    std::vector<cv::Point2f> blade_points(object.kpt.begin(), object.kpt.begin() + 4);
    fanblades.emplace_back(FanBlade(blade_points, blade_center, _target));
  };

  append_fanblade(final_result);
  for (int i = 0; i < static_cast<int>(target_results.size()); ++i) {
    if (i == best_idx) {
      continue;
    }
    append_fanblade(target_results[i]);
  }
  if (cost != nullptr) {
    cost->constructed_fanblades = static_cast<std::uint32_t>(fanblades.size());
    cost->fanblade_build_ns += cost_ns(fanblade_build_begin, CostClock::now());
  }

  if (fanblades.empty()) {
    handle_lose(timestamp);
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::NoFanblades;
    }
    return std::nullopt;
  }

  // 4. 获取 R 标中心 (额外将 YOLO 检测到的 R 标作为第4个参数传入)
  const auto r_center_begin = CostClock::now();
  auto r_center = get_r_center(
    fanblades, bgr_img, final_result.label, final_result.kpt[4],
    cost != nullptr ? &cost->r_search : nullptr);
  if (cost != nullptr) {
    cost->r_center_ns += cost_ns(r_center_begin, CostClock::now());
  }

  // 5. 生成 PowerRune 并进行 3D 解算
  PowerRune powerrune;
  powerrune.type = BIG;
  powerrune.fanblades = fanblades;
  powerrune.r_center = r_center;
  powerrune.r_search_debug = last_r_search_debug_;
  powerrune.target_switched = target_switched;
  powerrune.switch_deferred = false;

  // 依赖解算器进行 PnP，为后续卡尔曼滤波提供 XYZ 世界坐标
  const auto pnp_begin = CostClock::now();
  solver.set_image_size(bgr_img.size(), cost != nullptr ? &cost->pnp : nullptr);
  solver.solve(powerrune, cost != nullptr ? &cost->pnp : nullptr);
  if (cost != nullptr) {
    cost->pnp_solve_ns += cost_ns(pnp_begin, CostClock::now());
  }

  if (powerrune.is_unsolve()) {
    handle_lose(timestamp);
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::PnpUnsolved;
    }
    return std::nullopt;
  }

  // 6. 更新状态
  const auto commit_begin = CostClock::now();
  status_ = TRACK;
  lose_ = 0;
  locked_target_missing_since_.reset();
  if (!powerrune.fanblades.empty() &&
      is_finite_point(powerrune.target().center) &&
      is_finite_point(powerrune.r_center)) {
    locked_target_angle_ = blade_angle_around_r(powerrune.target().center, powerrune.r_center);
    locked_target_update_time_ = timestamp;
  }
  last_powerrune_ = std::make_optional(powerrune);
  if (cost != nullptr) {
    cost->reject_state_commit_ns += cost_ns(commit_begin, CostClock::now());
  }
  return last_powerrune_;
}

// 小符逻辑
std::optional<PowerRune> Buff_Detector::solve_small_buff(
  cv::Mat & bgr_img,
  const Solver & solver,
  const std::vector<YOLO11_BUFF::Object> & results,
  BuffDetectorCostSample * cost)
{
  if (results.empty()) {
    handle_lose();
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::RawResultsEmpty;
    }
    return std::nullopt;
  }

  const auto classification_begin = CostClock::now();
  auto result_it = std::find_if(
    results.begin(),
    results.end(),
    [](const YOLO11_BUFF::Object & object) {
      return has_rune_keypoints(object);
    });
  if (cost != nullptr) {
    cost->target_candidates_examined = static_cast<std::uint32_t>(
      result_it == results.end()
        ? results.size()
        : static_cast<std::size_t>(std::distance(results.begin(), result_it)) + 1);
    cost->target_candidates = result_it == results.end() ? 0u : 1u;
    cost->result_classification_ns += cost_ns(classification_begin, CostClock::now());
  }
  if (result_it == results.end()) {
    handle_lose();
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::NoRuneCandidate;
    }
    return std::nullopt;
  }

  const auto fanblade_build_begin = CostClock::now();
  std::vector<FanBlade> fanblades;
  const auto & result = *result_it;
  cv::Point2f blade_center(0, 0);
  for (int i = 0; i < 4; ++i) {
      blade_center += result.kpt[i];
  }
  blade_center /= 4.0;

  // ===== 核心修复点：只把前 4 个角点给 FanBlade =====
  std::vector<cv::Point2f> blade_points(result.kpt.begin(), result.kpt.begin() + 4);
  fanblades.emplace_back(FanBlade(blade_points, blade_center, _target));
  if (cost != nullptr) {
    cost->constructed_fanblades = static_cast<std::uint32_t>(fanblades.size());
    cost->fanblade_build_ns += cost_ns(fanblade_build_begin, CostClock::now());
  }

  // 传参传入 YOLO 识别到的第5个关键点（R标）
  const auto r_center_begin = CostClock::now();
  auto r_center = get_r_center(
    fanblades, bgr_img, result.label, result.kpt[4],
    cost != nullptr ? &cost->r_search : nullptr);
  if (cost != nullptr) {
    cost->r_center_ns += cost_ns(r_center_begin, CostClock::now());
  }

  PowerRune powerrune;
  powerrune.type = SMALL;
  powerrune.fanblades = fanblades;
  powerrune.r_center = r_center;
  powerrune.r_search_debug = last_r_search_debug_;

  // 依赖解算器进行 PnP
  const auto pnp_begin = CostClock::now();
  solver.set_image_size(bgr_img.size(), cost != nullptr ? &cost->pnp : nullptr);
  solver.solve(powerrune, cost != nullptr ? &cost->pnp : nullptr);
  if (cost != nullptr) {
    cost->pnp_solve_ns += cost_ns(pnp_begin, CostClock::now());
  }

  if (powerrune.is_unsolve()) {
    handle_lose();
    if (cost != nullptr) {
      cost->early_exit = BuffDetectorEarlyExit::PnpUnsolved;
    }
    return std::nullopt;
  }

  const auto commit_begin = CostClock::now();
  status_ = TRACK;
  lose_ = 0;
  locked_target_missing_since_.reset();
  locked_target_angle_.reset();
  locked_target_update_time_.reset();
  last_powerrune_ = std::make_optional(powerrune);
  if (cost != nullptr) {
    cost->reject_state_commit_ns += cost_ns(commit_begin, CostClock::now());
  }
  return last_powerrune_;
}

} // namespace auto_buff
