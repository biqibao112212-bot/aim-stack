#include "aim_sim_bridge/pipeline.hpp"
#include "aim_sim_bridge/debug_telemetry.hpp"

#include "AngleSolver.h"
#include "firecontrol.h"
#include "generalDeclaration.h"
#include "mt_detector_tensorrt.hpp"
#include "params.h"
#include "robotestimator.h"
#include "trajectory.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace aim_sim_bridge
{
namespace
{

std::uint8_t taskModeFor(TargetMode mode)
{
    switch (mode) {
    case TargetMode::Outpost:
        return rm::FeedBackData::TASK_MODE::HIT_OUTPOST;
    case TargetMode::Armor:
    default:
        return rm::FeedBackData::TASK_MODE::AUTO_SHOT;
    }
}

std::optional<int> optionalPositiveEnvInt(const char* key)
{
    const char* raw = std::getenv(key);
    if (raw == nullptr || raw[0] == '\0') return std::nullopt;
    try {
        const std::string text(raw);
        std::size_t parsed = 0;
        const int value = std::stoi(text, &parsed);
        if (parsed != text.size() || value <= 0) return std::nullopt;
        return value;
    } catch (...) {
        return std::nullopt;
    }
}

std::string envString(const char* key, const std::string& fallback = {})
{
    const char* raw = std::getenv(key);
    if (raw == nullptr || raw[0] == '\0') return fallback;
    return raw;
}

bool envFlag(const char* key)
{
    const char* raw = std::getenv(key);
    if (raw == nullptr || raw[0] == '\0') return false;
    const std::string text(raw);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" &&
           text != "OFF";
}

bool isPositiveFinite(double value)
{
    return std::isfinite(value) && value > 0.05;
}

std::uint64_t systemNowNs()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::uint64_t elapsedNs(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end)
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

double armorDistanceMeters(const std::shared_ptr<rm::Armor>& armor)
{
    if (!armor) return std::numeric_limits<double>::quiet_NaN();

    if (isPositiveFinite(armor->dis)) {
        return static_cast<double>(armor->dis) / 1000.0;
    }
    if (armor->ypd.allFinite() && isPositiveFinite(armor->ypd.z())) {
        return armor->ypd.z();
    }
    if (armor->armorPosition.allFinite()) {
        const double norm = armor->armorPosition.norm();
        if (isPositiveFinite(norm)) return norm;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double commandDistanceMeters(
    const rm::Estimator& estimator, const std::vector<std::shared_ptr<rm::Armor>>& solved)
{
    if (isPositiveFinite(estimator.distance_)) return estimator.distance_;

    const double tracked_distance = armorDistanceMeters(estimator._trackedArmor);
    if (isPositiveFinite(tracked_distance)) return tracked_distance;

    const double last_tracked_distance = armorDistanceMeters(estimator._last_trackedArmor);
    if (isPositiveFinite(last_tracked_distance)) return last_tracked_distance;

    if (!solved.empty()) {
        const double solved_distance = armorDistanceMeters(solved.front());
        if (isPositiveFinite(solved_distance)) return solved_distance;
    }

    return std::numeric_limits<double>::quiet_NaN();
}

std::string point2Json(const cv::Point2f& point)
{
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) return "null";
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"x\":" << point.x << ",\"y\":" << point.y << "}";
    return out.str();
}

std::string points2Json(const std::vector<cv::Point2f>& points)
{
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (i > 0) out << ',';
        out << point2Json(points[i]);
    }
    out << ']';
    return out.str();
}

std::string intVectorJson(const std::vector<int>& values)
{
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << values[i];
    }
    out << ']';
    return out.str();
}

std::string rectJson(const cv::Rect& rect)
{
    std::ostringstream out;
    out << "{\"x\":" << rect.x
        << ",\"y\":" << rect.y
        << ",\"width\":" << rect.width
        << ",\"height\":" << rect.height << "}";
    return out.str();
}

std::string refinementJson(
    const std::vector<cv::Point2f>& detector_vertices,
    const std::vector<std::shared_ptr<rm::Armor>>& solved)
{
    std::ostringstream out;
    out << std::setprecision(10) << '{';
    bool first = true;
    if (detector_vertices.size() != 4 || solved.empty() || !solved.front() ||
        solved.front()->vertex.size() != detector_vertices.size()) {
        debug::appendBool(out, "available", false, first);
        out << '}';
        return out.str();
    }

    double sum = 0.0;
    double max_delta = 0.0;
    for (std::size_t i = 0; i < detector_vertices.size(); ++i) {
        const double delta = cv::norm(solved.front()->vertex[i] - detector_vertices[i]);
        sum += delta;
        max_delta = std::max(max_delta, delta);
    }
    debug::appendBool(out, "available", true, first);
    debug::appendNumber(out, "mean_delta_px", sum / detector_vertices.size(), first);
    debug::appendNumber(out, "max_delta_px", max_delta, first);
    debug::appendRaw(out, "detector_vertices_px", points2Json(detector_vertices), first);
    debug::appendRaw(out, "pnp_vertices_px", points2Json(solved.front()->vertex), first);
    out << '}';
    return out.str();
}

struct DualFocalDecision
{
    bool enabled = false;
    std::string selected = "wide_6mm";
    std::string reason = "disabled";
    double wide_focal_mm = 6.0;
    double precision_focal_mm = 16.0;
    double scale = 2.0;
    double selected_focal_mm = 6.0;
    double distance_m = std::numeric_limits<double>::quiet_NaN();
    bool target_inside_precision_fov = false;
    double precision_half_width_px_in_wide = std::numeric_limits<double>::quiet_NaN();
    double precision_half_height_px_in_wide = std::numeric_limits<double>::quiet_NaN();
    double precision_min_margin_px = std::numeric_limits<double>::quiet_NaN();
    double wide_armor_width_px = std::numeric_limits<double>::quiet_NaN();
    double precision_equivalent_armor_width_px = std::numeric_limits<double>::quiet_NaN();
    double expected_error_ratio_vs_wide = std::numeric_limits<double>::quiet_NaN();
    bool switch_pending = false;
    std::string next_selected = "wide_6mm";
};

std::string focalProfileId(const char* prefix, double focal_mm)
{
    std::ostringstream out;
    out << prefix << '_';
    if (std::abs(focal_mm - std::round(focal_mm)) < 1e-6) {
        out << static_cast<int>(std::round(focal_mm));
    } else {
        out << std::fixed << std::setprecision(1) << focal_mm;
    }
    out << "mm";
    return out.str();
}

double wrapAngleDeg(double angle)
{
    while (angle > 180.0) angle -= 360.0;
    while (angle < -180.0) angle += 360.0;
    return angle;
}

double smoothScalarCommand(
    double previous,
    double current,
    const AimBridgeConfig& config,
    bool angular_wrap)
{
    double delta = angular_wrap ? wrapAngleDeg(current - previous) : current - previous;
    const double abs_delta = std::abs(delta);
    const double deadband = std::max(0.0, config.command_smoothing_deadband_deg);
    const double passthrough = std::max(deadband, config.command_smoothing_passthrough_deg);
    const double alpha = std::clamp(config.command_smoothing_alpha, 0.0, 1.0);

    if (abs_delta <= deadband) {
        return previous;
    }
    if (abs_delta <= passthrough) {
        return previous + alpha * delta;
    }
    return current;
}

double armorWidthPx(const std::shared_ptr<rm::Armor>& armor)
{
    if (!armor || armor->vertex.size() != 4) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double top = cv::norm(armor->vertex[2] - armor->vertex[1]);
    const double bottom = cv::norm(armor->vertex[3] - armor->vertex[0]);
    return (top + bottom) * 0.5;
}

DualFocalDecision makeWideDecision(const AimBridgeConfig& config, const std::string& reason)
{
    DualFocalDecision decision;
    decision.enabled = config.dual_focal_enabled;
    decision.selected = focalProfileId("wide", config.wide_focal_mm);
    decision.reason = reason;
    decision.wide_focal_mm = config.wide_focal_mm;
    decision.precision_focal_mm = config.precision_focal_mm;
    decision.scale = config.precision_focal_mm > 0.0 && config.wide_focal_mm > 0.0
        ? config.precision_focal_mm / config.wide_focal_mm
        : 1.0;
    decision.selected_focal_mm = config.wide_focal_mm;
    decision.expected_error_ratio_vs_wide = 1.0;
    decision.next_selected = decision.selected;
    return decision;
}

std::string dualFocalJson(const DualFocalDecision& decision)
{
    std::ostringstream out;
    out << std::setprecision(10) << '{';
    bool first = true;
    debug::appendBool(out, "enabled", decision.enabled, first);
    debug::appendString(out, "selected", decision.selected, first);
    debug::appendString(out, "reason", decision.reason, first);
    debug::appendNumber(out, "wide_focal_mm", decision.wide_focal_mm, first);
    debug::appendNumber(out, "precision_focal_mm", decision.precision_focal_mm, first);
    debug::appendNumber(out, "scale", decision.scale, first);
    debug::appendNumber(out, "selected_focal_mm", decision.selected_focal_mm, first);
    debug::appendNumber(out, "distance_m", decision.distance_m, first);
    debug::appendBool(
        out, "target_inside_precision_fov", decision.target_inside_precision_fov, first);
    debug::appendNumber(
        out, "precision_half_width_px_in_wide",
        decision.precision_half_width_px_in_wide, first);
    debug::appendNumber(
        out, "precision_half_height_px_in_wide",
        decision.precision_half_height_px_in_wide, first);
    debug::appendNumber(
        out, "precision_min_margin_px", decision.precision_min_margin_px, first);
    debug::appendNumber(out, "wide_armor_width_px", decision.wide_armor_width_px, first);
    debug::appendNumber(
        out, "precision_equivalent_armor_width_px",
        decision.precision_equivalent_armor_width_px, first);
    debug::appendNumber(
        out, "expected_error_ratio_vs_wide",
        decision.expected_error_ratio_vs_wide, first);
    debug::appendBool(out, "switch_pending", decision.switch_pending, first);
    debug::appendString(out, "next_selected", decision.next_selected, first);
    out << '}';
    return out.str();
}

cv::Mat cropScaledCameraMatrix(
    const cv::Mat& base_camera_matrix, const cv::Rect& roi, double scale_x, double scale_y)
{
    cv::Mat out;
    base_camera_matrix.convertTo(out, CV_64F);
    out.at<double>(0, 0) *= scale_x;
    out.at<double>(1, 1) *= scale_y;
    out.at<double>(0, 2) = (out.at<double>(0, 2) - roi.x) * scale_x;
    out.at<double>(1, 2) = (out.at<double>(1, 2) - roi.y) * scale_y;
    return out;
}

cv::Rect precisionSourceRoi(
    int image_width, int image_height, const cv::Mat& base_camera_matrix, double scale)
{
    if (image_width <= 0 || image_height <= 0 || scale <= 1.0) {
        return cv::Rect(0, 0, image_width, image_height);
    }

    const int crop_width = std::max(8, static_cast<int>(std::round(image_width / scale)));
    const int crop_height = std::max(8, static_cast<int>(std::round(image_height / scale)));
    const double cx = !base_camera_matrix.empty() ? base_camera_matrix.at<double>(0, 2)
                                                  : (image_width - 1) * 0.5;
    const double cy = !base_camera_matrix.empty() ? base_camera_matrix.at<double>(1, 2)
                                                  : (image_height - 1) * 0.5;
    int x = static_cast<int>(std::round(cx - crop_width * 0.5));
    int y = static_cast<int>(std::round(cy - crop_height * 0.5));
    x = std::max(0, std::min(x, image_width - crop_width));
    y = std::max(0, std::min(y, image_height - crop_height));
    return cv::Rect(x, y, crop_width, crop_height);
}

cv::Mat cameraMatrixForFrame(const SimFrame& frame, const rm::AngleSolver& angle_solver)
{
    cv::Mat out;
    if (frame.has_camera_matrix_override && !frame.camera_matrix_override.empty()) {
        frame.camera_matrix_override.convertTo(out, CV_64F);
        if (out.rows == 3 && out.cols == 3) {
            return out;
        }
        out.release();
    }

    const cv::Mat configured = angle_solver.configuredCameraMatrix();
    if (!configured.empty()) {
        configured.convertTo(out, CV_64F);
    }
    return out;
}

std::vector<cv::Point2f> sourceVerticesFromActiveFrame(
    const SimFrame& frame, const std::vector<cv::Point2f>& active_vertices)
{
    std::vector<cv::Point2f> source_vertices;
    source_vertices.reserve(active_vertices.size());
    const double sx = frame.virtual_scale_x > 1e-6 ? frame.virtual_scale_x : 1.0;
    const double sy = frame.virtual_scale_y > 1e-6 ? frame.virtual_scale_y : 1.0;
    for (const auto& point : active_vertices) {
        source_vertices.emplace_back(
            static_cast<float>(frame.source_roi_px.x + point.x / sx),
            static_cast<float>(frame.source_roi_px.y + point.y / sy));
    }
    return source_vertices;
}

std::string eigen3Json(const Eigen::Vector3d& value)
{
    if (!value.allFinite()) return "null";
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"x\":" << value.x()
        << ",\"y\":" << value.y()
        << ",\"z\":" << value.z() << "}";
    return out.str();
}

double matAt3x1(const cv::Mat& mat, int row)
{
    if (mat.empty() || mat.rows < 3 || mat.cols < 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (mat.depth() == CV_64F) return mat.at<double>(row, 0);
    if (mat.depth() == CV_32F) return static_cast<double>(mat.at<float>(row, 0));
    return std::numeric_limits<double>::quiet_NaN();
}

std::string mat3x1Json(const cv::Mat& mat)
{
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"x\":" << (std::isfinite(matAt3x1(mat, 0)) ? std::to_string(matAt3x1(mat, 0)) : "null")
        << ",\"y\":" << (std::isfinite(matAt3x1(mat, 1)) ? std::to_string(matAt3x1(mat, 1)) : "null")
        << ",\"z\":" << (std::isfinite(matAt3x1(mat, 2)) ? std::to_string(matAt3x1(mat, 2)) : "null")
        << "}";
    return out.str();
}

std::string jsonNumber(double value)
{
    std::ostringstream out;
    out << std::setprecision(10);
    if (std::isfinite(value)) {
        out << value;
    } else {
        out << "null";
    }
    return out.str();
}

std::string eigenVectorJson(const Eigen::VectorXd& value)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    for (Eigen::Index i = 0; i < value.size(); ++i) {
        if (i > 0) out << ',';
        out << jsonNumber(value(i));
    }
    out << ']';
    return out.str();
}

std::string eigenMatrix3Json(const Eigen::Matrix3d& value)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    for (int r = 0; r < 3; ++r) {
        if (r > 0) out << ',';
        out << '[';
        for (int c = 0; c < 3; ++c) {
            if (c > 0) out << ',';
            out << jsonNumber(value(r, c));
        }
        out << ']';
    }
    out << ']';
    return out.str();
}

std::string eigenMatrixJson(const Eigen::MatrixXd& value)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    for (Eigen::Index r = 0; r < value.rows(); ++r) {
        if (r > 0) out << ',';
        out << '[';
        for (Eigen::Index c = 0; c < value.cols(); ++c) {
            if (c > 0) out << ',';
            out << jsonNumber(value(r, c));
        }
        out << ']';
    }
    out << ']';
    return out.str();
}

std::string eigenVector4Json(const Eigen::Vector4d& value)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    for (int i = 0; i < 4; ++i) {
        if (i > 0) out << ',';
        out << jsonNumber(value(i));
    }
    out << ']';
    return out.str();
}

const char* observationRejectReasonName(int reason)
{
    using Tracker = RobotEstimator::YpdAngleTracker;
    switch (reason) {
    case Tracker::kObservationAccepted:
        return "accepted";
    case Tracker::kObservationSkippedByPairSelection:
        return "skipped_by_pair_selection";
    case Tracker::kObservationSkippedByPrimarySelection:
        return "skipped_by_primary_selection";
    case Tracker::kObservationRejectedByPriorNis:
        return "rejected_by_prior_nis";
    case Tracker::kObservationRejectedByPhysicalGate:
        return "rejected_by_physical_gate";
    case Tracker::kObservationSkippedByBatchCapacity:
        return "skipped_by_batch_capacity";
    default:
        return "not_evaluated";
    }
}

std::string observationDiagnosticsJson(
    const std::vector<RobotEstimator::YpdAngleTracker::ObservationDiagnostic>& diagnostics)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    for (std::size_t i = 0; i < diagnostics.size(); ++i) {
        if (i > 0) out << ',';
        const auto& diagnostic = diagnostics[i];
        out << '{';
        bool first = true;
        debug::appendInt(out, "observation_index", diagnostic.observation_index, first);
        debug::appendInt(out, "matched_slot", diagnostic.matched_slot, first);
        debug::appendBool(out, "accepted", diagnostic.accepted, first);
        debug::appendInt(out, "reject_reason", diagnostic.reject_reason, first);
        debug::appendString(
            out, "reject_reason_name",
            observationRejectReasonName(diagnostic.reject_reason), first);
        debug::appendInt(
            out, "physical_reject_reason", diagnostic.physical_reject_reason, first);
        debug::appendBool(out, "yaw_pi_flip", diagnostic.yaw_pi_flip, first);
        debug::appendBool(
            out, "freeze_normal_geometry", diagnostic.freeze_normal_geometry, first);
        debug::appendNumber(
            out, "max_center_jump_m", diagnostic.max_center_jump_m, first);
        debug::appendRaw(out, "observation_ypd_yaw", eigenVector4Json(diagnostic.observation), first);
        debug::appendRaw(
            out, "prior_predicted_measurement_ypd_yaw",
            eigenVector4Json(diagnostic.prior_predicted_measurement), first);
        debug::appendRaw(
            out, "prior_innovation_ypd_yaw",
            eigenVector4Json(diagnostic.prior_innovation), first);
        debug::appendNumber(out, "prior_nis", diagnostic.prior_nis, first);
        debug::appendRaw(
            out, "posterior_innovation_ypd_yaw",
            eigenVector4Json(diagnostic.posterior_innovation), first);
        debug::appendNumber(out, "posterior_nis", diagnostic.posterior_nis, first);
        debug::appendInt(
            out, "update_count_before", diagnostic.update_count_before, first);
        debug::appendInt(out, "update_count_after", diagnostic.update_count_after, first);
        out << '}';
    }
    out << ']';
    return out.str();
}

std::string velocityHistoryJson(
    const std::deque<RobotEstimator::YpdAngleTracker::MotionSample>& history,
    double current_frame_yaw)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    bool first_sample = true;
    for (const auto& sample : history) {
        if (!first_sample) out << ',';
        first_sample = false;
        double reexpressed_x = std::numeric_limits<double>::quiet_NaN();
        double reexpressed_y = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(sample.frame_yaw) && std::isfinite(current_frame_yaw) &&
            std::isfinite(sample.center_x) && std::isfinite(sample.center_y)) {
            const double relative_yaw = std::remainder(
                sample.frame_yaw - current_frame_yaw, 2.0 * M_PI);
            const double c = std::cos(relative_yaw);
            const double s = std::sin(relative_yaw);
            reexpressed_x = c * sample.center_x - s * sample.center_y;
            reexpressed_y = s * sample.center_x + c * sample.center_y;
        }
        out << '{';
        bool first = true;
        debug::appendNumber(out, "t_s", sample.t_s, first);
        debug::appendNumber(out, "original_x_m", sample.center_x, first);
        debug::appendNumber(out, "original_y_m", sample.center_y, first);
        debug::appendNumber(out, "original_frame_yaw_rad", sample.frame_yaw, first);
        debug::appendNumber(out, "reexpressed_current_x_m", reexpressed_x, first);
        debug::appendNumber(out, "reexpressed_current_y_m", reexpressed_y, first);
        debug::appendInt(out, "source", sample.source, first);
        debug::appendInt(out, "group_id", sample.group_id, first);
        out << '}';
    }
    out << ']';
    return out.str();
}

std::string cvMatJson(const cv::Mat& mat)
{
    if (mat.empty()) return "null";

    cv::Mat as_double;
    mat.convertTo(as_double, CV_64F);
    std::ostringstream out;
    out << std::setprecision(10);
    const bool vector_like = as_double.rows == 1 || as_double.cols == 1;
    if (vector_like) {
        out << '[';
        for (int i = 0; i < static_cast<int>(as_double.total()); ++i) {
            if (i > 0) out << ',';
            const int row = as_double.rows == 1 ? 0 : i;
            const int col = as_double.rows == 1 ? i : 0;
            out << jsonNumber(as_double.at<double>(row, col));
        }
        out << ']';
        return out.str();
    }

    out << '[';
    for (int r = 0; r < as_double.rows; ++r) {
        if (r > 0) out << ',';
        out << '[';
        for (int c = 0; c < as_double.cols; ++c) {
            if (c > 0) out << ',';
            out << jsonNumber(as_double.at<double>(r, c));
        }
        out << ']';
    }
    out << ']';
    return out.str();
}

std::string movementName(rm::MOVEMENT movement)
{
    switch (movement) {
    case rm::STATIC:
        return "static";
    case rm::TRANSLATION:
        return "translation";
    case rm::SPINNING:
        return "spinning";
    case rm::TRANSPIN:
        return "translation+spin";
    default:
        return "unknown";
    }
}

std::string firePhaseModeName(rm::FirePhaseMode mode)
{
    switch (mode) {
    case rm::FirePhaseMode::None:
        return "none";
    case rm::FirePhaseMode::Auto:
        return "auto";
    case rm::FirePhaseMode::Single:
        return "single";
    default:
        return "unknown";
    }
}

std::string fireRuntimeStatsJson(const rm::FireControlRuntimeStats& stats)
{
    std::ostringstream out;
    out << std::setprecision(10) << '{';
    bool first = true;
    debug::appendNumber(out, "wall_control_dt_ms", stats.wall_control_dt_ms, first);
    debug::appendNumber(out, "frame_control_dt_ms", stats.frame_control_dt_ms, first);
    debug::appendNumber(out, "control_dt_ms", stats.control_dt_ms, first);
    debug::appendNumber(out, "planner_us", stats.planner_us, first);
    debug::appendNumber(out, "yaw_ctrl_us", stats.yaw_ctrl_us, first);
    debug::appendNumber(out, "yaw_solve_us", stats.yaw_solve_us, first);
    debug::appendNumber(out, "pitch_ctrl_us", stats.pitch_ctrl_us, first);
    debug::appendNumber(out, "pitch_solve_us", stats.pitch_solve_us, first);
    debug::appendNumber(out, "total_us", stats.total_us, first);
    debug::appendBool(out, "planner_active", stats.planner_active, first);
    debug::appendBool(out, "preview_mpc_active", stats.preview_mpc_active, first);
    debug::appendBool(out, "target_detected", stats.target_detected, first);
    out << '}';
    return out.str();
}

std::string calibrationJson(const rm::AngleSolver& angle_solver)
{
    std::ostringstream out;
    out << std::setprecision(10) << '{';
    bool first = true;
    debug::appendBool(
        out, "extrinsic_enabled", angle_solver.cameraGimbalExtrinsicEnabled(), first);
    debug::appendBool(
        out, "extrinsic_from_config", angle_solver.cameraGimbalExtrinsicFromConfig(), first);
    debug::appendString(
        out, "camera_frame", "OpenCV: +x right, +y down, +z forward", first);
    debug::appendString(
        out, "gimbal_frame", "solver gimbal frame, configured by R_camera2gimbal",
        first);
    debug::appendNumber(out, "legacy_h_m", angle_solver.legacyHeightM(), first);
    debug::appendNumber(out, "aiming_cx_px", angle_solver.aimingOffsetCxPx(), first);
    debug::appendNumber(out, "aiming_cy_px", angle_solver.aimingOffsetCyPx(), first);
    debug::appendBool(
        out, "apply_aiming_offset_to_intrinsics",
        angle_solver.applyAimingOffsetToIntrinsics(), first);
    debug::appendRaw(
        out, "R_camera2gimbal",
        eigenMatrix3Json(angle_solver.cameraToGimbalRotation()), first);
    debug::appendRaw(
        out, "t_camera2gimbal_m",
        eigen3Json(angle_solver.cameraToGimbalTranslationM()), first);
    debug::appendRaw(
        out, "camera_matrix_configured",
        cvMatJson(angle_solver.configuredCameraMatrix()), first);
    debug::appendRaw(
        out, "camera_matrix_effective",
        cvMatJson(angle_solver._cam_instant_matrix), first);
    debug::appendRaw(
        out, "distortion_coeffs",
        cvMatJson(angle_solver.configuredDistortionCoeffs()), first);
    out << '}';
    return out.str();
}

std::string armorTypeName(rm::ArmorType type)
{
    const int index = static_cast<int>(type);
    if (index >= 0 && index < 3) return rm::ARMOR_TYPE_STR[index];
    return "unknown";
}

double yawFromPositionDeg(const Eigen::Vector3d& position)
{
    if (!position.allFinite()) return std::numeric_limits<double>::quiet_NaN();
    return std::atan2(position.y(), position.x()) * rm::R2D;
}

double pitchFromPositionDeg(const Eigen::Vector3d& position)
{
    if (!position.allFinite()) return std::numeric_limits<double>::quiet_NaN();
    const double xy = std::hypot(position.x(), position.y());
    return std::atan2(position.z(), xy) * rm::R2D;
}

double imageYawDeg(const cv::Point2f& point, const cv::Mat& camera_matrix)
{
    if (!std::isfinite(point.x) || camera_matrix.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double fx = camera_matrix.at<double>(0, 0);
    const double cx = camera_matrix.at<double>(0, 2);
    return std::atan2(static_cast<double>(point.x) - cx, fx) * rm::R2D;
}

double imagePitchDownPositiveDeg(const cv::Point2f& point, const cv::Mat& camera_matrix)
{
    if (!std::isfinite(point.y) || camera_matrix.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double fy = camera_matrix.at<double>(1, 1);
    const double cy = camera_matrix.at<double>(1, 2);
    return std::atan2(static_cast<double>(point.y) - cy, fy) * rm::R2D;
}

double imagePitchCorrectedCommandDeg(
    double current_pitch_deg,
    const std::shared_ptr<rm::Armor>& armor,
    const cv::Mat& camera_matrix)
{
    if (!armor) return std::numeric_limits<double>::quiet_NaN();
    const double image_pitch_down_deg = imagePitchDownPositiveDeg(armor->center, camera_matrix);
    if (!std::isfinite(current_pitch_deg) || !std::isfinite(image_pitch_down_deg)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return current_pitch_deg - image_pitch_down_deg;
}

bool finiteImagePoint(const cv::Point2f& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y);
}

double imageYawCorrectedCommandDegForPoint(
    double current_yaw_deg,
    const cv::Point2f& center,
    const cv::Mat& camera_matrix)
{
    const double image_yaw_deg = imageYawDeg(center, camera_matrix);
    if (!std::isfinite(current_yaw_deg) || !std::isfinite(image_yaw_deg)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return current_yaw_deg + image_yaw_deg;
}

double imagePitchCorrectedCommandDegForPoint(
    double current_pitch_deg,
    const cv::Point2f& center,
    const cv::Mat& camera_matrix)
{
    const double image_pitch_down_deg = imagePitchDownPositiveDeg(center, camera_matrix);
    if (!std::isfinite(current_pitch_deg) || !std::isfinite(image_pitch_down_deg)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return current_pitch_deg - image_pitch_down_deg;
}

double ballisticImagePitchCommandDegForPoint(
    double current_pitch_deg,
    const cv::Point2f& center,
    const cv::Mat& camera_matrix,
    double distance_m,
    double bullet_speed_mps)
{
    const double line_pitch_deg =
        imagePitchCorrectedCommandDegForPoint(current_pitch_deg, center, camera_matrix);
    if (!std::isfinite(line_pitch_deg) || !isPositiveFinite(distance_m) ||
        !isPositiveFinite(bullet_speed_mps)) {
        return line_pitch_deg;
    }

    const double line_pitch_rad = line_pitch_deg * rm::D2R;
    const double horizontal_m = distance_m * std::cos(line_pitch_rad);
    const double height_m = distance_m * std::sin(line_pitch_rad);
    if (!isPositiveFinite(horizontal_m)) return line_pitch_deg;

    tools::Trajectory trajectory(bullet_speed_mps, horizontal_m, height_m);
    if (trajectory.unsolvable || !std::isfinite(trajectory.pitch)) {
        return line_pitch_deg;
    }
    return trajectory.pitch * rm::R2D;
}

double ballisticImagePitchCommandDeg(
    double current_pitch_deg,
    const std::shared_ptr<rm::Armor>& armor,
    const cv::Mat& camera_matrix,
    double distance_m,
    double bullet_speed_mps)
{
    const double line_pitch_deg =
        imagePitchCorrectedCommandDeg(current_pitch_deg, armor, camera_matrix);
    if (!std::isfinite(line_pitch_deg) || !isPositiveFinite(distance_m) ||
        !isPositiveFinite(bullet_speed_mps)) {
        return line_pitch_deg;
    }

    const double line_pitch_rad = line_pitch_deg * rm::D2R;
    const double horizontal_m = distance_m * std::cos(line_pitch_rad);
    const double height_m = distance_m * std::sin(line_pitch_rad);
    if (!isPositiveFinite(horizontal_m)) return line_pitch_deg;

    tools::Trajectory trajectory(bullet_speed_mps, horizontal_m, height_m);
    if (trajectory.unsolvable || !std::isfinite(trajectory.pitch)) {
        return line_pitch_deg;
    }
    return trajectory.pitch * rm::R2D;
}

double physicalArmorWidthMeters(int number, rm::ArmorType type)
{
    if (type == rm::ArmorType::LARGE || number == 1) {
        return 0.225;
    }
    return 0.135;
}

double detectedArmorWidthPx(const std::vector<cv::Point2f>& vertices)
{
    if (vertices.size() != 4) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double top = cv::norm(vertices[2] - vertices[1]);
    const double bottom = cv::norm(vertices[3] - vertices[0]);
    const double width = (top + bottom) * 0.5;
    return isPositiveFinite(width) ? width : std::numeric_limits<double>::quiet_NaN();
}

double estimateDetectedDistanceMeters(
    const std::vector<cv::Point2f>& vertices,
    int number,
    rm::ArmorType type,
    const cv::Mat& camera_matrix)
{
    if (camera_matrix.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double width_px = detectedArmorWidthPx(vertices);
    const double fx = camera_matrix.at<double>(0, 0);
    if (!isPositiveFinite(width_px) || !isPositiveFinite(fx)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return physicalArmorWidthMeters(number, type) * fx / width_px;
}

double imageYawCorrectedCommandDeg(
    double current_yaw_deg,
    const std::shared_ptr<rm::Armor>& armor,
    const cv::Mat& camera_matrix)
{
    if (!armor) return std::numeric_limits<double>::quiet_NaN();
    const double image_yaw_deg = imageYawDeg(armor->center, camera_matrix);
    if (!std::isfinite(current_yaw_deg) || !std::isfinite(image_yaw_deg)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return current_yaw_deg + image_yaw_deg;
}

bool imageAlignedForSimulatorFire(
    const std::shared_ptr<rm::Armor>& armor,
    const cv::Mat& camera_matrix)
{
    if (!armor) return false;
    const double yaw_error_deg = imageYawDeg(armor->center, camera_matrix);
    const double pitch_error_deg = imagePitchDownPositiveDeg(armor->center, camera_matrix);
    return std::isfinite(yaw_error_deg) && std::isfinite(pitch_error_deg) &&
           std::abs(yaw_error_deg) <= 1.2 && std::abs(pitch_error_deg) <= 1.2;
}

bool imageYawAlignedForSimulatorFire(
    const std::shared_ptr<rm::Armor>& armor,
    const cv::Mat& camera_matrix)
{
    if (!armor) return false;
    const double yaw_error_deg = imageYawDeg(armor->center, camera_matrix);
    return std::isfinite(yaw_error_deg) && std::abs(yaw_error_deg) <= 1.2;
}

std::shared_ptr<rm::Armor> pitchReferenceArmor(
    const std::vector<std::shared_ptr<rm::Armor>>& solved)
{
    if (!solved.empty() && solved.front()) return solved.front();
    return nullptr;
}

std::shared_ptr<rm::Armor> selectedReferenceArmor(
    const rm::Estimator& estimator, const std::vector<std::shared_ptr<rm::Armor>>& solved)
{
    const int primary = estimator._current_primary_observation_index;
    if (primary >= 0 &&
        static_cast<std::size_t>(primary) < estimator._current_tracker_input_armors.size()) {
        const auto& armor = estimator._current_tracker_input_armors[primary];
        return std::make_shared<rm::Armor>(armor);
    }
    if (estimator._trackedArmor) return estimator._trackedArmor;
    if (estimator._last_trackedArmor) return estimator._last_trackedArmor;
    return pitchReferenceArmor(solved);
}

std::string doubleVectorJson(const std::vector<double>& values)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) out << ',';
        out << jsonNumber(values[index]);
    }
    out << ']';
    return out.str();
}

std::string parallelJointCandidatesJson(
    const std::vector<rm::ParallelJointPnPCandidate>& candidates)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (index > 0) out << ',';
        const auto& candidate = candidates[index];
        out << '{';
        bool first = true;
        debug::appendUInt(out, "id", candidate.id, first);
        debug::appendBool(out, "selected", candidate.selected, first);
        debug::appendNumber(out, "yaw_rad", candidate.yaw_rad, first);
        debug::appendNumber(out, "yaw_deg", candidate.yaw_rad * rm::R2D, first);
        debug::appendNumber(
            out, "coarse_seed_yaw_deg", candidate.coarse_seed_yaw_rad * rm::R2D,
            first);
        debug::appendRaw(out, "rvec", mat3x1Json(candidate.rVec), first);
        debug::appendRaw(out, "tvec_mm", mat3x1Json(candidate.tVec), first);
        debug::appendNumber(
            out, "reprojection_rms_px", candidate.reprojection_error_px, first);
        debug::appendNumber(
            out, "reprojection_max_px", candidate.max_reprojection_error_px, first);
        debug::appendRaw(
            out, "corner_residual_px", doubleVectorJson(candidate.corner_residual_px),
            first);
        debug::appendNumber(
            out, "translation_linear_condition",
            candidate.translation_linear_condition, first);
        debug::appendNumber(
            out, "translation_information_condition",
            candidate.translation_information_condition, first);
        debug::appendNumber(
            out, "yaw_sensitivity_deg_per_px",
            candidate.yaw_sensitivity_deg_per_px, first);
        debug::appendBool(
            out, "yaw_sensitivity_valid", candidate.yaw_sensitivity_valid, first);
        debug::appendInt(out, "iterations", candidate.iterations, first);
        debug::appendBool(out, "converged", candidate.converged, first);
        debug::appendBool(out, "improved", candidate.improved, first);
        debug::appendBool(out, "positive_depth", candidate.positive_depth, first);
        debug::appendBool(out, "search_bound_hit", candidate.search_bound_hit, first);
        out << '}';
    }
    out << ']';
    return out.str();
}

std::string pnpParallelAbJson(const rm::Armor& armor)
{
    std::ostringstream out;
    out << std::setprecision(10) << '{';
    bool first = true;
    debug::appendString(out, "schema", "pnp_joint_ab.v2", first);
    debug::appendString(out, "orientation_frame", "tracker_chassis", first);
    debug::appendString(out, "pose_projection", "exposure_camera", first);
    debug::appendString(
        out, "camera_parameters", "per_frame_effective_intrinsics_and_calibration_distortion",
        first);
    debug::appendString(out, "tracker_consumes", "exposure_constrained_chassis_yaw", first);
    debug::appendNumber(
        out, "fixed_tilt_deg",
        armor.number == rm::Armor::LABEL::OUTPOST ? -15.0 : 15.0, first);

    std::ostringstream legacy;
    legacy << '{';
    bool legacy_first = true;
    debug::appendString(legacy, "method", "fixed_tvec_yaw_grid", legacy_first);
    debug::appendNumber(
        legacy, "yaw_absolute_deg",
        (std::isfinite(armor.legacy_camera_fixed_yaw)
             ? armor.legacy_camera_fixed_yaw
             : armor.yaw_absolute) * rm::R2D,
        legacy_first);
    debug::appendNumber(
        legacy, "reprojection_rms_px",
        armor.legacy_constrained_reprojection_error_px, legacy_first);
    debug::appendNumber(
        legacy, "reprojection_max_px",
        armor.legacy_constrained_max_reprojection_error_px, legacy_first);
    debug::appendRaw(
        legacy, "corner_residual_px",
        doubleVectorJson(armor.legacy_constrained_corner_residual_px), legacy_first);
    debug::appendRaw(legacy, "tvec_mm", mat3x1Json(armor.tVec), legacy_first);
    legacy << '}';
    debug::appendRaw(out, "legacy", legacy.str(), first);
    debug::appendNumber(out, "corrected_chassis_yaw_deg", armor.yaw_absolute * rm::R2D, first);

    std::ostringstream joint_refined;
    joint_refined << '{';
    bool refined_first = true;
    debug::appendString(
        joint_refined, "method", "fixed_tilt_joint_yaw_translation_lm", refined_first);
    debug::appendString(
        joint_refined, "corners", "subpixel_refined_bl_tl_tr_br", refined_first);
    debug::appendNumber(
        joint_refined, "solve_us", armor.parallel_joint_solve_us, refined_first);
    debug::appendNumber(
        joint_refined, "constrained_reprojection_rms_px",
        armor.exposure_constrained_reprojection_error_px, refined_first);
    debug::appendNumber(
        joint_refined, "constrained_reprojection_max_px",
        armor.exposure_constrained_max_reprojection_error_px, refined_first);
    debug::appendRaw(
        joint_refined, "constrained_corner_residual_px",
        doubleVectorJson(armor.exposure_constrained_corner_residual_px), refined_first);
    debug::appendRaw(
        joint_refined, "candidates",
        parallelJointCandidatesJson(armor.parallel_joint_candidates), refined_first);
    joint_refined << '}';
    debug::appendRaw(out, "joint_refined", joint_refined.str(), first);

    std::ostringstream joint_raw;
    joint_raw << '{';
    bool raw_first = true;
    debug::appendString(
        joint_raw, "method", "fixed_tilt_joint_yaw_translation_lm", raw_first);
    debug::appendString(joint_raw, "corners", "detector_raw_bl_tl_tr_br", raw_first);
    debug::appendNumber(
        joint_raw, "solve_us", armor.parallel_joint_raw_solve_us, raw_first);
    debug::appendRaw(
        joint_raw, "candidates",
        parallelJointCandidatesJson(armor.parallel_joint_raw_candidates), raw_first);
    joint_raw << '}';
    debug::appendRaw(out, "joint_raw", joint_raw.str(), first);
    out << '}';
    return out.str();
}

std::string armorJson(
    const std::shared_ptr<rm::Armor>& armor,
    const cv::Mat& camera_matrix,
    bool include_parallel_pnp = false)
{
    if (!armor) return "null";

    std::ostringstream out;
    out << std::setprecision(10) << '{';
    bool first = true;
    debug::appendUInt(out, "observation_id", armor->observation_id, first);
    debug::appendInt(out, "number", armor->number, first);
    debug::appendInt(out, "color", armor->color, first);
    debug::appendString(out, "type", armorTypeName(armor->type), first);
    debug::appendRaw(out, "center_px", point2Json(armor->center), first);
    debug::appendRaw(out, "pnp_vertices_px", points2Json(armor->vertex), first);
    debug::appendString(out, "pnp_vertex_order", "bl,tl,tr,br as built from lights", first);
    debug::appendRaw(out, "corner_covariance_px2", "null", first);
    debug::appendString(out, "corner_covariance_status", "unavailable", first);
    std::ostringstream pnp_candidates;
    pnp_candidates << '[';
    for (std::size_t index = 0; index < armor->pnp_candidates.size(); ++index) {
        if (index > 0) pnp_candidates << ',';
        const auto& candidate = armor->pnp_candidates[index];
        pnp_candidates << '{';
        bool candidate_first = true;
        debug::appendUInt(pnp_candidates, "id", candidate.id, candidate_first);
        debug::appendUInt(
            pnp_candidates, "solver_solution_index", candidate.solver_solution_index,
            candidate_first);
        debug::appendBool(pnp_candidates, "selected", candidate.selected, candidate_first);
        debug::appendNumber(
            pnp_candidates, "reprojection_error_px", candidate.reprojection_error_px,
            candidate_first);
        debug::appendString(
            pnp_candidates, "corner_order", "detector_canonical_bl_tl_tr_br",
            candidate_first);
        debug::appendString(
            pnp_candidates, "polarity", "nominal_not_a_reversed_normal_hypothesis",
            candidate_first);
        debug::appendRaw(pnp_candidates, "rvec", mat3x1Json(candidate.rVec), candidate_first);
        debug::appendRaw(
            pnp_candidates, "tvec_mm", mat3x1Json(candidate.tVec), candidate_first);
        pnp_candidates << '}';
    }
    pnp_candidates << ']';
    debug::appendRaw(out, "pnp_candidates", pnp_candidates.str(), first);
    debug::appendRaw(out, "relative_slot_candidates", "[0,1,2,3]", first);
    // Armor outward normal is unique under the detector/model contract. Do not
    // manufacture a reversed-normal branch in downstream estimators.
    debug::appendRaw(out, "normal_polarity_candidates", "[1]", first);
    debug::appendBool(out, "semantic_hypotheses_deferred_to_estimator", false, first);
    debug::appendRaw(out, "rvec", mat3x1Json(armor->rVec), first);
    debug::appendRaw(out, "tvec_mm", mat3x1Json(armor->tVec), first);
    debug::appendRaw(out, "position_m", eigen3Json(armor->armorPosition), first);
    debug::appendRaw(out, "ypd", eigen3Json(armor->ypd), first);
    debug::appendNumber(out, "ypd_yaw_deg", armor->ypd.x() * rm::R2D, first);
    debug::appendNumber(out, "ypd_pitch_deg", armor->ypd.y() * rm::R2D, first);
    debug::appendNumber(out, "position_yaw_deg", yawFromPositionDeg(armor->armorPosition), first);
    debug::appendNumber(out, "position_pitch_deg", pitchFromPositionDeg(armor->armorPosition), first);
    debug::appendNumber(out, "image_yaw_deg", imageYawDeg(armor->center, camera_matrix), first);
    debug::appendNumber(
        out, "image_pitch_down_positive_deg",
        imagePitchDownPositiveDeg(armor->center, camera_matrix), first);
    debug::appendNumber(out, "armor_yaw_deg", armor->yaw * rm::R2D, first);
    debug::appendNumber(out, "armor_yaw_absolute_deg", armor->yaw_absolute * rm::R2D, first);
    debug::appendNumber(out, "armor_yaw_raw_deg", armor->yaw_raw * rm::R2D, first);
    debug::appendNumber(out, "distance_mm", armor->dis, first);
    debug::appendNumber(out, "distance_to_image_center_px", armor->distanceToImageCenter, first);
    if (include_parallel_pnp &&
        (!armor->parallel_joint_candidates.empty() ||
         !armor->parallel_joint_raw_candidates.empty())) {
        debug::appendRaw(out, "pnp_ab", pnpParallelAbJson(*armor), first);
    }
    out << '}';
    return out.str();
}

std::string detectorCandidatesJson(
    const std::vector<rm::ArmorForDetect>& armors,
    const std::vector<std::shared_ptr<rm::Armor>>& solved,
    int active_target_number)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    for (std::size_t index = 0; index < armors.size(); ++index) {
        if (index > 0) out << ',';
        const auto& armor = armors[index];
        out << '{';
        bool first = true;
        debug::appendUInt(out, "observation_id", armor.observation_id, first);
        debug::appendInt(out, "number", armor.number, first);
        debug::appendInt(out, "color", armor.color, first);
        debug::appendString(out, "type", armorTypeName(armor.type), first);
        debug::appendNumber(out, "confidence", armor.confidence, first);
        debug::appendRaw(out, "center_px", point2Json(armor.center), first);
        debug::appendRaw(out, "raw_corners_px", points2Json(armor.vertex), first);
        debug::appendString(out, "raw_corner_order", "bl,tl,tr,br", first);
        debug::appendRaw(out, "raw_corner_covariance_px2", "null", first);
        debug::appendString(out, "raw_corner_covariance_status", "unavailable", first);
        const bool matched = std::any_of(
            solved.begin(), solved.end(), [&](const std::shared_ptr<rm::Armor>& candidate) {
                return candidate && candidate->observation_id == armor.observation_id;
            });
        const char* reject_reason = matched
                                        ? "accepted_by_pnp"
                                        : (active_target_number > 0 &&
                                                   armor.number != active_target_number
                                               ? "active_target_filter"
                                               : "no_finite_pnp_solution_or_nonpositive_distance");
        debug::appendString(out, "reject_reason", reject_reason, first);
        out << '}';
    }
    out << ']';
    return out.str();
}

Eigen::Matrix3d exposureGimbalPoseRotation(
    const rm::AngleSolver& solver, const rm::FrameMeta& frame_meta)
{
    const double yaw = frame_meta.poseEuler.yaw * rm::D2R;
    const double pitch = frame_meta.poseEuler.pitch * rm::D2R;
    Eigen::Matrix3d yaw_rotation;
    yaw_rotation << std::cos(yaw), 0.0, std::sin(yaw), 0.0, 1.0, 0.0,
        -std::sin(yaw), 0.0, std::cos(yaw);
    Eigen::Matrix3d pitch_rotation;
    pitch_rotation << 1.0, 0.0, 0.0, 0.0, std::cos(pitch), -std::sin(pitch),
        0.0, std::sin(pitch), std::cos(pitch);
    const Eigen::Matrix3d camera_to_gimbal = solver.cameraToGimbalRotation();
    return camera_to_gimbal * yaw_rotation * pitch_rotation * camera_to_gimbal.transpose();
}

std::string armorsJson(
    const std::vector<std::shared_ptr<rm::Armor>>& armors,
    const cv::Mat& camera_matrix,
    bool include_parallel_pnp = false)
{
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < armors.size(); ++i) {
        if (i > 0) out << ',';
        out << armorJson(armors[i], camera_matrix, include_parallel_pnp);
    }
    out << ']';
    return out.str();
}

std::string armorValueJson(const rm::Armor& armor, const cv::Mat& camera_matrix)
{
    return armorJson(std::make_shared<rm::Armor>(armor), camera_matrix);
}

std::string armorValuesJson(
    const std::vector<rm::Armor>& armors, const cv::Mat& camera_matrix)
{
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < armors.size(); ++i) {
        if (i > 0) out << ',';
        out << armorValueJson(armors[i], camera_matrix);
    }
    out << ']';
    return out.str();
}

std::string predictedArmorSlotsJson(const std::vector<Eigen::Vector4d>& slots)
{
    std::ostringstream out;
    out << std::setprecision(10) << '[';
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (i > 0) out << ',';
        out << "{\"id\":" << i
            << ",\"x\":" << jsonNumber(slots[i](0))
            << ",\"y\":" << jsonNumber(slots[i](1))
            << ",\"z\":" << jsonNumber(slots[i](2))
            << ",\"yaw\":" << jsonNumber(slots[i](3))
            << ",\"yaw_deg\":" << jsonNumber(slots[i](3) * rm::R2D)
            << '}';
    }
    out << ']';
    return out.str();
}

void normalizeOutpostDetections(std::vector<rm::ArmorForDetect>& armors, TargetMode mode)
{
    if (mode != TargetMode::Outpost) return;
    for (auto& armor : armors) {
        armor.number = 6;
        armor.classfication_result = "6";
    }
}

rm::Frame makeVivsionnFrame(const SimFrame& input)
{
    rm::Frame frame;
    frame.srcImg = input.bgr_image;
    frame.source_producer_epoch = input.source_producer_epoch;
    frame.source_image_seq = input.source_image_seq;
    frame.source_capture_timestamp_ns = input.source_capture_timestamp_ns;
    frame.poseEuler.roll = 0.0f;
    frame.poseEuler.yaw = static_cast<float>(input.gimbal_yaw_deg);
    frame.poseEuler.pitch = static_cast<float>(input.gimbal_pitch_deg);
    frame.bullet_speed = input.bullet_speed_mps;
    frame.timeStamp = input.timestamp_ms;
    frame.usb_timeStamp = input.timestamp_ms;
    frame.simulator_state_age_s = input.simulator_state_age_s;
    frame.startTime = std::chrono::high_resolution_clock::now() -
        std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
            std::chrono::duration<double, std::milli>(input.timestamp_ms));

    frame.fb.task_mode = taskModeFor(input.target_mode);
    frame.fb.self_team = rm::FeedBackData::SELF_TEAM::SELF_BLUE;
    frame.fb.heat = 0;
    frame.fb.heat_cap = 600;
    frame.fb.bullet_speed = static_cast<float>(input.bullet_speed_mps);
    frame.fb.gimbal_roll = frame.poseEuler.roll;
    frame.fb.gimbal_yaw = frame.poseEuler.yaw;
    frame.fb.gimbal_pitch = frame.poseEuler.pitch;
    frame.fb.yaw_speed = static_cast<float>(input.gimbal_yaw_speed_deg_s);
    frame.fb.__reserved[0] = 1;
    frame.fb.set_task_mode_telemetry(frame.fb.task_mode, frame.fb.task_mode);
    return frame;
}

AimCommand fromControlData(
    const rm::ControlData& control, double distance_m, const std::string& backend)
{
    AimCommand command;
    command.backend = backend;
    command.has_target =
        control.aiming_state == rm::ControlData::AIMING_STATE::TARGET_DETECTED;
    command.yaw_deg = control.gimbal_yaw;
    command.pitch_deg = control.gimbal_pitch;
    command.distance_m =
        command.has_target && isPositiveFinite(distance_m) ? distance_m : -1.0;
    command.raw_shot_mode = control.shot_mode;
    command.fire_advice =
        control.shot_mode == rm::ControlData::SHOT_MODE::AUTO_FIRE ||
        control.shot_mode == rm::ControlData::SHOT_MODE::SHOT_ONCE;
    return command;
}

std::string aimCommandJson(const AimCommand& command)
{
    std::ostringstream out;
    out << std::setprecision(10) << '{';
    bool first = true;
    debug::appendBool(out, "has_target", command.has_target, first);
    debug::appendNumber(out, "yaw_deg", command.yaw_deg, first);
    debug::appendNumber(out, "pitch_deg", command.pitch_deg, first);
    debug::appendNumber(out, "distance_m", command.distance_m, first);
    debug::appendBool(out, "fire_advice", command.fire_advice, first);
    debug::appendInt(out, "raw_shot_mode", command.raw_shot_mode, first);
    debug::appendString(out, "selected_camera", command.selected_camera, first);
    debug::appendNumber(out, "selected_focal_mm", command.selected_focal_mm, first);
    debug::appendString(out, "backend", command.backend, first);
    debug::appendNumber(out, "runtime_yolo_ms", command.runtime_yolo_ms, first);
    debug::appendNumber(out, "runtime_solve_ms", command.runtime_solve_ms, first);
    debug::appendNumber(out, "runtime_track_aim_ms", command.runtime_track_aim_ms, first);
    debug::appendNumber(
        out, "runtime_pipeline_delay_ms", command.runtime_pipeline_delay_ms, first);
    out << '}';
    return out.str();
}

struct CommandPipelineDebug
{
    std::string policy = "fire_control";
    std::string source = "fire_control";
    AimCommand fire_control_command;
    AimCommand pre_smoothing_command;
    double image_yaw_command_deg = std::numeric_limits<double>::quiet_NaN();
    double ballistic_pitch_command_deg = std::numeric_limits<double>::quiet_NaN();
    double fallback_yaw_command_deg = std::numeric_limits<double>::quiet_NaN();
    double fallback_pitch_command_deg = std::numeric_limits<double>::quiet_NaN();
    double detected_distance_m = std::numeric_limits<double>::quiet_NaN();
    double fallback_distance_m = std::numeric_limits<double>::quiet_NaN();
    bool reference_armor_available = false;
    bool can_fallback_lock = false;
    bool image_yaw_override_used = false;
    bool ballistic_pitch_override_used = false;
};

class VivsionnAimPipeline final : public IAimPipeline
{
public:
    explicit VivsionnAimPipeline(AimBridgeConfig config)
        : config_(std::move(config)),
          armor_detector_(std::make_unique<rm::MultiThreadDetectorTRT>(
              config_.armor_detector_config, false))
    {
        Params params;
        armor_detector_->setAttackAllColors(params.DEBUG_ATTACK_ALL_ARMOR_COLORS);
        std::cerr << "[aim_sim_bridge] attack_all_armor_colors="
                  << (params.DEBUG_ATTACK_ALL_ARMOR_COLORS ? "on" : "off")
                  << std::endl;
        active_target_number_ = optionalPositiveEnvInt("AIM_SIM_ACTIVE_TARGET_NUMBER");
        if (active_target_number_) {
            std::cerr << "[aim_sim_bridge] active_target_number_filter="
                      << *active_target_number_ << std::endl;
        }
        const std::string command_policy = envString("AIM_SIM_COMMAND_POLICY", "fire_control");
        fire_control_direct_command_ =
            command_policy == "fire_control" || command_policy == "fire-control" ||
            command_policy == "planner" || command_policy == "plan";
        std::cerr << "[aim_sim_bridge] command_policy="
                  << (fire_control_direct_command_ ? "fire_control" : "sim_image_override")
                  << std::endl;
        completion_worker_ = std::thread(&VivsionnAimPipeline::completionLoop, this);
    }

    ~VivsionnAimPipeline() override
    {
        armor_detector_->stop();
        if (completion_worker_.joinable()) completion_worker_.join();
    }

    AimCommand process(const SimFrame& input) override
    {
        if (!input.bgr_image.empty()) {
            const auto submission_begin = std::chrono::steady_clock::now();
            SimFrame profile_frame = makeProfileFrame(input);
            rm::Frame frame = makeVivsionnFrame(profile_frame);
            {
                std::lock_guard<std::mutex> lock(submitted_mutex_);
                submitted_profiles_.push_back({profile_frame, input.target_mode});
            }
            const bool overwritten = armor_detector_->push(frame, submission_begin);
            stage_telemetry_.recordSubmission(
                elapsedNs(submission_begin, std::chrono::steady_clock::now()), overwritten);
            if (overwritten) {
                std::lock_guard<std::mutex> lock(submitted_mutex_);
                detail::eraseOldestOverwrittenRawSuffixItem(
                    submitted_profiles_, rm::K_PIPELINE_DEPTH);
            }
        }

        AimCommand completed;
        if (completion_queue_.tryTake(completed)) {
            stage_telemetry_.recordDeliveredCompletion();
            return completed;
        }
        completed.backend = backendName();
        return completed;
    }

    std::string backendName() const override
    {
        return "vivsionn_trt";
    }

    AimPipelineCounters counters() const override
    {
        AimPipelineCounters out = stage_telemetry_.snapshot();
        const rm::DetectorTimingSnapshot detector = armor_detector_->timingSnapshot();
        out.detector_active_slots = detector.active_slots;
        out.detector_profile_enabled = detector.enabled;
        out.detector_profile_timing_event_count = detector.timing_event_count;
        out.detector_profile_sample_stride = detector.sample_stride;
        out.detector_profile_completed = detector.completed;
        out.detector_profile_errors = detector.errors;
        out.detector_raw_queue_wait_ns = detector.raw_queue_wait_ns;
        out.detector_slot_wait_ns = detector.slot_wait_ns;
        out.detector_launcher_host_ns = detector.launcher_host_ns;
        out.detector_pending_order_wait_ns = detector.pending_order_wait_ns;
        out.detector_event_wait_ns = detector.event_wait_ns;
        out.detector_fp_convert_ns = detector.fp_convert_ns;
        out.detector_postprocess_nms_ns = detector.postprocess_nms_ns;
        out.detector_completion_bookkeeping_ns = detector.completion_bookkeeping_ns;
        out.detector_profile_wall_ns = detector.wall_ns;
        out.detector_gpu_h2d_ns = detector.gpu_h2d_ns;
        out.detector_gpu_preprocess_ns = detector.gpu_preprocess_ns;
        out.detector_gpu_trt_ns = detector.gpu_trt_ns;
        out.detector_gpu_d2h_ns = detector.gpu_d2h_ns;
        out.detector_gpu_stream_ns = detector.gpu_stream_ns;
        return out;
    }

private:
    struct SubmittedProfile
    {
        SimFrame frame;
        TargetMode mode = TargetMode::Armor;
    };

    void completionLoop()
    {
        while (armor_detector_->running()) {
            auto result = armor_detector_->pop();
            rm::Frame& detected = std::get<2>(result);
            if (detected.srcImg.empty() && !armor_detector_->running()) break;

            const auto detector_done = std::chrono::steady_clock::now();
            stage_telemetry_.recordDetectorCompletion(
                elapsedNs(std::get<1>(result), detector_done));

            SubmittedProfile submitted;
            bool found = false;
            {
                std::lock_guard<std::mutex> lock(submitted_mutex_);
                found = detail::takeFirstMatchingAndErasePrefix(
                    submitted_profiles_,
                    [&detected](const SubmittedProfile& item) {
                        return item.frame.source_producer_epoch == detected.source_producer_epoch &&
                               item.frame.source_image_seq == detected.source_image_seq &&
                               item.frame.source_capture_timestamp_ns ==
                                   detected.source_capture_timestamp_ns;
                    },
                    submitted);
            }
            if (!found) continue;

            AimCommand command = processArmorOrOutpost(
                std::move(result), submitted.frame, submitted.mode);
            completion_queue_.publish(std::move(command));
        }
    }

    SimFrame makeProfileFrame(const SimFrame& input)
    {
        SimFrame out = input;
        out.source_image_width = input.bgr_image.cols;
        out.source_image_height = input.bgr_image.rows;
        out.source_roi_px = cv::Rect(0, 0, input.bgr_image.cols, input.bgr_image.rows);
        out.virtual_scale_x = 1.0;
        out.virtual_scale_y = 1.0;
        out.camera_profile_id = focalProfileId("wide", config_.wide_focal_mm);
        out.camera_focal_mm = config_.wide_focal_mm;

        cv::Mat base_k = cameraMatrixForFrame(input, angle_solver_);
        if (!base_k.empty()) {
            base_k.convertTo(out.camera_matrix_override, CV_64F);
            out.has_camera_matrix_override = true;
        }

        if (!config_.dual_focal_enabled || !precision_mode_active_ ||
            config_.wide_focal_mm <= 0.0 ||
            config_.precision_focal_mm <= config_.wide_focal_mm || input.bgr_image.empty()) {
            return out;
        }

        const double scale = config_.precision_focal_mm / config_.wide_focal_mm;
        const cv::Rect roi =
            precisionSourceRoi(input.bgr_image.cols, input.bgr_image.rows, base_k, scale);
        if (roi.width <= 0 || roi.height <= 0 || roi.width > input.bgr_image.cols ||
            roi.height > input.bgr_image.rows) {
            precision_mode_active_ = false;
            return out;
        }

        cv::Mat cropped = input.bgr_image(roi);
        cv::resize(cropped, out.bgr_image, input.bgr_image.size(), 0.0, 0.0, cv::INTER_LINEAR);
        out.source_roi_px = roi;
        out.virtual_scale_x = static_cast<double>(input.bgr_image.cols) / roi.width;
        out.virtual_scale_y = static_cast<double>(input.bgr_image.rows) / roi.height;
        out.camera_profile_id = focalProfileId("precision", config_.precision_focal_mm);
        out.camera_focal_mm = config_.precision_focal_mm;
        if (!base_k.empty()) {
            out.camera_matrix_override =
                cropScaledCameraMatrix(base_k, roi, out.virtual_scale_x, out.virtual_scale_y);
            out.has_camera_matrix_override = true;
        }
        return out;
    }

    DualFocalDecision updateDualFocalState(
        const SimFrame& frame,
        const std::vector<std::shared_ptr<rm::Armor>>& solved,
        double distance_m,
        bool current_target_frame)
    {
        DualFocalDecision decision =
            makeWideDecision(config_, config_.dual_focal_enabled ? "wide_acquisition" : "disabled");
        decision.selected = frame.camera_profile_id;
        decision.selected_focal_mm = frame.camera_focal_mm;
        const std::string wide_profile = focalProfileId("wide", config_.wide_focal_mm);
        const std::string precision_profile =
            focalProfileId("precision", config_.precision_focal_mm);
        decision.next_selected = precision_mode_active_ ? precision_profile : wide_profile;

        if (!config_.dual_focal_enabled || config_.precision_focal_mm <= config_.wide_focal_mm) {
            precision_mode_active_ = false;
            decision.enabled = config_.dual_focal_enabled;
            decision.reason = config_.dual_focal_enabled ? "invalid_focal_ratio" : "disabled";
            decision.next_selected = wide_profile;
            decision.switch_pending = decision.selected != decision.next_selected;
            return decision;
        }

        const double scale = config_.precision_focal_mm / config_.wide_focal_mm;
        decision.enabled = true;
        decision.wide_focal_mm = config_.wide_focal_mm;
        decision.precision_focal_mm = config_.precision_focal_mm;
        decision.scale = scale;
        decision.distance_m = distance_m;
        decision.expected_error_ratio_vs_wide = 1.0 / scale;
        decision.precision_half_width_px_in_wide =
            frame.source_image_width > 0 ? frame.source_image_width / (2.0 * scale)
                                         : std::numeric_limits<double>::quiet_NaN();
        decision.precision_half_height_px_in_wide =
            frame.source_image_height > 0 ? frame.source_image_height / (2.0 * scale)
                                          : std::numeric_limits<double>::quiet_NaN();

        bool next_precision = precision_mode_active_;
        if (!current_target_frame || solved.empty() || !solved.front()) {
            next_precision = false;
            decision.reason = precision_mode_active_ ? "leave_precision_target_lost"
                                                     : "wide_no_current_target";
        } else {
            const auto source_vertices =
                sourceVerticesFromActiveFrame(frame, solved.front()->vertex);
            const double source_width =
                source_vertices.size() == 4
                    ? (cv::norm(source_vertices[2] - source_vertices[1]) +
                       cv::norm(source_vertices[3] - source_vertices[0])) *
                          0.5
                    : std::numeric_limits<double>::quiet_NaN();
            decision.wide_armor_width_px = source_width;
            decision.precision_equivalent_armor_width_px =
                std::isfinite(source_width) ? source_width * scale
                                            : std::numeric_limits<double>::quiet_NaN();

            const cv::Mat base_k = cameraMatrixForFrame(frame, angle_solver_);
            const cv::Rect roi = precisionSourceRoi(
                frame.source_image_width, frame.source_image_height, base_k, scale);
            double min_margin = std::numeric_limits<double>::infinity();
            for (const auto& point : source_vertices) {
                min_margin = std::min(min_margin, static_cast<double>(point.x - roi.x));
                min_margin = std::min(
                    min_margin, static_cast<double>(roi.x + roi.width - point.x));
                min_margin = std::min(min_margin, static_cast<double>(point.y - roi.y));
                min_margin = std::min(
                    min_margin, static_cast<double>(roi.y + roi.height - point.y));
            }
            if (!std::isfinite(min_margin)) {
                min_margin = std::numeric_limits<double>::quiet_NaN();
            }
            decision.precision_min_margin_px = min_margin;
            decision.target_inside_precision_fov =
                std::isfinite(min_margin) && min_margin >= config_.precision_fov_margin_px;

            if (!isPositiveFinite(distance_m)) {
                next_precision = false;
                decision.reason = "wide_invalid_distance";
            } else if (precision_mode_active_) {
                if (distance_m <= config_.precision_leave_distance_m) {
                    next_precision = false;
                    decision.reason = "leave_precision_close_range";
                } else if (!decision.target_inside_precision_fov) {
                    next_precision = false;
                    decision.reason = "leave_precision_crop_margin";
                } else {
                    next_precision = true;
                    decision.reason = "hold_precision";
                }
            } else {
                if (distance_m < config_.precision_enter_distance_m) {
                    next_precision = false;
                    decision.reason = "wide_close_range";
                } else if (!decision.target_inside_precision_fov) {
                    next_precision = false;
                    decision.reason = "wide_target_not_inside_precision_crop";
                } else {
                    next_precision = true;
                    decision.reason = "enter_precision_distance_and_margin";
                }
            }
        }

        const bool previous_precision = precision_mode_active_;
        precision_mode_active_ = next_precision;
        decision.next_selected = precision_mode_active_ ? precision_profile : wide_profile;
        decision.switch_pending = previous_precision != precision_mode_active_;
        return decision;
    }

    AimCommand processArmorOrOutpost(
        std::tuple<std::vector<rm::ArmorForDetect>,
                   std::chrono::steady_clock::time_point, rm::Frame> result,
        const SimFrame& profile_frame, TargetMode mode)
    {
        std::vector<rm::ArmorForDetect> armors = std::move(std::get<0>(result));
        rm::Frame detected_frame = std::move(std::get<2>(result));
        normalizeOutpostDetections(armors, mode);
        for (std::size_t index = 0; index < armors.size(); ++index) {
            armors[index].observation_id = static_cast<std::uint32_t>(index);
        }
        const std::vector<rm::ArmorForDetect> detector_candidates = armors;
        const std::size_t raw_detected_count = armors.size();
        int active_target_number = -1;
        std::size_t active_target_rejected_count = 0;
        if (mode == TargetMode::Armor && active_target_number_) {
            active_target_number = *active_target_number_;
            const std::size_t before_filter = armors.size();
            armors.erase(
                std::remove_if(
                    armors.begin(), armors.end(),
                    [active_target_number](const rm::ArmorForDetect& armor) {
                        return armor.number != active_target_number;
                    }),
                armors.end());
            active_target_rejected_count = before_filter - armors.size();
        }
        const std::size_t detected_count = armors.size();
        int first_detected_number = -1;
        int first_detected_color = -1;
        float first_detected_confidence = 0.0f;
        cv::Point2f first_detected_center;
        std::size_t first_detected_vertex_count = 0;
        std::vector<cv::Point2f> first_detected_vertices;
        rm::ArmorType first_detected_type = rm::ArmorType::INVALID;
        if (!armors.empty()) {
            const auto& armor = armors.front();
            first_detected_number = armor.number;
            first_detected_color = armor.color;
            first_detected_confidence = armor.confidence;
            first_detected_center = armor.center;
            first_detected_vertex_count = armor.vertex.size();
            first_detected_vertices = armor.vertex;
            first_detected_type = armor.type;
        }

        rm::FrameMeta frame_meta(detected_frame);
        angle_solver_.loadMeta(frame_meta, detected_frame.srcImg);
        if (profile_frame.has_camera_matrix_override) {
            angle_solver_.setCameraIntrinsicsOverride(profile_frame.camera_matrix_override);
        }
        const auto solve_begin = std::chrono::steady_clock::now();
        std::vector<std::shared_ptr<rm::Armor>> solved =
            angle_solver_.solveArmors(std::move(armors));
        stage_telemetry_.recordSolveCompletion(
            elapsedNs(solve_begin, std::chrono::steady_clock::now()));

        const auto tracker_aim_begin = std::chrono::steady_clock::now();
        if (!last_camera_profile_id_.empty() &&
            last_camera_profile_id_ != profile_frame.camera_profile_id) {
            estimator_.resetForTaskModeSwitch();
            fire_control_.resetExecutionState();
            last_drive_command_ = AimCommand{};
            last_drive_command_hold_frames_ = kMaxDriveCommandHoldFrames;
            last_smoothed_command_.reset();
        }
        last_camera_profile_id_ = profile_frame.camera_profile_id;

        if (last_task_mode_ != 0 && last_task_mode_ != frame_meta.fb.task_mode) {
            estimator_.resetForTaskModeSwitch();
            fire_control_.resetExecutionState();
            last_smoothed_command_.reset();
        }
        last_task_mode_ = frame_meta.fb.task_mode;

        estimator_.latency = 0.0;
        estimator_.loadMeta(frame_meta);
        estimator_.trackerUpdate(solved, angle_solver_);
        if (active_target_number > 0 && estimator_._trackedArmor &&
            estimator_._trackedArmor->number != active_target_number) {
            estimator_.resetForTaskModeSwitch();
            fire_control_.resetExecutionState();
            last_drive_command_ = AimCommand{};
            last_drive_command_hold_frames_ = kMaxDriveCommandHoldFrames;
            last_smoothed_command_.reset();
        }

        fire_control_.loadMeta(frame_meta);
        rm::ControlData control = fire_control_.calControlData(angle_solver_, estimator_);
        const double command_distance_m = commandDistanceMeters(estimator_, solved);
        if (isPositiveFinite(command_distance_m)) {
            last_valid_target_distance_m_ = command_distance_m;
        }
        AimCommand command = fromControlData(control, command_distance_m, backendName());
        command.backend = backendName();
        CommandPipelineDebug command_pipeline;
        command_pipeline.policy =
            fire_control_direct_command_ ? "fire_control" : "sim_image_override";
        command_pipeline.fire_control_command = command;
        const bool current_target_frame =
            !solved.empty() && estimator_._detectedFlag &&
            !estimator_._current_obs_armors.empty() &&
            !estimator_._current_tracker_input_armors.empty() &&
            estimator_._current_primary_observation_index >= 0;
        const auto reference_armor = selectedReferenceArmor(estimator_, solved);
        const double image_yaw_command_deg = imageYawCorrectedCommandDeg(
            frame_meta.poseEuler.yaw, reference_armor, angle_solver_._cam_instant_matrix);
        const double ballistic_pitch_command_deg = ballisticImagePitchCommandDeg(
            frame_meta.poseEuler.pitch, reference_armor, angle_solver_._cam_instant_matrix,
            command_distance_m, frame_meta.fb.bullet_speed);
        command_pipeline.reference_armor_available = reference_armor != nullptr;
        command_pipeline.image_yaw_command_deg = image_yaw_command_deg;
        command_pipeline.ballistic_pitch_command_deg = ballistic_pitch_command_deg;
        const bool has_detection_center =
            detected_count > 0 && finiteImagePoint(first_detected_center);
        const cv::Point2f lock_fallback_center =
            reference_armor ? reference_armor->center : first_detected_center;
        const bool can_fallback_lock =
            (reference_armor != nullptr || has_detection_center) &&
            finiteImagePoint(lock_fallback_center);
        const double detected_distance_m = estimateDetectedDistanceMeters(
            first_detected_vertices, first_detected_number, first_detected_type,
            angle_solver_._cam_instant_matrix);
        command_pipeline.can_fallback_lock = can_fallback_lock;
        command_pipeline.detected_distance_m = detected_distance_m;
        if (!isPositiveFinite(command_distance_m) && isPositiveFinite(detected_distance_m)) {
            last_valid_target_distance_m_ = detected_distance_m;
        }
        double fallback_distance_m = command_distance_m;
        if (!isPositiveFinite(fallback_distance_m)) {
            fallback_distance_m = detected_distance_m;
        }
        if (!isPositiveFinite(fallback_distance_m)) {
            fallback_distance_m = last_valid_target_distance_m_;
        }
        if (!isPositiveFinite(fallback_distance_m)) {
            fallback_distance_m = 3.0;
        }
        command_pipeline.fallback_distance_m = fallback_distance_m;
        const double fallback_yaw_command_deg = imageYawCorrectedCommandDegForPoint(
            frame_meta.poseEuler.yaw, lock_fallback_center, angle_solver_._cam_instant_matrix);
        const double fallback_pitch_command_deg = ballisticImagePitchCommandDegForPoint(
            frame_meta.poseEuler.pitch, lock_fallback_center, angle_solver_._cam_instant_matrix,
            fallback_distance_m, frame_meta.fb.bullet_speed);
        command_pipeline.fallback_yaw_command_deg = fallback_yaw_command_deg;
        command_pipeline.fallback_pitch_command_deg = fallback_pitch_command_deg;
        const DualFocalDecision dual_focal_decision =
            updateDualFocalState(profile_frame, solved, command_distance_m, current_target_frame);
        if (fire_control_direct_command_) {
            command_pipeline.source =
                command.has_target ? "fire_control_direct" : "fire_control_no_target";
            if (command.has_target) {
                last_drive_command_ = command;
                last_drive_command_hold_frames_ = 0;
            } else {
                command.distance_m = -1.0;
                command.fire_advice = false;
            }
        } else {
            if (command.has_target && reference_armor) {
                command_pipeline.source = "image_ballistic_override";
                if (std::isfinite(image_yaw_command_deg)) {
                    command.yaw_deg = image_yaw_command_deg;
                    command_pipeline.image_yaw_override_used = true;
                }
                if (std::isfinite(ballistic_pitch_command_deg)) {
                    command.pitch_deg = ballistic_pitch_command_deg;
                    command_pipeline.ballistic_pitch_override_used = true;
                }
            }
            if (command.has_target) {
                last_drive_command_ = command;
                last_drive_command_hold_frames_ = 0;
            } else if (can_fallback_lock) {
                command_pipeline.source = "fallback_lock";
                command.has_target = true;
                command.distance_m = fallback_distance_m;
                command.fire_advice = false;
                if (std::isfinite(fallback_yaw_command_deg)) {
                    command.yaw_deg = fallback_yaw_command_deg;
                } else {
                    command.yaw_deg = frame_meta.poseEuler.yaw;
                }
                if (std::isfinite(fallback_pitch_command_deg)) {
                    command.pitch_deg = fallback_pitch_command_deg;
                } else {
                    command.pitch_deg = frame_meta.poseEuler.pitch;
                }
                last_drive_command_ = command;
                last_drive_command_hold_frames_ = 0;
            } else if (
                detected_count > 0 &&
                last_drive_command_.has_target &&
                last_drive_command_hold_frames_ < kMaxDriveCommandHoldFrames) {
                command_pipeline.source = "held_last";
                command = last_drive_command_;
                command.fire_advice = false;
                ++last_drive_command_hold_frames_;
            } else {
                command_pipeline.source = "no_target";
                command.has_target = false;
                command.distance_m = -1.0;
                command.fire_advice = false;
            }
        }
        command.selected_camera = dual_focal_decision.selected;
        command.selected_focal_mm = dual_focal_decision.selected_focal_mm;
        if (!fire_control_direct_command_ && command.has_target && reference_armor) {
            command.fire_advice =
                imageYawAlignedForSimulatorFire(reference_armor, angle_solver_._cam_instant_matrix);
        }
        command_pipeline.pre_smoothing_command = command;
        applyCommandSmoothing(command);
        stage_telemetry_.recordTrackerAimCompletion(
            elapsedNs(tracker_aim_begin, std::chrono::steady_clock::now()));
        const auto finalize_begin = std::chrono::steady_clock::now();
        command.completed_vision_result = true;
        command.source_producer_epoch = detected_frame.source_producer_epoch;
        command.source_image_seq = detected_frame.source_image_seq;
        command.source_capture_timestamp_ns = detected_frame.source_capture_timestamp_ns;
        command.vision_completion_timestamp_ns = systemNowNs();
        writePipelineTelemetry(
            frame_meta, mode, detected_frame, profile_frame, dual_focal_decision,
            raw_detected_count, detected_count, active_target_number,
            active_target_rejected_count, first_detected_number,
            first_detected_color, first_detected_confidence, first_detected_center,
            first_detected_vertices, detector_candidates, solved, control, command, command_distance_m,
            command_pipeline);
        reportRecognitionDiagnostics(
            detected_count, solved, control, first_detected_number, first_detected_color,
            first_detected_confidence, first_detected_center, first_detected_vertex_count,
            command_distance_m);
        const auto final_done = std::chrono::steady_clock::now();
        stage_telemetry_.recordFinalCompletion(
            elapsedNs(finalize_begin, final_done),
            elapsedNs(std::get<1>(result), final_done));
        return command;
    }

    void applyCommandSmoothing(AimCommand& command)
    {
        if (!config_.command_smoothing_enabled || !command.has_target ||
            !std::isfinite(command.yaw_deg) || !std::isfinite(command.pitch_deg)) {
            last_smoothed_command_.reset();
            return;
        }

        if (!last_smoothed_command_.has_value() ||
            last_smoothed_command_->selected_camera != command.selected_camera) {
            last_smoothed_command_ = command;
            return;
        }

        command.yaw_deg = smoothScalarCommand(
            last_smoothed_command_->yaw_deg, command.yaw_deg, config_, true);
        command.pitch_deg = smoothScalarCommand(
            last_smoothed_command_->pitch_deg, command.pitch_deg, config_, false);
        last_smoothed_command_ = command;
    }

    void writePipelineTelemetry(
        const rm::FrameMeta& frame_meta,
        TargetMode mode,
        const rm::Frame& frame,
        const SimFrame& profile_frame,
        const DualFocalDecision& dual_focal_decision,
        std::size_t raw_detected_count,
        std::size_t detected_count,
        int active_target_number,
        std::size_t active_target_rejected_count,
        int first_detected_number,
        int first_detected_color,
        float first_detected_confidence,
        const cv::Point2f& first_detected_center,
        const std::vector<cv::Point2f>& first_detected_vertices,
        const std::vector<rm::ArmorForDetect>& detector_candidates,
        const std::vector<std::shared_ptr<rm::Armor>>& solved,
        const rm::ControlData& control,
        const AimCommand& command,
        double command_distance_m,
        const CommandPipelineDebug& command_pipeline)
    {
        const std::string path = debug::envPath("AIM_SIM_DEBUG_PIPELINE_JSON");
        const std::string jsonl_path = debug::envPath("AIM_SIM_DEBUG_PIPELINE_JSONL");
        const bool write_every_frame =
            !jsonl_path.empty() || envFlag("AIM_SIM_DEBUG_PIPELINE_EVERY_FRAME");
        if (path.empty() && jsonl_path.empty()) return;

        const auto now = std::chrono::steady_clock::now();
        if (!write_every_frame && last_debug_write_.time_since_epoch().count() != 0 &&
            now - last_debug_write_ < std::chrono::milliseconds(100)) {
            return;
        }
        last_debug_write_ = now;

        std::ostringstream detector_json;
        detector_json << std::setprecision(10) << '{';
        bool detector_first = true;
        debug::appendUInt(detector_json, "raw_count", raw_detected_count, detector_first);
        debug::appendUInt(detector_json, "count", detected_count, detector_first);
        debug::appendRaw(
            detector_json, "candidates",
            detectorCandidatesJson(detector_candidates, solved, active_target_number),
            detector_first);
        debug::appendInt(
            detector_json, "active_target_number", active_target_number, detector_first);
        debug::appendUInt(
            detector_json, "active_target_rejected_count", active_target_rejected_count,
            detector_first);
        if (detected_count > 0) {
            std::ostringstream first_json;
            first_json << std::setprecision(10) << '{';
            bool first = true;
            debug::appendInt(first_json, "number", first_detected_number, first);
            debug::appendInt(first_json, "color", first_detected_color, first);
            debug::appendNumber(first_json, "confidence", first_detected_confidence, first);
            debug::appendRaw(first_json, "center_px", point2Json(first_detected_center), first);
            debug::appendRaw(
                first_json, "detector_vertices_px", points2Json(first_detected_vertices), first);
            debug::appendString(
                first_json, "detector_vertex_order_note",
                "raw detector ArmorForDetect.vertex before AngleSolver", first);
            first_json << '}';
            debug::appendRaw(detector_json, "first", first_json.str(), detector_first);
        }
        detector_json << '}';

        std::ostringstream tracker_json;
        tracker_json << '{';
        bool tracker_first = true;
        const bool ypd_tracker_ready =
            estimator_.ypd_angle_tracker_ && estimator_.ypd_angle_tracker_->isInitialized();
        const std::vector<Eigen::Vector4d> predicted_armor_slots =
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->getPredictedArmorStates()
                              : std::vector<Eigen::Vector4d>{};
        debug::appendBool(tracker_json, "detected", estimator_._detectedFlag, tracker_first);
        debug::appendString(
            tracker_json, "tracker_state",
            estimator_.trackerStateStr[estimator_.tracker_state], tracker_first);
        debug::appendString(
            tracker_json, "update_state",
            estimator_.UpdateStateStr[estimator_.update_state], tracker_first);
        debug::appendUInt(
            tracker_json, "observation_count", estimator_._current_obs_armors.size(),
            tracker_first);
        debug::appendUInt(
            tracker_json, "tracker_input_count",
            estimator_._current_tracker_input_armors.size(), tracker_first);
        debug::appendInt(
            tracker_json, "primary_observation_index",
            estimator_._current_primary_observation_index, tracker_first);
        debug::appendBool(tracker_json, "ypd_initialized", ypd_tracker_ready, tracker_first);
        debug::appendInt(
            tracker_json, "tracked_id",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->getTrackedId() : -1,
            tracker_first);
        debug::appendInt(
            tracker_json, "tracked_armors_num",
            static_cast<int>(estimator_.tracked_armors_num), tracker_first);
        debug::appendInt(tracker_json, "jump_flag", estimator_.jump_flag, tracker_first);
        debug::appendBool(
            tracker_json, "reset_this_frame", estimator_.ypd_reset_this_frame,
            tracker_first);
        debug::appendString(
            tracker_json, "reset_reason", estimator_.ypd_reset_reason,
            tracker_first);
        debug::appendBool(
            tracker_json, "diverged", estimator_.ypd_reset_diverged,
            tracker_first);
        debug::appendBool(
            tracker_json, "bad_convergence", estimator_.ypd_reset_bad_convergence,
            tracker_first);
        debug::appendNumber(
            tracker_json, "radius_primary", estimator_.ypd_debug_primary_radius,
            tracker_first);
        debug::appendNumber(
            tracker_json, "radius_secondary", estimator_.ypd_debug_secondary_radius,
            tracker_first);
        debug::appendNumber(
            tracker_json, "radius_delta", estimator_.ypd_debug_delta_radius,
            tracker_first);
        debug::appendNumber(
            tracker_json, "height_delta", estimator_.ypd_debug_height_delta,
            tracker_first);
        debug::appendInt(
            tracker_json, "ypd_update_count", estimator_.ypd_debug_update_count,
            tracker_first);
        debug::appendInt(
            tracker_json, "actual_update_count",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->updateCount() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "recent_nis_failures",
            estimator_.ypd_debug_recent_nis_failures, tracker_first);
        debug::appendInt(
            tracker_json, "nis_window", estimator_.ypd_debug_nis_window,
            tracker_first);
        debug::appendInt(
            tracker_json, "physical_rejection_count",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastPhysicalRejectionCount() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "physical_rejection_reason",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastPhysicalRejectionReason() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_pair_required",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalPairRequired() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_pair_found",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalPairFound() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_update_class",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalUpdateClass() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_accepted_count",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalAcceptedCount() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_pair_accepted_count",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalPairAcceptedCount() : 0,
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_pair_score",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalPairScore()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_pair_center_gap_m",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalPairCenterGap()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_pair_center_jump_m",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalPairCenterJump()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_pair_center_x_m",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalPairCenterX()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_pair_center_y_m",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalPairCenterY()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_single_center_count",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalSingleCenterCount()
                              : 0,
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_single_center_x_m",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalSingleCenterX()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_single_center_y_m",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalSingleCenterY()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_history_size",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocityHistorySize() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_observation_source",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityObservationSource()
                : 0,
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_sample_t_s",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocitySampleTime()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_sample_x_m",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocitySampleX()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_sample_y_m",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocitySampleY()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_sample_frame_yaw_rad",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocitySampleFrameYaw()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_sample_group_id",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocitySampleGroupId()
                : -1,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_fit_sample_count",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitSampleCount() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_fit_accepted",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitAccepted() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_fit_reject_reason",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitRejectReason() : 0,
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_time_span_s",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitTimeSpan()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_net_displacement_m",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitNetDisplacement()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_rms_m",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitRms()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_raw_speed_mps",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitRawSpeed()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_raw_vx_mps",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitRawVx()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_raw_vy_mps",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitRawVy()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_frame_yaw_rate_rad_s",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitFrameYawRate()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_frame_yaw_mean_rad",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitFrameYawMean()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_frame_yaw_span_rad",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitFrameYawSpan()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_fit_pair_sample_count",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitPairSampleCount()
                : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_fit_single_sample_count",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitSingleSampleCount()
                : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_fit_group_count",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitGroupCount()
                : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_fit_grouped_used",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitGroupedUsed()
                : 0,
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_grouped_speed_mps",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitGroupedSpeed()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_grouped_rms_m",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitGroupedRms()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_velocity_fit_rot_comp_used",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitRotCompUsed()
                : 0,
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_rot_comp_speed_mps",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitRotCompSpeed()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_frame_transform_pos_yaw_speed_mps",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitFrameTransformPosYawSpeed()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_frame_transform_pos_yaw_rms_m",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitFrameTransformPosYawRms()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_frame_transform_neg_yaw_speed_mps",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitFrameTransformNegYawSpeed()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_frame_transform_neg_yaw_rms_m",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitFrameTransformNegYawRms()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_applied_speed_mps",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitAppliedSpeed()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_applied_vx_mps",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitAppliedVx()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_velocity_fit_applied_vy_mps",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalVelocityFitAppliedVy()
                : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_yaw_rate_history_size",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawRateHistorySize() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_yaw_rate_observation_source",
            ypd_tracker_ready
                ? estimator_.ypd_angle_tracker_->lastNormalYawRateObservationSource()
                : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_yaw_observation_count",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawObservationCount()
                              : 0,
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_yaw_observation_t_s",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawObservationTime()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_yaw_observation_raw_rad",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawObservationRaw()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_yaw_observation_unwrapped_rad",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawObservationUnwrapped()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_yaw_rate_fit_sample_count",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawRateFitSampleCount()
                              : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_yaw_rate_fit_accepted",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawRateFitAccepted() : 0,
            tracker_first);
        debug::appendInt(
            tracker_json, "normal_yaw_rate_fit_reject_reason",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawRateFitRejectReason()
                              : 0,
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_yaw_rate_fit_time_span_s",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawRateFitTimeSpan()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_yaw_rate_fit_rms_rad",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawRateFitRms()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_yaw_rate_fit_raw_rad_s",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawRateFitRaw()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendNumber(
            tracker_json, "normal_yaw_rate_fit_applied_rad_s",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->lastNormalYawRateFitApplied()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendRaw(
            tracker_json, "match_ids", intVectorJson(estimator_.ypd_debug_last_match_ids),
            tracker_first);
        debug::appendRaw(
            tracker_json, "current_match_ids",
            intVectorJson(estimator_._current_obs_match_ids), tracker_first);
        debug::appendRaw(
            tracker_json, "observation_update_diagnostics",
            ypd_tracker_ready
                ? observationDiagnosticsJson(
                      estimator_.ypd_angle_tracker_->lastObservationDiagnostics())
                : "[]",
            tracker_first);
        debug::appendRaw(
            tracker_json, "state_covariance_11x11",
            ypd_tracker_ready
                ? eigenMatrixJson(estimator_.ypd_angle_tracker_->getCovariance())
                : "[]",
            tracker_first);
        debug::appendNumber(
            tracker_json, "velocity_history_reference_frame_yaw_rad",
            ypd_tracker_ready ? estimator_.ypd_angle_tracker_->currentFrameYaw()
                              : std::numeric_limits<double>::quiet_NaN(),
            tracker_first);
        debug::appendRaw(
            tracker_json, "normal_velocity_history_samples",
            ypd_tracker_ready
                ? velocityHistoryJson(
                      estimator_.ypd_angle_tracker_->normalCenterVelocityHistory(),
                      estimator_.ypd_angle_tracker_->currentFrameYaw())
                : "[]",
            tracker_first);
        debug::appendRaw(
            tracker_json, "tracker_input_armors",
            armorValuesJson(estimator_._current_tracker_input_armors,
                angle_solver_._cam_instant_matrix),
            tracker_first);
        debug::appendRaw(
            tracker_json, "predicted_armor_slots",
            predictedArmorSlotsJson(predicted_armor_slots), tracker_first);
        debug::appendNumber(tracker_json, "raw_distance_m", estimator_.distance_, tracker_first);
        debug::appendString(
            tracker_json, "movement", movementName(estimator_.movement), tracker_first);
        debug::appendInt(
            tracker_json, "movement_code", static_cast<int>(estimator_.movement), tracker_first);
        debug::appendNumber(tracker_json, "v_t", estimator_.v_t, tracker_first);
        debug::appendNumber(tracker_json, "v_r", estimator_.v_r, tracker_first);
        debug::appendNumber(tracker_json, "v_xy", estimator_.v_xy, tracker_first);
        debug::appendRaw(
            tracker_json, "last_velocity", eigen3Json(estimator_._last_velocity),
            tracker_first);
        debug::appendRaw(tracker_json, "target_accel", eigen3Json(estimator_.target_ac), tracker_first);
        debug::appendRaw(
            tracker_json, "measurement", eigenVectorJson(estimator_._measurement),
            tracker_first);
        debug::appendRaw(
            tracker_json, "target_state", eigenVectorJson(estimator_._targetStateMat),
            tracker_first);
        debug::appendRaw(
            tracker_json, "target_state11d", eigenVectorJson(estimator_._targetState11d),
            tracker_first);
        debug::appendRaw(
            tracker_json, "legacy_target_state",
            eigenVectorJson(estimator_._legacyTargetStateMat), tracker_first);
        debug::appendRaw(
            tracker_json, "ypd_angle_target_state",
            eigenVectorJson(estimator_._ypdAngleTargetStateMat), tracker_first);
        debug::appendRaw(
            tracker_json, "ypd_state_pre_predict",
            eigenVectorJson(estimator_.ypd_debug_pre_predict_state11d), tracker_first);
        debug::appendRaw(
            tracker_json, "ypd_state_prior",
            eigenVectorJson(estimator_.ypd_debug_prior_state11d), tracker_first);
        debug::appendRaw(
            tracker_json, "ypd_state_posterior",
            eigenVectorJson(estimator_.ypd_debug_posterior_state11d), tracker_first);
        debug::appendRaw(
            tracker_json, "ypd_state_before_reset",
            eigenVectorJson(estimator_.ypd_debug_reset_state11d), tracker_first);
        debug::appendRaw(
            tracker_json, "ypd_cov_diag_pre_predict",
            eigenVectorJson(estimator_.ypd_debug_pre_predict_cov_diag), tracker_first);
        debug::appendRaw(
            tracker_json, "ypd_cov_diag_prior",
            eigenVectorJson(estimator_.ypd_debug_prior_cov_diag), tracker_first);
        debug::appendRaw(
            tracker_json, "ypd_cov_diag_posterior",
            eigenVectorJson(estimator_.ypd_debug_posterior_cov_diag), tracker_first);
        debug::appendRaw(
            tracker_json, "ypd_cov_diag_before_reset",
            eigenVectorJson(estimator_.ypd_debug_reset_cov_diag), tracker_first);
        debug::appendBool(
            tracker_json, "fire_motion_uniform", estimator_.fire_motion_uniform,
            tracker_first);
        debug::appendBool(
            tracker_json, "fire_observation_stable", estimator_.fire_observation_stable,
            tracker_first);
        debug::appendBool(
            tracker_json, "fire_motion_translation_blocked",
            estimator_.fire_motion_translation_blocked, tracker_first);
        debug::appendNumber(
            tracker_json, "fire_motion_center_accel_metric",
            estimator_.fire_motion_center_accel_metric, tracker_first);
        debug::appendNumber(
            tracker_json, "fire_motion_omega_metric", estimator_.fire_motion_omega_metric,
            tracker_first);
        debug::appendNumber(
            tracker_json, "fire_motion_translation_burst_metric",
            estimator_.fire_motion_translation_burst_metric, tracker_first);
        debug::appendNumber(
            tracker_json, "fire_motion_translation_drift_metric",
            estimator_.fire_motion_translation_drift_metric, tracker_first);
        debug::appendRaw(
            tracker_json, "tracked_armor",
            armorJson(estimator_._trackedArmor, angle_solver_._cam_instant_matrix),
            tracker_first);
        tracker_json << '}';

        const rm::FireControlRuntimeStats& runtime_stats = fire_control_.lastRuntimeStats();
        std::ostringstream control_json;
        control_json << std::setprecision(10) << '{';
        bool control_first = true;
        debug::appendInt(control_json, "aiming_state", control.aiming_state, control_first);
        debug::appendInt(control_json, "shot_mode", control.shot_mode, control_first);
        debug::appendNumber(control_json, "yaw_deg", control.gimbal_yaw, control_first);
        debug::appendNumber(control_json, "pitch_deg", control.gimbal_pitch, control_first);
        debug::appendNumber(control_json, "yaw_error", control.yaw_error, control_first);
        debug::appendNumber(control_json, "command_distance_m", command_distance_m, control_first);
        debug::appendNumber(
            control_json, "raw_command_yaw_deg", fire_control_.raw_command_yaw_deg_,
            control_first);
        debug::appendNumber(
            control_json, "raw_command_pitch_deg", fire_control_.raw_command_pitch_deg_,
            control_first);
        debug::appendNumber(
            control_json, "filtered_command_yaw_deg", fire_control_.filtered_command_yaw_deg_,
            control_first);
        debug::appendNumber(
            control_json, "filtered_command_pitch_deg",
            fire_control_.filtered_command_pitch_deg_, control_first);
        debug::appendNumber(control_json, "yaw_speed_deg_s", fire_control_._yaw_speed, control_first);
        debug::appendNumber(
            control_json, "simulator_state_age_s", fire_control_.simulator_state_age_s_,
            control_first);
        debug::appendBool(
            control_json, "yaw_speed_feedback_initialized",
            fire_control_.yaw_speed_feedback_initialized_, control_first);
        debug::appendBool(
            control_json, "planner_active", runtime_stats.planner_active, control_first);
        debug::appendBool(
            control_json, "preview_mpc_active", runtime_stats.preview_mpc_active,
            control_first);
        debug::appendBool(
            control_json, "yaw_planner_active", fire_control_.yaw_planner_active_,
            control_first);
        std::ostringstream yaw_plan_json;
        yaw_plan_json << std::setprecision(10) << '{';
        bool yaw_plan_first = true;
        debug::appendBool(
            yaw_plan_json, "valid", fire_control_.last_yaw_plan_valid_, yaw_plan_first);
        debug::appendInt(
            yaw_plan_json, "selected_armor_index",
            fire_control_.last_plan_selected_armor_index_, yaw_plan_first);
        debug::appendNumber(
            yaw_plan_json, "execution_delay_s",
            fire_control_.last_plan_execution_delay_s_, yaw_plan_first);
        debug::appendNumber(
            yaw_plan_json, "estimated_fly_time_s",
            fire_control_.last_plan_estimated_fly_time_s_, yaw_plan_first);
        debug::appendNumber(
            yaw_plan_json, "target_yaw_deg",
            fire_control_.last_plan_target_yaw_deg_, yaw_plan_first);
        debug::appendNumber(
            yaw_plan_json, "target_yaw_vel_deg_s",
            fire_control_.last_plan_target_yaw_vel_deg_s_, yaw_plan_first);
        debug::appendNumber(
            yaw_plan_json, "impact_delta_angle_deg",
            fire_control_.last_plan_impact_delta_angle_deg_, yaw_plan_first);
        debug::appendRaw(
            yaw_plan_json, "target_pos_m",
            eigen3Json(fire_control_.last_plan_target_pos_), yaw_plan_first);
        debug::appendRaw(
            yaw_plan_json, "zero_vxy_target_pos_m",
            eigen3Json(fire_control_.last_plan_zero_vxy_target_pos_), yaw_plan_first);
        debug::appendNumber(
            yaw_plan_json, "zero_vxy_delta_m",
            fire_control_.last_plan_zero_vxy_delta_m_, yaw_plan_first);
        yaw_plan_json << '}';
        debug::appendRaw(control_json, "yaw_plan", yaw_plan_json.str(), control_first);
        debug::appendBool(
            control_json, "yaw_static_mpc_bypass_active",
            fire_control_.yaw_static_mpc_bypass_active_, control_first);
        debug::appendString(
            control_json, "fire_phase_mode",
            firePhaseModeName(fire_control_.fire_phase_mode_), control_first);
        debug::appendNumber(
            control_json, "fire_next_slot_delay_ms",
            fire_control_.fire_next_slot_delay_ms_, control_first);
        debug::appendNumber(
            control_json, "fire_tolerance_deg", fire_control_.fire_tolerance_deg_,
            control_first);
        debug::appendNumber(
            control_json, "fire_cmd_delta_deg", fire_control_.fire_cmd_delta_deg_,
            control_first);
        debug::appendNumber(
            control_json, "fire_follow_error_deg", fire_control_.fire_follow_error_deg_,
            control_first);
        debug::appendNumber(
            control_json, "fire_pitch_cmd_delta_deg",
            fire_control_.fire_pitch_cmd_delta_deg_, control_first);
        debug::appendNumber(
            control_json, "fire_pitch_follow_error_deg",
            fire_control_.fire_pitch_follow_error_deg_, control_first);
        debug::appendNumber(
            control_json, "fire_first_slot_error_deg",
            fire_control_.fire_first_slot_error_deg_, control_first);
        debug::appendNumber(
            control_json, "yaw_preview_tracking_error_deg",
            fire_control_.yaw_preview_tracking_error_deg_, control_first);
        debug::appendNumber(
            control_json, "yaw_fire_check_error_deg",
            fire_control_.yaw_fire_check_error_deg_, control_first);
        debug::appendInt(
            control_json, "fire_viable_slot_count",
            fire_control_.fire_viable_slot_count_, control_first);
        debug::appendBool(
            control_json, "fire_burst_active", fire_control_.fire_burst_active_,
            control_first);
        debug::appendBool(
            control_json, "fire_mechanical_hold_active",
            fire_control_.fire_mechanical_hold_active_, control_first);
        debug::appendBool(
            control_json, "fire_gate_valid", fire_control_.fire_gate_valid_,
            control_first);
        debug::appendBool(
            control_json, "fire_gate_mcu_permit", fire_control_.fire_gate_mcu_permit_,
            control_first);
        debug::appendBool(
            control_json, "fire_gate_command_stable",
            fire_control_.fire_gate_command_stable_, control_first);
        debug::appendBool(
            control_json, "fire_gate_follow", fire_control_.fire_gate_follow_,
            control_first);
        debug::appendBool(
            control_json, "fire_gate_preview", fire_control_.fire_gate_preview_,
            control_first);
        debug::appendBool(
            control_json, "fire_gate_impact_angle",
            fire_control_.fire_gate_impact_angle_, control_first);
        debug::appendBool(
            control_json, "fire_gate_motion_uniform",
            fire_control_.fire_gate_motion_uniform_, control_first);
        debug::appendBool(
            control_json, "fire_gate_observation_stable",
            fire_control_.fire_gate_observation_stable_, control_first);
        debug::appendBool(
            control_json, "fire_gate_slot_window",
            fire_control_.fire_gate_slot_window_, control_first);
        debug::appendBool(
            control_json, "yaw_preview_tracking_valid",
            fire_control_.yaw_preview_tracking_valid_, control_first);
        debug::appendBool(
            control_json, "yaw_fire_check_valid", fire_control_.yaw_fire_check_valid_,
            control_first);
        debug::appendRaw(
            control_json, "runtime", fireRuntimeStatsJson(runtime_stats), control_first);
        control_json << '}';

        const std::string command_json = aimCommandJson(command);

        std::ostringstream command_pipeline_json;
        command_pipeline_json << std::setprecision(10) << '{';
        bool command_pipeline_first = true;
        debug::appendString(
            command_pipeline_json, "policy", command_pipeline.policy,
            command_pipeline_first);
        debug::appendString(
            command_pipeline_json, "source", command_pipeline.source,
            command_pipeline_first);
        debug::appendBool(
            command_pipeline_json, "reference_armor_available",
            command_pipeline.reference_armor_available, command_pipeline_first);
        debug::appendBool(
            command_pipeline_json, "can_fallback_lock",
            command_pipeline.can_fallback_lock, command_pipeline_first);
        debug::appendNumber(
            command_pipeline_json, "image_yaw_command_deg",
            command_pipeline.image_yaw_command_deg, command_pipeline_first);
        debug::appendNumber(
            command_pipeline_json, "ballistic_pitch_command_deg",
            command_pipeline.ballistic_pitch_command_deg, command_pipeline_first);
        debug::appendNumber(
            command_pipeline_json, "fallback_yaw_command_deg",
            command_pipeline.fallback_yaw_command_deg, command_pipeline_first);
        debug::appendNumber(
            command_pipeline_json, "fallback_pitch_command_deg",
            command_pipeline.fallback_pitch_command_deg, command_pipeline_first);
        debug::appendNumber(
            command_pipeline_json, "detected_distance_m",
            command_pipeline.detected_distance_m, command_pipeline_first);
        debug::appendNumber(
            command_pipeline_json, "fallback_distance_m",
            command_pipeline.fallback_distance_m, command_pipeline_first);
        debug::appendBool(
            command_pipeline_json, "image_yaw_override_used",
            command_pipeline.image_yaw_override_used, command_pipeline_first);
        debug::appendBool(
            command_pipeline_json, "ballistic_pitch_override_used",
            command_pipeline.ballistic_pitch_override_used, command_pipeline_first);
        debug::appendRaw(
            command_pipeline_json, "fire_control_command",
            aimCommandJson(command_pipeline.fire_control_command), command_pipeline_first);
        debug::appendRaw(
            command_pipeline_json, "pre_smoothing_command",
            aimCommandJson(command_pipeline.pre_smoothing_command), command_pipeline_first);
        debug::appendRaw(
            command_pipeline_json, "post_smoothing_command", command_json,
            command_pipeline_first);
        command_pipeline_json << '}';

        std::ostringstream camera_json;
        camera_json << std::setprecision(10) << '{';
        bool camera_first = true;
        debug::appendString(
            camera_json, "profile_id", profile_frame.camera_profile_id, camera_first);
        debug::appendNumber(camera_json, "focal_mm", profile_frame.camera_focal_mm, camera_first);
        debug::appendRaw(camera_json, "source_roi_px", rectJson(profile_frame.source_roi_px), camera_first);
        debug::appendInt(
            camera_json, "source_image_width", profile_frame.source_image_width, camera_first);
        debug::appendInt(
            camera_json, "source_image_height", profile_frame.source_image_height, camera_first);
        debug::appendNumber(
            camera_json, "virtual_scale_x", profile_frame.virtual_scale_x, camera_first);
        debug::appendNumber(
            camera_json, "virtual_scale_y", profile_frame.virtual_scale_y, camera_first);
        debug::appendString(
            camera_json, "matrix_source",
            profile_frame.has_camera_matrix_override ? "sim_profile_override" : "param_yaml",
            camera_first);
        if (!angle_solver_._cam_instant_matrix.empty()) {
            debug::appendNumber(
                camera_json, "fx", angle_solver_._cam_instant_matrix.at<double>(0, 0),
                camera_first);
            debug::appendNumber(
                camera_json, "fy", angle_solver_._cam_instant_matrix.at<double>(1, 1),
                camera_first);
            debug::appendNumber(
                camera_json, "cx", angle_solver_._cam_instant_matrix.at<double>(0, 2),
                camera_first);
            debug::appendNumber(
                camera_json, "cy", angle_solver_._cam_instant_matrix.at<double>(1, 2),
                camera_first);
        }
        debug::appendInt(camera_json, "image_width", frame.srcImg.cols, camera_first);
        debug::appendInt(camera_json, "image_height", frame.srcImg.rows, camera_first);
        camera_json << '}';

        std::ostringstream out;
        out << std::setprecision(10) << '{';
        bool first = true;
        debug::appendString(out, "source", "vivsionn_pipeline", first);
        debug::appendString(out, "backend", backendName(), first);
        debug::appendString(out, "mode", toString(mode), first);
        debug::appendUInt(out, "frame_count", recognition_frame_count_ + 1, first);
        debug::appendUInt(out, "source_producer_epoch", frame.source_producer_epoch, first);
        debug::appendUInt(out, "source_image_seq", frame.source_image_seq, first);
        debug::appendUInt(
            out, "source_capture_timestamp_ns", frame.source_capture_timestamp_ns, first);
        debug::appendUInt(
            out, "vision_completion_timestamp_ns", command.vision_completion_timestamp_ns,
            first);
        debug::appendNumber(out, "frame_timestamp_ms", frame_meta.timeStamp, first);
        debug::appendNumber(out, "usb_timestamp_ms", frame_meta.usb_timeStamp, first);
        debug::appendNumber(
            out, "simulator_state_age_s", frame_meta.simulator_state_age_s, first);
        debug::appendNumber(out, "input_gimbal_yaw_deg", frame_meta.poseEuler.yaw, first);
        debug::appendNumber(out, "input_gimbal_pitch_deg", frame_meta.poseEuler.pitch, first);
        debug::appendNumber(out, "input_gimbal_roll_deg", frame_meta.poseEuler.roll, first);
        debug::appendNumber(out, "input_gimbal_yaw_speed_deg_s", frame_meta.fb.yaw_speed, first);
        std::ostringstream observation_contract;
        observation_contract << std::setprecision(10) << '{';
        bool observation_first = true;
        debug::appendString(
            observation_contract, "schema", "observation_candidate.v1", observation_first);
        debug::appendUInt(
            observation_contract, "exposure_timestamp_ns", frame.source_capture_timestamp_ns,
            observation_first);
        debug::appendUInt(
            observation_contract, "gimbal_pose_timestamp_ns",
            profile_frame.gimbal_pose_timestamp_ns, observation_first);
        debug::appendBool(
            observation_contract, "gimbal_pose_exposure_matched",
            profile_frame.gimbal_pose_exposure_matched, observation_first);
        debug::appendBool(
            observation_contract, "tracker_world_transform_exposure_matched",
            profile_frame.tracker_world_transform_exposure_matched,
            observation_first);
        debug::appendRaw(
            observation_contract, "tracker_origin_world_ros_m",
            eigen3Json(Eigen::Vector3d(
                profile_frame.tracker_origin_world_ros_m[0],
                profile_frame.tracker_origin_world_ros_m[1],
                profile_frame.tracker_origin_world_ros_m[2])),
            observation_first);
        debug::appendRaw(
            observation_contract, "tracker_frame_rpy_world_ros_rad",
            eigen3Json(Eigen::Vector3d(
                profile_frame.tracker_frame_rpy_world_ros_rad[0],
                profile_frame.tracker_frame_rpy_world_ros_rad[1],
                profile_frame.tracker_frame_rpy_world_ros_rad[2])),
            observation_first);
        std::ostringstream tracker_quaternion;
        tracker_quaternion << std::setprecision(10) << '['
                           << profile_frame.tracker_gimbal_quaternion_world_wxyz[0] << ','
                           << profile_frame.tracker_gimbal_quaternion_world_wxyz[1] << ','
                           << profile_frame.tracker_gimbal_quaternion_world_wxyz[2] << ','
                           << profile_frame.tracker_gimbal_quaternion_world_wxyz[3] << ']';
        debug::appendRaw(
            observation_contract, "tracker_gimbal_quaternion_world_wxyz",
            tracker_quaternion.str(), observation_first);
        debug::appendRaw(
            observation_contract, "camera_origin_world_ros_m",
            eigen3Json(Eigen::Vector3d(
                profile_frame.camera_origin_world_ros_m[0],
                profile_frame.camera_origin_world_ros_m[1],
                profile_frame.camera_origin_world_ros_m[2])),
            observation_first);
        std::ostringstream camera_quaternion;
        camera_quaternion << std::setprecision(10) << '['
                          << profile_frame.camera_quaternion_world_wxyz[0] << ','
                          << profile_frame.camera_quaternion_world_wxyz[1] << ','
                          << profile_frame.camera_quaternion_world_wxyz[2] << ','
                          << profile_frame.camera_quaternion_world_wxyz[3] << ']';
        debug::appendRaw(
            observation_contract, "camera_quaternion_world_wxyz",
            camera_quaternion.str(), observation_first);
        debug::appendString(
            observation_contract, "tracker_world_transform_policy",
            "world vector -> inverse exposure gimbal quaternion -> R_gimbal_pose_to_tracker; origin is exposure gimbal world position",
            observation_first);
        debug::appendString(
            observation_contract, "timestamp_semantics",
            "source capture timestamp; gimbal pose is the pose attached to this frame",
            observation_first);
        debug::appendString(
            observation_contract, "camera_frame", "opencv_camera", observation_first);
        debug::appendString(
            observation_contract, "gimbal_frame", "solver_gimbal", observation_first);
        debug::appendString(
            observation_contract, "tracker_frame", "ypd_tracker", observation_first);
        debug::appendRaw(
            observation_contract, "R_camera2gimbal",
            eigenMatrix3Json(angle_solver_.cameraToGimbalRotation()), observation_first);
        debug::appendRaw(
            observation_contract, "t_camera2gimbal_m",
            eigen3Json(angle_solver_.cameraToGimbalTranslationM()), observation_first);
        debug::appendRaw(
            observation_contract, "R_gimbal_pose_to_tracker",
            eigenMatrix3Json(exposureGimbalPoseRotation(angle_solver_, frame_meta)),
            observation_first);
        debug::appendString(
            observation_contract, "translation_gimbal_pose_to_tracker",
            "zero_in_current_solver_contract", observation_first);
        debug::appendString(
            observation_contract, "corner_covariance_status", "unavailable",
            observation_first);
        debug::appendString(
            observation_contract, "semantic_branch_policy",
            "PnP enumerates pose solutions; armor outward normal polarity is unique",
            observation_first);
        observation_contract << '}';
        debug::appendRaw(out, "observation_contract", observation_contract.str(), first);
        debug::appendRaw(out, "camera", camera_json.str(), first);
        debug::appendRaw(out, "dual_focal", dualFocalJson(dual_focal_decision), first);
        debug::appendRaw(out, "calibration", calibrationJson(angle_solver_), first);
        debug::appendRaw(out, "detector", detector_json.str(), first);
        debug::appendRaw(out, "subpixel_refinement", refinementJson(first_detected_vertices, solved), first);
        debug::appendUInt(out, "solved_count", solved.size(), first);
        debug::appendUInt(
            out, "pnp_rejected_count",
            detected_count >= solved.size() ? detected_count - solved.size() : 0, first);
        debug::appendString(
            out, "pnp_reject_reason_when_unmatched",
            "no_finite_pnp_solution_or_nonpositive_distance", first);
        debug::appendRaw(
            out, "solved_armors",
            armorsJson(solved, angle_solver_._cam_instant_matrix, true), first);
        debug::appendRaw(
            out, "first_solved",
            solved.empty()
                ? "null"
                : armorJson(solved.front(), angle_solver_._cam_instant_matrix, true),
            first);
        debug::appendRaw(out, "tracker", tracker_json.str(), first);
        debug::appendRaw(out, "fire_control", control_json.str(), first);
        debug::appendRaw(out, "command_pipeline", command_pipeline_json.str(), first);
        debug::appendRaw(out, "aim_command", command_json, first);
        out << '}';

        const std::string payload = out.str();
        debug::writeJsonFile(path, payload);
        debug::appendJsonLine(jsonl_path, payload);
    }

    void reportRecognitionDiagnostics(
        std::size_t detected_count,
        const std::vector<std::shared_ptr<rm::Armor>>& solved,
        const rm::ControlData& control,
        int first_detected_number,
        int first_detected_color,
        float first_detected_confidence,
        const cv::Point2f& first_detected_center,
        std::size_t first_detected_vertex_count,
        double command_distance_m)
    {
        const auto now = std::chrono::steady_clock::now();
        ++recognition_frame_count_;
        if (now - last_recognition_report_ < std::chrono::seconds(1)) {
            return;
        }
        last_recognition_report_ = now;

        std::cerr << "[aim_sim_bridge] recog frame=" << recognition_frame_count_
                  << " det=" << detected_count
                  << " solved=" << solved.size()
                  << " tracker_detected=" << (estimator_._detectedFlag ? "yes" : "no")
                  << " tracker_state=" << estimator_.trackerStateStr[estimator_.tracker_state]
                  << " update_state=" << estimator_.UpdateStateStr[estimator_.update_state]
                  << " obs=" << estimator_._current_obs_armors.size()
                  << " tracker_inputs=" << estimator_._current_tracker_input_armors.size()
                  << " primary=" << estimator_._current_primary_observation_index
                  << " raw_dist=" << estimator_.distance_
                  << " cmd_dist=" << command_distance_m
                  << " ctrl_state=0x" << std::hex << static_cast<int>(control.aiming_state)
                  << " shot=0x" << static_cast<int>(control.shot_mode) << std::dec;

        if (!estimator_._current_obs_match_ids.empty()) {
            std::cerr << " match_ids=[";
            for (std::size_t i = 0; i < estimator_._current_obs_match_ids.size(); ++i) {
                if (i > 0) std::cerr << ",";
                std::cerr << estimator_._current_obs_match_ids[i];
            }
            std::cerr << "]";
        }

        if (estimator_._trackedArmor) {
            std::cerr << " tracked{num=" << estimator_._trackedArmor->number
                      << " type=" << static_cast<int>(estimator_._trackedArmor->type)
                      << " color=" << estimator_._trackedArmor->color << "}";
        }

        if (detected_count > 0) {
            std::cerr << " first_det{num=" << first_detected_number
                      << " color=" << first_detected_color
                      << " conf=" << first_detected_confidence
                      << " center=(" << first_detected_center.x << ","
                      << first_detected_center.y << ")"
                      << " vertices=" << first_detected_vertex_count << "}";
        }

        if (!solved.empty()) {
            const auto& armor = solved.front();
            std::cerr << " first_solved{num=" << armor->number
                      << " type=" << static_cast<int>(armor->type)
                      << " color=" << armor->color
                      << " pos=(" << armor->armorPosition.x() << ","
                      << armor->armorPosition.y() << ","
                      << armor->armorPosition.z() << ")"
                      << " ypd=(" << armor->ypd.x() << ","
                      << armor->ypd.y() << "," << armor->ypd.z() << ")"
                      << " dis=" << armor->dis << "}";
            std::cerr << " solved_list=[";
            for (std::size_t i = 0; i < solved.size(); ++i) {
                if (i > 0) std::cerr << ",";
                const auto& item = solved[i];
                if (!item) {
                    std::cerr << "null";
                    continue;
                }
                std::cerr << "{n=" << item->number
                          << ",t=" << static_cast<int>(item->type)
                          << ",c=" << item->color << "}";
            }
            std::cerr << "]";
        }

        std::cerr << std::endl;
    }

    AimBridgeConfig config_;
    std::unique_ptr<rm::MultiThreadDetectorTRT> armor_detector_;
    rm::AngleSolver angle_solver_;
    rm::Estimator estimator_;
    rm::FireControl fire_control_;
    std::uint8_t last_task_mode_ = 0;
    static constexpr int kMaxDriveCommandHoldFrames = 30;
    AimCommand last_drive_command_{};
    int last_drive_command_hold_frames_ = kMaxDriveCommandHoldFrames;
    double last_valid_target_distance_m_ = std::numeric_limits<double>::quiet_NaN();
    std::optional<AimCommand> last_smoothed_command_;
    std::optional<int> active_target_number_;
    bool fire_control_direct_command_ = false;
    bool precision_mode_active_ = false;
    std::string last_camera_profile_id_;
    std::uint64_t recognition_frame_count_ = 0;
    std::chrono::steady_clock::time_point last_recognition_report_ =
        std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_debug_write_{};
    std::thread completion_worker_;
    mutable std::mutex submitted_mutex_;
    std::vector<SubmittedProfile> submitted_profiles_;
    detail::BoundedCompletionQueue<AimCommand, detail::kCompletionQueueCapacity>
        completion_queue_;
    detail::AimPipelineStageTelemetry stage_telemetry_;
};

}  // namespace

std::unique_ptr<IAimPipeline> createAimPipeline(const AimBridgeConfig& config)
{
    return std::make_unique<VivsionnAimPipeline>(config);
}

}  // namespace aim_sim_bridge
