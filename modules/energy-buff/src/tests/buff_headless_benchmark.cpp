#include "buff_rune_pipeline.hpp"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr double kPixelTolerance = 1e-4;
constexpr double kPoseTolerance = 1e-5;
constexpr double kTrackerTolerance = 1e-5;
constexpr double kControlToleranceDeg = 0.05;
constexpr double kDefaultGimbalYawDeg = -42.9719439;
constexpr double kDefaultGimbalPitchDeg = -2.9794088;

struct Comparison
{
    int failures = 0;
    int reported_failures = 0;
    double max_control_delta_deg = 0.0;
    double max_pose_delta = 0.0;

    void fail(const std::string& message)
    {
        ++failures;
        if (reported_failures < 24) {
            ++reported_failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    void exact(bool condition, const std::string& message)
    {
        if (!condition) fail(message);
    }

    void near(double lhs, double rhs, double tolerance, const std::string& message)
    {
        if (std::isnan(lhs) && std::isnan(rhs)) return;
        if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
            if (lhs != rhs) fail(message + " (non-finite mismatch)");
            return;
        }
        const double delta = std::abs(lhs - rhs);
        if (delta > tolerance) {
            fail(
                message + " delta=" + std::to_string(delta) +
                " tolerance=" + std::to_string(tolerance));
        }
    }
};

rm::Frame makeFrame(
    const cv::Mat& image,
    std::uint64_t sequence,
    double source_time_ms,
    bool include_debug_image,
    bool big_buff = false,
    double gimbal_yaw_deg = kDefaultGimbalYawDeg,
    double gimbal_pitch_deg = kDefaultGimbalPitchDeg)
{
    rm::Frame frame{};
    frame.srcImg = image;
    if (include_debug_image) {
        frame.debugImg = image.clone();
    }
    frame.source_producer_epoch = 0xE12B16u;
    frame.source_image_seq = sequence;
    frame.source_capture_timestamp_ns =
        1'000'000'000ULL + static_cast<std::uint64_t>(std::llround(source_time_ms * 1e6));
    // Pose paired with the default live golden frame in
    // e10-energy-manual-state-live-current-visible/manual_view_pose.json.
    frame.poseEuler.roll = 0.0f;
    frame.poseEuler.yaw = static_cast<float>(gimbal_yaw_deg);
    frame.poseEuler.pitch = static_cast<float>(gimbal_pitch_deg);
    frame.bullet_speed = 25.0;
    frame.timeStamp = source_time_ms;
    frame.usb_timeStamp = source_time_ms;
    frame.simulator_state_age_s = 0.0;
    frame.fb.task_mode = big_buff
        ? rm::FeedBackData::TASK_MODE::HIT_BIG_BUFF
        : rm::FeedBackData::TASK_MODE::HIT_SMALL_BUFF;
    frame.fb.self_team = rm::FeedBackData::SELF_TEAM::SELF_BLUE;
    frame.fb.heat = 0;
    frame.fb.heat_cap = 200;
    frame.fb.bullet_speed = 25.0f;
    frame.fb.gimbal_roll = 0.0f;
    frame.fb.gimbal_yaw = frame.poseEuler.yaw;
    frame.fb.gimbal_pitch = frame.poseEuler.pitch;
    frame.fb.yaw_speed = 0.0f;
    frame.fb.__reserved[0] = 1;
    frame.startTime = std::chrono::high_resolution_clock::now();
    return frame;
}

bool waitForResult(
    auto_buff::BuffRunePipeline& pipeline,
    auto_buff::BuffRuneResult* result,
    std::chrono::milliseconds timeout)
{
    const auto deadline = Clock::now() + timeout;
    while (Clock::now() < deadline) {
        if (pipeline.tryPopLatest(result)) return true;
        std::this_thread::sleep_for(100us);
    }
    return false;
}

void comparePoint(
    Comparison& comparison,
    const cv::Point2f& lhs,
    const cv::Point2f& rhs,
    double tolerance,
    const std::string& name)
{
    comparison.near(lhs.x, rhs.x, tolerance, name + ".x");
    comparison.near(lhs.y, rhs.y, tolerance, name + ".y");
}

void comparePointVector(
    Comparison& comparison,
    const std::vector<cv::Point2f>& lhs,
    const std::vector<cv::Point2f>& rhs,
    double tolerance,
    const std::string& name)
{
    comparison.exact(lhs.size() == rhs.size(), name + ".size");
    const std::size_t count = std::min(lhs.size(), rhs.size());
    for (std::size_t index = 0; index < count; ++index) {
        comparePoint(
            comparison, lhs[index], rhs[index], tolerance,
            name + "[" + std::to_string(index) + "]");
    }
}

void compareEigen3(
    Comparison& comparison,
    const Eigen::Vector3d& lhs,
    const Eigen::Vector3d& rhs,
    double tolerance,
    const std::string& name)
{
    for (int index = 0; index < 3; ++index) {
        const double delta = std::abs(lhs[index] - rhs[index]);
        comparison.max_pose_delta = std::max(comparison.max_pose_delta, delta);
        comparison.near(
            lhs[index], rhs[index], tolerance,
            name + "[" + std::to_string(index) + "]");
    }
}

void compareBlade(
    Comparison& comparison,
    const auto_buff::FanBlade& lhs,
    const auto_buff::FanBlade& rhs,
    const std::string& name)
{
    comparison.exact(lhs.type == rhs.type, name + ".type");
    comparison.exact(lhs.solved == rhs.solved, name + ".solved");
    comparePoint(comparison, lhs.center, rhs.center, kPixelTolerance, name + ".center");
    comparePointVector(comparison, lhs.points, rhs.points, kPixelTolerance, name + ".points");
    comparison.near(lhs.angle, rhs.angle, kPoseTolerance, name + ".angle");
    compareEigen3(
        comparison, lhs.rune_xyz_in_world, rhs.rune_xyz_in_world,
        kPoseTolerance, name + ".rune_xyz");
    compareEigen3(
        comparison, lhs.rune_ypd_in_world, rhs.rune_ypd_in_world,
        kPoseTolerance, name + ".rune_ypd");
    compareEigen3(
        comparison, lhs.ypr_in_world, rhs.ypr_in_world,
        kPoseTolerance, name + ".ypr");
    compareEigen3(
        comparison, lhs.blade_xyz_in_world, rhs.blade_xyz_in_world,
        kPoseTolerance, name + ".blade_xyz");
    compareEigen3(
        comparison, lhs.blade_ypd_in_world, rhs.blade_ypd_in_world,
        kPoseTolerance, name + ".blade_ypd");
    comparePointVector(
        comparison, lhs.pnp_observed_points, rhs.pnp_observed_points,
        kPixelTolerance, name + ".pnp_observed");
    comparePointVector(
        comparison, lhs.pnp_input_reprojected_points, rhs.pnp_input_reprojected_points,
        kPixelTolerance, name + ".pnp_input_reprojected");
    comparePointVector(
        comparison, lhs.pnp_reprojected_points, rhs.pnp_reprojected_points,
        kPixelTolerance, name + ".pnp_reprojected");
    comparison.exact(
        lhs.pnp_point_errors_px.size() == rhs.pnp_point_errors_px.size(),
        name + ".pnp_point_errors.size");
    for (std::size_t index = 0;
         index < std::min(lhs.pnp_point_errors_px.size(), rhs.pnp_point_errors_px.size());
         ++index) {
        comparison.near(
            lhs.pnp_point_errors_px[index], rhs.pnp_point_errors_px[index],
            kPixelTolerance, name + ".pnp_point_error[" + std::to_string(index) + "]");
    }
    comparePoint(
        comparison, lhs.pnp_model_center, rhs.pnp_model_center,
        kPixelTolerance, name + ".pnp_model_center");
    comparison.near(
        lhs.pnp_reproj_error_px, rhs.pnp_reproj_error_px,
        kPixelTolerance, name + ".pnp_reproj_error");
    comparison.near(lhs.pnp_score, rhs.pnp_score, kPoseTolerance, name + ".pnp_score");
    comparison.near(
        lhs.pnp_model_center_error_px, rhs.pnp_model_center_error_px,
        kPixelTolerance, name + ".pnp_model_center_error");
    comparison.near(
        lhs.pnp_model_center_radial_error_px, rhs.pnp_model_center_radial_error_px,
        kPixelTolerance, name + ".pnp_model_center_radial_error");
    comparison.near(
        lhs.pnp_model_center_tangent_error_px, rhs.pnp_model_center_tangent_error_px,
        kPixelTolerance, name + ".pnp_model_center_tangent_error");
    comparison.exact(lhs.pnp_method == rhs.pnp_method, name + ".pnp_method");
    comparison.exact(lhs.pnp_order == rhs.pnp_order, name + ".pnp_order");
}

void compareRSearch(
    Comparison& comparison,
    const std::optional<auto_buff::RSearchDebug>& lhs,
    const std::optional<auto_buff::RSearchDebug>& rhs,
    const std::string& name)
{
    comparison.exact(lhs.has_value() == rhs.has_value(), name + ".presence");
    if (!lhs.has_value() || !rhs.has_value()) return;
    comparePoint(comparison, lhs->yolo_center, rhs->yolo_center, kPixelTolerance, name + ".yolo");
    comparePoint(comparison, lhs->prior_center, rhs->prior_center, kPixelTolerance, name + ".prior");
    comparePoint(comparison, lhs->raw_center, rhs->raw_center, kPixelTolerance, name + ".raw");
    comparison.exact(lhs->roi_rect == rhs->roi_rect, name + ".roi_rect");
    comparison.exact(lhs->template_rect == rhs->template_rect, name + ".template_rect");
    comparison.near(
        lhs->template_score, rhs->template_score, kPoseTolerance, name + ".template_score");
    comparison.exact(lhs->template_hits == rhs->template_hits, name + ".template_hits");
    comparison.near(lhs->radius, rhs->radius, kPixelTolerance, name + ".radius");
    comparison.exact(lhs->total_contours == rhs->total_contours, name + ".total_contours");
    comparison.exact(lhs->accepted_count == rhs->accepted_count, name + ".accepted_count");
    comparison.exact(lhs->used_template == rhs->used_template, name + ".used_template");
    comparison.exact(lhs->used_hold_center == rhs->used_hold_center, name + ".used_hold");
    comparison.exact(
        lhs->used_contour_center == rhs->used_contour_center, name + ".used_contour");
    comparePointVector(
        comparison, lhs->accepted_centers, rhs->accepted_centers,
        kPixelTolerance, name + ".accepted_centers");
    comparison.exact(
        lhs->accepted_contour_points == rhs->accepted_contour_points,
        name + ".accepted_contour_points");
    comparison.exact(
        lhs->masked_roi.size() == rhs->masked_roi.size() &&
            lhs->masked_roi.type() == rhs->masked_roi.type(),
        name + ".masked_roi contract");
    if (!lhs->masked_roi.empty() && !rhs->masked_roi.empty() &&
        lhs->masked_roi.size() == rhs->masked_roi.size() &&
        lhs->masked_roi.type() == rhs->masked_roi.type()) {
        comparison.exact(
            cv::norm(lhs->masked_roi, rhs->masked_roi, cv::NORM_INF) == 0.0,
            name + ".masked_roi pixels");
    }
}

void compareRune(
    Comparison& comparison,
    const std::optional<auto_buff::PowerRune>& lhs,
    const std::optional<auto_buff::PowerRune>& rhs,
    const std::string& name)
{
    comparison.exact(lhs.has_value() == rhs.has_value(), name + ".presence");
    if (!lhs.has_value() || !rhs.has_value()) return;
    comparison.exact(lhs->type == rhs->type, name + ".type");
    comparison.exact(lhs->target_switched == rhs->target_switched, name + ".target_switched");
    comparison.exact(lhs->switch_deferred == rhs->switch_deferred, name + ".switch_deferred");
    comparison.exact(
        lhs->selected_phase_index == rhs->selected_phase_index,
        name + ".selected_phase_index");
    comparison.near(
        lhs->selected_roll_offset, rhs->selected_roll_offset,
        kTrackerTolerance, name + ".selected_roll_offset");
    comparePoint(comparison, lhs->r_center, rhs->r_center, kPixelTolerance, name + ".r_center");
    compareEigen3(
        comparison, lhs->xyz_in_world, rhs->xyz_in_world,
        kPoseTolerance, name + ".xyz");
    compareEigen3(
        comparison, lhs->ypd_in_world, rhs->ypd_in_world,
        kPoseTolerance, name + ".ypd");
    compareEigen3(
        comparison, lhs->ypr_in_world, rhs->ypr_in_world,
        kPoseTolerance, name + ".ypr");
    compareEigen3(
        comparison, lhs->blade_xyz_in_world, rhs->blade_xyz_in_world,
        kPoseTolerance, name + ".blade_xyz");
    compareEigen3(
        comparison, lhs->blade_ypd_in_world, rhs->blade_ypd_in_world,
        kPoseTolerance, name + ".blade_ypd");
    comparison.exact(lhs->fanblades.size() == rhs->fanblades.size(), name + ".fanblades.size");
    for (std::size_t index = 0;
         index < std::min(lhs->fanblades.size(), rhs->fanblades.size());
         ++index) {
        compareBlade(
            comparison, lhs->fanblades[index], rhs->fanblades[index],
            name + ".fanblades[" + std::to_string(index) + "]");
    }
    compareRSearch(comparison, lhs->r_search_debug, rhs->r_search_debug, name + ".r_search");
}

void compareTracker(
    Comparison& comparison,
    const auto_buff::BuffTracker::DebugSnapshot& lhs,
    const auto_buff::BuffTracker::DebugSnapshot& rhs,
    const std::string& name)
{
    comparison.exact(lhs.initialized == rhs.initialized, name + ".initialized");
    comparison.exact(lhs.lost == rhs.lost, name + ".lost");
    comparison.exact(lhs.fit_valid == rhs.fit_valid, name + ".fit_valid");
    comparison.exact(lhs.switch_deferred == rhs.switch_deferred, name + ".switch_deferred");
    comparison.exact(lhs.target_switched == rhs.target_switched, name + ".target_switched");
    comparison.exact(lhs.direction == rhs.direction, name + ".direction");
    comparison.exact(lhs.history_size == rhs.history_size, name + ".history_size");
    comparison.exact(
        lhs.selected_phase_index == rhs.selected_phase_index,
        name + ".selected_phase_index");
    comparison.exact(lhs.phase_origin_index == rhs.phase_origin_index, name + ".phase_origin_index");
    comparison.exact(lhs.reinit_reason == rhs.reinit_reason, name + ".reinit_reason");
    comparison.exact(
        lhs.speed_measurement_status == rhs.speed_measurement_status,
        name + ".speed_measurement_status");

    const auto near = [&](double a, double b, const char* field) {
        comparison.near(a, b, kTrackerTolerance, name + "." + field);
    };
    near(lhs.observed_roll, rhs.observed_roll, "observed_roll");
    near(lhs.observed_speed_raw, rhs.observed_speed_raw, "observed_speed_raw");
    near(lhs.observed_speed, rhs.observed_speed, "observed_speed");
    near(lhs.filtered_roll, rhs.filtered_roll, "filtered_roll");
    near(lhs.filtered_speed, rhs.filtered_speed, "filtered_speed");
    near(lhs.filtered_speed_raw, rhs.filtered_speed_raw, "filtered_speed_raw");
    near(lhs.predicted_roll, rhs.predicted_roll, "predicted_roll");
    near(lhs.predicted_speed, rhs.predicted_speed, "predicted_speed");
    near(lhs.curve_speed_now, rhs.curve_speed_now, "curve_speed_now");
    near(lhs.curve_speed_raw, rhs.curve_speed_raw, "curve_speed_raw");
    near(lhs.curve_speed_after_predict, rhs.curve_speed_after_predict, "curve_speed_after_predict");
    near(
        lhs.curve_speed_after_blade_update, rhs.curve_speed_after_blade_update,
        "curve_speed_after_blade_update");
    near(
        lhs.curve_speed_before_speed_update, rhs.curve_speed_before_speed_update,
        "curve_speed_before_speed_update");
    near(
        lhs.curve_speed_after_speed_update, rhs.curve_speed_after_speed_update,
        "curve_speed_after_speed_update");
    near(lhs.curve_phi_after_predict, rhs.curve_phi_after_predict, "curve_phi_after_predict");
    near(
        lhs.curve_phi_after_blade_update, rhs.curve_phi_after_blade_update,
        "curve_phi_after_blade_update");
    near(
        lhs.curve_phi_before_speed_update, rhs.curve_phi_before_speed_update,
        "curve_phi_before_speed_update");
    near(
        lhs.curve_phi_after_speed_update, rhs.curve_phi_after_speed_update,
        "curve_phi_after_speed_update");
    near(
        lhs.speed_measurement_predicted, rhs.speed_measurement_predicted,
        "speed_measurement_predicted");
    near(
        lhs.speed_measurement_innovation, rhs.speed_measurement_innovation,
        "speed_measurement_innovation");
    near(
        lhs.speed_measurement_noise, rhs.speed_measurement_noise,
        "speed_measurement_noise");
    near(lhs.selected_roll_offset, rhs.selected_roll_offset, "selected_roll_offset");
    near(lhs.fit_a, rhs.fit_a, "fit_a");
    near(lhs.fit_w, rhs.fit_w, "fit_w");
    near(lhs.fit_phi, rhs.fit_phi, "fit_phi");

    for (std::size_t index = 0; index < lhs.blade_observations.size(); ++index) {
        const auto& a = lhs.blade_observations[index];
        const auto& b = rhs.blade_observations[index];
        const std::string blade = name + ".blade[" + std::to_string(index) + "]";
        comparison.exact(a.present == b.present, blade + ".present");
        comparison.exact(a.solved == b.solved, blade + ".solved");
        comparison.exact(a.selected == b.selected, blade + ".selected");
        comparison.near(
            a.assoc_global_roll_rad, b.assoc_global_roll_rad,
            kTrackerTolerance, blade + ".assoc_global_roll");
    }
}

void compareShotGate(
    Comparison& comparison,
    const auto_buff::BuffShotGateSnapshot& lhs,
    const auto_buff::BuffShotGateSnapshot& rhs,
    const std::string& name)
{
    comparison.exact(lhs.requested == rhs.requested, name + ".requested");
    comparison.exact(lhs.allowed == rhs.allowed, name + ".allowed");
    comparison.exact(lhs.pending_detected == rhs.pending_detected, name + ".pending_detected");
    comparison.exact(lhs.r_center_ok == rhs.r_center_ok, name + ".r_center_ok");
    comparison.exact(lhs.pnp_ok == rhs.pnp_ok, name + ".pnp_ok");
    comparison.exact(lhs.tracker_ok == rhs.tracker_ok, name + ".tracker_ok");
    comparison.exact(lhs.gimbal_ok == rhs.gimbal_ok, name + ".gimbal_ok");
    comparison.exact(lhs.stable_ok == rhs.stable_ok, name + ".stable_ok");
    comparison.exact(lhs.stable_frames == rhs.stable_frames, name + ".stable_frames");
    comparison.exact(lhs.reason_code == rhs.reason_code, name + ".reason_code");
    comparison.near(
        lhs.yaw_error_deg, rhs.yaw_error_deg,
        kControlToleranceDeg, name + ".yaw_error_deg");
    comparison.near(
        lhs.pitch_error_deg, rhs.pitch_error_deg,
        kControlToleranceDeg, name + ".pitch_error_deg");
    comparison.near(
        lhs.pnp_reproj_error_px, rhs.pnp_reproj_error_px,
        kPixelTolerance, name + ".pnp_reproj_error");
    comparison.near(
        lhs.pnp_model_center_error_px, rhs.pnp_model_center_error_px,
        kPixelTolerance, name + ".pnp_model_center_error");
}

void compareControl(
    Comparison& comparison,
    const rm::ControlData& lhs,
    const rm::ControlData& rhs,
    const std::string& name)
{
    comparison.exact(lhs.SOF == rhs.SOF, name + ".SOF");
    comparison.exact(lhs.EOF == rhs.EOF, name + ".EOF");
    comparison.exact(lhs.shot_mode == rhs.shot_mode, name + ".shot_mode");
    comparison.exact(lhs.shot_buff_mode == rhs.shot_buff_mode, name + ".shot_buff_mode");
    comparison.exact(lhs.aiming_state == rhs.aiming_state, name + ".aiming_state");
    const double yaw_delta = std::abs(lhs.gimbal_yaw - rhs.gimbal_yaw);
    const double pitch_delta = std::abs(lhs.gimbal_pitch - rhs.gimbal_pitch);
    const double error_delta = std::abs(lhs.yaw_error - rhs.yaw_error);
    comparison.max_control_delta_deg = std::max(
        comparison.max_control_delta_deg,
        std::max(yaw_delta, std::max(pitch_delta, error_delta)));
    comparison.near(
        lhs.gimbal_yaw, rhs.gimbal_yaw,
        kControlToleranceDeg, name + ".gimbal_yaw");
    comparison.near(
        lhs.gimbal_pitch, rhs.gimbal_pitch,
        kControlToleranceDeg, name + ".gimbal_pitch");
    comparison.near(
        lhs.yaw_error, rhs.yaw_error,
        kControlToleranceDeg, name + ".yaw_error");
}

struct ManifestExpected
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

struct ManifestFrame
{
    rm::Frame frame;
    std::uint64_t live_completion_sequence = 0;
    std::uint64_t live_completion_timestamp_ns = 0;
    ManifestExpected expected;
};

struct GoldenManifest
{
    std::vector<ManifestFrame> frames;
    std::size_t trigger_index = 0;
    std::optional<std::string> runtime_param_yaml;
    std::optional<std::string> runtime_param_yaml_fnv1a64;
};

const YAML::Node requireField(const YAML::Node& node, const char* key)
{
    const YAML::Node value = node[key];
    if (!value) throw std::runtime_error(std::string("missing manifest field: ") + key);
    return value;
}

double nullableDouble(const YAML::Node& node)
{
    return !node || node.IsNull()
        ? std::numeric_limits<double>::quiet_NaN()
        : node.as<double>();
}

std::optional<std::string> fileFnv1a64(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) return std::nullopt;
    std::uint64_t hash = 14695981039346656037ULL;
    std::array<char, 16 * 1024> buffer{};
    while (input.good()) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        for (std::streamsize index = 0; index < count; ++index) {
            hash ^= static_cast<unsigned char>(buffer[static_cast<std::size_t>(index)]);
            hash *= 1099511628211ULL;
        }
    }
    if (!input.eof()) return std::nullopt;
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << hash;
    return text.str();
}

Eigen::Vector3d parseVector3(const YAML::Node& node, const char* field)
{
    if (!node.IsSequence() || node.size() != 3) {
        throw std::runtime_error(std::string(field) + " must contain exactly 3 values");
    }
    return Eigen::Vector3d(
        nullableDouble(node[0]), nullableDouble(node[1]), nullableDouble(node[2]));
}

cv::Point2f parsePoint2(const YAML::Node& node, const char* field)
{
    if (!node.IsSequence() || node.size() != 2) {
        throw std::runtime_error(std::string(field) + " must contain exactly 2 values");
    }
    return cv::Point2f(
        static_cast<float>(nullableDouble(node[0])),
        static_cast<float>(nullableDouble(node[1])));
}

std::array<int, 4> parseInt4(const YAML::Node& node, const char* field)
{
    if (!node.IsSequence() || node.size() != 4) {
        throw std::runtime_error(std::string(field) + " must contain exactly 4 values");
    }
    return {{node[0].as<int>(), node[1].as<int>(), node[2].as<int>(), node[3].as<int>()}};
}

int compareIdentity(const rm::Frame& lhs, const rm::Frame& rhs)
{
    if (lhs.source_producer_epoch != rhs.source_producer_epoch) {
        return lhs.source_producer_epoch < rhs.source_producer_epoch ? -1 : 1;
    }
    if (lhs.source_image_seq != rhs.source_image_seq) {
        return lhs.source_image_seq < rhs.source_image_seq ? -1 : 1;
    }
    return 0;
}

std::optional<GoldenManifest> loadGoldenManifest(const std::string& manifest_path)
{
    namespace fs = std::filesystem;
    try {
        const fs::path path = fs::absolute(fs::path(manifest_path)).lexically_normal();
        const YAML::Node root = YAML::LoadFile(path.string());
        if (requireField(root, "schema").as<std::string>() !=
            "aim_buff_exact_valid_sequence_v1") {
            throw std::runtime_error("unsupported exact-valid sequence schema");
        }
        if (!requireField(root, "capture_complete").as<bool>() ||
            !requireField(root, "order_valid").as<bool>()) {
            throw std::runtime_error("manifest is incomplete or source order is invalid");
        }
        if (requireField(root, "ring_truncated").as<bool>() ||
            requireField(root, "evicted_frames").as<std::uint64_t>() != 0 ||
            requireField(root, "unretainable_frames").as<std::uint64_t>() != 0) {
            throw std::runtime_error("manifest history is truncated; replay must fail closed");
        }
        const YAML::Node frame_nodes = requireField(root, "frames");
        if (!frame_nodes.IsSequence() || frame_nodes.size() == 0 ||
            frame_nodes.size() > auto_buff::BuffExactValidSequenceCapture::kHardMaxFrames) {
            throw std::runtime_error("manifest frame count is empty or exceeds hard bound");
        }
        if (requireField(root, "retained_frames").as<std::size_t>() != frame_nodes.size()) {
            throw std::runtime_error("retained_frames does not match frames array");
        }

        GoldenManifest manifest;
        if (const YAML::Node param_yaml = root["runtime_param_yaml"];
            param_yaml && !param_yaml.IsNull()) {
            manifest.runtime_param_yaml = param_yaml.as<std::string>();
        }
        if (const YAML::Node param_hash = root["runtime_param_yaml_fnv1a64"];
            param_hash && !param_hash.IsNull()) {
            manifest.runtime_param_yaml_fnv1a64 = param_hash.as<std::string>();
        }
        manifest.trigger_index = requireField(root, "trigger_index").as<std::size_t>();
        if (manifest.trigger_index + 1 != frame_nodes.size()) {
            throw std::runtime_error("trigger must be the last retained completion");
        }
        manifest.frames.reserve(frame_nodes.size());
        std::uint64_t previous_completion = 0;
        std::size_t total_image_bytes = 0;

        for (std::size_t index = 0; index < frame_nodes.size(); ++index) {
            const YAML::Node node = frame_nodes[index];
            if (requireField(node, "index").as<std::size_t>() != index) {
                throw std::runtime_error("manifest frame indices are not contiguous");
            }
            if (requireField(node, "color_order").as<std::string>() != "BGR" ||
                requireField(node, "image_type").as<int>() != CV_8UC3) {
                throw std::runtime_error("manifest image contract must be BGR CV_8UC3");
            }
            const fs::path image_file(requireField(node, "image_file").as<std::string>());
            if (image_file.empty() || image_file.is_absolute() || image_file.has_parent_path()) {
                throw std::runtime_error("manifest image_file must be a local filename");
            }
            const int declared_width = requireField(node, "image_width").as<int>();
            const int declared_height = requireField(node, "image_height").as<int>();
            const std::size_t declared_bytes =
                requireField(node, "image_storage_bytes").as<std::size_t>();
            if (declared_width <= 0 || declared_height <= 0 || declared_bytes == 0 ||
                declared_bytes > auto_buff::BuffExactValidSequenceCapture::kHardMaxBytes ||
                static_cast<std::size_t>(declared_width) >
                    auto_buff::BuffExactValidSequenceCapture::kHardMaxBytes / 3U /
                        static_cast<std::size_t>(declared_height) ||
                static_cast<std::size_t>(declared_width) *
                        static_cast<std::size_t>(declared_height) * 3U > declared_bytes ||
                total_image_bytes > auto_buff::BuffExactValidSequenceCapture::kHardMaxBytes -
                    declared_bytes) {
                throw std::runtime_error("manifest image dimensions/storage exceed hard byte bound");
            }
            total_image_bytes += declared_bytes;
            const fs::path image_path = path.parent_path() / image_file;
            ManifestFrame item;
            item.frame = rm::Frame{};
            item.frame.srcImg = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (item.frame.srcImg.empty()) {
                throw std::runtime_error("failed to load manifest image: " + image_path.string());
            }
            if (item.frame.srcImg.cols != declared_width ||
                item.frame.srcImg.rows != declared_height ||
                item.frame.srcImg.type() != CV_8UC3) {
                throw std::runtime_error("manifest image dimensions/type mismatch");
            }

            auto& frame = item.frame;
            frame.source_producer_epoch =
                requireField(node, "source_producer_epoch").as<std::uint64_t>();
            frame.source_image_seq =
                requireField(node, "source_image_seq").as<std::uint64_t>();
            frame.source_capture_timestamp_ns =
                requireField(node, "source_capture_timestamp_ns").as<std::uint64_t>();
            frame.timeStamp = nullableDouble(requireField(node, "frame_timestamp_ms"));
            frame.usb_timeStamp = nullableDouble(requireField(node, "frame_usb_timestamp_ms"));
            frame.simulator_state_age_s =
                nullableDouble(requireField(node, "simulator_state_age_s"));
            frame.bullet_speed = nullableDouble(requireField(node, "bullet_speed"));
            frame.poseEuler.roll = static_cast<float>(
                nullableDouble(requireField(node, "gimbal_roll_deg")));
            frame.poseEuler.yaw = static_cast<float>(
                nullableDouble(requireField(node, "gimbal_yaw_deg")));
            frame.poseEuler.pitch = static_cast<float>(
                nullableDouble(requireField(node, "gimbal_pitch_deg")));
            frame.fb.SOF = static_cast<std::uint8_t>(requireField(node, "fb_sof").as<unsigned>());
            frame.fb.task_mode = static_cast<std::uint8_t>(
                requireField(node, "fb_task_mode").as<unsigned>());
            frame.fb.self_team = static_cast<std::uint8_t>(
                requireField(node, "fb_self_team").as<unsigned>());
            frame.fb.heat = requireField(node, "fb_heat").as<std::uint16_t>();
            frame.fb.heat_cap = requireField(node, "fb_heat_cap").as<std::uint16_t>();
            frame.fb.bullet_speed = static_cast<float>(
                nullableDouble(requireField(node, "fb_bullet_speed")));
            frame.fb.gimbal_roll = static_cast<float>(
                nullableDouble(requireField(node, "fb_gimbal_roll_deg")));
            frame.fb.gimbal_yaw = static_cast<float>(
                nullableDouble(requireField(node, "fb_gimbal_yaw_deg")));
            frame.fb.gimbal_pitch = static_cast<float>(
                nullableDouble(requireField(node, "fb_gimbal_pitch_deg")));
            frame.fb.yaw_speed = static_cast<float>(
                nullableDouble(requireField(node, "fb_yaw_speed")));
            frame.fb.__reserved[0] =
                requireField(node, "fb_mcu_fire_permit").as<bool>() ? 1 : 0;
            frame.fb.set_task_mode_telemetry(
                static_cast<std::uint8_t>(
                    requireField(node, "fb_raw_task_mode").as<unsigned>()),
                static_cast<std::uint8_t>(
                    requireField(node, "fb_head_mapped_task_mode").as<unsigned>()));
            frame.fb.EOF = static_cast<std::uint8_t>(requireField(node, "fb_eof").as<unsigned>());
            frame.startTime = std::chrono::high_resolution_clock::now();

            item.live_completion_sequence =
                requireField(node, "completion_sequence").as<std::uint64_t>();
            item.live_completion_timestamp_ns =
                requireField(node, "completion_timestamp_ns").as<std::uint64_t>();
            if (frame.source_producer_epoch == 0 || frame.source_image_seq == 0 ||
                item.live_completion_sequence == 0 ||
                (previous_completion != 0 &&
                 item.live_completion_sequence <= previous_completion)) {
                throw std::runtime_error("manifest contains invalid or regressing identity/order");
            }
            if (!manifest.frames.empty() &&
                compareIdentity(manifest.frames.back().frame, frame) >= 0) {
                throw std::runtime_error("manifest source identities are not strictly increasing");
            }
            previous_completion = item.live_completion_sequence;

            auto& expected = item.expected;
            expected.rune_present = requireField(node, "expected_rune_present").as<bool>();
            expected.rune_type = requireField(node, "expected_rune_type").as<int>();
            expected.solved_blades = requireField(node, "expected_solved_blades").as<std::size_t>();
            expected.target_solved = requireField(node, "expected_target_solved").as<bool>();
            expected.has_control = requireField(node, "expected_has_control").as<bool>();
            expected.current_target_control =
                requireField(node, "expected_current_target_control").as<bool>();
            expected.switch_deferred =
                requireField(node, "expected_switch_deferred").as<bool>();
            expected.target_switched =
                requireField(node, "expected_target_switched").as<bool>();
            expected.selected_target_index =
                requireField(node, "expected_selected_target_index").as<int>();
            expected.control_aiming_state =
                requireField(node, "expected_control_aiming_state").as<int>();
            expected.control_shot_mode =
                requireField(node, "expected_control_shot_mode").as<int>();
            expected.control_shot_buff_mode =
                requireField(node, "expected_control_shot_buff_mode").as<int>();
            expected.control_gimbal_yaw_deg =
                nullableDouble(requireField(node, "expected_control_gimbal_yaw_deg"));
            expected.control_gimbal_pitch_deg =
                nullableDouble(requireField(node, "expected_control_gimbal_pitch_deg"));
            expected.control_yaw_error_deg =
                nullableDouble(requireField(node, "expected_control_yaw_error_deg"));
            expected.r_center = parsePoint2(requireField(node, "expected_r_center"), "expected_r_center");
            expected.rune_xyz = parseVector3(requireField(node, "expected_rune_xyz"), "expected_rune_xyz");
            expected.rune_ypd = parseVector3(requireField(node, "expected_rune_ypd"), "expected_rune_ypd");
            expected.rune_ypr = parseVector3(requireField(node, "expected_rune_ypr"), "expected_rune_ypr");
            expected.blade_xyz = parseVector3(requireField(node, "expected_blade_xyz"), "expected_blade_xyz");
            expected.blade_ypd = parseVector3(requireField(node, "expected_blade_ypd"), "expected_blade_ypd");
            expected.target_pnp_method = requireField(node, "expected_target_pnp_method").as<int>();
            expected.target_pnp_order = parseInt4(
                requireField(node, "expected_target_pnp_order"), "expected_target_pnp_order");
            expected.target_pnp_reproj_error_px = nullableDouble(
                requireField(node, "expected_target_pnp_reproj_error_px"));
            expected.target_pnp_score =
                nullableDouble(requireField(node, "expected_target_pnp_score"));
            expected.target_pnp_model_center_error_px = nullableDouble(
                requireField(node, "expected_target_pnp_model_center_error_px"));
            manifest.frames.push_back(std::move(item));
        }

        const auto& trigger = manifest.frames.back();
        if (requireField(root, "trigger_source_producer_epoch").as<std::uint64_t>() !=
                trigger.frame.source_producer_epoch ||
            requireField(root, "trigger_source_image_seq").as<std::uint64_t>() !=
                trigger.frame.source_image_seq ||
            requireField(root, "trigger_completion_sequence").as<std::uint64_t>() !=
                trigger.live_completion_sequence ||
            !trigger.expected.rune_present || trigger.expected.solved_blades == 0 ||
            !trigger.expected.current_target_control) {
            throw std::runtime_error("manifest trigger is not a current valid rune/PnP/control result");
        }
        return manifest;
    } catch (const std::exception& error) {
        std::cerr << "Failed to load exact-valid sequence manifest: " << error.what() << '\n';
        return std::nullopt;
    }
}

std::optional<std::vector<auto_buff::BuffRuneResult>> runSequential(
    const std::string& config_path,
    const cv::Mat& image,
    bool emit_debug_artifacts,
    int frame_count,
    bool big_buff,
    double gimbal_yaw_deg,
    double gimbal_pitch_deg)
{
    std::unique_ptr<auto_buff::BuffRunePipeline> pipeline;
    if (emit_debug_artifacts) {
        // Intentionally exercise the one-argument compatibility/default path.
        pipeline = std::make_unique<auto_buff::BuffRunePipeline>(config_path);
    } else {
        pipeline = std::make_unique<auto_buff::BuffRunePipeline>(
            config_path, auto_buff::BuffRunePipelineOptions{false});
    }

    std::vector<auto_buff::BuffRuneResult> results;
    results.reserve(static_cast<std::size_t>(frame_count));
    const auto start = Clock::now() + 100ms;
    for (int index = 0; index < frame_count; ++index) {
        std::this_thread::sleep_until(start + std::chrono::milliseconds(index * 100));
        const std::uint64_t sequence = static_cast<std::uint64_t>(index + 1);
        pipeline->push(makeFrame(
            image, sequence, 10'000.0 + index * 100.0, true, big_buff,
            gimbal_yaw_deg, gimbal_pitch_deg));
        auto_buff::BuffRuneResult result;
        if (!waitForResult(*pipeline, &result, 3s)) {
            std::cerr << "Timed out waiting for sequential result " << sequence
                      << " emit_debug_artifacts=" << emit_debug_artifacts << '\n';
            return std::nullopt;
        }
        if (result.frame.source_image_seq != sequence) {
            std::cerr << "Sequential result identity mismatch expected=" << sequence
                      << " actual=" << result.frame.source_image_seq << '\n';
            return std::nullopt;
        }
        result.frame.srcImg.release();
        result.frame.debugImg.release();
        result.frame.yoloImg.release();
        results.push_back(std::move(result));
    }
    return results;
}

int runEquivalence(
    const std::string& config_path,
    const cv::Mat& image,
    int frame_count,
    bool big_buff,
    double gimbal_yaw_deg,
    double gimbal_pitch_deg)
{
    const auto debug = runSequential(
        config_path, image, true, frame_count, big_buff, gimbal_yaw_deg, gimbal_pitch_deg);
    if (!debug.has_value()) return 1;
    const auto headless = runSequential(
        config_path, image, false, frame_count, big_buff, gimbal_yaw_deg, gimbal_pitch_deg);
    if (!headless.has_value()) return 1;

    Comparison comparison;
    comparison.exact(debug->size() == headless->size(), "result vector size");
    int debug_overlay_frames = 0;
    int debug_timed_frames = 0;
    int rune_frames = 0;
    int solved_pnp_frames = 0;
    int target_control_frames = 0;
    for (std::size_t index = 0; index < std::min(debug->size(), headless->size()); ++index) {
        const auto& lhs = (*debug)[index];
        const auto& rhs = (*headless)[index];
        const std::string name = "frame[" + std::to_string(index) + "]";

        comparison.exact(lhs.debug_artifacts_emitted, name + ".debug flag true");
        comparison.exact(!rhs.debug_artifacts_emitted, name + ".headless flag false");
        comparison.exact(rhs.overlay.empty(), name + ".headless overlay empty");
        comparison.exact(
            rhs.debug_artifact_ms == 0.0,
            name + ".headless debug duration zero");
        if (!lhs.overlay.empty()) ++debug_overlay_frames;
        if (lhs.debug_artifact_ms > 0.0) ++debug_timed_frames;
        if (lhs.rune.has_value()) {
            ++rune_frames;
            if (std::any_of(
                    lhs.rune->fanblades.begin(), lhs.rune->fanblades.end(),
                    [](const auto_buff::FanBlade& blade) { return blade.solved; })) {
                ++solved_pnp_frames;
            }
        }
        if (lhs.control.aiming_state == rm::ControlData::AIMING_STATE::TARGET_DETECTED) {
            ++target_control_frames;
        }

        comparison.exact(lhs.has_control == rhs.has_control, name + ".has_control");
        comparison.exact(
            lhs.frame.source_producer_epoch == rhs.frame.source_producer_epoch,
            name + ".source_producer_epoch");
        comparison.exact(
            lhs.frame.source_image_seq == rhs.frame.source_image_seq,
            name + ".source_image_seq");
        comparison.exact(
            lhs.frame.source_capture_timestamp_ns == rhs.frame.source_capture_timestamp_ns,
            name + ".source_capture_timestamp_ns");
        comparison.exact(
            lhs.completion_sequence == index + 1 && rhs.completion_sequence == index + 1,
            name + ".completion_sequence");
        comparison.exact(
            lhs.completion_timestamp_ns != 0 && rhs.completion_timestamp_ns != 0,
            name + ".completion_timestamp_nonzero");
        comparison.exact(
            lhs.completion_counters.essential_completed == lhs.completion_sequence &&
                rhs.completion_counters.essential_completed == rhs.completion_sequence,
            name + ".completion_local_counter");
        comparison.exact(
            lhs.switch_deferred == rhs.switch_deferred,
            name + ".switch_deferred");
        comparison.exact(
            lhs.target_switched == rhs.target_switched,
            name + ".target_switched");
        comparison.exact(
            lhs.selected_target_index == rhs.selected_target_index,
            name + ".selected_target_index");
        compareRune(comparison, lhs.rune, rhs.rune, name + ".rune");
        compareTracker(comparison, lhs.tracker_debug, rhs.tracker_debug, name + ".tracker");
        compareShotGate(comparison, lhs.shot_gate, rhs.shot_gate, name + ".shot_gate");
        compareControl(comparison, lhs.control, rhs.control, name + ".control");
    }
    comparison.exact(debug_overlay_frames > 0, "debug path emitted at least one overlay");
    comparison.exact(
        debug_timed_frames == frame_count,
        "debug path recorded debug artifact time for every frame");
    comparison.exact(rune_frames > 0, "golden sequence produced at least one rune");
    comparison.exact(solved_pnp_frames > 0, "golden sequence produced at least one solved PnP");
    comparison.exact(target_control_frames > 0, "golden sequence produced target control");

    std::cout << std::fixed << std::setprecision(6)
              << "equivalence_frames=" << frame_count
              << " failures=" << comparison.failures
              << " mode=" << (big_buff ? "big" : "small")
              << " gimbal_yaw_deg=" << gimbal_yaw_deg
              << " gimbal_pitch_deg=" << gimbal_pitch_deg
              << " debug_overlay_frames=" << debug_overlay_frames
              << " rune_frames=" << rune_frames
              << " solved_pnp_frames=" << solved_pnp_frames
              << " target_control_frames=" << target_control_frames
              << " max_control_delta_deg=" << comparison.max_control_delta_deg
              << " max_pose_delta=" << comparison.max_pose_delta
              << " control_tolerance_deg=" << kControlToleranceDeg
              << " pose_tolerance=" << kPoseTolerance << '\n';
    return comparison.failures == 0 ? 0 : 1;
}

std::optional<std::vector<auto_buff::BuffRuneResult>> runManifestSequential(
    const std::string& config_path,
    const GoldenManifest& manifest,
    bool emit_debug_artifacts)
{
    auto_buff::BuffRunePipeline pipeline(
        config_path,
        auto_buff::BuffRunePipelineOptions{emit_debug_artifacts, false});
    std::vector<auto_buff::BuffRuneResult> results;
    results.reserve(manifest.frames.size());
    std::uint64_t last_completion = 0;
    rm::Frame last_source{};
    bool has_last_source = false;

    for (std::size_t index = 0; index < manifest.frames.size(); ++index) {
        rm::Frame frame = manifest.frames[index].frame;
        frame.debugImg = emit_debug_artifacts ? frame.srcImg.clone() : cv::Mat{};
        frame.yoloImg.release();
        frame.startTime = std::chrono::high_resolution_clock::now();
        const std::uint64_t expected_epoch = frame.source_producer_epoch;
        const std::uint64_t expected_sequence = frame.source_image_seq;
        pipeline.push(std::move(frame));

        auto_buff::BuffRuneResult result;
        if (!waitForResult(pipeline, &result, 5s)) {
            std::cerr << "Timed out waiting for manifest frame " << index
                      << " source=" << expected_epoch << ':' << expected_sequence
                      << " debug=" << emit_debug_artifacts << '\n';
            return std::nullopt;
        }
        if (result.frame.source_producer_epoch != expected_epoch ||
            result.frame.source_image_seq != expected_sequence) {
            std::cerr << "Manifest result identity mismatch index=" << index
                      << " expected=" << expected_epoch << ':' << expected_sequence
                      << " actual=" << result.frame.source_producer_epoch << ':'
                      << result.frame.source_image_seq << '\n';
            return std::nullopt;
        }
        if (result.completion_sequence == 0 ||
            (last_completion != 0 && result.completion_sequence <= last_completion) ||
            (has_last_source && compareIdentity(last_source, result.frame) >= 0)) {
            std::cerr << "Manifest replay produced non-increasing completion/source order at index="
                      << index << '\n';
            return std::nullopt;
        }
        last_completion = result.completion_sequence;
        last_source = result.frame;
        has_last_source = true;
        result.frame.srcImg.release();
        result.frame.debugImg.release();
        result.frame.yoloImg.release();
        results.push_back(std::move(result));
    }
    return results;
}

void compareManifestExpected(
    Comparison& comparison,
    const auto_buff::BuffRuneResult& result,
    const ManifestFrame& recorded,
    const std::string& name)
{
    const auto& expected = recorded.expected;
    comparison.exact(
        result.frame.source_producer_epoch == recorded.frame.source_producer_epoch,
        name + ".live_source_producer_epoch");
    comparison.exact(
        result.frame.source_image_seq == recorded.frame.source_image_seq,
        name + ".live_source_image_seq");
    comparison.exact(
        result.frame.source_capture_timestamp_ns == recorded.frame.source_capture_timestamp_ns,
        name + ".live_source_capture_timestamp_ns");
    comparison.exact(result.rune.has_value() == expected.rune_present, name + ".live_rune_present");
    comparison.exact(result.has_control == expected.has_control, name + ".live_has_control");
    comparison.exact(
        result.switch_deferred == expected.switch_deferred,
        name + ".live_switch_deferred");
    comparison.exact(
        result.target_switched == expected.target_switched,
        name + ".live_target_switched");
    comparison.exact(
        result.selected_target_index == expected.selected_target_index,
        name + ".live_selected_target_index");
    comparison.exact(
        static_cast<int>(result.control.aiming_state) == expected.control_aiming_state,
        name + ".live_control_aiming_state");
    comparison.exact(
        static_cast<int>(result.control.shot_mode) == expected.control_shot_mode,
        name + ".live_control_shot_mode");
    comparison.exact(
        static_cast<int>(result.control.shot_buff_mode) == expected.control_shot_buff_mode,
        name + ".live_control_shot_buff_mode");
    comparison.near(
        result.control.gimbal_yaw, expected.control_gimbal_yaw_deg,
        kControlToleranceDeg, name + ".live_control_gimbal_yaw");
    comparison.near(
        result.control.gimbal_pitch, expected.control_gimbal_pitch_deg,
        kControlToleranceDeg, name + ".live_control_gimbal_pitch");
    comparison.near(
        result.control.yaw_error, expected.control_yaw_error_deg,
        kControlToleranceDeg, name + ".live_control_yaw_error");

    std::size_t solved_blades = 0;
    bool current_target_control = false;
    if (result.rune.has_value()) {
        const auto& rune = *result.rune;
        comparison.exact(
            static_cast<int>(rune.type) == expected.rune_type,
            name + ".live_rune_type");
        solved_blades = static_cast<std::size_t>(std::count_if(
            rune.fanblades.begin(), rune.fanblades.end(),
            [](const auto_buff::FanBlade& blade) { return blade.solved; }));
        comparePoint(comparison, rune.r_center, expected.r_center, kPixelTolerance, name + ".live_r_center");
        compareEigen3(comparison, rune.xyz_in_world, expected.rune_xyz, kPoseTolerance, name + ".live_rune_xyz");
        compareEigen3(comparison, rune.ypd_in_world, expected.rune_ypd, kPoseTolerance, name + ".live_rune_ypd");
        compareEigen3(comparison, rune.ypr_in_world, expected.rune_ypr, kPoseTolerance, name + ".live_rune_ypr");
        compareEigen3(comparison, rune.blade_xyz_in_world, expected.blade_xyz, kPoseTolerance, name + ".live_blade_xyz");
        compareEigen3(comparison, rune.blade_ypd_in_world, expected.blade_ypd, kPoseTolerance, name + ".live_blade_ypd");
        if (!rune.fanblades.empty()) {
            const auto& target = rune.target();
            comparison.exact(target.solved == expected.target_solved, name + ".live_target_solved");
            comparison.exact(target.pnp_method == expected.target_pnp_method, name + ".live_target_pnp_method");
            comparison.exact(target.pnp_order == expected.target_pnp_order, name + ".live_target_pnp_order");
            comparison.near(
                target.pnp_reproj_error_px, expected.target_pnp_reproj_error_px,
                kPixelTolerance, name + ".live_target_pnp_reproj_error");
            comparison.near(
                target.pnp_score, expected.target_pnp_score,
                kPoseTolerance, name + ".live_target_pnp_score");
            comparison.near(
                target.pnp_model_center_error_px,
                expected.target_pnp_model_center_error_px,
                kPixelTolerance, name + ".live_target_pnp_model_center_error");
        }
        current_target_control =
            solved_blades > 0 && result.has_control &&
            result.control.aiming_state == rm::ControlData::AIMING_STATE::TARGET_DETECTED;
    }
    comparison.exact(solved_blades == expected.solved_blades, name + ".live_solved_blades");
    comparison.exact(
        current_target_control == expected.current_target_control,
        name + ".live_current_target_control");
}

int runManifestReplay(const std::string& config_path, const std::string& manifest_path)
{
    const auto manifest = loadGoldenManifest(manifest_path);
    if (!manifest.has_value()) return 1;
    const char* active_param_yaml = std::getenv("AIM_SIM_PARAM_YAML");
    if (active_param_yaml == nullptr || active_param_yaml[0] == '\0') {
        std::cerr
            << "Manifest replay requires explicit AIM_SIM_PARAM_YAML before pipeline "
               "construction; refusing silent repository calibration fallback\n";
        return 2;
    }
    if (manifest->runtime_param_yaml.has_value()) {
        namespace fs = std::filesystem;
        std::error_code expected_error;
        std::error_code active_error;
        const fs::path expected = fs::weakly_canonical(
            fs::path(*manifest->runtime_param_yaml), expected_error);
        const fs::path active = fs::weakly_canonical(
            fs::path(active_param_yaml), active_error);
        if (expected_error || active_error || expected != active) {
            std::cerr << "Manifest calibration provenance mismatch: captured="
                      << *manifest->runtime_param_yaml << " active=" << active_param_yaml
                      << '\n';
            return 2;
        }
        if (!manifest->runtime_param_yaml_fnv1a64.has_value()) {
            std::cerr << "Manifest calibration provenance is missing its content hash\n";
            return 2;
        }
        const auto active_hash = fileFnv1a64(active);
        if (!active_hash.has_value() ||
            *active_hash != *manifest->runtime_param_yaml_fnv1a64) {
            std::cerr << "Manifest calibration content hash mismatch: captured="
                      << *manifest->runtime_param_yaml_fnv1a64 << " active="
                      << (active_hash.has_value() ? *active_hash : std::string("unreadable"))
                      << '\n';
            return 2;
        }
    } else {
        std::cerr
            << "Warning: v1 manifest has no recorded calibration provenance; using "
               "explicit AIM_SIM_PARAM_YAML="
            << active_param_yaml << '\n';
    }
    const auto debug = runManifestSequential(config_path, *manifest, true);
    if (!debug.has_value()) return 1;
    const auto headless = runManifestSequential(config_path, *manifest, false);
    if (!headless.has_value()) return 1;

    Comparison comparison;
    comparison.exact(debug->size() == manifest->frames.size(), "debug manifest result count");
    comparison.exact(headless->size() == manifest->frames.size(), "headless manifest result count");
    std::size_t rune_frames = 0;
    std::size_t solved_pnp_frames = 0;
    std::size_t current_target_control_frames = 0;
    const std::size_t count = std::min(
        manifest->frames.size(), std::min(debug->size(), headless->size()));
    for (std::size_t index = 0; index < count; ++index) {
        const auto& lhs = (*debug)[index];
        const auto& rhs = (*headless)[index];
        const std::string name = "manifest_frame[" + std::to_string(index) + "]";
        comparison.exact(lhs.debug_artifacts_emitted, name + ".debug_flag");
        comparison.exact(!rhs.debug_artifacts_emitted, name + ".headless_flag");
        comparison.exact(rhs.overlay.empty(), name + ".headless_overlay_empty");
        comparison.exact(rhs.debug_artifact_ms == 0.0, name + ".headless_debug_ms_zero");
        comparison.exact(lhs.has_control == rhs.has_control, name + ".has_control");
        comparison.exact(lhs.completion_sequence == rhs.completion_sequence, name + ".completion_sequence");
        comparison.exact(
            lhs.frame.source_producer_epoch == rhs.frame.source_producer_epoch &&
                lhs.frame.source_image_seq == rhs.frame.source_image_seq &&
                lhs.frame.source_capture_timestamp_ns == rhs.frame.source_capture_timestamp_ns,
            name + ".source_identity");
        comparison.exact(lhs.switch_deferred == rhs.switch_deferred, name + ".switch_deferred");
        comparison.exact(lhs.target_switched == rhs.target_switched, name + ".target_switched");
        comparison.exact(
            lhs.selected_target_index == rhs.selected_target_index,
            name + ".selected_target_index");
        compareRune(comparison, lhs.rune, rhs.rune, name + ".rune");
        compareTracker(comparison, lhs.tracker_debug, rhs.tracker_debug, name + ".tracker");
        compareShotGate(comparison, lhs.shot_gate, rhs.shot_gate, name + ".shot_gate");
        compareControl(comparison, lhs.control, rhs.control, name + ".control");
        compareManifestExpected(comparison, rhs, manifest->frames[index], name);

        if (rhs.rune.has_value()) {
            ++rune_frames;
            if (std::any_of(
                    rhs.rune->fanblades.begin(), rhs.rune->fanblades.end(),
                    [](const auto_buff::FanBlade& blade) { return blade.solved; })) {
                ++solved_pnp_frames;
            }
        }
        if (rhs.rune.has_value() && rhs.has_control &&
            rhs.control.aiming_state == rm::ControlData::AIMING_STATE::TARGET_DETECTED) {
            ++current_target_control_frames;
        }
    }
    comparison.exact(rune_frames > 0, "manifest replay produced at least one rune");
    comparison.exact(solved_pnp_frames > 0, "manifest replay produced at least one solved PnP");
    comparison.exact(
        current_target_control_frames > 0,
        "manifest replay produced at least one current target control");
    if (count > manifest->trigger_index) {
        comparison.exact(
            (*headless)[manifest->trigger_index].rune.has_value() &&
                (*headless)[manifest->trigger_index].has_control &&
                (*headless)[manifest->trigger_index].control.aiming_state ==
                    rm::ControlData::AIMING_STATE::TARGET_DETECTED,
            "manifest trigger replayed as current valid control");
        if (const char* trace = std::getenv("AIM_BUFF_MANIFEST_TRACE");
            trace != nullptr && std::string(trace) == "1") {
            const auto& result = (*headless)[manifest->trigger_index];
            std::cout << std::setprecision(12)
                      << "manifest_trace has_control=" << result.has_control
                      << " aiming_state=" << static_cast<int>(result.control.aiming_state)
                      << " shot_mode=" << static_cast<int>(result.control.shot_mode)
                      << " shot_buff_mode=" << static_cast<int>(result.control.shot_buff_mode)
                      << " control_yaw=" << result.control.gimbal_yaw
                      << " control_pitch=" << result.control.gimbal_pitch
                      << " control_yaw_error=" << result.control.yaw_error
                      << " shot_requested=" << result.shot_gate.requested
                      << " shot_allowed=" << result.shot_gate.allowed
                      << " shot_reason=" << result.shot_gate.reason_code
                      << " shot_stable_frames=" << result.shot_gate.stable_frames
                      << " shot_pending=" << result.shot_gate.pending_detected
                      << " shot_r_center_ok=" << result.shot_gate.r_center_ok
                      << " shot_pnp_ok=" << result.shot_gate.pnp_ok
                      << " shot_tracker_ok=" << result.shot_gate.tracker_ok
                      << " shot_gimbal_ok=" << result.shot_gate.gimbal_ok
                      << " shot_stable_ok=" << result.shot_gate.stable_ok
                      << " shot_yaw_error=" << result.shot_gate.yaw_error_deg
                      << " shot_pitch_error=" << result.shot_gate.pitch_error_deg
                      << " shot_pnp_error=" << result.shot_gate.pnp_reproj_error_px
                      << " shot_center_error=" << result.shot_gate.pnp_model_center_error_px;
            if (result.rune.has_value()) {
                const auto& rune = *result.rune;
                std::cout << " r_center=" << rune.r_center.x << ',' << rune.r_center.y
                          << " rune_xyz=" << rune.xyz_in_world[0] << ','
                          << rune.xyz_in_world[1] << ',' << rune.xyz_in_world[2]
                          << " rune_ypd=" << rune.ypd_in_world[0] << ','
                          << rune.ypd_in_world[1] << ',' << rune.ypd_in_world[2]
                          << " rune_ypr=" << rune.ypr_in_world[0] << ','
                          << rune.ypr_in_world[1] << ',' << rune.ypr_in_world[2]
                          << " blade_xyz=" << rune.blade_xyz_in_world[0] << ','
                          << rune.blade_xyz_in_world[1] << ',' << rune.blade_xyz_in_world[2]
                          << " blade_ypd=" << rune.blade_ypd_in_world[0] << ','
                          << rune.blade_ypd_in_world[1] << ',' << rune.blade_ypd_in_world[2];
                if (!rune.fanblades.empty()) {
                    const auto& target = rune.target();
                    std::cout << " target_solved=" << target.solved
                              << " pnp_method=" << target.pnp_method
                              << " pnp_order=" << target.pnp_order[0] << ','
                              << target.pnp_order[1] << ',' << target.pnp_order[2] << ','
                              << target.pnp_order[3]
                              << " pnp_reproj=" << target.pnp_reproj_error_px
                              << " pnp_score=" << target.pnp_score
                              << " pnp_center_error=" << target.pnp_model_center_error_px;
                }
            }
            std::cout << '\n';
        }
    }

    std::cout << std::fixed << std::setprecision(6)
              << "manifest_replay_frames=" << count
              << " failures=" << comparison.failures
              << " rune_frames=" << rune_frames
              << " solved_pnp_frames=" << solved_pnp_frames
              << " current_target_control_frames=" << current_target_control_frames
              << " max_control_delta_deg=" << comparison.max_control_delta_deg
              << " max_pose_delta=" << comparison.max_pose_delta << '\n';
    const bool observation_profile_requested =
        auto_buff::buffObservationSupersetEnabledFromValue(
            std::getenv("AIM_BUFF_REFINE_PROFILE"));
    const bool observation_profile_ready =
        !headless->empty() &&
        headless->back().completion_counters.observation_superset_ready;
    if (observation_profile_requested && !observation_profile_ready) {
        const std::uint64_t fallbacks = headless->empty()
            ? 0
            : headless->back().completion_counters.observation_proposal_fallbacks;
        std::cerr
            << "Observation-superset replay used gated scaffolding only: ready=false, "
               "fallbacks="
            << fallbacks
            << "; semantic fallback equality is not profile acceptance\n";
        return comparison.failures == 0 ? 3 : 1;
    }
    return comparison.failures == 0 ? 0 : 1;
}

int runOrderedBoundary(
    const std::string& config_path,
    const cv::Mat& image,
    int frame_count)
{
    try {
        if (!auto_buff::buffOrderedCommitEnabledFromValue(
                std::getenv("AIM_BUFF_ORDERED_COMMIT"))) {
            std::cerr << "ordered-boundary requires AIM_BUFF_ORDERED_COMMIT=1\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "Invalid ordered-commit selector: " << error.what() << '\n';
        return 2;
    }

    auto_buff::BuffRunePipeline pipeline(
        config_path, auto_buff::BuffRunePipelineOptions{false, true});
    bool identity_error = false;
    std::uint64_t previous_epoch = 0;
    std::uint64_t previous_source = 0;
    std::uint64_t previous_completion = 0;
    for (int index = 0; index < frame_count; ++index) {
        const std::uint64_t source = static_cast<std::uint64_t>(index + 1);
        pipeline.push(makeFrame(
            image, source, 40'000.0 + index * 20.0, false));
        auto_buff::BuffRuneResult result;
        if (!waitForResult(pipeline, &result, 5s)) {
            std::cerr << "Ordered-boundary timeout source=" << source << '\n';
            return 1;
        }
        if (result.frame.source_producer_epoch != 0xE12B16u ||
            result.frame.source_image_seq != source ||
            result.completion_sequence != source ||
            (previous_completion != 0 &&
             result.completion_sequence <= previous_completion) ||
            (previous_epoch != 0 &&
             (result.frame.source_producer_epoch < previous_epoch ||
              (result.frame.source_producer_epoch == previous_epoch &&
               result.frame.source_image_seq <= previous_source)))) {
            identity_error = true;
        }
        previous_epoch = result.frame.source_producer_epoch;
        previous_source = result.frame.source_image_seq;
        previous_completion = result.completion_sequence;
    }

    const auto counters = pipeline.counters();
    const auto samples = pipeline.completionSamples();
    bool sample_order_error = samples.size() != static_cast<std::size_t>(frame_count);
    std::uint64_t previous_sample = 0;
    for (const auto& sample : samples) {
        if (sample.completion_sequence == 0 ||
            (previous_sample != 0 && sample.completion_sequence <= previous_sample)) {
            sample_order_error = true;
        }
        previous_sample = sample.completion_sequence;
    }
    const bool exact_boundary =
        counters.ordered_commit_inline && !counters.track_thread_started &&
        counters.solve_completed == static_cast<std::uint64_t>(frame_count) &&
        counters.solve_completed == counters.essential_completed &&
        counters.solve_completed == counters.published_results &&
        counters.detection_queue_overwrites == 0 &&
        counters.ordered_commit_failures == 0 &&
        counters.solve_completed ==
            counters.published_results + counters.ordered_commit_failures;
    const bool pass = exact_boundary && !identity_error && !sample_order_error;
    std::cout << "ordered_boundary_frames=" << frame_count
              << " solve_completed=" << counters.solve_completed
              << " essential_completed=" << counters.essential_completed
              << " published_results=" << counters.published_results
              << " detection_queue_overwrites=" << counters.detection_queue_overwrites
              << " ordered_commit_failures=" << counters.ordered_commit_failures
              << " ordered_commit_inline="
              << (counters.ordered_commit_inline ? "true" : "false")
              << " track_thread_started="
              << (counters.track_thread_started ? "true" : "false")
              << " completion_samples=" << samples.size()
              << " identity_error=" << (identity_error ? "true" : "false")
              << " sample_order_error=" << (sample_order_error ? "true" : "false")
              << " pass=" << (pass ? "true" : "false") << '\n';
    if (!pass) return 1;

    pipeline.reset();
    const int reset_frames = std::min(frame_count, 3);
    bool reset_identity_error = false;
    for (int index = 0; index < reset_frames; ++index) {
        const std::uint64_t source = static_cast<std::uint64_t>(10'001 + index);
        pipeline.push(makeFrame(
            image, source, 50'000.0 + index * 20.0, false));
        auto_buff::BuffRuneResult result;
        if (!waitForResult(pipeline, &result, 5s)) return 1;
        if (result.frame.source_image_seq != source ||
            result.completion_sequence != static_cast<std::uint64_t>(index + 1)) {
            reset_identity_error = true;
        }
    }
    const auto reset_counters = pipeline.counters();
    const auto reset_samples = pipeline.completionSamples();
    const bool solver_prior_reset =
        !reset_samples.empty() &&
        !reset_samples.front().solve_cost.detector.pnp.prior_pose_valid;
    const bool reset_pass =
        !reset_identity_error && solver_prior_reset &&
        reset_counters.solve_completed == static_cast<std::uint64_t>(reset_frames) &&
        reset_counters.solve_completed == reset_counters.essential_completed &&
        reset_counters.solve_completed == reset_counters.published_results &&
        reset_counters.detection_queue_overwrites == 0 &&
        reset_counters.ordered_commit_failures == 0;
    std::cout << "ordered_boundary_reset_frames=" << reset_frames
              << " solve_completed=" << reset_counters.solve_completed
              << " essential_completed=" << reset_counters.essential_completed
              << " published_results=" << reset_counters.published_results
              << " solver_prior_reset=" << (solver_prior_reset ? "true" : "false")
              << " identity_error=" << (reset_identity_error ? "true" : "false")
              << " pass=" << (reset_pass ? "true" : "false") << '\n';
    return reset_pass ? 0 : 1;
}

struct StageSample
{
    std::uint64_t sequence = 0;
    double yolo_ms = 0.0;
    double solve_ms = 0.0;
    double essential_ms = 0.0;
    double debug_ms = 0.0;
};

double percentile(std::vector<double> values, double quantile)
{
    values.erase(
        std::remove_if(values.begin(), values.end(), [](double value) {
            return !std::isfinite(value) || value < 0.0;
        }),
        values.end());
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(std::clamp(quantile, 0.0, 1.0) * static_cast<double>(values.size())));
    const std::size_t index = rank == 0 ? 0 : rank - 1;
    return values[std::min(index, values.size() - 1)];
}

double percentile95(std::vector<double> values)
{
    return percentile(std::move(values), 0.95);
}

double mean(const std::vector<double>& values)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.0;
    std::size_t count = 0;
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0) continue;
        sum += value;
        ++count;
    }
    return count == 0
        ? std::numeric_limits<double>::quiet_NaN()
        : sum / static_cast<double>(count);
}

double nsToMs(std::uint64_t value)
{
    return static_cast<double>(value) * 1e-6;
}

void drainResult(
    auto_buff::BuffRunePipeline& pipeline,
    std::vector<StageSample>* samples,
    std::uint64_t* last_sequence,
    bool* sequence_error)
{
    auto_buff::BuffRuneResult result;
    while (pipeline.tryPopLatest(&result)) {
        if (*last_sequence != 0 && result.completion_sequence <= *last_sequence) {
            *sequence_error = true;
        }
        *last_sequence = result.completion_sequence;
        samples->push_back(StageSample{
            result.completion_sequence,
            result.yolo_ms,
            result.solve_ms,
            result.essential_track_aim_ms,
            result.debug_artifact_ms});
    }
}

int runBenchmark(
    const std::string& config_path,
    const cv::Mat& image,
    double duration_s,
    double input_hz,
    double gimbal_yaw_deg,
    double gimbal_pitch_deg)
{
    const char* disable_template_cache =
        std::getenv("AIM_BUFF_DISABLE_R_TEMPLATE_CACHE");
    const bool template_cache_enabled =
        disable_template_cache == nullptr || std::string(disable_template_cache) != "1";
    auto_buff::BuffRunePipeline pipeline(
        config_path, auto_buff::BuffRunePipelineOptions{false, true});

    // Warm the engine and ordered state path without contaminating measurement
    // counters.  Waiting per frame also verifies the basic result path first.
    for (int index = 0; index < 12; ++index) {
        pipeline.push(makeFrame(
            image, static_cast<std::uint64_t>(index + 1), 20'000.0 + index * 20.0,
            false, false, gimbal_yaw_deg, gimbal_pitch_deg));
        auto_buff::BuffRuneResult warm_result;
        if (!waitForResult(pipeline, &warm_result, 3s)) {
            std::cerr << "Warmup timed out at frame " << index << '\n';
            return 1;
        }
    }
    pipeline.reset();
    std::this_thread::sleep_for(100ms);

    std::vector<StageSample> samples;
    samples.reserve(static_cast<std::size_t>(duration_s * 260.0));
    std::uint64_t last_sequence = 0;
    bool sequence_error = false;
    std::uint64_t source_sequence = 1;

    std::atomic<bool> stop_consumer{false};
    std::thread consumer([&] {
        while (!stop_consumer.load(std::memory_order_relaxed)) {
            const std::size_t size_before = samples.size();
            drainResult(pipeline, &samples, &last_sequence, &sequence_error);
            if (samples.size() == size_before) {
                std::this_thread::sleep_for(50us);
            }
        }
        drainResult(pipeline, &samples, &last_sequence, &sequence_error);
    });

    const auto start = Clock::now();
    const auto end = start + std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(duration_s));
    const auto input_period = std::chrono::duration<double>(1.0 / input_hz);
    auto next_input = start;

    while (Clock::now() < end) {
        auto now = Clock::now();
        while (now >= next_input && next_input < end) {
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(next_input - start).count();
            pipeline.push(makeFrame(
                image, source_sequence, 30'000.0 + elapsed_ms, false, false,
                gimbal_yaw_deg, gimbal_pitch_deg));
            ++source_sequence;
            next_input += std::chrono::duration_cast<Clock::duration>(input_period);
            now = Clock::now();
        }
        const auto wake = std::min(next_input, end);
        if (wake > Clock::now()) std::this_thread::sleep_until(wake);
    }

    const auto measured = pipeline.counters();
    const std::uint64_t measured_completion_sequence = measured.essential_completed;

    // Let the producer publish any result whose essential boundary was inside
    // the measurement window.  The dedicated consumer minimizes mailbox loss;
    // later completions are filtered by completion_sequence below.
    const auto drain_deadline = Clock::now() + 2s;
    while (Clock::now() < drain_deadline) {
        const auto current = pipeline.counters();
        if (current.published_results >= measured_completion_sequence) {
            break;
        }
        std::this_thread::sleep_for(100us);
    }
    std::this_thread::sleep_for(100ms);
    stop_consumer.store(true, std::memory_order_relaxed);
    consumer.join();

    const auto scheduler_drain_deadline = Clock::now() + 2s;
    while (Clock::now() < scheduler_drain_deadline) {
        if (auto_buff::buffProposalDrainAccepted(pipeline.counters())) break;
        std::this_thread::sleep_for(100us);
    }
    const auto completion_samples = pipeline.completionSamples();
    const auto final_counters = pipeline.counters();

    std::vector<double> yolo;
    std::vector<double> solve;
    std::vector<double> essential;
    std::vector<double> solve_outer;
    std::vector<double> observation_proposal_total;
    std::vector<double> observation_candidate_scan;
    std::vector<double> observation_fallback_solve;
    std::vector<double> observation_ordered_commit;
    std::vector<double> packet_setup;
    std::vector<double> mutex_wait;
    std::vector<double> set_image_size;
    std::vector<double> set_pose;
    std::vector<double> detector_wall;
    std::vector<double> detector_internal;
    std::vector<double> solve_unaccounted;
    std::vector<double> r_total;
    std::vector<double> r_prior;
    std::vector<double> r_roi;
    std::vector<double> r_preprocess;
    std::vector<double> r_mask;
    std::vector<double> r_template;
    std::vector<double> r_find_contours;
    std::vector<double> r_contour_filter;
    std::vector<double> r_debug;
    std::vector<double> pnp_total;
    std::vector<double> pnp_fast_solve;
    std::vector<double> pnp_fast_score;
    std::vector<double> pnp_fallback_solve;
    std::vector<double> pnp_fallback_score;
    std::vector<double> pnp_debug;
    std::uint64_t cost_samples = 0;
    std::uint64_t raw_candidates = 0;
    std::uint64_t target_candidates = 0;
    std::uint64_t hit_candidates = 0;
    std::uint64_t constructed_fanblades = 0;
    std::uint64_t template_scales = 0;
    std::uint64_t template_builds = 0;
    std::uint64_t template_cache_hits = 0;
    std::uint64_t match_template_calls = 0;
    std::uint64_t template_result_pixels = 0;
    std::uint64_t contours_total = 0;
    std::uint64_t contours_accepted = 0;
    std::uint64_t pnp_fast_calls = 0;
    std::uint64_t pnp_fallback_invocations = 0;
    std::uint64_t pnp_fallback_calls = 0;
    std::uint64_t pnp_solved_samples = 0;
    std::uint64_t pnp_solved_blades = 0;
    std::uint64_t debug_copied_bytes = 0;
    std::uint64_t observation_candidates = 0;
    std::uint64_t observation_pnp_proposals = 0;
    std::uint64_t observation_union_roi_pixels = 0;
    std::uint64_t observation_template_result_pixels = 0;
    std::uint64_t observation_cap_events = 0;
    std::uint64_t observation_fallback_events = 0;
    std::uint64_t observation_scratch_allocations = 0;
    std::uint64_t observation_scratch_reuses = 0;
    std::uint64_t observation_response_cells_scanned = 0;
    std::uint64_t observation_support_rejected_cells = 0;
    std::uint64_t observation_distance_tested_cells = 0;
    std::uint64_t observation_contour_copy_bytes_avoided = 0;
    std::vector<double> observation_worker_context_setup;
    std::array<std::uint64_t, 7> early_exits{};
    std::uint64_t last_cost_sequence = 0;
    std::size_t eligible_output_samples = 0;
    for (const auto& sample : samples) {
        if (sample.sequence == 0 || sample.sequence > measured_completion_sequence) continue;
        ++eligible_output_samples;
        if (sample.debug_ms != 0.0) sequence_error = true;
    }
    std::size_t eligible_completion_samples = 0;
    for (const auto& sample : completion_samples) {
        if (sample.completion_sequence == 0 ||
            sample.completion_sequence > measured_completion_sequence) {
            continue;
        }
        ++eligible_completion_samples;
        yolo.push_back(sample.yolo_ms);
        solve.push_back(sample.solve_ms);
        essential.push_back(sample.essential_track_aim_ms);
        if (sample.debug_artifact_ms != 0.0) sequence_error = true;

        if (last_cost_sequence != 0 && sample.completion_sequence <= last_cost_sequence) {
            sequence_error = true;
        }
        last_cost_sequence = sample.completion_sequence;
        const auto& pipeline_cost = sample.solve_cost;
        const auto& observation_cost = pipeline_cost.observation_superset;
        const auto& detector_cost = pipeline_cost.detector;
        const auto& r_cost = detector_cost.r_search;
        const auto& pnp_cost = detector_cost.pnp;
        if (pipeline_cost.outer_total_ns == 0) {
            continue;
        }
        ++cost_samples;
        solve_outer.push_back(nsToMs(pipeline_cost.outer_total_ns));
        if (observation_cost.enabled) {
            observation_proposal_total.push_back(nsToMs(observation_cost.proposal_total_ns));
            observation_candidate_scan.push_back(nsToMs(observation_cost.candidate_scan_ns));
            observation_fallback_solve.push_back(nsToMs(observation_cost.fallback_solve_ns));
            observation_ordered_commit.push_back(nsToMs(observation_cost.ordered_commit_ns));
            observation_candidates += observation_cost.retained_candidate_count;
            observation_pnp_proposals += observation_cost.pnp_proposal_count;
            observation_union_roi_pixels += observation_cost.union_roi_pixels;
            observation_template_result_pixels += observation_cost.template_result_pixels;
            observation_cap_events += observation_cost.cap_events;
            observation_fallback_events += observation_cost.fallback_events;
            observation_scratch_allocations += observation_cost.scratch_allocations;
            observation_scratch_reuses += observation_cost.scratch_reuses;
            observation_response_cells_scanned += observation_cost.response_cells_scanned;
            observation_support_rejected_cells += observation_cost.support_rejected_cells;
            observation_distance_tested_cells += observation_cost.distance_tested_cells;
            observation_contour_copy_bytes_avoided +=
                observation_cost.contour_copy_bytes_avoided;
            observation_worker_context_setup.push_back(
                nsToMs(observation_cost.worker_context_setup_ns));
        }
        packet_setup.push_back(nsToMs(pipeline_cost.packet_setup_ns));
        mutex_wait.push_back(nsToMs(pipeline_cost.solve_mutex_wait_ns));
        set_image_size.push_back(nsToMs(pipeline_cost.set_image_size_ns));
        set_pose.push_back(nsToMs(pipeline_cost.set_pose_ns));
        detector_wall.push_back(nsToMs(pipeline_cost.detector_total_ns));
        detector_internal.push_back(nsToMs(detector_cost.total_ns));
        solve_unaccounted.push_back(nsToMs(pipeline_cost.unaccounted_ns));
        r_total.push_back(nsToMs(r_cost.total_ns));
        r_prior.push_back(nsToMs(r_cost.prior_ns));
        r_roi.push_back(nsToMs(r_cost.roi_setup_ns));
        r_preprocess.push_back(nsToMs(r_cost.preprocess_ns));
        r_mask.push_back(nsToMs(r_cost.circle_mask_ns));
        r_template.push_back(nsToMs(r_cost.template_scale_search_ns));
        r_find_contours.push_back(nsToMs(r_cost.find_contours_ns));
        r_contour_filter.push_back(nsToMs(r_cost.contour_filter_score_ns));
        r_debug.push_back(nsToMs(r_cost.debug_materialization_ns));
        pnp_total.push_back(nsToMs(pnp_cost.total_ns));
        pnp_fast_solve.push_back(nsToMs(pnp_cost.fast_solvepnp_ns));
        pnp_fast_score.push_back(nsToMs(pnp_cost.fast_reprojection_score_ns));
        pnp_fallback_solve.push_back(nsToMs(pnp_cost.fallback_solvepnp_ns));
        pnp_fallback_score.push_back(nsToMs(pnp_cost.fallback_reprojection_score_ns));
        pnp_debug.push_back(nsToMs(pnp_cost.debug_materialization_ns));

        raw_candidates += detector_cost.raw_candidates;
        target_candidates += detector_cost.target_candidates;
        hit_candidates += detector_cost.hit_candidates;
        constructed_fanblades += detector_cost.constructed_fanblades;
        template_scales += r_cost.template_scale_count;
        template_builds += r_cost.template_builds;
        template_cache_hits += r_cost.template_cache_hits;
        match_template_calls += r_cost.match_template_calls;
        template_result_pixels += r_cost.template_result_pixels;
        contours_total += r_cost.contours_total;
        contours_accepted += r_cost.contours_accepted;
        debug_copied_bytes += r_cost.debug_copied_bytes;
        pnp_fast_calls += pnp_cost.fast_pnp_calls;
        pnp_fallback_invocations += pnp_cost.fallback_invocations;
        pnp_fallback_calls += pnp_cost.fallback_pnp_calls;
        pnp_solved_samples += pnp_cost.solved ? 1U : 0U;
        pnp_solved_blades += pnp_cost.solved_blades;
        const auto early_exit_index = static_cast<std::size_t>(detector_cost.early_exit);
        if (early_exit_index < early_exits.size()) {
            ++early_exits[early_exit_index];
        }
    }

    const double completion_hz =
        static_cast<double>(measured.essential_completed) / duration_s;
    const double yolo_p95 = percentile95(std::move(yolo));
    const double solve_p95 = percentile95(std::move(solve));
    const double essential_p95 = percentile95(std::move(essential));
    const double slowest_p95 = std::max(yolo_p95, std::max(solve_p95, essential_p95));
    const double sample_coverage = measured.essential_completed == 0
        ? 0.0
        : static_cast<double>(eligible_completion_samples) /
              static_cast<double>(measured.essential_completed);
    const double cost_sample_coverage = measured.essential_completed == 0
        ? 0.0
        : static_cast<double>(cost_samples) /
              static_cast<double>(measured.essential_completed);
    const bool cost_coverage_ok = cost_sample_coverage >= 0.99;
    const bool observation_profile_gate =
        !measured.observation_superset_enabled ||
        (measured.observation_superset_ready &&
         measured.observation_proposal_fallbacks == 0 &&
         measured.observation_identity_failures == 0 &&
         mean(observation_proposal_total) <= 13.636 &&
         percentile95(observation_ordered_commit) <= 2.0);
    const double proposal_accepted_ratio = final_counters.proposal_completed == 0
        ? 0.0
        : static_cast<double>(final_counters.proposal_committed) /
              static_cast<double>(final_counters.proposal_completed);
    double proposal_aggregate_service_hz = 0.0;
    for (std::size_t index = 0; index < 4; ++index) {
        if (final_counters.proposal_worker_total_ns[index] != 0) {
            proposal_aggregate_service_hz +=
                static_cast<double>(final_counters.proposal_worker_completed[index]) * 1e9 /
                static_cast<double>(final_counters.proposal_worker_total_ns[index]);
        }
    }
    const auto proposal_bound = final_counters.proposal_worker_count;
    const bool proposal_drain_gate =
        auto_buff::buffProposalDrainAccepted(final_counters);
    const bool proposal_scheduler_gate =
        final_counters.proposal_worker_count == 1 ||
        (proposal_accepted_ratio >= 0.97 &&
         proposal_drain_gate &&
         final_counters.proposal_input_max_occupancy <= proposal_bound &&
         final_counters.proposal_reorder_max_occupancy <= proposal_bound &&
         final_counters.proposal_max_inflight <= proposal_bound &&
         final_counters.proposal_terminal_failures == 0 &&
         final_counters.proposal_cancelled == 0 &&
         final_counters.proposal_stale == 0 &&
         final_counters.ordered_commit_failures == 0);
    const bool proposal_capacity_gate = auto_buff::buffProposalCapacityAccepted(
        final_counters.proposal_worker_count, proposal_aggregate_service_hz);
    const std::uint64_t inferred_output_overwrites =
        measured.essential_completed > eligible_output_samples
        ? measured.essential_completed - static_cast<std::uint64_t>(eligible_output_samples)
        : 0;
    const bool acceptance_pass =
        completion_hz >= 220.0 && slowest_p95 <= 4.0 &&
        sample_coverage >= 0.99 && cost_coverage_ok && observation_profile_gate &&
        proposal_scheduler_gate && proposal_capacity_gate &&
        !sequence_error;

    std::cout << std::fixed << std::setprecision(3)
              << "benchmark_duration_s=" << duration_s
              << " requested_input_hz=" << input_hz
              << " gimbal_yaw_deg=" << gimbal_yaw_deg
              << " gimbal_pitch_deg=" << gimbal_pitch_deg
              << " template_cache_enabled=" << (template_cache_enabled ? "true" : "false")
              << " pushed=" << measured.pushed_frames
              << " input_overwrites=" << measured.input_queue_overwrites
              << " yolo_completed=" << measured.yolo_completed
              << " yolo_overwrites=" << measured.yolo_queue_overwrites
              << " solve_completed=" << measured.solve_completed
              << " detection_overwrites=" << measured.detection_queue_overwrites
              << " essential_completed=" << measured.essential_completed
              << " completion_hz=" << completion_hz
              << " completion_samples=" << eligible_completion_samples
              << " sample_coverage=" << sample_coverage
              << " cost_samples=" << cost_samples
              << " cost_sample_coverage=" << cost_sample_coverage
              << " popped_unique=" << eligible_output_samples
              << " inferred_output_overwrites=" << inferred_output_overwrites
              << " sequence_error=" << (sequence_error ? "true" : "false")
              << " yolo_p95_ms=" << yolo_p95
              << " solve_p95_ms=" << solve_p95
              << " essential_p95_ms=" << essential_p95
              << " slowest_serial_p95_ms=" << slowest_p95
              << " observation_profile_gate="
              << (observation_profile_gate ? "true" : "false")
              << " acceptance_pass=" << (acceptance_pass ? "true" : "false")
              << '\n';

    std::cout << "proposal_scheduler"
              << " workers=" << final_counters.proposal_worker_count
              << " submitted=" << final_counters.proposal_submitted
              << " completed=" << final_counters.proposal_completed
              << " committed=" << final_counters.proposal_committed
              << " accepted_ratio=" << proposal_accepted_ratio
              << " active=" << final_counters.proposal_active_workers
              << " max_active=" << final_counters.proposal_max_active_workers
              << " input_occupancy=" << final_counters.proposal_input_occupancy
              << " input_max=" << final_counters.proposal_input_max_occupancy
              << " reorder_occupancy=" << final_counters.proposal_reorder_occupancy
              << " reorder_max=" << final_counters.proposal_reorder_max_occupancy
              << " inflight=" << final_counters.proposal_inflight
              << " max_inflight=" << final_counters.proposal_max_inflight
              << " aggregate_service_hz=" << proposal_aggregate_service_hz
              << " head_gap_events=" << final_counters.proposal_terminal_gaps
              << " head_wait_ms=" << nsToMs(final_counters.proposal_head_wait_ns)
              << " failures=" << final_counters.proposal_terminal_failures
              << " cancelled=" << final_counters.proposal_cancelled
              << " stale=" << final_counters.proposal_stale
              << " gate=" << (proposal_scheduler_gate ? "true" : "false");
    std::cout << " drain_gate=" << (proposal_drain_gate ? "true" : "false");
    std::cout << " capacity_gate=" << (proposal_capacity_gate ? "true" : "false");
    for (std::size_t index = 0; index < 4; ++index) {
        const double worker_mean_ms = final_counters.proposal_worker_completed[index] == 0
            ? 0.0
            : nsToMs(final_counters.proposal_worker_total_ns[index]) /
                  static_cast<double>(final_counters.proposal_worker_completed[index]);
        std::cout << " worker" << index << "_completed="
                  << final_counters.proposal_worker_completed[index]
                  << " worker" << index << "_mean_ms=" << worker_mean_ms;
    }
    std::cout << '\n';

    const auto p50 = [](const std::vector<double>& values) {
        return percentile(values, 0.50);
    };
    const auto p95 = [](const std::vector<double>& values) {
        return percentile(values, 0.95);
    };
    const auto p99 = [](const std::vector<double>& values) {
        return percentile(values, 0.99);
    };
    std::cout << "cost_percentiles_ms"
              << " solve_outer_mean=" << mean(solve_outer)
              << " solve_outer_p50=" << p50(solve_outer)
              << " solve_outer_p95=" << p95(solve_outer)
              << " solve_outer_p99=" << p99(solve_outer)
              << " detector_wall_p50=" << p50(detector_wall)
              << " detector_wall_mean=" << mean(detector_wall)
              << " detector_wall_p95=" << p95(detector_wall)
              << " detector_wall_p99=" << p99(detector_wall)
              << " detector_internal_p95=" << p95(detector_internal)
              << " r_total_p50=" << p50(r_total)
              << " r_total_mean=" << mean(r_total)
              << " r_total_p95=" << p95(r_total)
              << " r_total_p99=" << p99(r_total)
              << " r_template_p50=" << p50(r_template)
              << " r_template_mean=" << mean(r_template)
              << " r_template_p95=" << p95(r_template)
              << " r_template_p99=" << p99(r_template)
              << " r_find_contours_p50=" << p50(r_find_contours)
              << " r_find_contours_p95=" << p95(r_find_contours)
              << " r_find_contours_p99=" << p99(r_find_contours)
              << " pnp_total_p50=" << p50(pnp_total)
              << " pnp_total_mean=" << mean(pnp_total)
              << " pnp_total_p95=" << p95(pnp_total)
              << " pnp_total_p99=" << p99(pnp_total)
              << '\n';

    std::cout << "observation_superset_cost_ms"
              << " enabled=" << (measured.observation_superset_enabled ? "true" : "false")
              << " ready=" << (measured.observation_superset_ready ? "true" : "false")
              << " proposal_mean=" << mean(observation_proposal_total)
              << " proposal_p95=" << p95(observation_proposal_total)
              << " candidate_scan_p95=" << p95(observation_candidate_scan)
              << " fallback_solve_mean=" << mean(observation_fallback_solve)
              << " fallback_solve_p95=" << p95(observation_fallback_solve)
              << " ordered_commit_mean=" << mean(observation_ordered_commit)
              << " ordered_commit_p95=" << p95(observation_ordered_commit)
              << " candidates=" << observation_candidates
              << " pnp_proposals=" << observation_pnp_proposals
              << " union_roi_pixels=" << observation_union_roi_pixels
              << " template_result_pixels=" << observation_template_result_pixels
              << " cap_events=" << observation_cap_events
              << " fallback_events=" << observation_fallback_events
              << " identity_gaps=" << measured.observation_identity_gaps
              << " identity_failures=" << measured.observation_identity_failures
              << " scratch_allocations=" << observation_scratch_allocations
              << " scratch_reuses=" << observation_scratch_reuses
              << " response_cells_scanned=" << observation_response_cells_scanned
              << " support_rejected_cells=" << observation_support_rejected_cells
              << " distance_tested_cells=" << observation_distance_tested_cells
              << " contour_copy_bytes_avoided="
              << observation_contour_copy_bytes_avoided
              << " worker_context_setup_mean="
              << mean(observation_worker_context_setup)
              << '\n';

    std::cout << "cost_components_ms"
              << " packet_setup_p95=" << p95(packet_setup)
              << " mutex_wait_p95=" << p95(mutex_wait)
              << " set_image_size_p95=" << p95(set_image_size)
              << " set_pose_p95=" << p95(set_pose)
              << " solve_unaccounted_p95=" << p95(solve_unaccounted)
              << " r_prior_p95=" << p95(r_prior)
              << " r_roi_p95=" << p95(r_roi)
              << " r_preprocess_p95=" << p95(r_preprocess)
              << " r_mask_p95=" << p95(r_mask)
              << " r_contour_filter_p95=" << p95(r_contour_filter)
              << " r_debug_p95=" << p95(r_debug)
              << " pnp_fast_solve_p95=" << p95(pnp_fast_solve)
              << " pnp_fast_score_p95=" << p95(pnp_fast_score)
              << " pnp_fallback_solve_p95=" << p95(pnp_fallback_solve)
              << " pnp_fallback_score_p95=" << p95(pnp_fallback_score)
              << " pnp_debug_p95=" << p95(pnp_debug)
              << '\n';

    const auto per_cost_sample = [cost_samples](std::uint64_t value) {
        return cost_samples == 0
            ? std::numeric_limits<double>::quiet_NaN()
            : static_cast<double>(value) / static_cast<double>(cost_samples);
    };
    std::cout << "cost_counters"
              << " raw_candidates_avg=" << per_cost_sample(raw_candidates)
              << " target_candidates_avg=" << per_cost_sample(target_candidates)
              << " hit_candidates_avg=" << per_cost_sample(hit_candidates)
              << " fanblades_avg=" << per_cost_sample(constructed_fanblades)
              << " template_scales_avg=" << per_cost_sample(template_scales)
              << " template_builds_avg=" << per_cost_sample(template_builds)
              << " template_cache_hits_avg=" << per_cost_sample(template_cache_hits)
              << " match_template_calls_avg=" << per_cost_sample(match_template_calls)
              << " template_result_pixels_avg=" << per_cost_sample(template_result_pixels)
              << " contours_total_avg=" << per_cost_sample(contours_total)
              << " contours_accepted_avg=" << per_cost_sample(contours_accepted)
              << " r_debug_copied_bytes_avg=" << per_cost_sample(debug_copied_bytes)
              << " pnp_fast_calls_avg=" << per_cost_sample(pnp_fast_calls)
              << " pnp_fallback_invocations_avg=" << per_cost_sample(pnp_fallback_invocations)
              << " pnp_fallback_calls_avg=" << per_cost_sample(pnp_fallback_calls)
              << " pnp_solved_samples=" << pnp_solved_samples
              << " pnp_solved_blades=" << pnp_solved_blades
              << " early_none=" << early_exits[0]
              << " early_raw_empty=" << early_exits[1]
              << " early_no_rune=" << early_exits[2]
              << " early_switch_deferred=" << early_exits[3]
              << " early_selection_failed=" << early_exits[4]
              << " early_no_fanblades=" << early_exits[5]
              << " early_pnp_unsolved=" << early_exits[6]
              << '\n';

    // Missing/duplicate identity or an empty measurement is a harness failure.
    // Missing the performance target is reported as acceptance_pass=false but
    // remains a successful, reproducible measurement.
    return measured.essential_completed > 0 && !sequence_error && cost_coverage_ok ? 0 : 1;
}

auto_buff::BuffRuneResult makeCaptureTestResult(
    const cv::Mat& image,
    std::uint64_t source_sequence,
    std::uint64_t completion_sequence,
    bool current_valid = false)
{
    auto_buff::BuffRuneResult result;
    result.frame.srcImg = image;
    result.frame.source_producer_epoch = 1234;
    result.frame.source_image_seq = source_sequence;
    result.frame.source_capture_timestamp_ns = 1'000'000 + source_sequence;
    result.frame.timeStamp = static_cast<double>(source_sequence);
    result.frame.usb_timeStamp = static_cast<double>(source_sequence);
    result.frame.bullet_speed = 25.0;
    result.frame.fb.task_mode = rm::FeedBackData::TASK_MODE::HIT_SMALL_BUFF;
    result.completion_sequence = completion_sequence;
    result.completion_timestamp_ns = 2'000'000 + completion_sequence;
    if (current_valid) {
        auto_buff::PowerRune rune;
        rune.type = auto_buff::SMALL;
        rune.r_center = cv::Point2f(1.0f, 1.0f);
        auto_buff::FanBlade target;
        target.type = auto_buff::_target;
        target.solved = true;
        target.pnp_order = {{0, 1, 2, 3}};
        rune.fanblades.push_back(target);
        result.rune = rune;
        result.has_control = true;
        result.control.aiming_state = rm::ControlData::AIMING_STATE::TARGET_DETECTED;
    }
    return result;
}

int runCaptureSelfTest(const std::string& output_root)
{
    namespace fs = std::filesystem;
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: capture self-test " << message << '\n';
        }
    };
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::absolute(fs::path(output_root)) /
        ("buff_exact_valid_selftest_" + std::to_string(nonce));
    const cv::Mat image(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));

    try {
        check(!auto_buff::buffOrderedCommitEnabledFromValue(nullptr),
              "missing ordered selector selects legacy");
        check(!auto_buff::buffOrderedCommitEnabledFromValue(""),
              "empty ordered selector selects legacy");
        check(!auto_buff::buffOrderedCommitEnabledFromValue("0"),
              "exact zero ordered selector selects legacy");
        check(auto_buff::buffOrderedCommitEnabledFromValue("1"),
              "exact one ordered selector selects inline");
        bool invalid_selector_threw = false;
        try {
            (void)auto_buff::buffOrderedCommitEnabledFromValue("true");
        } catch (const std::invalid_argument&) {
            invalid_selector_threw = true;
        }
        check(invalid_selector_threw, "invalid ordered selector hard-failed");
        check(!auto_buff::buffObservationSupersetEnabledFromValue(nullptr),
              "missing refine selector selects legacy");
        check(!auto_buff::buffObservationSupersetEnabledFromValue(""),
              "empty refine selector selects legacy");
        check(!auto_buff::buffObservationSupersetEnabledFromValue("legacy"),
              "legacy refine selector selects legacy");
        check(auto_buff::buffObservationSupersetEnabledFromValue("observation_superset"),
              "observation-superset selector is accepted");
        bool invalid_refine_selector_threw = false;
        try {
            (void)auto_buff::buffObservationSupersetEnabledFromValue("superset");
        } catch (const std::invalid_argument&) {
            invalid_refine_selector_threw = true;
        }
        check(invalid_refine_selector_threw, "invalid refine selector hard-failed");
        {
            auto_buff::BuffExactValidSequenceCapture disabled("", 2, 24);
            auto result = makeCaptureTestResult(image, 1, 1);
            const int before = image.u->refcount;
            disabled.observeCompletion(result);
            const auto diagnostics = disabled.diagnostics();
            check(!diagnostics.armed, "disabled capture is not armed");
            check(diagnostics.retained_frames == 0, "disabled capture retained no frames");
            check(image.u->refcount == before, "disabled capture added no shallow reference");
        }
        {
            auto_buff::BuffExactValidSequenceCapture bounded(
                (root / "bounded").string(), 2, 24);
            auto first = makeCaptureTestResult(image, 1, 1);
            const int before = image.u->refcount;
            bounded.observeCompletion(first);
            check(image.u->refcount == before + 1, "armed capture shallow-retained exactly once");
            bounded.observeCompletion(makeCaptureTestResult(image, 2, 2));
            bounded.observeCompletion(makeCaptureTestResult(image, 3, 3));
            const auto diagnostics = bounded.diagnostics();
            const auto identities = bounded.retainedIdentities();
            check(diagnostics.retained_frames == 2, "frame bound retained two frames");
            check(diagnostics.retained_bytes == 24, "byte accounting retained 24 bytes");
            check(diagnostics.evicted_frames == 1 && diagnostics.ring_truncated,
                  "frame bound explicitly recorded eviction/truncation");
            check(identities.size() == 2 && identities[0].image_sequence == 2 &&
                      identities[1].image_sequence == 3,
                  "frame bound evicted oldest in commit order");
        }
        {
            auto_buff::BuffExactValidSequenceCapture byte_bounded(
                (root / "byte_bounded").string(), 4, 18);
            byte_bounded.observeCompletion(makeCaptureTestResult(image, 1, 1));
            byte_bounded.observeCompletion(makeCaptureTestResult(image, 2, 2));
            const auto diagnostics = byte_bounded.diagnostics();
            check(diagnostics.retained_frames == 1 && diagnostics.retained_bytes == 12,
                  "byte bound evicted before exceeding configured cap");
            check(diagnostics.evicted_frames == 1 && diagnostics.ring_truncated,
                  "byte-bound eviction was explicit");
        }
        {
            auto_buff::BuffExactValidSequenceCapture unretainable(
                (root / "unretainable").string(), 4, 8);
            unretainable.observeCompletion(makeCaptureTestResult(image, 1, 1));
            const auto diagnostics = unretainable.diagnostics();
            check(diagnostics.retained_frames == 0 && diagnostics.unretainable_frames == 1,
                  "oversize frame was rejected");
            check(diagnostics.ring_truncated, "oversize rejection marked truncation");
        }
        {
            const fs::path invalid_dir = root / "invalid_order";
            auto_buff::BuffExactValidSequenceCapture invalid(invalid_dir.string(), 8, 256);
            invalid.observeCompletion(makeCaptureTestResult(image, 1, 1));
            invalid.observeCompletion(makeCaptureTestResult(image, 1, 2));
            invalid.observeCompletion(makeCaptureTestResult(image, 2, 3, true));
            const auto diagnostics = invalid.diagnostics();
            check(!diagnostics.order_valid && diagnostics.duplicate_identity_rejects == 1,
                  "duplicate identity invalidated sequence");
            check(!diagnostics.complete, "invalid sequence did not complete");
            check(!fs::exists(invalid_dir), "invalid armed sequence wrote nothing");
        }
        {
            const fs::path valid_dir = root / "valid";
            auto_buff::BuffExactValidSequenceCapture valid(valid_dir.string(), 8, 256);
            valid.observeCompletion(makeCaptureTestResult(image, 1, 1));
            valid.observeCompletion(makeCaptureTestResult(image, 2, 2, true));
            const auto diagnostics = valid.diagnostics();
            check(diagnostics.complete, "valid current completion flushed manifest");
            std::vector<fs::path> manifests;
            if (fs::exists(valid_dir)) {
                for (const auto& entry : fs::directory_iterator(valid_dir)) {
                    if (entry.path().extension() == ".json") manifests.push_back(entry.path());
                }
            }
            check(manifests.size() == 1, "valid capture wrote exactly one manifest");
            if (manifests.size() == 1) {
                const auto parsed = loadGoldenManifest(manifests.front().string());
                check(parsed.has_value() && parsed->frames.size() == 2,
                      "written manifest passed strict parser");
                if (const char* param_yaml = std::getenv("AIM_SIM_PARAM_YAML");
                    param_yaml != nullptr && param_yaml[0] != '\0' && parsed.has_value()) {
                    check(parsed->runtime_param_yaml == std::optional<std::string>(param_yaml),
                          "manifest recorded runtime calibration path");
                    check(parsed->runtime_param_yaml_fnv1a64 ==
                              fileFnv1a64(std::filesystem::path(param_yaml)),
                          "manifest recorded runtime calibration content hash");
                }
            }
        }
        {
            const fs::path truncated_dir = root / "truncated";
            auto_buff::BuffExactValidSequenceCapture truncated(
                truncated_dir.string(), 1, 256);
            truncated.observeCompletion(makeCaptureTestResult(image, 1, 1));
            truncated.observeCompletion(makeCaptureTestResult(image, 2, 2, true));
            std::vector<fs::path> manifests;
            if (fs::exists(truncated_dir)) {
                for (const auto& entry : fs::directory_iterator(truncated_dir)) {
                    if (entry.path().extension() == ".json") manifests.push_back(entry.path());
                }
            }
            check(manifests.size() == 1, "truncated capture explicitly wrote one manifest");
            if (manifests.size() == 1) {
                check(!loadGoldenManifest(manifests.front().string()).has_value(),
                      "strict replay parser rejected truncated history");
            }
        }
        if (fs::exists(root)) fs::remove_all(root);
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL: capture self-test exception: " << error.what() << '\n';
        if (fs::exists(root)) fs::remove_all(root);
    }
    std::cout << "capture_selftest_failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}

void printUsage(const char* executable)
{
    std::cerr
        << "Usage:\n  " << executable
        << " equivalence <config.yaml> <image> [frames] [small|big] [gimbal_yaw_deg] [gimbal_pitch_deg]\n  "
        << executable
        << " benchmark <config.yaml> <image> [duration_s] [input_hz] [gimbal_yaw_deg] [gimbal_pitch_deg]\n  "
        << executable
        << " manifest-replay <config.yaml> <manifest.json>\n  "
        << executable
        << " ordered-boundary <config.yaml> <image> [frames] (requires AIM_BUFF_ORDERED_COMMIT=1)\n  "
        << executable
        << " capture-selftest <temporary-output-root>\n";
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "capture-selftest") {
        if (argc != 3) {
            printUsage(argv[0]);
            return 2;
        }
        return runCaptureSelfTest(argv[2]);
    }
    if (mode == "manifest-replay") {
        if (argc != 4) {
            printUsage(argv[0]);
            return 2;
        }
        return runManifestReplay(argv[2], argv[3]);
    }
    if (argc < 4) {
        printUsage(argv[0]);
        return 2;
    }
    const std::string config_path = argv[2];
    const std::string image_path = argv[3];
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << '\n';
        return 2;
    }

    if (mode == "equivalence") {
        const int frames = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 10;
        const bool big_buff = argc >= 6 && std::string(argv[5]) == "big";
        const double gimbal_yaw_deg = argc >= 7 ? std::atof(argv[6]) : kDefaultGimbalYawDeg;
        const double gimbal_pitch_deg = argc >= 8 ? std::atof(argv[7]) : kDefaultGimbalPitchDeg;
        if (!std::isfinite(gimbal_yaw_deg) || !std::isfinite(gimbal_pitch_deg)) {
            printUsage(argv[0]);
            return 2;
        }
        return runEquivalence(
            config_path, image, frames, big_buff, gimbal_yaw_deg, gimbal_pitch_deg);
    }
    if (mode == "ordered-boundary") {
        const int frames = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 20;
        return runOrderedBoundary(config_path, image, frames);
    }
    if (mode == "benchmark") {
        const double duration_s = argc >= 5 ? std::atof(argv[4]) : 30.0;
        const double input_hz = argc >= 6 ? std::atof(argv[5]) : 300.0;
        const double gimbal_yaw_deg = argc >= 7 ? std::atof(argv[6]) : kDefaultGimbalYawDeg;
        const double gimbal_pitch_deg = argc >= 8 ? std::atof(argv[7]) : kDefaultGimbalPitchDeg;
        if (!std::isfinite(duration_s) || duration_s <= 0.0 ||
            !std::isfinite(input_hz) || input_hz <= 0.0 ||
            !std::isfinite(gimbal_yaw_deg) || !std::isfinite(gimbal_pitch_deg)) {
            printUsage(argv[0]);
            return 2;
        }
        return runBenchmark(
            config_path, image, duration_s, input_hz, gimbal_yaw_deg, gimbal_pitch_deg);
    }

    printUsage(argv[0]);
    return 2;
}
