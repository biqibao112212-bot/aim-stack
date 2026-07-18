#include "buff_rune_pipeline.hpp"

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
constexpr double kHudPredictionDebugDtS = 0.100;
constexpr double kPastPredictionOverlayDtS = 0.200;
constexpr double kPastPredictionOverlayMatchMs = 35.0;
constexpr double kPastPredictionOverlayRetentionMs = 350.0;
constexpr double kFixed100PredictionEvalDtS = 0.100;
constexpr double kFixed200PredictionEvalDtS = 0.200;
constexpr double kHistoricalPredictionMaxSourceLagMs = 35.0;
constexpr double kHistoricalPredictionMaxSeedSpeed = 2.35;
constexpr int kHistoricalPredictionMinHistorySize = 8;
constexpr int kCsvFlushEveryRows = 50;

std::string resolve_config_path(const std::string& config_path)
{
    namespace fs = std::filesystem;

    const fs::path raw(config_path);
    if (raw.is_absolute() && fs::exists(raw)) {
        return raw.string();
    }

    std::vector<fs::path> candidates;
    candidates.push_back(raw);
    candidates.push_back(fs::current_path() / raw);

    fs::path cursor = fs::current_path();
    while (!cursor.empty() && cursor.has_parent_path() && cursor != cursor.parent_path()) {
        candidates.push_back(cursor / raw);
        candidates.push_back(cursor / "src/BuffDetector/buff_config.yaml");
        cursor = cursor.parent_path();
    }

    for (const auto& candidate : candidates) {
        if (!candidate.empty() && fs::exists(candidate)) {
            return candidate.string();
        }
    }

    return config_path;
}

double finiteOr(double value, double fallback)
{
    return std::isfinite(value) ? value : fallback;
}

double nanValue()
{
    return std::numeric_limits<double>::quiet_NaN();
}

std::uint64_t systemNowNs()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::uint64_t costDurationNs(const std::chrono::steady_clock::time_point& begin)
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
}

double angleErrorDeg(double target_deg, double measured_deg)
{
    if (!std::isfinite(target_deg) || !std::isfinite(measured_deg)) {
        return nanValue();
    }
    return std::abs(std::remainder(target_deg - measured_deg, 360.0));
}

bool isGoodHistoricalPredictionSeed(
    const Eigen::VectorXd& state,
    int direction,
    int history_size,
    int reinit_reason,
    bool switch_deferred,
    bool target_switched)
{
    if (direction == 0 || reinit_reason != 0 || switch_deferred || target_switched) {
        return false;
    }
    if (history_size < kHistoricalPredictionMinHistorySize || state.size() < 10 ||
        !state.array().isFinite().all()) {
        return false;
    }
    const double speed = std::abs(state[6]);
    return std::isfinite(speed) && speed <= kHistoricalPredictionMaxSeedSpeed;
}

bool isFinitePoint(const cv::Point2f& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y);
}

bool isDrawablePoint(const cv::Point2f& point, const cv::Mat& image)
{
    if (!isFinitePoint(point) || image.empty()) return false;
    const float margin_x = static_cast<float>(std::max(image.cols, 1));
    const float margin_y = static_cast<float>(std::max(image.rows, 1));
    return point.x > -margin_x && point.x < image.cols + margin_x &&
           point.y > -margin_y && point.y < image.rows + margin_y;
}

cv::Rect clampRectToImage(const cv::Rect& rect, const cv::Mat& image)
{
    if (image.empty()) return {};
    const cv::Rect bounds(0, 0, image.cols, image.rows);
    return rect & bounds;
}

bool isFiniteWorldPoint(const Eigen::Vector3d& point)
{
    return point.array().isFinite().all();
}

uint32_t rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
{
    return (static_cast<uint32_t>(r) << 24) | (static_cast<uint32_t>(g) << 16) |
        (static_cast<uint32_t>(b) << 8) | static_cast<uint32_t>(a);
}

void appendOverlayBox2D(
    rm::DebugHudSnapshot& hud,
    const std::string& entity_path,
    const cv::Rect2f& rect,
    uint32_t color,
    float radius,
    std::string label = {},
    bool show_label = false)
{
    if (entity_path.empty() || !std::isfinite(rect.x) || !std::isfinite(rect.y) ||
        !std::isfinite(rect.width) || !std::isfinite(rect.height) || rect.width <= 0.0f ||
        rect.height <= 0.0f) {
        return;
    }
    hud.boxes2d.push_back({entity_path, rect, std::move(label), color, radius, show_label});
}

void appendOverlayPoint2D(
    rm::DebugHudSnapshot& hud,
    const std::string& entity_path,
    const cv::Point2f& position,
    uint32_t color,
    float radius,
    std::string label = {},
    bool show_label = false)
{
    if (entity_path.empty() || !isFinitePoint(position)) return;
    hud.points2d.push_back({entity_path, position, std::move(label), color, radius, show_label});
}

void appendOverlayLineStrip2D(
    rm::DebugHudSnapshot& hud,
    const std::string& entity_path,
    std::vector<cv::Point2f> points,
    uint32_t color,
    float radius,
    bool closed,
    std::string label = {},
    bool show_label = false)
{
    points.erase(
        std::remove_if(points.begin(), points.end(), [](const cv::Point2f& point) {
            return !isFinitePoint(point);
        }),
        points.end());
    if (entity_path.empty() || points.size() < 2) return;
    if (closed && points.front() != points.back()) {
        points.push_back(points.front());
    }
    hud.lines2d.push_back(
        {entity_path, std::move(points), std::move(label), color, radius, closed, show_label});
}

void appendOverlayPoint3D(
    rm::DebugHudSnapshot& hud,
    const std::string& entity_path,
    const Eigen::Vector3d& position,
    uint32_t color,
    float radius,
    std::string label = {},
    bool show_label = false)
{
    if (entity_path.empty() || !isFiniteWorldPoint(position)) return;
    hud.points3d.push_back({entity_path, position, std::move(label), color, radius, show_label});
}

void appendOverlayLineStrip3D(
    rm::DebugHudSnapshot& hud,
    const std::string& entity_path,
    std::vector<Eigen::Vector3d> points,
    uint32_t color,
    float radius,
    bool closed,
    std::string label = {},
    bool show_label = false)
{
    points.erase(
        std::remove_if(points.begin(), points.end(), [](const Eigen::Vector3d& point) {
            return !isFiniteWorldPoint(point);
        }),
        points.end());
    if (entity_path.empty() || points.size() < 2) return;
    if (closed && points.front() != points.back()) {
        points.push_back(points.front());
    }
    hud.lines3d.push_back(
        {entity_path, std::move(points), std::move(label), color, radius, closed, show_label});
}

const char* yoloCandidateLabel(int label)
{
    switch (label) {
        case 0: return "R-target";
        case 1: return "R-hit";
        case 2: return "B-target";
        case 3: return "B-hit";
        default: return "unknown";
    }
}

uint32_t yoloCandidateColor(int label)
{
    switch (label) {
        case 0:
        case 2:
            return rgba(0, 255, 0);
        case 1:
        case 3:
            return rgba(255, 64, 64);
        default:
            return rgba(255, 255, 255);
    }
}

struct ProjectedBladeOverlayState
{
    Eigen::Vector3d rune_center_world = Eigen::Vector3d::Constant(nanValue());
    Eigen::Vector3d blade_center_world = Eigen::Vector3d::Constant(nanValue());
    std::vector<Eigen::Vector3d> blade_outline_world;
    std::vector<Eigen::Vector3d> blade_keypoints_world;
    std::vector<cv::Point2f> projected_points;
};

constexpr int kBuffGuideRingSegments = 64;
constexpr int kBuffGuidePhaseCount = 5;

Eigen::Vector3d solverModelPointToEigen(const cv::Point3f& point)
{
    return Eigen::Vector3d(point.x, point.y, point.z);
}

Eigen::Vector3d buffLocalPolarPoint(double radius_m, double angle_rad)
{
    return Eigen::Vector3d(0.0, radius_m * std::sin(angle_rad), radius_m * std::cos(angle_rad));
}

Eigen::Vector3d transformBuffLocalPoint(
    const Eigen::Vector3d& local_point,
    const Eigen::Vector3d& rune_center_world,
    const Eigen::Vector3d& rune_ypr_world)
{
    return tools::rotation_matrix(rune_ypr_world) * local_point + rune_center_world;
}

std::array<double, 3> buffGuideRadii(const auto_buff::Solver& solver)
{
    const auto& object_points = solver.get_object_points();
    if (object_points.size() < 5) {
        return {0.57, 0.70, 0.83};
    }
    return {
        std::hypot(object_points[2].y, object_points[2].z),
        std::hypot(object_points[4].y, object_points[4].z),
        std::hypot(object_points[0].y, object_points[0].z)};
}

std::vector<Eigen::Vector3d> buildBuffGuideRingWorld(
    const Eigen::Vector3d& rune_center_world,
    const Eigen::Vector3d& rune_ypr_world,
    double radius_m)
{
    std::vector<Eigen::Vector3d> ring_world;
    ring_world.reserve(kBuffGuideRingSegments);
    for (int index = 0; index < kBuffGuideRingSegments; ++index) {
        const double angle = 2.0 * CV_PI * static_cast<double>(index) /
                             static_cast<double>(kBuffGuideRingSegments);
        ring_world.push_back(transformBuffLocalPoint(
            buffLocalPolarPoint(radius_m, angle), rune_center_world, rune_ypr_world));
    }
    return ring_world;
}

std::vector<Eigen::Vector3d> buildSolvedBladeOutlineWorld(
    const auto_buff::Solver& solver,
    const Eigen::Vector3d& rune_center_world,
    const Eigen::Vector3d& rune_ypr_world)
{
    const auto& object_points = solver.get_object_points();
    std::vector<Eigen::Vector3d> outline_world;
    outline_world.reserve(std::min<size_t>(4, object_points.size()));
    for (size_t index = 0; index < std::min<size_t>(4, object_points.size()); ++index) {
        outline_world.push_back(transformBuffLocalPoint(
            solverModelPointToEigen(object_points[index]), rune_center_world, rune_ypr_world));
    }
    return outline_world;
}

void appendBuffGuideSkeleton3D(
    rm::DebugHudSnapshot& hud,
    const auto_buff::Solver& solver,
    const Eigen::Vector3d& rune_center_world,
    const Eigen::Vector3d& rune_ypr_world)
{
    if (!isFiniteWorldPoint(rune_center_world) || !rune_ypr_world.array().isFinite().all()) {
        return;
    }

    const auto guide_radii = buffGuideRadii(solver);
    appendOverlayLineStrip3D(
        hud,
        "buff/overlay3d/model_guides/rings",
        buildBuffGuideRingWorld(rune_center_world, rune_ypr_world, guide_radii[0]),
        rgba(120, 160, 220, 140),
        0.004f,
        true,
        "",
        false);
    appendOverlayLineStrip3D(
        hud,
        "buff/overlay3d/model_guides/rings",
        buildBuffGuideRingWorld(rune_center_world, rune_ypr_world, guide_radii[1]),
        rgba(180, 200, 230, 120),
        0.004f,
        true,
        "",
        false);
    appendOverlayLineStrip3D(
        hud,
        "buff/overlay3d/model_guides/rings",
        buildBuffGuideRingWorld(rune_center_world, rune_ypr_world, guide_radii[2]),
        rgba(120, 160, 220, 140),
        0.004f,
        true,
        "",
        false);

    for (int phase_index = 0; phase_index < kBuffGuidePhaseCount; ++phase_index) {
        const double angle = 2.0 * CV_PI * static_cast<double>(phase_index) /
                             static_cast<double>(kBuffGuidePhaseCount);
        appendOverlayLineStrip3D(
            hud,
            "buff/overlay3d/model_guides/spokes",
            {
                transformBuffLocalPoint(Eigen::Vector3d::Zero(), rune_center_world, rune_ypr_world),
                transformBuffLocalPoint(
                    buffLocalPolarPoint(guide_radii[2], angle), rune_center_world, rune_ypr_world),
            },
            phase_index == 0 ? rgba(0, 220, 255, 180) : rgba(150, 170, 200, 96),
            phase_index == 0 ? 0.006f : 0.004f,
            false,
            "",
            false);
    }
}

ProjectedBladeOverlayState projectBladeOverlayState(
    const auto_buff::Solver& solver,
    auto_buff::BuffTracker& tracker,
    const Eigen::VectorXd& state,
    double target_roll_offset)
{
    ProjectedBladeOverlayState overlay;
    if (state.size() < 10 || !state.array().isFinite().all()) return overlay;

    overlay.rune_center_world = tracker.point_buff2world(Eigen::Vector3d::Zero(), state);
    overlay.blade_center_world = tracker.point_buff2world(
        Eigen::Vector3d(0.0, 0.0, 0.7), state, target_roll_offset);
    if (!isFiniteWorldPoint(overlay.rune_center_world)) return overlay;

    const auto& object_points = solver.get_object_points();
    overlay.blade_outline_world.reserve(std::min<size_t>(4, object_points.size()));
    overlay.blade_keypoints_world.reserve(std::min<size_t>(4, object_points.size()));
    for (size_t index = 0; index < std::min<size_t>(4, object_points.size()); ++index) {
        const Eigen::Vector3d model_point = solverModelPointToEigen(object_points[index]);
        const Eigen::Vector3d world_point = tracker.point_buff2world(model_point, state, target_roll_offset);
        overlay.blade_outline_world.push_back(world_point);
        overlay.blade_keypoints_world.push_back(world_point);
    }

    overlay.projected_points = solver.reproject_buff(
        overlay.rune_center_world, state[4], state[5] + target_roll_offset);
    return overlay;
}

void drawProjectedBlade(
    cv::Mat& image,
    const std::vector<cv::Point2f>& points,
    const cv::Scalar& color,
    int thickness)
{
    if (points.size() < 5) return;

    std::vector<cv::Point> polygon;
    polygon.reserve(4);
    for (int i = 0; i < 4; ++i) {
        if (!isDrawablePoint(points[i], image)) return;
        polygon.emplace_back(
            static_cast<int>(std::lround(points[i].x)),
            static_cast<int>(std::lround(points[i].y)));
    }
    cv::polylines(image, std::vector<std::vector<cv::Point>>{polygon}, true, color, thickness);

    if (isDrawablePoint(points[4], image)) {
        cv::circle(image, points[4], 4, color, -1, cv::LINE_AA);
    }
}

void drawPredictionOverlay(
    cv::Mat& image,
    const auto_buff::Solver& solver,
    auto_buff::BuffTracker& tracker,
    double predict_dt_s,
    bool command_valid)
{
    if (image.empty() || tracker.is_lost()) return;

    const Eigen::VectorXd current_state = tracker.get_state();
    if (current_state.size() < 6 || !current_state.array().isFinite().all()) return;

    const double safe_dt_s = std::clamp(predict_dt_s, 0.0, 1.0);
    const double target_roll_offset = tracker.selected_target_roll_offset();
    const Eigen::VectorXd predicted_state = tracker.predict(safe_dt_s);
    if (predicted_state.size() < 6 || !predicted_state.array().isFinite().all()) return;

    const Eigen::Vector3d current_center =
        tracker.point_buff2world(Eigen::Vector3d::Zero(), current_state);
    const Eigen::Vector3d predicted_center =
        tracker.point_buff2world(Eigen::Vector3d::Zero(), predicted_state);
    if (!current_center.array().isFinite().all() || !predicted_center.array().isFinite().all()) {
        return;
    }

    const auto current_points =
        solver.reproject_buff(current_center, current_state[4], current_state[5] + target_roll_offset);
    const auto predicted_points =
        solver.reproject_buff(
            predicted_center, predicted_state[4], predicted_state[5] + target_roll_offset);
    if (current_points.size() < 5 || predicted_points.size() < 5) return;

    const cv::Point2f current_hit = current_points[4];
    const cv::Point2f predicted_hit = predicted_points[4];
    const cv::Scalar current_color(255, 255, 0);
    const cv::Scalar predicted_color(255, 0, 255);
    const cv::Scalar line_color(255, 255, 255);

    drawProjectedBlade(image, current_points, current_color, 1);
    drawProjectedBlade(image, predicted_points, predicted_color, 2);

    if (isDrawablePoint(current_hit, image) && isDrawablePoint(predicted_hit, image)) {
        cv::line(image, current_hit, predicted_hit, line_color, 1, cv::LINE_AA);
        cv::drawMarker(
            image, current_hit, current_color, cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
        cv::drawMarker(
            image, predicted_hit, predicted_color, cv::MARKER_TILTED_CROSS, 24, 2,
            cv::LINE_AA);
        cv::circle(image, predicted_hit, 10, predicted_color, 2, cv::LINE_AA);

        const std::string label = cv::format(
            "pred %.0fms %s", safe_dt_s * 1000.0, command_valid ? "CTRL" : "NOCTRL");
        cv::putText(
            image, label, predicted_hit + cv::Point2f(12.0f, -12.0f),
            cv::FONT_HERSHEY_SIMPLEX, 0.55, predicted_color, 2, cv::LINE_AA);
    }
}

double normalizeFullTurn(double raw_roll, double reference_roll)
{
    if (!std::isfinite(raw_roll) || !std::isfinite(reference_roll)) {
        return raw_roll;
    }
    return raw_roll + std::round((reference_roll - raw_roll) / CV_2PI) * CV_2PI;
}

int observedBladePhaseIndex(double observed_roll, double reference_global_roll)
{
    constexpr double step = CV_2PI / 5.0;
    int best_index = 0;
    double best_error = std::numeric_limits<double>::max();
    for (int k = 0; k < 5; ++k) {
        const double offset = static_cast<double>(k) * step;
        const double global_roll = normalizeFullTurn(observed_roll - offset, reference_global_roll);
        const double error = std::abs(global_roll - reference_global_roll);
        if (error < best_error) {
            best_error = error;
            best_index = k;
        }
    }
    return best_index;
}

int normalizePhaseIndex(int index)
{
    int normalized = index % 5;
    if (normalized < 0) normalized += 5;
    return normalized;
}

int logicalPhaseIndex(int raw_phase_index, int phase_origin_index)
{
    if (raw_phase_index < 0 || phase_origin_index < 0) {
        return -1;
    }
    return normalizePhaseIndex(raw_phase_index - phase_origin_index);
}

int phaseIndexFromOffset(double offset)
{
    if (!std::isfinite(offset)) return -1;
    constexpr double step = CV_2PI / 5.0;
    return normalizePhaseIndex(static_cast<int>(std::lround(offset / step)));
}

void drawAllBladePhaseLabels(
    cv::Mat& image,
    const auto_buff::Solver& solver,
    const auto_buff::BuffTracker& tracker)
{
    if (image.empty()) return;

    const auto tracker_debug = tracker.debugSnapshot(0.0);
    if (!tracker_debug.initialized) return;

    const Eigen::VectorXd state = tracker.get_state();
    if (state.size() < 6 || !state.array().isFinite().all()) return;

    const Eigen::Vector3d rune_center =
        tracker.point_buff2world(Eigen::Vector3d::Zero(), state);
    if (!rune_center.array().isFinite().all()) return;

    constexpr double step = CV_2PI / 5.0;
    const int phase_origin_index = tracker_debug.phase_origin_index;
    const bool phase_origin_locked = phase_origin_index >= 0;
    int selected_phase = tracker_debug.selected_phase_index;
    if ((selected_phase < 0 || selected_phase >= 5) && phase_origin_locked) {
        selected_phase =
            logicalPhaseIndex(phaseIndexFromOffset(tracker_debug.selected_roll_offset), phase_origin_index);
    }

    const cv::Scalar r_color = tracker_debug.lost ? cv::Scalar(90, 90, 90) : cv::Scalar(180, 180, 180);
    for (int phase = 0; phase < 5; ++phase) {
        const int raw_phase = phase_origin_locked
            ? normalizePhaseIndex(phase + phase_origin_index)
            : phase;
        const auto points =
            solver.reproject_buff(rune_center, state[4], state[5] + static_cast<double>(raw_phase) * step);
        if (points.size() < 6) continue;

        const cv::Point2f label_pos = points[4];
        if (!isDrawablePoint(label_pos, image)) continue;

        const bool selected = phase == selected_phase;
        const cv::Scalar color = selected
            ? cv::Scalar(0, 255, 255)
            : (tracker_debug.lost ? cv::Scalar(120, 120, 120) : cv::Scalar(255, 255, 255));
        const std::string label = phase_origin_locked
            ? (selected ? cv::format("T%d", phase) : cv::format("%d", phase))
            : (selected ? "T?" : "?");
        const cv::Point text_pos(
            static_cast<int>(std::lround(label_pos.x + 10.0f)),
            static_cast<int>(std::lround(label_pos.y + 6.0f)));

        if (isDrawablePoint(points[5], image)) {
            cv::line(image, points[5], label_pos, r_color, 1, cv::LINE_AA);
        }
        cv::circle(image, label_pos, selected ? 9 : 7, color, 2, cv::LINE_AA);
        if (!selected) {
            cv::putText(
                image, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
            cv::putText(
                image, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                color, 2, cv::LINE_AA);
        }
    }
}

void drawObservedBladeLabels(
    cv::Mat& image,
    const auto_buff::PowerRune& rune,
    const auto_buff::BuffTracker& tracker)
{
    if (image.empty() || rune.fanblades.empty() || tracker.is_lost()) return;

    const Eigen::VectorXd state = tracker.get_state();
    if (state.size() < 6 || !state.array().isFinite().all()) return;
    const auto tracker_debug = tracker.debugSnapshot(0.0);
    const int phase_origin_index = tracker_debug.phase_origin_index;
    const bool phase_origin_locked = phase_origin_index >= 0;
    const double reference_roll = state[5];

    for (size_t i = 0; i < rune.fanblades.size(); ++i) {
        const auto& blade = rune.fanblades[i];
        if (!blade.solved || !blade.ypr_in_world.array().isFinite().all()) continue;

        cv::Point2f label_pos = blade.center;
        if (!isFinitePoint(label_pos) && blade.points.size() >= 4) {
            label_pos = cv::Point2f(0.0f, 0.0f);
            for (int j = 0; j < 4; ++j) label_pos += blade.points[j];
            label_pos *= 0.25f;
        }
        if (!isDrawablePoint(label_pos, image)) continue;

        const int raw_phase_index = observedBladePhaseIndex(blade.ypr_in_world[2], reference_roll);
        const int phase_index = logicalPhaseIndex(raw_phase_index, phase_origin_index);
        const bool selected = i == 0;
        const std::string label = phase_origin_locked
            ? cv::format("%s%d", selected ? "T" : "B", phase_index)
            : (selected ? "T?" : "B?");
        const cv::Scalar color = selected ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 255, 0);
        const cv::Point text_pos(
            static_cast<int>(std::lround(label_pos.x + 8.0f)),
            static_cast<int>(std::lround(label_pos.y - 8.0f)));

        cv::circle(image, label_pos, selected ? 7 : 5, color, 2, cv::LINE_AA);
        if (!selected) {
            cv::putText(
                image, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
            cv::putText(
                image, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                color, 2, cv::LINE_AA);
        }
    }
}

void drawRLogoOverlay(cv::Mat& image, const auto_buff::PowerRune& rune, bool draw_binary_mask)
{
    if (image.empty() || !isDrawablePoint(rune.r_center, image)) return;

    const cv::Scalar shadow(0, 0, 0);
    if (rune.r_search_debug.has_value()) {
        const auto& debug = *rune.r_search_debug;
        const cv::Rect roi_rect = clampRectToImage(debug.roi_rect, image);
        if (roi_rect.area() > 0) {
            cv::rectangle(image, roi_rect, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }
        if (debug.radius > 1.0 && isDrawablePoint(debug.prior_center, image)) {
            cv::circle(
                image,
                debug.prior_center,
                static_cast<int>(std::lround(debug.radius)),
                cv::Scalar(255, 0, 0),
                1,
                cv::LINE_AA);
            cv::circle(image, debug.prior_center, 4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
        }
        if (isDrawablePoint(debug.yolo_center, image)) {
            cv::circle(image, debug.yolo_center, 4, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
        }
        if (isDrawablePoint(debug.raw_center, image)) {
            cv::circle(image, debug.raw_center, 7, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
        }
        const cv::Rect template_rect = clampRectToImage(debug.template_rect, image);
        if (template_rect.area() > 0) {
            cv::rectangle(image, template_rect, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        }
        for (const auto& contour : debug.accepted_contour_points) {
            if (contour.empty()) continue;
            std::vector<cv::Point> shifted;
            shifted.reserve(contour.size());
            for (const auto& point : contour) {
                shifted.emplace_back(point.x + debug.roi_rect.x, point.y + debug.roi_rect.y);
            }
            cv::polylines(image, shifted, true, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }
        for (const auto& center : debug.accepted_centers) {
            if (isDrawablePoint(center, image)) {
                cv::drawMarker(
                    image, center, cv::Scalar(255, 255, 0), cv::MARKER_TILTED_CROSS, 12, 1, cv::LINE_AA);
            }
        }

        if (draw_binary_mask && !debug.masked_roi.empty()) {
            cv::Mat roi_preview_gray;
            if (debug.masked_roi.channels() == 1) {
                roi_preview_gray = debug.masked_roi;
            } else {
                cv::cvtColor(debug.masked_roi, roi_preview_gray, cv::COLOR_BGR2GRAY);
            }
            const int preview_w = std::min(220, std::max(80, image.cols / 5));
            const double scale =
                static_cast<double>(preview_w) / static_cast<double>(std::max(roi_preview_gray.cols, 1));
            const int preview_h = std::clamp(
                static_cast<int>(std::lround(roi_preview_gray.rows * scale)),
                40,
                std::max(40, image.rows / 4));
            cv::Mat roi_preview;
            cv::resize(roi_preview_gray, roi_preview, cv::Size(preview_w, preview_h), 0.0, 0.0, cv::INTER_NEAREST);
            cv::cvtColor(roi_preview, roi_preview, cv::COLOR_GRAY2BGR);

            const int preview_x = std::max(0, image.cols - preview_w - 12);
            const int preview_y = std::max(0, image.rows - preview_h - 12);
            cv::Rect preview_rect(preview_x, preview_y, preview_w, preview_h);
            if ((preview_rect & cv::Rect(0, 0, image.cols, image.rows)) == preview_rect) {
                cv::rectangle(image, preview_rect + cv::Size(4, 4), shadow, -1);
                roi_preview.copyTo(image(preview_rect));
                cv::rectangle(image, preview_rect, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                const cv::Point label_pos(preview_x, std::max(12, preview_y - 5));
                cv::putText(
                    image, "R mask", label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    shadow, 3, cv::LINE_AA);
                cv::putText(
                    image, "R mask", label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        }

        const cv::Point text_pos(
            static_cast<int>(std::lround(rune.r_center.x + 16.0f)),
            static_cast<int>(std::lround(rune.r_center.y - 16.0f)));
        const std::string label = cv::format(
            "R roi %.0f tpl %.2f cnt %d/%d %s",
            debug.radius,
            debug.template_score,
            debug.accepted_count,
            debug.total_contours,
            debug.used_template
                ? "templ"
                : (debug.used_hold_center ? "hold" : (debug.used_contour_center ? "contour" : "prior")));
        cv::putText(
            image, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
            shadow, 3, cv::LINE_AA);
        cv::putText(
            image, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    const cv::Scalar color(0, 255, 0);
    const cv::Point2f p = rune.r_center;
    cv::circle(image, p, 12, color, 2, cv::LINE_AA);
    cv::circle(image, p, 3, color, -1, cv::LINE_AA);
    cv::drawMarker(image, p, color, cv::MARKER_CROSS, 28, 2, cv::LINE_AA);

    const cv::Point text_pos(
        static_cast<int>(std::lround(p.x + 14.0f)),
        static_cast<int>(std::lround(p.y + 18.0f)));
    cv::putText(
        image, "R", text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.8,
        shadow, 4, cv::LINE_AA);
    cv::putText(
        image, "R", text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.8,
        color, 2, cv::LINE_AA);
}

const char* pnpMethodName(int method)
{
    switch (method) {
        case cv::SOLVEPNP_ITERATIVE:
            return "ITER";
        case cv::SOLVEPNP_IPPE:
            return "IPPE";
        default:
            return "PNP";
    }
}

void drawPnpDebugOverlay(cv::Mat& image, const auto_buff::PowerRune& rune)
{
    if (image.empty() || rune.fanblades.empty()) return;

    constexpr std::array<const char*, 5> point_labels = {"T", "L", "B", "R", "R0"};
    const cv::Scalar observed_color(0, 255, 255);
    const cv::Scalar r_observed_color(0, 255, 0);
    const cv::Scalar reprojected_color(255, 0, 255);
    const cv::Scalar line_color(255, 255, 255);
    const cv::Scalar center_color(255, 255, 0);
    const cv::Scalar shadow(0, 0, 0);

    for (size_t blade_idx = 0; blade_idx < rune.fanblades.size(); ++blade_idx) {
        const auto& blade = rune.fanblades[blade_idx];
        if (!blade.solved) continue;

        const size_t point_count =
            std::min(blade.pnp_observed_points.size(), blade.pnp_input_reprojected_points.size());
        for (size_t i = 0; i < point_count; ++i) {
            const cv::Point2f observed = blade.pnp_observed_points[i];
            const cv::Point2f reprojected = blade.pnp_input_reprojected_points[i];
            if (isDrawablePoint(observed, image)) {
                const cv::Scalar color = (i == 4) ? r_observed_color : observed_color;
                cv::circle(image, observed, 5, color, -1, cv::LINE_AA);
                cv::putText(
                    image,
                    point_labels[std::min<size_t>(i, point_labels.size() - 1)],
                    observed + cv::Point2f(5.0f, -5.0f),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.42,
                    shadow,
                    3,
                    cv::LINE_AA);
                cv::putText(
                    image,
                    point_labels[std::min<size_t>(i, point_labels.size() - 1)],
                    observed + cv::Point2f(5.0f, -5.0f),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv::LINE_AA);
            }
            if (isDrawablePoint(reprojected, image)) {
                cv::circle(image, reprojected, 5, reprojected_color, 2, cv::LINE_AA);
            }
            if (isDrawablePoint(observed, image) && isDrawablePoint(reprojected, image)) {
                cv::line(image, observed, reprojected, line_color, 1, cv::LINE_AA);
            }
        }

        if (isDrawablePoint(blade.center, image)) {
            cv::drawMarker(
                image, blade.center, center_color, cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
        }
        if (isDrawablePoint(blade.pnp_model_center, image)) {
            cv::drawMarker(
                image,
                blade.pnp_model_center,
                reprojected_color,
                cv::MARKER_TILTED_CROSS,
                22,
                2,
                cv::LINE_AA);
            cv::circle(image, blade.pnp_model_center, 8, reprojected_color, 2, cv::LINE_AA);
            if (isDrawablePoint(blade.center, image)) {
                cv::line(image, blade.center, blade.pnp_model_center, line_color, 1, cv::LINE_AA);
            }
        }

        if (blade_idx == 0 && isDrawablePoint(blade.center, image)) {
            const std::string label = cv::format(
                "PnP %s err %.2f ctr %.1f R/T %.1f/%.1f ord %d%d%d%d",
                pnpMethodName(blade.pnp_method),
                blade.pnp_reproj_error_px,
                blade.pnp_model_center_error_px,
                blade.pnp_model_center_radial_error_px,
                blade.pnp_model_center_tangent_error_px,
                blade.pnp_order[0],
                blade.pnp_order[1],
                blade.pnp_order[2],
                blade.pnp_order[3]);
            const cv::Point text_pos(
                static_cast<int>(std::lround(blade.center.x + 14.0f)),
                static_cast<int>(std::lround(blade.center.y + 24.0f)));
            cv::putText(
                image, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                shadow, 3, cv::LINE_AA);
            cv::putText(
                image, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                reprojected_color, 1, cv::LINE_AA);
        }
    }
}

double stateAgeSeconds(
    std::chrono::steady_clock::time_point frame_timestamp,
    std::chrono::steady_clock::time_point now)
{
    if (frame_timestamp.time_since_epoch().count() == 0) return 0.0;
    const double age_s = std::chrono::duration<double>(now - frame_timestamp).count();
    if (!std::isfinite(age_s)) return 0.0;
    return std::clamp(age_s, 0.0, 0.2);
}

double framePipelineAgeSeconds(
    const rm::Frame& frame,
    std::chrono::steady_clock::time_point frame_timestamp,
    std::chrono::steady_clock::time_point now)
{
    if (frame_timestamp.time_since_epoch().count() != 0) {
        return stateAgeSeconds(frame_timestamp, now);
    }

    if (
        frame.startTime.time_since_epoch().count() != 0 &&
        std::isfinite(frame.timeStamp) &&
        frame.timeStamp > 0.0) {
        const auto capture_tp = frame.startTime +
            std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
                std::chrono::duration<double, std::milli>(frame.timeStamp));
        const double age_s =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - capture_tp)
                .count();
        if (std::isfinite(age_s)) {
            return std::clamp(age_s, 0.0, 0.2);
        }
    }
    return stateAgeSeconds(frame_timestamp, now);
}

double clampPipelineDelay(double delay_s)
{
    if (!std::isfinite(delay_s)) return 0.0;
    return std::clamp(delay_s, 0.0, 0.2);
}

double buildProcessingDelayFloorSeconds(double infer_ms, double track_aim_ms, double queue_ms = 0.0)
{
    return clampPipelineDelay(std::max(0.0, infer_ms + track_aim_ms + queue_ms) * 0.001);
}

bool envFlagEnabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr) return false;
    const std::string text(value);
    return !text.empty() && text != "0" && text != "false" && text != "FALSE";
}

std::filesystem::path resolvePredictionLogPath()
{
    const char* env_path = std::getenv("BUFF_PREDICTION_LOG_PATH");
    if (env_path != nullptr && env_path[0] != '\0') {
        return std::filesystem::path(env_path);
    }
    return std::filesystem::current_path() / "build/buff_prediction_debug.csv";
}

std::filesystem::path resolvePredictionErrorLogPath()
{
    const char* env_path = std::getenv("BUFF_PREDICTION_ERROR_LOG_PATH");
    if (env_path != nullptr && env_path[0] != '\0') {
        return std::filesystem::path(env_path);
    }
    return std::filesystem::current_path() / "build/buff_prediction_error.csv";
}

double frameTimeMs(const rm::Frame& frame)
{
    if (std::isfinite(frame.usb_timeStamp) && frame.usb_timeStamp > 0.0) {
        return frame.usb_timeStamp;
    }
    return finiteOr(frame.timeStamp, nanValue());
}

double debugOverlayTimeMs(
    const rm::Frame& frame,
    std::chrono::steady_clock::time_point frame_timestamp)
{
    const double frame_time_ms = frameTimeMs(frame);
    if (std::isfinite(frame_time_ms)) {
        return frame_time_ms;
    }
    if (frame_timestamp.time_since_epoch().count() != 0) {
        return std::chrono::duration<double, std::milli>(
            frame_timestamp.time_since_epoch()).count();
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void writeCsvValue(std::ofstream& file, double value)
{
    if (std::isfinite(value)) {
        file << value;
    } else {
        file << "nan";
    }
}

void writeCsvPoint(std::ofstream& file, const cv::Point2f& point)
{
    writeCsvValue(file, point.x);
    file << ',';
    writeCsvValue(file, point.y);
}

void writeCsvVector(std::ofstream& file, const Eigen::Vector3d& vector)
{
    writeCsvValue(file, vector.x());
    file << ',';
    writeCsvValue(file, vector.y());
    file << ',';
    writeCsvValue(file, vector.z());
}

Eigen::Vector3d safeYpdFromWorldPoint(const Eigen::Vector3d& point)
{
    if (!point.array().isFinite().all()) {
        return Eigen::Vector3d::Constant(nanValue());
    }
    return tools::xyz2ypd(point);
}

double pointDistance(const cv::Point2f& a, const cv::Point2f& b)
{
    if (!isFinitePoint(a) || !isFinitePoint(b)) {
        return nanValue();
    }
    return cv::norm(a - b);
}

std::pair<double, double> radialTangentialError(
    const cv::Point2f& predicted,
    const cv::Point2f& actual,
    const cv::Point2f& r_center)
{
    if (!isFinitePoint(predicted) || !isFinitePoint(actual) || !isFinitePoint(r_center)) {
        return {nanValue(), nanValue()};
    }
    const cv::Point2f radial = actual - r_center;
    const double radius = cv::norm(radial);
    if (!std::isfinite(radius) || radius < 1e-3) {
        return {nanValue(), nanValue()};
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

double yawPitchDistanceMrad(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
    if (!a.array().isFinite().all() || !b.array().isFinite().all()) {
        return nanValue();
    }
    const double yaw_error = tools::limit_rad(a[0] - b[0]);
    const double pitch_error = a[1] - b[1];
    return std::hypot(yaw_error, pitch_error) * 1000.0;
}

double rollDifference(double a, double b)
{
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return nanValue();
    }
    return tools::limit_rad(a - b);
}

class PredictionErrorCsvLogger
{
private:
    struct PendingPrediction
    {
        uint64_t seq = 0;
        std::string eval_mode;
        double time_ms = nanValue();
        double target_time_ms = nanValue();
        double eval_dt_s = nanValue();
        cv::Point2f predicted_center = cv::Point2f(nanValue(), nanValue());
        Eigen::Vector3d predicted_blade_ypd = Eigen::Vector3d::Constant(nanValue());
        bool pred_detected = false;
        bool pred_tracker_lost = true;
        double pred_time_since_detection_ms = nanValue();
        double pred_target_roll_offset = nanValue();
        int pred_selected_phase_index = -1;
        int pred_voter_direction = 0;
        int pred_history_size = 0;
        bool pred_switch_deferred = false;
        bool pred_target_switched = false;
        int pred_reinit_reason = 0;
        double pred_roll = nanValue();
        double pred_speed = nanValue();
        double pred_a = nanValue();
        double pred_w = nanValue();
        double pred_phi = nanValue();
    };

public:
    void observeAndQueue(
        uint64_t seq,
        const char* eval_mode,
        int stage,
        double time_ms,
        double target_time_ms,
        double eval_dt_s,
        bool pred_detected,
        bool pred_tracker_lost,
        double pred_target_roll_offset,
        int pred_selected_phase_index,
        int pred_voter_direction,
        int pred_history_size,
        bool pred_switch_deferred,
        bool pred_target_switched,
        int pred_reinit_reason,
        const std::optional<auto_buff::PowerRune>& rune,
        const cv::Point2f& predicted_center,
        const Eigen::Vector3d& predicted_blade_ypd,
        double pred_roll,
        double pred_speed,
        double pred_a,
        double pred_w,
        double pred_phi)
    {
        if (stage != 0 || !std::isfinite(time_ms)) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!ensureOpen()) return;

        if (rune.has_value() && !rune->fanblades.empty() && !rune->is_unsolve()) {
            last_detected_time_ms_ = time_ms;
        }
        const double time_since_detection_ms =
            std::isfinite(last_detected_time_ms_) ? time_ms - last_detected_time_ms_ : nanValue();

        logReadyObservations(time_ms, rune);

        if (
            std::isfinite(target_time_ms) && target_time_ms > time_ms &&
            isFinitePoint(predicted_center) && predicted_blade_ypd.array().isFinite().all()) {
            PendingPrediction prediction{
                seq,
                eval_mode == nullptr ? "" : eval_mode,
                time_ms,
                target_time_ms,
                eval_dt_s,
                predicted_center,
                predicted_blade_ypd,
                pred_detected,
                pred_tracker_lost,
                time_since_detection_ms,
                pred_target_roll_offset,
                pred_selected_phase_index,
                pred_voter_direction,
                pred_history_size,
                pred_switch_deferred,
                pred_target_switched,
                pred_reinit_reason,
                pred_roll,
                pred_speed,
                pred_a,
                pred_w,
                pred_phi};
            const auto insert_it = std::upper_bound(
                pending_.begin(),
                pending_.end(),
                prediction.target_time_ms,
                [](double target_time_ms, const PendingPrediction& pending) {
                    return target_time_ms < pending.target_time_ms;
                });
            pending_.insert(insert_it, std::move(prediction));
            while (pending_.size() > 512) {
                pending_.pop_front();
            }
        }
    }

    void logCurrent(
        uint64_t seq,
        const char* eval_mode,
        double pred_time_ms,
        double actual_time_ms,
        double eval_dt_s,
        bool pred_detected,
        bool pred_tracker_lost,
        double pred_time_since_detection_ms,
        double pred_target_roll_offset,
        int pred_selected_phase_index,
        int pred_voter_direction,
        int pred_history_size,
        bool pred_switch_deferred,
        bool pred_target_switched,
        int pred_reinit_reason,
        const std::optional<auto_buff::PowerRune>& rune,
        const cv::Point2f& predicted_center,
        const Eigen::Vector3d& predicted_blade_ypd,
        double pred_roll,
        double pred_speed,
        double pred_a,
        double pred_w,
        double pred_phi)
    {
        if (!std::isfinite(pred_time_ms) || !std::isfinite(actual_time_ms) ||
            actual_time_ms <= pred_time_ms || !isFinitePoint(predicted_center) ||
            !predicted_blade_ypd.array().isFinite().all()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!ensureOpen() || !rune.has_value() || rune->fanblades.empty() || rune->is_unsolve()) {
            return;
        }

        PendingPrediction prediction{
            seq,
            eval_mode == nullptr ? "" : eval_mode,
            pred_time_ms,
            actual_time_ms,
            eval_dt_s,
            predicted_center,
            predicted_blade_ypd,
            pred_detected,
            pred_tracker_lost,
            pred_time_since_detection_ms,
            pred_target_roll_offset,
            pred_selected_phase_index,
            pred_voter_direction,
            pred_history_size,
            pred_switch_deferred,
            pred_target_switched,
            pred_reinit_reason,
            pred_roll,
            pred_speed,
            pred_a,
            pred_w,
            pred_phi,
        };
        writeObservation(prediction, actual_time_ms, 0.0, rune);
    }

private:

    bool ensureOpen()
    {
        if (checked_) return file_.is_open();
        checked_ = true;

        if (!envFlagEnabled("BUFF_PREDICTION_LOG") && !envFlagEnabled("BUFF_PREDICTION_ERROR_LOG")) {
            return false;
        }

        const std::filesystem::path path = resolvePredictionErrorLogPath();
        std::error_code ec;
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path(), ec);
        }
        file_.open(path, std::ios::out | std::ios::trunc);
        if (!file_.is_open()) {
            tools::logger()->warn("Failed to open buff prediction error log: {}", path.string());
            return false;
        }

        file_ << "pred_seq,eval_mode,eval_dt_s,"
              << "pred_detected,pred_tracker_lost,pred_time_since_detection_ms,"
              << "pred_voter_direction,pred_reinit_reason,"
              << "visible_blades,target_angle_error_mrad,"
              << "target_radial_error_px,target_tangent_error_px,"
              << "nearest_angle_error_mrad,"
              << "pred_roll_rad,pred_speed_rad_s,pred_target_roll_rad,"
              << "actual_target_roll_rad,"
              << "target_signed_roll_error_rad,"
              << "target_signed_roll_lead_rad,"
              << "pred_big_a,pred_big_w,pred_big_phi\n";
        tools::logger()->info("Buff prediction error log path: {}", path.string());
        return true;
    }

    void logReadyObservations(double time_ms, const std::optional<auto_buff::PowerRune>& rune)
    {
        constexpr double kMaxMatchDelayMs = 25.0;
        while (!pending_.empty() && pending_.front().target_time_ms <= time_ms) {
            PendingPrediction pending = pending_.front();
            pending_.pop_front();

            const double match_dt_ms = time_ms - pending.target_time_ms;
            if (match_dt_ms > kMaxMatchDelayMs || !rune.has_value() || rune->fanblades.empty() ||
                rune->is_unsolve()) {
                continue;
            }

            writeObservation(pending, time_ms, match_dt_ms, rune);
        }
    }

    void writeObservation(
        const PendingPrediction& pending,
        double actual_time_ms,
        double match_dt_ms,
        const std::optional<auto_buff::PowerRune>& rune)
    {
        if (!rune.has_value() || rune->fanblades.empty() || rune->is_unsolve()) {
            return;
        }

        double target_angle_error_mrad = nanValue();
        double target_radial_error_px = nanValue();
        double target_tangent_error_px = nanValue();
        double actual_target_roll = nanValue();
        if (!rune->fanblades.empty()) {
            const auto& target = rune->target();
            target_angle_error_mrad =
                yawPitchDistanceMrad(pending.predicted_blade_ypd, target.blade_ypd_in_world);
            const auto radial_tangent =
                radialTangentialError(pending.predicted_center, target.center, rune->r_center);
            target_radial_error_px = radial_tangent.first;
            target_tangent_error_px = radial_tangent.second;
            actual_target_roll = target.ypr_in_world[2];
        }

        double nearest_angle_error_mrad = nanValue();
        int visible_blades = 0;
        for (const auto& blade : rune->fanblades) {
            if (!blade.solved || !blade.blade_ypd_in_world.array().isFinite().all()) {
                continue;
            }
            visible_blades++;
            const double angle_error =
                yawPitchDistanceMrad(pending.predicted_blade_ypd, blade.blade_ypd_in_world);
            if (std::isfinite(angle_error) &&
                (!std::isfinite(nearest_angle_error_mrad) || angle_error < nearest_angle_error_mrad)) {
                nearest_angle_error_mrad = angle_error;
            }
        }
        const double pred_target_roll = std::isfinite(pending.pred_roll)
            ? pending.pred_roll + pending.pred_target_roll_offset
            : nanValue();
        const double target_signed_roll_error =
            rollDifference(pred_target_roll, actual_target_roll);
        const double direction = static_cast<double>(pending.pred_voter_direction);
        const double target_signed_roll_lead =
            direction != 0.0 && std::isfinite(target_signed_roll_error)
                ? direction * target_signed_roll_error
                : nanValue();

        file_ << pending.seq << ',' << std::fixed << std::setprecision(6);
        file_ << pending.eval_mode << ',';
        writeCsvValue(file_, pending.eval_dt_s);
        file_ << ',' << (pending.pred_detected ? 1 : 0)
              << ',' << (pending.pred_tracker_lost ? 1 : 0)
              << ',';
        writeCsvValue(file_, pending.pred_time_since_detection_ms);
        file_ << ',' << pending.pred_voter_direction
              << ',' << pending.pred_reinit_reason
              << ',' << visible_blades << ',';
        writeCsvValue(file_, target_angle_error_mrad);
        file_ << ',';
        writeCsvValue(file_, target_radial_error_px);
        file_ << ',';
        writeCsvValue(file_, target_tangent_error_px);
        file_ << ',';
        writeCsvValue(file_, nearest_angle_error_mrad);
        file_ << ',';
        writeCsvValue(file_, pending.pred_roll);
        file_ << ',';
        writeCsvValue(file_, pending.pred_speed);
        file_ << ',';
        writeCsvValue(file_, pred_target_roll);
        file_ << ',';
        writeCsvValue(file_, actual_target_roll);
        file_ << ',';
        writeCsvValue(file_, target_signed_roll_error);
        file_ << ',';
        writeCsvValue(file_, target_signed_roll_lead);
        file_ << ',';
        writeCsvValue(file_, pending.pred_a);
        file_ << ',';
        writeCsvValue(file_, pending.pred_w);
        file_ << ',';
        writeCsvValue(file_, pending.pred_phi);
        file_ << '\n';
        flushPeriodically();
    }

    void flushPeriodically()
    {
        if (++unflushed_rows_ >= kCsvFlushEveryRows) {
            file_.flush();
            unflushed_rows_ = 0;
        }
    }

    bool checked_ = false;
    std::ofstream file_;
    std::mutex mutex_;
    std::deque<PendingPrediction> pending_;
    double last_detected_time_ms_ = nanValue();
    int unflushed_rows_ = 0;
};

PredictionErrorCsvLogger& predictionErrorCsvLogger()
{
    static PredictionErrorCsvLogger logger;
    return logger;
}

// ---------------------------------------------------------------------------
// ObservationDumpCsvLogger
// Enabled by env flag BUFF_OBSERVATION_DUMP_LOG=1.
// Writes one row per stage-0 frame with the tracker's roll observations and
// EKF state needed for offline ground-truth fitting.
// ---------------------------------------------------------------------------
class ObservationDumpCsvLogger
{
public:
    void log(
        uint64_t seq,
        double timestamp_ms,
        int stage,
        const auto_buff::BuffTracker::DebugSnapshot& dbg)
    {
        if (stage != 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ensureOpen()) return;

        file_ << seq << ',';
        writeCsvValue(file_, timestamp_ms);
        file_ << ',' << stage << ',';
        writeCsvValue(file_, dbg.observed_roll);
        for (int i = 0; i < 5; ++i) {
            const auto& b = dbg.blade_observations[i];
            file_ << ',' << (b.present ? 1 : 0)
                  << ',' << (b.solved  ? 1 : 0)
                  << ',';
            writeCsvValue(file_, b.assoc_global_roll_rad);
            file_ << ',' << (b.selected ? 1 : 0);
        }
        file_ << ',' << dbg.phase_origin_index
              << ',' << dbg.history_size
              << ',';
        writeCsvValue(file_, dbg.filtered_roll);
        file_ << ',';
        writeCsvValue(file_, dbg.filtered_speed);
        file_ << ',';
        writeCsvValue(file_, dbg.curve_speed_now);
        file_ << ',';
        writeCsvValue(file_, dbg.fit_a);
        file_ << ',';
        writeCsvValue(file_, dbg.fit_w);
        file_ << ',';
        writeCsvValue(file_, dbg.fit_phi);
        file_ << '\n';
        flushPeriodically();
    }

private:
    bool ensureOpen()
    {
        if (checked_) return file_.is_open();
        checked_ = true;
        if (!envFlagEnabled("BUFF_OBSERVATION_DUMP_LOG")) return false;

        const char* env_path = std::getenv("BUFF_OBSERVATION_DUMP_PATH");
        const std::filesystem::path path =
            (env_path != nullptr && env_path[0] != '\0')
                ? std::filesystem::path(env_path)
                : std::filesystem::current_path() / "build/buff_observation_dump.csv";

        std::error_code ec;
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path(), ec);
        file_.open(path, std::ios::out | std::ios::trunc);
        if (!file_.is_open()) {
            tools::logger()->warn("Failed to open buff observation dump: {}", path.string());
            return false;
        }
        file_ << "seq,time_ms,stage,observed_roll_rad";
        for (int i = 0; i < 5; ++i) {
            file_ << ",blade" << i << "_present"
                  << ",blade" << i << "_solved"
                  << ",blade" << i << "_assoc_global_roll_rad"
                  << ",blade" << i << "_selected";
        }
        file_ << ",phase_origin_index,history_size"
              << ",filtered_roll_rad,filtered_speed_rad_s,curve_speed_rad_s"
              << ",fit_a,fit_w,fit_phi\n";
        tools::logger()->info("Buff observation dump path: {}", path.string());
        return true;
    }

    void flushPeriodically()
    {
        if (++unflushed_rows_ >= kCsvFlushEveryRows) {
            file_.flush();
            unflushed_rows_ = 0;
        }
    }

    bool checked_ = false;
    std::ofstream file_;
    std::mutex mutex_;
    int unflushed_rows_ = 0;
};

ObservationDumpCsvLogger& observationDumpCsvLogger()
{
    static ObservationDumpCsvLogger logger;
    return logger;
}

class PredictionCsvLogger
{
public:
    void log(
        const rm::Frame& frame,
        const std::optional<auto_buff::PowerRune>& rune,
        auto_buff::BuffTracker& tracker,
        const auto_buff::Solver& solver,
        const auto_buff::Aimer& aimer,
        const auto_buff::AimCommand& command,
        const auto_buff::BuffShotGateSnapshot& shot_gate,
        bool switch_deferred,
        bool target_switched,
        int stage,
        double bullet_speed,
        double pipeline_delay_s,
        double base_predict_time_s,
        double fly_time_s)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool write_prediction_log = ensureOpen();
        const bool write_error_log =
            envFlagEnabled("BUFF_PREDICTION_LOG") || envFlagEnabled("BUFF_PREDICTION_ERROR_LOG");
        if (!write_prediction_log && !write_error_log) return;

        const uint64_t seq = seq_++;
        const double timestamp_ms = frameTimeMs(frame);
        const bool detected = rune.has_value() && !rune->fanblades.empty() && !rune->is_unsolve();
        if (stage == 0 && std::isfinite(last_stage0_time_ms_) && timestamp_ms + 1.0 < last_stage0_time_ms_) {
            historical_fixed200_seeds_.clear();
            last_stage0_detected_time_ms_ = nanValue();
        }
        double stage0_time_since_detection_ms = nanValue();
        if (stage == 0) {
            if (detected) {
                last_stage0_detected_time_ms_ = timestamp_ms;
            }
            stage0_time_since_detection_ms =
                std::isfinite(last_stage0_detected_time_ms_)
                    ? timestamp_ms - last_stage0_detected_time_ms_
                    : nanValue();
            last_stage0_time_ms_ = timestamp_ms;
        }
        const bool tracker_lost = tracker.is_lost();
        const double total_dt_s = base_predict_time_s + (command.control ? fly_time_s : 0.0);
        const double safe_dt_s = std::clamp(total_dt_s, 0.0, 1.0);
        const auto tracker_debug = tracker.debugSnapshot(safe_dt_s);
        const double target_roll_offset = tracker_debug.selected_roll_offset;
        const bool log_switch_deferred = switch_deferred || tracker_debug.switch_deferred;
        const bool log_target_switched =
            target_switched || tracker_debug.target_switched ||
            (detected && rune->target_switched);
        const auto& pitch_debug = aimer.get_last_pitch_debug();
        const Eigen::Vector3d debug_target_ypd =
            pitch_debug.target_in_world.array().isFinite().all()
                ? safeYpdFromWorldPoint(pitch_debug.target_in_world)
                : Eigen::Vector3d::Constant(nanValue());

        cv::Point2f actual_center(nanValue(), nanValue());
        cv::Point2f actual_r(nanValue(), nanValue());
        Eigen::Vector3d actual_rune_ypd = Eigen::Vector3d::Constant(nanValue());
        Eigen::Vector3d actual_blade_ypd = Eigen::Vector3d::Constant(nanValue());
        double actual_roll = nanValue();
        double pnp_reproj_error_px = nanValue();
        double pnp_score = nanValue();
        int pnp_method = -1;
        std::array<int, 4> pnp_order = {0, 1, 2, 3};
        std::array<double, 5> pnp_point_errors_px = {
            nanValue(), nanValue(), nanValue(), nanValue(), nanValue()};
        cv::Point2f pnp_model_center(nanValue(), nanValue());
        double pnp_model_center_error_px = nanValue();
        double pnp_model_center_radial_error_px = nanValue();
        double pnp_model_center_tangent_error_px = nanValue();

        if (detected) {
            const auto& target = rune->target();
            actual_center = target.center;
            actual_r = rune->r_center;
            actual_rune_ypd = rune->ypd_in_world;
            actual_blade_ypd = rune->blade_ypd_in_world;
            actual_roll = rune->ypr_in_world[2];
            pnp_reproj_error_px = target.pnp_reproj_error_px;
            pnp_score = target.pnp_score;
            pnp_method = target.pnp_method;
            pnp_order = target.pnp_order;
            for (size_t i = 0; i < pnp_point_errors_px.size(); ++i) {
                if (target.pnp_point_errors_px.size() > i) {
                    pnp_point_errors_px[i] = target.pnp_point_errors_px[i];
                }
            }
            pnp_model_center = target.pnp_model_center;
            pnp_model_center_error_px = target.pnp_model_center_error_px;
            pnp_model_center_radial_error_px = target.pnp_model_center_radial_error_px;
            pnp_model_center_tangent_error_px = target.pnp_model_center_tangent_error_px;
        }

        Eigen::VectorXd current_state;
        Eigen::VectorXd predicted_state;
        Eigen::VectorXd fixed_100ms_state;
        Eigen::VectorXd fixed_200ms_state;
        double fixed_200ms_target_roll_offset = target_roll_offset;
        cv::Point2f filtered_center(nanValue(), nanValue());
        cv::Point2f predicted_center(nanValue(), nanValue());
        cv::Point2f fixed_100ms_center(nanValue(), nanValue());
        cv::Point2f fixed_200ms_center(nanValue(), nanValue());
        Eigen::Vector3d filtered_blade = Eigen::Vector3d::Constant(nanValue());
        Eigen::Vector3d predicted_blade = Eigen::Vector3d::Constant(nanValue());
        Eigen::Vector3d fixed_100ms_blade = Eigen::Vector3d::Constant(nanValue());
        Eigen::Vector3d fixed_200ms_blade = Eigen::Vector3d::Constant(nanValue());
        Eigen::Vector3d filtered_blade_ypd = Eigen::Vector3d::Constant(nanValue());
        Eigen::Vector3d predicted_blade_ypd = Eigen::Vector3d::Constant(nanValue());
        Eigen::Vector3d fixed_100ms_blade_ypd = Eigen::Vector3d::Constant(nanValue());
        Eigen::Vector3d fixed_200ms_blade_ypd = Eigen::Vector3d::Constant(nanValue());
        double filtered_radial_error_px = nanValue();
        double filtered_tangent_error_px = nanValue();
        double predicted_radial_error_px = nanValue();
        double predicted_tangent_error_px = nanValue();
        double fixed_100ms_radial_error_px = nanValue();
        double fixed_100ms_tangent_error_px = nanValue();
        double fixed_200ms_radial_error_px = nanValue();
        double fixed_200ms_tangent_error_px = nanValue();
        double observed_roll_delta = rollDifference(tracker_debug.observed_roll, last_observed_roll_);
        double filtered_roll_delta = rollDifference(tracker_debug.filtered_roll, last_filtered_roll_);
        double predicted_roll_delta = rollDifference(tracker_debug.predicted_roll, last_predicted_roll_);
        // Legacy CSV contract:
        // - observed_speed keeps the tracker-side measurement used for updates.
        // - observed_speed_raw keeps the single-frame clamped derivative used by HUD.
        double observed_speed_raw_delta =
            std::isfinite(tracker_debug.observed_speed_raw) &&
                std::isfinite(last_observed_speed_raw_)
                ? tracker_debug.observed_speed_raw - last_observed_speed_raw_
                : nanValue();
        double observed_speed_delta =
            std::isfinite(tracker_debug.observed_speed) && std::isfinite(last_observed_speed_)
                ? tracker_debug.observed_speed - last_observed_speed_
                : nanValue();
        double filtered_speed_delta =
            std::isfinite(tracker_debug.filtered_speed) && std::isfinite(last_filtered_speed_)
                ? tracker_debug.filtered_speed - last_filtered_speed_
                : nanValue();
        double filtered_speed_raw_delta =
            std::isfinite(tracker_debug.filtered_speed_raw) &&
                std::isfinite(last_filtered_speed_raw_)
                ? tracker_debug.filtered_speed_raw - last_filtered_speed_raw_
                : nanValue();
        double predicted_speed_delta =
            std::isfinite(tracker_debug.predicted_speed) && std::isfinite(last_predicted_speed_)
                ? tracker_debug.predicted_speed - last_predicted_speed_
                : nanValue();
        double curve_speed_delta =
            std::isfinite(tracker_debug.curve_speed_now) && std::isfinite(last_curve_speed_now_)
                ? tracker_debug.curve_speed_now - last_curve_speed_now_
                : nanValue();
        double curve_speed_raw_delta =
            std::isfinite(tracker_debug.curve_speed_raw) && std::isfinite(last_curve_speed_raw_)
                ? tracker_debug.curve_speed_raw - last_curve_speed_raw_
                : nanValue();
        double curve_speed_after_predict_delta =
            std::isfinite(tracker_debug.curve_speed_after_predict) &&
                std::isfinite(last_curve_speed_after_predict_)
                ? tracker_debug.curve_speed_after_predict - last_curve_speed_after_predict_
                : nanValue();
        double curve_speed_after_blade_delta =
            std::isfinite(tracker_debug.curve_speed_after_blade_update) &&
                std::isfinite(last_curve_speed_after_blade_update_)
                ? tracker_debug.curve_speed_after_blade_update -
                    last_curve_speed_after_blade_update_
                : nanValue();
        double curve_speed_after_speed_delta =
            std::isfinite(tracker_debug.curve_speed_after_speed_update) &&
                std::isfinite(last_curve_speed_after_speed_update_)
                ? tracker_debug.curve_speed_after_speed_update -
                    last_curve_speed_after_speed_update_
                : nanValue();
        double curve_phi_after_predict_delta =
            rollDifference(
                tracker_debug.curve_phi_after_predict,
                last_curve_phi_after_predict_);
        double curve_phi_after_blade_delta =
            rollDifference(
                tracker_debug.curve_phi_after_blade_update,
                last_curve_phi_after_blade_update_);
        double curve_phi_after_speed_delta =
            rollDifference(
                tracker_debug.curve_phi_after_speed_update,
                last_curve_phi_after_speed_update_);
        double curve_phi_blade_correction =
            rollDifference(
                tracker_debug.curve_phi_after_blade_update,
                tracker_debug.curve_phi_after_predict);
        double curve_phi_speed_correction =
            rollDifference(
                tracker_debug.curve_phi_after_speed_update,
                tracker_debug.curve_phi_before_speed_update);
        double curve_speed_blade_correction =
            std::isfinite(tracker_debug.curve_speed_after_blade_update) &&
                std::isfinite(tracker_debug.curve_speed_after_predict)
                ? tracker_debug.curve_speed_after_blade_update -
                    tracker_debug.curve_speed_after_predict
                : nanValue();
        double curve_speed_measurement_correction =
            std::isfinite(tracker_debug.curve_speed_after_speed_update) &&
                std::isfinite(tracker_debug.curve_speed_before_speed_update)
                ? tracker_debug.curve_speed_after_speed_update -
                    tracker_debug.curve_speed_before_speed_update
                : nanValue();
        double fit_a_delta =
            std::isfinite(tracker_debug.fit_a) && std::isfinite(last_fit_a_)
                ? tracker_debug.fit_a - last_fit_a_
                : nanValue();
        double fit_w_delta =
            std::isfinite(tracker_debug.fit_w) && std::isfinite(last_fit_w_)
                ? tracker_debug.fit_w - last_fit_w_
                : nanValue();
        double fit_phi_delta = rollDifference(tracker_debug.fit_phi, last_fit_phi_);
        std::optional<HistoricalPredictionMatch> fixed_200ms_match;

        if (!tracker_lost) {
            current_state = tracker.get_state();
            if (current_state.size() >= 10 && current_state.array().isFinite().all()) {
                const auto projectPredictedState =
                    [&](const Eigen::VectorXd& state,
                        double state_target_roll_offset,
                        cv::Point2f* center,
                        Eigen::Vector3d* blade_world,
                        Eigen::Vector3d* blade_ypd) {
                        if (center == nullptr || blade_world == nullptr || blade_ypd == nullptr) {
                            return;
                        }
                        if (state.size() < 10 || !state.array().isFinite().all()) {
                            return;
                        }
                        const Eigen::Vector3d center_world =
                            tracker.point_buff2world(Eigen::Vector3d::Zero(), state);
                        *blade_world = tracker.point_buff2world(
                            Eigen::Vector3d(0.0, 0.0, 0.7), state, state_target_roll_offset);
                        *blade_ypd = safeYpdFromWorldPoint(*blade_world);
                        if (center_world.array().isFinite().all()) {
                            const auto points = solver.reproject_buff(
                                center_world,
                                state[4],
                                state[5] + state_target_roll_offset);
                            if (points.size() > 4) {
                                *center = points[4];
                            }
                        }
                    };

                predicted_state = tracker.predict(safe_dt_s);
                fixed_100ms_state = tracker.predict(kFixed100PredictionEvalDtS);
                if (stage == 0) {
                    fixed_200ms_match = buildHistoricalPredictionMatch(timestamp_ms, tracker);
                    if (fixed_200ms_match.has_value()) {
                        fixed_200ms_state = fixed_200ms_match->predicted_state;
                        fixed_200ms_target_roll_offset =
                            fixed_200ms_match->seed.target_roll_offset;
                    } else {
                        fixed_200ms_state = tracker.predict(kFixed200PredictionEvalDtS);
                    }
                } else {
                    fixed_200ms_state = tracker.predict(kFixed200PredictionEvalDtS);
                }

                const Eigen::Vector3d filtered_center_world =
                    tracker.point_buff2world(Eigen::Vector3d::Zero(), current_state);
                filtered_blade =
                    tracker.target_point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7), current_state);
                filtered_blade_ypd = safeYpdFromWorldPoint(filtered_blade);

                if (filtered_center_world.array().isFinite().all()) {
                    const auto points = solver.reproject_buff(
                        filtered_center_world,
                        current_state[4],
                        current_state[5] + target_roll_offset);
                    if (points.size() > 4) {
                        filtered_center = points[4];
                    }
                }

                projectPredictedState(
                    predicted_state,
                    target_roll_offset,
                    &predicted_center,
                    &predicted_blade,
                    &predicted_blade_ypd);
                projectPredictedState(
                    fixed_100ms_state,
                    target_roll_offset,
                    &fixed_100ms_center,
                    &fixed_100ms_blade,
                    &fixed_100ms_blade_ypd);
                projectPredictedState(
                    fixed_200ms_state,
                    fixed_200ms_target_roll_offset,
                    &fixed_200ms_center,
                    &fixed_200ms_blade,
                    &fixed_200ms_blade_ypd);
            }
        }

        if (detected && isFinitePoint(actual_center) && isFinitePoint(actual_r)) {
            const auto filtered_error =
                radialTangentialError(filtered_center, actual_center, actual_r);
            filtered_radial_error_px = filtered_error.first;
            filtered_tangent_error_px = filtered_error.second;
            const auto predicted_error =
                radialTangentialError(predicted_center, actual_center, actual_r);
            predicted_radial_error_px = predicted_error.first;
            predicted_tangent_error_px = predicted_error.second;
            const auto fixed_100ms_error =
                radialTangentialError(fixed_100ms_center, actual_center, actual_r);
            fixed_100ms_radial_error_px = fixed_100ms_error.first;
            fixed_100ms_tangent_error_px = fixed_100ms_error.second;
            const auto fixed_200ms_error =
                radialTangentialError(fixed_200ms_center, actual_center, actual_r);
            fixed_200ms_radial_error_px = fixed_200ms_error.first;
            fixed_200ms_tangent_error_px = fixed_200ms_error.second;
        }

        predictionErrorCsvLogger().observeAndQueue(
            seq,
            "dynamic",
            stage,
            timestamp_ms,
            timestamp_ms + safe_dt_s * 1000.0,
            safe_dt_s,
            detected,
            tracker_lost,
            target_roll_offset,
            tracker_debug.selected_phase_index,
            tracker_debug.direction,
            tracker_debug.history_size,
            log_switch_deferred,
            log_target_switched,
            tracker_debug.reinit_reason,
            rune,
            predicted_center,
            predicted_blade_ypd,
            predicted_state.size() > 5 ? predicted_state[5] : nanValue(),
            predicted_state.size() > 6 ? predicted_state[6] : nanValue(),
            predicted_state.size() > 7 ? predicted_state[7] : nanValue(),
            predicted_state.size() > 8 ? predicted_state[8] : nanValue(),
            predicted_state.size() > 9 ? predicted_state[9] : nanValue());
        predictionErrorCsvLogger().observeAndQueue(
            seq,
            "fixed_100ms",
            stage,
            timestamp_ms,
            timestamp_ms + kFixed100PredictionEvalDtS * 1000.0,
            kFixed100PredictionEvalDtS,
            detected,
            tracker_lost,
            target_roll_offset,
            tracker_debug.selected_phase_index,
            tracker_debug.direction,
            tracker_debug.history_size,
            log_switch_deferred,
            log_target_switched,
            tracker_debug.reinit_reason,
            rune,
            fixed_100ms_center,
            fixed_100ms_blade_ypd,
            fixed_100ms_state.size() > 5 ? fixed_100ms_state[5] : nanValue(),
            fixed_100ms_state.size() > 6 ? fixed_100ms_state[6] : nanValue(),
            fixed_100ms_state.size() > 7 ? fixed_100ms_state[7] : nanValue(),
            fixed_100ms_state.size() > 8 ? fixed_100ms_state[8] : nanValue(),
            fixed_100ms_state.size() > 9 ? fixed_100ms_state[9] : nanValue());
        if (stage == 0 && fixed_200ms_match.has_value()) {
            const auto& match = *fixed_200ms_match;
            predictionErrorCsvLogger().logCurrent(
                match.seed.seq,
                "fixed_200ms",
                match.seed.time_ms,
                timestamp_ms,
                match.eval_dt_s,
                match.seed.pred_detected,
                match.seed.pred_tracker_lost,
                match.seed.pred_time_since_detection_ms,
                match.seed.target_roll_offset,
                match.seed.pred_selected_phase_index,
                match.seed.pred_voter_direction,
                match.seed.pred_history_size,
                match.seed.pred_switch_deferred,
                match.seed.pred_target_switched,
                match.seed.pred_reinit_reason,
                rune,
                fixed_200ms_center,
                fixed_200ms_blade_ypd,
                fixed_200ms_state.size() > 5 ? fixed_200ms_state[5] : nanValue(),
                fixed_200ms_state.size() > 6 ? fixed_200ms_state[6] : nanValue(),
                fixed_200ms_state.size() > 7 ? fixed_200ms_state[7] : nanValue(),
                fixed_200ms_state.size() > 8 ? fixed_200ms_state[8] : nanValue(),
                fixed_200ms_state.size() > 9 ? fixed_200ms_state[9] : nanValue());
        } else {
            predictionErrorCsvLogger().observeAndQueue(
                seq,
                "fixed_200ms",
                stage,
                timestamp_ms,
                timestamp_ms + kFixed200PredictionEvalDtS * 1000.0,
                kFixed200PredictionEvalDtS,
                detected,
                tracker_lost,
                target_roll_offset,
                tracker_debug.selected_phase_index,
                tracker_debug.direction,
                tracker_debug.history_size,
                log_switch_deferred,
                log_target_switched,
                tracker_debug.reinit_reason,
                rune,
                fixed_200ms_center,
                fixed_200ms_blade_ypd,
                fixed_200ms_state.size() > 5 ? fixed_200ms_state[5] : nanValue(),
                fixed_200ms_state.size() > 6 ? fixed_200ms_state[6] : nanValue(),
                fixed_200ms_state.size() > 7 ? fixed_200ms_state[7] : nanValue(),
                fixed_200ms_state.size() > 8 ? fixed_200ms_state[8] : nanValue(),
                fixed_200ms_state.size() > 9 ? fixed_200ms_state[9] : nanValue());
        }

        if (!write_prediction_log) return;

        file_ << seq << ','
              << std::fixed << std::setprecision(6);
        writeCsvValue(file_, timestamp_ms);
        file_ << ',' << stage
              << ',' << static_cast<int>(frame.fb.task_mode)
              << ',' << (detected ? 1 : 0)
              << ',' << (tracker_lost ? 1 : 0)
              << ',' << (log_switch_deferred ? 1 : 0)
              << ',' << (log_target_switched ? 1 : 0)
              << ',';
        writeCsvValue(file_, target_roll_offset);
        file_ << ',' << tracker_debug.selected_phase_index
              << ',' << tracker_debug.direction
              << ',' << tracker_debug.history_size
              << ',' << tracker_debug.reinit_reason
              << ',' << (tracker_debug.fit_valid ? 1 : 0)
              << ',';
        writeCsvValue(file_, tracker_debug.observed_roll);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.observed_speed);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.observed_speed_raw);
        file_ << ',';
        writeCsvValue(file_, observed_roll_delta);
        file_ << ',';
        writeCsvValue(file_, observed_speed_delta);
        file_ << ',';
        writeCsvValue(file_, observed_speed_raw_delta);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.filtered_roll);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.filtered_speed);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.filtered_speed_raw);
        file_ << ',';
        writeCsvValue(file_, filtered_roll_delta);
        file_ << ',';
        writeCsvValue(file_, filtered_speed_delta);
        file_ << ',';
        writeCsvValue(file_, filtered_speed_raw_delta);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.predicted_roll);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.predicted_speed);
        file_ << ',';
        writeCsvValue(file_, predicted_roll_delta);
        file_ << ',';
        writeCsvValue(file_, predicted_speed_delta);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_speed_now);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_speed_raw);
        file_ << ',';
        writeCsvValue(file_, curve_speed_delta);
        file_ << ',';
        writeCsvValue(file_, curve_speed_raw_delta);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_speed_after_predict);
        file_ << ',';
        writeCsvValue(file_, curve_speed_after_predict_delta);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_speed_after_blade_update);
        file_ << ',';
        writeCsvValue(file_, curve_speed_after_blade_delta);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_speed_before_speed_update);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_speed_after_speed_update);
        file_ << ',';
        writeCsvValue(file_, curve_speed_after_speed_delta);
        file_ << ',';
        writeCsvValue(file_, curve_speed_blade_correction);
        file_ << ',';
        writeCsvValue(file_, curve_speed_measurement_correction);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_phi_after_predict);
        file_ << ',';
        writeCsvValue(file_, curve_phi_after_predict_delta);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_phi_after_blade_update);
        file_ << ',';
        writeCsvValue(file_, curve_phi_after_blade_delta);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_phi_before_speed_update);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.curve_phi_after_speed_update);
        file_ << ',';
        writeCsvValue(file_, curve_phi_after_speed_delta);
        file_ << ',';
        writeCsvValue(file_, curve_phi_blade_correction);
        file_ << ',';
        writeCsvValue(file_, curve_phi_speed_correction);
        file_ << ',' << tracker_debug.speed_measurement_status << ',';
        writeCsvValue(file_, tracker_debug.speed_measurement_predicted);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.speed_measurement_innovation);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.speed_measurement_noise);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.fit_a);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.fit_w);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.fit_phi);
        file_ << ',';
        writeCsvValue(file_, fit_a_delta);
        file_ << ',';
        writeCsvValue(file_, fit_w_delta);
        file_ << ',';
        writeCsvValue(file_, fit_phi_delta);
        file_ << ',';
        writeCsvValue(file_, tracker_debug.selected_roll_offset);
        file_ << ',' << (shot_gate.allowed ? 1 : 0)
              << ',' << shot_gate.reason_code
              << ',' << shot_gate.stable_frames
              << ',' << (shot_gate.pending_detected ? 1 : 0)
              << ',' << (shot_gate.r_center_ok ? 1 : 0)
              << ',' << (shot_gate.pnp_ok ? 1 : 0)
              << ',' << (shot_gate.tracker_ok ? 1 : 0)
              << ',' << (shot_gate.gimbal_ok ? 1 : 0)
              << ',' << (shot_gate.stable_ok ? 1 : 0)
              << ',';
        writeCsvValue(file_, shot_gate.yaw_error_deg);
        file_ << ',';
        writeCsvValue(file_, shot_gate.pitch_error_deg);
        file_ << ',';
        writeCsvValue(file_, shot_gate.pnp_reproj_error_px);
        file_ << ',';
        writeCsvValue(file_, shot_gate.pnp_model_center_error_px);
        file_ << ',' << (command.control ? 1 : 0)
              << ',' << (command.shoot ? 1 : 0) << ',';
        writeCsvValue(file_, bullet_speed);
        file_ << ',';
        writeCsvValue(file_, pipeline_delay_s);
        file_ << ',';
        writeCsvValue(file_, base_predict_time_s);
        file_ << ',';
        writeCsvValue(file_, command.control ? fly_time_s : 0.0);
        file_ << ',';
        writeCsvValue(file_, safe_dt_s);
        file_ << ',';
        writeCsvValue(file_, timestamp_ms + safe_dt_s * 1000.0);
        file_ << ',';
        writeCsvValue(file_, command.yaw);
        file_ << ',';
        writeCsvValue(file_, command.pitch);
        file_ << ',';
        writeCsvValue(file_, frame.poseEuler.yaw);
        file_ << ',';
        writeCsvValue(file_, frame.poseEuler.pitch);
        file_ << ',';
        writeCsvValue(file_, frame.poseEuler.roll);
        file_ << ',';
        writeCsvValue(file_, frame.fb.yaw_speed);
        file_ << ',' << (pitch_debug.valid ? 1 : 0) << ',';
        writeCsvVector(file_, pitch_debug.target_in_buff);
        file_ << ',';
        writeCsvVector(file_, pitch_debug.target_in_world);
        file_ << ',';
        writeCsvVector(file_, debug_target_ypd);
        file_ << ',';
        writeCsvValue(file_, pitch_debug.horizontal_distance);
        file_ << ',';
        writeCsvValue(file_, pitch_debug.height);
        file_ << ',';
        writeCsvValue(file_, pitch_debug.raw_yaw);
        file_ << ',';
        writeCsvValue(file_, pitch_debug.ballistic_pitch);
        file_ << ',';
        writeCsvValue(file_, pitch_debug.pitch_offset);
        file_ << ',';
        writeCsvValue(file_, pitch_debug.pitch_rate);
        file_ << ',';
        writeCsvValue(file_, pitch_debug.pitch_lead);
        file_ << ',';
        writeCsvValue(file_, pitch_debug.solved_pitch);
        file_ << ',';
        writeCsvValue(file_, pitch_debug.command_pitch);
        file_ << ',';
        writeCsvPoint(file_, actual_center);
        file_ << ',';
        writeCsvPoint(file_, actual_r);
        file_ << ',';
        writeCsvVector(file_, actual_rune_ypd);
        file_ << ',';
        writeCsvValue(file_, actual_roll);
        file_ << ',';
        writeCsvVector(file_, actual_blade_ypd);
        file_ << ',';
        writeCsvValue(file_, pnp_reproj_error_px);
        file_ << ',';
        writeCsvValue(file_, pnp_score);
        file_ << ',' << pnp_method
              << ',' << pnp_order[0]
              << ',' << pnp_order[1]
              << ',' << pnp_order[2]
              << ',' << pnp_order[3]
              << ',';
        for (size_t i = 0; i < pnp_point_errors_px.size(); ++i) {
            if (i > 0) file_ << ',';
            writeCsvValue(file_, pnp_point_errors_px[i]);
        }
        file_ << ',';
        writeCsvPoint(file_, pnp_model_center);
        file_ << ',';
        writeCsvValue(file_, pnp_model_center_error_px);
        file_ << ',';
        writeCsvValue(file_, pnp_model_center_radial_error_px);
        file_ << ',';
        writeCsvValue(file_, pnp_model_center_tangent_error_px);
        file_ << ',';
        writeCsvPoint(file_, filtered_center);
        file_ << ',';
        writeCsvValue(file_, filtered_radial_error_px);
        file_ << ',';
        writeCsvValue(file_, filtered_tangent_error_px);
        file_ << ',';
        writeStatePrefix(current_state, 10);
        file_ << ',';
        writeCsvVector(file_, filtered_blade_ypd);
        file_ << ',';
        writeCsvPoint(file_, predicted_center);
        file_ << ',';
        writeCsvValue(file_, predicted_radial_error_px);
        file_ << ',';
        writeCsvValue(file_, predicted_tangent_error_px);
        file_ << ',';
        writeStatePrefix(predicted_state, 10);
        file_ << ',';
        writeCsvVector(file_, predicted_blade);
        file_ << ',';
        writeCsvVector(file_, predicted_blade_ypd);
        file_ << ',';
        writeCsvPoint(file_, fixed_100ms_center);
        file_ << ',';
        writeCsvValue(file_, fixed_100ms_radial_error_px);
        file_ << ',';
        writeCsvValue(file_, fixed_100ms_tangent_error_px);
        file_ << ',';
        writeCsvVector(file_, fixed_100ms_blade_ypd);
        file_ << ',';
        writeCsvPoint(file_, fixed_200ms_center);
        file_ << ',';
        writeCsvValue(file_, fixed_200ms_radial_error_px);
        file_ << ',';
        writeCsvValue(file_, fixed_200ms_tangent_error_px);
        file_ << ',';
        writeCsvVector(file_, fixed_200ms_blade_ypd);
        file_ << '\n';
        flushPeriodically();
        updateLastDebugState(tracker_debug);
        observationDumpCsvLogger().log(seq, timestamp_ms, stage, tracker_debug);
        if (stage == 0) {
            enqueueHistoricalPredictionSeed(
                seq,
                timestamp_ms,
                tracker,
                current_state,
                tracker_debug,
                detected,
                tracker_lost,
                stage0_time_since_detection_ms,
                log_switch_deferred,
                log_target_switched);
        }
    }

private:
    bool ensureOpen()
    {
        if (checked_) return file_.is_open();
        checked_ = true;

        if (!envFlagEnabled("BUFF_PREDICTION_LOG")) return false;

        const std::filesystem::path path = resolvePredictionLogPath();
        std::error_code ec;
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path(), ec);
        }
        file_.open(path, std::ios::out | std::ios::trunc);
        if (!file_.is_open()) {
            tools::logger()->warn("Failed to open buff prediction log: {}", path.string());
            return false;
        }

	        file_ << "seq,time_ms,stage,task_mode,detected,tracker_lost,"
	              << "switch_deferred,target_switched,selected_target_roll_offset,"
              << "selected_phase_index,voter_direction,history_size,reinit_reason,"
              << "fit_valid,tracker_observed_roll,"
              << "tracker_observed_speed,tracker_observed_speed_raw,"
              << "tracker_observed_roll_delta,tracker_observed_speed_delta,"
              << "tracker_observed_speed_raw_delta,"
              << "tracker_filtered_roll,tracker_filtered_speed,tracker_filtered_speed_raw,"
              << "tracker_filtered_roll_delta,tracker_filtered_speed_delta,"
              << "tracker_filtered_speed_raw_delta,"
              << "tracker_predicted_roll,tracker_predicted_speed,"
              << "tracker_predicted_roll_delta,tracker_predicted_speed_delta,"
              << "tracker_curve_speed_now,tracker_curve_speed_raw,"
              << "tracker_curve_speed_delta,tracker_curve_speed_raw_delta,"
              << "tracker_curve_speed_after_predict,tracker_curve_speed_after_predict_delta,"
              << "tracker_curve_speed_after_blade_update,"
              << "tracker_curve_speed_after_blade_update_delta,"
              << "tracker_curve_speed_before_speed_update,"
              << "tracker_curve_speed_after_speed_update,"
              << "tracker_curve_speed_after_speed_update_delta,"
              << "tracker_curve_speed_blade_correction,"
              << "tracker_curve_speed_measurement_correction,"
              << "tracker_curve_phi_after_predict,tracker_curve_phi_after_predict_delta,"
              << "tracker_curve_phi_after_blade_update,"
              << "tracker_curve_phi_after_blade_update_delta,"
              << "tracker_curve_phi_before_speed_update,"
              << "tracker_curve_phi_after_speed_update,"
              << "tracker_curve_phi_after_speed_update_delta,"
              << "tracker_curve_phi_blade_correction,"
              << "tracker_curve_phi_speed_correction,"
              << "tracker_speed_measurement_status,"
              << "tracker_speed_measurement_predicted,"
              << "tracker_speed_measurement_innovation,"
              << "tracker_speed_measurement_noise,"
              << "tracker_fit_a,tracker_fit_w,tracker_fit_phi,"
              << "tracker_fit_a_delta,tracker_fit_w_delta,tracker_fit_phi_delta,"
              << "tracker_selected_roll_offset,"
              << "shot_gate_allowed,shot_gate_reason,shot_gate_stable_frames,"
              << "shot_gate_pending,shot_gate_r_center,shot_gate_pnp,"
              << "shot_gate_tracker,shot_gate_gimbal,shot_gate_stable,"
              << "shot_gate_yaw_error_deg,shot_gate_pitch_error_deg,"
              << "shot_gate_pnp_reproj_error_px,shot_gate_model_center_error_px,"
	              << "control,shoot,bullet_speed,"
              << "pipeline_delay_s,base_predict_time_s,fly_time_s,total_dt_s,target_time_ms,"
              << "cmd_yaw_rad,cmd_pitch_rad,"
              << "frame_yaw_deg,frame_pitch_deg,frame_roll_deg,frame_yaw_speed_deg_s,"
              << "pitch_debug_valid,"
              << "dbg_target_buff_x_m,dbg_target_buff_y_m,dbg_target_buff_z_m,"
              << "dbg_target_world_x_m,dbg_target_world_y_m,dbg_target_world_z_m,"
              << "dbg_target_yaw_rad,dbg_target_pitch_rad,dbg_target_dist_m,"
              << "dbg_horizontal_m,dbg_height_m,dbg_raw_yaw_rad,dbg_ballistic_pitch_rad,"
              << "dbg_pitch_offset_rad,dbg_pitch_rate_rad_s,dbg_pitch_lead_rad,"
              << "dbg_solved_pitch_rad,dbg_command_pitch_rad,"
              << "actual_px_x,actual_px_y,actual_r_px_x,actual_r_px_y,"
              << "actual_rune_yaw_rad,actual_rune_pitch_rad,actual_rune_dist_m,actual_roll_rad,"
              << "actual_blade_yaw_rad,actual_blade_pitch_rad,actual_blade_dist_m,"
              << "pnp_reproj_error_px,pnp_score,pnp_method,"
              << "pnp_order0,pnp_order1,pnp_order2,pnp_order3,"
              << "pnp_err_top_px,pnp_err_left_px,pnp_err_bottom_px,pnp_err_right_px,pnp_err_r_px,"
              << "pnp_model_center_px_x,pnp_model_center_px_y,"
              << "pnp_model_center_error_px,pnp_model_center_radial_error_px,"
              << "pnp_model_center_tangent_error_px,"
              << "filtered_px_x,filtered_px_y,filtered_radial_error_px,filtered_tangent_error_px,"
              << "filtered_rune_yaw_rad,filtered_yaw_rate_rad_s,filtered_rune_pitch_rad,"
              << "filtered_rune_dist_m,filtered_plane_yaw_rad,filtered_roll_rad,"
              << "filtered_omega_rad_s,filtered_big_a,filtered_big_w,filtered_big_phi,"
              << "filtered_blade_yaw_rad,filtered_blade_pitch_rad,filtered_blade_dist_m,"
              << "pred_px_x,pred_px_y,pred_radial_error_px,pred_tangent_error_px,"
              << "pred_rune_yaw_rad,pred_yaw_rate_rad_s,pred_rune_pitch_rad,pred_rune_dist_m,"
              << "pred_plane_yaw_rad,pred_roll_rad,pred_omega_rad_s,pred_big_a,pred_big_w,pred_big_phi,"
              << "pred_blade_x_m,pred_blade_y_m,pred_blade_z_m,"
              << "pred_blade_yaw_rad,pred_blade_pitch_rad,pred_blade_dist_m,"
              << "fixed100_px_x,fixed100_px_y,fixed100_radial_error_px,fixed100_tangent_error_px,"
              << "fixed100_blade_yaw_rad,fixed100_blade_pitch_rad,fixed100_blade_dist_m,"
              << "fixed200_px_x,fixed200_px_y,fixed200_radial_error_px,fixed200_tangent_error_px,"
              << "fixed200_blade_yaw_rad,fixed200_blade_pitch_rad,fixed200_blade_dist_m\n";
        tools::logger()->info("Buff prediction log path: {}", path.string());
        return true;
    }

    void writeStatePrefix(const Eigen::VectorXd& state, int count)
    {
        for (int i = 0; i < count; ++i) {
            if (i > 0) file_ << ',';
            if (state.size() > i) {
                writeCsvValue(file_, state[i]);
            } else {
                writeCsvValue(file_, nanValue());
            }
        }
    }

    void updateLastDebugState(const auto_buff::BuffTracker::DebugSnapshot& snapshot)
    {
        last_observed_roll_ = snapshot.observed_roll;
        last_filtered_roll_ = snapshot.filtered_roll;
        last_predicted_roll_ = snapshot.predicted_roll;
        last_observed_speed_raw_ = snapshot.observed_speed_raw;
        last_observed_speed_ = snapshot.observed_speed;
        last_filtered_speed_ = snapshot.filtered_speed;
        last_filtered_speed_raw_ = snapshot.filtered_speed_raw;
        last_predicted_speed_ = snapshot.predicted_speed;
        last_curve_speed_now_ = snapshot.curve_speed_now;
        last_curve_speed_raw_ = snapshot.curve_speed_raw;
        last_curve_speed_after_predict_ = snapshot.curve_speed_after_predict;
        last_curve_speed_after_blade_update_ = snapshot.curve_speed_after_blade_update;
        last_curve_speed_after_speed_update_ = snapshot.curve_speed_after_speed_update;
        last_curve_phi_after_predict_ = snapshot.curve_phi_after_predict;
        last_curve_phi_after_blade_update_ = snapshot.curve_phi_after_blade_update;
        last_curve_phi_after_speed_update_ = snapshot.curve_phi_after_speed_update;
        last_fit_a_ = snapshot.fit_a;
        last_fit_w_ = snapshot.fit_w;
        last_fit_phi_ = snapshot.fit_phi;
    }

    struct HistoricalPredictionSeed
    {
        uint64_t seq = 0;
        double time_ms = nanValue();
        double relative_time_s = nanValue();
        Eigen::VectorXd state;
        double target_roll_offset = nanValue();
        int pred_voter_direction = 0;
        bool pred_detected = false;
        bool pred_tracker_lost = true;
        double pred_time_since_detection_ms = nanValue();
        int pred_selected_phase_index = -1;
        int pred_history_size = 0;
        bool pred_switch_deferred = false;
        bool pred_target_switched = false;
        int pred_reinit_reason = 0;
    };

    struct HistoricalPredictionMatch
    {
        HistoricalPredictionSeed seed;
        double eval_dt_s = 0.0;
        Eigen::VectorXd predicted_state;
    };

    std::optional<HistoricalPredictionMatch> buildHistoricalPredictionMatch(
        double current_time_ms,
        auto_buff::BuffTracker& tracker) const
    {
        if (!std::isfinite(current_time_ms)) {
            return std::nullopt;
        }

        const double target_source_time_ms =
            current_time_ms - kFixed200PredictionEvalDtS * 1000.0;
        for (auto it = historical_fixed200_seeds_.rbegin();
             it != historical_fixed200_seeds_.rend();
             ++it) {
            if (!std::isfinite(it->time_ms) || it->time_ms > target_source_time_ms) {
                continue;
            }
            if (target_source_time_ms - it->time_ms > kHistoricalPredictionMaxSourceLagMs) {
                return std::nullopt;
            }
            if (!isGoodHistoricalPredictionSeed(
                    it->state,
                    it->pred_voter_direction,
                    it->pred_history_size,
                    it->pred_reinit_reason,
                    it->pred_switch_deferred,
                    it->pred_target_switched)) {
                continue;
            }

            HistoricalPredictionMatch match;
            match.seed = *it;
            match.eval_dt_s = std::max(0.0, (current_time_ms - it->time_ms) * 0.001);
            match.predicted_state = tracker.predict_from_state(
                it->state,
                match.eval_dt_s,
                it->pred_voter_direction,
                it->relative_time_s);
            if (match.predicted_state.size() < 10 ||
                !match.predicted_state.array().isFinite().all()) {
                return std::nullopt;
            }
            return match;
        }
        return std::nullopt;
    }

    void enqueueHistoricalPredictionSeed(
        uint64_t seq,
        double time_ms,
        auto_buff::BuffTracker& tracker,
        const Eigen::VectorXd& current_state,
        const auto_buff::BuffTracker::DebugSnapshot& tracker_debug,
        bool pred_detected,
        bool pred_tracker_lost,
        double pred_time_since_detection_ms,
        bool pred_switch_deferred,
        bool pred_target_switched)
    {
        if (!std::isfinite(time_ms) || pred_tracker_lost ||
            !isGoodHistoricalPredictionSeed(
                current_state,
                tracker_debug.direction,
                tracker_debug.history_size,
                tracker_debug.reinit_reason,
                pred_switch_deferred,
                pred_target_switched)) {
            historical_fixed200_seeds_.clear();
            return;
        }

        HistoricalPredictionSeed seed;
        seed.seq = seq;
        seed.time_ms = time_ms;
        seed.relative_time_s = tracker.current_relative_time_s();
        seed.state = current_state;
        seed.target_roll_offset = tracker_debug.selected_roll_offset;
        seed.pred_voter_direction = tracker_debug.direction;
        seed.pred_detected = pred_detected;
        seed.pred_tracker_lost = pred_tracker_lost;
        seed.pred_time_since_detection_ms = pred_time_since_detection_ms;
        seed.pred_selected_phase_index = tracker_debug.selected_phase_index;
        seed.pred_history_size = tracker_debug.history_size;
        seed.pred_switch_deferred = pred_switch_deferred;
        seed.pred_target_switched = pred_target_switched;
        seed.pred_reinit_reason = tracker_debug.reinit_reason;
        historical_fixed200_seeds_.push_back(std::move(seed));

        while (!historical_fixed200_seeds_.empty() &&
               historical_fixed200_seeds_.front().time_ms <
                   time_ms - kPastPredictionOverlayRetentionMs) {
            historical_fixed200_seeds_.pop_front();
        }
        while (historical_fixed200_seeds_.size() > 256) {
            historical_fixed200_seeds_.pop_front();
        }
    }

    void flushPeriodically()
    {
        if (++unflushed_rows_ >= kCsvFlushEveryRows) {
            file_.flush();
            unflushed_rows_ = 0;
        }
    }

    bool checked_ = false;
    uint64_t seq_ = 0;
    std::ofstream file_;
    std::mutex mutex_;
    int unflushed_rows_ = 0;
    double last_observed_roll_ = nanValue();
    double last_filtered_roll_ = nanValue();
    double last_predicted_roll_ = nanValue();
    double last_observed_speed_raw_ = nanValue();
    double last_observed_speed_ = nanValue();
    double last_filtered_speed_ = nanValue();
    double last_filtered_speed_raw_ = nanValue();
    double last_predicted_speed_ = nanValue();
    double last_curve_speed_now_ = nanValue();
    double last_curve_speed_raw_ = nanValue();
    double last_curve_speed_after_predict_ = nanValue();
    double last_curve_speed_after_blade_update_ = nanValue();
    double last_curve_speed_after_speed_update_ = nanValue();
    double last_curve_phi_after_predict_ = nanValue();
    double last_curve_phi_after_blade_update_ = nanValue();
    double last_curve_phi_after_speed_update_ = nanValue();
    double last_fit_a_ = nanValue();
    double last_fit_w_ = nanValue();
    double last_fit_phi_ = nanValue();
    double last_stage0_time_ms_ = nanValue();
    double last_stage0_detected_time_ms_ = nanValue();
    std::deque<HistoricalPredictionSeed> historical_fixed200_seeds_;
};

PredictionCsvLogger& predictionCsvLogger()
{
    static PredictionCsvLogger logger;
    return logger;
}

}  // namespace

namespace auto_buff
{

bool buffOrderedCommitEnabledFromValue(const char* value)
{
    if (value == nullptr || value[0] == '\0' || std::string(value) == "0") {
        return false;
    }
    if (std::string(value) == "1") return true;
    throw std::invalid_argument(
        "AIM_BUFF_ORDERED_COMMIT must be missing, empty, exact 0, or exact 1");
}

bool buffObservationSupersetEnabledFromValue(const char* value)
{
    if (value == nullptr || value[0] == '\0' || std::string(value) == "legacy") {
        return false;
    }
    if (std::string(value) == "observation_superset") return true;
    throw std::invalid_argument(
        "AIM_BUFF_REFINE_PROFILE must be missing, empty, exact legacy, or "
        "exact observation_superset");
}

int buffProposalWorkersFromValue(const char* value, bool observation_profile)
{
    if (!observation_profile) return 1;
    if (value == nullptr || value[0] == '\0' || std::string(value) == "1") return 1;
    if (std::string(value) == "3") return 3;
    if (std::string(value) == "4") return 4;
    throw std::invalid_argument(
        "AIM_BUFF_PROPOSAL_WORKERS must be missing, empty, exact 1, exact 3, or exact 4");
}

bool buffProposalCapacityAccepted(
    std::uint32_t worker_count, double aggregate_service_hz) noexcept
{
    return worker_count != 4 ||
        (std::isfinite(aggregate_service_hz) && aggregate_service_hz > 250.0);
}

bool buffProposalDrainAccepted(const BuffRunePipelineCounters& counters) noexcept
{
    if (counters.proposal_worker_count == 1) return true;
    return counters.proposal_submitted == counters.proposal_completed &&
        counters.proposal_completed == counters.proposal_committed &&
        counters.proposal_inflight == 0 &&
        counters.proposal_input_occupancy == 0 &&
        counters.proposal_reorder_occupancy == 0 &&
        counters.proposal_active_workers == 0;
}

namespace
{

int compareSourceIdentity(
    const BuffExactValidSourceIdentity& lhs,
    const BuffExactValidSourceIdentity& rhs)
{
    if (lhs.producer_epoch != rhs.producer_epoch) {
        return lhs.producer_epoch < rhs.producer_epoch ? -1 : 1;
    }
    if (lhs.image_sequence != rhs.image_sequence) {
        return lhs.image_sequence < rhs.image_sequence ? -1 : 1;
    }
    return 0;
}

std::size_t retainedImageStorageBytes(const cv::Mat& image)
{
    // A shallow capture is safe only when OpenCV owns/refcounts the backing
    // allocation.  UMatData::size accounts the full retained allocation even
    // if the Mat happens to be an ROI.
    if (image.empty() || image.u == nullptr || image.u->size == 0) {
        return 0;
    }
    return image.u->size;
}

void writeJsonNumber(std::ostream& output, double value)
{
    if (std::isfinite(value)) {
        output << value;
    } else {
        output << "null";
    }
}

void writeJsonString(std::ostream& output, const std::string& value)
{
    output << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (ch < 0x20) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned>(ch) << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(ch);
            }
        }
    }
    output << '"';
}

std::optional<std::uint64_t> fileFnv1a64(const std::filesystem::path& path)
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
    return input.eof() ? std::optional<std::uint64_t>(hash) : std::nullopt;
}

std::string hexU64(std::uint64_t value)
{
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << value;
    return text.str();
}

void writeJsonVector3(std::ostream& output, const Eigen::Vector3d& value)
{
    output << '[';
    for (int index = 0; index < 3; ++index) {
        if (index != 0) output << ", ";
        writeJsonNumber(output, value[index]);
    }
    output << ']';
}

std::string sequenceImageFilename(
    std::size_t index,
    const rm::Frame& frame)
{
    std::ostringstream name;
    name << "sequence_" << std::setw(6) << std::setfill('0') << index << '_'
         << frame.source_producer_epoch << '_' << frame.source_image_seq << ".png";
    return name.str();
}

}  // namespace

BuffExactValidSequenceCapture::BuffExactValidSequenceCapture(
    std::string capture_dir,
    std::size_t max_frames,
    std::size_t max_bytes)
    : capture_dir_(std::move(capture_dir)),
      max_frames_(std::clamp<std::size_t>(max_frames, 1, kHardMaxFrames)),
      max_bytes_(std::clamp<std::size_t>(max_bytes, 1, kHardMaxBytes))
{
    diagnostics_.armed = !capture_dir_.empty();
    diagnostics_.max_frames = max_frames_;
    diagnostics_.max_bytes = max_bytes_;
}

void BuffExactValidSequenceCapture::reset()
{
    if (diagnostics_.complete) return;
    retained_.clear();
    has_last_observed_identity_ = false;
    last_observed_identity_ = {};
    last_observed_completion_sequence_ = 0;
    const bool armed = diagnostics_.armed;
    diagnostics_ = {};
    diagnostics_.armed = armed;
    diagnostics_.max_frames = max_frames_;
    diagnostics_.max_bytes = max_bytes_;
}

void BuffExactValidSequenceCapture::evictOldest()
{
    if (retained_.empty()) return;
    const auto& evicted = retained_.front();
    const BuffExactValidSourceIdentity identity{
        evicted.frame.source_producer_epoch,
        evicted.frame.source_image_seq};
    if (!diagnostics_.has_first_evicted_identity) {
        diagnostics_.has_first_evicted_identity = true;
        diagnostics_.first_evicted_identity = identity;
        diagnostics_.first_evicted_completion_sequence = evicted.completion_sequence;
    }
    diagnostics_.has_last_evicted_identity = true;
    diagnostics_.last_evicted_identity = identity;
    diagnostics_.last_evicted_completion_sequence = evicted.completion_sequence;
    ++diagnostics_.evicted_frames;
    diagnostics_.evicted_bytes += evicted.image_storage_bytes;
    diagnostics_.ring_truncated = true;
    diagnostics_.retained_bytes -= evicted.image_storage_bytes;
    retained_.pop_front();
}

void BuffExactValidSequenceCapture::refreshRetainedBounds()
{
    diagnostics_.retained_frames = retained_.size();
    diagnostics_.has_first_retained_identity = !retained_.empty();
    diagnostics_.has_last_retained_identity = !retained_.empty();
    if (retained_.empty()) {
        diagnostics_.first_retained_identity = {};
        diagnostics_.last_retained_identity = {};
        diagnostics_.first_retained_completion_sequence = 0;
        diagnostics_.last_retained_completion_sequence = 0;
        return;
    }
    diagnostics_.first_retained_identity = {
        retained_.front().frame.source_producer_epoch,
        retained_.front().frame.source_image_seq};
    diagnostics_.last_retained_identity = {
        retained_.back().frame.source_producer_epoch,
        retained_.back().frame.source_image_seq};
    diagnostics_.first_retained_completion_sequence =
        retained_.front().completion_sequence;
    diagnostics_.last_retained_completion_sequence =
        retained_.back().completion_sequence;
}

void BuffExactValidSequenceCapture::observeCompletion(const BuffRuneResult& result)
{
    if (!diagnostics_.armed || diagnostics_.complete) return;

    ++diagnostics_.observed_completions;
    const BuffExactValidSourceIdentity identity{
        result.frame.source_producer_epoch,
        result.frame.source_image_seq};
    if (identity.producer_epoch == 0 || identity.image_sequence == 0) {
        ++diagnostics_.invalid_identity_rejects;
        diagnostics_.order_valid = false;
        tools::logger()->error(
            "Buff exact-valid sequence rejected invalid source identity {}:{} completion={}",
            identity.producer_epoch,
            identity.image_sequence,
            result.completion_sequence);
        return;
    }
    if (result.completion_sequence == 0 ||
        (last_observed_completion_sequence_ != 0 &&
         result.completion_sequence <= last_observed_completion_sequence_)) {
        ++diagnostics_.completion_order_rejects;
        diagnostics_.order_valid = false;
        tools::logger()->error(
            "Buff exact-valid sequence rejected completion order last={} current={}",
            last_observed_completion_sequence_,
            result.completion_sequence);
        return;
    }
    if (has_last_observed_identity_) {
        const int order = compareSourceIdentity(identity, last_observed_identity_);
        if (order <= 0) {
            if (order == 0) {
                ++diagnostics_.duplicate_identity_rejects;
            } else {
                ++diagnostics_.regression_identity_rejects;
            }
            diagnostics_.order_valid = false;
            tools::logger()->error(
                "Buff exact-valid sequence rejected non-increasing source identity "
                "last={}:{} current={}:{} completion={}",
                last_observed_identity_.producer_epoch,
                last_observed_identity_.image_sequence,
                identity.producer_epoch,
                identity.image_sequence,
                result.completion_sequence);
            return;
        }
    }
    has_last_observed_identity_ = true;
    last_observed_identity_ = identity;
    last_observed_completion_sequence_ = result.completion_sequence;

    const std::size_t image_storage_bytes = retainedImageStorageBytes(result.frame.srcImg);
    if (result.frame.srcImg.empty() || result.frame.srcImg.type() != CV_8UC3 ||
        image_storage_bytes == 0 || image_storage_bytes > max_bytes_) {
        ++diagnostics_.unretainable_frames;
        diagnostics_.unretainable_bytes += image_storage_bytes;
        diagnostics_.ring_truncated = true;
        tools::logger()->error(
            "Buff exact-valid sequence cannot shallow-retain committed frame "
            "source={}:{} completion={} type={} bytes={} max_bytes={}",
            identity.producer_epoch,
            identity.image_sequence,
            result.completion_sequence,
            result.frame.srcImg.empty() ? -1 : result.frame.srcImg.type(),
            image_storage_bytes,
            max_bytes_);
        return;
    }

    while (!retained_.empty() &&
           (retained_.size() >= max_frames_ ||
            diagnostics_.retained_bytes > max_bytes_ - image_storage_bytes)) {
        evictOldest();
    }

    CapturedCompletion captured;
    captured.frame = result.frame;
    captured.frame.debugImg.release();
    captured.frame.yoloImg.release();
    captured.completion_sequence = result.completion_sequence;
    captured.completion_timestamp_ns = result.completion_timestamp_ns;
    captured.image_storage_bytes = image_storage_bytes;
    auto& expected = captured.expected;
    expected.rune_present = result.rune.has_value();
    expected.has_control = result.has_control;
    expected.switch_deferred = result.switch_deferred;
    expected.target_switched = result.target_switched;
    expected.selected_target_index = result.selected_target_index;
    expected.control_aiming_state = static_cast<int>(result.control.aiming_state);
    expected.control_shot_mode = static_cast<int>(result.control.shot_mode);
    expected.control_shot_buff_mode = static_cast<int>(result.control.shot_buff_mode);
    expected.control_gimbal_yaw_deg = result.control.gimbal_yaw;
    expected.control_gimbal_pitch_deg = result.control.gimbal_pitch;
    expected.control_yaw_error_deg = result.control.yaw_error;
    if (result.rune.has_value()) {
        const auto& rune = *result.rune;
        expected.rune_type = static_cast<int>(rune.type);
        expected.solved_blades = static_cast<std::size_t>(std::count_if(
            rune.fanblades.begin(), rune.fanblades.end(),
            [](const FanBlade& blade) { return blade.solved; }));
        expected.r_center = rune.r_center;
        expected.rune_xyz = rune.xyz_in_world;
        expected.rune_ypd = rune.ypd_in_world;
        expected.rune_ypr = rune.ypr_in_world;
        expected.blade_xyz = rune.blade_xyz_in_world;
        expected.blade_ypd = rune.blade_ypd_in_world;
        if (!rune.fanblades.empty()) {
            const auto& target = rune.target();
            expected.target_solved = target.solved;
            expected.target_pnp_method = target.pnp_method;
            expected.target_pnp_order = target.pnp_order;
            expected.target_pnp_reproj_error_px = target.pnp_reproj_error_px;
            expected.target_pnp_score = target.pnp_score;
            expected.target_pnp_model_center_error_px =
                target.pnp_model_center_error_px;
        }
    }
    expected.current_target_control =
        expected.rune_present && expected.solved_blades > 0 && result.has_control &&
        result.control.aiming_state == rm::ControlData::AIMING_STATE::TARGET_DETECTED;

    diagnostics_.retained_bytes += image_storage_bytes;
    ++diagnostics_.accepted_completions;
    retained_.push_back(std::move(captured));
    refreshRetainedBounds();

    // Every committed completion is retained first.  The persisted command is
    // insufficient: the just-committed result must itself contain a rune, a
    // solved blade, and TARGET_DETECTED control.
    if (!retained_.back().expected.current_target_control) return;
    if (!diagnostics_.order_valid) {
        tools::logger()->error(
            "Buff exact-valid trigger ignored because committed source order is invalid");
        return;
    }
    if (writeSequenceManifest(result)) {
        diagnostics_.complete = true;
    }
}

BuffExactValidCaptureDiagnostics BuffExactValidSequenceCapture::diagnostics() const
{
    return diagnostics_;
}

std::vector<BuffExactValidSourceIdentity>
BuffExactValidSequenceCapture::retainedIdentities() const
{
    std::vector<BuffExactValidSourceIdentity> identities;
    identities.reserve(retained_.size());
    for (const auto& captured : retained_) {
        identities.push_back({
            captured.frame.source_producer_epoch,
            captured.frame.source_image_seq});
    }
    return identities;
}

bool BuffExactValidSequenceCapture::writeSequenceManifest(const BuffRuneResult& trigger)
{
    namespace fs = std::filesystem;
    if (retained_.empty()) return false;
    const auto& last = retained_.back();
    if (last.frame.source_producer_epoch != trigger.frame.source_producer_epoch ||
        last.frame.source_image_seq != trigger.frame.source_image_seq ||
        last.completion_sequence != trigger.completion_sequence) {
        tools::logger()->error(
            "Buff exact-valid trigger is not the last retained committed completion");
        return false;
    }

    try {
        const fs::path capture_dir(capture_dir_);
        fs::create_directories(capture_dir);
        const std::string base_name =
            "exact_valid_sequence_" +
            std::to_string(trigger.frame.source_producer_epoch) + "_" +
            std::to_string(trigger.frame.source_image_seq);
        const fs::path manifest_path = capture_dir / (base_name + ".json");
        const fs::path temporary_manifest_path = capture_dir / (base_name + ".json.tmp");
        if (fs::exists(manifest_path)) {
            tools::logger()->error(
                "Buff exact-valid sequence manifest already exists: {}",
                manifest_path.string());
            return false;
        }

        std::vector<std::string> image_filenames;
        image_filenames.reserve(retained_.size());
        for (std::size_t index = 0; index < retained_.size(); ++index) {
            const auto& captured = retained_[index];
            const std::string filename = sequenceImageFilename(index, captured.frame);
            const fs::path image_path = capture_dir / filename;
            if (!cv::imwrite(image_path.string(), captured.frame.srcImg)) {
                tools::logger()->error(
                    "Buff exact-valid sequence image write failed: {}", image_path.string());
                return false;
            }
            image_filenames.push_back(filename);
        }

        std::ofstream manifest(temporary_manifest_path, std::ios::out | std::ios::trunc);
        if (!manifest.is_open()) {
            tools::logger()->error(
                "Buff exact-valid sequence manifest open failed: {}",
                temporary_manifest_path.string());
            return false;
        }
        manifest << std::setprecision(17)
                 << "{\n"
                 << "  \"schema\": \"aim_buff_exact_valid_sequence_v1\",\n"
                 << "  \"runtime_param_yaml\": ";
        const char* param_yaml = std::getenv("AIM_SIM_PARAM_YAML");
        if (param_yaml != nullptr && param_yaml[0] != '\0') {
            writeJsonString(manifest, param_yaml);
        } else {
            manifest << "null";
        }
        manifest << ",\n  \"runtime_param_yaml_fnv1a64\": ";
        if (param_yaml != nullptr && param_yaml[0] != '\0') {
            const auto hash = fileFnv1a64(std::filesystem::path(param_yaml));
            if (hash.has_value()) {
                writeJsonString(manifest, hexU64(*hash));
            } else {
                manifest << "null";
            }
        } else {
            manifest << "null";
        }
        manifest << ",\n"
                 << "  \"capture_complete\": true,\n"
                 << "  \"order_valid\": " << (diagnostics_.order_valid ? "true" : "false")
                 << ",\n"
                 << "  \"ring_truncated\": "
                 << (diagnostics_.ring_truncated ? "true" : "false") << ",\n"
                 << "  \"hard_max_frames\": " << kHardMaxFrames << ",\n"
                 << "  \"hard_max_bytes\": " << kHardMaxBytes << ",\n"
                 << "  \"configured_max_frames\": " << max_frames_ << ",\n"
                 << "  \"configured_max_bytes\": " << max_bytes_ << ",\n"
                 << "  \"observed_completions\": "
                 << diagnostics_.observed_completions << ",\n"
                 << "  \"accepted_completions\": "
                 << diagnostics_.accepted_completions << ",\n"
                 << "  \"retained_frames\": " << retained_.size() << ",\n"
                 << "  \"retained_bytes\": " << diagnostics_.retained_bytes << ",\n"
                 << "  \"evicted_frames\": " << diagnostics_.evicted_frames << ",\n"
                 << "  \"evicted_bytes\": " << diagnostics_.evicted_bytes << ",\n"
                 << "  \"unretainable_frames\": "
                 << diagnostics_.unretainable_frames << ",\n"
                 << "  \"unretainable_bytes\": "
                 << diagnostics_.unretainable_bytes << ",\n"
                 << "  \"first_retained_source_producer_epoch\": "
                 << diagnostics_.first_retained_identity.producer_epoch << ",\n"
                 << "  \"first_retained_source_image_seq\": "
                 << diagnostics_.first_retained_identity.image_sequence << ",\n"
                 << "  \"first_retained_completion_sequence\": "
                 << diagnostics_.first_retained_completion_sequence << ",\n"
                 << "  \"last_retained_source_producer_epoch\": "
                 << diagnostics_.last_retained_identity.producer_epoch << ",\n"
                 << "  \"last_retained_source_image_seq\": "
                 << diagnostics_.last_retained_identity.image_sequence << ",\n"
                 << "  \"last_retained_completion_sequence\": "
                 << diagnostics_.last_retained_completion_sequence << ",\n"
                 << "  \"first_evicted_source_producer_epoch\": "
                 << diagnostics_.first_evicted_identity.producer_epoch << ",\n"
                 << "  \"first_evicted_source_image_seq\": "
                 << diagnostics_.first_evicted_identity.image_sequence << ",\n"
                 << "  \"first_evicted_completion_sequence\": "
                 << diagnostics_.first_evicted_completion_sequence << ",\n"
                 << "  \"last_evicted_source_producer_epoch\": "
                 << diagnostics_.last_evicted_identity.producer_epoch << ",\n"
                 << "  \"last_evicted_source_image_seq\": "
                 << diagnostics_.last_evicted_identity.image_sequence << ",\n"
                 << "  \"last_evicted_completion_sequence\": "
                 << diagnostics_.last_evicted_completion_sequence << ",\n"
                 << "  \"trigger_index\": " << (retained_.size() - 1) << ",\n"
                 << "  \"trigger_source_producer_epoch\": "
                 << trigger.frame.source_producer_epoch << ",\n"
                 << "  \"trigger_source_image_seq\": "
                 << trigger.frame.source_image_seq << ",\n"
                 << "  \"trigger_completion_sequence\": "
                 << trigger.completion_sequence << ",\n"
                 << "  \"frames\": [\n";

        for (std::size_t index = 0; index < retained_.size(); ++index) {
            const auto& captured = retained_[index];
            const auto& frame = captured.frame;
            const auto& expected = captured.expected;
            manifest << "    {\n"
                     << "      \"index\": " << index << ",\n"
                     << "      \"image_file\": \"" << image_filenames[index] << "\",\n"
                     << "      \"color_order\": \"BGR\",\n"
                     << "      \"image_width\": " << frame.srcImg.cols << ",\n"
                     << "      \"image_height\": " << frame.srcImg.rows << ",\n"
                     << "      \"image_type\": " << frame.srcImg.type() << ",\n"
                     << "      \"image_storage_bytes\": "
                     << captured.image_storage_bytes << ",\n"
                     << "      \"source_producer_epoch\": "
                     << frame.source_producer_epoch << ",\n"
                     << "      \"source_image_seq\": " << frame.source_image_seq << ",\n"
                     << "      \"source_capture_timestamp_ns\": "
                     << frame.source_capture_timestamp_ns << ",\n"
                     << "      \"frame_timestamp_ms\": ";
            writeJsonNumber(manifest, frame.timeStamp);
            manifest << ",\n      \"frame_usb_timestamp_ms\": ";
            writeJsonNumber(manifest, frame.usb_timeStamp);
            manifest << ",\n      \"simulator_state_age_s\": ";
            writeJsonNumber(manifest, frame.simulator_state_age_s);
            manifest << ",\n      \"bullet_speed\": ";
            writeJsonNumber(manifest, frame.bullet_speed);
            manifest << ",\n      \"gimbal_roll_deg\": ";
            writeJsonNumber(manifest, frame.poseEuler.roll);
            manifest << ",\n      \"gimbal_yaw_deg\": ";
            writeJsonNumber(manifest, frame.poseEuler.yaw);
            manifest << ",\n      \"gimbal_pitch_deg\": ";
            writeJsonNumber(manifest, frame.poseEuler.pitch);
            manifest << ",\n"
                     << "      \"fb_sof\": " << static_cast<unsigned>(frame.fb.SOF) << ",\n"
                     << "      \"fb_task_mode\": "
                     << static_cast<unsigned>(frame.fb.task_mode) << ",\n"
                     << "      \"fb_self_team\": "
                     << static_cast<unsigned>(frame.fb.self_team) << ",\n"
                     << "      \"fb_heat\": " << frame.fb.heat << ",\n"
                     << "      \"fb_heat_cap\": " << frame.fb.heat_cap << ",\n"
                     << "      \"fb_bullet_speed\": ";
            writeJsonNumber(manifest, frame.fb.bullet_speed);
            manifest << ",\n      \"fb_gimbal_roll_deg\": ";
            writeJsonNumber(manifest, frame.fb.gimbal_roll);
            manifest << ",\n      \"fb_gimbal_yaw_deg\": ";
            writeJsonNumber(manifest, frame.fb.gimbal_yaw);
            manifest << ",\n      \"fb_gimbal_pitch_deg\": ";
            writeJsonNumber(manifest, frame.fb.gimbal_pitch);
            manifest << ",\n      \"fb_yaw_speed\": ";
            writeJsonNumber(manifest, frame.fb.yaw_speed);
            manifest << ",\n"
                     << "      \"fb_mcu_fire_permit\": "
                     << (frame.fb.mcu_fire_permit() ? "true" : "false") << ",\n"
                     << "      \"fb_raw_task_mode\": "
                     << static_cast<unsigned>(frame.fb.raw_task_mode()) << ",\n"
                     << "      \"fb_head_mapped_task_mode\": "
                     << static_cast<unsigned>(frame.fb.head_mapped_task_mode()) << ",\n"
                     << "      \"fb_eof\": " << static_cast<unsigned>(frame.fb.EOF) << ",\n"
                     << "      \"completion_sequence\": "
                     << captured.completion_sequence << ",\n"
                     << "      \"completion_timestamp_ns\": "
                     << captured.completion_timestamp_ns << ",\n"
                     << "      \"expected_rune_present\": "
                     << (expected.rune_present ? "true" : "false") << ",\n"
                     << "      \"expected_rune_type\": " << expected.rune_type << ",\n"
                     << "      \"expected_solved_blades\": "
                     << expected.solved_blades << ",\n"
                     << "      \"expected_target_solved\": "
                     << (expected.target_solved ? "true" : "false") << ",\n"
                     << "      \"expected_has_control\": "
                     << (expected.has_control ? "true" : "false") << ",\n"
                     << "      \"expected_current_target_control\": "
                     << (expected.current_target_control ? "true" : "false") << ",\n"
                     << "      \"expected_switch_deferred\": "
                     << (expected.switch_deferred ? "true" : "false") << ",\n"
                     << "      \"expected_target_switched\": "
                     << (expected.target_switched ? "true" : "false") << ",\n"
                     << "      \"expected_selected_target_index\": "
                     << expected.selected_target_index << ",\n"
                     << "      \"expected_control_aiming_state\": "
                     << expected.control_aiming_state << ",\n"
                     << "      \"expected_control_shot_mode\": "
                     << expected.control_shot_mode << ",\n"
                     << "      \"expected_control_shot_buff_mode\": "
                     << expected.control_shot_buff_mode << ",\n"
                     << "      \"expected_control_gimbal_yaw_deg\": ";
            writeJsonNumber(manifest, expected.control_gimbal_yaw_deg);
            manifest << ",\n      \"expected_control_gimbal_pitch_deg\": ";
            writeJsonNumber(manifest, expected.control_gimbal_pitch_deg);
            manifest << ",\n      \"expected_control_yaw_error_deg\": ";
            writeJsonNumber(manifest, expected.control_yaw_error_deg);
            manifest << ",\n      \"expected_r_center\": [";
            writeJsonNumber(manifest, expected.r_center.x);
            manifest << ", ";
            writeJsonNumber(manifest, expected.r_center.y);
            manifest << "],\n      \"expected_rune_xyz\": ";
            writeJsonVector3(manifest, expected.rune_xyz);
            manifest << ",\n      \"expected_rune_ypd\": ";
            writeJsonVector3(manifest, expected.rune_ypd);
            manifest << ",\n      \"expected_rune_ypr\": ";
            writeJsonVector3(manifest, expected.rune_ypr);
            manifest << ",\n      \"expected_blade_xyz\": ";
            writeJsonVector3(manifest, expected.blade_xyz);
            manifest << ",\n      \"expected_blade_ypd\": ";
            writeJsonVector3(manifest, expected.blade_ypd);
            manifest << ",\n"
                     << "      \"expected_target_pnp_method\": "
                     << expected.target_pnp_method << ",\n"
                     << "      \"expected_target_pnp_order\": ["
                     << expected.target_pnp_order[0] << ", "
                     << expected.target_pnp_order[1] << ", "
                     << expected.target_pnp_order[2] << ", "
                     << expected.target_pnp_order[3] << "],\n"
                     << "      \"expected_target_pnp_reproj_error_px\": ";
            writeJsonNumber(manifest, expected.target_pnp_reproj_error_px);
            manifest << ",\n      \"expected_target_pnp_score\": ";
            writeJsonNumber(manifest, expected.target_pnp_score);
            manifest << ",\n      \"expected_target_pnp_model_center_error_px\": ";
            writeJsonNumber(manifest, expected.target_pnp_model_center_error_px);
            manifest << "\n    }" << (index + 1 == retained_.size() ? "\n" : ",\n");
        }
        manifest << "  ]\n}\n";
        manifest.flush();
        if (!manifest.good()) {
            tools::logger()->error(
                "Buff exact-valid sequence manifest write failed: {}",
                temporary_manifest_path.string());
            return false;
        }
        manifest.close();
        fs::rename(temporary_manifest_path, manifest_path);
        tools::logger()->info(
            "Buff exact-valid sequence saved: manifest={}, frames={}, bytes={}, "
            "truncated={}, evicted_frames={}, first={}:{}, last={}:{}, trigger_completion={}",
            manifest_path.string(),
            retained_.size(),
            diagnostics_.retained_bytes,
            diagnostics_.ring_truncated,
            diagnostics_.evicted_frames,
            diagnostics_.first_retained_identity.producer_epoch,
            diagnostics_.first_retained_identity.image_sequence,
            diagnostics_.last_retained_identity.producer_epoch,
            diagnostics_.last_retained_identity.image_sequence,
            trigger.completion_sequence);
        return true;
    } catch (const std::exception& error) {
        tools::logger()->error("Buff exact-valid sequence capture failed: {}", error.what());
        return false;
    }
}

BuffRunePipeline::BuffRunePipeline(
    const std::string& config_path,
    BuffRunePipelineOptions options)
    : config_path_(resolve_config_path(config_path)),
      emit_debug_artifacts_(options.emit_debug_artifacts),
      collect_completion_samples_(options.collect_completion_samples),
      observation_superset_enabled_(buffObservationSupersetEnabledFromValue(
          std::getenv("AIM_BUFF_REFINE_PROFILE"))),
      ordered_commit_inline_(
          buffOrderedCommitEnabledFromValue(std::getenv("AIM_BUFF_ORDERED_COMMIT")) ||
          observation_superset_enabled_),
      proposal_worker_count_(buffProposalWorkersFromValue(
          std::getenv("AIM_BUFF_PROPOSAL_WORKERS"), observation_superset_enabled_)),
      detector_(config_path_),
      solver_(config_path_),
      visual_solver_(config_path_),
      tracker_(config_path_),
      aimer_(config_path_)
{
    if (const char* capture_dir = std::getenv("AIM_BUFF_EXACT_VALID_CAPTURE_DIR");
        capture_dir != nullptr && capture_dir[0] != '\0') {
        exact_valid_capture_ =
            std::make_unique<BuffExactValidSequenceCapture>(capture_dir);
        tools::logger()->info(
            "Buff exact-valid ordered sequence capture armed: {} "
            "(max_frames={}, max_bytes={})",
            capture_dir,
            BuffExactValidSequenceCapture::kHardMaxFrames,
            BuffExactValidSequenceCapture::kHardMaxBytes);
    }
    try {
        const auto yaml = YAML::LoadFile(config_path_);
        const auto pipeline_yaml = yaml["buff_pipeline"];
        if (pipeline_yaml && pipeline_yaml["draw_yolo_results"]) {
            draw_yolo_results_ = pipeline_yaml["draw_yolo_results"].as<bool>();
        }
        if (pipeline_yaml && pipeline_yaml["draw_r_binary_mask"]) {
            draw_r_binary_mask_ = pipeline_yaml["draw_r_binary_mask"].as<bool>();
        }
        if (pipeline_yaml && pipeline_yaml["shot_gate_enabled"]) {
            shot_gate_enabled_ = pipeline_yaml["shot_gate_enabled"].as<bool>();
        }
        if (pipeline_yaml && pipeline_yaml["shot_gate_min_stable_frames"]) {
            shot_gate_min_stable_frames_ =
                std::max(1, pipeline_yaml["shot_gate_min_stable_frames"].as<int>());
        }
        if (pipeline_yaml && pipeline_yaml["shot_gate_max_pnp_reproj_error_px"]) {
            shot_gate_max_pnp_reproj_error_px_ =
                pipeline_yaml["shot_gate_max_pnp_reproj_error_px"].as<double>();
        }
        if (pipeline_yaml && pipeline_yaml["shot_gate_max_model_center_error_px"]) {
            shot_gate_max_model_center_error_px_ =
                pipeline_yaml["shot_gate_max_model_center_error_px"].as<double>();
        }
        if (pipeline_yaml && pipeline_yaml["shot_gate_max_yaw_error_deg"]) {
            shot_gate_max_yaw_error_deg_ =
                pipeline_yaml["shot_gate_max_yaw_error_deg"].as<double>();
        }
        if (pipeline_yaml && pipeline_yaml["shot_gate_max_pitch_error_deg"]) {
            shot_gate_max_pitch_error_deg_ =
                pipeline_yaml["shot_gate_max_pitch_error_deg"].as<double>();
        }
    } catch (const std::exception& e) {
        tools::logger()->warn("Failed to load buff pipeline config: {}", e.what());
    }
    if (!std::isfinite(shot_gate_max_pnp_reproj_error_px_) ||
        shot_gate_max_pnp_reproj_error_px_ <= 0.0) {
        shot_gate_max_pnp_reproj_error_px_ = 28.0;
    }
    if (!std::isfinite(shot_gate_max_model_center_error_px_) ||
        shot_gate_max_model_center_error_px_ <= 0.0) {
        shot_gate_max_model_center_error_px_ = 8.0;
    }
    if (!std::isfinite(shot_gate_max_yaw_error_deg_) ||
        shot_gate_max_yaw_error_deg_ <= 0.0) {
        shot_gate_max_yaw_error_deg_ = 1.5;
    }
    if (!std::isfinite(shot_gate_max_pitch_error_deg_) ||
        shot_gate_max_pitch_error_deg_ <= 0.0) {
        shot_gate_max_pitch_error_deg_ = 1.5;
    }
    fps_start_tp_ = std::chrono::steady_clock::now();
    infer_thread_ = std::thread(&BuffRunePipeline::inferLoop, this);
    if (observation_superset_enabled_ && proposal_worker_count_ > 1) {
        proposal_worker_threads_.reserve(static_cast<std::size_t>(proposal_worker_count_));
        for (std::size_t index = 0;
             index < static_cast<std::size_t>(proposal_worker_count_); ++index) {
            proposal_worker_threads_.emplace_back(
                &BuffRunePipeline::proposalWorkerLoop, this, index);
        }
        solve_thread_ = std::thread(&BuffRunePipeline::proposalCommitLoop, this);
    } else {
        solve_thread_ = std::thread(&BuffRunePipeline::solveLoop, this);
    }
    if (!ordered_commit_inline_) {
        track_thread_ = std::thread(&BuffRunePipeline::trackAimLoop, this);
    }
    tools::logger()->info(
        "Buff rune pipeline config: {}, emit_debug_artifacts={}, collect_completion_samples={}, "
        "ordered_commit={}, refine_profile={}, proposal_workers={}, "
        "draw_yolo_results={}, "
        "draw_r_binary_mask={}, "
        "shot_gate(enabled={}, stable_frames={}, pnp_reproj<={:.2f}px, center<={:.2f}px, "
        "yaw<={:.2f}deg, pitch<={:.2f}deg)",
        config_path_,
        emit_debug_artifacts_,
        collect_completion_samples_,
        ordered_commit_inline_ ? "inline" : "legacy",
        observation_superset_enabled_ ? "observation_superset" : "legacy",
        proposal_worker_count_,
        draw_yolo_results_,
        draw_r_binary_mask_,
        shot_gate_enabled_,
        shot_gate_min_stable_frames_,
        shot_gate_max_pnp_reproj_error_px_,
        shot_gate_max_model_center_error_px_,
        shot_gate_max_yaw_error_deg_,
        shot_gate_max_pitch_error_deg_);
    if (observation_superset_enabled_) {
        tools::logger()->info(
            "Observation-superset prototype enabled: bounded canonical R artifacts and "
            "fixed-R exhaustive PnP are active; unsupported frames use explicit legacy fallback");
    }
}

BuffRunePipeline::~BuffRunePipeline()
{
    stop_.store(true, std::memory_order_relaxed);
    input_cv_.notify_all();
    yolo_cv_.notify_all();
    detection_cv_.notify_all();
    proposal_input_cv_.notify_all();
    proposal_reorder_cv_.notify_all();
    if (infer_thread_.joinable()) {
        infer_thread_.join();
    }
    for (auto& worker : proposal_worker_threads_) {
        if (worker.joinable()) worker.join();
    }
    if (solve_thread_.joinable()) {
        solve_thread_.join();
    }
    if (track_thread_.joinable()) {
        track_thread_.join();
    }
}

void BuffRunePipeline::push(rm::Frame frame)
{
    if (observation_superset_enabled_ && proposal_worker_count_ > 1 &&
        frame.source_producer_epoch != 0) {
        std::lock_guard<std::mutex> epoch_lock(push_identity_mutex_);
        if (last_pushed_epoch_ != 0 &&
            last_pushed_epoch_ != frame.source_producer_epoch) {
            tools::logger()->info(
                "Buff proposal producer epoch transition {} -> {}; resetting ordered state",
                last_pushed_epoch_, frame.source_producer_epoch);
            reset();
        }
        last_pushed_epoch_ = frame.source_producer_epoch;
    }
    pushed_frames_.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(input_mutex_);
        if (latest_input_.has_value()) {
            input_queue_overwrites_.fetch_add(1, std::memory_order_relaxed);
        }
        latest_input_ = std::move(frame);
    }
    input_cv_.notify_one();
}

void BuffRunePipeline::reset()
{
    std::uint64_t cancelled_on_reset = 0;
    std::unique_lock<std::mutex> ordered_commit_lock(
        ordered_commit_mutex_, std::defer_lock);
    if (ordered_commit_inline_) ordered_commit_lock.lock();
    {
        std::scoped_lock transition_lock(proposal_mutex_, observation_identity_mutex_);
        proposal_accepting_ = false;
        generation_.fetch_add(1, std::memory_order_acq_rel);
        cancelled_on_reset = proposal_inflight_;
        proposal_input_.clear();
        proposal_reorder_.clear();
        proposal_worker_has_terminal_.fill(false);
        proposal_inflight_ = 0;
        next_commit_sequence_ = next_proposal_sequence_;
        last_observation_epoch_ = 0;
        last_observation_sequence_ = 0;
    }
    proposal_input_cv_.notify_all();
    proposal_reorder_cv_.notify_all();
    {
        std::unique_lock<std::mutex> proposal_lock(proposal_mutex_);
        proposal_reorder_cv_.wait(proposal_lock, [this] {
            return proposal_active_workers_.load(std::memory_order_relaxed) == 0;
        });
    }
    {
        std::lock_guard<std::mutex> lock(input_mutex_);
        latest_input_.reset();
    }
    {
        std::lock_guard<std::mutex> lock(yolo_mutex_);
        latest_yolo_.reset();
    }
    {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        latest_detection_.reset();
    }
    output_mailbox_.clear();
    {
        std::lock_guard<std::mutex> lock(solve_mutex_);
        detector_.reset();
        solver_.reset();
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        visual_solver_.reset();
        tracker_.reset();
        aimer_.reset();
        historical_prediction_seeds_.clear();
        debug_prediction_overlays_.clear();
        last_valid_control_.reset();
        shot_gate_stable_frames_ = 0;
    }
    {
        std::lock_guard<std::mutex> lock(time_mutex_);
        time_base_ready_ = false;
        time_base_ms_ = 0.0;
        time_base_tp_ = std::chrono::steady_clock::time_point{};
        fps_ = 0;
        fps_count_ = 0;
        fps_start_tp_ = std::chrono::steady_clock::time_point{};
    }
    {
        std::lock_guard<std::mutex> lock(completion_samples_mutex_);
        completion_samples_.clear();
    }
    if (exact_valid_capture_) {
        exact_valid_capture_->reset();
    }
    pushed_frames_.store(0, std::memory_order_relaxed);
    input_queue_overwrites_.store(0, std::memory_order_relaxed);
    yolo_completed_.store(0, std::memory_order_relaxed);
    yolo_queue_overwrites_.store(0, std::memory_order_relaxed);
    solve_completed_.store(0, std::memory_order_relaxed);
    detection_queue_overwrites_.store(0, std::memory_order_relaxed);
    ordered_commit_failures_.store(0, std::memory_order_relaxed);
    observation_proposal_attempts_.store(0, std::memory_order_relaxed);
    observation_proposal_fallbacks_.store(0, std::memory_order_relaxed);
    observation_proposal_candidates_.store(0, std::memory_order_relaxed);
    observation_pnp_proposals_.store(0, std::memory_order_relaxed);
    observation_union_roi_pixels_.store(0, std::memory_order_relaxed);
    observation_template_result_pixels_.store(0, std::memory_order_relaxed);
    observation_cap_events_.store(0, std::memory_order_relaxed);
    observation_identity_gaps_.store(0, std::memory_order_relaxed);
    observation_identity_failures_.store(0, std::memory_order_relaxed);
    observation_ready_consumptions_.store(0, std::memory_order_relaxed);
    proposal_submitted_.store(0, std::memory_order_relaxed);
    proposal_completed_.store(0, std::memory_order_relaxed);
    proposal_committed_.store(0, std::memory_order_relaxed);
    proposal_active_workers_.store(0, std::memory_order_relaxed);
    proposal_max_active_workers_.store(0, std::memory_order_relaxed);
    proposal_input_max_occupancy_.store(0, std::memory_order_relaxed);
    proposal_reorder_max_occupancy_.store(0, std::memory_order_relaxed);
    proposal_max_inflight_.store(0, std::memory_order_relaxed);
    proposal_terminal_gaps_.store(0, std::memory_order_relaxed);
    proposal_terminal_failures_.store(0, std::memory_order_relaxed);
    proposal_cancelled_.store(cancelled_on_reset, std::memory_order_relaxed);
    proposal_stale_.store(0, std::memory_order_relaxed);
    proposal_head_wait_ns_.store(0, std::memory_order_relaxed);
    for (std::size_t index = 0; index < kProposalMaxWorkers; ++index) {
        proposal_worker_completed_[index].store(0, std::memory_order_relaxed);
        proposal_worker_total_ns_[index].store(0, std::memory_order_relaxed);
    }
    essential_completed_.store(0, std::memory_order_relaxed);
    published_results_.store(0, std::memory_order_relaxed);
    popped_results_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(proposal_mutex_);
        proposal_accepting_ = true;
    }
    yolo_cv_.notify_all();
    detection_cv_.notify_all();
    proposal_input_cv_.notify_all();
    proposal_reorder_cv_.notify_all();
}

bool BuffRunePipeline::tryPopLatest(BuffRuneResult* result)
{
    const bool popped = output_mailbox_.tryPopLatest(result);
    if (popped) {
        popped_results_.fetch_add(1, std::memory_order_relaxed);
    }
    return popped;
}

BuffRunePipelineCounters BuffRunePipeline::counters() const noexcept
{
    BuffRunePipelineCounters snapshot;
    snapshot.ordered_commit_inline = ordered_commit_inline_;
    snapshot.track_thread_started = track_thread_.joinable();
    snapshot.observation_superset_enabled = observation_superset_enabled_;
    snapshot.observation_superset_ready =
        observation_ready_consumptions_.load(std::memory_order_relaxed) > 0;
    snapshot.pushed_frames = pushed_frames_.load(std::memory_order_relaxed);
    snapshot.input_queue_overwrites =
        input_queue_overwrites_.load(std::memory_order_relaxed);
    snapshot.yolo_completed = yolo_completed_.load(std::memory_order_relaxed);
    snapshot.yolo_queue_overwrites =
        yolo_queue_overwrites_.load(std::memory_order_relaxed);
    snapshot.solve_completed = solve_completed_.load(std::memory_order_relaxed);
    snapshot.detection_queue_overwrites =
        detection_queue_overwrites_.load(std::memory_order_relaxed);
    snapshot.ordered_commit_failures =
        ordered_commit_failures_.load(std::memory_order_relaxed);
    snapshot.observation_proposal_attempts =
        observation_proposal_attempts_.load(std::memory_order_relaxed);
    snapshot.observation_proposal_fallbacks =
        observation_proposal_fallbacks_.load(std::memory_order_relaxed);
    snapshot.observation_proposal_candidates =
        observation_proposal_candidates_.load(std::memory_order_relaxed);
    snapshot.observation_pnp_proposals =
        observation_pnp_proposals_.load(std::memory_order_relaxed);
    snapshot.observation_union_roi_pixels =
        observation_union_roi_pixels_.load(std::memory_order_relaxed);
    snapshot.observation_template_result_pixels =
        observation_template_result_pixels_.load(std::memory_order_relaxed);
    snapshot.observation_cap_events =
        observation_cap_events_.load(std::memory_order_relaxed);
    snapshot.observation_identity_gaps =
        observation_identity_gaps_.load(std::memory_order_relaxed);
    snapshot.observation_identity_failures =
        observation_identity_failures_.load(std::memory_order_relaxed);
    snapshot.proposal_worker_count = static_cast<std::uint32_t>(proposal_worker_count_);
    snapshot.proposal_submitted = proposal_submitted_.load(std::memory_order_relaxed);
    snapshot.proposal_completed = proposal_completed_.load(std::memory_order_relaxed);
    snapshot.proposal_committed = proposal_committed_.load(std::memory_order_relaxed);
    snapshot.proposal_active_workers = proposal_active_workers_.load(std::memory_order_relaxed);
    snapshot.proposal_max_active_workers =
        proposal_max_active_workers_.load(std::memory_order_relaxed);
    snapshot.proposal_input_max_occupancy =
        proposal_input_max_occupancy_.load(std::memory_order_relaxed);
    snapshot.proposal_reorder_max_occupancy =
        proposal_reorder_max_occupancy_.load(std::memory_order_relaxed);
    snapshot.proposal_terminal_gaps = proposal_terminal_gaps_.load(std::memory_order_relaxed);
    snapshot.proposal_terminal_failures =
        proposal_terminal_failures_.load(std::memory_order_relaxed);
    snapshot.proposal_cancelled = proposal_cancelled_.load(std::memory_order_relaxed);
    snapshot.proposal_stale = proposal_stale_.load(std::memory_order_relaxed);
    snapshot.proposal_head_wait_ns = proposal_head_wait_ns_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(proposal_mutex_);
        snapshot.proposal_input_occupancy =
            static_cast<std::uint32_t>(proposal_input_.size());
        snapshot.proposal_reorder_occupancy =
            static_cast<std::uint32_t>(proposal_reorder_.size());
        snapshot.proposal_inflight = static_cast<std::uint32_t>(proposal_inflight_);
    }
    snapshot.proposal_max_inflight =
        proposal_max_inflight_.load(std::memory_order_relaxed);
    for (std::size_t index = 0; index < kProposalMaxWorkers; ++index) {
        snapshot.proposal_worker_completed[index] =
            proposal_worker_completed_[index].load(std::memory_order_relaxed);
        snapshot.proposal_worker_total_ns[index] =
            proposal_worker_total_ns_[index].load(std::memory_order_relaxed);
    }
    snapshot.essential_completed = essential_completed_.load(std::memory_order_relaxed);
    snapshot.published_results = published_results_.load(std::memory_order_relaxed);
    snapshot.popped_results = popped_results_.load(std::memory_order_relaxed);
    return snapshot;
}

std::vector<BuffRuneCompletionSample> BuffRunePipeline::completionSamples() const
{
    std::lock_guard<std::mutex> lock(completion_samples_mutex_);
    return completion_samples_;
}

void BuffRunePipeline::inferLoop()
{
    while (!stop_.load(std::memory_order_relaxed)) {
        rm::Frame frame;
        uint64_t local_generation = 0;
        {
            std::unique_lock<std::mutex> lock(input_mutex_);
            input_cv_.wait(lock, [this] {
                return latest_input_.has_value() || stop_.load(std::memory_order_relaxed);
            });
            if (stop_.load(std::memory_order_relaxed)) break;
            local_generation = generation_.load(std::memory_order_acquire);
            frame = std::move(*latest_input_);
            latest_input_.reset();
        }

        if (frame.srcImg.empty()) continue;

        try {
            const auto yolo_begin = std::chrono::steady_clock::now();
            BuffYoloPacket packet = runYolo(std::move(frame), local_generation);
            packet.yolo_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - yolo_begin).count();
            if (local_generation != generation_.load(std::memory_order_acquire)) {
                continue;
            }
            yolo_completed_.fetch_add(1, std::memory_order_relaxed);
            if (observation_superset_enabled_ && proposal_worker_count_ > 1) {
                recordObservationIdentity(packet);
                std::unique_lock<std::mutex> lock(proposal_mutex_);
                proposal_input_cv_.wait(lock, [this, local_generation] {
                    return stop_.load(std::memory_order_relaxed) ||
                           generation_.load(std::memory_order_acquire) != local_generation ||
                           (proposal_accepting_ &&
                            proposal_inflight_ <
                              static_cast<std::size_t>(proposal_worker_count_));
                });
                if (stop_.load(std::memory_order_relaxed) ||
                    generation_.load(std::memory_order_acquire) != local_generation) {
                    proposal_cancelled_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                packet.proposal_sequence = next_proposal_sequence_++;
                proposal_input_.push_back(std::move(packet));
                ++proposal_inflight_;
                const auto inflight = static_cast<std::uint32_t>(proposal_inflight_);
                auto max_inflight = proposal_max_inflight_.load(std::memory_order_relaxed);
                while (max_inflight < inflight &&
                       !proposal_max_inflight_.compare_exchange_weak(
                           max_inflight, inflight, std::memory_order_relaxed)) {}
                proposal_submitted_.fetch_add(1, std::memory_order_relaxed);
                const auto occupancy = static_cast<std::uint32_t>(proposal_input_.size());
                auto maximum = proposal_input_max_occupancy_.load(std::memory_order_relaxed);
                while (maximum < occupancy &&
                       !proposal_input_max_occupancy_.compare_exchange_weak(
                           maximum, occupancy, std::memory_order_relaxed)) {}
                lock.unlock();
                proposal_input_cv_.notify_one();
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(yolo_mutex_);
                if (latest_yolo_.has_value()) {
                    yolo_queue_overwrites_.fetch_add(1, std::memory_order_relaxed);
                }
                latest_yolo_ = std::move(packet);
            }
            yolo_cv_.notify_one();
        } catch (const std::exception& e) {
            tools::logger()->error("Buff infer pipeline exception: {}", e.what());
        }
    }
}

void BuffRunePipeline::recordObservationIdentity(const BuffYoloPacket& packet)
{
    std::scoped_lock lock(proposal_mutex_, observation_identity_mutex_);
    if (!proposal_accepting_ ||
        packet.generation != generation_.load(std::memory_order_acquire)) return;
    const auto epoch = packet.frame.source_producer_epoch;
    const auto sequence = packet.frame.source_image_seq;
    if (epoch == 0 || sequence == 0 ||
        (last_observation_epoch_ == epoch && sequence <= last_observation_sequence_)) {
        observation_identity_failures_.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error(
            "observation-superset proposal source identity is invalid or non-increasing");
    }
    if (last_observation_epoch_ == epoch && last_observation_sequence_ != 0 &&
        sequence > last_observation_sequence_ + 1) {
        observation_identity_gaps_.fetch_add(
            sequence - last_observation_sequence_ - 1, std::memory_order_relaxed);
    }
    last_observation_epoch_ = epoch;
    last_observation_sequence_ = sequence;
}

void BuffRunePipeline::buildObservationSupersetScaffold(
    BuffYoloPacket* packet, BuffCanonicalWorkerScratch* scratch)
{
    if (packet == nullptr || !observation_superset_enabled_) return;
    const auto proposal_begin = std::chrono::steady_clock::now();
    BuffYoloPacket::ObservationSupersetProposal proposal;
    proposal.producer_epoch = packet->frame.source_producer_epoch;
    proposal.image_sequence = packet->frame.source_image_seq;
    proposal.cost.enabled = true;

    if (proposal_worker_count_ == 1) {
        std::lock_guard<std::mutex> lock(observation_identity_mutex_);
        if (proposal.producer_epoch == 0 || proposal.image_sequence == 0 ||
            (last_observation_epoch_ == proposal.producer_epoch &&
             proposal.image_sequence <= last_observation_sequence_)) {
            observation_identity_failures_.fetch_add(1, std::memory_order_relaxed);
            throw std::runtime_error(
                "observation-superset proposal source identity is invalid or non-increasing");
        }
        if (last_observation_epoch_ == proposal.producer_epoch &&
            last_observation_sequence_ != 0 &&
            proposal.image_sequence > last_observation_sequence_ + 1) {
            observation_identity_gaps_.fetch_add(
                proposal.image_sequence - last_observation_sequence_ - 1,
                std::memory_order_relaxed);
        }
        last_observation_epoch_ = proposal.producer_epoch;
        last_observation_sequence_ = proposal.image_sequence;
    }

    const auto scan_begin = std::chrono::steady_clock::now();
    proposal.observation = detector_.build_canonical_observation(
        packet->mode, packet->frame.srcImg, packet->candidates, scratch);
    proposal.union_roi = proposal.observation.union_roi;
    proposal.cost.candidate_scan_ns = costDurationNs(scan_begin);
    proposal.cost.retained_candidate_count =
        static_cast<std::uint32_t>(packet->candidates.size());
    proposal.cost.target_hypothesis_count =
        static_cast<std::uint32_t>(proposal.observation.targets.size());
    proposal.cost.hit_context_count =
        static_cast<std::uint32_t>(proposal.observation.hit_context.size());
    proposal.cost.anchor_count =
        static_cast<std::uint32_t>(proposal.observation.anchors.size());
    proposal.cost.scale_count =
        static_cast<std::uint32_t>(proposal.observation.scale_responses.size());
    proposal.cost.contour_count =
        static_cast<std::uint32_t>(proposal.observation.contours.size());
    proposal.cost.union_roi_pixels = proposal.observation.union_roi_pixels;
    proposal.cost.template_result_pixels =
        proposal.observation.template_result_pixels;
    proposal.cost.r_preprocess_ns = proposal.observation.preprocess_ns;
    proposal.cost.r_template_ns = proposal.observation.template_ns;
    proposal.cost.r_contour_ns = proposal.observation.contour_ns;
    proposal.cost.scratch_allocations = proposal.observation.scratch_allocations;
    proposal.cost.scratch_reuses = proposal.observation.scratch_reuses;
    proposal.cost.response_cells_scanned = proposal.observation.response_cells_scanned;
    proposal.cost.support_rejected_cells = proposal.observation.support_rejected_cells;
    proposal.cost.distance_tested_cells = proposal.observation.distance_tested_cells;
    proposal.cost.contour_copy_bytes_avoided =
        proposal.observation.contour_copy_bytes_avoided;
    proposal.cost.cap_events = proposal.observation.cap_events;
    proposal.cost.coverage_complete = proposal.observation.ready;
    proposal.cost.proposal_ready = false; // PnP extraction completes in solveRune.
    proposal.cost.used_legacy_fallback =
        proposal.observation.requires_legacy_fallback;
    proposal.cost.fallback_events = proposal.cost.used_legacy_fallback ? 1U : 0U;
    proposal.cost.fallback_reason =
        static_cast<std::uint32_t>(proposal.observation.fallback_reason);
    proposal.candidates = std::move(packet->candidates);
    proposal.cost.proposal_total_ns = costDurationNs(proposal_begin);

    observation_proposal_attempts_.fetch_add(1, std::memory_order_relaxed);
    if (proposal.cost.used_legacy_fallback) {
        observation_proposal_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    }
    observation_proposal_candidates_.fetch_add(
        proposal.cost.retained_candidate_count, std::memory_order_relaxed);
    observation_union_roi_pixels_.fetch_add(
        proposal.cost.union_roi_pixels, std::memory_order_relaxed);
    observation_template_result_pixels_.fetch_add(
        proposal.cost.template_result_pixels, std::memory_order_relaxed);
    observation_cap_events_.fetch_add(
        proposal.cost.cap_events, std::memory_order_relaxed);
    packet->observation_proposal = std::move(proposal);
}

void BuffRunePipeline::completeObservationProposal(
    BuffYoloPacket* packet, Solver* scratch_solver)
{
    if (packet == nullptr || scratch_solver == nullptr ||
        !packet->observation_proposal.has_value()) return;
    auto& proposal = *packet->observation_proposal;
    if (!proposal.observation.ready || proposal.observation.requires_legacy_fallback) return;

    const auto context_begin = std::chrono::steady_clock::now();
    scratch_solver->set_R_gimbal2world(gimbalQuaternion(packet->frame));
    proposal.solver_frame = scratch_solver->makeFrameContext(packet->frame.srcImg.size());
    std::vector<SolverPnpHypothesis> hypotheses;
    hypotheses.reserve(proposal.observation.targets.size());
    for (const auto& target : proposal.observation.targets) {
        const auto choice = std::find_if(
            proposal.observation.r_choices.begin(), proposal.observation.r_choices.end(),
            [&](const BuffCanonicalRChoice& value) {
                return value.hypothesis_index == target.hypothesis_index;
            });
        if (choice == proposal.observation.r_choices.end()) {
            proposal.cost.used_legacy_fallback = true;
            proposal.cost.coverage_complete = false;
            proposal.cost.fallback_reason = static_cast<std::uint32_t>(
                BuffObservationFallbackReason::MissingCoverage);
            break;
        }
        SolverPnpHypothesis hypothesis;
        hypothesis.hypothesis_index = target.hypothesis_index;
        hypothesis.source_pixel_signature = proposal.observation.template_result_pixels ^
            (static_cast<std::uint64_t>(target.hypothesis_index) << 56U);
        hypothesis.r_center = choice->r_center;
        hypothesis.fanblades_view = &target.fanblades;
        hypotheses.push_back(std::move(hypothesis));
    }
    proposal.cost.worker_context_setup_ns = costDurationNs(context_begin);
    if (!proposal.cost.used_legacy_fallback) {
        const auto begin = std::chrono::steady_clock::now();
        proposal.pnp = scratch_solver->buildExhaustiveProposal(
            hypotheses, proposal.solver_frame);
        proposal.cost.pnp_extract_ns = costDurationNs(begin);
        proposal.cost.pnp_proposal_count =
            static_cast<std::uint32_t>(proposal.pnp.solutions.size());
        const bool complete = proposal.pnp.status == ExhaustivePnpStatus::Ready &&
            proposal.pnp.expected_solution_count == proposal.pnp.solutions.size();
        proposal.cost.coverage_complete = proposal.cost.coverage_complete && complete;
        proposal.cost.proposal_ready = proposal.cost.coverage_complete;
        if (!complete) {
            proposal.cost.used_legacy_fallback = true;
            proposal.cost.fallback_reason = static_cast<std::uint32_t>(
                BuffObservationFallbackReason::PnpCap);
            ++proposal.cost.cap_events;
        }
        proposal.cost.proposal_total_ns += proposal.cost.pnp_extract_ns;
    }
}

void BuffRunePipeline::proposalWorkerLoop(std::size_t worker_index)
{
    Solver scratch_solver(config_path_);
    BuffCanonicalWorkerScratch canonical_scratch;
    while (!stop_.load(std::memory_order_relaxed)) {
        BuffYoloPacket packet;
        auto begin = std::chrono::steady_clock::time_point{};
        {
            std::unique_lock<std::mutex> lock(proposal_mutex_);
            proposal_input_cv_.wait(lock, [this, worker_index] {
                return stop_.load(std::memory_order_relaxed) ||
                    (!proposal_worker_has_terminal_[worker_index] &&
                     !proposal_input_.empty());
            });
            if (stop_.load(std::memory_order_relaxed)) break;
            packet = std::move(proposal_input_.front());
            proposal_input_.pop_front();
            begin = std::chrono::steady_clock::now();
            const auto active = proposal_active_workers_.fetch_add(
                1, std::memory_order_relaxed) + 1;
            auto maximum = proposal_max_active_workers_.load(std::memory_order_relaxed);
            while (maximum < active &&
                   !proposal_max_active_workers_.compare_exchange_weak(
                       maximum, active, std::memory_order_relaxed)) {}
        }
        proposal_input_cv_.notify_all();

        ProposalTerminal terminal;
        terminal.sequence = packet.proposal_sequence;
        terminal.generation = packet.generation;
        terminal.worker_index = worker_index;
        try {
            buildObservationSupersetScaffold(&packet, &canonical_scratch);
            completeObservationProposal(&packet, &scratch_solver);
            terminal.success = true;
        } catch (const std::exception& error) {
            terminal.error = error.what();
        } catch (...) {
            terminal.error = "unknown proposal worker failure";
        }
        terminal.packet = std::move(packet);
        const auto elapsed = costDurationNs(begin);

        std::unique_lock<std::mutex> lock(proposal_mutex_);
        if (terminal.generation != generation_.load(std::memory_order_acquire)) {
            proposal_stale_.fetch_add(1, std::memory_order_relaxed);
            proposal_active_workers_.fetch_sub(1, std::memory_order_relaxed);
            lock.unlock();
            proposal_reorder_cv_.notify_all();
            continue;
        }
        proposal_reorder_cv_.wait(lock, [this] {
            return stop_.load(std::memory_order_relaxed) ||
                   proposal_reorder_.size() <
                     static_cast<std::size_t>(proposal_worker_count_);
        });
        if (stop_.load(std::memory_order_relaxed)) {
            proposal_active_workers_.fetch_sub(1, std::memory_order_relaxed);
            lock.unlock();
            proposal_reorder_cv_.notify_all();
            break;
        }
        if (proposal_reorder_.count(terminal.sequence) != 0) {
            proposal_terminal_failures_.fetch_add(1, std::memory_order_relaxed);
            proposal_stale_.fetch_add(1, std::memory_order_relaxed);
            proposal_active_workers_.fetch_sub(1, std::memory_order_relaxed);
            lock.unlock();
            proposal_reorder_cv_.notify_all();
            continue; // The existing terminal still closes this exact ticket.
        }
        proposal_worker_has_terminal_[worker_index] = true;
        proposal_reorder_.emplace(terminal.sequence, std::move(terminal));
        proposal_completed_.fetch_add(1, std::memory_order_relaxed);
        proposal_worker_completed_[worker_index].fetch_add(1, std::memory_order_relaxed);
        proposal_worker_total_ns_[worker_index].fetch_add(elapsed, std::memory_order_relaxed);
        const auto occupancy = static_cast<std::uint32_t>(proposal_reorder_.size());
        auto reorder_max = proposal_reorder_max_occupancy_.load(std::memory_order_relaxed);
        while (reorder_max < occupancy &&
               !proposal_reorder_max_occupancy_.compare_exchange_weak(
                   reorder_max, occupancy, std::memory_order_relaxed)) {}
        proposal_active_workers_.fetch_sub(1, std::memory_order_relaxed);
        lock.unlock();
        proposal_reorder_cv_.notify_all();
    }
}

bool BuffRunePipeline::commitOrderedPacket(BuffYoloPacket packet)
{
    if (packet.generation != generation_.load(std::memory_order_acquire)) {
        proposal_stale_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (packet.frame.fb.task_mode != rm::FeedBackData::TASK_MODE::HIT_BIG_BUFF &&
        packet.frame.fb.task_mode != rm::FeedBackData::TASK_MODE::HIT_SMALL_BUFF) {
        throw std::runtime_error("three-worker committer received non-buff task");
    }
    const auto solve_begin = std::chrono::steady_clock::now();
    BuffDetectionPacket detection = solveRune(std::move(packet));
    const auto solve_end = std::chrono::steady_clock::now();
    detection.solve_ms = std::chrono::duration<double, std::milli>(
        solve_end - solve_begin).count();
    detection.solve_cost.outer_total_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            solve_end - solve_begin).count());
    detection.solve_cost.unaccounted_ns =
        detection.solve_cost.outer_total_ns > detection.solve_cost.accounted_ns
        ? detection.solve_cost.outer_total_ns - detection.solve_cost.accounted_ns : 0;
    if (detection.solve_cost.observation_superset.enabled &&
        detection.solve_cost.observation_superset.used_legacy_fallback) {
        detection.solve_cost.observation_superset.fallback_solve_ns =
            detection.solve_cost.outer_total_ns;
    }
    detection.infer_ms = detection.yolo_ms + detection.solve_ms;

    std::lock_guard<std::mutex> ordered_lock(ordered_commit_mutex_);
    if (detection.generation != generation_.load(std::memory_order_acquire)) {
        proposal_stale_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    solve_completed_.fetch_add(1, std::memory_order_relaxed);
    const auto commit_begin = std::chrono::steady_clock::now();
    BuffRuneResult result = buildResult(std::move(detection));
    result.solve_cost.observation_superset.ordered_commit_ns =
        costDurationNs(commit_begin);
    recordCompletionSample(result);
    output_mailbox_.publish(std::move(result));
    published_results_.fetch_add(1, std::memory_order_relaxed);
    proposal_committed_.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> proposal_lock(proposal_mutex_);
        if (proposal_inflight_ > 0) --proposal_inflight_;
    }
    proposal_input_cv_.notify_all();
    return true;
}

void BuffRunePipeline::proposalCommitLoop()
{
    while (!stop_.load(std::memory_order_relaxed)) {
        ProposalTerminal terminal;
        {
            std::unique_lock<std::mutex> lock(proposal_mutex_);
            bool head_gap = false;
            auto head_wait_begin = std::chrono::steady_clock::time_point{};
            while (!stop_.load(std::memory_order_relaxed) &&
                   proposal_reorder_.count(next_commit_sequence_) == 0) {
                if (!head_gap && !proposal_reorder_.empty()) {
                    head_gap = true;
                    head_wait_begin = std::chrono::steady_clock::now();
                    proposal_terminal_gaps_.fetch_add(1, std::memory_order_relaxed);
                }
                proposal_reorder_cv_.wait(lock);
            }
            if (head_gap) {
                proposal_head_wait_ns_.fetch_add(
                    costDurationNs(head_wait_begin), std::memory_order_relaxed);
            }
            if (stop_.load(std::memory_order_relaxed)) break;
            auto entry = proposal_reorder_.find(next_commit_sequence_);
            if (entry == proposal_reorder_.end()) continue;
            terminal = std::move(entry->second);
            proposal_reorder_.erase(entry);
            ++next_commit_sequence_;
            proposal_worker_has_terminal_[terminal.worker_index] = false;
        }
        proposal_reorder_cv_.notify_all();
        proposal_input_cv_.notify_all();

        if (terminal.generation != generation_.load(std::memory_order_acquire)) {
            proposal_stale_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (!terminal.success) {
            proposal_terminal_failures_.fetch_add(1, std::memory_order_relaxed);
            observation_proposal_fallbacks_.fetch_add(1, std::memory_order_relaxed);
            tools::logger()->error(
                "Buff proposal ticket {} failed; ordered legacy fallback: {}",
                terminal.sequence, terminal.error);
            if (terminal.packet.observation_proposal.has_value()) {
                auto& proposal = *terminal.packet.observation_proposal;
                proposal.cost.used_legacy_fallback = true;
                proposal.cost.proposal_ready = false;
                proposal.cost.coverage_complete = false;
                proposal.cost.fallback_events = 1;
                proposal.cost.fallback_reason = static_cast<std::uint32_t>(
                    BuffObservationFallbackReason::NumericalFailure);
            }
        }
        bool committed = false;
        try {
            committed = commitOrderedPacket(std::move(terminal.packet));
        } catch (const std::exception& error) {
            ordered_commit_failures_.fetch_add(1, std::memory_order_relaxed);
            tools::logger()->error("Buff proposal ordered commit failure: {}", error.what());
        } catch (...) {
            ordered_commit_failures_.fetch_add(1, std::memory_order_relaxed);
            tools::logger()->error("Buff proposal ordered commit failure: unknown");
        }
        if (!committed) {
            std::lock_guard<std::mutex> lock(proposal_mutex_);
            if (terminal.generation == generation_.load(std::memory_order_acquire) &&
                proposal_inflight_ > 0) {
                --proposal_inflight_;
            }
        }
        proposal_input_cv_.notify_all();
    }
}

void BuffRunePipeline::solveLoop()
{
    while (!stop_.load(std::memory_order_relaxed)) {
        BuffYoloPacket packet;
        {
            std::unique_lock<std::mutex> lock(yolo_mutex_);
            yolo_cv_.wait(lock, [this] {
                return latest_yolo_.has_value() || stop_.load(std::memory_order_relaxed);
            });
            if (stop_.load(std::memory_order_relaxed)) break;
            packet = std::move(*latest_yolo_);
            latest_yolo_.reset();
        }

        if (packet.generation != generation_.load(std::memory_order_acquire)) {
            continue;
        }
        if (ordered_commit_inline_ &&
            packet.frame.fb.task_mode != rm::FeedBackData::TASK_MODE::HIT_BIG_BUFF &&
            packet.frame.fb.task_mode != rm::FeedBackData::TASK_MODE::HIT_SMALL_BUFF) {
            tools::logger()->error(
                "Buff ordered-commit rejected non-buff task before solve: {}",
                static_cast<unsigned>(packet.frame.fb.task_mode));
            continue;
        }

        try {
            buildObservationSupersetScaffold(&packet);
            const auto solve_begin = std::chrono::steady_clock::now();
            BuffDetectionPacket detection = solveRune(std::move(packet));
            const auto solve_end = std::chrono::steady_clock::now();
            detection.solve_ms =
                std::chrono::duration<double, std::milli>(solve_end - solve_begin).count();
            detection.solve_cost.outer_total_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    solve_end - solve_begin).count());
            detection.solve_cost.unaccounted_ns =
                detection.solve_cost.outer_total_ns > detection.solve_cost.accounted_ns
                ? detection.solve_cost.outer_total_ns - detection.solve_cost.accounted_ns
                : 0;
            if (detection.solve_cost.observation_superset.enabled &&
                detection.solve_cost.observation_superset.used_legacy_fallback) {
                detection.solve_cost.observation_superset.fallback_solve_ns =
                    detection.solve_cost.outer_total_ns;
            }
            detection.infer_ms = detection.yolo_ms + detection.solve_ms;
            if (ordered_commit_inline_) {
                // Reset owns the same boundary mutex before changing generation
                // or clearing state. Thus a solved packet is either cancelled
                // before it is counted, or receives exactly one ordered
                // tracker/aim/control completion and publication.
                std::lock_guard<std::mutex> ordered_lock(ordered_commit_mutex_);
                if (detection.generation != generation_.load(std::memory_order_acquire)) {
                    continue;
                }
                solve_completed_.fetch_add(1, std::memory_order_relaxed);
                try {
                    const auto commit_begin = std::chrono::steady_clock::now();
                    BuffRuneResult result = buildResult(std::move(detection));
                    result.solve_cost.observation_superset.ordered_commit_ns =
                        costDurationNs(commit_begin);
                    recordCompletionSample(result);
                    output_mailbox_.publish(std::move(result));
                    published_results_.fetch_add(1, std::memory_order_relaxed);
                } catch (const std::exception& e) {
                    ordered_commit_failures_.fetch_add(1, std::memory_order_relaxed);
                    tools::logger()->error(
                        "Buff ordered-commit terminal failure after successful solve: {}",
                        e.what());
                } catch (...) {
                    ordered_commit_failures_.fetch_add(1, std::memory_order_relaxed);
                    tools::logger()->error(
                        "Buff ordered-commit terminal failure after successful solve: unknown");
                }
                continue;
            }
            if (detection.generation != generation_.load(std::memory_order_acquire)) {
                continue;
            }
            solve_completed_.fetch_add(1, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(detection_mutex_);
                if (latest_detection_.has_value()) {
                    detection_queue_overwrites_.fetch_add(1, std::memory_order_relaxed);
                }
                latest_detection_ = std::move(detection);
            }
            detection_cv_.notify_one();
        } catch (const std::exception& e) {
            tools::logger()->error("Buff solve pipeline exception: {}", e.what());
        }
    }
}

void BuffRunePipeline::trackAimLoop()
{
    while (!stop_.load(std::memory_order_relaxed)) {
        BuffDetectionPacket packet;
        {
            std::unique_lock<std::mutex> lock(detection_mutex_);
            detection_cv_.wait(lock, [this] {
                return latest_detection_.has_value() || stop_.load(std::memory_order_relaxed);
            });
            if (stop_.load(std::memory_order_relaxed)) break;
            packet = std::move(*latest_detection_);
            latest_detection_.reset();
        }

        if (packet.generation != generation_.load(std::memory_order_acquire)) {
            continue;
        }

        try {
            BuffRuneResult result = buildResult(std::move(packet));
            recordCompletionSample(result);
            if (result.frame.srcImg.empty()) continue;
            if (result.has_control &&
                result.frame.fb.task_mode != rm::FeedBackData::TASK_MODE::HIT_BIG_BUFF &&
                result.frame.fb.task_mode != rm::FeedBackData::TASK_MODE::HIT_SMALL_BUFF) {
                continue;
            }
            output_mailbox_.publish(std::move(result));
            published_results_.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& e) {
            tools::logger()->error("Buff track pipeline exception: {}", e.what());
        }
    }
}

BuffYoloPacket BuffRunePipeline::runYolo(rm::Frame frame, uint64_t generation)
{
    BuffYoloPacket packet;
    packet.frame = std::move(frame);
    packet.generation = generation;

    const bool is_big =
        packet.frame.fb.task_mode == rm::FeedBackData::TASK_MODE::HIT_BIG_BUFF;
    packet.mode = is_big ? BIG : SMALL;
    const bool draw_yolo =
        emit_debug_artifacts_ && draw_yolo_results_ && !packet.frame.debugImg.empty();
    cv::Mat* work_img = &packet.frame.srcImg;
    if (draw_yolo && !packet.frame.debugImg.empty()) {
        work_img = &packet.frame.debugImg;
    }

    packet.frame_timestamp = timestampForFrame(packet.frame);
    packet.bullet_speed =
        packet.frame.bullet_speed > 1.0 ? packet.frame.bullet_speed : packet.frame.fb.bullet_speed;

    {
        std::lock_guard<std::mutex> lock(infer_mutex_);
        packet.candidates = detector_.detect_candidates(packet.mode, *work_img, draw_yolo);
    }

    return packet;
}

BuffDetectionPacket BuffRunePipeline::solveRune(BuffYoloPacket packet)
{
    const auto packet_setup_begin = std::chrono::steady_clock::now();
    BuffDetectionPacket detection;
    detection.frame = std::move(packet.frame);
    detection.frame_timestamp = packet.frame_timestamp;
    detection.bullet_speed = packet.bullet_speed;
    detection.mode = packet.mode;
    detection.generation = packet.generation;
    detection.yolo_ms = packet.yolo_ms;
    if (packet.observation_proposal.has_value()) {
        detection.solve_cost.observation_superset = packet.observation_proposal->cost;
    }

    const auto* solve_candidates = packet.observation_proposal.has_value()
        ? &packet.observation_proposal->candidates
        : &packet.candidates;

    cv::Mat* work_img = nullptr;
    if (!detection.frame.srcImg.empty()) {
        work_img = &detection.frame.srcImg;
    } else {
        work_img = &detection.frame.debugImg;
    }
    detection.solve_cost.packet_setup_ns = costDurationNs(packet_setup_begin);
    if (work_img == nullptr || work_img->empty()) {
        detection.solve_cost.accounted_ns = detection.solve_cost.packet_setup_ns;
        return detection;
    }

    {
        const auto mutex_wait_begin = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(solve_mutex_);
        detection.solve_cost.solve_mutex_wait_ns = costDurationNs(mutex_wait_begin);

        const auto image_size_begin = std::chrono::steady_clock::now();
        solver_.set_image_size(work_img->size(), &detection.solve_cost.detector.pnp);
        detection.solve_cost.set_image_size_ns = costDurationNs(image_size_begin);

        const auto pose_begin = std::chrono::steady_clock::now();
        solver_.set_R_gimbal2world(gimbalQuaternion(detection.frame));
        detection.solve_cost.set_pose_ns = costDurationNs(pose_begin);

        if (packet.observation_proposal.has_value() &&
            packet.observation_proposal->observation.ready &&
            !packet.observation_proposal->observation.requires_legacy_fallback) {
            auto& observation_proposal = *packet.observation_proposal;
            if (!detector_.canonical_observation_commit_supported(
                    observation_proposal.observation)) {
                observation_proposal.cost.used_legacy_fallback = true;
                observation_proposal.cost.coverage_complete = false;
                observation_proposal.cost.proposal_ready = false;
                observation_proposal.cost.fallback_reason = static_cast<std::uint32_t>(
                    BuffObservationFallbackReason::HoldAmbiguity);
            }
            if (proposal_worker_count_ > 1 &&
                !observation_proposal.cost.used_legacy_fallback) {
                const SolverFrameContext ordered_frame = solver_.makeFrameContext(
                    work_img->size(), &detection.solve_cost.detector.pnp);
                const auto same_mat = [](const cv::Mat& lhs, const cv::Mat& rhs) {
                    return lhs.size() == rhs.size() && lhs.type() == rhs.type() &&
                           !lhs.empty() && cv::norm(lhs, rhs, cv::NORM_INF) == 0.0;
                };
                const bool calibration_matches =
                    same_mat(ordered_frame.camera_matrix,
                             observation_proposal.solver_frame.camera_matrix) &&
                    same_mat(ordered_frame.distort_coeffs,
                             observation_proposal.solver_frame.distort_coeffs) &&
                    ordered_frame.R_camera2gimbal.isApprox(
                        observation_proposal.solver_frame.R_camera2gimbal, 0.0) &&
                    ordered_frame.t_camera2gimbal.isApprox(
                        observation_proposal.solver_frame.t_camera2gimbal, 0.0) &&
                    ordered_frame.R_gimbal2world.isApprox(
                        observation_proposal.solver_frame.R_gimbal2world, 1e-12);
                if (!calibration_matches) {
                    observation_proposal.cost.used_legacy_fallback = true;
                    observation_proposal.cost.coverage_complete = false;
                    observation_proposal.cost.proposal_ready = false;
                    observation_proposal.cost.fallback_reason = static_cast<std::uint32_t>(
                        BuffObservationFallbackReason::PnpRMismatch);
                }
                const bool precomputed_complete =
                    observation_proposal.pnp.status == ExhaustivePnpStatus::Ready &&
                    observation_proposal.pnp.expected_solution_count ==
                        observation_proposal.pnp.solutions.size() &&
                    observation_proposal.cost.proposal_ready;
                if (!observation_proposal.cost.used_legacy_fallback &&
                    !precomputed_complete) {
                    observation_proposal.cost.used_legacy_fallback = true;
                    observation_proposal.cost.coverage_complete = false;
                    observation_proposal.cost.proposal_ready = false;
                    observation_proposal.cost.fallback_reason = static_cast<std::uint32_t>(
                        BuffObservationFallbackReason::PnpCap);
                } else if (!observation_proposal.cost.used_legacy_fallback) {
                    observation_pnp_proposals_.fetch_add(
                        observation_proposal.cost.pnp_proposal_count,
                        std::memory_order_relaxed);
                }
            }
            if (proposal_worker_count_ == 1) {
            observation_proposal.solver_frame = solver_.makeFrameContext(
                work_img->size(), &detection.solve_cost.detector.pnp);
            std::vector<SolverPnpHypothesis> pnp_hypotheses;
            pnp_hypotheses.reserve(observation_proposal.observation.targets.size());
            for (const auto& target : observation_proposal.observation.targets) {
                if (observation_proposal.cost.used_legacy_fallback) {
                    break;
                }
                const auto r_choice = std::find_if(
                    observation_proposal.observation.r_choices.begin(),
                    observation_proposal.observation.r_choices.end(),
                    [&](const BuffCanonicalRChoice& value) {
                        return value.hypothesis_index == target.hypothesis_index;
                    });
                if (r_choice == observation_proposal.observation.r_choices.end()) {
                    observation_proposal.cost.used_legacy_fallback = true;
                    observation_proposal.cost.fallback_reason = static_cast<std::uint32_t>(
                        BuffObservationFallbackReason::MissingCoverage);
                    break;
                }
                SolverPnpHypothesis hypothesis;
                hypothesis.hypothesis_index = target.hypothesis_index;
                hypothesis.source_pixel_signature =
                    observation_proposal.observation.template_result_pixels ^
                    (static_cast<std::uint64_t>(target.hypothesis_index) << 56U);
                hypothesis.r_center = r_choice->r_center;
                hypothesis.fanblades = target.fanblades;
                pnp_hypotheses.push_back(std::move(hypothesis));
            }
            if (!observation_proposal.cost.used_legacy_fallback) {
                const auto pnp_extract_begin = std::chrono::steady_clock::now();
                observation_proposal.pnp = solver_.buildExhaustiveProposal(
                    pnp_hypotheses, observation_proposal.solver_frame,
                    &detection.solve_cost.detector.pnp);
                observation_proposal.cost.pnp_extract_ns =
                    costDurationNs(pnp_extract_begin);
                observation_proposal.cost.pnp_proposal_count =
                    static_cast<std::uint32_t>(observation_proposal.pnp.solutions.size());
                const bool pnp_complete =
                    observation_proposal.pnp.status == ExhaustivePnpStatus::Ready &&
                    observation_proposal.pnp.expected_solution_count ==
                        observation_proposal.pnp.solutions.size();
                observation_proposal.cost.coverage_complete =
                    observation_proposal.cost.coverage_complete && pnp_complete;
                observation_proposal.cost.proposal_ready =
                    observation_proposal.cost.coverage_complete;
                if (!pnp_complete) {
                    observation_proposal.cost.used_legacy_fallback = true;
                    observation_proposal.cost.fallback_reason = static_cast<std::uint32_t>(
                        BuffObservationFallbackReason::PnpCap);
                    ++observation_proposal.cost.cap_events;
                    observation_cap_events_.fetch_add(1, std::memory_order_relaxed);
                }
                observation_proposal.cost.proposal_total_ns +=
                    observation_proposal.cost.pnp_extract_ns;
                observation_pnp_proposals_.fetch_add(
                    observation_proposal.cost.pnp_proposal_count,
                    std::memory_order_relaxed);
            }
            }
            if (observation_proposal.cost.used_legacy_fallback &&
                observation_proposal.cost.fallback_events == 0) {
                observation_proposal.cost.fallback_events = 1;
                observation_proposal_fallbacks_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        const auto detector_begin = std::chrono::steady_clock::now();
        if (packet.observation_proposal.has_value() &&
            packet.observation_proposal->cost.proposal_ready &&
            !packet.observation_proposal->cost.used_legacy_fallback) {
            auto& proposal = *packet.observation_proposal;
            const auto reduce_begin = std::chrono::steady_clock::now();
            detection.rune = detector_.solve_canonical_observation(
                detection.mode, proposal.observation, solver_, proposal.solver_frame,
                proposal.pnp, packet.frame_timestamp, &detection.solve_cost.detector);
            proposal.cost.pnp_reduce_ns = costDurationNs(reduce_begin);
            observation_ready_consumptions_.fetch_add(1, std::memory_order_relaxed);
        } else {
            detection.rune = detector_.solve_candidates(
                detection.mode, *work_img, solver_, *solve_candidates, packet.frame_timestamp,
                &detection.solve_cost.detector);
        }
        detection.solve_cost.detector_total_ns = costDurationNs(detector_begin);

        if (packet.observation_proposal.has_value()) {
            detection.solve_cost.observation_superset = packet.observation_proposal->cost;
        }

        const auto snapshot_begin = std::chrono::steady_clock::now();
        detection.switch_deferred = detector_.last_switch_deferred();
        detection.target_switched =
            detection.rune.has_value() ? detection.rune->target_switched : detector_.last_target_switched();
        detection.selected_target_index = detector_.last_selected_target_index();
        detection.solve_cost.state_snapshot_ns = costDurationNs(snapshot_begin);
    }

    const auto snapshot_begin = std::chrono::steady_clock::now();
    if (packet.observation_proposal.has_value()) {
        detection.candidates = std::move(packet.observation_proposal->candidates);
    } else {
        detection.candidates = std::move(packet.candidates);
    }
    detection.solve_cost.state_snapshot_ns += costDurationNs(snapshot_begin);
    detection.solve_cost.accounted_ns =
        detection.solve_cost.packet_setup_ns +
        detection.solve_cost.solve_mutex_wait_ns +
        detection.solve_cost.set_image_size_ns +
        detection.solve_cost.set_pose_ns +
        detection.solve_cost.detector_total_ns +
        detection.solve_cost.state_snapshot_ns;

    return detection;
}

BuffRuneResult BuffRunePipeline::buildResult(BuffDetectionPacket packet)
{
    const auto track_begin = std::chrono::steady_clock::now();
    BuffRuneResult result;
    result.frame = std::move(packet.frame);
    result.infer_ms = packet.infer_ms;
    result.yolo_ms = packet.yolo_ms;
    result.solve_ms = packet.solve_ms;
    result.solve_cost = packet.solve_cost;
    result.frame_timestamp = packet.frame_timestamp;
    result.switch_deferred = packet.switch_deferred;
    result.target_switched = packet.target_switched;
    result.selected_target_index = packet.selected_target_index;
    result.debug_artifacts_emitted = emit_debug_artifacts_;
    if (emit_debug_artifacts_) {
        result.overlay.clear();
        result.overlay.rerun_skip_image = true;
    }

    updateFps();
    result.fps = fps_;
    AimCommand aim_command;
    std::chrono::steady_clock::time_point debug_begin{};
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        result.rune = std::move(packet.rune);
        tracker_.update(result.rune, packet.frame_timestamp, packet.switch_deferred);
        const auto aim_timestamp = std::chrono::steady_clock::now();
        const double timestamp_age_s =
            framePipelineAgeSeconds(result.frame, packet.frame_timestamp, aim_timestamp);
        const double pipeline_delay_s =
            std::max(timestamp_age_s, buildProcessingDelayFloorSeconds(packet.infer_ms, 0.0));
        aim_command = aimer_.aim(
            tracker_,
            packet.bullet_speed,
            result.frame.poseEuler.yaw,
            result.frame.fb.yaw_speed,
            aim_timestamp,
            pipeline_delay_s);
        result.pipeline_delay_ms = aimer_.get_last_pipeline_delay() * 1000.0;
        result.predict_ms = aimer_.get_last_total_predict_time() * 1000.0;
        result.base_predict_ms = aimer_.get_last_base_predict_time() * 1000.0;
        result.fly_time_ms = aimer_.get_last_fly_time() * 1000.0;
        result.tracker_debug = tracker_.debugSnapshot(kHudPredictionDebugDtS);
        result.shot_gate = evaluateShotGate(
            result.frame,
            aim_command,
            result.rune,
            result.tracker_debug,
            result.switch_deferred,
            result.target_switched,
            true);
        if (result.shot_gate.allowed) {
            aimer_.markShotFired();
        }
        result.control = makeControl(result.frame, aim_command, result.shot_gate.allowed);

        // This is the completed auto-aim boundary: ordered detector/R-center/PnP
        // work has already committed, and tracker/aimer/shot-gate/control are all
        // ready.  Debug artifacts below are not part of the Talos control result.
        const auto essential_ready = std::chrono::steady_clock::now();
        result.has_control = true;
        result.essential_track_aim_ms =
            std::chrono::duration<double, std::milli>(essential_ready - track_begin).count();
        result.result_timestamp = essential_ready;
        result.completion_timestamp_ns = systemNowNs();
        result.completion_sequence =
            essential_completed_.fetch_add(1, std::memory_order_relaxed) + 1;
        result.completion_counters = counters();
        debug_begin = essential_ready;

        if (emit_debug_artifacts_) {
        visual_solver_.set_image_size(
            result.frame.debugImg.empty() ? result.frame.srcImg.size() : result.frame.debugImg.size());
        visual_solver_.set_R_gimbal2world(gimbalQuaternion(result.frame));
        const double debug_time_ms = debugOverlayTimeMs(result.frame, packet.frame_timestamp);

        for (const auto& candidate : packet.candidates) {
            const std::string label =
                cv::format("%s %.2f", yoloCandidateLabel(candidate.label), candidate.prob);
            const uint32_t color = yoloCandidateColor(candidate.label);
            appendOverlayBox2D(
                result.overlay,
                "buff/overlay2d/yolo_boxes",
                candidate.rect,
                color,
                2.0f,
                label,
                false);
            for (size_t index = 0; index < candidate.kpt.size(); ++index) {
                const bool is_r_point = index == 4;
                appendOverlayPoint2D(
                    result.overlay,
                    "buff/overlay2d/yolo_keypoints",
                    candidate.kpt[index],
                    is_r_point ? rgba(255, 255, 0) : rgba(255, 64, 64),
                    is_r_point ? 5.0f : 3.5f,
                    is_r_point ? "R" : std::to_string(index),
                    false);
            }
        }

        if (result.rune.has_value()) {
            const auto& rune = *result.rune;
            appendOverlayPoint2D(
                result.overlay,
                "buff/overlay2d/r_center",
                rune.r_center,
                rgba(255, 255, 0),
                5.0f,
                "R",
                false);

            std::vector<cv::Point2f> target_polygon;
            const auto& target = rune.target();
            for (size_t index = 0; index < std::min<size_t>(4, target.points.size()); ++index) {
                target_polygon.push_back(target.points[index]);
            }
            appendOverlayLineStrip2D(
                result.overlay,
                "buff/overlay2d/actual_target",
                std::move(target_polygon),
                rgba(0, 255, 255),
                2.0f,
                true,
                "actual",
                false);

            appendBuffGuideSkeleton3D(
                result.overlay,
                visual_solver_,
                rune.xyz_in_world,
                rune.ypr_in_world);

            appendOverlayPoint3D(
                result.overlay,
                "buff/overlay3d/rune_center",
                rune.xyz_in_world,
                rgba(80, 160, 255),
                0.03f,
                "rune",
                false);
            appendOverlayPoint3D(
                result.overlay,
                "buff/overlay3d/actual_target_blade",
                rune.blade_xyz_in_world,
                rgba(0, 255, 170),
                0.03f,
                "actual target",
                false);

            const auto target_outline_world =
                buildSolvedBladeOutlineWorld(visual_solver_, rune.xyz_in_world, rune.ypr_in_world);
            appendOverlayLineStrip3D(
                result.overlay,
                "buff/overlay3d/actual/target_outline",
                target_outline_world,
                rgba(0, 255, 170),
                0.01f,
                true,
                "",
                false);
            for (const auto& point : target_outline_world) {
                appendOverlayPoint3D(
                    result.overlay,
                    "buff/overlay3d/actual/target_keypoints",
                    point,
                    rgba(255, 196, 64),
                    0.018f,
                    "",
                    false);
            }

            appendOverlayLineStrip3D(
                result.overlay,
                "buff/overlay3d/actual_target_link",
                {rune.xyz_in_world, rune.blade_xyz_in_world},
                rgba(0, 255, 170),
                0.01f,
                false,
                "",
                false);

            for (size_t index = 0; index < rune.fanblades.size(); ++index) {
                const auto& blade = rune.fanblades[index];
                if (blade.solved && index != 0) {
                    appendOverlayLineStrip3D(
                        result.overlay,
                        "buff/overlay3d/actual/observed_outlines",
                        buildSolvedBladeOutlineWorld(
                            visual_solver_, blade.rune_xyz_in_world, blade.ypr_in_world),
                        rgba(200, 200, 200, 120),
                        0.006f,
                        true,
                        "",
                        false);
                }
                appendOverlayPoint3D(
                    result.overlay,
                    "buff/overlay3d/blade_centers",
                    blade.blade_xyz_in_world,
                    index == 0 ? rgba(0, 255, 170) : rgba(200, 200, 200, 180),
                    index == 0 ? 0.028f : 0.02f,
                    std::string(),
                    false);
            }
        }

        if (!tracker_.is_lost()) {
            const Eigen::VectorXd current_state = tracker_.get_state();
            const double target_roll_offset = tracker_.selected_target_roll_offset();
            const auto current_overlay = projectBladeOverlayState(
                visual_solver_,
                tracker_,
                current_state,
                target_roll_offset);
            if (!result.rune.has_value() && current_state.size() >= 6 && current_state.array().isFinite().all()) {
                appendBuffGuideSkeleton3D(
                    result.overlay,
                    visual_solver_,
                    current_overlay.rune_center_world,
                    Eigen::Vector3d(current_state[4], 0.0, current_state[5] + target_roll_offset));
            }

            appendOverlayLineStrip3D(
                result.overlay,
                "buff/overlay3d/tracker/current_outline",
                current_overlay.blade_outline_world,
                rgba(0, 220, 255),
                0.008f,
                true,
                "",
                false);
            for (const auto& point : current_overlay.blade_keypoints_world) {
                appendOverlayPoint3D(
                    result.overlay,
                    "buff/overlay3d/tracker/current_keypoints",
                    point,
                    rgba(0, 220, 255, 180),
                    0.014f,
                    "",
                    false);
            }
            appendOverlayPoint3D(
                result.overlay,
                "buff/overlay3d/tracker/current_target",
                current_overlay.blade_center_world,
                rgba(0, 220, 255),
                0.028f,
                "",
                false);
            appendOverlayLineStrip3D(
                result.overlay,
                "buff/overlay3d/tracker/current_link",
                {current_overlay.rune_center_world, current_overlay.blade_center_world},
                rgba(0, 220, 255),
                0.008f,
                false,
                "",
                false);

            const auto historical_fixed_200 = buildHistoricalPredictionMatch(debug_time_ms);
            if (historical_fixed_200.has_value()) {
                const auto fixed_200ms_overlay = projectBladeOverlayState(
                    visual_solver_,
                    tracker_,
                    historical_fixed_200->predicted_state,
                    historical_fixed_200->target_roll_offset);
                appendOverlayLineStrip2D(
                    result.overlay,
                    "buff/overlay2d/fixed_200ms",
                    fixed_200ms_overlay.projected_points,
                    rgba(255, 0, 255),
                    2.0f,
                    true,
                    "fixed 200ms",
                    false);
                appendOverlayPoint3D(
                    result.overlay,
                    "buff/overlay3d/fixed_200ms_target",
                    fixed_200ms_overlay.blade_center_world,
                    rgba(255, 0, 255),
                    0.03f,
                    "fixed 200ms",
                    false);
                appendOverlayLineStrip3D(
                    result.overlay,
                    "buff/overlay3d/tracker/fixed_200ms_outline",
                    fixed_200ms_overlay.blade_outline_world,
                    rgba(255, 0, 255),
                    0.008f,
                    true,
                    "",
                    false);
                for (const auto& point : fixed_200ms_overlay.blade_keypoints_world) {
                    appendOverlayPoint3D(
                        result.overlay,
                        "buff/overlay3d/tracker/fixed_200ms_keypoints",
                        point,
                        rgba(255, 0, 255, 180),
                        0.014f,
                        "",
                        false);
                }
                appendOverlayLineStrip3D(
                    result.overlay,
                    "buff/overlay3d/fixed_200ms_link",
                    {fixed_200ms_overlay.rune_center_world, fixed_200ms_overlay.blade_center_world},
                    rgba(255, 0, 255),
                    0.01f,
                    false,
                    "",
                    false);
            }
        }

        predictionCsvLogger().log(
            result.frame, result.rune, tracker_, visual_solver_, aimer_, aim_command,
            result.shot_gate,
            result.switch_deferred, result.target_switched,
            0, packet.bullet_speed, aimer_.get_last_pipeline_delay(),
            aimer_.get_last_base_predict_time(), aimer_.get_last_fly_time());

        if (!result.frame.debugImg.empty()) {
            drawAllBladePhaseLabels(result.frame.debugImg, visual_solver_, tracker_);
            if (result.rune.has_value()) {
                drawObservedBladeLabels(result.frame.debugImg, *result.rune, tracker_);
                drawRLogoOverlay(result.frame.debugImg, *result.rune, draw_r_binary_mask_);
                drawPnpDebugOverlay(result.frame.debugImg, *result.rune);
            }
            const double predict_dt_s = aim_command.control
                                            ? aimer_.get_last_total_predict_time()
                                            : aimer_.get_last_base_predict_time();
            drawPredictionOverlay(
                result.frame.debugImg, visual_solver_, tracker_, predict_dt_s, aim_command.control);
            drawPastPredictionOverlay(result.frame.debugImg, debug_time_ms);
        }
        enqueueDebugPredictionOverlay(debug_time_ms);
        }
    }

    if (emit_debug_artifacts_ && !result.frame.debugImg.empty()) {
        const cv::Scalar green(0, 255, 0);
        cv::putText(
            result.frame.debugImg, cv::format("Buff FPS: %d", result.fps), cv::Point(20, 70),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, green, 2);
    }

    const auto track_end = std::chrono::steady_clock::now();
    if (emit_debug_artifacts_) {
        result.debug_artifact_ms =
            std::chrono::duration<double, std::milli>(track_end - debug_begin).count();
    }
    result.track_aim_ms =
        std::chrono::duration<double, std::milli>(track_end - track_begin).count();
    maybeCaptureExactValid(result);

    return result;
}

void BuffRunePipeline::maybeCaptureExactValid(const BuffRuneResult& result)
{
    if (exact_valid_capture_) {
        exact_valid_capture_->observeCompletion(result);
    }
}

void BuffRunePipeline::recordCompletionSample(const BuffRuneResult& result)
{
    if (!collect_completion_samples_) return;
    std::lock_guard<std::mutex> lock(completion_samples_mutex_);
    completion_samples_.push_back(BuffRuneCompletionSample{
        result.completion_sequence,
        result.yolo_ms,
        result.solve_ms,
        result.essential_track_aim_ms,
        result.debug_artifact_ms,
        result.solve_cost});
}

void BuffRunePipeline::enqueueDebugPredictionOverlay(double current_time_ms)
{
    enqueueHistoricalPredictionSeed(current_time_ms, tracker_.debugSnapshot(0.0));
}

void BuffRunePipeline::drawPastPredictionOverlay(cv::Mat& image, double current_time_ms)
{
    if (image.empty() || !std::isfinite(current_time_ms)) {
        return;
    }

    const auto historical_prediction = buildHistoricalPredictionMatch(current_time_ms);
    if (!historical_prediction.has_value()) {
        return;
    }

    const auto overlay = projectBladeOverlayState(
        visual_solver_,
        tracker_,
        historical_prediction->predicted_state,
        historical_prediction->target_roll_offset);
    const cv::Scalar red(0, 0, 255);
    const auto& points = overlay.projected_points;
    drawProjectedBlade(image, points, red, 2);
    if (points.size() > 4 && isDrawablePoint(points[4], image)) {
        cv::drawMarker(image, points[4], red, cv::MARKER_CROSS, 26, 2, cv::LINE_AA);
        cv::circle(image, points[4], 12, red, 2, cv::LINE_AA);
        const std::string label = cv::format(
            "pred -%.0fms",
            std::max(0.0, current_time_ms - historical_prediction->source_time_ms));
        cv::putText(
            image, label, points[4] + cv::Point2f(12.0f, 22.0f),
            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(
            image, label, points[4] + cv::Point2f(12.0f, 22.0f),
            cv::FONT_HERSHEY_SIMPLEX, 0.55, red, 2, cv::LINE_AA);
    }
}

void BuffRunePipeline::enqueueHistoricalPredictionSeed(
    double current_time_ms,
    const BuffTracker::DebugSnapshot& tracker_debug)
{
    if (!std::isfinite(current_time_ms) || tracker_.is_lost()) {
        historical_prediction_seeds_.clear();
        return;
    }

    const Eigen::VectorXd current_state = tracker_.get_state();
    if (current_state.size() < 10 || !current_state.array().isFinite().all()) {
        historical_prediction_seeds_.clear();
        return;
    }

    HistoricalPredictionSeed seed;
    seed.time_ms = current_time_ms;
    seed.relative_time_s = tracker_.current_relative_time_s();
    seed.state = current_state;
    seed.target_roll_offset = tracker_debug.selected_roll_offset;
    seed.voter_direction = tracker_debug.direction;
    seed.history_size = tracker_debug.history_size;
    seed.reinit_reason = tracker_debug.reinit_reason;
    seed.switch_deferred = tracker_debug.switch_deferred;
    seed.target_switched = tracker_debug.target_switched;
    if (!isGoodHistoricalPredictionSeed(
            seed.state,
            seed.voter_direction,
            seed.history_size,
            seed.reinit_reason,
            seed.switch_deferred,
            seed.target_switched)) {
        historical_prediction_seeds_.clear();
        return;
    }
    historical_prediction_seeds_.push_back(std::move(seed));

    while (!historical_prediction_seeds_.empty() &&
           historical_prediction_seeds_.front().time_ms <
               current_time_ms - kPastPredictionOverlayRetentionMs) {
        historical_prediction_seeds_.pop_front();
    }
    while (historical_prediction_seeds_.size() > 256) {
        historical_prediction_seeds_.pop_front();
    }
}

std::optional<BuffRunePipeline::HistoricalPredictionMatch>
BuffRunePipeline::buildHistoricalPredictionMatch(double current_time_ms) const
{
    if (!std::isfinite(current_time_ms)) {
        return std::nullopt;
    }

    const double target_source_time_ms =
        current_time_ms - kPastPredictionOverlayDtS * 1000.0;
    for (auto it = historical_prediction_seeds_.rbegin();
         it != historical_prediction_seeds_.rend();
         ++it) {
        if (!std::isfinite(it->time_ms) || it->time_ms > target_source_time_ms) {
            continue;
        }
        if (target_source_time_ms - it->time_ms > kHistoricalPredictionMaxSourceLagMs) {
            return std::nullopt;
        }
        if (!isGoodHistoricalPredictionSeed(
                it->state,
                it->voter_direction,
                it->history_size,
                it->reinit_reason,
                it->switch_deferred,
                it->target_switched)) {
            continue;
        }

        HistoricalPredictionMatch match;
        match.source_time_ms = it->time_ms;
        match.eval_dt_s = std::max(0.0, (current_time_ms - it->time_ms) * 0.001);
        match.target_roll_offset = it->target_roll_offset;
        match.predicted_state = tracker_.predict_from_state(
            it->state,
            match.eval_dt_s,
            it->voter_direction,
            it->relative_time_s);
        if (match.predicted_state.size() < 10 ||
            !match.predicted_state.array().isFinite().all()) {
            return std::nullopt;
        }
        return match;
    }

    return std::nullopt;
}

bool BuffRunePipeline::refreshControlWithFeedback(
    BuffRuneResult* result,
    const rm::FeedBackData& live_feedback,
    std::chrono::steady_clock::time_point timestamp)
{
    if (result == nullptr || !result->has_control) return false;

    result->frame.fb = live_feedback;
    result->frame.poseEuler.pitch = live_feedback.gimbal_pitch;
    result->frame.poseEuler.yaw = live_feedback.gimbal_yaw;
    result->frame.poseEuler.roll = live_feedback.gimbal_roll;

    const bool preserve_shot =
        result->control.shot_mode == rm::ControlData::SHOT_MODE::SHOT_ONCE;
    const double bullet_speed = result->frame.bullet_speed > 1.0
                                    ? result->frame.bullet_speed
                                    : result->frame.fb.bullet_speed;

    AimCommand refreshed_command;
    const double timestamp_age_s =
        framePipelineAgeSeconds(result->frame, result->frame_timestamp, timestamp);
    const double queue_ms =
        result->result_timestamp.time_since_epoch().count() == 0
            ? 0.0
            : std::max(
                  0.0,
                  std::chrono::duration<double, std::milli>(
                      timestamp - result->result_timestamp).count());
    const double pipeline_delay_s = std::max(
        timestamp_age_s,
        buildProcessingDelayFloorSeconds(result->infer_ms, result->track_aim_ms, queue_ms));
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        refreshed_command = aimer_.aim(
            tracker_,
            bullet_speed,
            result->frame.poseEuler.yaw,
            live_feedback.yaw_speed,
            timestamp,
            pipeline_delay_s);
        result->pipeline_delay_ms = aimer_.get_last_pipeline_delay() * 1000.0;
        result->predict_ms = aimer_.get_last_total_predict_time() * 1000.0;
        result->base_predict_ms = aimer_.get_last_base_predict_time() * 1000.0;
        result->fly_time_ms = aimer_.get_last_fly_time() * 1000.0;
        result->tracker_debug = tracker_.debugSnapshot(kHudPredictionDebugDtS);
        AimCommand gate_command = refreshed_command;
        if (preserve_shot) {
            gate_command.shoot = true;
        }
        result->shot_gate = evaluateShotGate(
            result->frame,
            gate_command,
            result->rune,
            result->tracker_debug,
            result->switch_deferred,
            result->target_switched,
            false);
        if (result->shot_gate.allowed) {
            aimer_.markShotFired();
        }
        predictionCsvLogger().log(
            result->frame, result->rune, tracker_, visual_solver_, aimer_, gate_command,
            result->shot_gate,
            result->switch_deferred, result->target_switched,
            1, bullet_speed, aimer_.get_last_pipeline_delay(),
            aimer_.get_last_base_predict_time(), aimer_.get_last_fly_time());
        refreshed_command = gate_command;
    }

    result->control = makeControl(result->frame, refreshed_command, result->shot_gate.allowed);

    return true;
}

std::chrono::steady_clock::time_point BuffRunePipeline::timestampForFrame(const rm::Frame& frame)
{
    const double frame_ms =
        frame.usb_timeStamp > 0.0 ? frame.usb_timeStamp : finiteOr(frame.timeStamp, 0.0);
    const auto now = std::chrono::steady_clock::now();
    if (frame_ms <= 0.0) return now;

    std::lock_guard<std::mutex> lock(time_mutex_);
    if (!time_base_ready_) {
        time_base_ready_ = true;
        time_base_ms_ = frame_ms;
        time_base_tp_ = now;
        return now;
    }

    const auto delta =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(frame_ms - time_base_ms_));
    return time_base_tp_ + delta;
}

Eigen::Quaterniond BuffRunePipeline::gimbalQuaternion(const rm::Frame& frame) const
{
    // tools::rotation_matrix uses +yaw -> +world-y and +pitch -> -world-z.
    // Gimbal command/feedback convention in the main auto-aim path is the opposite:
    // +yaw turns toward image right (-world-y), +pitch raises the barrel (+world-z).
    const Eigen::Vector3d ypr(
        -finiteOr(frame.poseEuler.yaw, 0.0) * rm::D2R,
        -finiteOr(frame.poseEuler.pitch, 0.0) * rm::D2R,
        finiteOr(frame.poseEuler.roll, 0.0) * rm::D2R);
    return Eigen::Quaterniond(tools::rotation_matrix(ypr));
}

BuffShotGateSnapshot BuffRunePipeline::evaluateShotGate(
    const rm::Frame& frame,
    const AimCommand& command,
    const std::optional<PowerRune>& rune,
    const BuffTracker::DebugSnapshot& tracker_debug,
    bool switch_deferred,
    bool target_switched,
    bool update_stability)
{
    BuffShotGateSnapshot gate;
    gate.requested = command.shoot;

    gate.pending_detected =
        rune.has_value() && !rune->fanblades.empty() && !rune->is_unsolve();
    if (gate.pending_detected) {
        const auto& target = rune->target();
        gate.r_center_ok = isFinitePoint(rune->r_center) && target.points.size() >= 4;
        gate.pnp_reproj_error_px = target.pnp_reproj_error_px;
        gate.pnp_model_center_error_px = target.pnp_model_center_error_px;
        gate.pnp_ok =
            target.solved &&
            std::isfinite(gate.pnp_reproj_error_px) &&
            std::isfinite(gate.pnp_model_center_error_px) &&
            gate.pnp_reproj_error_px <= shot_gate_max_pnp_reproj_error_px_ &&
            gate.pnp_model_center_error_px <= shot_gate_max_model_center_error_px_;
    }

    gate.tracker_ok =
        !tracker_debug.lost &&
        tracker_debug.reinit_reason == 0 &&
        !switch_deferred &&
        !target_switched;

    if (command.control) {
        gate.yaw_error_deg =
            angleErrorDeg(command.yaw * rm::R2D, finiteOr(frame.poseEuler.yaw, 0.0));
        gate.pitch_error_deg =
            std::abs(command.pitch * rm::R2D - finiteOr(frame.poseEuler.pitch, 0.0));
        gate.gimbal_ok =
            std::isfinite(gate.yaw_error_deg) &&
            std::isfinite(gate.pitch_error_deg) &&
            gate.yaw_error_deg <= shot_gate_max_yaw_error_deg_ &&
            gate.pitch_error_deg <= shot_gate_max_pitch_error_deg_;
    }

    const bool stability_sample_ok =
        gate.pending_detected && gate.r_center_ok && gate.pnp_ok && gate.tracker_ok;
    if (update_stability) {
        if (stability_sample_ok) {
            shot_gate_stable_frames_ =
                std::min(shot_gate_stable_frames_ + 1, shot_gate_min_stable_frames_ + 1000);
        } else {
            shot_gate_stable_frames_ = 0;
        }
    }
    gate.stable_frames = shot_gate_stable_frames_;
    gate.stable_ok = gate.stable_frames >= shot_gate_min_stable_frames_;

    if (!gate.requested) {
        gate.reason_code = 1;
    } else if (!shot_gate_enabled_) {
        gate.allowed = true;
        gate.reason_code = 0;
    } else if (!gate.pending_detected) {
        gate.reason_code = 2;
    } else if (!gate.r_center_ok) {
        gate.reason_code = 3;
    } else if (!gate.pnp_ok) {
        gate.reason_code = 4;
    } else if (!gate.tracker_ok) {
        gate.reason_code = 5;
    } else if (!gate.gimbal_ok) {
        gate.reason_code = 6;
    } else if (!gate.stable_ok) {
        gate.reason_code = 7;
    } else {
        gate.allowed = true;
        gate.reason_code = 0;
    }

    return gate;
}

rm::ControlData BuffRunePipeline::makeControl(
    const rm::Frame& frame, const AimCommand& command, bool shot_allowed)
{
    rm::ControlData control;
    control.shot_buff_mode = buffModeFlag(frame);

    if (command.control) {
        control.gimbal_yaw = static_cast<float>(command.yaw * rm::R2D);
        control.gimbal_pitch = static_cast<float>(command.pitch * rm::R2D);
        control.aiming_state = rm::ControlData::AIMING_STATE::TARGET_DETECTED;
        control.shot_mode =
            shot_allowed ? rm::ControlData::SHOT_MODE::SHOT_ONCE
                         : rm::ControlData::SHOT_MODE::AIM_ONLY;
        last_valid_control_ = control;
    } else {
        if (last_valid_control_.has_value()) {
            control.gimbal_yaw = last_valid_control_->gimbal_yaw;
            control.gimbal_pitch = last_valid_control_->gimbal_pitch;
        } else {
            control.gimbal_yaw = frame.poseEuler.yaw;
            control.gimbal_pitch = frame.poseEuler.pitch;
        }
        control.aiming_state = rm::ControlData::AIMING_STATE::AIMMING_NO_TARGET;
        control.shot_mode = rm::ControlData::SHOT_MODE::DO_NOTHING;
    }

    return control;
}

uint8_t BuffRunePipeline::buffModeFlag(const rm::Frame& frame) const
{
    if (frame.fb.task_mode == rm::FeedBackData::TASK_MODE::HIT_BIG_BUFF) {
        return 2;
    }
    if (frame.fb.task_mode == rm::FeedBackData::TASK_MODE::HIT_SMALL_BUFF) {
        return 1;
    }
    return rm::ControlData::SHOT_BUFF_MODE::SHOT_BUFF_OFF;
}

void BuffRunePipeline::updateFps()
{
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(time_mutex_);
    if (fps_start_tp_.time_since_epoch().count() == 0) {
        fps_start_tp_ = now;
    }

    fps_count_++;
    if (now - fps_start_tp_ >= std::chrono::seconds(1)) {
        fps_ = fps_count_;
        fps_count_ = 0;
        fps_start_tp_ = now;
    }
}

}  // namespace auto_buff
