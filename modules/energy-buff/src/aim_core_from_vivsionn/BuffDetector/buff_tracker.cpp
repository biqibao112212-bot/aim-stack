#include "buff_tracker.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace
{
Eigen::VectorXd measurement_difference(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    Eigen::VectorXd d = a - b;
    if (d.size() > 0) {
        d[0] = tools::limit_rad(d[0]);
    }
    if (d.size() > 4) {
        d[4] = tools::limit_rad(d[4]);
    }
    return d;
}

template <typename T>
void read_optional_scalar(const YAML::Node& yaml, const std::string& key, T& value)
{
    if (!yaml || !yaml[key]) {
        return;
    }
    value = yaml[key].as<T>();
}

template <typename T>
void read_env_scalar(const char* name, T& value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return;
    }
    std::stringstream stream(raw);
    T parsed{};
    if (stream >> parsed) {
        value = parsed;
    }
}

constexpr double kBigBuffBaseSpeed = 2.090;
constexpr double kBigBuffAInit = 0.9125;
constexpr double kBigBuffWInit = 1.942;
constexpr double kBigBuffAMin = 0.780;
constexpr double kBigBuffAMax = 1.045;
constexpr double kBigBuffWMin = 1.884;
constexpr double kBigBuffWMax = 2.000;
constexpr double kBigBuffSpeedHardMax = 2.35;
constexpr double kBigBuffPredictionDeltaSlackRad = 0.02;

double clamp_big_buff_speed_magnitude(double speed)
{
    if (!std::isfinite(speed)) {
        return speed;
    }
    return std::clamp(std::abs(speed), 0.0, kBigBuffSpeedHardMax);
}

bool env_flag_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

double clamp_or_default(double value, double min_value, double max_value, double default_value)
{
    if (!std::isfinite(value)) {
        return default_value;
    }
    return std::clamp(value, min_value, max_value);
}

bool has_big_buff_curve_model(double a, double w)
{
    return std::isfinite(a) && std::isfinite(w) && a >= kBigBuffAMin && w >= kBigBuffWMin;
}

double big_buff_speed(double a, double w, double phi, double t)
{
    a = clamp_or_default(a, kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
    w = clamp_or_default(w, kBigBuffWMin, kBigBuffWMax, kBigBuffWInit);
    if (!std::isfinite(phi)) {
        phi = 0.0;
    }
    return a * std::sin(w * t + phi) + (kBigBuffBaseSpeed - a);
}

double big_buff_angle(double a, double w, double phi, double t)
{
    a = clamp_or_default(a, kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
    w = clamp_or_default(w, kBigBuffWMin, kBigBuffWMax, kBigBuffWInit);
    if (!std::isfinite(phi)) {
        phi = 0.0;
    }
    return -a / w * std::cos(w * t + phi) + (kBigBuffBaseSpeed - a) * t;
}

double big_buff_angle_delta(double a, double w, double phi, double t_start, double t_end)
{
    return big_buff_angle(a, w, phi, t_end) - big_buff_angle(a, w, phi, t_start);
}

double big_runev2_speed(double a, double phase)
{
    a = clamp_or_default(a, kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
    if (!std::isfinite(phase)) {
        phase = 0.0;
    }
    return a * std::sin(phase) + (kBigBuffBaseSpeed - a);
}

double big_runev2_curve_speed_from_state(const Eigen::VectorXd& x)
{
    if (x.size() < 10 || !x.array().isFinite().all()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::abs(big_runev2_speed(x[7], x[9]));
}

double big_runev2_curve_phi_from_state(const Eigen::VectorXd& x)
{
    if (x.size() < 10 || !x.array().isFinite().all()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return x[9];
}

double big_runev2_angle_delta(double a, double w, double phase, double dt)
{
    a = clamp_or_default(a, kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
    w = clamp_or_default(w, kBigBuffWMin, kBigBuffWMax, kBigBuffWInit);
    if (!std::isfinite(phase)) {
        phase = 0.0;
    }
    dt = std::max(0.0, dt);
    return (kBigBuffBaseSpeed - a) * dt +
        a / w * (std::cos(phase) - std::cos(phase + w * dt));
}

double big_buff_theta(double w, double phi, double t)
{
    if (!std::isfinite(w) || !std::isfinite(phi) || !std::isfinite(t)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return tools::limit_rad(w * t + phi);
}

template <typename Iterator>
std::optional<double> estimate_linear_rate_from_history(
    Iterator begin_it,
    Iterator end_it,
    std::size_t max_samples,
    double max_window_s)
{
    if (begin_it == end_it || max_samples < 2 || !std::isfinite(max_window_s) || max_window_s <= 0.0) {
        return std::nullopt;
    }

    const auto last_it = end_it - 1;
    const double latest_t = last_it->t;
    if (!std::isfinite(latest_t)) {
        return std::nullopt;
    }

    std::array<double, 16> times{};
    std::array<double, 16> angles{};
    max_samples = std::min(max_samples, times.size());
    std::size_t count = 0;
    for (auto it = last_it;; --it) {
        if (!std::isfinite(it->t) || !std::isfinite(it->angle)) {
            break;
        }
        if (latest_t - it->t > max_window_s && count >= 2) {
            break;
        }
        times[count] = it->t;
        angles[count] = it->angle;
        count++;
        if (count >= max_samples || it == begin_it) {
            break;
        }
    }
    if (count < 2) {
        return std::nullopt;
    }

    double sum_t = 0.0;
    double sum_a = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum_t += times[i];
        sum_a += angles[i];
    }
    const double mean_t = sum_t / static_cast<double>(count);
    const double mean_a = sum_a / static_cast<double>(count);

    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double dt = times[i] - mean_t;
        numerator += dt * (angles[i] - mean_a);
        denominator += dt * dt;
    }
    if (!std::isfinite(numerator) || !std::isfinite(denominator) || denominator < 1e-6) {
        return std::nullopt;
    }
    return numerator / denominator;
}

struct CurveBlendState
{
    double alpha = 1.0;
    double fallback_speed = std::numeric_limits<double>::quiet_NaN();
};

CurveBlendState compute_curve_blend_state(
    double curve_speed_now,
    double current_speed,
    double observed_speed,
    bool observed_speed_valid,
    int history_size,
    bool fit_valid)
{
    const double abs_current_speed = std::abs(current_speed);
    CurveBlendState state;
    state.fallback_speed = observed_speed_valid && std::isfinite(observed_speed)
        ? 0.6 * abs_current_speed + 0.4 * observed_speed
        : abs_current_speed;

    const auto normalize_range = [](double value, double lo, double hi) {
        if (!std::isfinite(value) || hi <= lo) {
            return 0.0;
        }
        return std::clamp((value - lo) / (hi - lo), 0.0, 1.0);
    };

    const double speed_alpha = normalize_range(std::abs(curve_speed_now), 0.7, 1.4);
    const double fit_alpha = fit_valid ? 1.0 : 0.4;
    const double history_alpha = normalize_range(static_cast<double>(history_size), 20.0, 60.0);
    double consistency_alpha = 1.0;
    if (observed_speed_valid && std::isfinite(observed_speed)) {
        consistency_alpha = 1.0 - std::clamp(
            std::abs(observed_speed - std::abs(curve_speed_now)) / 0.8,
            0.0,
            1.0);
    }

    state.alpha = speed_alpha * fit_alpha * history_alpha * consistency_alpha;
    if (!std::isfinite(state.alpha)) {
        state.alpha = 0.2;
    }
    state.alpha = std::clamp(state.alpha, 0.2, 0.95);
    return state;
}

double normalize_roll_measurement(double raw_roll, double reference_roll)
{
    const double step = CV_2PI / 5.0;
    double best_roll = raw_roll;
    double min_diff = std::numeric_limits<double>::max();

    for (int k = -2; k <= 2; ++k) {
        double candidate = raw_roll + k * step;
        const double n = std::round((reference_roll - candidate) / CV_2PI);
        candidate += n * CV_2PI;
        const double diff = std::abs(candidate - reference_roll);
        if (diff < min_diff) {
            min_diff = diff;
            best_roll = candidate;
        }
    }

    return best_roll;
}

double normalize_full_turn_measurement(double raw_roll, double reference_roll)
{
    if (!std::isfinite(raw_roll) || !std::isfinite(reference_roll)) {
        return raw_roll;
    }
    return raw_roll + std::round((reference_roll - raw_roll) / CV_2PI) * CV_2PI;
}

double phase_offset_distance(double a, double b)
{
    constexpr double step = CV_2PI / 5.0;
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return 0.0;
    }
    double d = std::fmod(std::abs(a - b), step);
    if (d < 0.0) {
        d += step;
    }
    return std::min(d, step - d);
}

int normalize_phase_index(int index)
{
    int normalized = index % 5;
    if (normalized < 0) {
        normalized += 5;
    }
    return normalized;
}

double ypd_angle_error(const Eigen::Vector3d & observed, const Eigen::Vector3d & predicted)
{
    if (!observed.array().isFinite().all() || !predicted.array().isFinite().all()) {
        return std::numeric_limits<double>::max();
    }

    const double yaw_error = tools::limit_rad(observed[0] - predicted[0]);
    const double pitch_error = observed[1] - predicted[1];
    return std::hypot(yaw_error, pitch_error);
}

struct PhaseAssociation
{
    int index = 0;
    double offset = 0.0;
    double global_roll = 0.0;
};

PhaseAssociation associate_blade_phase(double observed_roll, double reference_global_roll)
{
    constexpr double step = CV_2PI / 5.0;

    PhaseAssociation best;
    double best_error = std::numeric_limits<double>::max();
    for (int k = 0; k < 5; ++k) {
        const double offset = static_cast<double>(k) * step;
        const double global_roll =
            normalize_full_turn_measurement(observed_roll - offset, reference_global_roll);
        const double error = std::abs(global_roll - reference_global_roll);
        if (error < best_error) {
            best_error = error;
            best.index = k;
            best.offset = offset;
            best.global_roll = global_roll;
        }
    }
    return best;
}

struct BladeObservation
{
    int blade_index = 0;
    bool selected = false;
    PhaseAssociation phase;
    Eigen::Vector3d rune_ypd = Eigen::Vector3d::Zero();
    Eigen::Vector3d blade_ypd = Eigen::Vector3d::Zero();
};

struct GlobalRollObservation
{
    double roll = 0.0;
    int used_count = 0;
};

std::vector<BladeObservation> build_blade_observations(
    const auto_buff::PowerRune& rune,
    double reference_global_roll)
{
    std::vector<BladeObservation> observations;
    observations.reserve(std::max<size_t>(1, rune.fanblades.size()));

    for (size_t i = 0; i < rune.fanblades.size(); ++i) {
        const auto& blade = rune.fanblades[i];
        if (!blade.solved) {
            continue;
        }
        BladeObservation observation;
        observation.blade_index = static_cast<int>(i);
        observation.selected = (i == 0);
        observation.phase = associate_blade_phase(blade.ypr_in_world[2], reference_global_roll);
        observation.rune_ypd = blade.rune_ypd_in_world;
        observation.blade_ypd = blade.blade_ypd_in_world;
        observations.push_back(observation);
    }

    if (observations.empty() && !rune.is_unsolve()) {
        BladeObservation observation;
        observation.blade_index = 0;
        observation.selected = true;
        observation.phase = associate_blade_phase(rune.ypr_in_world[2], reference_global_roll);
        observation.rune_ypd = rune.ypd_in_world;
        observation.blade_ypd = rune.blade_ypd_in_world;
        observations.push_back(observation);
    }

    return observations;
}

std::optional<GlobalRollObservation> robust_global_roll_observation(
    const std::vector<BladeObservation>& observations,
    double reference_global_roll)
{
    if (observations.empty() || !std::isfinite(reference_global_roll)) {
        return std::nullopt;
    }

    const auto selected_it =
        std::find_if(observations.begin(), observations.end(), [](const auto& observation) {
            return observation.selected && std::isfinite(observation.phase.global_roll);
        });
    if (selected_it != observations.end()) {
        return GlobalRollObservation{selected_it->phase.global_roll, 1};
    }

    size_t best_index = 0;
    double best_abs_residual = std::numeric_limits<double>::max();
    bool has_finite_residual = false;
    for (size_t i = 0; i < observations.size(); ++i) {
        const double residual = observations[i].phase.global_roll - reference_global_roll;
        const double abs_residual = std::abs(residual);
        if (std::isfinite(abs_residual) && abs_residual < best_abs_residual) {
            best_abs_residual = abs_residual;
            best_index = i;
            has_finite_residual = true;
        }
    }
    if (!has_finite_residual) {
        return std::nullopt;
    }

    const double best_roll = observations[best_index].phase.global_roll;
    if (observations.size() == 1) {
        return GlobalRollObservation{best_roll, 1};
    }

    constexpr double step = CV_2PI / 5.0;
    constexpr double kMaxResidualGate = step * 0.45;
    constexpr double kMinResidualGate = step * 0.16;
    constexpr double kRelativeGate = step * 0.28;
    constexpr double kClusterGate = step * 0.35;
    constexpr double kWeightScale = step * 0.20;

    const double adaptive_gate =
        std::min(kMaxResidualGate, std::max(kMinResidualGate, best_abs_residual + kRelativeGate));
    double weighted_residual_sum = 0.0;
    double weight_sum = 0.0;
    int used_count = 0;

    for (const auto& observation : observations) {
        const double roll = observation.phase.global_roll;
        const double residual = roll - reference_global_roll;
        if (!std::isfinite(roll) || !std::isfinite(residual)) {
            continue;
        }

        const double abs_residual = std::abs(residual);
        const double cluster_delta = std::abs(roll - best_roll);
        if (abs_residual > adaptive_gate || cluster_delta > kClusterGate) {
            continue;
        }

        const double normalized = abs_residual / kWeightScale;
        const double weight = 1.0 / (1.0 + normalized * normalized);
        weighted_residual_sum += weight * residual;
        weight_sum += weight;
        used_count++;
    }

    if (weight_sum <= 1e-9 || used_count == 0) {
        return GlobalRollObservation{best_roll, 1};
    }

    return GlobalRollObservation{
        reference_global_roll + weighted_residual_sum / weight_sum,
        used_count};
}
}

namespace auto_buff
{

// ================= Voter Implementation =================
void Voter::vote(double angle_last, double angle_now) {
    if (std::abs(clockwise_) > 50) return;
    double diff = angle_now - angle_last;
    if (diff > CV_PI) diff -= CV_2PI;
    if (diff < -CV_PI) diff += CV_2PI;
    if (diff > 0.001) clockwise_++;
    else if (diff < -0.001) clockwise_--;
}

int Voter::clockwise() const {
    if (clockwise_ > 5) return 1;
    if (clockwise_ < -5) return -1;
    return 0;
}

void Voter::seed_direction(int direction) {
    if (direction > 0) {
        clockwise_ = 6;
    } else if (direction < 0) {
        clockwise_ = -6;
    } else {
        clockwise_ = 0;
    }
}

// ================= BuffTracker Implementation =================

BuffTracker::BuffTracker() {}

BuffTracker::BuffTracker(const std::string& config_path)
{
    const auto yaml = YAML::LoadFile(config_path);
    const auto tracker_yaml = yaml["buff_tracker"];
    read_optional_scalar(tracker_yaml, "lost_timeout_s", lost_timeout_s_);
    read_optional_scalar(tracker_yaml, "big_lost_timeout_s", big_lost_timeout_s_);
    read_optional_scalar(tracker_yaml, "big_model_reset_timeout_s", big_model_reset_timeout_s_);
    read_optional_scalar(tracker_yaml, "big_curve_ekf_fit_enabled", big_curve_ekf_fit_enabled_);
    read_optional_scalar(tracker_yaml, "big_phase_process_noise", big_phase_process_noise_);
    read_optional_scalar(tracker_yaml, "big_a_process_noise", big_a_process_noise_);
    read_optional_scalar(tracker_yaml, "big_w_process_noise", big_w_process_noise_);
    read_optional_scalar(tracker_yaml, "big_measurement_noise_scale", big_measurement_noise_scale_);
    read_optional_scalar(tracker_yaml, "big_speed_measurement_enabled", big_speed_measurement_enabled_);
    read_optional_scalar(tracker_yaml, "big_speed_measurement_noise", big_speed_measurement_noise_);
    read_optional_scalar(tracker_yaml, "big_speed_measurement_gate", big_speed_measurement_gate_);
    read_optional_scalar(
        tracker_yaml, "big_speed_measurement_adaptive_scale",
        big_speed_measurement_adaptive_scale_);
    read_optional_scalar(
        tracker_yaml, "big_speed_measurement_correction_limit",
        big_speed_measurement_correction_limit_);
    read_optional_scalar(tracker_yaml, "big_curve_speed_slew_limit", big_curve_speed_slew_limit_);
    read_optional_scalar(
        tracker_yaml, "big_speed_measurement_window_samples",
        big_speed_measurement_window_samples_);
    read_optional_scalar(
        tracker_yaml, "big_speed_measurement_window_s", big_speed_measurement_window_s_);
    read_optional_scalar(
        tracker_yaml, "big_speed_measurement_min_history",
        big_speed_measurement_min_history_);
    read_optional_scalar(
        tracker_yaml, "big_curve_phi_correction_limit",
        big_curve_phi_correction_limit_);
    read_optional_scalar(tracker_yaml, "big_phi_seed_frames", big_phi_seed_frames_);

    read_env_scalar("BUFF_BIG_PHASE_PROCESS_NOISE", big_phase_process_noise_);
    read_env_scalar("BUFF_BIG_A_PROCESS_NOISE", big_a_process_noise_);
    read_env_scalar("BUFF_BIG_W_PROCESS_NOISE", big_w_process_noise_);
    read_env_scalar("BUFF_BIG_SPEED_MEASUREMENT_ENABLED", big_speed_measurement_enabled_);
    read_env_scalar("BUFF_BIG_SPEED_MEASUREMENT_NOISE", big_speed_measurement_noise_);
    read_env_scalar(
        "BUFF_BIG_SPEED_MEASUREMENT_ADAPTIVE_SCALE",
        big_speed_measurement_adaptive_scale_);
    read_env_scalar(
        "BUFF_BIG_SPEED_MEASUREMENT_CORRECTION_LIMIT",
        big_speed_measurement_correction_limit_);
    read_env_scalar("BUFF_BIG_CURVE_SPEED_SLEW_LIMIT", big_curve_speed_slew_limit_);
    read_env_scalar("BUFF_BIG_SPEED_MEASUREMENT_WINDOW_S", big_speed_measurement_window_s_);
    read_env_scalar(
        "BUFF_BIG_SPEED_MEASUREMENT_WINDOW_SAMPLES",
        big_speed_measurement_window_samples_);
    read_env_scalar("BUFF_BIG_SPEED_MEASUREMENT_MIN_HISTORY", big_speed_measurement_min_history_);
    read_env_scalar("BUFF_BIG_CURVE_PHI_CORRECTION_LIMIT", big_curve_phi_correction_limit_);
    read_env_scalar("BUFF_BIG_PHI_SEED_FRAMES", big_phi_seed_frames_);

    if (!std::isfinite(lost_timeout_s_) || lost_timeout_s_ < 0.0) {
        lost_timeout_s_ = 0.35;
    }
    if (!std::isfinite(big_lost_timeout_s_) || big_lost_timeout_s_ < 0.0) {
        big_lost_timeout_s_ = 0.08;
    }
    if (big_phi_seed_frames_ < 0) {
        big_phi_seed_frames_ = 5;
    }
    if (!std::isfinite(big_model_reset_timeout_s_) || big_model_reset_timeout_s_ < big_lost_timeout_s_) {
        big_model_reset_timeout_s_ = std::max(lost_timeout_s_, big_lost_timeout_s_);
    }
    if (!std::isfinite(big_phase_process_noise_) || big_phase_process_noise_ < 0.0) {
        big_phase_process_noise_ = 0.02;
    }
    if (!std::isfinite(big_a_process_noise_) || big_a_process_noise_ < 0.0) {
        big_a_process_noise_ = 0.0001;
    }
    if (!std::isfinite(big_w_process_noise_) || big_w_process_noise_ < 0.0) {
        big_w_process_noise_ = 0.00001;
    }
    if (!std::isfinite(big_measurement_noise_scale_) || big_measurement_noise_scale_ < 1.0) {
        big_measurement_noise_scale_ = 1.0;
    }
    if (!std::isfinite(big_speed_measurement_noise_) || big_speed_measurement_noise_ <= 0.0) {
        big_speed_measurement_noise_ = 0.20;
    }
    if (!std::isfinite(big_speed_measurement_gate_) || big_speed_measurement_gate_ <= 0.0) {
        big_speed_measurement_gate_ = 1.20;
    }
    if (!std::isfinite(big_speed_measurement_adaptive_scale_) ||
        big_speed_measurement_adaptive_scale_ < 0.0) {
        big_speed_measurement_adaptive_scale_ = 0.0;
    }
    if (!std::isfinite(big_speed_measurement_correction_limit_) ||
        big_speed_measurement_correction_limit_ < 0.0) {
        big_speed_measurement_correction_limit_ = 0.0;
    }
    if (!std::isfinite(big_curve_speed_slew_limit_) || big_curve_speed_slew_limit_ < 0.0) {
        big_curve_speed_slew_limit_ = 0.0;
    }
    if (big_speed_measurement_window_samples_ < 2) {
        big_speed_measurement_window_samples_ = 12;
    }
    big_speed_measurement_window_samples_ =
        std::min(big_speed_measurement_window_samples_, 16);
    if (!std::isfinite(big_speed_measurement_window_s_) ||
        big_speed_measurement_window_s_ <= 0.0) {
        big_speed_measurement_window_s_ = 0.20;
    }
    if (big_speed_measurement_min_history_ < 0) {
        big_speed_measurement_min_history_ = 0;
    }
    if (!std::isfinite(big_curve_phi_correction_limit_) ||
        big_curve_phi_correction_limit_ < 0.0) {
        big_curve_phi_correction_limit_ = 0.0;
    }

    tools::logger()->info(
        "Buff tracker config -> lost_timeout_s={:.3f}, big_lost_timeout_s={:.3f}, big_model_reset_timeout_s={:.3f}, big_ekf_fit={}, q(phase={:.4f}, a={:.5f}, w={:.6f}), big_measurement_noise_scale={:.2f}, big_speed_measurement(enabled={}, noise={:.4f}, gate={:.2f}, adaptive_scale={:.2f}, correction_limit={:.4f}, window_samples={}, window_s={:.3f}, min_history={}), speed_slew_limit={:.3f}, phi_correction_limit={:.4f}",
        lost_timeout_s_, big_lost_timeout_s_, big_model_reset_timeout_s_,
        big_curve_ekf_fit_enabled_,
        big_phase_process_noise_,
        big_a_process_noise_,
        big_w_process_noise_,
        big_measurement_noise_scale_,
        big_speed_measurement_enabled_,
        big_speed_measurement_noise_,
        big_speed_measurement_gate_,
        big_speed_measurement_adaptive_scale_,
        big_speed_measurement_correction_limit_,
        big_speed_measurement_window_samples_,
        big_speed_measurement_window_s_,
        big_speed_measurement_min_history_,
        big_curve_speed_slew_limit_,
        big_curve_phi_correction_limit_);
}

void BuffTracker::reset()
{
    ekf_ = tools::ExtendedKalmanFilter{};
    voter_.reset();
    history_info_.clear();
    last_time_ = std::chrono::steady_clock::time_point{};
    start_time_ = std::chrono::steady_clock::time_point{};
    last_angle_ = 0.0;
    last_observed_speed_raw_ = std::numeric_limits<double>::quiet_NaN();
    last_observed_speed_ = std::numeric_limits<double>::quiet_NaN();
    smoothed_curve_speed_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_speed_after_predict_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_speed_after_blade_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_speed_before_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_speed_after_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_phi_after_predict_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_phi_after_blade_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_phi_before_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_phi_after_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_speed_measurement_predicted_ = std::numeric_limits<double>::quiet_NaN();
    debug_speed_measurement_innovation_ = std::numeric_limits<double>::quiet_NaN();
    debug_speed_measurement_noise_ = std::numeric_limits<double>::quiet_NaN();
    debug_speed_measurement_status_ = 0;
    selected_target_roll_offset_ = 0.0;
    big_curve_prediction_enabled_ = false;
    lost_since_.reset();
    last_rune_type_.reset();
    last_switch_deferred_ = false;
    last_target_switched_ = false;
    phase_origin_index_ = -1;
    selected_phase_index_ = -1;
    last_reinit_reason_ = 0;
    is_initialized_ = false;
    is_lost_ = true;
}

void BuffTracker::update(
    const std::optional<PowerRune>& rune,
    std::chrono::steady_clock::time_point timestamp,
    bool switch_deferred)
{
    last_switch_deferred_ = switch_deferred;
    last_target_switched_ = false;
    last_reinit_reason_ = 0;

    if (!rune.has_value()) {
        if (!lost_since_.has_value()) {
            lost_since_ = timestamp;
        }
        if (switch_deferred) {
            is_lost_ = true;
        }

        const double lost_elapsed_s =
            std::max(0.0, tools::delta_time(timestamp, *lost_since_));
        const double timeout_s =
            last_rune_type_.has_value() && *last_rune_type_ == BIG
                ? big_lost_timeout_s_
                : lost_timeout_s_;
        if (lost_elapsed_s >= timeout_s) {
            is_lost_ = true;
            if (!last_rune_type_.has_value() || *last_rune_type_ != BIG ||
                lost_elapsed_s >= big_model_reset_timeout_s_) {
                is_initialized_ = false;
                voter_.reset();
                history_info_.clear();
                big_curve_prediction_enabled_ = false;
                lost_since_.reset();
                last_rune_type_.reset();
                phase_origin_index_ = -1;
            }
        }
        return;
    }

    last_rune_type_ = rune->type;

    if (!is_initialized_) {
        start_time_ = timestamp;
        init_ekf(rune.value());
        is_initialized_ = true;
        last_time_ = timestamp;
        is_lost_ = false;
        lost_since_.reset();
        return;
    }

    const bool reacquired_after_lost = lost_since_.has_value();
    const double curve_pause_s =
        reacquired_after_lost && lost_since_.has_value()
        ? std::max(0.0, tools::delta_time(timestamp, *lost_since_))
        : 0.0;
    double dt = tools::delta_time(timestamp, last_time_);
    dt = std::clamp(dt, 1e-3, 0.1);
    last_time_ = timestamp;
    is_lost_ = false;
    lost_since_.reset();

    const bool target_switch_segment =
        rune->type == BIG && should_segment_for_target_switch(rune.value());
    const bool reacquire_reinit =
        rune->type == BIG && reacquired_after_lost &&
        should_reinitialize_big_reacquire(rune.value());
    if (target_switch_segment || reacquire_reinit) {
        last_target_switched_ = target_switch_segment || rune->target_switched;
        reinitialize_big_reacquire(
            rune.value(), timestamp, target_switch_segment ? 2 : 1, curve_pause_s);
        return;
    }

    if (rune->type == BIG) {
        double relative_t = tools::delta_time(timestamp, start_time_);
        double reference_angle = ekf_.x.size() > 5 && std::isfinite(ekf_.x[5])
            ? ekf_.x[5]
            : (history_info_.empty() ? last_angle_ : history_info_.back().angle);
        const int dir = voter_.clockwise();
        if (dir != 0) {
            if (can_use_big_buff_curve_model(ekf_.x)) {
                const double angle_delta = big_curve_ekf_fit_enabled_
                    ? big_runev2_angle_delta(ekf_.x[7], ekf_.x[8], ekf_.x[9], dt)
                    : big_buff_angle_delta(
                        ekf_.x[7], ekf_.x[8], ekf_.x[9],
                        std::max(0.0, relative_t - dt), relative_t);
                reference_angle += static_cast<double>(dir) * angle_delta;
            } else {
                reference_angle += static_cast<double>(dir) * std::abs(ekf_.x[6]) * dt;
            }
        }
        const auto observations = build_blade_observations(rune.value(), reference_angle);
        const auto global_roll_observation =
            robust_global_roll_observation(observations, reference_angle);
        const double cur_angle = global_roll_observation.has_value()
            ? global_roll_observation->roll
            : normalize_roll_measurement(rune->ypr_in_world[2], reference_angle);
        history_info_.push_back({relative_t, cur_angle});
        if (history_info_.size() > 250) history_info_.pop_front();
    }

    update_ekf(rune.value(), dt);
}

void BuffTracker::init_ekf(const PowerRune& p)
{
    const bool has_primary_blade = !p.fanblades.empty() && p.fanblades.front().solved;
    const Eigen::Vector3d rune_ypd =
        has_primary_blade ? p.fanblades.front().rune_ypd_in_world : p.ypd_in_world;
    const Eigen::Vector3d primary_ypr =
        has_primary_blade ? p.fanblades.front().ypr_in_world : p.ypr_in_world;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(10);
    x0 << rune_ypd[0], 0.0, rune_ypd[1], rune_ypd[2],
          primary_ypr[0], primary_ypr[2],
          (p.type == SMALL ? 0.436 : 1.1775),
          (p.type == SMALL ? 0.0 : kBigBuffAInit),
          (p.type == SMALL ? 0.0 : kBigBuffWInit),
          0.0;

    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(10, 10);
    P0.diagonal() << 1, 1, 1, 1, 1, 10, 10, 0.1, 0.1, 1;

    auto x_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        Eigen::VectorXd c = a + b;
        c[0] = tools::limit_rad(c[0]); c[2] = tools::limit_rad(c[2]); c[4] = tools::limit_rad(c[4]);
        return c;
    };
    ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
    last_angle_ = primary_ypr[2];
    selected_target_roll_offset_ = 0.0;
    selected_phase_index_ = -1;
    big_curve_prediction_enabled_ = (p.type == BIG);
}

void BuffTracker::maybe_lock_phase_origin(int raw_phase_index)
{
    if (phase_origin_index_ >= 0 || raw_phase_index < 0) {
        return;
    }
    phase_origin_index_ = normalize_phase_index(raw_phase_index);
}

int BuffTracker::logical_phase_index(int raw_phase_index) const
{
    if (raw_phase_index < 0 || phase_origin_index_ < 0) {
        return -1;
    }
    return normalize_phase_index(raw_phase_index - phase_origin_index_);
}

std::optional<double> BuffTracker::estimate_big_observed_speed() const
{
    return estimate_linear_rate_from_history(
        history_info_.begin(),
        history_info_.end(),
        static_cast<std::size_t>(big_speed_measurement_window_samples_),
        big_speed_measurement_window_s_);
}

BuffTracker::ObservedSpeedMeasurement BuffTracker::buildObservedSpeedMeasurement(
    PowerRune_type rune_type,
    double observed_roll_delta,
    double dt) const
{
    ObservedSpeedMeasurement measurement;
    measurement.raw_unbounded = std::abs(observed_roll_delta) / std::max(dt, 1e-3);
    measurement.raw_clamped = rune_type == BIG
        ? clamp_big_buff_speed_magnitude(measurement.raw_unbounded)
        : measurement.raw_unbounded;
    measurement.measurement_unbounded = measurement.raw_unbounded;
    measurement.measurement = measurement.raw_clamped;

    if (rune_type == BIG) {
        const auto estimated_speed = estimate_big_observed_speed();
        if (estimated_speed.has_value() && std::isfinite(*estimated_speed)) {
            measurement.measurement_unbounded = std::abs(*estimated_speed);
            measurement.measurement =
                clamp_big_buff_speed_magnitude(measurement.measurement_unbounded);
        }
    }

    measurement.valid =
        rune_type == BIG && std::isfinite(measurement.measurement_unbounded) &&
        measurement.measurement_unbounded < 3.5 && std::abs(observed_roll_delta) < 0.35;
    return measurement;
}

bool BuffTracker::can_use_big_buff_curve_model(const Eigen::VectorXd& x) const
{
    return big_curve_prediction_enabled_ && x.size() >= 10 && has_big_buff_curve_model(x[7], x[8]);
}

bool BuffTracker::should_reinitialize_big_reacquire(const PowerRune& p) const
{
    if (!is_initialized_ || ekf_.x.size() < 10) {
        return true;
    }

    const auto observations = build_blade_observations(p, ekf_.x[5]);
    if (observations.empty()) {
        return true;
    }

    const auto selected_it = std::find_if(observations.begin(), observations.end(), [](const auto& obs) {
        return obs.selected;
    });
    const BladeObservation& observation =
        selected_it == observations.end() ? observations.front() : *selected_it;

    const Eigen::VectorXd predicted_measurement = measurement_model(ekf_.x, observation.phase.offset);
    if (predicted_measurement.size() < 7 || !predicted_measurement.array().isFinite().all()) {
        return true;
    }

    const Eigen::Vector3d predicted_blade_ypd = predicted_measurement.tail(3);
    const double blade_angle_error = ypd_angle_error(observation.blade_ypd, predicted_blade_ypd);
    const double blade_dist_error = std::abs(observation.blade_ypd[2] - predicted_blade_ypd[2]);

    // 大符激活时目标扇叶会熄灭并切到另一块随机点亮装甲。临时丢失后如果
    // 观测已经明显不在当前 EKF 扇叶上，继续用旧状态会把新目标当离群点拒掉。
    return blade_angle_error > 0.12 || blade_dist_error > 0.45;
}

bool BuffTracker::should_segment_for_target_switch(const PowerRune& p) const
{
    if (p.type != BIG || !is_initialized_ || ekf_.x.size() < 10) {
        return false;
    }
    if (p.target_switched) {
        return true;
    }

    const auto observations = build_blade_observations(p, ekf_.x[5]);
    const auto selected_it = std::find_if(observations.begin(), observations.end(), [](const auto& obs) {
        return obs.selected;
    });
    if (selected_it == observations.end()) {
        return false;
    }

    constexpr double step = CV_2PI / 5.0;
    return phase_offset_distance(selected_it->phase.offset, selected_target_roll_offset_) > step * 0.45;
}

void BuffTracker::reinitialize_big_reacquire(
    const PowerRune& p,
    std::chrono::steady_clock::time_point timestamp,
    int reason,
    double curve_pause_s)
{
    const Eigen::VectorXd previous_state = ekf_.x;
    const auto previous_start_time = start_time_;
    const int previous_direction = voter_.clockwise();
    const bool previous_curve_available =
        previous_state.size() >= 10 && has_big_buff_curve_model(previous_state[7], previous_state[8]);

    tools::logger()->debug(
        "[BuffTracker] Reinitialize big buff, reason={}, blade_ypd=({:.3f}, {:.3f}, {:.3f})",
        reason, p.blade_ypd_in_world[0], p.blade_ypd_in_world[1], p.blade_ypd_in_world[2]);

    last_time_ = timestamp;
    init_ekf(p);
    smoothed_curve_speed_ = std::numeric_limits<double>::quiet_NaN();
    last_reinit_reason_ = reason;
    const bool can_inherit_motion =
        previous_state.size() >= 10 && previous_state.array().isFinite().all();
    if (can_inherit_motion) {
        const auto observations = build_blade_observations(p, previous_state[5]);
        const auto selected_it =
            std::find_if(observations.begin(), observations.end(), [](const auto& obs) {
                return obs.selected;
            });
        const BladeObservation* selected_observation =
            selected_it == observations.end() ? nullptr : &(*selected_it);

        ekf_.x[1] = previous_state[1];
        if (selected_observation != nullptr) {
            ekf_.x[5] = selected_observation->phase.global_roll;
            selected_target_roll_offset_ = selected_observation->phase.offset;
            maybe_lock_phase_origin(selected_observation->phase.index);
            selected_phase_index_ = logical_phase_index(selected_observation->phase.index);
        } else {
            ekf_.x[5] = normalize_full_turn_measurement(p.ypr_in_world[2], previous_state[5]);
            selected_target_roll_offset_ = 0.0;
            selected_phase_index_ = -1;
        }
        ekf_.x[6] = previous_state[6];
        ekf_.x[7] = previous_state[7];
        ekf_.x[8] = previous_state[8];
        ekf_.x[9] = previous_state[9];
        last_angle_ = ekf_.x[5];
        start_time_ = previous_start_time;
        if (curve_pause_s > 0.0) {
            start_time_ += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(curve_pause_s));
        }
        if (previous_curve_available) {
            const double rebased_t = std::max(0.0, tools::delta_time(timestamp, start_time_));
            ekf_.x[6] = big_curve_ekf_fit_enabled_
                ? std::abs(big_runev2_speed(ekf_.x[7], ekf_.x[9]))
                : std::abs(big_buff_speed(ekf_.x[7], ekf_.x[8], ekf_.x[9], rebased_t));
        }
    } else {
        start_time_ = timestamp;
    }
    voter_.reset();
    if (previous_direction != 0) {
        voter_.seed_direction(previous_direction);
    }
    history_info_.clear();
    big_curve_prediction_enabled_ =
        p.type == BIG && has_big_buff_curve_model(ekf_.x[7], ekf_.x[8]);
    last_observed_speed_ = std::numeric_limits<double>::quiet_NaN();
    is_initialized_ = true;
    is_lost_ = false;
    lost_since_.reset();
}

void BuffTracker::update_ekf(const PowerRune& p, double dt)
{
    const double current_roll = ekf_.x[5];
    const auto observations = build_blade_observations(p, current_roll);
    if (observations.empty()) {
        return;
    }

    // Capture per-blade debug data keyed by phase slot (0-4).
    last_blade_debug_ = {};
    const bool is_fallback = observations.size() == 1 && !p.is_unsolve() &&
                             (p.fanblades.empty() || !p.fanblades[0].solved);
    for (const auto& obs : observations) {
        const int slot = ((obs.phase.index % 5) + 5) % 5;
        auto& info = last_blade_debug_[slot];
        info.present = true;
        info.solved = !is_fallback;
        info.assoc_global_roll_rad = obs.phase.global_roll;
        info.selected = obs.selected;
    }

    const auto selected_it =
        std::find_if(observations.begin(), observations.end(), [](const auto& obs) {
            return obs.selected;
        });
    const BladeObservation* selected_observation =
        selected_it == observations.end() ? nullptr : &(*selected_it);
    const BladeObservation& primary_observation =
        selected_observation == nullptr ? observations.front() : *selected_observation;
    if (selected_observation != nullptr) {
        selected_target_roll_offset_ = selected_observation->phase.offset;
        selected_phase_index_ = logical_phase_index(selected_observation->phase.index);
    }

    double measured_roll = primary_observation.phase.global_roll;
    const double last_measured_roll = last_angle_;
    const double observed_roll_delta = measured_roll - last_measured_roll;
    const auto observed_speed_measurement =
        buildObservedSpeedMeasurement(p.type, observed_roll_delta, dt);
    const double observed_speed = observed_speed_measurement.measurement;
    last_observed_speed_raw_ = observed_speed_measurement.raw_clamped;
    last_observed_speed_ = observed_speed;
    debug_curve_speed_after_predict_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_speed_after_blade_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_speed_before_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_speed_after_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_phi_after_predict_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_phi_after_blade_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_phi_before_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_curve_phi_after_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    debug_speed_measurement_predicted_ = std::numeric_limits<double>::quiet_NaN();
    debug_speed_measurement_innovation_ = std::numeric_limits<double>::quiet_NaN();
    debug_speed_measurement_noise_ = big_speed_measurement_noise_;
    debug_speed_measurement_status_ = 0;
    const bool observed_speed_valid = observed_speed_measurement.valid;

    voter_.vote(last_angle_, measured_roll);
    last_angle_ = measured_roll;
    int dir = voter_.clockwise();

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(10, 10);
    F(0, 1) = dt;
    double active_dir = (dir == 0) ? (measured_roll > current_roll ? 1 : -1) : dir;
    const bool use_original_big_model =
        p.type == BIG && big_curve_ekf_fit_enabled_ && can_use_big_buff_curve_model(ekf_.x);
    F(5, 6) = use_original_big_model ? 0.0 : active_dir * dt;
    if (use_original_big_model) {
        const double a = clamp_or_default(ekf_.x[7], kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
        const double phase = std::isfinite(ekf_.x[9]) ? ekf_.x[9] : 0.0;
        const double sin_phase = std::sin(phase);
        const double cos_phase = std::cos(phase);
        F(5, 7) = active_dir * dt * (sin_phase - 1.0);
        F(5, 9) = active_dir * dt * a * cos_phase;
        F(6, 7) = sin_phase - 1.0;
        F(6, 9) = a * cos_phase;
        F(9, 8) = dt;
    }

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(10, 10) * 1e-6;
    Q(6, 6) = (p.type == BIG) ? 2e-2 : 1e-9;
    if (use_original_big_model) {
        Q(6, 6) = 1e-3;
        Q(7, 7) = big_a_process_noise_;
        Q(8, 8) = big_w_process_noise_;
        Q(9, 9) = big_phase_process_noise_;
    }

    // Phi one-shot seed: wait until history_info_ has big_phi_seed_frames_ samples so
    // the rolling-window speed estimate is stable, then invert the speed formula to
    // place x[9] near the correct phase in one step instead of drifting over ~60 frames.
    // Only fires EXACTLY once per tracker lifetime (when history_size == seed window).
    // Skipped if observed_speed is above the model ceiling (noisy spike) or out of range.
    if (use_original_big_model &&
        big_phi_seed_frames_ > 0 &&
        static_cast<int>(history_info_.size()) == big_phi_seed_frames_ &&
        observed_speed_valid &&
        observed_speed > 0.1 &&
        observed_speed < kBigBuffBaseSpeed * 0.98 &&
        ekf_.x.size() >= 10) {
        const double a_s = clamp_or_default(ekf_.x[7], kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
        const double base_s = kBigBuffBaseSpeed - a_s;
        const double sin_arg = (observed_speed - base_s) / a_s;
        if (std::isfinite(sin_arg) && std::abs(sin_arg) <= 1.0) {
            // Ascending branch: phi ∈ [-π/2, π/2]  → speed still increasing
            // Descending branch: phi = limit_rad(π - asin) → speed decreasing
            const double phi_asc  = std::asin(sin_arg);
            const double phi_desc = tools::limit_rad(M_PI - phi_asc);
            double phi_seed = phi_asc;  // default: ascending
            // Use the last few history samples to determine speed trend
            if (history_info_.size() >= 4) {
                const auto it0 = history_info_.rbegin();
                const auto it1 = std::next(it0);
                const auto it3 = std::next(std::next(it1));
                const double dt_recent = it0->t - it1->t;
                const double dt_early  = it1->t - it3->t;
                if (dt_recent > 1e-4 && dt_early > 1e-4) {
                    const double spd_recent = std::abs(it0->angle - it1->angle) / dt_recent;
                    const double spd_early  = std::abs(it1->angle - it3->angle) / dt_early;
                    if (spd_recent < spd_early - 0.05) {
                        phi_seed = phi_desc;  // speed trend is decreasing → descending branch
                    }
                }
            }
            ekf_.x[9] = phi_seed;
            // Tighten phi covariance to moderate confidence so EKF can still correct if needed
            if (ekf_.P.rows() > 9 && ekf_.P(9, 9) > 0.05) {
                ekf_.P(9, 9) = 0.05;
            }
            tools::logger()->debug(
                "[Tracker] Phi seed applied: obs_speed={:.3f} sin_arg={:.3f} phi={:.3f}rad ({:.1f}deg)",
                observed_speed, sin_arg, phi_seed, phi_seed * 180.0 / M_PI);
        }
    }

    auto f = [=](const Eigen::VectorXd& x) {
        Eigen::VectorXd x_next = x;
        x_next[0] = tools::limit_rad(x[0] + x[1] * dt);
        if (p.type == BIG && big_curve_ekf_fit_enabled_ && can_use_big_buff_curve_model(x)) {
            const double a = clamp_or_default(x[7], kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
            const double w = clamp_or_default(x[8], kBigBuffWMin, kBigBuffWMax, kBigBuffWInit);
            const double phase = std::isfinite(x[9]) ? x[9] : 0.0;
            const double speed = big_runev2_speed(a, phase);
            x_next[5] = x[5] + active_dir * speed * dt;
            x_next[6] = std::abs(speed);
            x_next[7] = a;
            x_next[8] = w;
            x_next[9] = tools::limit_rad(phase + w * dt);
        } else if (p.type == BIG && dir != 0 && can_use_big_buff_curve_model(x)) {
            const double t_end = tools::delta_time(last_time_, start_time_);
            const double t_start = std::max(0.0, t_end - dt);
            const double curve_speed_now = std::abs(big_buff_speed(x[7], x[8], x[9], t_start));
            const double curve_speed_end = std::abs(big_buff_speed(x[7], x[8], x[9], t_end));
            const double curve_delta = big_buff_angle_delta(x[7], x[8], x[9], t_start, t_end);
            const auto blend = compute_curve_blend_state(
                curve_speed_now,
                x[6],
                observed_speed,
                observed_speed_valid,
                static_cast<int>(history_info_.size()),
                can_use_big_buff_curve_model(x));
            const double fallback_delta = blend.fallback_speed * dt;
            x_next[6] = blend.alpha * curve_speed_end + (1.0 - blend.alpha) * blend.fallback_speed;
            x_next[5] = x[5] + active_dir * (
                blend.alpha * curve_delta + (1.0 - blend.alpha) * fallback_delta);
        } else {
            x_next[5] = x[5] + active_dir * x[6] * dt;
        }
        return x_next;
    };
    ekf_.predict(F, Q, f);
    if (use_original_big_model) {
        debug_curve_speed_after_predict_ = big_runev2_curve_speed_from_state(ekf_.x);
        debug_curve_phi_after_predict_ = big_runev2_curve_phi_from_state(ekf_.x);
    }
    const double phase_before_measurement =
        ekf_.x.size() > 9 && std::isfinite(ekf_.x[9]) ? ekf_.x[9] : std::numeric_limits<double>::quiet_NaN();
    if (observed_speed_valid && !big_curve_ekf_fit_enabled_) {
        const double current_curve_speed = can_use_big_buff_curve_model(ekf_.x)
            ? std::abs(big_buff_speed(
                ekf_.x[7],
                ekf_.x[8],
                ekf_.x[9],
                std::max(0.0, tools::delta_time(last_time_, start_time_))))
            : std::abs(ekf_.x[6]);
        const auto blend = compute_curve_blend_state(
            current_curve_speed,
            ekf_.x[6],
            observed_speed,
            observed_speed_valid,
            static_cast<int>(history_info_.size()),
            can_use_big_buff_curve_model(ekf_.x));
        ekf_.x[6] = blend.alpha * std::abs(ekf_.x[6]) + (1.0 - blend.alpha) * observed_speed;
    }

    for (const auto& observation : observations) {
        if (p.type == BIG && !observation.selected) {
            continue;
        }
        Eigen::VectorXd z(7);
        z << observation.rune_ypd[0], observation.rune_ypd[1], observation.rune_ypd[2],
             observation.phase.global_roll,
             observation.blade_ypd[0], observation.blade_ypd[1], observation.blade_ypd[2];

        Eigen::MatrixXd H = h_jacobian(ekf_.x, observation.phase.offset);
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(7, 7);
        R.diagonal() << 1e-2, 1e-2, 1.0, (p.type == BIG ? 0.02 : 0.5), 1e-2, 1e-2, 1.0;
        if (p.type == BIG) {
            R *= big_measurement_noise_scale_;
        }
        const Eigen::VectorXd z_pred = measurement_model(ekf_.x, observation.phase.offset);
        const Eigen::VectorXd innovation = measurement_difference(z, z_pred);

        const double dist_gate = std::max(0.35, 0.08 * std::max(z[2], z[6]));
        const double roll_gate = p.type == BIG
            ? std::max(0.80, std::abs(ekf_.x[6]) * dt * 4.0 + 0.30)
            : std::max(0.28, std::abs(ekf_.x[6]) * dt * 2.5 + 0.15);
        const double blade_angle_gate = p.type == BIG ? 0.20 : 0.10;
        const std::array<double, 7> gates = {
            0.08, 0.08, dist_gate, roll_gate, blade_angle_gate, blade_angle_gate, dist_gate};

        int gross_outlier_count = 0;
        for (int i = 0; i < innovation.size(); ++i) {
            const double gate = std::max(1e-6, gates[i]);
            const double ratio = std::abs(innovation[i]) / gate;
            if (ratio > 1.0) {
                R(i, i) *= std::min(400.0, 25.0 * ratio * ratio);
            }
            if (ratio > 2.5) {
                gross_outlier_count++;
            }
        }

        if (gross_outlier_count >= 2 || std::abs(innovation[3]) > 3.0 * roll_gate) {
            tools::logger()->debug(
                "[Tracker] Reject blade measurement, blade={}, k={}, innovation(yaw={:.3f}, pitch={:.3f}, dist={:.3f}, roll={:.3f})",
                observation.blade_index, observation.phase.index, innovation[0], innovation[1],
                innovation[2], innovation[3]);
            continue;
        }

        auto h = [&](const Eigen::VectorXd& x) {
            return measurement_model(x, observation.phase.offset);
        };
        auto z_d = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            return measurement_difference(a, b);
        };
        ekf_.update(z, H, R, h, z_d);
    }
    if (use_original_big_model) {
        debug_curve_speed_after_blade_update_ = big_runev2_curve_speed_from_state(ekf_.x);
        debug_curve_phi_after_blade_update_ = big_runev2_curve_phi_from_state(ekf_.x);
        debug_curve_speed_before_speed_update_ = debug_curve_speed_after_blade_update_;
        debug_curve_phi_before_speed_update_ = debug_curve_phi_after_blade_update_;
    }

    if (
        use_original_big_model &&
        big_speed_measurement_enabled_ &&
        observed_speed_valid &&
        static_cast<int>(history_info_.size()) >= big_speed_measurement_min_history_ &&
        ekf_.x.size() >= 10 &&
        ekf_.x.array().isFinite().all()) {
        const double a = clamp_or_default(ekf_.x[7], kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
        const double phase = std::isfinite(ekf_.x[9]) ? ekf_.x[9] : 0.0;
        const double signed_predicted_speed = big_runev2_speed(a, phase);
        const double predicted_speed = std::abs(signed_predicted_speed);
        const double speed_innovation = observed_speed - predicted_speed;
        debug_curve_speed_before_speed_update_ = predicted_speed;
        debug_curve_phi_before_speed_update_ = phase;
        debug_speed_measurement_predicted_ = predicted_speed;
        debug_speed_measurement_innovation_ = speed_innovation;
        debug_speed_measurement_noise_ = big_speed_measurement_noise_;
        debug_speed_measurement_status_ = 2;
        if (std::abs(speed_innovation) <= big_speed_measurement_gate_) {
            Eigen::VectorXd z_speed(1);
            z_speed[0] = observed_speed;

            Eigen::MatrixXd H_speed = Eigen::MatrixXd::Zero(1, 10);
            const double speed_abs_sign = signed_predicted_speed < 0.0 ? -1.0 : 1.0;
            H_speed(0, 7) = speed_abs_sign * (std::sin(phase) - 1.0);
            H_speed(0, 9) = speed_abs_sign * a * std::cos(phase);

            const double adaptive_noise_scale = 1.0 +
                big_speed_measurement_adaptive_scale_ *
                    std::pow(std::abs(speed_innovation) / big_speed_measurement_gate_, 2.0);
            const double effective_speed_measurement_noise =
                big_speed_measurement_noise_ * adaptive_noise_scale;
            debug_speed_measurement_noise_ = effective_speed_measurement_noise;
            const double speed_before_update = predicted_speed;
            const double phi_before_speed_update = ekf_.x[9];
            Eigen::MatrixXd R_speed = Eigen::MatrixXd::Identity(1, 1) *
                effective_speed_measurement_noise;
            auto h_speed = [](const Eigen::VectorXd& x) {
                Eigen::VectorXd z(1);
                z[0] = std::abs(big_runev2_speed(x[7], x[9]));
                return z;
            };
            auto z_speed_sub = [](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) {
                return lhs - rhs;
            };
            ekf_.update(z_speed, H_speed, R_speed, h_speed, z_speed_sub);
            debug_speed_measurement_status_ = 1;
            if (
                big_speed_measurement_correction_limit_ > 0.0 &&
                ekf_.x.size() >= 10 &&
                ekf_.x.array().isFinite().all()) {
                const double speed_after_update = big_runev2_curve_speed_from_state(ekf_.x);
                const double speed_correction = speed_after_update - speed_before_update;
                if (std::abs(speed_correction) > big_speed_measurement_correction_limit_) {
                    const double limited_speed =
                        speed_before_update +
                        std::copysign(
                            big_speed_measurement_correction_limit_,
                            speed_correction);
                    const double a_limited =
                        clamp_or_default(ekf_.x[7], kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
                    const double signed_target_speed =
                        std::copysign(limited_speed, signed_predicted_speed);
                    const double sin_arg =
                        (signed_target_speed - (kBigBuffBaseSpeed - a_limited)) / a_limited;
                    if (std::isfinite(sin_arg) && std::abs(sin_arg) <= 1.0) {
                        const double phi_candidate_a = std::asin(sin_arg);
                        const double phi_candidate_b =
                            tools::limit_rad(M_PI - phi_candidate_a);
                        const double diff_a =
                            std::abs(tools::limit_rad(phi_candidate_a - phi_before_speed_update));
                        const double diff_b =
                            std::abs(tools::limit_rad(phi_candidate_b - phi_before_speed_update));
                        ekf_.x[9] = tools::limit_rad(
                            diff_a <= diff_b ? phi_candidate_a : phi_candidate_b);
                        debug_speed_measurement_status_ = 3;
                    }
                }
            }
        } else {
            debug_speed_measurement_status_ = -1;
            tools::logger()->debug(
                "[Tracker] Reject speed measurement, observed={:.3f}, predicted={:.3f}, gate={:.3f}",
                observed_speed, predicted_speed, big_speed_measurement_gate_);
        }
    }
    if (use_original_big_model) {
        debug_curve_speed_after_speed_update_ = big_runev2_curve_speed_from_state(ekf_.x);
        debug_curve_phi_after_speed_update_ = big_runev2_curve_phi_from_state(ekf_.x);
    }

    if (
        use_original_big_model &&
        big_curve_phi_correction_limit_ > 0.0 &&
        std::isfinite(phase_before_measurement) &&
        ekf_.x.size() >= 10 &&
        std::isfinite(ekf_.x[9])) {
        const double phase_correction = tools::limit_rad(ekf_.x[9] - phase_before_measurement);
        if (std::abs(phase_correction) > big_curve_phi_correction_limit_) {
            ekf_.x[9] = tools::limit_rad(
                phase_before_measurement +
                std::copysign(big_curve_phi_correction_limit_, phase_correction));
        }
    }
    if (use_original_big_model && big_curve_phi_correction_limit_ > 0.0) {
        debug_curve_speed_after_speed_update_ = big_runev2_curve_speed_from_state(ekf_.x);
        debug_curve_phi_after_speed_update_ = big_runev2_curve_phi_from_state(ekf_.x);
    }

    if (p.type == BIG && big_curve_ekf_fit_enabled_ && ekf_.x.size() >= 10) {
        ekf_.x[7] = clamp_or_default(ekf_.x[7], kBigBuffAMin, kBigBuffAMax, kBigBuffAInit);
        ekf_.x[8] = clamp_or_default(ekf_.x[8], kBigBuffWMin, kBigBuffWMax, kBigBuffWInit);
        ekf_.x[9] = tools::limit_rad(ekf_.x[9]);
        const double raw_curve_speed = std::abs(big_runev2_speed(ekf_.x[7], ekf_.x[9]));
        if (
            big_curve_speed_slew_limit_ > 0.0 &&
            std::isfinite(smoothed_curve_speed_) &&
            std::isfinite(raw_curve_speed)) {
            const double max_delta = big_curve_speed_slew_limit_ * std::max(0.0, dt);
            const double delta = std::clamp(
                raw_curve_speed - smoothed_curve_speed_,
                -max_delta,
                max_delta);
            smoothed_curve_speed_ += delta;
        } else {
            smoothed_curve_speed_ = raw_curve_speed;
        }
        ekf_.x[6] = raw_curve_speed;
    } else {
        ekf_.x[6] = std::abs(ekf_.x[6]);
        smoothed_curve_speed_ = ekf_.x[6];
    }
}

BuffTracker::DebugSnapshot BuffTracker::debugSnapshot(double predict_dt_s) const
{
    DebugSnapshot snapshot;
    snapshot.initialized = is_initialized_;
    snapshot.lost = is_lost_;
    snapshot.switch_deferred = last_switch_deferred_;
    snapshot.target_switched = last_target_switched_;
    snapshot.direction = voter_.clockwise();
    snapshot.history_size = static_cast<int>(history_info_.size());
    snapshot.selected_phase_index = selected_phase_index_;
    snapshot.phase_origin_index = phase_origin_index_;
    snapshot.reinit_reason = last_reinit_reason_;
    snapshot.fit_valid =
        ekf_.x.size() >= 10 && can_use_big_buff_curve_model(ekf_.x);
    snapshot.observed_roll = last_angle_;
    snapshot.observed_speed_raw = last_observed_speed_raw_;
    snapshot.observed_speed = last_observed_speed_;
    snapshot.curve_speed_after_predict = debug_curve_speed_after_predict_;
    snapshot.curve_speed_after_blade_update = debug_curve_speed_after_blade_update_;
    snapshot.curve_speed_before_speed_update = debug_curve_speed_before_speed_update_;
    snapshot.curve_speed_after_speed_update = debug_curve_speed_after_speed_update_;
    snapshot.curve_phi_after_predict = debug_curve_phi_after_predict_;
    snapshot.curve_phi_after_blade_update = debug_curve_phi_after_blade_update_;
    snapshot.curve_phi_before_speed_update = debug_curve_phi_before_speed_update_;
    snapshot.curve_phi_after_speed_update = debug_curve_phi_after_speed_update_;
    snapshot.speed_measurement_predicted = debug_speed_measurement_predicted_;
    snapshot.speed_measurement_innovation = debug_speed_measurement_innovation_;
    snapshot.speed_measurement_noise = debug_speed_measurement_noise_;
    snapshot.speed_measurement_status = debug_speed_measurement_status_;
    snapshot.selected_roll_offset = selected_target_roll_offset_;
    snapshot.blade_observations = last_blade_debug_;

    if (ekf_.x.size() >= 10 && ekf_.x.array().isFinite().all()) {
        snapshot.filtered_roll = ekf_.x[5];
        snapshot.filtered_speed = ekf_.x[6];
        snapshot.fit_a = ekf_.x[7];
        snapshot.fit_w = ekf_.x[8];
        snapshot.fit_phi = ekf_.x[9];
        if (can_use_big_buff_curve_model(ekf_.x)) {
            if (big_curve_ekf_fit_enabled_) {
                snapshot.curve_speed_raw = std::abs(big_runev2_speed(ekf_.x[7], ekf_.x[9]));
                snapshot.curve_speed_now = std::isfinite(smoothed_curve_speed_)
                    ? smoothed_curve_speed_
                    : snapshot.curve_speed_raw;
                snapshot.filtered_speed = snapshot.curve_speed_now;
            } else {
                const double t_now = std::max(0.0, tools::delta_time(last_time_, start_time_));
                snapshot.curve_speed_raw =
                    std::abs(big_buff_speed(ekf_.x[7], ekf_.x[8], ekf_.x[9], t_now));
                snapshot.curve_speed_now = snapshot.curve_speed_raw;
            }
            snapshot.filtered_speed_raw = snapshot.curve_speed_raw;
        }

        Eigen::VectorXd x_pred = predict_from_state(
            ekf_.x,
            predict_dt_s,
            snapshot.direction,
            current_relative_time_s());
        snapshot.predicted_roll = x_pred[5];
        snapshot.predicted_speed = std::abs(x_pred[6]);
    }

    return snapshot;
}

Eigen::VectorXd BuffTracker::predict(double dt)
{
    return predict_from_state(ekf_.x, dt, voter_.clockwise(), current_relative_time_s());
}

double BuffTracker::current_relative_time_s() const
{
    if (!is_initialized_) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::max(0.0, tools::delta_time(last_time_, start_time_));
}

Eigen::VectorXd BuffTracker::predict_from_state(
    const Eigen::VectorXd& state,
    double dt,
    int direction,
    double source_relative_time_s) const
{
    Eigen::VectorXd x_p = state;
    if (x_p.size() < 6 || !std::isfinite(dt)) {
        return x_p;
    }

    const double safe_dt = std::max(0.0, dt);
    if (x_p.size() > 1 && std::isfinite(x_p[0]) && std::isfinite(x_p[1])) {
        x_p[0] += x_p[1] * safe_dt;
    }
    if (direction == 0 || safe_dt <= 0.0) {
        return x_p;
    }

    if (!can_use_big_buff_curve_model(x_p)) {
        if (x_p.size() > 6 && std::isfinite(x_p[5]) && std::isfinite(x_p[6])) {
            const double speed = clamp_big_buff_speed_magnitude(x_p[6]);
            x_p[5] += direction * speed * safe_dt;
            x_p[6] = speed;
        }
        return x_p;
    }

    const double source_speed = x_p.size() > 6 && std::isfinite(x_p[6])
        ? clamp_big_buff_speed_magnitude(x_p[6])
        : std::numeric_limits<double>::quiet_NaN();
    double curve_speed_future = 0.0;
    double curve_delta = 0.0;
    if (big_curve_ekf_fit_enabled_) {
        curve_delta = big_runev2_angle_delta(x_p[7], x_p[8], x_p[9], safe_dt);
        x_p[9] = tools::limit_rad(x_p[9] + x_p[8] * safe_dt);
        curve_speed_future = std::abs(big_runev2_speed(x_p[7], x_p[9]));
    } else {
        double t_s = source_relative_time_s;
        if (!std::isfinite(t_s)) {
            t_s = current_relative_time_s();
        }
        t_s = std::max(0.0, t_s);
        const double t_e = t_s + safe_dt;
        curve_speed_future = std::abs(big_buff_speed(x_p[7], x_p[8], x_p[9], t_e));
        curve_delta = big_buff_angle_delta(x_p[7], x_p[8], x_p[9], t_s, t_e);
    }
    if (
        big_curve_speed_slew_limit_ > 0.0 &&
        std::isfinite(big_curve_speed_slew_limit_) &&
        std::isfinite(source_speed) &&
        std::isfinite(curve_speed_future)) {
        const double max_speed_delta = big_curve_speed_slew_limit_ * safe_dt;
        curve_speed_future = std::clamp(
            curve_speed_future,
            std::max(0.0, source_speed - max_speed_delta),
            std::min(kBigBuffSpeedHardMax, source_speed + max_speed_delta));
        if (std::isfinite(curve_delta)) {
            const double max_curve_delta =
                0.5 * (source_speed + curve_speed_future) * safe_dt +
                kBigBuffPredictionDeltaSlackRad;
            if (std::abs(curve_delta) > max_curve_delta) {
                curve_delta = std::copysign(max_curve_delta, curve_delta);
            }
        }
    }
    x_p[5] += direction * curve_delta;
    x_p[6] = curve_speed_future;
    return x_p;
}

Eigen::Vector3d BuffTracker::point_buff2world(const Eigen::Vector3d& pb, const Eigen::VectorXd& x) const {
    return point_buff2world(pb, x, 0.0);
}

Eigen::Vector3d BuffTracker::point_buff2world(
    const Eigen::Vector3d& pb,
    const Eigen::VectorXd& x,
    double roll_offset) const {
    Eigen::Matrix3d R = tools::rotation_matrix(Eigen::Vector3d(x[4], 0.0, x[5] + roll_offset));
    Eigen::Vector3d Rc(x[3]*cos(x[2])*cos(x[0]), x[3]*cos(x[2])*sin(x[0]), x[3]*sin(x[2]));
    return R * pb + Rc;
}

Eigen::Vector3d BuffTracker::target_point_buff2world(
    const Eigen::Vector3d& pb,
    const Eigen::VectorXd& x) const {
    return point_buff2world(pb, x, selected_target_roll_offset_);
}

Eigen::VectorXd BuffTracker::measurement_model(const Eigen::VectorXd& x) const
{
    return measurement_model(x, 0.0);
}

Eigen::VectorXd BuffTracker::measurement_model(const Eigen::VectorXd& x, double roll_offset) const
{
    Eigen::VectorXd z_p(7);
    z_p.head(3) = Eigen::Vector3d(x[0], x[2], x[3]);
    z_p[3] = x[5];
    const Eigen::Vector3d blade_point =
        point_buff2world(Eigen::Vector3d(0, 0, 0.7), x, roll_offset);
    z_p.tail(3) = tools::xyz2ypd(blade_point);
    return z_p;
}

Eigen::MatrixXd BuffTracker::h_jacobian(const Eigen::VectorXd& x) const
{
    return h_jacobian(x, 0.0);
}

Eigen::MatrixXd BuffTracker::h_jacobian(const Eigen::VectorXd& x, double roll_offset) const {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(7, 10);
    static const double eps_table[10] = {1e-4, 1e-4, 1e-4, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4};

    for (int i = 0; i < x.size(); ++i) {
        const double eps = eps_table[i];
        Eigen::VectorXd x_plus = x;
        Eigen::VectorXd x_minus = x;
        x_plus[i] += eps;
        x_minus[i] -= eps;

        if (i == 0 || i == 4) {
            x_plus[i] = tools::limit_rad(x_plus[i]);
            x_minus[i] = tools::limit_rad(x_minus[i]);
        }

        const Eigen::VectorXd diff =
            measurement_difference(
                measurement_model(x_plus, roll_offset), measurement_model(x_minus, roll_offset));
        H.col(i) = diff / (2.0 * eps);
    }
    return H;
}

} // namespace auto_buff
