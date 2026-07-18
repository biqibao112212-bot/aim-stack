#include "buff_aimer.hpp"
#include "params.h"
#include "tools/logger.hpp"
#include "tools/trajectory.hpp"
#include "tools/math_tools.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
constexpr double kRadToDeg = 180.0 / CV_PI;
constexpr double kDegToRad = CV_PI / 180.0;

double clamp_value(double value, double lower, double upper)
{
    return std::max(lower, std::min(value, upper));
}

double low_pass_filter(double previous, double sample, double alpha)
{
    const double clamped_alpha = clamp_value(alpha, 0.0, 1.0);
    if (!std::isfinite(previous)) {
        return std::isfinite(sample) ? sample : 0.0;
    }
    if (!std::isfinite(sample)) {
        return previous;
    }
    return clamped_alpha * previous + (1.0 - clamped_alpha) * sample;
}

double command_yaw_from_world(const Eigen::Vector3d& p_world, double yaw_offset)
{
    // Match the main auto-aim planner convention: world +y maps to negative gimbal yaw.
    return tools::limit_rad(std::atan2(-p_world[1], p_world[0]) + yaw_offset);
}
}

namespace auto_buff
{
Aimer::Aimer(const std::string& config_path)
    : config_path_(config_path),
      buff_runtime_config_(load_buff_runtime_config(config_path))
{
    auto yaml = YAML::LoadFile(config_path);
    const auto aimer_yaml = yaml["buff_aimer"];
    yaw_offset_ = aimer_yaml["yaw_offset"].as<double>() / 57.3;
    pitch_offset_ = aimer_yaml["pitch_offset"].as<double>() / 57.3;
    fire_gap_time_ = aimer_yaml["fire_gap_time"].as<double>();
    predict_time_ = aimer_yaml["predict_time"].as<double>();
    if (aimer_yaml["pitch_velocity_lead_time"]) {
        pitch_velocity_lead_time_ = aimer_yaml["pitch_velocity_lead_time"].as<double>();
    }
    if (!std::isfinite(pitch_velocity_lead_time_)) {
        pitch_velocity_lead_time_ = 0.0;
    }
    last_fire_t_ = std::chrono::steady_clock::now();
    params_ = std::make_unique<Params>();
    configure_yaw_mpc_from_params();
}

Aimer::~Aimer() = default;

void Aimer::reset()
{
    last_fly_time_ = 0.0;
    last_pipeline_delay_ = 0.0;
    last_base_predict_time_ = 0.0;
    last_total_predict_time_ = 0.0;
    last_pitch_debug_ = PitchDebugSnapshot{};
    last_fire_t_ = std::chrono::steady_clock::now();
    last_aim_t_.reset();
    yaw_mpc_initialized_ = false;
    yaw_rate_feedback_initialized_ = false;
    filtered_yaw_rate_deg_s_ = 0.0;
}

void Aimer::markShotFired()
{
    last_fire_t_ = std::chrono::steady_clock::now();
}

AimCommand Aimer::aim(BuffTracker& tracker, double bullet_speed) {
    return aim(
        tracker,
        bullet_speed,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::chrono::steady_clock::now());
}

AimCommand Aimer::aim(
    BuffTracker& tracker,
    double bullet_speed,
    double measured_yaw_deg,
    double measured_yaw_rate_deg_s,
    std::chrono::steady_clock::time_point timestamp,
    double pipeline_delay_s) {
    AimCommand cmd;
    if (params_ && params_->reload()) {
        configure_yaw_mpc_from_params();
        buff_runtime_config_ = load_buff_runtime_config(config_path_);
    }
    last_pipeline_delay_ = clamp_value(pipeline_delay_s, 0.0, 0.2);
    last_base_predict_time_ = base_predict_time(last_pipeline_delay_);
    last_total_predict_time_ = last_base_predict_time_;
    last_pitch_debug_ = PitchDebugSnapshot{};
    if (tracker.is_lost()) {
        yaw_mpc_initialized_ = false;
        last_aim_t_.reset();
        return cmd;
    }

    if (bullet_speed < 10) bullet_speed = 24;

    double yaw = 0, pitch = 0;
    if (solve_trajectory(tracker, bullet_speed, yaw, pitch, pipeline_delay_s)) {
        cmd.yaw = apply_yaw_mpc(
            tracker, yaw, measured_yaw_deg, measured_yaw_rate_deg_s, timestamp,
            last_base_predict_time_);
        cmd.pitch = pitch;
        last_pitch_debug_.command_pitch = cmd.pitch;
        cmd.control = true;

        auto now = std::chrono::steady_clock::now();
        if (tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
            cmd.shoot = true;
        }
    }
    return cmd;
}

void Aimer::configure_yaw_mpc_from_params()
{
    if (!params_) {
        yaw_mpc_enabled_ = false;
        return;
    }

    rm::SecondOrderPositionMPCConfig config;
    config.model_dt_s = std::max(params_->SECOND_ORDER_CTRL_MODEL_DT_S, 1e-3);
    config.horizon = std::max(params_->SECOND_ORDER_CTRL_HORIZON, 4);
    config.track_q = std::max(params_->SECOND_ORDER_CTRL_TRACK_Q, 0.0);
    config.rate_q = std::max(params_->SECOND_ORDER_CTRL_RATE_Q, 0.0);
    config.command_q = std::max(params_->SECOND_ORDER_CTRL_COMMAND_Q, 0.0);
    config.delta_r = std::max(params_->SECOND_ORDER_CTRL_DELTA_R, 1e-6);
    config.input_gain = std::max(params_->SECOND_ORDER_CTRL_YAW_K, 0.0);
    config.wn_rad_s = params_->SECOND_ORDER_CTRL_YAW_WN_RAD_S;
    config.zeta = params_->SECOND_ORDER_CTRL_YAW_ZETA;
    config.input_lag_s = std::max(params_->SECOND_ORDER_CTRL_YAW_DELAY_S, 0.0);
    config.max_rate_deg_s = params_->SECOND_ORDER_CTRL_YAW_MAX_RATE_DEG_S;
    config.max_lead_deg = params_->SECOND_ORDER_CTRL_YAW_MAX_LEAD_DEG;
    config.max_state_rate_deg_s = params_->SECOND_ORDER_CTRL_YAW_MAX_STATE_RATE_DEG_S;
    config.output_stage_ratio =
        clamp_value(params_->SECOND_ORDER_CTRL_OUTPUT_STAGE_RATIO, 0.0, 1.0);

    yaw_rate_lpf_alpha_ =
        clamp_value(params_->SECOND_ORDER_CTRL_YAW_FEEDBACK_LPF_ALPHA, 0.0, 1.0);
    yaw_mpc_enabled_ = params_->AIM_COMMAND_CTRL_MODE == 2;
    yaw_mpc_config_ = config;
    yaw_mpc_.configure(yaw_mpc_config_);
}

bool Aimer::build_yaw_reference(
    BuffTracker& tracker,
    Eigen::VectorXd& yaw_ref_deg,
    Eigen::VectorXd& yaw_rate_ref_deg_s,
    double base_predict_time_s) const
{
    const int horizon = std::max(yaw_mpc_config_.horizon, 4);
    const double model_dt_s = std::max(yaw_mpc_config_.model_dt_s, 1e-3);
    yaw_ref_deg.resize(horizon);
    yaw_rate_ref_deg_s.resize(horizon);

    std::vector<double> yaw_ref_rad;
    yaw_ref_rad.reserve(horizon);
    const Eigen::Vector3d target_in_buff = target_point_in_buff();
    const double base_dt_s = std::max(base_predict_time_s, 0.0) +
                             std::max(last_fly_time_, 0.0);
    for (int i = 0; i < horizon; ++i) {
        const double dt_s = base_dt_s + static_cast<double>(i) * model_dt_s;
        const Eigen::VectorXd x_pred = tracker.predict(dt_s);
        const Eigen::Vector3d p_world =
            tracker.target_point_buff2world(target_in_buff, x_pred);
        if (!p_world.array().isFinite().all()) {
            yaw_ref_deg.resize(0);
            yaw_rate_ref_deg_s.resize(0);
            return false;
        }

        const double yaw_rad = command_yaw_from_world(p_world, yaw_offset_);
        yaw_ref_rad.push_back(yaw_rad);
        yaw_ref_deg[i] = yaw_rad * kRadToDeg;
    }

    yaw_rate_ref_deg_s.setZero();
    for (int i = 1; i < horizon; ++i) {
        const double delta_rad = tools::limit_rad(yaw_ref_rad[i] - yaw_ref_rad[i - 1]);
        yaw_rate_ref_deg_s[i] = delta_rad * kRadToDeg / model_dt_s;
    }
    if (horizon > 1) {
        yaw_rate_ref_deg_s[0] = yaw_rate_ref_deg_s[1];
    }
    return true;
}

double Aimer::apply_yaw_mpc(
    BuffTracker& tracker,
    double raw_yaw_rad,
    double measured_yaw_deg,
    double measured_yaw_rate_deg_s,
    std::chrono::steady_clock::time_point timestamp,
    double base_predict_time_s)
{
    if (buff_runtime_config_.correction_enabled) {
        yaw_mpc_initialized_ = false;
        yaw_rate_feedback_initialized_ = false;
        last_aim_t_.reset();
        return raw_yaw_rad;
    }

    if (!yaw_mpc_enabled_ || !std::isfinite(measured_yaw_deg)) {
        yaw_mpc_initialized_ = false;
        last_aim_t_.reset();
        return raw_yaw_rad;
    }

    Eigen::VectorXd yaw_ref_deg;
    Eigen::VectorXd yaw_rate_ref_deg_s;
    if (!build_yaw_reference(tracker, yaw_ref_deg, yaw_rate_ref_deg_s, base_predict_time_s)) {
        return raw_yaw_rad;
    }

    const double measured_rate_sample =
        std::isfinite(measured_yaw_rate_deg_s) ? measured_yaw_rate_deg_s : filtered_yaw_rate_deg_s_;
    if (!yaw_rate_feedback_initialized_) {
        filtered_yaw_rate_deg_s_ =
            std::isfinite(measured_rate_sample) ? measured_rate_sample : 0.0;
        yaw_rate_feedback_initialized_ = true;
    } else {
        filtered_yaw_rate_deg_s_ = low_pass_filter(
            filtered_yaw_rate_deg_s_, measured_rate_sample, yaw_rate_lpf_alpha_);
    }

    double applied_dt_s = std::max(yaw_mpc_config_.model_dt_s, 1e-3);
    if (last_aim_t_.has_value()) {
        applied_dt_s = tools::delta_time(timestamp, *last_aim_t_);
    }
    const bool reset_mpc =
        !yaw_mpc_initialized_ || !last_aim_t_.has_value() ||
        !std::isfinite(applied_dt_s) || applied_dt_s <= 0.0 || applied_dt_s > 0.25;
    if (reset_mpc) {
        yaw_mpc_.reset(measured_yaw_deg, filtered_yaw_rate_deg_s_);
        yaw_mpc_initialized_ = true;
        applied_dt_s = std::max(yaw_mpc_config_.model_dt_s, 1e-3);
    }
    last_aim_t_ = timestamp;

    yaw_mpc_.configure(yaw_mpc_config_);
    yaw_mpc_.setPreviewWindow(2, 2);
    const double command_yaw_deg = yaw_mpc_.updateTrajectory(
        yaw_ref_deg,
        yaw_rate_ref_deg_s,
        measured_yaw_deg,
        filtered_yaw_rate_deg_s_,
        applied_dt_s);

    if (!std::isfinite(command_yaw_deg)) {
        return raw_yaw_rad;
    }

    tools::logger()->debug(
        "[BuffAimer] yaw raw={:.2f}deg mpc={:.2f}deg measured={:.2f}deg ref0={:.2f}deg",
        raw_yaw_rad * kRadToDeg,
        command_yaw_deg,
        measured_yaw_deg,
        yaw_ref_deg.size() > 0 ? yaw_ref_deg[0] : raw_yaw_rad * kRadToDeg);

    return command_yaw_deg * kDegToRad;
}

double Aimer::base_predict_time(double pipeline_delay_s) const
{
    const double state_age_s = clamp_value(pipeline_delay_s, 0.0, 0.2);
    const double configured_delay_s = std::max(0.0, predict_time_);
    const double execution_delay_s =
        params_ ? std::max(0.0, static_cast<double>(params_->HORIZONTAL_DELAY_TIME)) : 0.0;
    return clamp_value(state_age_s + configured_delay_s + execution_delay_s, 0.0, 0.5);
}

Eigen::Vector3d Aimer::target_point_in_buff() const
{
    if (buff_runtime_config_.correction_enabled) {
        return Eigen::Vector3d::Zero();
    }
    return Eigen::Vector3d(0.0, 0.0, 0.7);
}

double Aimer::estimate_target_pitch_rate(BuffTracker& tracker, double predict_dt_s) const
{
    const double sample_dt_s =
        std::clamp(std::max(yaw_mpc_config_.model_dt_s, 0.004), 0.004, 0.02);
    const double safe_dt_s = std::max(0.0, predict_dt_s);

    const Eigen::VectorXd x0 = tracker.predict(safe_dt_s);
    const Eigen::VectorXd x1 = tracker.predict(safe_dt_s + sample_dt_s);
    if (x0.size() < 10 || x1.size() < 10 ||
        !x0.array().isFinite().all() || !x1.array().isFinite().all()) {
        return 0.0;
    }

    const Eigen::Vector3d target_in_buff = target_point_in_buff();
    const Eigen::Vector3d p0 =
        tracker.target_point_buff2world(target_in_buff, x0);
    const Eigen::Vector3d p1 =
        tracker.target_point_buff2world(target_in_buff, x1);
    if (!p0.array().isFinite().all() || !p1.array().isFinite().all()) {
        return 0.0;
    }

    const double pitch0 = tools::xyz2ypd(p0)[1];
    const double pitch1 = tools::xyz2ypd(p1)[1];
    if (!std::isfinite(pitch0) || !std::isfinite(pitch1)) {
        return 0.0;
    }

    return (pitch1 - pitch0) / sample_dt_s;
}

bool Aimer::solve_trajectory(
    BuffTracker& tracker, double v, double& yaw, double& pitch, double pipeline_delay_s) {
    last_pipeline_delay_ = clamp_value(pipeline_delay_s, 0.0, 0.2);
    const double base_dt_s = base_predict_time(last_pipeline_delay_);
    last_base_predict_time_ = base_dt_s;
    last_total_predict_time_ = base_dt_s;

    // 1. 初始预测：流水线帧龄 + 固定补偿 + 执行延迟
    const Eigen::Vector3d target_in_buff = target_point_in_buff();
    Eigen::VectorXd x_pred = tracker.predict(base_dt_s);
    Eigen::Vector3d p_world = tracker.target_point_buff2world(target_in_buff, x_pred);

    double d = std::hypot(p_world[0], p_world[1]);
    double h = p_world[2];

    // 2. 弹道解算迭代 (考虑飞行时间)
    for (int i = 0; i < 2; ++i) {
        tools::Trajectory traj(v, d, h);
        if (traj.unsolvable) return false;

        // ============= 新增：记录最新的飞行时间 =============
        last_fly_time_ = traj.fly_time;
        // ==================================================

        // 重新预测位置 = 流水线/执行延迟 + 飞行时间
        x_pred = tracker.predict(base_dt_s + traj.fly_time);
        p_world = tracker.target_point_buff2world(target_in_buff, x_pred);

        d = std::hypot(p_world[0], p_world[1]);
        h = p_world[2];

        // 最后一次迭代计算发射角
        if (i == 1) {
             tools::Trajectory final_traj(v, d, h);
             yaw = command_yaw_from_world(p_world, yaw_offset_);
             const double total_predict_dt_s = base_dt_s + traj.fly_time;
             last_total_predict_time_ = total_predict_dt_s;
             const double pitch_rate = estimate_target_pitch_rate(tracker, total_predict_dt_s);
             const double pitch_velocity_lead = pitch_rate * pitch_velocity_lead_time_;
             pitch = final_traj.pitch + pitch_offset_ + pitch_velocity_lead;
             last_pitch_debug_.valid = true;
             last_pitch_debug_.target_in_buff = target_in_buff;
             last_pitch_debug_.target_in_world = p_world;
             last_pitch_debug_.horizontal_distance = d;
             last_pitch_debug_.height = h;
             last_pitch_debug_.raw_yaw = yaw;
             last_pitch_debug_.ballistic_pitch = final_traj.pitch;
             last_pitch_debug_.pitch_offset = pitch_offset_;
             last_pitch_debug_.pitch_rate = pitch_rate;
             last_pitch_debug_.pitch_lead = pitch_velocity_lead;
             last_pitch_debug_.solved_pitch = pitch;
             tools::logger()->debug(
                 "[BuffAimer] pitch base={:.3f}deg rate={:.3f}deg/s lead_time={:.3f}s lead={:.3f}deg",
                 (final_traj.pitch + pitch_offset_) * kRadToDeg,
                 pitch_rate * kRadToDeg,
                 pitch_velocity_lead_time_,
                 pitch_velocity_lead * kRadToDeg);
        }
    }
    return true;
}
}
