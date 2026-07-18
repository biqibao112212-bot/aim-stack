#include "firecontrol.h"

#include "trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>

namespace rm
{

namespace {

constexpr double kAutoFireMpcCheckAheadS = 0.05;
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
constexpr double kFireImpactDirectionEpsilonRadS = 0.2;

double elapsed_us(const std::chrono::steady_clock::time_point& begin)
{
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - begin)
        .count();
}

double normalize_angle_deg(double angle)
{
    double result = std::fmod(angle + 180.0, 360.0);
    if (result < 0.0) result += 360.0;
    return result - 180.0;
}

double angle_diff_deg(double a, double b)
{
    return std::abs(normalize_angle_deg(a - b));
}

double closest_equivalent_angle_deg(double reference_deg, double angle_deg)
{
    return reference_deg + normalize_angle_deg(angle_deg - reference_deg);
}

const char* fire_phase_mode_name(FirePhaseMode mode)
{
    switch (mode) {
    case FirePhaseMode::Auto:
        return "auto";
    case FirePhaseMode::Single:
        return "single";
    case FirePhaseMode::None:
    default:
        return "none";
    }
}

double clamp_value(double value, double lower, double upper)
{
    return std::max(lower, std::min(value, upper));
}

double fire_yaw_tolerance_deg(double distance_m, const Params& params)
{
    const double lower = std::min(
        params.FIRE_YAW_TOLERANCE_MIN_DEG, params.FIRE_YAW_TOLERANCE_MAX_DEG);
    const double upper = std::max(
        params.FIRE_YAW_TOLERANCE_MIN_DEG, params.FIRE_YAW_TOLERANCE_MAX_DEG);
    const double miss_tolerance_m =
        std::isfinite(params.FIRE_YAW_MISS_TOLERANCE_M) &&
                params.FIRE_YAW_MISS_TOLERANCE_M > 0.0
            ? params.FIRE_YAW_MISS_TOLERANCE_M
            : 0.055;
    if (!std::isfinite(distance_m) || distance_m <= 1e-3) return upper;

    const double tolerance_deg = std::atan(miss_tolerance_m / distance_m) * kRadToDeg;
    return clamp_value(tolerance_deg, lower, upper);
}

bool fire_impact_angle_in_window(
    double impact_delta_angle_rad, double omega_rad_s, const Params& params)
{
    if (!std::isfinite(impact_delta_angle_rad)) return false;

    const double configured_enter_deg =
        std::max(0.0, params.FIRE_ARMOR_IMPACT_ENTER_ANGLE_DEG);
    const double configured_leave_deg =
        std::max(0.0, params.FIRE_ARMOR_IMPACT_LEAVE_ANGLE_DEG);
    const double enter_rad = std::max(configured_enter_deg, configured_leave_deg) * kDegToRad;
    const double leave_rad = std::min(configured_enter_deg, configured_leave_deg) * kDegToRad;

    if (std::abs(omega_rad_s) <= kFireImpactDirectionEpsilonRadS) {
        return std::abs(impact_delta_angle_rad) <= enter_rad;
    }
    return omega_rad_s > 0.0
               ? (impact_delta_angle_rad >= -enter_rad &&
                     impact_delta_angle_rad <= leave_rad)
               : (impact_delta_angle_rad <= enter_rad &&
                     impact_delta_angle_rad >= -leave_rad);
}

bool fire_impact_angle_ref_in_window(
    const Eigen::VectorXd& impact_delta_angle_ref_deg, int index, double omega_rad_s,
    const Params& params, double* impact_delta_angle_deg = nullptr)
{
    if (impact_delta_angle_deg != nullptr) {
        *impact_delta_angle_deg = std::numeric_limits<double>::quiet_NaN();
    }
    if (index < 0 || index >= impact_delta_angle_ref_deg.size()) {
        return false;
    }

    const double delta_deg = impact_delta_angle_ref_deg(index);
    if (impact_delta_angle_deg != nullptr) {
        *impact_delta_angle_deg = delta_deg;
    }
    return fire_impact_angle_in_window(delta_deg * kDegToRad, omega_rad_s, params);
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

double shot_slot_window_max_error_deg(
    const Eigen::VectorXd& predicted_yaw_deg, const Eigen::VectorXd& reference_yaw_deg,
    int horizon, double model_dt_s, int slot_index, double pre_ms, double post_ms)
{
    if (horizon <= 1 || model_dt_s <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int clamped_slot_index = std::clamp(slot_index, 1, std::max(1, horizon - 1));
    const int pre_steps = std::max(
        0, static_cast<int>(std::ceil(std::max(0.0, pre_ms) * 1e-3 / model_dt_s)));
    const int post_steps = std::max(
        0, static_cast<int>(std::ceil(std::max(0.0, post_ms) * 1e-3 / model_dt_s)));
    const int start_index = std::clamp(clamped_slot_index - pre_steps, 1, horizon - 1);
    const int end_index = std::clamp(clamped_slot_index + post_steps, start_index, horizon - 1);

    double max_error_deg = 0.0;
    for (int index = start_index; index <= end_index; ++index) {
        const double slot_error_deg =
            std::abs(reference_yaw_deg(index) - predicted_yaw_deg(index));
        if (!std::isfinite(slot_error_deg)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        max_error_deg = std::max(max_error_deg, slot_error_deg);
    }
    return max_error_deg;
}

bool is_finite_vec3(const Eigen::Vector3d& value)
{
    return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}

double sanitize_dt_upper_bound(double configured_dt_s)
{
    if (!std::isfinite(configured_dt_s) || configured_dt_s <= 1e-3) {
        return 0.01;
    }
    return configured_dt_s;
}

double control_dt_upper_bound_sec(const Params& params)
{
    if (params.AIM_COMMAND_CTRL_MODE == 2) {
        return sanitize_dt_upper_bound(params.SECOND_ORDER_CTRL_MODEL_DT_S);
    }
    return sanitize_dt_upper_bound(params.SECOND_ORDER_CTRL_MODEL_DT_S);
}

const char* aim_command_ctrl_mode_name(int mode)
{
    switch (mode) {
        case 0:
            return "legacy";
        case 1:
            return "legacy";
        case 2:
            return "mpc2";
        default:
            return "unknown";
    }
}

bool static_target_bypasses_second_order_mpc(const Params& params, MOVEMENT movement)
{
    return params.AIM_COMMAND_CTRL_MODE == 2 && movement == STATIC;
}

std::filesystem::path resolve_firecontrol_debug_log_path()
{
    namespace fs = std::filesystem;

    const fs::path cwd = fs::current_path();
    if (cwd.filename() == "build") {
        return cwd / "firecontrol_shot_debug.csv";
    }

    const fs::path build_dir = cwd / "build";
    if (fs::exists(build_dir) && fs::is_directory(build_dir)) {
        return build_dir / "firecontrol_shot_debug.csv";
    }

    return cwd / "firecontrol_shot_debug.csv";
}

void write_firecontrol_debug_log_header(std::ofstream& out)
{
    out << "time_stamp_ms,control_dt_ms,target_detected,planner_active,"
           "preview_start_idx,preview_end_idx,preview_error_deg,preview_valid,"
           "fire_check_idx,fire_check_time_ms,fire_check_error_deg,fire_check_valid,"
           "current_yaw_deg,current_yaw_rate_deg_s,legacy_yaw_deg,base_planner_yaw_deg,"
           "final_cmd_yaw_deg,final_cmd_pitch_deg,state_age_ms,movement,shot_mode,gate_valid,gate_mcu_permit,gate_stable,"
           "gate_follow,gate_preview,gate_impact_angle,impact_delta_angle_deg,"
           "gate_slot_window,gate_motion_uniform,gate_observation_stable,"
           "burst_active,fire_phase,next_slot_ms,mechanical_hold,viable_slot_count,first_slot_error_deg,"
           "pitch_cmd_delta_deg,pitch_follow_error_deg,"
           "motion_translation_burst_metric,motion_translation_drift_metric\n";
}

struct FireControlDebugLogSink
{
    std::mutex mutex;
    std::ofstream out;
    bool initialized = false;
    bool enabled = false;
};

FireControlDebugLogSink& firecontrol_debug_log_sink()
{
    static FireControlDebugLogSink sink;
    return sink;
}

}  // namespace

FireControl::FireControl()
{
    _params = std::make_unique<Params>();
    if (_params->DEBUG_LOG_FIRECONTROL_CSV) {
        auto& sink = firecontrol_debug_log_sink();
        const std::filesystem::path log_path = resolve_firecontrol_debug_log_path();
        std::lock_guard<std::mutex> lock(sink.mutex);
        if (!sink.initialized) {
            sink.out.open(log_path, std::ios::out | std::ios::trunc);
            if (sink.out.is_open()) {
                write_firecontrol_debug_log_header(sink.out);
                sink.out << std::fixed << std::setprecision(6);
                sink.out.flush();
                sink.enabled = true;
                std::cerr << "[FireControl] debug log path: " << log_path << std::endl;
            } else {
                std::cerr << "[FireControl] failed to open debug log: " << log_path << std::endl;
            }
            sink.initialized = true;
        }
        debug_firecontrol_log_enabled_ = sink.enabled;
    }
    shoot_time = timer.tic();
}

FireControlTargetSnapshot FireControl::makeTargetSnapshot(
    const rm::Estimator& estimator, double state_age_s)
{
    FireControlTargetSnapshot snapshot;
    snapshot.detected_flag = estimator._detectedFlag;
    snapshot.aim_point = estimator._aimPoint;
    snapshot.target_state11d = estimator._targetState11d;
    snapshot.target_state_mat = estimator._targetStateMat;
    snapshot.tracked_armors_num = static_cast<int>(estimator.tracked_armors_num);
    snapshot.target_jumped = estimator.jump_flag != 0;
    snapshot.movement = estimator.movement;
    snapshot.motion_uniform = estimator.fire_motion_uniform;
    snapshot.observation_stable = estimator.fire_observation_stable;
    snapshot.motion_translation_burst_metric =
        estimator.fire_motion_translation_burst_metric;
    snapshot.motion_translation_drift_metric =
        estimator.fire_motion_translation_drift_metric;
    snapshot.state_age_s = clamp_value(state_age_s, 0.0, 0.2);

    if (estimator._trackedArmor) {
        snapshot.tracked_armor_valid = true;
        snapshot.tracked_armor_position = estimator._trackedArmor->armorPosition;
        snapshot.tracked_armor_right = estimator._trackedArmor->armorR;
        snapshot.tracked_armor_left = estimator._trackedArmor->armorL;
    }

    return snapshot;
}

void FireControl::loadFrame(rm::Frame& frame)
{
    loadMeta(FrameMeta(frame));
    attachDebugImage(frame.debugImg);
}

void FireControl::loadMeta(const FrameMeta& frame_meta)
{
    _gimbalPose = frame_meta.poseEuler;
    mcu_fire_permit_ = frame_meta.fb.mcu_fire_permit();
    _bulletSpeed = frame_meta.bullet_speed;
    _nowTime = frame_meta.timeStamp;
    const double measured_yaw_rate_deg_s =
        std::isfinite(frame_meta.fb.yaw_speed) ? frame_meta.fb.yaw_speed : _yaw_speed;
    if (!yaw_speed_feedback_initialized_) {
        _yaw_speed = measured_yaw_rate_deg_s;
        yaw_speed_feedback_initialized_ = true;
    } else {
        _yaw_speed = low_pass_filter(
            _yaw_speed, measured_yaw_rate_deg_s,
            clamp_value(_params->SECOND_ORDER_CTRL_YAW_FEEDBACK_LPF_ALPHA, 0.0, 1.0));
    }
    _heat_cap = frame_meta.fb.heat_cap;

    const auto now_tp = std::chrono::steady_clock::now();
    double wall_dt_s = std::numeric_limits<double>::quiet_NaN();
    if (control_timer_initialized_) {
        wall_dt_s = std::chrono::duration<double>(now_tp - last_control_tp_).count();
    }
    last_control_tp_ = now_tp;
    control_timer_initialized_ = true;

    double frame_dt_s = std::numeric_limits<double>::quiet_NaN();
    if (last_control_timestamp_ms_ > 0.0 && frame_meta.timeStamp > last_control_timestamp_ms_) {
        frame_dt_s = (frame_meta.timeStamp - last_control_timestamp_ms_) / 1000.0;
    }

    wall_control_dt_sec_ = wall_dt_s;
    frame_control_dt_sec_ = frame_dt_s;
    simulator_state_age_s_ = clamp_value(frame_meta.simulator_state_age_s, 0.0, 0.2);

    double measured_dt_s = wall_dt_s;
    if (!std::isfinite(measured_dt_s) || measured_dt_s <= 0.0) {
        measured_dt_s = frame_dt_s;
    }

    const double control_dt_upper_s = control_dt_upper_bound_sec(*_params);
    if (std::isfinite(measured_dt_s) && measured_dt_s > 0.0) {
        control_dt_sec_ = clamp_value(measured_dt_s, 1e-3, control_dt_upper_s);
    } else {
        control_dt_sec_ = control_dt_upper_s;
    }

    last_control_timestamp_ms_ = frame_meta.timeStamp;
    _params->reload();
}

void FireControl::attachDebugImage(const cv::Mat& image)
{
    _debugImg = image;
}

void FireControl::attachDebugHud(DebugHudSnapshot* hud)
{
    _debugHud = hud;
}

void FireControl::resetPlannerTrackingState(double fallback_yaw_deg)
{
    legacy_command_yaw_deg_ = fallback_yaw_deg;
    base_planner_yaw_deg_ = fallback_yaw_deg;
    yaw_planner_active_ = false;
    clearLastYawPlan();
    yaw_preview_tracking_error_deg_ = 0.0;
    yaw_preview_tracking_valid_ = false;
    yaw_fire_check_error_deg_ = 0.0;
    yaw_fire_check_valid_ = false;
    yaw_fire_check_index_ = 0;
    yaw_fire_check_time_ms_ = 0.0;
    yaw_preview_start_index_ = 2;
    yaw_preview_end_index_ = 2;
    yaw_static_mpc_bypass_active_ = false;
    fire_impact_delta_angle_valid_ = false;
    fire_impact_delta_angle_deg_ = std::numeric_limits<double>::quiet_NaN();
    fire_impact_delta_angle_ref_deg_.resize(0);
}

void FireControl::clearLastYawPlan()
{
    last_yaw_plan_valid_ = false;
    last_plan_selected_armor_index_ = -1;
    last_plan_execution_delay_s_ = std::numeric_limits<double>::quiet_NaN();
    last_plan_estimated_fly_time_s_ = std::numeric_limits<double>::quiet_NaN();
    last_plan_target_yaw_deg_ = std::numeric_limits<double>::quiet_NaN();
    last_plan_target_yaw_vel_deg_s_ = std::numeric_limits<double>::quiet_NaN();
    last_plan_impact_delta_angle_deg_ = std::numeric_limits<double>::quiet_NaN();
    last_plan_target_pos_ =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    last_plan_zero_vxy_target_pos_ =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    last_plan_zero_vxy_delta_m_ = std::numeric_limits<double>::quiet_NaN();
}

void FireControl::clearAutoFireState(bool clear_last_command)
{
    if (clear_last_command) {
        has_last_auto_command_ = false;
    }
    clearFirePhaseState();
    fire_auto_restart_cooldown_until_ = {};
    fire_gate_valid_ = false;
    fire_gate_command_stable_ = false;
    fire_gate_follow_ = false;
    fire_gate_mcu_permit_ = false;
    fire_gate_preview_ = false;
    fire_gate_impact_angle_ = false;
    fire_gate_slot_window_ = false;
    fire_gate_motion_uniform_ = false;
    fire_gate_observation_stable_ = false;
    fire_tolerance_deg_ = 0.0;
    fire_cmd_delta_deg_ = 0.0;
    fire_follow_error_deg_ = 0.0;
    fire_pitch_cmd_delta_deg_ = 0.0;
    fire_pitch_follow_error_deg_ = 0.0;
    fire_viable_slot_count_ = 0;
    fire_first_slot_error_deg_ = std::numeric_limits<double>::quiet_NaN();
}

void FireControl::clearFirePhaseState()
{
    fire_burst_active_ = false;
    fire_single_window_latched_ = false;
    fire_burst_hold_deadline_ = {};
    fire_phase_mode_ = FirePhaseMode::None;
    fire_next_slot_tp_ = {};
    fire_next_slot_delay_ms_ = std::numeric_limits<double>::quiet_NaN();
    fire_single_pulse_sent_ = false;
    fire_mechanical_hold_active_ = false;
}

void FireControl::applyNoTargetControlState()
{
    _controlData.shot_mode = ControlData::SHOT_MODE::DO_NOTHING;
    _controlData.aiming_state = ControlData::AIMING_STATE::AIMMING_NO_TARGET;
    _controlData.yaw_error = 0;
    resetPlannerTrackingState(_gimbalPose.yaw);
    raw_command_yaw_deg_ = _gimbalPose.yaw;
    raw_command_pitch_deg_ = _gimbalPose.pitch;
    resetAimCommandController();
    clearAutoFireState(true);
}

void FireControl::resetExecutionState()
{
    resetAimCommandController();
    resetPlannerTrackingState(_gimbalPose.yaw);
    clearAutoFireState(true);
}

void FireControl::resetAimCommandController()
{
    yaw_second_order_mpc_.reset(_gimbalPose.yaw, _yaw_speed);
    pitch_second_order_mpc_.reset(_gimbalPose.pitch, 0.0);
    filtered_command_yaw_deg_ = _gimbalPose.yaw;
    filtered_command_pitch_deg_ = _gimbalPose.pitch;
}

rm::PlannerConfig FireControl::buildYawPlannerConfig() const
{
    rm::PlannerConfig config;
    config.yaw_offset = 0.0;
    config.execution_delay_s = std::max(0.0, static_cast<double>(_params->HORIZONTAL_DELAY_TIME));
    config.preview_dt_s = std::max(_params->SECOND_ORDER_CTRL_MODEL_DT_S, 1e-3);
    config.preview_horizon = std::max(_params->SECOND_ORDER_CTRL_HORIZON, 4);
    config.armor_enter_angle_deg = std::max(0.0, _params->PLANNER_ARMOR_ENTER_ANGLE_DEG);
    config.armor_leave_angle_deg = std::max(0.0, _params->PLANNER_ARMOR_LEAVE_ANGLE_DEG);
    return config;
}

rm::SecondOrderPositionMPCConfig FireControl::buildSecondOrderAimConfig(bool is_pitch) const
{
    rm::SecondOrderPositionMPCConfig config;
    config.model_dt_s = std::max(_params->SECOND_ORDER_CTRL_MODEL_DT_S, 1e-3);
    config.horizon = std::max(_params->SECOND_ORDER_CTRL_HORIZON, 4);
    config.track_q = std::max(_params->SECOND_ORDER_CTRL_TRACK_Q, 0.0);
    config.rate_q = std::max(_params->SECOND_ORDER_CTRL_RATE_Q, 0.0);
    config.command_q = std::max(_params->SECOND_ORDER_CTRL_COMMAND_Q, 0.0);
    config.delta_r = std::max(_params->SECOND_ORDER_CTRL_DELTA_R, 1e-6);
    config.input_gain = std::max(
        is_pitch ? _params->SECOND_ORDER_CTRL_PITCH_K : _params->SECOND_ORDER_CTRL_YAW_K, 0.0);
    config.wn_rad_s = is_pitch ? _params->SECOND_ORDER_CTRL_PITCH_WN_RAD_S
                               : _params->SECOND_ORDER_CTRL_YAW_WN_RAD_S;
    config.zeta = is_pitch ? _params->SECOND_ORDER_CTRL_PITCH_ZETA
                           : _params->SECOND_ORDER_CTRL_YAW_ZETA;
    config.input_lag_s = std::max(
        is_pitch ? _params->SECOND_ORDER_CTRL_PITCH_DELAY_S
                 : _params->SECOND_ORDER_CTRL_YAW_DELAY_S,
        0.0);
    config.max_rate_deg_s = is_pitch ? _params->SECOND_ORDER_CTRL_PITCH_MAX_RATE_DEG_S
                                     : _params->SECOND_ORDER_CTRL_YAW_MAX_RATE_DEG_S;
    config.max_lead_deg = is_pitch ? _params->SECOND_ORDER_CTRL_PITCH_MAX_LEAD_DEG
                                   : _params->SECOND_ORDER_CTRL_YAW_MAX_LEAD_DEG;
    config.max_state_rate_deg_s =
        is_pitch ? _params->SECOND_ORDER_CTRL_PITCH_MAX_STATE_RATE_DEG_S
                 : _params->SECOND_ORDER_CTRL_YAW_MAX_STATE_RATE_DEG_S;
    config.output_stage_ratio =
        clamp_value(_params->SECOND_ORDER_CTRL_OUTPUT_STAGE_RATIO, 0.0, 1.0);
    return config;
}

rm::Planner& FireControl::ensureYawPlanner()
{
    const rm::PlannerConfig config = buildYawPlannerConfig();
    if (!yaw_planner_) {
        yaw_planner_ = std::make_unique<rm::Planner>(config);
    } else {
        yaw_planner_->configure(config);
    }
    return *yaw_planner_;
}

void FireControl::updateYawFireCheckFromMpc()
{
    yaw_fire_check_error_deg_ = 0.0;
    yaw_fire_check_valid_ = false;
    yaw_fire_check_index_ = 0;
    yaw_fire_check_time_ms_ = 0.0;

    const Eigen::VectorXd& predicted_yaw_deg = yaw_second_order_mpc_.lastPredictedYawTrajectoryDeg();
    const Eigen::VectorXd& reference_yaw_deg = yaw_second_order_mpc_.lastReferenceTrajectoryDeg();
    const int horizon = std::min<int>(predicted_yaw_deg.size(), reference_yaw_deg.size());
    if (horizon <= 0) return;

    const double model_dt_s = std::max(_params->SECOND_ORDER_CTRL_MODEL_DT_S, 1e-3);
    const int fire_check_index = std::clamp(
        static_cast<int>(std::lround(kAutoFireMpcCheckAheadS / model_dt_s)), 0, horizon - 1);
    const double fire_check_error_deg =
        std::abs(reference_yaw_deg(fire_check_index) - predicted_yaw_deg(fire_check_index));
    if (!std::isfinite(fire_check_error_deg)) return;

    yaw_fire_check_index_ = fire_check_index;
    yaw_fire_check_time_ms_ = static_cast<double>(fire_check_index) * model_dt_s * 1000.0;
    yaw_fire_check_error_deg_ = fire_check_error_deg;
    yaw_fire_check_valid_ = true;
}

double FireControl::computeShotCheckStartTimeS(bool first_shot_candidate) const
{
    const double model_dt_s = std::max(_params->SECOND_ORDER_CTRL_MODEL_DT_S, 1e-3);
    const double slot_period_s = fireSlotPeriodS();
    if (!first_shot_candidate) return slot_period_s;

    const double max_advance_s = std::max(0.0, slot_period_s - model_dt_s);
    const double raw_advance_s = _params->FIRE_FIRST_SHOT_ADVANCE_MS * 1e-3;
    const double advance_s = std::min(raw_advance_s, max_advance_s);
    return std::max(model_dt_s, slot_period_s - advance_s);
}

double FireControl::fireSlotPeriodS() const
{
    return 1.0 / std::max(_params->FIRE_RATE_HZ, 1e-3);
}

void FireControl::advanceFirePhase(const std::chrono::steady_clock::time_point& now)
{
    if (fire_phase_mode_ == FirePhaseMode::None || fire_next_slot_tp_ == std::chrono::steady_clock::time_point{}) {
        return;
    }

    if (fire_phase_mode_ == FirePhaseMode::Single && now >= fire_next_slot_tp_) {
        clearFirePhaseState();
        return;
    }

    const double slot_period_s = fireSlotPeriodS();
    const auto slot_period =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(slot_period_s));
    while (fire_phase_mode_ == FirePhaseMode::Auto && now >= fire_next_slot_tp_) {
        fire_next_slot_tp_ += slot_period;
    }
}

void FireControl::startFirePhase(
    FirePhaseMode mode, const std::chrono::steady_clock::time_point& now,
    double first_slot_time_s)
{
    const double model_dt_s = std::max(_params->SECOND_ORDER_CTRL_MODEL_DT_S, 1e-3);
    const double clamped_first_slot_s = std::max(first_slot_time_s, model_dt_s);
    fire_phase_mode_ = mode;
    fire_next_slot_tp_ =
        now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double>(clamped_first_slot_s));
    fire_next_slot_delay_ms_ = clamped_first_slot_s * 1000.0;
    fire_single_pulse_sent_ = false;
    fire_single_window_latched_ = true;
    fire_burst_active_ = mode == FirePhaseMode::Auto;
}

double FireControl::firePhaseFirstSlotTimeS(
    const std::chrono::steady_clock::time_point& now) const
{
    const double model_dt_s = std::max(_params->SECOND_ORDER_CTRL_MODEL_DT_S, 1e-3);
    if (fire_phase_mode_ == FirePhaseMode::None || fire_next_slot_tp_ == std::chrono::steady_clock::time_point{}) {
        return computeShotCheckStartTimeS(true);
    }

    const double delay_s = std::chrono::duration<double>(fire_next_slot_tp_ - now).count();
    return std::max(model_dt_s, delay_s);
}

int FireControl::countViableShotSlots(
    double tolerance_deg, double first_slot_time_s, double target_omega_rad_s,
    bool require_impact_angle_gate, bool mcu_fire_permit,
    double* first_slot_error_deg,
    double* first_viable_slot_time_s, bool* first_slot_impact_gate,
    double* first_slot_impact_delta_deg) const
{
    if (first_slot_error_deg != nullptr) {
        *first_slot_error_deg = std::numeric_limits<double>::quiet_NaN();
    }
    if (first_viable_slot_time_s != nullptr) {
        *first_viable_slot_time_s = std::numeric_limits<double>::quiet_NaN();
    }
    if (first_slot_impact_gate != nullptr) {
        *first_slot_impact_gate = false;
    }
    if (first_slot_impact_delta_deg != nullptr) {
        *first_slot_impact_delta_deg = std::numeric_limits<double>::quiet_NaN();
    }
    if (!std::isfinite(tolerance_deg) || tolerance_deg <= 0.0) return 0;

    const Eigen::VectorXd& predicted_yaw_deg = yaw_second_order_mpc_.lastPredictedYawTrajectoryDeg();
    const Eigen::VectorXd& reference_yaw_deg = yaw_second_order_mpc_.lastReferenceTrajectoryDeg();
    const int horizon = std::min<int>(predicted_yaw_deg.size(), reference_yaw_deg.size());
    if (horizon <= 1) return 0;

    const double model_dt_s = std::max(_params->SECOND_ORDER_CTRL_MODEL_DT_S, 1e-3);
    const double fire_rate_hz = std::max(_params->FIRE_RATE_HZ, 1e-3);
    const int slot_step = std::max(
        1, static_cast<int>(std::lround((1.0 / fire_rate_hz) / model_dt_s)));
    const int first_slot_index = std::clamp(
        static_cast<int>(std::lround(std::max(first_slot_time_s, model_dt_s) / model_dt_s)),
        1, horizon - 1);

    int viable_slot_count = 0;
    for (int index = first_slot_index; index < horizon; index += slot_step) {
        double impact_delta_deg = std::numeric_limits<double>::quiet_NaN();
        bool impact_angle_ok = true;
        if (index >= 0 && index < fire_impact_delta_angle_ref_deg_.size()) {
            impact_delta_deg = fire_impact_delta_angle_ref_deg_(index);
        }
        if (require_impact_angle_gate) {
            impact_angle_ok = fire_impact_angle_ref_in_window(
                fire_impact_delta_angle_ref_deg_, index, target_omega_rad_s, *_params,
                &impact_delta_deg);
        }
        if (index == first_slot_index) {
            if (first_slot_impact_gate != nullptr) {
                *first_slot_impact_gate = impact_angle_ok;
            }
            if (first_slot_impact_delta_deg != nullptr) {
                *first_slot_impact_delta_deg = impact_delta_deg;
            }
        }
        if (!impact_angle_ok) break;

        const double slot_error_deg = shot_slot_window_max_error_deg(
            predicted_yaw_deg, reference_yaw_deg, horizon, model_dt_s, index,
            _params->FIRE_SHOT_WINDOW_PRE_MS, _params->FIRE_SHOT_WINDOW_POST_MS);
        if (!std::isfinite(slot_error_deg)) break;
        if (first_slot_error_deg != nullptr && index == first_slot_index) {
            *first_slot_error_deg = slot_error_deg;
        }
        if (slot_error_deg >= tolerance_deg) break;
        if (!mcu_fire_permit) break;
        if (first_viable_slot_time_s != nullptr && viable_slot_count == 0) {
            *first_viable_slot_time_s = index * model_dt_s;
        }
        viable_slot_count++;
    }
    return viable_slot_count;
}

void FireControl::applyStaticDirectYawControl(bool planner_yaw_valid)
{
    const auto yaw_ctrl_begin = std::chrono::steady_clock::now();
    _controlData.gimbal_yaw = raw_command_yaw_deg_;
    yaw_static_mpc_bypass_active_ = true;
    yaw_planner_active_ = planner_yaw_valid;
    yaw_preview_tracking_error_deg_ = 0.0;
    yaw_preview_tracking_valid_ = false;
    yaw_fire_check_error_deg_ = 0.0;
    yaw_fire_check_valid_ = false;
    yaw_fire_check_index_ = 0;
    yaw_fire_check_time_ms_ = 0.0;

    yaw_second_order_mpc_.reset(_controlData.gimbal_yaw, _yaw_speed);
    last_runtime_stats_.yaw_ctrl_us = elapsed_us(yaw_ctrl_begin);
    last_runtime_stats_.yaw_solve_us = 0.0;
    last_runtime_stats_.planner_active = planner_yaw_valid;
    last_runtime_stats_.preview_mpc_active = false;
}

bool FireControl::tryBuildYawPlan(
    const FireControlTargetSnapshot& target_snapshot, rm::Plan* yaw_plan)
{
    if (yaw_plan == nullptr) return false;
    if (target_snapshot.target_state11d.size() < 11 ||
        target_snapshot.target_state11d.squaredNorm() <= 0.0) {
        return false;
    }

    const auto planner_begin = std::chrono::steady_clock::now();
    rm::Target planner_target(std::max(1, target_snapshot.tracked_armors_num));
    planner_target.sync_state(target_snapshot.target_state11d);
    planner_target.jumped = target_snapshot.target_jumped;
    if (target_snapshot.state_age_s > 1e-6) {
        planner_target.predict(std::min(target_snapshot.state_age_s, 0.2));
    }
    last_plan_execution_delay_s_ = buildYawPlannerConfig().execution_delay_s;
    *yaw_plan = ensureYawPlanner().plan(planner_target, _bulletSpeed);
    last_runtime_stats_.planner_us = elapsed_us(planner_begin);

    const bool planner_yaw_valid =
        yaw_plan->control && yaw_plan->yaw_ref.size() > 0 && std::isfinite(yaw_plan->target_yaw);
    if (!planner_yaw_valid) {
        clearLastYawPlan();
        return false;
    }

    last_yaw_plan_valid_ = true;
    last_plan_selected_armor_index_ = yaw_plan->selected_armor_index;
    last_plan_estimated_fly_time_s_ = yaw_plan->estimated_fly_time_s;
    last_plan_target_yaw_deg_ = normalize_angle_deg(yaw_plan->target_yaw * R2D);
    last_plan_target_yaw_vel_deg_s_ = yaw_plan->target_yaw_vel * R2D;
    last_plan_impact_delta_angle_deg_ = yaw_plan->impact_delta_angle * R2D;
    last_plan_target_pos_ = yaw_plan->target_pos;
    last_plan_zero_vxy_target_pos_ =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    last_plan_zero_vxy_delta_m_ = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd zero_vxy_state = target_snapshot.target_state11d;
    if (zero_vxy_state.size() >= 11) {
        zero_vxy_state(1) = 0.0;
        zero_vxy_state(3) = 0.0;
        rm::Target zero_vxy_target(std::max(1, target_snapshot.tracked_armors_num));
        zero_vxy_target.sync_state(zero_vxy_state);
        if (target_snapshot.state_age_s > 1e-6) {
            zero_vxy_target.predict(std::min(target_snapshot.state_age_s, 0.2));
        }
        if (std::isfinite(yaw_plan->estimated_fly_time_s) &&
            yaw_plan->estimated_fly_time_s > 0.0) {
            zero_vxy_target.predict(yaw_plan->estimated_fly_time_s);
        }
        const auto zero_vxy_armors = zero_vxy_target.armor_xyza_list();
        if (!zero_vxy_armors.empty()) {
            int zero_vxy_index = yaw_plan->selected_armor_index;
            if (zero_vxy_index < 0) zero_vxy_index = 0;
            if (zero_vxy_index >= static_cast<int>(zero_vxy_armors.size())) {
                zero_vxy_index = static_cast<int>(zero_vxy_armors.size()) - 1;
            }
            last_plan_zero_vxy_target_pos_ = zero_vxy_armors[zero_vxy_index].head<3>();
            if (is_finite_vec3(last_plan_target_pos_) &&
                is_finite_vec3(last_plan_zero_vxy_target_pos_)) {
                last_plan_zero_vxy_delta_m_ =
                    (last_plan_target_pos_ - last_plan_zero_vxy_target_pos_).norm();
            }
        }
    }
    base_planner_yaw_deg_ = normalize_angle_deg(yaw_plan->target_yaw * R2D);
    if (_params->AIM_COMMAND_CTRL_MODE == 2) {
        const int horizon = static_cast<int>(yaw_plan->yaw_ref.size());
        const int preview_index = std::clamp(2, 0, std::max(0, horizon - 1));
        yaw_preview_start_index_ = preview_index;
        yaw_preview_end_index_ = preview_index;
    }

    return true;
}

void FireControl::applyYawPlanOrFallback(
    const rm::Plan& yaw_plan, bool planner_yaw_valid, Eigen::Vector3d* pitch_target_pos)
{
    if (planner_yaw_valid) {
        raw_command_yaw_deg_ = normalize_angle_deg(yaw_plan.target_yaw * R2D);
        if (pitch_target_pos != nullptr && is_finite_vec3(yaw_plan.target_pos)) {
            *pitch_target_pos = yaw_plan.target_pos;
            raw_command_pitch_deg_ = gravityOffset(*pitch_target_pos) * R2D;
        }
        if (_params->AIM_COMMAND_CTRL_MODE == 2) {
            const auto yaw_ctrl_begin = std::chrono::steady_clock::now();
            yaw_second_order_mpc_.configure(buildSecondOrderAimConfig(false));
            const Eigen::VectorXd yaw_ref_deg = yaw_plan.yaw_ref * R2D;
            const Eigen::VectorXd yaw_rate_ref_deg_s = yaw_plan.yaw_rate_ref * R2D;
            yaw_second_order_mpc_.setPreviewWindow(
                yaw_preview_start_index_, yaw_preview_end_index_);
            _controlData.gimbal_yaw = yaw_second_order_mpc_.updateTrajectory(
                yaw_ref_deg, yaw_rate_ref_deg_s, _gimbalPose.yaw, _yaw_speed, control_dt_sec_);
            last_runtime_stats_.yaw_ctrl_us = elapsed_us(yaw_ctrl_begin);
            last_runtime_stats_.yaw_solve_us = yaw_second_order_mpc_.lastSolveUs();
            yaw_preview_tracking_error_deg_ = yaw_second_order_mpc_.lastPreviewTrackingErrorDeg();
            yaw_preview_tracking_valid_ = yaw_second_order_mpc_.lastPreviewTrackingValid();
            updateYawFireCheckFromMpc();
            yaw_planner_active_ = true;
            last_runtime_stats_.planner_active = true;
            last_runtime_stats_.preview_mpc_active = true;
            return;
        }

        const auto yaw_ctrl_begin = std::chrono::steady_clock::now();
        _controlData.gimbal_yaw =
            applyAimCommandController(raw_command_yaw_deg_, _gimbalPose.yaw, false);
        last_runtime_stats_.yaw_ctrl_us = elapsed_us(yaw_ctrl_begin);
        if (_params->AIM_COMMAND_CTRL_MODE == 2) {
            last_runtime_stats_.yaw_solve_us = yaw_second_order_mpc_.lastSolveUs();
        }
        return;
    }

    const auto yaw_ctrl_begin = std::chrono::steady_clock::now();
    _controlData.gimbal_yaw =
        applyAimCommandController(raw_command_yaw_deg_, _gimbalPose.yaw, false);
    last_runtime_stats_.yaw_ctrl_us = elapsed_us(yaw_ctrl_begin);
    if (_params->AIM_COMMAND_CTRL_MODE == 2) {
        last_runtime_stats_.yaw_solve_us = yaw_second_order_mpc_.lastSolveUs();
    }
}

void FireControl::appendFireControlDebugLog(const FireControlTargetSnapshot& target_snapshot)
{
    if (!debug_firecontrol_log_enabled_) {
        return;
    }

    auto& sink = firecontrol_debug_log_sink();
    std::lock_guard<std::mutex> lock(sink.mutex);
    if (!sink.enabled || !sink.out.is_open()) return;

    sink.out
        << _nowTime << ','
        << control_dt_sec_ * 1000.0 << ','
        << (last_runtime_stats_.target_detected ? 1 : 0) << ','
        << (yaw_planner_active_ ? 1 : 0) << ','
        << yaw_preview_start_index_ << ','
        << yaw_preview_end_index_ << ','
        << yaw_preview_tracking_error_deg_ << ','
        << (yaw_preview_tracking_valid_ ? 1 : 0) << ','
        << yaw_fire_check_index_ << ','
        << yaw_fire_check_time_ms_ << ','
        << yaw_fire_check_error_deg_ << ','
        << (yaw_fire_check_valid_ ? 1 : 0) << ','
        << _gimbalPose.yaw << ','
        << _yaw_speed << ','
        << legacy_command_yaw_deg_ << ','
        << base_planner_yaw_deg_ << ','
        << _controlData.gimbal_yaw << ','
        << _controlData.gimbal_pitch << ','
        << target_snapshot.state_age_s * 1000.0 << ','
        << static_cast<int>(target_snapshot.movement) << ','
        << static_cast<int>(_controlData.shot_mode) << ','
        << (fire_gate_valid_ ? 1 : 0) << ','
        << (fire_gate_mcu_permit_ ? 1 : 0) << ','
        << (fire_gate_command_stable_ ? 1 : 0) << ','
        << (fire_gate_follow_ ? 1 : 0) << ','
        << (fire_gate_preview_ ? 1 : 0) << ','
        << (fire_gate_impact_angle_ ? 1 : 0) << ','
        << fire_impact_delta_angle_deg_ << ','
        << (fire_gate_slot_window_ ? 1 : 0) << ','
        << (fire_gate_motion_uniform_ ? 1 : 0) << ','
        << (fire_gate_observation_stable_ ? 1 : 0) << ','
        << (fire_burst_active_ ? 1 : 0) << ','
        << static_cast<int>(fire_phase_mode_) << ','
        << fire_next_slot_delay_ms_ << ','
        << (fire_mechanical_hold_active_ ? 1 : 0) << ','
        << fire_viable_slot_count_ << ','
        << fire_first_slot_error_deg_ << ','
        << fire_pitch_cmd_delta_deg_ << ','
        << fire_pitch_follow_error_deg_ << ','
        << target_snapshot.motion_translation_burst_metric << ','
        << target_snapshot.motion_translation_drift_metric
        << '\n';

    static int unflushed_rows = 0;
    if (++unflushed_rows >= 50) {
        sink.out.flush();
        unflushed_rows = 0;
    }
}

const FireControlRuntimeStats& FireControl::lastRuntimeStats() const
{
    return last_runtime_stats_;
}

double FireControl::applyAimCommandController(
    double raw_command_deg, double measured_deg, bool is_pitch)
{
    try {
        // Pitch keeps legacy ballistic command; MPC applies to yaw only.
        if (is_pitch) return raw_command_deg;

        if (_params->AIM_COMMAND_CTRL_MODE == 0) return raw_command_deg;

        if (_params->AIM_COMMAND_CTRL_MODE == 2) {
            const rm::SecondOrderPositionMPCConfig config = buildSecondOrderAimConfig(is_pitch);
            const double measured_rate_deg_s =
                is_pitch ? std::numeric_limits<double>::quiet_NaN() : _yaw_speed;
            auto& controller = is_pitch ? pitch_second_order_mpc_ : yaw_second_order_mpc_;
            controller.configure(config);
            return controller.update(
                raw_command_deg, measured_deg, measured_rate_deg_s, control_dt_sec_);
        }

        return raw_command_deg;
    } catch (...) {
        return raw_command_deg;
    }
}

uint8_t FireControl::decideShotMode(
    const FireControlTargetSnapshot& target_snapshot, bool aim_point_valid)
{
    const auto now = std::chrono::steady_clock::now();
    advanceFirePhase(now);

    const bool static_mpc_bypass =
        yaw_static_mpc_bypass_active_ &&
        static_target_bypasses_second_order_mpc(*_params, target_snapshot.movement);
    fire_gate_valid_ = aim_point_valid && target_snapshot.tracked_armor_valid;
    fire_gate_command_stable_ = false;
    fire_gate_follow_ = false;
    fire_gate_mcu_permit_ = target_snapshot.mcu_fire_permit;
    fire_gate_preview_ = true;
    fire_gate_impact_angle_ = false;
    fire_gate_slot_window_ = false;
    fire_gate_motion_uniform_ = target_snapshot.motion_uniform;
    fire_gate_observation_stable_ = target_snapshot.observation_stable;
    fire_tolerance_deg_ = fire_yaw_tolerance_deg(
        std::numeric_limits<double>::quiet_NaN(), *_params);
    fire_cmd_delta_deg_ = 999.0;
    fire_follow_error_deg_ = 999.0;
    fire_pitch_cmd_delta_deg_ = 999.0;
    fire_pitch_follow_error_deg_ = 999.0;
    fire_viable_slot_count_ = 0;
    fire_first_slot_error_deg_ = std::numeric_limits<double>::quiet_NaN();
    fire_next_slot_delay_ms_ = std::numeric_limits<double>::quiet_NaN();
    fire_mechanical_hold_active_ = false;

    if (target_snapshot.target_state_mat.size() > 2) {
        const double target_distance =
            std::hypot(target_snapshot.target_state_mat(0), target_snapshot.target_state_mat(2));
        fire_tolerance_deg_ = fire_yaw_tolerance_deg(target_distance, *_params);
    }

    const double target_omega_rad_s =
        target_snapshot.target_state11d.size() > 7 ? target_snapshot.target_state11d(7) : 0.0;

    if (has_last_auto_command_) {
        fire_cmd_delta_deg_ = angle_diff_deg(_controlData.gimbal_yaw, last_auto_command_.gimbal_yaw);
        fire_follow_error_deg_ = angle_diff_deg(_gimbalPose.yaw, last_auto_command_.gimbal_yaw);
        fire_pitch_cmd_delta_deg_ =
            angle_diff_deg(_controlData.gimbal_pitch, last_auto_command_.gimbal_pitch);
        fire_pitch_follow_error_deg_ =
            angle_diff_deg(_gimbalPose.pitch, last_auto_command_.gimbal_pitch);
        fire_gate_command_stable_ =
            fire_cmd_delta_deg_ < fire_tolerance_deg_ * _params->FIRE_COMMAND_STABLE_RATIO &&
            fire_pitch_cmd_delta_deg_ < fire_tolerance_deg_ * _params->FIRE_COMMAND_STABLE_RATIO;
        fire_gate_follow_ =
            fire_follow_error_deg_ < fire_tolerance_deg_ &&
            fire_pitch_follow_error_deg_ < fire_tolerance_deg_;
    }

    if (static_mpc_bypass) {
        yaw_fire_check_error_deg_ = angle_diff_deg(_controlData.gimbal_yaw, _gimbalPose.yaw);
        yaw_fire_check_valid_ = std::isfinite(yaw_fire_check_error_deg_);
        fire_follow_error_deg_ = yaw_fire_check_error_deg_;
        fire_pitch_follow_error_deg_ =
            angle_diff_deg(_controlData.gimbal_pitch, _gimbalPose.pitch);
        fire_cmd_delta_deg_ = yaw_fire_check_error_deg_;
        fire_pitch_cmd_delta_deg_ = fire_pitch_follow_error_deg_;
        const bool current_error_ok =
            yaw_fire_check_valid_ && yaw_fire_check_error_deg_ < fire_tolerance_deg_ &&
            std::isfinite(fire_pitch_follow_error_deg_) &&
            fire_pitch_follow_error_deg_ < fire_tolerance_deg_;
        yaw_fire_check_index_ = 0;
        yaw_fire_check_time_ms_ = 0.0;
        fire_gate_command_stable_ = current_error_ok;
        fire_gate_follow_ = current_error_ok;
        fire_gate_preview_ = current_error_ok;
    } else if (yaw_fire_check_valid_) {
        fire_gate_preview_ = yaw_fire_check_error_deg_ < fire_tolerance_deg_;
    } else if (yaw_preview_tracking_valid_) {
        fire_gate_preview_ = yaw_preview_tracking_error_deg_ < fire_tolerance_deg_;
    }

    const bool auto_fire_enabled = _params->AUTO_SHOT_SWITCH != 0;
    const int enter_slot_count = std::max(1, _params->FIRE_AUTO_ENTER_SLOT_COUNT);
    const int hold_slot_count = std::max(1, _params->FIRE_AUTO_HOLD_SLOT_COUNT);
    const int static_ready_slot_count = std::max(enter_slot_count, hold_slot_count);
    const double first_slot_time_s = firePhaseFirstSlotTimeS(now);
    const bool require_impact_angle_gate =
        !static_mpc_bypass && std::abs(target_omega_rad_s) > kFireImpactDirectionEpsilonRadS;
    fire_next_slot_delay_ms_ = first_slot_time_s * 1000.0;

    if (static_mpc_bypass) {
        const int impact_horizon = static_cast<int>(fire_impact_delta_angle_ref_deg_.size());
        const double model_dt_s = std::max(_params->SECOND_ORDER_CTRL_MODEL_DT_S, 1e-3);
        const int first_slot_index =
            impact_horizon > 0
                ? std::clamp(
                      static_cast<int>(
                          std::lround(std::max(first_slot_time_s, model_dt_s) / model_dt_s)),
                      0, impact_horizon - 1)
                : -1;
        fire_gate_impact_angle_ = true;
        fire_impact_delta_angle_deg_ =
            first_slot_index >= 0
                ? fire_impact_delta_angle_ref_deg_(first_slot_index)
                : std::numeric_limits<double>::quiet_NaN();
        fire_first_slot_error_deg_ =
            yaw_fire_check_valid_ ? yaw_fire_check_error_deg_
                                  : std::numeric_limits<double>::quiet_NaN();
        fire_viable_slot_count_ =
            fire_gate_preview_ && fire_gate_follow_ && fire_gate_mcu_permit_
                ? static_ready_slot_count
                : 0;
    } else {
        bool first_slot_impact_gate = false;
        double first_slot_impact_delta_deg =
            std::numeric_limits<double>::quiet_NaN();
        fire_viable_slot_count_ = countViableShotSlots(
            fire_tolerance_deg_, first_slot_time_s, target_omega_rad_s,
            require_impact_angle_gate, fire_gate_mcu_permit_,
            &fire_first_slot_error_deg_, nullptr,
            &first_slot_impact_gate, &first_slot_impact_delta_deg);
        fire_gate_impact_angle_ =
            !require_impact_angle_gate ||
            (fire_impact_delta_angle_valid_ && first_slot_impact_gate);
        fire_impact_delta_angle_deg_ = first_slot_impact_delta_deg;
        fire_gate_preview_ =
            std::isfinite(fire_first_slot_error_deg_) &&
            fire_first_slot_error_deg_ < fire_tolerance_deg_;
    }
    fire_gate_slot_window_ = fire_viable_slot_count_ >= hold_slot_count;

    const bool hard_gate_ok = auto_fire_enabled && fire_gate_valid_;
    const bool mechanical_hold_ready =
        fire_phase_mode_ == FirePhaseMode::Auto && now < fire_burst_hold_deadline_;
    const bool auto_restart_cooling_down =
        now < fire_auto_restart_cooldown_until_;
    fire_mechanical_hold_active_ =
        mechanical_hold_ready && hard_gate_ok && fire_viable_slot_count_ < hold_slot_count;
    const bool burst_enter_ready = fire_viable_slot_count_ >= enter_slot_count;
    const bool burst_hold_ready = fire_viable_slot_count_ >= hold_slot_count;
    const bool single_window_ready = fire_viable_slot_count_ == 1;
    const auto auto_hold_deadline_from_next_slot =
        [&]() -> std::chrono::steady_clock::time_point {
            const double hold_ms = std::max(0.0, _params->FIRE_AUTO_MIN_BURST_MS);
            const auto hold_duration =
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double, std::milli>(hold_ms));
            const auto committed_slot_tp =
                fire_next_slot_tp_ == std::chrono::steady_clock::time_point{}
                    ? now
                    : fire_next_slot_tp_;
            return committed_slot_tp + hold_duration;
        };

    uint8_t shot_mode = ControlData::SHOT_MODE::AIM_ONLY;
    // AUTO_FIRE has mechanical inertia: do not retract it before one拨盘 step can finish.
    if (!hard_gate_ok) {
        clearFirePhaseState();
    } else if (fire_phase_mode_ == FirePhaseMode::Auto) {
        if (burst_hold_ready) {
            fire_burst_active_ = true;
            fire_burst_hold_deadline_ = auto_hold_deadline_from_next_slot();
            shot_mode = ControlData::SHOT_MODE::AUTO_FIRE;
        } else if (mechanical_hold_ready) {
            fire_burst_active_ = true;
            fire_mechanical_hold_active_ = true;
            shot_mode = ControlData::SHOT_MODE::AUTO_FIRE;
        } else {
            clearFirePhaseState();
            const double cooldown_ms =
                std::max(0.0, _params->FIRE_AUTO_RESTART_COOLDOWN_MS);
            fire_auto_restart_cooldown_until_ =
                now + std::chrono::milliseconds(
                          static_cast<int>(std::lround(cooldown_ms)));
        }
    } else if (fire_phase_mode_ == FirePhaseMode::Single) {
        if (!fire_single_pulse_sent_) {
            fire_single_pulse_sent_ = true;
            shot_mode = ControlData::SHOT_MODE::SHOT_ONCE;
        }
    } else if (auto_restart_cooling_down) {
        fire_next_slot_delay_ms_ =
            std::chrono::duration<double, std::milli>(
                fire_auto_restart_cooldown_until_ - now)
                .count();
    } else if (burst_enter_ready) {
        startFirePhase(FirePhaseMode::Auto, now, first_slot_time_s);
        fire_burst_hold_deadline_ = auto_hold_deadline_from_next_slot();
        shot_mode = ControlData::SHOT_MODE::AUTO_FIRE;
    } else if (single_window_ready) {
        startFirePhase(FirePhaseMode::Single, now, first_slot_time_s);
        fire_single_pulse_sent_ = true;
        shot_mode = ControlData::SHOT_MODE::SHOT_ONCE;
    } else {
        clearFirePhaseState();
    }
    if (fire_phase_mode_ != FirePhaseMode::None) {
        fire_next_slot_delay_ms_ = firePhaseFirstSlotTimeS(now) * 1000.0;
    } else if (hard_gate_ok) {
        fire_next_slot_delay_ms_ = first_slot_time_s * 1000.0;
    }

    last_auto_command_ = _controlData;
    has_last_auto_command_ = true;
    return shot_mode;
}

ControlData FireControl::calControlData(rm::AngleSolver& angleSolver, rm::Estimator& estimator)
{
    FireControlTargetSnapshot snapshot = makeTargetSnapshot(estimator, simulator_state_age_s_);
    snapshot.mcu_fire_permit = mcu_fire_permit_;
    return calControlData(angleSolver, snapshot);
}

ControlData FireControl::calControlData(
    rm::AngleSolver& angleSolver, const FireControlTargetSnapshot& target_snapshot)
{
    const auto total_begin = std::chrono::steady_clock::now();
    last_runtime_stats_ = {};
    last_runtime_stats_.wall_control_dt_ms = wall_control_dt_sec_ * 1000.0;
    last_runtime_stats_.frame_control_dt_ms = frame_control_dt_sec_ * 1000.0;
    last_runtime_stats_.control_dt_ms = control_dt_sec_ * 1000.0;
    last_runtime_stats_.target_detected = target_snapshot.detected_flag;
    if (!target_snapshot.detected_flag) {
        applyNoTargetControlState();
    } else {
        cv::Vec2f aim_point_on_image = angleSolver.calculateImagePoint(target_snapshot.aim_point);
        cv::Vec2f aim_point_angle = angleSolver.calculateAngleDev(aim_point_on_image);
        const double legacy_yaw_command_deg = _gimbalPose.yaw + aim_point_angle[0] * R2D;
        resetPlannerTrackingState(legacy_yaw_command_deg);
        yaw_dev_d = aim_point_angle[0] * R2D;

        if (target_snapshot.tracked_armor_valid) {
            _controlData.yaw_error = angleSolver.calculateAngleDev(
                angleSolver.calculateImagePoint(target_snapshot.tracked_armor_position))[0];
        } else {
            _controlData.yaw_error = 0;
        }
        raw_command_yaw_deg_ = legacy_yaw_command_deg;
        Eigen::Vector3d pitch_target_pos = target_snapshot.aim_point;
        raw_command_pitch_deg_ = gravityOffset(pitch_target_pos) * R2D;

        const bool bypass_mpc_for_static =
            static_target_bypasses_second_order_mpc(*_params, target_snapshot.movement);
        rm::Plan yaw_plan;
        const bool planner_yaw_valid = tryBuildYawPlan(target_snapshot, &yaw_plan);
        fire_impact_delta_angle_valid_ =
            planner_yaw_valid && yaw_plan.impact_delta_angle_valid &&
            yaw_plan.impact_delta_angle_ref.size() > 0;
        if (fire_impact_delta_angle_valid_) {
            fire_impact_delta_angle_ref_deg_ = yaw_plan.impact_delta_angle_ref * R2D;
            fire_impact_delta_angle_deg_ = fire_impact_delta_angle_ref_deg_(0);
        } else {
            fire_impact_delta_angle_ref_deg_.resize(0);
            fire_impact_delta_angle_deg_ = std::numeric_limits<double>::quiet_NaN();
        }
        if (bypass_mpc_for_static) {
            if (planner_yaw_valid) {
                raw_command_yaw_deg_ = normalize_angle_deg(yaw_plan.target_yaw * R2D);
                if (is_finite_vec3(yaw_plan.target_pos)) {
                    pitch_target_pos = yaw_plan.target_pos;
                    raw_command_pitch_deg_ = gravityOffset(pitch_target_pos) * R2D;
                }
            }
            applyStaticDirectYawControl(planner_yaw_valid);
        } else {
            applyYawPlanOrFallback(yaw_plan, planner_yaw_valid, &pitch_target_pos);
        }

        yaw_dev_d = normalize_angle_deg(_controlData.gimbal_yaw - _gimbalPose.yaw);

        const auto pitch_ctrl_begin = std::chrono::steady_clock::now();
        _controlData.gimbal_pitch =
            applyAimCommandController(raw_command_pitch_deg_, _gimbalPose.pitch, true);
        last_runtime_stats_.pitch_ctrl_us = elapsed_us(pitch_ctrl_begin);
        last_runtime_stats_.pitch_solve_us = 0.0;
        filtered_command_yaw_deg_ = _controlData.gimbal_yaw;
        filtered_command_pitch_deg_ = _controlData.gimbal_pitch;

        const bool aim_point_valid =
            is_finite_vec3(pitch_target_pos) && std::isfinite(_controlData.gimbal_yaw) &&
            std::isfinite(_controlData.gimbal_pitch);
        _controlData.shot_mode = decideShotMode(target_snapshot, aim_point_valid);

        _controlData.aiming_state = ControlData::AIMING_STATE::TARGET_DETECTED;
    }
    appendFireControlDebugLog(target_snapshot);
    last_runtime_stats_.total_us = elapsed_us(total_begin);
    return _controlData;
}

void FireControl::showResults()
{
    if (!_params->DEBUG_SWITCH) return;
    string str = "[" + to_string(_controlData.gimbal_pitch) + " " + to_string(_controlData.gimbal_yaw) +
                 "]  " + shotModeStr[_controlData.shot_mode];
    string fire_dbg =
        "fire tol=" + to_string(fire_tolerance_deg_).substr(0, 4) +
        " cmd=" + to_string(fire_cmd_delta_deg_).substr(0, 4) +
        " pcmd=" + to_string(fire_pitch_cmd_delta_deg_).substr(0, 4) +
        " m50=" + to_string(yaw_fire_check_error_deg_).substr(0, 4) +
        " sloterr=" + to_string(fire_first_slot_error_deg_).substr(0, 4) +
        " impact=" + to_string(fire_impact_delta_angle_deg_).substr(0, 5) +
        " slot=" + to_string(fire_viable_slot_count_) +
        " next=" + to_string(fire_next_slot_delay_ms_).substr(0, 4) +
        " phase=" + fire_phase_mode_name(fire_phase_mode_) +
        " follow=" + to_string(fire_follow_error_deg_).substr(0, 4) +
        " pfollow=" + to_string(fire_pitch_follow_error_deg_).substr(0, 4) +
        " gate=" + string(fire_gate_valid_ ? "V" : "x") +
        string(fire_gate_mcu_permit_ ? "U" : "x") +
        string(fire_gate_command_stable_ ? "C" : "x") +
        string(fire_gate_follow_ ? "F" : "x") +
        string(fire_gate_preview_ ? "P" : "x") +
        string(fire_gate_impact_angle_ ? "A" : "x") +
        string(fire_gate_slot_window_ ? "S" : "x") +
        string(fire_gate_motion_uniform_ ? "M" : "x") +
        string(fire_gate_observation_stable_ ? "O" : "x") +
        string(fire_burst_active_ ? "B" : "x") +
        string(fire_mechanical_hold_active_ ? "H" : "x");
    
    string ctrl_dbg =
        "ctrl yaw=" + string(yaw_static_mpc_bypass_active_ ? "static-direct" :
            (yaw_planner_active_ ? "preview2" :
             aim_command_ctrl_mode_name(_params->AIM_COMMAND_CTRL_MODE))) +
        " pitch=legacy" +
        " yaw " + to_string(raw_command_yaw_deg_).substr(0, 5) + "->" +
        to_string(filtered_command_yaw_deg_).substr(0, 5) + " pitch " +
        to_string(raw_command_pitch_deg_).substr(0, 5) + "->" +
        to_string(filtered_command_pitch_deg_).substr(0, 5);
    const string bullet_dbg = "bullet: " + to_string(_bulletSpeed).substr(0, 4);

    if (_debugHud != nullptr) {
        _debugHud->upsert("firecontrol.command", str, "top_left", 30, "#66ff66");
        _debugHud->upsert("firecontrol.fire_gate", fire_dbg, "top_left", 31, "#66ff66");
        _debugHud->upsert("firecontrol.controller", ctrl_dbg, "top_left", 32, "#66ff66");
        _debugHud->upsert("firecontrol.bullet", bullet_dbg, "bottom_left", 30, "#c77dff");

        const double sample_time_ms =
            std::isfinite(_nowTime) && _nowTime > 0.0
                ? _nowTime
                : std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
        _debugHud->addSample(
            "yaw.cmd", "yaw", "cmd", _controlData.gimbal_yaw, sample_time_ms, "deg",
            "#38bdf8");
        _debugHud->addSample(
            "yaw.feedback", "yaw", "feedback", _gimbalPose.yaw, sample_time_ms, "deg",
            "#fbbf24");
        _debugHud->addSample(
            "pitch.cmd", "pitch", "cmd", _controlData.gimbal_pitch, sample_time_ms, "deg",
            "#38bdf8");
        _debugHud->addSample(
            "pitch.feedback", "pitch", "feedback", _gimbalPose.pitch, sample_time_ms, "deg",
            "#fbbf24");

        const bool auto_fire = _controlData.shot_mode == ControlData::SHOT_MODE::AUTO_FIRE;
        const bool shot_once = _controlData.shot_mode == ControlData::SHOT_MODE::SHOT_ONCE;
        _debugHud->addSample(
            "fire.auto_fire", "fire command", "AUTO_FIRE", auto_fire ? 1.0 : 0.0,
            sample_time_ms, "", "#22c55e");
        _debugHud->addSample(
            "fire.shot_once", "fire command", "SHOT_ONCE", shot_once ? 1.0 : 0.0,
            sample_time_ms, "", "#f59e0b");
        return;
    }
    putText(_debugImg, fire_dbg, Point(25, 45), FONT_HERSHEY_PLAIN, 1.2, cvex::GREEN, 1);
    putText(_debugImg, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.5, cvex::GREEN, 1);
    putText(_debugImg, fire_dbg, Point(25, 45), FONT_HERSHEY_PLAIN, 1.2, cvex::GREEN, 1);
    putText(_debugImg, ctrl_dbg, Point(25, 65), FONT_HERSHEY_PLAIN, 1.2, cvex::GREEN, 1);
    putText(_debugImg, bullet_dbg, Point(10, 240), FONT_HERSHEY_PLAIN, 2, PURPLE);
}

double FireControl::gravityOffset(double theta, double euclideanDistance)
{
    double v = _bulletSpeed;

    double d = euclideanDistance * 10;
    double temp = d - floor(d);
    if (temp >= 0.5) {
        d = floor(d) + 1;
    }
    if (temp < 0.5) {
        d = floor(d);
    }

    euclideanDistance = d / 10;
    double g = 9.8;
    double x = euclideanDistance * cos(theta);
    double y = euclideanDistance * sin(theta);
    double _theta =
        atan(-(v * v / g / x / x) * (-x + sqrt(x * x - 2 * (g * x * x / v / v) * (g * x * x / 2 / v / v + y))));

    if (x * x - 2 * (g * x * x / v / v) * (g * x * x / 2 / v / v + y) <= 0) {
        return 0;
    }

    return _theta - theta;
}

double FireControl::gravityOffset(Eigen::Vector3d pos)
{
    double x = norm(Vec2d(pos[0], pos[1]));
    double y = pos[2];

    if (!_params->GRAVITY_OFFSET_SWITCH)
        return atan(y / x);

    tools::Trajectory trajectory(_bulletSpeed, x, y);
    if (trajectory.unsolvable) {
        return atan(y / x);
    }

    return trajectory.pitch;
}

bool FireControl::intervalFiring(double time)
{
    static bool shot_flag = 0;
    if (!shot_flag) {
        shot_flag = !shot_flag;
        timer.tic();
    }

    double delta = timer.toc();

    if (delta < time * 1000) {
        return false;
    } else {
        timer.tic();
        return true;
    }
}

double FireControl::getDistance(Eigen::Vector3d p)
{
    return sqrt(pow(p.x(), 2) + pow(p.y(), 2) + pow(p.z(), 2));
}

double FireControl::outpost(double now_time)
{
    now_time = _nowTime - now_time;
    int end = outpostQ.size() - 1;
    if (now_time <= outpostQ[0].t) return outpostQ[end].t;
    for (size_t i = 0; i < outpostQ.size(); i++) {
        if (now_time > outpostQ[end - i].t) {
            if (direction * int(outpostQ[end - i + 1].angle >= outpostQ[end - i].angle) > 0) {
                return (outpostQ[end - i + 1].angle + outpostQ[end - i].angle) / 2;
            } else {
                _controlData.shot_mode = ControlData::SHOT_MODE::AIM_ONLY;
                return (outpostQ[end - i + 1].angle + outpostQ[end - i].angle) / 2;
            }
        }
    }
    return _controlData.gimbal_yaw;
}

} // namespace rm
