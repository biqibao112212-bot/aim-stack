// Copyright 2022 Chen Jun

#include "robotestimator.h"
#include "extendedkalmanfilter.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <iostream>
#include <limits>
#include <numeric>  
#include <sstream>
#include <string>

using namespace std;

namespace rm
{

static inline double normalize_angle(double angle)
{
    const double result = fmod(angle + M_PI, 2.0 * M_PI);
    if (result <= 0.0) return result + M_PI;
    return result - M_PI;
}

namespace {

struct AimChoice {
    bool valid = false;
    Eigen::Vector4d xyza = Eigen::Vector4d::Zero();
};

struct LegacyTrackerLogFields {
    double xc = std::numeric_limits<double>::quiet_NaN();
    double vxc = std::numeric_limits<double>::quiet_NaN();
    double yc = std::numeric_limits<double>::quiet_NaN();
    double vyc = std::numeric_limits<double>::quiet_NaN();
    double z = std::numeric_limits<double>::quiet_NaN();
    double vz = std::numeric_limits<double>::quiet_NaN();
    double yaw = std::numeric_limits<double>::quiet_NaN();
    double vyaw = std::numeric_limits<double>::quiet_NaN();
    double r = std::numeric_limits<double>::quiet_NaN();
};

struct MatchedArmorBatch {
    std::vector<std::shared_ptr<rm::Armor>> armor_ptrs;
    std::vector<RobotEstimator::ArmorObservation> observations;
    std::vector<rm::Armor> tracker_input_armors;
    std::shared_ptr<rm::Armor> primary_armor_ptr = nullptr;
    int primary_observation_index = -1;
};

constexpr int kArmorColorBlue = 0;
constexpr int kArmorColorRed = 1;
constexpr int kArmorColorGray = 2;
constexpr int kArmorColorPurple = 3;

bool is_neutral_armor_color(int color)
{
    return color == kArmorColorGray || color == kArmorColorPurple;
}

bool is_team_armor_color(int color)
{
    return color == kArmorColorBlue || color == kArmorColorRed;
}

constexpr int kMode1SelectionLogSlots = 4;
bool env_flag_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return false;
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" &&
           text != "OFF";
}

std::filesystem::path resolve_mode1_selection_log_path()
{
    namespace fs = std::filesystem;

    if (const char* env_path = std::getenv("AIM_SIM_MODE1_SELECTION_CSV");
        env_path != nullptr && env_path[0] != '\0') {
        return fs::path(env_path);
    }

    const fs::path cwd = fs::current_path();
    if (cwd.filename() == "build") {
        return cwd / "mode1_selection_debug.csv";
    }

    const fs::path build_dir = cwd / "build";
    if (fs::exists(build_dir) && fs::is_directory(build_dir)) {
        return build_dir / "mode1_selection_debug.csv";
    }

    return cwd / "mode1_selection_debug.csv";
}

Eigen::VectorXd covariance_diag_or_zero(const Eigen::MatrixXd& covariance)
{
    Eigen::VectorXd diag = Eigen::VectorXd::Zero(11);
    const Eigen::Index n = std::min<Eigen::Index>(diag.size(), covariance.rows());
    for (Eigen::Index i = 0; i < n; ++i) {
        if (i < covariance.cols()) diag(i) = covariance(i, i);
    }
    return diag;
}

double state_value_or_nan(const Eigen::VectorXd& state, Eigen::Index index)
{
    if (index < 0 || index >= state.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return state(index);
}

void write_state_key_fields(std::ofstream& out, const Eigen::VectorXd& state)
{
    out << state_value_or_nan(state, 0) << ","
        << state_value_or_nan(state, 1) << ","
        << state_value_or_nan(state, 2) << ","
        << state_value_or_nan(state, 3) << ","
        << state_value_or_nan(state, 6) << ","
        << state_value_or_nan(state, 7) << ","
        << state_value_or_nan(state, 8) << ","
        << state_value_or_nan(state, 9);
}

RobotEstimator::YpdAngleTracker::GeometryRecoveryConfig make_ypd_recovery_config(
    const Params& params)
{
    RobotEstimator::YpdAngleTracker::GeometryRecoveryConfig config;
    config.recovery_window_frames = params.YPD_GEOMETRY_RECOVERY_WINDOW_FRAMES;
    config.recovery_cooldown_frames = params.YPD_GEOMETRY_RECOVERY_COOLDOWN_FRAMES;
    config.mismatch_required_streak =
        params.YPD_GEOMETRY_RECOVERY_MISMATCH_REQUIRED_STREAK;
    config.min_matched_count = params.YPD_GEOMETRY_RECOVERY_MIN_MATCHED_COUNT;
    config.residual_z_sigma_threshold = params.YPD_GEOMETRY_RECOVERY_Z_SIGMA_THRESHOLD;
    config.residual_xy_sigma_threshold = params.YPD_GEOMETRY_RECOVERY_XY_SIGMA_THRESHOLD;
    config.covariance_inflation_scale = params.YPD_GEOMETRY_RECOVERY_COV_INFLATION_SCALE;
    config.min_dr_variance = params.YPD_GEOMETRY_RECOVERY_MIN_DR_VARIANCE;
    config.min_h_variance = params.YPD_GEOMETRY_RECOVERY_MIN_H_VARIANCE;
    return config;
}

void apply_ypd_recovery_config(
    const std::shared_ptr<RobotEstimator::YpdAngleTracker>& tracker, const Params& params)
{
    if (!tracker) return;
    tracker->setGeometryRecoveryConfig(make_ypd_recovery_config(params));
}

using MatchCostMatrix =
    std::array<std::array<double, kMode1SelectionLogSlots>, kMode1SelectionLogSlots>;

MatchCostMatrix build_match_cost_matrix(const std::vector<Armor>& observations,
    const std::vector<Eigen::Vector4d>& predicted_armors)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    MatchCostMatrix cost_matrix;
    for (auto& row : cost_matrix) {
        row.fill(nan);
    }

    const int obs_limit =
        std::min(kMode1SelectionLogSlots, static_cast<int>(observations.size()));
    const int pred_limit =
        std::min(kMode1SelectionLogSlots, static_cast<int>(predicted_armors.size()));

    for (int obs_index = 0; obs_index < obs_limit; ++obs_index) {
        const Armor& obs_armor = observations[obs_index];
        const double obs_camera_yaw =
            std::atan2(obs_armor.armorPosition.y(), obs_armor.armorPosition.x());

        for (int armor_id = 0; armor_id < pred_limit; ++armor_id) {
            const Eigen::Vector4d& pred_xyza = predicted_armors[armor_id];
            const double pred_camera_yaw = std::atan2(pred_xyza(1), pred_xyza(0));
            cost_matrix[obs_index][armor_id] =
                std::abs(normalize_angle(obs_armor.yaw - pred_xyza(3))) +
                std::abs(normalize_angle(obs_camera_yaw - pred_camera_yaw));
        }
    }

    return cost_matrix;
}

double compute_ambiguity_index(
    const MatchCostMatrix& cost_matrix, int obs_count, int pred_count)
{
    std::vector<double> best_values;
    double min_cost_gap = std::numeric_limits<double>::quiet_NaN();
    const int obs_limit = std::min(kMode1SelectionLogSlots, obs_count);
    const int pred_limit = std::min(kMode1SelectionLogSlots, pred_count);

    for (int obs_index = 0; obs_index < obs_limit; ++obs_index) {
        std::vector<double> costs;
        costs.reserve(pred_limit);
        for (int armor_id = 0; armor_id < pred_limit; ++armor_id) {
            const double cost = cost_matrix[obs_index][armor_id];
            if (std::isfinite(cost)) {
                costs.push_back(cost);
            }
        }

        if (costs.empty()) {
            continue;
        }

        std::sort(costs.begin(), costs.end());
        best_values.push_back(costs.front());

        if (costs.size() < 2) {
            continue;
        }

        const double cost_gap = costs[1] - costs[0];
        if (!std::isfinite(min_cost_gap) || cost_gap < min_cost_gap) {
            min_cost_gap = cost_gap;
        }
    }

    if (best_values.empty() || !std::isfinite(min_cost_gap) || min_cost_gap <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double mean_best_cost =
        std::accumulate(best_values.begin(), best_values.end(), 0.0) / best_values.size();
    return mean_best_cost / min_cost_gap;
}

bool armor_observation_less(
    const std::shared_ptr<rm::Armor>& lhs, const std::shared_ptr<rm::Armor>& rhs)
{
    if (!lhs) return static_cast<bool>(rhs);
    if (!rhs) return false;

    if (lhs->distanceToImageCenter != rhs->distanceToImageCenter) {
        return lhs->distanceToImageCenter < rhs->distanceToImageCenter;
    }
    if (lhs->number != rhs->number) {
        return lhs->number < rhs->number;
    }
    if (lhs->type != rhs->type) {
        return static_cast<int>(lhs->type) < static_cast<int>(rhs->type);
    }
    if (lhs->center.x != rhs->center.x) {
        return lhs->center.x < rhs->center.x;
    }
    return lhs->center.y < rhs->center.y;
}

bool armor_matches_tracked_target(const std::shared_ptr<rm::Armor>& armor,
    const std::shared_ptr<rm::Armor>& tracked_armor, bool ignore_same_number)
{
    if (!armor) return false;
    if (!tracked_armor) return true;
    if (armor->type != tracked_armor->type) return false;
    if (!ignore_same_number && armor->number != tracked_armor->number) return false;
    return true;
}

Eigen::VectorXd legacy_state_from_tracker(
    const Eigen::VectorXd& state11d, const Eigen::Vector4d& armor_xyza, double radius)
{
    Eigen::VectorXd state9 = Eigen::VectorXd::Zero(9);
    if (state11d.size() < 11) return state9;

    state9(EKF_CENTER_X) = state11d(0);
    state9(EKF_CENTER_V_X) = state11d(1);
    state9(EKF_CENTER_Y) = state11d(2);
    state9(EKF_CENTER_V_Y) = state11d(3);
    state9(EKF_ARMOR_Z) = armor_xyza(2);
    state9(EKF_ARMOR_V_Z) = state11d(5);
    state9(EKF_ARMOR_YAW) = armor_xyza(3);
    state9(EKF_ARMOR_V_YAW) = state11d(7);
    state9(EKF_ROTATION_RADIUS) = radius;
    return state9;
}

template <typename TrackerT>
void sync_robot_msg_from_tracker(RobotMsg& robot, const TrackerT& tracker)
{
    const auto state11d = tracker.getState();
    const auto armor_xyza_list = tracker.getPredictedArmorStates();
    const int armor_count = static_cast<int>(armor_xyza_list.size());

    robot.idx = static_cast<size_t>(std::max(0, tracker.getTrackedId()));
    robot.direction = state11d.size() > 7 ? state11d(7) : 0.0;

    for (int i = 0; i < 4; ++i) {
        if (i < armor_count) {
            robot.determined[i] = true;
            robot.armor_zs[i] = armor_xyza_list[i](2);
            robot.armor_rs[i] = tracker.getArmorRadius(i);
        } else {
            robot.determined[i] = false;
            robot.armor_zs[i] = 0.0;
            robot.armor_rs[i] = 0.0;
        }
    }
}

template <typename TrackerT>
void refresh_tracker_snapshot(const std::shared_ptr<TrackerT>& tracker, Eigen::VectorXd& state11d,
    Eigen::VectorXd& state9)
{
    state11d = Eigen::VectorXd::Zero(11);
    state9 = Eigen::VectorXd::Zero(9);
    if (!tracker || !tracker->isInitialized()) return;

    state11d = tracker->getState();
    const int tracked_id =
        std::clamp(tracker->getTrackedId(), 0, std::max(0, tracker->getArmorNum() - 1));
    const Eigen::Vector4d armor_xyza = tracker->getPredictedArmorState(tracked_id);
    state9 = legacy_state_from_tracker(state11d, armor_xyza, tracker->getArmorRadius(tracked_id));
}

LegacyTrackerLogFields extract_log_fields(
    const Eigen::VectorXd& state9, const Eigen::VectorXd& state11d)
{
    LegacyTrackerLogFields fields;
    if (state9.size() < 9) return fields;
    if (state9.squaredNorm() <= 0.0 && state11d.squaredNorm() <= 0.0) return fields;

    fields.xc = state9(EKF_CENTER_X);
    fields.vxc = state9(EKF_CENTER_V_X);
    fields.yc = state9(EKF_CENTER_Y);
    fields.vyc = state9(EKF_CENTER_V_Y);
    fields.z = state11d.size() >= 11 ? state11d(4) : state9(EKF_ARMOR_Z);
    fields.vz = state9(EKF_ARMOR_V_Z);
    fields.yaw = state9(EKF_ARMOR_YAW);
    fields.vyaw = state9(EKF_ARMOR_V_YAW);
    fields.r = state9(EKF_ROTATION_RADIUS);
    return fields;
}

std::vector<rm::Armor> copy_tracker_armors(
    const std::vector<std::shared_ptr<rm::Armor>>& matched_armors)
{
    std::vector<rm::Armor> tracker_armors;
    tracker_armors.reserve(matched_armors.size());
    for (const auto& armor_ptr : matched_armors) {
        tracker_armors.push_back(*armor_ptr);
    }
    return tracker_armors;
}

void normalize_tracker_armors_as_outpost(std::vector<rm::Armor>& armors)
{
    for (auto& armor : armors) {
        armor.number = rm::Armor::LABEL::OUTPOST;
    }
}

Eigen::VectorXd predict_legacy_target_state(const Eigen::VectorXd& current_state, double dt)
{
    Eigen::VectorXd predicted_state = current_state;
    if (predicted_state.size() < 9) return predicted_state;

    predicted_state(EKF_CENTER_X) += predicted_state(EKF_CENTER_V_X) * dt;
    predicted_state(EKF_CENTER_Y) += predicted_state(EKF_CENTER_V_Y) * dt;
    predicted_state(EKF_ARMOR_Z) += predicted_state(EKF_ARMOR_V_Z) * dt;
    predicted_state(EKF_ARMOR_YAW) += predicted_state(EKF_ARMOR_V_YAW) * dt;
    return predicted_state;
}

MatchedArmorBatch collect_matching_armors(const std::vector<std::shared_ptr<rm::Armor>>& sorted_armors,
    const std::shared_ptr<rm::Armor>& tracked_armor, bool ignore_same_number)
{
    MatchedArmorBatch batch;
    for (const auto& armor : sorted_armors) {
        if (!armor) continue;
        if (std::isnan(armor->armorPosition.x()) || std::isnan(armor->armorPosition.y())) continue;
        if (!armor_matches_tracked_target(armor, tracked_armor, ignore_same_number)) continue;

        RobotEstimator::ArmorObservation obs;
        obs.pos = armor->armorPosition;
        obs.yaw = armor->yaw;
        batch.observations.push_back(obs);
        batch.armor_ptrs.push_back(armor);
    }

    batch.tracker_input_armors = copy_tracker_armors(batch.armor_ptrs);
    if (!batch.armor_ptrs.empty()) {
        batch.primary_observation_index = 0;
        batch.primary_armor_ptr = batch.armor_ptrs.front();
    }
    return batch;
}

Eigen::Vector3d xyz_to_ypd(const Eigen::Vector3d& xyz)
{
    const double x = xyz.x();
    const double y = xyz.y();
    const double z = xyz.z();
    const double yaw = std::atan2(y, x);
    const double pitch = std::atan2(z, std::sqrt(x * x + y * y));
    const double distance = std::sqrt(x * x + y * y + z * z);
    return {yaw, pitch, distance};
}

Eigen::Vector3d observation_ypd_for_log(const rm::Armor& armor)
{
    if (armor.has_explicit_ypd && armor.ypd.allFinite()) {
        return armor.ypd;
    }
    return xyz_to_ypd(armor.armorPosition);
}

void write_mode1_selection_log_header(std::ofstream& log_file)
{
    log_file << "timestamp,usb_timestamp,dt_ms,tracker_state,update_state,"
             << "tracked_id,primary_obs_index,primary_match_id,"
             << "obs_count,pred_count,detected,pos_diff,ambiguity_index,"
             << "tracker_update_count,tracker_converged,tracker_last_nis,"
             << "tracker_recent_nis_failures,tracker_nis_window,"
             << "tracker_physical_rejection_count,tracker_physical_rejection_reason,"
             << "tracker_normal_pair_required,tracker_normal_pair_found,"
             << "tracker_normal_update_class,tracker_normal_accepted_count,"
             << "tracker_normal_pair_accepted_count,tracker_normal_pair_score,"
             << "tracker_normal_pair_center_gap_m,tracker_normal_pair_center_jump_m,"
             << "tracker_normal_pair_center_x_m,tracker_normal_pair_center_y_m,"
             << "tracker_normal_single_center_count,"
             << "tracker_normal_single_center_x_m,tracker_normal_single_center_y_m,"
             << "tracker_normal_velocity_history_size,"
             << "tracker_normal_velocity_observation_source,"
             << "tracker_normal_velocity_sample_t_s,"
             << "tracker_normal_velocity_sample_x_m,tracker_normal_velocity_sample_y_m,"
             << "tracker_normal_velocity_sample_frame_yaw_rad,"
             << "tracker_normal_velocity_sample_group_id,"
             << "tracker_normal_velocity_fit_sample_count,"
             << "tracker_normal_velocity_fit_accepted,"
             << "tracker_normal_velocity_fit_reject_reason,"
              << "tracker_normal_velocity_fit_time_span_s,"
              << "tracker_normal_velocity_fit_net_displacement_m,"
              << "tracker_normal_velocity_fit_rms_m,"
              << "tracker_normal_velocity_fit_raw_speed_mps,"
              << "tracker_normal_velocity_fit_frame_yaw_rate_rad_s,"
              << "tracker_normal_velocity_fit_frame_yaw_mean_rad,"
              << "tracker_normal_velocity_fit_frame_yaw_span_rad,"
              << "tracker_normal_velocity_fit_pair_sample_count,"
              << "tracker_normal_velocity_fit_single_sample_count,"
              << "tracker_normal_velocity_fit_group_count,"
              << "tracker_normal_velocity_fit_grouped_used,"
              << "tracker_normal_velocity_fit_grouped_speed_mps,"
              << "tracker_normal_velocity_fit_grouped_rms_m,"
              << "tracker_normal_velocity_fit_rot_comp_used,"
              << "tracker_normal_velocity_fit_rot_comp_speed_mps,"
              << "tracker_normal_velocity_fit_frame_transform_pos_yaw_speed_mps,"
              << "tracker_normal_velocity_fit_frame_transform_pos_yaw_rms_m,"
              << "tracker_normal_velocity_fit_frame_transform_neg_yaw_speed_mps,"
              << "tracker_normal_velocity_fit_frame_transform_neg_yaw_rms_m,"
              << "tracker_normal_velocity_fit_applied_speed_mps,"
              << "tracker_normal_yaw_rate_history_size,"
              << "tracker_normal_yaw_rate_observation_source,"
              << "tracker_normal_yaw_observation_count,"
              << "tracker_normal_yaw_observation_t_s,"
              << "tracker_normal_yaw_observation_raw_rad,"
              << "tracker_normal_yaw_observation_unwrapped_rad,"
              << "tracker_normal_yaw_rate_fit_sample_count,"
              << "tracker_normal_yaw_rate_fit_accepted,"
              << "tracker_normal_yaw_rate_fit_reject_reason,"
              << "tracker_normal_yaw_rate_fit_time_span_s,"
              << "tracker_normal_yaw_rate_fit_rms_rad,"
              << "tracker_normal_yaw_rate_fit_raw_rad_s,"
              << "tracker_normal_yaw_rate_fit_applied_rad_s,"
              << "tracker_diverged,tracker_bad_convergence,"
             << "tracker_reset_this_frame,tracker_reset_reason,"
             << "tracker_radius_primary,tracker_radius_secondary,"
             << "tracker_radius_delta,tracker_height_delta,"
             << "pre_xc,pre_vx,pre_yc,pre_vy,pre_yaw,pre_vyaw,pre_r,pre_dr,"
             << "prior_xc,prior_vx,prior_yc,prior_vy,prior_yaw,prior_vyaw,prior_r,prior_dr,"
             << "post_xc,post_vx,post_yc,post_vy,post_yaw,post_vyaw,post_r,post_dr,"
             << "reset_xc,reset_vx,reset_yc,reset_vy,reset_yaw,reset_vyaw,reset_r,reset_dr,"
             << "tracker_match_id0,tracker_match_id1,tracker_match_id2,tracker_match_id3,"
             << "fire_motion_uniform,motion_center_accel_reg,motion_omega_reg,"
             << "motion_prior_vyaw,motion_post_vyaw,"
             << "motion_prior_vx,motion_prior_vy,"
             << "motion_post_vx,motion_post_vy,"
             << "motion_center_update_norm,motion_velocity_update_norm,"
             << "motion_speed_update_abs,"
             << "motion_translation_burst_probe,"
             << "motion_translation_drift_probe,"
             << "motion_translation_burst_gate_metric,"
             << "motion_translation_gate_pass,"
             << "motion_translation_dual_probe_norm";

    for (int obs_index = 0; obs_index < kMode1SelectionLogSlots; ++obs_index) {
        log_file << ",obs" << obs_index << "_number"
                 << ",obs" << obs_index << "_type"
                 << ",obs" << obs_index << "_img_dist"
                 << ",obs" << obs_index << "_x"
                 << ",obs" << obs_index << "_y"
                 << ",obs" << obs_index << "_z"
                 << ",obs" << obs_index << "_yaw"
                 << ",obs" << obs_index << "_yaw_abs"
                 << ",obs" << obs_index << "_camera_yaw"
                 << ",obs" << obs_index << "_ypd_yaw"
                 << ",obs" << obs_index << "_ypd_pitch"
                 << ",obs" << obs_index << "_ypd_dist"
                 << ",obs" << obs_index << "_match_id";
    }

    for (int armor_id = 0; armor_id < kMode1SelectionLogSlots; ++armor_id) {
        log_file << ",pred" << armor_id << "_x"
                 << ",pred" << armor_id << "_y"
                 << ",pred" << armor_id << "_z"
                 << ",pred" << armor_id << "_yaw"
                 << ",pred" << armor_id << "_camera_yaw"
                 << ",pred" << armor_id << "_radius";
    }

    for (int obs_index = 0; obs_index < kMode1SelectionLogSlots; ++obs_index) {
        for (int armor_id = 0; armor_id < kMode1SelectionLogSlots; ++armor_id) {
            log_file << ",cost_obs" << obs_index << "_id" << armor_id;
        }
    }

    log_file << "\n";
}

template <typename TrackerT>
void predict_tracker_if_ready(const std::shared_ptr<TrackerT>& tracker, double dt)
{
    if (tracker && tracker->isInitialized()) {
        tracker->predict(dt);
    }
}

template <typename TrackerT>
std::vector<int> update_tracker_batch(std::shared_ptr<TrackerT>& tracker,
    const std::vector<rm::Armor>& tracker_input_armors, int primary_observation_index,
    int tracked_armor_count, double frame_yaw_rad, double frame_yaw_rate_rad_s)
{
    if (tracker_input_armors.empty() || primary_observation_index < 0 ||
        primary_observation_index >= static_cast<int>(tracker_input_armors.size())) {
        return {};
    }

    if (!tracker) {
        tracker = std::make_shared<TrackerT>();
    }
    tracker->setFrameYaw(frame_yaw_rad);
    tracker->setFrameYawRate(frame_yaw_rate_rad_s);
    if (!tracker->isInitialized()) {
        tracker->init(tracker_input_armors[primary_observation_index], tracked_armor_count);
        tracker->selectTrackedId(tracker_input_armors[primary_observation_index]);
    }
    tracker->updateBatch(tracker_input_armors, primary_observation_index);
    return tracker->lastBatchMatchIds();
}

void copy_mode1_primary_observation_to_tracked_armor(
    const std::shared_ptr<rm::Armor>& tracked_armor, const rm::Armor& primary_tracker_armor)
{
    if (!tracked_armor) return;

    tracked_armor->armorPosition = primary_tracker_armor.armorPosition;
    tracked_armor->ypd = primary_tracker_armor.ypd;
    tracked_armor->yaw = primary_tracker_armor.yaw;
    tracked_armor->yaw_absolute = primary_tracker_armor.yaw_absolute;
    tracked_armor->yaw_raw = primary_tracker_armor.yaw_raw;
    tracked_armor->has_explicit_ypd = primary_tracker_armor.has_explicit_ypd;
    tracked_armor->dis = primary_tracker_armor.dis;
    tracked_armor->rVec = primary_tracker_armor.rVec.clone();
    tracked_armor->tVec = primary_tracker_armor.tVec.clone();
}

double estimate_fly_time(const Eigen::Vector3d& pos, double bullet_speed)
{
    if (bullet_speed < 10.0) bullet_speed = 15.0;
    return std::hypot(pos.x(), pos.y()) / bullet_speed + 0.008;
}

AimChoice choose_spin_aim_point(
    const Target& target, ArmorsNum armors_num, int& lock_id, double omega,
    AIM_MODE aim_mode)
{
    AimChoice choice;
    const auto armor_xyza_list = target.armor_xyza_list();
    if (armor_xyza_list.empty()) return choice;
    choice.xyza = armor_xyza_list[0];

    const Eigen::VectorXd state = target.ekf_x();
    if (!target.jumped) {
        choice.valid = true;
        return choice;
    }

    const double center_yaw = std::atan2(state[2], state[0]);
    std::vector<double> delta_angle_list;
    delta_angle_list.reserve(armor_xyza_list.size());
    for (const auto& armor_xyza : armor_xyza_list) {
        delta_angle_list.push_back(normalize_angle(armor_xyza[3] - center_yaw));
    }

    if (aim_mode == TRACK_HARD) {
        choice.valid = true;
        return choice;
    }

    if (state.size() > 8 && std::abs(state[8]) <= 2.0 &&
        armors_num != ArmorsNum::OUTPOST_3) {
        std::vector<int> id_list;
        for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); ++i) {
            if (std::abs(delta_angle_list[i]) > 60.0 * D2R) continue;
            id_list.push_back(i);
        }

        if (id_list.empty()) {
            return choice;
        }

        if (id_list.size() > 1) {
            const int id0 = id_list[0];
            const int id1 = id_list[1];
            if (lock_id != id0 && lock_id != id1) {
                lock_id =
                    (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1]))
                    ? id0
                    : id1;
            }
            if (lock_id >= 0 && lock_id < static_cast<int>(armor_xyza_list.size())) {
                choice.valid = true;
                choice.xyza = armor_xyza_list[lock_id];
            }
            return choice;
        }

        lock_id = -1;
        choice.valid = true;
        choice.xyza = armor_xyza_list[id_list[0]];
        return choice;
    }

    const double coming_angle = armors_num == ArmorsNum::OUTPOST_3 ? 70.0 * D2R : 55.0 * D2R;
    const double leaving_angle = armors_num == ArmorsNum::OUTPOST_3 ? 30.0 * D2R : 20.0 * D2R;

    for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); ++i) {
        if (std::abs(delta_angle_list[i]) > coming_angle) continue;
        if (omega > 0.0 && delta_angle_list[i] < leaving_angle) {
            choice.valid = true;
            choice.xyza = armor_xyza_list[i];
            return choice;
        }
        if (omega < 0.0 && delta_angle_list[i] > -leaving_angle) {
            choice.valid = true;
            choice.xyza = armor_xyza_list[i];
            return choice;
        }
    }

    return choice;
}

} // namespace

Estimator::Estimator()
    : _measurement(Eigen::VectorXd::Zero(4)),
      _targetStateMat(Eigen::VectorXd::Zero(9)),
      _targetState11d(Eigen::VectorXd::Zero(11)),
      _legacyTargetStateMat(Eigen::VectorXd::Zero(9)),
      _legacyTargetState11d(Eigen::VectorXd::Zero(11)),
      _ypdAngleTargetStateMat(Eigen::VectorXd::Zero(9)),
      _ypdAngleTargetState11d(Eigen::VectorXd::Zero(11))
{
    _params = make_unique<Params>();
    _mode1SelectionLogEnabled =
        _params->DEBUG_LOG_MODE1_SELECTION_CSV ||
        env_flag_enabled("AIM_SIM_DEBUG_MODE1_SELECTION_CSV") ||
        (std::getenv("AIM_SIM_MODE1_SELECTION_CSV") != nullptr &&
         std::getenv("AIM_SIM_MODE1_SELECTION_CSV")[0] != '\0');
    if (_mode1SelectionLogEnabled) {
        const std::filesystem::path log_path = resolve_mode1_selection_log_path();
        _mode1SelectionLogFile.open(log_path, std::ios::out | std::ios::trunc);
        if (_mode1SelectionLogFile.is_open()) {
            write_mode1_selection_log_header(_mode1SelectionLogFile);
            _mode1SelectionLogFile.flush();
            std::cerr << "[Estimator] mode1 selection log path: " << log_path
                      << std::endl;
        } else {
            _mode1SelectionLogEnabled = false;
            std::cerr << "[Estimator] failed to open mode1 selection log: "
                      << log_path << std::endl;
        }
    }
    _dt = 0.03;
 
    _lastTime = 0;
    _timeStamp = 0;
    _last_usb_time = 0; 

    planner_.reset();
    
    predict_target_ = Target(4);

    resetTrackersForMode();

    _last_obs_armor_pos = Eigen::Vector3d::Zero();
    
    tracker_state = LOST;
    update_state = NOMATCH;
    _last_velocity << 0, 0, 0;

}

void Estimator::loadFrame(Frame &frame)
{
    loadMeta(FrameMeta(frame));
    if (!frame.debugImg.empty()) {
        attachDebugImage(frame.debugImg);
    } else {
        attachDebugImage(frame.srcImg);
    }
}

void Estimator::loadMeta(const FrameMeta& frame_meta)
{
    double curr_sys_time = frame_meta.timeStamp;
    double curr_usb_time = frame_meta.usb_timeStamp;

    if (_last_usb_time == 0 || std::abs(curr_usb_time - _last_usb_time) > 100000.0 || std::isnan(_last_usb_time)) 
    {
        _last_usb_time = curr_usb_time - 10.0; 
        resetTrackersForMode();
    }

    double dt_ms = curr_usb_time - _last_usb_time;
    
    if (dt_ms <= 0 || dt_ms > 100.0 || std::isnan(dt_ms)) {
        _dt = 0.006; 
    } else {
        _dt = dt_ms / 1000.0; 
    }

    _last_usb_time = curr_usb_time; 
    _lastTime = curr_sys_time;      
    _timeStamp = curr_sys_time;     

    _bulletSpeed = frame_meta.bullet_speed;
    _gimbal_pitch_d = frame_meta.poseEuler.pitch;
    _gimbal_yaw_d = frame_meta.poseEuler.yaw;
    _gimbal_yaw_speed = frame_meta.fb.yaw_speed;
    _startTime = frame_meta.startTime;
    _params->reload();
    apply_ypd_recovery_config(ypd_angle_tracker_, *_params);
    _lostThreshold = _params->LOST_THRESHOLD;
    update_state = NOMATCH;
}

void Estimator::attachDebugImage(const cv::Mat& image)
{
    _debugImg = image;
}

void Estimator::attachDebugHud(DebugHudSnapshot* hud)
{
    _debugHud = hud;
}

void Estimator::init(const std::vector<std::shared_ptr<Armor>> armors)
{
    _current_obs_armors.clear();
    _current_tracker_input_armors.clear();
    _current_obs_match_ids.clear();
    _current_primary_observation_index = -1;
    jump_flag = 0;
    spin_aim_lock_id_ = -1;

    if (armors.empty()) return;

    std::vector<std::shared_ptr<Armor>> sorted_armors = armors;
    std::stable_sort(sorted_armors.begin(), sorted_armors.end(), armor_observation_less);

    auto first_valid_armor =
        std::find_if(sorted_armors.begin(), sorted_armors.end(),
            [](const std::shared_ptr<Armor>& armor) {
                return armor && !is_neutral_armor_color(armor->color);
            });
    if (first_valid_armor == sorted_armors.end()) {
        first_valid_armor = std::find_if(sorted_armors.begin(), sorted_armors.end(),
            [](const std::shared_ptr<Armor>& armor) {
                return armor != nullptr;
            });
    }
    if (first_valid_armor == sorted_armors.end()) return;
    _trackedArmor = *first_valid_armor;
    _tracked_neutral_color_frames =
        is_neutral_armor_color(_trackedArmor->color) ? 1 : 0;

    updateArmorsNum(_trackedArmor);

    if(_trackedArmor->number != Armor::LABEL::OUTPOST)
        robot.init(_trackedArmor);

    initEKF(_trackedArmor);

    tracker_state = DETECTING;

    robot_center_top_on_image_of_last_frame = Point2f(-1,-1);
    robot_center_top_on_image_of_last_armor = Point2f(-1,-1);
    robot_center_top_on_image = Point2f(-1,-1);
}

bool Estimator::update(const std::vector<std::shared_ptr<Armor>> armors)
{
    const bool ignore_same_number = _params->IGNORE_SAMENUM_CONDITION_SWITCH;
    const double previous_target_armor_z = _targetStateMat(EKF_ARMOR_Z);
    const int tracked_armor_count = static_cast<int>(tracked_armors_num);

    std::vector<std::shared_ptr<Armor>> sorted_observations = armors;
    std::stable_sort(
        sorted_observations.begin(), sorted_observations.end(), armor_observation_less);

    const Eigen::VectorXd predicted_legacy_state =
        predict_legacy_target_state(_targetStateMat, _dt);
    MatchedArmorBatch matched_batch =
        collect_matching_armors(sorted_observations, _trackedArmor, ignore_same_number);

    const int neutral_grace_frames =
        std::max(0, _params ? _params->ARMOR_NEUTRAL_GRACE_FRAMES : 20);
    if (matched_batch.primary_armor_ptr &&
        is_neutral_armor_color(matched_batch.primary_armor_ptr->color) &&
        _tracked_neutral_color_frames >= neutral_grace_frames) {
        matched_batch = MatchedArmorBatch{};
    }

    _current_obs_armors = matched_batch.observations;
    _current_tracker_input_armors = matched_batch.tracker_input_armors;
    if (tracked_armors_num == ArmorsNum::OUTPOST_3) {
        normalize_tracker_armors_as_outpost(_current_tracker_input_armors);
        matched_batch.tracker_input_armors = _current_tracker_input_armors;
    }
    _current_obs_match_ids.assign(_current_obs_armors.size(), -1);
    _current_primary_observation_index = matched_batch.primary_observation_index;

    if (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized()) {
        ypd_debug_pre_predict_state11d = ypd_angle_tracker_->getState();
        ypd_debug_pre_predict_cov_diag =
            covariance_diag_or_zero(ypd_angle_tracker_->getCovariance());
    } else {
        ypd_debug_pre_predict_state11d = Eigen::VectorXd::Zero(11);
        ypd_debug_pre_predict_cov_diag = Eigen::VectorXd::Zero(11);
    }

    const double frame_yaw_rad = std::isfinite(_gimbal_yaw_d)
        ? _gimbal_yaw_d * D2R
        : std::numeric_limits<double>::quiet_NaN();
    const double frame_yaw_rate_rad_s = std::isfinite(_gimbal_yaw_speed)
        ? _gimbal_yaw_speed * D2R
        : std::numeric_limits<double>::quiet_NaN();
    if (ypd_angle_tracker_) {
        ypd_angle_tracker_->setFrameYaw(frame_yaw_rad);
        ypd_angle_tracker_->setFrameYawRate(frame_yaw_rate_rad_s);
    }

    predict_tracker_if_ready(ypd_angle_tracker_, _dt);

    refreshTrackerSnapshots();
    applyActiveTrackerState();

    syncRobotFromActiveTracker();

    if (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized()) {
        ypd_debug_prior_state11d = ypd_angle_tracker_->getState();
        ypd_debug_prior_cov_diag =
            covariance_diag_or_zero(ypd_angle_tracker_->getCovariance());
    } else {
        ypd_debug_prior_state11d = Eigen::VectorXd::Zero(11);
        ypd_debug_prior_cov_diag = Eigen::VectorXd::Zero(11);
    }

    const bool has_primary_observation =
        matched_batch.primary_armor_ptr &&
        matched_batch.primary_observation_index >= 0 &&
        matched_batch.primary_observation_index <
            static_cast<int>(matched_batch.tracker_input_armors.size());

    if (has_primary_observation) {
        _detectedFlag = true;
        _trackedArmor = matched_batch.primary_armor_ptr;
        if (is_neutral_armor_color(_trackedArmor->color)) {
            ++_tracked_neutral_color_frames;
        } else if (is_team_armor_color(_trackedArmor->color)) {
            _tracked_neutral_color_frames = 0;
        }

        const Armor& primary_tracker_armor =
            matched_batch.tracker_input_armors[matched_batch.primary_observation_index];
        const Eigen::Vector3d& primary_observation_pos = primary_tracker_armor.armorPosition;

        _pred_armor_pos = getArmorPositionFromState(predicted_legacy_state);
        Eigen::Vector2d predicted_position_2d(_pred_armor_pos.x(), _pred_armor_pos.y());
        _posDiff = (predicted_position_2d -
                    Eigen::Vector2d(primary_observation_pos.x(), primary_observation_pos.y()))
                       .norm();
        const double observation_jump_distance =
            (primary_observation_pos - _last_obs_armor_pos).norm();
        const bool observation_jump =
            observation_jump_distance > 0.15 && tracker_state != DETECTING;

        if (!_last_trackedArmor) _last_trackedArmor = matched_batch.primary_armor_ptr;

        std::vector<int> tracker_match_ids;
        if (ypd_angle_tracker_) {
            ypd_angle_tracker_->noteObservationJump(observation_jump);
        }
        tracker_match_ids = update_tracker_batch(ypd_angle_tracker_,
            matched_batch.tracker_input_armors, matched_batch.primary_observation_index,
            tracked_armor_count, frame_yaw_rad, frame_yaw_rate_rad_s);

        _current_obs_match_ids = tracker_match_ids;
        if (matched_batch.primary_observation_index >= 0 &&
            matched_batch.primary_observation_index < static_cast<int>(tracker_match_ids.size()) &&
            tracker_match_ids[matched_batch.primary_observation_index] != 0) {
            jump_flag = 1;
        }

        refreshTrackerSnapshots();
        applyActiveTrackerState();

        syncRobotFromActiveTracker();

        copy_mode1_primary_observation_to_tracked_armor(_trackedArmor, primary_tracker_armor);

        update_state = observation_jump ? ARMORJUMP : MATCH_ARMOR_FOUND;
        _last_obs_armor_pos = primary_observation_pos;
    } else {
        refreshTrackerSnapshots();
        applyActiveTrackerState();

        const bool active_tracker_ready =
            ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized();
        if (!active_tracker_ready) {
            _targetStateMat = predicted_legacy_state;
            _targetState11d.setZero();
        }
        syncRobotFromActiveTracker();
        update_state = NOMATCH;
        _detectedFlag = false; 
        _current_obs_match_ids.assign(_current_obs_armors.size(), -1);
    }

    if (update_state != NOMATCH && matched_batch.primary_armor_ptr) {
        _trackedArmor = matched_batch.primary_armor_ptr;
    }
    if (_trackedArmor) {
        _trackedAngle_d = _trackedArmor->yaw_absolute * R2D;
    }

    if (std::abs(_targetStateMat(EKF_ARMOR_YAW)) > 50.0) {
        _targetStateMat(EKF_ARMOR_Z) = previous_target_armor_z;
    }
    if (_trackedArmor) {
        _trackedArmor->armorPosition.z() = _targetStateMat(EKF_ARMOR_Z);
    }

    if (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized()) {
        ypd_debug_posterior_state11d = ypd_angle_tracker_->getState();
        ypd_debug_posterior_cov_diag =
            covariance_diag_or_zero(ypd_angle_tracker_->getCovariance());
    } else {
        ypd_debug_posterior_state11d = Eigen::VectorXd::Zero(11);
        ypd_debug_posterior_cov_diag = Eigen::VectorXd::Zero(11);
    }

    _last_trackedArmor = _trackedArmor; 
    
    update_tracker_state();
    const bool reset_due_diverged =
        ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized() &&
        ypd_angle_tracker_->diverged();
    const bool reset_due_bad_convergence =
        ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized() &&
        ypd_angle_tracker_->badConvergence();
    if (reset_due_diverged || reset_due_bad_convergence) {
        captureYpdTrackerDebug(reset_due_diverged, reset_due_bad_convergence);
        resetActiveTrackerAndMarkLost();
    }
    updateFireSafetyState();
    return true;
}

bool Estimator::activeTrackerNeedsReset() const
{
    return ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized() &&
           (ypd_angle_tracker_->diverged() || ypd_angle_tracker_->badConvergence());
}

void Estimator::clearTrackerEstimateState()
{
    _targetStateMat = Eigen::VectorXd::Zero(9);
    _targetState11d = Eigen::VectorXd::Zero(11);
    _legacyTargetStateMat = Eigen::VectorXd::Zero(9);
    _legacyTargetState11d = Eigen::VectorXd::Zero(11);
    _ypdAngleTargetStateMat = Eigen::VectorXd::Zero(9);
    _ypdAngleTargetState11d = Eigen::VectorXd::Zero(11);
    _last_obs_armor_pos = Eigen::Vector3d::Zero();
    _pred_armor_pos = Eigen::Vector3d::Zero();
    _aimPoint = Eigen::Vector3d::Zero();
    _robotCenter = Eigen::Vector3d::Zero();
}

void Estimator::resetFireSafetyState()
{
    fire_motion_uniform = true;
    fire_observation_stable = false;
    fire_motion_center_accel_metric = std::numeric_limits<double>::quiet_NaN();
    fire_motion_omega_metric = std::numeric_limits<double>::quiet_NaN();
    fire_motion_translation_burst_metric = std::numeric_limits<double>::quiet_NaN();
    fire_motion_translation_drift_metric = std::numeric_limits<double>::quiet_NaN();
    fire_motion_translation_blocked = false;
    fire_observation_hold_frames_ = 0;
}

void Estimator::resetActiveTrackerAndMarkLost()
{
    if (!ypd_angle_tracker_) {
        ypd_angle_tracker_ = std::make_shared<RobotEstimator::YpdAngleTracker>();
    } else {
        ypd_angle_tracker_->reset();
    }
    apply_ypd_recovery_config(ypd_angle_tracker_, *_params);

    clearTrackerEstimateState();
    resetFireSafetyState();
    tracker_state = LOST;
    update_state = NOMATCH;
    detect_count_ = 0;
    lost_count_ = 0;
    _tracked_neutral_color_frames = 0;
    spin_count = 0;
    low_speed = 0;
    jump_flag = 0;
    spin_aim_lock_id_ = -1;
    _detectedFlag = false;
    _isSpinning = false;
    _trackedArmor.reset();
    _anotherArmor.reset();
    _last_trackedArmor.reset();
    _current_obs_armors.clear();
    _current_tracker_input_armors.clear();
    _current_obs_match_ids.clear();
    _current_primary_observation_index = -1;
    robot_center_top_on_image = Point2f(-1, -1);
    robot_center_top_on_image_of_last_frame = Point2f(-1, -1);
    robot_center_top_on_image_of_last_armor = Point2f(-1, -1);
}

void Estimator::updateFireSafetyState()
{
    fire_motion_center_accel_metric = std::numeric_limits<double>::quiet_NaN();
    fire_motion_omega_metric = std::numeric_limits<double>::quiet_NaN();
    fire_motion_translation_burst_metric = std::numeric_limits<double>::quiet_NaN();
    fire_motion_translation_drift_metric = std::numeric_limits<double>::quiet_NaN();
    fire_motion_translation_blocked = false;

    if (_params->FIRE_BLOCK_ON_ARMORJUMP && update_state == ARMORJUMP) {
        fire_observation_hold_frames_ = std::max(
            fire_observation_hold_frames_,
            std::max(0, _params->FIRE_ARMORJUMP_BLOCK_FRAMES));
    } else if (fire_observation_hold_frames_ > 0) {
        fire_observation_hold_frames_--;
    }
    fire_observation_stable = fire_observation_hold_frames_ <= 0;

    fire_motion_uniform = true;
}

void Estimator::resetTrackersForMode()
{
    clearYpdResetDebug();
    tracker_state = LOST;
    update_state = NOMATCH;
    detect_count_ = 0;
    lost_count_ = 0;
    _tracked_neutral_color_frames = 0;
    spin_count = 0;
    low_speed = 0;
    pour_water = 0;
    jump_flag = 0;
    spin_aim_lock_id_ = -1;
    _detectedFlag = false;
    _isSpinning = false;
    _trackedArmor.reset();
    _anotherArmor.reset();
    _last_trackedArmor.reset();
    _current_obs_armors.clear();
    _current_tracker_input_armors.clear();
    _current_obs_match_ids.clear();
    _current_primary_observation_index = -1;
    clearTrackerEstimateState();
    resetFireSafetyState();
    robot_center_top_on_image = Point2f(-1, -1);
    robot_center_top_on_image_of_last_frame = Point2f(-1, -1);
    robot_center_top_on_image_of_last_armor = Point2f(-1, -1);
    predict_target_ = Target(4);

    if (!ypd_angle_tracker_) {
        ypd_angle_tracker_ = std::make_shared<RobotEstimator::YpdAngleTracker>();
    } else {
        ypd_angle_tracker_->reset();
    }
    apply_ypd_recovery_config(ypd_angle_tracker_, *_params);
}

void Estimator::resetForTaskModeSwitch()
{
    resetTrackersForMode();
}

void Estimator::refreshTrackerSnapshots()
{
    _legacyTargetStateMat.setZero();
    _legacyTargetState11d.setZero();
    _ypdAngleTargetStateMat.setZero();
    _ypdAngleTargetState11d.setZero();

    refresh_tracker_snapshot(ypd_angle_tracker_, _ypdAngleTargetState11d, _ypdAngleTargetStateMat);
}

void Estimator::applyActiveTrackerState()
{
    if (_ypdAngleTargetState11d.size() >= 11 && _ypdAngleTargetState11d.squaredNorm() > 0.0) {
        _targetState11d = _ypdAngleTargetState11d;
        _targetStateMat = _ypdAngleTargetStateMat;
        return;
    }
    _targetState11d = Eigen::VectorXd::Zero(11);
    _targetStateMat = Eigen::VectorXd::Zero(9);
}

void Estimator::syncRobotFromActiveTracker()
{
    if (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized()) {
        sync_robot_msg_from_tracker(robot, *ypd_angle_tracker_);
    }
}

void Estimator::initEKF(const std::shared_ptr<Armor> & armors)
{
    if (!armors) return;
    clearTrackerEstimateState();
    Armor active_armor_for_tracker = *armors;

    if (!ypd_angle_tracker_) {
        ypd_angle_tracker_ = std::make_shared<RobotEstimator::YpdAngleTracker>();
    }
    apply_ypd_recovery_config(ypd_angle_tracker_, *_params);
    ypd_angle_tracker_->init(active_armor_for_tracker, static_cast<int>(tracked_armors_num));
    ypd_angle_tracker_->selectTrackedId(active_armor_for_tracker);
    refreshTrackerSnapshots();
    applyActiveTrackerState();

    _last_obs_armor_pos = active_armor_for_tracker.armorPosition;

    armors->armorPosition = active_armor_for_tracker.armorPosition;
    armors->ypd = active_armor_for_tracker.ypd;
    armors->yaw = active_armor_for_tracker.yaw;
    armors->yaw_absolute = active_armor_for_tracker.yaw_absolute;
    armors->yaw_raw = active_armor_for_tracker.yaw_raw;
    armors->has_explicit_ypd = active_armor_for_tracker.has_explicit_ypd;
    armors->dis = active_armor_for_tracker.dis;
    armors->rVec = active_armor_for_tracker.rVec.clone();
    armors->tVec = active_armor_for_tracker.tVec.clone();

    syncRobotFromActiveTracker();
}

void Estimator::updateArmorsNum(const std::shared_ptr<Armor> & armor)
{
    if (!armor) return;
    tracked_armors_num =
        (armor->number == Armor::LABEL::OUTPOST) ? ArmorsNum::OUTPOST_3 : ArmorsNum::NORMAL_4;
    predict_target_ = Target(static_cast<int>(tracked_armors_num));
}

void Estimator::clearYpdResetDebug()
{
    ypd_reset_this_frame = false;
    ypd_reset_diverged = false;
    ypd_reset_bad_convergence = false;
    ypd_reset_reason = "none";
    ypd_debug_primary_radius = std::numeric_limits<double>::quiet_NaN();
    ypd_debug_secondary_radius = std::numeric_limits<double>::quiet_NaN();
    ypd_debug_delta_radius = std::numeric_limits<double>::quiet_NaN();
    ypd_debug_height_delta = std::numeric_limits<double>::quiet_NaN();
    ypd_debug_update_count = 0;
    ypd_debug_recent_nis_failures = 0;
    ypd_debug_nis_window = 0;
    ypd_debug_last_match_ids.clear();
    ypd_debug_reset_state11d = Eigen::VectorXd::Zero(11);
    ypd_debug_reset_cov_diag = Eigen::VectorXd::Zero(11);
}

void Estimator::captureYpdTrackerDebug(bool diverged, bool bad_convergence)
{
    clearYpdResetDebug();
    ypd_reset_this_frame = diverged || bad_convergence;
    ypd_reset_diverged = diverged;
    ypd_reset_bad_convergence = bad_convergence;
    if (diverged && bad_convergence) {
        ypd_reset_reason = "diverged+bad_convergence";
    } else if (diverged) {
        ypd_reset_reason = "diverged";
    } else if (bad_convergence) {
        ypd_reset_reason = "bad_convergence";
    }

    if (!ypd_angle_tracker_ || !ypd_angle_tracker_->isInitialized()) return;

    const Eigen::VectorXd state = ypd_angle_tracker_->getState();
    ypd_debug_reset_state11d = state;
    ypd_debug_reset_cov_diag =
        covariance_diag_or_zero(ypd_angle_tracker_->getCovariance());
    if (state.size() > 8) {
        ypd_debug_primary_radius = state(8);
    }
    if (state.size() > 9) {
        ypd_debug_delta_radius = state(9);
        if (std::isfinite(ypd_debug_primary_radius)) {
            ypd_debug_secondary_radius = ypd_debug_primary_radius + state(9);
        }
    }
    if (state.size() > 10) {
        ypd_debug_height_delta = state(10);
    }
    ypd_debug_update_count = ypd_angle_tracker_->updateCount();
    ypd_debug_recent_nis_failures = ypd_angle_tracker_->recentNisFailureCount();
    ypd_debug_nis_window = ypd_angle_tracker_->nisWindowSize();
    ypd_debug_last_match_ids = ypd_angle_tracker_->lastBatchMatchIds();
}

void Estimator::update_tracker_state()
{
    const State previous_state = tracker_state;
    bool matched = (update_state == MATCH_ARMOR_FOUND || update_state == ARMORJUMP);
    if (tracker_state == DETECTING) {
        if (matched) {
            detect_count_++;
            if (detect_count_ > _trackingThreshold) {
                detect_count_ = 0;
                tracker_state = TRACKING;
            }
        } else {
            detect_count_ = 0;
            tracker_state = LOST;
        }
    } else if (tracker_state == TRACKING) {
        if (!matched) {
            tracker_state = TEMP_LOST;
            lost_count_++;
        }
    } else if (tracker_state == TEMP_LOST) {
        if (!matched) {
            if (lost_count_ > _lostThreshold) {
                lost_count_ = 0;
                tracker_state = LOST;
            } else lost_count_++;
        } else {
            tracker_state = TRACKING;
            lost_count_ = 0;
        }
    }

    if (tracker_state == LOST && previous_state != LOST && ypd_angle_tracker_) {
        ypd_angle_tracker_->clearGeometryRecoveryHistory();
    }
}

Eigen::Vector3d Estimator::getArmorPositionFromState(const Eigen::VectorXd & x)
{
    double xc = x(EKF_CENTER_X), yc = x(EKF_CENTER_Y), za = x(EKF_ARMOR_Z);
    double yaw = x(EKF_ARMOR_YAW);
    double r = x(EKF_ROTATION_RADIUS);
    double xa = xc - r * cos(yaw);
    double ya = yc - r * sin(yaw);
    return Eigen::Vector3d(xa, ya, za);
}

void Estimator::getHitPoint()
{
    if (!_trackedArmor) return;

    const double center_z =
        _targetState11d.size() >= 11 ? _targetState11d(4) : _targetStateMat(EKF_ARMOR_Z);
    _robotCenter << _targetStateMat(EKF_CENTER_X), _targetStateMat(EKF_CENTER_Y), center_z;
    
    _center_yaw_rad = atan(abs(_targetStateMat(EKF_CENTER_Y))/abs(_targetStateMat(EKF_CENTER_X)));
    if(_targetStateMat(EKF_CENTER_X) <= 0 && _targetStateMat(EKF_CENTER_Y) >= 0)   
        _center_yaw_rad = _center_yaw_rad - CV_PI;
    else if(_targetStateMat(EKF_CENTER_X) <= 0 && _targetStateMat(EKF_CENTER_Y) <= 0)  
        _center_yaw_rad = CV_PI - _center_yaw_rad;
    else if(_targetStateMat(EKF_CENTER_X) >= 0 && _targetStateMat(EKF_CENTER_Y) >= 0)   
        _center_yaw_rad = -_center_yaw_rad;

    if(abs(_targetStateMat(EKF_ARMOR_V_YAW)) < 0.1) spin_count = 0;
    else spin_count++;

    _isSpinning = spin_count > 10;

    if (!_isSpinning) {
        aimTranposeTarget();
    } else {
        double fly_dis = _trackedArmor->armorPosition.norm();
        if(abs(_targetStateMat(EKF_ARMOR_V_YAW)) < 5 && fly_dis > 1.2) low_speed ++;
        else low_speed = 0;
        if (low_speed >= 10) aimRotatingTarget(AIM_MODE::TRACK_ARMOR_WITHIN_RANGE);
        else aimRotatingTarget(AIM_MODE::TRACK_CENTER);
    }
}
void Estimator::aimTranposeTarget()
{
    if (!_trackedArmor) return;

    _isSpinning = false;

    double xc = _targetStateMat(EKF_CENTER_X);
    double yc = _targetStateMat(EKF_CENTER_Y);
    double vx = _targetStateMat(EKF_CENTER_V_X);
    double vy = _targetStateMat(EKF_CENTER_V_Y);
    double z  = _targetStateMat(EKF_ARMOR_Z);
    double yaw = _targetStateMat(EKF_ARMOR_YAW);
    double r   = _targetStateMat(EKF_ROTATION_RADIUS);

    double v_xy = sqrt(vx * vx + vy * vy);            
    static int trans_count = 0;
    if (v_xy >= 0.2) trans_count++;
    else trans_count = 0;
    if (trans_count >= 5) movement = TRANSLATION;
    else movement = STATIC;
    
    double dis = _trackedArmor->armorPosition.norm();
    double predict_time_delta = dis / _bulletSpeed + _params->HORIZONTAL_DELAY_TIME + latency / 1000.0;

    double pred_xc = xc + vx * predict_time_delta;
    double pred_yc = yc + vy * predict_time_delta;
    double pred_z  = z; 

    double vyaw = _targetStateMat(EKF_ARMOR_V_YAW);
    double pred_yaw = yaw + vyaw * predict_time_delta;

    double pred_xa = pred_xc - r * cos(pred_yaw);
    double pred_ya = pred_yc - r * sin(pred_yaw);

    _aimPoint << pred_xa, pred_ya, pred_z;
    _last_velocity << vx, vy, 0;
}

void Estimator::aimRotatingTarget(AIM_MODE aim_mode)
{
    if (!_trackedArmor) return;

    _aim_mode = aim_mode;
    _hitPoint = _trackedArmor->armorPosition;
    double vx = _targetStateMat(EKF_CENTER_V_X);
    double vy = _targetStateMat(EKF_CENTER_V_Y);
    double linear_speed = std::hypot(vx, vy);
    movement = linear_speed > 0.2 ? TRANSPIN : SPINNING;

    double delay_time =
        _params->SPIN_DELAY_TIME_SWITCH ? _params->SPIN_DELAY_TIME_s : _params->HORIZONTAL_DELAY_TIME;

    Eigen::VectorXd delayed_state = _targetState11d;
    if (delayed_state.size() != 11) {
        delayed_state = Eigen::VectorXd::Zero(11);
        delayed_state(0) = _targetStateMat(EKF_CENTER_X);
        delayed_state(1) = _targetStateMat(EKF_CENTER_V_X);
        delayed_state(2) = _targetStateMat(EKF_CENTER_Y);
        delayed_state(3) = _targetStateMat(EKF_CENTER_V_Y);
        delayed_state(4) = _targetStateMat(EKF_ARMOR_Z);
        delayed_state(5) = _targetStateMat(EKF_ARMOR_V_Z);
        delayed_state(6) = _targetStateMat(EKF_ARMOR_YAW);
        delayed_state(7) = _targetStateMat(EKF_ARMOR_V_YAW);
        delayed_state(8) = _targetStateMat(EKF_ROTATION_RADIUS);
    }

    delayed_state(0) += delayed_state(1) * delay_time;
    delayed_state(2) += delayed_state(3) * delay_time;
    delayed_state(4) += delayed_state(5) * delay_time;
    delayed_state(6) = normalize_angle(delayed_state(6) + delayed_state(7) * delay_time);

    predict_target_.sync_state(delayed_state);
    predict_target_.jumped = jump_flag != 0;

    double fly_time = estimate_fly_time(_trackedArmor->armorPosition, _bulletSpeed);
    AimChoice choice;
    for (int iter = 0; iter < 8; ++iter) {
        Target iter_target = predict_target_;
        iter_target.predict(fly_time);
        choice = choose_spin_aim_point(
            iter_target, tracked_armors_num, spin_aim_lock_id_, delayed_state(7), aim_mode);

        if (!choice.valid) break;

        double next_fly_time = estimate_fly_time(choice.xyza.head<3>(), _bulletSpeed);
        if (std::abs(next_fly_time - fly_time) < 0.001) {
            fly_time = next_fly_time;
            break;
        }
        fly_time = next_fly_time;
    }

    if (!choice.valid) {
        const auto fallback_armors = predict_target_.armor_xyza_list();
        if (!fallback_armors.empty()) {
            int fallback_id = 0;
            if (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized()) {
                fallback_id = ypd_angle_tracker_->getTrackedId();
            }
            if (fallback_id < 0) fallback_id = 0;
            if (fallback_id >= static_cast<int>(fallback_armors.size())) {
                fallback_id = static_cast<int>(fallback_armors.size()) - 1;
            }

            _aimPoint = fallback_armors[fallback_id].head<3>();
        } else {
            _aimPoint = getArmorPositionFromState(_targetStateMat);
        }
        return;
    }

    _aimPoint = choice.xyza.head<3>();
}
void Estimator::trackerUpdate(std::vector<std::shared_ptr<Armor>> armors, AngleSolver & anglesolver)
{
    _angleSolverPtr = &anglesolver;
    clearYpdResetDebug();

    bool init_flag = false;
    if (tracker_state == State::LOST) {
        init(armors); 
        init_flag = true;
    } else update(armors); 

    if (init_flag && ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized()) {
        const Eigen::VectorXd init_state = ypd_angle_tracker_->getState();
        const Eigen::VectorXd init_cov_diag =
            covariance_diag_or_zero(ypd_angle_tracker_->getCovariance());
        ypd_debug_pre_predict_state11d = init_state;
        ypd_debug_prior_state11d = init_state;
        ypd_debug_posterior_state11d = init_state;
        ypd_debug_pre_predict_cov_diag = init_cov_diag;
        ypd_debug_prior_cov_diag = init_cov_diag;
        ypd_debug_posterior_cov_diag = init_cov_diag;
    }

    if((tracker_state == State::LOST) || init_flag) {
        _current_obs_match_ids.assign(_current_obs_armors.size(), -1);
        _current_primary_observation_index = -1;
        _detectedFlag = false;
        _lastTime = _timeStamp;
    } else {
        _detectedFlag = true;
        getHitPoint(); 
        _lastTime = _timeStamp;
    }

    logMode1SelectionDebug();
}

void Estimator::logMode1SelectionDebug()
{
    if (!_mode1SelectionLogEnabled || !_mode1SelectionLogFile.is_open()) {
        return;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int tracked_id = (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->getTrackedId()
        : -1;
    const int pred_count = (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->getArmorNum()
        : 0;
    const int primary_match_id =
        (_current_primary_observation_index >= 0 &&
         _current_primary_observation_index < static_cast<int>(_current_obs_match_ids.size()))
        ? _current_obs_match_ids[_current_primary_observation_index]
        : -1;

    std::vector<Eigen::Vector4d> predicted_armors;
    predicted_armors.reserve(kMode1SelectionLogSlots);
    if (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized()) {
        predicted_armors = ypd_angle_tracker_->getPredictedArmorStates();
    }
    const MatchCostMatrix match_cost_matrix =
        build_match_cost_matrix(_current_tracker_input_armors, predicted_armors);
    const double ambiguity_index = compute_ambiguity_index(
        match_cost_matrix, static_cast<int>(_current_tracker_input_armors.size()), pred_count);
    const double tracker_last_nis =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastNis()
        : nan;
    const bool tracker_ready =
        ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized();
    const Eigen::VectorXd tracker_state11d =
        tracker_ready ? ypd_angle_tracker_->getState() : Eigen::VectorXd();
    const int tracker_recent_nis_failures =
        tracker_ready ? ypd_angle_tracker_->recentNisFailureCount()
                      : ypd_debug_recent_nis_failures;
    const int tracker_nis_window =
        tracker_ready ? ypd_angle_tracker_->nisWindowSize() : ypd_debug_nis_window;
    const int tracker_physical_rejection_count =
        tracker_ready ? ypd_angle_tracker_->lastPhysicalRejectionCount() : 0;
    const int tracker_physical_rejection_reason =
        tracker_ready ? ypd_angle_tracker_->lastPhysicalRejectionReason() : 0;
    const int tracker_normal_pair_required =
        tracker_ready ? ypd_angle_tracker_->lastNormalPairRequired() : 0;
    const int tracker_normal_pair_found =
        tracker_ready ? ypd_angle_tracker_->lastNormalPairFound() : 0;
    const int tracker_normal_update_class =
        tracker_ready ? ypd_angle_tracker_->lastNormalUpdateClass() : 0;
    const int tracker_normal_accepted_count =
        tracker_ready ? ypd_angle_tracker_->lastNormalAcceptedCount() : 0;
    const int tracker_normal_pair_accepted_count =
        tracker_ready ? ypd_angle_tracker_->lastNormalPairAcceptedCount() : 0;
    const double tracker_normal_pair_score =
        tracker_ready ? ypd_angle_tracker_->lastNormalPairScore() : nan;
    const double tracker_normal_pair_center_gap_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalPairCenterGap() : nan;
    const double tracker_normal_pair_center_jump_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalPairCenterJump() : nan;
    const double tracker_normal_pair_center_x_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalPairCenterX() : nan;
    const double tracker_normal_pair_center_y_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalPairCenterY() : nan;
    const int tracker_normal_single_center_count =
        tracker_ready ? ypd_angle_tracker_->lastNormalSingleCenterCount() : 0;
    const double tracker_normal_single_center_x_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalSingleCenterX() : nan;
    const double tracker_normal_single_center_y_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalSingleCenterY() : nan;
    const int tracker_normal_velocity_history_size =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityHistorySize() : 0;
    const int tracker_normal_velocity_observation_source =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityObservationSource() : 0;
    const double tracker_normal_velocity_sample_t_s =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocitySampleTime() : nan;
    const double tracker_normal_velocity_sample_x_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocitySampleX() : nan;
    const double tracker_normal_velocity_sample_y_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocitySampleY() : nan;
    const double tracker_normal_velocity_sample_frame_yaw_rad =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocitySampleFrameYaw() : nan;
    const int tracker_normal_velocity_sample_group_id =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocitySampleGroupId() : -1;
    const int tracker_normal_velocity_fit_sample_count =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitSampleCount() : 0;
    const int tracker_normal_velocity_fit_accepted =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitAccepted() : 0;
    const int tracker_normal_velocity_fit_reject_reason =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitRejectReason() : 0;
    const double tracker_normal_velocity_fit_time_span_s =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitTimeSpan() : nan;
    const double tracker_normal_velocity_fit_net_displacement_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitNetDisplacement() : nan;
    const double tracker_normal_velocity_fit_rms_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitRms() : nan;
    const double tracker_normal_velocity_fit_raw_speed_mps =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitRawSpeed() : nan;
    const double tracker_normal_velocity_fit_frame_yaw_rate_rad_s =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitFrameYawRate() : nan;
    const double tracker_normal_velocity_fit_frame_yaw_mean_rad =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitFrameYawMean() : nan;
    const double tracker_normal_velocity_fit_frame_yaw_span_rad =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitFrameYawSpan() : nan;
    const int tracker_normal_velocity_fit_pair_sample_count =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitPairSampleCount() : 0;
    const int tracker_normal_velocity_fit_single_sample_count =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitSingleSampleCount() : 0;
    const int tracker_normal_velocity_fit_group_count =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitGroupCount() : 0;
    const int tracker_normal_velocity_fit_grouped_used =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitGroupedUsed() : 0;
    const double tracker_normal_velocity_fit_grouped_speed_mps =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitGroupedSpeed() : nan;
    const double tracker_normal_velocity_fit_grouped_rms_m =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitGroupedRms() : nan;
    const int tracker_normal_velocity_fit_rot_comp_used =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitRotCompUsed() : 0;
    const double tracker_normal_velocity_fit_rot_comp_speed_mps =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitRotCompSpeed() : nan;
    const double tracker_normal_velocity_fit_frame_transform_pos_yaw_speed_mps =
        tracker_ready
            ? ypd_angle_tracker_->lastNormalVelocityFitFrameTransformPosYawSpeed()
            : nan;
    const double tracker_normal_velocity_fit_frame_transform_pos_yaw_rms_m =
        tracker_ready
            ? ypd_angle_tracker_->lastNormalVelocityFitFrameTransformPosYawRms()
            : nan;
    const double tracker_normal_velocity_fit_frame_transform_neg_yaw_speed_mps =
        tracker_ready
            ? ypd_angle_tracker_->lastNormalVelocityFitFrameTransformNegYawSpeed()
            : nan;
    const double tracker_normal_velocity_fit_frame_transform_neg_yaw_rms_m =
        tracker_ready
            ? ypd_angle_tracker_->lastNormalVelocityFitFrameTransformNegYawRms()
            : nan;
    const double tracker_normal_velocity_fit_applied_speed_mps =
        tracker_ready ? ypd_angle_tracker_->lastNormalVelocityFitAppliedSpeed() : nan;
    const int tracker_normal_yaw_rate_history_size =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawRateHistorySize() : 0;
    const int tracker_normal_yaw_rate_observation_source =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawRateObservationSource() : 0;
    const int tracker_normal_yaw_observation_count =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawObservationCount() : 0;
    const double tracker_normal_yaw_observation_t_s =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawObservationTime() : nan;
    const double tracker_normal_yaw_observation_raw_rad =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawObservationRaw() : nan;
    const double tracker_normal_yaw_observation_unwrapped_rad =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawObservationUnwrapped() : nan;
    const int tracker_normal_yaw_rate_fit_sample_count =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawRateFitSampleCount() : 0;
    const int tracker_normal_yaw_rate_fit_accepted =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawRateFitAccepted() : 0;
    const int tracker_normal_yaw_rate_fit_reject_reason =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawRateFitRejectReason() : 0;
    const double tracker_normal_yaw_rate_fit_time_span_s =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawRateFitTimeSpan() : nan;
    const double tracker_normal_yaw_rate_fit_rms_rad =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawRateFitRms() : nan;
    const double tracker_normal_yaw_rate_fit_raw_rad_s =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawRateFitRaw() : nan;
    const double tracker_normal_yaw_rate_fit_applied_rad_s =
        tracker_ready ? ypd_angle_tracker_->lastNormalYawRateFitApplied() : nan;
    const bool tracker_diverged =
        tracker_ready ? ypd_angle_tracker_->diverged() : ypd_reset_diverged;
    const bool tracker_bad_convergence =
        tracker_ready ? ypd_angle_tracker_->badConvergence() : ypd_reset_bad_convergence;
    const double tracker_radius_primary =
        tracker_state11d.size() > 8 ? tracker_state11d(8) : ypd_debug_primary_radius;
    const double tracker_radius_delta =
        tracker_state11d.size() > 9 ? tracker_state11d(9) : ypd_debug_delta_radius;
    const double tracker_radius_secondary =
        tracker_state11d.size() > 9 ? tracker_state11d(8) + tracker_state11d(9)
                                    : ypd_debug_secondary_radius;
    const double tracker_height_delta =
        tracker_state11d.size() > 10 ? tracker_state11d(10) : ypd_debug_height_delta;
    const std::vector<int>& tracker_match_ids =
        tracker_ready ? ypd_angle_tracker_->lastBatchMatchIds() : ypd_debug_last_match_ids;
    const double motion_prior_vyaw =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastMotionPriorYawRate()
        : nan;
    const double motion_post_vyaw =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastMotionPosteriorYawRate()
        : nan;
    const double motion_prior_vx =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastMotionPriorVx()
        : nan;
    const double motion_prior_vy =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastMotionPriorVy()
        : nan;
    const double motion_post_vx =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastMotionPosteriorVx()
        : nan;
    const double motion_post_vy =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastMotionPosteriorVy()
        : nan;
    const double motion_center_update_norm =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastMotionCenterUpdateNorm()
        : nan;
    const double motion_velocity_update_norm =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastMotionVelocityUpdateNorm()
        : nan;
    const double motion_speed_update_abs =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->lastMotionSpeedUpdateAbs()
        : nan;
    const double motion_translation_burst_probe = fire_motion_center_accel_metric;
    const double motion_translation_drift_probe = fire_motion_translation_drift_metric;
    const double motion_translation_burst_gate_metric =
        fire_motion_translation_burst_metric;
    const double motion_translation_dual_probe_norm = nan;
    const int tracker_update_count =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized())
        ? ypd_angle_tracker_->updateCount()
        : 0;
    const int tracker_converged =
        (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized() &&
         ypd_angle_tracker_->convergedStatus())
        ? 1
        : 0;

    _mode1SelectionLogFile << _timeStamp << ","
                           << _last_usb_time << ","
                           << _dt * 1000.0 << ","
                           << static_cast<int>(tracker_state) << ","
                           << static_cast<int>(update_state) << ","
                           << tracked_id << ","
                           << _current_primary_observation_index << ","
                           << primary_match_id << ","
                           << _current_tracker_input_armors.size() << ","
                           << pred_count << ","
                           << static_cast<int>(_detectedFlag) << ","
                           << _posDiff << ","
                           << ambiguity_index << ","
                           << tracker_update_count << ","
                           << tracker_converged << ","
                           << tracker_last_nis << ","
                           << tracker_recent_nis_failures << ","
                           << tracker_nis_window << ","
                           << tracker_physical_rejection_count << ","
                           << tracker_physical_rejection_reason << ","
                           << tracker_normal_pair_required << ","
                           << tracker_normal_pair_found << ","
                           << tracker_normal_update_class << ","
                           << tracker_normal_accepted_count << ","
                           << tracker_normal_pair_accepted_count << ","
                           << tracker_normal_pair_score << ","
                           << tracker_normal_pair_center_gap_m << ","
                           << tracker_normal_pair_center_jump_m << ","
                           << tracker_normal_pair_center_x_m << ","
                           << tracker_normal_pair_center_y_m << ","
                           << tracker_normal_single_center_count << ","
                           << tracker_normal_single_center_x_m << ","
                           << tracker_normal_single_center_y_m << ","
                           << tracker_normal_velocity_history_size << ","
                           << tracker_normal_velocity_observation_source << ","
                           << tracker_normal_velocity_sample_t_s << ","
                           << tracker_normal_velocity_sample_x_m << ","
                           << tracker_normal_velocity_sample_y_m << ","
                           << tracker_normal_velocity_sample_frame_yaw_rad << ","
                           << tracker_normal_velocity_sample_group_id << ","
                           << tracker_normal_velocity_fit_sample_count << ","
                           << tracker_normal_velocity_fit_accepted << ","
                           << tracker_normal_velocity_fit_reject_reason << ","
                           << tracker_normal_velocity_fit_time_span_s << ","
                           << tracker_normal_velocity_fit_net_displacement_m << ","
                           << tracker_normal_velocity_fit_rms_m << ","
                           << tracker_normal_velocity_fit_raw_speed_mps << ","
                           << tracker_normal_velocity_fit_frame_yaw_rate_rad_s << ","
                           << tracker_normal_velocity_fit_frame_yaw_mean_rad << ","
                           << tracker_normal_velocity_fit_frame_yaw_span_rad << ","
                           << tracker_normal_velocity_fit_pair_sample_count << ","
                           << tracker_normal_velocity_fit_single_sample_count << ","
                           << tracker_normal_velocity_fit_group_count << ","
                           << tracker_normal_velocity_fit_grouped_used << ","
                           << tracker_normal_velocity_fit_grouped_speed_mps << ","
                           << tracker_normal_velocity_fit_grouped_rms_m << ","
                           << tracker_normal_velocity_fit_rot_comp_used << ","
                           << tracker_normal_velocity_fit_rot_comp_speed_mps << ","
                           << tracker_normal_velocity_fit_frame_transform_pos_yaw_speed_mps << ","
                           << tracker_normal_velocity_fit_frame_transform_pos_yaw_rms_m << ","
                           << tracker_normal_velocity_fit_frame_transform_neg_yaw_speed_mps << ","
                           << tracker_normal_velocity_fit_frame_transform_neg_yaw_rms_m << ","
                           << tracker_normal_velocity_fit_applied_speed_mps << ","
                           << tracker_normal_yaw_rate_history_size << ","
                           << tracker_normal_yaw_rate_observation_source << ","
                           << tracker_normal_yaw_observation_count << ","
                           << tracker_normal_yaw_observation_t_s << ","
                           << tracker_normal_yaw_observation_raw_rad << ","
                           << tracker_normal_yaw_observation_unwrapped_rad << ","
                           << tracker_normal_yaw_rate_fit_sample_count << ","
                           << tracker_normal_yaw_rate_fit_accepted << ","
                           << tracker_normal_yaw_rate_fit_reject_reason << ","
                           << tracker_normal_yaw_rate_fit_time_span_s << ","
                           << tracker_normal_yaw_rate_fit_rms_rad << ","
                           << tracker_normal_yaw_rate_fit_raw_rad_s << ","
                           << tracker_normal_yaw_rate_fit_applied_rad_s << ","
                           << static_cast<int>(tracker_diverged) << ","
                           << static_cast<int>(tracker_bad_convergence) << ","
                           << static_cast<int>(ypd_reset_this_frame) << ","
                           << ypd_reset_reason << ","
                           << tracker_radius_primary << ","
                           << tracker_radius_secondary << ","
                           << tracker_radius_delta << ","
                           << tracker_height_delta << ",";
    write_state_key_fields(_mode1SelectionLogFile, ypd_debug_pre_predict_state11d);
    _mode1SelectionLogFile << ",";
    write_state_key_fields(_mode1SelectionLogFile, ypd_debug_prior_state11d);
    _mode1SelectionLogFile << ",";
    write_state_key_fields(_mode1SelectionLogFile, ypd_debug_posterior_state11d);
    _mode1SelectionLogFile << ",";
    write_state_key_fields(_mode1SelectionLogFile, ypd_debug_reset_state11d);
    _mode1SelectionLogFile << ",";
    for (int match_index = 0; match_index < kMode1SelectionLogSlots; ++match_index) {
        if (match_index > 0) _mode1SelectionLogFile << ",";
        const int match_id =
            match_index < static_cast<int>(tracker_match_ids.size())
            ? tracker_match_ids[match_index]
            : -1;
        _mode1SelectionLogFile << match_id;
    }
    _mode1SelectionLogFile << ","
                           << static_cast<int>(fire_motion_uniform) << ","
                           << fire_motion_center_accel_metric << ","
                           << fire_motion_omega_metric << ","
                           << motion_prior_vyaw << ","
                           << motion_post_vyaw << ","
                           << motion_prior_vx << ","
                           << motion_prior_vy << ","
                           << motion_post_vx << ","
                           << motion_post_vy << ","
                           << motion_center_update_norm << ","
                           << motion_velocity_update_norm << ","
                           << motion_speed_update_abs << ","
                           << motion_translation_burst_probe << ","
                           << motion_translation_drift_probe << ","
                           << motion_translation_burst_gate_metric << ","
                           << static_cast<int>(fire_motion_translation_blocked) << ","
                           << motion_translation_dual_probe_norm;

    for (int obs_index = 0; obs_index < kMode1SelectionLogSlots; ++obs_index) {
        if (obs_index < static_cast<int>(_current_tracker_input_armors.size())) {
            const Armor& obs_armor = _current_tracker_input_armors[obs_index];
            const Eigen::Vector3d obs_ypd = observation_ypd_for_log(obs_armor);
            const double obs_camera_yaw =
                std::atan2(obs_armor.armorPosition.y(), obs_armor.armorPosition.x());
            const int matched_id =
                obs_index < static_cast<int>(_current_obs_match_ids.size())
                ? _current_obs_match_ids[obs_index]
                : -1;

            _mode1SelectionLogFile << ","
                                   << obs_armor.number << ","
                                   << static_cast<int>(obs_armor.type) << ","
                                   << obs_armor.distanceToImageCenter << ","
                                   << obs_armor.armorPosition.x() << ","
                                   << obs_armor.armorPosition.y() << ","
                                   << obs_armor.armorPosition.z() << ","
                                   << obs_armor.yaw << ","
                                   << obs_armor.yaw_absolute << ","
                                   << obs_camera_yaw << ","
                                   << obs_ypd(0) << ","
                                   << obs_ypd(1) << ","
                                   << obs_ypd(2) << ","
                                   << matched_id;
        } else {
            _mode1SelectionLogFile << ",-1,-1," << nan << "," << nan << "," << nan << ","
                                   << nan << "," << nan << "," << nan << "," << nan << ","
                                   << nan << "," << nan << "," << nan << ",-1";
        }
    }

    for (int armor_id = 0; armor_id < kMode1SelectionLogSlots; ++armor_id) {
        if (armor_id < static_cast<int>(predicted_armors.size())) {
            const Eigen::Vector4d& pred_xyza = predicted_armors[armor_id];
            const double pred_camera_yaw = std::atan2(pred_xyza(1), pred_xyza(0));
            const double pred_radius = ypd_angle_tracker_->getArmorRadius(armor_id);
            _mode1SelectionLogFile << ","
                                   << pred_xyza(0) << ","
                                   << pred_xyza(1) << ","
                                   << pred_xyza(2) << ","
                                   << pred_xyza(3) << ","
                                   << pred_camera_yaw << ","
                                   << pred_radius;
        } else {
            _mode1SelectionLogFile << "," << nan << "," << nan << "," << nan << ","
                                   << nan << "," << nan << "," << nan;
        }
    }

    for (int obs_index = 0; obs_index < kMode1SelectionLogSlots; ++obs_index) {
        for (int armor_id = 0; armor_id < kMode1SelectionLogSlots; ++armor_id) {
            _mode1SelectionLogFile << "," << match_cost_matrix[obs_index][armor_id];
        }
    }

    _mode1SelectionLogFile << "\n";
    _mode1SelectionLogFile.flush();
}

Eigen::Vector3d Estimator::calArmorPos(double angle, double r, double z)
{
    return Eigen::Vector3d(_targetStateMat(EKF_CENTER_X) - r * cos(angle - _center_yaw_rad),
                           _targetStateMat(EKF_CENTER_Y) - r * sin(angle - _center_yaw_rad), z);
}

bool Estimator::Judge_by_IOU() { return true; }
bool Estimator::isDetected() { return !(tracker_state == LOST); }
bool Estimator::shotNow(chrono::high_resolution_clock::time_point nowTime)
{
    double delta = 1000 * std::chrono::duration_cast<std::chrono::duration<double>>(nowTime-_startTime).count();
    return (abs(delta - _shot_time) < 10);
}

void Estimator::showResults(AngleSolver & angleSolver)
{
    if (!_params->DEBUG_SWITCH) return;

    const double nis_threshold = 0.711;
    bool tracker_ready = false;
    double last_nis = std::numeric_limits<double>::quiet_NaN();
    double last_geometry_residual_xy_over_sigma_dr = std::numeric_limits<double>::quiet_NaN();
    double last_geometry_residual_z_over_sigma_h = std::numeric_limits<double>::quiet_NaN();
    int geometry_recovery_inflation_count = 0;
    int nis_failures = 0;
    int nis_window = 0;
    bool bad_convergence = false;
    bool diverged = false;

    if (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized()) {
        tracker_ready = true;
        last_nis = ypd_angle_tracker_->lastNis();
        last_geometry_residual_xy_over_sigma_dr =
            ypd_angle_tracker_->lastGeometryResidualXyOverSigmaDr();
        last_geometry_residual_z_over_sigma_h =
            ypd_angle_tracker_->lastGeometryResidualZOverSigmaH();
        geometry_recovery_inflation_count =
            ypd_angle_tracker_->geometryRecoveryInflationCount();
        nis_failures = ypd_angle_tracker_->recentNisFailureCount();
        nis_window = ypd_angle_tracker_->nisWindowSize();
        bad_convergence = ypd_angle_tracker_->badConvergence();
        diverged = ypd_angle_tracker_->diverged();
    }

    const bool nis_failed = tracker_ready && std::isfinite(last_nis) && last_nis > nis_threshold;
    const bool geometry_z_alert = tracker_ready &&
        std::isfinite(last_geometry_residual_z_over_sigma_h) &&
        last_geometry_residual_z_over_sigma_h > _params->YPD_GEOMETRY_RECOVERY_Z_SIGMA_THRESHOLD;
    const bool geometry_xy_alert = tracker_ready &&
        std::isfinite(last_geometry_residual_xy_over_sigma_dr) &&
        last_geometry_residual_xy_over_sigma_dr >
            _params->YPD_GEOMETRY_RECOVERY_XY_SIGMA_THRESHOLD;
    const cv::Scalar nis_color =
        (diverged || bad_convergence) ? cv::Scalar(0, 0, 255)
        : (nis_failed ? cv::Scalar(0, 215, 255) : cvex::GREEN);
    const cv::Scalar geometry_z_color =
        geometry_z_alert ? cv::Scalar(0, 0, 255) : cvex::GREEN;
    const cv::Scalar geometry_xy_color =
        geometry_xy_alert ? cv::Scalar(0, 215, 255) : cvex::GREEN;
    const cv::Scalar geometry_fix_color =
        geometry_recovery_inflation_count > 0 ? cv::Scalar(0, 215, 255) : cvex::GREEN;

    const std::string fps_dbg = "FPS: " + to_string(getFPS.getFreq());
    const std::string tracker_dbg = "tracker_state: " + trackerStateStr[tracker_state];
    const std::string update_dbg = "update_state: " + UpdateStateStr[update_state];
    const std::string ekf_mode_dbg = "tracker: ypd+angle";
    const std::string nis_dbg = tracker_ready
        ? "nis  : " + to_string(last_nis).substr(0, 6) + " / " +
              to_string(nis_threshold).substr(0, 6) + (nis_failed ? " fail" : " pass")
        : "nis  : n/a";
    const std::string nis_count_dbg = tracker_ready
        ? "nisN : " + to_string(nis_failures) + "/" + to_string(nis_window) +
              " bad=" + to_string(static_cast<int>(bad_convergence)) +
              " div=" + to_string(static_cast<int>(diverged))
        : "";
    const std::string geom_z_dbg =
        (tracker_ready && std::isfinite(last_geometry_residual_z_over_sigma_h)
            ? "geom z/h: " + to_string(last_geometry_residual_z_over_sigma_h).substr(0, 5) +
                  " / " +
                  to_string(_params->YPD_GEOMETRY_RECOVERY_Z_SIGMA_THRESHOLD).substr(0, 4)
            : "geom z/h: n/a");
    const std::string geom_xy_dbg =
        (tracker_ready && std::isfinite(last_geometry_residual_xy_over_sigma_dr)
            ? "geom xy/dr: " +
                  to_string(last_geometry_residual_xy_over_sigma_dr).substr(0, 5) + " / " +
                  to_string(_params->YPD_GEOMETRY_RECOVERY_XY_SIGMA_THRESHOLD).substr(0, 4)
            : "geom xy/dr: n/a");
    const std::string geom_fix_dbg =
        "geom fix: " + to_string(geometry_recovery_inflation_count);
    const std::string dt_dbg = "dt   : " + to_string(_dt * 1000).substr(0,4);
    const std::string xc_dbg = "xc   : " + to_string(_targetStateMat(0)).substr(0,4);
    const std::string yc_dbg = "yc   : " + to_string(_targetStateMat(2)).substr(0,4);
    const std::string zc_dbg = "zc   : " + to_string(_targetStateMat(4)).substr(0,4);
    const std::string vxc_dbg =
        (_targetStateMat(1) > 0 ? "vxc  :+" : "vxc  :") + to_string(_targetStateMat(1)).substr(0,5);
    const std::string vyc_dbg =
        (_targetStateMat(3) > 0 ? "vyc  :+" : "vyc  :") + to_string(_targetStateMat(3)).substr(0,5);
    const std::string vzc_dbg =
        (_targetStateMat(5) > 0 ? "vzc  :+" : "vzc  :") + to_string(_targetStateMat(5)).substr(0,5);
    const std::string yaw_dbg = "yaw  : " + to_string(_targetStateMat(6) * R2D).substr(0,5);
    const std::string vyaw_dbg =
        "vyaw : " + to_string(_targetStateMat(EKF_ARMOR_V_YAW)).substr(0,5);
    const std::string radius_dbg = "r    : " + to_string(_targetStateMat(8)).substr(0,4);
    const std::string yaw_diff_dbg =
        "yaw_diff: " + to_string(abs(_yawDiff)).substr(0,4) + " < " +
        to_string(max_match_yaw_diff_).substr(0,4);
    const std::string pos_diff_dbg =
        "position_diff: " + to_string(_posDiff).substr(0,4) + " < " +
        to_string(max_match_distance_).substr(0,4);

    const std::string fire_motion_status_dbg = "fire motion: uniform gate disabled";
    const std::string fire_motion_obs_dbg =
        "obs   : " + string(fire_observation_stable ? "stable" : "blocked") +
        " jump_hold=" + to_string(fire_observation_hold_frames_);

    if (_debugHud != nullptr) {

        _debugHud->upsert("estimator.fps", fps_dbg, "top_right", 10, "#c77dff");
        _debugHud->upsert("estimator.tracker_state", tracker_dbg, "top_right", 11, "#c77dff");
        _debugHud->upsert("estimator.update_state", update_dbg, "top_right", 12, "#c77dff");
        _debugHud->upsert(
            "estimator.ekf_mode", ekf_mode_dbg, "top_right", 13, "#ffe66d");
        _debugHud->upsert(
            "estimator.nis", nis_dbg, "top_right", 14,
            tracker_ready ? (diverged || bad_convergence ? "#ff4d4f"
                                                         : (nis_failed ? "#ffd166" : "#66ff66"))
                          : "#b4b4b4");
        if (!nis_count_dbg.empty()) {
            _debugHud->upsert(
                "estimator.nis_count", nis_count_dbg, "top_right", 15,
                diverged || bad_convergence ? "#ff4d4f"
                                            : (nis_failed ? "#ffd166" : "#66ff66"));
        }
        _debugHud->upsert(
            "estimator.geom_z_sigma", geom_z_dbg, "top_right", 16,
            geometry_z_alert ? "#ff4d4f" : "#66ff66");
        _debugHud->upsert(
            "estimator.geom_xy_sigma", geom_xy_dbg, "top_right", 17,
            geometry_xy_alert ? "#ffd166" : "#66ff66");
        _debugHud->upsert(
            "estimator.geom_fix_count", geom_fix_dbg, "top_right", 18,
            geometry_recovery_inflation_count > 0 ? "#ffd166" : "#66ff66");
        _debugHud->upsert("estimator.dt", dt_dbg, "bottom_left", 100, "#39c5bb");
        _debugHud->upsert("estimator.xc", xc_dbg, "bottom_left", 101, "#39c5bb");
        _debugHud->upsert("estimator.yc", yc_dbg, "bottom_left", 102, "#39c5bb");
        _debugHud->upsert("estimator.zc", zc_dbg, "bottom_left", 103, "#39c5bb");
        _debugHud->upsert("estimator.vxc", vxc_dbg, "bottom_left", 104, "#39c5bb");
        _debugHud->upsert("estimator.vyc", vyc_dbg, "bottom_left", 105, "#39c5bb");
        _debugHud->upsert("estimator.vzc", vzc_dbg, "bottom_left", 106, "#39c5bb");
        _debugHud->upsert("estimator.yaw", yaw_dbg, "bottom_left", 107, "#39c5bb");
        _debugHud->upsert("estimator.vyaw", vyaw_dbg, "bottom_left", 108, "#39c5bb");
        _debugHud->upsert("estimator.radius", radius_dbg, "bottom_left", 109, "#39c5bb");
        _debugHud->upsert("estimator.yaw_diff", yaw_diff_dbg, "bottom_left", 110, "#c77dff");
        _debugHud->upsert("estimator.pos_diff", pos_diff_dbg, "bottom_left", 111, "#c77dff");
        _debugHud->upsert(
            "fire.motion.status", fire_motion_status_dbg, "bottom_right", 150, "#66ff66");
        _debugHud->upsert(
            "fire.observation.status", fire_motion_obs_dbg, "bottom_right", 151,
            fire_observation_stable ? "#66ff66" : "#ffd166");
    } else {
        putText(_debugImg, fps_dbg, Point(25,100), FONT_HERSHEY_PLAIN, 2, PURPLE);
        putText(_debugImg, tracker_dbg, Point(700, 50), FONT_HERSHEY_PLAIN, 2, PURPLE);
        putText(_debugImg, update_dbg, Point(700, 70), FONT_HERSHEY_PLAIN, 2, PURPLE,
                update_state==ARMORJUMP ? 4 : 1);
        putText(
            _debugImg, ekf_mode_dbg, Point(700, 90), FONT_HERSHEY_PLAIN, 2,
            cvex::YELLOW);
        if (tracker_ready) {
            putText(
                _debugImg, nis_dbg, Point(700, 110), FONT_HERSHEY_PLAIN, 1.6, nis_color,
                nis_failed ? 2 : 1);
            putText(
                _debugImg, nis_count_dbg, Point(700, 128), FONT_HERSHEY_PLAIN, 1.6,
                nis_color, 1);
        } else {
            putText(
                _debugImg, nis_dbg, Point(700, 110), FONT_HERSHEY_PLAIN, 1.6,
                cv::Scalar(180, 180, 180), 1);
        }
        putText(
            _debugImg, geom_z_dbg, Point(700, 146), FONT_HERSHEY_PLAIN, 1.6,
            geometry_z_color, 1);
        putText(
            _debugImg, geom_xy_dbg, Point(700, 164), FONT_HERSHEY_PLAIN, 1.6,
            geometry_xy_color, 1);
        putText(
            _debugImg, geom_fix_dbg, Point(700, 182), FONT_HERSHEY_PLAIN, 1.6,
            geometry_fix_color, 1);
        putText(_debugImg, dt_dbg, Point(10, 280), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, xc_dbg, Point(10, 300), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, yc_dbg, Point(10, 320), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, zc_dbg, Point(10, 340), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, vxc_dbg, Point(10, 360), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, vyc_dbg, Point(10, 380), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, vzc_dbg, Point(10, 400), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, yaw_dbg, Point(10, 420), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, vyaw_dbg, Point(10, 440), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, radius_dbg, Point(10, 460), FONT_HERSHEY_PLAIN, 2, cvex::MIKU);
        putText(_debugImg, yaw_diff_dbg, Point(10, 500), FONT_HERSHEY_PLAIN, 2, PURPLE);
        putText(_debugImg, pos_diff_dbg, Point(10, 520), FONT_HERSHEY_PLAIN, 2, PURPLE);
        putText(
            _debugImg, fire_motion_status_dbg, Point(700, 220), FONT_HERSHEY_PLAIN, 1.6,
            cvex::GREEN, 1);
        putText(
            _debugImg, fire_motion_obs_dbg, Point(700, 238), FONT_HERSHEY_PLAIN, 1.6,
            fire_observation_stable ? cvex::GREEN : cv::Scalar(0, 211, 255), 1);
    }

    auto project_tracker_point = [&](const Eigen::VectorXd& state, Point2f& out) -> bool {
        if (state.size() < 9 || state.squaredNorm() <= 0.0) return false;
        Eigen::Vector3d pos3d = getArmorPositionFromState(state);
        if (!std::isfinite(pos3d.x()) || !std::isfinite(pos3d.y()) || !std::isfinite(pos3d.z())) return false;
        out = angleSolver.calculateImagePoint(pos3d);
        return std::isfinite(out.x) && std::isfinite(out.y);
    };

    Point2f active_pt;
    const bool has_active_point = project_tracker_point(_targetStateMat, active_pt);
    if (has_active_point) {
        drawMarker(_debugImg, active_pt, cvex::YELLOW, cv::MARKER_CROSS, 20, 2);
        putText(_debugImg, "EKF", active_pt + Point2f(8.0f, -8.0f), FONT_HERSHEY_PLAIN, 1.2, cvex::YELLOW, 1);
    }

    int obs_drawn = 0;
    double obs_err_sum = 0.0;
    for (size_t i = 0; i < _current_obs_armors.size(); ++i)
    {
        const auto& obs = _current_obs_armors[i];
        Point2f obs_pt = angleSolver.calculateImagePoint(obs.pos);
        if (!std::isfinite(obs_pt.x) || !std::isfinite(obs_pt.y)) continue;

        circle(_debugImg, obs_pt, 5, cv::Scalar(0, 255, 0), -1);
        std::string obs_label = "obs" + to_string(i);
        if (_params->DEBUG_DRAW_OBS_MATCH_LABELS &&
            i < _current_obs_match_ids.size() && _current_obs_match_ids[i] >= 0) {
            obs_label += "->id" + to_string(_current_obs_match_ids[i]);
        }
        putText(
            _debugImg, obs_label, obs_pt + Point2f(6.0f, 14.0f),
            FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1);

        if (has_active_point) {
            line(_debugImg, obs_pt, active_pt, cv::Scalar(0, 200, 200), 1);
            obs_err_sum += cv::norm(obs_pt - active_pt);
            obs_drawn++;
        }
    }

    if (obs_drawn > 0) {
        const double mean_obs_err = obs_err_sum / static_cast<double>(obs_drawn);
        const std::string obs_err_dbg =
            "obs->ekf err(px): " + to_string(mean_obs_err).substr(0, 5) +
            " n=" + to_string(obs_drawn);
        if (_debugHud != nullptr) {
            _debugHud->upsert("estimator.obs_err", obs_err_dbg, "bottom_left", 112, "#ffe66d");
        } else {
            putText(
                _debugImg, obs_err_dbg, Point(10, 560), FONT_HERSHEY_PLAIN, 1.5,
                cvex::YELLOW, 2);
        }
    }

    if (!_detectedFlag || !_trackedArmor) return;

    Point2f robot_center_bottom_on_image = robot_center_top_on_image + Point2f(0, 200);
    arrowedLine(_debugImg, robot_center_bottom_on_image, robot_center_top_on_image, cvex::CYAN);

    Point2f robot_center_on_image = angleSolver.calculateImagePoint(_robotCenter);
    if (_params->DRAW_TARGET_SWITCH) {
        circle(_debugImg, _trackedArmor->center, 11, Scalar(55, 255, 55), 3);
    }
    // World-yaw positive is counter-clockwise in math coordinates.
    // On the image plane (y down), the perceived rotation direction is flipped.
    const double display_direction = robot.direction;
    auto p1 = robot_center_on_image + Point2f(10,0);
    auto p2 = robot_center_on_image - Point2f(10,0);
    if (display_direction > 0) swap(p1, p2);
    arrowedLine(_debugImg, p1, p2, cvex::CYAN, 2, 8, 0, 0.5);


    if (tracked_armors_num != ArmorsNum::OUTPOST_3)
    {
        vector<Eigen::Vector3d> points;
        vector<double> thetas;
        const int tracked_idx = static_cast<int>(robot.idx);
        auto wrap_idx = [](int idx) {
            idx %= 4;
            if (idx < 0) idx += 4;
            return idx;
        };
        for (int i = 0; i < 4; i++)
        {
            int idx = wrap_idx(display_direction < 0 ? tracked_idx - i : tracked_idx + i);

            double theta;
            if (display_direction < 0)
                theta = _targetStateMat(EKF_ARMOR_YAW) - (i + 2) * CV_PI / 2;
            else
                theta = _targetStateMat(EKF_ARMOR_YAW) + (i + 2) * CV_PI / 2;

            double r = robot.armor_rs[idx] * 0.9;

            Eigen::Vector3d radius_vec(r * cos(theta), r * sin(theta), 0);
            Eigen::Vector3d test_point = _robotCenter + radius_vec;

            test_point.z() = robot.armor_zs[idx];
            if (!robot.determined[idx]) break;
            thetas.push_back(theta);
            points.push_back(test_point);
        }

        angleSolver.drawArmor(
            _debugImg, _trackedArmor->type, points, thetas, display_direction, cvex::PINK);
    }
    else
    {
        vector<Eigen::Vector3d> points;
        vector<double> thetas;
        const double display_vyaw = -_targetStateMat(EKF_ARMOR_V_YAW);
        bool used_tracker_geometry = false;
        if (ypd_angle_tracker_ && ypd_angle_tracker_->isInitialized()) {
            const auto armor_xyza_list = ypd_angle_tracker_->getPredictedArmorStates();
            if (armor_xyza_list.size() == 3) {
                for (const auto& armor_xyza : armor_xyza_list) {
                    points.push_back(armor_xyza.head<3>());
                    thetas.push_back(armor_xyza(3));
                }
                used_tracker_geometry = true;
            }
        }
        if (!used_tracker_geometry) {
            for (int i = 0; i < 3; i++) {
                double r = _targetStateMat(EKF_ROTATION_RADIUS);
                double theta = display_vyaw <= 0
                    ? _targetStateMat(EKF_ARMOR_YAW) - (i + 0.5) * CV_PI / 3 * 2
                    : _targetStateMat(EKF_ARMOR_YAW) + (i + 0.5) * CV_PI / 3 * 2;
                Eigen::Vector3d radius_vec(r * cos(theta), r * sin(theta), 0);
                Eigen::Vector3d test_point = _robotCenter + radius_vec;
                test_point.z() = _targetStateMat(EKF_ARMOR_Z);
                thetas.push_back(theta);
                points.push_back(test_point);
            }
        }
        angleSolver.drawArmor(
            _debugImg, _trackedArmor->type, points, thetas, display_vyaw, cvex::PINK);
    }

    for (int i = 0; i < int(tracked_armors_num); i++)
    {
        const std::string armor_z_dbg = to_string(robot.armor_zs[i]);
        if (_debugHud != nullptr) {
            _debugHud->upsert(
                "estimator.armor_z." + to_string(i), armor_z_dbg, "bottom_right", 120 + i,
                i == robot.idx ? "#f0a6ff" : "#c77dff");
        } else {
            putText(
                _debugImg, armor_z_dbg, Point(1000, 500 + i * 30), FONT_HERSHEY_PLAIN,
                1.5, PURPLE, i == robot.idx ? 2 : 1);
        }
    }

}

void Estimator::drawObservationMatchLabels(AngleSolver& angleSolver)
{
    if (_debugImg.empty() || !_params->DEBUG_DRAW_OBS_MATCH_LABELS) return;

    for (size_t i = 0; i < _current_obs_armors.size(); ++i) {
        const auto& obs = _current_obs_armors[i];
        Point2f obs_pt = angleSolver.calculateImagePoint(obs.pos);
        if (!std::isfinite(obs_pt.x) || !std::isfinite(obs_pt.y)) continue;

        std::string label = "obs" + to_string(i);
        if (i < _current_obs_match_ids.size() && _current_obs_match_ids[i] >= 0) {
            label += "->id" + to_string(_current_obs_match_ids[i]);
        }

        putText(
            _debugImg, label, obs_pt + Point2f(6.0f, -8.0f),
            FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 255), 2);
    }
}

} // namespace rm
