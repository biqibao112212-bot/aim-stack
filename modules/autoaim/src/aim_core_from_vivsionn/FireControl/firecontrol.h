#ifndef FIRECONTROL_H
#define FIRECONTROL_H

#include <chrono>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "generalDeclaration.h"
#include "AngleSolver.h"
#include "second_order_position_mpc.h"
#include "robotestimator.h"
#include "params.h"
#include "Serial.h"

#include "timer.h"

namespace rm
{

struct FireControlRuntimeStats
{
    double wall_control_dt_ms = 0.0;
    double frame_control_dt_ms = 0.0;
    double control_dt_ms = 0.0;
    double planner_us = 0.0;
    double yaw_ctrl_us = 0.0;
    double yaw_solve_us = 0.0;
    double pitch_ctrl_us = 0.0;
    double pitch_solve_us = 0.0;
    double total_us = 0.0;
    bool planner_active = false;
    bool preview_mpc_active = false;
    bool target_detected = false;
};

struct FireControlTargetSnapshot
{
    bool detected_flag = false;
    Eigen::Vector3d aim_point = Eigen::Vector3d::Zero();
    bool tracked_armor_valid = false;
    Eigen::Vector3d tracked_armor_position = Eigen::Vector3d::Zero();
    cv::Point2f tracked_armor_right = cv::Point2f(0.0f, 0.0f);
    cv::Point2f tracked_armor_left = cv::Point2f(0.0f, 0.0f);
    Eigen::VectorXd target_state11d = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd target_state_mat = Eigen::VectorXd::Zero(9);
    int tracked_armors_num = 4;
    bool target_jumped = false;
    MOVEMENT movement = STATIC;
    double state_age_s = 0.0;
    bool mcu_fire_permit = false;
    bool motion_uniform = false;
    bool observation_stable = false;
    double motion_translation_burst_metric = std::numeric_limits<double>::quiet_NaN();
    double motion_translation_drift_metric = std::numeric_limits<double>::quiet_NaN();
};

enum class FirePhaseMode
{
    None = 0,
    Auto = 1,
    Single = 2
};

class FireControl
{
public:
    FireControl();
    void loadFrame(rm::Frame & frame);
    void loadMeta(const rm::FrameMeta& frame_meta);
    void attachDebugImage(const cv::Mat& image);
    void attachDebugHud(DebugHudSnapshot* hud);

    ControlData calControlData(rm::AngleSolver & angleSolver, rm::Estimator & estimator);
    ControlData calControlData(
        rm::AngleSolver& angleSolver, const FireControlTargetSnapshot& target_snapshot);
    static FireControlTargetSnapshot makeTargetSnapshot(
        const rm::Estimator& estimator, double state_age_s = 0.0);
    void resetExecutionState();

    void showResults();
    /**
     * @brief 返回重力补偿后的pitch角度rad
     * @param v 弹速(m/s)
     * @param theta 仰角(rad)
     * @param _euclideanDistance 距离(mm)
     * @return
     */
    double gravityOffset(double theta, double euclideanDistance);
    double gravityOffset(Eigen::Vector3d pos);

    bool intervalFiring(double time);

    double getDistance(Eigen::Vector3d);
    uint8_t decideShotMode(const FireControlTargetSnapshot& target_snapshot, bool aim_point_valid);
    void resetAimCommandController();
    double applyAimCommandController(double raw_command_deg, double measured_deg, bool is_pitch);
    rm::PlannerConfig buildYawPlannerConfig() const;
    rm::SecondOrderPositionMPCConfig buildSecondOrderAimConfig(bool is_pitch) const;
    rm::Planner& ensureYawPlanner();
    const FireControlRuntimeStats& lastRuntimeStats() const;

private:
    void resetPlannerTrackingState(double fallback_yaw_deg);
    void clearLastYawPlan();
    void clearAutoFireState(bool clear_last_command);
    void clearFirePhaseState();
    void applyNoTargetControlState();
    void updateYawFireCheckFromMpc();
    void applyStaticDirectYawControl(bool planner_yaw_valid);
    double computeShotCheckStartTimeS(bool first_shot_candidate) const;
    double fireSlotPeriodS() const;
    void advanceFirePhase(const std::chrono::steady_clock::time_point& now);
    void startFirePhase(
        FirePhaseMode mode, const std::chrono::steady_clock::time_point& now,
        double first_slot_time_s);
    double firePhaseFirstSlotTimeS(const std::chrono::steady_clock::time_point& now) const;
    int countViableShotSlots(
        double tolerance_deg, double first_slot_time_s, double target_omega_rad_s,
        bool require_impact_angle_gate, bool mcu_fire_permit,
        double* first_slot_error_deg = nullptr,
        double* first_viable_slot_time_s = nullptr,
        bool* first_slot_impact_gate = nullptr,
        double* first_slot_impact_delta_deg = nullptr) const;
    bool tryBuildYawPlan(const FireControlTargetSnapshot& target_snapshot, rm::Plan* yaw_plan);
    void applyYawPlanOrFallback(
        const rm::Plan& yaw_plan, bool planner_yaw_valid, Eigen::Vector3d* pitch_target_pos);
    void appendFireControlDebugLog(const FireControlTargetSnapshot& target_snapshot);


public:
    std::unique_ptr<Params> _params;

    rm::ControlData _controlData;

    double _nowTime;///< ms
    rm::GimbalData    _gimbalPose;
    bool            mcu_fire_permit_ = false;
    double last_pitch, last_yaw;
    double          _yaw_speed = 0.0;
    bool            yaw_speed_feedback_initialized_ = false;
    double          _bulletSpeed;
    int             _heat_cap;
    Mat             _debugImg;
    DebugHudSnapshot* _debugHud = nullptr;
    bool           first_shoot;
    tp         buff_shot_time;
    int jump_count = 0;
    bool has_last_auto_command_ = false;
    rm::ControlData last_auto_command_;
    bool fire_burst_active_ = false;
    bool fire_single_window_latched_ = false;
    std::chrono::steady_clock::time_point fire_burst_hold_deadline_{};
    std::chrono::steady_clock::time_point fire_auto_restart_cooldown_until_{};
    FirePhaseMode fire_phase_mode_ = FirePhaseMode::None;
    std::chrono::steady_clock::time_point fire_next_slot_tp_{};
    double fire_next_slot_delay_ms_ = std::numeric_limits<double>::quiet_NaN();
    bool fire_single_pulse_sent_ = false;
    bool fire_mechanical_hold_active_ = false;
    double fire_tolerance_deg_ = 0;
    double fire_cmd_delta_deg_ = 0;
    double fire_follow_error_deg_ = 0;
    double fire_pitch_cmd_delta_deg_ = 0;
    double fire_pitch_follow_error_deg_ = 0;
    double fire_first_slot_error_deg_ = std::numeric_limits<double>::quiet_NaN();
    double yaw_preview_tracking_error_deg_ = 0;
    double yaw_fire_check_error_deg_ = 0;
    int fire_viable_slot_count_ = 0;
    bool fire_gate_command_stable_ = false;
    bool fire_gate_follow_ = false;
    bool fire_gate_mcu_permit_ = false;
    bool fire_gate_valid_ = false;
    bool fire_gate_preview_ = false;
    bool fire_gate_impact_angle_ = false;
    bool fire_gate_motion_uniform_ = false;
    bool fire_gate_observation_stable_ = false;
    bool fire_gate_slot_window_ = false;
    bool yaw_preview_tracking_valid_ = false;
    bool yaw_fire_check_valid_ = false;
    bool yaw_static_mpc_bypass_active_ = false;
    double raw_command_yaw_deg_ = 0;
    double raw_command_pitch_deg_ = 0;
    double filtered_command_yaw_deg_ = 0;
    double filtered_command_pitch_deg_ = 0;
    bool yaw_planner_active_ = false;
    bool last_yaw_plan_valid_ = false;
    int last_plan_selected_armor_index_ = -1;
    double last_plan_execution_delay_s_ = std::numeric_limits<double>::quiet_NaN();
    double last_plan_estimated_fly_time_s_ = std::numeric_limits<double>::quiet_NaN();
    double last_plan_target_yaw_deg_ = std::numeric_limits<double>::quiet_NaN();
    double last_plan_target_yaw_vel_deg_s_ = std::numeric_limits<double>::quiet_NaN();
    double last_plan_impact_delta_angle_deg_ = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d last_plan_target_pos_ =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d last_plan_zero_vxy_target_pos_ =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    double last_plan_zero_vxy_delta_m_ = std::numeric_limits<double>::quiet_NaN();
    int yaw_preview_start_index_ = 2;
    int yaw_preview_end_index_ = 2;
    int yaw_fire_check_index_ = 0;
    double yaw_fire_check_time_ms_ = 0.0;
    double legacy_command_yaw_deg_ = 0.0;
    double base_planner_yaw_deg_ = 0.0;
    bool fire_impact_delta_angle_valid_ = false;
    double fire_impact_delta_angle_deg_ = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd fire_impact_delta_angle_ref_deg_;
    bool debug_firecontrol_log_enabled_ = false;
    double wall_control_dt_sec_ = 0.0;
    double frame_control_dt_sec_ = 0.0;
    double control_dt_sec_ = 0.01;
    double simulator_state_age_s_ = 0.0;
    double last_control_timestamp_ms_ = 0.0;
    std::chrono::steady_clock::time_point last_control_tp_{};
    bool control_timer_initialized_ = false;
    rm::SecondOrderPositionMPC yaw_second_order_mpc_;
    rm::SecondOrderPositionMPC pitch_second_order_mpc_;
    std::unique_ptr<rm::Planner> yaw_planner_;
    FireControlRuntimeStats last_runtime_stats_;

    Timer timer;
    tp shoot_time;
public:
    double pitch_dev_d = 0;
    double yaw_dev_d = 0;

    double target_pitch = 0;

    // 前哨站

    double direction = 0; // 按照正负判断旋转方向
    deque<AngleWithTime> outpostQ;
    double outpost(double now_time);
};

}

#endif // FIRECONTROL_H
