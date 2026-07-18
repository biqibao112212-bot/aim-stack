// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__TRACKER_HPP_
#define ARMOR_PROCESSOR__TRACKER_HPP_

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <array>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "AngleSolver.h"
#include "generalDeclaration.h"
#include "opencv_extended.h"
#include "params.h"
#include "planner.h"
#include "quadratic_regressor.h"
#include "target.h"
#include "ypd_angle_tracker.h"

using cvex::PURPLE;
using namespace cv;

namespace RobotEstimator
{
struct ArmorObservation
{
    Eigen::Vector3d pos;
    double yaw = 0.0;
};
}  // namespace RobotEstimator

namespace rm
{

enum class ArmorsNum { NORMAL_4 = 4, OUTPOST_3 = 3 };
enum AIM_MODE
{
    TRACK_ARMOR_WITHIN_RANGE,
    TRACK_CENTER,
    TRACK_ARMOR_ALL_TIME,
    TRACK_HARD
};

class Estimator
{
public:
    Estimator();

    std::array<double, 4> plate_z_array_ = {0.0, 0.0, 0.0, 0.0};
    QuadraticRegressor reg_xc{20};
    QuadraticRegressor reg_yc{20};
    QuadraticRegressor reg_z{20};
    QuadraticRegressor reg_yaw{25};
    QuadraticRegressor reg_r{40};

    std::vector<RobotEstimator::ArmorObservation> _current_obs_armors;
    std::vector<Armor> _current_tracker_input_armors;
    std::vector<int> _current_obs_match_ids;
    int _current_primary_observation_index = -1;
    std::ofstream _mode1SelectionLogFile;
    bool _mode1SelectionLogEnabled = false;
    bool ypd_reset_this_frame = false;
    bool ypd_reset_diverged = false;
    bool ypd_reset_bad_convergence = false;
    std::string ypd_reset_reason = "none";
    double ypd_debug_primary_radius = std::numeric_limits<double>::quiet_NaN();
    double ypd_debug_secondary_radius = std::numeric_limits<double>::quiet_NaN();
    double ypd_debug_delta_radius = std::numeric_limits<double>::quiet_NaN();
    double ypd_debug_height_delta = std::numeric_limits<double>::quiet_NaN();
    int ypd_debug_update_count = 0;
    int ypd_debug_recent_nis_failures = 0;
    int ypd_debug_nis_window = 0;
    std::vector<int> ypd_debug_last_match_ids;
    Eigen::VectorXd ypd_debug_pre_predict_state11d = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd ypd_debug_prior_state11d = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd ypd_debug_posterior_state11d = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd ypd_debug_reset_state11d = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd ypd_debug_pre_predict_cov_diag = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd ypd_debug_prior_cov_diag = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd ypd_debug_posterior_cov_diag = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd ypd_debug_reset_cov_diag = Eigen::VectorXd::Zero(11);

    void loadFrame(Frame& frame);
    void loadMeta(const FrameMeta& frame_meta);
    void attachDebugImage(const cv::Mat& image);
    void attachDebugHud(DebugHudSnapshot* hud);
    void init(const std::vector<std::shared_ptr<Armor>> armors);
    bool update(const std::vector<std::shared_ptr<Armor>> armors);
    void getHitPoint();
    void aimTranposeTarget();
    void aimRotatingTarget(AIM_MODE aim_mode);
    void trackerUpdate(std::vector<std::shared_ptr<Armor>> armors, AngleSolver& anglesolver);
    bool Judge_by_IOU();
    Eigen::Vector3d calArmorPos(double angle, double r, double z);
    void showResults(AngleSolver& angleSolver);
    void drawObservationMatchLabels(AngleSolver& angleSolver);
    bool isDetected();
    bool shotNow(std::chrono::high_resolution_clock::time_point nowTime);
    void resetForTaskModeSwitch();

    std::shared_ptr<RobotEstimator::YpdAngleTracker> ypd_angle_tracker_;
    RobotMsg robot;

    int _trackingThreshold = 2;
    int _lostThreshold;

    enum State {
        LOST = 0,
        DETECTING = 1,
        TRACKING = 2,
        TEMP_LOST = 3
    } tracker_state;

    const string trackerStateStr[4] = {"lost", "detecting", "tracking", "temp lost"};

    enum UpdateState
    {
        MATCH_ARMOR_FOUND,
        ARMORJUMP,
        NOMATCH,
        LOW_CONFIDENCE
    } update_state;

    const string UpdateStateStr[4] = {"match armor found", "armor jump", "no match armor", "low confidence"};

    int jump_flag = 0;
    double latency = 0;
    double _bulletSpeed = 0;
    std::shared_ptr<Armor> _trackedArmor;
    std::shared_ptr<Armor> _anotherArmor;
    std::shared_ptr<Armor> _last_trackedArmor;
    double _trackedAngle_d;
    ArmorsNum tracked_armors_num;
    double distance_;
    double nowangle;
    double _posDiff;
    double _yawDiff;

    Eigen::VectorXd _measurement;
    Eigen::VectorXd _targetStateMat;
    Eigen::VectorXd _targetState11d;
    Eigen::VectorXd _legacyTargetStateMat;
    Eigen::VectorXd _legacyTargetState11d;
    Eigen::VectorXd _ypdAngleTargetStateMat;
    Eigen::VectorXd _ypdAngleTargetState11d;
    Eigen::Vector3d _last_obs_armor_pos;

    bool armor_flag = 0;
    double _last_usb_time;
    bool _detectedFlag;
    bool fire_motion_uniform = true;
    bool fire_observation_stable = false;
    double fire_motion_center_accel_metric = std::numeric_limits<double>::quiet_NaN();
    double fire_motion_omega_metric = std::numeric_limits<double>::quiet_NaN();
    double fire_motion_translation_burst_metric = std::numeric_limits<double>::quiet_NaN();
    double fire_motion_translation_drift_metric = std::numeric_limits<double>::quiet_NaN();
    bool fire_motion_translation_blocked = false;
    bool _isSpinning = false;
    Eigen::Vector3d _hitPoint;
    Eigen::Vector3d _aimPoint;
    Eigen::Vector3d _robotCenter;
    double _gimbal_pitch_d = 0;
    double _center_yaw_rad;
    bool high_heat_cap = 0;
    bool sentry = 0;
    bool hero = 0;
    Mat _debugImg;
    DebugHudSnapshot* _debugHud = nullptr;
    Eigen::Vector3d _pred_armor_pos;
    Eigen::Vector3d _last_predict_point;

    double _shot_time;

    std::shared_ptr<Planner> planner_;
    Target predict_target_;

private:
    void logMode1SelectionDebug();
    void resetTrackersForMode();
    void initEKF(const std::shared_ptr<Armor>& a);
    void refreshTrackerSnapshots();
    void applyActiveTrackerState();
    void syncRobotFromActiveTracker();
    bool activeTrackerNeedsReset() const;
    void clearYpdResetDebug();
    void captureYpdTrackerDebug(bool diverged, bool bad_convergence);
    void clearTrackerEstimateState();
    void resetActiveTrackerAndMarkLost();
    void resetFireSafetyState();
    void updateFireSafetyState();
    void updateArmorsNum(const std::shared_ptr<Armor>&);
    void update_tracker_state();
    void match_armors(const std::vector<std::shared_ptr<Armor>> armors);
    Eigen::Vector3d getArmorPositionFromState(const Eigen::VectorXd& x);

    double max_match_distance_ = 0.2;
    double max_match_yaw_diff_ = 0.8;

    int detect_count_ = 0;
    int lost_count_ = 0;
    int _tracked_neutral_color_frames = 0;
    int fire_observation_hold_frames_ = 0;
    int spin_count = 0;
    int low_speed = 0;
    int pour_water = 0;

    double _dt;
    double _lastTime;
    double _timeStamp;
    std::chrono::high_resolution_clock::time_point _startTime;
    std::chrono::high_resolution_clock::time_point _last_jump_time;
    int spin_aim_lock_id_ = -1;
    double interval_time;

    double _gimbal_yaw_d = 0;
    double _gimbal_yaw_speed = 0;

    AIM_MODE _aim_mode;

public:
    double v_t = 0;
    double v_r = 0;
    double v_xy = 0;
    MOVEMENT movement = STATIC;
    Eigen::Vector3d _last_velocity;
    Eigen::Vector3d target_ac;
    Point2f robot_center_top_on_image;
    Point2f robot_center_top_on_image_of_last_frame;
    Point2f robot_center_top_on_image_of_last_armor;

private:
    AngleSolver* _angleSolverPtr;

public:
    std::unique_ptr<Params> _params;
    Timer getFPS;
};

} // namespace rm

#endif // ARMOR_PROCESSOR__TRACKER_HPP_
