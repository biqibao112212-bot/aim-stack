#ifndef AUTO_BUFF__TRACKER_HPP
#define AUTO_BUFF__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include "buff_type.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_buff
{

class Voter
{
public:
    Voter() : clockwise_(0) {}
    void vote(double angle_last, double angle_now);
    int clockwise() const;
    void reset() { clockwise_ = 0; }
    void seed_direction(int direction);
private:
    int clockwise_;
};

class BuffTracker
{
public:
    struct DebugSnapshot
    {
        bool initialized = false;
        bool lost = true;
        bool fit_valid = false;
        bool switch_deferred = false;
        bool target_switched = false;
        int direction = 0;
        int history_size = 0;
        int selected_phase_index = -1;
        int phase_origin_index = -1;
        int reinit_reason = 0;
        double observed_roll = std::numeric_limits<double>::quiet_NaN();
        // Single-frame roll derivative after the hard speed clamp.
        double observed_speed_raw = std::numeric_limits<double>::quiet_NaN();
        // Tracker-side speed measurement used by blend / speed update.
        double observed_speed = std::numeric_limits<double>::quiet_NaN();
        double filtered_roll = std::numeric_limits<double>::quiet_NaN();
        // Current EKF/model-side speed estimate after updates.
        double filtered_speed = std::numeric_limits<double>::quiet_NaN();
        double filtered_speed_raw = std::numeric_limits<double>::quiet_NaN();
        double predicted_roll = std::numeric_limits<double>::quiet_NaN();
        // Short-horizon speed from predict_from_state().
        double predicted_speed = std::numeric_limits<double>::quiet_NaN();
        double curve_speed_now = std::numeric_limits<double>::quiet_NaN();
        double curve_speed_raw = std::numeric_limits<double>::quiet_NaN();
        double curve_speed_after_predict = std::numeric_limits<double>::quiet_NaN();
        double curve_speed_after_blade_update = std::numeric_limits<double>::quiet_NaN();
        double curve_speed_before_speed_update = std::numeric_limits<double>::quiet_NaN();
        double curve_speed_after_speed_update = std::numeric_limits<double>::quiet_NaN();
        double curve_phi_after_predict = std::numeric_limits<double>::quiet_NaN();
        double curve_phi_after_blade_update = std::numeric_limits<double>::quiet_NaN();
        double curve_phi_before_speed_update = std::numeric_limits<double>::quiet_NaN();
        double curve_phi_after_speed_update = std::numeric_limits<double>::quiet_NaN();
        double speed_measurement_predicted = std::numeric_limits<double>::quiet_NaN();
        double speed_measurement_innovation = std::numeric_limits<double>::quiet_NaN();
        double speed_measurement_noise = std::numeric_limits<double>::quiet_NaN();
        int speed_measurement_status = 0;
        double selected_roll_offset = std::numeric_limits<double>::quiet_NaN();
        double fit_a = std::numeric_limits<double>::quiet_NaN();
        double fit_w = std::numeric_limits<double>::quiet_NaN();
        double fit_phi = std::numeric_limits<double>::quiet_NaN();
        // Per-blade (phase slot 0-4) observation info captured during update_ekf().
        struct BladeDebugInfo {
            bool present = false;
            bool solved = false;   // true = per-blade PnP, false = rune-level fallback
            double assoc_global_roll_rad = std::numeric_limits<double>::quiet_NaN();
            bool selected = false;
        };
        std::array<BladeDebugInfo, 5> blade_observations{};
    };

    BuffTracker();
    explicit BuffTracker(const std::string& config_path);
    void reset();
    void update(
        const std::optional<PowerRune>& rune,
        std::chrono::steady_clock::time_point timestamp,
        bool switch_deferred = false);
    Eigen::VectorXd predict(double dt);
    Eigen::VectorXd predict_from_state(
        const Eigen::VectorXd& state,
        double dt,
        int direction,
        double source_relative_time_s = std::numeric_limits<double>::quiet_NaN()) const;
    Eigen::VectorXd get_state() const { return ekf_.x; }
    bool is_lost() const { return is_lost_; }
    double selected_target_roll_offset() const { return selected_target_roll_offset_; }
    double current_relative_time_s() const;
    DebugSnapshot debugSnapshot(double predict_dt_s) const;

    // 【核心修复】移至 Public，允许 Aimer 访问
    Eigen::Vector3d point_buff2world(const Eigen::Vector3d& point_in_buff, const Eigen::VectorXd& x) const;
    Eigen::Vector3d point_buff2world(
      const Eigen::Vector3d& point_in_buff, const Eigen::VectorXd& x, double roll_offset) const;
    Eigen::Vector3d target_point_buff2world(const Eigen::Vector3d& point_in_buff, const Eigen::VectorXd& x) const;

private:
    struct ObservedSpeedMeasurement
    {
            double raw_unbounded = std::numeric_limits<double>::quiet_NaN();
            double raw_clamped = std::numeric_limits<double>::quiet_NaN();
            double measurement_unbounded = std::numeric_limits<double>::quiet_NaN();
            double measurement = std::numeric_limits<double>::quiet_NaN();
            bool valid = false;
    };

  std::optional<double> estimate_big_observed_speed() const;
    ObservedSpeedMeasurement buildObservedSpeedMeasurement(
            PowerRune_type rune_type,
            double observed_roll_delta,
            double dt) const;
  void init_ekf(const PowerRune& p);
  void update_ekf(const PowerRune& p, double dt);
  bool should_reinitialize_big_reacquire(const PowerRune& p) const;
  bool should_segment_for_target_switch(const PowerRune& p) const;
  void reinitialize_big_reacquire(
      const PowerRune& p,
      std::chrono::steady_clock::time_point timestamp,
      int reason,
      double curve_pause_s = 0.0);
  void maybe_lock_phase_origin(int raw_phase_index);
  int logical_phase_index(int raw_phase_index) const;
  bool can_use_big_buff_curve_model(const Eigen::VectorXd& x) const;
  Eigen::VectorXd measurement_model(const Eigen::VectorXd& x) const;
  Eigen::VectorXd measurement_model(const Eigen::VectorXd& x, double roll_offset) const;

  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd& x) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd& x, double roll_offset) const;

    tools::ExtendedKalmanFilter ekf_;
    Voter voter_;

    struct ObsNode { double t; double angle; };
    std::deque<ObsNode> history_info_;

    std::chrono::steady_clock::time_point last_time_;
    std::chrono::steady_clock::time_point start_time_;
    double last_angle_ = 0.0;
    double last_observed_speed_raw_ = std::numeric_limits<double>::quiet_NaN();
    double last_observed_speed_ = std::numeric_limits<double>::quiet_NaN();
    double smoothed_curve_speed_ = std::numeric_limits<double>::quiet_NaN();
    double debug_curve_speed_after_predict_ = std::numeric_limits<double>::quiet_NaN();
    double debug_curve_speed_after_blade_update_ = std::numeric_limits<double>::quiet_NaN();
    double debug_curve_speed_before_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    double debug_curve_speed_after_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    double debug_curve_phi_after_predict_ = std::numeric_limits<double>::quiet_NaN();
    double debug_curve_phi_after_blade_update_ = std::numeric_limits<double>::quiet_NaN();
    double debug_curve_phi_before_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    double debug_curve_phi_after_speed_update_ = std::numeric_limits<double>::quiet_NaN();
    double debug_speed_measurement_predicted_ = std::numeric_limits<double>::quiet_NaN();
    double debug_speed_measurement_innovation_ = std::numeric_limits<double>::quiet_NaN();
    double debug_speed_measurement_noise_ = std::numeric_limits<double>::quiet_NaN();
    int debug_speed_measurement_status_ = 0;
    double selected_target_roll_offset_ = 0.0;
    int phase_origin_index_ = -1;
    double lost_timeout_s_ = 0.35;
    double big_lost_timeout_s_ = 0.08;
    double big_model_reset_timeout_s_ = 0.35;
    bool big_curve_ekf_fit_enabled_ = true;
    double big_phase_process_noise_ = 0.02;
    double big_a_process_noise_ = 0.0001;
    double big_w_process_noise_ = 0.00001;
    double big_measurement_noise_scale_ = 4.0;
    int big_phi_seed_frames_ = 15;
    bool big_speed_measurement_enabled_ = true;
    double big_speed_measurement_noise_ = 0.20;
    double big_speed_measurement_gate_ = 1.20;
    double big_speed_measurement_adaptive_scale_ = 0.0;
    double big_speed_measurement_correction_limit_ = 0.0;
    double big_curve_speed_slew_limit_ = 0.0;
    int big_speed_measurement_window_samples_ = 12;
    double big_speed_measurement_window_s_ = 0.20;
    int big_speed_measurement_min_history_ = 0;
    double big_curve_phi_correction_limit_ = 0.0;
    bool big_curve_prediction_enabled_ = false;
    std::optional<std::chrono::steady_clock::time_point> lost_since_;
    std::optional<PowerRune_type> last_rune_type_;
    bool last_switch_deferred_ = false;
    bool last_target_switched_ = false;
    int selected_phase_index_ = -1;
    int last_reinit_reason_ = 0;
    bool is_initialized_ = false;
    bool is_lost_ = true;
    // Per-blade observation data captured in update_ekf(), keyed by phase slot 0-4.
    std::array<DebugSnapshot::BladeDebugInfo, 5> last_blade_debug_{};
};

} // namespace auto_buff

#endif
