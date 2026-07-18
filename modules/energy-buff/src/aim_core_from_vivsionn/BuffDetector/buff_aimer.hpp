#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include "buff_tracker.hpp"
#include "buff_runtime_config.hpp"
#include "second_order_position_mpc.h"
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>

class Params;

namespace auto_buff
{
struct AimCommand
{
    bool control = false;
    bool shoot = false;
    double yaw = 0.0;
    double pitch = 0.0;
};

struct PitchDebugSnapshot
{
    bool valid = false;
    Eigen::Vector3d target_in_buff = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d target_in_world = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    double horizontal_distance = std::numeric_limits<double>::quiet_NaN();
    double height = std::numeric_limits<double>::quiet_NaN();
    double raw_yaw = std::numeric_limits<double>::quiet_NaN();
    double ballistic_pitch = std::numeric_limits<double>::quiet_NaN();
    double pitch_offset = std::numeric_limits<double>::quiet_NaN();
    double pitch_rate = std::numeric_limits<double>::quiet_NaN();
    double pitch_lead = std::numeric_limits<double>::quiet_NaN();
    double solved_pitch = std::numeric_limits<double>::quiet_NaN();
    double command_pitch = std::numeric_limits<double>::quiet_NaN();
};

class Aimer
{
public:
    Aimer(const std::string& config_path);
    ~Aimer();
    void reset();

    AimCommand aim(BuffTracker& tracker, double bullet_speed);
    AimCommand aim(
        BuffTracker& tracker,
        double bullet_speed,
        double measured_yaw_deg,
        double measured_yaw_rate_deg_s,
        std::chrono::steady_clock::time_point timestamp,
        double pipeline_delay_s = 0.0);

    // ================= 新增：获取预测时间接口 =================
    // 获取系统的固定预测延迟
    double get_predict_time() const { return predict_time_; }
    double get_last_pipeline_delay() const { return last_pipeline_delay_; }
    double get_last_base_predict_time() const { return last_base_predict_time_; }
    double get_last_total_predict_time() const { return last_total_predict_time_; }

    // 获取上一次弹道解算出的子弹飞行时间
    double get_last_fly_time() const { return last_fly_time_; }
    const PitchDebugSnapshot& get_last_pitch_debug() const { return last_pitch_debug_; }
    void markShotFired();
    // =========================================================

private:
    bool solve_trajectory(
        BuffTracker& tracker, double v, double& yaw, double& pitch, double pipeline_delay_s);
    double estimate_target_pitch_rate(BuffTracker& tracker, double predict_dt_s) const;
    bool build_yaw_reference(
        BuffTracker& tracker,
        Eigen::VectorXd& yaw_ref_deg,
        Eigen::VectorXd& yaw_rate_ref_deg_s,
        double base_predict_time_s) const;
    void configure_yaw_mpc_from_params();
    double base_predict_time(double pipeline_delay_s) const;
    Eigen::Vector3d target_point_in_buff() const;
    double apply_yaw_mpc(
        BuffTracker& tracker,
        double raw_yaw_rad,
        double measured_yaw_deg,
        double measured_yaw_rate_deg_s,
        std::chrono::steady_clock::time_point timestamp,
        double base_predict_time_s);

    double yaw_offset_;
    double pitch_offset_;
    double fire_gap_time_;
    double predict_time_;
    double pitch_velocity_lead_time_ = 0.0;
    std::string config_path_;
    BuffRuntimeConfig buff_runtime_config_;

    double last_fly_time_ = 0.0; // 新增：保存子弹飞行时间
    double last_pipeline_delay_ = 0.0;
    double last_base_predict_time_ = 0.0;
    double last_total_predict_time_ = 0.0;
    PitchDebugSnapshot last_pitch_debug_;

    std::chrono::steady_clock::time_point last_fire_t_;
    std::optional<std::chrono::steady_clock::time_point> last_aim_t_;
    std::unique_ptr<Params> params_;
    rm::SecondOrderPositionMPC yaw_mpc_;
    rm::SecondOrderPositionMPCConfig yaw_mpc_config_;
    bool yaw_mpc_enabled_ = false;
    bool yaw_mpc_initialized_ = false;
    bool yaw_rate_feedback_initialized_ = false;
    double filtered_yaw_rate_deg_s_ = 0.0;
    double yaw_rate_lpf_alpha_ = 0.87;
};
}
#endif
