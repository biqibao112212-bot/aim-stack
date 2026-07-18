#ifndef RM_SECOND_ORDER_POSITION_MPC_H
#define RM_SECOND_ORDER_POSITION_MPC_H

#include "tinympc/tiny_api.hpp"

namespace rm
{

struct SecondOrderPositionMPCConfig
{
    double model_dt_s = 0.01;
    int horizon = 16;
    double track_q = 6.0;
    double rate_q = 0.02;
    double command_q = 0.5;
    bool command_track_target = true;
    double delta_r = 1.5;
    double input_gain = 1.0;
    double input_lag_s = 0.0;
    double wn_rad_s = 18.0;
    double zeta = 0.85;
    double max_rate_deg_s = 720.0;
    double max_lead_deg = 6.0;
    double max_state_rate_deg_s = 720.0;
    double output_stage_ratio = 0.0;
};

class SecondOrderPositionMPC
{
public:
    SecondOrderPositionMPC() = default;
    ~SecondOrderPositionMPC();

    void configure(const SecondOrderPositionMPCConfig& config);
    void reset(double measured_deg, double measured_rate_deg_s = 0.0);
    void setPreviewWindow(int start_index, int end_index);
    double update(
        double target_deg, double measured_deg, double measured_rate_deg_s, double applied_dt_s);
    double updateTrajectory(
        const Eigen::VectorXd& target_deg_traj, const Eigen::VectorXd& target_rate_deg_s_traj,
        double measured_deg, double measured_rate_deg_s, double applied_dt_s);
    double lastPreviewTrackingErrorDeg() const;
    bool lastPreviewTrackingValid() const;
    const Eigen::VectorXd& lastPredictedYawTrajectoryDeg() const;
    const Eigen::VectorXd& lastPredictedCommandTrajectoryDeg() const;
    const Eigen::VectorXd& lastReferenceTrajectoryDeg() const;
    const Eigen::VectorXd& lastReferenceRateTrajectoryDegS() const;
    int lastOutputStateIndex() const;
    double lastSolveUs() const;
    double lastUpdateUs() const;

private:
    SecondOrderPositionMPCConfig config_;
    TinySolver* solver_ = nullptr;
    bool initialized_ = false;
    double last_command_deg_ = 0.0;
    double last_effective_command_deg_ = 0.0;
    double last_measured_deg_ = 0.0;
    double filtered_rate_deg_s_ = 0.0;
    double last_preview_error_deg_ = 0.0;
    bool last_preview_valid_ = false;
    Eigen::VectorXd last_predicted_yaw_traj_deg_;
    Eigen::VectorXd last_predicted_command_traj_deg_;
    Eigen::VectorXd last_reference_traj_deg_;
    Eigen::VectorXd last_reference_rate_traj_deg_s_;
    int last_output_state_index_ = 0;
    double last_solve_us_ = 0.0;
    double last_update_us_ = 0.0;
    int preview_window_start_index_ = 2;
    int preview_window_end_index_ = 2;
    bool dirty_ = true;

    void destroySolver();
    void rebuildSolver();
    Eigen::VectorXd unwrapReferenceTrajectory(
        const Eigen::VectorXd& target_deg_traj, double reference_deg, int horizon) const;
    Eigen::VectorXd buildRateReferenceTrajectory(
        const Eigen::VectorXd& target_unwrapped_deg_traj,
        const Eigen::VectorXd& target_rate_deg_s_traj, double reference_dt_s, int horizon) const;

    static double normalizeAngleDeg(double angle_deg);
    static double closestEquivalentAngleDeg(double reference_deg, double angle_deg);
};

}  // namespace rm

#endif  // RM_SECOND_ORDER_POSITION_MPC_H
