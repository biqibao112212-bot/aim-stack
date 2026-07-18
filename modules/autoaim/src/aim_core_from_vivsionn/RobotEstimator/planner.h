#ifndef RM_PLANNER_H
#define RM_PLANNER_H

#include <Eigen/Dense>

#include <optional>

#include "target.h"

namespace rm
{

struct PlannerConfig
{
    double yaw_offset = 0.0;
    double execution_delay_s = 0.0;
    double preview_dt_s = 0.01;
    int preview_horizon = 16;
    double armor_enter_angle_deg = 50.0;
    double armor_leave_angle_deg = 30.0;
};

struct Plan
{
    bool control = false;
    double target_yaw = 0.0;
    double target_yaw_vel = 0.0;
    int selected_armor_index = -1;
    double estimated_fly_time_s = 0.0;
    bool impact_delta_angle_valid = false;
    double impact_delta_angle = 0.0;
    Eigen::VectorXd impact_delta_angle_ref;
    Eigen::VectorXd yaw_ref;
    Eigen::VectorXd yaw_rate_ref;
    Eigen::Vector3d target_pos = Eigen::Vector3d::Zero();
};

class Planner
{
public:
    Planner() = default;
    explicit Planner(const PlannerConfig& config);

    void configure(const PlannerConfig& config);
    Plan plan(Target target, double bullet_speed) const;
    Plan plan(std::optional<Target> target, double bullet_speed) const;

private:
    struct AimSolution
    {
        bool valid = false;
        double yaw = 0.0;
        double fly_time = 0.0;
        double impact_delta_angle = 0.0;
        double planar_distance = 0.0;
        int selected_index = -1;
        Eigen::Vector3d aim_pos = Eigen::Vector3d::Zero();
    };

    PlannerConfig config_;
    mutable int last_selected_armor_index_ = -1;
    mutable std::optional<double> last_fly_time_ = std::nullopt;
    mutable Eigen::VectorXd last_yaw_ref_;
    mutable int last_armor_count_ = -1;
    mutable int source_aim_lock_id_ = -1;

    double limitRad(double angle) const;
    double closestEquivalentRad(double reference, double angle) const;
    AimSolution solveAimForArmor(
        Target target, double bullet_speed, int armor_index,
        std::optional<double> initial_fly_time = std::nullopt) const;
    double scoreAimSolution(
        const Target& target, const AimSolution& solution, int preferred_index,
        std::optional<double> continuity_yaw = std::nullopt) const;
    AimSolution solveAim(
        Target target, double bullet_speed, int preferred_index = -1,
        std::optional<double> initial_fly_time = std::nullopt,
        std::optional<double> continuity_yaw = std::nullopt) const;
    int selectSourceAimIndex(const Target& target, int* lock_id) const;
    AimSolution solveSourceAim(
        Target target, double bullet_speed, int* lock_id,
        std::optional<double> initial_fly_time = std::nullopt) const;
    void buildYawReference(
        Target target, double bullet_speed, Eigen::VectorXd* yaw_ref,
        Eigen::VectorXd* yaw_rate_ref, Eigen::Vector3d* first_target_pos,
        int initial_preferred_index = -1, int* first_selected_index = nullptr,
        std::optional<double> initial_fly_time = std::nullopt,
        double* first_fly_time = nullptr,
        double* first_impact_delta_angle = nullptr,
        Eigen::VectorXd* impact_delta_angle_ref = nullptr) const;
};

} // namespace rm

#endif // RM_PLANNER_H
