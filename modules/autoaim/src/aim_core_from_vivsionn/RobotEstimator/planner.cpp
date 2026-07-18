#include "planner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rm
{

namespace {

constexpr double kGravity = 9.7833;
constexpr double kDegToRad = M_PI / 180.0;
constexpr double kLowSpeedSelectionOmegaRadS = 4.0;
constexpr double kLowSpeedMaxDeltaRad = 60.0 * kDegToRad;
constexpr double kOutpostComingAngleRad = 70.0 * kDegToRad;
constexpr double kOutpostLeavingAngleRad = 30.0 * kDegToRad;
constexpr double kAimSwitchPenaltyRad = 10.0 * kDegToRad;
constexpr double kAimContinuityPenaltyWeight = 1.0;
constexpr double kAimDirectionalPenaltyWeight = 0.2;
constexpr double kAimExcessAnglePenaltyWeight = 4.0;
constexpr double kAimRelaxedAngleExtraRad = 12.0 * kDegToRad;
constexpr double kAimSolveToleranceS = 1e-3;
constexpr int kAimSolveMaxIterations = 8;

bool same_config(const PlannerConfig& lhs, const PlannerConfig& rhs)
{
    return lhs.yaw_offset == rhs.yaw_offset &&
           lhs.execution_delay_s == rhs.execution_delay_s &&
           lhs.preview_dt_s == rhs.preview_dt_s &&
           lhs.preview_horizon == rhs.preview_horizon &&
           lhs.armor_enter_angle_deg == rhs.armor_enter_angle_deg &&
           lhs.armor_leave_angle_deg == rhs.armor_leave_angle_deg;
}

struct BallisticTrajectory
{
    bool unsolvable = false;
    double fly_time = 0.0;

    BallisticTrajectory(double v0, double d, double h)
    {
        if (v0 <= 1e-6 || d <= 1e-6) {
            unsolvable = true;
            return;
        }

        const double a = kGravity * d * d / (2.0 * v0 * v0);
        const double b = -d;
        const double c = a + h;
        const double delta = b * b - 4.0 * a * c;

        if (delta < 0.0) {
            unsolvable = true;
            return;
        }

        const double tan_pitch_1 = (-b + std::sqrt(delta)) / (2.0 * a);
        const double tan_pitch_2 = (-b - std::sqrt(delta)) / (2.0 * a);
        const double pitch_1 = std::atan(tan_pitch_1);
        const double pitch_2 = std::atan(tan_pitch_2);
        const double t_1 = d / (v0 * std::cos(pitch_1));
        const double t_2 = d / (v0 * std::cos(pitch_2));

        fly_time = t_1 < t_2 ? t_1 : t_2;
    }
};

} // namespace

Planner::Planner(const PlannerConfig& config)
{
    configure(config);
}

void Planner::configure(const PlannerConfig& config)
{
    if (same_config(config_, config)) return;
    config_ = config;
    last_selected_armor_index_ = -1;
    last_fly_time_ = std::nullopt;
    last_yaw_ref_.resize(0);
    last_armor_count_ = -1;
    source_aim_lock_id_ = -1;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed) const
{
    if (!target.has_value()) {
        last_selected_armor_index_ = -1;
        last_fly_time_ = std::nullopt;
        last_yaw_ref_.resize(0);
        last_armor_count_ = -1;
        source_aim_lock_id_ = -1;
        return {};
    }
    return plan(*target, bullet_speed);
}

Plan Planner::plan(Target target, double bullet_speed) const
{
    Plan plan;
    const int horizon = std::max(config_.preview_horizon, 4);
    if (horizon <= 0) return plan;

    const auto armor_xyza_list = target.armor_xyza_list();
    const int armor_count = static_cast<int>(armor_xyza_list.size());
    if (last_armor_count_ >= 0 && armor_count != last_armor_count_) {
        last_selected_armor_index_ = -1;
        last_fly_time_ = std::nullopt;
        last_yaw_ref_.resize(0);
        source_aim_lock_id_ = -1;
    }
    last_armor_count_ = armor_count;

    if (config_.execution_delay_s > 1e-6) {
        target.predict(config_.execution_delay_s);
    }

    if (bullet_speed < 10.0 || bullet_speed > 25.0) {
        bullet_speed = 22.0;
    }

    try {
        int first_selected_index = last_selected_armor_index_;
        double first_fly_time = 0.0;
        double first_impact_delta_angle = 0.0;
        buildYawReference(
            target, bullet_speed, &plan.yaw_ref, &plan.yaw_rate_ref, &plan.target_pos,
            last_selected_armor_index_, &first_selected_index, last_fly_time_, &first_fly_time,
            &first_impact_delta_angle, &plan.impact_delta_angle_ref);
        plan.selected_armor_index = first_selected_index;
        plan.estimated_fly_time_s = first_fly_time;
        plan.impact_delta_angle = first_impact_delta_angle;
        plan.impact_delta_angle_valid = plan.impact_delta_angle_ref.size() > 0;
        last_yaw_ref_ = plan.yaw_ref;
        last_selected_armor_index_ = first_selected_index;
        if (first_fly_time > 1e-6) {
            last_fly_time_ = first_fly_time;
        }
    } catch (const std::exception&) {
        return {};
    }

    if (plan.yaw_ref.size() == 0 || plan.yaw_rate_ref.size() == 0) return {};

    plan.control = true;
    plan.target_yaw = plan.yaw_ref(0);
    plan.target_yaw_vel = plan.yaw_rate_ref(0);
    return plan;
}

double Planner::limitRad(double angle) const
{
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double Planner::closestEquivalentRad(double reference, double angle) const
{
    return reference + limitRad(angle - reference);
}

Planner::AimSolution Planner::solveAimForArmor(
    Target target, double bullet_speed, int armor_index,
    std::optional<double> initial_fly_time) const
{
    const auto initial_armors = target.armor_xyza_list();
    if (initial_armors.empty()) {
        throw std::runtime_error("No valid armor for yaw planning");
    }
    if (armor_index < 0 || armor_index >= static_cast<int>(initial_armors.size())) {
        throw std::runtime_error("Invalid armor index for yaw planning");
    }

    const auto solve_at_time =
        [&](double fly_time) -> AimSolution {
            Target impact_target = target;
            if (fly_time > 1e-6) impact_target.predict(fly_time);

            const auto armor_xyza_list = impact_target.armor_xyza_list();
            if (armor_index < 0 || armor_index >= static_cast<int>(armor_xyza_list.size())) {
                throw std::runtime_error("Invalid armor index for yaw planning");
            }

            const Eigen::Vector3d xyz = armor_xyza_list[armor_index].head<3>();
            const double horizontal_distance = xyz.head<2>().norm();
            BallisticTrajectory bullet_traj(bullet_speed, horizontal_distance, xyz.z());
            if (bullet_traj.unsolvable) {
                throw std::runtime_error("Unsolvable bullet trajectory");
            }

            const Eigen::VectorXd impact_state = impact_target.ekf_x();
            const double center_yaw = std::atan2(impact_state[2], impact_state[0]);

            AimSolution solution;
            solution.valid = true;
            solution.yaw = limitRad(std::atan2(-xyz.y(), xyz.x()) + config_.yaw_offset);
            solution.fly_time = bullet_traj.fly_time;
            solution.impact_delta_angle = limitRad(armor_xyza_list[armor_index][3] - center_yaw);
            solution.planar_distance = horizontal_distance;
            solution.selected_index = armor_index;
            solution.aim_pos = xyz;
            return solution;
        };

    double fly_time = 0.0;
    if (initial_fly_time.has_value() && initial_fly_time.value() > 1e-6) {
        fly_time = initial_fly_time.value();
    } else {
        const Eigen::Vector3d xyz = initial_armors[armor_index].head<3>();
        const double horizontal_distance = xyz.head<2>().norm();
        BallisticTrajectory bullet_traj(bullet_speed, horizontal_distance, xyz.z());
        if (bullet_traj.unsolvable) {
            throw std::runtime_error("Unsolvable bullet trajectory");
        }
        fly_time = bullet_traj.fly_time;
    }

    for (int iter = 0; iter < kAimSolveMaxIterations; ++iter) {
        const AimSolution solution = solve_at_time(fly_time);
        if (std::abs(solution.fly_time - fly_time) < kAimSolveToleranceS) {
            return solution;
        }
        fly_time = solution.fly_time;
    }

    return solve_at_time(fly_time);
}

Planner::AimSolution Planner::solveAim(
    Target target, double bullet_speed, int preferred_index,
    std::optional<double> initial_fly_time, std::optional<double> continuity_yaw) const
{
    const auto initial_armors = target.armor_xyza_list();
    if (initial_armors.empty()) {
        throw std::runtime_error("No valid armor for yaw planning");
    }

    AimSolution best_solution;
    double best_score = std::numeric_limits<double>::infinity();
    AimSolution preferred_solution;
    double preferred_score = std::numeric_limits<double>::infinity();
    bool preferred_solution_valid = false;

    for (int armor_index = 0; armor_index < static_cast<int>(initial_armors.size()); ++armor_index) {
        AimSolution solution;
        try {
            solution = solveAimForArmor(target, bullet_speed, armor_index, initial_fly_time);
        } catch (const std::exception&) {
            continue;
        }

        const double score =
            scoreAimSolution(target, solution, preferred_index, continuity_yaw);
        if (score + 1e-9 < best_score) {
            best_score = score;
            best_solution = solution;
        }
        if (armor_index == preferred_index) {
            preferred_solution = solution;
            preferred_score = score;
            preferred_solution_valid = true;
        }
    }

    if (!best_solution.valid) {
        throw std::runtime_error("No valid armor for yaw planning");
    }
    if (preferred_solution_valid &&
        preferred_score <= best_score + 0.5 * kAimSwitchPenaltyRad) {
        return preferred_solution;
    }
    return best_solution;
}

int Planner::selectSourceAimIndex(const Target& target, int* lock_id) const
{
    const auto armor_xyza_list = target.armor_xyza_list();
    if (armor_xyza_list.empty()) return -1;
    if (!target.jumped) return 0;

    const Eigen::VectorXd state = target.ekf_x();
    if (state.size() < 9) return -1;

    const bool is_outpost = armor_xyza_list.size() == 3;
    const double center_yaw = std::atan2(state[2], state[0]);
    std::vector<double> delta_angle_list;
    delta_angle_list.reserve(armor_xyza_list.size());
    for (const auto& armor_xyza : armor_xyza_list) {
        delta_angle_list.push_back(limitRad(armor_xyza[3] - center_yaw));
    }

    if (std::abs(state[8]) <= 2.0 && !is_outpost) {
        std::vector<int> id_list;
        for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); ++i) {
            if (std::abs(delta_angle_list[i]) > kLowSpeedMaxDeltaRad) continue;
            id_list.push_back(i);
        }
        if (id_list.empty()) return -1;

        if (id_list.size() > 1 && lock_id != nullptr) {
            const int id0 = id_list[0];
            const int id1 = id_list[1];
            if (*lock_id != id0 && *lock_id != id1) {
                *lock_id =
                    (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1]))
                    ? id0
                    : id1;
            }
            if (*lock_id >= 0 &&
                *lock_id < static_cast<int>(armor_xyza_list.size())) {
                return *lock_id;
            }
        }

        if (lock_id != nullptr) *lock_id = -1;
        return id_list[0];
    }

    const double configured_enter_deg = std::max(0.0, config_.armor_enter_angle_deg);
    const double configured_leave_deg = std::max(0.0, config_.armor_leave_angle_deg);
    const double normal_coming_angle =
        std::max(configured_enter_deg, configured_leave_deg) * kDegToRad;
    const double normal_leaving_angle =
        std::min(configured_enter_deg, configured_leave_deg) * kDegToRad;
    const double coming_angle = is_outpost ? kOutpostComingAngleRad : normal_coming_angle;
    const double leaving_angle = is_outpost ? kOutpostLeavingAngleRad : normal_leaving_angle;
    const double omega = state.size() > 7 ? state[7] : 0.0;

    for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); ++i) {
        if (std::abs(delta_angle_list[i]) > coming_angle) continue;
        if (omega > 0.0 && delta_angle_list[i] < leaving_angle) return i;
        if (omega < 0.0 && delta_angle_list[i] > -leaving_angle) return i;
    }

    return -1;
}

Planner::AimSolution Planner::solveSourceAim(
    Target target, double bullet_speed, int* lock_id,
    std::optional<double> initial_fly_time) const
{
    double fly_time =
        initial_fly_time.has_value() && initial_fly_time.value() > 1e-6
        ? initial_fly_time.value()
        : 0.0;
    int previous_selected_index = -1;
    AimSolution solution;

    for (int iter = 0; iter < kAimSolveMaxIterations; ++iter) {
        Target impact_target = target;
        if (fly_time > 1e-6) {
            impact_target.predict(fly_time);
        }

        const int selected_index = selectSourceAimIndex(impact_target, lock_id);
        if (selected_index < 0) {
            throw std::runtime_error("No valid source aim armor");
        }

        std::optional<double> fly_time_hint = std::nullopt;
        if (fly_time > 1e-6) {
            fly_time_hint = fly_time;
        }
        solution = solveAimForArmor(target, bullet_speed, selected_index, fly_time_hint);
        if (!solution.valid) {
            throw std::runtime_error("Invalid source aim solution");
        }

        if (selected_index == previous_selected_index &&
            std::abs(solution.fly_time - fly_time) < kAimSolveToleranceS) {
            return solution;
        }
        if (std::abs(solution.fly_time - fly_time) < kAimSolveToleranceS) {
            return solution;
        }

        previous_selected_index = selected_index;
        fly_time = solution.fly_time;
    }

    if (!solution.valid) {
        throw std::runtime_error("No valid source aim solution");
    }
    return solution;
}

double Planner::scoreAimSolution(
    const Target& target, const AimSolution& solution, int preferred_index,
    std::optional<double> continuity_yaw) const
{
    if (!solution.valid) return std::numeric_limits<double>::infinity();

    const auto armor_xyza_list = target.armor_xyza_list();
    const Eigen::VectorXd state = target.ekf_x();
    const double omega = state.size() > 7 ? state[7] : 0.0;
    const bool is_outpost = armor_xyza_list.size() == 3;
    const bool low_speed = std::abs(omega) <= kLowSpeedSelectionOmegaRadS && !is_outpost;

    const double configured_enter_deg = std::max(0.0, config_.armor_enter_angle_deg);
    const double configured_leave_deg = std::max(0.0, config_.armor_leave_angle_deg);
    const double normal_coming_angle =
        std::max(configured_enter_deg, configured_leave_deg) * kDegToRad;
    const double normal_leaving_angle =
        std::min(configured_enter_deg, configured_leave_deg) * kDegToRad;
    const double coming_angle = is_outpost ? kOutpostComingAngleRad : normal_coming_angle;
    const double leaving_angle = is_outpost ? kOutpostLeavingAngleRad : normal_leaving_angle;
    const double preferred_angle_limit =
        (low_speed ? kLowSpeedMaxDeltaRad : coming_angle) + kAimRelaxedAngleExtraRad;

    const double abs_delta = std::abs(solution.impact_delta_angle);
    double score = abs_delta;
    if (abs_delta > preferred_angle_limit) {
        score += kAimExcessAnglePenaltyWeight * (abs_delta - preferred_angle_limit);
    }

    if (preferred_index >= 0 && solution.selected_index != preferred_index) {
        score += kAimSwitchPenaltyRad;
    }

    if (continuity_yaw.has_value()) {
        score +=
            kAimContinuityPenaltyWeight *
            std::abs(limitRad(solution.yaw - continuity_yaw.value()));
    }

    if (!low_speed) {
        if (omega > 0.0 && solution.impact_delta_angle > leaving_angle) {
            score +=
                kAimDirectionalPenaltyWeight * (solution.impact_delta_angle - leaving_angle);
        } else if (omega < 0.0 && solution.impact_delta_angle < -leaving_angle) {
            score +=
                kAimDirectionalPenaltyWeight * (-leaving_angle - solution.impact_delta_angle);
        }
    }

    score += 1e-3 * solution.planar_distance;
    return score;
}

void Planner::buildYawReference(
    Target target, double bullet_speed, Eigen::VectorXd* yaw_ref, Eigen::VectorXd* yaw_rate_ref,
    Eigen::Vector3d* first_target_pos, int initial_preferred_index, int* first_selected_index,
    std::optional<double> initial_fly_time, double* first_fly_time,
    double* first_impact_delta_angle, Eigen::VectorXd* impact_delta_angle_ref) const
{
    if (!yaw_ref || !yaw_rate_ref) {
        throw std::runtime_error("Yaw reference buffer is null");
    }

    const int horizon = std::max(config_.preview_horizon, 4);
    const double dt = std::max(config_.preview_dt_s, 1e-3);
    *yaw_ref = Eigen::VectorXd::Zero(horizon);
    *yaw_rate_ref = Eigen::VectorXd::Zero(horizon);
    if (impact_delta_angle_ref) {
        *impact_delta_angle_ref = Eigen::VectorXd::Zero(horizon);
    }

    (void)initial_preferred_index;
    int source_lock_id = source_aim_lock_id_;
    std::optional<double> fly_time_hint = initial_fly_time;
    std::optional<double> continuity_yaw = std::nullopt;
    if (last_yaw_ref_.size() > 1) {
        continuity_yaw = last_yaw_ref_(1);
    } else if (last_yaw_ref_.size() > 0) {
        continuity_yaw = last_yaw_ref_(0);
    }

    for (int i = 0; i < horizon; ++i) {
        if (i > 0) target.predict(dt);

        const AimSolution solution =
            solveSourceAim(target, bullet_speed, &source_lock_id, fly_time_hint);
        if (!solution.valid) {
            throw std::runtime_error("Invalid aim solution");
        }
        const double wrapped_yaw = solution.yaw;
        fly_time_hint = solution.fly_time;
        if (impact_delta_angle_ref) {
            (*impact_delta_angle_ref)(i) = solution.impact_delta_angle;
        }
        if (i == 0) {
            (*yaw_ref)(0) =
                continuity_yaw.has_value()
                    ? closestEquivalentRad(continuity_yaw.value(), wrapped_yaw)
                    : wrapped_yaw;
            continuity_yaw = (*yaw_ref)(0);
            if (first_target_pos) *first_target_pos = solution.aim_pos;
            if (first_selected_index) *first_selected_index = solution.selected_index;
            if (first_fly_time) *first_fly_time = solution.fly_time;
            source_aim_lock_id_ = source_lock_id;
            if (first_impact_delta_angle) {
                *first_impact_delta_angle = solution.impact_delta_angle;
            }
            continue;
        }

        (*yaw_ref)(i) = closestEquivalentRad((*yaw_ref)(i - 1), wrapped_yaw);
        continuity_yaw = (*yaw_ref)(i);
    }

    if (horizon == 1) return;

    (*yaw_rate_ref)(0) = ((*yaw_ref)(1) - (*yaw_ref)(0)) / dt;
    for (int i = 1; i < horizon - 1; ++i) {
        (*yaw_rate_ref)(i) = ((*yaw_ref)(i + 1) - (*yaw_ref)(i - 1)) / (2.0 * dt);
    }
    (*yaw_rate_ref)(horizon - 1) = ((*yaw_ref)(horizon - 1) - (*yaw_ref)(horizon - 2)) / dt;
}

} // namespace rm
