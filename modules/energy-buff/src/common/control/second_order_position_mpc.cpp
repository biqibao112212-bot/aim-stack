#include "second_order_position_mpc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace rm
{
/**
 * 4-state second-order position-command MPC
 *
 * x = [theta, omega, a, c]^T
 *   theta: measured yaw angle
 *   omega: measured yaw rate
 *   a    : effective input after a first-order lag that approximates pure delay
 *   c    : commanded yaw angle sent to the lower controller
 *
 * u = delta_c
 *   c(k+1) = c(k) + u(k)
 *
 * Plant model:
 *   theta_ddot + 2*zeta*wn*theta_dot + wn^2*theta = K*wn^2*a
 *   tau*a_dot = c - a
 */

namespace {

double clamp_value(double value, double lower, double upper)
{
    return std::max(lower, std::min(value, upper));
}

bool same_config(const SecondOrderPositionMPCConfig& lhs, const SecondOrderPositionMPCConfig& rhs)
{
    return lhs.model_dt_s == rhs.model_dt_s && lhs.horizon == rhs.horizon &&
           lhs.track_q == rhs.track_q && lhs.rate_q == rhs.rate_q &&
           lhs.command_q == rhs.command_q &&
           lhs.command_track_target == rhs.command_track_target &&
           lhs.delta_r == rhs.delta_r &&
           lhs.input_gain == rhs.input_gain && lhs.input_lag_s == rhs.input_lag_s &&
           lhs.wn_rad_s == rhs.wn_rad_s && lhs.zeta == rhs.zeta &&
           lhs.max_rate_deg_s == rhs.max_rate_deg_s &&
           lhs.max_lead_deg == rhs.max_lead_deg &&
           lhs.max_state_rate_deg_s == rhs.max_state_rate_deg_s &&
           lhs.output_stage_ratio == rhs.output_stage_ratio;
}

}  // namespace

SecondOrderPositionMPC::~SecondOrderPositionMPC()
{
    destroySolver();
}

void SecondOrderPositionMPC::configure(const SecondOrderPositionMPCConfig& config)
{
    if (!same_config(config_, config)) {
        config_ = config;
        dirty_ = true;
    }
}

void SecondOrderPositionMPC::reset(double measured_deg, double measured_rate_deg_s)
{
    last_command_deg_ = normalizeAngleDeg(measured_deg);
    last_effective_command_deg_ = last_command_deg_;
    last_measured_deg_ = last_command_deg_;
    filtered_rate_deg_s_ = std::isfinite(measured_rate_deg_s) ? measured_rate_deg_s : 0.0;
    last_preview_error_deg_ = 0.0;
    last_preview_valid_ = false;
    last_predicted_yaw_traj_deg_.resize(0);
    last_predicted_command_traj_deg_.resize(0);
    last_reference_traj_deg_.resize(0);
    last_reference_rate_traj_deg_s_.resize(0);
    last_output_state_index_ = 0;
    last_solve_us_ = 0.0;
    last_update_us_ = 0.0;
    initialized_ = true;
}

void SecondOrderPositionMPC::setPreviewWindow(int start_index, int end_index)
{
    preview_window_start_index_ = std::max(0, start_index);
    preview_window_end_index_ = std::max(preview_window_start_index_, end_index);
}

double SecondOrderPositionMPC::update(
    double target_deg, double measured_deg, double measured_rate_deg_s, double applied_dt_s)
{
    const int horizon = std::max(config_.horizon, 4);
    Eigen::VectorXd target_traj = Eigen::VectorXd::Constant(horizon, target_deg);
    Eigen::VectorXd target_rate_traj = Eigen::VectorXd::Zero(horizon);
    return updateTrajectory(
        target_traj, target_rate_traj, measured_deg, measured_rate_deg_s, applied_dt_s);
}

/**
 * @brief 更新二阶位置 MPC 的轨迹并返回新的命令角度（度）
 *
 * 功能概述：
 * - 根据测量值和目标轨迹构建 MPC 求解器的初始状态、参考轨迹和约束，调用求解器，
 *   从预测结果中读取命令角度并返回（归一化到常用角度范围）。
 *
 * 参数：
 * @param target_deg_traj           目标位置轨迹（度），按时间顺序的向量。
 * @param target_rate_deg_s_traj    目标速度轨迹（度/秒）；若不可用或为非有限值，速度参考由位置差分估算。
 * @param measured_deg              当前测量的角度（度）。
 * @param measured_rate_deg_s       当前测量的角速度（度/秒），若为非有限值则用差分估算。
 * @param applied_dt_s              本次调用的时间间隔（秒），用于速率/控制步长计算与限幅。
 *
 * 返回值：
 * @return 新的命令角度（度），已归一化并保存在 last_command_deg_ 中。
 *
 * 内部要点（简要）：
 * - horizon = std::max(config_.horizon, 4)；保证最小预测步数。
 * - 构造并设置初始状态 x0 = [measured_unwrapped, filtered_rate_deg_s_, last_effective_input, last_command]。
 * - 对输入位置轨迹做 unwrap（避免 ±180° 跳变），并基于 model_dt_s 构建速度参考序列。
 * - u_ref 设为 0（期望的控制增量参考为零）。
 * - 施加状态/控制约束（最大前瞻角 max_lead_deg、状态速率 max_state_rate_deg_s、控制增量限幅等）。
 * - 调用 tiny_solve 求解，读取配置对应预测步的命令输出，
 *   更新 last_preview_error_deg_、last_preview_valid_、last_command_deg_、last_measured_deg_ 等成员。
 *
 * x_ref 矩阵说明：
 * - x_ref 是传给求解器的参考状态矩阵，维度为 4 × horizon（4 行，horizon 列）。
 *   - 行 0（索引 0）：位置参考（度），在代码中赋为 target_unwrapped。
 *   - 行 1（索引 1）：速度参考（度/秒），在代码中赋为 buildRateReferenceTrajectory 的结果。
 *   - 行 2（索引 2）：有效输入状态参考（度），当前不单独加权。
 *   - 行 3（索引 3）：命令位置参考（度），用于与 max_lead_deg 相关的约束/参考。
 * - 列索引对应预测时域内的时间步（0..horizon-1），第 0 列通常表示当前时刻的参考。
 *
 * 注意：
 * - x_ref 的行含义和维度是固定的（4×horizon），必须与求解器预期的状态维度一致。
 * - 函数内部通过 unwrap 等操作保证参考轨迹在角度环绕处的连续性，以避免不合理的跨越跳变。
 */
double SecondOrderPositionMPC::updateTrajectory(
    const Eigen::VectorXd& target_deg_traj, const Eigen::VectorXd& target_rate_deg_s_traj,
    double measured_deg, double measured_rate_deg_s, double applied_dt_s)
{
    const auto update_begin = std::chrono::steady_clock::now();
    if (!initialized_) reset(measured_deg, measured_rate_deg_s);
    if (dirty_ || solver_ == nullptr) rebuildSolver();

    const int horizon = std::max(config_.horizon, 4);
    const double dt_upper_s = std::max(config_.model_dt_s, 1e-3);
    double rate_dt_s = dt_upper_s;
    if (std::isfinite(applied_dt_s) && applied_dt_s > 0.0) {
        rate_dt_s = applied_dt_s;
    }
    rate_dt_s = clamp_value(rate_dt_s, 1e-3, dt_upper_s);
    const double measured_unwrapped =
        closestEquivalentAngleDeg(last_measured_deg_, measured_deg);
    /**
     * @brief last_command_unwrapped
     *
     * The previous commanded angle (in degrees) shifted into the same continuous
     * (unwrapped) rotation range as the current measured angle so that angular
     * differences across the wrap boundary are computed correctly.
     *
     * Computed as the equivalent angle to last_command_deg_ that is closest to
     * measured_unwrapped (units: degrees). Used for smooth error/delta calculations
     * in the controller/MPC.
     *
     * 注：x0 是 MPC 的初始状态向量（例如位置、速度等），表示预测时域起点的系统状态。
     */
    const double last_command_unwrapped =
        closestEquivalentAngleDeg(measured_unwrapped, last_command_deg_);
    const double last_effective_command_unwrapped =
        closestEquivalentAngleDeg(last_command_unwrapped, last_effective_command_deg_);

    const double max_state_rate =
        config_.max_state_rate_deg_s > 0.0 ? config_.max_state_rate_deg_s : 1e17;
    double measured_rate_raw = std::isfinite(measured_rate_deg_s)
                                   ? measured_rate_deg_s
                                   : (measured_unwrapped - last_measured_deg_) / rate_dt_s;
    measured_rate_raw = clamp_value(measured_rate_raw, -max_state_rate, max_state_rate);
    filtered_rate_deg_s_ = measured_rate_raw;

    Eigen::Vector4d x0;
    x0 << measured_unwrapped, filtered_rate_deg_s_, last_effective_command_unwrapped,
        last_command_unwrapped;
    tiny_set_x0(solver_, x0);

    const Eigen::VectorXd target_unwrapped =
        unwrapReferenceTrajectory(target_deg_traj, last_command_unwrapped, horizon);
    const Eigen::VectorXd target_rate_ref = buildRateReferenceTrajectory(
        target_unwrapped, target_rate_deg_s_traj, rate_dt_s, horizon);

    Eigen::MatrixXd x_ref = Eigen::MatrixXd::Zero(4, horizon);
    x_ref.row(0) = target_unwrapped.transpose();
    x_ref.row(1) = target_rate_ref.transpose();
    x_ref.row(2).setConstant(last_effective_command_unwrapped);
    if (config_.command_track_target) {
        x_ref.row(3) = target_unwrapped.transpose();
    } else {
        x_ref.row(3).setConstant(last_command_unwrapped);
    }
    tiny_set_x_ref(solver_, x_ref);

    Eigen::MatrixXd u_ref = Eigen::MatrixXd::Zero(1, horizon - 1);
    tiny_set_u_ref(solver_, u_ref);

    Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(4, horizon, -1e17);
    Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(4, horizon, 1e17);
    if (config_.max_lead_deg > 0.0) {
        const double lower = measured_unwrapped - config_.max_lead_deg;
        const double upper = measured_unwrapped + config_.max_lead_deg;
        for (int col = 1; col < horizon; ++col) {
            x_min(3, col) = lower;
            x_max(3, col) = upper;
        }
    }
    if (config_.max_state_rate_deg_s > 0.0) {
        x_min.row(1).setConstant(-config_.max_state_rate_deg_s);
        x_max.row(1).setConstant(config_.max_state_rate_deg_s);
    }

    const double max_delta_deg =
        config_.max_rate_deg_s > 0.0 ? config_.max_rate_deg_s * rate_dt_s : 1e17;
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, horizon - 1, -max_delta_deg);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, horizon - 1, max_delta_deg);
    tiny_set_bound_constraints(solver_, x_min, x_max, u_min, u_max);

    const auto solve_begin = std::chrono::steady_clock::now();
    tiny_solve(solver_);
    last_solve_us_ = std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - solve_begin)
                         .count();

    const int output_state_index = std::clamp(
        static_cast<int>(std::lround(
            clamp_value(config_.output_stage_ratio, 0.0, 1.0) * static_cast<double>(horizon - 1))),
        0, horizon - 1);
    last_output_state_index_ = output_state_index;

    double command_unwrapped = last_command_unwrapped;
    if (output_state_index <= 0) {
        double delta_command_deg = solver_->work->u(0, 0);
        if (!std::isfinite(delta_command_deg)) {
            delta_command_deg = 0.0;
        }
        command_unwrapped += delta_command_deg;
    } else {
        const double predicted_command_deg = solver_->work->x(3, output_state_index);
        if (std::isfinite(predicted_command_deg)) {
            command_unwrapped = predicted_command_deg;
        }
    }

    const int preview_start_index =
        std::clamp(preview_window_start_index_, 0, horizon - 1);
    const int preview_end_index =
        std::clamp(preview_window_end_index_, preview_start_index, horizon - 1);
    last_preview_error_deg_ = 0.0;
    last_preview_valid_ = false;
    for (int i = preview_start_index; i <= preview_end_index; ++i) {
        const double preview_error_deg =
            std::abs(x_ref(0, i) - solver_->work->x(0, i));
        if (!std::isfinite(preview_error_deg)) continue;
        if (!last_preview_valid_ || preview_error_deg > last_preview_error_deg_) {
            last_preview_error_deg_ = preview_error_deg;
        }
        last_preview_valid_ = true;
    }
    last_predicted_yaw_traj_deg_ = solver_->work->x.row(0).transpose();
    last_predicted_command_traj_deg_ = solver_->work->x.row(3).transpose();
    last_reference_traj_deg_ = target_unwrapped;
    last_reference_rate_traj_deg_s_ = target_rate_ref;

    double effective_command_unwrapped = last_effective_command_unwrapped;
    if (solver_->work->x.cols() > 1) {
        const double predicted_effective_command_deg = solver_->work->x(2, 1);
        if (std::isfinite(predicted_effective_command_deg)) {
            effective_command_unwrapped = predicted_effective_command_deg;
        }
    }

    last_effective_command_deg_ = normalizeAngleDeg(effective_command_unwrapped);
    last_command_deg_ = normalizeAngleDeg(command_unwrapped);
    last_measured_deg_ = measured_unwrapped;
    last_update_us_ = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - update_begin)
                          .count();
    return last_command_deg_;
}

double SecondOrderPositionMPC::lastPreviewTrackingErrorDeg() const
{
    return last_preview_error_deg_;
}

bool SecondOrderPositionMPC::lastPreviewTrackingValid() const
{
    return last_preview_valid_;
}

const Eigen::VectorXd& SecondOrderPositionMPC::lastPredictedYawTrajectoryDeg() const
{
    return last_predicted_yaw_traj_deg_;
}

const Eigen::VectorXd& SecondOrderPositionMPC::lastPredictedCommandTrajectoryDeg() const
{
    return last_predicted_command_traj_deg_;
}

const Eigen::VectorXd& SecondOrderPositionMPC::lastReferenceTrajectoryDeg() const
{
    return last_reference_traj_deg_;
}

const Eigen::VectorXd& SecondOrderPositionMPC::lastReferenceRateTrajectoryDegS() const
{
    return last_reference_rate_traj_deg_s_;
}

int SecondOrderPositionMPC::lastOutputStateIndex() const
{
    return last_output_state_index_;
}

double SecondOrderPositionMPC::lastSolveUs() const
{
    return last_solve_us_;
}

double SecondOrderPositionMPC::lastUpdateUs() const
{
    return last_update_us_;
}

void SecondOrderPositionMPC::destroySolver()
{
    if (solver_ == nullptr) return;
    delete solver_->solution;
    delete solver_->cache;
    delete solver_->settings;
    delete solver_->work;
    delete solver_;
    solver_ = nullptr;
}

void SecondOrderPositionMPC::rebuildSolver()
{
    destroySolver();

    const double model_dt_s = std::max(config_.model_dt_s, 1e-3);
    const double input_gain = std::max(config_.input_gain, 0.0);
    const double input_lag_s = std::max(config_.input_lag_s, 0.0);
    const double wn = std::max(config_.wn_rad_s, 1e-3);
    const double zeta = std::max(config_.zeta, 0.0);
    const int horizon = std::max(config_.horizon, 4);
    const double wn2 = wn * wn;
    const double input_alpha =
        input_lag_s > 1e-6 ? std::exp(-model_dt_s / input_lag_s) : 0.0;
    const double input_blend = 1.0 - input_alpha;

    const double theta_theta = 1.0 - 0.5 * wn2 * model_dt_s * model_dt_s;
    const double theta_rate = model_dt_s - zeta * wn * model_dt_s * model_dt_s;
    const double theta_effective_command =
        0.5 * input_gain * wn2 * model_dt_s * model_dt_s;
    const double omega_theta = -wn2 * model_dt_s;
    const double omega_rate = 1.0 - 2.0 * zeta * wn * model_dt_s;
    const double omega_effective_command = input_gain * wn2 * model_dt_s;

    Eigen::MatrixXd A(4, 4);
    A << theta_theta, theta_rate, theta_effective_command, 0.0,
         omega_theta, omega_rate, omega_effective_command, 0.0,
         0.0, 0.0, input_alpha, input_blend,
         0.0, 0.0, 0.0, 1.0;

    Eigen::MatrixXd B(4, 1);
    B << 0.0,
         0.0,
         input_blend,
         1.0;

    Eigen::VectorXd f = Eigen::VectorXd::Zero(4);
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    Q(0, 0) = std::max(config_.track_q, 0.0);
    Q(1, 1) = std::max(config_.rate_q, 0.0);
    Q(3, 3) = std::max(config_.command_q, 0.0);

    Eigen::Matrix<double, 1, 1> R;
    R(0, 0) = std::max(config_.delta_r, 1e-6);

    const int status = tiny_setup(&solver_, A, B, f, Q, R, 1.0, 4, 1, horizon, 0);
    if (status != 0 || solver_ == nullptr) {
        throw std::runtime_error("SecondOrderPositionMPC tiny_setup failed");
    }

    Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(4, horizon, -1e17);
    Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(4, horizon, 1e17);
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, horizon - 1, -1e17);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, horizon - 1, 1e17);
    tiny_set_bound_constraints(solver_, x_min, x_max, u_min, u_max);
    solver_->settings->max_iter = 20;
    dirty_ = false;
}

Eigen::VectorXd SecondOrderPositionMPC::unwrapReferenceTrajectory(
    const Eigen::VectorXd& target_deg_traj, double reference_deg, int horizon) const
{
    Eigen::VectorXd unwrapped = Eigen::VectorXd::Constant(horizon, reference_deg);
    if (target_deg_traj.size() <= 0) return unwrapped;

    const int copy_count = std::min<int>(target_deg_traj.size(), horizon);
    double prev_unwrapped = closestEquivalentAngleDeg(reference_deg, target_deg_traj(0));
    unwrapped(0) = prev_unwrapped;
    for (int i = 1; i < copy_count; ++i) {
        prev_unwrapped = closestEquivalentAngleDeg(prev_unwrapped, target_deg_traj(i));
        unwrapped(i) = prev_unwrapped;
    }
    for (int i = copy_count; i < horizon; ++i) {
        unwrapped(i) = prev_unwrapped;
    }
    return unwrapped;
}

Eigen::VectorXd SecondOrderPositionMPC::buildRateReferenceTrajectory(
    const Eigen::VectorXd& target_unwrapped_deg_traj,
    const Eigen::VectorXd& target_rate_deg_s_traj, double reference_dt_s, int horizon) const
{
    Eigen::VectorXd rate_ref = Eigen::VectorXd::Zero(horizon);
    if (horizon <= 0) return rate_ref;

    const double dt_s = std::max(reference_dt_s, 1e-3);
    if (horizon > 1) {
        rate_ref(0) = (target_unwrapped_deg_traj(1) - target_unwrapped_deg_traj(0)) / dt_s;
        for (int i = 1; i < horizon - 1; ++i) {
            rate_ref(i) = (target_unwrapped_deg_traj(i + 1) - target_unwrapped_deg_traj(i - 1)) /
                          (2.0 * dt_s);
        }
        rate_ref(horizon - 1) =
            (target_unwrapped_deg_traj(horizon - 1) - target_unwrapped_deg_traj(horizon - 2)) /
            dt_s;
    }

    const int rate_copy_count = std::min<int>(target_rate_deg_s_traj.size(), horizon);
    for (int i = 0; i < rate_copy_count; ++i) {
        if (std::isfinite(target_rate_deg_s_traj(i))) {
            rate_ref(i) = target_rate_deg_s_traj(i);
        }
    }

    if (config_.max_state_rate_deg_s > 0.0) {
        const double limit = config_.max_state_rate_deg_s;
        for (int i = 0; i < horizon; ++i) {
            if (!std::isfinite(rate_ref(i))) continue;
            rate_ref(i) = clamp_value(rate_ref(i), -limit, limit);
        }
    }

    return rate_ref;
}

double SecondOrderPositionMPC::normalizeAngleDeg(double angle_deg)
{
    double result = std::fmod(angle_deg + 180.0, 360.0);
    if (result < 0.0) result += 360.0;
    return result - 180.0;
}

double SecondOrderPositionMPC::closestEquivalentAngleDeg(double reference_deg, double angle_deg)
{
    return reference_deg + normalizeAngleDeg(angle_deg - reference_deg);
}

}  // namespace rm
