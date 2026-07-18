#include "target.h"

#include <algorithm>

namespace rm
{

static double limit_rad(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

namespace {

constexpr double kMinArmorRadiusM = 0.05;
constexpr double kMaxArmorRadiusM = 0.60;
constexpr double kMaxOutpostHeightOffsetM = 0.60;
constexpr double kOutpostPlaneToRadialYawOffsetRad = 153.0 * M_PI / 180.0;

double clamp_radius(double radius)
{
    return std::clamp(radius, kMinArmorRadiusM, kMaxArmorRadiusM);
}

void clamp_geometry_state(Eigen::VectorXd& x, int armor_num)
{
    if (x.size() < 11) return;

    x(8) = clamp_radius(x(8));
    if (armor_num == 4) {
        const double secondary_radius = clamp_radius(x(8) + x(9));
        x(9) = secondary_radius - x(8);
    } else if (armor_num == 3) {
        x(9) = std::clamp(x(9), -kMaxOutpostHeightOffsetM, kMaxOutpostHeightOffsetM);
        x(10) = std::clamp(x(10), -kMaxOutpostHeightOffsetM, kMaxOutpostHeightOffsetM);
    } else {
        x(9) = 0.0;
        x(10) = 0.0;
    }
}

double radius_from_state(const Eigen::VectorXd& x, int armor_num, int id)
{
    if (x.size() < 11) return 0.20;
    const bool use_secondary_radius = (armor_num == 4) && (id == 1 || id == 3);
    return clamp_radius(use_secondary_radius ? x(8) + x(9) : x(8));
}

double armor_height_offset_from_state(const Eigen::VectorXd& x, int armor_num, int id)
{
    if (x.size() < 11) return 0.0;
    if (armor_num == 4) {
        return (id == 1 || id == 3) ? x(10) : 0.0;
    }
    if (armor_num == 3) {
        if (id == 1) return x(9);
        if (id == 2) return x(10);
    }
    return 0.0;
}

double armor_radial_sign(int armor_num)
{
    return armor_num == 3 ? 1.0 : -1.0;
}

double observed_yaw_from_radial_yaw(double radial_yaw, int armor_num)
{
    if (armor_num != 3) return radial_yaw;
    return limit_rad(radial_yaw - kOutpostPlaneToRadialYawOffsetRad);
}

} // namespace

Target::Target(int armor_num) : armor_num_(armor_num)
{
    x_ = Eigen::VectorXd::Zero(11);
}

void Target::sync_state(const Eigen::VectorXd& rm_state)
{
    if (x_.size() != 11) x_ = Eigen::VectorXd::Zero(11);

    int copy_size = std::min<int>(11, rm_state.size());
    for (int i = 0; i < copy_size; i++) {
        x_(i) = rm_state(i);
    }
    for (int i = copy_size; i < 11; ++i) {
        x_(i) = 0.0;
    }
    clamp_geometry_state(x_, armor_num_);
}

void Target::predict(double dt)
{
    x_(0) += x_(1) * dt;
    x_(2) += x_(3) * dt;
    x_(4) += x_(5) * dt;
    x_(6) += x_(7) * dt;
    x_(6) = limit_rad(x_(6));
    clamp_geometry_state(x_, armor_num_);
}

Eigen::VectorXd Target::ekf_x() const
{
    return x_;
}

Eigen::Vector3d Target::get_armor_pos(double yaw_base, int id) const
{
    const double angle = limit_rad(yaw_base + id * (2 * M_PI / armor_num_));
    const double r = radius_from_state(x_, armor_num_, id);
    const double z = x_(4) + armor_height_offset_from_state(x_, armor_num_, id);
    const double radial_sign = armor_radial_sign(armor_num_);
    return {
        x_(0) + radial_sign * r * std::cos(angle),
        x_(2) + radial_sign * r * std::sin(angle),
        z};
}

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
    std::vector<Eigen::Vector4d> list;
    for (int i = 0; i < armor_num_; i++) {
        const double radial_yaw = limit_rad(x_(6) + i * (2 * M_PI / armor_num_));
        Eigen::Vector3d pos = get_armor_pos(x_(6), i);
        double yaw = observed_yaw_from_radial_yaw(radial_yaw, armor_num_);
        list.push_back({pos(0), pos(1), pos(2), yaw});
    }
    return list;
}

} // namespace rm
