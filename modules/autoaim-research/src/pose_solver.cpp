#include "autoaim_research/pose_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "tools/math_tools.hpp"

namespace autoaim_research {

namespace {
Eigen::Matrix3d eigenRotation(const cv::Mat& matrix) {
  Eigen::Matrix3d result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result(row, column) = matrix.at<double>(row, column);
    }
  }
  return result;
}

bool finiteVec(const cv::Vec3d& value) {
  return std::isfinite(value[0]) && std::isfinite(value[1]) &&
         std::isfinite(value[2]);
}
}  // namespace

TongjiPoseSolver::TongjiPoseSolver(cv::Mat camera_matrix, cv::Mat distortion,
                                   ArmorGeometry geometry)
    : geometry_(geometry) {
  camera_matrix.convertTo(camera_matrix_, CV_64F);
  distortion.convertTo(distortion_, CV_64F);
  if (camera_matrix_.rows != 3 || camera_matrix_.cols != 3) {
    throw std::runtime_error("camera matrix must be 3x3");
  }
  if (!(geometry_.small_width_m > 0.0 && geometry_.big_width_m > 0.0 &&
        geometry_.height_m > 0.0)) {
    throw std::runtime_error("armor dimensions must be positive");
  }
}

std::vector<cv::Point3f> TongjiPoseSolver::objectPoints(
    auto_aim::ArmorType type) const {
  const float width = static_cast<float>(
      type == auto_aim::ArmorType::big ? geometry_.big_width_m
                                        : geometry_.small_width_m);
  const float height = static_cast<float>(geometry_.height_m);
  return {{0.0F, width / 2.0F, height / 2.0F},
          {0.0F, -width / 2.0F, height / 2.0F},
          {0.0F, -width / 2.0F, -height / 2.0F},
          {0.0F, width / 2.0F, -height / 2.0F}};
}

PoseResult TongjiPoseSolver::solve(auto_aim::Armor& armor,
                                   const ExposurePose& exposure) const {
  PoseResult result;
  if (armor.points.size() != 4 || !validExposurePose(exposure)) return result;

  const auto model = objectPoints(armor.type);
  if (!cv::solvePnP(model, armor.points, camera_matrix_, distortion_, result.rvec,
                    result.tvec, false, cv::SOLVEPNP_IPPE) ||
      !finiteVec(result.rvec) || !finiteVec(result.tvec) || result.tvec[2] <= 0.0) {
    return result;
  }

  const Eigen::Vector3d optical_position(result.tvec[0], result.tvec[1],
                                         result.tvec[2]);
  armor.xyz_in_world = opticalPointToResearch(exposure, optical_position);
  const Eigen::Matrix3d gimbal_from_world =
      exposure.world_from_gimbal.normalized().toRotationMatrix().transpose();
  armor.xyz_in_gimbal = gimbal_from_world * armor.xyz_in_world;

  cv::Mat rotation_cv;
  cv::Rodrigues(result.rvec, rotation_cv);
  const Eigen::Matrix3d optical_from_armor = eigenRotation(rotation_cv);
  const Eigen::Matrix3d world_from_armor =
      worldFromOptical(exposure) * optical_from_armor;
  const Eigen::Matrix3d gimbal_from_armor =
      gimbal_from_world * world_from_armor;
  armor.ypr_in_gimbal = tools::eulers(gimbal_from_armor, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(world_from_armor, 2, 1, 0);
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  armor.yaw_raw = armor.ypr_in_world[0];

  const bool balance = armor.type == auto_aim::ArmorType::big &&
      (armor.name == auto_aim::ArmorName::three ||
       armor.name == auto_aim::ArmorName::four ||
       armor.name == auto_aim::ArmorName::five);
  if (!balance) armor.ypr_in_world[0] = constrainedYaw(armor, exposure);

  result.reprojection_rms_px = reprojectionRms(armor, result.rvec, result.tvec);
  result.valid = std::isfinite(result.reprojection_rms_px) &&
                 armor.xyz_in_world.allFinite() && armor.ypr_in_world.allFinite();
  return result;
}

double TongjiPoseSolver::reprojectionRms(const auto_aim::Armor& armor,
                                         const cv::Vec3d& rvec,
                                         const cv::Vec3d& tvec) const {
  std::vector<cv::Point2f> projected;
  cv::projectPoints(objectPoints(armor.type), rvec, tvec, camera_matrix_, distortion_,
                    projected);
  double squared_error = 0.0;
  for (std::size_t index = 0; index < projected.size(); ++index) {
    const cv::Point2f delta = projected[index] - armor.points[index];
    squared_error += static_cast<double>(delta.dot(delta));
  }
  return std::sqrt(squared_error / static_cast<double>(projected.size()));
}

double TongjiPoseSolver::constrainedYaw(const auto_aim::Armor& armor,
                                        const ExposurePose& exposure) const {
  const double gimbal_yaw =
      tools::eulers(exposure.world_from_gimbal.normalized().toRotationMatrix(),
                    2, 1, 0)[0];
  constexpr int search_range_degrees = 140;
  const double yaw_begin = tools::limit_rad(
      gimbal_yaw - search_range_degrees * 0.5 * CV_PI / 180.0);
  const double pitch = (armor.name == auto_aim::ArmorName::outpost ? -15.0 : 15.0) *
                       CV_PI / 180.0;
  const Eigen::Vector3d position_world =
      armor.xyz_in_world + exposure.gimbal_position_world_m;
  double best_yaw = armor.ypr_in_world[0];
  double best_error = std::numeric_limits<double>::infinity();

  for (int degree = 0; degree < search_range_degrees; ++degree) {
    const double yaw = tools::limit_rad(yaw_begin + degree * CV_PI / 180.0);
    const double sy = std::sin(yaw);
    const double cy = std::cos(yaw);
    const double sp = std::sin(pitch);
    const double cp = std::cos(pitch);
    Eigen::Matrix3d world_from_armor;
    world_from_armor << cy * cp, -sy, cy * sp,
                        sy * cp,  cy, sy * sp,
                             -sp, 0.0, cp;

    const Eigen::Matrix3d optical_from_armor =
        worldFromOptical(exposure).transpose() * world_from_armor;
    const Eigen::Vector3d optical_position =
        worldFromOptical(exposure).transpose() *
        (position_world - exposure.camera_position_world_m);
    cv::Mat rotation_cv(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        rotation_cv.at<double>(row, column) = optical_from_armor(row, column);
      }
    }
    cv::Vec3d rvec;
    cv::Rodrigues(rotation_cv, rvec);
    const cv::Vec3d tvec(optical_position.x(), optical_position.y(),
                         optical_position.z());
    std::vector<cv::Point2f> projected;
    cv::projectPoints(objectPoints(armor.type), rvec, tvec, camera_matrix_,
                      distortion_, projected);
    double error = 0.0;
    for (std::size_t index = 0; index < projected.size(); ++index) {
      error += cv::norm(projected[index] - armor.points[index]);
    }
    if (error < best_error) {
      best_error = error;
      best_yaw = yaw;
    }
  }
  return best_yaw;
}

}  // namespace autoaim_research
