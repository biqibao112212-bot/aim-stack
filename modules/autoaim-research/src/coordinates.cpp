#include "autoaim_research/coordinates.hpp"

#include <cmath>

namespace autoaim_research {

Eigen::Matrix3d cameraLinkFromOptical() {
  Eigen::Matrix3d rotation;
  rotation << 0.0, 0.0, 1.0,
             -1.0, 0.0, 0.0,
              0.0, -1.0, 0.0;
  return rotation;
}

Eigen::Matrix3d worldFromOptical(const ExposurePose& pose) {
  return pose.world_from_camera_link.normalized().toRotationMatrix() *
         cameraLinkFromOptical();
}

Eigen::Vector3d opticalPointToResearch(
    const ExposurePose& pose, const Eigen::Vector3d& optical_point_m) {
  const Eigen::Vector3d point_world =
      worldFromOptical(pose) * optical_point_m + pose.camera_position_world_m;
  return point_world - pose.gimbal_position_world_m;
}

Eigen::Vector3d researchPointToOptical(
    const ExposurePose& pose, const Eigen::Vector3d& research_point_m) {
  const Eigen::Vector3d point_world =
      research_point_m + pose.gimbal_position_world_m;
  return worldFromOptical(pose).transpose() *
         (point_world - pose.camera_position_world_m);
}

bool validExposurePose(const ExposurePose& pose) {
  const double camera_norm = pose.world_from_camera_link.norm();
  const double gimbal_norm = pose.world_from_gimbal.norm();
  return pose.camera_position_world_m.allFinite() &&
         pose.gimbal_position_world_m.allFinite() &&
         pose.world_from_camera_link.coeffs().allFinite() &&
         pose.world_from_gimbal.coeffs().allFinite() &&
         std::isfinite(camera_norm) && std::isfinite(gimbal_norm) &&
         camera_norm > 0.5 && gimbal_norm > 0.5;
}

}  // namespace autoaim_research
