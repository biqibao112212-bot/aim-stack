#pragma once

#include <Eigen/Geometry>

namespace autoaim_research {

// Exact pose published for the same exposure as the image. All positions and
// rotations use the SDK ROS-odom convention: x forward, y left, z up.
struct ExposurePose {
  Eigen::Vector3d camera_position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond world_from_camera_link{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d gimbal_position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond world_from_gimbal{Eigen::Quaterniond::Identity()};
};

// OpenCV optical (right, down, forward) -> ROS camera_link
// (forward, left, up).
Eigen::Matrix3d cameraLinkFromOptical();
Eigen::Matrix3d worldFromOptical(const ExposurePose& pose);
Eigen::Vector3d opticalPointToResearch(
    const ExposurePose& pose, const Eigen::Vector3d& optical_point_m);
Eigen::Vector3d researchPointToOptical(
    const ExposurePose& pose, const Eigen::Vector3d& research_point_m);
bool validExposurePose(const ExposurePose& pose);

}  // namespace autoaim_research
