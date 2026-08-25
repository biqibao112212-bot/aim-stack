#pragma once

#include <opencv2/opencv.hpp>

#include "autoaim_research/coordinates.hpp"
#include "tasks/auto_aim/armor.hpp"

namespace autoaim_research {

struct ArmorGeometry {
  double small_width_m = 0.135;
  double big_width_m = 0.225;
  double height_m = 0.055;
};

struct PoseResult {
  bool valid = false;
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  double reprojection_rms_px = 0.0;
};

// Tongji IPPE solver adapted to Daedalus' fixed calibration and exact
// exposure pose. The resulting Armor position is expressed in a ROS-odom
// oriented frame whose origin is the exposure-time gimbal pivot.
class TongjiPoseSolver {
 public:
  TongjiPoseSolver(cv::Mat camera_matrix, cv::Mat distortion,
                   ArmorGeometry geometry = {});

  PoseResult solve(auto_aim::Armor& armor, const ExposurePose& exposure) const;

 private:
  std::vector<cv::Point3f> objectPoints(auto_aim::ArmorType type) const;
  double reprojectionRms(const auto_aim::Armor& armor, const cv::Vec3d& rvec,
                         const cv::Vec3d& tvec) const;
  double constrainedYaw(const auto_aim::Armor& armor,
                        const ExposurePose& exposure) const;

  cv::Mat camera_matrix_;
  cv::Mat distortion_;
  ArmorGeometry geometry_;
};

}  // namespace autoaim_research
