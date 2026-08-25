#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <list>

#include <opencv2/calib3d.hpp>

#include "autoaim_research/coordinates.hpp"
#include "autoaim_research/pose_solver.hpp"
#include "autoaim_research/research_tracker.hpp"
#include "tools/math_tools.hpp"

namespace {
auto_aim::Armor makeArmor(const std::vector<cv::Point2f>& points) {
  auto_aim::Armor armor(1, 3, 0.95F, cv::Rect(0, 0, 100, 40), points);
  armor.priority = auto_aim::ArmorPriority::third;
  armor.duplicated = false;
  return armor;
}
}  // namespace

int main() {
  using namespace autoaim_research;

  ExposurePose exposure;
  exposure.camera_position_world_m = {1.0, 2.0, 3.0};
  exposure.gimbal_position_world_m = {0.5, 1.5, 2.5};
  const Eigen::Vector3d optical_point{1.0, 2.0, 3.0};
  const Eigen::Vector3d research = opticalPointToResearch(exposure, optical_point);
  const Eigen::Vector3d expected{3.5, -0.5, -1.5};
  assert((research - expected).norm() < 1e-12);
  assert((researchPointToOptical(exposure, research) - optical_point).norm() < 1e-12);

  cv::Mat camera = (cv::Mat_<double>(3, 3) <<
      1303.67532368147, 0.0, 720.0,
      0.0, 1303.67532368147, 540.0,
      0.0, 0.0, 1.0);
  cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
  const std::vector<cv::Point3f> object_points{
      {0.0F, 0.0675F, 0.0275F}, {0.0F, -0.0675F, 0.0275F},
      {0.0F, -0.0675F, -0.0275F}, {0.0F, 0.0675F, -0.0275F}};
  const cv::Vec3d expected_rvec{0.0, -CV_PI / 2.0, 0.0};
  const cv::Vec3d expected_tvec{0.10, -0.05, 4.0};
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, expected_rvec, expected_tvec, camera, distortion,
                    image_points);
  auto armor = makeArmor(image_points);
  exposure.camera_position_world_m.setZero();
  exposure.gimbal_position_world_m.setZero();
  TongjiPoseSolver solver(camera, distortion);
  const PoseResult pose = solver.solve(armor, exposure);
  assert(pose.valid);
  assert(std::abs(pose.tvec[0] - expected_tvec[0]) < 1e-3);
  assert(std::abs(pose.tvec[1] - expected_tvec[1]) < 1e-3);
  assert(std::abs(pose.tvec[2] - expected_tvec[2]) < 1e-3);
  assert(pose.reprojection_rms_px < 1e-2);
  assert((armor.xyz_in_gimbal - armor.xyz_in_world).norm() < 1e-12);

  TongjiResearchTracker tracker;
  std::vector<cv::Point2f> dummy{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  TrackerSnapshot snapshot;
  for (std::uint64_t frame = 0; frame < 6; ++frame) {
    auto observation = makeArmor(dummy);
    observation.xyz_in_world = {4.0, 0.2, 0.1};
    observation.ypr_in_world = {0.1, 0.0, 0.0};
    observation.ypd_in_world = tools::xyz2ypd(observation.xyz_in_world);
    snapshot = tracker.update({observation}, 1'000'000'000ULL + frame * 10'000'000ULL);
  }
  assert(snapshot.has_estimate);
  assert(snapshot.state_vector.size() == 11);
  assert(snapshot.state_vector.allFinite());

  std::cout << "autoaim_research_core_test: ok\n";
  return 0;
}
