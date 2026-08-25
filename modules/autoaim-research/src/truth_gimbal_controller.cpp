#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#include <Eigen/Geometry>

#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/udp_gimbal_client.hpp>

namespace {
using namespace daedalus::sim::sdk::v1;

std::atomic<bool> running{true};
void stopHandler(int) { running.store(false); }

Eigen::Matrix3d cameraLinkFromOptical() {
  Eigen::Matrix3d rotation;
  rotation << 0.0, 0.0, 1.0,
             -1.0, 0.0, 0.0,
              0.0, -1.0, 0.0;
  return rotation;
}

const GroundTruthTarget* selectActiveRangeTarget(
    const GroundTruthBatch& truth, const ExposureState& exposure,
    std::uint8_t target_number) {
  const Eigen::Vector3d camera(exposure.camera_position_world[0],
                               exposure.camera_position_world[1],
                               exposure.camera_position_world[2]);
  const GroundTruthTarget* best = nullptr;
  double best_distance = 0.0;
  for (std::uint32_t index = 0; index < truth.target_count; ++index) {
    const auto& target = truth.targets[index];
    if (target.armor_label != target_number ||
        (target.state_flags & (kGroundTruthTargetHasWorldState |
                               kGroundTruthTargetHasArmorGeometry)) !=
            (kGroundTruthTargetHasWorldState |
             kGroundTruthTargetHasArmorGeometry) ||
        target.world_state_frame != kGroundTruthFrameRosOdom ||
        target.armor_count != 4) {
      continue;
    }
    bool scene_visible = false;
    for (std::uint8_t slot = 0; slot < target.armor_count; ++slot) {
      scene_visible = scene_visible ||
                      target.armors[slot].visibility !=
                          kGroundTruthVisibilityHidden;
    }
    if (!scene_visible) continue;
    const Eigen::Vector3d position(target.position[0], target.position[1],
                                   target.position[2]);
    const double distance = (position - camera).norm();
    if (distance > 1.0 && distance > best_distance) {
      best = &target;
      best_distance = distance;
    }
  }
  return best;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string ipc_directory;
    double lookahead_s = 0.03;
    double gain = 1.0;
    int target_number = 3;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      auto value = [&]() -> std::string {
        if (index + 1 >= argc) throw std::runtime_error("missing value for " + argument);
        return argv[++index];
      };
      if (argument == "--ipc-dir") ipc_directory = value();
      else if (argument == "--lookahead-s") lookahead_s = std::stod(value());
      else if (argument == "--gain") gain = std::stod(value());
      else if (argument == "--target") target_number = std::stoi(value());
      else throw std::runtime_error("unknown argument: " + argument);
    }
    if (ipc_directory.empty()) throw std::runtime_error("--ipc-dir is required");
    if ((target_number != 1 && target_number != 3) ||
        !std::isfinite(lookahead_s) || lookahead_s < 0.0 ||
        !std::isfinite(gain) || gain <= 0.0 || gain > 1.0) {
      throw std::runtime_error("invalid lookahead or gain");
    }

    TalosMetadataMapping mapping;
    const auto opened = mapping.open(ipc_directory + "/talos_ipc_meta");
    if (!opened) throw std::runtime_error(opened.message);
    const auto reader_result = mapping.reader();
    if (!reader_result) throw std::runtime_error(reader_result.status.message);
    const TalosMetadataReader reader = *reader_result.value;
    UdpGimbalClient gimbal;
    std::signal(SIGINT, stopHandler);
    std::signal(SIGTERM, stopHandler);

    std::uint64_t last_sequence = 0;
    std::uint64_t command_count = 0;
    std::uint64_t exact_count = 0;
    std::uint64_t target_count = 0;
    std::uint64_t behind_count = 0;
    while (running.load()) {
      const auto latest = reader.readLatestPose(kCameraPoseIndex);
      if (!latest || latest.value->frame_seq == 0 ||
          latest.value->frame_seq == last_sequence) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      const auto exact =
          reader.readGroundTruthForFrame(latest.value->frame_seq);
      if (!exact || exact.value->ground_truth.timestamp_ns !=
                        exact.value->exposure_state.timestamp_ns ||
          (exact.value->exposure_state.state_flags &
           (kExposureStateHasGimbalWorldPose | kExposureStateHasCameraWorldPose)) !=
              (kExposureStateHasGimbalWorldPose | kExposureStateHasCameraWorldPose)) {
        continue;
      }
      ++exact_count;
      last_sequence = latest.value->frame_seq;
      const auto* target = selectActiveRangeTarget(
          exact.value->ground_truth, exact.value->exposure_state,
          static_cast<std::uint8_t>(target_number));
      if (target == nullptr) continue;
      ++target_count;

      const auto& exposure = exact.value->exposure_state;
      const Eigen::Vector3d camera(exposure.camera_position_world[0],
                                   exposure.camera_position_world[1],
                                   exposure.camera_position_world[2]);
      Eigen::Vector3d target_world(target->position[0], target->position[1],
                                   target->position[2]);
      target_world += lookahead_s * Eigen::Vector3d(
          target->velocity[0], target->velocity[1], target->velocity[2]);
      Eigen::Quaterniond world_from_camera_link(
          exposure.camera_quaternion_world_wxyz[0],
          exposure.camera_quaternion_world_wxyz[1],
          exposure.camera_quaternion_world_wxyz[2],
          exposure.camera_quaternion_world_wxyz[3]);
      if (!world_from_camera_link.coeffs().allFinite() ||
          world_from_camera_link.norm() < 0.5) {
        continue;
      }
      const Eigen::Matrix3d world_from_optical =
          world_from_camera_link.normalized().toRotationMatrix() *
          cameraLinkFromOptical();
      const Eigen::Vector3d optical =
          world_from_optical.transpose() * (target_world - camera);
      if (!optical.allFinite() || optical.z() <= 0.1) {
        ++behind_count;
        continue;
      }
      const double yaw_error = std::atan2(optical.x(), optical.z());
      const double pitch_error =
          std::atan2(optical.y(), std::hypot(optical.x(), optical.z()));
      constexpr double radians_to_degrees = 180.0 / 3.14159265358979323846;

      UdpGimbalCommand command;
      command.yaw_deg = static_cast<float>(
          (exposure.gimbal_yaw_rad - gain * yaw_error) * radians_to_degrees);
      command.pitch_deg = static_cast<float>(
          90.0 + (exposure.gimbal_pitch_rad - gain * pitch_error) *
                     radians_to_degrees);
      command.distance_m = static_cast<float>(optical.norm());
      command.fire_advice = false;
      const auto sent = gimbal.sendTracked(command);
      if (!sent) throw std::runtime_error(sent.status.message);
      ++command_count;
      if (command_count == 1 || command_count % 250 == 0) {
        std::cout << "truth_gimbal command_count=" << command_count
                  << " frame_seq=" << last_sequence
                  << " yaw_error_deg=" << yaw_error * radians_to_degrees
                  << " pitch_error_deg=" << pitch_error * radians_to_degrees
                  << " command_yaw_deg=" << *command.yaw_deg
                  << " command_pitch_deg=" << *command.pitch_deg << '\n';
      }
    }
    std::cout << "truth_gimbal stopped command_count=" << command_count
              << " exact_count=" << exact_count
              << " target_count=" << target_count
              << " behind_count=" << behind_count << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "autoaim_research_truth_gimbal: " << error.what() << '\n';
    return 1;
  }
}
