#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>

#include "autoaim_research/coordinates.hpp"
#include "autoaim_research/detector.hpp"
#include "autoaim_research/pose_solver.hpp"
#include "autoaim_research/research_tracker.hpp"

namespace {
using namespace daedalus::sim::sdk::v1;

std::atomic<bool> running{true};

void stopHandler(int) { running.store(false); }

struct RuntimeConfig {
  std::string simulator_release_root;
  std::string ipc_directory;
  std::string model_path;
  std::string engine_path;
  std::string inference_backend;
  std::string output_path;
  float score_threshold = 0.5F;
  float nms_threshold = 0.45F;
  autoaim_research::ArmorGeometry geometry;
  autoaim_research::TrackerConfig tracker;
  std::uint64_t max_frames = 0;
  double duration_s = 0.0;
};

RuntimeConfig loadConfig(const std::string& path) {
  cv::FileStorage file(path, cv::FileStorage::READ);
  if (!file.isOpened()) throw std::runtime_error("cannot open config: " + path);
  RuntimeConfig config;
  file["simulator_release_root"] >> config.simulator_release_root;
  file["ipc_directory"] >> config.ipc_directory;
  file["model_path"] >> config.model_path;
  file["engine_path"] >> config.engine_path;
  file["inference_backend"] >> config.inference_backend;
  file["score_threshold"] >> config.score_threshold;
  file["nms_threshold"] >> config.nms_threshold;
  file["small_armor_width_m"] >> config.geometry.small_width_m;
  file["big_armor_width_m"] >> config.geometry.big_width_m;
  file["armor_height_m"] >> config.geometry.height_m;
  file["min_detect_count"] >> config.tracker.min_detect_count;
  file["max_temp_lost_count"] >> config.tracker.max_temp_lost_count;
  file["max_observation_gap_s"] >> config.tracker.max_observation_gap_s;
  file["initial_radius_m"] >> config.tracker.initial_radius_m;
  file["armor_count"] >> config.tracker.armor_count;
  double max_frames = 0.0;
  file["max_frames"] >> max_frames;
  config.max_frames = max_frames > 0.0 ? static_cast<std::uint64_t>(max_frames) : 0;
  file["duration_s"] >> config.duration_s;
  if (!std::isfinite(config.duration_s) || config.duration_s < 0.0) {
    throw std::runtime_error("duration_s must be finite and non-negative");
  }
  return config;
}

void parseArguments(int argc, char** argv, RuntimeConfig* config,
                    std::string* config_path) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (index + 1 >= argc) throw std::runtime_error("missing value for " + argument);
      return argv[++index];
    };
    if (argument == "--config") *config_path = value();
    else if (argument == "--model") config->model_path = value();
    else if (argument == "--engine") config->engine_path = value();
    else if (argument == "--backend") config->inference_backend = value();
    else if (argument == "--ipc-dir") config->ipc_directory = value();
    else if (argument == "--output") config->output_path = value();
    else if (argument == "--max-frames") config->max_frames = std::stoull(value());
    else if (argument == "--duration-s") config->duration_s = std::stod(value());
    else throw std::runtime_error("unknown argument: " + argument);
  }
}

cv::Mat frameToBgr(const TcpImageFrame& frame) {
  const int width = static_cast<int>(frame.header.width);
  const int height = static_cast<int>(frame.header.height);
  cv::Mat bgr;
  if (frame.header.format == tcp_image::PixelFormat::Rgba32) {
    const cv::Mat rgba(height, width, CV_8UC4,
                       const_cast<std::uint8_t*>(frame.payload.data()));
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
  } else {
    const cv::Mat rgb(height, width, CV_8UC3,
                      const_cast<std::uint8_t*>(frame.payload.data()));
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  }
  return bgr;
}

autoaim_research::ExposurePose exposurePose(const ExposureState& source) {
  autoaim_research::ExposurePose pose;
  pose.camera_position_world_m = Eigen::Vector3d(
      source.camera_position_world[0], source.camera_position_world[1],
      source.camera_position_world[2]);
  pose.gimbal_position_world_m = Eigen::Vector3d(
      source.gimbal_position_world[0], source.gimbal_position_world[1],
      source.gimbal_position_world[2]);
  pose.world_from_camera_link = Eigen::Quaterniond(
      source.camera_quaternion_world_wxyz[0], source.camera_quaternion_world_wxyz[1],
      source.camera_quaternion_world_wxyz[2], source.camera_quaternion_world_wxyz[3]);
  pose.world_from_gimbal = Eigen::Quaterniond(
      source.gimbal_quaternion_world_wxyz[0], source.gimbal_quaternion_world_wxyz[1],
      source.gimbal_quaternion_world_wxyz[2], source.gimbal_quaternion_world_wxyz[3]);
  return pose;
}

struct TruthMatch {
  const GroundTruthTarget* target = nullptr;
  int armor_slot = -1;
  Eigen::Vector3d armor_position_m{Eigen::Vector3d::Zero()};
  double center_error_m = std::numeric_limits<double>::infinity();
  double armor_error_m = std::numeric_limits<double>::infinity();
  double view_angle_error_rad = std::numeric_limits<double>::infinity();
};

Eigen::Vector3d targetCenter(const GroundTruthTarget& target,
                             const ExposureState& exposure) {
  return {target.position[0] - exposure.gimbal_position_world[0],
          target.position[1] - exposure.gimbal_position_world[1],
          target.position[2] - exposure.gimbal_position_world[2]};
}

const auto_aim::Armor* primaryObservation(
    const std::list<auto_aim::Armor>& solved) {
  if (solved.empty()) return nullptr;
  return &*std::min_element(
      solved.begin(), solved.end(),
      [](const auto_aim::Armor& left, const auto_aim::Armor& right) {
        constexpr double center_x = 720.0;
        constexpr double center_y = 540.0;
        return std::hypot(left.center.x - center_x, left.center.y - center_y) <
               std::hypot(right.center.x - center_x, right.center.y - center_y);
      });
}

TruthMatch selectResearchTarget(const GroundTruthBatch& truth,
                                const ExposureState& exposure,
                                const std::list<auto_aim::Armor>& solved,
                                const autoaim_research::TrackerSnapshot& tracker) {
  TruthMatch match;
  const auto_aim::Armor* observation = primaryObservation(solved);
  const bool has_observation = observation != nullptr;
  const bool has_tracker_center =
      tracker.has_estimate && tracker.state_vector.size() == 11;
  const Eigen::Vector3d tracker_center = has_tracker_center
      ? Eigen::Vector3d(tracker.state_vector[0], tracker.state_vector[2],
                        tracker.state_vector[4])
      : Eigen::Vector3d::Zero();
  const Eigen::Quaterniond world_from_camera =
      exposurePose(exposure).world_from_camera_link.normalized();
  const Eigen::Vector3d camera_forward_world =
      world_from_camera * Eigen::Vector3d::UnitX();

  for (std::uint32_t index = 0; index < truth.target_count; ++index) {
    const auto& target = truth.targets[index];
    if ((target.state_flags & kGroundTruthTargetHasWorldState) == 0 ||
        target.world_state_frame != kGroundTruthFrameRosOdom) {
      continue;
    }
    const Eigen::Vector3d center = targetCenter(target, exposure);
    const double center_error = has_tracker_center
        ? (tracker_center - center).norm()
        : std::numeric_limits<double>::infinity();
    const Eigen::Vector3d camera_to_target(
        target.position[0] - exposure.camera_position_world[0],
        target.position[1] - exposure.camera_position_world[1],
        target.position[2] - exposure.camera_position_world[2]);
    double view_angle_error = std::numeric_limits<double>::infinity();
    if (camera_to_target.allFinite() && camera_to_target.norm() > 1e-9) {
      const double cosine = std::clamp(
          camera_forward_world.dot(camera_to_target.normalized()), -1.0, 1.0);
      view_angle_error = std::acos(cosine);
    }

    double armor_error = std::numeric_limits<double>::infinity();
    int armor_slot = -1;
    Eigen::Vector3d armor_position = Eigen::Vector3d::Zero();
    if (observation != nullptr &&
        (target.state_flags & kGroundTruthTargetHasWorldOrientation) != 0 &&
        (target.state_flags & kGroundTruthTargetHasArmorGeometry) != 0 &&
        target.armor_count == 4) {
      const Eigen::Quaterniond world_from_target(
          target.world_quaternion_wxyz[0], target.world_quaternion_wxyz[1],
          target.world_quaternion_wxyz[2], target.world_quaternion_wxyz[3]);
      if (world_from_target.coeffs().allFinite() &&
          world_from_target.norm() > 0.5) {
        for (std::uint8_t slot = 0; slot < target.armor_count; ++slot) {
          const auto& armor = target.armors[slot];
          if (armor.visibility == kGroundTruthVisibilityHidden) continue;
          const Eigen::Vector3d local(armor.relative_position[0],
                                      armor.relative_position[1],
                                      armor.relative_position[2]);
          const Eigen::Vector3d candidate =
              center + world_from_target.normalized() * local;
          const double error = (observation->xyz_in_world - candidate).norm();
          if (error < armor_error) {
            armor_error = error;
            armor_slot = armor.relative_slot;
            armor_position = candidate;
          }
        }
      }
    }

    const double score = has_tracker_center
        ? center_error
        : (has_observation ? armor_error : view_angle_error);
    const double best_score = has_tracker_center
        ? match.center_error_m
        : (has_observation ? match.armor_error_m
                           : match.view_angle_error_rad);
    if (score < best_score) {
      match.target = &target;
      match.center_error_m = center_error;
      match.armor_error_m = armor_error;
      match.view_angle_error_rad = view_angle_error;
      match.armor_slot = armor_slot;
      match.armor_position_m = armor_position;
    }
  }
  return match;
}

void writeVector(std::ostream& output, const Eigen::VectorXd& values) {
  output << '[';
  for (Eigen::Index index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

void writeFiniteOrNull(std::ostream& output, double value) {
  if (std::isfinite(value)) output << value;
  else output << "null";
}

struct PipelineTiming {
  double color_convert_ms = 0.0;
  double pnp_ms = 0.0;
  double tracker_ms = 0.0;
  double total_ms = 0.0;
};

void writeFrame(std::ostream& output, const TcpImageFrame& frame,
                const GroundTruthExposureSnapshot& exact,
                const std::list<auto_aim::Armor>& solved,
                const autoaim_research::TrackerSnapshot& tracker,
                const std::string& inference_backend,
                const autoaim_research::DetectorTiming& detector_timing,
                const PipelineTiming& pipeline_timing) {
  const auto_aim::Armor* observation = primaryObservation(solved);
  const TruthMatch match = selectResearchTarget(
      exact.ground_truth, exact.exposure_state, solved, tracker);
  const GroundTruthTarget* truth = match.target;
  output << std::setprecision(12) << '{'
         << "\"schema\":\"autoaim-research-frame-v2\""
         << ",\"producer_epoch\":" << frame.header.producer_epoch
         << ",\"frame_seq\":" << frame.header.source_sequence
         << ",\"timestamp_ns\":" << frame.header.capture_timestamp_ns
         << ",\"inference_backend\":\"" << inference_backend << "\""
         << ",\"timing_ms\":{"
         << "\"color_convert\":" << pipeline_timing.color_convert_ms
         << ",\"detector_host_prepare\":"
         << detector_timing.host_prepare_ms
         << ",\"detector_gpu_preprocess\":"
         << detector_timing.gpu_preprocess_ms
         << ",\"detector_inference\":" << detector_timing.inference_ms
         << ",\"detector_output_copy\":"
         << detector_timing.output_copy_ms
         << ",\"detector_postprocess\":"
         << detector_timing.postprocess_ms
         << ",\"detector_total\":" << detector_timing.total_ms
         << ",\"pnp\":" << pipeline_timing.pnp_ms
         << ",\"tracker\":" << pipeline_timing.tracker_ms
         << ",\"pipeline_total\":" << pipeline_timing.total_ms << '}'
         << ",\"detection_count\":" << solved.size()
         << ",\"tracker_state\":\"" << tracker.state << "\""
         << ",\"ekf_state\":";
  if (tracker.has_estimate) writeVector(output, tracker.state_vector);
  else output << "null";
  output << ",\"ekf_estimate\":";
  if (!tracker.has_estimate || tracker.state_vector.size() != 11) {
    output << "null";
  } else {
    const auto& state = tracker.state_vector;
    output << "{\"center_m\":[" << state[0] << ',' << state[2] << ','
           << state[4] << "],\"velocity_mps\":[" << state[1] << ','
           << state[3] << ',' << state[5] << "],\"yaw_rad\":" << state[6]
           << ",\"omega_rad_s\":" << state[7]
           << ",\"radius_even_m\":" << state[8]
           << ",\"radius_odd_m\":" << state[8] + state[9]
           << ",\"height_even_m\":" << state[4]
           << ",\"height_odd_m\":" << state[4] + state[10] << '}';
  }
  output << ",\"primary_pnp\":";
  if (observation == nullptr) {
    output << "null";
  } else {
    const auto& armor = *observation;
    output << "{\"xyz_m\":[" << armor.xyz_in_world.x() << ','
           << armor.xyz_in_world.y() << ',' << armor.xyz_in_world.z()
           << "],\"yaw_rad\":" << armor.ypr_in_world[0]
           << ",\"confidence\":" << armor.confidence
           << ",\"name\":\"" << auto_aim::ARMOR_NAMES[armor.name] << "\""
           << ",\"type\":\"" << auto_aim::ARMOR_TYPES[armor.type] << "\"}";
  }
  output << ",\"truth\":";
  if (truth == nullptr) {
    output << "null";
  } else {
    output << "{\"target_id\":" << truth->target_id
           << ",\"match_center_error_m\":";
    writeFiniteOrNull(output, match.center_error_m);
    output << ",\"match_armor_error_m\":";
    writeFiniteOrNull(output, match.armor_error_m);
    output << ",\"match_view_angle_error_rad\":";
    writeFiniteOrNull(output, match.view_angle_error_rad);
    output << ",\"matched_armor_slot\":" << match.armor_slot
           << ",\"armor_m\":";
    if (match.armor_slot < 0) output << "null";
    else output << '[' << match.armor_position_m.x() << ','
                << match.armor_position_m.y() << ','
                << match.armor_position_m.z() << ']';
    output << ",\"center_m\":["
           << truth->position[0] - exact.exposure_state.gimbal_position_world[0] << ','
           << truth->position[1] - exact.exposure_state.gimbal_position_world[1] << ','
           << truth->position[2] - exact.exposure_state.gimbal_position_world[2]
           << "],\"center_world_m\":[" << truth->position[0] << ','
           << truth->position[1] << ',' << truth->position[2]
           << "],\"research_origin_world_m\":["
           << exact.exposure_state.gimbal_position_world[0] << ','
           << exact.exposure_state.gimbal_position_world[1] << ','
           << exact.exposure_state.gimbal_position_world[2]
           << "],\"velocity_mps\":[" << truth->velocity[0] << ','
           << truth->velocity[1] << ',' << truth->velocity[2]
           << "],\"yaw_rad\":" << truth->yaw
           << ",\"omega_rad_s\":" << truth->vyaw
           << ",\"radius_even_m\":" << truth->radius_even
           << ",\"radius_odd_m\":" << truth->radius_odd << '}';
  }
  output << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string config_path = "config/research.yaml";
    RuntimeConfig overrides;
    parseArguments(argc, argv, &overrides, &config_path);
    RuntimeConfig config = loadConfig(config_path);
    if (!overrides.model_path.empty()) config.model_path = overrides.model_path;
    if (!overrides.engine_path.empty()) config.engine_path = overrides.engine_path;
    if (!overrides.inference_backend.empty()) {
      config.inference_backend = overrides.inference_backend;
    }
    if (!overrides.ipc_directory.empty()) config.ipc_directory = overrides.ipc_directory;
    if (!overrides.output_path.empty()) config.output_path = overrides.output_path;
    if (overrides.max_frames != 0) config.max_frames = overrides.max_frames;
    if (overrides.duration_s > 0.0) config.duration_s = overrides.duration_s;

    if (!std::filesystem::is_regular_file(
            std::filesystem::path(config.simulator_release_root) / "release.json")) {
      throw std::runtime_error("locked simulator release is missing");
    }
    if (!config.output_path.empty() && std::filesystem::exists(config.output_path)) {
      throw std::runtime_error("refusing to overwrite existing output: " +
                               config.output_path);
    }

    autoaim_research::DetectorConfig detector_config;
    detector_config.model_path = config.model_path;
    detector_config.engine_path = config.engine_path;
    detector_config.inference_backend = config.inference_backend;
    detector_config.score_threshold = config.score_threshold;
    detector_config.nms_threshold = config.nms_threshold;
    autoaim_research::TongjiYoloDetector detector(std::move(detector_config));
    autoaim_research::TongjiResearchTracker tracker(config.tracker);

    ContestClientOptions options;
    options.ipc_directory = config.ipc_directory;
    ContestClient simulator(options);
    const ClientStatus connected = simulator.connect();
    if (!connected) throw std::runtime_error(connected.message);
    const auto health = simulator.health();
    if (!health || health.value->product_version != "1.4.0-learning-r1" ||
        health.value->distribution_profile != "learning" ||
        health.value->competition_eligible || !health.value->online_ground_truth_enabled ||
        health.value->future_truth_included) {
      throw std::runtime_error("runtime is not the locked 1.4.0-learning-r1 profile");
    }
    const auto scene = simulator.selectScene(ContestScene::ShootingRange);
    if (!scene) throw std::runtime_error(scene.status.message);

    TalosMetadataMapping mapping;
    const ClientStatus mapped =
        mapping.open(config.ipc_directory + "/talos_ipc_meta");
    if (!mapped) throw std::runtime_error(mapped.message);
    const auto reader_result = mapping.reader();
    if (!reader_result) throw std::runtime_error(reader_result.status.message);
    const TalosMetadataReader reader = *reader_result.value;
    const auto camera_info = reader.readCameraInfo();
    if (!camera_info) throw std::runtime_error(camera_info.status.message);
    cv::Mat camera = (cv::Mat_<double>(3, 3) << camera_info.value->fx, 0.0,
                      camera_info.value->cx, 0.0, camera_info.value->fy,
                      camera_info.value->cy, 0.0, 0.0, 1.0);
    cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
    for (int index = 0; index < 5; ++index) {
      distortion.at<double>(0, index) = camera_info.value->distortion[index];
    }
    autoaim_research::TongjiPoseSolver solver(camera, distortion,
                                              config.geometry);

    std::ofstream file;
    std::ostream* output = &std::cout;
    if (!config.output_path.empty()) {
      file.open(config.output_path, std::ios::out | std::ios::binary);
      if (!file) throw std::runtime_error("cannot create output: " + config.output_path);
      output = &file;
    }
    std::signal(SIGINT, stopHandler);
    std::signal(SIGTERM, stopHandler);

    std::uint64_t after_sequence = 0;
    std::uint64_t accepted = 0;
    std::uint64_t first_timestamp_ns = 0;
    std::uint64_t frame_join_retries = 0;
    std::uint64_t frame_timeouts = 0;
    int consecutive_join_retries = 0;
    while (running.load() && (config.max_frames == 0 || accepted < config.max_frames)) {
      const auto frame_result = simulator.nextFrame(after_sequence);
      if (!frame_result) {
        if (frame_result.status.error == ClientError::Timeout) {
          ++frame_timeouts;
          continue;
        }
        const bool recoverable_join_miss =
            (frame_result.status.error == ClientError::ProtocolError ||
             frame_result.status.error == ClientError::UnstableSnapshot) &&
            frame_result.status.message.find("ground-truth history") !=
                std::string::npos;
        if (recoverable_join_miss) {
          ++frame_join_retries;
          ++consecutive_join_retries;
          if (consecutive_join_retries > 1000) {
            throw std::runtime_error(
                "ground-truth join did not recover after 1000 retries");
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        throw std::runtime_error(frame_result.status.message);
      }
      consecutive_join_retries = 0;
      const auto& frame = *frame_result.value;
      after_sequence = frame.image.header.source_sequence;
      const auto exact = reader.readGroundTruthForFrame(after_sequence);
      if (!exact || exact.value->producer_epoch != frame.image.header.producer_epoch ||
          exact.value->ground_truth.timestamp_ns !=
              frame.image.header.capture_timestamp_ns ||
          exact.value->exposure_state.timestamp_ns !=
              frame.image.header.capture_timestamp_ns) {
        continue;
      }
      if ((exact.value->exposure_state.state_flags &
           (kExposureStateHasCameraWorldPose | kExposureStateHasGimbalWorldPose)) !=
          (kExposureStateHasCameraWorldPose | kExposureStateHasGimbalWorldPose)) {
        continue;
      }

      const auto pipeline_begin = std::chrono::steady_clock::now();
      const cv::Mat bgr = frameToBgr(frame.image);
      const auto color_end = std::chrono::steady_clock::now();
      auto detected = detector.detect(bgr);
      const auto detector_end = std::chrono::steady_clock::now();
      std::list<auto_aim::Armor> solved;
      const auto pose = exposurePose(exact.value->exposure_state);
      for (auto& armor : detected) {
        const auto result = solver.solve(armor, pose);
        if (result.valid) solved.push_back(armor);
      }
      const auto pnp_end = std::chrono::steady_clock::now();
      const auto tracker_snapshot =
          tracker.update(solved, frame.image.header.capture_timestamp_ns);
      const auto tracker_end = std::chrono::steady_clock::now();
      const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
      };
      PipelineTiming timing;
      timing.color_convert_ms = milliseconds(pipeline_begin, color_end);
      timing.pnp_ms = milliseconds(detector_end, pnp_end);
      timing.tracker_ms = milliseconds(pnp_end, tracker_end);
      timing.total_ms = milliseconds(pipeline_begin, tracker_end);
      writeFrame(*output, frame.image, *exact.value, solved, tracker_snapshot,
                 detector.backendName(), detector.lastTiming(), timing);
      ++accepted;
      if (first_timestamp_ns == 0) {
        first_timestamp_ns = frame.image.header.capture_timestamp_ns;
      }
      if (config.duration_s > 0.0 &&
          frame.image.header.capture_timestamp_ns >= first_timestamp_ns &&
          static_cast<double>(frame.image.header.capture_timestamp_ns -
                              first_timestamp_ns) *
                  1e-9 >=
              config.duration_s) {
        break;
      }
    }
    std::cerr << "autoaim_research_runner complete backend="
              << detector.backendName() << " accepted=" << accepted
              << " frame_join_retries=" << frame_join_retries
              << " frame_timeouts=" << frame_timeouts << '\n';
    simulator.close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "autoaim_research_runner: " << error.what() << "\n";
    return 1;
  }
}
