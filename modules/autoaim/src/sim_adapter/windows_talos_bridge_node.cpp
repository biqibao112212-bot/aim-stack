#include "aim_sim_bridge/pipeline.hpp"
#include "aim_sim_bridge/stage3_capture.hpp"

#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>
#include <daedalus_sim_sdk/udp_gimbal_client.hpp>

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

namespace {
using namespace daedalus::sim::sdk::v1;

const char* value(int argc, char** argv, const char* key, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == key) return argv[i + 1];
  return fallback;
}

int integer(int argc, char** argv, const char* key, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) {
      try { return std::stoi(argv[i + 1]); } catch (...) { return fallback; }
    }
  }
  try { return fallback; }
  catch (...) { return fallback; }
}
}  // namespace

int main(int argc, char** argv) {
  const std::string ipc_dir = value(argc, argv, "--ipc-dir", ".");
  const std::string engine = value(argc, argv, "--armor-engine", "models/armor.engine");
  const int seconds = integer(argc, argv, "--duration-seconds", 30);
  const std::string param_yaml = value(argc, argv, "--param-yaml", "config/param.sim.yaml");
  const std::string frame_log_path = value(argc, argv, "--frame-log", "");
  _putenv_s("AIM_SIM_PARAM_YAML", param_yaml.c_str());

  std::ofstream frame_log;
  if (!frame_log_path.empty()) {
    frame_log.open(frame_log_path, std::ios::out | std::ios::trunc);
    if (!frame_log) { std::cerr << "frame log open failed: " << frame_log_path << '\n'; return 5; }
  }

  aim_sim_bridge::AimBridgeConfig config;
  config.armor_detector_config = engine;
  // A no-target result has no valid aim point.  Leaving the simulator gimbal
  // untouched prevents the bridge from overriding shooting-range truth-gimbal
  // alignment while the detector is acquiring its first valid target.
  config.publish_no_target = false;
  config.pre_tracker_observation_sink = aim_sim_bridge::createStage3ObservationSinkFromEnv();
  auto pipeline = aim_sim_bridge::createAimPipeline(config);
  TcpImageClient images;
  const auto connected = images.connect();
  if (!connected) { std::cerr << "tcp connect failed: " << connected.message << '\n'; return 2; }

  TalosMetadataMapping metadata;
  const auto opened = metadata.open((std::filesystem::path(ipc_dir) / "talos_ipc_meta").string());
  if (!opened) { std::cerr << "metadata open failed: " << opened.message << '\n'; return 3; }
  UdpGimbalClient commands;
  std::uint64_t sequence = 0, processed = 0, sent = 0;
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(seconds)) {
    const auto image = images.waitForLatest(sequence, std::chrono::milliseconds(1000));
    if (!image) continue;
    sequence = image.value->header.source_sequence;
    const auto reader_result = metadata.reader();
    if (!reader_result) continue;
    const auto gimbal = reader_result.value->readGimbalStateForFrame(sequence);
    const auto camera = reader_result.value->readCameraInfo();
    const int type = image.value->header.format == tcp_image::PixelFormat::Rgba32 ? CV_8UC4 : CV_8UC3;
    // TcpImageClient exposes a const view. OpenCV's header constructor takes a
    // mutable pointer even when the source is read-only; cvtColor writes only
    // to frame.bgr_image below.
    auto* pixels = const_cast<std::uint8_t*>(image.value->payload.data());
    cv::Mat raw(static_cast<int>(image.value->header.height), static_cast<int>(image.value->header.width), type,
                pixels);
    aim_sim_bridge::SimFrame frame;
    if (type == CV_8UC4) cv::cvtColor(raw, frame.bgr_image, cv::COLOR_RGBA2BGR);
    else cv::cvtColor(raw, frame.bgr_image, cv::COLOR_RGB2BGR);
    frame.source_producer_epoch = image.value->header.producer_epoch;
    frame.source_image_seq = sequence;
    frame.source_capture_timestamp_ns = image.value->header.capture_timestamp_ns;
    frame.source_image_width = raw.cols;
    frame.source_image_height = raw.rows;
    if (gimbal) {
      frame.gimbal_pose_timestamp_ns = gimbal.value->timestamp_ns;
      frame.gimbal_pose_exposure_matched = gimbal.value->frame_seq == sequence;
      frame.gimbal_yaw_deg = gimbal.value->yaw_deg;
      frame.gimbal_pitch_deg = gimbal.value->pitch_deg;
      frame.gimbal_yaw_speed_deg_s = gimbal.value->yaw_velocity_deg_s;
    }
    if (camera) {
      frame.has_camera_matrix_override = true;
      frame.camera_matrix_override = (cv::Mat_<double>(3, 3) << camera.value->fx, 0., camera.value->cx,
          0., camera.value->fy, camera.value->cy, 0., 0., 1.);
    }
    const auto process_started = std::chrono::steady_clock::now();
    const auto output = pipeline->process(frame);
    const auto process_elapsed = std::chrono::steady_clock::now() - process_started;
    ++processed;
    bool command_sent = false;
    if (output.has_target || config.publish_no_target) {
      UdpGimbalCommand command;
      command.yaw_deg = static_cast<float>(output.yaw_deg);
      command.pitch_deg = static_cast<float>(output.pitch_deg);
      command.distance_m = static_cast<float>(output.distance_m);
      command.fire_advice = output.fire_advice;
      if (commands.send(command)) { ++sent; command_sent = true; }
    }
    if (frame_log) {
      const auto elapsed = std::chrono::steady_clock::now() - start;
      frame_log << "{\"elapsed_ms\":"
                << std::chrono::duration<double, std::milli>(elapsed).count()
                << ",\"source_sequence\":" << sequence
                << ",\"capture_timestamp_ns\":" << frame.source_capture_timestamp_ns
                << ",\"process_ms\":"
                << std::chrono::duration<double, std::milli>(process_elapsed).count()
                << ",\"has_target\":" << (output.has_target ? "true" : "false")
                << ",\"udp_sent\":" << (command_sent ? "true" : "false") << "}\n";
    }
  }
  const auto c = pipeline->counters();
  std::cout << "{\"processed\":" << processed << ",\"udp_sent\":" << sent
            << ",\"submitted\":" << c.submitted << ",\"detector_completed\":" << c.detector_completed
            << ",\"final_completed\":" << c.final_completed << ",\"delivered_completed\":" << c.delivered_completed << "}\n";
  return processed == 0 ? 4 : 0;
}
