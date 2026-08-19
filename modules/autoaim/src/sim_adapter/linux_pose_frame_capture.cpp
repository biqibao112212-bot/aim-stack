#include <daedalus_sim_sdk/client_result.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/talos_v1.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>
#include <daedalus_sim_sdk/tcp_image_v1.hpp>

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace {

using daedalus::sim::sdk::v1::ClientError;
using daedalus::sim::sdk::v1::ExposureState;
using daedalus::sim::sdk::v1::TalosCompatibility;
using daedalus::sim::sdk::v1::TalosMetadataMapping;
using daedalus::sim::sdk::v1::TalosMetadataReader;
using daedalus::sim::sdk::v1::TcpImageClient;
using daedalus::sim::sdk::v1::TcpImageFrame;
using daedalus::sim::sdk::v1::UdpEndpoint;
using daedalus::sim::sdk::v1::tcp_image::PixelFormat;

struct Options {
  std::filesystem::path ipc_dir;
  std::filesystem::path output_dir;
  std::string tcp_host = "127.0.0.1";
  std::uint16_t tcp_port = daedalus::sim::sdk::v1::kTcpImagePort;
  bool until_eof = false;
};

std::string argument(int argc, char** argv, const std::string& name) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) return argv[index + 1];
  }
  return {};
}

bool hasArgument(int argc, char** argv, const std::string& name) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] == name) return true;
  }
  return false;
}

Options parseOptions(int argc, char** argv) {
  Options options;
  options.ipc_dir = argument(argc, argv, "--ipc-dir");
  options.output_dir = argument(argc, argv, "--output-dir");
  const std::string host = argument(argc, argv, "--tcp-host");
  if (!host.empty()) options.tcp_host = host;
  const std::string port = argument(argc, argv, "--tcp-port");
  if (!port.empty()) {
    std::size_t consumed = 0;
    const unsigned long parsed = std::stoul(port, &consumed, 10);
    if (consumed != port.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::uint16_t>::max()) {
      throw std::invalid_argument("--tcp-port must be in 1..65535");
    }
    options.tcp_port = static_cast<std::uint16_t>(parsed);
  }
  options.until_eof = hasArgument(argc, argv, "--until-eof");
  if (options.ipc_dir.empty() || options.output_dir.empty() ||
      !options.until_eof) {
    throw std::invalid_argument(
        "required: --ipc-dir DIR --output-dir DIR --until-eof");
  }
  return options;
}

void appendArray(std::ostream& out, const float* values, std::size_t count) {
  out << '[';
  for (std::size_t index = 0; index < count; ++index) {
    if (index != 0) out << ',';
    out << values[index];
  }
  out << ']';
}

void writeExclusive(const std::filesystem::path& path,
                    const std::vector<std::uint8_t>& payload) {
  const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (descriptor < 0) {
    throw std::runtime_error("cannot create " + path.string() + ": " +
                             std::strerror(errno));
  }
  std::size_t offset = 0;
  while (offset < payload.size()) {
    const ssize_t written = ::write(
        descriptor, payload.data() + offset, payload.size() - offset);
    if (written <= 0) {
      const std::string message = std::strerror(errno);
      ::close(descriptor);
      throw std::runtime_error("cannot write " + path.string() + ": " +
                               message);
    }
    offset += static_cast<std::size_t>(written);
  }
  if (::close(descriptor) != 0) {
    throw std::runtime_error("cannot close " + path.string());
  }
}

TalosMetadataMapping openMapping(const std::filesystem::path& path) {
  TalosMetadataMapping mapping;
  std::string last_error;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(20);
  while (std::chrono::steady_clock::now() < deadline) {
    const auto status = mapping.open(path.string());
    if (status) return mapping;
    last_error = status.message;
    mapping.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error("cannot open compatible metadata: " + last_error);
}

TcpImageClient connectImages(const Options& options) {
  TcpImageClient client(UdpEndpoint{options.tcp_host, options.tcp_port});
  std::string last_error;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(20);
  while (std::chrono::steady_clock::now() < deadline) {
    const auto status = client.connect();
    if (status) return client;
    last_error = status.message;
    client.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error("cannot connect to TCP images: " + last_error);
}

void writeEvent(std::ostream& out, const TcpImageFrame& frame,
                const std::filesystem::path& relative_path) {
  const auto& header = frame.header;
  out << "{\"producer_epoch\":" << header.producer_epoch
      << ",\"frame_seq\":" << header.source_sequence
      << ",\"timestamp_ns\":" << header.capture_timestamp_ns
      << ",\"width\":" << header.width
      << ",\"height\":" << header.height
      << ",\"payload_bytes\":" << header.payload_bytes
      << ",\"pixel_format\":\"rgba32\""
      << ",\"raw_rgba_file\":\"" << relative_path.generic_string()
      << "\"}\n";
  out.flush();
  if (!out) throw std::runtime_error("failed to persist capture event");
}

void writeExposure(std::ostream& out, const TcpImageFrame& frame,
                   const ExposureState& exposure) {
  const auto& header = frame.header;
  out << std::setprecision(17)
      << "{\"schema_version\":\"aim-stack.exposure-frame/1\""
      << ",\"producer_epoch\":" << header.producer_epoch
      << ",\"frame_seq\":" << header.source_sequence
      << ",\"timestamp_ns\":" << header.capture_timestamp_ns
      << ",\"state_flags\":" << exposure.state_flags
      << ",\"world_frame\":" << static_cast<unsigned>(exposure.world_frame)
      << ",\"chassis_position_world_m\":";
  appendArray(out, exposure.chassis_position_world, 3);
  out << ",\"chassis_quaternion_world_wxyz\":";
  appendArray(out, exposure.chassis_quaternion_world_wxyz, 4);
  out << ",\"chassis_rpy_world_rad\":";
  appendArray(out, exposure.chassis_rpy_world, 3);
  out << ",\"gimbal_position_world_m\":";
  appendArray(out, exposure.gimbal_position_world, 3);
  out << ",\"gimbal_quaternion_world_wxyz\":";
  appendArray(out, exposure.gimbal_quaternion_world_wxyz, 4);
  out << ",\"camera_position_world_m\":";
  appendArray(out, exposure.camera_position_world, 3);
  out << ",\"camera_quaternion_world_wxyz\":";
  appendArray(out, exposure.camera_quaternion_world_wxyz, 4);
  out << ",\"gimbal_yaw_rad\":" << exposure.gimbal_yaw_rad
      << ",\"gimbal_pitch_rad\":" << exposure.gimbal_pitch_rad
      << ",\"source\":\"DaedalusSimSdk-1.3.1/readExposureStateForFrame\""
      << ",\"online_target_truth_read\":false"
      << ",\"future_truth_included\":false}\n";
  out.flush();
  if (!out) throw std::runtime_error("failed to persist exposure state");
}

int run(const Options& options) {
  if (!std::filesystem::is_directory(options.output_dir)) {
    throw std::invalid_argument("--output-dir must already exist");
  }
  const auto frames_dir = options.output_dir / "frames";
  if (!std::filesystem::create_directory(frames_dir)) {
    throw std::runtime_error("refusing to reuse frames directory");
  }
  std::ofstream events(options.output_dir / "capture-events.jsonl",
                       std::ios::out | std::ios::app);
  std::ofstream exposures(options.output_dir / "exposure-states.jsonl",
                          std::ios::out | std::ios::app);
  if (!events || !exposures) throw std::runtime_error("cannot create ledgers");

  auto mapping = openMapping(options.ipc_dir / "talos_ipc_meta");
  auto made_reader = mapping.reader();
  if (!made_reader ||
      made_reader.value->compatibility() != TalosCompatibility::Compatible) {
    throw std::runtime_error("metadata reader is not compatible");
  }
  TalosMetadataReader reader = *made_reader.value;
  auto images = connectImages(options);

  const auto ready = options.output_dir / "collector-ready";
  std::ofstream ready_stream(ready, std::ios::out | std::ios::app);
  if (!ready_stream) throw std::runtime_error("cannot create collector-ready");
  ready_stream << "connected\n";
  ready_stream.close();

  std::uint64_t epoch = 0;
  std::uint64_t previous = 0;
  std::uint64_t frame_count = 0;
  std::uint64_t exposure_count = 0;
  while (true) {
    auto received = images.waitForLatest(previous, std::chrono::milliseconds(1000));
    if (!received) {
      if ((received.status.error == ClientError::PeerClosed ||
           received.status.error == ClientError::ReceiveFailed) &&
          frame_count > 0) {
        break;
      }
      if (received.status.error == ClientError::Timeout && images.connected()) {
        continue;
      }
      throw std::runtime_error("TCP image receive failed: " +
                               received.status.message);
    }
    const TcpImageFrame& frame = *received.value;
    const auto& header = frame.header;
    if (header.format != PixelFormat::Rgba32 || header.width != 1440 ||
        header.height != 1080 || frame.payload.size() != header.payload_bytes) {
      throw std::runtime_error("capture requires complete 1440x1080 RGBA32 frames");
    }
    if ((epoch != 0 && epoch != header.producer_epoch) ||
        header.source_sequence <= previous) {
      throw std::runtime_error("producer epoch changed or frame sequence regressed");
    }
    epoch = header.producer_epoch;
    previous = header.source_sequence;
    const std::string filename = std::to_string(header.producer_epoch) + "_" +
                                 std::to_string(header.source_sequence) + "_" +
                                 std::to_string(header.capture_timestamp_ns) +
                                 ".rgba";
    const std::filesystem::path relative =
        std::filesystem::path("frames") / filename;
    writeExclusive(options.output_dir / relative, frame.payload);
    writeEvent(events, frame, relative);
    ++frame_count;

    auto exposure = reader.readExposureStateForFrame(header.source_sequence);
    if (!exposure) continue;
    const std::uint32_t required =
        daedalus::sim::sdk::v1::kExposureStateHasChassisWorldPose |
        daedalus::sim::sdk::v1::kExposureStateHasGimbalWorldPose |
        daedalus::sim::sdk::v1::kExposureStateHasCameraWorldPose;
    if (exposure.value->frame_seq != header.source_sequence ||
        exposure.value->timestamp_ns != header.capture_timestamp_ns ||
        (exposure.value->state_flags & required) != required) {
      continue;
    }
    writeExposure(exposures, frame, *exposure.value);
    ++exposure_count;
  }
  images.close();
  std::cout << "linux_pose_frame_capture_ok frames=" << frame_count
            << " exposures=" << exposure_count << " producer_epoch=" << epoch
            << " last_frame_seq=" << previous << '\n';
  return frame_count > 0 && exposure_count > 0 ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parseOptions(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "aim_sim_linux_pose_frame_capture: " << error.what() << '\n';
    return 2;
  }
}
