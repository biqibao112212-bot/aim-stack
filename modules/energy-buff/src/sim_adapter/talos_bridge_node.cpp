#include "aim_sim_bridge/debug_telemetry.hpp"
#include "aim_sim_bridge/fixed_rate_command_loop.hpp"
#include "aim_sim_bridge/pipeline.hpp"
#include "aim_sim_bridge/sim_command.hpp"
#include "aim_sim_bridge/tcp_image_receiver.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <netdb.h>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kShmMagic = 0x54414C05;
constexpr std::uint32_t kShmVersion = 6;
constexpr std::uint32_t kImageWidth = 1280;
constexpr std::uint32_t kImageHeight = 720;
constexpr std::uint32_t kImageChannels = 3;
constexpr std::size_t kMaxImagePayloadBytes =
    kImageWidth * kImageHeight * kImageChannels;
// Talos v6 keeps fixed maximum-size slot addresses even when ImageMeta declares
// a smaller tightly packed RGB payload.
constexpr std::size_t kImageSlotStrideBytes = kMaxImagePayloadBytes;
constexpr std::size_t kImagePoolSize = kImageSlotStrideBytes * 3;
constexpr std::size_t kMetaSize = 76992;
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr std::size_t kGimbalPoseIndex = 0;
constexpr std::size_t kOdomPoseIndex = 1;
constexpr std::uint8_t kFlagNew = 0x80;
constexpr std::uint8_t kIndexMask = 0x03;

struct alignas(32) ImageMeta {
  std::uint64_t seq;
  std::uint64_t timestamp_ns;
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t buffer_id;
  std::uint8_t format;
  std::uint8_t pad[6];
};
static_assert(sizeof(ImageMeta) == 32);

struct alignas(64) PoseMeta {
  std::uint64_t frame_seq;
  float position[3];
  float quaternion[4];
  std::uint64_t timestamp_ns;
  std::uint8_t pad[16];
};
static_assert(sizeof(PoseMeta) == 64);

struct alignas(32) GimbalCmd {
  std::uint64_t timestamp_ns;
  float yaw_deg;
  float pitch_deg;
  float distance_m;
  std::uint8_t fire_advice;
  std::uint8_t pad[11];
};
static_assert(sizeof(GimbalCmd) == 32);

struct alignas(64) CameraInfo {
  std::uint64_t timestamp_ns;
  double fx;
  double fy;
  double cx;
  double cy;
  double distortion[5];
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t pad[24];
};
static_assert(sizeof(CameraInfo) == 128);

struct alignas(64) ChassisObservation {
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  float dt_s;
  float v_body[2];
  float wz_radps;
  float wheel_linear_mps[4];
  float wheel_angular_radps[4];
  float a_body[2];
  float alpha_z_radps2;
  float rpy_rad[3];
  float gyro_xyz_radps[3];
  float accel_xyz_mps2[3];
  std::uint8_t pad[16];
};
static_assert(sizeof(ChassisObservation) == 128);

struct alignas(64) RuntimeState {
  std::uint64_t timestamp_ns;
  std::uint8_t following;
  std::uint8_t pad1[3];
  float gimbal_yaw_rad;
  float gimbal_pitch_rad;
  std::uint8_t pad[44];
};
static_assert(sizeof(RuntimeState) == 64);

constexpr std::size_t kGroundTruthMaxTargets = 16;
constexpr std::size_t kGroundTruthMaxRunes = 4;
constexpr std::size_t kGroundTruthMaxArmorsPerTarget = 4;
constexpr std::uint8_t kGroundTruthVisibilityUnknown = 0;
constexpr std::uint8_t kGroundTruthVisibilityHidden = 1;
constexpr std::uint8_t kGroundTruthFrameUnknown = 0;
constexpr std::uint8_t kGroundTruthFrameRosOdom = 1;
constexpr std::uint8_t kGroundTruthFrameChassisLocalRos = 2;
constexpr std::uint32_t kGroundTruthTargetHasWorldState = 1U << 0;
constexpr std::uint32_t kGroundTruthTargetHasWorldOrientation = 1U << 1;
constexpr std::uint32_t kGroundTruthTargetHasArmorGeometry = 1U << 2;
constexpr std::uint32_t kExposureStateHasChassisWorldPose = 1U << 0;
constexpr std::uint32_t kExposureStateHasGimbalWorldPose = 1U << 1;
constexpr std::uint32_t kExposureStateHasCameraWorldPose = 1U << 2;
constexpr std::size_t kGroundTruthHistorySlots = 16;

struct alignas(32) GroundTruthArmor {
  std::uint8_t relative_slot;
  std::uint8_t visibility;
  std::uint8_t pad1[2];
  float relative_position[3];
  float outward_normal[3];
  float relative_yaw;
};
static_assert(sizeof(GroundTruthArmor) == 32);
static_assert(offsetof(GroundTruthArmor, relative_position) == 4);
static_assert(offsetof(GroundTruthArmor, outward_normal) == 16);
static_assert(offsetof(GroundTruthArmor, relative_yaw) == 28);

struct alignas(32) GroundTruthTarget {
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint8_t team;
  std::uint8_t armor_label;
  std::uint8_t is_outpost;
  std::uint8_t armor_count;
  float position[3];
  float vyaw;
  float yaw;
  float velocity[3];
  float radius_even;
  float radius_odd;
  float armor_height;
  GroundTruthArmor armors[kGroundTruthMaxArmorsPerTarget];
  std::uint64_t target_id;
  float world_quaternion_wxyz[4];
  std::uint8_t world_state_frame;
  std::uint8_t armor_geometry_frame;
  std::uint8_t pad2[2];
  std::uint32_t state_flags;
};
static_assert(sizeof(GroundTruthTarget) == 224);
static_assert(offsetof(GroundTruthTarget, position) == 20);
static_assert(offsetof(GroundTruthTarget, vyaw) == 32);
static_assert(offsetof(GroundTruthTarget, yaw) == 36);
static_assert(offsetof(GroundTruthTarget, velocity) == 40);
static_assert(offsetof(GroundTruthTarget, radius_even) == 52);
static_assert(offsetof(GroundTruthTarget, armors) == 64);
static_assert(offsetof(GroundTruthTarget, target_id) == 192);
static_assert(offsetof(GroundTruthTarget, world_quaternion_wxyz) == 200);
static_assert(offsetof(GroundTruthTarget, world_state_frame) == 216);
static_assert(offsetof(GroundTruthTarget, state_flags) == 220);

struct alignas(64) GroundTruthRune {
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint8_t team;
  std::uint8_t rune_mode;
  std::uint8_t mechanism_state;
  std::uint8_t pad1;
  float r_center_odom[3];
  float radius;
  float current_angle;
  float v_roll;
  std::int32_t direction;
  float sin_amplitude;
  float sin_omega;
  float sin_phase;
  float sin_offset;
  float relative_time;
  std::int32_t blade_id;
  std::uint8_t target_activations[5];
  std::uint8_t pad[20];
};
static_assert(sizeof(GroundTruthRune) == 128);

struct alignas(64) GroundTruthBatch {
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t target_count;
  std::uint32_t rune_count;
  GroundTruthTarget targets[kGroundTruthMaxTargets];
  GroundTruthRune runes[kGroundTruthMaxRunes];
  std::uint8_t pad[64];
};
static_assert(sizeof(GroundTruthBatch) == 4224);
static_assert(offsetof(GroundTruthBatch, targets) == 32);
static_assert(offsetof(GroundTruthBatch, runes) == 3648);

struct alignas(64) ExposureState {
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t state_flags;
  std::uint8_t world_frame;
  std::uint8_t pad1[3];
  float chassis_position_world[3];
  float chassis_quaternion_world_wxyz[4];
  float chassis_rpy_world[3];
  float gimbal_position_world[3];
  float gimbal_quaternion_world_wxyz[4];
  float camera_position_world[3];
  float camera_quaternion_world_wxyz[4];
  std::uint8_t pad[8];
};
static_assert(sizeof(ExposureState) == 128);
static_assert(offsetof(ExposureState, chassis_position_world) == 24);
static_assert(offsetof(ExposureState, gimbal_position_world) == 64);
static_assert(offsetof(ExposureState, camera_position_world) == 92);

struct alignas(64) GroundTruthHistorySlot {
  std::uint64_t commit_seq;
  std::uint8_t pad1[56];
  GroundTruthBatch ground_truth;
  ExposureState exposure_state;
};
static_assert(sizeof(GroundTruthHistorySlot) == 4416);
static_assert(offsetof(GroundTruthHistorySlot, ground_truth) == 64);
static_assert(offsetof(GroundTruthHistorySlot, exposure_state) == 4288);

struct alignas(64) GroundTruthHistory {
  std::uint64_t next_publication;
  std::uint8_t pad1[56];
  GroundTruthHistorySlot slots[kGroundTruthHistorySlots];
};
static_assert(sizeof(GroundTruthHistory) == 70720);
static_assert(offsetof(GroundTruthHistory, slots) == 64);

struct GimbalFeedback {
  double yaw_deg = 0.0;
  double pitch_deg = 0.0;
  std::uint64_t frame_seq = 0;
  std::uint64_t timestamp_ns = 0;
};

template <typename Slot, std::size_t Size> struct alignas(64) TripleBuffer {
  std::uint8_t state;
  std::uint8_t write_idx;
  std::uint8_t read_idx;
  std::uint8_t pad[61];
  Slot slots[3];
};

using ImageTripleBuffer = TripleBuffer<ImageMeta, 192>;
using PoseTripleBuffer = TripleBuffer<PoseMeta, 256>;
using GimbalTripleBuffer = TripleBuffer<GimbalCmd, 192>;
static_assert(sizeof(ImageTripleBuffer) == 192);
static_assert(sizeof(PoseTripleBuffer) == 256);
static_assert(sizeof(GimbalTripleBuffer) == 192);

struct alignas(64) ShmHeader {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t created_ns;
  std::uint64_t heartbeat_ns;
  std::uint32_t image_width;
  std::uint32_t image_height;
  std::uint8_t pad[32];
};
static_assert(sizeof(ShmHeader) == 64);

struct ShmMetaRegion {
  ShmHeader header;
  ImageTripleBuffer image;
  PoseTripleBuffer poses[5];
  GimbalTripleBuffer gimbal_cmd;
  CameraInfo camera_info;
  ChassisObservation chassis_observation;
  GroundTruthBatch ground_truth;
  RuntimeState runtime_state;
  GroundTruthHistory ground_truth_history;
};
static_assert(sizeof(ShmMetaRegion) == kMetaSize);
static_assert(offsetof(ShmMetaRegion, gimbal_cmd) == 1536);
static_assert(offsetof(ShmMetaRegion, camera_info) == 1728);
static_assert(offsetof(ShmMetaRegion, chassis_observation) == 1856);
static_assert(offsetof(ShmMetaRegion, ground_truth) == 1984);
static_assert(offsetof(ShmMetaRegion, runtime_state) == 6208);
static_assert(offsetof(ShmMetaRegion, ground_truth_history) == 6272);

class MmapRegion {
public:
  MmapRegion() = default;
  MmapRegion(const MmapRegion &) = delete;
  MmapRegion &operator=(const MmapRegion &) = delete;

  ~MmapRegion() { close(); }

  bool openExisting(const std::filesystem::path &path, std::size_t size) {
    close();
    fd_ = ::open(path.string().c_str(), O_RDWR);
    if (fd_ < 0) {
      return false;
    }
    size_ = size;
    data_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      ::close(fd_);
      fd_ = -1;
      return false;
    }
    return true;
  }

  void close() {
    if (data_ != nullptr) {
      ::munmap(data_, size_);
      data_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    size_ = 0;
  }

  template <typename T> T *as() { return reinterpret_cast<T *>(data_); }

  std::uint8_t *bytes() { return reinterpret_cast<std::uint8_t *>(data_); }

private:
  int fd_ = -1;
  void *data_ = nullptr;
  std::size_t size_ = 0;
};

class FileHandle {
public:
  FileHandle() = default;
  FileHandle(const FileHandle &) = delete;
  FileHandle &operator=(const FileHandle &) = delete;

  ~FileHandle() { close(); }

  bool openExisting(const std::filesystem::path &path, int flags) {
    close();
    fd_ = ::open(path.string().c_str(), flags);
    return fd_ >= 0;
  }

  void close() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int fd() const { return fd_; }

private:
  int fd_ = -1;
};

bool readFullyAt(int fd, void *data, std::size_t size, std::uint64_t offset) {
  auto *out = static_cast<std::uint8_t *>(data);
  std::size_t done = 0;
  while (done < size) {
    const ssize_t n =
        ::pread(fd, out + done, size - done, static_cast<off_t>(offset + done));
    if (n <= 0) {
      return false;
    }
    done += static_cast<std::size_t>(n);
  }
  return true;
}

struct ExactExposureTruth {
  GroundTruthBatch ground_truth{};
  ExposureState exposure_state{};
  std::uint64_t commit_seq = 0;
  std::size_t slot_index = 0;
};

bool selectExactExposureTruth(const GroundTruthHistory &history,
                              std::uint64_t frame_seq,
                              std::uint64_t timestamp_ns,
                              ExactExposureTruth *out) {
  if (out == nullptr || frame_seq == 0 || timestamp_ns == 0)
    return false;
  for (std::size_t i = 0; i < kGroundTruthHistorySlots; ++i) {
    const GroundTruthHistorySlot &slot = history.slots[i];
    if (slot.commit_seq == 0 || (slot.commit_seq & 1U) != 0 ||
        slot.ground_truth.frame_seq != frame_seq ||
        slot.ground_truth.timestamp_ns != timestamp_ns ||
        slot.exposure_state.frame_seq != frame_seq ||
        slot.exposure_state.timestamp_ns != timestamp_ns) {
      continue;
    }
    out->ground_truth = slot.ground_truth;
    out->exposure_state = slot.exposure_state;
    out->commit_seq = slot.commit_seq;
    out->slot_index = i;
    return true;
  }
  return false;
}

bool readExactExposureTruth(int fd, std::uint64_t frame_seq,
                            std::uint64_t timestamp_ns,
                            ExactExposureTruth *out) {
  if (fd < 0 || out == nullptr || frame_seq == 0 || timestamp_ns == 0)
    return false;
  const std::uint64_t slots_base =
      offsetof(ShmMetaRegion, ground_truth_history) +
      offsetof(GroundTruthHistory, slots);
  for (std::size_t i = 0; i < kGroundTruthHistorySlots; ++i) {
    const std::uint64_t slot_base =
        slots_base + static_cast<std::uint64_t>(i) *
                         sizeof(GroundTruthHistorySlot);
    std::uint64_t before = 0;
    if (!readFullyAt(fd, &before, sizeof(before), slot_base) || before == 0 ||
        (before & 1U) != 0) {
      continue;
    }
    ExactExposureTruth candidate{};
    if (!readFullyAt(fd, &candidate.ground_truth,
                     sizeof(candidate.ground_truth),
                     slot_base + offsetof(GroundTruthHistorySlot,
                                          ground_truth)) ||
        !readFullyAt(fd, &candidate.exposure_state,
                     sizeof(candidate.exposure_state),
                     slot_base + offsetof(GroundTruthHistorySlot,
                                          exposure_state))) {
      continue;
    }
    std::uint64_t after = 0;
    if (!readFullyAt(fd, &after, sizeof(after), slot_base) || before != after ||
        (after & 1U) != 0) {
      continue;
    }
    if (candidate.ground_truth.frame_seq != frame_seq ||
        candidate.ground_truth.timestamp_ns != timestamp_ns ||
        candidate.exposure_state.frame_seq != frame_seq ||
        candidate.exposure_state.timestamp_ns != timestamp_ns) {
      continue;
    }
    candidate.commit_seq = after;
    candidate.slot_index = i;
    *out = candidate;
    return true;
  }
  return false;
}

bool writeFullyAt(int fd, const void *data, std::size_t size,
                  std::uint64_t offset) {
  const auto *in = static_cast<const std::uint8_t *>(data);
  std::size_t done = 0;
  while (done < size) {
    const ssize_t n =
        ::pwrite(fd, in + done, size - done, static_cast<off_t>(offset + done));
    if (n <= 0) {
      return false;
    }
    done += static_cast<std::size_t>(n);
  }
  return true;
}

std::uint64_t nowNs() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::string argValue(int argc, char **argv, const std::string &key,
                     std::string fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      return argv[i + 1];
    }
  }
  return fallback;
}

std::string lowerAscii(std::string value) {
  for (char &c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

bool parseBoolValue(const std::string &value, bool fallback) {
  if (value.empty())
    return fallback;
  const std::string lower = lowerAscii(value);
  if (lower == "1" || lower == "true" || lower == "yes" || lower == "on")
    return true;
  if (lower == "0" || lower == "false" || lower == "no" || lower == "off")
    return false;
  return fallback;
}

double argDouble(int argc, char **argv, const std::string &key,
                 double fallback) {
  const std::string value = argValue(argc, argv, key, "");
  if (value.empty())
    return fallback;
  return std::stod(value);
}

int argInt(int argc, char **argv, const std::string &key, int fallback) {
  const std::string value = argValue(argc, argv, key, "");
  if (value.empty())
    return fallback;
  return std::stoi(value);
}

bool argBool(int argc, char **argv, const std::string &key, bool fallback) {
  const std::string value = argValue(argc, argv, key, "");
  return parseBoolValue(value, fallback);
}

std::string envValue(const char *key, const std::string &fallback) {
  const char *value = std::getenv(key);
  if (value == nullptr || value[0] == '\0')
    return fallback;
  return value;
}

bool envBoolValue(const char *key, bool fallback) {
  return parseBoolValue(envValue(key, ""), fallback);
}

std::optional<int> parseIntValue(const std::string &value) {
  if (value.empty())
    return std::nullopt;
  try {
    std::size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    return consumed == value.size() ? std::optional<int>(parsed) : std::nullopt;
  } catch (...) {
    return std::nullopt;
  }
}

enum class ImageTransport {
  File,
  Tcp,
};

std::optional<ImageTransport> parseImageTransport(const std::string &value) {
  const std::string lower = lowerAscii(value);
  if (lower == "file")
    return ImageTransport::File;
  if (lower == "tcp")
    return ImageTransport::Tcp;
  return std::nullopt;
}

const char *imageTransportLabel(ImageTransport transport) {
  return transport == ImageTransport::Tcp ? "tcp" : "file";
}

bool defaultTalosCommandEnabled(ImageTransport transport) {
  return transport == ImageTransport::File;
}

std::string defaultWindowsUdpHost() {
  std::ifstream route("/proc/net/route");
  std::string line;
  std::getline(route, line);
  while (std::getline(route, line)) {
    std::istringstream fields(line);
    std::string iface;
    std::string destination_hex;
    std::string gateway_hex;
    fields >> iface >> destination_hex >> gateway_hex;
    if (destination_hex == "00000000" && !gateway_hex.empty()) {
      const auto gateway =
          static_cast<std::uint32_t>(std::stoul(gateway_hex, nullptr, 16));
      std::ostringstream host;
      host << (gateway & 0xff) << "." << ((gateway >> 8) & 0xff) << "."
           << ((gateway >> 16) & 0xff) << "." << ((gateway >> 24) & 0xff);
      return host.str();
    }
  }

  std::ifstream resolv("/etc/resolv.conf");
  while (std::getline(resolv, line)) {
    std::istringstream fields(line);
    std::string key;
    std::string value;
    fields >> key >> value;
    if (key == "nameserver" && !value.empty()) {
      return value;
    }
  }
  return "127.0.0.1";
}

class UdpSender {
public:
  UdpSender() = default;
  UdpSender(const UdpSender &) = delete;
  UdpSender &operator=(const UdpSender &) = delete;

  ~UdpSender() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  bool open(const std::string &host, int port) {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) {
      return false;
    }

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;

    addrinfo *result = nullptr;
    const std::string service = std::to_string(port);
    if (::getaddrinfo(host.c_str(), service.c_str(), &hints, &result) != 0 ||
        result == nullptr) {
      ::close(fd_);
      fd_ = -1;
      return false;
    }

    std::memcpy(&addr_, result->ai_addr, sizeof(sockaddr_in));
    addr_len_ = static_cast<socklen_t>(result->ai_addrlen);
    ::freeaddrinfo(result);
    return true;
  }

  [[nodiscard]] bool
  send(const aim_sim_bridge::SimulatorCommandFields &command) {
    if (fd_ < 0)
      return false;

    std::ostringstream payload;
    payload << "{\"yaw_deg\":" << command.yaw_deg
            << ",\"pitch_deg\":" << command.pitch_deg
            << ",\"distance_m\":" << command.distance_m
            << ",\"fire_advice\":" << (command.fire_advice ? "true" : "false")
            << "}";
    const std::string text = payload.str();
    const ssize_t sent =
        ::sendto(fd_, text.data(), text.size(), 0,
                 reinterpret_cast<const sockaddr *>(&addr_), addr_len_);
    return sent == static_cast<ssize_t>(text.size());
  }

private:
  int fd_ = -1;
  sockaddr_in addr_{};
  socklen_t addr_len_ = sizeof(addr_);
};

bool consumeImage(ImageTripleBuffer *buffer, ImageMeta *out) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::uint8_t expected = __atomic_load_n(&buffer->state, __ATOMIC_ACQUIRE);
    if ((expected & kFlagNew) == 0) {
      return false;
    }

    const std::uint8_t ready_idx = expected & kIndexMask;
    const std::uint8_t desired = buffer->read_idx;
    if (__atomic_compare_exchange_n(&buffer->state, &expected, desired, true,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      buffer->read_idx = ready_idx;
      *out = buffer->slots[ready_idx];
      return true;
    }
  }
  return false;
}

bool poseToGimbalFeedback(const PoseMeta &pose, GimbalFeedback *out) {
  if (out == nullptr || pose.timestamp_ns == 0)
    return false;

  double w = pose.quaternion[0];
  double x = pose.quaternion[1];
  double y = pose.quaternion[2];
  double z = pose.quaternion[3];
  if (!std::isfinite(w) || !std::isfinite(x) || !std::isfinite(y) ||
      !std::isfinite(z)) {
    return false;
  }

  const double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (!std::isfinite(norm) || norm < 1e-6)
    return false;
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;

  const double sinp = 2.0 * (w * y - z * x);
  const double clamped_sinp = std::clamp(sinp, -1.0, 1.0);
  const double pitch_rad = std::asin(clamped_sinp);
  const double yaw_rad =
      std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  if (!std::isfinite(yaw_rad) || !std::isfinite(pitch_rad))
    return false;

  out->yaw_deg = yaw_rad * kRadToDeg;
  out->pitch_deg = pitch_rad * kRadToDeg;
  out->frame_seq = pose.frame_seq;
  out->timestamp_ns = pose.timestamp_ns;
  return true;
}

double normalizeAngleDeg(double angle) {
  double result = std::fmod(angle + 180.0, 360.0);
  if (result < 0.0)
    result += 360.0;
  return result - 180.0;
}

bool selectLatestGimbalFeedback(const PoseTripleBuffer &buffer,
                                GimbalFeedback *out) {
  GimbalFeedback best{};
  bool found = false;
  for (const PoseMeta &pose : buffer.slots) {
    GimbalFeedback candidate{};
    if (!poseToGimbalFeedback(pose, &candidate))
      continue;
    if (!found || candidate.timestamp_ns > best.timestamp_ns ||
        (candidate.timestamp_ns == best.timestamp_ns &&
         candidate.frame_seq > best.frame_seq)) {
      best = candidate;
      found = true;
    }
  }
  if (found && out != nullptr) {
    *out = best;
  }
  return found;
}

void publishGimbalCommand(GimbalTripleBuffer *buffer,
                          const GimbalCmd &command) {
  buffer->slots[buffer->write_idx] = command;
  const std::uint8_t old = __atomic_exchange_n(
      &buffer->state, buffer->write_idx | kFlagNew, __ATOMIC_ACQ_REL);
  buffer->write_idx = old & kIndexMask;
}

bool checkedImagePayloadBytes(const ImageMeta &meta,
                              std::size_t *payload_bytes) {
  if (payload_bytes == nullptr || meta.width == 0 || meta.height == 0 ||
      meta.width > kImageWidth || meta.height > kImageHeight) {
    return false;
  }
  const std::size_t width = static_cast<std::size_t>(meta.width);
  const std::size_t height = static_cast<std::size_t>(meta.height);
  if (width > std::numeric_limits<std::size_t>::max() / height) {
    return false;
  }
  const std::size_t pixels = width * height;
  if (pixels > std::numeric_limits<std::size_t>::max() / kImageChannels) {
    return false;
  }
  const std::size_t bytes = pixels * kImageChannels;
  if (bytes > kImageSlotStrideBytes) {
    return false;
  }
  *payload_bytes = bytes;
  return true;
}

std::size_t imageSlotOffset(std::uint8_t buffer_id) {
  return static_cast<std::size_t>(buffer_id) * kImageSlotStrideBytes;
}

bool isValidImageMeta(const ImageMeta &meta) {
  std::size_t payload_bytes = 0;
  return meta.seq != 0 && meta.buffer_id < 3 &&
         checkedImagePayloadBytes(meta, &payload_bytes);
}

bool isValidMetaSnapshot(const ShmMetaRegion &meta) {
  return meta.header.magic == kShmMagic && meta.header.version == kShmVersion &&
         meta.header.created_ns != 0 &&
         meta.header.image_width == kImageWidth &&
         meta.header.image_height == kImageHeight;
}

enum class EpochObservation {
  First,
  Same,
  Restart,
};

struct ImageSequenceState {
  EpochObservation observe(std::uint64_t epoch) {
    if (!has_epoch) {
      has_epoch = true;
      producer_epoch = epoch;
      return EpochObservation::First;
    }
    if (producer_epoch == epoch) {
      return EpochObservation::Same;
    }
    producer_epoch = epoch;
    has_last_accepted = false;
    last_accepted_seq = 0;
    return EpochObservation::Restart;
  }

  void commit(std::uint64_t seq) {
    last_accepted_seq = seq;
    has_last_accepted = true;
  }

  bool has_epoch = false;
  std::uint64_t producer_epoch = 0;
  bool has_last_accepted = false;
  std::uint64_t last_accepted_seq = 0;
};

enum class TcpImageSelectionKind {
  Candidate,
  EpochMismatch,
  Duplicate,
  Regression,
  InvalidFrame,
};

const char *tcpPixelFormatLabel(aim_sim_bridge::tcp_image::PixelFormat format) {
  switch (format) {
  case aim_sim_bridge::tcp_image::PixelFormat::Rgb24:
    return "rgb24";
  case aim_sim_bridge::tcp_image::PixelFormat::Rgba32:
    return "rgba32";
  }
  return "unknown";
}

TcpImageSelectionKind
selectTcpImageStrict(const aim_sim_bridge::tcp_image::Frame &frame,
                     const ImageSequenceState &sequence,
                     std::uint64_t metadata_epoch) {
  std::uint32_t expected_payload_bytes = 0;
  if (frame.producer_epoch == 0 || frame.source_sequence == 0 ||
      !aim_sim_bridge::tcp_image::checkedPayloadBytes(
          frame.width, frame.height, frame.format, &expected_payload_bytes) ||
      frame.pixel_bytes.size() !=
          static_cast<std::size_t>(expected_payload_bytes)) {
    return TcpImageSelectionKind::InvalidFrame;
  }
  if (!sequence.has_epoch || sequence.producer_epoch != metadata_epoch ||
      frame.producer_epoch != metadata_epoch) {
    return TcpImageSelectionKind::EpochMismatch;
  }
  if (!sequence.has_last_accepted ||
      frame.source_sequence > sequence.last_accepted_seq) {
    return TcpImageSelectionKind::Candidate;
  }
  if (frame.source_sequence == sequence.last_accepted_seq) {
    return TcpImageSelectionKind::Duplicate;
  }
  return TcpImageSelectionKind::Regression;
}

bool tcpFrameToBgr(const aim_sim_bridge::tcp_image::Frame &frame,
                   cv::Mat *bgr) {
  if (bgr == nullptr)
    return false;
  std::uint32_t expected_payload_bytes = 0;
  if (!aim_sim_bridge::tcp_image::checkedPayloadBytes(
          frame.width, frame.height, frame.format, &expected_payload_bytes) ||
      frame.pixel_bytes.size() !=
          static_cast<std::size_t>(expected_payload_bytes) ||
      frame.width >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      frame.height >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  const int cv_type =
      frame.format == aim_sim_bridge::tcp_image::PixelFormat::Rgb24 ? CV_8UC3
                                                                    : CV_8UC4;
  const int conversion =
      frame.format == aim_sim_bridge::tcp_image::PixelFormat::Rgb24
          ? cv::COLOR_RGB2BGR
          : cv::COLOR_RGBA2BGR;
  try {
    const cv::Mat source(static_cast<int>(frame.height),
                         static_cast<int>(frame.width), cv_type,
                         const_cast<std::uint8_t *>(frame.pixel_bytes.data()));
    cv::cvtColor(source, *bgr, conversion);
  } catch (const cv::Exception &) {
    bgr->release();
    return false;
  }
  return !bgr->empty() && bgr->cols == static_cast<int>(frame.width) &&
         bgr->rows == static_cast<int>(frame.height) && bgr->type() == CV_8UC3;
}

ImageMeta tcpFrameImageMeta(const aim_sim_bridge::tcp_image::Frame &frame) {
  ImageMeta meta{};
  meta.seq = frame.source_sequence;
  meta.timestamp_ns = frame.capture_timestamp_ns;
  meta.width = frame.width;
  meta.height = frame.height;
  meta.buffer_id = std::numeric_limits<std::uint8_t>::max();
  meta.format = static_cast<std::uint8_t>(frame.format);
  return meta;
}

struct TcpBridgeMetrics {
  void recordTaken() { ++taken_frames; }

  void recordSelected(const aim_sim_bridge::tcp_image::Frame &frame) {
    ++selected_frames;
    last_format = frame.format;
    last_width = frame.width;
    last_height = frame.height;
    last_epoch = frame.producer_epoch;
    last_sequence = frame.source_sequence;
    last_capture_timestamp_ns = frame.capture_timestamp_ns;
    const std::uint64_t observed_ns = nowNs();
    if (frame.capture_timestamp_ns != 0 &&
        observed_ns >= frame.capture_timestamp_ns) {
      latest_selected_source_age_ms =
          static_cast<double>(observed_ns - frame.capture_timestamp_ns) * 1e-6;
      selected_source_age_available = true;
      ++selected_source_age_samples;
    } else {
      latest_selected_source_age_ms = std::numeric_limits<double>::quiet_NaN();
      selected_source_age_available = false;
      ++invalid_selected_source_age_samples;
    }
  }

  std::uint64_t taken_frames = 0;
  std::uint64_t selected_frames = 0;
  std::uint64_t epoch_mismatch_rejects = 0;
  std::uint64_t duplicate_rejects = 0;
  std::uint64_t regression_rejects = 0;
  std::uint64_t invalid_frame_rejects = 0;
  std::uint64_t color_conversion_failures = 0;
  aim_sim_bridge::tcp_image::PixelFormat last_format =
      aim_sim_bridge::tcp_image::PixelFormat::Rgb24;
  std::uint32_t last_width = 0;
  std::uint32_t last_height = 0;
  std::uint64_t last_epoch = 0;
  std::uint64_t last_sequence = 0;
  std::uint64_t last_capture_timestamp_ns = 0;
  std::uint64_t selected_source_age_samples = 0;
  std::uint64_t invalid_selected_source_age_samples = 0;
  double latest_selected_source_age_ms =
      std::numeric_limits<double>::quiet_NaN();
  bool selected_source_age_available = false;
};

struct ImageTransportTelemetry {
  ImageTransport transport = ImageTransport::File;
  std::string tcp_host;
  std::uint16_t tcp_port = 0;
  bool tcp_receiver_running = false;
  aim_sim_bridge::tcp_image::ReceiverCounters tcp_receiver;
  TcpBridgeMetrics tcp_bridge;
};

enum class ImageSelectionKind {
  Candidate,
  NoValidImage,
  Duplicate,
  Regression,
};

struct ImageSelection {
  ImageSelectionKind kind = ImageSelectionKind::NoValidImage;
  ImageMeta image{};
};

ImageSelection selectLatestImageStrict(const ShmMetaRegion &meta,
                                       const ImageSequenceState &sequence) {
  bool found = false;
  ImageMeta newest{};
  for (const ImageMeta &candidate : meta.image.slots) {
    if (!isValidImageMeta(candidate)) {
      continue;
    }
    if (!found || candidate.seq > newest.seq) {
      newest = candidate;
      found = true;
    }
  }

  if (!found) {
    return {};
  }
  if (!sequence.has_last_accepted || newest.seq > sequence.last_accepted_seq) {
    return {ImageSelectionKind::Candidate, newest};
  }
  if (newest.seq == sequence.last_accepted_seq) {
    return {ImageSelectionKind::Duplicate, newest};
  }
  return {ImageSelectionKind::Regression, newest};
}

bool imageReadStillConsistent(const ShmMetaRegion &meta,
                              std::uint64_t producer_epoch,
                              const ImageMeta &image) {
  if (!isValidMetaSnapshot(meta) || meta.header.created_ns != producer_epoch) {
    return false;
  }
  for (const ImageMeta &candidate : meta.image.slots) {
    if (isValidImageMeta(candidate) && candidate.seq == image.seq &&
        candidate.timestamp_ns == image.timestamp_ns &&
        candidate.buffer_id == image.buffer_id &&
        candidate.width == image.width && candidate.height == image.height) {
      return true;
    }
  }
  return false;
}

struct IngressMetrics {
  void recordAccepted(std::chrono::steady_clock::time_point now) {
    ++unique_accepted_frames;
    rolling_accepts.push_back(now);
    prune(now);
  }

  double rollingHz(std::chrono::steady_clock::time_point now) {
    prune(now);
    return static_cast<double>(rolling_accepts.size());
  }

  void prune(std::chrono::steady_clock::time_point now) {
    const auto cutoff = now - std::chrono::seconds(1);
    while (!rolling_accepts.empty() && rolling_accepts.front() <= cutoff) {
      rolling_accepts.pop_front();
    }
  }

  std::uint64_t unique_accepted_frames = 0;
  std::uint64_t sequence_gap_frames = 0;
  std::uint64_t duplicate_rejects = 0;
  std::uint64_t regression_rejects = 0;
  std::uint64_t inconsistent_image_reads = 0;
  std::uint64_t producer_epoch_changes = 0;
  std::uint64_t image_read_failures = 0;
  std::deque<std::chrono::steady_clock::time_point> rolling_accepts;
};

enum class CompletionObservation {
  NotCompleted,
  Accepted,
  InvalidIdentity,
  StaleEpoch,
  Duplicate,
  Regression,
};

struct CompletedVisionSample {
  std::chrono::steady_clock::time_point observed_at{};
  double source_to_completion_age_ms = std::numeric_limits<double>::quiet_NaN();
};

// A completion is counted only once per (producer epoch, image sequence), and
// only when the pipeline marks that it crossed its post-track/aim boundary.
// This makes repeated publication of one AimCommand rate-neutral.
struct CompletedVisionMetrics {
  void observeProducerEpoch(std::uint64_t epoch) {
    if (epoch == 0)
      return;
    if (!has_active_producer_epoch) {
      has_active_producer_epoch = true;
      active_producer_epoch = epoch;
      return;
    }
    if (active_producer_epoch == epoch)
      return;

    active_producer_epoch = epoch;
    has_sequence_in_active_epoch = false;
    last_sequence_in_active_epoch = 0;
    ++producer_epoch_resets;
  }

  CompletionObservation
  record(const aim_sim_bridge::AimCommand &command,
         std::chrono::steady_clock::time_point observed_at) {
    if (!command.completed_vision_result) {
      return CompletionObservation::NotCompleted;
    }
    ++completion_events_observed;

    if (command.source_producer_epoch == 0 || command.source_image_seq == 0) {
      ++invalid_identity_rejects;
      return CompletionObservation::InvalidIdentity;
    }
    if (!has_active_producer_epoch ||
        command.source_producer_epoch != active_producer_epoch) {
      ++stale_epoch_rejects;
      return CompletionObservation::StaleEpoch;
    }
    if (has_sequence_in_active_epoch) {
      if (command.source_image_seq == last_sequence_in_active_epoch) {
        ++duplicate_sequence_rejects;
        return CompletionObservation::Duplicate;
      }
      if (command.source_image_seq < last_sequence_in_active_epoch) {
        ++regression_sequence_rejects;
        return CompletionObservation::Regression;
      }
      if (command.source_image_seq > last_sequence_in_active_epoch + 1) {
        completed_sequence_gap_frames +=
            command.source_image_seq - last_sequence_in_active_epoch - 1;
      }
    }

    has_sequence_in_active_epoch = true;
    last_sequence_in_active_epoch = command.source_image_seq;
    has_last_completed_result = true;
    last_completed_epoch = command.source_producer_epoch;
    last_completed_seq = command.source_image_seq;
    last_source_capture_timestamp_ns = command.source_capture_timestamp_ns;
    last_completion_timestamp_ns = command.vision_completion_timestamp_ns;
    ++processed_result_count;

    double age_ms = std::numeric_limits<double>::quiet_NaN();
    if (command.source_capture_timestamp_ns != 0 &&
        command.vision_completion_timestamp_ns >=
            command.source_capture_timestamp_ns) {
      age_ms = static_cast<double>(command.vision_completion_timestamp_ns -
                                   command.source_capture_timestamp_ns) *
               1e-6;
      latest_source_to_completion_age_ms = age_ms;
      lifetime_age_sum_ms += age_ms;
      ++lifetime_age_samples;
    } else {
      latest_source_to_completion_age_ms =
          std::numeric_limits<double>::quiet_NaN();
      ++invalid_age_samples;
    }

    rolling_completions.push_back({observed_at, age_ms});
    prune(observed_at);
    return CompletionObservation::Accepted;
  }

  double rollingHz(std::chrono::steady_clock::time_point now) {
    prune(now);
    return static_cast<double>(rolling_completions.size());
  }

  double rollingMeanAgeMs() const {
    double sum = 0.0;
    std::size_t count = 0;
    for (const auto &sample : rolling_completions) {
      if (std::isfinite(sample.source_to_completion_age_ms)) {
        sum += sample.source_to_completion_age_ms;
        ++count;
      }
    }
    return count > 0 ? sum / static_cast<double>(count)
                     : std::numeric_limits<double>::quiet_NaN();
  }

  double rollingP95AgeMs() const {
    std::vector<double> ages;
    ages.reserve(rolling_completions.size());
    for (const auto &sample : rolling_completions) {
      if (std::isfinite(sample.source_to_completion_age_ms)) {
        ages.push_back(sample.source_to_completion_age_ms);
      }
    }
    if (ages.empty())
      return std::numeric_limits<double>::quiet_NaN();
    std::sort(ages.begin(), ages.end());
    const std::size_t nearest_rank = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(ages.size())));
    return ages[std::max<std::size_t>(1, nearest_rank) - 1];
  }

  double lifetimeMeanAgeMs() const {
    return lifetime_age_samples > 0
               ? lifetime_age_sum_ms / static_cast<double>(lifetime_age_samples)
               : std::numeric_limits<double>::quiet_NaN();
  }

  void prune(std::chrono::steady_clock::time_point now) {
    const auto cutoff = now - std::chrono::seconds(1);
    while (!rolling_completions.empty() &&
           rolling_completions.front().observed_at <= cutoff) {
      rolling_completions.pop_front();
    }
  }

  bool has_active_producer_epoch = false;
  std::uint64_t active_producer_epoch = 0;
  bool has_sequence_in_active_epoch = false;
  std::uint64_t last_sequence_in_active_epoch = 0;
  bool has_last_completed_result = false;
  std::uint64_t last_completed_epoch = 0;
  std::uint64_t last_completed_seq = 0;
  std::uint64_t last_source_capture_timestamp_ns = 0;
  std::uint64_t last_completion_timestamp_ns = 0;
  std::uint64_t completion_events_observed = 0;
  std::uint64_t processed_result_count = 0;
  std::uint64_t completed_sequence_gap_frames = 0;
  std::uint64_t duplicate_sequence_rejects = 0;
  std::uint64_t regression_sequence_rejects = 0;
  std::uint64_t stale_epoch_rejects = 0;
  std::uint64_t invalid_identity_rejects = 0;
  std::uint64_t producer_epoch_resets = 0;
  std::uint64_t invalid_age_samples = 0;
  double latest_source_to_completion_age_ms =
      std::numeric_limits<double>::quiet_NaN();
  double lifetime_age_sum_ms = 0.0;
  std::uint64_t lifetime_age_samples = 0;
  std::deque<CompletedVisionSample> rolling_completions;
};

bool hasArg(int argc, char **argv, const std::string &key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

int runSequenceSelectionSelfTest() {
  auto fail = [](const char *message) {
    std::cerr << "sequence self-test failed: " << message << '\n';
    return 1;
  };
  auto image = [](std::uint64_t seq, std::uint8_t buffer_id) {
    ImageMeta out{};
    out.seq = seq;
    out.timestamp_ns = seq * 1000;
    out.width = kImageWidth;
    out.height = kImageHeight;
    out.buffer_id = buffer_id;
    return out;
  };

  ShmMetaRegion meta{};
  meta.header.magic = kShmMagic;
  meta.header.version = kShmVersion;
  meta.header.created_ns = 100;
  meta.header.image_width = kImageWidth;
  meta.header.image_height = kImageHeight;
  meta.image.slots[0] = image(6, 0);
  meta.image.slots[1] = image(7, 1);

  ImageSequenceState sequence;
  if (sequence.observe(100) != EpochObservation::First)
    return fail("first epoch");
  auto selected = selectLatestImageStrict(meta, sequence);
  if (selected.kind != ImageSelectionKind::Candidate ||
      selected.image.seq != 7) {
    return fail("select newest initial sequence");
  }
  sequence.commit(selected.image.seq);
  if (selectLatestImageStrict(meta, sequence).kind !=
      ImageSelectionKind::Duplicate) {
    return fail("do not fall back from duplicate latest slot");
  }
  meta.image.slots[0] = image(4, 0);
  meta.image.slots[1] = image(5, 1);
  if (selectLatestImageStrict(meta, sequence).kind !=
      ImageSelectionKind::Regression) {
    return fail("reject regressed snapshot");
  }
  meta.image.slots[0] = image(8, 2);
  selected = selectLatestImageStrict(meta, sequence);
  if (selected.kind != ImageSelectionKind::Candidate ||
      !imageReadStillConsistent(meta, 100, selected.image)) {
    return fail("consistent increasing candidate");
  }
  meta.image.slots[0].buffer_id = 1;
  if (imageReadStillConsistent(meta, 100, selected.image)) {
    return fail("detect overwritten image metadata");
  }
  if (sequence.last_accepted_seq != 7) {
    return fail("inconsistent read advanced sequence");
  }
  if (sequence.observe(200) != EpochObservation::Restart ||
      sequence.has_last_accepted) {
    return fail("producer restart resets sequence");
  }
  meta.header.created_ns = 200;
  meta.image.slots[0] = image(2, 0);
  meta.image.slots[1] = {};
  selected = selectLatestImageStrict(meta, sequence);
  if (selected.kind != ImageSelectionKind::Candidate ||
      selected.image.seq != 2) {
    return fail("accept lower sequence in new epoch");
  }

  ImageMeta small = image(3, 2);
  small.width = 640;
  small.height = 360;
  std::size_t payload_bytes = 0;
  if (!isValidImageMeta(small) ||
      !checkedImagePayloadBytes(small, &payload_bytes) ||
      payload_bytes != 640u * 360u * kImageChannels ||
      imageSlotOffset(small.buffer_id) != 2u * kImageSlotStrideBytes) {
    return fail("smaller payload keeps fixed maximum slot stride");
  }
  ImageMeta invalid = small;
  invalid.width = 0;
  if (isValidImageMeta(invalid))
    return fail("reject zero width");
  invalid = small;
  invalid.height = kImageHeight + 1;
  if (isValidImageMeta(invalid))
    return fail("reject height above maximum");
  invalid = small;
  invalid.width = std::numeric_limits<std::uint32_t>::max();
  invalid.height = std::numeric_limits<std::uint32_t>::max();
  if (checkedImagePayloadBytes(invalid, &payload_bytes)) {
    return fail("reject unbounded or overflowing dimensions");
  }

  auto completedCommand = [](std::uint64_t epoch, std::uint64_t seq,
                             std::uint64_t capture_ns,
                             std::uint64_t completion_ns) {
    aim_sim_bridge::AimCommand command;
    command.completed_vision_result = true;
    command.source_producer_epoch = epoch;
    command.source_image_seq = seq;
    command.source_capture_timestamp_ns = capture_ns;
    command.vision_completion_timestamp_ns = completion_ns;
    return command;
  };
  const auto completion_test_start =
      std::chrono::steady_clock::time_point(std::chrono::seconds(10));
  CompletedVisionMetrics completion_metrics;
  completion_metrics.observeProducerEpoch(100);
  const aim_sim_bridge::AimCommand first_completion =
      completedCommand(100, 10, 1'000'000'000, 1'005'000'000);
  if (completion_metrics.record(first_completion, completion_test_start) !=
      CompletionObservation::Accepted) {
    return fail("accept first completed vision result");
  }
  for (int resend = 0; resend < 250; ++resend) {
    if (completion_metrics.record(first_completion,
                                  completion_test_start +
                                      std::chrono::microseconds(resend + 1)) !=
        CompletionObservation::Duplicate) {
      return fail("reject a resent completed result");
    }
  }
  if (completion_metrics.processed_result_count != 1 ||
      completion_metrics.duplicate_sequence_rejects != 250) {
    return fail("250 Hz resend does not inflate completed count");
  }
  if (completion_metrics.record(
          completedCommand(100, 9, 1'001'000'000, 1'006'000'000),
          completion_test_start + std::chrono::milliseconds(2)) !=
      CompletionObservation::Regression) {
    return fail("reject regressed completed result");
  }
  if (completion_metrics.record(
          completedCommand(100, 13, 1'010'000'000, 1'017'000'000),
          completion_test_start + std::chrono::milliseconds(3)) !=
      CompletionObservation::Accepted) {
    return fail("accept increasing completed result");
  }
  if (completion_metrics.completed_sequence_gap_frames != 2) {
    return fail("account completed sequence gaps");
  }
  if (completion_metrics.record(
          completedCommand(99, 14, 1'020'000'000, 1'025'000'000),
          completion_test_start + std::chrono::milliseconds(4)) !=
      CompletionObservation::StaleEpoch) {
    return fail("reject stale producer epoch completion");
  }
  if (completion_metrics.record(
          completedCommand(100, 0, 1'020'000'000, 1'025'000'000),
          completion_test_start + std::chrono::milliseconds(5)) !=
      CompletionObservation::InvalidIdentity) {
    return fail("reject missing completed-result identity");
  }
  aim_sim_bridge::AimCommand not_completed;
  if (completion_metrics.record(not_completed,
                                completion_test_start +
                                    std::chrono::milliseconds(6)) !=
      CompletionObservation::NotCompleted) {
    return fail("do not count incomplete pipeline calls");
  }
  completion_metrics.observeProducerEpoch(200);
  if (completion_metrics.record(
          completedCommand(200, 2, 2'000'000'000, 2'004'000'000),
          completion_test_start + std::chrono::milliseconds(7)) !=
      CompletionObservation::Accepted) {
    return fail("accept lower sequence after producer restart");
  }
  if (completion_metrics.processed_result_count != 3 ||
      completion_metrics.completion_events_observed != 256 ||
      completion_metrics.regression_sequence_rejects != 1 ||
      completion_metrics.stale_epoch_rejects != 1 ||
      completion_metrics.invalid_identity_rejects != 1 ||
      completion_metrics.producer_epoch_resets != 1 ||
      completion_metrics.last_completed_epoch != 200 ||
      completion_metrics.last_completed_seq != 2 ||
      completion_metrics.rollingHz(completion_test_start +
                                   std::chrono::milliseconds(500)) != 3.0 ||
      std::abs(completion_metrics.latest_source_to_completion_age_ms - 4.0) >
          1e-9 ||
      std::abs(completion_metrics.rollingP95AgeMs() - 7.0) > 1e-9) {
    return fail("completed vision metric totals and age");
  }
  std::cout << "sequence self-test passed\n";
  return 0;
}

int runImageTransportIntegrationSelfTest() {
  const auto fail = [](const char *message) {
    std::cerr << "image transport integration self-test failed: " << message
              << '\n';
    return 1;
  };

  if (parseImageTransport("file") != ImageTransport::File ||
      parseImageTransport("TCP") != ImageTransport::Tcp ||
      parseImageTransport("udp").has_value() ||
      !defaultTalosCommandEnabled(ImageTransport::File) ||
      defaultTalosCommandEnabled(ImageTransport::Tcp) ||
      !parseBoolValue("true", false) || parseBoolValue("off", true) ||
      parseIntValue("5602") != std::optional<int>(5602) ||
      parseIntValue("bad").has_value() || parseIntValue("5602x").has_value()) {
    return fail("transport and Talos-default parsing");
  }

  aim_sim_bridge::tcp_image::FrameHeader header;
  header.format = aim_sim_bridge::tcp_image::PixelFormat::Rgba32;
  header.width = 1;
  header.height = 1;
  header.payload_bytes = 4;
  header.producer_epoch = 44;
  header.source_sequence = 7;
  header.capture_timestamp_ns = 123456789;
  aim_sim_bridge::tcp_image::WireHeader wire{};
  if (aim_sim_bridge::tcp_image::encodeHeader(header, &wire) !=
      aim_sim_bridge::tcp_image::HeaderStatus::Ok) {
    return fail("encode v1 codec header");
  }
  const auto decoded =
      aim_sim_bridge::tcp_image::decodeHeader(wire.data(), wire.size());
  if (!decoded.ok() || decoded.header.format != header.format ||
      decoded.header.producer_epoch != header.producer_epoch ||
      decoded.header.source_sequence != header.source_sequence ||
      decoded.header.capture_timestamp_ns != header.capture_timestamp_ns) {
    return fail("decode v1 codec identity and format");
  }

  auto makeFrame = [](aim_sim_bridge::tcp_image::PixelFormat format,
                      std::uint64_t epoch, std::uint64_t sequence,
                      std::vector<std::uint8_t> pixels, std::uint32_t width,
                      std::uint32_t height) {
    aim_sim_bridge::tcp_image::Frame frame;
    frame.format = format;
    frame.width = width;
    frame.height = height;
    frame.producer_epoch = epoch;
    frame.source_sequence = sequence;
    frame.capture_timestamp_ns = 123456789;
    frame.pixel_bytes = std::move(pixels);
    return frame;
  };

  ImageSequenceState sequence;
  if (sequence.observe(44) != EpochObservation::First) {
    return fail("initialize TCP selection epoch");
  }
  const auto rgb = makeFrame(aim_sim_bridge::tcp_image::PixelFormat::Rgb24, 44,
                             7, {255, 0, 0, 0, 255, 0}, 2, 1);
  if (selectTcpImageStrict(rgb, sequence, 44) !=
      TcpImageSelectionKind::Candidate) {
    return fail("select increasing TCP identity");
  }
  sequence.commit(rgb.source_sequence);
  if (selectTcpImageStrict(rgb, sequence, 44) !=
          TcpImageSelectionKind::Duplicate ||
      selectTcpImageStrict(
          makeFrame(aim_sim_bridge::tcp_image::PixelFormat::Rgb24, 44, 6,
                    {1, 2, 3}, 1, 1),
          sequence, 44) != TcpImageSelectionKind::Regression ||
      selectTcpImageStrict(
          makeFrame(aim_sim_bridge::tcp_image::PixelFormat::Rgb24, 45, 8,
                    {1, 2, 3}, 1, 1),
          sequence, 44) != TcpImageSelectionKind::EpochMismatch ||
      selectTcpImageStrict(
          makeFrame(aim_sim_bridge::tcp_image::PixelFormat::Rgb24, 44, 8,
                    {1, 2}, 1, 1),
          sequence, 44) != TcpImageSelectionKind::InvalidFrame) {
    return fail("reject duplicate/regression/epoch/payload errors");
  }

  cv::Mat bgr;
  if (!tcpFrameToBgr(rgb, &bgr) || bgr.rows != 1 || bgr.cols != 2 ||
      bgr.at<cv::Vec3b>(0, 0) != cv::Vec3b(0, 0, 255) ||
      bgr.at<cv::Vec3b>(0, 1) != cv::Vec3b(0, 255, 0)) {
    return fail("RGB24 to BGR conversion");
  }
  const auto rgba = makeFrame(aim_sim_bridge::tcp_image::PixelFormat::Rgba32,
                              44, 8, {10, 20, 30, 40}, 1, 1);
  if (!tcpFrameToBgr(rgba, &bgr) ||
      bgr.at<cv::Vec3b>(0, 0) != cv::Vec3b(30, 20, 10)) {
    return fail("RGBA32 to BGR conversion and alpha discard");
  }
  const ImageMeta synthetic = tcpFrameImageMeta(rgba);
  if (synthetic.seq != 8 || synthetic.timestamp_ns != 123456789 ||
      synthetic.width != 1 || synthetic.height != 1 ||
      synthetic.buffer_id != std::numeric_limits<std::uint8_t>::max() ||
      synthetic.format != 2) {
    return fail("TCP identity to pipeline metadata");
  }

  std::cout << "image transport integration self-test passed\n";
  return 0;
}

[[nodiscard]] bool publishGimbalCommandFile(int meta_fd,
                                            const ShmMetaRegion &snapshot,
                                            const GimbalCmd &command) {
  std::uint8_t write_idx = snapshot.gimbal_cmd.write_idx;
  if (write_idx >= 3) {
    write_idx = 0;
  }

  const std::uint64_t base = offsetof(ShmMetaRegion, gimbal_cmd);
  const std::uint64_t slot_offset =
      base + offsetof(GimbalTripleBuffer, slots) +
      static_cast<std::uint64_t>(write_idx) * sizeof(GimbalCmd);
  if (!writeFullyAt(meta_fd, &command, sizeof(command), slot_offset)) {
    return false;
  }

  const std::uint8_t state = write_idx | kFlagNew;
  if (!writeFullyAt(meta_fd, &state, sizeof(state),
                    base + offsetof(GimbalTripleBuffer, state))) {
    return false;
  }

  std::uint8_t next_write_idx = snapshot.gimbal_cmd.state & kIndexMask;
  if (next_write_idx >= 3) {
    next_write_idx = 0;
  }
  return writeFullyAt(meta_fd, &next_write_idx, sizeof(next_write_idx),
                      base + offsetof(GimbalTripleBuffer, write_idx));
}

GimbalCmd toTalosCommand(const aim_sim_bridge::AimCommand &command,
                         const aim_sim_bridge::AimBridgeConfig &config) {
  GimbalCmd out{};
  out.timestamp_ns = nowNs();
  if (!command.has_target) {
    out.distance_m = -1.0f;
    return out;
  }

  // Daedalus local gimbal yaw is opposite to the vivsionn yaw convention.
  out.yaw_deg = static_cast<float>(-command.yaw_deg);
  // Talos decodes local_joint = -encoded - 90deg. Encode the same optical
  // target as UDP: local_joint = -mount_pitch - optical_pitch.
  out.pitch_deg =
      static_cast<float>(command.pitch_deg - config.sim_pitch_neutral_deg);
  out.distance_m =
      static_cast<float>(command.distance_m > 0.0 ? command.distance_m : 1.0);
  out.fire_advice =
      static_cast<std::uint8_t>(config.enable_fire && command.fire_advice);
  return out;
}

std::optional<aim_sim_bridge::control::SteadyTimePoint>
sourceSteadyTimeFromWall(
    std::uint64_t capture_timestamp_ns, std::uint64_t completion_timestamp_ns,
    std::uint64_t observed_wall_ns,
    aim_sim_bridge::control::SteadyTimePoint observed_steady) {
  if (capture_timestamp_ns == 0 || completion_timestamp_ns == 0 ||
      completion_timestamp_ns < capture_timestamp_ns ||
      observed_wall_ns < completion_timestamp_ns) {
    return std::nullopt;
  }

  const std::uint64_t age_ns = observed_wall_ns - capture_timestamp_ns;
  if (age_ns >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  return observed_steady -
         std::chrono::nanoseconds(static_cast<std::int64_t>(age_ns));
}

void preserveVisionProvenanceForSafeOutput(
    aim_sim_bridge::control::CommandSelection<aim_sim_bridge::AimCommand>
        &selection) {
  if ((selection.disposition !=
           aim_sim_bridge::control::CommandDisposition::NoTarget &&
       selection.disposition !=
           aim_sim_bridge::control::CommandDisposition::Stale) ||
      !selection.source_payload) {
    return;
  }

  const auto &source = *selection.source_payload;
  selection.output.completed_vision_result = source.completed_vision_result;
  selection.output.source_producer_epoch = source.source_producer_epoch;
  selection.output.source_image_seq = source.source_image_seq;
  selection.output.source_capture_timestamp_ns =
      source.source_capture_timestamp_ns;
  selection.output.vision_completion_timestamp_ns =
      source.vision_completion_timestamp_ns;
}

enum class WallCommandSubmitStatus : std::uint8_t {
  Accepted,
  IgnoredNotCompleted,
  IgnoredDisabled,
  Duplicate,
  RejectedOlder,
  InvalidIdentityOrTimestamp,
};

const char *transportPublishStatusLabel(
    aim_sim_bridge::control::TransportPublishStatus status) {
  switch (status) {
  case aim_sim_bridge::control::TransportPublishStatus::NotAttempted:
    return "not_attempted";
  case aim_sim_bridge::control::TransportPublishStatus::Succeeded:
    return "succeeded";
  case aim_sim_bridge::control::TransportPublishStatus::Failed:
    return "failed";
  }
  return "unknown";
}

struct WallCommandPublisherSnapshot {
  bool has_output = false;
  bool following_enabled = false;
  bool talos_command_enabled = true;
  std::uint64_t tick_index = 0;
  std::uint64_t published_timestamp_ns = 0;
  aim_sim_bridge::control::CommandDisposition disposition =
      aim_sim_bridge::control::CommandDisposition::Unavailable;
  aim_sim_bridge::control::SourceIdentity source;
  bool source_age_available = false;
  double source_age_ms = 0.0;
  aim_sim_bridge::AimCommand aim_command;
  aim_sim_bridge::SimulatorCommandFields simulator_command;
  GimbalCmd talos_command{};
  aim_sim_bridge::control::TransportPublishStatus udp_publish =
      aim_sim_bridge::control::TransportPublishStatus::NotAttempted;
  aim_sim_bridge::control::TransportPublishStatus talos_publish =
      aim_sim_bridge::control::TransportPublishStatus::NotAttempted;
  aim_sim_bridge::control::TickMetricsSnapshot metrics;

  std::uint64_t submit_accepted_count = 0;
  std::uint64_t submit_not_completed_count = 0;
  std::uint64_t submit_disabled_count = 0;
  std::uint64_t submit_duplicate_count = 0;
  std::uint64_t submit_rejected_older_count = 0;
  std::uint64_t submit_invalid_count = 0;
  std::uint64_t clear_count = 0;
};

class WallCommandPublisher {
public:
  WallCommandPublisher(std::filesystem::path meta_path,
                       aim_sim_bridge::AimBridgeConfig config, bool enable_udp,
                       bool enable_talos_command, std::string udp_host,
                       int udp_port)
      : meta_path_(std::move(meta_path)), config_(std::move(config)),
        enable_udp_(enable_udp), enable_talos_command_(enable_talos_command),
        udp_host_(std::move(udp_host)), udp_port_(udp_port),
        latest_command_(aim_sim_bridge::AimCommand{}),
        worker_([this]() { run(); }) {}

  WallCommandPublisher(const WallCommandPublisher &) = delete;
  WallCommandPublisher &operator=(const WallCommandPublisher &) = delete;

  ~WallCommandPublisher() {
    stop_requested_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void setFollowing(bool enabled) {
    std::lock_guard<std::mutex> lock(publication_gate_);
    following_enabled_.store(enabled, std::memory_order_release);
    latest_command_.clear();
    clear_count_.fetch_add(1, std::memory_order_relaxed);
  }

  void clearForProducerEpochRestart() {
    std::lock_guard<std::mutex> lock(publication_gate_);
    latest_command_.clear();
    clear_count_.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] WallCommandSubmitStatus
  submitCompleted(const aim_sim_bridge::AimCommand &command) {
    if (!command.completed_vision_result) {
      submit_not_completed_count_.fetch_add(1, std::memory_order_relaxed);
      return WallCommandSubmitStatus::IgnoredNotCompleted;
    }
    if (command.source_producer_epoch == 0 || command.source_image_seq == 0) {
      invalidateLatest();
      return WallCommandSubmitStatus::InvalidIdentityOrTimestamp;
    }

    const auto observed_steady = aim_sim_bridge::control::SteadyClock::now();
    const auto source_time = sourceSteadyTimeFromWall(
        command.source_capture_timestamp_ns,
        command.vision_completion_timestamp_ns, nowNs(), observed_steady);
    if (!source_time) {
      invalidateLatest();
      return WallCommandSubmitStatus::InvalidIdentityOrTimestamp;
    }

    std::lock_guard<std::mutex> lock(publication_gate_);
    if (!following_enabled_.load(std::memory_order_acquire)) {
      submit_disabled_count_.fetch_add(1, std::memory_order_relaxed);
      return WallCommandSubmitStatus::IgnoredDisabled;
    }

    const aim_sim_bridge::control::TimestampedCommand<
        aim_sim_bridge::AimCommand>
        timed{
            {command.source_producer_epoch, command.source_image_seq},
            *source_time,
            command.has_target,
            command,
        };
    switch (latest_command_.publish(timed)) {
    case aim_sim_bridge::control::CommandPublishStatus::Accepted:
      submit_accepted_count_.fetch_add(1, std::memory_order_relaxed);
      return WallCommandSubmitStatus::Accepted;
    case aim_sim_bridge::control::CommandPublishStatus::DuplicateSource:
      submit_duplicate_count_.fetch_add(1, std::memory_order_relaxed);
      return WallCommandSubmitStatus::Duplicate;
    case aim_sim_bridge::control::CommandPublishStatus::RejectedOlderSource:
      submit_rejected_older_count_.fetch_add(1, std::memory_order_relaxed);
      return WallCommandSubmitStatus::RejectedOlder;
    }
    submit_rejected_older_count_.fetch_add(1, std::memory_order_relaxed);
    return WallCommandSubmitStatus::RejectedOlder;
  }

  [[nodiscard]] WallCommandPublisherSnapshot snapshot() const {
    WallCommandPublisherSnapshot result;
    {
      std::lock_guard<std::mutex> lock(output_mutex_);
      result = last_output_;
    }
    result.following_enabled =
        following_enabled_.load(std::memory_order_acquire);
    result.talos_command_enabled = enable_talos_command_;
    result.metrics = scheduler_.metrics().snapshot();
    result.submit_accepted_count =
        submit_accepted_count_.load(std::memory_order_relaxed);
    result.submit_not_completed_count =
        submit_not_completed_count_.load(std::memory_order_relaxed);
    result.submit_disabled_count =
        submit_disabled_count_.load(std::memory_order_relaxed);
    result.submit_duplicate_count =
        submit_duplicate_count_.load(std::memory_order_relaxed);
    result.submit_rejected_older_count =
        submit_rejected_older_count_.load(std::memory_order_relaxed);
    result.submit_invalid_count =
        submit_invalid_count_.load(std::memory_order_relaxed);
    result.clear_count = clear_count_.load(std::memory_order_relaxed);
    return result;
  }

private:
  void invalidateLatest() {
    std::lock_guard<std::mutex> lock(publication_gate_);
    latest_command_.clear();
    submit_invalid_count_.fetch_add(1, std::memory_order_relaxed);
    clear_count_.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] bool publishTalos(FileHandle &meta_file,
                                  const GimbalCmd &command) {
    if (meta_file.fd() < 0 && !meta_file.openExisting(meta_path_, O_RDWR)) {
      return false;
    }

    ShmMetaRegion snapshot{};
    if (!readFullyAt(meta_file.fd(), &snapshot, sizeof(snapshot), 0) ||
        !isValidMetaSnapshot(snapshot)) {
      meta_file.close();
      return false;
    }
    if (!publishGimbalCommandFile(meta_file.fd(), snapshot, command)) {
      meta_file.close();
      return false;
    }
    return true;
  }

  aim_sim_bridge::control::TickWorkResult
  publishTick(const aim_sim_bridge::control::TickContext &context,
              FileHandle &talos_meta_file, UdpSender &udp_sender) {
    aim_sim_bridge::control::CommandSelection<aim_sim_bridge::AimCommand>
        selection;
    aim_sim_bridge::SimulatorCommandFields simulator_command;
    GimbalCmd talos_command{};
    aim_sim_bridge::control::TransportPublishStatus udp_status =
        aim_sim_bridge::control::TransportPublishStatus::NotAttempted;
    aim_sim_bridge::control::TransportPublishStatus talos_status =
        aim_sim_bridge::control::TransportPublishStatus::NotAttempted;

    {
      std::lock_guard<std::mutex> gate(publication_gate_);
      if (following_enabled_.load(std::memory_order_acquire)) {
        selection = latest_command_.select(context.started_at);
      } else {
        selection.output = aim_sim_bridge::AimCommand{};
      }

      // Keep the transport command canonical-safe while retaining all B-I3
      // provenance fields for completed no-target and expired publications.
      preserveVisionProvenanceForSafeOutput(selection);

      simulator_command =
          aim_sim_bridge::toSimulatorCommand(selection.output, config_);
      talos_command = toTalosCommand(selection.output, config_);
      if (enable_talos_command_) {
        talos_status =
            publishTalos(talos_meta_file, talos_command)
                ? aim_sim_bridge::control::TransportPublishStatus::Succeeded
                : aim_sim_bridge::control::TransportPublishStatus::Failed;
      }
      if (enable_udp_) {
        udp_status =
            udp_sender.send(simulator_command)
                ? aim_sim_bridge::control::TransportPublishStatus::Succeeded
                : aim_sim_bridge::control::TransportPublishStatus::Failed;
      }

      WallCommandPublisherSnapshot output;
      output.has_output = true;
      output.following_enabled =
          following_enabled_.load(std::memory_order_relaxed);
      output.talos_command_enabled = enable_talos_command_;
      output.tick_index = context.tick_index;
      output.published_timestamp_ns = nowNs();
      output.disposition = selection.disposition;
      output.source = selection.source;
      output.source_age_available = selection.source_age.has_value();
      output.source_age_ms =
          selection.source_age
              ? std::chrono::duration<double, std::milli>(*selection.source_age)
                    .count()
              : 0.0;
      output.aim_command = selection.output;
      output.simulator_command = simulator_command;
      output.talos_command = talos_command;
      output.udp_publish = udp_status;
      output.talos_publish = talos_status;
      std::lock_guard<std::mutex> output_lock(output_mutex_);
      last_output_ = std::move(output);
    }

    return {
        selection.disposition,
        selection.source_age,
        udp_status,
        talos_status,
    };
  }

  void run() {
    FileHandle talos_meta_file;
    UdpSender udp_sender;
    if (!enable_talos_command_) {
      std::cerr << "aim_sim_talos_bridge wall publisher Talos command output "
                   "disabled; UDP remains independent\n";
    }
    if (enable_udp_) {
      if (udp_sender.open(udp_host_, udp_port_)) {
        std::cerr << "aim_sim_talos_bridge wall publisher UDP output udp://"
                  << udp_host_ << ":" << udp_port_ << "\n";
      } else {
        std::cerr
            << "aim_sim_talos_bridge wall publisher UDP unavailable udp://"
            << udp_host_ << ":" << udp_port_ << "\n";
      }
    }

    scheduler_.run(
        [this]() { return stop_requested_.load(std::memory_order_acquire); },
        [this, &talos_meta_file,
         &udp_sender](const aim_sim_bridge::control::TickContext &context) {
          return publishTick(context, talos_meta_file, udp_sender);
        });
  }

  std::filesystem::path meta_path_;
  aim_sim_bridge::AimBridgeConfig config_;
  bool enable_udp_ = false;
  bool enable_talos_command_ = true;
  std::string udp_host_;
  int udp_port_ = 0;

  aim_sim_bridge::control::LatestCommandState<aim_sim_bridge::AimCommand>
      latest_command_;
  aim_sim_bridge::control::SteadyRateScheduler scheduler_;
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> following_enabled_{false};
  mutable std::mutex publication_gate_;
  mutable std::mutex output_mutex_;
  WallCommandPublisherSnapshot last_output_;

  std::atomic<std::uint64_t> submit_accepted_count_{0};
  std::atomic<std::uint64_t> submit_not_completed_count_{0};
  std::atomic<std::uint64_t> submit_disabled_count_{0};
  std::atomic<std::uint64_t> submit_duplicate_count_{0};
  std::atomic<std::uint64_t> submit_rejected_older_count_{0};
  std::atomic<std::uint64_t> submit_invalid_count_{0};
  std::atomic<std::uint64_t> clear_count_{0};
  std::thread worker_;
};

int runWallControlIntegrationSelfTest() {
  const auto fail = [](const char *message) {
    std::cerr << "wall control integration self-test failed: " << message
              << "\n";
    return 1;
  };

  using namespace std::chrono_literals;
  const auto base =
      aim_sim_bridge::control::SteadyTimePoint(std::chrono::seconds(10));
  const auto converted = sourceSteadyTimeFromWall(1'000'000'000, 1'005'000'000,
                                                  1'010'000'000, base);
  if (!converted || *converted != base - 10ms) {
    return fail("wall-to-steady source age conversion");
  }
  if (sourceSteadyTimeFromWall(0, 1, 2, base) ||
      sourceSteadyTimeFromWall(10, 9, 11, base) ||
      sourceSteadyTimeFromWall(10, 12, 11, base)) {
    return fail("invalid wall timestamp rejection");
  }

  aim_sim_bridge::AimBridgeConfig config;
  config.enable_fire = true;
  aim_sim_bridge::AimCommand command;
  command.completed_vision_result = true;
  command.source_producer_epoch = 77;
  command.source_image_seq = 1234;
  command.source_capture_timestamp_ns = 1'000'000'000;
  command.vision_completion_timestamp_ns = 1'005'000'000;
  command.has_target = true;
  command.yaw_deg = 4.5;
  command.pitch_deg = -1.25;
  command.distance_m = 3.0;
  command.fire_advice = true;

  const auto optical_udp = aim_sim_bridge::toSimulatorCommand(command, config);
  const auto optical_talos = toTalosCommand(command, config);
  if (std::abs(optical_udp.pitch_deg - 66.25) > 1e-9 ||
      std::abs(static_cast<double>(optical_talos.pitch_deg) + 66.25) > 1e-6) {
    return fail("optical pitch to UDP/Talos mount conversion");
  }

  aim_sim_bridge::control::LatestCommandState<aim_sim_bridge::AimCommand> state(
      aim_sim_bridge::AimCommand{});
  if (state.publish({{77, 1234}, base, true, command}) !=
      aim_sim_bridge::control::CommandPublishStatus::Accepted) {
    return fail("completed command acceptance");
  }
  const auto fresh = state.select(base + 1ms);
  const auto repeated = state.select(base + 4ms);
  const auto identity_preserved = [](const aim_sim_bridge::AimCommand &output) {
    return output.completed_vision_result &&
           output.source_producer_epoch == 77 &&
           output.source_image_seq == 1234 &&
           output.source_capture_timestamp_ns == 1'000'000'000 &&
           output.vision_completion_timestamp_ns == 1'005'000'000;
  };
  if (fresh.disposition != aim_sim_bridge::control::CommandDisposition::Fresh ||
      repeated.disposition !=
          aim_sim_bridge::control::CommandDisposition::Repeated ||
      !identity_preserved(fresh.output) ||
      !identity_preserved(repeated.output)) {
    return fail("B-I3 identity preservation across repeated ticks");
  }

  auto stale = state.select(base + 251ms);
  preserveVisionProvenanceForSafeOutput(stale);
  const auto stale_udp =
      aim_sim_bridge::toSimulatorCommand(stale.output, config);
  const auto stale_talos = toTalosCommand(stale.output, config);
  if (stale.disposition != aim_sim_bridge::control::CommandDisposition::Stale ||
      !identity_preserved(stale.output) || stale_udp.distance_m != -1.0 ||
      stale_udp.fire_advice || stale_talos.distance_m != -1.0f ||
      stale_talos.fire_advice != 0) {
    return fail("stale provenance and safe UDP/Talos no-target");
  }

  aim_sim_bridge::control::TimestampedCommand<aim_sim_bridge::AimCommand>
      duplicate{{77, 1234}, base + 300ms, true, command};
  if (state.publish(duplicate) !=
          aim_sim_bridge::control::CommandPublishStatus::DuplicateSource ||
      state.select(base + 301ms).disposition !=
          aim_sim_bridge::control::CommandDisposition::Stale) {
    return fail("duplicate cannot refresh stale source");
  }

  auto no_target_command = command;
  no_target_command.source_image_seq = 1235;
  no_target_command.has_target = false;
  no_target_command.fire_advice = false;
  if (state.publish({{77, 1235}, base + 400ms, false, no_target_command}) !=
      aim_sim_bridge::control::CommandPublishStatus::Accepted) {
    return fail("completed no-target command acceptance");
  }
  auto no_target = state.select(base + 401ms);
  preserveVisionProvenanceForSafeOutput(no_target);
  if (no_target.disposition !=
          aim_sim_bridge::control::CommandDisposition::NoTarget ||
      !no_target.output.completed_vision_result ||
      no_target.output.source_producer_epoch != 77 ||
      no_target.output.source_image_seq != 1235 ||
      no_target.output.source_capture_timestamp_ns != 1'000'000'000 ||
      no_target.output.vision_completion_timestamp_ns != 1'005'000'000 ||
      aim_sim_bridge::toSimulatorCommand(no_target.output, config).distance_m !=
          -1.0 ||
      aim_sim_bridge::toSimulatorCommand(no_target.output, config)
          .fire_advice) {
    return fail("completed no-target provenance and safe output");
  }
  state.clear();
  const auto cleared = state.select(base + 302ms);
  if (cleared.disposition !=
          aim_sim_bridge::control::CommandDisposition::Unavailable ||
      aim_sim_bridge::toSimulatorCommand(cleared.output, config).fire_advice) {
    return fail("epoch/F5 clear remains safe");
  }

  std::cout << "wall control integration self-test passed\n";
  return 0;
}

std::string float3Json(const float values[3]) {
  std::ostringstream out;
  out << std::setprecision(10) << '[' << values[0] << ',' << values[1] << ','
      << values[2] << ']';
  return out.str();
}

std::string double3Json(const double values[3]) {
  std::ostringstream out;
  out << std::setprecision(10) << '[' << values[0] << ',' << values[1] << ','
      << values[2] << ']';
  return out.str();
}

std::string float4Json(const float values[4]) {
  std::ostringstream out;
  out << std::setprecision(10) << '[' << values[0] << ',' << values[1] << ','
      << values[2] << ',' << values[3] << ']';
  return out.str();
}

const char *groundTruthFrameLabel(std::uint8_t frame) {
  switch (frame) {
  case kGroundTruthFrameRosOdom:
    return "ros_odom";
  case kGroundTruthFrameChassisLocalRos:
    return "chassis_local_ros";
  case kGroundTruthFrameUnknown:
  default:
    return "unknown";
  }
}

std::string groundTruthJson(const GroundTruthBatch &ground_truth) {
  std::ostringstream out;
  out << std::setprecision(10) << '{';
  bool first = true;
  aim_sim_bridge::debug::appendUInt(out, "frame_seq", ground_truth.frame_seq,
                                    first);
  aim_sim_bridge::debug::appendUInt(out, "timestamp_ns",
                                    ground_truth.timestamp_ns, first);
  aim_sim_bridge::debug::appendUInt(out, "target_count",
                                    ground_truth.target_count, first);
  aim_sim_bridge::debug::appendUInt(out, "rune_count", ground_truth.rune_count,
                                    first);

  std::ostringstream targets;
  targets << '[';
  const std::uint32_t count = std::min<std::uint32_t>(
      ground_truth.target_count,
      static_cast<std::uint32_t>(kGroundTruthMaxTargets));
  for (std::uint32_t i = 0; i < count; ++i) {
    if (i > 0)
      targets << ',';
    const GroundTruthTarget &target = ground_truth.targets[i];
    targets << '{';
    bool target_first = true;
    aim_sim_bridge::debug::appendUInt(targets, "frame_seq", target.frame_seq,
                                      target_first);
    aim_sim_bridge::debug::appendUInt(targets, "timestamp_ns",
                                      target.timestamp_ns, target_first);
    aim_sim_bridge::debug::appendUInt(targets, "target_id", target.target_id,
                                      target_first);
    aim_sim_bridge::debug::appendString(targets, "target_id_scope",
                                        "simulator_run", target_first);
    aim_sim_bridge::debug::appendUInt(targets, "team", target.team,
                                      target_first);
    aim_sim_bridge::debug::appendUInt(targets, "armor_label",
                                      target.armor_label, target_first);
    aim_sim_bridge::debug::appendBool(targets, "is_outpost",
                                      target.is_outpost != 0, target_first);
    aim_sim_bridge::debug::appendUInt(targets, "armor_count",
                                      target.armor_count, target_first);
    aim_sim_bridge::debug::appendUInt(targets, "state_flags",
                                      target.state_flags, target_first);
    aim_sim_bridge::debug::appendBool(
        targets, "has_world_state",
        (target.state_flags & kGroundTruthTargetHasWorldState) != 0,
        target_first);
    aim_sim_bridge::debug::appendBool(
        targets, "has_world_orientation",
        (target.state_flags & kGroundTruthTargetHasWorldOrientation) != 0,
        target_first);
    aim_sim_bridge::debug::appendBool(
        targets, "has_chassis_local_armor_geometry",
        (target.state_flags & kGroundTruthTargetHasArmorGeometry) != 0,
        target_first);
    aim_sim_bridge::debug::appendUInt(targets, "world_state_frame_code",
                                      target.world_state_frame, target_first);
    aim_sim_bridge::debug::appendString(
        targets, "world_state_frame",
        groundTruthFrameLabel(target.world_state_frame), target_first);
    aim_sim_bridge::debug::appendUInt(
        targets, "armor_geometry_frame_code", target.armor_geometry_frame,
        target_first);
    aim_sim_bridge::debug::appendString(
        targets, "armor_geometry_frame",
        groundTruthFrameLabel(target.armor_geometry_frame), target_first);
    aim_sim_bridge::debug::appendRaw(
        targets, "world_position_m", float3Json(target.position), target_first);
    aim_sim_bridge::debug::appendRaw(
        targets, "world_velocity_mps", float3Json(target.velocity),
        target_first);
    aim_sim_bridge::debug::appendRaw(
        targets, "world_quaternion_wxyz",
        float4Json(target.world_quaternion_wxyz), target_first);
    aim_sim_bridge::debug::appendNumber(targets, "world_vyaw_rad_s",
                                        target.vyaw, target_first);
    aim_sim_bridge::debug::appendNumber(targets, "world_yaw_rad", target.yaw,
                                        target_first);
    // Legacy aliases retained for existing telemetry consumers.
    aim_sim_bridge::debug::appendRaw(targets, "position_m",
                                     float3Json(target.position), target_first);
    aim_sim_bridge::debug::appendRaw(targets, "velocity_mps",
                                     float3Json(target.velocity), target_first);
    aim_sim_bridge::debug::appendNumber(targets, "vyaw_rad_s", target.vyaw,
                                        target_first);
    aim_sim_bridge::debug::appendNumber(targets, "yaw_rad", target.yaw,
                                        target_first);
    aim_sim_bridge::debug::appendNumber(targets, "radius_even_m",
                                        target.radius_even, target_first);
    aim_sim_bridge::debug::appendNumber(targets, "radius_odd_m",
                                        target.radius_odd, target_first);
    aim_sim_bridge::debug::appendNumber(targets, "armor_height_m",
                                        target.armor_height, target_first);

    std::ostringstream armors;
    armors << '[';
    const std::uint32_t armor_count = std::min<std::uint32_t>(
        target.armor_count,
        static_cast<std::uint32_t>(kGroundTruthMaxArmorsPerTarget));
    for (std::uint32_t armor_index = 0; armor_index < armor_count;
         ++armor_index) {
      if (armor_index > 0)
        armors << ',';
      const GroundTruthArmor &armor = target.armors[armor_index];
      armors << '{';
      bool armor_first = true;
      aim_sim_bridge::debug::appendUInt(armors, "relative_slot",
                                        armor.relative_slot, armor_first);
      aim_sim_bridge::debug::appendUInt(armors, "visibility_code",
                                        armor.visibility, armor_first);
      aim_sim_bridge::debug::appendString(
          armors, "visibility",
          armor.visibility == kGroundTruthVisibilityHidden ? "hidden"
                                                          : "unknown",
          armor_first);
      aim_sim_bridge::debug::appendBool(
          armors, "visibility_known",
          armor.visibility != kGroundTruthVisibilityUnknown, armor_first);
      aim_sim_bridge::debug::appendRaw(
          armors, "relative_position_m", float3Json(armor.relative_position),
          armor_first);
      aim_sim_bridge::debug::appendRaw(
          armors, "chassis_local_position_m",
          float3Json(armor.relative_position), armor_first);
      aim_sim_bridge::debug::appendRaw(
          armors, "outward_normal", float3Json(armor.outward_normal),
          armor_first);
      aim_sim_bridge::debug::appendRaw(
          armors, "chassis_local_outward_normal",
          float3Json(armor.outward_normal), armor_first);
      aim_sim_bridge::debug::appendNumber(
          armors, "relative_yaw_rad", armor.relative_yaw, armor_first);
      armors << '}';
    }
    armors << ']';
    aim_sim_bridge::debug::appendRaw(targets, "armors", armors.str(),
                                     target_first);
    targets << '}';
  }
  targets << ']';
  aim_sim_bridge::debug::appendRaw(out, "targets", targets.str(), first);
  out << '}';
  return out.str();
}

int runGroundTruthLayoutSelfTest() {
  GroundTruthBatch ground_truth{};
  ground_truth.frame_seq = 42;
  ground_truth.timestamp_ns = 123456789;
  ground_truth.target_count = 1;
  GroundTruthTarget &target = ground_truth.targets[0];
  target.frame_seq = ground_truth.frame_seq;
  target.timestamp_ns = ground_truth.timestamp_ns;
  target.target_id = 987654321;
  target.armor_count = 1;
  target.radius_even = 0.30F;
  target.radius_odd = 0.20F;
  target.armor_height = 0.12F;
  target.armors[0].relative_slot = 0;
  target.armors[0].visibility = kGroundTruthVisibilityUnknown;
  target.armors[0].relative_position[0] = 0.30F;
  target.armors[0].outward_normal[0] = 1.0F;
  target.world_quaternion_wxyz[0] = 1.0F;
  target.world_state_frame = kGroundTruthFrameRosOdom;
  target.armor_geometry_frame = kGroundTruthFrameChassisLocalRos;
  target.state_flags = kGroundTruthTargetHasWorldState |
                       kGroundTruthTargetHasWorldOrientation |
                       kGroundTruthTargetHasArmorGeometry;

  const std::string json = groundTruthJson(ground_truth);
  const auto require = [&json](const char *needle) {
    return json.find(needle) != std::string::npos;
  };
  if (!require("\"frame_seq\":42") ||
      !require("\"timestamp_ns\":123456789") ||
      !require("\"target_id\":987654321") ||
      !require("\"world_state_frame\":\"ros_odom\"") ||
      !require("\"armor_geometry_frame\":\"chassis_local_ros\"") ||
      !require("\"world_quaternion_wxyz\":[1,0,0,0]") ||
      !require("\"has_world_state\":true") ||
      !require("\"has_chassis_local_armor_geometry\":true") ||
      !require("\"radius_even_m\":0.3") ||
      !require("\"visibility\":\"unknown\"") ||
      !require("\"visibility_known\":false") ||
      !require("\"relative_position_m\":[0.3000000119,0,0]")) {
    std::cerr << "ground truth layout self-test failed: " << json << '\n';
    return 1;
  }

  GroundTruthHistory history{};
  history.slots[1].commit_seq = 3; // in-progress write must never be read
  history.slots[1].ground_truth = ground_truth;
  history.slots[1].exposure_state.frame_seq = ground_truth.frame_seq;
  history.slots[1].exposure_state.timestamp_ns = ground_truth.timestamp_ns;
  ExactExposureTruth selected{};
  if (selectExactExposureTruth(history, ground_truth.frame_seq,
                               ground_truth.timestamp_ns, &selected)) {
    std::cerr << "ground truth layout self-test accepted odd seqlock\n";
    return 1;
  }
  history.slots[2].commit_seq = 4;
  history.slots[2].ground_truth = ground_truth;
  history.slots[2].exposure_state.frame_seq = ground_truth.frame_seq;
  history.slots[2].exposure_state.timestamp_ns = ground_truth.timestamp_ns;
  if (!selectExactExposureTruth(history, ground_truth.frame_seq,
                                ground_truth.timestamp_ns, &selected) ||
      selected.slot_index != 2 || selected.commit_seq != 4 ||
      selectExactExposureTruth(history, ground_truth.frame_seq + 1,
                               ground_truth.timestamp_ns, &selected)) {
    std::cerr << "ground truth layout self-test exact ring selection failed\n";
    return 1;
  }
  std::cout << "ground truth layout self-test passed\n";
  return 0;
}

void writeBridgeTelemetry(
    const std::string &path, const std::string &jsonl_path,
    aim_sim_bridge::TargetMode mode,
    std::uint64_t processed_frames, const ImageSequenceState &sequence,
    const IngressMetrics &ingress, double unique_ingress_rolling_hz,
    double unique_ingress_lifetime_hz,
    const CompletedVisionMetrics &completed_vision,
    double completed_vision_rolling_hz, double completed_vision_lifetime_hz,
    bool following, const ImageMeta &image_meta, const CameraInfo &camera_info,
    const RuntimeState &runtime_state, const GroundTruthBatch &ground_truth,
    bool has_feedback, const GimbalFeedback &feedback, double chassis_yaw_deg,
    double simulator_local_gimbal_yaw_deg, double vivsionn_gimbal_yaw_deg,
    bool has_runtime_gimbal, double runtime_gimbal_yaw_deg,
    double runtime_gimbal_pitch_deg, const char *gimbal_feedback_source,
    const aim_sim_bridge::SimFrame &frame,
    const aim_sim_bridge::AimCommand &aim,
    const WallCommandPublisherSnapshot &wall_control,
    const ImageTransportTelemetry &image_transport, double detector_fps,
    double bridge_elapsed_s) {
  if (path.empty())
    return;

  std::ostringstream image_json;
  image_json << '{';
  bool image_first = true;
  aim_sim_bridge::debug::appendBool(image_json, "has_processed_frame",
                                    image_meta.seq != 0, image_first);
  aim_sim_bridge::debug::appendUInt(image_json, "seq", image_meta.seq,
                                    image_first);
  aim_sim_bridge::debug::appendUInt(image_json, "timestamp_ns",
                                    image_meta.timestamp_ns, image_first);
  aim_sim_bridge::debug::appendUInt(image_json, "width", image_meta.width,
                                    image_first);
  aim_sim_bridge::debug::appendUInt(image_json, "height", image_meta.height,
                                    image_first);
  aim_sim_bridge::debug::appendUInt(image_json, "buffer_id",
                                    image_meta.buffer_id, image_first);
  aim_sim_bridge::debug::appendString(
      image_json, "transport", imageTransportLabel(image_transport.transport),
      image_first);
  aim_sim_bridge::debug::appendString(
      image_json, "pixel_format",
      image_transport.transport == ImageTransport::Tcp
          ? tcpPixelFormatLabel(image_transport.tcp_bridge.last_format)
          : "rgb24",
      image_first);
  image_json << '}';

  const auto &tcp_receiver = image_transport.tcp_receiver;
  const auto &tcp_bridge = image_transport.tcp_bridge;
  const double tcp_complete_lifetime_hz =
      bridge_elapsed_s > 1e-6
          ? static_cast<double>(tcp_receiver.complete_frames) / bridge_elapsed_s
          : 0.0;
  const double tcp_accepted_lifetime_hz =
      bridge_elapsed_s > 1e-6
          ? static_cast<double>(tcp_receiver.accepted_frames) / bridge_elapsed_s
          : 0.0;
  const double tcp_selected_lifetime_hz =
      bridge_elapsed_s > 1e-6
          ? static_cast<double>(tcp_bridge.selected_frames) / bridge_elapsed_s
          : 0.0;
  const double tcp_wire_mib_s =
      bridge_elapsed_s > 1e-6
          ? static_cast<double>(tcp_receiver.wire_bytes_received) /
                bridge_elapsed_s / (1024.0 * 1024.0)
          : 0.0;
  const double tcp_wire_gbit_s =
      bridge_elapsed_s > 1e-6
          ? static_cast<double>(tcp_receiver.wire_bytes_received) * 8.0 /
                bridge_elapsed_s / 1e9
          : 0.0;

  std::ostringstream transport_json;
  transport_json << std::setprecision(10) << '{';
  bool transport_first = true;
  aim_sim_bridge::debug::appendString(
      transport_json, "mode", imageTransportLabel(image_transport.transport),
      transport_first);
  aim_sim_bridge::debug::appendBool(
      transport_json, "tcp_enabled",
      image_transport.transport == ImageTransport::Tcp, transport_first);
  aim_sim_bridge::debug::appendString(
      transport_json, "tcp_host", image_transport.tcp_host, transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "tcp_port",
                                    image_transport.tcp_port, transport_first);
  aim_sim_bridge::debug::appendBool(transport_json, "tcp_receiver_running",
                                    image_transport.tcp_receiver_running,
                                    transport_first);
  aim_sim_bridge::debug::appendBool(transport_json, "has_selected_frame",
                                    tcp_bridge.selected_frames > 0,
                                    transport_first);
  aim_sim_bridge::debug::appendString(
      transport_json, "latest_pixel_format",
      tcpPixelFormatLabel(tcp_bridge.last_format), transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "latest_width",
                                    tcp_bridge.last_width, transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "latest_height",
                                    tcp_bridge.last_height, transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "latest_epoch",
                                    tcp_bridge.last_epoch, transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "latest_sequence",
                                    tcp_bridge.last_sequence, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "latest_capture_timestamp_ns",
      tcp_bridge.last_capture_timestamp_ns, transport_first);
  aim_sim_bridge::debug::appendBool(
      transport_json, "selected_source_age_available",
      tcp_bridge.selected_source_age_available, transport_first);
  aim_sim_bridge::debug::appendNumber(
      transport_json, "latest_selected_source_age_ms",
      tcp_bridge.latest_selected_source_age_ms, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "selected_source_age_samples",
      tcp_bridge.selected_source_age_samples, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "invalid_selected_source_age_samples",
      tcp_bridge.invalid_selected_source_age_samples, transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "main_taken_frames",
                                    tcp_bridge.taken_frames, transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "main_selected_frames",
                                    tcp_bridge.selected_frames,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "main_epoch_mismatch_rejects",
      tcp_bridge.epoch_mismatch_rejects, transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "main_duplicate_rejects",
                                    tcp_bridge.duplicate_rejects,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "main_regression_rejects",
                                    tcp_bridge.regression_rejects,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "main_invalid_frame_rejects",
      tcp_bridge.invalid_frame_rejects, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "main_color_conversion_failures",
      tcp_bridge.color_conversion_failures, transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "connect_attempts",
                                    tcp_receiver.connect_attempts,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "connect_successes",
                                    tcp_receiver.connect_successes,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "connect_failures",
                                    tcp_receiver.connect_failures,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "reconnect_attempts",
                                    tcp_receiver.reconnect_attempts,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "disconnects",
                                    tcp_receiver.disconnects, transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "headers_received",
                                    tcp_receiver.headers_received,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "complete_frames",
                                    tcp_receiver.complete_frames,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "accepted_frames",
                                    tcp_receiver.accepted_frames,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "replaced_frames",
                                    tcp_receiver.replaced_frames,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "duplicate_frames",
                                    tcp_receiver.duplicate_frames,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "regression_frames",
                                    tcp_receiver.regression_frames,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "invalid_headers",
                                    tcp_receiver.invalid_headers,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "read_failures",
                                    tcp_receiver.read_failures,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "allocation_failures",
                                    tcp_receiver.allocation_failures,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(transport_json, "wire_bytes_received",
                                    tcp_receiver.wire_bytes_received,
                                    transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "header_read_duration_count",
      tcp_receiver.header_read_duration_count, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "header_read_duration_ns_total",
      tcp_receiver.header_read_duration_ns_total, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "header_read_duration_ns_max",
      tcp_receiver.header_read_duration_ns_max, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "payload_read_duration_count",
      tcp_receiver.payload_read_duration_count, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "payload_read_duration_ns_total",
      tcp_receiver.payload_read_duration_ns_total, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "payload_read_duration_ns_max",
      tcp_receiver.payload_read_duration_ns_max, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "connection_lifetime_count",
      tcp_receiver.connection_lifetime_count, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "connection_lifetime_ns_total",
      tcp_receiver.connection_lifetime_ns_total, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "connection_lifetime_ns_max",
      tcp_receiver.connection_lifetime_ns_max, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "receiver_source_age_samples",
      tcp_receiver.source_age_samples, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "receiver_invalid_source_age_samples",
      tcp_receiver.invalid_source_age_samples, transport_first);
  aim_sim_bridge::debug::appendBool(
      transport_json, "receiver_source_age_available",
      tcp_receiver.source_age_available, transport_first);
  aim_sim_bridge::debug::appendNumber(
      transport_json, "receiver_latest_source_age_ms",
      tcp_receiver.latest_source_age_ms, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "receiver_last_accepted_epoch",
      tcp_receiver.last_accepted_epoch, transport_first);
  aim_sim_bridge::debug::appendUInt(
      transport_json, "receiver_last_accepted_sequence",
      tcp_receiver.last_accepted_sequence, transport_first);
  aim_sim_bridge::debug::appendNumber(
      transport_json, "complete_frame_lifetime_hz", tcp_complete_lifetime_hz,
      transport_first);
  aim_sim_bridge::debug::appendNumber(
      transport_json, "accepted_frame_lifetime_hz", tcp_accepted_lifetime_hz,
      transport_first);
  aim_sim_bridge::debug::appendNumber(
      transport_json, "selected_frame_lifetime_hz", tcp_selected_lifetime_hz,
      transport_first);
  aim_sim_bridge::debug::appendNumber(transport_json, "wire_mib_s",
                                      tcp_wire_mib_s, transport_first);
  aim_sim_bridge::debug::appendNumber(transport_json, "wire_gbit_s",
                                      tcp_wire_gbit_s, transport_first);
  transport_json << '}';

  std::ostringstream ingress_json;
  ingress_json << std::setprecision(10) << '{';
  bool ingress_first = true;
  aim_sim_bridge::debug::appendString(
      ingress_json, "transport", imageTransportLabel(image_transport.transport),
      ingress_first);
  aim_sim_bridge::debug::appendUInt(ingress_json, "producer_epoch",
                                    sequence.producer_epoch, ingress_first);
  aim_sim_bridge::debug::appendUInt(ingress_json, "producer_epoch_changes",
                                    ingress.producer_epoch_changes,
                                    ingress_first);
  aim_sim_bridge::debug::appendBool(ingress_json, "has_last_accepted_seq",
                                    sequence.has_last_accepted, ingress_first);
  aim_sim_bridge::debug::appendUInt(ingress_json, "last_accepted_seq",
                                    sequence.last_accepted_seq, ingress_first);
  aim_sim_bridge::debug::appendUInt(ingress_json, "unique_accepted_frames",
                                    ingress.unique_accepted_frames,
                                    ingress_first);
  aim_sim_bridge::debug::appendUInt(ingress_json, "source_sequence_gap_frames",
                                    ingress.sequence_gap_frames, ingress_first);
  aim_sim_bridge::debug::appendUInt(ingress_json, "duplicate_sequence_rejects",
                                    ingress.duplicate_rejects, ingress_first);
  aim_sim_bridge::debug::appendUInt(ingress_json, "regression_sequence_rejects",
                                    ingress.regression_rejects, ingress_first);
  aim_sim_bridge::debug::appendUInt(ingress_json, "inconsistent_image_reads",
                                    ingress.inconsistent_image_reads,
                                    ingress_first);
  aim_sim_bridge::debug::appendUInt(ingress_json, "image_read_failures",
                                    ingress.image_read_failures, ingress_first);
  aim_sim_bridge::debug::appendNumber(ingress_json, "rolling_window_s", 1.0,
                                      ingress_first);
  aim_sim_bridge::debug::appendNumber(ingress_json, "unique_ingress_rolling_hz",
                                      unique_ingress_rolling_hz, ingress_first);
  aim_sim_bridge::debug::appendNumber(
      ingress_json, "unique_ingress_lifetime_hz", unique_ingress_lifetime_hz,
      ingress_first);
  ingress_json << '}';

  std::ostringstream completion_json;
  completion_json << std::setprecision(10) << '{';
  bool completion_first = true;
  aim_sim_bridge::debug::appendString(completion_json, "counting_boundary",
                                      "post_yolo_solve_pnp_track_aim",
                                      completion_first);
  aim_sim_bridge::debug::appendString(completion_json, "identity",
                                      "producer_epoch_image_seq",
                                      completion_first);
  aim_sim_bridge::debug::appendString(completion_json, "resend_policy",
                                      "duplicate_identity_rejected",
                                      completion_first);
  aim_sim_bridge::debug::appendUInt(completion_json, "active_producer_epoch",
                                    completed_vision.active_producer_epoch,
                                    completion_first);
  aim_sim_bridge::debug::appendUInt(completion_json, "producer_epoch_resets",
                                    completed_vision.producer_epoch_resets,
                                    completion_first);
  aim_sim_bridge::debug::appendBool(
      completion_json, "has_last_completed_result",
      completed_vision.has_last_completed_result, completion_first);
  aim_sim_bridge::debug::appendUInt(completion_json, "last_completed_epoch",
                                    completed_vision.last_completed_epoch,
                                    completion_first);
  aim_sim_bridge::debug::appendUInt(completion_json, "last_completed_seq",
                                    completed_vision.last_completed_seq,
                                    completion_first);
  aim_sim_bridge::debug::appendUInt(
      completion_json, "last_source_capture_timestamp_ns",
      completed_vision.last_source_capture_timestamp_ns, completion_first);
  aim_sim_bridge::debug::appendUInt(
      completion_json, "last_completion_timestamp_ns",
      completed_vision.last_completion_timestamp_ns, completion_first);
  aim_sim_bridge::debug::appendUInt(
      completion_json, "completion_events_observed",
      completed_vision.completion_events_observed, completion_first);
  aim_sim_bridge::debug::appendUInt(completion_json, "processed_result_count",
                                    completed_vision.processed_result_count,
                                    completion_first);
  aim_sim_bridge::debug::appendUInt(completion_json, "unique_completed_results",
                                    completed_vision.processed_result_count,
                                    completion_first);
  aim_sim_bridge::debug::appendUInt(
      completion_json, "source_sequence_gap_frames",
      completed_vision.completed_sequence_gap_frames, completion_first);
  aim_sim_bridge::debug::appendUInt(
      completion_json, "duplicate_sequence_rejects",
      completed_vision.duplicate_sequence_rejects, completion_first);
  aim_sim_bridge::debug::appendUInt(
      completion_json, "regression_sequence_rejects",
      completed_vision.regression_sequence_rejects, completion_first);
  aim_sim_bridge::debug::appendUInt(completion_json, "stale_epoch_rejects",
                                    completed_vision.stale_epoch_rejects,
                                    completion_first);
  aim_sim_bridge::debug::appendUInt(completion_json, "invalid_identity_rejects",
                                    completed_vision.invalid_identity_rejects,
                                    completion_first);
  aim_sim_bridge::debug::appendUInt(completion_json, "invalid_age_samples",
                                    completed_vision.invalid_age_samples,
                                    completion_first);
  aim_sim_bridge::debug::appendNumber(completion_json, "rolling_window_s", 1.0,
                                      completion_first);
  aim_sim_bridge::debug::appendNumber(
      completion_json, "completed_vision_rolling_hz",
      completed_vision_rolling_hz, completion_first);
  aim_sim_bridge::debug::appendNumber(
      completion_json, "completed_vision_lifetime_hz",
      completed_vision_lifetime_hz, completion_first);
  aim_sim_bridge::debug::appendNumber(
      completion_json, "latest_source_to_completion_age_ms",
      completed_vision.latest_source_to_completion_age_ms, completion_first);
  aim_sim_bridge::debug::appendNumber(
      completion_json, "rolling_mean_source_to_completion_age_ms",
      completed_vision.rollingMeanAgeMs(), completion_first);
  aim_sim_bridge::debug::appendNumber(
      completion_json, "rolling_p95_source_to_completion_age_ms",
      completed_vision.rollingP95AgeMs(), completion_first);
  aim_sim_bridge::debug::appendNumber(
      completion_json, "lifetime_mean_source_to_completion_age_ms",
      completed_vision.lifetimeMeanAgeMs(), completion_first);
  completion_json << '}';

  std::ostringstream camera_json;
  camera_json << std::setprecision(10) << '{';
  bool camera_first = true;
  aim_sim_bridge::debug::appendNumber(camera_json, "fx", camera_info.fx,
                                      camera_first);
  aim_sim_bridge::debug::appendNumber(camera_json, "fy", camera_info.fy,
                                      camera_first);
  aim_sim_bridge::debug::appendNumber(camera_json, "cx", camera_info.cx,
                                      camera_first);
  aim_sim_bridge::debug::appendNumber(camera_json, "cy", camera_info.cy,
                                      camera_first);
  aim_sim_bridge::debug::appendUInt(camera_json, "width", camera_info.width,
                                    camera_first);
  aim_sim_bridge::debug::appendUInt(camera_json, "height", camera_info.height,
                                    camera_first);
  camera_json << '}';

  std::ostringstream gimbal_json;
  gimbal_json << std::setprecision(10) << '{';
  bool gimbal_first = true;
  aim_sim_bridge::debug::appendBool(gimbal_json, "has_feedback", has_feedback,
                                    gimbal_first);
  aim_sim_bridge::debug::appendUInt(gimbal_json, "feedback_frame_seq",
                                    feedback.frame_seq, gimbal_first);
  aim_sim_bridge::debug::appendUInt(gimbal_json, "feedback_timestamp_ns",
                                    feedback.timestamp_ns, gimbal_first);
  aim_sim_bridge::debug::appendInt(gimbal_json, "feedback_frame_seq_delta",
                                   static_cast<long long>(feedback.frame_seq) -
                                       static_cast<long long>(image_meta.seq),
                                   gimbal_first);
  aim_sim_bridge::debug::appendNumber(
      gimbal_json, "feedback_minus_image_ms",
      has_feedback && feedback.timestamp_ns != 0 && image_meta.timestamp_ns != 0
          ? (static_cast<double>(feedback.timestamp_ns) -
             static_cast<double>(image_meta.timestamp_ns)) *
                1e-6
          : std::numeric_limits<double>::quiet_NaN(),
      gimbal_first);
  aim_sim_bridge::debug::appendNumber(gimbal_json, "global_yaw_deg",
                                      feedback.yaw_deg, gimbal_first);
  aim_sim_bridge::debug::appendNumber(gimbal_json, "chassis_yaw_deg",
                                      chassis_yaw_deg, gimbal_first);
  aim_sim_bridge::debug::appendNumber(gimbal_json, "simulator_local_yaw_deg",
                                      simulator_local_gimbal_yaw_deg,
                                      gimbal_first);
  aim_sim_bridge::debug::appendNumber(gimbal_json, "vivsionn_yaw_deg",
                                      vivsionn_gimbal_yaw_deg, gimbal_first);
  aim_sim_bridge::debug::appendString(
      gimbal_json, "yaw_adapter", "simulator_to_vivsionn_negate", gimbal_first);
  aim_sim_bridge::debug::appendNumber(gimbal_json, "pitch_deg",
                                      feedback.pitch_deg, gimbal_first);
  aim_sim_bridge::debug::appendBool(gimbal_json, "has_runtime_gimbal",
                                    has_runtime_gimbal, gimbal_first);
  aim_sim_bridge::debug::appendUInt(gimbal_json, "runtime_timestamp_ns",
                                    runtime_state.timestamp_ns, gimbal_first);
  aim_sim_bridge::debug::appendNumber(
      gimbal_json, "runtime_minus_image_ms",
      has_runtime_gimbal && runtime_state.timestamp_ns != 0 &&
              image_meta.timestamp_ns != 0
          ? (static_cast<double>(runtime_state.timestamp_ns) -
             static_cast<double>(image_meta.timestamp_ns)) *
                1e-6
          : std::numeric_limits<double>::quiet_NaN(),
      gimbal_first);
  aim_sim_bridge::debug::appendNumber(gimbal_json, "runtime_local_yaw_deg",
                                      runtime_gimbal_yaw_deg, gimbal_first);
  aim_sim_bridge::debug::appendNumber(gimbal_json, "runtime_pitch_deg",
                                      runtime_gimbal_pitch_deg, gimbal_first);
  aim_sim_bridge::debug::appendString(gimbal_json, "feedback_source",
                                      gimbal_feedback_source, gimbal_first);
  gimbal_json << '}';

  std::ostringstream input_json;
  input_json << std::setprecision(10) << '{';
  bool input_first = true;
  aim_sim_bridge::debug::appendUInt(input_json, "source_producer_epoch",
                                    frame.source_producer_epoch, input_first);
  aim_sim_bridge::debug::appendUInt(input_json, "source_image_seq",
                                    frame.source_image_seq, input_first);
  aim_sim_bridge::debug::appendUInt(input_json, "source_capture_timestamp_ns",
                                    frame.source_capture_timestamp_ns,
                                    input_first);
  aim_sim_bridge::debug::appendUInt(input_json, "gimbal_pose_timestamp_ns",
                                    frame.gimbal_pose_timestamp_ns,
                                    input_first);
  aim_sim_bridge::debug::appendBool(
      input_json, "gimbal_pose_exposure_matched",
      frame.gimbal_pose_exposure_matched, input_first);
  aim_sim_bridge::debug::appendBool(
      input_json, "tracker_world_transform_exposure_matched",
      frame.tracker_world_transform_exposure_matched, input_first);
  const double tracker_origin_world_ros_m[3] = {
      frame.tracker_origin_world_ros_m[0], frame.tracker_origin_world_ros_m[1],
      frame.tracker_origin_world_ros_m[2]};
  const double tracker_frame_rpy_world_ros_rad[3] = {
      frame.tracker_frame_rpy_world_ros_rad[0],
      frame.tracker_frame_rpy_world_ros_rad[1],
      frame.tracker_frame_rpy_world_ros_rad[2]};
  aim_sim_bridge::debug::appendRaw(
      input_json, "tracker_origin_world_ros_m",
      double3Json(tracker_origin_world_ros_m), input_first);
  aim_sim_bridge::debug::appendRaw(
      input_json, "tracker_frame_rpy_world_ros_rad",
      double3Json(tracker_frame_rpy_world_ros_rad), input_first);
  aim_sim_bridge::debug::appendNumber(input_json, "gimbal_yaw_deg",
                                      frame.gimbal_yaw_deg, input_first);
  aim_sim_bridge::debug::appendNumber(input_json, "gimbal_pitch_deg",
                                      frame.gimbal_pitch_deg, input_first);
  aim_sim_bridge::debug::appendNumber(input_json, "gimbal_yaw_speed_deg_s",
                                      frame.gimbal_yaw_speed_deg_s,
                                      input_first);
  aim_sim_bridge::debug::appendNumber(input_json, "simulator_state_age_s",
                                      frame.simulator_state_age_s, input_first);
  aim_sim_bridge::debug::appendNumber(input_json, "bullet_speed_mps",
                                      frame.bullet_speed_mps, input_first);
  aim_sim_bridge::debug::appendNumber(input_json, "timestamp_ms",
                                      frame.timestamp_ms, input_first);
  input_json << '}';

  std::ostringstream aim_json;
  aim_json << std::setprecision(10) << '{';
  bool aim_first = true;
  aim_sim_bridge::debug::appendBool(aim_json, "has_target", aim.has_target,
                                    aim_first);
  aim_sim_bridge::debug::appendNumber(aim_json, "yaw_deg", aim.yaw_deg,
                                      aim_first);
  aim_sim_bridge::debug::appendNumber(aim_json, "pitch_deg", aim.pitch_deg,
                                      aim_first);
  aim_sim_bridge::debug::appendNumber(aim_json, "distance_m", aim.distance_m,
                                      aim_first);
  aim_sim_bridge::debug::appendBool(aim_json, "fire_advice", aim.fire_advice,
                                    aim_first);
  aim_sim_bridge::debug::appendInt(aim_json, "raw_shot_mode", aim.raw_shot_mode,
                                   aim_first);
  aim_sim_bridge::debug::appendString(aim_json, "selected_camera",
                                      aim.selected_camera, aim_first);
  aim_sim_bridge::debug::appendNumber(aim_json, "selected_focal_mm",
                                      aim.selected_focal_mm, aim_first);
  aim_sim_bridge::debug::appendString(aim_json, "backend", aim.backend,
                                      aim_first);
  aim_sim_bridge::debug::appendBool(aim_json, "completed_vision_result",
                                    aim.completed_vision_result, aim_first);
  aim_sim_bridge::debug::appendUInt(aim_json, "source_producer_epoch",
                                    aim.source_producer_epoch, aim_first);
  aim_sim_bridge::debug::appendUInt(aim_json, "source_image_seq",
                                    aim.source_image_seq, aim_first);
  aim_sim_bridge::debug::appendUInt(aim_json, "source_capture_timestamp_ns",
                                    aim.source_capture_timestamp_ns, aim_first);
  aim_sim_bridge::debug::appendUInt(aim_json, "vision_completion_timestamp_ns",
                                    aim.vision_completion_timestamp_ns,
                                    aim_first);
  aim_sim_bridge::debug::appendNumber(aim_json, "runtime_yolo_ms",
                                      aim.runtime_yolo_ms, aim_first);
  aim_sim_bridge::debug::appendNumber(aim_json, "runtime_solve_ms",
                                      aim.runtime_solve_ms, aim_first);
  aim_sim_bridge::debug::appendNumber(aim_json, "runtime_track_aim_ms",
                                      aim.runtime_track_aim_ms, aim_first);
  aim_sim_bridge::debug::appendNumber(aim_json, "runtime_pipeline_delay_ms",
                                      aim.runtime_pipeline_delay_ms, aim_first);
  aim_json << '}';

  std::ostringstream simulator_json;
  simulator_json << std::setprecision(10) << '{';
  bool simulator_first = true;
  aim_sim_bridge::debug::appendNumber(simulator_json, "yaw_deg",
                                      wall_control.simulator_command.yaw_deg,
                                      simulator_first);
  aim_sim_bridge::debug::appendNumber(simulator_json, "pitch_deg",
                                      wall_control.simulator_command.pitch_deg,
                                      simulator_first);
  aim_sim_bridge::debug::appendNumber(simulator_json, "distance_m",
                                      wall_control.simulator_command.distance_m,
                                      simulator_first);
  aim_sim_bridge::debug::appendBool(simulator_json, "fire_advice",
                                    wall_control.simulator_command.fire_advice,
                                    simulator_first);
  simulator_json << '}';

  std::ostringstream talos_json;
  talos_json << std::setprecision(10) << '{';
  bool talos_first = true;
  aim_sim_bridge::debug::appendBool(
      talos_json, "enabled", wall_control.talos_command_enabled, talos_first);
  aim_sim_bridge::debug::appendNumber(
      talos_json, "yaw_deg", wall_control.talos_command.yaw_deg, talos_first);
  aim_sim_bridge::debug::appendNumber(talos_json, "pitch_deg",
                                      wall_control.talos_command.pitch_deg,
                                      talos_first);
  aim_sim_bridge::debug::appendNumber(talos_json, "distance_m",
                                      wall_control.talos_command.distance_m,
                                      talos_first);
  aim_sim_bridge::debug::appendBool(talos_json, "fire_advice",
                                    wall_control.talos_command.fire_advice != 0,
                                    talos_first);
  talos_json << '}';

  const auto &control_metrics = wall_control.metrics;
  std::ostringstream control_json;
  control_json << std::setprecision(10) << '{';
  bool control_first = true;
  aim_sim_bridge::debug::appendBool(control_json, "has_output",
                                    wall_control.has_output, control_first);
  aim_sim_bridge::debug::appendBool(control_json, "following_enabled",
                                    wall_control.following_enabled,
                                    control_first);
  aim_sim_bridge::debug::appendBool(control_json, "talos_command_enabled",
                                    wall_control.talos_command_enabled,
                                    control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "tick_index",
                                    wall_control.tick_index, control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "published_timestamp_ns",
                                    wall_control.published_timestamp_ns,
                                    control_first);
  aim_sim_bridge::debug::appendString(
      control_json, "disposition",
      aim_sim_bridge::control::toString(wall_control.disposition),
      control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "source_producer_epoch",
                                    wall_control.source.epoch, control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "source_image_seq",
                                    wall_control.source.sequence,
                                    control_first);
  aim_sim_bridge::debug::appendBool(control_json, "source_age_available",
                                    wall_control.source_age_available,
                                    control_first);
  aim_sim_bridge::debug::appendNumber(
      control_json, "source_age_ms",
      wall_control.source_age_available
          ? wall_control.source_age_ms
          : std::numeric_limits<double>::quiet_NaN(),
      control_first);
  aim_sim_bridge::debug::appendBool(control_json, "has_target",
                                    wall_control.aim_command.has_target,
                                    control_first);
  aim_sim_bridge::debug::appendNumber(
      control_json, "yaw_deg", wall_control.aim_command.yaw_deg, control_first);
  aim_sim_bridge::debug::appendNumber(control_json, "pitch_deg",
                                      wall_control.aim_command.pitch_deg,
                                      control_first);
  aim_sim_bridge::debug::appendNumber(control_json, "distance_m",
                                      wall_control.aim_command.distance_m,
                                      control_first);
  aim_sim_bridge::debug::appendBool(control_json, "fire_advice",
                                    wall_control.aim_command.fire_advice,
                                    control_first);
  aim_sim_bridge::debug::appendBool(
      control_json, "completed_vision_result",
      wall_control.aim_command.completed_vision_result, control_first);
  aim_sim_bridge::debug::appendUInt(
      control_json, "source_capture_timestamp_ns",
      wall_control.aim_command.source_capture_timestamp_ns, control_first);
  aim_sim_bridge::debug::appendUInt(
      control_json, "vision_completion_timestamp_ns",
      wall_control.aim_command.vision_completion_timestamp_ns, control_first);
  aim_sim_bridge::debug::appendString(
      control_json, "udp_publish_status",
      transportPublishStatusLabel(wall_control.udp_publish), control_first);
  aim_sim_bridge::debug::appendString(
      control_json, "talos_publish_status",
      transportPublishStatusLabel(wall_control.talos_publish), control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "submit_accepted_count",
                                    wall_control.submit_accepted_count,
                                    control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "submit_not_completed_count",
                                    wall_control.submit_not_completed_count,
                                    control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "submit_disabled_count",
                                    wall_control.submit_disabled_count,
                                    control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "submit_duplicate_count",
                                    wall_control.submit_duplicate_count,
                                    control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "submit_rejected_older_count",
                                    wall_control.submit_rejected_older_count,
                                    control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "submit_invalid_count",
                                    wall_control.submit_invalid_count,
                                    control_first);
  aim_sim_bridge::debug::appendUInt(control_json, "clear_count",
                                    wall_control.clear_count, control_first);
  control_json << '}';

  const double udp_success_hz =
      control_metrics.udp_publish_attempt_count > 0
          ? control_metrics.wall_tick_hz *
                static_cast<double>(control_metrics.udp_publish_success_count) /
                static_cast<double>(control_metrics.udp_publish_attempt_count)
          : 0.0;
  const double talos_success_hz =
      control_metrics.talos_publish_attempt_count > 0
          ? control_metrics.wall_tick_hz *
                static_cast<double>(
                    control_metrics.talos_publish_success_count) /
                static_cast<double>(control_metrics.talos_publish_attempt_count)
          : 0.0;

  std::ostringstream frequency_json;
  frequency_json << std::setprecision(10) << '{';
  bool frequency_first = true;
  aim_sim_bridge::debug::appendNumber(frequency_json, "detector_fps",
                                      detector_fps, frequency_first);
  aim_sim_bridge::debug::appendNumber(frequency_json, "bridge_processed_fps",
                                      detector_fps, frequency_first);
  aim_sim_bridge::debug::appendNumber(frequency_json,
                                      "legacy_detector_fps_lifetime_compat",
                                      detector_fps, frequency_first);
  aim_sim_bridge::debug::appendNumber(frequency_json, "bridge_loop_lifetime_hz",
                                      detector_fps, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "unique_ingress_rolling_hz", unique_ingress_rolling_hz,
      frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "unique_ingress_lifetime_hz", unique_ingress_lifetime_hz,
      frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "completed_vision_rolling_hz",
      completed_vision_rolling_hz, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "completed_vision_lifetime_hz",
      completed_vision_lifetime_hz, frequency_first);
  aim_sim_bridge::debug::appendString(
      frequency_json, "command_output_semantics", "wall_scheduler_tick_compat",
      frequency_first);
  aim_sim_bridge::debug::appendNumber(frequency_json, "command_output_hz",
                                      control_metrics.wall_tick_hz,
                                      frequency_first);
  aim_sim_bridge::debug::appendNumber(frequency_json, "udp_command_output_hz",
                                      udp_success_hz, frequency_first);
  aim_sim_bridge::debug::appendNumber(frequency_json, "talos_command_output_hz",
                                      talos_success_hz, frequency_first);
  aim_sim_bridge::debug::appendBool(
      frequency_json, "wall_control_talos_command_enabled",
      wall_control.talos_command_enabled, frequency_first);
  aim_sim_bridge::debug::appendNumber(frequency_json, "wall_control_target_hz",
                                      control_metrics.target_hz,
                                      frequency_first);
  aim_sim_bridge::debug::appendNumber(frequency_json, "wall_control_tick_hz",
                                      control_metrics.wall_tick_hz,
                                      frequency_first);
  aim_sim_bridge::debug::appendUInt(frequency_json, "wall_control_tick_count",
                                    control_metrics.tick_count,
                                    frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_period_sample_count",
      control_metrics.period_sample_count, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_period_mean_ms",
      control_metrics.period_mean_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_period_p50_ms",
      control_metrics.period_p50_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_period_p95_ms",
      control_metrics.period_p95_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_period_p99_ms",
      control_metrics.period_p99_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_period_max_ms",
      control_metrics.period_max_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(frequency_json, "wall_control_max_gap_ms",
                                      control_metrics.period_max_ms,
                                      frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_period_abs_error_p99_ms",
      control_metrics.period_abs_error_p99_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_lateness_p50_ms",
      control_metrics.lateness_p50_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_lateness_p95_ms",
      control_metrics.lateness_p95_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_lateness_p99_ms",
      control_metrics.lateness_p99_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_lateness_max_ms",
      control_metrics.lateness_max_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_execution_p99_ms",
      control_metrics.execution_p99_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_execution_max_ms",
      control_metrics.execution_max_ms, frequency_first);
  aim_sim_bridge::debug::appendBool(
      frequency_json, "wall_control_source_age_available",
      control_metrics.source_age_available, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_source_age_sample_count",
      control_metrics.source_age_sample_count, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_source_age_latest_ms",
      control_metrics.source_age_available
          ? control_metrics.source_age_latest_ms
          : std::numeric_limits<double>::quiet_NaN(),
      frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_source_age_p99_ms",
      control_metrics.source_age_p99_ms, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_source_age_max_ms",
      control_metrics.source_age_max_ms, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_missed_deadline_count",
      control_metrics.missed_deadline_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_overrun_count",
      control_metrics.overrun_count, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_missed_deadline_ratio",
      control_metrics.missed_deadline_ratio, frequency_first);
  aim_sim_bridge::debug::appendNumber(
      frequency_json, "wall_control_overrun_ratio",
      control_metrics.overrun_ratio, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_unavailable_count",
      control_metrics.unavailable_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_no_target_count",
      control_metrics.no_target_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(frequency_json, "wall_control_fresh_count",
                                    control_metrics.fresh_count,
                                    frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_repeated_command_count",
      control_metrics.repeated_command_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(frequency_json, "wall_control_stale_count",
                                    control_metrics.stale_count,
                                    frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_invalid_timestamp_count",
      control_metrics.invalid_timestamp_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_udp_publish_attempt_count",
      control_metrics.udp_publish_attempt_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_udp_publish_success_count",
      control_metrics.udp_publish_success_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_udp_publish_failure_count",
      control_metrics.udp_publish_failure_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_talos_publish_attempt_count",
      control_metrics.talos_publish_attempt_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_talos_publish_success_count",
      control_metrics.talos_publish_success_count, frequency_first);
  aim_sim_bridge::debug::appendUInt(
      frequency_json, "wall_control_talos_publish_failure_count",
      control_metrics.talos_publish_failure_count, frequency_first);
  frequency_json << '}';

  std::ostringstream out;
  out << std::setprecision(10) << '{';
  bool first = true;
  aim_sim_bridge::debug::appendString(out, "source", "talos_bridge", first);
  aim_sim_bridge::debug::appendString(out, "mode",
                                      aim_sim_bridge::toString(mode), first);
  aim_sim_bridge::debug::appendUInt(out, "processed_frames", processed_frames,
                                    first);
  aim_sim_bridge::debug::appendUInt(out, "unique_ingress_frames",
                                    ingress.unique_accepted_frames, first);
  aim_sim_bridge::debug::appendUInt(out, "completed_vision_results",
                                    completed_vision.processed_result_count,
                                    first);
  aim_sim_bridge::debug::appendBool(out, "simulator_following", following,
                                    first);
  aim_sim_bridge::debug::appendUInt(out, "written_timestamp_ns", nowNs(),
                                    first);
  aim_sim_bridge::debug::appendRaw(out, "image", image_json.str(), first);
  aim_sim_bridge::debug::appendRaw(out, "image_transport", transport_json.str(),
                                   first);
  aim_sim_bridge::debug::appendRaw(out, "ingress", ingress_json.str(), first);
  aim_sim_bridge::debug::appendRaw(out, "vision_completion",
                                   completion_json.str(), first);
  aim_sim_bridge::debug::appendRaw(out, "camera", camera_json.str(), first);
  aim_sim_bridge::debug::appendRaw(out, "gimbal", gimbal_json.str(), first);
  const bool ground_truth_exposure_match =
      ground_truth.frame_seq != 0 && ground_truth.frame_seq == image_meta.seq &&
      ground_truth.timestamp_ns == image_meta.timestamp_ns;
  aim_sim_bridge::debug::appendBool(
      out, "ground_truth_exposure_match", ground_truth_exposure_match, first);
  aim_sim_bridge::debug::appendBool(out, "ground_truth_history_exact_match",
                                    ground_truth_exposure_match, first);
  aim_sim_bridge::debug::appendString(
      out, "ground_truth_match_source",
      ground_truth_exposure_match ? "v6_exact_ring" : "missing_fail_closed",
      first);
  aim_sim_bridge::debug::appendUInt(out, "ground_truth_history_slots",
                                    kGroundTruthHistorySlots, first);
  aim_sim_bridge::debug::appendInt(
      out, "ground_truth_frame_seq_delta",
      static_cast<long long>(ground_truth.frame_seq) -
          static_cast<long long>(image_meta.seq),
      first);
  aim_sim_bridge::debug::appendNumber(
      out, "ground_truth_minus_image_ms",
      ground_truth.frame_seq != 0
          ? (static_cast<double>(ground_truth.timestamp_ns) -
             static_cast<double>(image_meta.timestamp_ns)) *
                1e-6
          : std::numeric_limits<double>::quiet_NaN(),
      first);
  aim_sim_bridge::debug::appendRaw(out, "ground_truth",
                                   groundTruthJson(ground_truth), first);
  aim_sim_bridge::debug::appendRaw(out, "pipeline_input", input_json.str(),
                                   first);
  aim_sim_bridge::debug::appendRaw(out, "aim_command", aim_json.str(), first);
  aim_sim_bridge::debug::appendRaw(out, "wall_control_publication",
                                   control_json.str(), first);
  aim_sim_bridge::debug::appendRaw(out, "simulator_udp_command",
                                   simulator_json.str(), first);
  aim_sim_bridge::debug::appendRaw(out, "talos_command", talos_json.str(),
                                   first);
  aim_sim_bridge::debug::appendRaw(out, "frequency", frequency_json.str(),
                                   first);
  out << '}';

  const std::string payload = out.str();
  aim_sim_bridge::debug::writeJsonFile(path, payload);
  aim_sim_bridge::debug::appendJsonLine(jsonl_path, payload);
}

} // namespace

int main(int argc, char **argv) {
  if (hasArg(argc, argv, "--self-test-sequence")) {
    return runSequenceSelectionSelfTest();
  }
  if (hasArg(argc, argv, "--self-test-wall-control")) {
    return runWallControlIntegrationSelfTest();
  }
  if (hasArg(argc, argv, "--self-test-image-transport")) {
    return runImageTransportIntegrationSelfTest();
  }
  if (hasArg(argc, argv, "--self-test-ground-truth")) {
    return runGroundTruthLayoutSelfTest();
  }

  aim_sim_bridge::AimBridgeConfig config;
  config.default_mode =
      aim_sim_bridge::parseTargetMode(argValue(argc, argv, "--mode", "small_buff"),
                                      aim_sim_bridge::TargetMode::SmallBuff);
  config.bullet_speed_mps = argDouble(argc, argv, "--bullet-speed", 22.0);
  config.sim_pitch_neutral_deg =
      argDouble(argc, argv, "--talos-pitch-neutral", 65.0);
  config.enable_fire = argBool(argc, argv, "--enable-fire", true);
  config.publish_no_target = true;
  config.buff_config_path =
      argValue(argc, argv, "--buff-config", "config/buff_config.sim.yaml");

  const std::string param_yaml =
      argValue(argc, argv, "--param-yaml", "config/param.sim.yaml");
  setenv("AIM_SIM_PARAM_YAML", param_yaml.c_str(), 1);

  const bool enable_udp = argBool(argc, argv, "--enable-udp", true);
  const std::string udp_host =
      argValue(argc, argv, "--udp-host",
               envValue("AIM_SIM_UDP_HOST", defaultWindowsUdpHost()));
  const int udp_port = argInt(argc, argv, "--udp-port", 5601);
  const std::string bridge_debug_json =
      aim_sim_bridge::debug::envPath("AIM_SIM_DEBUG_BRIDGE_JSON");
  const std::string bridge_debug_jsonl =
      aim_sim_bridge::debug::envPath("AIM_SIM_DEBUG_BRIDGE_JSONL");

  const std::string image_transport_value =
      argValue(argc, argv, "--image-transport",
               envValue("AIM_SIM_IMAGE_TRANSPORT", "file"));
  const std::optional<ImageTransport> parsed_image_transport =
      parseImageTransport(image_transport_value);
  if (!parsed_image_transport) {
    std::cerr << "aim_sim_talos_bridge invalid image transport '"
              << image_transport_value << "' (expected file|tcp)\n";
    return 2;
  }
  const ImageTransport image_transport = *parsed_image_transport;
  std::string tcp_image_host;
  int tcp_image_port = 5602;
  if (image_transport == ImageTransport::Tcp) {
    tcp_image_host =
        argValue(argc, argv, "--tcp-image-host",
                 envValue("AIM_SIM_TCP_IMAGE_HOST", defaultWindowsUdpHost()));
    const std::string tcp_image_port_value =
        argValue(argc, argv, "--tcp-image-port",
                 envValue("AIM_SIM_TCP_IMAGE_PORT", "5602"));
    const std::optional<int> parsed_tcp_image_port =
        parseIntValue(tcp_image_port_value);
    if (!parsed_tcp_image_port || *parsed_tcp_image_port <= 0 ||
        *parsed_tcp_image_port > 65535) {
      std::cerr << "aim_sim_talos_bridge invalid TCP image port '"
                << tcp_image_port_value << "' (expected integer 1..65535)\n";
      return 2;
    }
    tcp_image_port = *parsed_tcp_image_port;
  }
  const bool enable_talos_command =
      argBool(argc, argv, "--enable-talos-command",
              envBoolValue("AIM_SIM_ENABLE_TALOS_COMMAND",
                           defaultTalosCommandEnabled(image_transport)));

  const std::filesystem::path ipc_dir =
      argValue(argc, argv, "--ipc-dir", "/tmp");
  const auto meta_path = ipc_dir / "talos_ipc_meta";
  const auto pool_path = ipc_dir / "talos_ipc_image_pool";

  std::cerr << "aim_sim_talos_bridge waiting for Talos IPC in "
            << ipc_dir.string() << "\n";

  FileHandle meta_file;
  FileHandle image_pool_file;
  ShmMetaRegion meta_snapshot{};
  std::deque<GroundTruthBatch> ground_truth_history;
  std::deque<ChassisObservation> chassis_observation_history;
  std::deque<PoseMeta> gimbal_pose_history;
  std::deque<PoseMeta> odom_pose_history;
  while (true) {
    bool opened = meta_file.openExisting(meta_path, O_RDWR);
    if (opened && image_transport == ImageTransport::File) {
      opened = image_pool_file.openExisting(pool_path, O_RDONLY);
    }
    const bool valid =
        opened &&
        readFullyAt(meta_file.fd(), &meta_snapshot, sizeof(meta_snapshot), 0) &&
        isValidMetaSnapshot(meta_snapshot);
    if (valid) {
      break;
    }

    meta_file.close();
    image_pool_file.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }

  std::unique_ptr<aim_sim_bridge::tcp_image::Receiver> tcp_image_receiver;
  auto pipeline = aim_sim_bridge::createAimPipeline(config);
  if (image_transport == ImageTransport::Tcp) {
    aim_sim_bridge::tcp_image::ReceiverConfig receiver_config;
    receiver_config.host = tcp_image_host;
    receiver_config.port = static_cast<std::uint16_t>(tcp_image_port);
    tcp_image_receiver =
        std::make_unique<aim_sim_bridge::tcp_image::Receiver>(receiver_config);
    std::string receiver_error;
    if (!tcp_image_receiver->start(&receiver_error)) {
      std::cerr << "aim_sim_talos_bridge failed to start TCP image receiver "
                << tcp_image_host << ':' << tcp_image_port << ": "
                << receiver_error << '\n';
      return 2;
    }
  }

  std::cerr << "aim_sim_talos_bridge started backend="
            << pipeline->backendName()
            << " mode=" << aim_sim_bridge::toString(config.default_mode)
            << " fire=" << (config.enable_fire ? "on" : "off")
            << " image_transport=" << imageTransportLabel(image_transport)
            << " talos_command=" << (enable_talos_command ? "on" : "off")
            << " udp_command=" << (enable_udp ? "on" : "off");
  if (image_transport == ImageTransport::Tcp) {
    std::cerr << " tcp_image=" << tcp_image_host << ':' << tcp_image_port;
  }
  std::cerr << '\n';
  WallCommandPublisher wall_command_publisher(
      meta_path, config, enable_udp, enable_talos_command, udp_host, udp_port);

  std::uint64_t processed = 0;
  ImageSequenceState image_sequence;
  IngressMetrics ingress_metrics;
  CompletedVisionMetrics completed_vision_metrics;
  bool was_following = false;
  bool logged_real_feedback = false;
  bool logged_feedback_fallback = false;
  bool have_last_vivsionn_yaw = false;
  double last_vivsionn_gimbal_yaw_deg = 0.0;
  std::uint64_t last_vivsionn_gimbal_timestamp_ns = 0;
  std::vector<std::uint8_t> rgb_buffer;
  if (image_transport == ImageTransport::File) {
    rgb_buffer.reserve(kMaxImagePayloadBytes);
  }
  ImageTransportTelemetry image_transport_telemetry;
  image_transport_telemetry.transport = image_transport;
  image_transport_telemetry.tcp_host =
      image_transport == ImageTransport::Tcp ? tcp_image_host : "";
  image_transport_telemetry.tcp_port =
      image_transport == ImageTransport::Tcp
          ? static_cast<std::uint16_t>(tcp_image_port)
          : 0;
  const auto report_period = std::chrono::seconds(2);
  const auto bridge_started = std::chrono::steady_clock::now();
  auto last_report = std::chrono::steady_clock::now();
  auto last_debug_write = std::chrono::steady_clock::time_point{};
  ImageMeta telemetry_image_meta{};
  GroundTruthBatch telemetry_ground_truth{};
  bool telemetry_has_feedback = false;
  GimbalFeedback telemetry_feedback{};
  double telemetry_chassis_yaw_deg = std::numeric_limits<double>::quiet_NaN();
  double telemetry_simulator_local_gimbal_yaw_deg =
      std::numeric_limits<double>::quiet_NaN();
  double telemetry_vivsionn_gimbal_yaw_deg =
      std::numeric_limits<double>::quiet_NaN();
  bool telemetry_has_runtime_gimbal = false;
  double telemetry_runtime_gimbal_yaw_deg =
      std::numeric_limits<double>::quiet_NaN();
  double telemetry_runtime_gimbal_pitch_deg =
      std::numeric_limits<double>::quiet_NaN();
  const char *telemetry_gimbal_feedback_source = "unavailable";
  aim_sim_bridge::SimFrame telemetry_frame;
  aim_sim_bridge::AimCommand telemetry_aim;

  const auto write_telemetry_if_due =
      [&](std::chrono::steady_clock::time_point now, bool following) {
        if ((bridge_debug_json.empty() && bridge_debug_jsonl.empty()) ||
            (bridge_debug_jsonl.empty() &&
             last_debug_write.time_since_epoch().count() != 0 &&
             now - last_debug_write < std::chrono::milliseconds(100))) {
          return;
        }

        const double elapsed_s =
            std::chrono::duration<double>(now - bridge_started).count();
        const double detector_fps =
            elapsed_s > 1e-6 ? static_cast<double>(processed) / elapsed_s : 0.0;
        const double unique_ingress_rolling_hz = ingress_metrics.rollingHz(now);
        const double unique_ingress_lifetime_hz =
            elapsed_s > 1e-6
                ? static_cast<double>(ingress_metrics.unique_accepted_frames) /
                      elapsed_s
                : 0.0;
        const double completed_vision_rolling_hz =
            completed_vision_metrics.rollingHz(now);
        const double completed_vision_lifetime_hz =
            elapsed_s > 1e-6
                ? static_cast<double>(
                      completed_vision_metrics.processed_result_count) /
                      elapsed_s
                : 0.0;
        if (tcp_image_receiver) {
          image_transport_telemetry.tcp_receiver_running =
              tcp_image_receiver->running();
          image_transport_telemetry.tcp_receiver =
              tcp_image_receiver->counters();
        }
        const WallCommandPublisherSnapshot wall_control =
            wall_command_publisher.snapshot();
        writeBridgeTelemetry(
            bridge_debug_json, bridge_debug_jsonl, config.default_mode,
            processed, image_sequence,
            ingress_metrics, unique_ingress_rolling_hz,
            unique_ingress_lifetime_hz, completed_vision_metrics,
            completed_vision_rolling_hz, completed_vision_lifetime_hz,
            following, telemetry_image_meta, meta_snapshot.camera_info,
            meta_snapshot.runtime_state, telemetry_ground_truth,
            telemetry_has_feedback, telemetry_feedback,
            telemetry_chassis_yaw_deg, telemetry_simulator_local_gimbal_yaw_deg,
            telemetry_vivsionn_gimbal_yaw_deg, telemetry_has_runtime_gimbal,
            telemetry_runtime_gimbal_yaw_deg,
            telemetry_runtime_gimbal_pitch_deg,
            telemetry_gimbal_feedback_source, telemetry_frame, telemetry_aim,
            wall_control, image_transport_telemetry, detector_fps, elapsed_s);
        last_debug_write = now;
      };

  while (true) {
    if (!readFullyAt(meta_file.fd(), &meta_snapshot, sizeof(meta_snapshot),
                     0)) {
      if (image_transport == ImageTransport::Tcp) {
        write_telemetry_if_due(std::chrono::steady_clock::now(), was_following);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    if (!isValidMetaSnapshot(meta_snapshot)) {
      if (image_transport == ImageTransport::Tcp) {
        write_telemetry_if_due(std::chrono::steady_clock::now(), was_following);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    const auto remember_ground_truth = [&](const GroundTruthBatch &candidate) {
      if (candidate.frame_seq == 0)
        return;
      if (ground_truth_history.empty() ||
          ground_truth_history.back().frame_seq != candidate.frame_seq) {
        ground_truth_history.push_back(candidate);
        while (ground_truth_history.size() > 16)
          ground_truth_history.pop_front();
      }
    };
    const auto remember_chassis = [&](const ChassisObservation &candidate) {
      if (candidate.frame_seq == 0)
        return;
      if (chassis_observation_history.empty() ||
          chassis_observation_history.back().frame_seq != candidate.frame_seq) {
        chassis_observation_history.push_back(candidate);
        while (chassis_observation_history.size() > 16)
          chassis_observation_history.pop_front();
      }
    };
    remember_ground_truth(meta_snapshot.ground_truth);
    remember_chassis(meta_snapshot.chassis_observation);
    const auto remember_pose_buffer = [](const PoseTripleBuffer &buffer,
                                         std::deque<PoseMeta> *history) {
      if (history == nullptr)
        return;
      for (const PoseMeta &candidate : buffer.slots) {
        if (candidate.frame_seq == 0 || candidate.timestamp_ns == 0)
          continue;
        const bool already_present = std::any_of(
            history->begin(), history->end(), [&](const PoseMeta &stored) {
              return stored.frame_seq == candidate.frame_seq &&
                     stored.timestamp_ns == candidate.timestamp_ns;
            });
        if (!already_present)
          history->push_back(candidate);
      }
      while (history->size() > 64)
        history->pop_front();
    };
    remember_pose_buffer(meta_snapshot.poses[kGimbalPoseIndex],
                         &gimbal_pose_history);
    remember_pose_buffer(meta_snapshot.poses[kOdomPoseIndex],
                         &odom_pose_history);

    const EpochObservation epoch_observation =
        image_sequence.observe(meta_snapshot.header.created_ns);
    completed_vision_metrics.observeProducerEpoch(
        image_sequence.producer_epoch);
    if (epoch_observation == EpochObservation::Restart) {
      ++ingress_metrics.producer_epoch_changes;
      wall_command_publisher.clearForProducerEpochRestart();
    }

    const bool following = meta_snapshot.runtime_state.following != 0;
    if (!following) {
      if (was_following) {
        std::cerr << "aim_sim_talos_bridge paused: simulator auto-aim is OFF\n";
        wall_command_publisher.setFollowing(false);
        was_following = false;
      }
      if (image_transport == ImageTransport::Tcp) {
        write_telemetry_if_due(std::chrono::steady_clock::now(), false);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }
    if (!was_following) {
      std::cerr << "aim_sim_talos_bridge active: simulator auto-aim is ON\n";
      wall_command_publisher.setFollowing(true);
      was_following = true;
    }

    ImageMeta image_meta{};
    cv::Mat bgr;
    if (image_transport == ImageTransport::File) {
      const ImageSelection selection =
          selectLatestImageStrict(meta_snapshot, image_sequence);
      if (selection.kind != ImageSelectionKind::Candidate) {
        if (selection.kind == ImageSelectionKind::Duplicate) {
          ++ingress_metrics.duplicate_rejects;
        } else if (selection.kind == ImageSelectionKind::Regression) {
          ++ingress_metrics.regression_rejects;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      image_meta = selection.image;

      std::size_t payload_bytes = 0;
      if (!checkedImagePayloadBytes(image_meta, &payload_bytes)) {
        ++ingress_metrics.inconsistent_image_reads;
        continue;
      }
      rgb_buffer.resize(payload_bytes);
      const std::size_t offset = imageSlotOffset(image_meta.buffer_id);
      if (!readFullyAt(image_pool_file.fd(), rgb_buffer.data(), payload_bytes,
                       offset)) {
        ++ingress_metrics.image_read_failures;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      ShmMetaRegion verified_meta{};
      if (!readFullyAt(meta_file.fd(), &verified_meta, sizeof(verified_meta),
                       0) ||
          !imageReadStillConsistent(
              verified_meta, image_sequence.producer_epoch, image_meta)) {
        ++ingress_metrics.inconsistent_image_reads;
        continue;
      }
      if (image_sequence.has_last_accepted &&
          image_meta.seq > image_sequence.last_accepted_seq + 1) {
        ingress_metrics.sequence_gap_frames +=
            image_meta.seq - image_sequence.last_accepted_seq - 1;
      }
      image_sequence.commit(image_meta.seq);
      ingress_metrics.recordAccepted(std::chrono::steady_clock::now());

      const cv::Mat rgb(static_cast<int>(image_meta.height),
                        static_cast<int>(image_meta.width), CV_8UC3,
                        rgb_buffer.data());
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    } else {
      aim_sim_bridge::tcp_image::Frame tcp_frame;
      if (!tcp_image_receiver ||
          !tcp_image_receiver->tryTakeLatest(&tcp_frame)) {
        write_telemetry_if_due(std::chrono::steady_clock::now(), following);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      image_transport_telemetry.tcp_bridge.recordTaken();
      const TcpImageSelectionKind tcp_selection = selectTcpImageStrict(
          tcp_frame, image_sequence, meta_snapshot.header.created_ns);
      if (tcp_selection != TcpImageSelectionKind::Candidate) {
        switch (tcp_selection) {
        case TcpImageSelectionKind::EpochMismatch:
          ++image_transport_telemetry.tcp_bridge.epoch_mismatch_rejects;
          break;
        case TcpImageSelectionKind::Duplicate:
          ++image_transport_telemetry.tcp_bridge.duplicate_rejects;
          ++ingress_metrics.duplicate_rejects;
          break;
        case TcpImageSelectionKind::Regression:
          ++image_transport_telemetry.tcp_bridge.regression_rejects;
          ++ingress_metrics.regression_rejects;
          break;
        case TcpImageSelectionKind::InvalidFrame:
          ++image_transport_telemetry.tcp_bridge.invalid_frame_rejects;
          break;
        case TcpImageSelectionKind::Candidate:
          break;
        }
        write_telemetry_if_due(std::chrono::steady_clock::now(), following);
        continue;
      }
      if (!tcpFrameToBgr(tcp_frame, &bgr)) {
        ++image_transport_telemetry.tcp_bridge.color_conversion_failures;
        write_telemetry_if_due(std::chrono::steady_clock::now(), following);
        continue;
      }
      image_meta = tcpFrameImageMeta(tcp_frame);
      if (image_sequence.has_last_accepted &&
          image_meta.seq > image_sequence.last_accepted_seq + 1) {
        ingress_metrics.sequence_gap_frames +=
            image_meta.seq - image_sequence.last_accepted_seq - 1;
      }
      image_sequence.commit(image_meta.seq);
      ingress_metrics.recordAccepted(std::chrono::steady_clock::now());
      image_transport_telemetry.tcp_bridge.recordSelected(tcp_frame);
    }

    GroundTruthBatch exposure_ground_truth{};
    ExactExposureTruth exact_exposure{};
    const bool has_exact_exposure_truth = readExactExposureTruth(
        meta_file.fd(), image_meta.seq, image_meta.timestamp_ns,
        &exact_exposure);
    if (has_exact_exposure_truth)
      exposure_ground_truth = exact_exposure.ground_truth;

    ChassisObservation exposure_chassis{};
    const bool has_exposure_chassis =
        has_exact_exposure_truth &&
        (exact_exposure.exposure_state.state_flags &
         kExposureStateHasChassisWorldPose) != 0;
    if (has_exposure_chassis) {
      exposure_chassis.frame_seq = exact_exposure.exposure_state.frame_seq;
      exposure_chassis.timestamp_ns =
          exact_exposure.exposure_state.timestamp_ns;
      std::copy_n(exact_exposure.exposure_state.chassis_rpy_world, 3,
                  exposure_chassis.rpy_rad);
    }

    GimbalFeedback feedback{};
    PoseMeta exposure_gimbal_pose{};
    const bool has_ring_gimbal_pose =
        has_exact_exposure_truth &&
        (exact_exposure.exposure_state.state_flags &
         kExposureStateHasGimbalWorldPose) != 0;
    if (has_ring_gimbal_pose) {
      exposure_gimbal_pose.frame_seq = exact_exposure.exposure_state.frame_seq;
      exposure_gimbal_pose.timestamp_ns =
          exact_exposure.exposure_state.timestamp_ns;
      std::copy_n(exact_exposure.exposure_state.gimbal_position_world, 3,
                  exposure_gimbal_pose.position);
      std::copy_n(
          exact_exposure.exposure_state.gimbal_quaternion_world_wxyz, 4,
          exposure_gimbal_pose.quaternion);
    }
    const bool has_exposure_feedback =
        has_ring_gimbal_pose &&
        poseToGimbalFeedback(exposure_gimbal_pose, &feedback);
    const bool has_feedback = has_exposure_feedback || selectLatestGimbalFeedback(
        meta_snapshot.poses[kGimbalPoseIndex], &feedback);
    const ChassisObservation &chassis_for_frame =
        has_exposure_chassis ? exposure_chassis : meta_snapshot.chassis_observation;
    const double chassis_yaw_deg =
        static_cast<double>(chassis_for_frame.rpy_rad[2]) * kRadToDeg;
    const bool has_runtime_gimbal =
        meta_snapshot.runtime_state.timestamp_ns != 0 &&
        std::isfinite(meta_snapshot.runtime_state.gimbal_yaw_rad) &&
        std::isfinite(meta_snapshot.runtime_state.gimbal_pitch_rad);
    const double runtime_gimbal_yaw_deg =
        has_runtime_gimbal
            ? static_cast<double>(meta_snapshot.runtime_state.gimbal_yaw_rad) *
                  kRadToDeg
            : std::numeric_limits<double>::quiet_NaN();
    const double runtime_gimbal_pitch_deg =
        has_runtime_gimbal ? static_cast<double>(
                                 meta_snapshot.runtime_state.gimbal_pitch_rad) *
                                 kRadToDeg
                           : std::numeric_limits<double>::quiet_NaN();
    const double quaternion_local_gimbal_yaw_deg =
        has_feedback ? normalizeAngleDeg(feedback.yaw_deg - chassis_yaw_deg)
                     : 0.0;
    const bool has_exposure_pose = has_exposure_feedback && has_exposure_chassis;
    PoseMeta exposure_odom_pose{};
    const bool has_exposure_odom = has_ring_gimbal_pose;
    if (has_exposure_odom)
      exposure_odom_pose = exposure_gimbal_pose;
    // The solver needs the exposure-time optical pose, not the raw local
    // gimbal joint. The fixed camera mount makes those angles different.
    const double simulator_local_gimbal_yaw_deg =
        has_exposure_pose
            ? quaternion_local_gimbal_yaw_deg
            : (has_runtime_gimbal ? runtime_gimbal_yaw_deg
                                  : quaternion_local_gimbal_yaw_deg);
    const double vivsionn_gimbal_yaw_deg = -simulator_local_gimbal_yaw_deg;
    const std::uint64_t gimbal_timestamp_ns =
        has_exposure_pose
            ? feedback.timestamp_ns
            : (has_runtime_gimbal ? meta_snapshot.runtime_state.timestamp_ns
                                  : feedback.timestamp_ns);
    double vivsionn_gimbal_yaw_speed_deg_s = 0.0;
    if (have_last_vivsionn_yaw &&
        gimbal_timestamp_ns > last_vivsionn_gimbal_timestamp_ns) {
      const double dt_s =
          static_cast<double>(gimbal_timestamp_ns -
                              last_vivsionn_gimbal_timestamp_ns) *
          1e-9;
      if (dt_s > 1e-6 && dt_s < 0.5) {
        vivsionn_gimbal_yaw_speed_deg_s =
            normalizeAngleDeg(vivsionn_gimbal_yaw_deg -
                              last_vivsionn_gimbal_yaw_deg) /
            dt_s;
      }
    }
    if (gimbal_timestamp_ns != 0) {
      have_last_vivsionn_yaw = true;
      last_vivsionn_gimbal_yaw_deg = vivsionn_gimbal_yaw_deg;
      last_vivsionn_gimbal_timestamp_ns = gimbal_timestamp_ns;
    }
    const double local_gimbal_pitch_deg =
        has_exposure_pose
            ? feedback.pitch_deg
            : (has_runtime_gimbal
                   ? runtime_gimbal_pitch_deg
                   : (has_feedback ? feedback.pitch_deg : 0.0));
    const char *gimbal_feedback_source =
        has_exposure_pose
            ? "exposure_matched_optical_pose"
            : (has_runtime_gimbal
                   ? "runtime_state_local_joint_fallback"
                   : (has_feedback ? "talos_quaternion_latest_fallback"
                                   : "zero_fallback"));
    if (has_feedback && !logged_real_feedback) {
      std::cerr << "aim_sim_talos_bridge using Talos gimbal pose feedback"
                << " frame=" << feedback.frame_seq
                << " global_yaw=" << feedback.yaw_deg
                << " chassis_yaw=" << chassis_yaw_deg
                << " quat_local_yaw=" << quaternion_local_gimbal_yaw_deg
                << " quat_pitch=" << feedback.pitch_deg
                << " runtime_local_yaw=" << runtime_gimbal_yaw_deg
                << " runtime_pitch=" << runtime_gimbal_pitch_deg
                << " vivsionn_yaw=" << vivsionn_gimbal_yaw_deg
                << " source=" << gimbal_feedback_source << "\n";
      logged_real_feedback = true;
    } else if (!has_feedback && !logged_feedback_fallback) {
      std::cerr << "aim_sim_talos_bridge warning: Talos gimbal pose feedback "
                   "missing; "
                << "using zero yaw/pitch fallback\n";
      logged_feedback_fallback = true;
    }

    aim_sim_bridge::SimFrame frame;
    frame.bgr_image = std::move(bgr);
    frame.source_producer_epoch = image_sequence.producer_epoch;
    frame.source_image_seq = image_meta.seq;
    frame.source_capture_timestamp_ns = image_meta.timestamp_ns;
    frame.gimbal_pose_timestamp_ns = gimbal_timestamp_ns;
    frame.gimbal_pose_exposure_matched =
        has_exposure_pose && gimbal_timestamp_ns == image_meta.timestamp_ns;
    frame.tracker_world_transform_exposure_matched =
        frame.gimbal_pose_exposure_matched && has_exposure_chassis &&
        has_exposure_odom;
    if (has_exposure_odom) {
      frame.tracker_origin_world_ros_m = cv::Vec3d(
          exposure_odom_pose.position[0], exposure_odom_pose.position[1],
          exposure_odom_pose.position[2]);
      frame.tracker_gimbal_quaternion_world_wxyz = cv::Vec4d(
          exposure_odom_pose.quaternion[0], exposure_odom_pose.quaternion[1],
          exposure_odom_pose.quaternion[2], exposure_odom_pose.quaternion[3]);
    }
    if (has_exposure_chassis) {
      frame.tracker_frame_rpy_world_ros_rad = cv::Vec3d(
          exposure_chassis.rpy_rad[0], exposure_chassis.rpy_rad[1],
          exposure_chassis.rpy_rad[2]);
    }
    if (has_exact_exposure_truth &&
        (exact_exposure.exposure_state.state_flags &
         kExposureStateHasCameraWorldPose) != 0) {
      frame.camera_origin_world_ros_m = cv::Vec3d(
          exact_exposure.exposure_state.camera_position_world[0],
          exact_exposure.exposure_state.camera_position_world[1],
          exact_exposure.exposure_state.camera_position_world[2]);
      frame.camera_quaternion_world_wxyz = cv::Vec4d(
          exact_exposure.exposure_state.camera_quaternion_world_wxyz[0],
          exact_exposure.exposure_state.camera_quaternion_world_wxyz[1],
          exact_exposure.exposure_state.camera_quaternion_world_wxyz[2],
          exact_exposure.exposure_state.camera_quaternion_world_wxyz[3]);
    }
    frame.timestamp_ms = static_cast<double>(image_meta.timestamp_ns) * 1e-6;
    frame.gimbal_yaw_deg = vivsionn_gimbal_yaw_deg;
    frame.gimbal_pitch_deg = local_gimbal_pitch_deg;
    frame.gimbal_yaw_speed_deg_s = vivsionn_gimbal_yaw_speed_deg_s;
    if (has_runtime_gimbal &&
        meta_snapshot.runtime_state.timestamp_ns > image_meta.timestamp_ns) {
      frame.simulator_state_age_s = std::clamp(
          (static_cast<double>(meta_snapshot.runtime_state.timestamp_ns) -
           static_cast<double>(image_meta.timestamp_ns)) *
              1e-9,
          0.0, 0.2);
    } else {
      frame.simulator_state_age_s = 0.0;
    }
    frame.bullet_speed_mps = config.bullet_speed_mps;
    frame.target_mode = config.default_mode;
    const CameraInfo &camera_info = meta_snapshot.camera_info;
    if (std::isfinite(camera_info.fx) && std::isfinite(camera_info.fy) &&
        std::isfinite(camera_info.cx) && std::isfinite(camera_info.cy) &&
        camera_info.fx > 0.0 && camera_info.fy > 0.0) {
      frame.camera_matrix_override =
          (cv::Mat_<double>(3, 3) << camera_info.fx, 0.0, camera_info.cx, 0.0,
           camera_info.fy, camera_info.cy, 0.0, 0.0, 1.0);
      frame.has_camera_matrix_override = true;
    }

    // A captured image and a live local gimbal joint are not a valid pose pair.
    // The simulator camera has a fixed optical mount offset, so substituting the
    // runtime joint pitch for an older exposure can create a large false target
    // height. Fail closed and retain the last valid command until an exact
    // exposure-time optical pose is available.
    if (!frame.gimbal_pose_exposure_matched) {
      static std::uint64_t unmatched_exposure_pose_drop_count = 0;
      ++unmatched_exposure_pose_drop_count;
      if (unmatched_exposure_pose_drop_count <= 3 ||
          (unmatched_exposure_pose_drop_count % 100) == 0) {
        std::cerr << "aim_sim_talos_bridge dropping frame without exposure-"
                     "matched optical pose"
                  << " seq=" << image_meta.seq
                  << " capture_ns=" << image_meta.timestamp_ns
                  << " pose_ns=" << gimbal_timestamp_ns
                  << " count=" << unmatched_exposure_pose_drop_count << "\n";
      }
      continue;
    }

    aim_sim_bridge::AimCommand aim = pipeline->process(frame);
    completed_vision_metrics.record(aim, std::chrono::steady_clock::now());
    const WallCommandSubmitStatus submit_status =
        wall_command_publisher.submitCompleted(aim);
    (void)submit_status;

    telemetry_image_meta = image_meta;
    telemetry_ground_truth = exposure_ground_truth;
    telemetry_has_feedback = has_feedback;
    telemetry_feedback = feedback;
    telemetry_chassis_yaw_deg = chassis_yaw_deg;
    telemetry_simulator_local_gimbal_yaw_deg = simulator_local_gimbal_yaw_deg;
    telemetry_vivsionn_gimbal_yaw_deg = vivsionn_gimbal_yaw_deg;
    telemetry_has_runtime_gimbal = has_runtime_gimbal;
    telemetry_runtime_gimbal_yaw_deg = runtime_gimbal_yaw_deg;
    telemetry_runtime_gimbal_pitch_deg = runtime_gimbal_pitch_deg;
    telemetry_gimbal_feedback_source = gimbal_feedback_source;
    telemetry_frame = frame;
    telemetry_frame.bgr_image.release();
    telemetry_aim = aim;

    ++processed;
    const auto now = std::chrono::steady_clock::now();
    write_telemetry_if_due(now, following);

    if (now - last_report >= report_period) {
      std::cerr << "aim_sim_talos_bridge frames=" << processed
                << " target=" << (aim.has_target ? "yes" : "no")
                << " fire=" << (aim.fire_advice ? "yes" : "no")
                << " yaw=" << aim.yaw_deg << " pitch=" << aim.pitch_deg
                << " dist=" << aim.distance_m << "\n";
      last_report = now;
    }
  }
}
