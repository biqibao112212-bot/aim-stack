#include <daedalus_sim_sdk/scene_control_client.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

namespace {

using daedalus::sim::sdk::v1::ClientResult;
using daedalus::sim::sdk::v1::RangeMotionMode;
using daedalus::sim::sdk::v1::RangeTargetMotion;
using daedalus::sim::sdk::v1::SceneControlClient;
using daedalus::sim::sdk::v1::SceneControlOptions;
using daedalus::sim::sdk::v1::SceneControlResponse;
using daedalus::sim::sdk::v1::SceneControlStatus;
using daedalus::sim::sdk::v1::SceneMode;

std::string envOr(const char *name, const std::string &fallback) {
  const char *value = std::getenv(name);
  return value == nullptr || *value == '\0' ? fallback : value;
}

float envFloat(const char *name, float fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char *end = nullptr;
  const float parsed = std::strtof(value, &end);
  return end != value && *end == '\0' && std::isfinite(parsed) ? parsed : fallback;
}

std::uint8_t envTarget(const char *name, std::uint8_t fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end != value && *end == '\0' && (parsed == 1 || parsed == 3)
             ? static_cast<std::uint8_t>(parsed)
             : fallback;
}

void argumentValue(int argc, char **argv, const std::string &name,
                   std::string *value) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == name) {
      *value = argv[i + 1];
      return;
    }
  }
}

bool responseOk(const char *operation,
                const ClientResult<SceneControlResponse> &result) {
  if (!result.ok()) {
    std::cerr << "scene_control operation=" << operation
              << " status=error detail=" << result.status.message << '\n';
    return false;
  }
  const auto &response = *result.value;
  if (response.status != SceneControlStatus::Ok) {
    std::cerr << "scene_control operation=" << operation
              << " status=not_ok message=" << response.message << '\n';
    return false;
  }
  std::cout << "{\"operation\":\"" << operation
            << "\",\"status\":\"ok\",\"command_id\":"
            << response.command_id << ",\"applied_frame_seq\":"
            << response.applied_frame_seq << ",\"timestamp_ns\":"
            << response.timestamp_ns << "}\n";
  return true;
}

template <typename Request>
ClientResult<SceneControlResponse> retryRequest(const Request &request) {
  ClientResult<SceneControlResponse> last;
  for (int attempt = 0; attempt < 5; ++attempt) {
    last = request();
    if (last.ok() && last.value->status == SceneControlStatus::Ok) {
      return last;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  return last;
}

}  // namespace

int main(int argc, char **argv) {
  std::string host = envOr("AIM_SIM_SCENE_CONTROL_HOST", "127.0.0.1");
  std::string session = envOr("AIM_SIM_SCENE_CONTROL_SESSION", "aim-b-g2");
  const std::uint8_t target = envTarget("AIM_SIM_RANGE_TARGET_NUMBER", 3);
  const float spin_deg_s = std::max(0.0F, envFloat("AIM_SIM_RANGE_SPIN_DEG_S", 30.0F));
  argumentValue(argc, argv, "--host", &host);
  argumentValue(argc, argv, "--session", &session);

  SceneControlOptions options;
  options.endpoint.host = host;
  options.endpoint.port = 5603;
  options.session_id = session;
  options.timeout = std::chrono::milliseconds(300);
  SceneControlClient control(options);

  if (!responseOk("create_session",
                  retryRequest([&control] { return control.createSession(); })) ||
      !responseOk("set_scene", retryRequest([&control] {
        return control.setScene(SceneMode::ShootingRange);
      }))) {
    return 2;
  }

  RangeTargetMotion motion;
  motion.target = target;
  motion.mode = RangeMotionMode::Spin;
  motion.direction_deg = 90.0F;
  motion.linear_speed_mps = 0.0F;
  motion.linear_span_m = 0.0F;
  motion.spin_deg_s = spin_deg_s;
  const std::string operation = target == 1 ? "set_target_1_spin" : "set_target_3_spin";
  if (!responseOk(operation.c_str(), retryRequest([&control, motion] {
    return control.setRangeTargetMotion(motion);
  }))) {
    return 2;
  }

  std::cout << "scene_control_g2_ready host=" << host
            << " session=" << session << " target=" << static_cast<int>(target)
            << " mode=spin spin_deg_s=" << spin_deg_s << "\n";
  return 0;
}
