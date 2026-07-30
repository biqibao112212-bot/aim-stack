#include <daedalus_sim_sdk/scene_control_client.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_set>

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

bool hasArgument(int argc, char **argv, const std::string &name) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == name) return true;
  return false;
}

int argumentCount(int argc, char **argv, const std::string &name) {
  int count = 0;
  for (int i = 1; i < argc; ++i)
    count += std::string(argv[i]) == name ? 1 : 0;
  return count;
}

bool strictArgument(int argc, char **argv, const std::string &name,
                    std::string *value, std::string *error) {
  int matches = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) != name) continue;
    ++matches;
    if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
      *error = "missing value for " + name;
      return false;
    }
    *value = argv[i + 1];
  }
  if (matches != 1) {
    *error = matches == 0 ? "missing required " + name : "duplicate " + name;
    return false;
  }
  return true;
}

bool strictDouble(const std::string &raw, double *value) {
  char *end = nullptr;
  const double parsed = std::strtod(raw.c_str(), &end);
  if (end == raw.c_str() || *end != '\0' || !std::isfinite(parsed)) return false;
  *value = parsed;
  return true;
}

bool parseStage3Args(int argc, char **argv, std::string *host,
                    std::string *session, RangeTargetMotion *motion,
                    std::string *error) {
  const std::unordered_set<std::string> allowed = {
      "--stage3", "--stage3-update", "--host", "--session", "--target", "--mode",
      "--direction-deg", "--linear-speed-mps", "--linear-span-m",
      "--spin-deg-s"};
  const int initialize_count = argumentCount(argc, argv, "--stage3");
  const int update_count = argumentCount(argc, argv, "--stage3-update");
  if (initialize_count + update_count != 1) {
    *error = "exactly one of --stage3 or --stage3-update is required";
    return false;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (allowed.count(arg) == 0) {
      *error = "unknown Stage3 option " + arg;
      return false;
    }
    if (arg != "--stage3" && arg != "--stage3-update") {
      if (++i >= argc || std::string(argv[i]).rfind("--", 0) == 0) {
        *error = "missing value for " + arg;
        return false;
      }
    }
  }
  std::string raw;
  if (!strictArgument(argc, argv, "--host", host, error) ||
      !strictArgument(argc, argv, "--session", session, error) ||
      !strictArgument(argc, argv, "--target", &raw, error)) return false;
  if (host->empty() || session->empty()) {
    *error = "--host and --session must be non-empty";
    return false;
  }
  if (raw != "3") {
    *error = "Stage3 only controls target 3";
    return false;
  }
  if (!strictArgument(argc, argv, "--mode", &raw, error)) return false;
  if (raw == "stationary") motion->mode = RangeMotionMode::Stationary;
  else if (raw == "linear") motion->mode = RangeMotionMode::Linear;
  else if (raw == "spin") motion->mode = RangeMotionMode::Spin;
  else if (raw == "linear_and_spin") motion->mode = RangeMotionMode::LinearAndSpin;
  else { *error = "invalid Stage3 mode " + raw; return false; }
  double value = 0.0;
  if (!strictArgument(argc, argv, "--direction-deg", &raw, error) ||
      !strictDouble(raw, &value) || value < -360.0 || value > 360.0) {
    *error = "invalid --direction-deg"; return false;
  }
  motion->direction_deg = static_cast<float>(value);
  if (!strictArgument(argc, argv, "--linear-speed-mps", &raw, error) ||
      !strictDouble(raw, &value) || value < 0.0 || value > 3.0) {
    *error = "invalid --linear-speed-mps (expected 0..3)"; return false;
  }
  motion->linear_speed_mps = static_cast<float>(value);
  if (!strictArgument(argc, argv, "--linear-span-m", &raw, error) ||
      !strictDouble(raw, &value) || value < 0.0 || value > 8.0) {
    *error = "invalid --linear-span-m (expected 0..8)"; return false;
  }
  motion->linear_span_m = static_cast<float>(value);
  if (!strictArgument(argc, argv, "--spin-deg-s", &raw, error) ||
      !strictDouble(raw, &value) || std::abs(value) > 15.0 * 180.0 / 3.14159265358979323846) {
    *error = "invalid --spin-deg-s (expected abs <= 15 rad/s)"; return false;
  }
  motion->spin_deg_s = static_cast<float>(value);
  motion->target = 3;
  const bool has_linear = motion->linear_speed_mps > 0.0F ||
                          motion->linear_span_m > 0.0F;
  const bool has_spin = std::abs(motion->spin_deg_s) > 0.0F;
  const bool mode_consistent =
      (motion->mode == RangeMotionMode::Stationary && !has_linear && !has_spin) ||
      (motion->mode == RangeMotionMode::Linear &&
       motion->linear_speed_mps > 0.0F && motion->linear_span_m > 0.0F && !has_spin) ||
      (motion->mode == RangeMotionMode::Spin && !has_linear && has_spin) ||
      (motion->mode == RangeMotionMode::LinearAndSpin &&
       motion->linear_speed_mps > 0.0F && motion->linear_span_m > 0.0F && has_spin);
  if (!mode_consistent) {
    *error = "motion parameters are inconsistent with --mode";
    return false;
  }
  return true;
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
  const bool stage3_initialize = hasArgument(argc, argv, "--stage3");
  const bool stage3_update = hasArgument(argc, argv, "--stage3-update");
  if (stage3_initialize || stage3_update) {
    std::string host = envOr("AIM_SIM_SCENE_CONTROL_HOST", "127.0.0.1");
    std::string session = envOr("AIM_SIM_SCENE_CONTROL_SESSION", "stage3");
    RangeTargetMotion motion;
    std::string error;
    if (!parseStage3Args(argc, argv, &host, &session, &motion, &error)) {
      std::cerr << "scene_control Stage3 argument error: " << error << '\n';
      return 2;
    }
    SceneControlOptions options;
    options.endpoint.host = host;
    options.endpoint.port = 5603;
    options.session_id = session;
    options.timeout = std::chrono::milliseconds(300);
    SceneControlClient control(options);
    if (stage3_initialize) {
      if (!responseOk("create_session", retryRequest([&control] { return control.createSession(); })) ||
          !responseOk("set_scene", retryRequest([&control] { return control.setScene(SceneMode::ShootingRange); }))) {
        return 2;
      }
    }
    if (!responseOk("set_target_3_motion",
                    retryRequest([&control, motion] {
                      return control.setRangeTargetMotion(motion);
                    }))) {
      return 2;
    }
    std::cout << (stage3_initialize ? "scene_control_stage3_ready" :
                                      "scene_control_stage3_updated")
              << " host=" << host << " session=" << session << " target=3\n";
    return 0;
  }
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
