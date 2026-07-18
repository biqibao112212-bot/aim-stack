#include "aim_sim_bridge/sim_command.hpp"

#include <algorithm>
#include <cctype>

namespace aim_sim_bridge
{

const char* toString(TargetMode mode)
{
    switch (mode) {
    case TargetMode::Armor:
        return "armor";
    case TargetMode::Outpost:
        return "outpost";
    }
    return "armor";
}

TargetMode parseTargetMode(const std::string& value, TargetMode fallback)
{
    std::string key = value;
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (key == "armor" || key == "auto_shot" || key == "robot") return TargetMode::Armor;
    if (key == "outpost" || key == "hit_outpost") return TargetMode::Outpost;
    return fallback;
}

SimulatorCommandFields toSimulatorCommand(
    const AimCommand& command, const AimBridgeConfig& config)
{
    SimulatorCommandFields fields;
    if (!command.has_target) {
        fields.distance_m = -1.0;
        fields.fire_advice = false;
        return fields;
    }

    // Daedalus local gimbal yaw is opposite to the vivsionn yaw convention.
    fields.yaw_deg = -command.yaw_deg;
    // FireControl pitch is optical elevation. Positive optical elevation
    // requires a more-negative simulator joint angle because of the fixed
    // camera mount: encoded = optical_zero - optical_pitch.
    fields.pitch_deg = config.sim_pitch_neutral_deg - command.pitch_deg;
    fields.distance_m = command.distance_m > 0.0 ? command.distance_m : 1.0;
    fields.fire_advice = config.enable_fire && command.fire_advice;
    return fields;
}

}  // namespace aim_sim_bridge
