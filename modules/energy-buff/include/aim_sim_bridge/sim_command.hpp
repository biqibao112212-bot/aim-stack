#pragma once

#include "aim_sim_bridge/aim_types.hpp"

namespace aim_sim_bridge
{

struct SimulatorCommandFields
{
    double yaw_deg = 0.0;
    double pitch_deg = 90.0;
    double distance_m = -1.0;
    bool fire_advice = false;
};

SimulatorCommandFields toSimulatorCommand(
    const AimCommand& command, const AimBridgeConfig& config);

}  // namespace aim_sim_bridge

