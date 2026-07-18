#pragma once

#include "aim_sim_bridge/aim_types.hpp"

#include <memory>
#include <string>

namespace aim_sim_bridge
{

class IAimPipeline
{
public:
    virtual ~IAimPipeline() = default;

    virtual AimCommand process(const SimFrame& frame) = 0;
    virtual std::string backendName() const = 0;
};

std::unique_ptr<IAimPipeline> createAimPipeline(const AimBridgeConfig& config);

}  // namespace aim_sim_bridge

