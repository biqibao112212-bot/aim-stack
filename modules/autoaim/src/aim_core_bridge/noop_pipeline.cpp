#include "aim_sim_bridge/pipeline.hpp"

#include <iostream>
#include <utility>

namespace aim_sim_bridge
{
namespace
{

class NoopAimPipeline final : public IAimPipeline
{
public:
    explicit NoopAimPipeline(AimBridgeConfig config) : config_(std::move(config)) {}

    AimCommand process(const SimFrame& frame) override
    {
        (void)frame;
        if (!warned_) {
            std::cerr
                << "[aim_sim_bridge] AIM_SIM_WITH_VIVSIONN_TRT=OFF; publishing no-target "
                   "commands. Rebuild with -DAIM_SIM_WITH_VIVSIONN_TRT=ON on a machine "
                   "with TensorRT/CUDA and model engines to run vivsionn detection."
                << std::endl;
            warned_ = true;
        }
        AimCommand command;
        command.backend = backendName();
        command.has_target = false;
        return command;
    }

    std::string backendName() const override
    {
        return "noop";
    }

private:
    AimBridgeConfig config_;
    bool warned_ = false;
};

}  // namespace

std::unique_ptr<IAimPipeline> createAimPipeline(const AimBridgeConfig& config)
{
    return std::make_unique<NoopAimPipeline>(config);
}

}  // namespace aim_sim_bridge
