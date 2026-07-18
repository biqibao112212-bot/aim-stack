#include "aim_sim_bridge/pipeline.hpp"

#include "buff_rune_pipeline.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <utility>

namespace aim_sim_bridge
{
namespace
{

std::uint8_t taskModeFor(TargetMode mode)
{
    return mode == TargetMode::BigBuff
        ? rm::FeedBackData::TASK_MODE::HIT_BIG_BUFF
        : rm::FeedBackData::TASK_MODE::HIT_SMALL_BUFF;
}

rm::Frame makeBuffFrame(const SimFrame& input)
{
    rm::Frame frame;
    frame.srcImg = input.bgr_image;
    frame.source_producer_epoch = input.source_producer_epoch;
    frame.source_image_seq = input.source_image_seq;
    frame.source_capture_timestamp_ns = input.source_capture_timestamp_ns;
    frame.poseEuler.roll = 0.0F;
    frame.poseEuler.yaw = static_cast<float>(input.gimbal_yaw_deg);
    frame.poseEuler.pitch = static_cast<float>(input.gimbal_pitch_deg);
    frame.bullet_speed = input.bullet_speed_mps;
    frame.timeStamp = input.timestamp_ms;
    frame.usb_timeStamp = input.timestamp_ms;
    frame.simulator_state_age_s = input.simulator_state_age_s;
    frame.startTime = std::chrono::high_resolution_clock::now() -
        std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
            std::chrono::duration<double, std::milli>(input.timestamp_ms));
    frame.fb.task_mode = taskModeFor(input.target_mode);
    frame.fb.self_team = rm::FeedBackData::SELF_TEAM::SELF_BLUE;
    frame.fb.heat = 0;
    frame.fb.heat_cap = 600;
    frame.fb.bullet_speed = static_cast<float>(input.bullet_speed_mps);
    frame.fb.gimbal_roll = frame.poseEuler.roll;
    frame.fb.gimbal_yaw = frame.poseEuler.yaw;
    frame.fb.gimbal_pitch = frame.poseEuler.pitch;
    frame.fb.yaw_speed = static_cast<float>(input.gimbal_yaw_speed_deg_s);
    frame.fb.__reserved[0] = 1;
    frame.fb.set_task_mode_telemetry(frame.fb.task_mode, frame.fb.task_mode);
    return frame;
}

AimCommand fromControlData(const rm::ControlData& control)
{
    AimCommand command;
    command.backend = "energy_buff_trt";
    command.has_target =
        control.aiming_state == rm::ControlData::AIMING_STATE::TARGET_DETECTED;
    command.yaw_deg = control.gimbal_yaw;
    command.pitch_deg = control.gimbal_pitch;
    command.distance_m = command.has_target ? 1.0 : -1.0;
    command.raw_shot_mode = control.shot_mode;
    command.fire_advice =
        control.shot_mode == rm::ControlData::SHOT_MODE::AUTO_FIRE ||
        control.shot_mode == rm::ControlData::SHOT_MODE::SHOT_ONCE;
    return command;
}

class BuffAimPipeline final : public IAimPipeline
{
public:
    explicit BuffAimPipeline(AimBridgeConfig config) : config_(std::move(config)) {}

    AimCommand process(const SimFrame& input) override
    {
        if (input.bgr_image.empty()) {
            AimCommand command;
            command.backend = backendName();
            return command;
        }
        if (!pipeline_) {
            pipeline_ = std::make_unique<auto_buff::BuffRunePipeline>(
                config_.buff_config_path, auto_buff::BuffRunePipelineOptions{false});
        }
        pipeline_->push(makeBuffFrame(input));
        auto_buff::BuffRuneResult result;
        if (!pipeline_->tryPopLatest(&result)) {
            AimCommand command;
            command.backend = backendName();
            return command;
        }
        AimCommand command = fromControlData(result.control);
        command.runtime_yolo_ms = result.yolo_ms;
        command.runtime_solve_ms = result.solve_ms;
        command.runtime_track_aim_ms = result.essential_track_aim_ms;
        command.runtime_pipeline_delay_ms = result.pipeline_delay_ms;
        command.completed_vision_result = true;
        command.source_producer_epoch = result.frame.source_producer_epoch;
        command.source_image_seq = result.frame.source_image_seq;
        command.source_capture_timestamp_ns = result.frame.source_capture_timestamp_ns;
        command.vision_completion_timestamp_ns = result.completion_timestamp_ns;
        return command;
    }

    std::string backendName() const override { return "energy_buff_trt"; }

private:
    AimBridgeConfig config_;
    std::unique_ptr<auto_buff::BuffRunePipeline> pipeline_;
};

}  // namespace

std::unique_ptr<IAimPipeline> createAimPipeline(const AimBridgeConfig& config)
{
    return std::make_unique<BuffAimPipeline>(config);
}

}  // namespace aim_sim_bridge
