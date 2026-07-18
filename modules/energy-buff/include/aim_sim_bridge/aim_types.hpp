#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <limits>
#include <string>

namespace aim_sim_bridge
{

enum class TargetMode
{
    SmallBuff,
    BigBuff
};

struct AimBridgeConfig
{
    TargetMode default_mode = TargetMode::SmallBuff;
    double bullet_speed_mps = 22.0;
    // Encoded transport pitch corresponding to zero optical aim. The
    // simulator camera is mounted 25 degrees from the gimbal joint.
    double sim_pitch_neutral_deg = 65.0;
    bool enable_fire = true;
    bool publish_no_target = true;
    std::string buff_config_path;
};

struct SimFrame
{
    cv::Mat bgr_image;
    std::uint64_t source_producer_epoch = 0;
    std::uint64_t source_image_seq = 0;
    std::uint64_t source_capture_timestamp_ns = 0;
    std::uint64_t gimbal_pose_timestamp_ns = 0;
    bool gimbal_pose_exposure_matched = false;
    bool tracker_world_transform_exposure_matched = false;
    cv::Vec3d tracker_origin_world_ros_m = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d tracker_frame_rpy_world_ros_rad = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec4d tracker_gimbal_quaternion_world_wxyz = cv::Vec4d(1.0, 0.0, 0.0, 0.0);
    cv::Vec3d camera_origin_world_ros_m = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec4d camera_quaternion_world_wxyz = cv::Vec4d(1.0, 0.0, 0.0, 0.0);
    double timestamp_ms = 0.0;
    double gimbal_yaw_deg = 0.0;
    double gimbal_pitch_deg = 0.0;
    double gimbal_yaw_speed_deg_s = 0.0;
    double simulator_state_age_s = 0.0;
    double bullet_speed_mps = 22.0;
    TargetMode target_mode = TargetMode::SmallBuff;
    std::string camera_profile_id = "wide_6mm";
    double camera_focal_mm = 6.0;
    bool has_camera_matrix_override = false;
    cv::Mat camera_matrix_override;
    cv::Rect source_roi_px;
    int source_image_width = 0;
    int source_image_height = 0;
    double virtual_scale_x = 1.0;
    double virtual_scale_y = 1.0;
};

struct AimCommand
{
    bool has_target = false;
    double yaw_deg = 0.0;
    double pitch_deg = 0.0;
    double distance_m = -1.0;
    bool fire_advice = false;
    std::uint8_t raw_shot_mode = 0;
    std::string selected_camera = "wide_6mm";
    double selected_focal_mm = 6.0;
    std::string backend;
    // True only for a result that crossed the active detector, solve/PnP,
    // tracker, and aim boundary. Command resends must preserve this identity.
    bool completed_vision_result = false;
    std::uint64_t source_producer_epoch = 0;
    std::uint64_t source_image_seq = 0;
    std::uint64_t source_capture_timestamp_ns = 0;
    std::uint64_t vision_completion_timestamp_ns = 0;
    double runtime_yolo_ms = std::numeric_limits<double>::quiet_NaN();
    double runtime_solve_ms = std::numeric_limits<double>::quiet_NaN();
    double runtime_track_aim_ms = std::numeric_limits<double>::quiet_NaN();
    double runtime_pipeline_delay_ms = std::numeric_limits<double>::quiet_NaN();
};

const char* toString(TargetMode mode);
TargetMode parseTargetMode(const std::string& value, TargetMode fallback = TargetMode::SmallBuff);

}  // namespace aim_sim_bridge
