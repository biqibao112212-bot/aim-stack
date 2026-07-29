#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace aim_sim_bridge
{

struct PreTrackerArmorObservation
{
    std::uint32_t observation_id = 0;
    int detector_number = 0;
    int detector_color = 0;
    int detector_type = 0;
    bool finite_valid = false;
    std::array<double, 3> position_m{};
    std::array<double, 3> camera_tvec_m{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};
    double yaw_rad = std::numeric_limits<double>::quiet_NaN();
    double yaw_absolute_rad = std::numeric_limits<double>::quiet_NaN();
    double reprojection_rms_px = std::numeric_limits<double>::quiet_NaN();
    double reprojection_max_px = std::numeric_limits<double>::quiet_NaN();
};

struct PreTrackerObservationFrame
{
    std::string session_id;
    std::uint64_t producer_epoch = 0;
    std::uint64_t frame_seq = 0;
    std::uint64_t timestamp_ns = 0;
    std::uint64_t gimbal_pose_timestamp_ns = 0;
    bool gimbal_pose_exposure_matched = false;
    bool tracker_world_transform_exposure_matched = false;
    std::array<double, 3> tracker_origin_world_ros_m{};
    std::array<double, 3> camera_origin_world_ros_m{};
    std::array<double, 4> tracker_gimbal_quaternion_world_wxyz{1.0, 0.0, 0.0, 0.0};
    std::array<double, 4> camera_quaternion_world_wxyz{1.0, 0.0, 0.0, 0.0};
    double gimbal_yaw_deg = 0.0;
    double gimbal_pitch_deg = 0.0;
    double gimbal_yaw_speed_deg_s = 0.0;
    std::string camera_profile_id;
    std::string position_contract = "calibrated-camera-gimbal-extrinsic-v1";
    bool camera_gimbal_extrinsic_from_config = false;
    std::array<double, 9> R_camera2gimbal{};
    std::array<double, 3> t_camera2gimbal_m{};
    std::vector<PreTrackerArmorObservation> armors;
};

class IPreTrackerObservationSink
{
public:
    virtual ~IPreTrackerObservationSink() = default;
    virtual bool submit(PreTrackerObservationFrame frame) = 0;
    virtual bool healthy() const = 0;
    virtual std::uint64_t submitted() const = 0;
    virtual std::uint64_t failed() const = 0;
};

enum class TargetMode
{
    Armor,
    Outpost
};

struct AimBridgeConfig
{
    TargetMode default_mode = TargetMode::Armor;
    double bullet_speed_mps = 22.0;
    // Encoded transport pitch corresponding to zero optical aim. The
    // simulator camera is mounted 25 degrees from the gimbal joint.
    double sim_pitch_neutral_deg = 65.0;
    bool enable_fire = true;
    bool publish_no_target = true;
    // The deployed and collection default is the native 6 mm full frame.
    // Precision cropping remains an explicit diagnostic opt-in only.
    bool dual_focal_enabled = false;
    double wide_focal_mm = 6.0;
    double precision_focal_mm = 16.0;
    double precision_enter_distance_m = 3.2;
    double precision_leave_distance_m = 2.8;
    double precision_fov_margin_px = 28.0;
    bool command_smoothing_enabled = true;
    double command_smoothing_deadband_deg = 0.03;
    double command_smoothing_alpha = 0.65;
    double command_smoothing_passthrough_deg = 1.5;
    std::string armor_detector_config;
    std::shared_ptr<IPreTrackerObservationSink> pre_tracker_observation_sink;
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
    TargetMode target_mode = TargetMode::Armor;
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
TargetMode parseTargetMode(const std::string& value, TargetMode fallback = TargetMode::Armor);

}  // namespace aim_sim_bridge
