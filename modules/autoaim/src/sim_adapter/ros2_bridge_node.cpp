#include "aim_sim_bridge/pipeline.hpp"
#include "aim_sim_bridge/sim_command.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rm_interfaces/msg/gimbal_cmd.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace
{

constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

double stampToMs(const builtin_interfaces::msg::Time& stamp)
{
    return static_cast<double>(stamp.sec) * 1000.0 +
           static_cast<double>(stamp.nanosec) * 1e-6;
}

bool encodingEquals(const std::string& lhs, const char* rhs)
{
    if (lhs.size() != std::strlen(rhs)) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
            std::tolower(static_cast<unsigned char>(rhs[i]))) {
            return false;
        }
    }
    return true;
}

bool imageToBgr(const sensor_msgs::msg::Image& image, cv::Mat* output)
{
    if (output == nullptr || image.height == 0 || image.width == 0 || image.data.empty()) {
        return false;
    }

    const int height = static_cast<int>(image.height);
    const int width = static_cast<int>(image.width);
    const size_t expected_step_rgb = static_cast<size_t>(width) * 3;
    if (image.step < expected_step_rgb) {
        return false;
    }

    if (encodingEquals(image.encoding, "rgb8")) {
        cv::Mat rgb(height, width, CV_8UC3, const_cast<std::uint8_t*>(image.data.data()), image.step);
        cv::cvtColor(rgb, *output, cv::COLOR_RGB2BGR);
        return true;
    }

    if (encodingEquals(image.encoding, "bgr8")) {
        cv::Mat bgr(height, width, CV_8UC3, const_cast<std::uint8_t*>(image.data.data()), image.step);
        *output = bgr.clone();
        return true;
    }

    if (encodingEquals(image.encoding, "mono8")) {
        cv::Mat mono(height, width, CV_8UC1, const_cast<std::uint8_t*>(image.data.data()), image.step);
        cv::cvtColor(mono, *output, cv::COLOR_GRAY2BGR);
        return true;
    }

    return false;
}

void setEnvIfNotEmpty(const char* key, const std::string& value)
{
    if (value.empty()) return;
#if defined(_WIN32)
    _putenv_s(key, value.c_str());
#else
    setenv(key, value.c_str(), 1);
#endif
}

}  // namespace

namespace aim_sim_bridge
{

class AimSimBridgeNode final : public rclcpp::Node
{
public:
    AimSimBridgeNode() : Node("aim_sim_bridge_node")
    {
        config_.bullet_speed_mps = declare_parameter<double>("bullet_speed_mps", 22.0);
        config_.sim_pitch_neutral_deg =
            declare_parameter<double>("sim_pitch_neutral_deg", 65.0);
        config_.enable_fire = declare_parameter<bool>("enable_fire", true);
        config_.publish_no_target = declare_parameter<bool>("publish_no_target", true);
        config_.dual_focal_enabled = declare_parameter<bool>("dual_focal_enabled", false);
        config_.wide_focal_mm = declare_parameter<double>("wide_focal_mm", 6.0);
        config_.precision_focal_mm = declare_parameter<double>("precision_focal_mm", 16.0);
        config_.precision_enter_distance_m =
            declare_parameter<double>("precision_enter_distance_m", 3.2);
        config_.precision_leave_distance_m =
            declare_parameter<double>("precision_leave_distance_m", 2.8);
        config_.precision_fov_margin_px =
            declare_parameter<double>("precision_fov_margin_px", 28.0);
        config_.armor_detector_config =
            declare_parameter<std::string>("armor_detector_config", "");
        target_mode_param_ = declare_parameter<std::string>("target_mode", "armor");
        config_.default_mode = parseTargetMode(target_mode_param_, TargetMode::Armor);

        use_last_command_as_feedback_ =
            declare_parameter<bool>("use_last_command_as_feedback", true);
        use_tf_feedback_ = declare_parameter<bool>("use_tf_feedback", false);
        const std::string param_yaml =
            declare_parameter<std::string>("param_yaml", "config/param.sim.yaml");
        setEnvIfNotEmpty("AIM_SIM_PARAM_YAML", param_yaml);

        pipeline_ = createAimPipeline(config_);

        command_pub_ = create_publisher<rm_interfaces::msg::GimbalCmd>(
            "/rm_gimbal/cmd", rclcpp::SensorDataQoS());
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                handleImage(*msg);
            });
        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
                handleCameraInfo(*msg);
            });
        tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
            "/tf", rclcpp::SensorDataQoS(),
            [this](tf2_msgs::msg::TFMessage::ConstSharedPtr msg) {
                handleTf(*msg);
            });

        RCLCPP_INFO(
            get_logger(),
            "aim_sim_bridge started backend=%s mode=%s fire=%s pitch_neutral=%.1f",
            pipeline_->backendName().c_str(), toString(config_.default_mode),
            config_.enable_fire ? "on" : "off", config_.sim_pitch_neutral_deg);
    }

private:
    void handleCameraInfo(const sensor_msgs::msg::CameraInfo& msg)
    {
        if (camera_info_logged_) return;
        camera_info_logged_ = true;
        RCLCPP_INFO(
            get_logger(), "camera_info: %ux%u fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5]);
    }

    void handleTf(const tf2_msgs::msg::TFMessage& msg)
    {
        if (!use_tf_feedback_) return;

        for (const auto& tf : msg.transforms) {
            if (tf.child_frame_id != "gimbal_link") continue;
            const auto& q = tf.transform.rotation;
            const double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
            const double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
            const double roll = std::atan2(sinr_cosp, cosr_cosp);

            const double sinp = 2.0 * (q.w * q.y - q.z * q.x);
            const double pitch = std::abs(sinp) >= 1.0
                ? std::copysign(3.14159265358979323846 / 2.0, sinp)
                : std::asin(sinp);

            const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
            const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
            const double yaw = std::atan2(siny_cosp, cosy_cosp);

            (void)roll;
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            feedback_yaw_deg_ = yaw * kRadToDeg;
            feedback_pitch_deg_ = pitch * kRadToDeg;
        }
    }

    void handleImage(const sensor_msgs::msg::Image& image)
    {
        cv::Mat bgr;
        if (!imageToBgr(image, &bgr)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "unsupported image encoding or layout: encoding=%s step=%u size=%zu",
                image.encoding.c_str(), image.step, image.data.size());
            return;
        }

        const std::string target_mode_text = get_parameter("target_mode").as_string();
        const TargetMode target_mode = parseTargetMode(target_mode_text, config_.default_mode);

        double yaw = 0.0;
        double pitch = 0.0;
        {
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            yaw = feedback_yaw_deg_;
            pitch = feedback_pitch_deg_;
        }

        SimFrame frame;
        frame.bgr_image = std::move(bgr);
        frame.timestamp_ms = stampToMs(image.header.stamp);
        frame.gimbal_yaw_deg = yaw;
        frame.gimbal_pitch_deg = pitch;
        frame.bullet_speed_mps = config_.bullet_speed_mps;
        frame.target_mode = target_mode;

        AimCommand aim = pipeline_->process(frame);
        if (!aim.has_target && !config_.publish_no_target) {
            return;
        }

        const SimulatorCommandFields sim_cmd = toSimulatorCommand(aim, config_);
        rm_interfaces::msg::GimbalCmd msg;
        msg.header.stamp = get_clock()->now();
        msg.header.frame_id = "aim_sim_bridge";
        msg.yaw = sim_cmd.yaw_deg;
        msg.pitch = sim_cmd.pitch_deg;
        msg.yaw_diff = 0.0;
        msg.pitch_diff = 0.0;
        msg.distance = sim_cmd.distance_m;
        msg.fire_advice = sim_cmd.fire_advice;
        command_pub_->publish(msg);

        if (use_last_command_as_feedback_ && aim.has_target) {
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            feedback_yaw_deg_ = aim.yaw_deg;
            feedback_pitch_deg_ = aim.pitch_deg;
        }
    }

    AimBridgeConfig config_;
    std::string target_mode_param_;
    bool use_last_command_as_feedback_ = true;
    bool use_tf_feedback_ = false;
    bool camera_info_logged_ = false;
    std::mutex feedback_mutex_;
    double feedback_yaw_deg_ = 0.0;
    double feedback_pitch_deg_ = 0.0;
    std::unique_ptr<IAimPipeline> pipeline_;

    rclcpp::Publisher<rm_interfaces::msg::GimbalCmd>::SharedPtr command_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
};

}  // namespace aim_sim_bridge

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<aim_sim_bridge::AimSimBridgeNode>());
    rclcpp::shutdown();
    return 0;
}
