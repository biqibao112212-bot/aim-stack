#include "aim_sim_bridge/stage3_capture.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

namespace aim_sim_bridge
{
namespace
{

constexpr std::size_t kMaxPendingObservationLines = 8192;

std::string envString(const char* name)
{
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

void appendJsonString(std::ostringstream& out, const std::string& value)
{
    out << '"';
    for (const char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    out << '"';
}

void appendNumberOrNull(std::ostringstream& out, double value)
{
    if (std::isfinite(value)) out << value;
    else out << "null";
}

}  // namespace

Stage3ObservationJsonlSink::Stage3ObservationJsonlSink(
    std::filesystem::path path, std::string session_id)
    : path_(std::move(path)), session_id_(std::move(session_id))
{
    std::error_code error;
    if (path_.has_parent_path()) {
        std::filesystem::create_directories(path_.parent_path(), error);
    }
    if (error) {
        healthy_ = false;
        return;
    }
    stream_.open(path_, std::ios::out | std::ios::app);
    healthy_ = stream_.good();
    if (healthy_) worker_ = std::thread(&Stage3ObservationJsonlSink::workerLoop, this);
}

Stage3ObservationJsonlSink::~Stage3ObservationJsonlSink()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    ready_.notify_one();
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    stream_.flush();
    stream_.close();
    std::cerr << "stage3 observation summary submitted=" << submitted_
              << " failed=" << failed_ << " healthy="
              << (healthy_ ? "true" : "false") << '\n';
}

bool Stage3ObservationJsonlSink::writeLine(const std::string& line)
{
    if (!healthy_ || !stream_.good()) return false;
    stream_ << line << '\n';
    stream_.flush();
    if (!stream_.good()) {
        healthy_ = false;
        return false;
    }
    return true;
}

void Stage3ObservationJsonlSink::workerLoop()
{
    while (true) {
        std::string line;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (pending_.empty() && stopping_) return;
            line = std::move(pending_.front());
            pending_.pop_front();
            if (!writeLine(line)) {
                ++failed_;
                // Drop queued lines after the first durable I/O failure.
                pending_.clear();
                return;
            }
            ++submitted_;
        }
    }
}

bool Stage3ObservationJsonlSink::submit(PreTrackerObservationFrame frame)
{
    std::ostringstream out;
    out << std::setprecision(17) << '{';
    out << "\"schema_version\":\"stage3-observation-v2\",\"session_id\":";
    appendJsonString(out, frame.session_id.empty() ? session_id_ : frame.session_id);
    out << ",\"producer_epoch\":" << frame.producer_epoch
        << ",\"frame_seq\":" << frame.frame_seq
        << ",\"timestamp_ns\":" << frame.timestamp_ns
        << ",\"gimbal_pose_timestamp_ns\":" << frame.gimbal_pose_timestamp_ns
        << ",\"gimbal_pose_exposure_matched\":"
        << (frame.gimbal_pose_exposure_matched ? "true" : "false")
        << ",\"tracker_world_transform_exposure_matched\":"
        << (frame.tracker_world_transform_exposure_matched ? "true" : "false")
        << ",\"tracker_origin_world_ros_m\":[";
    for (std::size_t i = 0; i < frame.tracker_origin_world_ros_m.size(); ++i) {
        if (i != 0) out << ',';
        appendNumberOrNull(out, frame.tracker_origin_world_ros_m[i]);
    }
    out << "],\"camera_origin_world_ros_m\":[";
    for (std::size_t i = 0; i < frame.camera_origin_world_ros_m.size(); ++i) {
        if (i != 0) out << ',';
        appendNumberOrNull(out, frame.camera_origin_world_ros_m[i]);
    }
    out << "],\"tracker_gimbal_quaternion_world_wxyz\":[";
    for (std::size_t i = 0; i < frame.tracker_gimbal_quaternion_world_wxyz.size(); ++i) {
        if (i != 0) out << ',';
        appendNumberOrNull(out, frame.tracker_gimbal_quaternion_world_wxyz[i]);
    }
    out << "],\"camera_quaternion_world_wxyz\":[";
    for (std::size_t i = 0; i < frame.camera_quaternion_world_wxyz.size(); ++i) {
        if (i != 0) out << ',';
        appendNumberOrNull(out, frame.camera_quaternion_world_wxyz[i]);
    }
    out << "],\"gimbal_yaw_deg\":" << frame.gimbal_yaw_deg
        << ",\"gimbal_pitch_deg\":" << frame.gimbal_pitch_deg
        << ",\"gimbal_yaw_speed_deg_s\":" << frame.gimbal_yaw_speed_deg_s
        << ",\"camera_profile_id\":";
    appendJsonString(out, frame.camera_profile_id);
    out << ",\"position_contract\":";
    appendJsonString(out, frame.position_contract);
    out << ",\"camera_gimbal_extrinsic_from_config\":"
        << (frame.camera_gimbal_extrinsic_from_config ? "true" : "false")
        << ",\"R_camera2gimbal\":[";
    for (std::size_t i = 0; i < frame.R_camera2gimbal.size(); ++i) {
        if (i != 0) out << ',';
        appendNumberOrNull(out, frame.R_camera2gimbal[i]);
    }
    out << "],\"t_camera2gimbal_m\":[";
    for (std::size_t i = 0; i < frame.t_camera2gimbal_m.size(); ++i) {
        if (i != 0) out << ',';
        appendNumberOrNull(out, frame.t_camera2gimbal_m[i]);
    }
    out << ']';
    out << ",\"armors\":[";
    for (std::size_t i = 0; i < frame.armors.size(); ++i) {
        const auto& armor = frame.armors[i];
        if (i != 0) out << ',';
        out << "{\"observation_index\":" << armor.observation_id
            << ",\"detector_number\":" << armor.detector_number
            << ",\"detector_color\":" << armor.detector_color
            << ",\"detector_type\":" << armor.detector_type
            << ",\"valid\":" << (armor.finite_valid ? "true" : "false")
            << ",\"position_m\":[";
        for (std::size_t component = 0; component < armor.position_m.size(); ++component) {
            if (component != 0) out << ',';
            appendNumberOrNull(out, armor.position_m[component]);
        }
        out << "],\"camera_tvec_m\":[";
        for (std::size_t component = 0; component < armor.camera_tvec_m.size(); ++component) {
            if (component != 0) out << ',';
            appendNumberOrNull(out, armor.camera_tvec_m[component]);
        }
        out << "],\"yaw_rad\":";
        appendNumberOrNull(out, armor.yaw_rad);
        out << ",\"yaw_absolute_rad\":";
        appendNumberOrNull(out, armor.yaw_absolute_rad);
        out << ",\"reprojection_rms_px\":";
        appendNumberOrNull(out, armor.reprojection_rms_px);
        out << ",\"reprojection_max_px\":";
        appendNumberOrNull(out, armor.reprojection_max_px);
        out << '}';
    }
    out << "]}";

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!healthy_ || stopping_) {
            ++failed_;
            return false;
        }
        if (pending_.size() >= kMaxPendingObservationLines) {
            healthy_ = false;
            ++failed_;
            return false;
        }
        pending_.push_back(out.str());
    }
    ready_.notify_one();
    return true;
}

bool Stage3ObservationJsonlSink::healthy() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return healthy_;
}

std::uint64_t Stage3ObservationJsonlSink::submitted() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return submitted_;
}

std::uint64_t Stage3ObservationJsonlSink::failed() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return failed_;
}

std::shared_ptr<IPreTrackerObservationSink> createStage3ObservationSinkFromEnv()
{
    const std::string path = envString("AIM_SIM_STAGE3_OBSERVATIONS");
    if (path.empty()) return {};
    return std::make_shared<Stage3ObservationJsonlSink>(
        std::filesystem::path(path), envString("AIM_SIM_STAGE3_SESSION_ID"));
}

}  // namespace aim_sim_bridge
