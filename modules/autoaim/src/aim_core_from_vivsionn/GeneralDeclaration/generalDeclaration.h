#ifndef GENERALDECLARATION_H
#define GENERALDECLARATION_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#define OUTPOST_ROTATE_D 553.0

#undef EOF // 这里的EOF会与其他的EOF冲突

using namespace std;

namespace rm
{

enum MOVEMENT// 目标运动模式，用于辅助响应
{
    STATIC,     ///< 静止状态
    TRANSLATION, ///< 横移
    SPINNING,   ///< 旋转
    TRANSPIN    ///< 平移+旋转
};

enum ArmorColor
{
    RED_ARMOR,
    BLUE_ARMOR,
    GRAY_ARMOR, ///< 熄灭灯条
    PURPLE_ARMOR
};

const std::string ArmorColorStr[4] = {"RED", "BLUE", "GRAY", "PURPLE"};

constexpr double D2R = CV_PI / 180.0;
constexpr double R2D = 180.0 / CV_PI;

struct AngleWithTime
{
    double angle;///< 角度
    double t;///< 秒
};

struct PosWithTime
{
    Eigen::Vector3d pos;///< 三维坐标/m
    double t;   ///< 时间戳/秒
};

enum class ArmorType
{
    SMALL = 0,
    LARGE,
    INVALID ///< cant recognize
};
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

struct Light : public cv::RotatedRect
{
  Light() = default;

  explicit Light(cv::RotatedRect box) : cv::RotatedRect(box)
  {
    cv::Point2f p[4];
    box.points(p);
    std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
    top = (p[0] + p[1]) / 2;
    bottom = (p[2] + p[3]) / 2;

    length = cv::norm(top - bottom);
    width = cv::norm(p[0] - p[1]);

    tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
    tilt_angle = tilt_angle / CV_PI * 180;
  }

  int color = 0;
  cv::Point2f top, bottom;
  double length;
  double width;
  float tilt_angle; ///< 倾斜角
};

struct ArmorForDetect
{
  // [修复] 加上这个宏，防止 Eigen 内存未对齐导致的崩溃
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::uint32_t observation_id = std::numeric_limits<std::uint32_t>::max();

  ArmorForDetect() = default;
  ArmorForDetect(const Light & l1, const Light & l2)
  {
    if (l1.center.x < l2.center.x)
    {
      left_light = l1, right_light = l2;
    }
    else
    {
      left_light = l2, right_light = l1;
    }
    center = (left_light.center + right_light.center) / 2;
    color = left_light.color;
  }

  // Light pairs part
  Light left_light, right_light;
  cv::Point2f center;
  ArmorType type = ArmorType::INVALID;

  // Number part
  cv::Mat number_img;
  std::string number_string;
  int number;
  float confidence;
  std::string classfication_result;

  int color = 0;

  // position
  std::vector<cv::Point2f> vertex;  //vertex for PNP (bl, tl, tr, br)

  Eigen::Vector3d armorPosition;
  double yaw;

  float dis; //mm
  double distanceToImageCenter;
};

struct PnPCandidate
{
    enum class CornerOrder : std::uint8_t
    {
        DETECTOR_CANONICAL = 0 // ArmorForDetect order: bl, tl, tr, br.
    };

    enum class Polarity : std::int8_t
    {
        NOMINAL = 1 // No reversed-normal semantic hypothesis was generated.
    };

    std::uint32_t id = 0;
    std::uint32_t solver_solution_index = 0;
    cv::Mat rVec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tVec = cv::Mat::zeros(3, 1, CV_64FC1);
    double reprojection_error_px = std::numeric_limits<double>::infinity();
    CornerOrder corner_order = CornerOrder::DETECTOR_CANONICAL;
    Polarity polarity = Polarity::NOMINAL;
    bool selected = false;

    PnPCandidate() = default;

    PnPCandidate(const PnPCandidate& other)
        : id(other.id),
          solver_solution_index(other.solver_solution_index),
          rVec(other.rVec.clone()),
          tVec(other.tVec.clone()),
          reprojection_error_px(other.reprojection_error_px),
          corner_order(other.corner_order),
          polarity(other.polarity),
          selected(other.selected)
    {
    }

    PnPCandidate& operator=(const PnPCandidate& other)
    {
        if (this == &other) return *this;
        id = other.id;
        solver_solution_index = other.solver_solution_index;
        rVec = other.rVec.clone();
        tVec = other.tVec.clone();
        reprojection_error_px = other.reprojection_error_px;
        corner_order = other.corner_order;
        polarity = other.polarity;
        selected = other.selected;
        return *this;
    }

    PnPCandidate(PnPCandidate&&) noexcept = default;
    PnPCandidate& operator=(PnPCandidate&&) noexcept = default;
};

// Diagnostic-only constrained armor pose candidate. It is deliberately
// separate from PnPCandidate: IPPE ranks a free 6-DoF planar pose, while this
// candidate fixes the armor tilt and jointly optimizes camera-frame yaw+tvec.
// The tracker and legacy selected pose never consume this sidecar.
struct ParallelJointPnPCandidate
{
    std::uint32_t id = 0;
    double yaw_rad = std::numeric_limits<double>::quiet_NaN();
    cv::Mat rVec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tVec = cv::Mat::zeros(3, 1, CV_64FC1);
    double reprojection_error_px = std::numeric_limits<double>::infinity();
    double max_reprojection_error_px = std::numeric_limits<double>::infinity();
    std::vector<double> corner_residual_px;
    double translation_linear_condition = std::numeric_limits<double>::infinity();
    double translation_information_condition = std::numeric_limits<double>::infinity();
    double yaw_sensitivity_deg_per_px = std::numeric_limits<double>::infinity();
    bool yaw_sensitivity_valid = false;
    double coarse_seed_yaw_rad = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    bool converged = false;
    bool improved = false;
    bool positive_depth = false;
    bool search_bound_hit = false;
    bool selected = false;

    ParallelJointPnPCandidate() = default;

    ParallelJointPnPCandidate(const ParallelJointPnPCandidate& other)
        : id(other.id),
          yaw_rad(other.yaw_rad),
          rVec(other.rVec.clone()),
          tVec(other.tVec.clone()),
          reprojection_error_px(other.reprojection_error_px),
          max_reprojection_error_px(other.max_reprojection_error_px),
          corner_residual_px(other.corner_residual_px),
          translation_linear_condition(other.translation_linear_condition),
          translation_information_condition(other.translation_information_condition),
          yaw_sensitivity_deg_per_px(other.yaw_sensitivity_deg_per_px),
          yaw_sensitivity_valid(other.yaw_sensitivity_valid),
          coarse_seed_yaw_rad(other.coarse_seed_yaw_rad),
          iterations(other.iterations),
          converged(other.converged),
          improved(other.improved),
          positive_depth(other.positive_depth),
          search_bound_hit(other.search_bound_hit),
          selected(other.selected)
    {
    }

    ParallelJointPnPCandidate& operator=(const ParallelJointPnPCandidate& other)
    {
        if (this == &other) return *this;
        id = other.id;
        yaw_rad = other.yaw_rad;
        rVec = other.rVec.clone();
        tVec = other.tVec.clone();
        reprojection_error_px = other.reprojection_error_px;
        max_reprojection_error_px = other.max_reprojection_error_px;
        corner_residual_px = other.corner_residual_px;
        translation_linear_condition = other.translation_linear_condition;
        translation_information_condition = other.translation_information_condition;
        yaw_sensitivity_deg_per_px = other.yaw_sensitivity_deg_per_px;
        yaw_sensitivity_valid = other.yaw_sensitivity_valid;
        coarse_seed_yaw_rad = other.coarse_seed_yaw_rad;
        iterations = other.iterations;
        converged = other.converged;
        improved = other.improved;
        positive_depth = other.positive_depth;
        search_bound_hit = other.search_bound_hit;
        selected = other.selected;
        return *this;
    }

    ParallelJointPnPCandidate(ParallelJointPnPCandidate&&) noexcept = default;
    ParallelJointPnPCandidate& operator=(ParallelJointPnPCandidate&&) noexcept = default;
};

struct Armor
{
    // [修复] 加上这个宏，防止 Eigen 内存未对齐导致的崩溃
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::uint32_t observation_id = std::numeric_limits<std::uint32_t>::max();

    int number;
    int color = 0;

    ArmorType type;   //big or small
    cv::Rect rect;
    std::vector<cv::Point2f> vertex;//vertex for PNP (tl, tr, br, bl)

    cv::Point2f center = cv::Point(0, 0); ///< 装甲板中心的图像坐标
    cv::Point2f hitPointR, hitPointL, hitPointU, hitPointD;
    cv::Point2f armorR, armorL, armorU, armorD; // 用于计算开火范围
    cv::Mat rVec = cv::Mat::zeros(3, 1, CV_64FC1); // 旋转矢量
    cv::Mat tVec = cv::Mat::zeros(3, 1, CV_64FC1); // 平移矢量

    Eigen::Vector3d armorPosition; ///< 装甲板在惯性系下的3D坐标(m)
    std::vector<PnPCandidate> pnp_candidates;
    std::vector<ParallelJointPnPCandidate> parallel_joint_candidates;
    std::vector<ParallelJointPnPCandidate> parallel_joint_raw_candidates;
    double legacy_constrained_reprojection_error_px =
        std::numeric_limits<double>::infinity();
    double legacy_constrained_max_reprojection_error_px =
        std::numeric_limits<double>::infinity();
    std::vector<double> legacy_constrained_corner_residual_px;
    double exposure_constrained_reprojection_error_px =
        std::numeric_limits<double>::infinity();
    double exposure_constrained_max_reprojection_error_px =
        std::numeric_limits<double>::infinity();
    std::vector<double> exposure_constrained_corner_residual_px;
    double parallel_joint_solve_us = 0.0;
    double parallel_joint_raw_solve_us = 0.0;
    Eigen::Vector3d ypd = Eigen::Vector3d::Zero(); ///< yaw / pitch / distance
    Eigen::Vector3d hitPosRight, hitPosLeft, hitPosUp, hitPosDown;
    double yaw; ///< 相对于机器人中心的角度（弧度），云台转，这个角度不变
    // Corrected constrained-PnP yaw in the tracker/chassis frame.  The
    // +15-degree armor tilt is also defined in this frame and is projected
    // through the exposure gimbal pose before solving.
    double yaw_absolute;
    double yaw_raw = 0.0; ///< PnP姿态直接给出的装甲板yaw
    double legacy_camera_fixed_yaw = std::numeric_limits<double>::quiet_NaN();
    double R;
    bool has_explicit_ypd = false;

    float dis; ///< mm
    double distanceToImageCenter;

    enum LABEL
    {
        HERO = 1,
        SENTRY = 7,
        BASE = 8,
        OUTPOST = 9
    };

    float getDis(cv::Point2f p, cv::Point2f q){return std::sqrt(pow(p.x-q.x,2)+pow(p.y-q.y,2));}

    bool judge_hit(cv::Point2f p)
    {
        return (getDis(p, center) <= getDis(vertex[0], vertex[1])/2.0);
    }

    Armor() = default;
    Armor(const rm::ArmorForDetect & armor)
    {
        observation_id = armor.observation_id;
        const bool has_explicit_type = armor.type != ArmorType::INVALID;
        type = has_explicit_type ? armor.type : ArmorType::SMALL;
        this->color = armor.color;

        //number
        if(armor.number == 1)
        {
            number = 1;
            if (!has_explicit_type) type = ArmorType::LARGE;
        }
        else if(armor.number == 2)
            number = 2;
        else if(armor.number == 3)
            number = 3;
        else if(armor.number == 4)
            number = 4;
        else if(armor.number == 5)
            number = 5;
        else if(armor.number == 0)
            number = SENTRY;
        else if(armor.number == 7 || armor.number == 8)
            number = 8;
        else if(armor.number == 6)
            number = OUTPOST;

        // pose
        if (armor.vertex.size() == 4) {
            vertex = armor.vertex;
        } else {
            vertex.emplace_back(armor.left_light.bottom);
            vertex.emplace_back(armor.left_light.top);
            vertex.emplace_back(armor.right_light.top);
            vertex.emplace_back(armor.right_light.bottom);
        }

        auto finite_point = [](const cv::Point2f& point) {
            return std::isfinite(point.x) && std::isfinite(point.y);
        };
        const cv::Point2f vertex_left_center = (vertex[0] + vertex[1]) / 2.0f;
        const cv::Point2f vertex_right_center = (vertex[2] + vertex[3]) / 2.0f;
        const cv::Point2f left_center =
            finite_point(armor.left_light.center) ? armor.left_light.center : vertex_left_center;
        const cv::Point2f right_center =
            finite_point(armor.right_light.center) ? armor.right_light.center : vertex_right_center;
        const cv::Point2f left_axis = vertex[1] - vertex[0];

        center = finite_point(armor.center) ? armor.center : (right_center + left_center)/2.0;
        hitPointR = left_center + 4.0/5.0*(right_center - left_center);
        hitPointL = left_center + 1.0/5.0*(right_center - left_center);
        hitPointU = center + left_axis*0.8;
        hitPointD = center - left_axis*0.8;

        armorR = right_center;
        armorL = left_center;
        armorU = center + left_axis*0.8;
        armorD = center - left_axis*0.8;
    }
};

const uint8_t start_of_frame = 0x33, end_of_frame = 0xEE;
struct ControlData
{
    uint8_t  SOF = start_of_frame;
    uint8_t  _reserved[3] = {0,};     
    float    gimbal_pitch = 0;        
    float    gimbal_yaw = 0;          
    float    yaw_error = 0;
    uint8_t  shot_mode = DO_NOTHING;  
    uint8_t  shot_buff_mode = SHOT_BUFF_OFF;    
    uint8_t  aiming_state = AIMMING_NO_TARGET;  
    uint8_t  EOF = end_of_frame;

    enum AIMING_STATE
    {
        NO_COM                  = (uint8_t)0x00,
        NO_CAMERA               = (uint8_t)0x11, 
        AIMMING_NO_TARGET       = (uint8_t)0x22, 
        TARGET_DETECTED         = (uint8_t)0x33  
    };

    enum SHOT_BUFF_MODE
    {
        SHOT_BUFF_OFF = (uint8_t)0x00,  
        SHOT_BUFF_ON  = (uint8_t)0x01  
    };

    enum SHOT_MODE
    {
        DO_NOTHING    =   (uint8_t)0x00,  
        AIM_ONLY      =   (uint8_t)0x01,  
        AUTO_FIRE     =   (uint8_t)0x02,  
        SHOT_ONCE     =   (uint8_t)0x03   
    };

};

const std::string shotModeStr[4] = {"DO_NOTHING", "AIM_ONLY",
                                    "AUTO_FIRE", "SHOT_ONCE"};

struct FeedBackData
{
    uint8_t  SOF = start_of_frame;
    uint8_t  task_mode;
    uint8_t  self_team;     
    uint8_t  _reserved = 0; 
    uint16_t  heat ;
    uint16_t  heat_cap;   
    float    bullet_speed;  
    float    gimbal_roll;   
    float    gimbal_yaw;    
    float    gimbal_pitch;  
    float    yaw_speed;     
    uint8_t __reserved[3] = {0,};  // [0]: MCU fire permit flag from feedback tail.
    uint8_t  EOF = end_of_frame;

    bool mcu_fire_permit() const
    {
        return __reserved[0] != 0;
    }

    uint8_t raw_task_mode() const
    {
        return __reserved[1];
    }

    uint8_t head_raw_task_mode() const
    {
        return __reserved[1];
    }

    uint8_t head_mapped_task_mode() const
    {
        return __reserved[2];
    }

    void set_task_mode_telemetry(uint8_t raw_task_mode, uint8_t mapped_task_mode)
    {
        __reserved[1] = raw_task_mode;
        __reserved[2] = mapped_task_mode;
    }

    enum TASK_MODE
    {
        AUTO_SHOT = (uint8_t)0x01, 
        HIT_BIG_BUFF  = (uint8_t)0x02, 
        HIT_SMALL_BUFF = (uint8_t)0x03,  
        HIT_OUTPOST = (uint8_t)0x04 
    };

    enum SELF_TEAM
    {
        SELF_RED  = (uint8_t)0xAA,
        SELF_BLUE = (uint8_t)0XBB
    };
};


struct GimbalData
{
    float roll;
    float yaw;
    float pitch;
};

// [修复] 删除所有手动写的拷贝构造函数、移动构造函数和赋值操作符
// 直接使用 default，让编译器处理 cv::Mat 的浅拷贝，这才是安全的做法。
struct Frame
{
    cv::Mat srcImg; 
    cv::Mat debugImg; 
    cv::Mat yoloImg; 
    std::uint64_t source_producer_epoch = 0;
    std::uint64_t source_image_seq = 0;
    std::uint64_t source_capture_timestamp_ns = 0;
    GimbalData poseEuler;
    double bullet_speed;
    
    // [Double Track Time System]
    double timeStamp;           // System time (ms) - Used for logic & communication
    double usb_timeStamp = 0;   // Hardware time (ms) - Used for high-precision dt
    double simulator_state_age_s = 0.0;
    
    FeedBackData fb;
    std::chrono::high_resolution_clock::time_point startTime;

    // Default constructors handle cv::Mat shallow copies correctly
    Frame() = default;
    Frame(const Frame&) = default;
    Frame(Frame&&) = default;
    Frame& operator=(const Frame&) = default;
    Frame& operator=(Frame&&) = default;

    void setBulletSpeed(double speed)
    {
        this->bullet_speed = speed;
    }
};

struct FrameMeta
{
    GimbalData poseEuler;
    double bullet_speed = 0.0;

    // [Double Track Time System]
    double timeStamp = 0.0;
    double usb_timeStamp = 0.0;
    double simulator_state_age_s = 0.0;

    FeedBackData fb;
    std::chrono::high_resolution_clock::time_point startTime{};

    FrameMeta() = default;
    explicit FrameMeta(const Frame& frame)
        : poseEuler(frame.poseEuler),
          bullet_speed(frame.bullet_speed),
          timeStamp(frame.timeStamp),
          usb_timeStamp(frame.usb_timeStamp),
          simulator_state_age_s(frame.simulator_state_age_s),
          fb(frame.fb),
          startTime(frame.startTime)
    {
    }
};

struct DebugHudLine
{
    std::string id;
    std::string text;
    std::string section = "top_left";
    std::string color = "#e8e8e8";
    int order = 0;
};

struct DebugSeriesSample
{
    std::string id;
    std::string group = "runtime";
    std::string name;
    std::string unit;
    std::string color = "#e8e8e8";
    double value = 0.0;
    double timestamp_ms = 0.0;
};

struct DebugOverlayBox2D
{
    std::string entity_path;
    cv::Rect2f rect;
    std::string label;
    uint32_t rgba = 0xFFFFFFFFu;
    float radius = 1.0f;
    bool show_label = false;
};

struct DebugOverlayPoint2D
{
    std::string entity_path;
    cv::Point2f position;
    std::string label;
    uint32_t rgba = 0xFFFFFFFFu;
    float radius = 3.0f;
    bool show_label = false;
};

struct DebugOverlayLineStrip2D
{
    std::string entity_path;
    std::vector<cv::Point2f> points;
    std::string label;
    uint32_t rgba = 0xFFFFFFFFu;
    float radius = 1.0f;
    bool closed = false;
    bool show_label = false;
};

struct DebugOverlayPoint3D
{
    std::string entity_path;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    std::string label;
    uint32_t rgba = 0xFFFFFFFFu;
    float radius = 0.02f;
    bool show_label = false;
};

struct DebugOverlayLineStrip3D
{
    std::string entity_path;
    std::vector<Eigen::Vector3d> points;
    std::string label;
    uint32_t rgba = 0xFFFFFFFFu;
    float radius = 0.01f;
    bool closed = false;
    bool show_label = false;
};

struct DebugHudSnapshot
{
    // Browser HUD entries should keep stable ids and coarse sections so new
    // fields can be appended without changing the client-side layout contract.
    bool rerun_skip_image = false;
    std::vector<DebugHudLine> lines;
    std::vector<DebugSeriesSample> samples;
    std::vector<DebugOverlayBox2D> boxes2d;
    std::vector<DebugOverlayPoint2D> points2d;
    std::vector<DebugOverlayLineStrip2D> lines2d;
    std::vector<DebugOverlayPoint3D> points3d;
    std::vector<DebugOverlayLineStrip3D> lines3d;

    void clear()
    {
        rerun_skip_image = false;
        lines.clear();
        samples.clear();
        boxes2d.clear();
        points2d.clear();
        lines2d.clear();
        points3d.clear();
        lines3d.clear();
    }

    bool empty() const
    {
        return lines.empty() && samples.empty() && boxes2d.empty() && points2d.empty() &&
            lines2d.empty() && points3d.empty() && lines3d.empty();
    }

    void upsert(
        const std::string& id, const std::string& text, const std::string& section, int order,
        const std::string& color = "#e8e8e8")
    {
        auto it = std::find_if(lines.begin(), lines.end(), [&](const DebugHudLine& line) {
            return line.id == id;
        });
        if (it == lines.end()) {
            lines.push_back(DebugHudLine{id, text, section, color, order});
            return;
        }
        it->text = text;
        it->section = section;
        it->color = color;
        it->order = order;
    }

    void addSample(
        const std::string& id, const std::string& group, const std::string& name,
        double value, double timestamp_ms, const std::string& unit = "",
        const std::string& color = "#e8e8e8")
    {
        if (!std::isfinite(value)) return;
        samples.push_back(DebugSeriesSample{id, group, name, unit, color, value, timestamp_ms});
    }
};

struct DebugFramePacket
{
    cv::Mat srcImg;
    cv::Mat debugImg;
    double timeStamp = 0.0;
    double usb_timeStamp = 0.0;
    FeedBackData fb;
    DebugHudSnapshot hud;

    DebugFramePacket() = default;

    DebugFramePacket(const Frame& frame, bool keep_src, bool keep_debug)
        : timeStamp(frame.timeStamp), usb_timeStamp(frame.usb_timeStamp), fb(frame.fb)
    {
        if (keep_src) srcImg = frame.srcImg;
        if (keep_debug) debugImg = frame.debugImg;
    }
};

struct SolvedArmorsPacket
{
    std::vector<std::shared_ptr<Armor>> armors;
    FrameMeta meta;
};

struct ArmorsAndFrame
{
    std::vector<std::shared_ptr<Armor>> armors;
    Frame frame;
};

struct RobotMsg
{
    int num; 
    double armor_zs[4]; 
    double armor_rs[4]; 
    bool determined[4]; 
    size_t idx = 0; 
    double direction = 0; 

public:
    void init(std::shared_ptr<Armor> armor)
    {
        num = armor->number;
        idx = 0;
        for(size_t i=1;i<4;i++)
        {
            determined[i] = false;
            armor_rs[i] = 0.2;
        }
        determined[idx] = true;
        armor_zs[idx] = armor->armorPosition.z();
        direction = 0;
    }

    void setR(double r)
    {
        armor_rs[idx] = r;
    }

    void updateRobotMsg(std::shared_ptr<Armor> armor)
    {
        determined[idx] = true;
        armor_zs[idx] = armor->armorPosition.z();
    }

    int nextArmorIdx()
    {
        return (idx + (direction<0?1:3)) % 4;
    }
    int nextArmorIdx(int i)
    {
        return (i + (direction<0?1:3)) % 4;
    }

    int preArmorIdx()
    {
        return (idx + (direction<0?3:1))%4;
    }
    int preArmorIdx(int i)
    {
        return (i + (direction<0?3:1))%4;
    }

    void nextArmor()
    {
        idx = nextArmorIdx();
    }

    double getThisArmorZ()
    {
        return armor_zs[idx];
    }

    double getNextArmorZ()
    {
        return armor_zs[nextArmorIdx()];
    }
};

}; // namespace rm

#endif // GENERALDECLARATION_H
