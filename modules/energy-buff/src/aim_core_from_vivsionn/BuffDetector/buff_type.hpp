#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <Eigen/Dense>
#include <limits>
#include <optional>

namespace auto_buff
{

enum PowerRune_type { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };

// 扇叶数据结构
struct FanBlade
{
    cv::Point2f center;               // 扇页中心 (像素坐标)
    std::vector<cv::Point2f> points;  // 关键点 (0:上, 1:左, 2:下, 3:右, 4:R标)
    double angle = 0.0;               // 相对于 R 标的角度 (弧度)
    FanBlade_type type;               // 类型

    bool solved = false;
    Eigen::Vector3d rune_xyz_in_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d rune_ypd_in_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d blade_xyz_in_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d blade_ypd_in_world = Eigen::Vector3d::Zero();

    std::vector<cv::Point2f> pnp_observed_points;           // PnP 输入图像点：上/左/下/右/R
    std::vector<cv::Point2f> pnp_input_reprojected_points;  // 与输入图像点一一对应的重投影点
    std::vector<cv::Point2f> pnp_reprojected_points;        // 完整模型点重投影：上/左/下/右/中心/R
    std::vector<double> pnp_point_errors_px;                // 上/左/下/右/R 的重投影误差
    cv::Point2f pnp_model_center{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()};
    double pnp_reproj_error_px = std::numeric_limits<double>::quiet_NaN();
    double pnp_score = std::numeric_limits<double>::quiet_NaN();
    double pnp_model_center_error_px = std::numeric_limits<double>::quiet_NaN();
    double pnp_model_center_radial_error_px = std::numeric_limits<double>::quiet_NaN();
    double pnp_model_center_tangent_error_px = std::numeric_limits<double>::quiet_NaN();
    int pnp_method = -1;
    std::array<int, 4> pnp_order = {0, 1, 2, 3};

    FanBlade() = default;

    // 简化构造函数
    FanBlade(const std::vector<cv::Point2f>& kpt, cv::Point2f c, FanBlade_type t)
        : center(c), points(kpt), type(t) {}
};

struct RSearchDebug
{
    cv::Point2f yolo_center{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()};
    cv::Point2f prior_center{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()};
    cv::Point2f raw_center{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()};
    cv::Rect roi_rect;
    std::vector<std::vector<cv::Point>> accepted_contour_points;
    std::vector<cv::Point2f> accepted_centers;
    cv::Mat masked_roi;
    cv::Rect template_rect;
    double template_score = 0.0;
    int template_hits = 0;
    double radius = 0.0;
    int total_contours = 0;
    int accepted_count = 0;
    bool used_template = false;
    bool used_hold_center = false;
    bool used_contour_center = false;
};

// 能量机关整体数据结构
struct PowerRune
{
    PowerRune_type type;              // 大符 or 小符
    cv::Point2f r_center;             // R 标中心 (像素坐标)
    std::vector<FanBlade> fanblades;  // 所有检测到的扇叶
    bool target_switched = false;     // Detector 判断本帧已切到新待击打扇叶
    bool switch_deferred = false;     // 旧目标消失但还在等待切换确认
    int selected_phase_index = -1;    // Tracker 关联出的目标相位编号
    double selected_roll_offset = std::numeric_limits<double>::quiet_NaN();
    std::optional<RSearchDebug> r_search_debug = std::nullopt;

    // --- 以下数据由 Solver 解算后填充 ---

    Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();      // R 标世界坐标 (x, y, z)
    Eigen::Vector3d ypd_in_world = Eigen::Vector3d::Zero();      // R 标球坐标 (yaw, pitch, dist)
    Eigen::Vector3d ypr_in_world = Eigen::Vector3d::Zero();      // 姿态 (yaw, pitch, roll)

    Eigen::Vector3d blade_xyz_in_world = Eigen::Vector3d::Zero(); // 击打扇叶中心世界坐标
    Eigen::Vector3d blade_ypd_in_world = Eigen::Vector3d::Zero(); // 击打扇叶中心球坐标

    PowerRune() = default;

    // 辅助函数：获取当前需要击打的目标扇叶
    // (Detector 会确保 fanblades[0] 就是目标)
    FanBlade& target() { return fanblades[0]; }
    const FanBlade& target() const { return fanblades[0]; }

    // 判断是否解算失败 (根据 Solver 结果判断，例如坐标是否为0)
    bool is_unsolve() const {
        return xyz_in_world.isZero(1e-5);
    }
};

} // namespace auto_buff

#endif // BUFF__TYPE_HPP
