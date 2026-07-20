#ifndef PARAMS_H
#define PARAMS_H

#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <timer.h>
#include "runtime_paths.h"

using namespace cv;

namespace
{

struct ParamsCalibrationCache
{
    cv::Mat radial_distortion;
    cv::Mat camera_matrix;
    bool loaded = false;
};

inline const ParamsCalibrationCache& getParamsCalibrationCache(const std::string& yaml_path)
{
    static const ParamsCalibrationCache cache = [yaml_path]() {
        ParamsCalibrationCache value;
        FileStorage fs(yaml_path, FileStorage::READ);
        if (!fs.isOpened()) {
            return value;
        }

        fs["RADIAL_DISTORTION"] >> value.radial_distortion;
        fs["CAMERA_MATRIX"] >> value.camera_matrix;
        value.loaded =
            !value.radial_distortion.empty() &&
            !value.camera_matrix.empty();
        return value;
    }();
    return cache;
}

inline cv::Mat defaultCameraToGimbalRotation()
{
    return (cv::Mat_<double>(3, 3) << 0, 0, 1, -1, 0, 0, 0, -1, 0);
}

inline cv::Mat defaultCameraToGimbalTranslation()
{
    return cv::Mat::zeros(3, 1, CV_64F);
}

inline bool isValidCameraToGimbalExtrinsic(const cv::Mat& rotation, const cv::Mat& translation)
{
    if (rotation.rows != 3 || rotation.cols != 3 || translation.total() != 3 ||
        !cv::checkRange(rotation) || !cv::checkRange(translation)) {
        return false;
    }
    const cv::Mat identity_error = rotation.t() * rotation - cv::Mat::eye(3, 3, CV_64F);
    const double determinant = cv::determinant(rotation);
    const double translation_norm_m = cv::norm(translation);
    return cv::norm(identity_error, cv::NORM_INF) <= 1e-5 &&
        std::abs(determinant - 1.0) <= 1e-5 && translation_norm_m <= 5.0;
}

inline bool readNumericSequence(const cv::FileNode& node, std::vector<double>* values)
{
    if (node.empty() || !node.isSeq() || values == nullptr) return false;
    values->clear();
    for (const auto& item : node) {
        values->push_back(static_cast<double>(item));
    }
    return true;
}

inline cv::Mat readMatrix3x3(
    const cv::FileStorage& fs, const std::vector<std::string>& keys, const cv::Mat& fallback,
    bool* found = nullptr)
{
    for (const auto& key : keys) {
        const cv::FileNode node = fs[key];
        if (node.empty()) continue;

        cv::Mat mat;
        node >> mat;
        if (!mat.empty()) {
            mat.convertTo(mat, CV_64F);
            if (mat.rows == 3 && mat.cols == 3) {
                if (found) *found = true;
                return mat.clone();
            }
        }

        std::vector<double> values;
        if (readNumericSequence(node, &values) && values.size() == 9) {
            if (found) *found = true;
            return cv::Mat(3, 3, CV_64F, values.data()).clone();
        }
    }
    if (found) *found = false;
    return fallback.clone();
}

inline cv::Mat readVector3(
    const cv::FileStorage& fs, const std::vector<std::string>& keys, const cv::Mat& fallback,
    bool* found = nullptr)
{
    for (const auto& key : keys) {
        const cv::FileNode node = fs[key];
        if (node.empty()) continue;

        cv::Mat mat;
        node >> mat;
        if (!mat.empty()) {
            mat.convertTo(mat, CV_64F);
            if (mat.total() == 3) {
                if (found) *found = true;
                return mat.reshape(1, 3).clone();
            }
        }

        std::vector<double> values;
        if (readNumericSequence(node, &values) && values.size() == 3) {
            if (found) *found = true;
            return cv::Mat(3, 1, CV_64F, values.data()).clone();
        }
    }
    if (found) *found = false;
    return fallback.clone();
}

}  // namespace

/**
 * @brief parameters
 */
class Params
{
public:
    bool DEBUG_SWITCH;
    bool DEBUG_DRAW_DETECTOR_OVERLAY = true;
    bool DEBUG_DRAW_SOLVED_ARMOR_OVERLAY = true;
    bool DEBUG_DRAW_OBS_MATCH_LABELS = true;
    bool DEBUG_DRAW_ESTIMATOR_OVERLAY = true;
    bool DEBUG_DRAW_FIRECONTROL_OVERLAY = true;
    bool DEBUG_LOG_MODE1_SELECTION_CSV = false;
    bool DEBUG_LOG_FIRECONTROL_CSV = false;
    bool DEBUG_ATTACK_ALL_ARMOR_COLORS = false;
    bool SUBPIXEL_REFINE_KEYPOINTS = true;
    int SUBPIXEL_REFINE_THRESHOLD = 140;
    double SUBPIXEL_REFINE_MAX_MOVE_PX = 5.0;
    double SUBPIXEL_REFINE_ROI_WIDTH_RATIO = 0.55;
    bool SUBPIXEL_REFINE_PARALLEL_LIGHTBAR_CONSTRAINT = true;
    bool RECORD_SWITCH;
    bool CAMERA_UPSIDE_DOWN;
    bool FLIP_CAMERA_INTRINSICS_WITH_IMAGE = false;
    int CAMERA_IMAGE_WIDTH = 1440;
    int CAMERA_IMAGE_HEIGHT = 1080;
    /* camera */
    int AUTOSHOT_EXPOSURE_TIME;
    int AUTOSHOT_GAIN;
    int OUTPOST_EXPOSURE_TIME;
    int OUTPOST_GAIN;

    /* get target */
    bool DRAW_TARGET_SWITCH;
    int LOST_THRESHOLD;
    int YPD_GEOMETRY_RECOVERY_WINDOW_FRAMES = 24;
    int YPD_GEOMETRY_RECOVERY_COOLDOWN_FRAMES = 12;
    int YPD_GEOMETRY_RECOVERY_MISMATCH_REQUIRED_STREAK = 2;
    int YPD_GEOMETRY_RECOVERY_MIN_MATCHED_COUNT = 2;
    double YPD_GEOMETRY_RECOVERY_Z_SIGMA_THRESHOLD = 3.0;
    double YPD_GEOMETRY_RECOVERY_XY_SIGMA_THRESHOLD = 2.0;
    double YPD_GEOMETRY_RECOVERY_COV_INFLATION_SCALE = 48.0;
    double YPD_GEOMETRY_RECOVERY_MIN_DR_VARIANCE = 2.5e-3;
    double YPD_GEOMETRY_RECOVERY_MIN_H_VARIANCE = 6.25e-4;

    /* aiming */
    /*
                       _ooOoo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      O\  =  /O
                   ____/`---'\____
                 .'  \\|     |//  `.
                /  \\|||  :  |||//  \
               /  _||||| -:- |||||-  \
               |   | \\\  -  /// |   |
               | \_|  ''\-/''  |   |
               \  .-\__  `-`  ___/-. /
             ___`. .'  /-.-\  `. . __
          ."" '<  `.___\_<|>_/___.'  >'"".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `-.   \_ __\ /__ _/   .-` /  /
    ======`-.____`-.___\_____/___.-`____.-'======
                       `=-='
    */
    bool GRAVITY_OFFSET_SWITCH;   //1=on 0=off, including gravity and yaw predict
    double AIMING_CX;     // 偏右调大
    double AIMING_CY;       // 偏下调大
    bool CAMERA_GIMBAL_EXTRINSIC_ENABLED = true;
    bool CAMERA_GIMBAL_EXTRINSIC_FROM_CONFIG = false;
    bool APPLY_AIMING_OFFSET_TO_INTRINSICS = false;
    cv::Mat R_CAMERA2GIMBAL = defaultCameraToGimbalRotation();
    cv::Mat T_CAMERA2GIMBAL = defaultCameraToGimbalTranslation();
    int AUTO_SHOT_SWITCH;
    bool IGNORE_SAMENUM_CONDITION_SWITCH;   // 1=忽略只能追踪同编号装甲板的条件
    int ARMOR_NEUTRAL_GRACE_FRAMES = 20;
    double VELOCITY;// 偏下调小
    float HORIZONTAL_DELAY_TIME;  // 滞后调大
    float SPIN_DELAY_TIME_s;
    bool SPIN_DELAY_TIME_SWITCH;
    double FIRE_YAW_MISS_TOLERANCE_M = 0.055;
    double FIRE_YAW_TOLERANCE_MIN_DEG = 0.9;
    double FIRE_YAW_TOLERANCE_MAX_DEG = 2.4;
    double FIRE_ARMOR_IMPACT_ENTER_ANGLE_DEG = 50.0;
    double FIRE_ARMOR_IMPACT_LEAVE_ANGLE_DEG = 30.0;
    double FIRE_COMMAND_STABLE_RATIO;
    double FIRE_RATE_HZ = 20.0;
    double FIRE_FIRST_SHOT_ADVANCE_MS = 0.0;
    double FIRE_SHOT_WINDOW_PRE_MS = 0.0;
    double FIRE_SHOT_WINDOW_POST_MS = 0.0;
    int FIRE_AUTO_ENTER_SLOT_COUNT = 2;
    int FIRE_AUTO_HOLD_SLOT_COUNT = 1;
    double FIRE_AUTO_MIN_BURST_MS = 70.0;
    double FIRE_AUTO_RESTART_COOLDOWN_MS = 50.0;
    bool FIRE_BLOCK_ON_ARMORJUMP = true;
    int FIRE_ARMORJUMP_BLOCK_FRAMES = 4;
    double CONTROL_LOOP_HZ = 250.0;
    double PLANNER_ARMOR_ENTER_ANGLE_DEG = 50.0;
    double PLANNER_ARMOR_LEAVE_ANGLE_DEG = 30.0;
    int AIM_COMMAND_CTRL_MODE;
    double SECOND_ORDER_CTRL_MODEL_DT_S;
    int SECOND_ORDER_CTRL_HORIZON;
    double SECOND_ORDER_CTRL_TRACK_Q;
    double SECOND_ORDER_CTRL_RATE_Q;
    double SECOND_ORDER_CTRL_COMMAND_Q;
    double SECOND_ORDER_CTRL_DELTA_R;
    double SECOND_ORDER_CTRL_YAW_K = 1.0;
    double SECOND_ORDER_CTRL_PITCH_K = 1.0;
    double SECOND_ORDER_CTRL_YAW_WN_RAD_S;
    double SECOND_ORDER_CTRL_PITCH_WN_RAD_S;
    double SECOND_ORDER_CTRL_YAW_ZETA;
    double SECOND_ORDER_CTRL_PITCH_ZETA;
    double SECOND_ORDER_CTRL_YAW_DELAY_S = 0.0;
    double SECOND_ORDER_CTRL_PITCH_DELAY_S = 0.0;
    double SECOND_ORDER_CTRL_YAW_MAX_RATE_DEG_S;
    double SECOND_ORDER_CTRL_PITCH_MAX_RATE_DEG_S;
    double SECOND_ORDER_CTRL_YAW_MAX_LEAD_DEG;
    double SECOND_ORDER_CTRL_PITCH_MAX_LEAD_DEG;
    double SECOND_ORDER_CTRL_YAW_MAX_STATE_RATE_DEG_S;
    double SECOND_ORDER_CTRL_PITCH_MAX_STATE_RATE_DEG_S;
    double SECOND_ORDER_CTRL_OUTPUT_STAGE_RATIO = 0.0;
    double SECOND_ORDER_CTRL_YAW_FEEDBACK_LPF_ALPHA = 0.87;

    cv::Mat RADIAL_DISTORTION;
    cv::Mat CAMERA_MATRIX;

    rm::Timer timer; // reload 需要

    double last_exp, last_gain; // 用于记录之前的增益
    bool camera_reset_flag = 0;
public:
    Params()
    {
        load();
    }

    /**
     * @brief 动态地从yaml文件中读取参数
     * @return
     */
    bool reload()
    {
        if (timer.calTime(1)) // 每隔1秒load一次
        {
            last_exp = AUTOSHOT_EXPOSURE_TIME;
            last_gain = AUTOSHOT_GAIN;
            load();

            if (last_exp != AUTOSHOT_EXPOSURE_TIME || last_gain != AUTOSHOT_GAIN)
                camera_reset_flag = 1;
            else camera_reset_flag = 0;
            return true;
        }
        return false;
    }

    void load()
    {
        const char* override_yaml = std::getenv("AIM_SIM_PARAM_YAML");
        const std::string yaml_path =
            (override_yaml != nullptr && override_yaml[0] != '\0')
                ? rm::runtime_paths::resolveExistingPath(override_yaml).string()
                : rm::runtime_paths::repoPath("src/param.yaml").string();
        FileStorage fs(yaml_path, FileStorage::READ);
        if (!fs.isOpened())
        {
            printf("未找到yaml文件，请检查路径！%s\n", yaml_path.c_str());
            exit(0);
        }


        fs["DEBUG_SWITCH"] >> DEBUG_SWITCH;
        const FileNode draw_detector_overlay_node = fs["DEBUG_DRAW_DETECTOR_OVERLAY"];
        if (!draw_detector_overlay_node.empty()) {
            draw_detector_overlay_node >> DEBUG_DRAW_DETECTOR_OVERLAY;
        } else {
            DEBUG_DRAW_DETECTOR_OVERLAY = true;
        }
        const FileNode draw_solved_overlay_node = fs["DEBUG_DRAW_SOLVED_ARMOR_OVERLAY"];
        if (!draw_solved_overlay_node.empty()) {
            draw_solved_overlay_node >> DEBUG_DRAW_SOLVED_ARMOR_OVERLAY;
        } else {
            DEBUG_DRAW_SOLVED_ARMOR_OVERLAY = true;
        }
        const FileNode draw_obs_match_labels_node = fs["DEBUG_DRAW_OBS_MATCH_LABELS"];
        if (!draw_obs_match_labels_node.empty()) {
            draw_obs_match_labels_node >> DEBUG_DRAW_OBS_MATCH_LABELS;
        } else {
            DEBUG_DRAW_OBS_MATCH_LABELS = true;
        }
        const FileNode draw_estimator_overlay_node = fs["DEBUG_DRAW_ESTIMATOR_OVERLAY"];
        if (!draw_estimator_overlay_node.empty()) {
            draw_estimator_overlay_node >> DEBUG_DRAW_ESTIMATOR_OVERLAY;
        } else {
            DEBUG_DRAW_ESTIMATOR_OVERLAY = true;
        }
        const FileNode draw_firecontrol_overlay_node = fs["DEBUG_DRAW_FIRECONTROL_OVERLAY"];
        if (!draw_firecontrol_overlay_node.empty()) {
            draw_firecontrol_overlay_node >> DEBUG_DRAW_FIRECONTROL_OVERLAY;
        } else {
            DEBUG_DRAW_FIRECONTROL_OVERLAY = true;
        }
        const FileNode debug_log_mode1_selection_node = fs["DEBUG_LOG_MODE1_SELECTION_CSV"];
        if (!debug_log_mode1_selection_node.empty()) {
            debug_log_mode1_selection_node >> DEBUG_LOG_MODE1_SELECTION_CSV;
        } else {
            DEBUG_LOG_MODE1_SELECTION_CSV = false;
        }
        const FileNode debug_log_firecontrol_node = fs["DEBUG_LOG_FIRECONTROL_CSV"];
        if (!debug_log_firecontrol_node.empty()) {
            debug_log_firecontrol_node >> DEBUG_LOG_FIRECONTROL_CSV;
        } else {
            DEBUG_LOG_FIRECONTROL_CSV = false;
        }
        const FileNode attack_all_colors_node = fs["DEBUG_ATTACK_ALL_ARMOR_COLORS"];
        if (!attack_all_colors_node.empty()) {
            attack_all_colors_node >> DEBUG_ATTACK_ALL_ARMOR_COLORS;
        } else {
            DEBUG_ATTACK_ALL_ARMOR_COLORS = false;
        }
        const FileNode subpixel_refine_node = fs["SUBPIXEL_REFINE_KEYPOINTS"];
        if (!subpixel_refine_node.empty()) {
            subpixel_refine_node >> SUBPIXEL_REFINE_KEYPOINTS;
        } else {
            SUBPIXEL_REFINE_KEYPOINTS = true;
        }
        const FileNode subpixel_threshold_node = fs["SUBPIXEL_REFINE_THRESHOLD"];
        if (!subpixel_threshold_node.empty()) {
            subpixel_threshold_node >> SUBPIXEL_REFINE_THRESHOLD;
        } else {
            SUBPIXEL_REFINE_THRESHOLD = 140;
        }
        const FileNode subpixel_max_move_node = fs["SUBPIXEL_REFINE_MAX_MOVE_PX"];
        if (!subpixel_max_move_node.empty()) {
            subpixel_max_move_node >> SUBPIXEL_REFINE_MAX_MOVE_PX;
        } else {
            SUBPIXEL_REFINE_MAX_MOVE_PX = 5.0;
        }
        const FileNode subpixel_roi_width_node = fs["SUBPIXEL_REFINE_ROI_WIDTH_RATIO"];
        if (!subpixel_roi_width_node.empty()) {
            subpixel_roi_width_node >> SUBPIXEL_REFINE_ROI_WIDTH_RATIO;
        } else {
            SUBPIXEL_REFINE_ROI_WIDTH_RATIO = 0.55;
        }
        const FileNode subpixel_parallel_node =
            fs["SUBPIXEL_REFINE_PARALLEL_LIGHTBAR_CONSTRAINT"];
        if (!subpixel_parallel_node.empty()) {
            subpixel_parallel_node >> SUBPIXEL_REFINE_PARALLEL_LIGHTBAR_CONSTRAINT;
        } else {
            SUBPIXEL_REFINE_PARALLEL_LIGHTBAR_CONSTRAINT = true;
        }
        fs["RECORD_SWITCH"] >> RECORD_SWITCH;
        fs["CAMERA_UPSIDE_DOWN"] >> CAMERA_UPSIDE_DOWN;
        const FileNode flip_intrinsics_node = fs["FLIP_CAMERA_INTRINSICS_WITH_IMAGE"];
        if (!flip_intrinsics_node.empty()) {
            flip_intrinsics_node >> FLIP_CAMERA_INTRINSICS_WITH_IMAGE;
        } else {
            FLIP_CAMERA_INTRINSICS_WITH_IMAGE = false;
        }
        const FileNode camera_image_width_node = fs["CAMERA_IMAGE_WIDTH"];
        if (!camera_image_width_node.empty()) {
            camera_image_width_node >> CAMERA_IMAGE_WIDTH;
        } else {
            CAMERA_IMAGE_WIDTH = 1440;
        }
        const FileNode camera_image_height_node = fs["CAMERA_IMAGE_HEIGHT"];
        if (!camera_image_height_node.empty()) {
            camera_image_height_node >> CAMERA_IMAGE_HEIGHT;
        } else {
            CAMERA_IMAGE_HEIGHT = 1080;
        }
        fs["SPIN_DELAY_TIME_SWITCH"] >> SPIN_DELAY_TIME_SWITCH;

        /* camera */
        fs["AUTOSHOT_EXPOSURE_TIME"] >> AUTOSHOT_EXPOSURE_TIME;
        fs["AUTOSHOT_GAIN"] >> AUTOSHOT_GAIN;

        /* get target */
        fs["DRAW_TARGET_SWITCH"] >> DRAW_TARGET_SWITCH;
        fs["LOST_THRESHOLD"] >> LOST_THRESHOLD;
        const FileNode ypd_recovery_window_node = fs["YPD_GEOMETRY_RECOVERY_WINDOW_FRAMES"];
        if (!ypd_recovery_window_node.empty()) {
            ypd_recovery_window_node >> YPD_GEOMETRY_RECOVERY_WINDOW_FRAMES;
        } else {
            YPD_GEOMETRY_RECOVERY_WINDOW_FRAMES = 24;
        }
        const FileNode ypd_recovery_cooldown_node = fs["YPD_GEOMETRY_RECOVERY_COOLDOWN_FRAMES"];
        if (!ypd_recovery_cooldown_node.empty()) {
            ypd_recovery_cooldown_node >> YPD_GEOMETRY_RECOVERY_COOLDOWN_FRAMES;
        } else {
            YPD_GEOMETRY_RECOVERY_COOLDOWN_FRAMES = 12;
        }
        const FileNode ypd_recovery_streak_node =
            fs["YPD_GEOMETRY_RECOVERY_MISMATCH_REQUIRED_STREAK"];
        if (!ypd_recovery_streak_node.empty()) {
            ypd_recovery_streak_node >> YPD_GEOMETRY_RECOVERY_MISMATCH_REQUIRED_STREAK;
        } else {
            YPD_GEOMETRY_RECOVERY_MISMATCH_REQUIRED_STREAK = 2;
        }
        const FileNode ypd_recovery_min_matched_node =
            fs["YPD_GEOMETRY_RECOVERY_MIN_MATCHED_COUNT"];
        if (!ypd_recovery_min_matched_node.empty()) {
            ypd_recovery_min_matched_node >> YPD_GEOMETRY_RECOVERY_MIN_MATCHED_COUNT;
        } else {
            YPD_GEOMETRY_RECOVERY_MIN_MATCHED_COUNT = 2;
        }
        const FileNode ypd_recovery_z_sigma_node =
            fs["YPD_GEOMETRY_RECOVERY_Z_SIGMA_THRESHOLD"];
        if (!ypd_recovery_z_sigma_node.empty()) {
            ypd_recovery_z_sigma_node >> YPD_GEOMETRY_RECOVERY_Z_SIGMA_THRESHOLD;
        } else {
            YPD_GEOMETRY_RECOVERY_Z_SIGMA_THRESHOLD = 3.0;
        }
        const FileNode ypd_recovery_xy_sigma_node =
            fs["YPD_GEOMETRY_RECOVERY_XY_SIGMA_THRESHOLD"];
        if (!ypd_recovery_xy_sigma_node.empty()) {
            ypd_recovery_xy_sigma_node >> YPD_GEOMETRY_RECOVERY_XY_SIGMA_THRESHOLD;
        } else {
            YPD_GEOMETRY_RECOVERY_XY_SIGMA_THRESHOLD = 2.0;
        }
        const FileNode ypd_recovery_cov_scale_node =
            fs["YPD_GEOMETRY_RECOVERY_COV_INFLATION_SCALE"];
        if (!ypd_recovery_cov_scale_node.empty()) {
            ypd_recovery_cov_scale_node >> YPD_GEOMETRY_RECOVERY_COV_INFLATION_SCALE;
        } else {
            YPD_GEOMETRY_RECOVERY_COV_INFLATION_SCALE = 48.0;
        }
        const FileNode ypd_recovery_min_dr_var_node =
            fs["YPD_GEOMETRY_RECOVERY_MIN_DR_VARIANCE"];
        if (!ypd_recovery_min_dr_var_node.empty()) {
            ypd_recovery_min_dr_var_node >> YPD_GEOMETRY_RECOVERY_MIN_DR_VARIANCE;
        } else {
            YPD_GEOMETRY_RECOVERY_MIN_DR_VARIANCE = 2.5e-3;
        }
        const FileNode ypd_recovery_min_h_var_node =
            fs["YPD_GEOMETRY_RECOVERY_MIN_H_VARIANCE"];
        if (!ypd_recovery_min_h_var_node.empty()) {
            ypd_recovery_min_h_var_node >> YPD_GEOMETRY_RECOVERY_MIN_H_VARIANCE;
        } else {
            YPD_GEOMETRY_RECOVERY_MIN_H_VARIANCE = 6.25e-4;
        }

        /* aiming */
        fs["GRAVITY_OFFSET_SWITCH"] >> GRAVITY_OFFSET_SWITCH;
        fs["AIMING_CX"] >> AIMING_CX;
        fs["AIMING_CY"] >> AIMING_CY;
        const FileNode extrinsic_enabled_node = fs["CAMERA_GIMBAL_EXTRINSIC_ENABLED"];
        if (!extrinsic_enabled_node.empty()) {
            extrinsic_enabled_node >> CAMERA_GIMBAL_EXTRINSIC_ENABLED;
        } else {
            CAMERA_GIMBAL_EXTRINSIC_ENABLED = true;
        }
        const FileNode aim_offset_intrinsics_node = fs["APPLY_AIMING_OFFSET_TO_INTRINSICS"];
        if (!aim_offset_intrinsics_node.empty()) {
            aim_offset_intrinsics_node >> APPLY_AIMING_OFFSET_TO_INTRINSICS;
        } else {
            APPLY_AIMING_OFFSET_TO_INTRINSICS = false;
        }
        bool found_rotation = false;
        bool found_translation = false;
        R_CAMERA2GIMBAL = readMatrix3x3(
            fs, {"R_CAMERA2GIMBAL", "R_camera2gimbal"}, defaultCameraToGimbalRotation(),
            &found_rotation);
        T_CAMERA2GIMBAL = readVector3(
            fs, {"T_CAMERA2GIMBAL", "t_camera2gimbal"}, defaultCameraToGimbalTranslation(),
            &found_translation);
        CAMERA_GIMBAL_EXTRINSIC_FROM_CONFIG = found_rotation && found_translation;
        if (CAMERA_GIMBAL_EXTRINSIC_FROM_CONFIG &&
            !isValidCameraToGimbalExtrinsic(R_CAMERA2GIMBAL, T_CAMERA2GIMBAL)) {
            CAMERA_GIMBAL_EXTRINSIC_FROM_CONFIG = false;
        }
        if (CAMERA_GIMBAL_EXTRINSIC_ENABLED && !CAMERA_GIMBAL_EXTRINSIC_FROM_CONFIG) {
            throw std::runtime_error(
                "CAMERA_GIMBAL_EXTRINSIC_ENABLED requires a valid calibrated R and T");
        }
        fs["AUTO_SHOT_SWITCH"] >> AUTO_SHOT_SWITCH;
        fs["IGNORE_SAMENUM_CONDITION_SWITCH"] >> IGNORE_SAMENUM_CONDITION_SWITCH;
        const FileNode armor_neutral_grace_frames_node = fs["ARMOR_NEUTRAL_GRACE_FRAMES"];
        if (!armor_neutral_grace_frames_node.empty()) {
            armor_neutral_grace_frames_node >> ARMOR_NEUTRAL_GRACE_FRAMES;
        } else {
            ARMOR_NEUTRAL_GRACE_FRAMES = 20;
        }
        if (ARMOR_NEUTRAL_GRACE_FRAMES < 0) ARMOR_NEUTRAL_GRACE_FRAMES = 0;
#ifdef HERO
        fs["V"] >> VELOCITY;
#else
        fs["V"] >> VELOCITY;// 偏下调小
#endif
        fs["HORIZONTAL_DELAY_TIME"] >> HORIZONTAL_DELAY_TIME;
        fs["SPIN_DELAY_TIME"] >> SPIN_DELAY_TIME_s;
        SPIN_DELAY_TIME_s /= 1000.0;
        const FileNode fire_yaw_miss_tolerance_m_node = fs["FIRE_YAW_MISS_TOLERANCE_M"];
        if (!fire_yaw_miss_tolerance_m_node.empty()) {
            fire_yaw_miss_tolerance_m_node >> FIRE_YAW_MISS_TOLERANCE_M;
        } else {
            FIRE_YAW_MISS_TOLERANCE_M = 0.055;
        }
        const FileNode fire_yaw_tolerance_min_deg_node = fs["FIRE_YAW_TOLERANCE_MIN_DEG"];
        if (!fire_yaw_tolerance_min_deg_node.empty()) {
            fire_yaw_tolerance_min_deg_node >> FIRE_YAW_TOLERANCE_MIN_DEG;
        } else {
            FIRE_YAW_TOLERANCE_MIN_DEG = 0.9;
        }
        const FileNode fire_yaw_tolerance_max_deg_node = fs["FIRE_YAW_TOLERANCE_MAX_DEG"];
        if (!fire_yaw_tolerance_max_deg_node.empty()) {
            fire_yaw_tolerance_max_deg_node >> FIRE_YAW_TOLERANCE_MAX_DEG;
        } else {
            FIRE_YAW_TOLERANCE_MAX_DEG = 2.4;
        }
        const FileNode fire_armor_impact_enter_angle_node =
            fs["FIRE_ARMOR_IMPACT_ENTER_ANGLE_DEG"];
        if (!fire_armor_impact_enter_angle_node.empty()) {
            fire_armor_impact_enter_angle_node >> FIRE_ARMOR_IMPACT_ENTER_ANGLE_DEG;
        } else {
            FIRE_ARMOR_IMPACT_ENTER_ANGLE_DEG = 50.0;
        }
        const FileNode fire_armor_impact_leave_angle_node =
            fs["FIRE_ARMOR_IMPACT_LEAVE_ANGLE_DEG"];
        if (!fire_armor_impact_leave_angle_node.empty()) {
            fire_armor_impact_leave_angle_node >> FIRE_ARMOR_IMPACT_LEAVE_ANGLE_DEG;
        } else {
            FIRE_ARMOR_IMPACT_LEAVE_ANGLE_DEG = 30.0;
        }
        fs["FIRE_COMMAND_STABLE_RATIO"] >> FIRE_COMMAND_STABLE_RATIO;
        const FileNode fire_rate_hz_node = fs["FIRE_RATE_HZ"];
        if (!fire_rate_hz_node.empty()) {
            fire_rate_hz_node >> FIRE_RATE_HZ;
        } else {
            FIRE_RATE_HZ = 20.0;
        }
        const FileNode fire_first_shot_advance_ms_node = fs["FIRE_FIRST_SHOT_ADVANCE_MS"];
        if (!fire_first_shot_advance_ms_node.empty()) {
            fire_first_shot_advance_ms_node >> FIRE_FIRST_SHOT_ADVANCE_MS;
        } else {
            FIRE_FIRST_SHOT_ADVANCE_MS = 0.0;
        }
        const FileNode fire_shot_window_pre_ms_node = fs["FIRE_SHOT_WINDOW_PRE_MS"];
        if (!fire_shot_window_pre_ms_node.empty()) {
            fire_shot_window_pre_ms_node >> FIRE_SHOT_WINDOW_PRE_MS;
        } else {
            FIRE_SHOT_WINDOW_PRE_MS = 0.0;
        }
        const FileNode fire_shot_window_post_ms_node = fs["FIRE_SHOT_WINDOW_POST_MS"];
        if (!fire_shot_window_post_ms_node.empty()) {
            fire_shot_window_post_ms_node >> FIRE_SHOT_WINDOW_POST_MS;
        } else {
            FIRE_SHOT_WINDOW_POST_MS = 0.0;
        }
        const FileNode fire_auto_enter_slot_count_node = fs["FIRE_AUTO_ENTER_SLOT_COUNT"];
        if (!fire_auto_enter_slot_count_node.empty()) {
            fire_auto_enter_slot_count_node >> FIRE_AUTO_ENTER_SLOT_COUNT;
        } else {
            FIRE_AUTO_ENTER_SLOT_COUNT = 2;
        }
        const FileNode fire_auto_hold_slot_count_node = fs["FIRE_AUTO_HOLD_SLOT_COUNT"];
        if (!fire_auto_hold_slot_count_node.empty()) {
            fire_auto_hold_slot_count_node >> FIRE_AUTO_HOLD_SLOT_COUNT;
        } else {
            FIRE_AUTO_HOLD_SLOT_COUNT = 1;
        }
        const FileNode fire_auto_min_burst_ms_node = fs["FIRE_AUTO_MIN_BURST_MS"];
        if (!fire_auto_min_burst_ms_node.empty()) {
            fire_auto_min_burst_ms_node >> FIRE_AUTO_MIN_BURST_MS;
        } else {
            FIRE_AUTO_MIN_BURST_MS = 70.0;
        }
        const FileNode fire_auto_restart_cooldown_ms_node =
            fs["FIRE_AUTO_RESTART_COOLDOWN_MS"];
        if (!fire_auto_restart_cooldown_ms_node.empty()) {
            fire_auto_restart_cooldown_ms_node >> FIRE_AUTO_RESTART_COOLDOWN_MS;
        } else {
            FIRE_AUTO_RESTART_COOLDOWN_MS = 50.0;
        }
        const FileNode fire_block_on_armorjump_node = fs["FIRE_BLOCK_ON_ARMORJUMP"];
        if (!fire_block_on_armorjump_node.empty()) {
            fire_block_on_armorjump_node >> FIRE_BLOCK_ON_ARMORJUMP;
        } else {
            FIRE_BLOCK_ON_ARMORJUMP = true;
        }
        const FileNode fire_armorjump_block_frames_node =
            fs["FIRE_ARMORJUMP_BLOCK_FRAMES"];
        if (!fire_armorjump_block_frames_node.empty()) {
            fire_armorjump_block_frames_node >> FIRE_ARMORJUMP_BLOCK_FRAMES;
        } else {
            FIRE_ARMORJUMP_BLOCK_FRAMES = 4;
        }
        const FileNode control_loop_hz_node = fs["CONTROL_LOOP_HZ"];
        if (!control_loop_hz_node.empty()) {
            control_loop_hz_node >> CONTROL_LOOP_HZ;
        } else {
            CONTROL_LOOP_HZ = 250.0;
        }
        const FileNode planner_armor_enter_angle_node = fs["PLANNER_ARMOR_ENTER_ANGLE_DEG"];
        if (!planner_armor_enter_angle_node.empty()) {
            planner_armor_enter_angle_node >> PLANNER_ARMOR_ENTER_ANGLE_DEG;
        } else {
            PLANNER_ARMOR_ENTER_ANGLE_DEG = 50.0;
        }
        const FileNode planner_armor_leave_angle_node = fs["PLANNER_ARMOR_LEAVE_ANGLE_DEG"];
        if (!planner_armor_leave_angle_node.empty()) {
            planner_armor_leave_angle_node >> PLANNER_ARMOR_LEAVE_ANGLE_DEG;
        } else {
            PLANNER_ARMOR_LEAVE_ANGLE_DEG = 30.0;
        }
        fs["AIM_COMMAND_CTRL_MODE"] >> AIM_COMMAND_CTRL_MODE;
        fs["SECOND_ORDER_CTRL_MODEL_DT_S"] >> SECOND_ORDER_CTRL_MODEL_DT_S;
        fs["SECOND_ORDER_CTRL_HORIZON"] >> SECOND_ORDER_CTRL_HORIZON;
        fs["SECOND_ORDER_CTRL_TRACK_Q"] >> SECOND_ORDER_CTRL_TRACK_Q;
        fs["SECOND_ORDER_CTRL_RATE_Q"] >> SECOND_ORDER_CTRL_RATE_Q;
        fs["SECOND_ORDER_CTRL_COMMAND_Q"] >> SECOND_ORDER_CTRL_COMMAND_Q;
        fs["SECOND_ORDER_CTRL_DELTA_R"] >> SECOND_ORDER_CTRL_DELTA_R;
        const FileNode yaw_k_node = fs["SECOND_ORDER_CTRL_YAW_K"];
        if (!yaw_k_node.empty()) {
            yaw_k_node >> SECOND_ORDER_CTRL_YAW_K;
        }
        const FileNode pitch_k_node = fs["SECOND_ORDER_CTRL_PITCH_K"];
        if (!pitch_k_node.empty()) {
            pitch_k_node >> SECOND_ORDER_CTRL_PITCH_K;
        }
        fs["SECOND_ORDER_CTRL_YAW_WN_RAD_S"] >> SECOND_ORDER_CTRL_YAW_WN_RAD_S;
        fs["SECOND_ORDER_CTRL_PITCH_WN_RAD_S"] >> SECOND_ORDER_CTRL_PITCH_WN_RAD_S;
        fs["SECOND_ORDER_CTRL_YAW_ZETA"] >> SECOND_ORDER_CTRL_YAW_ZETA;
        fs["SECOND_ORDER_CTRL_PITCH_ZETA"] >> SECOND_ORDER_CTRL_PITCH_ZETA;
        const FileNode yaw_delay_node = fs["SECOND_ORDER_CTRL_YAW_DELAY_S"];
        if (!yaw_delay_node.empty()) {
            yaw_delay_node >> SECOND_ORDER_CTRL_YAW_DELAY_S;
        }
        const FileNode pitch_delay_node = fs["SECOND_ORDER_CTRL_PITCH_DELAY_S"];
        if (!pitch_delay_node.empty()) {
            pitch_delay_node >> SECOND_ORDER_CTRL_PITCH_DELAY_S;
        }
        fs["SECOND_ORDER_CTRL_YAW_MAX_RATE_DEG_S"] >> SECOND_ORDER_CTRL_YAW_MAX_RATE_DEG_S;
        fs["SECOND_ORDER_CTRL_PITCH_MAX_RATE_DEG_S"] >> SECOND_ORDER_CTRL_PITCH_MAX_RATE_DEG_S;
        fs["SECOND_ORDER_CTRL_YAW_MAX_LEAD_DEG"] >> SECOND_ORDER_CTRL_YAW_MAX_LEAD_DEG;
        fs["SECOND_ORDER_CTRL_PITCH_MAX_LEAD_DEG"] >> SECOND_ORDER_CTRL_PITCH_MAX_LEAD_DEG;
        fs["SECOND_ORDER_CTRL_YAW_MAX_STATE_RATE_DEG_S"] >> SECOND_ORDER_CTRL_YAW_MAX_STATE_RATE_DEG_S;
        fs["SECOND_ORDER_CTRL_PITCH_MAX_STATE_RATE_DEG_S"] >> SECOND_ORDER_CTRL_PITCH_MAX_STATE_RATE_DEG_S;
        const FileNode output_stage_ratio_node = fs["SECOND_ORDER_CTRL_OUTPUT_STAGE_RATIO"];
        if (!output_stage_ratio_node.empty()) {
            output_stage_ratio_node >> SECOND_ORDER_CTRL_OUTPUT_STAGE_RATIO;
        }
        const FileNode yaw_feedback_lpf_alpha_node = fs["SECOND_ORDER_CTRL_YAW_FEEDBACK_LPF_ALPHA"];
        if (!yaw_feedback_lpf_alpha_node.empty()) {
            yaw_feedback_lpf_alpha_node >> SECOND_ORDER_CTRL_YAW_FEEDBACK_LPF_ALPHA;
        }

        fs["OUTPOST_EXPOSURE_TIME"] >> OUTPOST_EXPOSURE_TIME;
        fs["OUTPOST_GAIN"] >> OUTPOST_GAIN;


        /* 标定矩阵是静态配置，首次读取后复用，避免在 TRT 初始化后再次触发 OpenCV matrix 反序列化。 */
        const ParamsCalibrationCache& calibration_cache = getParamsCalibrationCache(yaml_path);
        if (calibration_cache.loaded) {
            RADIAL_DISTORTION = calibration_cache.radial_distortion;
            CAMERA_MATRIX = calibration_cache.camera_matrix;
        } else {
            fs["RADIAL_DISTORTION"] >> RADIAL_DISTORTION;
            fs["CAMERA_MATRIX"] >> CAMERA_MATRIX;
        }
    };
};

#endif // PARAMS_H
