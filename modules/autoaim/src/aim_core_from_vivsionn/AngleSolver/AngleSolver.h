#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "generalDeclaration.h"
#include "opencv_extended.h"
#include "params.h"

using namespace std;
using namespace cv;

namespace rm
{

// Coordinate contract used by the armor tracker and planner:
// OpenCV camera (+x right, +y down, +z forward) ->
// tracker (+x forward, +y left, +z up).
// These functions intentionally make the axis permutation explicit. Do not
// replace them with a calibration rotation unless the tracker-frame contract
// and the live regression test are updated together.
Eigen::Vector3d cameraPointToTrackerConvention(
    const Eigen::Vector3d& camera_point, double gimbal_pitch_rad, double gimbal_yaw_rad);
Eigen::Vector3d trackerPointToCameraConvention(
    const Eigen::Vector3d& tracker_point, double gimbal_pitch_rad, double gimbal_yaw_rad);

class AngleSolver
{
public:
    struct AngleSolverParam
    {
        cv::Mat CAM_MATRIX;
        cv::Mat DISTORTION_COEFF;

        std::vector<cv::Point3f> POINT_3D_OF_ARMOR_BIG = {
            cv::Point3f(0, 112.5, -27.5),
            cv::Point3f(0, 112.5, 27.5),
            cv::Point3f(0, -112.5, 27.5),
            cv::Point3f(0, -112.5, -27.5)};
        std::vector<cv::Point3f> POINT_3D_OF_ARMOR_SMALL = {
            cv::Point3f(0, 67.50, -27.5),
            cv::Point3f(0, 67.50, 27.5),
            cv::Point3f(0, -67.50, 27.5),
            cv::Point3f(0, -67.50, -27.5)};
    } _ASparams;

public:
    AngleSolver();
    void loadFrame(rm::Frame&);
    void loadMeta(const rm::FrameMeta&, const cv::Mat& debug_img = cv::Mat());
    void setCameraIntrinsicsOverride(const cv::Mat& camera_matrix, const cv::Mat& distortion = cv::Mat());
    void attachDebugHud(DebugHudSnapshot* hud);

    std::vector<std::shared_ptr<Armor>> solveArmors(std::vector<ArmorForDetect>);

    cv::Vec2f calculateAngleDev(const cv::Point2f Center_of_armor);
    Eigen::Vector3d calculateCamPoint(const cv::Point2f Cente_of_armor, double dis);
    Eigen::Vector3d calculateGimblePoint(const cv::Point2f Cente_of_armor, double dis);
    Eigen::Vector3d calculateGimblePointFromTvec(const cv::Mat& tvec);
    Eigen::Vector3d cameraPointToGimbal(const Eigen::Vector3d& camera_point) const;
    Eigen::Vector3d gimbalPointToCamera(const Eigen::Vector3d& gimbal_point) const;
    Eigen::Matrix3d cameraToGimbalRotation() const;
    Eigen::Vector3d cameraToGimbalTranslationM() const;
    bool cameraGimbalExtrinsicEnabled() const;
    bool cameraGimbalExtrinsicFromConfig() const;
    double legacyHeightM() const;
    double aimingOffsetCxPx() const;
    double aimingOffsetCyPx() const;
    bool applyAimingOffsetToIntrinsics() const;
    cv::Mat configuredCameraMatrix() const;
    cv::Mat configuredDistortionCoeffs() const;

    double calculateDistance(Armor& armor, ArmorType objectType);
    static std::vector<PnPCandidate> solvePlanarPnPCandidates(
        const std::vector<cv::Point3f>& armor_points,
        const std::vector<cv::Point2f>& image_points,
        const cv::Mat& camera_matrix,
        const cv::Mat& distortion_coeffs = cv::Mat());

    float calculateDistanceToCenter(const cv::Point2f& image_point);
    cv::Point2f calculateImagePoint(Eigen::Vector3d gimbal_point);
    vector<cv::Point2f> calculateImagePoint(vector<Eigen::Vector3d> gimbal_point);

    void refineKeypoints(std::vector<cv::Point2f>& corners, const cv::Mat& gray_img);
    double optimizeYaw(
        const std::vector<cv::Point3f>& points_3d, const std::vector<cv::Point2f>& points_2d,
        const cv::Mat& tvec, double armor_tilt_rad);

    void drawArmor(Mat& image, ArmorType type, Eigen::Vector3d armor_pos, double yaw, const Scalar& color);
    void drawArmor(
        Mat& image, ArmorType type, vector<Eigen::Vector3d> armor_pos, vector<double> yaw,
        double direction, const Scalar& color);

    void showResults();
    void setSolvedArmorsForDebug(const std::vector<std::shared_ptr<Armor>>& armors);

    double _xErr = 0;
    double _yErr = 0;
    double _euclideanDistance = 0;
    cv::Mat _rVec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat _tVec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat _cam_instant_matrix;
    cv::Mat _grayImg;
    cv::Mat _debugImg;
    DebugHudSnapshot* _debugHud = nullptr;
    GimbalData _gimbal_pose;

private:
    std::vector<std::shared_ptr<Armor>> _armors;
    std::unique_ptr<Params> _params;
    std::vector<cv::Point2f> _point_2d_of_armor;
    Eigen::Matrix3d _R_camera2gimbal = Eigen::Matrix3d::Identity();
    Eigen::Vector3d _t_camera2gimbal_m = Eigen::Vector3d::Zero();
    bool _camera_gimbal_extrinsic_enabled = true;
    bool _camera_gimbal_extrinsic_from_config = false;

    void AngleSolverParamsInit()
    {
        _ASparams.CAM_MATRIX = _params->CAMERA_MATRIX.clone();
        _ASparams.DISTORTION_COEFF = _params->RADIAL_DISTORTION.clone();
        if (_params->CAMERA_UPSIDE_DOWN && _params->FLIP_CAMERA_INTRINSICS_WITH_IMAGE &&
            _params->CAMERA_IMAGE_WIDTH > 0 && _params->CAMERA_IMAGE_HEIGHT > 0) {
            _ASparams.CAM_MATRIX.at<double>(0, 2) =
                static_cast<double>(_params->CAMERA_IMAGE_WIDTH - 1) -
                _ASparams.CAM_MATRIX.at<double>(0, 2);
            _ASparams.CAM_MATRIX.at<double>(1, 2) =
                static_cast<double>(_params->CAMERA_IMAGE_HEIGHT - 1) -
                _ASparams.CAM_MATRIX.at<double>(1, 2);
        }
        if (_params->APPLY_AIMING_OFFSET_TO_INTRINSICS) {
            _ASparams.CAM_MATRIX.at<double>(2) += _params->AIMING_CX;
            _ASparams.CAM_MATRIX.at<double>(5) += _params->AIMING_CY;
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                _R_camera2gimbal(r, c) = _params->R_CAMERA2GIMBAL.at<double>(r, c);
            }
            _t_camera2gimbal_m(r) = _params->T_CAMERA2GIMBAL.at<double>(r, 0);
        }
        _camera_gimbal_extrinsic_enabled = _params->CAMERA_GIMBAL_EXTRINSIC_ENABLED;
        _camera_gimbal_extrinsic_from_config = _params->CAMERA_GIMBAL_EXTRINSIC_FROM_CONFIG;
    }
};

} // namespace rm
