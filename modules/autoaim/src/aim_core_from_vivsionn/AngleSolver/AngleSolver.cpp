#include "AngleSolver.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

namespace rm
{

namespace
{

constexpr double kArmorTiltRad = 15.0 * D2R;

double armorTiltForNumber(int armor_number)
{
    return armor_number == Armor::LABEL::OUTPOST ? -kArmorTiltRad : kArmorTiltRad;
}

double outpostYawInGimbalFrame(double camera_yaw, double gimbal_pitch, double gimbal_yaw)
{
    const double tilt = armorTiltForNumber(Armor::LABEL::OUTPOST);
    const double cy = std::cos(camera_yaw);
    const double sy = std::sin(camera_yaw);
    const double ct = std::cos(tilt);
    const double st = std::sin(tilt);

    cv::Mat R_pitch_armor =
        (cv::Mat_<double>(3, 3) << ct, 0, st, 0, 1, 0, -st, 0, ct);
    cv::Mat R_yaw_armor =
        (cv::Mat_<double>(3, 3) << cy, -sy, 0, sy, cy, 0, 0, 0, 1);
    cv::Mat R_base = (cv::Mat_<double>(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);

    cv::Mat normal_camera = R_base * R_yaw_armor * R_pitch_armor *
                            (cv::Mat_<double>(3, 1) << 1, 0, 0);

    const double cp = std::cos(gimbal_pitch);
    const double sp = std::sin(gimbal_pitch);
    const double cg = std::cos(gimbal_yaw);
    const double sg = std::sin(gimbal_yaw);
    cv::Mat R_pitch_gimbal =
        (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cp, -sp, 0, sp, cp);
    cv::Mat R_yaw_gimbal =
        (cv::Mat_<double>(3, 3) << cg, 0, sg, 0, 1, 0, -sg, 0, cg);

    cv::Mat normal_abs = R_yaw_gimbal * R_pitch_gimbal * normal_camera;
    const double tracker_x = normal_abs.at<double>(2, 0);
    const double tracker_y = -normal_abs.at<double>(0, 0);
    return std::atan2(tracker_y, tracker_x);
}

double normalYawInGimbalFrame(
    double nx, double ny, double nz, double gimbal_pitch, double gimbal_yaw)
{
    const double cp = std::cos(gimbal_pitch);
    const double sp = std::sin(gimbal_pitch);
    const double cg = std::cos(gimbal_yaw);
    const double sg = std::sin(gimbal_yaw);

    const double pitched_x = nx;
    const double pitched_y = cp * ny - sp * nz;
    const double pitched_z = sp * ny + cp * nz;

    const double abs_x = cg * pitched_x + sg * pitched_z;
    const double abs_z = -sg * pitched_x + cg * pitched_z;
    const double tracker_x = abs_z;
    const double tracker_y = -abs_x;
    return std::atan2(tracker_y, tracker_x);
}

double outpostYawFromRvecInGimbalFrame(
    const cv::Mat& rvec, double gimbal_pitch, double gimbal_yaw)
{
    if (rvec.empty()) return std::numeric_limits<double>::quiet_NaN();

    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    cv::Mat normal_cam = rotation * (cv::Mat_<double>(3, 1) << 1, 0, 0);
    const double normal_norm = cv::norm(normal_cam);
    if (!std::isfinite(normal_norm) || normal_norm <= 1e-9) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    normal_cam /= normal_norm;
    return normalYawInGimbalFrame(
        normal_cam.at<double>(0, 0), normal_cam.at<double>(1, 0),
        normal_cam.at<double>(2, 0), gimbal_pitch, gimbal_yaw);
}

cv::Point2f projectCameraPoint(
    const Eigen::Vector3d& cam_point, const cv::Mat& camera_matrix,
    const cv::Mat& distortion_coeff)
{
    const std::vector<cv::Point3f> object_points = {cv::Point3f(0.0f, 0.0f, 0.0f)};
    const cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    tvec.at<double>(0, 0) = cam_point.x();
    tvec.at<double>(1, 0) = cam_point.y();
    tvec.at<double>(2, 0) = cam_point.z();

    std::vector<cv::Point2f> image_points;
    cv::projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion_coeff, image_points);
    return image_points.empty() ? cv::Point2f() : image_points.front();
}

} // namespace

Eigen::Matrix3d gimbalPoseRotationFromCameraConvention(
    const Eigen::Matrix3d& R_camera2gimbal, double gimbal_pitch_rad, double gimbal_yaw_rad)
{
    Eigen::Matrix3d R_yaw, R_pitch;
    R_yaw << cos(gimbal_yaw_rad), 0, sin(gimbal_yaw_rad), 0, 1, 0, -sin(gimbal_yaw_rad), 0,
        cos(gimbal_yaw_rad);
    R_pitch << 1, 0, 0, 0, cos(gimbal_pitch_rad), -sin(gimbal_pitch_rad), 0,
        sin(gimbal_pitch_rad), cos(gimbal_pitch_rad);
    const Eigen::Matrix3d R_camera_pose = R_yaw * R_pitch;
    return R_camera2gimbal * R_camera_pose * R_camera2gimbal.transpose();
}

Eigen::Matrix3d currentGimbalPoseRotation(const AngleSolver& solver)
{
    return gimbalPoseRotationFromCameraConvention(
        solver.cameraToGimbalRotation(), solver._gimbal_pose.pitch * D2R,
        solver._gimbal_pose.yaw * D2R);
}

Eigen::Vector3d cameraPointToTrackerConvention(
    const Eigen::Vector3d& camera_point, double gimbal_pitch_rad, double gimbal_yaw_rad)
{
    Eigen::Matrix3d yaw_rotation;
    yaw_rotation << std::cos(gimbal_yaw_rad), 0.0, std::sin(gimbal_yaw_rad), 0.0, 1.0, 0.0,
        -std::sin(gimbal_yaw_rad), 0.0, std::cos(gimbal_yaw_rad);
    Eigen::Matrix3d pitch_rotation;
    // The bridge pitch is optical elevation. Stabilizing a camera-frame point
    // into the tracker frame requires the inverse optical pitch rotation.
    const double stabilizing_pitch_rad = -gimbal_pitch_rad;
    pitch_rotation << 1.0, 0.0, 0.0, 0.0, std::cos(stabilizing_pitch_rad),
        -std::sin(stabilizing_pitch_rad), 0.0, std::sin(stabilizing_pitch_rad),
        std::cos(stabilizing_pitch_rad);

    const Eigen::Vector3d posed_camera = yaw_rotation * pitch_rotation * camera_point;
    return Eigen::Vector3d(posed_camera.z(), -posed_camera.x(), -posed_camera.y());
}

Eigen::Vector3d trackerPointToCameraConvention(
    const Eigen::Vector3d& tracker_point, double gimbal_pitch_rad, double gimbal_yaw_rad)
{
    const Eigen::Vector3d posed_camera(
        -tracker_point.y(), -tracker_point.z(), tracker_point.x());
    Eigen::Matrix3d yaw_rotation;
    yaw_rotation << std::cos(gimbal_yaw_rad), 0.0, std::sin(gimbal_yaw_rad), 0.0, 1.0, 0.0,
        -std::sin(gimbal_yaw_rad), 0.0, std::cos(gimbal_yaw_rad);
    Eigen::Matrix3d pitch_rotation;
    const double stabilizing_pitch_rad = -gimbal_pitch_rad;
    pitch_rotation << 1.0, 0.0, 0.0, 0.0, std::cos(stabilizing_pitch_rad),
        -std::sin(stabilizing_pitch_rad), 0.0, std::sin(stabilizing_pitch_rad),
        std::cos(stabilizing_pitch_rad);
    return (yaw_rotation * pitch_rotation).transpose() * posed_camera;
}

AngleSolver::AngleSolver()
{
    _params = std::make_unique<Params>();
    AngleSolverParamsInit();
    _cam_instant_matrix = _ASparams.CAM_MATRIX.clone();
}

void AngleSolver::loadFrame(Frame& frame)
{
    loadMeta(FrameMeta(frame), frame.debugImg);
}

void AngleSolver::loadMeta(const FrameMeta& frame_meta, const cv::Mat& debug_img)
{
    _debugImg = debug_img;
    _gimbal_pose = frame_meta.poseEuler;
    if (_params->reload()) {
        AngleSolverParamsInit();
        _cam_instant_matrix = _ASparams.CAM_MATRIX.clone();
    }
}

void AngleSolver::setCameraIntrinsicsOverride(
    const cv::Mat& camera_matrix, const cv::Mat& distortion)
{
    if (camera_matrix.empty() || camera_matrix.rows != 3 || camera_matrix.cols != 3) {
        return;
    }

    cv::Mat camera_matrix_64;
    camera_matrix.convertTo(camera_matrix_64, CV_64F);
    _ASparams.CAM_MATRIX = camera_matrix_64.clone();
    _cam_instant_matrix = camera_matrix_64.clone();

    if (!distortion.empty()) {
        distortion.convertTo(_ASparams.DISTORTION_COEFF, CV_64F);
    }
}

void AngleSolver::attachDebugHud(DebugHudSnapshot* hud)
{
    _debugHud = hud;
}

Eigen::Vector3d AngleSolver::cameraPointToGimbal(const Eigen::Vector3d& camera_point) const
{
    if (!_camera_gimbal_extrinsic_enabled) return camera_point;
    return _R_camera2gimbal * camera_point + _t_camera2gimbal_m * 1000.0;
}

Eigen::Vector3d AngleSolver::gimbalPointToCamera(const Eigen::Vector3d& gimbal_point) const
{
    if (!_camera_gimbal_extrinsic_enabled) return gimbal_point;
    return _R_camera2gimbal.transpose() * (gimbal_point - _t_camera2gimbal_m * 1000.0);
}

Eigen::Matrix3d AngleSolver::cameraToGimbalRotation() const
{
    return _camera_gimbal_extrinsic_enabled ? _R_camera2gimbal : Eigen::Matrix3d::Identity();
}

Eigen::Vector3d AngleSolver::cameraToGimbalTranslationM() const
{
    return _camera_gimbal_extrinsic_enabled ? _t_camera2gimbal_m : Eigen::Vector3d::Zero();
}

bool AngleSolver::cameraGimbalExtrinsicEnabled() const
{
    return _camera_gimbal_extrinsic_enabled;
}

bool AngleSolver::cameraGimbalExtrinsicFromConfig() const
{
    return _camera_gimbal_extrinsic_from_config;
}

double AngleSolver::legacyHeightM() const
{
    return _params->H;
}

double AngleSolver::aimingOffsetCxPx() const
{
    return _params->AIMING_CX;
}

double AngleSolver::aimingOffsetCyPx() const
{
    return _params->AIMING_CY;
}

bool AngleSolver::applyAimingOffsetToIntrinsics() const
{
    return _params->APPLY_AIMING_OFFSET_TO_INTRINSICS;
}

cv::Mat AngleSolver::configuredCameraMatrix() const
{
    return _params->CAMERA_MATRIX.clone();
}

cv::Mat AngleSolver::configuredDistortionCoeffs() const
{
    return _params->RADIAL_DISTORTION.clone();
}

std::vector<std::shared_ptr<Armor>> AngleSolver::solveArmors(std::vector<ArmorForDetect> armors_jun)
{
    if (!_armors.empty()) _armors.clear();

    if (!_debugImg.empty() && _debugImg.channels() == 3) {
        cv::cvtColor(_debugImg, _grayImg, cv::COLOR_BGR2GRAY);
    } else {
        _grayImg.release();
    }

    for (auto& armor_jun : armors_jun) {
        if (!_grayImg.empty()) {
            refineKeypoints(armor_jun.vertex, _grayImg);
        }

        Armor armor(armor_jun);
        armor.rect = cvex::scaleRect(boundingRect(armor.vertex), Vec2f(1.5, 1.5));

        armor.dis = calculateDistance(armor, armor.type);
        armor.armorPosition = calculateGimblePointFromTvec(armor.tVec) / 1000.0;

        armor.hitPosRight = calculateGimblePoint(armor.hitPointR, armor.dis) / 1000.0;
        armor.hitPosLeft = calculateGimblePoint(armor.hitPointL, armor.dis) / 1000.0;
        armor.hitPosUp = calculateGimblePoint(armor.hitPointU, armor.dis) / 1000.0;
        armor.hitPosDown = calculateGimblePoint(armor.hitPointD, armor.dis) / 1000.0;

        std::vector<cv::Point3f> point_3d_of_armor =
            armor.type == ArmorType::LARGE ? _ASparams.POINT_3D_OF_ARMOR_BIG
                                           : _ASparams.POINT_3D_OF_ARMOR_SMALL;
        // `optimizeYaw` already returns the camera-relative yaw convention that the
        // downstream tracker expects. Flipping it here makes the relative yaw drift
        // with gimbal rotation for a static armor.
        armor.yaw_absolute =
            optimizeYaw(point_3d_of_armor, armor.vertex, armor.tVec, armorTiltForNumber(armor.number));
        if (armor.number == Armor::LABEL::OUTPOST) {
            armor.yaw = outpostYawFromRvecInGimbalFrame(
                armor.rVec, _gimbal_pose.pitch * D2R, _gimbal_pose.yaw * D2R);
            if (!std::isfinite(armor.yaw)) {
                armor.yaw = outpostYawInGimbalFrame(
                    armor.yaw_absolute, _gimbal_pose.pitch * D2R, _gimbal_pose.yaw * D2R);
            }
        } else {
            armor.yaw = armor.yaw_absolute - _gimbal_pose.yaw * D2R;
        }
        armor.distanceToImageCenter = calculateDistanceToCenter(armor.center);
        if (armor.number == Armor::LABEL::OUTPOST) {
            const cv::Point2f optical_center(
                _cam_instant_matrix.at<double>(0, 2) - _params->AIMING_CX,
                _cam_instant_matrix.at<double>(1, 2) - _params->AIMING_CY);
            armor.distanceToImageCenter = cv::norm(armor.center - optical_center);
        }

        _armors.push_back(make_shared<Armor>(armor));
    }

    _armors.erase(
        std::remove_if(
            _armors.begin(), _armors.end(),
            [](const shared_ptr<Armor>& armor) {
                return !armor || !armor->armorPosition.allFinite() ||
                       !std::isfinite(armor->dis) || armor->dis <= 0.0;
            }),
        _armors.end());

    return _armors;
}

cv::Vec2f AngleSolver::calculateAngleDev(const cv::Point2f Center_of_armor)
{
    double x = Center_of_armor.x;
    double y = Center_of_armor.y;

    double fx = _cam_instant_matrix.at<double>(0, 0);
    double fy = _cam_instant_matrix.at<double>(1, 1);
    double cx = _cam_instant_matrix.at<double>(0, 2);
    double cy = _cam_instant_matrix.at<double>(1, 2);

    cv::Point2f point;
    std::vector<cv::Point2f> in;
    std::vector<cv::Point2f> out;
    in.push_back(cv::Point2f(x, y));

    undistortPoints(
        in, out, _cam_instant_matrix, _ASparams.DISTORTION_COEFF, noArray(),
        _cam_instant_matrix);
    point = out.front();

    double rxNew = (point.x - cx) / fx;
    double ryNew = (point.y - cy) / fy;

    _xErr = atan(rxNew);
    _yErr = atan(ryNew);

    return cv::Vec2f(_xErr, _yErr);
}

float AngleSolver::calculateDistanceToCenter(const cv::Point2f& image_point)
{
    float cx = _cam_instant_matrix.at<double>(0, 2);
    float cy = _cam_instant_matrix.at<double>(1, 2);
    return cv::norm(image_point - cv::Point2f(cx, cy));
}

Eigen::Vector3d AngleSolver::calculateCamPoint(const cv::Point2f Center_of_armor, double dis)
{
    cv::Vec2f Err = calculateAngleDev(Center_of_armor);

    double tanx = tan(Err[0]);
    double tany = tan(Err[1]);

    double z = sqrt((dis * dis) / (1 + tanx * tanx + tany * tany));
    double x = z * tanx;
    double y = z * tany;

    Eigen::Vector3d camPoint3D;
    camPoint3D << x, y, z;

    return camPoint3D;
}

Eigen::Vector3d AngleSolver::calculateGimblePoint(const cv::Point2f Center_of_armor, double dis)
{
    Eigen::Vector3d camPoint3D = calculateCamPoint(Center_of_armor, dis);
    camPoint3D.y() -= _params->H * 1000.0;
    return cameraPointToTrackerConvention(
        camPoint3D, _gimbal_pose.pitch * D2R, _gimbal_pose.yaw * D2R);
}

double AngleSolver::calculateDistance(Armor& armor, ArmorType objectType)
{
    _point_2d_of_armor = armor.vertex;
    std::vector<cv::Point3f> point_3d_of_armor =
        objectType == ArmorType::LARGE ? _ASparams.POINT_3D_OF_ARMOR_BIG
                                       : _ASparams.POINT_3D_OF_ARMOR_SMALL;

    armor.pnp_candidates = solvePlanarPnPCandidates(
        point_3d_of_armor, _point_2d_of_armor, _cam_instant_matrix,
        _ASparams.DISTORTION_COEFF);

    if (armor.pnp_candidates.empty()) {
        _rVec = cv::Mat::zeros(3, 1, CV_64FC1);
        _tVec = cv::Mat::zeros(3, 1, CV_64FC1);
        armor.rVec = _rVec.clone();
        armor.tVec = _tVec.clone();
        _euclideanDistance = 0.0;
        return _euclideanDistance;
    }

    _rVec = armor.pnp_candidates.front().rVec.clone();
    _tVec = armor.pnp_candidates.front().tVec.clone();
    armor.rVec = _rVec.clone();
    armor.tVec = _tVec.clone();

    _euclideanDistance = calculateGimblePointFromTvec(_tVec).norm();

    return _euclideanDistance;
}

std::vector<PnPCandidate> AngleSolver::solvePlanarPnPCandidates(
    const std::vector<cv::Point3f>& armor_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& distortion_coeffs)
{
    std::vector<PnPCandidate> candidates;
    if (armor_points.size() < 4 || armor_points.size() != image_points.size() ||
        camera_matrix.rows != 3 || camera_matrix.cols != 3 || !cv::checkRange(camera_matrix) ||
        (!distortion_coeffs.empty() && !cv::checkRange(distortion_coeffs))) {
        return candidates;
    }

    for (const auto& point : armor_points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            return candidates;
        }
    }
    for (const auto& point : image_points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) return candidates;
    }

    std::vector<cv::Point3f> points_ippe;
    points_ippe.reserve(armor_points.size());
    for (const auto& point : armor_points) {
        points_ippe.emplace_back(point.y, point.z, point.x);
    }

    std::vector<cv::Mat> rvecs_ippe;
    std::vector<cv::Mat> tvecs;
    int reported_solution_count = 0;
    try {
        reported_solution_count = cv::solvePnPGeneric(
            points_ippe, image_points, camera_matrix, distortion_coeffs, rvecs_ippe, tvecs,
            false, cv::SOLVEPNP_IPPE);
    } catch (const cv::Exception&) {
        return candidates;
    }
    if (reported_solution_count <= 0) return candidates;

    const cv::Mat ippe_from_armor =
        (cv::Mat_<double>(3, 3) << 0, 1, 0, 0, 0, 1, 1, 0, 0);
    const std::size_t solution_count = std::min(
        static_cast<std::size_t>(reported_solution_count),
        std::min(rvecs_ippe.size(), tvecs.size()));
    candidates.reserve(solution_count);

    for (std::size_t solution_index = 0; solution_index < solution_count; ++solution_index) {
        if (rvecs_ippe[solution_index].total() != 3 || tvecs[solution_index].total() != 3) {
            continue;
        }
        cv::Mat rvec_ippe;
        cv::Mat tvec;
        try {
            rvecs_ippe[solution_index].reshape(1, 3).convertTo(rvec_ippe, CV_64F);
            tvecs[solution_index].reshape(1, 3).convertTo(tvec, CV_64F);
        } catch (const cv::Exception&) {
            continue;
        }
        if (!cv::checkRange(rvec_ippe) || !cv::checkRange(tvec)) {
            continue;
        }

        cv::Mat rotation_ippe;
        cv::Mat rotation_armor;
        cv::Mat rvec_armor;
        try {
            cv::Rodrigues(rvec_ippe, rotation_ippe);
            rotation_armor = rotation_ippe * ippe_from_armor;
            cv::Rodrigues(rotation_armor, rvec_armor);
        } catch (const cv::Exception&) {
            continue;
        }
        if (!cv::checkRange(rvec_armor)) continue;

        std::vector<cv::Point2f> projected_points;
        try {
            cv::projectPoints(
                armor_points, rvec_armor, tvec, camera_matrix, distortion_coeffs,
                projected_points);
        } catch (const cv::Exception&) {
            continue;
        }
        if (projected_points.size() != image_points.size()) continue;

        double squared_error_sum = 0.0;
        bool finite_projection = true;
        for (std::size_t point_index = 0; point_index < projected_points.size(); ++point_index) {
            const cv::Point2f& projected = projected_points[point_index];
            if (!std::isfinite(projected.x) || !std::isfinite(projected.y)) {
                finite_projection = false;
                break;
            }
            const cv::Point2f residual = projected - image_points[point_index];
            squared_error_sum += static_cast<double>(residual.x) * residual.x +
                                 static_cast<double>(residual.y) * residual.y;
        }
        if (!finite_projection) continue;

        const double reprojection_error =
            std::sqrt(squared_error_sum / static_cast<double>(projected_points.size()));
        if (!std::isfinite(reprojection_error)) continue;

        PnPCandidate candidate;
        candidate.solver_solution_index = static_cast<std::uint32_t>(solution_index);
        candidate.rVec = rvec_armor.clone();
        candidate.tVec = tvec.clone();
        candidate.reprojection_error_px = reprojection_error;
        candidates.push_back(std::move(candidate));
    }

    auto matrix_element = [](const cv::Mat& vector, int index) {
        return vector.at<double>(index, 0);
    };
    std::sort(
        candidates.begin(), candidates.end(), [&](const PnPCandidate& lhs, const PnPCandidate& rhs) {
            if (lhs.reprojection_error_px != rhs.reprojection_error_px) {
                return lhs.reprojection_error_px < rhs.reprojection_error_px;
            }
            for (int index = 0; index < 3; ++index) {
                const double lhs_value = matrix_element(lhs.tVec, index);
                const double rhs_value = matrix_element(rhs.tVec, index);
                if (lhs_value != rhs_value) return lhs_value < rhs_value;
            }
            for (int index = 0; index < 3; ++index) {
                const double lhs_value = matrix_element(lhs.rVec, index);
                const double rhs_value = matrix_element(rhs.rVec, index);
                if (lhs_value != rhs_value) return lhs_value < rhs_value;
            }
            return lhs.solver_solution_index < rhs.solver_solution_index;
        });

    for (std::size_t index = 0; index < candidates.size(); ++index) {
        candidates[index].id = static_cast<std::uint32_t>(index);
        candidates[index].selected = index == 0;
    }
    return candidates;
}

Eigen::Vector3d AngleSolver::calculateGimblePointFromTvec(const cv::Mat& tvec)
{
    Eigen::Vector3d camPoint3D;
    camPoint3D << tvec.at<double>(0, 0),
        tvec.at<double>(1, 0) - _params->H * 1000.0, tvec.at<double>(2, 0);
    return cameraPointToTrackerConvention(
        camPoint3D, _gimbal_pose.pitch * D2R, _gimbal_pose.yaw * D2R);
}

cv::Point2f AngleSolver::calculateImagePoint(Eigen::Vector3d gimbal_point)
{
    Eigen::Vector3d camPoint3D = trackerPointToCameraConvention(
        gimbal_point, _gimbal_pose.pitch * D2R, _gimbal_pose.yaw * D2R);
    camPoint3D.y() += _params->H * 1000.0;
    return projectCameraPoint(camPoint3D, _cam_instant_matrix, _ASparams.DISTORTION_COEFF);
}

vector<cv::Point2f> AngleSolver::calculateImagePoint(vector<Eigen::Vector3d> gimbal_points)
{
    vector<Point2f> image_points;
    for (auto& gimbal_point : gimbal_points) {
        Eigen::Vector3d camPoint3D = trackerPointToCameraConvention(
            gimbal_point, _gimbal_pose.pitch * D2R, _gimbal_pose.yaw * D2R);
        camPoint3D.y() += _params->H * 1000.0;
        image_points.push_back(
            projectCameraPoint(camPoint3D, _cam_instant_matrix, _ASparams.DISTORTION_COEFF));
    }

    return image_points;
}

double AngleSolver::optimizeYaw(
    const std::vector<cv::Point3f>& points_3d, const std::vector<cv::Point2f>& points_2d,
    const cv::Mat& tvec, double armor_tilt_rad)
{
    double min_error = DBL_MAX;
    double best_yaw_deg = 0.0;

    double p_rad = armor_tilt_rad;
    cv::Mat R_pitch =
        (cv::Mat_<double>(3, 3) << cos(p_rad), 0, sin(p_rad), 0, 1, 0, -sin(p_rad), 0, cos(p_rad));

    cv::Mat R_base = (cv::Mat_<double>(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);

    for (double yaw_deg = -80.0; yaw_deg <= 80.0; yaw_deg += 2.0) {
        double y_rad = yaw_deg * CV_PI / 180.0;
        cv::Mat R_yaw = (cv::Mat_<double>(3, 3)
                             << cos(y_rad), -sin(y_rad), 0, sin(y_rad), cos(y_rad), 0, 0, 0, 1);

        cv::Mat R_total = R_base * R_yaw * R_pitch;
        cv::Mat rvec_search;
        cv::Rodrigues(R_total, rvec_search);

        std::vector<cv::Point2f> projected;
        cv::projectPoints(
            points_3d, rvec_search, tvec, _cam_instant_matrix, _ASparams.DISTORTION_COEFF,
            projected);

        double err = 0;
        for (int i = 0; i < 4; i++) err += cv::norm(projected[i] - points_2d[i]);

        if (err < min_error) {
            min_error = err;
            best_yaw_deg = yaw_deg;
        }
    }

    double coarse_best = best_yaw_deg;
    min_error = DBL_MAX;
    for (double yaw_deg = coarse_best - 2.0; yaw_deg <= coarse_best + 2.0; yaw_deg += 0.1) {
        double y_rad = yaw_deg * CV_PI / 180.0;
        cv::Mat R_yaw = (cv::Mat_<double>(3, 3)
                             << cos(y_rad), -sin(y_rad), 0, sin(y_rad), cos(y_rad), 0, 0, 0, 1);

        cv::Mat R_total = R_base * R_yaw * R_pitch;
        cv::Mat rvec_search;
        cv::Rodrigues(R_total, rvec_search);

        std::vector<cv::Point2f> projected;
        cv::projectPoints(
            points_3d, rvec_search, tvec, _cam_instant_matrix, _ASparams.DISTORTION_COEFF,
            projected);

        double err = 0;
        for (int i = 0; i < 4; i++) err += cv::norm(projected[i] - points_2d[i]);

        if (err < min_error) {
            min_error = err;
            best_yaw_deg = yaw_deg;
        }
    }

    return best_yaw_deg * D2R;
}

void AngleSolver::drawArmor(
    Mat& image, ArmorType type, Eigen::Vector3d center_vec, double yaw, const Scalar& color)
{
    vector<Point3f> points_3d =
        type == ArmorType::LARGE ? _ASparams.POINT_3D_OF_ARMOR_BIG : _ASparams.POINT_3D_OF_ARMOR_SMALL;
    vector<Point2f> points_2d;
    for (Point3f p : points_3d) {
        Eigen::Vector3d light_vec(0, 0, 0);
        light_vec.z() = p.z / 1000.0 * cos(15 * D2R);
        light_vec.x() = -p.y / 1000.0 * sin(yaw) - p.z / 1000.0 * sin(15 * D2R) * cos(yaw);
        light_vec.y() = +p.y / 1000.0 * cos(yaw) - p.z / 1000.0 * sin(15 * D2R) * sin(yaw);
        Point2f pp = calculateImagePoint(light_vec + center_vec);
        points_2d.push_back(pp);
    }
    for (int i = 0; i < 4; i++) {
        line(image, points_2d[i], points_2d[(i + 1) % 4], color, 2);
    }
}

void AngleSolver::drawArmor(
    Mat& image, ArmorType type, vector<Eigen::Vector3d> center_vecs, vector<double> yaws,
    double direction, const Scalar& color)
{
    vector<vector<Point2f>> points_2ds;
    for (size_t i = 0; i < center_vecs.size(); i++) {
        double yaw = yaws[i];
        Eigen::Vector3d center_vec = center_vecs[i];
        vector<Point3f> points_3d =
            type == ArmorType::LARGE ? _ASparams.POINT_3D_OF_ARMOR_BIG
                                     : _ASparams.POINT_3D_OF_ARMOR_SMALL;
        vector<Point2f> points_2d;
        for (Point3f p : points_3d) {
            Eigen::Vector3d light_vec(0, 0, 0);
            if (center_vecs.size() == 3) {
                light_vec.z() = p.z / 1000.0 * cos(15 * D2R);
                light_vec.x() =
                    -p.y / 1000.0 * sin(yaw) - p.z / 1000.0 * sin(15 * D2R) * cos(yaw);
                light_vec.y() =
                    +p.y / 1000.0 * cos(yaw) - p.z / 1000.0 * sin(15 * D2R) * sin(yaw);
            } else {
                light_vec.z() = p.z / 1000.0 * cos(15 * D2R);
                light_vec.x() =
                    -p.y / 1000.0 * sin(yaw) - p.z / 1000.0 * sin(15 * D2R) * cos(yaw);
                light_vec.y() =
                    +p.y / 1000.0 * cos(yaw) - p.z / 1000.0 * sin(15 * D2R) * sin(yaw);
            }
            Point2f pp = calculateImagePoint(light_vec + center_vec);
            points_2d.push_back(pp);
        }

        points_2ds.push_back(points_2d);

        for (int i = 0; i < 4; i++) {
            line(image, points_2d[i], points_2d[(i + 1) % 4], color, 2);
        }
    }

    if (points_2ds.size() == 1) return;

    int size = points_2ds.size();
    for (int i = 0; (size == 4) ? i < size : i < size - 1; i++) {
        if (direction < 0) {
            line(image, points_2ds[i][2], points_2ds[(i + 1) % size][1], cvex::GREEN, 1);
            line(image, points_2ds[i][3], points_2ds[(i + 1) % size][0], cvex::GREEN, 1);
        } else {
            line(image, points_2ds[i][1], points_2ds[(i + 1) % size][2], cvex::GREEN, 1);
            line(image, points_2ds[i][0], points_2ds[(i + 1) % size][3], cvex::GREEN, 1);
        }
    }
}

void AngleSolver::showResults()
{
    if (!_params->DEBUG_SWITCH) return;
    const std::string pose_dbg =
        "pose: [p:" + to_string(_gimbal_pose.pitch) + ", y:" + to_string(_gimbal_pose.yaw) + "]";
    if (_debugHud != nullptr) {
        _debugHud->upsert("angle_solver.pose", pose_dbg, "top_left", 10, "#e68caf");
    } else {
        putText(
            _debugImg, pose_dbg,
            Point(10, 60), FONT_HERSHEY_PLAIN, 1, Scalar(230, 140, 175));
    }

    for (auto& armor : _armors) {
        if (_params->DRAW_TARGET_SWITCH) {
            putText(
                _debugImg, "dis: " + to_string(armor->dis).substr(0, 6) + "mm",
                armor->rect.br() + Point(0, 10), FONT_HERSHEY_PLAIN, 1, cvex::PURPLE, 2);
            putText(
                _debugImg, "yaw: " + to_string(armor->yaw_absolute * R2D).substr(0, 6),
                armor->rect.br() + Point(0, 25), FONT_HERSHEY_PLAIN, 1, cvex::PURPLE, 2);

            Point2f armor_up = armor->armorU;
            Point2f armor_down = armor->armorD;
            Point2f armor_center = armor->center;
            Point2f arrow_tail = armor_center + (armor_down - armor_center) * 0.6f;
            Point2f arrow_head = armor_center + (armor_up - armor_center) * 0.9f;

            arrowedLine(
                _debugImg, arrow_tail, arrow_head, cvex::RED, 2, LINE_AA, 0, 0.25);
            circle(_debugImg, armor_up, 4, cvex::GREEN, -1, LINE_AA);
            circle(_debugImg, armor_down, 4, cvex::RED, -1, LINE_AA);
            putText(
                _debugImg, "U", armor_up + Point2f(4.0f, -4.0f), FONT_HERSHEY_PLAIN, 1,
                cvex::GREEN, 1);
            putText(
                _debugImg, "D", armor_down + Point2f(4.0f, 12.0f), FONT_HERSHEY_PLAIN, 1,
                cvex::RED, 1);
        }
    }

    float cx = _cam_instant_matrix.at<double>(0, 2);
    float cy = _cam_instant_matrix.at<double>(1, 2);
    Point cam_center(cx, cy);
    drawMarker(_debugImg, cam_center, cvex::PURPLE, MARKER_CROSS, 20, 2);
}

void AngleSolver::setSolvedArmorsForDebug(const std::vector<std::shared_ptr<Armor>>& armors)
{
    _armors = armors;
}

void AngleSolver::refineKeypoints(std::vector<cv::Point2f>& corners, const cv::Mat& gray_img)
{
    if (corners.size() != 4 || gray_img.empty()) return;
    if (!_params->SUBPIXEL_REFINE_KEYPOINTS) return;

    auto is_finite_corner = [](const cv::Point2f& point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
    };

    auto is_valid_subpix_corner = [&](const cv::Point2f& point) {
        return is_finite_corner(point) && point.x >= 0 && point.y >= 0 &&
               point.x < gray_img.cols && point.y < gray_img.rows;
    };

    for (const auto& point : corners) {
        if (!is_finite_corner(point)) {
            return;
        }
    }

    std::pair<int, int> lightbars[2] = {{0, 1}, {3, 2}};
    const std::vector<cv::Point2f> original_corners = corners;
    std::vector<cv::Point2f> refined_corners = corners;
    const int threshold = std::max(1, std::min(254, _params->SUBPIXEL_REFINE_THRESHOLD));
    const double max_move_px = std::max(0.0, _params->SUBPIXEL_REFINE_MAX_MOVE_PX);
    const double roi_width_ratio =
        std::max(0.20, std::min(1.20, _params->SUBPIXEL_REFINE_ROI_WIDTH_RATIO));

    for (int i = 0; i < 2; i++) {
        int idx_bot = lightbars[i].first;
        int idx_top = lightbars[i].second;
        cv::Point2f pt_bot = corners[idx_bot];
        cv::Point2f pt_top = corners[idx_top];

        cv::Point2f center = (pt_bot + pt_top) / 2.0f;
        double length = cv::norm(pt_top - pt_bot);
        if (length < 4.0) continue;
        double width = length * roi_width_ratio;

        std::vector<cv::Point2f> bright_points;
        int x_min = std::max(0, (int)(std::min(pt_bot.x, pt_top.x) - width));
        int x_max = std::min(gray_img.cols - 1, (int)(std::max(pt_bot.x, pt_top.x) + width));
        int y_min = std::max(0, (int)(std::min(pt_bot.y, pt_top.y) - length * 0.5));
        int y_max = std::min(gray_img.rows - 1, (int)(std::max(pt_bot.y, pt_top.y) + length * 0.5));

        int local_max = 0;
        for (int y = y_min; y <= y_max; y++) {
            const uchar* row_ptr = gray_img.ptr<uchar>(y);
            for (int x = x_min; x <= x_max; x++) {
                local_max = std::max(local_max, static_cast<int>(row_ptr[x]));
            }
        }
        const int local_threshold = std::min(threshold, std::max(30, local_max - 35));

        for (int y = y_min; y <= y_max; y++) {
            const uchar* row_ptr = gray_img.ptr<uchar>(y);
            for (int x = x_min; x <= x_max; x++) {
                if (row_ptr[x] >= local_threshold) {
                    bright_points.push_back(cv::Point2f(x, y));
                }
            }
        }

        if (bright_points.size() > 10) {
            cv::Mat data_pts = cv::Mat(bright_points.size(), 2, CV_64FC1);
            for (size_t k = 0; k < bright_points.size(); ++k) {
                data_pts.at<double>(k, 0) = bright_points[k].x;
                data_pts.at<double>(k, 1) = bright_points[k].y;
            }
            cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
            cv::Point2f pca_center(
                static_cast<float>(pca_analysis.mean.at<double>(0, 0)),
                static_cast<float>(pca_analysis.mean.at<double>(0, 1)));
            cv::Point2f main_axis(
                pca_analysis.eigenvectors.at<double>(0, 0), pca_analysis.eigenvectors.at<double>(0, 1));

            cv::Point2f dir_orig = pt_top - pt_bot;
            if (main_axis.x * dir_orig.x + main_axis.y * dir_orig.y < 0) {
                main_axis = -main_axis;
            }

            std::vector<double> projections;
            for (const auto& pt : bright_points) {
                projections.push_back((pt.x - pca_center.x) * main_axis.x + (pt.y - pca_center.y) * main_axis.y);
            }
            std::sort(projections.begin(), projections.end());

            double proj_bot = projections[projections.size() * 0.05];
            double proj_top = projections[projections.size() * 0.95];

            refined_corners[idx_bot] = pca_center + main_axis * proj_bot;
            refined_corners[idx_top] = pca_center + main_axis * proj_top;
        }
    }

    std::vector<int> valid_corner_indices;
    std::vector<cv::Point2f> subpix_corners;
    valid_corner_indices.reserve(refined_corners.size());
    subpix_corners.reserve(refined_corners.size());

    for (size_t i = 0; i < refined_corners.size(); ++i) {
        if (!is_valid_subpix_corner(refined_corners[i]) && is_valid_subpix_corner(corners[i])) {
            refined_corners[i] = corners[i];
        }

        if (is_valid_subpix_corner(refined_corners[i])) {
            valid_corner_indices.push_back(static_cast<int>(i));
            subpix_corners.push_back(refined_corners[i]);
        }
    }

    if (!subpix_corners.empty()) {
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
        cv::cornerSubPix(gray_img, subpix_corners, cv::Size(4, 4), cv::Size(-1, -1), criteria);

        for (size_t i = 0; i < subpix_corners.size(); ++i) {
            int corner_index = valid_corner_indices[i];
            if (is_valid_subpix_corner(subpix_corners[i]) &&
                cv::norm(subpix_corners[i] - refined_corners[corner_index]) < 3.0) {
                refined_corners[corner_index] = subpix_corners[i];
            }
        }
    }

    cv::Point2f vec_L = refined_corners[1] - refined_corners[0];
    cv::Point2f vec_R = refined_corners[2] - refined_corners[3];

    double len_L = cv::norm(vec_L);
    double len_R = cv::norm(vec_R);

    if (_params->SUBPIXEL_REFINE_PARALLEL_LIGHTBAR_CONSTRAINT && len_L > 1e-3 && len_R > 1e-3) {
        cv::Point2f dir_L = vec_L / len_L;
        cv::Point2f dir_R = vec_R / len_R;

        double dot_prod = dir_L.x * dir_R.x + dir_L.y * dir_R.y;
        double len_diff = std::abs(len_L - len_R) / std::max(len_L, len_R);

        if (dot_prod > 0.96 && len_diff < 0.20) {
            cv::Point2f avg_dir = (dir_L + dir_R);
            avg_dir /= cv::norm(avg_dir);
            double avg_len = (len_L + len_R) / 2.0;

            cv::Point2f mid_L = (refined_corners[0] + refined_corners[1]) / 2.0;
            cv::Point2f mid_R = (refined_corners[3] + refined_corners[2]) / 2.0;

            cv::Point2f target_L_top = mid_L + avg_dir * (avg_len / 2.0);
            cv::Point2f target_L_bot = mid_L - avg_dir * (avg_len / 2.0);
            cv::Point2f target_R_top = mid_R + avg_dir * (avg_len / 2.0);
            cv::Point2f target_R_bot = mid_R - avg_dir * (avg_len / 2.0);

            double alpha = 0.2;
            refined_corners[0] = refined_corners[0] * (1 - alpha) + target_L_bot * alpha;
            refined_corners[1] = refined_corners[1] * (1 - alpha) + target_L_top * alpha;
            refined_corners[2] = refined_corners[2] * (1 - alpha) + target_R_top * alpha;
            refined_corners[3] = refined_corners[3] * (1 - alpha) + target_R_bot * alpha;
        }
    }

    if (max_move_px > 0.0) {
        for (size_t i = 0; i < refined_corners.size(); ++i) {
            if (!is_valid_subpix_corner(refined_corners[i]) ||
                cv::norm(refined_corners[i] - original_corners[i]) > max_move_px) {
                refined_corners[i] = original_corners[i];
            }
        }
    }

    corners = refined_corners;
}

} // namespace rm
