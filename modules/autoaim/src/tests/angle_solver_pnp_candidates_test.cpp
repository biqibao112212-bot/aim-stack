#include "AngleSolver.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

std::vector<cv::Point3f> smallArmorPoints()
{
    return {
        cv::Point3f(0.0f, 67.5f, -27.5f),
        cv::Point3f(0.0f, 67.5f, 27.5f),
        cv::Point3f(0.0f, -67.5f, 27.5f),
        cv::Point3f(0.0f, -67.5f, -27.5f)};
}

std::vector<cv::Point2f> makePlanarObservation(
    const std::vector<cv::Point3f>& armor_points,
    const cv::Mat& camera_matrix)
{
    std::vector<cv::Point3f> points_ippe;
    for (const auto& point : armor_points) {
        points_ippe.emplace_back(point.y, point.z, point.x);
    }

    const cv::Mat rvec_ippe = (cv::Mat_<double>(3, 1) << 0.18, -0.31, 0.07);
    const cv::Mat tvec = (cv::Mat_<double>(3, 1) << 45.0, -22.0, 1800.0);
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(
        points_ippe, rvec_ippe, tvec, camera_matrix, cv::Mat(), image_points);
    return image_points;
}

cv::Mat cameraFromTrackerAtExposure(double gimbal_pitch_rad, double gimbal_yaw_rad)
{
    const double cy = std::cos(gimbal_yaw_rad);
    const double sy = std::sin(gimbal_yaw_rad);
    const double cp = std::cos(gimbal_pitch_rad);
    const double sp = std::sin(gimbal_pitch_rad);
    const cv::Mat yaw_inverse =
        (cv::Mat_<double>(3, 3) << cy, 0, -sy, 0, 1, 0, sy, 0, cy);
    const cv::Mat pitch_inverse =
        (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cp, sp, 0, -sp, cp);
    const cv::Mat base =
        (cv::Mat_<double>(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);
    return pitch_inverse * yaw_inverse * base;
}

cv::Mat constrainedArmorRotation(
    double yaw_rad, double tilt_rad, const cv::Mat& camera_from_tracker)
{
    const double cy = std::cos(yaw_rad);
    const double sy = std::sin(yaw_rad);
    const double ct = std::cos(tilt_rad);
    const double st = std::sin(tilt_rad);
    const cv::Mat yaw =
        (cv::Mat_<double>(3, 3) << cy, -sy, 0, sy, cy, 0, 0, 0, 1);
    const cv::Mat pitch =
        (cv::Mat_<double>(3, 3) << ct, 0, st, 0, 1, 0, -st, 0, ct);
    return camera_from_tracker * yaw * pitch;
}

std::vector<cv::Point2f> makeConstrainedObservation(
    const std::vector<cv::Point3f>& armor_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& distortion,
    double yaw_rad,
    double tilt_rad,
    double depth_mm,
    double gimbal_pitch_rad = 0.0,
    double gimbal_yaw_rad = 0.0)
{
    cv::Mat rvec;
    cv::Rodrigues(
        constrainedArmorRotation(
            yaw_rad, tilt_rad,
            cameraFromTrackerAtExposure(gimbal_pitch_rad, gimbal_yaw_rad)),
        rvec);
    const cv::Mat tvec =
        (cv::Mat_<double>(3, 1) << 120.0, -35.0, depth_mm);
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(
        armor_points, rvec, tvec, camera_matrix, distortion, image_points);
    return image_points;
}

void verifyNonzeroExposurePose(
    const std::vector<cv::Point3f>& armor_points,
    const cv::Mat& camera_matrix)
{
    const cv::Mat distortion =
        (cv::Mat_<double>(1, 5) << -0.08, 0.02, 0.0004, -0.0003, 0.0);
    const double yaw_rad = 37.0 * rm::D2R;
    const double tilt_rad = 15.0 * rm::D2R;
    const double gimbal_pitch_rad = 7.0 * rm::D2R;
    const double gimbal_yaw_rad = -11.0 * rm::D2R;
    const double depth_mm = 5000.0;
    const std::vector<cv::Point2f> image_points = makeConstrainedObservation(
        armor_points, camera_matrix, distortion, yaw_rad, tilt_rad, depth_mm,
        gimbal_pitch_rad, gimbal_yaw_rad);

    const auto candidates = rm::AngleSolver::solveParallelJointPnPCandidatesAtExposure(
        armor_points, image_points, camera_matrix, distortion, tilt_rad,
        gimbal_pitch_rad, gimbal_yaw_rad);
    require(!candidates.empty(), "exposure-pose solver returned no synthetic candidate");
    require(std::abs(candidates.front().yaw_rad - yaw_rad) < 0.1 * rm::D2R,
            "exposure-pose solver did not recover chassis yaw");
    require(candidates.front().reprojection_error_px < 1e-3,
            "exposure-pose solver synthetic reprojection error is too large");

    const cv::Mat exact_tvec =
        (cv::Mat_<double>(3, 1) << 120.0, -35.0, depth_mm);
    double max_error_px = 0.0;
    const double exact_error_px =
        rm::AngleSolver::constrainedPoseReprojectionErrorAtExposure(
            armor_points, image_points, camera_matrix, distortion, yaw_rad,
            tilt_rad, gimbal_pitch_rad, gimbal_yaw_rad, exact_tvec, &max_error_px);
    require(exact_error_px < 1e-4 && max_error_px < 1e-4,
            "exposure-pose reprojection diagnostic rejects exact pose");

    const auto zero_pose = rm::AngleSolver::solveParallelJointPnPCandidates(
        armor_points, makeConstrainedObservation(
                          armor_points, camera_matrix, distortion, yaw_rad, tilt_rad,
                          depth_mm),
        camera_matrix, distortion, tilt_rad);
    const auto zero_pose_explicit = rm::AngleSolver::solveParallelJointPnPCandidatesAtExposure(
        armor_points, makeConstrainedObservation(
                          armor_points, camera_matrix, distortion, yaw_rad, tilt_rad,
                          depth_mm),
        camera_matrix, distortion, tilt_rad, 0.0, 0.0);
    require(!zero_pose.empty() && !zero_pose_explicit.empty(),
            "zero-pose solver regression returned no candidate");
    require(std::abs(zero_pose.front().yaw_rad - zero_pose_explicit.front().yaw_rad) <
                1e-9,
            "legacy and explicit zero-pose models diverged");
}

void verifyParallelJointSolver(
    const std::vector<cv::Point3f>& armor_points,
    const cv::Mat& camera_matrix)
{
    const cv::Mat distortion =
        (cv::Mat_<double>(1, 5) << -0.08, 0.02, 0.0004, -0.0003, 0.0);
    const std::vector<double> yaw_deg_cases = {-70.0, -30.0, 0.0, 30.0, 70.0};
    const std::vector<double> depths_mm = {3000.0, 5000.0, 7000.0};
    const double tilt_rad = 15.0 * rm::D2R;
    for (const double depth_mm : depths_mm) {
        for (const double yaw_deg : yaw_deg_cases) {
            const double yaw_rad = yaw_deg * rm::D2R;
            const std::vector<cv::Point2f> image_points = makeConstrainedObservation(
                armor_points, camera_matrix, distortion, yaw_rad, tilt_rad,
                depth_mm);
            const std::vector<rm::ParallelJointPnPCandidate> candidates =
                rm::AngleSolver::solveParallelJointPnPCandidates(
                    armor_points, image_points, camera_matrix, distortion, tilt_rad);
            require(!candidates.empty(), "joint solver returned no synthetic candidate");
            const auto& selected = candidates.front();
            require(selected.selected, "joint solver did not select rank zero");
            require(selected.positive_depth, "joint solver selected nonpositive depth");
            require(
                std::abs(selected.yaw_rad - yaw_rad) < 0.1 * rm::D2R,
                "joint solver did not recover synthetic yaw");
            require(
                selected.reprojection_error_px < 1e-3,
                "joint solver synthetic reprojection error is too large");
            require(
                yaw_deg == 0.0 ||
                    (selected.yaw_sensitivity_valid &&
                     std::isfinite(selected.yaw_sensitivity_deg_per_px) &&
                     selected.yaw_sensitivity_deg_per_px > 0.0),
                "joint solver yaw sensitivity is invalid");

            double max_error_px = 0.0;
            std::vector<double> corner_error_px;
            const cv::Mat exact_tvec =
                (cv::Mat_<double>(3, 1) << 120.0, -35.0, depth_mm);
            const double exact_error_px =
                rm::AngleSolver::constrainedPoseReprojectionError(
                    armor_points, image_points, camera_matrix, distortion,
                    yaw_rad, tilt_rad, exact_tvec, &max_error_px,
                    &corner_error_px);
            require(
                exact_error_px < 1e-4 && max_error_px < 1e-4 &&
                    corner_error_px.size() == armor_points.size(),
                "constrained reprojection diagnostic rejects exact pose");
        }
    }

    std::vector<cv::Point2f> invalid_points = makeConstrainedObservation(
        armor_points, camera_matrix, distortion, 20.0 * rm::D2R,
        15.0 * rm::D2R, 3000.0);
    invalid_points[2].x = std::numeric_limits<float>::quiet_NaN();
    require(
        rm::AngleSolver::solveParallelJointPnPCandidates(
            armor_points, invalid_points, camera_matrix, distortion,
            15.0 * rm::D2R)
            .empty(),
        "joint solver accepted invalid observations");
}

} // namespace

int main()
{
    try {
        const cv::Mat camera_matrix =
            (cv::Mat_<double>(3, 3) << 1280.0, 0.0, 640.0, 0.0, 1275.0, 512.0, 0.0, 0.0, 1.0);
        const std::vector<cv::Point3f> armor_points = smallArmorPoints();
        verifyParallelJointSolver(armor_points, camera_matrix);
        verifyNonzeroExposurePose(armor_points, camera_matrix);
        const std::vector<cv::Point2f> image_points =
            makePlanarObservation(armor_points, camera_matrix);

        std::vector<rm::PnPCandidate> candidates =
            rm::AngleSolver::solvePlanarPnPCandidates(
                armor_points, image_points, camera_matrix);
        std::vector<cv::Point3f> points_ippe;
        for (const auto& point : armor_points) {
            points_ippe.emplace_back(point.y, point.z, point.x);
        }
        std::vector<cv::Mat> raw_rvecs;
        std::vector<cv::Mat> raw_tvecs;
        const int raw_solution_count = cv::solvePnPGeneric(
            points_ippe, image_points, camera_matrix, cv::Mat(), raw_rvecs, raw_tvecs,
            false, cv::SOLVEPNP_IPPE);
        require(raw_solution_count >= 2, "IPPE did not produce its two planar solutions");
        require(
            candidates.size() == static_cast<std::size_t>(raw_solution_count),
            "not every finite IPPE solution was retained");
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            require(candidates[index].id == index, "candidate ids are not deterministic ranks");
            require(candidates[index].selected == (index == 0), "selected flag mismatch");
            require(
                candidates[index].corner_order ==
                    rm::PnPCandidate::CornerOrder::DETECTOR_CANONICAL,
                "candidate corner order is not detector-canonical");
            require(
                candidates[index].polarity == rm::PnPCandidate::Polarity::NOMINAL,
                "candidate incorrectly claims a generated polarity hypothesis");
            require(cv::checkRange(candidates[index].rVec), "candidate rvec is not finite");
            require(cv::checkRange(candidates[index].tVec), "candidate tvec is not finite");
            require(
                std::isfinite(candidates[index].reprojection_error_px),
                "candidate reprojection error is not finite");
            if (index > 0) {
                require(
                    candidates[index - 1].reprojection_error_px <=
                        candidates[index].reprojection_error_px,
                    "candidates are not sorted by reprojection error");
            }
        }

        const std::vector<rm::PnPCandidate> repeated =
            rm::AngleSolver::solvePlanarPnPCandidates(
                armor_points, image_points, camera_matrix);
        require(repeated.size() == candidates.size(), "candidate count is not repeatable");
        for (std::size_t index = 0; index < repeated.size(); ++index) {
            require(
                repeated[index].solver_solution_index == candidates[index].solver_solution_index,
                "candidate ordering is not repeatable");
            require(
                cv::norm(repeated[index].rVec - candidates[index].rVec) < 1e-12 &&
                    cv::norm(repeated[index].tVec - candidates[index].tVec) < 1e-9,
                "candidate pose is not repeatable");
        }

        std::vector<rm::PnPCandidate> copied = candidates;
        require(
            copied.front().rVec.data != candidates.front().rVec.data &&
                copied.front().tVec.data != candidates.front().tVec.data,
            "candidate copy aliases pose matrices");
        const double copied_rvec_value = copied.front().rVec.at<double>(0, 0);
        candidates.front().rVec.at<double>(0, 0) += 1.0;
        require(
            copied.front().rVec.at<double>(0, 0) == copied_rvec_value,
            "candidate copy is not deep-owned");

        rm::AngleSolver solver;
        solver.setCameraIntrinsicsOverride(camera_matrix);
        rm::Armor armor;
        armor.type = rm::ArmorType::SMALL;
        armor.vertex = image_points;
        const double distance = solver.calculateDistance(armor, armor.type);
        require(distance > 0.0 && std::isfinite(distance), "legacy distance is invalid");
        require(!armor.pnp_candidates.empty(), "legacy solve did not publish candidates");
        require(
            cv::norm(armor.rVec - armor.pnp_candidates.front().rVec) < 1e-12 &&
                cv::norm(armor.tVec - armor.pnp_candidates.front().tVec) < 1e-9,
            "legacy selected pose is not candidate zero");

        rm::Armor invalid_armor;
        invalid_armor.type = rm::ArmorType::SMALL;
        invalid_armor.vertex = {
            cv::Point2f(std::numeric_limits<float>::quiet_NaN(), 10.0f)};
        const double invalid_distance = solver.calculateDistance(invalid_armor, invalid_armor.type);
        require(invalid_distance == 0.0, "invalid solve did not return a safe distance");
        require(invalid_armor.pnp_candidates.empty(), "invalid solve retained candidates");
        require(
            cv::checkRange(invalid_armor.rVec) && cv::checkRange(invalid_armor.tVec),
            "invalid solve populated NaNs");

        // Regression from the 2026-07-11 live failure: a 3.64 m OpenCV depth
        // must remain tracker forward distance. The broken transform placed it
        // in tracker z and made FireControl command about -79 degrees pitch.
        const Eigen::Vector3d live_camera_point_mm(111.802147, 175.895243, 3640.185614);
        const double live_pitch_rad = -0.9220832586 * rm::D2R;
        const double live_yaw_rad = -21.48591232 * rm::D2R;
        const Eigen::Vector3d live_tracker_point_mm = rm::cameraPointToTrackerConvention(
            live_camera_point_mm, live_pitch_rad, live_yaw_rad);
        require(
            std::abs(live_tracker_point_mm.x() - 3430.365912) < 1e-3 &&
                std::abs(live_tracker_point_mm.y() - 1230.131067) < 1e-3 &&
                std::abs(live_tracker_point_mm.z() + 117.292072) < 1e-3,
            "camera depth no longer maps to tracker forward/left/up axes");
        require(
            live_tracker_point_mm.x() > 3000.0 && std::abs(live_tracker_point_mm.z()) < 500.0,
            "live armor point would again be interpreted as a target below the robot");
        const Eigen::Vector3d live_round_trip_mm = rm::trackerPointToCameraConvention(
            live_tracker_point_mm, live_pitch_rad, live_yaw_rad);
        require(
            (live_round_trip_mm - live_camera_point_mm).norm() < 1e-9,
            "camera/tracker coordinate conversion is not invertible");

        // Production coordinate contract: compose the exposure optical pose
        // with the calibrated camera->gimbal rigid transform.  R is not a
        // replacement for the tracker axis convention; it recovers ^T R_G.
        solver._gimbal_pose.pitch = -0.9220832586;
        solver._gimbal_pose.yaw = -21.48591232;
        require(solver.cameraGimbalExtrinsicEnabled(), "sim extrinsic is disabled");
        require(solver.cameraGimbalExtrinsicFromConfig(), "sim extrinsic is not calibrated config");
        Eigen::Matrix3d R_tracker_camera;
        const double configured_pitch_rad =
            static_cast<double>(solver._gimbal_pose.pitch) * rm::D2R;
        const double configured_yaw_rad =
            static_cast<double>(solver._gimbal_pose.yaw) * rm::D2R;
        R_tracker_camera.col(0) = rm::cameraPointToTrackerConvention(
            Eigen::Vector3d::UnitX(), configured_pitch_rad, configured_yaw_rad);
        R_tracker_camera.col(1) = rm::cameraPointToTrackerConvention(
            Eigen::Vector3d::UnitY(), configured_pitch_rad, configured_yaw_rad);
        R_tracker_camera.col(2) = rm::cameraPointToTrackerConvention(
            Eigen::Vector3d::UnitZ(), configured_pitch_rad, configured_yaw_rad);
        const Eigen::Matrix3d R_camera_gimbal = solver.cameraToGimbalRotation();
        const Eigen::Vector3d t_camera_gimbal_mm =
            solver.cameraToGimbalTranslationM() * 1000.0;
        const Eigen::Matrix3d R_tracker_gimbal =
            R_tracker_camera * R_camera_gimbal.transpose();
        const Eigen::Vector3d expected_tracker_mm = R_tracker_gimbal *
            (R_camera_gimbal * live_camera_point_mm + t_camera_gimbal_mm);
        const Eigen::Vector3d calibrated_tracker_mm =
            solver.cameraPointToTracker(live_camera_point_mm);
        require(
            (calibrated_tracker_mm - expected_tracker_mm).norm() < 1e-9,
            "calibrated camera->gimbal->tracker composition is incorrect");
        require(
            (solver.trackerPointToCamera(calibrated_tracker_mm) - live_camera_point_mm).norm() < 1e-9,
            "calibrated production coordinate conversion is not invertible");
        require(
            (solver.gimbalPointToCamera(solver.cameraPointToGimbal(live_camera_point_mm)) -
             live_camera_point_mm).norm() < 1e-9,
            "configured camera/gimbal extrinsic is not invertible");

        cv::Mat live_tvec = (cv::Mat_<double>(3, 1) <<
            live_camera_point_mm.x(), live_camera_point_mm.y(), live_camera_point_mm.z());
        require(
            (solver.calculateGimblePointFromTvec(live_tvec) - calibrated_tracker_mm).norm() < 1e-9,
            "PnP tvec does not use the calibrated production transform");
        const cv::Point2f projected = solver.calculateImagePoint(calibrated_tracker_mm);
        std::vector<cv::Point2f> expected_projection;
        cv::projectPoints(
            std::vector<cv::Point3f>{cv::Point3f(0.0f, 0.0f, 0.0f)},
            cv::Mat::zeros(3, 1, CV_64F), live_tvec, camera_matrix,
            solver.configuredDistortionCoeffs(), expected_projection);
        require(
            !expected_projection.empty() && cv::norm(projected - expected_projection.front()) < 1e-3,
            "tracker->camera inverse projection disagrees with cv::projectPoints");

        std::cout << "angle_solver_pnp_candidates_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "angle_solver_pnp_candidates_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
