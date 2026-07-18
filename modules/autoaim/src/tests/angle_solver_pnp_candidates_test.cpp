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

} // namespace

int main()
{
    try {
        const cv::Mat camera_matrix =
            (cv::Mat_<double>(3, 3) << 1280.0, 0.0, 640.0, 0.0, 1275.0, 512.0, 0.0, 0.0, 1.0);
        const std::vector<cv::Point3f> armor_points = smallArmorPoints();
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

        std::cout << "angle_solver_pnp_candidates_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "angle_solver_pnp_candidates_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
