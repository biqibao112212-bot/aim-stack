#include "ypd_angle_tracker.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <initializer_list>
#include <numeric>

namespace RobotEstimator {

namespace {

constexpr int kStateDim = 11;
constexpr int kPrimaryRadiusIndex = 8;
constexpr int kDeltaRadiusIndex = 9;
constexpr int kHeightDiffIndex = 10;
constexpr double kMaxOutpostHeightOffsetM = 0.60;
constexpr int kMotionHistoryCapacity = 128;
constexpr int kUpdateMetricHistoryCapacity = 128;
constexpr double kMinArmorRadiusM = 0.05;
constexpr double kMaxValidArmorRadiusM = 0.50;
constexpr double kOutpostRadiusM = 0.2765;
constexpr double kOutpostRadiusPriorSigmaM = 0.018;
constexpr double kOutpostPlaneToRadialYawOffsetRad = 153.0 * M_PI / 180.0;
constexpr double kReferenceNisThreshold = 0.711;
constexpr double kOutpostHeightStepM = 0.105;
constexpr double kOutpostHeightPhaseSigmaM = 0.035;
constexpr double kOutpostHeightPhaseLockMargin = 9.0;
constexpr double kOutpostHeightPhaseRelockMargin = 36.0;
constexpr int kOutpostHeightPhaseCenterCapacity = 96;
constexpr int kOutpostHeightPhaseMinCenterSamples = 36;
constexpr int kOutpostHeightPhaseMinSamplesPerId = 6;
constexpr int kOutpostHeightPhaseRelockMinStreak = 12;
constexpr int kOutpostHeightPhaseRelockCooldownSamples = 96;
constexpr int kOutpostPriorGateMinUpdates = 8;
constexpr int kOutpostReinitAfterRejectedUpdates = 4;
constexpr double kOutpostPriorNisRejectThreshold = 16.0;
constexpr double kMaxNormalSingleUpdateCenterJumpM = 0.30;
constexpr double kMaxNormalWeakSingleUpdateCenterJumpM = 0.12;
constexpr double kMaxNormalPairUpdateCenterJumpM = 0.36;
constexpr double kMaxNormalSingleUpdateVelocityJumpMps = 3.0;
constexpr double kMaxNormalPlanarSpeedMps = 4.0;
constexpr double kNormalPairYawGateRad = 35.0 * M_PI / 180.0;
constexpr double kNormalPairEarlyCenterGateM = 0.45;
constexpr double kNormalPairWeakCenterGateM = 0.28;
constexpr double kNormalPairCenterGapGateM = 0.08;
constexpr double kNormalPairMinRadiusM = 0.06;
constexpr double kNormalPairMaxRadiusM = 0.40;
constexpr double kNormalVelocityFitWindowS = 1.20;
constexpr int kNormalVelocityFitMinSamples = 4;
constexpr int kNormalVelocityGroupedFitMinSamples = 8;
constexpr int kNormalVelocityGroupedFitMinSamplesPerGroup = 2;
constexpr double kNormalVelocityDeadbandMps = 0.05;
constexpr double kNormalSpinVelocityDeadbandBaseMps = 0.15;
constexpr double kNormalVelocityMinDisplacementM = 0.10;
constexpr double kNormalSpinVelocityMinDisplacementM = 0.25;
constexpr double kNormalVelocityMaxFitRmsM = 0.20;
constexpr double kNormalSpinVelocityMaxFitRmsM = 0.15;
constexpr double kNormalVelocityMaxMps = 3.0;
constexpr double kNormalVelocityMaxStepMps = 0.35;
constexpr double kNormalVelocityFrameYawRateDeadbandRadS = 0.02;
constexpr double kNormalVelocityFrameYawRateMaxRadS = 2.0;
constexpr double kNormalSpinVelocityYawRateGateRadS = 2.0;
constexpr double kNormalSpinVelocityDeadbandMps = 0.55;
constexpr double kNormalVelocityHoldMinPriorSpeedMps = 0.35;
constexpr double kNormalVelocityHoldMinRawSpeedMps = 0.35;
constexpr double kNormalVelocityHoldMinDisplacementM = 0.45;
constexpr double kNormalVelocityHoldMaxRmsM = 0.20;
constexpr double kNormalVelocityHoldDecay = 0.92;
constexpr double kNormalSingleCenterVelocityCenterGateM = 0.50;
constexpr int kNormalSingleCenterVelocityMinUpdates = 8;
constexpr bool kNormalPairGeometryVelocityStateUpdateEnabled = false;
constexpr int kNormalCandidateVelocityFitMinSamples = 8;
constexpr double kNormalCandidateVelocityFitMinTimeSpanS = 0.25;
constexpr double kNormalCandidateVelocityMinDisplacementM = 0.20;
constexpr double kNormalCandidateVelocityMaxFitRmsM = 0.20;
constexpr double kNormalCandidateVelocityHighYawGateRadS = 0.50;
constexpr double kNormalYawRateFitWindowS = 0.70;
constexpr int kNormalYawRateFitMinSamples = 4;
constexpr double kNormalYawRateFitMinTimeSpanS = 0.08;
constexpr double kNormalYawRateFitMaxRmsRad = 0.28;
constexpr double kNormalYawRateDeadbandRadS = 0.08;
constexpr double kNormalYawRateMaxRadS = 4.80;
constexpr double kNormalYawRateMaxStepRadS = 0.45;
constexpr double kNormalYawRatePairBlend = 0.45;
constexpr double kNormalYawRateSingleBlend = 0.30;
constexpr int kPhysicalRejectNone = 0;
constexpr int kPhysicalRejectRadius = 1;
constexpr int kPhysicalRejectCenterJump = 2;
constexpr int kPhysicalRejectVelocityJump = 3;
constexpr int kPhysicalRejectPlanarSpeed = 4;
constexpr int kNormalVelocityFitNotEvaluated = 0;
constexpr int kNormalVelocityFitNotEnoughHistory = 1;
constexpr int kNormalVelocityFitNotEnoughWindowSamples = 2;
constexpr int kNormalVelocityFitBadTimeSpan = 3;
constexpr int kNormalVelocityFitNonFiniteVelocity = 4;
constexpr int kNormalVelocityFitLowDisplacement = 5;
constexpr int kNormalVelocityFitHighRms = 6;
constexpr int kNormalVelocityFitLowSpeed = 7;
constexpr int kNormalVelocityFitSpinLowSpeed = 8;
constexpr int kNormalVelocityFitAccepted = 9;
constexpr int kNormalVelocityFitHeldPrevious = 10;
constexpr int kNormalYawRateFitNotEvaluated = 0;
constexpr int kNormalYawRateFitNotEnoughHistory = 1;
constexpr int kNormalYawRateFitNotEnoughWindowSamples = 2;
constexpr int kNormalYawRateFitBadTimeSpan = 3;
constexpr int kNormalYawRateFitNonFiniteVelocity = 4;
constexpr int kNormalYawRateFitHighRms = 6;
constexpr int kNormalYawRateFitAccepted = 9;
// A locked height phase is part of the discrete id mapping. Re-locking it from
// already matched ids creates a feedback loop when a transient id mistake occurs.
constexpr bool kOutpostHeightPhaseRelockEnabled = false;

int normalSingleVelocityGroupId(int source, int armor_id)
{
    if (source <= 0 || armor_id < 0) return -1;
    return source * 100 + armor_id;
}

int normalPairVelocityGroupId(int id_a, int id_b)
{
    if (id_a < 0 || id_b < 0) return -1;
    const int lo = std::min(id_a, id_b);
    const int hi = std::max(id_a, id_b);
    return 1000 + lo * 4 + hi;
}

constexpr std::array<std::array<int, 3>, 6> kOutpostHeightRankCandidates = {{
    {{1, 0, -1}},
    {{0, -1, 1}},
    {{-1, 1, 0}},
    {{1, -1, 0}},
    {{-1, 0, 1}},
    {{0, 1, -1}},
}};
// The tracker id base can change after resets, so the height phase is inferred
// from the observed center-height consistency across all six permutations.

Eigen::MatrixXd makeDiagonal(const std::initializer_list<double>& values)
{
    Eigen::VectorXd diag(values.size());
    int idx = 0;
    for (double value : values) {
        diag(idx++) = value;
    }
    return diag.asDiagonal();
}

Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz)
{
    const double x = xyz.x();
    const double y = xyz.y();
    const double z = xyz.z();
    const double yaw = std::atan2(y, x);
    const double pitch = std::atan2(z, std::sqrt(x * x + y * y));
    const double distance = std::sqrt(x * x + y * y + z * z);
    return {yaw, pitch, distance};
}

Eigen::MatrixXd xyz2ypdJacobian(const Eigen::Vector3d& xyz)
{
    const double x = xyz.x();
    const double y = xyz.y();
    const double z = xyz.z();
    const double xy_sq = std::max(x * x + y * y, 1e-9);
    const double xy = std::sqrt(xy_sq);
    const double xyz_sq = std::max(x * x + y * y + z * z, 1e-9);
    const double pitch_den = z * z / xy_sq + 1.0;

    const double dyaw_dx = -y / xy_sq;
    const double dyaw_dy = x / xy_sq;

    const double dpitch_dx = -(x * z) / (pitch_den * std::pow(xy_sq, 1.5));
    const double dpitch_dy = -(y * z) / (pitch_den * std::pow(xy_sq, 1.5));
    const double dpitch_dz = 1.0 / (pitch_den * xy);

    const double ddistance_dx = x / std::sqrt(xyz_sq);
    const double ddistance_dy = y / std::sqrt(xyz_sq);
    const double ddistance_dz = z / std::sqrt(xyz_sq);

    Eigen::MatrixXd J(3, 3);
    J << dyaw_dx, dyaw_dy, 0.0,
         dpitch_dx, dpitch_dy, dpitch_dz,
         ddistance_dx, ddistance_dy, ddistance_dz;
    return J;
}

double nisThresholdForDim(int measurement_dim)
{
    (void)measurement_dim;
    return kReferenceNisThreshold;
}

double radiusFromState(const Eigen::VectorXd& x, int armor_num, int id)
{
    if (x.size() < kStateDim) return 0.20;
    const bool use_secondary_radius = (armor_num == 4) && (id == 1 || id == 3);
    return use_secondary_radius ? x(kPrimaryRadiusIndex) + x(kDeltaRadiusIndex)
                                : x(kPrimaryRadiusIndex);
}

bool normalArmorStatePhysical(const Eigen::VectorXd& x, int armor_num)
{
    if (x.size() < kStateDim) return false;
    if (!x.allFinite()) return false;

    const double primary_radius = x(kPrimaryRadiusIndex);
    if (!(primary_radius > kMinArmorRadiusM && primary_radius < kMaxValidArmorRadiusM)) {
        return false;
    }

    if (armor_num == 4) {
        const double secondary_radius = x(kPrimaryRadiusIndex) + x(kDeltaRadiusIndex);
        if (!(secondary_radius > kMinArmorRadiusM &&
              secondary_radius < kMaxValidArmorRadiusM)) {
            return false;
        }
    }

    return true;
}

int normalArmorUpdateRejectionReason(
    const Eigen::VectorXd& prior_state, const Eigen::VectorXd& candidate_state,
    int armor_num, double max_center_jump_m)
{
    if (!normalArmorStatePhysical(candidate_state, armor_num)) return kPhysicalRejectRadius;
    if (prior_state.size() < kStateDim || candidate_state.size() < kStateDim) {
        return kPhysicalRejectRadius;
    }

    const double center_jump =
        std::hypot(candidate_state(0) - prior_state(0),
                   candidate_state(2) - prior_state(2));
    if (center_jump > max_center_jump_m) {
        return kPhysicalRejectCenterJump;
    }

    const double velocity_jump =
        std::hypot(candidate_state(1) - prior_state(1),
                   candidate_state(3) - prior_state(3));
    if (velocity_jump > kMaxNormalSingleUpdateVelocityJumpMps) {
        return kPhysicalRejectVelocityJump;
    }

    const double planar_speed = std::hypot(candidate_state(1), candidate_state(3));
    if (planar_speed > kMaxNormalPlanarSpeedMps) return kPhysicalRejectPlanarSpeed;

    return kPhysicalRejectNone;
}

double armorHeightOffsetFromState(const Eigen::VectorXd& x, int armor_num, int id)
{
    if (x.size() < kStateDim) return 0.0;
    if (armor_num == 4) {
        return (id == 1 || id == 3) ? x(kHeightDiffIndex) : 0.0;
    }
    if (armor_num == 3) {
        if (id == 1) return x(kDeltaRadiusIndex);
        if (id == 2) return x(kHeightDiffIndex);
    }
    return 0.0;
}

double outpostHeightOffsetFromLockedPhase(int phase, int id)
{
    if (phase < 0 || phase >= static_cast<int>(kOutpostHeightRankCandidates.size()) ||
        id < 0 || id >= 3) {
        return 0.0;
    }
    return kOutpostHeightRankCandidates[phase][id] * kOutpostHeightStepM;
}

double armorRadialSign(int armor_num)
{
    return armor_num == 3 ? 1.0 : -1.0;
}

double normalizeAngleValue(double angle)
{
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double medianValue(std::vector<double> values)
{
    values.erase(
        std::remove_if(
            values.begin(), values.end(),
            [](double value) { return !std::isfinite(value); }),
        values.end());
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();

    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double median = values[mid];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        median = 0.5 * (median + values[mid - 1]);
    }
    return median;
}

double radialYawFromObservedYaw(double observed_yaw, int armor_num)
{
    if (armor_num != 3) return observed_yaw;
    return normalizeAngleValue(observed_yaw + kOutpostPlaneToRadialYawOffsetRad);
}

double observedYawFromRadialYaw(double radial_yaw, int armor_num)
{
    if (armor_num != 3) return radial_yaw;
    return normalizeAngleValue(radial_yaw - kOutpostPlaneToRadialYawOffsetRad);
}

double safeNormalizedMagnitude(double value, double variance)
{
    if (!std::isfinite(value) || !std::isfinite(variance) || variance <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return value / std::sqrt(variance);
}

} // namespace

YpdAngleTracker::YpdAngleTracker()
{
    reset();
}

void YpdAngleTracker::reset()
{
    initialized_ = false;
    is_outpost_ = false;
    armor_num_ = 4;
    tracked_id_ = 0;
    update_count_ = 0;
    is_converged_ = false;
    last_nis_ = 0.0;
    last_motion_prior_yaw_rate_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_posterior_yaw_rate_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_prior_vx_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_prior_vy_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_posterior_vx_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_posterior_vy_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_center_update_norm_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_velocity_update_norm_ =
        std::numeric_limits<double>::quiet_NaN();
    last_motion_speed_update_abs_ = std::numeric_limits<double>::quiet_NaN();
    last_physical_rejection_count_ = 0;
    last_physical_rejection_reason_ = kPhysicalRejectNone;
    velocity_update_history_.clear();
    motion_history_.clear();
    normal_center_velocity_history_.clear();
    normal_single_center_candidate_history_.clear();
    normal_yaw_observation_history_.clear();
    resetNormalObservationDiagnostics();
    tracker_time_sec_ = 0.0;
    recent_nis_failures_.clear();
    recent_nis_failures_.push_back(0);
    last_batch_match_ids_.clear();
    last_observation_diagnostics_.clear();
    clearGeometryRecoveryHistory();
    resetOutpostHeightPhase();
    x_ = Eigen::VectorXd::Zero(kStateDim);
    P_ = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
    I_ = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
}

void YpdAngleTracker::resetNormalObservationDiagnostics()
{
    last_normal_pair_required_ = 0;
    last_normal_pair_found_ = 0;
    last_normal_update_class_ = 0;
    last_normal_accepted_count_ = 0;
    last_normal_pair_accepted_count_ = 0;
    last_normal_pair_score_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_pair_center_gap_m_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_pair_center_jump_m_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_pair_center_x_m_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_pair_center_y_m_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_single_center_count_ = 0;
    last_normal_single_center_x_m_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_single_center_y_m_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_history_size_ =
        static_cast<int>(normal_center_velocity_history_.size());
    last_normal_velocity_observation_source_ = 0;
    last_normal_velocity_sample_t_s_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_sample_x_m_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_sample_y_m_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_sample_frame_yaw_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_sample_group_id_ = -1;
    last_normal_velocity_fit_sample_count_ = 0;
    last_normal_velocity_fit_accepted_ = 0;
    last_normal_velocity_fit_reject_reason_ = kNormalVelocityFitNotEvaluated;
    last_normal_velocity_fit_time_span_s_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_net_displacement_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_raw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_raw_vx_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_raw_vy_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_yaw_rate_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_yaw_mean_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_yaw_span_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_pair_sample_count_ = 0;
    last_normal_velocity_fit_single_sample_count_ = 0;
    last_normal_velocity_fit_group_count_ = 0;
    last_normal_velocity_fit_grouped_used_ = 0;
    last_normal_velocity_fit_grouped_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_grouped_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_rot_comp_used_ = 0;
    last_normal_velocity_fit_rot_comp_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_pos_yaw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_pos_yaw_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_neg_yaw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_neg_yaw_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_applied_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_applied_vx_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_applied_vy_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_rate_history_size_ =
        static_cast<int>(normal_yaw_observation_history_.size());
    last_normal_yaw_rate_observation_source_ = 0;
    last_normal_yaw_observation_count_ = 0;
    last_normal_yaw_observation_t_s_ = std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_observation_raw_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_observation_unwrapped_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_rate_fit_sample_count_ = 0;
    last_normal_yaw_rate_fit_accepted_ = 0;
    last_normal_yaw_rate_fit_reject_reason_ = kNormalYawRateFitNotEvaluated;
    last_normal_yaw_rate_fit_time_span_s_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_rate_fit_rms_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_rate_fit_raw_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_rate_fit_applied_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();
}

double YpdAngleTracker::normalizeAngle(double angle) const
{
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void YpdAngleTracker::resetOutpostHeightPhase()
{
    outpost_height_phase_scores_.fill(0.0);
    outpost_height_phase_observations_ = 0;
    outpost_height_phase_ = -1;
    last_outpost_height_id_ = -1;
    last_outpost_height_z_ = std::numeric_limits<double>::quiet_NaN();
    outpost_height_phase_relock_candidate_ = -1;
    outpost_height_phase_relock_streak_ = 0;
    outpost_height_phase_relock_cooldown_ = 0;
    outpost_height_phase_id_samples_.clear();
    outpost_height_phase_id_counts_.fill(0);
    for (auto& samples : outpost_height_phase_center_samples_) {
        samples.clear();
    }
}

bool YpdAngleTracker::outpostHeightPhaseLocked() const
{
    return armor_num_ == 3 && outpost_height_phase_ >= 0;
}

double YpdAngleTracker::outpostHeightOffsetForId(int id) const
{
    if (!outpostHeightPhaseLocked()) return 0.0;
    return outpostHeightOffsetFromLockedPhase(outpost_height_phase_, id);
}

void YpdAngleTracker::applyLockedOutpostHeightOffsets()
{
    if (!outpostHeightPhaseLocked() || x_.size() < kStateDim) return;
    x_(kDeltaRadiusIndex) = outpostHeightOffsetForId(1);
    x_(kHeightDiffIndex) = outpostHeightOffsetForId(2);
    P_(kDeltaRadiusIndex, kDeltaRadiusIndex) = 1e-6;
    P_(kHeightDiffIndex, kHeightDiffIndex) = 1e-6;
}

double YpdAngleTracker::outpostHeightPhaseScoreFromSamples(
    int phase, double* center_z) const
{
    if (center_z != nullptr) {
        *center_z = std::numeric_limits<double>::quiet_NaN();
    }
    if (phase < 0 || phase >= static_cast<int>(outpost_height_phase_center_samples_.size())) {
        return DBL_MAX;
    }

    std::vector<double> centers(
        outpost_height_phase_center_samples_[phase].begin(),
        outpost_height_phase_center_samples_[phase].end());
    const double median_center_z = medianValue(centers);
    if (!std::isfinite(median_center_z)) return DBL_MAX;
    if (center_z != nullptr) {
        *center_z = median_center_z;
    }

    const double sigma_sq = kOutpostHeightPhaseSigmaM * kOutpostHeightPhaseSigmaM;
    double score = 0.0;
    int count = 0;
    for (const double center : centers) {
        if (!std::isfinite(center)) continue;
        const double residual = center - median_center_z;
        score += residual * residual / sigma_sq;
        count++;
    }
    return count > 0 ? score : DBL_MAX;
}

bool YpdAngleTracker::outpostHeightPhaseHasEnoughIdCoverage() const
{
    for (int id = 0; id < 3; ++id) {
        if (outpost_height_phase_id_counts_[id] <
            kOutpostHeightPhaseMinSamplesPerId) {
            return false;
        }
    }
    return true;
}

void YpdAngleTracker::applyOutpostRadiusPrior()
{
    if (!is_outpost_ || x_.size() < kStateDim || P_.rows() < kStateDim ||
        P_.cols() < kStateDim) {
        return;
    }

    const double prior_variance = kOutpostRadiusPriorSigmaM * kOutpostRadiusPriorSigmaM;
    const double innovation_variance = P_(kPrimaryRadiusIndex, kPrimaryRadiusIndex) + prior_variance;
    if (!std::isfinite(innovation_variance) || innovation_variance <= 1e-12) return;

    Eigen::VectorXd K = P_.col(kPrimaryRadiusIndex) / innovation_variance;
    x_ += K * (kOutpostRadiusM - x_(kPrimaryRadiusIndex));
    x_(6) = normalizeAngle(x_(6));

    Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
    I_KH.col(kPrimaryRadiusIndex) -= K;
    Eigen::MatrixXd Rk = prior_variance * (K * K.transpose());
    P_ = I_KH * P_ * I_KH.transpose() + Rk;
}

void YpdAngleTracker::updateOutpostHeightPhase(const rm::Armor& armor, int matched_id)
{
    if (armor_num_ != 3 || matched_id < 0 || matched_id >= 3 ||
        !std::isfinite(armor.armorPosition.z())) {
        return;
    }

    if (outpost_height_phase_relock_cooldown_ > 0) {
        outpost_height_phase_relock_cooldown_--;
    }

    outpost_height_phase_id_samples_.push_back(matched_id);
    outpost_height_phase_id_counts_[matched_id]++;
    for (int phase = 0;
         phase < static_cast<int>(kOutpostHeightRankCandidates.size()); ++phase) {
        const double center_z =
            armor.armorPosition.z() -
            outpostHeightOffsetFromLockedPhase(phase, matched_id);
        auto& samples = outpost_height_phase_center_samples_[phase];
        samples.push_back(center_z);
    }
    while (static_cast<int>(outpost_height_phase_id_samples_.size()) >
           kOutpostHeightPhaseCenterCapacity) {
        const int old_id = outpost_height_phase_id_samples_.front();
        outpost_height_phase_id_samples_.pop_front();
        if (old_id >= 0 && old_id < static_cast<int>(outpost_height_phase_id_counts_.size())) {
            outpost_height_phase_id_counts_[old_id] =
                std::max(0, outpost_height_phase_id_counts_[old_id] - 1);
        }
        for (auto& samples : outpost_height_phase_center_samples_) {
            if (!samples.empty()) samples.pop_front();
        }
    }

    outpost_height_phase_observations_ =
        static_cast<int>(outpost_height_phase_id_samples_.size());
    std::array<double, 6> candidate_center_z{};
    candidate_center_z.fill(std::numeric_limits<double>::quiet_NaN());
    for (int phase = 0;
         phase < static_cast<int>(outpost_height_phase_scores_.size()); ++phase) {
        outpost_height_phase_scores_[phase] =
            outpostHeightPhaseScoreFromSamples(phase, &candidate_center_z[phase]);
    }

    if (outpost_height_phase_observations_ >= kOutpostHeightPhaseMinCenterSamples &&
        outpostHeightPhaseHasEnoughIdCoverage()) {
        int best_phase = -1;
        int second_phase = -1;
        for (int phase = 0;
             phase < static_cast<int>(outpost_height_phase_scores_.size()); ++phase) {
            if (best_phase < 0 ||
                outpost_height_phase_scores_[phase] <
                    outpost_height_phase_scores_[best_phase]) {
                second_phase = best_phase;
                best_phase = phase;
            } else if (second_phase < 0 ||
                       outpost_height_phase_scores_[phase] <
                           outpost_height_phase_scores_[second_phase]) {
                second_phase = phase;
            }
        }

        if (best_phase >= 0 && second_phase >= 0 &&
            std::isfinite(candidate_center_z[best_phase])) {
            const double margin =
                outpost_height_phase_scores_[second_phase] -
                outpost_height_phase_scores_[best_phase];
            if (!outpostHeightPhaseLocked() &&
                margin >= kOutpostHeightPhaseLockMargin) {
                outpost_height_phase_ = best_phase;
                x_(4) = candidate_center_z[best_phase];
                applyLockedOutpostHeightOffsets();
                outpost_height_phase_relock_candidate_ = -1;
                outpost_height_phase_relock_streak_ = 0;
            } else if (
                kOutpostHeightPhaseRelockEnabled &&
                outpostHeightPhaseLocked() && best_phase != outpost_height_phase_ &&
                outpost_height_phase_relock_cooldown_ == 0 &&
                margin >= kOutpostHeightPhaseLockMargin &&
                outpost_height_phase_scores_[outpost_height_phase_] -
                    outpost_height_phase_scores_[best_phase] >=
                    kOutpostHeightPhaseRelockMargin) {
                if (outpost_height_phase_relock_candidate_ == best_phase) {
                    outpost_height_phase_relock_streak_++;
                } else {
                    outpost_height_phase_relock_candidate_ = best_phase;
                    outpost_height_phase_relock_streak_ = 1;
                }

                if (outpost_height_phase_relock_streak_ >=
                    kOutpostHeightPhaseRelockMinStreak) {
                    outpost_height_phase_ = best_phase;
                    x_(4) = candidate_center_z[best_phase];
                    applyLockedOutpostHeightOffsets();
                    outpost_height_phase_relock_candidate_ = -1;
                    outpost_height_phase_relock_streak_ = 0;
                    outpost_height_phase_relock_cooldown_ =
                        kOutpostHeightPhaseRelockCooldownSamples;
                    outpost_height_phase_scores_.fill(0.0);
                    outpost_height_phase_observations_ = 0;
                    outpost_height_phase_id_samples_.clear();
                    outpost_height_phase_id_counts_.fill(0);
                    for (auto& samples : outpost_height_phase_center_samples_) {
                        samples.clear();
                    }
                }
            } else {
                outpost_height_phase_relock_candidate_ = -1;
                outpost_height_phase_relock_streak_ = 0;
            }
        }
    }

    last_outpost_height_id_ = matched_id;
    last_outpost_height_z_ = armor.armorPosition.z();
}

void YpdAngleTracker::init(const rm::Armor& armor, int armor_num)
{
    const double observed_radial_yaw = radialYawFromObservedYaw(armor.yaw, armor_num);

    reset();

    armor_num_ = armor_num;
    is_outpost_ = armor.number == rm::Armor::LABEL::OUTPOST;

    double radius = 0.20;
    if (armor_num_ == 3) radius = kOutpostRadiusM;

    const double yaw = observed_radial_yaw;
    const double radial_sign = armorRadialSign(armor_num_);
    const double observed_slot_yaw = yaw;
    const double center_x =
        armor.armorPosition.x() - radial_sign * radius * std::cos(observed_slot_yaw);
    const double center_y =
        armor.armorPosition.y() - radial_sign * radius * std::sin(observed_slot_yaw);
    const double center_z = armor.armorPosition.z();

    x_ = Eigen::VectorXd::Zero(kStateDim);
    x_ << center_x, 0.0, center_y, 0.0, center_z, 0.0, yaw, 0.0, radius, 0.0, 0.0;

    if (armor_num_ == 3) {
        P_ = makeDiagonal({1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0.09, 0.09});
    } else {
        P_ = makeDiagonal({1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1});
    }

    initialized_ = true;
    appendMotionSample();
}

bool YpdAngleTracker::converged()
{
    if (!initialized_) return false;
    const int min_updates = is_outpost_ ? 10 : 3;
    if (update_count_ > min_updates && !diverged()) {
        is_converged_ = true;
    }
    return is_converged_;
}

Eigen::MatrixXd YpdAngleTracker::buildProcessNoise(double dt) const
{
    double v1 = is_outpost_ ? 10.0 : 100.0;
    double v2 = is_outpost_ ? 0.1 : 400.0;
    double a = dt * dt * dt * dt / 4.0;
    double b = dt * dt * dt / 2.0;
    double c = dt * dt;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
    Q(0, 0) = a * v1;
    Q(0, 1) = b * v1;
    Q(1, 0) = b * v1;
    Q(1, 1) = c * v1;
    Q(2, 2) = a * v1;
    Q(2, 3) = b * v1;
    Q(3, 2) = b * v1;
    Q(3, 3) = c * v1;
    Q(4, 4) = a * v1;
    Q(4, 5) = b * v1;
    Q(5, 4) = b * v1;
    Q(5, 5) = c * v1;
    Q(6, 6) = a * v2;
    Q(6, 7) = b * v2;
    Q(7, 6) = b * v2;
    Q(7, 7) = c * v2;
    return Q;
}

void YpdAngleTracker::predict(double dt)
{
    if (!initialized_) return;
    if (dt <= 0.0) dt = 0.006;
    tracker_time_sec_ += dt;

    if (converged() && is_outpost_ && std::abs(x_(7)) > 2.0) {
        x_(7) = x_(7) > 0.0 ? 2.51 : -2.51;
    }

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;

    x_ = F * x_;
    x_(6) = normalizeAngle(x_(6));

    P_ = F * P_ * F.transpose() + buildProcessNoise(dt);
    applyLockedOutpostHeightOffsets();
}

void YpdAngleTracker::setGeometryRecoveryConfig(const GeometryRecoveryConfig& config)
{
    geometry_recovery_config_ = config;
}

void YpdAngleTracker::setFrameYaw(double yaw_rad)
{
    current_frame_yaw_rad_ = std::isfinite(yaw_rad)
        ? yaw_rad
        : std::numeric_limits<double>::quiet_NaN();
}

void YpdAngleTracker::setFrameYawRate(double yaw_rate_rad_s)
{
    current_frame_yaw_rate_rad_s_ = std::isfinite(yaw_rate_rad_s)
        ? yaw_rate_rad_s
        : std::numeric_limits<double>::quiet_NaN();
}

void YpdAngleTracker::clearGeometryRecoveryHistory()
{
    pending_observation_jump_hint_ = false;
    geometry_recovery_window_remaining_ = 0;
    geometry_mismatch_streak_ = 0;
    geometry_recovery_cooldown_ = 0;
    consecutive_rejected_updates_ = 0;
    last_geometry_residual_xy_over_sigma_dr_ = std::numeric_limits<double>::quiet_NaN();
    last_geometry_residual_z_over_sigma_h_ = std::numeric_limits<double>::quiet_NaN();
    geometry_recovery_inflation_count_ = 0;
}

void YpdAngleTracker::noteObservationJump(bool observation_jump)
{
    pending_observation_jump_hint_ =
        pending_observation_jump_hint_ || (initialized_ && observation_jump);
}

Eigen::Vector3d YpdAngleTracker::predictArmorPosition(
    const Eigen::VectorXd& state, int armor_id) const
{
    const double angle = normalizeAngle(state(6) + armor_id * 2.0 * M_PI / armor_num_);
    const double r = radiusFromState(state, armor_num_, armor_id);
    const double radial_sign = armorRadialSign(armor_num_);
    const double armor_x = state(0) + radial_sign * r * std::cos(angle);
    const double armor_y = state(2) + radial_sign * r * std::sin(angle);
    const double height_offset = outpostHeightPhaseLocked()
        ? outpostHeightOffsetForId(armor_id)
        : armorHeightOffsetFromState(state, armor_num_, armor_id);
    const double armor_z = state(4) + height_offset;
    return {armor_x, armor_y, armor_z};
}

double YpdAngleTracker::getArmorRadius(int id) const
{
    return radiusFromState(x_, armor_num_, id);
}

Eigen::Vector4d YpdAngleTracker::getPredictedArmorState(int armor_id) const
{
    const double angle = normalizeAngle(x_(6) + armor_id * 2.0 * M_PI / armor_num_);
    const Eigen::Vector3d xyz = predictArmorPosition(x_, armor_id);
    return {xyz.x(), xyz.y(), xyz.z(), observedYawFromRadialYaw(angle, armor_num_)};
}

std::vector<Eigen::Vector4d> YpdAngleTracker::getPredictedArmorStates() const
{
    std::vector<Eigen::Vector4d> list;
    list.reserve(armor_num_);
    for (int i = 0; i < armor_num_; ++i) {
        list.push_back(getPredictedArmorState(i));
    }
    return list;
}

Eigen::Vector4d YpdAngleTracker::predictArmorMeasurement(
    const Eigen::VectorXd& state, int armor_id) const
{
    const Eigen::Vector3d xyz = predictArmorPosition(state, armor_id);
    const Eigen::Vector3d ypd = xyz2ypd(xyz);
    const double angle = normalizeAngle(state(6) + armor_id * 2.0 * M_PI / armor_num_);
    return {ypd(0), ypd(1), ypd(2), observedYawFromRadialYaw(angle, armor_num_)};
}

Eigen::Vector3d YpdAngleTracker::extractObservationYpd(const rm::Armor& armor) const
{
    if (armor.has_explicit_ypd && armor.ypd.allFinite()) {
        return armor.ypd;
    }
    return xyz2ypd(armor.armorPosition);
}

double YpdAngleTracker::computeMatchCost(const rm::Armor& armor, int armor_id) const
{
    const Eigen::Vector4d xyza = getPredictedArmorState(armor_id);
    const double obs_camera_yaw = std::atan2(armor.armorPosition.y(), armor.armorPosition.x());
    const double pred_camera_yaw = std::atan2(xyza[1], xyza[0]);
    double armor_yaw_res = std::abs(normalizeAngle(armor.yaw - xyza[3]));
    if (!is_outpost_ && armor_num_ == 4) {
        armor_yaw_res = std::min(
            armor_yaw_res, std::abs(normalizeAngle(armor.yaw + M_PI - xyza[3])));
    }
    const double camera_yaw_res = std::abs(normalizeAngle(obs_camera_yaw - pred_camera_yaw));

    return armor_yaw_res + camera_yaw_res;
}

std::vector<int> YpdAngleTracker::assignArmorIds(const std::vector<rm::Armor>& armors) const
{
    const int armor_count = std::min<int>(armors.size(), armor_num_);
    std::vector<int> assignment(armors.size(), -1);
    if (armor_count <= 0) return assignment;

    std::vector<std::vector<double>> costs(
        armor_count, std::vector<double>(armor_num_, DBL_MAX));
    for (int i = 0; i < armor_count; ++i) {
        for (int id = 0; id < armor_num_; ++id) {
            costs[i][id] = computeMatchCost(armors[i], id);
        }
    }

    double best_total_cost = DBL_MAX;
    std::vector<int> best_local_assignment(armor_count, -1);
    std::vector<int> current_assignment(armor_count, -1);
    std::vector<bool> used_ids(armor_num_, false);

    std::function<void(int, double)> dfs = [&](int obs_index, double total_cost) {
        if (total_cost >= best_total_cost) return;
        if (obs_index >= armor_count) {
            best_total_cost = total_cost;
            best_local_assignment = current_assignment;
            return;
        }

        for (int id = 0; id < armor_num_; ++id) {
            if (used_ids[id]) continue;
            used_ids[id] = true;
            current_assignment[obs_index] = id;
            dfs(obs_index + 1, total_cost + costs[obs_index][id]);
            current_assignment[obs_index] = -1;
            used_ids[id] = false;
        }
    };
    dfs(0, 0.0);

    for (int i = 0; i < armor_count; ++i) {
        assignment[i] = best_local_assignment[i];
    }
    return assignment;
}

int YpdAngleTracker::selectBestArmorId(const rm::Armor& armor) const
{
    int best_id = tracked_id_;
    double best_angle_error = DBL_MAX;
    const double obs_camera_yaw = std::atan2(armor.armorPosition.y(), armor.armorPosition.x());

    std::vector<std::pair<Eigen::Vector4d, int>> xyza_with_id;
    xyza_with_id.reserve(armor_num_);
    for (int id = 0; id < armor_num_; ++id) {
        xyza_with_id.push_back({getPredictedArmorState(id), id});
    }

    std::sort(
        xyza_with_id.begin(), xyza_with_id.end(),
        [](const std::pair<Eigen::Vector4d, int>& lhs,
            const std::pair<Eigen::Vector4d, int>& rhs) {
            return lhs.first.head<3>().norm() < rhs.first.head<3>().norm();
        });

    const int candidate_count = std::min<int>(3, xyza_with_id.size());
    for (int i = 0; i < candidate_count; ++i) {
        const auto& xyza = xyza_with_id[i].first;
        const double pred_camera_yaw = std::atan2(xyza[1], xyza[0]);
        double yaw_error = std::abs(normalizeAngle(armor.yaw - xyza[3]));
        if (!is_outpost_ && armor_num_ == 4) {
            yaw_error = std::min(
                yaw_error, std::abs(normalizeAngle(armor.yaw + M_PI - xyza[3])));
        }
        const double angle_error =
            yaw_error + std::abs(normalizeAngle(obs_camera_yaw - pred_camera_yaw));

        if (angle_error < best_angle_error) {
            best_angle_error = angle_error;
            best_id = xyza_with_id[i].second;
        }
    }
    return best_id;
}

Eigen::MatrixXd YpdAngleTracker::buildMeasurementJacobian(
    const Eigen::VectorXd& state, int armor_id) const
{
    const double angle = normalizeAngle(state(6) + armor_id * 2.0 * M_PI / armor_num_);
    const bool use_secondary_radius = (armor_num_ == 4) && (armor_id == 1 || armor_id == 3);
    const double radial_sign = armorRadialSign(armor_num_);

    const double r = use_secondary_radius ? state(8) + state(9) : state(8);
    const double dx_da = -radial_sign * r * std::sin(angle);
    const double dy_da = radial_sign * r * std::cos(angle);

    const double dx_dr = radial_sign * std::cos(angle);
    const double dy_dr = radial_sign * std::sin(angle);
    const double dx_dl = use_secondary_radius ? radial_sign * std::cos(angle) : 0.0;
    const double dy_dl = use_secondary_radius ? radial_sign * std::sin(angle) : 0.0;
    const double dz_d_offset1 =
        (armor_num_ == 3 && !outpostHeightPhaseLocked() && armor_id == 1) ? 1.0 : 0.0;
    const double dz_d_offset2 =
        ((armor_num_ == 4 && (armor_id == 1 || armor_id == 3)) ||
         (armor_num_ == 3 && !outpostHeightPhaseLocked() && armor_id == 2))
            ? 1.0
            : 0.0;

    Eigen::MatrixXd H_xyza = Eigen::MatrixXd::Zero(4, kStateDim);
    H_xyza(0, 0) = 1.0;
    H_xyza(0, 6) = dx_da;
    H_xyza(0, 8) = dx_dr;
    H_xyza(0, 9) = dx_dl;

    H_xyza(1, 2) = 1.0;
    H_xyza(1, 6) = dy_da;
    H_xyza(1, 8) = dy_dr;
    H_xyza(1, 9) = dy_dl;

    H_xyza(2, 4) = 1.0;
    H_xyza(2, 9) = dz_d_offset1;
    H_xyza(2, 10) = dz_d_offset2;

    H_xyza(3, 6) = 1.0;

    const Eigen::Vector3d xyz = predictArmorPosition(state, armor_id);
    const Eigen::MatrixXd H_ypd = xyz2ypdJacobian(xyz);

    Eigen::MatrixXd H_ypda = Eigen::MatrixXd::Zero(4, 4);
    H_ypda.block(0, 0, 3, 3) = H_ypd;
    H_ypda(3, 3) = 1.0;
    return H_ypda * H_xyza;
}

Eigen::MatrixXd YpdAngleTracker::buildMeasurementNoise(const rm::Armor& armor) const
{
    const Eigen::Vector3d obs_ypd = extractObservationYpd(armor);
    const double center_yaw = std::atan2(armor.armorPosition.y(), armor.armorPosition.x());
    const double delta_angle =
        std::abs(normalizeAngle(armor.yaw - center_yaw));

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(4, 4);
    R(0, 0) = 4e-3;
    R(1, 1) = 4e-3;
    // Keep the same diagonal assignment as the reference tracker for behavior parity.
    R(2, 2) = std::log(std::abs(delta_angle) + 1.0) + 1.0;
    R(3, 3) = std::log(std::abs(obs_ypd(2)) + 1.0) / 200.0 + 9e-2;
    if (is_outpost_) {
        double distance_scale = 1.0;
        if (armor.center.y < 600.0) distance_scale *= 3.0;
        if (armor.center.y < 450.0) distance_scale *= 2.0;
        const double obs_pitch_deg = obs_ypd(1) * 180.0 / M_PI;
        if (obs_pitch_deg < 10.0) distance_scale *= 2.0;
        if (obs_pitch_deg < 5.0) distance_scale *= 1.5;
        R(2, 2) *= distance_scale;

    }
    return R;
}

void YpdAngleTracker::updateBatch(const std::vector<rm::Armor>& armors, int preferred_index)
{
    last_batch_match_ids_.clear();
    last_observation_diagnostics_.clear();
    resetNormalObservationDiagnostics();
    if (!initialized_ || armors.empty()) return;
    last_motion_prior_yaw_rate_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_posterior_yaw_rate_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_prior_vx_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_prior_vy_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_posterior_vx_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_posterior_vy_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_center_update_norm_ = std::numeric_limits<double>::quiet_NaN();
    last_motion_velocity_update_norm_ =
        std::numeric_limits<double>::quiet_NaN();
    last_motion_speed_update_abs_ = std::numeric_limits<double>::quiet_NaN();
    last_physical_rejection_count_ = 0;
    last_physical_rejection_reason_ = kPhysicalRejectNone;

    if (pending_observation_jump_hint_ && armor_num_ == 4) {
        geometry_recovery_window_remaining_ =
            std::max(0, geometry_recovery_config_.recovery_window_frames);
        geometry_mismatch_streak_ = 0;
    }
    pending_observation_jump_hint_ = false;
    if (geometry_recovery_cooldown_ > 0) {
        geometry_recovery_cooldown_--;
    }

    const int armor_count = std::min<int>(armors.size(), armor_num_);
    const double normal_slot_step = armor_num_ > 0
        ? 2.0 * M_PI / static_cast<double>(armor_num_)
        : 0.0;
    last_batch_match_ids_.assign(armors.size(), -1);
    std::vector<int> assignment = assignArmorIds(armors);
    last_observation_diagnostics_.resize(armors.size());
    for (int i = 0; i < static_cast<int>(armors.size()); ++i) {
        ObservationDiagnostic& diagnostic = last_observation_diagnostics_[i];
        diagnostic.observation_index = i;
        diagnostic.matched_slot = i < static_cast<int>(assignment.size())
            ? assignment[i]
            : -1;
        diagnostic.update_count_before = update_count_;
        diagnostic.update_count_after = update_count_;
        if (i >= armor_count) {
            diagnostic.reject_reason = kObservationSkippedByBatchCapacity;
        }
    }
    std::vector<double> corrected_yaws(
        armors.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<bool> strong_pair_member(armors.size(), false);
    Eigen::Vector2d normal_pair_center_measurement =
        Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector2d normal_pair_geometry_center_measurement =
        Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
    int normal_pair_group_id = -1;
    int normal_pair_geometry_group_id = -1;

    const bool normal_pair_required = !is_outpost_ && armor_num_ == 4 && armor_count >= 2;
    last_normal_pair_required_ = normal_pair_required ? 1 : 0;
    bool normal_pair_found = false;
    bool normal_pair_geometry_found = false;
    if (normal_pair_required) {
        struct PairCandidate {
            bool valid = false;
            int obs_a = -1;
            int obs_b = -1;
            int id_a = -1;
            int id_b = -1;
            double yaw_a = 0.0;
            double yaw_b = 0.0;
            double center_x = std::numeric_limits<double>::quiet_NaN();
            double center_y = std::numeric_limits<double>::quiet_NaN();
            double center_gap = std::numeric_limits<double>::quiet_NaN();
            double center_jump = std::numeric_limits<double>::quiet_NaN();
            double score = DBL_MAX;
        };

        PairCandidate best_pair;
        PairCandidate best_geometry_pair;
        const double center_gate =
            update_count_ < 3 ? kNormalPairEarlyCenterGateM : kNormalPairWeakCenterGateM;
        for (int obs_a = 0; obs_a < armor_count; ++obs_a) {
            if (!armors[obs_a].armorPosition.allFinite()) continue;
            const Eigen::Vector2d pos_a(
                armors[obs_a].armorPosition.x(), armors[obs_a].armorPosition.y());
            const double obs_camera_yaw_a =
                std::atan2(armors[obs_a].armorPosition.y(), armors[obs_a].armorPosition.x());

            for (int obs_b = obs_a + 1; obs_b < armor_count; ++obs_b) {
                if (!armors[obs_b].armorPosition.allFinite()) continue;
                const Eigen::Vector2d pos_b(
                    armors[obs_b].armorPosition.x(), armors[obs_b].armorPosition.y());
                const double obs_camera_yaw_b = std::atan2(
                    armors[obs_b].armorPosition.y(), armors[obs_b].armorPosition.x());

                for (int id_a = 0; id_a < armor_num_; ++id_a) {
                    for (int id_b = 0; id_b < armor_num_; ++id_b) {
                        if (id_a == id_b) continue;
                        const int id_delta = std::abs(id_a - id_b);
                        if (id_delta != 1 && id_delta != 3) continue;

                        const double expected_yaw_a =
                            normalizeAngle(x_(6) + id_a * normal_slot_step);
                        const double expected_yaw_b =
                            normalizeAngle(x_(6) + id_b * normal_slot_step);

                        for (int flip_a = 0; flip_a < 2; ++flip_a) {
                            const double yaw_a =
                                normalizeAngle(armors[obs_a].yaw + flip_a * M_PI);
                            const double yaw_res_a =
                                std::abs(normalizeAngle(yaw_a - expected_yaw_a));
                            if (yaw_res_a > kNormalPairYawGateRad) continue;

                            for (int flip_b = 0; flip_b < 2; ++flip_b) {
                                const double yaw_b =
                                    normalizeAngle(armors[obs_b].yaw + flip_b * M_PI);
                                const double yaw_res_b =
                                    std::abs(normalizeAngle(yaw_b - expected_yaw_b));
                                if (yaw_res_b > kNormalPairYawGateRad) continue;

                                const Eigen::Vector2d normal_a(
                                    std::cos(yaw_a), std::sin(yaw_a));
                                const Eigen::Vector2d normal_b(
                                    std::cos(yaw_b), std::sin(yaw_b));
                                Eigen::Matrix2d A;
                                A.col(0) = normal_a;
                                A.col(1) = -normal_b;
                                const double det = A.determinant();
                                if (!std::isfinite(det) || std::abs(det) < 0.15) continue;

                                const Eigen::Vector2d radii = A.inverse() * (pos_b - pos_a);
                                if (!radii.allFinite()) continue;
                                if (radii(0) < kNormalPairMinRadiusM ||
                                    radii(0) > kNormalPairMaxRadiusM ||
                                    radii(1) < kNormalPairMinRadiusM ||
                                    radii(1) > kNormalPairMaxRadiusM) {
                                    continue;
                                }

                                const Eigen::Vector2d center_a = pos_a + radii(0) * normal_a;
                                const Eigen::Vector2d center_b = pos_b + radii(1) * normal_b;
                                const double center_gap = (center_a - center_b).norm();
                                if (!std::isfinite(center_gap) ||
                                    center_gap > kNormalPairCenterGapGateM) {
                                    continue;
                                }

                                const Eigen::Vector2d center = 0.5 * (center_a + center_b);
                                const double radius_prior_a = getArmorRadius(id_a);
                                const double radius_prior_b = getArmorRadius(id_b);
                                const double radius_prior_residual =
                                    std::abs(radii(0) - radius_prior_a) +
                                    std::abs(radii(1) - radius_prior_b);
                                const double geometry_score =
                                    std::max(yaw_res_a, yaw_res_b) / kNormalPairYawGateRad +
                                    center_gap / kNormalPairCenterGapGateM +
                                    radius_prior_residual /
                                        std::max(0.01, kNormalPairMaxRadiusM - kNormalPairMinRadiusM);
                                if (geometry_score < best_geometry_pair.score) {
                                    best_geometry_pair.valid = true;
                                    best_geometry_pair.obs_a = obs_a;
                                    best_geometry_pair.obs_b = obs_b;
                                    best_geometry_pair.id_a = id_a;
                                    best_geometry_pair.id_b = id_b;
                                    best_geometry_pair.yaw_a = yaw_a;
                                    best_geometry_pair.yaw_b = yaw_b;
                                    best_geometry_pair.center_x = center.x();
                                    best_geometry_pair.center_y = center.y();
                                    best_geometry_pair.center_gap = center_gap;
                                    best_geometry_pair.center_jump =
                                        std::numeric_limits<double>::quiet_NaN();
                                    best_geometry_pair.score = geometry_score;
                                }
                                const double center_jump =
                                    std::hypot(center.x() - x_(0), center.y() - x_(2));
                                if (!std::isfinite(center_jump) ||
                                    center_jump > center_gate) {
                                    continue;
                                }

                                const Eigen::Vector3d pred_a = predictArmorPosition(x_, id_a);
                                const Eigen::Vector3d pred_b = predictArmorPosition(x_, id_b);
                                const double pred_camera_yaw_a =
                                    std::atan2(pred_a.y(), pred_a.x());
                                const double pred_camera_yaw_b =
                                    std::atan2(pred_b.y(), pred_b.x());
                                const double camera_residual =
                                    std::abs(normalizeAngle(obs_camera_yaw_a - pred_camera_yaw_a)) +
                                    std::abs(normalizeAngle(obs_camera_yaw_b - pred_camera_yaw_b));

                                const double score =
                                    center_jump / center_gate +
                                    std::max(yaw_res_a, yaw_res_b) / kNormalPairYawGateRad +
                                    center_gap / kNormalPairCenterGapGateM +
                                    0.25 * camera_residual;
                                if (score < best_pair.score) {
                                    best_pair.valid = true;
                                    best_pair.obs_a = obs_a;
                                    best_pair.obs_b = obs_b;
                                    best_pair.id_a = id_a;
                                    best_pair.id_b = id_b;
                                    best_pair.yaw_a = yaw_a;
                                    best_pair.yaw_b = yaw_b;
                                    best_pair.center_x = center.x();
                                    best_pair.center_y = center.y();
                                    best_pair.center_gap = center_gap;
                                    best_pair.center_jump = center_jump;
                                    best_pair.score = score;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (best_pair.valid) {
            normal_pair_found = true;
            assignment[best_pair.obs_a] = best_pair.id_a;
            assignment[best_pair.obs_b] = best_pair.id_b;
            corrected_yaws[best_pair.obs_a] = best_pair.yaw_a;
            corrected_yaws[best_pair.obs_b] = best_pair.yaw_b;
            strong_pair_member[best_pair.obs_a] = true;
            strong_pair_member[best_pair.obs_b] = true;
            normal_pair_center_measurement =
                Eigen::Vector2d(best_pair.center_x, best_pair.center_y);
            normal_pair_group_id =
                normalPairVelocityGroupId(best_pair.id_a, best_pair.id_b);
            last_normal_pair_found_ = 1;
            last_normal_pair_score_ = best_pair.score;
            last_normal_pair_center_gap_m_ = best_pair.center_gap;
            last_normal_pair_center_jump_m_ = best_pair.center_jump;
            last_normal_pair_center_x_m_ = best_pair.center_x;
            last_normal_pair_center_y_m_ = best_pair.center_y;
        }
        if (!normal_pair_found && best_geometry_pair.valid) {
            normal_pair_geometry_found = true;
            normal_pair_geometry_center_measurement =
                Eigen::Vector2d(best_geometry_pair.center_x, best_geometry_pair.center_y);
            normal_pair_geometry_group_id =
                normalPairVelocityGroupId(best_geometry_pair.id_a, best_geometry_pair.id_b);
        }
    }

    const bool use_primary_only =
        is_outpost_ && armor_count > 1 && preferred_index >= 0 &&
        preferred_index < armor_count && armors[preferred_index].center.y < 600.0;
    const int tracked_index =
        (preferred_index >= 0 && preferred_index < armor_count) ? preferred_index : 0;
    int accepted_count = 0;
    int primary_accepted_id = -1;
    int normal_accepted_count = 0;
    int strong_pair_accepted_count = 0;
    Eigen::Vector2d single_center_sum = Eigen::Vector2d::Zero();
    int single_center_count = 0;
    int single_center_group_id = -1;
    bool single_center_group_ambiguous = false;
    Eigen::Vector2d single_candidate_center_sum = Eigen::Vector2d::Zero();
    int single_candidate_center_count = 0;
    int single_candidate_center_group_id = -1;
    bool single_candidate_center_group_ambiguous = false;
    double yaw_observation_sin_sum = 0.0;
    double yaw_observation_cos_sum = 0.0;
    int yaw_observation_count = 0;
    int yaw_observation_source = 0;
    for (int i = 0; i < armor_count; ++i) {
        ObservationDiagnostic& diagnostic = last_observation_diagnostics_[i];
        if (normal_pair_found && !strong_pair_member[i]) {
            diagnostic.reject_reason = kObservationSkippedByPairSelection;
            continue;
        }
        const int matched_id =
            assignment[i] >= 0 ? assignment[i] : selectBestArmorId(armors[i]);
        diagnostic.matched_slot = matched_id;
        if (use_primary_only && i != preferred_index) {
            diagnostic.reject_reason = kObservationSkippedByPrimarySelection;
            continue;
        }
        rm::Armor observation = armors[i];
        if (!is_outpost_ && armor_num_ == 4) {
            if (std::isfinite(corrected_yaws[i])) {
                observation.yaw = corrected_yaws[i];
            } else {
                const Eigen::Vector4d predicted = getPredictedArmorState(matched_id);
                const double flipped_yaw = normalizeAngle(observation.yaw + M_PI);
                if (std::abs(normalizeAngle(flipped_yaw - predicted(3))) <
                    std::abs(normalizeAngle(observation.yaw - predicted(3)))) {
                    observation.yaw = flipped_yaw;
                }
            }
        }
        diagnostic.yaw_pi_flip =
            std::abs(std::abs(normalizeAngle(observation.yaw - armors[i].yaw)) - M_PI) <
            1e-6;

        const bool freeze_normal_geometry =
            !is_outpost_ && armor_num_ == 4 && !strong_pair_member[i];
        double max_center_jump_m = kMaxNormalSingleUpdateCenterJumpM;
        if (!is_outpost_ && armor_num_ == 4) {
            max_center_jump_m = strong_pair_member[i]
                ? kMaxNormalPairUpdateCenterJumpM
                : kMaxNormalWeakSingleUpdateCenterJumpM;
        }

        if (correctWithObservation(
                observation, matched_id, freeze_normal_geometry, max_center_jump_m,
                &diagnostic)) {
            last_batch_match_ids_[i] = matched_id;
            updateOutpostHeightPhase(observation, matched_id);
            accepted_count++;
            if (!is_outpost_ && armor_num_ == 4) {
                normal_accepted_count++;
                if (strong_pair_member[i]) {
                    strong_pair_accepted_count++;
                } else if (!normal_pair_found &&
                           update_count_ >= kNormalSingleCenterVelocityMinUpdates) {
                    const double radius = getArmorRadius(matched_id);
                    const Eigen::Vector2d normal(
                        std::cos(observation.yaw), std::sin(observation.yaw));
                    const Eigen::Vector2d pos(
                        observation.armorPosition.x(), observation.armorPosition.y());
                    const Eigen::Vector2d center =
                        pos - armorRadialSign(armor_num_) * radius * normal;
                    if (center.allFinite()) {
                        single_candidate_center_sum += center;
                        single_candidate_center_count++;
                        const int group_id =
                            normalSingleVelocityGroupId(3, matched_id);
                        if (single_candidate_center_group_id < 0 &&
                            !single_candidate_center_group_ambiguous) {
                            single_candidate_center_group_id = group_id;
                        } else if (single_candidate_center_group_id != group_id) {
                            single_candidate_center_group_ambiguous = true;
                            single_candidate_center_group_id = -1;
                        }
                    }
                    const double center_jump =
                        std::hypot(center.x() - x_(0), center.y() - x_(2));
                    if (center.allFinite() && std::isfinite(center_jump) &&
                        center_jump <= kNormalSingleCenterVelocityCenterGateM) {
                        single_center_sum += center;
                        single_center_count++;
                        const int group_id =
                            normalSingleVelocityGroupId(2, matched_id);
                        if (single_center_group_id < 0 &&
                            !single_center_group_ambiguous) {
                            single_center_group_id = group_id;
                        } else if (single_center_group_id != group_id) {
                            single_center_group_ambiguous = true;
                            single_center_group_id = -1;
                        }
                    }
                }
                const double base_yaw =
                    normalizeAngle(observation.yaw - matched_id * normal_slot_step);
                yaw_observation_sin_sum += std::sin(base_yaw);
                yaw_observation_cos_sum += std::cos(base_yaw);
                yaw_observation_count++;
                if (strong_pair_member[i]) {
                    yaw_observation_source = 1;
                } else if (yaw_observation_source == 0) {
                    yaw_observation_source = 2;
                }
            }
            if (i == tracked_index) {
                primary_accepted_id = matched_id;
            }
        }
    }

    if (is_outpost_ && accepted_count == 0 &&
        consecutive_rejected_updates_ >= kOutpostReinitAfterRejectedUpdates) {
        init(armors[tracked_index], armor_num_);
        tracked_id_ = selectBestArmorId(armors[tracked_index]);
        last_batch_match_ids_.assign(armors.size(), -1);
        last_batch_match_ids_[tracked_index] = tracked_id_;
        appendMotionSample();
        return;
    }

    applyLockedOutpostHeightOffsets();

    if (primary_accepted_id >= 0) {
        tracked_id_ = primary_accepted_id;
    }

    if (!is_outpost_ && armor_num_ == 4 && yaw_observation_count > 0) {
        const double yaw_measurement =
            std::atan2(yaw_observation_sin_sum, yaw_observation_cos_sum);
        appendNormalYawObservation(
            yaw_measurement, yaw_observation_source, yaw_observation_count);
        updateNormalYawRateFromHistory();
    }

    if (armor_num_ == 4) {
        const GeometryResidualSummary summary = summarizeGeometryResiduals(armors);
        last_geometry_residual_xy_over_sigma_dr_ = summary.mean_residual_xy_over_sigma_dr;
        last_geometry_residual_z_over_sigma_h_ = summary.mean_residual_z_over_sigma_h;
    } else {
        last_geometry_residual_xy_over_sigma_dr_ = std::numeric_limits<double>::quiet_NaN();
        last_geometry_residual_z_over_sigma_h_ = std::numeric_limits<double>::quiet_NaN();
    }

    last_normal_accepted_count_ = normal_accepted_count;
    last_normal_pair_accepted_count_ = strong_pair_accepted_count;
    if (!is_outpost_ && armor_num_ == 4) {
        if (normal_pair_found && strong_pair_accepted_count >= 2) {
            last_normal_update_class_ = 1;
        } else if (normal_pair_found) {
            last_normal_update_class_ = 2;
        } else if (normal_accepted_count > 0) {
            last_normal_update_class_ = 3;
        } else {
            last_normal_update_class_ = 0;
        }
    } else if (accepted_count > 0) {
        last_normal_update_class_ = 4;
    }

    Eigen::Vector2d velocity_center_measurement =
        Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
    int velocity_observation_source = 0;
    if (single_center_count > 0) {
        const Eigen::Vector2d single_center_measurement =
            single_center_sum / static_cast<double>(single_center_count);
        last_normal_single_center_count_ = single_center_count;
        last_normal_single_center_x_m_ = single_center_measurement.x();
        last_normal_single_center_y_m_ = single_center_measurement.y();
    }
    if (single_candidate_center_count > 0) {
        const Eigen::Vector2d candidate_center_measurement =
            single_candidate_center_sum /
            static_cast<double>(single_candidate_center_count);
        if (candidate_center_measurement.allFinite()) {
            MotionSample candidate_sample;
            candidate_sample.t_s = tracker_time_sec_;
            candidate_sample.center_x = candidate_center_measurement.x();
            candidate_sample.center_y = candidate_center_measurement.y();
            candidate_sample.yaw_rate = x_(7);
            candidate_sample.frame_yaw = current_frame_yaw_rad_;
            candidate_sample.frame_yaw_rate = current_frame_yaw_rate_rad_s_;
            candidate_sample.source = 3;
            candidate_sample.group_id =
                single_candidate_center_group_ambiguous
                    ? -1
                    : single_candidate_center_group_id;
            normal_single_center_candidate_history_.push_back(candidate_sample);
            while (static_cast<int>(normal_single_center_candidate_history_.size()) >
                   kMotionHistoryCapacity) {
                normal_single_center_candidate_history_.pop_front();
            }
        }
    }
    if (normal_pair_found && normal_pair_center_measurement.allFinite()) {
        velocity_center_measurement = normal_pair_center_measurement;
        velocity_observation_source = 1;
    } else if (kNormalPairGeometryVelocityStateUpdateEnabled &&
               normal_pair_geometry_found &&
               normal_pair_geometry_center_measurement.allFinite()) {
        velocity_center_measurement = normal_pair_geometry_center_measurement;
        // Source 4 is a geometry-only pair center. It passed yaw/radius/common-
        // center checks, but did not update center/radius because it failed the
        // center-jump gate against the current filter state. T87/T88 showed
        // that directly mixing it into ordinary velocity history is not a
        // final-acceptable state update, so the default path keeps it disabled.
        velocity_observation_source = 4;
    } else if (single_center_count > 0) {
        velocity_center_measurement =
            single_center_sum / static_cast<double>(single_center_count);
        velocity_observation_source = 2;
    }

    if (velocity_observation_source > 0 && velocity_center_measurement.allFinite()) {
        int velocity_group_id = -1;
        if (velocity_observation_source == 1) {
            velocity_group_id = normal_pair_group_id;
        } else if (velocity_observation_source == 4) {
            velocity_group_id = normal_pair_geometry_group_id;
        } else if (velocity_observation_source == 2 &&
                   !single_center_group_ambiguous) {
            velocity_group_id = single_center_group_id;
        }
        last_normal_velocity_observation_source_ = velocity_observation_source;
        last_normal_velocity_sample_t_s_ = tracker_time_sec_;
        last_normal_velocity_sample_x_m_ = velocity_center_measurement.x();
        last_normal_velocity_sample_y_m_ = velocity_center_measurement.y();
        last_normal_velocity_sample_frame_yaw_rad_ = current_frame_yaw_rad_;
        last_normal_velocity_sample_group_id_ = velocity_group_id;
        MotionSample center_sample;
        center_sample.t_s = tracker_time_sec_;
        center_sample.center_x = velocity_center_measurement.x();
        center_sample.center_y = velocity_center_measurement.y();
        center_sample.yaw_rate = x_(7);
        center_sample.frame_yaw = current_frame_yaw_rad_;
        center_sample.frame_yaw_rate = current_frame_yaw_rate_rad_s_;
        center_sample.source = velocity_observation_source;
        center_sample.group_id = velocity_group_id;
        normal_center_velocity_history_.push_back(center_sample);
        while (static_cast<int>(normal_center_velocity_history_.size()) >
               kMotionHistoryCapacity) {
            normal_center_velocity_history_.pop_front();
        }
        updateNormalCenterVelocityFromHistory();
    } else if (updateNormalCandidateCenterVelocityFromHistory()) {
        // Candidate fallback updates velocity only. It intentionally does not
        // move the center position or relax the center-jump gate.
    }
    last_normal_velocity_history_size_ =
        static_cast<int>(normal_center_velocity_history_.size());
    maybeRecoverGeometry(armors);
    appendMotionSample();
}

int YpdAngleTracker::selectTrackedId(const rm::Armor& armor)
{
    if (!initialized_) return tracked_id_;
    tracked_id_ = selectBestArmorId(armor);
    return tracked_id_;
}

void YpdAngleTracker::recordNis(double nis, int measurement_dim)
{
    last_nis_ = nis;

    int failed = 1;
    if (std::isfinite(nis)) {
        failed = nis > nisThresholdForDim(measurement_dim) ? 1 : 0;
    }

    recent_nis_failures_.push_back(failed);
    while (static_cast<int>(recent_nis_failures_.size()) > nis_window_size_) {
        recent_nis_failures_.pop_front();
    }
}

int YpdAngleTracker::recentNisFailureCount() const
{
    return std::accumulate(
        recent_nis_failures_.begin(), recent_nis_failures_.end(), 0);
}

bool YpdAngleTracker::badConvergence() const
{
    if (!initialized_ || recent_nis_failures_.empty()) return false;
    return recentNisFailureCount() * 5 >= nis_window_size_ * 2;
}

bool YpdAngleTracker::diverged() const
{
    if (!initialized_ || x_.size() < kStateDim) return false;
    const double primary_radius = x_(kPrimaryRadiusIndex);
    const bool primary_ok =
        primary_radius > kMinArmorRadiusM && primary_radius < kMaxValidArmorRadiusM;
    if (!primary_ok) return true;

    if (armor_num_ == 4) {
        const double secondary_radius = x_(kPrimaryRadiusIndex) + x_(kDeltaRadiusIndex);
        return !(secondary_radius > kMinArmorRadiusM &&
                 secondary_radius < kMaxValidArmorRadiusM);
    }

    if (armor_num_ == 3) {
        return !(std::isfinite(x_(kDeltaRadiusIndex)) &&
                 std::isfinite(x_(kHeightDiffIndex)) &&
                 std::abs(x_(kDeltaRadiusIndex)) <= kMaxOutpostHeightOffsetM &&
                 std::abs(x_(kHeightDiffIndex)) <= kMaxOutpostHeightOffsetM);
    }

    return false;
}

YpdAngleTracker::GeometryResidualSummary YpdAngleTracker::summarizeGeometryResiduals(
    const std::vector<rm::Armor>& armors) const
{
    GeometryResidualSummary summary;
    if (!initialized_ || armor_num_ != 4 || P_.rows() <= kHeightDiffIndex ||
        P_.cols() <= kHeightDiffIndex) {
        return summary;
    }

    std::vector<double> residual_xy_values;
    std::vector<double> residual_z_values;
    const int obs_limit = std::min(static_cast<int>(armors.size()),
        static_cast<int>(last_batch_match_ids_.size()));

    for (int obs_index = 0; obs_index < obs_limit; ++obs_index) {
        const int matched_id = last_batch_match_ids_[obs_index];
        if (matched_id < 0 || matched_id >= armor_num_) {
            continue;
        }

        const Eigen::Vector3d predicted_xyz = predictArmorPosition(x_, matched_id);
        const Eigen::Vector3d residual = armors[obs_index].armorPosition - predicted_xyz;
        residual_xy_values.push_back(residual.head<2>().norm());
        residual_z_values.push_back(std::abs(residual.z()));
    }

    summary.matched_count = static_cast<int>(residual_xy_values.size());
    if (summary.matched_count == 0) {
        return summary;
    }

    summary.mean_residual_xy = std::accumulate(
        residual_xy_values.begin(), residual_xy_values.end(), 0.0) / residual_xy_values.size();
    summary.mean_residual_z = std::accumulate(
        residual_z_values.begin(), residual_z_values.end(), 0.0) / residual_z_values.size();
    summary.mean_residual_xy_over_sigma_dr =
        safeNormalizedMagnitude(summary.mean_residual_xy, P_(kDeltaRadiusIndex, kDeltaRadiusIndex));
    summary.mean_residual_z_over_sigma_h =
        safeNormalizedMagnitude(summary.mean_residual_z, P_(kHeightDiffIndex, kHeightDiffIndex));
    return summary;
}

void YpdAngleTracker::inflateGeometryCovariance()
{
    if (P_.rows() <= kHeightDiffIndex || P_.cols() <= kHeightDiffIndex) {
        return;
    }

    const double inflation_scale =
        std::max(1.0, geometry_recovery_config_.covariance_inflation_scale);
    const double sqrt_scale = std::sqrt(inflation_scale);
    P_.row(kDeltaRadiusIndex) *= sqrt_scale;
    P_.col(kDeltaRadiusIndex) *= sqrt_scale;
    P_.row(kHeightDiffIndex) *= sqrt_scale;
    P_.col(kHeightDiffIndex) *= sqrt_scale;

    P_(kDeltaRadiusIndex, kDeltaRadiusIndex) =
        std::max(P_(kDeltaRadiusIndex, kDeltaRadiusIndex),
            geometry_recovery_config_.min_dr_variance);
    P_(kHeightDiffIndex, kHeightDiffIndex) =
        std::max(P_(kHeightDiffIndex, kHeightDiffIndex),
            geometry_recovery_config_.min_h_variance);
    P_ = 0.5 * (P_ + P_.transpose());
    geometry_recovery_inflation_count_++;
}

void YpdAngleTracker::maybeRecoverGeometry(const std::vector<rm::Armor>& armors)
{
    if (!initialized_ || armor_num_ != 4) return;
    if (geometry_recovery_window_remaining_ <= 0) return;
    if (geometry_recovery_cooldown_ > 0) return;

    const GeometryResidualSummary summary = summarizeGeometryResiduals(armors);
    last_geometry_residual_xy_over_sigma_dr_ = summary.mean_residual_xy_over_sigma_dr;
    last_geometry_residual_z_over_sigma_h_ = summary.mean_residual_z_over_sigma_h;
    geometry_recovery_window_remaining_--;

    const bool geometry_mismatch =
        summary.matched_count >= std::max(1, geometry_recovery_config_.min_matched_count) &&
        std::isfinite(summary.mean_residual_z_over_sigma_h) &&
        std::isfinite(summary.mean_residual_xy_over_sigma_dr) &&
        ((summary.mean_residual_z_over_sigma_h >
              geometry_recovery_config_.residual_z_sigma_threshold &&
          summary.mean_residual_xy_over_sigma_dr >
              geometry_recovery_config_.residual_xy_sigma_threshold) ||
         summary.mean_residual_z_over_sigma_h >
              (geometry_recovery_config_.residual_z_sigma_threshold + 1.0));

    if (!geometry_mismatch) {
        geometry_mismatch_streak_ = 0;
        return;
    }

    geometry_mismatch_streak_++;
    if (geometry_mismatch_streak_ <
        std::max(1, geometry_recovery_config_.mismatch_required_streak)) {
        return;
    }

    inflateGeometryCovariance();
    geometry_mismatch_streak_ = 0;
    geometry_recovery_window_remaining_ = 0;
    geometry_recovery_cooldown_ =
        std::max(0, geometry_recovery_config_.recovery_cooldown_frames);
    recent_nis_failures_.clear();
    recent_nis_failures_.push_back(0);
    last_nis_ = 0.0;
    is_converged_ = false;
}

bool YpdAngleTracker::correctWithObservation(
    const rm::Armor& armor, int matched_id, bool freeze_normal_geometry,
    double max_center_jump_m, ObservationDiagnostic* diagnostic)
{
    const Eigen::Vector3d obs_ypd = extractObservationYpd(armor);
    const Eigen::VectorXd prior_state = x_;

    if (diagnostic != nullptr) {
        diagnostic->matched_slot = matched_id;
        diagnostic->accepted = false;
        diagnostic->reject_reason = kObservationNotEvaluated;
        diagnostic->physical_reject_reason = kPhysicalRejectNone;
        diagnostic->freeze_normal_geometry = freeze_normal_geometry;
        diagnostic->max_center_jump_m = max_center_jump_m;
        diagnostic->update_count_before = update_count_;
        diagnostic->update_count_after = update_count_;
    }

    const Eigen::MatrixXd H = buildMeasurementJacobian(x_, matched_id);
    const Eigen::MatrixXd R = buildMeasurementNoise(armor);
    const Eigen::VectorXd z_pred = predictArmorMeasurement(x_, matched_id);

    double observation_yaw = armor.yaw;
    if (!is_outpost_ && armor_num_ == 4) {
        const double flipped_yaw = normalizeAngle(armor.yaw + M_PI);
        const double direct_residual = std::abs(normalizeAngle(observation_yaw - z_pred(3)));
        const double flipped_residual = std::abs(normalizeAngle(flipped_yaw - z_pred(3)));
        if (flipped_residual < direct_residual) {
            observation_yaw = flipped_yaw;
            if (diagnostic != nullptr) diagnostic->yaw_pi_flip = true;
        }
    }

    Eigen::VectorXd z(4);
    z << obs_ypd(0), obs_ypd(1), obs_ypd(2), observation_yaw;

    Eigen::VectorXd residual = z - z_pred;
    residual(0) = normalizeAngle(residual(0));
    residual(1) = normalizeAngle(residual(1));
    residual(3) = normalizeAngle(residual(3));

    if (diagnostic != nullptr) {
        diagnostic->observation = z;
        diagnostic->prior_predicted_measurement = z_pred;
        diagnostic->prior_innovation = residual;
    }

    const Eigen::MatrixXd innovation_cov = H * P_ * H.transpose() + R;
    const double prior_nis =
        residual.transpose() * innovation_cov.inverse() * residual;
    if (diagnostic != nullptr) diagnostic->prior_nis = prior_nis;
    if (is_outpost_ && update_count_ >= kOutpostPriorGateMinUpdates &&
        std::isfinite(prior_nis) && prior_nis > kOutpostPriorNisRejectThreshold) {
        consecutive_rejected_updates_++;
        recordNis(prior_nis, 4);
        if (diagnostic != nullptr) {
            diagnostic->reject_reason = kObservationRejectedByPriorNis;
        }
        return false;
    }

    Eigen::MatrixXd K = P_ * H.transpose() * innovation_cov.inverse();
    if (!is_outpost_ && armor_num_ == 4) {
        K.row(1).setZero();
        K.row(3).setZero();
        K.row(7).setZero();
    }
    if (freeze_normal_geometry && !is_outpost_ && armor_num_ == 4) {
        K.row(0).setZero();
        K.row(2).setZero();
        K.row(kPrimaryRadiusIndex).setZero();
        K.row(kDeltaRadiusIndex).setZero();
        K.row(kHeightDiffIndex).setZero();
    }
    const Eigen::MatrixXd KH = K * H;
    Eigen::MatrixXd candidate_P =
        (I_ - KH) * P_ * (I_ - KH).transpose() + K * R * K.transpose();
    Eigen::VectorXd candidate_x = x_ + K * residual;
    candidate_x(6) = normalizeAngle(candidate_x(6));

    const int physical_rejection_reason =
        is_outpost_ ? kPhysicalRejectNone
                    : normalArmorUpdateRejectionReason(
                          prior_state, candidate_x, armor_num_, max_center_jump_m);
    if (physical_rejection_reason != kPhysicalRejectNone) {
        last_physical_rejection_count_++;
        last_physical_rejection_reason_ = physical_rejection_reason;
        if (diagnostic != nullptr) {
            diagnostic->reject_reason = kObservationRejectedByPhysicalGate;
            diagnostic->physical_reject_reason = physical_rejection_reason;
        }
        return false;
    }

    P_ = candidate_P;
    x_ = candidate_x;
    applyOutpostRadiusPrior();
    recordMotionUpdate(prior_state, x_);

    Eigen::VectorXd posterior_residual = z - predictArmorMeasurement(x_, matched_id);
    posterior_residual(0) = normalizeAngle(posterior_residual(0));
    posterior_residual(1) = normalizeAngle(posterior_residual(1));
    posterior_residual(3) = normalizeAngle(posterior_residual(3));

    const Eigen::MatrixXd posterior_residual_cov = H * P_ * H.transpose() + R;
    const double posterior_nis =
        posterior_residual.transpose() * posterior_residual_cov.inverse() *
        posterior_residual;
    recordNis(posterior_nis, 4);
    update_count_++;
    consecutive_rejected_updates_ = 0;
    if (diagnostic != nullptr) {
        diagnostic->accepted = true;
        diagnostic->reject_reason = kObservationAccepted;
        diagnostic->posterior_innovation = posterior_residual;
        diagnostic->posterior_nis = posterior_nis;
        diagnostic->update_count_after = update_count_;
    }
    return true;
}

void YpdAngleTracker::appendMotionSample()
{
    if (!initialized_ || x_.size() < kStateDim) return;

    MotionSample sample;
    sample.t_s = tracker_time_sec_;
    sample.center_x = x_(0);
    sample.center_y = x_(2);
    sample.yaw_rate = x_(7);
    sample.frame_yaw = current_frame_yaw_rad_;
    sample.frame_yaw_rate = current_frame_yaw_rate_rad_s_;
    motion_history_.push_back(sample);
    while (static_cast<int>(motion_history_.size()) > kMotionHistoryCapacity) {
        motion_history_.pop_front();
    }
}

void YpdAngleTracker::appendNormalYawObservation(
    double yaw, int source, int observation_count)
{
    if (!initialized_ || is_outpost_ || armor_num_ != 4 ||
        !std::isfinite(yaw)) {
        return;
    }

    double unwrapped_yaw = normalizeAngle(yaw);
    if (!normal_yaw_observation_history_.empty()) {
        const double previous_yaw = normal_yaw_observation_history_.back().yaw;
        unwrapped_yaw =
            previous_yaw + normalizeAngle(unwrapped_yaw - normalizeAngle(previous_yaw));
    }

    YawObservationSample sample;
    sample.t_s = tracker_time_sec_;
    sample.yaw = unwrapped_yaw;
    sample.source = source;
    normal_yaw_observation_history_.push_back(sample);
    while (static_cast<int>(normal_yaw_observation_history_.size()) >
           kMotionHistoryCapacity) {
        normal_yaw_observation_history_.pop_front();
    }
    last_normal_yaw_rate_history_size_ =
        static_cast<int>(normal_yaw_observation_history_.size());
    last_normal_yaw_rate_observation_source_ = source;
    last_normal_yaw_observation_count_ = observation_count;
    last_normal_yaw_observation_t_s_ = tracker_time_sec_;
    last_normal_yaw_observation_raw_rad_ = normalizeAngle(yaw);
    last_normal_yaw_observation_unwrapped_rad_ = unwrapped_yaw;
}

void YpdAngleTracker::updateNormalYawRateFromHistory()
{
    if (!initialized_ || is_outpost_ || armor_num_ != 4 || x_.size() < kStateDim) {
        return;
    }

    last_normal_yaw_rate_history_size_ =
        static_cast<int>(normal_yaw_observation_history_.size());
    last_normal_yaw_rate_fit_sample_count_ = 0;
    last_normal_yaw_rate_fit_accepted_ = 0;
    last_normal_yaw_rate_fit_reject_reason_ = kNormalYawRateFitNotEvaluated;
    last_normal_yaw_rate_fit_time_span_s_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_rate_fit_rms_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_rate_fit_raw_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_yaw_rate_fit_applied_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();

    if (static_cast<int>(normal_yaw_observation_history_.size()) <
        kNormalYawRateFitMinSamples) {
        last_normal_yaw_rate_fit_reject_reason_ =
            kNormalYawRateFitNotEnoughHistory;
        return;
    }

    const double newest_t = normal_yaw_observation_history_.back().t_s;
    std::vector<YawObservationSample> samples;
    samples.reserve(normal_yaw_observation_history_.size());
    for (auto it = normal_yaw_observation_history_.rbegin();
         it != normal_yaw_observation_history_.rend(); ++it) {
        const double age = newest_t - it->t_s;
        if (!std::isfinite(age) || age > kNormalYawRateFitWindowS) break;
        if (std::isfinite(it->yaw)) {
            samples.push_back(*it);
        }
    }
    if (static_cast<int>(samples.size()) < kNormalYawRateFitMinSamples) {
        last_normal_yaw_rate_fit_sample_count_ = static_cast<int>(samples.size());
        last_normal_yaw_rate_fit_reject_reason_ =
            kNormalYawRateFitNotEnoughWindowSamples;
        return;
    }
    last_normal_yaw_rate_fit_sample_count_ = static_cast<int>(samples.size());

    double sum_t = 0.0;
    double sum_yaw = 0.0;
    for (const YawObservationSample& sample : samples) {
        const double t = sample.t_s - newest_t;
        sum_t += t;
        sum_yaw += sample.yaw;
    }
    const double inv_n = 1.0 / static_cast<double>(samples.size());
    const double mean_t = sum_t * inv_n;
    const double mean_yaw = sum_yaw * inv_n;

    double denom = 0.0;
    double num = 0.0;
    double oldest_t = newest_t;
    for (const YawObservationSample& sample : samples) {
        const double t = sample.t_s - newest_t;
        oldest_t = std::min(oldest_t, sample.t_s);
        const double dt = t - mean_t;
        denom += dt * dt;
        num += dt * (sample.yaw - mean_yaw);
    }
    const double time_span = newest_t - oldest_t;
    last_normal_yaw_rate_fit_time_span_s_ = time_span;
    if (time_span < kNormalYawRateFitMinTimeSpanS || denom <= 1e-9) {
        last_normal_yaw_rate_fit_reject_reason_ = kNormalYawRateFitBadTimeSpan;
        return;
    }

    double yaw_rate = num / denom;
    if (!std::isfinite(yaw_rate)) {
        last_normal_yaw_rate_fit_reject_reason_ =
            kNormalYawRateFitNonFiniteVelocity;
        return;
    }

    double residual_sq = 0.0;
    int pair_source_count = 0;
    int single_source_count = 0;
    for (const YawObservationSample& sample : samples) {
        const double t = sample.t_s - newest_t;
        const double fit_yaw = mean_yaw + yaw_rate * (t - mean_t);
        const double error = sample.yaw - fit_yaw;
        residual_sq += error * error;
        if (sample.source == 1) {
            pair_source_count++;
        } else if (sample.source == 2) {
            single_source_count++;
        }
    }
    const double fit_rms =
        std::sqrt(residual_sq / static_cast<double>(samples.size()));
    last_normal_yaw_rate_fit_rms_rad_ = fit_rms;
    last_normal_yaw_rate_fit_raw_rad_s_ = yaw_rate;
    if (!std::isfinite(fit_rms) || fit_rms > kNormalYawRateFitMaxRmsRad) {
        last_normal_yaw_rate_fit_reject_reason_ = kNormalYawRateFitHighRms;
        return;
    }

    if (std::abs(yaw_rate) < kNormalYawRateDeadbandRadS) {
        yaw_rate = 0.0;
    } else if (std::abs(yaw_rate) > kNormalYawRateMaxRadS) {
        yaw_rate = std::copysign(kNormalYawRateMaxRadS, yaw_rate);
    }

    const double blend =
        pair_source_count >= single_source_count
            ? kNormalYawRatePairBlend
            : kNormalYawRateSingleBlend;
    yaw_rate = x_(7) + blend * (yaw_rate - x_(7));

    const double delta = yaw_rate - x_(7);
    if (std::abs(delta) > kNormalYawRateMaxStepRadS) {
        yaw_rate = x_(7) + std::copysign(kNormalYawRateMaxStepRadS, delta);
    }

    x_(7) = yaw_rate;
    if (P_.rows() > 7 && P_.cols() > 7) {
        P_(7, 7) = std::min(P_(7, 7), 0.50);
    }
    last_normal_yaw_rate_fit_accepted_ = 1;
    last_normal_yaw_rate_fit_reject_reason_ = kNormalYawRateFitAccepted;
    last_normal_yaw_rate_fit_applied_rad_s_ = yaw_rate;
}

void YpdAngleTracker::updateNormalCenterVelocityFromHistory()
{
    if (!initialized_ || is_outpost_ || armor_num_ != 4 || x_.size() < kStateDim) {
        return;
    }
    last_normal_velocity_history_size_ =
        static_cast<int>(normal_center_velocity_history_.size());
    last_normal_velocity_fit_sample_count_ = 0;
    last_normal_velocity_fit_accepted_ = 0;
    last_normal_velocity_fit_reject_reason_ = kNormalVelocityFitNotEvaluated;
    last_normal_velocity_fit_time_span_s_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_net_displacement_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_raw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_yaw_rate_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_yaw_mean_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_yaw_span_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_pair_sample_count_ = 0;
    last_normal_velocity_fit_single_sample_count_ = 0;
    last_normal_velocity_fit_group_count_ = 0;
    last_normal_velocity_fit_grouped_used_ = 0;
    last_normal_velocity_fit_grouped_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_grouped_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_rot_comp_used_ = 0;
    last_normal_velocity_fit_rot_comp_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_pos_yaw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_pos_yaw_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_neg_yaw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_neg_yaw_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_applied_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();

    if (static_cast<int>(normal_center_velocity_history_.size()) <
        kNormalVelocityFitMinSamples) {
        last_normal_velocity_fit_reject_reason_ =
            kNormalVelocityFitNotEnoughHistory;
        return;
    }

    const double newest_t = normal_center_velocity_history_.back().t_s;
    std::vector<MotionSample> samples;
    samples.reserve(normal_center_velocity_history_.size());
    for (auto it = normal_center_velocity_history_.rbegin();
         it != normal_center_velocity_history_.rend(); ++it) {
        const double age = newest_t - it->t_s;
        if (!std::isfinite(age) || age > kNormalVelocityFitWindowS) break;
        if (std::isfinite(it->center_x) && std::isfinite(it->center_y)) {
            samples.push_back(*it);
        }
    }
    if (static_cast<int>(samples.size()) < kNormalVelocityFitMinSamples) {
        last_normal_velocity_fit_sample_count_ = static_cast<int>(samples.size());
        last_normal_velocity_fit_reject_reason_ =
            kNormalVelocityFitNotEnoughWindowSamples;
        return;
    }
    last_normal_velocity_fit_sample_count_ = static_cast<int>(samples.size());

    double sum_t = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_frame_yaw_rate = 0.0;
    int frame_yaw_rate_count = 0;
    double reference_frame_yaw = std::numeric_limits<double>::quiet_NaN();
    for (const MotionSample& sample : samples) {
        if (std::isfinite(sample.frame_yaw)) {
            reference_frame_yaw = sample.frame_yaw;
            break;
        }
    }
    double sum_frame_yaw_rel = 0.0;
    double min_frame_yaw_rel = std::numeric_limits<double>::infinity();
    double max_frame_yaw_rel = -std::numeric_limits<double>::infinity();
    int frame_yaw_count = 0;
    int pair_sample_count = 0;
    int single_sample_count = 0;
    for (const MotionSample& sample : samples) {
        const double t = sample.t_s - newest_t;
        sum_t += t;
        sum_x += sample.center_x;
        sum_y += sample.center_y;
        if (std::isfinite(sample.frame_yaw) && std::isfinite(reference_frame_yaw)) {
            const double rel_yaw = normalizeAngle(sample.frame_yaw - reference_frame_yaw);
            sum_frame_yaw_rel += rel_yaw;
            min_frame_yaw_rel = std::min(min_frame_yaw_rel, rel_yaw);
            max_frame_yaw_rel = std::max(max_frame_yaw_rel, rel_yaw);
            ++frame_yaw_count;
        }
        if (std::isfinite(sample.frame_yaw_rate)) {
            sum_frame_yaw_rate += sample.frame_yaw_rate;
            ++frame_yaw_rate_count;
        }
        if (sample.source == 1) {
            ++pair_sample_count;
        } else if (sample.source == 2) {
            ++single_sample_count;
        }
    }
    const double inv_n = 1.0 / static_cast<double>(samples.size());
    const double mean_t = sum_t * inv_n;
    const double mean_x = sum_x * inv_n;
    const double mean_y = sum_y * inv_n;
    const double mean_frame_yaw_rate =
        frame_yaw_rate_count > 0
            ? sum_frame_yaw_rate / static_cast<double>(frame_yaw_rate_count)
            : std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_yaw_rate_rad_s_ = mean_frame_yaw_rate;
    if (frame_yaw_count > 0 && std::isfinite(reference_frame_yaw)) {
        last_normal_velocity_fit_frame_yaw_mean_rad_ =
            normalizeAngle(reference_frame_yaw +
                           sum_frame_yaw_rel / static_cast<double>(frame_yaw_count));
        last_normal_velocity_fit_frame_yaw_span_rad_ =
            max_frame_yaw_rel - min_frame_yaw_rel;
    }
    last_normal_velocity_fit_pair_sample_count_ = pair_sample_count;
    last_normal_velocity_fit_single_sample_count_ = single_sample_count;

    double denom = 0.0;
    double num_x = 0.0;
    double num_y = 0.0;
    double oldest_t = newest_t;
    for (const MotionSample& sample : samples) {
        const double t = sample.t_s - newest_t;
        oldest_t = std::min(oldest_t, sample.t_s);
        const double dt = t - mean_t;
        denom += dt * dt;
        num_x += dt * (sample.center_x - mean_x);
        num_y += dt * (sample.center_y - mean_y);
    }
    const double time_span = newest_t - oldest_t;
    last_normal_velocity_fit_time_span_s_ = time_span;
    if (time_span < 0.08 || denom <= 1e-9) {
        last_normal_velocity_fit_reject_reason_ =
            kNormalVelocityFitBadTimeSpan;
        return;
    }

    double vx = num_x / denom;
    double vy = num_y / denom;
    if (!std::isfinite(vx) || !std::isfinite(vy)) {
        last_normal_velocity_fit_reject_reason_ =
            kNormalVelocityFitNonFiniteVelocity;
        return;
    }

    const MotionSample& newest = normal_center_velocity_history_.back();
    const MotionSample& oldest = samples.back();
    double net_displacement =
        std::hypot(newest.center_x - oldest.center_x, newest.center_y - oldest.center_y);
    double residual_sq = 0.0;
    for (const MotionSample& sample : samples) {
        const double t = sample.t_s - newest_t;
        const double fit_x = mean_x + vx * (t - mean_t);
        const double fit_y = mean_y + vy * (t - mean_t);
        const double ex = sample.center_x - fit_x;
        const double ey = sample.center_y - fit_y;
        residual_sq += ex * ex + ey * ey;
    }
    double fit_rms =
        std::sqrt(residual_sq / static_cast<double>(samples.size()));
    const double ordinary_vx = vx;
    const double ordinary_vy = vy;

    struct TransformedVelocitySample {
        double t = 0.0;
        double x = 0.0;
        double y = 0.0;
    };
    const auto compute_frame_transform_fit =
        [&](double yaw_sign, double* out_speed, double* out_rms) {
            if (!std::isfinite(reference_frame_yaw) || out_speed == nullptr ||
                out_rms == nullptr) {
                return;
            }
            std::vector<TransformedVelocitySample> transformed;
            transformed.reserve(samples.size());
            for (const MotionSample& sample : samples) {
                if (!std::isfinite(sample.frame_yaw) ||
                    !std::isfinite(sample.center_x) ||
                    !std::isfinite(sample.center_y)) {
                    continue;
                }
                const double rel_yaw = normalizeAngle(sample.frame_yaw - reference_frame_yaw);
                const double angle = yaw_sign * rel_yaw;
                const double c = std::cos(angle);
                const double s = std::sin(angle);
                TransformedVelocitySample transformed_sample;
                transformed_sample.t = sample.t_s - newest_t;
                transformed_sample.x = c * sample.center_x - s * sample.center_y;
                transformed_sample.y = s * sample.center_x + c * sample.center_y;
                transformed.push_back(transformed_sample);
            }
            if (static_cast<int>(transformed.size()) < kNormalVelocityFitMinSamples) {
                return;
            }
            double sum_tt = 0.0;
            double sum_xx = 0.0;
            double sum_yy = 0.0;
            for (const TransformedVelocitySample& sample : transformed) {
                sum_tt += sample.t;
                sum_xx += sample.x;
                sum_yy += sample.y;
            }
            const double inv_transformed_n =
                1.0 / static_cast<double>(transformed.size());
            const double transformed_mean_t = sum_tt * inv_transformed_n;
            const double transformed_mean_x = sum_xx * inv_transformed_n;
            const double transformed_mean_y = sum_yy * inv_transformed_n;
            double transformed_denom = 0.0;
            double transformed_num_x = 0.0;
            double transformed_num_y = 0.0;
            for (const TransformedVelocitySample& sample : transformed) {
                const double dt = sample.t - transformed_mean_t;
                transformed_denom += dt * dt;
                transformed_num_x += dt * (sample.x - transformed_mean_x);
                transformed_num_y += dt * (sample.y - transformed_mean_y);
            }
            if (transformed_denom <= 1e-9) {
                return;
            }
            const double transformed_vx = transformed_num_x / transformed_denom;
            const double transformed_vy = transformed_num_y / transformed_denom;
            if (!std::isfinite(transformed_vx) || !std::isfinite(transformed_vy)) {
                return;
            }
            double transformed_residual_sq = 0.0;
            for (const TransformedVelocitySample& sample : transformed) {
                const double fit_x =
                    transformed_mean_x + transformed_vx * (sample.t - transformed_mean_t);
                const double fit_y =
                    transformed_mean_y + transformed_vy * (sample.t - transformed_mean_t);
                const double ex = sample.x - fit_x;
                const double ey = sample.y - fit_y;
                transformed_residual_sq += ex * ex + ey * ey;
            }
            *out_speed = std::hypot(transformed_vx, transformed_vy);
            *out_rms = std::sqrt(
                transformed_residual_sq / static_cast<double>(transformed.size()));
        };
    compute_frame_transform_fit(
        1.0, &last_normal_velocity_fit_frame_transform_pos_yaw_speed_mps_,
        &last_normal_velocity_fit_frame_transform_pos_yaw_rms_m_);
    compute_frame_transform_fit(
        -1.0, &last_normal_velocity_fit_frame_transform_neg_yaw_speed_mps_,
        &last_normal_velocity_fit_frame_transform_neg_yaw_rms_m_);

    double yaw_rate_gate_value = std::abs(x_(7));
    if (std::isfinite(last_normal_yaw_rate_fit_raw_rad_s_)) {
        yaw_rate_gate_value =
            std::max(yaw_rate_gate_value, std::abs(last_normal_yaw_rate_fit_raw_rad_s_));
    }
    const bool high_yaw_rate = yaw_rate_gate_value > kNormalSpinVelocityYawRateGateRadS;
    const double min_displacement = high_yaw_rate
        ? kNormalSpinVelocityMinDisplacementM
        : kNormalVelocityMinDisplacementM;
    const double max_fit_rms = high_yaw_rate
        ? kNormalSpinVelocityMaxFitRmsM
        : kNormalVelocityMaxFitRmsM;
    const double velocity_deadband = high_yaw_rate
        ? kNormalSpinVelocityDeadbandBaseMps
        : kNormalVelocityDeadbandMps;

    struct GroupedVelocityFit {
        bool valid = false;
        int group_count = 0;
        double vx = std::numeric_limits<double>::quiet_NaN();
        double vy = std::numeric_limits<double>::quiet_NaN();
        double speed = std::numeric_limits<double>::quiet_NaN();
        double rms = std::numeric_limits<double>::quiet_NaN();
        double displacement = std::numeric_limits<double>::quiet_NaN();
    };

    const auto compute_grouped_velocity_fit = [&]() {
        GroupedVelocityFit out;
        std::vector<int> groups;
        std::vector<int> counts;
        for (const MotionSample& sample : samples) {
            if (!std::isfinite(sample.center_x) ||
                !std::isfinite(sample.center_y) ||
                sample.group_id < 0) {
                continue;
            }
            auto found = std::find(groups.begin(), groups.end(), sample.group_id);
            if (found == groups.end()) {
                groups.push_back(sample.group_id);
                counts.push_back(1);
            } else {
                ++counts[static_cast<int>(std::distance(groups.begin(), found))];
            }
        }
        std::vector<int> kept_groups;
        for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
            if (counts[i] >= kNormalVelocityGroupedFitMinSamplesPerGroup) {
                kept_groups.push_back(groups[i]);
            }
        }
        if (static_cast<int>(kept_groups.size()) < 2) return out;

        std::vector<const MotionSample*> kept_samples;
        for (const MotionSample& sample : samples) {
            if (!std::isfinite(sample.center_x) ||
                !std::isfinite(sample.center_y) ||
                sample.group_id < 0) {
                continue;
            }
            if (std::find(kept_groups.begin(), kept_groups.end(), sample.group_id) !=
                kept_groups.end()) {
                kept_samples.push_back(&sample);
            }
        }
        if (static_cast<int>(kept_samples.size()) <
                kNormalVelocityGroupedFitMinSamples ||
            static_cast<int>(kept_samples.size()) <=
                static_cast<int>(kept_groups.size()) + 1) {
            return out;
        }

        Eigen::MatrixXd A(
            static_cast<int>(kept_samples.size()),
            static_cast<int>(kept_groups.size()) + 1);
        Eigen::VectorXd bx(static_cast<int>(kept_samples.size()));
        Eigen::VectorXd by(static_cast<int>(kept_samples.size()));
        A.setZero();
        for (int row = 0; row < static_cast<int>(kept_samples.size()); ++row) {
            const MotionSample& sample = *kept_samples[row];
            A(row, 0) = sample.t_s - newest_t;
            const auto found =
                std::find(kept_groups.begin(), kept_groups.end(), sample.group_id);
            if (found == kept_groups.end()) continue;
            const int group_col =
                1 + static_cast<int>(std::distance(kept_groups.begin(), found));
            A(row, group_col) = 1.0;
            bx(row) = sample.center_x;
            by(row) = sample.center_y;
        }

        const Eigen::VectorXd coef_x =
            A.colPivHouseholderQr().solve(bx);
        const Eigen::VectorXd coef_y =
            A.colPivHouseholderQr().solve(by);
        if (coef_x.size() == 0 || coef_y.size() == 0 ||
            !std::isfinite(coef_x(0)) || !std::isfinite(coef_y(0))) {
            return out;
        }
        const Eigen::VectorXd residual_x = A * coef_x - bx;
        const Eigen::VectorXd residual_y = A * coef_y - by;
        double residual_grouped_sq = 0.0;
        for (int i = 0; i < residual_x.size(); ++i) {
            residual_grouped_sq +=
                residual_x(i) * residual_x(i) + residual_y(i) * residual_y(i);
        }
        out.valid = true;
        out.group_count = static_cast<int>(kept_groups.size());
        out.vx = coef_x(0);
        out.vy = coef_y(0);
        out.speed = std::hypot(out.vx, out.vy);
        out.rms = std::sqrt(
            residual_grouped_sq / static_cast<double>(kept_samples.size()));
        out.displacement = out.speed * time_span;
        return out;
    };

    const GroupedVelocityFit grouped_fit = compute_grouped_velocity_fit();
    last_normal_velocity_fit_group_count_ = grouped_fit.group_count;
    last_normal_velocity_fit_grouped_speed_mps_ = grouped_fit.speed;
    last_normal_velocity_fit_grouped_rms_m_ = grouped_fit.rms;
    // Diagnostic only for now. The grouped common-velocity model is physically
    // meaningful as a branch/source offset probe, but current simulator rows
    // show it can be lower than the ordinary raw fit. Do not write it into the
    // velocity state until it passes spin/linear/combined validation.
    last_normal_velocity_fit_net_displacement_m_ = net_displacement;
    last_normal_velocity_fit_rms_m_ = fit_rms;

    int reject_reason = kNormalVelocityFitAccepted;
    bool force_zero_velocity = false;
    if (net_displacement < min_displacement) {
        force_zero_velocity = true;
        reject_reason = kNormalVelocityFitLowDisplacement;
    } else if (!std::isfinite(fit_rms) || fit_rms > max_fit_rms) {
        force_zero_velocity = true;
        reject_reason = kNormalVelocityFitHighRms;
    }
    const double raw_speed = std::hypot(ordinary_vx, ordinary_vy);
    last_normal_velocity_fit_raw_speed_mps_ = raw_speed;
    last_normal_velocity_fit_raw_vx_mps_ = ordinary_vx;
    last_normal_velocity_fit_raw_vy_mps_ = ordinary_vy;
    if (std::isfinite(mean_frame_yaw_rate) &&
        std::abs(mean_frame_yaw_rate) > kNormalVelocityFrameYawRateDeadbandRadS &&
        std::abs(mean_frame_yaw_rate) <= kNormalVelocityFrameYawRateMaxRadS) {
        // Diagnostic only: center samples can carry a frame-rotation term, but
        // applying a speed-threshold compensation here would be an empirical
        // simulator fit. Keep the candidate visible in logs and leave the
        // accepted velocity governed by the raw center observations until the
        // coordinate-frame model is proven explicitly.
        const double comp_vx = ordinary_vx - mean_frame_yaw_rate * mean_y;
        const double comp_vy = ordinary_vy + mean_frame_yaw_rate * mean_x;
        const double comp_speed = std::hypot(comp_vx, comp_vy);
        if (std::isfinite(comp_speed)) {
            last_normal_velocity_fit_rot_comp_speed_mps_ = comp_speed;
        }
    }
    double speed = std::hypot(vx, vy);
    if (!std::isfinite(last_normal_velocity_fit_rot_comp_speed_mps_)) {
        last_normal_velocity_fit_rot_comp_speed_mps_ = speed;
    }
    if (!force_zero_velocity && speed < velocity_deadband) {
        force_zero_velocity = true;
        reject_reason = kNormalVelocityFitLowSpeed;
    }
    if (!force_zero_velocity && high_yaw_rate && speed < kNormalSpinVelocityDeadbandMps) {
        force_zero_velocity = true;
        reject_reason = kNormalVelocityFitSpinLowSpeed;
    }
    if (force_zero_velocity) {
        const double prior_speed = std::hypot(x_(1), x_(3));
        const bool can_hold_previous_velocity =
            high_yaw_rate &&
            (reject_reason == kNormalVelocityFitHighRms ||
             reject_reason == kNormalVelocityFitSpinLowSpeed) &&
            prior_speed >= kNormalVelocityHoldMinPriorSpeedMps &&
            speed >= kNormalVelocityHoldMinRawSpeedMps &&
            net_displacement >= kNormalVelocityHoldMinDisplacementM &&
            std::isfinite(fit_rms) &&
            fit_rms <= kNormalVelocityHoldMaxRmsM;
        if (can_hold_previous_velocity) {
            x_(1) *= kNormalVelocityHoldDecay;
            x_(3) *= kNormalVelocityHoldDecay;
            last_normal_velocity_fit_reject_reason_ =
                kNormalVelocityFitHeldPrevious;
            last_normal_velocity_fit_applied_speed_mps_ =
                std::hypot(x_(1), x_(3));
            last_normal_velocity_fit_applied_vx_mps_ = x_(1);
            last_normal_velocity_fit_applied_vy_mps_ = x_(3);
            return;
        }
        x_(1) = 0.0;
        x_(3) = 0.0;
        last_normal_velocity_fit_reject_reason_ = reject_reason;
        last_normal_velocity_fit_applied_speed_mps_ = 0.0;
        last_normal_velocity_fit_applied_vx_mps_ = 0.0;
        last_normal_velocity_fit_applied_vy_mps_ = 0.0;
        return;
    } else if (speed > kNormalVelocityMaxMps) {
        const double scale = kNormalVelocityMaxMps / speed;
        vx *= scale;
        vy *= scale;
        speed = kNormalVelocityMaxMps;
    }

    const double dvx = vx - x_(1);
    const double dvy = vy - x_(3);
    const double dv = std::hypot(dvx, dvy);
    if (dv > kNormalVelocityMaxStepMps && dv > 1e-9) {
        const double scale = kNormalVelocityMaxStepMps / dv;
        vx = x_(1) + dvx * scale;
        vy = x_(3) + dvy * scale;
    }

    x_(1) = vx;
    x_(3) = vy;
    last_normal_velocity_fit_accepted_ = 1;
    last_normal_velocity_fit_reject_reason_ = kNormalVelocityFitAccepted;
    last_normal_velocity_fit_applied_speed_mps_ = std::hypot(vx, vy);
    last_normal_velocity_fit_applied_vx_mps_ = vx;
    last_normal_velocity_fit_applied_vy_mps_ = vy;
}

bool YpdAngleTracker::updateNormalCandidateCenterVelocityFromHistory()
{
    if (!initialized_ || is_outpost_ || armor_num_ != 4 || x_.size() < kStateDim) {
        return false;
    }

    double yaw_rate_gate_value = std::abs(x_(7));
    if (std::isfinite(last_normal_yaw_rate_fit_raw_rad_s_)) {
        yaw_rate_gate_value =
            std::max(yaw_rate_gate_value, std::abs(last_normal_yaw_rate_fit_raw_rad_s_));
    }
    const bool high_yaw_candidate =
        yaw_rate_gate_value > kNormalCandidateVelocityHighYawGateRadS;
    if (high_yaw_candidate) {
        return false;
    }
    if (static_cast<int>(normal_single_center_candidate_history_.size()) <
        kNormalCandidateVelocityFitMinSamples) {
        return false;
    }

    const double newest_t = normal_single_center_candidate_history_.back().t_s;
    std::vector<MotionSample> samples;
    samples.reserve(normal_single_center_candidate_history_.size());
    for (auto it = normal_single_center_candidate_history_.rbegin();
         it != normal_single_center_candidate_history_.rend(); ++it) {
        const double age = newest_t - it->t_s;
        if (!std::isfinite(age) || age > kNormalVelocityFitWindowS) break;
        if (std::isfinite(it->center_x) && std::isfinite(it->center_y)) {
            samples.push_back(*it);
        }
    }
    if (static_cast<int>(samples.size()) < kNormalCandidateVelocityFitMinSamples) {
        return false;
    }

    double sum_t = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_frame_yaw_rate = 0.0;
    int frame_yaw_rate_count = 0;
    double reference_frame_yaw = std::numeric_limits<double>::quiet_NaN();
    for (const MotionSample& sample : samples) {
        if (std::isfinite(sample.frame_yaw)) {
            reference_frame_yaw = sample.frame_yaw;
            break;
        }
    }
    double sum_frame_yaw_rel = 0.0;
    double min_frame_yaw_rel = std::numeric_limits<double>::infinity();
    double max_frame_yaw_rel = -std::numeric_limits<double>::infinity();
    int frame_yaw_count = 0;
    for (const MotionSample& sample : samples) {
        const double t = sample.t_s - newest_t;
        sum_t += t;
        sum_x += sample.center_x;
        sum_y += sample.center_y;
        if (std::isfinite(sample.frame_yaw) && std::isfinite(reference_frame_yaw)) {
            const double rel_yaw = normalizeAngle(sample.frame_yaw - reference_frame_yaw);
            sum_frame_yaw_rel += rel_yaw;
            min_frame_yaw_rel = std::min(min_frame_yaw_rel, rel_yaw);
            max_frame_yaw_rel = std::max(max_frame_yaw_rel, rel_yaw);
            ++frame_yaw_count;
        }
        if (std::isfinite(sample.frame_yaw_rate)) {
            sum_frame_yaw_rate += sample.frame_yaw_rate;
            ++frame_yaw_rate_count;
        }
    }
    const double inv_n = 1.0 / static_cast<double>(samples.size());
    const double mean_t = sum_t * inv_n;
    const double mean_x = sum_x * inv_n;
    const double mean_y = sum_y * inv_n;

    double denom = 0.0;
    double num_x = 0.0;
    double num_y = 0.0;
    double oldest_t = newest_t;
    for (const MotionSample& sample : samples) {
        const double t = sample.t_s - newest_t;
        oldest_t = std::min(oldest_t, sample.t_s);
        const double dt = t - mean_t;
        denom += dt * dt;
        num_x += dt * (sample.center_x - mean_x);
        num_y += dt * (sample.center_y - mean_y);
    }
    const double time_span = newest_t - oldest_t;
    if (time_span < kNormalCandidateVelocityFitMinTimeSpanS || denom <= 1e-9) {
        return false;
    }

    double vx = num_x / denom;
    double vy = num_y / denom;
    if (!std::isfinite(vx) || !std::isfinite(vy)) {
        return false;
    }

    const MotionSample& newest = normal_single_center_candidate_history_.back();
    const MotionSample& oldest = samples.back();
    const double net_displacement =
        std::hypot(newest.center_x - oldest.center_x, newest.center_y - oldest.center_y);
    double residual_sq = 0.0;
    for (const MotionSample& sample : samples) {
        const double t = sample.t_s - newest_t;
        const double fit_x = mean_x + vx * (t - mean_t);
        const double fit_y = mean_y + vy * (t - mean_t);
        const double ex = sample.center_x - fit_x;
        const double ey = sample.center_y - fit_y;
        residual_sq += ex * ex + ey * ey;
    }
    const double fit_rms =
        std::sqrt(residual_sq / static_cast<double>(samples.size()));
    const double raw_vx = vx;
    const double raw_vy = vy;
    const double raw_speed = std::hypot(vx, vy);
    if (!std::isfinite(raw_speed) || !std::isfinite(fit_rms)) {
        return false;
    }
    if (net_displacement < kNormalCandidateVelocityMinDisplacementM ||
        fit_rms > kNormalCandidateVelocityMaxFitRmsM ||
        raw_speed < kNormalVelocityDeadbandMps) {
        return false;
    }

    double speed = raw_speed;
    if (speed > kNormalVelocityMaxMps) {
        const double scale = kNormalVelocityMaxMps / speed;
        vx *= scale;
        vy *= scale;
        speed = kNormalVelocityMaxMps;
    }

    const double dvx = vx - x_(1);
    const double dvy = vy - x_(3);
    const double dv = std::hypot(dvx, dvy);
    if (dv > kNormalVelocityMaxStepMps && dv > 1e-9) {
        const double scale = kNormalVelocityMaxStepMps / dv;
        vx = x_(1) + dvx * scale;
        vy = x_(3) + dvy * scale;
    }

    last_normal_velocity_observation_source_ = 3;
    last_normal_velocity_sample_t_s_ = newest.t_s;
    last_normal_velocity_sample_x_m_ = newest.center_x;
    last_normal_velocity_sample_y_m_ = newest.center_y;
    last_normal_velocity_sample_frame_yaw_rad_ = newest.frame_yaw;
    last_normal_velocity_sample_group_id_ = newest.group_id;
    last_normal_velocity_fit_sample_count_ = static_cast<int>(samples.size());
    last_normal_velocity_fit_accepted_ = 1;
    last_normal_velocity_fit_reject_reason_ = kNormalVelocityFitAccepted;
    last_normal_velocity_fit_time_span_s_ = time_span;
    last_normal_velocity_fit_net_displacement_m_ = net_displacement;
    last_normal_velocity_fit_rms_m_ = fit_rms;
    last_normal_velocity_fit_raw_speed_mps_ = raw_speed;
    last_normal_velocity_fit_raw_vx_mps_ = raw_vx;
    last_normal_velocity_fit_raw_vy_mps_ = raw_vy;
    last_normal_velocity_fit_frame_yaw_rate_rad_s_ =
        frame_yaw_rate_count > 0
            ? sum_frame_yaw_rate / static_cast<double>(frame_yaw_rate_count)
            : std::numeric_limits<double>::quiet_NaN();
    if (frame_yaw_count > 0 && std::isfinite(reference_frame_yaw)) {
        last_normal_velocity_fit_frame_yaw_mean_rad_ =
            normalizeAngle(reference_frame_yaw +
                           sum_frame_yaw_rel / static_cast<double>(frame_yaw_count));
        last_normal_velocity_fit_frame_yaw_span_rad_ =
            max_frame_yaw_rel - min_frame_yaw_rel;
    }
    last_normal_velocity_fit_pair_sample_count_ = 0;
    last_normal_velocity_fit_single_sample_count_ =
        static_cast<int>(samples.size());
    last_normal_velocity_fit_rot_comp_used_ = 0;
    last_normal_velocity_fit_rot_comp_speed_mps_ = raw_speed;
    last_normal_velocity_fit_frame_transform_pos_yaw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_pos_yaw_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_neg_yaw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    last_normal_velocity_fit_frame_transform_neg_yaw_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();

    x_(1) = vx;
    x_(3) = vy;
    last_normal_velocity_fit_applied_speed_mps_ = std::hypot(vx, vy);
    last_normal_velocity_fit_applied_vx_mps_ = vx;
    last_normal_velocity_fit_applied_vy_mps_ = vy;
    return true;
}

double YpdAngleTracker::posteriorYawRateLinearAccel(int window) const
{
    const int effective_window = std::max(2, window);
    if (static_cast<int>(motion_history_.size()) < effective_window) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto begin = motion_history_.end() - effective_window;
    const double t_base = motion_history_.back().t_s;
    double sum_t = 0.0;
    double sum_y = 0.0;
    double sum_tt = 0.0;
    double sum_ty = 0.0;
    int count = 0;

    for (auto it = begin; it != motion_history_.end(); ++it) {
        const double dt = it->t_s - t_base;
        const double yaw_rate = it->yaw_rate;
        if (!std::isfinite(dt) || !std::isfinite(yaw_rate)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        sum_t += dt;
        sum_y += yaw_rate;
        sum_tt += dt * dt;
        sum_ty += dt * yaw_rate;
        count++;
    }

    const double denom = static_cast<double>(count) * sum_tt - sum_t * sum_t;
    if (std::abs(denom) < 1e-9) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double slope =
        (static_cast<double>(count) * sum_ty - sum_t * sum_y) / denom;
    return std::abs(slope);
}

double YpdAngleTracker::posteriorCenterQuadraticAccel(int window) const
{
    const int effective_window = std::max(3, window);
    if (static_cast<int>(motion_history_.size()) < effective_window) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto begin = motion_history_.end() - effective_window;
    const double t_base = motion_history_.back().t_s;

    const auto fit_component = [&](double MotionSample::*member) {
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d B = Eigen::Vector3d::Zero();

        for (auto it = begin; it != motion_history_.end(); ++it) {
            const double dt = it->t_s - t_base;
            const double value = (*it).*member;
            if (!std::isfinite(dt) || !std::isfinite(value)) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            const double dt2 = dt * dt;
            const double dt3 = dt2 * dt;
            const double dt4 = dt3 * dt;

            A(0, 0) += dt4;
            A(0, 1) += dt3;
            A(0, 2) += dt2;
            A(1, 0) += dt3;
            A(1, 1) += dt2;
            A(1, 2) += dt;
            A(2, 0) += dt2;
            A(2, 1) += dt;
            A(2, 2) += 1.0;

            B(0) += dt2 * value;
            B(1) += dt * value;
            B(2) += value;
        }

        const Eigen::LDLT<Eigen::Matrix3d> ldlt(A);
        if (ldlt.info() != Eigen::Success) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const Eigen::Vector3d coeffs = ldlt.solve(B);
        if (!coeffs.allFinite()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 2.0 * coeffs(0);
    };

    const double accel_x = fit_component(&MotionSample::center_x);
    const double accel_y = fit_component(&MotionSample::center_y);
    if (!std::isfinite(accel_x) || !std::isfinite(accel_y)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::hypot(accel_x, accel_y);
}

double YpdAngleTracker::recentVelocityUpdateMean(int window) const
{
    const int effective_window = std::max(1, window);
    if (static_cast<int>(velocity_update_history_.size()) < effective_window) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto begin = velocity_update_history_.end() - effective_window;
    double sum = 0.0;
    int count = 0;
    for (auto it = begin; it != velocity_update_history_.end(); ++it) {
        if (!std::isfinite(*it)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        sum += *it;
        count++;
    }
    if (count <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / static_cast<double>(count);
}

void YpdAngleTracker::recordMotionUpdate(
    const Eigen::VectorXd& prior_state, const Eigen::VectorXd& posterior_state)
{
    if (prior_state.size() < kStateDim || posterior_state.size() < kStateDim) {
        return;
    }

    last_motion_prior_yaw_rate_ = prior_state(7);
    last_motion_posterior_yaw_rate_ = posterior_state(7);
    last_motion_prior_vx_ = prior_state(1);
    last_motion_prior_vy_ = prior_state(3);
    last_motion_posterior_vx_ = posterior_state(1);
    last_motion_posterior_vy_ = posterior_state(3);
    last_motion_center_update_norm_ =
        std::hypot(posterior_state(0) - prior_state(0),
                   posterior_state(2) - prior_state(2));
    last_motion_velocity_update_norm_ =
        std::hypot(posterior_state(1) - prior_state(1),
                   posterior_state(3) - prior_state(3));
    last_motion_speed_update_abs_ =
        std::abs(std::hypot(posterior_state(1), posterior_state(3)) -
                 std::hypot(prior_state(1), prior_state(3)));
    if (std::isfinite(last_motion_velocity_update_norm_)) {
        velocity_update_history_.push_back(last_motion_velocity_update_norm_);
        while (static_cast<int>(velocity_update_history_.size()) >
               kUpdateMetricHistoryCapacity) {
            velocity_update_history_.pop_front();
        }
    }
}

} // namespace RobotEstimator
