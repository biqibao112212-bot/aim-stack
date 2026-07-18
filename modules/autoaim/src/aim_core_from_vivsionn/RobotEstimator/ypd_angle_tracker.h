#pragma once

#include <deque>
#include <array>
#include <Eigen/Dense>
#include <limits>
#include <vector>

#include "generalDeclaration.h"

namespace RobotEstimator {

class YpdAngleTracker {
public:
    enum ObservationRejectReason {
        kObservationNotEvaluated = 0,
        kObservationAccepted = 1,
        kObservationSkippedByPairSelection = 2,
        kObservationSkippedByPrimarySelection = 3,
        kObservationRejectedByPriorNis = 4,
        kObservationRejectedByPhysicalGate = 5,
        kObservationSkippedByBatchCapacity = 6,
    };

    struct ObservationDiagnostic {
        int observation_index = -1;
        int matched_slot = -1;
        bool accepted = false;
        int reject_reason = kObservationNotEvaluated;
        int physical_reject_reason = 0;
        bool yaw_pi_flip = false;
        bool freeze_normal_geometry = false;
        double max_center_jump_m = std::numeric_limits<double>::quiet_NaN();
        Eigen::Vector4d observation = Eigen::Vector4d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        Eigen::Vector4d prior_predicted_measurement = Eigen::Vector4d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        Eigen::Vector4d prior_innovation = Eigen::Vector4d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        double prior_nis = std::numeric_limits<double>::quiet_NaN();
        Eigen::Vector4d posterior_innovation = Eigen::Vector4d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        double posterior_nis = std::numeric_limits<double>::quiet_NaN();
        int update_count_before = 0;
        int update_count_after = 0;
    };

    struct MotionSample {
        double t_s = 0.0;
        double center_x = 0.0;
        double center_y = 0.0;
        double yaw_rate = 0.0;
        double frame_yaw = std::numeric_limits<double>::quiet_NaN();
        double frame_yaw_rate = std::numeric_limits<double>::quiet_NaN();
        int source = 0;
        int group_id = -1;
    };

    struct YawObservationSample {
        double t_s = 0.0;
        double yaw = 0.0;
        int source = 0;
    };

    struct GeometryRecoveryConfig {
        int recovery_window_frames = 24;
        int recovery_cooldown_frames = 12;
        int mismatch_required_streak = 2;
        int min_matched_count = 2;
        double residual_z_sigma_threshold = 3.0;
        double residual_xy_sigma_threshold = 2.0;
        double covariance_inflation_scale = 36.0;
        double min_dr_variance = 2.5e-3;
        double min_h_variance = 6.25e-4;
    };

    YpdAngleTracker();

    void reset();
    void init(const rm::Armor& armor, int armor_num);
    void predict(double dt);
    void setFrameYaw(double yaw_rad);
    void setFrameYawRate(double yaw_rate_rad_s);
    void setGeometryRecoveryConfig(const GeometryRecoveryConfig& config);
    void noteObservationJump(bool observation_jump);
    void updateBatch(const std::vector<rm::Armor>& armors, int preferred_index = -1);
    int selectTrackedId(const rm::Armor& armor);
    void clearGeometryRecoveryHistory();
    const std::vector<int>& lastBatchMatchIds() const { return last_batch_match_ids_; }
    const std::vector<ObservationDiagnostic>& lastObservationDiagnostics() const
    {
        return last_observation_diagnostics_;
    }
    const std::deque<MotionSample>& normalCenterVelocityHistory() const
    {
        return normal_center_velocity_history_;
    }
    double currentFrameYaw() const { return current_frame_yaw_rad_; }

    bool isInitialized() const { return initialized_; }
    Eigen::VectorXd getState() const { return x_; }
    int getTrackedId() const { return tracked_id_; }
    int getArmorNum() const { return armor_num_; }
    double getArmorRadius(int id) const;
    Eigen::Vector4d getPredictedArmorState(int id) const;
    std::vector<Eigen::Vector4d> getPredictedArmorStates() const;
    bool diverged() const;
    bool badConvergence() const;
    int recentNisFailureCount() const;
    int nisWindowSize() const { return nis_window_size_; }
    double lastNis() const { return last_nis_; }
    Eigen::MatrixXd getCovariance() const { return P_; }
    int updateCount() const { return update_count_; }
    bool convergedStatus() const { return is_converged_; }
    double posteriorYawRateLinearAccel(int window) const;
    double posteriorCenterQuadraticAccel(int window) const;
    double recentVelocityUpdateMean(int window) const;
    double lastMotionPriorYawRate() const { return last_motion_prior_yaw_rate_; }
    double lastMotionPosteriorYawRate() const
    {
        return last_motion_posterior_yaw_rate_;
    }
    double lastMotionPriorVx() const { return last_motion_prior_vx_; }
    double lastMotionPriorVy() const { return last_motion_prior_vy_; }
    double lastMotionPosteriorVx() const { return last_motion_posterior_vx_; }
    double lastMotionPosteriorVy() const { return last_motion_posterior_vy_; }
    double lastMotionCenterUpdateNorm() const
    {
        return last_motion_center_update_norm_;
    }
    double lastMotionVelocityUpdateNorm() const
    {
        return last_motion_velocity_update_norm_;
    }
    double lastMotionSpeedUpdateAbs() const
    {
        return last_motion_speed_update_abs_;
    }
    int lastPhysicalRejectionCount() const { return last_physical_rejection_count_; }
    int lastPhysicalRejectionReason() const { return last_physical_rejection_reason_; }
    int lastNormalPairRequired() const { return last_normal_pair_required_; }
    int lastNormalPairFound() const { return last_normal_pair_found_; }
    int lastNormalUpdateClass() const { return last_normal_update_class_; }
    int lastNormalAcceptedCount() const { return last_normal_accepted_count_; }
    int lastNormalPairAcceptedCount() const { return last_normal_pair_accepted_count_; }
    double lastNormalPairScore() const { return last_normal_pair_score_; }
    double lastNormalPairCenterGap() const { return last_normal_pair_center_gap_m_; }
    double lastNormalPairCenterJump() const { return last_normal_pair_center_jump_m_; }
    double lastNormalPairCenterX() const { return last_normal_pair_center_x_m_; }
    double lastNormalPairCenterY() const { return last_normal_pair_center_y_m_; }
    int lastNormalSingleCenterCount() const { return last_normal_single_center_count_; }
    double lastNormalSingleCenterX() const { return last_normal_single_center_x_m_; }
    double lastNormalSingleCenterY() const { return last_normal_single_center_y_m_; }
    int lastNormalVelocityHistorySize() const { return last_normal_velocity_history_size_; }
    int lastNormalVelocityObservationSource() const
    {
        return last_normal_velocity_observation_source_;
    }
    double lastNormalVelocitySampleTime() const { return last_normal_velocity_sample_t_s_; }
    double lastNormalVelocitySampleX() const { return last_normal_velocity_sample_x_m_; }
    double lastNormalVelocitySampleY() const { return last_normal_velocity_sample_y_m_; }
    double lastNormalVelocitySampleFrameYaw() const
    {
        return last_normal_velocity_sample_frame_yaw_rad_;
    }
    int lastNormalVelocitySampleGroupId() const
    {
        return last_normal_velocity_sample_group_id_;
    }
    int lastNormalVelocityFitSampleCount() const
    {
        return last_normal_velocity_fit_sample_count_;
    }
    int lastNormalVelocityFitAccepted() const { return last_normal_velocity_fit_accepted_; }
    int lastNormalVelocityFitRejectReason() const
    {
        return last_normal_velocity_fit_reject_reason_;
    }
    double lastNormalVelocityFitTimeSpan() const
    {
        return last_normal_velocity_fit_time_span_s_;
    }
    double lastNormalVelocityFitNetDisplacement() const
    {
        return last_normal_velocity_fit_net_displacement_m_;
    }
    double lastNormalVelocityFitRms() const { return last_normal_velocity_fit_rms_m_; }
    double lastNormalVelocityFitRawSpeed() const
    {
        return last_normal_velocity_fit_raw_speed_mps_;
    }
    double lastNormalVelocityFitRawVx() const { return last_normal_velocity_fit_raw_vx_mps_; }
    double lastNormalVelocityFitRawVy() const { return last_normal_velocity_fit_raw_vy_mps_; }
    double lastNormalVelocityFitFrameYawRate() const
    {
        return last_normal_velocity_fit_frame_yaw_rate_rad_s_;
    }
    double lastNormalVelocityFitFrameYawMean() const
    {
        return last_normal_velocity_fit_frame_yaw_mean_rad_;
    }
    double lastNormalVelocityFitFrameYawSpan() const
    {
        return last_normal_velocity_fit_frame_yaw_span_rad_;
    }
    int lastNormalVelocityFitPairSampleCount() const
    {
        return last_normal_velocity_fit_pair_sample_count_;
    }
    int lastNormalVelocityFitSingleSampleCount() const
    {
        return last_normal_velocity_fit_single_sample_count_;
    }
    int lastNormalVelocityFitGroupCount() const
    {
        return last_normal_velocity_fit_group_count_;
    }
    int lastNormalVelocityFitGroupedUsed() const
    {
        return last_normal_velocity_fit_grouped_used_;
    }
    double lastNormalVelocityFitGroupedSpeed() const
    {
        return last_normal_velocity_fit_grouped_speed_mps_;
    }
    double lastNormalVelocityFitGroupedRms() const
    {
        return last_normal_velocity_fit_grouped_rms_m_;
    }
    int lastNormalVelocityFitRotCompUsed() const
    {
        return last_normal_velocity_fit_rot_comp_used_;
    }
    double lastNormalVelocityFitRotCompSpeed() const
    {
        return last_normal_velocity_fit_rot_comp_speed_mps_;
    }
    double lastNormalVelocityFitFrameTransformPosYawSpeed() const
    {
        return last_normal_velocity_fit_frame_transform_pos_yaw_speed_mps_;
    }
    double lastNormalVelocityFitFrameTransformPosYawRms() const
    {
        return last_normal_velocity_fit_frame_transform_pos_yaw_rms_m_;
    }
    double lastNormalVelocityFitFrameTransformNegYawSpeed() const
    {
        return last_normal_velocity_fit_frame_transform_neg_yaw_speed_mps_;
    }
    double lastNormalVelocityFitFrameTransformNegYawRms() const
    {
        return last_normal_velocity_fit_frame_transform_neg_yaw_rms_m_;
    }
    double lastNormalVelocityFitAppliedSpeed() const
    {
        return last_normal_velocity_fit_applied_speed_mps_;
    }
    double lastNormalVelocityFitAppliedVx() const
    {
        return last_normal_velocity_fit_applied_vx_mps_;
    }
    double lastNormalVelocityFitAppliedVy() const
    {
        return last_normal_velocity_fit_applied_vy_mps_;
    }
    int lastNormalYawRateHistorySize() const { return last_normal_yaw_rate_history_size_; }
    int lastNormalYawRateObservationSource() const
    {
        return last_normal_yaw_rate_observation_source_;
    }
    int lastNormalYawObservationCount() const { return last_normal_yaw_observation_count_; }
    double lastNormalYawObservationTime() const { return last_normal_yaw_observation_t_s_; }
    double lastNormalYawObservationRaw() const { return last_normal_yaw_observation_raw_rad_; }
    double lastNormalYawObservationUnwrapped() const
    {
        return last_normal_yaw_observation_unwrapped_rad_;
    }
    int lastNormalYawRateFitSampleCount() const
    {
        return last_normal_yaw_rate_fit_sample_count_;
    }
    int lastNormalYawRateFitAccepted() const { return last_normal_yaw_rate_fit_accepted_; }
    int lastNormalYawRateFitRejectReason() const
    {
        return last_normal_yaw_rate_fit_reject_reason_;
    }
    double lastNormalYawRateFitTimeSpan() const
    {
        return last_normal_yaw_rate_fit_time_span_s_;
    }
    double lastNormalYawRateFitRms() const { return last_normal_yaw_rate_fit_rms_rad_; }
    double lastNormalYawRateFitRaw() const
    {
        return last_normal_yaw_rate_fit_raw_rad_s_;
    }
    double lastNormalYawRateFitApplied() const
    {
        return last_normal_yaw_rate_fit_applied_rad_s_;
    }
    double lastGeometryResidualXyOverSigmaDr() const
    {
        return last_geometry_residual_xy_over_sigma_dr_;
    }
    double lastGeometryResidualZOverSigmaH() const
    {
        return last_geometry_residual_z_over_sigma_h_;
    }
    int geometryRecoveryInflationCount() const { return geometry_recovery_inflation_count_; }
    int outpostHeightPhase() const { return outpost_height_phase_; }
    bool isOutpostHeightPhaseLocked() const
    {
        return armor_num_ == 3 && outpost_height_phase_ >= 0;
    }
    int outpostHeightPhaseObservations() const
    {
        return outpost_height_phase_observations_;
    }
    double outpostHeightPhaseScore(int phase) const
    {
        if (phase < 0 || phase >= static_cast<int>(outpost_height_phase_scores_.size())) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return outpost_height_phase_scores_[phase];
    }
    int lastOutpostHeightId() const { return last_outpost_height_id_; }
    double lastOutpostHeightZ() const { return last_outpost_height_z_; }

private:
    struct GeometryResidualSummary {
        int matched_count = 0;
        double mean_residual_xy = std::numeric_limits<double>::quiet_NaN();
        double mean_residual_z = std::numeric_limits<double>::quiet_NaN();
        double mean_residual_xy_over_sigma_dr =
            std::numeric_limits<double>::quiet_NaN();
        double mean_residual_z_over_sigma_h =
            std::numeric_limits<double>::quiet_NaN();
    };

    double normalizeAngle(double angle) const;
    Eigen::Vector3d predictArmorPosition(const Eigen::VectorXd& state, int armor_id) const;
    Eigen::Vector4d predictArmorMeasurement(const Eigen::VectorXd& state, int armor_id) const;
    Eigen::Vector3d extractObservationYpd(const rm::Armor& armor) const;
    Eigen::MatrixXd buildMeasurementJacobian(const Eigen::VectorXd& state, int armor_id) const;
    Eigen::MatrixXd buildProcessNoise(double dt) const;
    Eigen::MatrixXd buildMeasurementNoise(const rm::Armor& armor) const;
    double computeMatchCost(const rm::Armor& armor, int armor_id) const;
    std::vector<int> assignArmorIds(const std::vector<rm::Armor>& armors) const;
    int selectBestArmorId(const rm::Armor& armor) const;
    bool converged();
    bool correctWithObservation(
        const rm::Armor& armor, int matched_id, bool freeze_normal_geometry,
        double max_center_jump_m, ObservationDiagnostic* diagnostic);
    GeometryResidualSummary summarizeGeometryResiduals(
        const std::vector<rm::Armor>& armors) const;
    void maybeRecoverGeometry(const std::vector<rm::Armor>& armors);
    void inflateGeometryCovariance();
    void recordNis(double nis, int measurement_dim);
    void appendMotionSample();
    void recordMotionUpdate(
        const Eigen::VectorXd& prior_state, const Eigen::VectorXd& posterior_state);
    void resetNormalObservationDiagnostics();
    void updateNormalCenterVelocityFromHistory();
    bool updateNormalCandidateCenterVelocityFromHistory();
    void appendNormalYawObservation(double yaw, int source, int observation_count);
    void updateNormalYawRateFromHistory();
    void resetOutpostHeightPhase();
    void updateOutpostHeightPhase(const rm::Armor& armor, int matched_id);
    double outpostHeightOffsetForId(int id) const;
    bool outpostHeightPhaseLocked() const;
    void applyLockedOutpostHeightOffsets();
    void applyOutpostRadiusPrior();
    double outpostHeightPhaseScoreFromSamples(int phase, double* center_z = nullptr) const;
    bool outpostHeightPhaseHasEnoughIdCoverage() const;

    bool initialized_ = false;
    bool is_outpost_ = false;
    int armor_num_ = 4;
    int tracked_id_ = 0;
    int update_count_ = 0;
    bool is_converged_ = false;
    double last_nis_ = 0.0;
    int nis_window_size_ = 100;
    std::deque<int> recent_nis_failures_{0};
    std::vector<int> last_batch_match_ids_;
    std::vector<ObservationDiagnostic> last_observation_diagnostics_;
    GeometryRecoveryConfig geometry_recovery_config_{};
    bool pending_observation_jump_hint_ = false;
    int geometry_recovery_window_remaining_ = 0;
    int geometry_mismatch_streak_ = 0;
    int geometry_recovery_cooldown_ = 0;
    int consecutive_rejected_updates_ = 0;
    double last_geometry_residual_xy_over_sigma_dr_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_geometry_residual_z_over_sigma_h_ =
        std::numeric_limits<double>::quiet_NaN();
    int geometry_recovery_inflation_count_ = 0;
    double last_motion_prior_yaw_rate_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_motion_posterior_yaw_rate_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_motion_prior_vx_ = std::numeric_limits<double>::quiet_NaN();
    double last_motion_prior_vy_ = std::numeric_limits<double>::quiet_NaN();
    double last_motion_posterior_vx_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_motion_posterior_vy_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_motion_center_update_norm_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_motion_velocity_update_norm_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_motion_speed_update_abs_ =
        std::numeric_limits<double>::quiet_NaN();
    double current_frame_yaw_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    double current_frame_yaw_rate_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();
    int last_physical_rejection_count_ = 0;
    int last_physical_rejection_reason_ = 0;
    int last_normal_pair_required_ = 0;
    int last_normal_pair_found_ = 0;
    int last_normal_update_class_ = 0;
    int last_normal_accepted_count_ = 0;
    int last_normal_pair_accepted_count_ = 0;
    double last_normal_pair_score_ = std::numeric_limits<double>::quiet_NaN();
    double last_normal_pair_center_gap_m_ = std::numeric_limits<double>::quiet_NaN();
    double last_normal_pair_center_jump_m_ = std::numeric_limits<double>::quiet_NaN();
    double last_normal_pair_center_x_m_ = std::numeric_limits<double>::quiet_NaN();
    double last_normal_pair_center_y_m_ = std::numeric_limits<double>::quiet_NaN();
    int last_normal_single_center_count_ = 0;
    double last_normal_single_center_x_m_ = std::numeric_limits<double>::quiet_NaN();
    double last_normal_single_center_y_m_ = std::numeric_limits<double>::quiet_NaN();
    int last_normal_velocity_history_size_ = 0;
    int last_normal_velocity_observation_source_ = 0;
    double last_normal_velocity_sample_t_s_ = std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_sample_x_m_ = std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_sample_y_m_ = std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_sample_frame_yaw_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    int last_normal_velocity_sample_group_id_ = -1;
    int last_normal_velocity_fit_sample_count_ = 0;
    int last_normal_velocity_fit_accepted_ = 0;
    int last_normal_velocity_fit_reject_reason_ = 0;
    double last_normal_velocity_fit_time_span_s_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_net_displacement_m_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_raw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_raw_vx_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_raw_vy_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_frame_yaw_rate_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_frame_yaw_mean_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_frame_yaw_span_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    int last_normal_velocity_fit_pair_sample_count_ = 0;
    int last_normal_velocity_fit_single_sample_count_ = 0;
    int last_normal_velocity_fit_group_count_ = 0;
    int last_normal_velocity_fit_grouped_used_ = 0;
    double last_normal_velocity_fit_grouped_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_grouped_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    int last_normal_velocity_fit_rot_comp_used_ = 0;
    double last_normal_velocity_fit_rot_comp_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_frame_transform_pos_yaw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_frame_transform_pos_yaw_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_frame_transform_neg_yaw_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_frame_transform_neg_yaw_rms_m_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_applied_speed_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_applied_vx_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_velocity_fit_applied_vy_mps_ =
        std::numeric_limits<double>::quiet_NaN();
    int last_normal_yaw_rate_history_size_ = 0;
    int last_normal_yaw_rate_observation_source_ = 0;
    int last_normal_yaw_observation_count_ = 0;
    double last_normal_yaw_observation_t_s_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_yaw_observation_raw_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_yaw_observation_unwrapped_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    int last_normal_yaw_rate_fit_sample_count_ = 0;
    int last_normal_yaw_rate_fit_accepted_ = 0;
    int last_normal_yaw_rate_fit_reject_reason_ = 0;
    double last_normal_yaw_rate_fit_time_span_s_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_yaw_rate_fit_rms_rad_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_yaw_rate_fit_raw_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_normal_yaw_rate_fit_applied_rad_s_ =
        std::numeric_limits<double>::quiet_NaN();
    std::deque<double> velocity_update_history_;
    std::deque<MotionSample> motion_history_;
    std::deque<MotionSample> normal_center_velocity_history_;
    std::deque<MotionSample> normal_single_center_candidate_history_;
    std::deque<YawObservationSample> normal_yaw_observation_history_;
    double tracker_time_sec_ = 0.0;
    std::array<double, 6> outpost_height_phase_scores_{};
    int outpost_height_phase_observations_ = 0;
    int outpost_height_phase_ = -1;
    int last_outpost_height_id_ = -1;
    double last_outpost_height_z_ = std::numeric_limits<double>::quiet_NaN();
    int outpost_height_phase_relock_candidate_ = -1;
    int outpost_height_phase_relock_streak_ = 0;
    int outpost_height_phase_relock_cooldown_ = 0;
    std::deque<int> outpost_height_phase_id_samples_;
    std::array<int, 3> outpost_height_phase_id_counts_{};
    std::array<std::deque<double>, 6> outpost_height_phase_center_samples_;

    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd I_;
};

} // namespace RobotEstimator
