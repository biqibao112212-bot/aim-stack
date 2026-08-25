#pragma once

#include <chrono>
#include <list>
#include <string>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/target.hpp"

namespace autoaim_research {

struct TrackerConfig {
  int min_detect_count = 3;
  int max_temp_lost_count = 15;
  double max_observation_gap_s = 0.1;
  double initial_radius_m = 0.2;
  int armor_count = 4;
};

struct TrackerSnapshot {
  std::string state{"lost"};
  bool has_estimate = false;
  Eigen::VectorXd state_vector;
};

// Single-target research wrapper around the upstream 11-state Target EKF.
// It deliberately omits aiming, fire control and gimbal command generation.
class TongjiResearchTracker {
 public:
  explicit TongjiResearchTracker(TrackerConfig config = {});

  TrackerSnapshot update(std::list<auto_aim::Armor> armors,
                         std::uint64_t timestamp_ns);
  const std::string& state() const noexcept { return state_; }

 private:
  bool initialize(const std::list<auto_aim::Armor>& armors,
                  std::chrono::steady_clock::time_point timestamp);
  bool updateTarget(const std::list<auto_aim::Armor>& armors,
                    std::chrono::steady_clock::time_point timestamp);
  void updateState(bool found);

  TrackerConfig config_;
  auto_aim::Target target_;
  bool target_initialized_ = false;
  std::string state_{"lost"};
  int detect_count_ = 0;
  int temp_lost_count_ = 0;
  std::uint64_t last_timestamp_ns_ = 0;
};

}  // namespace autoaim_research
