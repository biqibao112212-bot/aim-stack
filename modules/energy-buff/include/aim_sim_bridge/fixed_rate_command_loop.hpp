#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace aim_sim_bridge::control {

using SteadyClock = std::chrono::steady_clock;
using SteadyTimePoint = SteadyClock::time_point;
using SteadyDuration = SteadyClock::duration;

inline constexpr auto kDefaultControlPeriod = std::chrono::milliseconds(4);
inline constexpr auto kDefaultFreshnessTimeout = std::chrono::milliseconds(250);
inline constexpr std::size_t kDefaultMetricSampleCapacity = 20000;

enum class CommandDisposition : std::uint8_t {
  Unavailable,
  NoTarget,
  Fresh,
  Repeated,
  Stale,
  InvalidTimestamp,
};

enum class CommandPublishStatus : std::uint8_t {
  Accepted,
  DuplicateSource,
  RejectedOlderSource,
};

enum class TransportPublishStatus : std::uint8_t {
  NotAttempted,
  Succeeded,
  Failed,
};

const char *toString(CommandDisposition disposition) noexcept;

struct SourceIdentity {
  std::uint64_t epoch = 0;
  std::uint64_t sequence = 0;

  friend bool operator==(const SourceIdentity &lhs,
                         const SourceIdentity &rhs) noexcept {
    return lhs.epoch == rhs.epoch && lhs.sequence == rhs.sequence;
  }

  friend bool operator!=(const SourceIdentity &lhs,
                         const SourceIdentity &rhs) noexcept {
    return !(lhs == rhs);
  }
};

template <typename Payload> struct TimestampedCommand {
  SourceIdentity source;
  SteadyTimePoint source_time;
  bool has_target = false;
  Payload payload{};
};

template <typename Payload> struct CommandSelection {
  CommandDisposition disposition = CommandDisposition::Unavailable;
  SourceIdentity source;
  std::optional<SteadyDuration> source_age;
  // Preserves the accepted source payload for provenance/telemetry even when
  // output is replaced by the canonical safe no-target payload.
  std::optional<Payload> source_payload;
  Payload output{};

  [[nodiscard]] bool hasValidTarget() const noexcept {
    return disposition == CommandDisposition::Fresh ||
           disposition == CommandDisposition::Repeated;
  }
};

// Thread-safe latest-only command state. Duplicate publication never refreshes
// source_time, so a repeated publisher cannot make an old observation fresh.
template <typename Payload> class LatestCommandState {
  static_assert(std::is_copy_constructible_v<Payload> &&
                    std::is_copy_assignable_v<Payload>,
                "LatestCommandState payloads must be copyable");

public:
  explicit LatestCommandState(
      Payload safe_no_target,
      SteadyDuration freshness_timeout = kDefaultFreshnessTimeout)
      : safe_no_target_(std::move(safe_no_target)),
        freshness_timeout_(freshness_timeout) {}

  [[nodiscard]] CommandPublishStatus
  publish(const TimestampedCommand<Payload> &command) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_) {
      if (command.source == latest_->source) {
        return CommandPublishStatus::DuplicateSource;
      }
      if (command.source_time < latest_->source_time) {
        return CommandPublishStatus::RejectedOlderSource;
      }
      if (command.source.epoch == latest_->source.epoch &&
          command.source.sequence < latest_->source.sequence) {
        return CommandPublishStatus::RejectedOlderSource;
      }
    }

    latest_ = command;
    return CommandPublishStatus::Accepted;
  }

  [[nodiscard]] CommandSelection<Payload> select(SteadyTimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    CommandSelection<Payload> selection;
    selection.output = safe_no_target_;

    if (!latest_) {
      return selection;
    }

    selection.source = latest_->source;
    selection.source_payload = latest_->payload;
    if (now < latest_->source_time) {
      selection.disposition = CommandDisposition::InvalidTimestamp;
      return selection;
    }

    selection.source_age = now - latest_->source_time;
    if (*selection.source_age > freshness_timeout_) {
      selection.disposition = CommandDisposition::Stale;
      return selection;
    }
    if (!latest_->has_target) {
      selection.disposition = CommandDisposition::NoTarget;
      return selection;
    }

    selection.output = latest_->payload;
    if (last_selected_source_ && *last_selected_source_ == latest_->source) {
      selection.disposition = CommandDisposition::Repeated;
    } else {
      selection.disposition = CommandDisposition::Fresh;
      last_selected_source_ = latest_->source;
    }
    return selection;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_.reset();
    last_selected_source_.reset();
  }

  [[nodiscard]] SteadyDuration freshnessTimeout() const noexcept {
    return freshness_timeout_;
  }

private:
  mutable std::mutex mutex_;
  Payload safe_no_target_;
  SteadyDuration freshness_timeout_;
  std::optional<TimestampedCommand<Payload>> latest_;
  std::optional<SourceIdentity> last_selected_source_;
};

struct TickContext {
  std::uint64_t tick_index = 0;
  SteadyTimePoint deadline;
  SteadyTimePoint started_at;
  SteadyDuration lateness = SteadyDuration::zero();
};

struct TickWorkResult {
  CommandDisposition disposition = CommandDisposition::Unavailable;
  std::optional<SteadyDuration> source_age;
  TransportPublishStatus udp_publish = TransportPublishStatus::NotAttempted;
  TransportPublishStatus talos_publish = TransportPublishStatus::NotAttempted;
};

struct DeadlineAdvance {
  SteadyTimePoint next_deadline;
  std::uint64_t missed_deadlines = 0;
  bool overrun = false;
};

struct TickMetricsSnapshot {
  std::uint64_t tick_count = 0;
  std::uint64_t period_sample_count = 0;
  double target_hz = 250.0;
  double wall_tick_hz = 0.0;

  double period_mean_ms = 0.0;
  double period_p50_ms = 0.0;
  double period_p95_ms = 0.0;
  double period_p99_ms = 0.0;
  double period_max_ms = 0.0;
  double period_abs_error_p99_ms = 0.0;

  double lateness_p50_ms = 0.0;
  double lateness_p95_ms = 0.0;
  double lateness_p99_ms = 0.0;
  double lateness_max_ms = 0.0;

  double execution_p99_ms = 0.0;
  double execution_max_ms = 0.0;
  bool source_age_available = false;
  std::uint64_t source_age_sample_count = 0;
  double source_age_latest_ms = 0.0;
  double source_age_p99_ms = 0.0;
  double source_age_max_ms = 0.0;

  std::uint64_t missed_deadline_count = 0;
  std::uint64_t overrun_count = 0;
  double missed_deadline_ratio = 0.0;
  double overrun_ratio = 0.0;

  std::uint64_t unavailable_count = 0;
  std::uint64_t no_target_count = 0;
  std::uint64_t fresh_count = 0;
  std::uint64_t repeated_command_count = 0;
  std::uint64_t stale_count = 0;
  std::uint64_t invalid_timestamp_count = 0;

  std::uint64_t udp_publish_attempt_count = 0;
  std::uint64_t udp_publish_success_count = 0;
  std::uint64_t udp_publish_failure_count = 0;
  std::uint64_t talos_publish_attempt_count = 0;
  std::uint64_t talos_publish_success_count = 0;
  std::uint64_t talos_publish_failure_count = 0;
};

class TickMetrics {
public:
  explicit TickMetrics(
      SteadyDuration target_period = kDefaultControlPeriod,
      std::size_t sample_capacity = kDefaultMetricSampleCapacity);
  ~TickMetrics();

  TickMetrics(const TickMetrics &) = delete;
  TickMetrics &operator=(const TickMetrics &) = delete;
  TickMetrics(TickMetrics &&) = delete;
  TickMetrics &operator=(TickMetrics &&) = delete;

  void record(const TickContext &context, SteadyTimePoint finished_at,
              const TickWorkResult &work, std::uint64_t missed_deadlines,
              bool overrun);

  [[nodiscard]] TickMetricsSnapshot snapshot() const;
  void reset();

private:
  class Impl;
  Impl *impl_;
};

struct SteadyRateSchedulerConfig {
  std::size_t metric_sample_capacity = kDefaultMetricSampleCapacity;
};

class SteadyRateScheduler {
public:
  using StopPredicate = std::function<bool()>;
  using TickCallback = std::function<TickWorkResult(const TickContext &)>;

  explicit SteadyRateScheduler(SteadyRateSchedulerConfig config = {});

  // Serialized loop: the caller owns the thread. Deadlines are absolute and
  // overdue periods are skipped instead of replayed as catch-up bursts.
  void run(const StopPredicate &stop_requested, const TickCallback &on_tick);

  [[nodiscard]] TickMetrics &metrics() noexcept;
  [[nodiscard]] const TickMetrics &metrics() const noexcept;
  [[nodiscard]] SteadyDuration period() const noexcept;

  [[nodiscard]] static DeadlineAdvance
  advanceDeadline(SteadyTimePoint current_deadline, SteadyTimePoint finished_at,
                  SteadyDuration period);

private:
  TickMetrics metrics_;
};

} // namespace aim_sim_bridge::control
