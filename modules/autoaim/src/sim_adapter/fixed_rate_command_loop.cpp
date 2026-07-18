#include "aim_sim_bridge/fixed_rate_command_loop.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace aim_sim_bridge::control {
namespace {

double durationMs(SteadyDuration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

class BoundedSamples {
public:
  explicit BoundedSamples(std::size_t capacity)
      : values_(std::max<std::size_t>(capacity, 1), 0.0) {}

  void add(double value) {
    values_[next_] = value;
    next_ = (next_ + 1) % values_.size();
    size_ = std::min(size_ + 1, values_.size());
  }

  [[nodiscard]] std::vector<double> values() const {
    return std::vector<double>(values_.begin(), values_.begin() + size_);
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  void clear() noexcept {
    size_ = 0;
    next_ = 0;
  }

private:
  std::vector<double> values_;
  std::size_t size_ = 0;
  std::size_t next_ = 0;
};

struct SampleSummary {
  double mean = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double maximum = 0.0;
};

double nearestRank(const std::vector<double> &sorted, double quantile) {
  if (sorted.empty()) {
    return 0.0;
  }
  const double bounded = std::clamp(quantile, 0.0, 1.0);
  const auto rank = static_cast<std::size_t>(
      std::ceil(bounded * static_cast<double>(sorted.size())));
  const std::size_t index =
      std::clamp<std::size_t>(rank == 0 ? 0 : rank - 1, 0, sorted.size() - 1);
  return sorted[index];
}

SampleSummary summarize(std::vector<double> values) {
  if (values.empty()) {
    return {};
  }

  SampleSummary result;
  result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<double>(values.size());
  std::sort(values.begin(), values.end());
  result.p50 = nearestRank(values, 0.50);
  result.p95 = nearestRank(values, 0.95);
  result.p99 = nearestRank(values, 0.99);
  result.maximum = values.back();
  return result;
}

void recordTransport(TransportPublishStatus status, std::uint64_t &attempts,
                     std::uint64_t &successes, std::uint64_t &failures) {
  switch (status) {
  case TransportPublishStatus::NotAttempted:
    return;
  case TransportPublishStatus::Succeeded:
    ++attempts;
    ++successes;
    return;
  case TransportPublishStatus::Failed:
    ++attempts;
    ++failures;
    return;
  }
}

} // namespace

const char *toString(CommandDisposition disposition) noexcept {
  switch (disposition) {
  case CommandDisposition::Unavailable:
    return "unavailable";
  case CommandDisposition::NoTarget:
    return "no_target";
  case CommandDisposition::Fresh:
    return "fresh";
  case CommandDisposition::Repeated:
    return "repeated";
  case CommandDisposition::Stale:
    return "stale";
  case CommandDisposition::InvalidTimestamp:
    return "invalid_timestamp";
  }
  return "unknown";
}

class TickMetrics::Impl {
public:
  Impl(SteadyDuration target_period, std::size_t sample_capacity)
      : target_period(target_period), period_ms(sample_capacity),
        period_abs_error_ms(sample_capacity), lateness_ms(sample_capacity),
        execution_ms(sample_capacity), source_age_ms(sample_capacity) {
    if (target_period <= SteadyDuration::zero()) {
      throw std::invalid_argument("target period must be positive");
    }
  }

  void clearUnlocked() {
    last_start.reset();
    period_ms.clear();
    period_abs_error_ms.clear();
    lateness_ms.clear();
    execution_ms.clear();
    source_age_ms.clear();
    source_age_available = false;
    source_age_latest_ms = 0.0;

    tick_count = 0;
    missed_deadline_count = 0;
    overrun_count = 0;
    unavailable_count = 0;
    no_target_count = 0;
    fresh_count = 0;
    repeated_command_count = 0;
    stale_count = 0;
    invalid_timestamp_count = 0;
    udp_publish_attempt_count = 0;
    udp_publish_success_count = 0;
    udp_publish_failure_count = 0;
    talos_publish_attempt_count = 0;
    talos_publish_success_count = 0;
    talos_publish_failure_count = 0;
  }

  mutable std::mutex mutex;
  SteadyDuration target_period;
  std::optional<SteadyTimePoint> last_start;
  BoundedSamples period_ms;
  BoundedSamples period_abs_error_ms;
  BoundedSamples lateness_ms;
  BoundedSamples execution_ms;
  BoundedSamples source_age_ms;
  bool source_age_available = false;
  double source_age_latest_ms = 0.0;

  std::uint64_t tick_count = 0;
  std::uint64_t missed_deadline_count = 0;
  std::uint64_t overrun_count = 0;
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

TickMetrics::TickMetrics(SteadyDuration target_period,
                         std::size_t sample_capacity)
    : impl_(new Impl(target_period, sample_capacity)) {}

TickMetrics::~TickMetrics() { delete impl_; }

void TickMetrics::record(const TickContext &context,
                         SteadyTimePoint finished_at,
                         const TickWorkResult &work,
                         std::uint64_t missed_deadlines, bool overrun) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  ++impl_->tick_count;

  const double target_period_ms = durationMs(impl_->target_period);
  if (impl_->last_start) {
    const double period_ms =
        std::max(0.0, durationMs(context.started_at - *impl_->last_start));
    impl_->period_ms.add(period_ms);
    impl_->period_abs_error_ms.add(std::abs(period_ms - target_period_ms));
  }
  impl_->last_start = context.started_at;

  impl_->lateness_ms.add(std::max(0.0, durationMs(context.lateness)));
  impl_->execution_ms.add(
      std::max(0.0, durationMs(finished_at - context.started_at)));
  if (work.source_age && *work.source_age >= SteadyDuration::zero()) {
    impl_->source_age_available = true;
    impl_->source_age_latest_ms = durationMs(*work.source_age);
    impl_->source_age_ms.add(impl_->source_age_latest_ms);
  } else {
    impl_->source_age_available = false;
    impl_->source_age_latest_ms = 0.0;
  }

  impl_->missed_deadline_count += missed_deadlines;
  if (overrun) {
    ++impl_->overrun_count;
  }

  switch (work.disposition) {
  case CommandDisposition::Unavailable:
    ++impl_->unavailable_count;
    break;
  case CommandDisposition::NoTarget:
    ++impl_->no_target_count;
    break;
  case CommandDisposition::Fresh:
    ++impl_->fresh_count;
    break;
  case CommandDisposition::Repeated:
    ++impl_->repeated_command_count;
    break;
  case CommandDisposition::Stale:
    ++impl_->stale_count;
    break;
  case CommandDisposition::InvalidTimestamp:
    ++impl_->invalid_timestamp_count;
    break;
  }

  recordTransport(work.udp_publish, impl_->udp_publish_attempt_count,
                  impl_->udp_publish_success_count,
                  impl_->udp_publish_failure_count);
  recordTransport(work.talos_publish, impl_->talos_publish_attempt_count,
                  impl_->talos_publish_success_count,
                  impl_->talos_publish_failure_count);
}

TickMetricsSnapshot TickMetrics::snapshot() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const SampleSummary periods = summarize(impl_->period_ms.values());
  const SampleSummary period_errors =
      summarize(impl_->period_abs_error_ms.values());
  const SampleSummary lateness = summarize(impl_->lateness_ms.values());
  const SampleSummary execution = summarize(impl_->execution_ms.values());
  const SampleSummary source_age = summarize(impl_->source_age_ms.values());

  TickMetricsSnapshot result;
  result.tick_count = impl_->tick_count;
  result.period_sample_count = impl_->period_ms.size();
  result.target_hz =
      1.0 / std::chrono::duration<double>(impl_->target_period).count();
  result.wall_tick_hz = periods.mean > 0.0 ? 1000.0 / periods.mean : 0.0;

  result.period_mean_ms = periods.mean;
  result.period_p50_ms = periods.p50;
  result.period_p95_ms = periods.p95;
  result.period_p99_ms = periods.p99;
  result.period_max_ms = periods.maximum;
  result.period_abs_error_p99_ms = period_errors.p99;
  result.lateness_p50_ms = lateness.p50;
  result.lateness_p95_ms = lateness.p95;
  result.lateness_p99_ms = lateness.p99;
  result.lateness_max_ms = lateness.maximum;
  result.execution_p99_ms = execution.p99;
  result.execution_max_ms = execution.maximum;
  result.source_age_available = impl_->source_age_available;
  result.source_age_sample_count = impl_->source_age_ms.size();
  result.source_age_latest_ms = impl_->source_age_latest_ms;
  result.source_age_p99_ms = source_age.p99;
  result.source_age_max_ms = source_age.maximum;

  result.missed_deadline_count = impl_->missed_deadline_count;
  result.overrun_count = impl_->overrun_count;
  const double scheduled =
      static_cast<double>(impl_->tick_count + impl_->missed_deadline_count);
  result.missed_deadline_ratio =
      scheduled > 0.0
          ? static_cast<double>(impl_->missed_deadline_count) / scheduled
          : 0.0;
  result.overrun_ratio = impl_->tick_count > 0
                             ? static_cast<double>(impl_->overrun_count) /
                                   static_cast<double>(impl_->tick_count)
                             : 0.0;

  result.unavailable_count = impl_->unavailable_count;
  result.no_target_count = impl_->no_target_count;
  result.fresh_count = impl_->fresh_count;
  result.repeated_command_count = impl_->repeated_command_count;
  result.stale_count = impl_->stale_count;
  result.invalid_timestamp_count = impl_->invalid_timestamp_count;
  result.udp_publish_attempt_count = impl_->udp_publish_attempt_count;
  result.udp_publish_success_count = impl_->udp_publish_success_count;
  result.udp_publish_failure_count = impl_->udp_publish_failure_count;
  result.talos_publish_attempt_count = impl_->talos_publish_attempt_count;
  result.talos_publish_success_count = impl_->talos_publish_success_count;
  result.talos_publish_failure_count = impl_->talos_publish_failure_count;
  return result;
}

void TickMetrics::reset() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->clearUnlocked();
}

SteadyRateScheduler::SteadyRateScheduler(SteadyRateSchedulerConfig config)
    : metrics_(kDefaultControlPeriod, config.metric_sample_capacity) {}

void SteadyRateScheduler::run(const StopPredicate &stop_requested,
                              const TickCallback &on_tick) {
  if (!stop_requested || !on_tick) {
    throw std::invalid_argument("scheduler requires stop and tick callbacks");
  }

  std::uint64_t tick_index = 0;
  SteadyTimePoint deadline = SteadyClock::now() + kDefaultControlPeriod;
  while (!stop_requested()) {
    std::this_thread::sleep_until(deadline);
    if (stop_requested()) {
      break;
    }

    const SteadyTimePoint started_at = SteadyClock::now();
    const SteadyDuration lateness =
        started_at > deadline ? started_at - deadline : SteadyDuration::zero();
    const TickContext context{tick_index, deadline, started_at, lateness};
    const TickWorkResult work = on_tick(context);
    const SteadyTimePoint finished_at = SteadyClock::now();
    const DeadlineAdvance advance =
        advanceDeadline(deadline, finished_at, kDefaultControlPeriod);
    metrics_.record(context, finished_at, work, advance.missed_deadlines,
                    advance.overrun);
    deadline = advance.next_deadline;
    ++tick_index;
  }
}

TickMetrics &SteadyRateScheduler::metrics() noexcept { return metrics_; }

const TickMetrics &SteadyRateScheduler::metrics() const noexcept {
  return metrics_;
}

SteadyDuration SteadyRateScheduler::period() const noexcept {
  return kDefaultControlPeriod;
}

DeadlineAdvance
SteadyRateScheduler::advanceDeadline(SteadyTimePoint current_deadline,
                                     SteadyTimePoint finished_at,
                                     SteadyDuration period) {
  if (period <= SteadyDuration::zero()) {
    throw std::invalid_argument("scheduler period must be positive");
  }

  DeadlineAdvance result;
  result.next_deadline = current_deadline + period;
  if (finished_at <= result.next_deadline) {
    return result;
  }

  result.overrun = true;
  const SteadyDuration overdue = finished_at - result.next_deadline;
  result.missed_deadlines = static_cast<std::uint64_t>(overdue / period) + 1;
  result.next_deadline +=
      period * static_cast<SteadyDuration::rep>(result.missed_deadlines);
  return result;
}

} // namespace aim_sim_bridge::control
