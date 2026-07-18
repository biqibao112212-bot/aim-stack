#include "aim_sim_bridge/fixed_rate_command_loop.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using namespace std::chrono_literals;
using aim_sim_bridge::control::CommandDisposition;
using aim_sim_bridge::control::CommandPublishStatus;
using aim_sim_bridge::control::LatestCommandState;
using aim_sim_bridge::control::SourceIdentity;
using aim_sim_bridge::control::SteadyRateScheduler;
using aim_sim_bridge::control::SteadyTimePoint;
using aim_sim_bridge::control::TickContext;
using aim_sim_bridge::control::TickMetrics;
using aim_sim_bridge::control::TickWorkResult;
using aim_sim_bridge::control::TimestampedCommand;
using aim_sim_bridge::control::TransportPublishStatus;

struct TestCommand {
  bool has_target = false;
  int value = -1;
};

int failures = 0;

void check(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

void checkNear(double actual, double expected, double tolerance,
               const std::string &message) {
  check(std::abs(actual - expected) <= tolerance,
        message + " actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

SteadyTimePoint atUs(std::int64_t microseconds) {
  return SteadyTimePoint{} + std::chrono::microseconds(microseconds);
}

void testLatestCommandFreshRepeatedStaleAndNoResurrection() {
  const TestCommand safe{false, -1};
  LatestCommandState<TestCommand> state(safe, 250ms);

  const auto unavailable = state.select(atUs(0));
  check(unavailable.disposition == CommandDisposition::Unavailable,
        "empty state must be unavailable");
  check(!unavailable.output.has_target && unavailable.output.value == -1,
        "unavailable state must emit safe no-target");

  const TimestampedCommand<TestCommand> first{
      SourceIdentity{7, 10}, atUs(1'000'000), true, TestCommand{true, 42}};
  check(state.publish(first) == CommandPublishStatus::Accepted,
        "first command must be accepted");

  const auto fresh = state.select(atUs(1'010'000));
  check(fresh.disposition == CommandDisposition::Fresh,
        "first eligible tick must be fresh");
  check(fresh.hasValidTarget() && fresh.output.value == 42,
        "fresh command must preserve payload");

  const auto repeated = state.select(atUs(1'020'000));
  check(repeated.disposition == CommandDisposition::Repeated,
        "second tick on the same source must be repeated");
  check(repeated.hasValidTarget() && repeated.output.value == 42,
        "repeated command must remain eligible while fresh");

  const auto stale = state.select(atUs(1'251'000));
  check(stale.disposition == CommandDisposition::Stale,
        "source older than 250ms must be stale");
  check(!stale.hasValidTarget() && !stale.output.has_target,
        "stale source must emit safe no-target");

  TimestampedCommand<TestCommand> duplicate = first;
  duplicate.source_time = atUs(1'300'000);
  duplicate.payload.value = 99;
  check(state.publish(duplicate) == CommandPublishStatus::DuplicateSource,
        "duplicate source publication must not refresh source time");
  const auto still_stale = state.select(atUs(1'301'000));
  check(still_stale.disposition == CommandDisposition::Stale,
        "duplicate publication must not resurrect a stale source");
  check(!still_stale.output.has_target && still_stale.output.value == -1,
        "resurrection attempt must emit safe no-target");
  check(still_stale.source_payload && still_stale.source_payload->value == 42,
        "stale selection must retain source payload for provenance only");

  const TimestampedCommand<TestCommand> no_target{
      SourceIdentity{7, 11}, atUs(1'400'000), false, TestCommand{true, 123}};
  check(state.publish(no_target) == CommandPublishStatus::Accepted,
        "new no-target source must be accepted");
  const auto selected_no_target = state.select(atUs(1'401'000));
  check(selected_no_target.disposition == CommandDisposition::NoTarget,
        "fresh no-target source must be classified as no-target");
  check(!selected_no_target.output.has_target &&
            selected_no_target.output.value == -1,
        "no-target source must use canonical safe payload");
  check(selected_no_target.source_payload &&
            selected_no_target.source_payload->value == 123,
        "no-target selection must retain source payload for provenance only");

  state.clear();
  check(state.select(atUs(1'402'000)).disposition ==
            CommandDisposition::Unavailable,
        "clear must require a post-clear command");
}

void testLatestCommandRejectsOlderAndFutureSources() {
  LatestCommandState<TestCommand> state(TestCommand{false, -1}, 250ms);
  const TimestampedCommand<TestCommand> newest{
      SourceIdentity{3, 20}, atUs(2'000'000), true, TestCommand{true, 20}};
  const TimestampedCommand<TestCommand> older_time{
      SourceIdentity{3, 21}, atUs(1'900'000), true, TestCommand{true, 21}};
  const TimestampedCommand<TestCommand> older_sequence{
      SourceIdentity{3, 19}, atUs(2'100'000), true, TestCommand{true, 19}};

  check(state.publish(newest) == CommandPublishStatus::Accepted,
        "newest command must be accepted");
  check(state.publish(older_time) == CommandPublishStatus::RejectedOlderSource,
        "older source time must be rejected");
  check(state.publish(older_sequence) ==
            CommandPublishStatus::RejectedOlderSource,
        "older sequence in the same epoch must be rejected");
  const auto selected = state.select(atUs(2'010'000));
  check(selected.disposition == CommandDisposition::Fresh &&
            selected.output.value == 20,
        "rejected publications must not replace latest command");

  LatestCommandState<TestCommand> future_state(TestCommand{false, -1}, 250ms);
  const TimestampedCommand<TestCommand> future{
      SourceIdentity{4, 1}, atUs(3'100'000), true, TestCommand{true, 1}};
  check(future_state.publish(future) == CommandPublishStatus::Accepted,
        "future timestamp is retained for explicit safe classification");
  const auto invalid = future_state.select(atUs(3'000'000));
  check(invalid.disposition == CommandDisposition::InvalidTimestamp,
        "future source timestamp must be invalid");
  check(!invalid.output.has_target,
        "future source timestamp must emit safe no-target");
}

void testDeadlineAdvanceSkipsCatchUpBursts() {
  const SteadyTimePoint deadline = atUs(0);

  const auto early =
      SteadyRateScheduler::advanceDeadline(deadline, atUs(3'000), 4ms);
  check(early.next_deadline == atUs(4'000) && early.missed_deadlines == 0 &&
            !early.overrun,
        "on-time work must preserve the next absolute deadline");

  const auto exact =
      SteadyRateScheduler::advanceDeadline(deadline, atUs(4'000), 4ms);
  check(exact.next_deadline == atUs(4'000) && exact.missed_deadlines == 0 &&
            !exact.overrun,
        "finishing exactly at the next deadline must not invent a miss");

  const auto one_miss =
      SteadyRateScheduler::advanceDeadline(deadline, atUs(4'100), 4ms);
  check(one_miss.next_deadline == atUs(8'000) &&
            one_miss.missed_deadlines == 1 && one_miss.overrun,
        "overrun must skip one expired deadline instead of catch-up replay");

  const auto two_misses =
      SteadyRateScheduler::advanceDeadline(deadline, atUs(10'000), 4ms);
  check(two_misses.next_deadline == atUs(12'000) &&
            two_misses.missed_deadlines == 2 && two_misses.overrun,
        "long overrun must advance to the first future absolute deadline");

  SteadyRateScheduler scheduler;
  check(scheduler.period() == 4ms,
        "default scheduler period must be exactly 4ms");
}

void testDeterministicTimingStatistics() {
  TickMetrics metrics(4ms, 16);

  const TickWorkResult fresh{CommandDisposition::Fresh, 10ms,
                             TransportPublishStatus::Succeeded,
                             TransportPublishStatus::Succeeded};
  const TickWorkResult repeated{CommandDisposition::Repeated, 14ms,
                                TransportPublishStatus::Succeeded,
                                TransportPublishStatus::Failed};
  const TickWorkResult stale{CommandDisposition::Stale, 260ms,
                             TransportPublishStatus::Failed,
                             TransportPublishStatus::NotAttempted};
  const TickWorkResult no_target{CommandDisposition::NoTarget, 2ms,
                                 TransportPublishStatus::Succeeded,
                                 TransportPublishStatus::Succeeded};

  const TickContext tick0{0, atUs(0), atUs(100), 100us};
  metrics.record(tick0, atUs(600), fresh, 0, false);
  const TickContext tick1{1, atUs(4'000), atUs(4'200), 200us};
  metrics.record(tick1, atUs(4'700), repeated, 0, false);
  const TickContext tick2{2, atUs(8'000), atUs(8'000), 0us};
  metrics.record(tick2, atUs(8'500), stale, 0, false);
  const TickContext tick3{3, atUs(12'000), atUs(12'800), 800us};
  metrics.record(tick3, atUs(16'200), no_target, 1, true);

  const auto snapshot = metrics.snapshot();
  check(snapshot.tick_count == 4 && snapshot.period_sample_count == 3,
        "tick and period sample counts must be exact");
  checkNear(snapshot.target_hz, 250.0, 1e-9, "target rate");
  checkNear(snapshot.period_mean_ms, 12.7 / 3.0, 1e-9, "period mean");
  checkNear(snapshot.wall_tick_hz, 3000.0 / 12.7, 1e-6, "wall tick rate");
  checkNear(snapshot.period_p50_ms, 4.1, 1e-9, "period p50");
  checkNear(snapshot.period_p99_ms, 4.8, 1e-9, "period p99");
  checkNear(snapshot.period_max_ms, 4.8, 1e-9, "maximum gap");
  checkNear(snapshot.period_abs_error_p99_ms, 0.8, 1e-9, "period error p99");
  checkNear(snapshot.lateness_p99_ms, 0.8, 1e-9, "lateness p99");
  checkNear(snapshot.execution_p99_ms, 3.4, 1e-9, "execution p99");
  check(snapshot.source_age_available && snapshot.source_age_sample_count == 4,
        "source age availability and sample count must be explicit");
  checkNear(snapshot.source_age_latest_ms, 2.0, 1e-9, "latest source age");
  checkNear(snapshot.source_age_p99_ms, 260.0, 1e-9, "source age p99");

  check(snapshot.fresh_count == 1 && snapshot.repeated_command_count == 1 &&
            snapshot.stale_count == 1 && snapshot.no_target_count == 1,
        "command disposition buckets must be independent");
  check(snapshot.missed_deadline_count == 1 && snapshot.overrun_count == 1,
        "missed deadline and overrun counts must be exact");
  checkNear(snapshot.missed_deadline_ratio, 0.2, 1e-9, "missed ratio");
  checkNear(snapshot.overrun_ratio, 0.25, 1e-9, "overrun ratio");
  check(snapshot.udp_publish_attempt_count == 4 &&
            snapshot.udp_publish_success_count == 3 &&
            snapshot.udp_publish_failure_count == 1,
        "UDP attempt/success/failure counts must be separate");
  check(snapshot.talos_publish_attempt_count == 3 &&
            snapshot.talos_publish_success_count == 2 &&
            snapshot.talos_publish_failure_count == 1,
        "Talos attempt/success/failure counts must be separate");

  metrics.reset();
  const auto reset = metrics.snapshot();
  check(reset.tick_count == 0 && reset.period_sample_count == 0,
        "metrics reset must clear timing state and counters");
}

} // namespace

int main() {
  testLatestCommandFreshRepeatedStaleAndNoResurrection();
  testLatestCommandRejectsOlderAndFutureSources();
  testDeadlineAdvanceSkipsCatchUpBursts();
  testDeterministicTimingStatistics();

  if (failures != 0) {
    std::cerr << "fixed_rate_command_loop_test failures=" << failures << '\n';
    return 1;
  }
  std::cout << "fixed_rate_command_loop_test passed\n";
  return 0;
}
