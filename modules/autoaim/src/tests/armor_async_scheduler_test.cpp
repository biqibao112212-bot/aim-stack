
#include "aim_sim_bridge/pipeline.hpp"
#include "thread_safe_queue.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>
#include <vector>

int main()
{
    static_assert(aim_sim_bridge::detail::kCompletionQueueCapacity == 8);

    using RateGate = aim_sim_bridge::detail::DetectorSubmissionRateGate;
    using Clock = RateGate::Clock;
    const auto epoch = Clock::time_point{};

    RateGate unlimited;
    for (int index = 0; index < 12'300; ++index) {
        assert(unlimited.admit(
            epoch + std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(index / 205.0))));
    }

    RateGate capped(180);
    std::uint64_t admitted = 0;
    for (int index = 0; index < 12'300; ++index) {
        const auto now = epoch + std::chrono::duration_cast<Clock::duration>(
                                     std::chrono::duration<double>(index / 205.0));
        if (capped.admit(now)) ++admitted;
    }
    assert(admitted >= 10'740 && admitted <= 10'860);

    // A 207 Hz source with periodic scheduler stalls must retain the 165 Hz
    // gate phase instead of permanently resetting it after a late sample.
    RateGate jittered(165);
    admitted = 0;
    auto previous_now = epoch;
    constexpr std::chrono::milliseconds stalls[] = {
        std::chrono::milliseconds(8), std::chrono::milliseconds(12),
        std::chrono::milliseconds(16), std::chrono::milliseconds(20)};
    for (int index = 0; index < 207 * 60; ++index) {
        const auto nominal =
            epoch + std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(index / 207.0));
        auto now = nominal;
        if (index > 0 && index % (207 * 2) == 0) {
            now += stalls[(index / (207 * 2)) % 4];
        }
        now = std::max(now, previous_now + std::chrono::microseconds(1));
        previous_now = now;
        if (jittered.admit(now)) ++admitted;
    }
    const double jittered_hz = static_cast<double>(admitted) / 60.0;
    assert(jittered_hz >= 164.0 && jittered_hz <= 166.0);

    RateGate gap_phase(180);
    assert(gap_phase.admit(epoch));
    assert(!gap_phase.admit(epoch + std::chrono::milliseconds(1)));
    const auto after_gap = epoch + std::chrono::seconds(2);
    assert(gap_phase.admit(after_gap));
    assert(!gap_phase.admit(after_gap));
    assert(!gap_phase.admit(after_gap + std::chrono::milliseconds(1)));

    // A rate-limited input does not consume an already completed result.
    aim_sim_bridge::detail::BoundedCompletionQueue<int, 1> gated_completion;
    assert(!gated_completion.publish(7));
    assert(!gap_phase.admit(after_gap + std::chrono::milliseconds(2)));
    int gated_value = 0;
    assert(gated_completion.tryTake(gated_value) && gated_value == 7);

    const auto verify_raw_suffix_contract = [](int prefix_count) {
        tools::ThreadSafeQueue<int, true> raw(3);
        std::vector<int> profiles;
        for (int value = 100; value < 100 + prefix_count; ++value) {
            profiles.push_back(value);
        }
        for (int value : {1, 2, 3}) {
            profiles.push_back(value);
            assert(!raw.push(value));
        }

        for (int value : {4, 5}) {
            profiles.push_back(value);
            assert(raw.push(value));
            assert(aim_sim_bridge::detail::eraseOldestOverwrittenRawSuffixItem(
                profiles, 3));
        }

        std::vector<int> expected;
        for (int value = 100; value < 100 + prefix_count; ++value) {
            expected.push_back(value);
        }
        expected.insert(expected.end(), {3, 4, 5});
        assert(profiles == expected);

        for (int expected_raw : {3, 4, 5}) {
            int actual_raw = 0;
            assert(raw.wait_pop(actual_raw));
            assert(actual_raw == expected_raw);
        }

        int matched = -1;
        const int detected_identity = prefix_count > 0 ? 99 + prefix_count : 3;
        assert(aim_sim_bridge::detail::takeFirstMatchingAndErasePrefix(
            profiles,
            [detected_identity](int value) { return value == detected_identity; },
            matched));
        assert(matched == detected_identity);
        const std::vector<int> expected_after_detection =
            prefix_count > 0 ? std::vector<int>{3, 4, 5} : std::vector<int>{4, 5};
        assert(profiles == expected_after_detection);

        const auto before_missing = profiles;
        assert(!aim_sim_bridge::detail::takeFirstMatchingAndErasePrefix(
            profiles, [](int value) { return value == 999; }, matched));
        assert(profiles == before_missing);
        raw.stop();
    };
    verify_raw_suffix_contract(0);
    verify_raw_suffix_contract(1);
    verify_raw_suffix_contract(4);

    aim_sim_bridge::detail::AimPipelineStageTelemetry stages;
    const auto empty = stages.snapshot();
    assert(empty.submitted == 0);
    assert(empty.submission_rate_limited == 0);
    assert(empty.final_completed == 0);
    assert(empty.delivered_completed == 0);

    tools::ThreadSafeQueue<int, true> ingress(1);
    assert(!ingress.push(1));
    stages.recordSubmission(11, false);
    assert(ingress.push(2));
    stages.recordSubmission(13, true);
    stages.recordSubmissionRateLimited();

    aim_sim_bridge::detail::BoundedCompletionQueue<
        int, aim_sim_bridge::detail::kCompletionQueueCapacity> completions;
    std::vector<int> consumed;
    std::atomic<bool> stopped{false};
    std::thread worker([&] {
        int value = 0;
        while (ingress.wait_pop(value)) {
            if (!consumed.empty()) assert(value > consumed.back());
            consumed.push_back(value);
            stages.recordDetectorCompletion(17);
            stages.recordSolveCompletion(19);
            stages.recordTrackerAimCompletion(23);
            stages.recordFinalCompletion(29, 101);
            assert(!completions.publish(value));
        }
        stopped.store(true);
    });

    for (int attempt = 0; attempt < 100 && consumed.empty(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert((consumed == std::vector<int>{2}));

    int completion = 0;
    assert(completions.tryTake(completion));
    stages.recordDeliveredCompletion();
    assert(completion == 2);
    assert(!completions.tryTake(completion));
    assert(!completions.tryTake(completion));

    // Two four-result completion waves fit without loss, covering the burst
    // observed when the three detector slots and downstream worker catch up.
    for (int value = 10; value < 18; ++value) {
        assert(!completions.publish(value));
    }
    assert(completions.size() == 8);
    // A ninth completion retains the freshest bounded suffix.
    assert(completions.publish(18));
    assert(completions.size() == 8);
    for (int expected = 11; expected < 19; ++expected) {
        assert(completions.tryTake(completion));
        assert(completion == expected);
    }
    assert(!completions.tryTake(completion));

    for (int value = 20; value < 24; ++value) {
        assert(!completions.publish(value));
    }
    assert(completions.tryTake(completion) && completion == 20);
    assert(completions.tryTake(completion) && completion == 21);
    for (int value = 24; value < 30; ++value) {
        assert(!completions.publish(value));
    }
    assert(completions.size() == 8);
    for (int expected = 22; expected < 30; ++expected) {
        assert(completions.tryTake(completion));
        assert(completion == expected);
    }

    // A held/reused command is represented by the failed queue takes above.
    // It must not advance either the final or delivered completion boundary.
    const auto counters = stages.snapshot();
    assert(counters.submitted == 2);
    assert(counters.submission_rate_limited == 1);
    assert(counters.overwritten == 1);
    assert(counters.detector_completed == 1);
    assert(counters.solve_completed == 1);
    assert(counters.tracker_aim_completed == 1);
    assert(counters.final_completed == 1);
    assert(counters.delivered_completed == 1);
    assert(counters.submission_cpu_ns == 24);
    assert(counters.detector_latency_ns == 17);
    assert(counters.solve_cpu_ns == 19);
    assert(counters.tracker_aim_cpu_ns == 23);
    assert(counters.finalize_cpu_ns == 29);
    assert(counters.pipeline_latency_ns == 101);
    assert(counters.delivered_completed <= counters.final_completed);
    assert(counters.final_completed <= counters.tracker_aim_completed);
    assert(counters.tracker_aim_completed <= counters.solve_completed);
    assert(counters.solve_completed <= counters.detector_completed);
    assert(counters.detector_completed <=
           counters.submitted - counters.overwritten);

    ingress.stop();
    worker.join();
    assert(stopped.load());
    return 0;
}
