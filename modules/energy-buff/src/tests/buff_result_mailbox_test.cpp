#include "aim_sim_bridge/aim_types.hpp"
#include "latest_result_mailbox.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace
{

struct TestResult
{
    bool completed = false;
    std::uint64_t producer_epoch = 0;
    std::uint64_t image_seq = 0;
    std::uint64_t capture_timestamp_ns = 0;
    std::uint64_t completion_timestamp_ns = 0;
    std::string payload;
};

int failures = 0;

void check(bool condition, const char* message)
{
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void testEmptyPollIsNonDestructive()
{
    auto_buff::LatestResultMailbox<TestResult> mailbox;
    TestResult destination{true, 91, 92, 93, 94, "unchanged"};

    check(!mailbox.tryPopLatest(&destination), "empty poll must return immediately without result");
    check(
        destination.completed && destination.producer_epoch == 91 &&
            destination.image_seq == 92 && destination.capture_timestamp_ns == 93 &&
            destination.completion_timestamp_ns == 94 && destination.payload == "unchanged",
        "empty poll must not mutate the caller's destination");

    const TestResult later{true, 7, 10, 1'000, 1'250, "later"};
    mailbox.publish(later);
    TestResult received;
    check(mailbox.tryPopLatest(&received), "result published after an empty poll must remain available");
    check(
        received.completed && received.producer_epoch == later.producer_epoch &&
            received.image_seq == later.image_seq &&
            received.capture_timestamp_ns == later.capture_timestamp_ns &&
            received.completion_timestamp_ns == later.completion_timestamp_ns &&
            received.payload == later.payload,
        "real result identity and completion timestamp must be unchanged");
}

void testNullAndRepeatedPollsDoNotLoseOrReplayResults()
{
    auto_buff::LatestResultMailbox<TestResult> mailbox;
    mailbox.publish({true, 8, 20, 2'000, 2'300, "first"});
    check(!mailbox.tryPopLatest(nullptr), "null destination must be rejected");

    TestResult received;
    check(mailbox.tryPopLatest(&received), "null poll must not clear the available result");
    check(received.image_seq == 20 && received.payload == "first", "available result must survive null poll");

    TestResult sentinel{true, 99, 99, 99, 99, "sentinel"};
    check(!mailbox.tryPopLatest(&sentinel), "consumed result must not be replayed");
    check(sentinel.image_seq == 99 && sentinel.payload == "sentinel", "empty repeat poll must be non-destructive");
}

void testLatestOnlyReplacementAndExplicitClear()
{
    auto_buff::LatestResultMailbox<TestResult> mailbox;
    mailbox.publish({true, 9, 30, 3'000, 3'100, "older"});
    mailbox.publish({true, 9, 31, 3'200, 3'350, "newest"});

    TestResult received;
    check(mailbox.tryPopLatest(&received), "latest result must be available");
    check(received.image_seq == 31 && received.payload == "newest", "mailbox must preserve latest-only semantics");

    mailbox.publish({true, 9, 32, 3'400, 3'500, "clear-me"});
    mailbox.clear();
    check(!mailbox.tryPopLatest(&received), "explicit reset must clear the mailbox");
}

void testNoResultAimCommandHasNoCompletionIdentity()
{
    aim_sim_bridge::AimCommand no_result;
    no_result.backend = "vivsionn_trt";
    check(!no_result.completed_vision_result, "no-result command must not claim completion");
    check(
        no_result.source_producer_epoch == 0 && no_result.source_image_seq == 0 &&
            no_result.source_capture_timestamp_ns == 0 &&
            no_result.vision_completion_timestamp_ns == 0,
        "no-result command must not refresh a completion source identity");
}

}  // namespace

int main()
{
    testEmptyPollIsNonDestructive();
    testNullAndRepeatedPollsDoNotLoseOrReplayResults();
    testLatestOnlyReplacementAndExplicitClear();
    testNoResultAimCommandHasNoCompletionIdentity();

    if (failures != 0) {
        std::cerr << "buff_result_mailbox_test failures=" << failures << '\n';
        return 1;
    }
    std::cout << "buff_result_mailbox_test passed\n";
    return 0;
}
