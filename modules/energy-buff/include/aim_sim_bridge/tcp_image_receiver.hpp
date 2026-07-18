#pragma once

#include "aim_sim_bridge/tcp_image_protocol.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace aim_sim_bridge::tcp_image
{

struct SourceIdentity
{
    std::uint64_t producer_epoch = 0;
    std::uint64_t source_sequence = 0;
};

enum class IdentityOrder
{
    Regression,
    Duplicate,
    Newer,
};

[[nodiscard]] IdentityOrder compareIdentity(
    const SourceIdentity& candidate,
    const SourceIdentity& reference) noexcept;

struct Frame
{
    PixelFormat format = PixelFormat::Rgb24;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t producer_epoch = 0;
    std::uint64_t source_sequence = 0;
    std::uint64_t capture_timestamp_ns = 0;
    std::vector<std::uint8_t> pixel_bytes;

    [[nodiscard]] SourceIdentity identity() const noexcept
    {
        return {producer_epoch, source_sequence};
    }
};

enum class MailboxPublishStatus
{
    Accepted,
    Replaced,
    Duplicate,
    Regression,
    InvalidFrame,
};

class LatestFrameMailbox
{
public:
    [[nodiscard]] MailboxPublishStatus publish(Frame frame);
    [[nodiscard]] bool tryTakeNewest(Frame* frame);
    void clear();

private:
    std::mutex mutex_;
    std::optional<SourceIdentity> last_accepted_identity_;
    std::optional<Frame> latest_;
};

struct ReceiverConfig
{
    std::string host = "127.0.0.1";
    std::uint16_t port = 0;
    std::chrono::milliseconds connect_timeout{500};
    std::chrono::milliseconds reconnect_initial_backoff{10};
    std::chrono::milliseconds reconnect_max_backoff{250};
    std::chrono::milliseconds io_poll_interval{20};
};

struct ReceiverCounters
{
    std::uint64_t connect_attempts = 0;
    std::uint64_t connect_successes = 0;
    std::uint64_t connect_failures = 0;
    std::uint64_t reconnect_attempts = 0;
    std::uint64_t disconnects = 0;
    std::uint64_t headers_received = 0;
    std::uint64_t complete_frames = 0;
    std::uint64_t accepted_frames = 0;
    std::uint64_t replaced_frames = 0;
    std::uint64_t duplicate_frames = 0;
    std::uint64_t regression_frames = 0;
    std::uint64_t invalid_headers = 0;
    std::uint64_t read_failures = 0;
    std::uint64_t allocation_failures = 0;
    std::uint64_t wire_bytes_received = 0;
    std::uint64_t header_read_duration_count = 0;
    std::uint64_t header_read_duration_ns_total = 0;
    std::uint64_t header_read_duration_ns_max = 0;
    std::uint64_t payload_read_duration_count = 0;
    std::uint64_t payload_read_duration_ns_total = 0;
    std::uint64_t payload_read_duration_ns_max = 0;
    std::uint64_t connection_lifetime_count = 0;
    std::uint64_t connection_lifetime_ns_total = 0;
    std::uint64_t connection_lifetime_ns_max = 0;
    std::uint64_t source_age_samples = 0;
    std::uint64_t invalid_source_age_samples = 0;
    std::uint64_t last_accepted_epoch = 0;
    std::uint64_t last_accepted_sequence = 0;
    double latest_source_age_ms = 0.0;
    bool source_age_available = false;
};

class Receiver
{
public:
    explicit Receiver(ReceiverConfig config);
    ~Receiver();

    Receiver(const Receiver&) = delete;
    Receiver& operator=(const Receiver&) = delete;

    [[nodiscard]] bool start(std::string* error = nullptr);
    void stop() noexcept;
    [[nodiscard]] bool running() const noexcept;
    [[nodiscard]] bool tryTakeLatest(Frame* frame);
    [[nodiscard]] ReceiverCounters counters() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aim_sim_bridge::tcp_image
