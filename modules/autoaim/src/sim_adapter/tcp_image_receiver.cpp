#include "aim_sim_bridge/tcp_image_receiver.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <new>
#include <thread>
#include <utility>

namespace aim_sim_bridge::tcp_image
{
namespace
{

constexpr std::chrono::milliseconds kMaximumConnectTimeout{5000};
constexpr std::chrono::milliseconds kMaximumReconnectBackoff{5000};
constexpr std::chrono::milliseconds kMaximumIoPollInterval{100};

void recordDuration(
    std::atomic<std::uint64_t>& total_ns,
    std::atomic<std::uint64_t>& max_ns,
    std::chrono::steady_clock::time_point started) noexcept
{
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const auto raw_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    const std::uint64_t elapsed_ns =
        raw_ns > 0 ? static_cast<std::uint64_t>(raw_ns) : static_cast<std::uint64_t>(0);
    total_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    std::uint64_t observed = max_ns.load(std::memory_order_relaxed);
    while (observed < elapsed_ns &&
           !max_ns.compare_exchange_weak(
               observed, elapsed_ns, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

struct DurationSample
{
    std::atomic<std::uint64_t>& count;
    std::atomic<std::uint64_t>& total_ns;
    std::atomic<std::uint64_t>& max_ns;
    const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();

    ~DurationSample() noexcept
    {
        count.fetch_add(1U, std::memory_order_relaxed);
        recordDuration(total_ns, max_ns, started);
    }
};

std::uint64_t systemNowNs() noexcept
{
    const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
    const auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    return nanoseconds > 0 ? static_cast<std::uint64_t>(nanoseconds) : 0U;
}

int pollTimeoutMs(std::chrono::milliseconds timeout) noexcept
{
    const auto count = timeout.count();
    if (count <= 0) return 1;
    if (count >= static_cast<std::chrono::milliseconds::rep>(INT_MAX)) return INT_MAX;
    return static_cast<int>(count);
}

bool validConfig(const ReceiverConfig& config, std::string* error)
{
    auto fail = [error](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };

    in_addr address{};
    if (::inet_pton(AF_INET, config.host.c_str(), &address) != 1) {
        return fail("host must be a numeric IPv4 address");
    }
    if (config.port == 0U) return fail("port must be nonzero");
    if (config.connect_timeout <= std::chrono::milliseconds::zero() ||
        config.connect_timeout > kMaximumConnectTimeout) {
        return fail("connect_timeout must be within 1..5000ms");
    }
    if (config.reconnect_initial_backoff <= std::chrono::milliseconds::zero() ||
        config.reconnect_initial_backoff > kMaximumReconnectBackoff) {
        return fail("reconnect_initial_backoff must be within 1..5000ms");
    }
    if (config.reconnect_max_backoff < config.reconnect_initial_backoff ||
        config.reconnect_max_backoff > kMaximumReconnectBackoff) {
        return fail("reconnect_max_backoff must be >= initial and <=5000ms");
    }
    if (config.io_poll_interval <= std::chrono::milliseconds::zero() ||
        config.io_poll_interval > kMaximumIoPollInterval) {
        return fail("io_poll_interval must be within 1..100ms");
    }
    if (error != nullptr) error->clear();
    return true;
}

std::chrono::milliseconds doubledBackoff(
    std::chrono::milliseconds current,
    std::chrono::milliseconds maximum) noexcept
{
    if (current >= maximum || current.count() > maximum.count() / 2) return maximum;
    return std::min(current * 2, maximum);
}

enum class ReadStatus
{
    Complete,
    PeerClosed,
    Error,
    Stopped,
};

struct ReadOutcome
{
    ReadStatus status = ReadStatus::Error;
    std::size_t bytes_read = 0;
};

struct AtomicCounters
{
    std::atomic<std::uint64_t> connect_attempts{0};
    std::atomic<std::uint64_t> connect_successes{0};
    std::atomic<std::uint64_t> connect_failures{0};
    std::atomic<std::uint64_t> reconnect_attempts{0};
    std::atomic<std::uint64_t> disconnects{0};
    std::atomic<std::uint64_t> headers_received{0};
    std::atomic<std::uint64_t> complete_frames{0};
    std::atomic<std::uint64_t> accepted_frames{0};
    std::atomic<std::uint64_t> replaced_frames{0};
    std::atomic<std::uint64_t> duplicate_frames{0};
    std::atomic<std::uint64_t> regression_frames{0};
    std::atomic<std::uint64_t> invalid_headers{0};
    std::atomic<std::uint64_t> read_failures{0};
    std::atomic<std::uint64_t> allocation_failures{0};
    std::atomic<std::uint64_t> wire_bytes_received{0};
    std::atomic<std::uint64_t> header_read_duration_count{0};
    std::atomic<std::uint64_t> header_read_duration_ns_total{0};
    std::atomic<std::uint64_t> header_read_duration_ns_max{0};
    std::atomic<std::uint64_t> payload_read_duration_count{0};
    std::atomic<std::uint64_t> payload_read_duration_ns_total{0};
    std::atomic<std::uint64_t> payload_read_duration_ns_max{0};
    std::atomic<std::uint64_t> connection_lifetime_count{0};
    std::atomic<std::uint64_t> connection_lifetime_ns_total{0};
    std::atomic<std::uint64_t> connection_lifetime_ns_max{0};
    std::atomic<std::uint64_t> source_age_samples{0};
    std::atomic<std::uint64_t> invalid_source_age_samples{0};
    std::atomic<std::uint64_t> last_accepted_epoch{0};
    std::atomic<std::uint64_t> last_accepted_sequence{0};
    std::atomic<double> latest_source_age_ms{0.0};
    std::atomic<bool> source_age_available{false};

    [[nodiscard]] ReceiverCounters snapshot() const noexcept
    {
        ReceiverCounters out;
        out.connect_attempts = connect_attempts.load(std::memory_order_relaxed);
        out.connect_successes = connect_successes.load(std::memory_order_relaxed);
        out.connect_failures = connect_failures.load(std::memory_order_relaxed);
        out.reconnect_attempts = reconnect_attempts.load(std::memory_order_relaxed);
        out.disconnects = disconnects.load(std::memory_order_relaxed);
        out.headers_received = headers_received.load(std::memory_order_relaxed);
        out.complete_frames = complete_frames.load(std::memory_order_relaxed);
        out.accepted_frames = accepted_frames.load(std::memory_order_relaxed);
        out.replaced_frames = replaced_frames.load(std::memory_order_relaxed);
        out.duplicate_frames = duplicate_frames.load(std::memory_order_relaxed);
        out.regression_frames = regression_frames.load(std::memory_order_relaxed);
        out.invalid_headers = invalid_headers.load(std::memory_order_relaxed);
        out.read_failures = read_failures.load(std::memory_order_relaxed);
        out.allocation_failures = allocation_failures.load(std::memory_order_relaxed);
        out.wire_bytes_received = wire_bytes_received.load(std::memory_order_relaxed);
        out.header_read_duration_count =
            header_read_duration_count.load(std::memory_order_relaxed);
        out.header_read_duration_ns_total =
            header_read_duration_ns_total.load(std::memory_order_relaxed);
        out.header_read_duration_ns_max =
            header_read_duration_ns_max.load(std::memory_order_relaxed);
        out.payload_read_duration_count =
            payload_read_duration_count.load(std::memory_order_relaxed);
        out.payload_read_duration_ns_total =
            payload_read_duration_ns_total.load(std::memory_order_relaxed);
        out.payload_read_duration_ns_max =
            payload_read_duration_ns_max.load(std::memory_order_relaxed);
        out.connection_lifetime_count =
            connection_lifetime_count.load(std::memory_order_relaxed);
        out.connection_lifetime_ns_total =
            connection_lifetime_ns_total.load(std::memory_order_relaxed);
        out.connection_lifetime_ns_max =
            connection_lifetime_ns_max.load(std::memory_order_relaxed);
        out.source_age_samples = source_age_samples.load(std::memory_order_relaxed);
        out.invalid_source_age_samples =
            invalid_source_age_samples.load(std::memory_order_relaxed);
        out.last_accepted_epoch = last_accepted_epoch.load(std::memory_order_relaxed);
        out.last_accepted_sequence =
            last_accepted_sequence.load(std::memory_order_relaxed);
        out.latest_source_age_ms = latest_source_age_ms.load(std::memory_order_relaxed);
        out.source_age_available = source_age_available.load(std::memory_order_relaxed);
        return out;
    }
};

}  // namespace

IdentityOrder compareIdentity(
    const SourceIdentity& candidate,
    const SourceIdentity& reference) noexcept
{
    if (candidate.producer_epoch < reference.producer_epoch) {
        return IdentityOrder::Regression;
    }
    if (candidate.producer_epoch > reference.producer_epoch) {
        return IdentityOrder::Newer;
    }
    if (candidate.source_sequence < reference.source_sequence) {
        return IdentityOrder::Regression;
    }
    if (candidate.source_sequence == reference.source_sequence) {
        return IdentityOrder::Duplicate;
    }
    return IdentityOrder::Newer;
}

MailboxPublishStatus LatestFrameMailbox::publish(Frame frame)
{
    std::uint32_t expected_payload_bytes = 0;
    if (frame.producer_epoch == 0U || frame.source_sequence == 0U ||
        !checkedPayloadBytes(
            frame.width, frame.height, frame.format, &expected_payload_bytes) ||
        frame.pixel_bytes.size() != static_cast<std::size_t>(expected_payload_bytes)) {
        return MailboxPublishStatus::InvalidFrame;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const SourceIdentity candidate = frame.identity();
    if (last_accepted_identity_.has_value()) {
        const IdentityOrder order = compareIdentity(candidate, *last_accepted_identity_);
        if (order == IdentityOrder::Duplicate) return MailboxPublishStatus::Duplicate;
        if (order == IdentityOrder::Regression) return MailboxPublishStatus::Regression;
    }

    const bool replaced = latest_.has_value();
    last_accepted_identity_ = candidate;
    latest_ = std::move(frame);
    return replaced ? MailboxPublishStatus::Replaced : MailboxPublishStatus::Accepted;
}

bool LatestFrameMailbox::tryTakeNewest(Frame* frame)
{
    if (frame == nullptr) return false;
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || !latest_.has_value()) return false;

    *frame = std::move(*latest_);
    latest_.reset();
    return true;
}

void LatestFrameMailbox::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    last_accepted_identity_.reset();
    latest_.reset();
}

class Receiver::Impl
{
public:
    explicit Impl(ReceiverConfig receiver_config) : config(std::move(receiver_config)) {}

    ~Impl() { stopReceiver(); }

    bool startReceiver(std::string* error)
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if (worker.joinable()) {
            if (error != nullptr) *error = "receiver is already started";
            return false;
        }
        if (!validConfig(config, error)) return false;

        mailbox.clear();
        stop_requested.store(false, std::memory_order_release);
        running.store(true, std::memory_order_release);
        try {
            worker = std::thread(&Impl::workerLoop, this);
        } catch (const std::exception& exception) {
            running.store(false, std::memory_order_release);
            stop_requested.store(true, std::memory_order_release);
            if (error != nullptr) *error = exception.what();
            return false;
        } catch (...) {
            running.store(false, std::memory_order_release);
            stop_requested.store(true, std::memory_order_release);
            if (error != nullptr) *error = "failed to create receiver thread";
            return false;
        }
        if (error != nullptr) error->clear();
        return true;
    }

    void stopReceiver() noexcept
    {
        std::thread joining_thread;
        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex);
            stop_requested.store(true, std::memory_order_release);
            shutdownActiveSocket();
            backoff_cv.notify_all();
            if (worker.joinable()) joining_thread = std::move(worker);
        }
        if (joining_thread.joinable()) joining_thread.join();
        running.store(false, std::memory_order_release);
    }

    [[nodiscard]] bool isRunning() const noexcept
    {
        return running.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool tryTake(Frame* frame)
    {
        return mailbox.tryTakeNewest(frame);
    }

    [[nodiscard]] ReceiverCounters counterSnapshot() const noexcept
    {
        return metrics.snapshot();
    }

private:
    bool activateSocket(int socket_fd)
    {
        std::lock_guard<std::mutex> lock(socket_mutex);
        if (stop_requested.load(std::memory_order_acquire)) return false;
        active_socket = socket_fd;
        return true;
    }

    void closeActiveSocket(int socket_fd) noexcept
    {
        std::lock_guard<std::mutex> lock(socket_mutex);
        if (active_socket == socket_fd) active_socket = -1;
        (void)::close(socket_fd);
    }

    void shutdownActiveSocket() noexcept
    {
        std::lock_guard<std::mutex> lock(socket_mutex);
        if (active_socket >= 0) (void)::shutdown(active_socket, SHUT_RDWR);
    }

    int connectSocket()
    {
        const int socket_fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        if (socket_fd < 0) return -1;
        if (!activateSocket(socket_fd)) {
            (void)::close(socket_fd);
            return -1;
        }

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(config.port);
        if (::inet_pton(AF_INET, config.host.c_str(), &address.sin_addr) != 1) {
            closeActiveSocket(socket_fd);
            return -1;
        }

        const int connect_result = ::connect(
            socket_fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
        if (connect_result == 0) return socket_fd;
        if (connect_result < 0 && errno != EINPROGRESS) {
            closeActiveSocket(socket_fd);
            return -1;
        }

        const auto deadline = std::chrono::steady_clock::now() + config.connect_timeout;
        while (!stop_requested.load(std::memory_order_acquire)) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) break;
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            const auto wait_time = std::min(config.io_poll_interval, remaining);
            pollfd descriptor{socket_fd, static_cast<short>(POLLOUT | POLLERR | POLLHUP), 0};
            const int poll_result = ::poll(&descriptor, 1, pollTimeoutMs(wait_time));
            if (poll_result < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (poll_result == 0) continue;

            int socket_error = 0;
            socklen_t error_size = static_cast<socklen_t>(sizeof(socket_error));
            if (::getsockopt(
                    socket_fd, SOL_SOCKET, SO_ERROR, &socket_error, &error_size) == 0 &&
                socket_error == 0) {
                return socket_fd;
            }
            break;
        }

        closeActiveSocket(socket_fd);
        return -1;
    }

    ReadOutcome readExact(int socket_fd, std::uint8_t* destination, std::size_t bytes)
    {
        ReadOutcome outcome;
        outcome.status = ReadStatus::Complete;
        while (outcome.bytes_read < bytes) {
            if (stop_requested.load(std::memory_order_acquire)) {
                outcome.status = ReadStatus::Stopped;
                return outcome;
            }

            const ssize_t received = ::recv(
                socket_fd,
                destination + outcome.bytes_read,
                bytes - outcome.bytes_read,
                0);
            if (received > 0) {
                const auto received_bytes = static_cast<std::size_t>(received);
                outcome.bytes_read += received_bytes;
                metrics.wire_bytes_received.fetch_add(
                    static_cast<std::uint64_t>(received_bytes), std::memory_order_relaxed);
                continue;
            }
            if (received == 0) {
                outcome.status = stop_requested.load(std::memory_order_acquire)
                    ? ReadStatus::Stopped
                    : ReadStatus::PeerClosed;
                return outcome;
            }
            if (errno == EINTR) continue;
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                outcome.status = stop_requested.load(std::memory_order_acquire)
                    ? ReadStatus::Stopped
                    : ReadStatus::Error;
                return outcome;
            }

            pollfd descriptor{socket_fd, static_cast<short>(POLLIN | POLLERR | POLLHUP), 0};
            const int poll_result =
                ::poll(&descriptor, 1, pollTimeoutMs(config.io_poll_interval));
            if (poll_result < 0 && errno != EINTR) {
                outcome.status = stop_requested.load(std::memory_order_acquire)
                    ? ReadStatus::Stopped
                    : ReadStatus::Error;
                return outcome;
            }
        }
        return outcome;
    }

    bool waitBackoff(std::chrono::milliseconds delay)
    {
        std::unique_lock<std::mutex> lock(backoff_mutex);
        return backoff_cv.wait_for(lock, delay, [this] {
            return stop_requested.load(std::memory_order_acquire);
        });
    }

    void recordAcceptedIdentity(
        std::uint64_t producer_epoch,
        std::uint64_t source_sequence,
        std::uint64_t capture_timestamp_ns) noexcept
    {
        metrics.last_accepted_epoch.store(producer_epoch, std::memory_order_relaxed);
        metrics.last_accepted_sequence.store(source_sequence, std::memory_order_relaxed);

        const std::uint64_t completed_ns = systemNowNs();
        if (capture_timestamp_ns != 0U && completed_ns >= capture_timestamp_ns) {
            const double age_ms =
                static_cast<double>(completed_ns - capture_timestamp_ns) * 1e-6;
            metrics.latest_source_age_ms.store(age_ms, std::memory_order_relaxed);
            metrics.source_age_available.store(true, std::memory_order_relaxed);
            metrics.source_age_samples.fetch_add(1U, std::memory_order_relaxed);
        } else {
            metrics.invalid_source_age_samples.fetch_add(1U, std::memory_order_relaxed);
        }
    }

    void processConnectedSocket(int socket_fd)
    {
        DurationSample connection_lifetime{
            metrics.connection_lifetime_count,
            metrics.connection_lifetime_ns_total,
            metrics.connection_lifetime_ns_max};
        while (!stop_requested.load(std::memory_order_acquire)) {
            WireHeader wire_header{};
            const auto header_started = std::chrono::steady_clock::now();
            const ReadOutcome header_read =
                readExact(socket_fd, wire_header.data(), wire_header.size());
            metrics.header_read_duration_count.fetch_add(1U, std::memory_order_relaxed);
            recordDuration(
                metrics.header_read_duration_ns_total,
                metrics.header_read_duration_ns_max,
                header_started);
            if (header_read.status == ReadStatus::Stopped) return;
            if (header_read.status != ReadStatus::Complete) {
                if (header_read.status == ReadStatus::Error || header_read.bytes_read != 0U) {
                    metrics.read_failures.fetch_add(1U, std::memory_order_relaxed);
                }
                metrics.disconnects.fetch_add(1U, std::memory_order_relaxed);
                return;
            }
            metrics.headers_received.fetch_add(1U, std::memory_order_relaxed);

            const HeaderDecodeResult decoded =
                decodeHeader(wire_header.data(), wire_header.size());
            if (!decoded.ok()) {
                metrics.invalid_headers.fetch_add(1U, std::memory_order_relaxed);
                metrics.disconnects.fetch_add(1U, std::memory_order_relaxed);
                return;
            }

            std::vector<std::uint8_t> payload;
            try {
                payload.resize(static_cast<std::size_t>(decoded.header.payload_bytes));
            } catch (const std::bad_alloc&) {
                metrics.allocation_failures.fetch_add(1U, std::memory_order_relaxed);
                metrics.disconnects.fetch_add(1U, std::memory_order_relaxed);
                return;
            }

            const auto payload_started = std::chrono::steady_clock::now();
            const ReadOutcome payload_read =
                readExact(socket_fd, payload.data(), payload.size());
            metrics.payload_read_duration_count.fetch_add(1U, std::memory_order_relaxed);
            recordDuration(
                metrics.payload_read_duration_ns_total,
                metrics.payload_read_duration_ns_max,
                payload_started);
            if (payload_read.status == ReadStatus::Stopped) return;
            if (payload_read.status != ReadStatus::Complete) {
                metrics.read_failures.fetch_add(1U, std::memory_order_relaxed);
                metrics.disconnects.fetch_add(1U, std::memory_order_relaxed);
                return;
            }
            metrics.complete_frames.fetch_add(1U, std::memory_order_relaxed);

            const std::uint64_t producer_epoch = decoded.header.producer_epoch;
            const std::uint64_t source_sequence = decoded.header.source_sequence;
            const std::uint64_t capture_timestamp_ns = decoded.header.capture_timestamp_ns;
            Frame frame;
            frame.format = decoded.header.format;
            frame.width = decoded.header.width;
            frame.height = decoded.header.height;
            frame.producer_epoch = producer_epoch;
            frame.source_sequence = source_sequence;
            frame.capture_timestamp_ns = capture_timestamp_ns;
            frame.pixel_bytes = std::move(payload);

            const MailboxPublishStatus status = mailbox.publish(std::move(frame));
            switch (status) {
                case MailboxPublishStatus::Accepted:
                    metrics.accepted_frames.fetch_add(1U, std::memory_order_relaxed);
                    recordAcceptedIdentity(
                        producer_epoch, source_sequence, capture_timestamp_ns);
                    break;
                case MailboxPublishStatus::Replaced:
                    metrics.accepted_frames.fetch_add(1U, std::memory_order_relaxed);
                    metrics.replaced_frames.fetch_add(1U, std::memory_order_relaxed);
                    recordAcceptedIdentity(
                        producer_epoch, source_sequence, capture_timestamp_ns);
                    break;
                case MailboxPublishStatus::Duplicate:
                    metrics.duplicate_frames.fetch_add(1U, std::memory_order_relaxed);
                    break;
                case MailboxPublishStatus::Regression:
                    metrics.regression_frames.fetch_add(1U, std::memory_order_relaxed);
                    break;
                case MailboxPublishStatus::InvalidFrame:
                    metrics.invalid_headers.fetch_add(1U, std::memory_order_relaxed);
                    metrics.disconnects.fetch_add(1U, std::memory_order_relaxed);
                    return;
            }
        }
    }

    void workerLoop() noexcept
    {
        bool first_attempt = true;
        auto backoff = config.reconnect_initial_backoff;
        while (!stop_requested.load(std::memory_order_acquire)) {
            if (!first_attempt) {
                metrics.reconnect_attempts.fetch_add(1U, std::memory_order_relaxed);
            }
            first_attempt = false;
            metrics.connect_attempts.fetch_add(1U, std::memory_order_relaxed);

            const int socket_fd = connectSocket();
            if (socket_fd < 0) {
                if (stop_requested.load(std::memory_order_acquire)) break;
                metrics.connect_failures.fetch_add(1U, std::memory_order_relaxed);
                if (waitBackoff(backoff)) break;
                backoff = doubledBackoff(backoff, config.reconnect_max_backoff);
                continue;
            }

            metrics.connect_successes.fetch_add(1U, std::memory_order_relaxed);
            backoff = config.reconnect_initial_backoff;
            processConnectedSocket(socket_fd);
            closeActiveSocket(socket_fd);
            if (stop_requested.load(std::memory_order_acquire)) break;
            if (waitBackoff(backoff)) break;
        }
        running.store(false, std::memory_order_release);
    }

    ReceiverConfig config;
    LatestFrameMailbox mailbox;
    AtomicCounters metrics;
    std::atomic<bool> stop_requested{true};
    std::atomic<bool> running{false};
    std::mutex lifecycle_mutex;
    std::thread worker;
    std::mutex socket_mutex;
    int active_socket = -1;
    std::mutex backoff_mutex;
    std::condition_variable backoff_cv;
};

Receiver::Receiver(ReceiverConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

Receiver::~Receiver() = default;

bool Receiver::start(std::string* error)
{
    return impl_->startReceiver(error);
}

void Receiver::stop() noexcept
{
    impl_->stopReceiver();
}

bool Receiver::running() const noexcept
{
    return impl_->isRunning();
}

bool Receiver::tryTakeLatest(Frame* frame)
{
    return impl_->tryTake(frame);
}

ReceiverCounters Receiver::counters() const noexcept
{
    return impl_->counterSnapshot();
}

}  // namespace aim_sim_bridge::tcp_image
