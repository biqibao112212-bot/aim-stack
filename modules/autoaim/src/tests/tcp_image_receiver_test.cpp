#include "aim_sim_bridge/tcp_image_protocol.hpp"
#include "aim_sim_bridge/tcp_image_receiver.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <poll.h>
#include <string>
#include <sstream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{

namespace tcp = aim_sim_bridge::tcp_image;
using namespace std::chrono_literals;

int failures = 0;

void check(bool condition, const std::string& message)
{
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}
template <typename Predicate>
bool waitUntil(Predicate&& predicate, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(1ms);
    }
    return predicate();
}
std::uint64_t nowNs()
{
    const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    return value > 0 ? static_cast<std::uint64_t>(value) : 0U;
}

void closeFd(int* socket_fd)
{
    if (socket_fd == nullptr || *socket_fd < 0) return;
    (void)::shutdown(*socket_fd, SHUT_RDWR);
    (void)::close(*socket_fd);
    *socket_fd = -1;
}

class TestServer
{
public:
    TestServer()
    {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (listen_fd_ < 0) return;

        const int reuse = 1;
        if (::setsockopt(
                listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse,
                static_cast<socklen_t>(sizeof(reuse))) != 0) {
            closeFd(&listen_fd_);
            return;
        }

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        if (::bind(
                listen_fd_, reinterpret_cast<const sockaddr*>(&address),
                static_cast<socklen_t>(sizeof(address))) != 0 ||
            ::listen(listen_fd_, 8) != 0) {
            closeFd(&listen_fd_);
            return;
        }

        socklen_t address_size = static_cast<socklen_t>(sizeof(address));
        if (::getsockname(
                listen_fd_, reinterpret_cast<sockaddr*>(&address), &address_size) != 0) {
            closeFd(&listen_fd_);
            return;
        }
        port_ = ntohs(address.sin_port);
    }

    ~TestServer() { closeFd(&listen_fd_); }

    TestServer(const TestServer&) = delete;
    TestServer& operator=(const TestServer&) = delete;

    [[nodiscard]] bool valid() const noexcept { return listen_fd_ >= 0 && port_ != 0U; }
    [[nodiscard]] std::uint16_t port() const noexcept { return port_; }

    [[nodiscard]] int acceptClient(std::chrono::milliseconds timeout) const
    {
        if (listen_fd_ < 0) return -1;
        pollfd descriptor{listen_fd_, POLLIN, 0};
        const auto raw_timeout = timeout.count();
        const int timeout_ms = raw_timeout > 0 ? static_cast<int>(raw_timeout) : 1;
        const int poll_result = ::poll(&descriptor, static_cast<nfds_t>(1), timeout_ms);
        if (poll_result <= 0) return -1;
        return ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
    }

private:
    int listen_fd_ = -1;
    std::uint16_t port_ = 0;
};

tcp::FrameHeader makeHeader(
    tcp::PixelFormat format,
    std::uint32_t width,
    std::uint32_t height,
    std::uint64_t epoch,
    std::uint64_t sequence,
    std::uint64_t capture_timestamp_ns)
{
    tcp::FrameHeader header;
    header.format = format;
    header.width = width;
    header.height = height;
    header.producer_epoch = epoch;
    header.source_sequence = sequence;
    header.capture_timestamp_ns = capture_timestamp_ns;
    std::uint32_t payload_bytes = 0;
    if (tcp::checkedPayloadBytes(width, height, format, &payload_bytes)) {
        header.payload_bytes = payload_bytes;
    }
    return header;
}

std::vector<std::uint8_t> makePayload(const tcp::FrameHeader& header, std::uint8_t seed)
{
    std::vector<std::uint8_t> payload(static_cast<std::size_t>(header.payload_bytes));
    for (std::size_t index = 0; index < payload.size(); ++index) {
        payload[index] = static_cast<std::uint8_t>(
            (index + static_cast<std::size_t>(seed)) % static_cast<std::size_t>(251U));
    }
    return payload;
}

bool sendFragmented(
    int socket_fd,
    const std::uint8_t* data,
    std::size_t bytes,
    std::size_t maximum_chunk)
{
    if (socket_fd < 0 || data == nullptr || maximum_chunk == 0U) return false;
    std::size_t sent = 0;
    while (sent < bytes) {
        const std::size_t chunk = std::min(maximum_chunk, bytes - sent);
        const ssize_t result = ::send(socket_fd, data + sent, chunk, MSG_NOSIGNAL);
        if (result > 0) {
            sent += static_cast<std::size_t>(result);
            continue;
        }
        if (result < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

bool sendFrame(
    int socket_fd,
    const tcp::FrameHeader& header,
    const std::vector<std::uint8_t>& payload,
    std::size_t header_chunk,
    std::size_t payload_chunk)
{
    tcp::WireHeader wire{};
    if (tcp::encodeHeader(header, &wire) != tcp::HeaderStatus::Ok ||
        payload.size() != static_cast<std::size_t>(header.payload_bytes)) {
        return false;
    }
    return sendFragmented(socket_fd, wire.data(), wire.size(), header_chunk) &&
        sendFragmented(socket_fd, payload.data(), payload.size(), payload_chunk);
}

tcp::ReceiverConfig receiverConfig(std::uint16_t port)
{
    tcp::ReceiverConfig config;
    config.host = "127.0.0.1";
    config.port = port;
    config.connect_timeout = 300ms;
    config.reconnect_initial_backoff = 5ms;
    config.reconnect_max_backoff = 20ms;
    config.io_poll_interval = 5ms;
    return config;
}

void expectDecodeStatus(
    const tcp::WireHeader& wire,
    tcp::HeaderStatus expected,
    const char* message)
{
    const tcp::HeaderDecodeResult decoded = tcp::decodeHeader(wire.data(), wire.size());
    check(decoded.status == expected, std::string(message) + " actual=" + tcp::toString(decoded.status));
}

void testWireCodecAndBounds()
{
    check(tcp::kWireHeaderBytes == 64U, "wire header size must be 64 bytes");
    check(tcp::channelsFor(tcp::PixelFormat::Rgb24) == 3U, "RGB24 channel count");
    check(tcp::channelsFor(tcp::PixelFormat::Rgba32) == 4U, "RGBA32 channel count");

    std::uint32_t payload_bytes = 0;
    check(
        tcp::checkedPayloadBytes(1440U, 1080U, tcp::PixelFormat::Rgb24, &payload_bytes) &&
            payload_bytes == 4'665'600U,
        "maximum RGB24 payload must be exact");
    check(
        tcp::checkedPayloadBytes(1440U, 1080U, tcp::PixelFormat::Rgba32, &payload_bytes) &&
            payload_bytes == 6'220'800U,
        "maximum RGBA32 payload must be exact");
    check(
        !tcp::checkedPayloadBytes(0U, 1080U, tcp::PixelFormat::Rgb24, &payload_bytes) &&
            !tcp::checkedPayloadBytes(1441U, 1080U, tcp::PixelFormat::Rgb24, &payload_bytes) &&
            !tcp::checkedPayloadBytes(1440U, 1081U, tcp::PixelFormat::Rgba32, &payload_bytes) &&
            !tcp::checkedPayloadBytes(
                1U, 1U, static_cast<tcp::PixelFormat>(99U), &payload_bytes) &&
            !tcp::checkedPayloadBytes(1U, 1U, tcp::PixelFormat::Rgb24, nullptr),
        "payload helper must reject zero, bounds, format, and null output");

    const tcp::FrameHeader rgb_header = makeHeader(
        tcp::PixelFormat::Rgb24,
        1440U,
        1080U,
        0x0102030405060708ULL,
        0x1112131415161718ULL,
        0x2122232425262728ULL);
    tcp::WireHeader wire{};
    check(tcp::encodeHeader(rgb_header, &wire) == tcp::HeaderStatus::Ok, "encode valid RGB header");
    check(
        wire[0] == 0x54U && wire[1] == 0x49U && wire[2] == 0x4dU && wire[3] == 0x47U &&
            wire[4] == 0x00U && wire[5] == 0x01U && wire[6] == 0x00U && wire[7] == 0x40U,
        "golden magic/version/header bytes must be big endian");
    check(
        wire[8] == 0x00U && wire[9] == 0x01U && wire[12] == 0x00U && wire[13] == 0x00U &&
            wire[14] == 0x05U && wire[15] == 0xa0U && wire[16] == 0x00U && wire[17] == 0x00U &&
            wire[18] == 0x04U && wire[19] == 0x38U && wire[20] == 0x00U && wire[21] == 0x47U &&
            wire[22] == 0x31U && wire[23] == 0x00U,
        "golden RGB format/dimensions/payload bytes");
    check(
        wire[24] == 0x01U && wire[31] == 0x08U && wire[32] == 0x11U && wire[39] == 0x18U &&
            wire[40] == 0x21U && wire[47] == 0x28U,
        "golden 64-bit identities must be big endian");

    const tcp::HeaderDecodeResult decoded = tcp::decodeHeader(wire.data(), wire.size());
    check(
        decoded.ok() && decoded.header.format == tcp::PixelFormat::Rgb24 &&
            decoded.header.width == rgb_header.width && decoded.header.height == rgb_header.height &&
            decoded.header.payload_bytes == rgb_header.payload_bytes &&
            decoded.header.producer_epoch == rgb_header.producer_epoch &&
            decoded.header.source_sequence == rgb_header.source_sequence &&
            decoded.header.capture_timestamp_ns == rgb_header.capture_timestamp_ns,
        "RGB golden header must round trip exactly");

    const tcp::FrameHeader rgba_header = makeHeader(
        tcp::PixelFormat::Rgba32, 1440U, 1080U, 7U, 8U, 9U);
    tcp::WireHeader rgba_wire{};
    check(tcp::encodeHeader(rgba_header, &rgba_wire) == tcp::HeaderStatus::Ok, "encode RGBA header");
    check(
        rgba_wire[8] == 0x00U && rgba_wire[9] == 0x02U &&
            rgba_wire[20] == 0x00U && rgba_wire[21] == 0x5eU &&
            rgba_wire[22] == 0xecU && rgba_wire[23] == 0x00U,
        "RGBA format and exact payload must have stable golden bytes");
    check(
        tcp::decodeHeader(rgba_wire.data(), rgba_wire.size()).header.format ==
            tcp::PixelFormat::Rgba32,
        "RGBA format must round trip");

    check(
        tcp::decodeHeader(wire.data(), wire.size() - 1U).status ==
            tcp::HeaderStatus::WireSizeMismatch &&
            tcp::decodeHeader(nullptr, wire.size()).status == tcp::HeaderStatus::WireSizeMismatch &&
            tcp::encodeHeader(rgb_header, nullptr) == tcp::HeaderStatus::NullOutput,
        "codec must reject wrong wire size and null buffers");

    tcp::WireHeader invalid = wire;
    invalid[0] ^= 0x01U;
    expectDecodeStatus(invalid, tcp::HeaderStatus::InvalidMagic, "invalid magic");
    invalid = wire;
    invalid[5] = 0x02U;
    expectDecodeStatus(invalid, tcp::HeaderStatus::UnsupportedVersion, "unsupported version");
    invalid = wire;
    invalid[7] = 0x3fU;
    expectDecodeStatus(invalid, tcp::HeaderStatus::InvalidHeaderBytes, "invalid header bytes");
    invalid = wire;
    invalid[9] = 0x03U;
    expectDecodeStatus(invalid, tcp::HeaderStatus::UnsupportedFormat, "unsupported format");
    invalid = wire;
    invalid[11] = 0x01U;
    expectDecodeStatus(invalid, tcp::HeaderStatus::NonzeroFlags, "nonzero flags");
    invalid = wire;
    invalid[12] = invalid[13] = invalid[14] = invalid[15] = 0U;
    expectDecodeStatus(invalid, tcp::HeaderStatus::InvalidDimensions, "zero width");
    invalid = wire;
    invalid[23] ^= 0x01U;
    expectDecodeStatus(invalid, tcp::HeaderStatus::InvalidPayloadBytes, "payload mismatch");
    invalid = wire;
    for (std::size_t index = 24U; index < 32U; ++index) invalid[index] = 0U;
    expectDecodeStatus(invalid, tcp::HeaderStatus::InvalidIdentity, "zero epoch");
    invalid = wire;
    invalid[63] = 0x01U;
    expectDecodeStatus(invalid, tcp::HeaderStatus::NonzeroReserved, "nonzero reserved");
}

tcp::Frame makeMailboxFrame(
    tcp::PixelFormat format,
    std::uint64_t epoch,
    std::uint64_t sequence,
    std::uint8_t seed)
{
    const tcp::FrameHeader header = makeHeader(format, 2U, 2U, epoch, sequence, nowNs());
    tcp::Frame frame;
    frame.format = format;
    frame.width = header.width;
    frame.height = header.height;
    frame.producer_epoch = epoch;
    frame.source_sequence = sequence;
    frame.capture_timestamp_ns = header.capture_timestamp_ns;
    frame.pixel_bytes = makePayload(header, seed);
    return frame;
}

void testLatestMailboxIdentityAndOwnership()
{
    tcp::LatestFrameMailbox mailbox;
    check(
        mailbox.publish(makeMailboxFrame(tcp::PixelFormat::Rgb24, 10U, 5U, 1U)) ==
            tcp::MailboxPublishStatus::Accepted,
        "mailbox accepts first identity");
    check(
        mailbox.publish(makeMailboxFrame(tcp::PixelFormat::Rgb24, 10U, 5U, 2U)) ==
            tcp::MailboxPublishStatus::Duplicate,
        "mailbox rejects duplicate identity");
    check(
        mailbox.publish(makeMailboxFrame(tcp::PixelFormat::Rgb24, 10U, 4U, 3U)) ==
            tcp::MailboxPublishStatus::Regression,
        "mailbox rejects regressed sequence");
    const tcp::Frame newest = makeMailboxFrame(tcp::PixelFormat::Rgba32, 11U, 1U, 4U);
    const std::vector<std::uint8_t> expected_bytes = newest.pixel_bytes;
    check(
        mailbox.publish(newest) == tcp::MailboxPublishStatus::Replaced,
        "new epoch replaces unread older frame");

    tcp::Frame received;
    check(mailbox.tryTakeNewest(&received), "main-side nonblocking take gets newest frame");
    check(
        received.producer_epoch == 11U && received.source_sequence == 1U &&
            received.format == tcp::PixelFormat::Rgba32 &&
            received.pixel_bytes == expected_bytes,
        "taken frame preserves newest identity, format, and owned bytes");
    check(!mailbox.tryTakeNewest(&received), "mailbox never replays a taken frame");
    check(!mailbox.tryTakeNewest(nullptr), "mailbox rejects null take destination");

    tcp::Frame invalid = makeMailboxFrame(tcp::PixelFormat::Rgb24, 12U, 1U, 5U);
    invalid.pixel_bytes.pop_back();
    check(
        mailbox.publish(std::move(invalid)) == tcp::MailboxPublishStatus::InvalidFrame,
        "mailbox validates exact owned payload size");
    mailbox.clear();
    check(
        mailbox.publish(makeMailboxFrame(tcp::PixelFormat::Rgb24, 1U, 1U, 6U)) ==
            tcp::MailboxPublishStatus::Accepted,
        "explicit clear resets pending frame and identity horizon");
}

void testFragmentationLatestReplacementAndMonotonicCounters()
{
    TestServer server;
    check(server.valid(), "fragmentation test server setup");
    if (!server.valid()) return;

    tcp::Receiver receiver(receiverConfig(server.port()));
    std::string start_error;
    check(receiver.start(&start_error), "receiver starts: " + start_error);
    int client_fd = server.acceptClient(2s);
    check(client_fd >= 0, "receiver connects to fragmentation server");
    if (client_fd < 0) {
        receiver.stop();
        return;
    }

    const tcp::FrameHeader rgb =
        makeHeader(tcp::PixelFormat::Rgb24, 4U, 3U, 100U, 1U, nowNs());
    const std::vector<std::uint8_t> rgb_payload = makePayload(rgb, 11U);
    check(sendFrame(client_fd, rgb, rgb_payload, 1U, 2U), "send fragmented RGB frame");
    check(
        waitUntil([&receiver] { return receiver.counters().accepted_frames >= 1U; }, 2s),
        "receiver completes fragmented RGB frame");

    const tcp::FrameHeader rgba =
        makeHeader(tcp::PixelFormat::Rgba32, 3U, 2U, 100U, 2U, nowNs());
    const std::vector<std::uint8_t> rgba_payload = makePayload(rgba, 22U);
    check(sendFrame(client_fd, rgba, rgba_payload, 3U, 1U), "send fragmented RGBA frame");
    check(
        waitUntil(
            [&receiver] {
                const auto counters = receiver.counters();
                return counters.accepted_frames >= 2U && counters.replaced_frames >= 1U;
            },
            2s),
        "new complete frame replaces unread result");

    tcp::Frame received;
    check(receiver.tryTakeLatest(&received), "take latest fragmented frame");
    check(
        received.producer_epoch == 100U && received.source_sequence == 2U &&
            received.format == tcp::PixelFormat::Rgba32 &&
            received.width == 3U && received.height == 2U &&
            received.pixel_bytes == rgba_payload,
        "latest RGBA frame identity/format/owned bytes are exact");

    check(sendFrame(client_fd, rgba, rgba_payload, 64U, rgba_payload.size()), "send duplicate frame");
    const tcp::FrameHeader regression =
        makeHeader(tcp::PixelFormat::Rgb24, 4U, 3U, 100U, 1U, nowNs());
    check(
        sendFrame(client_fd, regression, makePayload(regression, 33U), 64U, 64U),
        "send regressed frame");
    check(
        waitUntil(
            [&receiver] {
                const auto counters = receiver.counters();
                return counters.duplicate_frames >= 1U && counters.regression_frames >= 1U;
            },
            2s),
        "duplicate and regression counters advance");
    check(!receiver.tryTakeLatest(&received), "duplicate/regression frames remain invisible");

    const auto before_stop = std::chrono::steady_clock::now();
    receiver.stop();
    const auto stop_elapsed = std::chrono::steady_clock::now() - before_stop;
    check(stop_elapsed < 1s, "stop/join interrupts an active socket deterministically");
    check(!receiver.running(), "receiver reports stopped after join");
    closeFd(&client_fd);

    const tcp::ReceiverCounters counters = receiver.counters();
    check(
        counters.headers_received >= 4U && counters.complete_frames >= 4U &&
            counters.accepted_frames == 2U && counters.replaced_frames == 1U &&
            counters.duplicate_frames == 1U && counters.regression_frames == 1U,
        "fragmentation/latest/identity counters are separated");
    check(
        counters.connect_attempts >= 1U && counters.connect_successes >= 1U &&
            counters.wire_bytes_received >=
                4U * static_cast<std::uint64_t>(tcp::kWireHeaderBytes) &&
            counters.last_accepted_epoch == 100U && counters.last_accepted_sequence == 2U &&
            counters.source_age_available && counters.source_age_samples == 2U,
        "connection, wire byte, identity, and source age metrics are exposed");
    check(
        counters.header_read_duration_count >= counters.headers_received &&
            counters.payload_read_duration_count >= counters.complete_frames &&
            counters.header_read_duration_ns_max <= counters.header_read_duration_ns_total &&
            counters.payload_read_duration_ns_max <= counters.payload_read_duration_ns_total &&
            counters.connection_lifetime_count >= 1U &&
            counters.connection_lifetime_ns_max <= counters.connection_lifetime_ns_total,
        "fragmented reader records bounded header/payload/connection timing");
}

void testPartialInvalidReconnectAndNoVisibility()
{
    TestServer server;
    check(server.valid(), "reconnect test server setup");
    if (!server.valid()) return;

    tcp::Receiver receiver(receiverConfig(server.port()));
    std::string start_error;
    check(receiver.start(&start_error), "reconnect receiver starts: " + start_error);

    int first_client = server.acceptClient(2s);
    check(first_client >= 0, "first receiver connection");
    if (first_client < 0) {
        receiver.stop();
        return;
    }

    const tcp::FrameHeader partial =
        makeHeader(tcp::PixelFormat::Rgba32, 4U, 2U, 200U, 1U, nowNs());
    const std::vector<std::uint8_t> partial_payload = makePayload(partial, 41U);
    tcp::WireHeader partial_wire{};
    check(tcp::encodeHeader(partial, &partial_wire) == tcp::HeaderStatus::Ok, "encode partial frame");
    check(
        sendFragmented(first_client, partial_wire.data(), partial_wire.size(), 5U) &&
            sendFragmented(
                first_client,
                partial_payload.data(),
                partial_payload.size() / 2U,
                3U),
        "send header and only half its payload");
    closeFd(&first_client);
    check(
        waitUntil(
            [&receiver] {
                const auto counters = receiver.counters();
                return counters.disconnects >= 1U && counters.read_failures >= 1U;
            },
            2s),
        "partial payload causes read failure and disconnect");
    tcp::Frame invisible;
    check(!receiver.tryTakeLatest(&invisible), "partial frame is never visible");

    int second_client = server.acceptClient(2s);
    check(second_client >= 0, "receiver reconnects after partial payload");
    if (second_client < 0) {
        receiver.stop();
        return;
    }
    tcp::WireHeader invalid_wire{};
    const tcp::FrameHeader valid_for_mutation =
        makeHeader(tcp::PixelFormat::Rgb24, 2U, 2U, 200U, 1U, nowNs());
    check(
        tcp::encodeHeader(valid_for_mutation, &invalid_wire) == tcp::HeaderStatus::Ok,
        "encode header before invalid mutation");
    invalid_wire[23] ^= 0x01U;
    check(
        sendFragmented(second_client, invalid_wire.data(), invalid_wire.size(), 2U),
        "send invalid exact-payload header");
    check(
        waitUntil([&receiver] { return receiver.counters().invalid_headers >= 1U; }, 2s),
        "invalid header rejected before payload allocation");
    closeFd(&second_client);
    check(!receiver.tryTakeLatest(&invisible), "invalid header frame is never visible");

    int third_client = server.acceptClient(2s);
    check(third_client >= 0, "receiver reconnects after invalid header");
    if (third_client < 0) {
        receiver.stop();
        return;
    }
    const tcp::FrameHeader complete =
        makeHeader(tcp::PixelFormat::Rgb24, 2U, 2U, 200U, 1U, nowNs());
    const std::vector<std::uint8_t> complete_payload = makePayload(complete, 52U);
    check(sendFrame(third_client, complete, complete_payload, 4U, 4U), "send post-reconnect frame");
    check(
        waitUntil([&receiver] { return receiver.counters().accepted_frames >= 1U; }, 2s),
        "complete frame after reconnect is accepted");
    check(receiver.tryTakeLatest(&invisible), "post-reconnect complete frame is visible");
    check(
        invisible.producer_epoch == 200U && invisible.source_sequence == 1U &&
            invisible.pixel_bytes == complete_payload,
        "post-reconnect frame preserves identity and owned bytes");

    receiver.stop();
    closeFd(&third_client);
    const tcp::ReceiverCounters counters = receiver.counters();
    check(
        counters.connect_successes >= 3U && counters.reconnect_attempts >= 2U &&
            counters.disconnects >= 2U && counters.read_failures >= 1U &&
            counters.invalid_headers >= 1U && counters.complete_frames == 1U &&
            counters.accepted_frames == 1U && counters.allocation_failures == 0U,
        "partial/invalid/reconnect counters are exact and partial frames stay incomplete");
}

void testConfigurationAndStoppedBackoffJoin()
{
    tcp::ReceiverConfig invalid;
    invalid.host = "localhost";
    invalid.port = 1234U;
    tcp::Receiver invalid_receiver(invalid);
    std::string error;
    check(!invalid_receiver.start(&error) && !error.empty(), "DNS names are rejected for deterministic numeric IPv4 connect");
    invalid_receiver.stop();

    TestServer reservation;
    check(reservation.valid(), "backoff stop test port reservation");
    if (!reservation.valid()) return;
    const std::uint16_t closed_port = reservation.port();
    // The listening socket stays open here, so connect is accepted by the kernel even without
    // accept(); stop still has to interrupt the connected read deterministically.
    tcp::Receiver receiver(receiverConfig(closed_port));
    check(receiver.start(&error), "stop test receiver starts");
    check(
        waitUntil([&receiver] { return receiver.counters().connect_successes >= 1U; }, 2s),
        "stop test establishes socket");
    const auto before_stop = std::chrono::steady_clock::now();
    receiver.stop();
    check(
        std::chrono::steady_clock::now() - before_stop < 1s,
        "stop/join is bounded while peer never sends a header");
}

void testTalosTcpMetadataAdmissionSourceOrder()
{
    const std::filesystem::path source_path =
        std::filesystem::path(__FILE__).parent_path().parent_path() /
        "sim_adapter" / "talos_bridge_node.cpp";
    std::ifstream input(source_path, std::ios::binary);
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string source = contents.str();
    check(input.good() || input.eof(), "Talos bridge source is readable for admission-order test");

    const std::size_t loop = source.find("TCP_METADATA_ADMISSION_TRY_TAKE");
    const std::size_t take = source.find("tryTakeLatest(&tcp_frame)", loop);
    const std::size_t taken_counter = source.find("recordTaken()", take);
    const std::size_t snapshot = source.find(
        "TCP_METADATA_ADMISSION_SNAPSHOT_AFTER_TAKE", taken_counter);
    const std::size_t metadata_read = source.find(
        "readFullyAt(meta_file.fd(), &meta_snapshot", snapshot);
    const std::size_t epoch_gate = source.find(
        "image_sequence.observe(meta_snapshot.header.created_ns)", metadata_read);
    const std::size_t following_gate = source.find(
        "meta_snapshot.runtime_state.following", epoch_gate);
    const std::size_t identity_gate = source.find(
        "selectTcpImageStrict(", following_gate);
    const std::size_t exact_exposure = source.find(
        "readExactExposureTruth(", identity_gate);

    check(
        loop != std::string::npos && take != std::string::npos &&
            taken_counter != std::string::npos && snapshot != std::string::npos &&
            metadata_read != std::string::npos && epoch_gate != std::string::npos &&
            following_gate != std::string::npos && identity_gate != std::string::npos &&
            exact_exposure != std::string::npos && loop < take && take < taken_counter &&
            taken_counter < snapshot && snapshot < metadata_read &&
            metadata_read < epoch_gate && epoch_gate < following_gate &&
            following_gate < identity_gate && identity_gate < exact_exposure,
        "TCP admission must take a frame before one metadata snapshot, then preserve epoch/following/identity/exact-exposure order");

    const std::size_t idle_branch = source.rfind("if (!tcp_image_receiver", take);
    const std::size_t idle_telemetry = source.find("write_telemetry_if_due", take);
    const std::size_t idle_wait = source.find(
        "sleep_for(std::chrono::milliseconds(1))", idle_telemetry);
    const std::size_t idle_continue = source.find("continue;", idle_wait);
    check(
        idle_branch != std::string::npos && idle_telemetry < idle_wait &&
            idle_wait < idle_continue && idle_continue < snapshot,
        "no-frame TCP admission writes due telemetry and waits 1 ms before any metadata snapshot");

    const std::size_t failure_counter = source.find(
        "metadata_read_validate_failures_after_take", metadata_read);
    check(
        failure_counter != std::string::npos && failure_counter < epoch_gate,
        "metadata failure after a destructive TCP take is explicitly counted");
}

}  // namespace

int main()
{
    testWireCodecAndBounds();
    testLatestMailboxIdentityAndOwnership();
    testFragmentationLatestReplacementAndMonotonicCounters();
    testPartialInvalidReconnectAndNoVisibility();
    testConfigurationAndStoppedBackoffJoin();
    testTalosTcpMetadataAdmissionSourceOrder();

    if (failures != 0) {
        std::cerr << "tcp_image_receiver_test failures=" << failures << '\n';
        return 1;
    }
    std::cout << "tcp_image_receiver_test passed\n";
    return 0;
}
