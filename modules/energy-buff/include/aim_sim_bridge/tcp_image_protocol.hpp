#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace aim_sim_bridge::tcp_image
{

constexpr std::uint32_t kWireMagic = 0x54494d47U;  // ASCII "TIMG"
constexpr std::uint16_t kWireVersion = 1;
constexpr std::uint16_t kWireHeaderBytes = 64;
constexpr std::uint32_t kMaxWidth = 1280;
constexpr std::uint32_t kMaxHeight = 720;
constexpr std::size_t kMaxPayloadBytes =
    static_cast<std::size_t>(kMaxWidth) * static_cast<std::size_t>(kMaxHeight) * 4U;

enum class PixelFormat : std::uint16_t
{
    Rgb24 = 1,
    Rgba32 = 2,
};

struct FrameHeader
{
    PixelFormat format = PixelFormat::Rgb24;
    std::uint16_t flags = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t payload_bytes = 0;
    std::uint64_t producer_epoch = 0;
    std::uint64_t source_sequence = 0;
    std::uint64_t capture_timestamp_ns = 0;
    std::uint64_t reserved0 = 0;
    std::uint64_t reserved1 = 0;
};

using WireHeader = std::array<std::uint8_t, kWireHeaderBytes>;

enum class HeaderStatus
{
    Ok,
    NullOutput,
    WireSizeMismatch,
    InvalidMagic,
    UnsupportedVersion,
    InvalidHeaderBytes,
    UnsupportedFormat,
    NonzeroFlags,
    InvalidDimensions,
    InvalidPayloadBytes,
    InvalidIdentity,
    NonzeroReserved,
};

struct HeaderDecodeResult
{
    HeaderStatus status = HeaderStatus::WireSizeMismatch;
    FrameHeader header{};

    [[nodiscard]] bool ok() const noexcept { return status == HeaderStatus::Ok; }
};

[[nodiscard]] std::uint32_t channelsFor(PixelFormat format) noexcept;
[[nodiscard]] bool checkedPayloadBytes(
    std::uint32_t width,
    std::uint32_t height,
    PixelFormat format,
    std::uint32_t* payload_bytes) noexcept;
[[nodiscard]] HeaderStatus validateHeader(const FrameHeader& header) noexcept;
[[nodiscard]] HeaderStatus encodeHeader(
    const FrameHeader& header,
    WireHeader* wire) noexcept;
[[nodiscard]] HeaderDecodeResult decodeHeader(
    const std::uint8_t* wire,
    std::size_t wire_size) noexcept;
[[nodiscard]] const char* toString(HeaderStatus status) noexcept;

}  // namespace aim_sim_bridge::tcp_image
