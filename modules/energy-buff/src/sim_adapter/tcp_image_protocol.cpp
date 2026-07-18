#include "aim_sim_bridge/tcp_image_protocol.hpp"

#include <limits>

namespace aim_sim_bridge::tcp_image
{
namespace
{

constexpr std::size_t kMagicOffset = 0;
constexpr std::size_t kVersionOffset = 4;
constexpr std::size_t kHeaderBytesOffset = 6;
constexpr std::size_t kFormatOffset = 8;
constexpr std::size_t kFlagsOffset = 10;
constexpr std::size_t kWidthOffset = 12;
constexpr std::size_t kHeightOffset = 16;
constexpr std::size_t kPayloadBytesOffset = 20;
constexpr std::size_t kProducerEpochOffset = 24;
constexpr std::size_t kSourceSequenceOffset = 32;
constexpr std::size_t kCaptureTimestampOffset = 40;
constexpr std::size_t kReserved0Offset = 48;
constexpr std::size_t kReserved1Offset = 56;

void writeU16(WireHeader& wire, std::size_t offset, std::uint16_t value) noexcept
{
    wire[offset] = static_cast<std::uint8_t>((value >> 8U) & 0xffU);
    wire[offset + 1U] = static_cast<std::uint8_t>(value & 0xffU);
}

void writeU32(WireHeader& wire, std::size_t offset, std::uint32_t value) noexcept
{
    wire[offset] = static_cast<std::uint8_t>((value >> 24U) & 0xffU);
    wire[offset + 1U] = static_cast<std::uint8_t>((value >> 16U) & 0xffU);
    wire[offset + 2U] = static_cast<std::uint8_t>((value >> 8U) & 0xffU);
    wire[offset + 3U] = static_cast<std::uint8_t>(value & 0xffU);
}

void writeU64(WireHeader& wire, std::size_t offset, std::uint64_t value) noexcept
{
    for (std::size_t index = 0; index < 8U; ++index) {
        const std::size_t shift = (7U - index) * 8U;
        wire[offset + index] = static_cast<std::uint8_t>((value >> shift) & 0xffU);
    }
}

std::uint16_t readU16(const std::uint8_t* wire, std::size_t offset) noexcept
{
    return static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(wire[offset]) << 8U) |
        static_cast<std::uint16_t>(wire[offset + 1U]));
}

std::uint32_t readU32(const std::uint8_t* wire, std::size_t offset) noexcept
{
    return (static_cast<std::uint32_t>(wire[offset]) << 24U) |
        (static_cast<std::uint32_t>(wire[offset + 1U]) << 16U) |
        (static_cast<std::uint32_t>(wire[offset + 2U]) << 8U) |
        static_cast<std::uint32_t>(wire[offset + 3U]);
}

std::uint64_t readU64(const std::uint8_t* wire, std::size_t offset) noexcept
{
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8U; ++index) {
        value = (value << 8U) | static_cast<std::uint64_t>(wire[offset + index]);
    }
    return value;
}

}  // namespace

std::uint32_t channelsFor(PixelFormat format) noexcept
{
    switch (format) {
        case PixelFormat::Rgb24: return 3U;
        case PixelFormat::Rgba32: return 4U;
        default: return 0U;
    }
}

bool checkedPayloadBytes(
    std::uint32_t width,
    std::uint32_t height,
    PixelFormat format,
    std::uint32_t* payload_bytes) noexcept
{
    if (payload_bytes == nullptr || width == 0U || height == 0U ||
        width > kMaxWidth || height > kMaxHeight) {
        return false;
    }
    const std::uint32_t channels = channelsFor(format);
    if (channels == 0U) return false;

    const std::uint64_t bytes = static_cast<std::uint64_t>(width) *
        static_cast<std::uint64_t>(height) * static_cast<std::uint64_t>(channels);
    if (bytes > static_cast<std::uint64_t>(kMaxPayloadBytes) ||
        bytes > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }
    *payload_bytes = static_cast<std::uint32_t>(bytes);
    return true;
}

HeaderStatus validateHeader(const FrameHeader& header) noexcept
{
    if (channelsFor(header.format) == 0U) return HeaderStatus::UnsupportedFormat;
    if (header.flags != 0U) return HeaderStatus::NonzeroFlags;

    std::uint32_t expected_payload_bytes = 0;
    if (!checkedPayloadBytes(
            header.width, header.height, header.format, &expected_payload_bytes)) {
        return HeaderStatus::InvalidDimensions;
    }
    if (header.payload_bytes != expected_payload_bytes) {
        return HeaderStatus::InvalidPayloadBytes;
    }
    if (header.producer_epoch == 0U || header.source_sequence == 0U) {
        return HeaderStatus::InvalidIdentity;
    }
    if (header.reserved0 != 0U || header.reserved1 != 0U) {
        return HeaderStatus::NonzeroReserved;
    }
    return HeaderStatus::Ok;
}

HeaderStatus encodeHeader(const FrameHeader& header, WireHeader* wire) noexcept
{
    if (wire == nullptr) return HeaderStatus::NullOutput;
    const HeaderStatus status = validateHeader(header);
    if (status != HeaderStatus::Ok) return status;

    wire->fill(0U);
    writeU32(*wire, kMagicOffset, kWireMagic);
    writeU16(*wire, kVersionOffset, kWireVersion);
    writeU16(*wire, kHeaderBytesOffset, kWireHeaderBytes);
    writeU16(*wire, kFormatOffset, static_cast<std::uint16_t>(header.format));
    writeU16(*wire, kFlagsOffset, header.flags);
    writeU32(*wire, kWidthOffset, header.width);
    writeU32(*wire, kHeightOffset, header.height);
    writeU32(*wire, kPayloadBytesOffset, header.payload_bytes);
    writeU64(*wire, kProducerEpochOffset, header.producer_epoch);
    writeU64(*wire, kSourceSequenceOffset, header.source_sequence);
    writeU64(*wire, kCaptureTimestampOffset, header.capture_timestamp_ns);
    writeU64(*wire, kReserved0Offset, header.reserved0);
    writeU64(*wire, kReserved1Offset, header.reserved1);
    return HeaderStatus::Ok;
}

HeaderDecodeResult decodeHeader(const std::uint8_t* wire, std::size_t wire_size) noexcept
{
    HeaderDecodeResult result;
    if (wire == nullptr || wire_size != kWireHeaderBytes) {
        result.status = HeaderStatus::WireSizeMismatch;
        return result;
    }
    if (readU32(wire, kMagicOffset) != kWireMagic) {
        result.status = HeaderStatus::InvalidMagic;
        return result;
    }
    if (readU16(wire, kVersionOffset) != kWireVersion) {
        result.status = HeaderStatus::UnsupportedVersion;
        return result;
    }
    if (readU16(wire, kHeaderBytesOffset) != kWireHeaderBytes) {
        result.status = HeaderStatus::InvalidHeaderBytes;
        return result;
    }

    result.header.format = static_cast<PixelFormat>(readU16(wire, kFormatOffset));
    result.header.flags = readU16(wire, kFlagsOffset);
    result.header.width = readU32(wire, kWidthOffset);
    result.header.height = readU32(wire, kHeightOffset);
    result.header.payload_bytes = readU32(wire, kPayloadBytesOffset);
    result.header.producer_epoch = readU64(wire, kProducerEpochOffset);
    result.header.source_sequence = readU64(wire, kSourceSequenceOffset);
    result.header.capture_timestamp_ns = readU64(wire, kCaptureTimestampOffset);
    result.header.reserved0 = readU64(wire, kReserved0Offset);
    result.header.reserved1 = readU64(wire, kReserved1Offset);
    result.status = validateHeader(result.header);
    return result;
}

const char* toString(HeaderStatus status) noexcept
{
    switch (status) {
        case HeaderStatus::Ok: return "ok";
        case HeaderStatus::NullOutput: return "null_output";
        case HeaderStatus::WireSizeMismatch: return "wire_size_mismatch";
        case HeaderStatus::InvalidMagic: return "invalid_magic";
        case HeaderStatus::UnsupportedVersion: return "unsupported_version";
        case HeaderStatus::InvalidHeaderBytes: return "invalid_header_bytes";
        case HeaderStatus::UnsupportedFormat: return "unsupported_format";
        case HeaderStatus::NonzeroFlags: return "nonzero_flags";
        case HeaderStatus::InvalidDimensions: return "invalid_dimensions";
        case HeaderStatus::InvalidPayloadBytes: return "invalid_payload_bytes";
        case HeaderStatus::InvalidIdentity: return "invalid_identity";
        case HeaderStatus::NonzeroReserved: return "nonzero_reserved";
        default: return "unknown";
    }
}

}  // namespace aim_sim_bridge::tcp_image
