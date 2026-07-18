#pragma once

#include <daedalus_sim_sdk/tcp_image_v1.hpp>

namespace aim_sim_bridge::tcp_image {

namespace sdk_tcp = daedalus::sim::sdk::v1::tcp_image;

inline constexpr auto kWireMagic = sdk_tcp::kWireMagic;
inline constexpr auto kWireVersion = sdk_tcp::kWireVersion;
inline constexpr auto kWireHeaderBytes = sdk_tcp::kWireHeaderBytes;
inline constexpr auto kMaxWidth = daedalus::sim::sdk::v1::kImageWidth;
inline constexpr auto kMaxHeight = daedalus::sim::sdk::v1::kImageHeight;
inline constexpr auto kMaxPayloadBytes = sdk_tcp::kMaxPayloadBytes;

using PixelFormat = sdk_tcp::PixelFormat;
using FrameHeader = sdk_tcp::FrameHeader;
using WireHeader = sdk_tcp::WireHeader;
using HeaderStatus = sdk_tcp::HeaderStatus;
using HeaderDecodeResult = sdk_tcp::HeaderDecodeResult;
using sdk_tcp::channelsFor;
using sdk_tcp::checkedPayloadBytes;
using sdk_tcp::decodeHeader;
using sdk_tcp::encodeHeader;
using sdk_tcp::validateHeader;

[[nodiscard]] inline const char* toString(HeaderStatus status) noexcept {
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
  }
  return "unknown";
}

}  // namespace aim_sim_bridge::tcp_image
