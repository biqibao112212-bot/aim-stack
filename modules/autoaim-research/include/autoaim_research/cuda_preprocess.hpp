#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace autoaim_research {

// Fuses letterbox resize, BGR->RGB, uint8->float conversion and HWC->CHW on
// one CUDA stream. Padding is placed on the right/bottom to preserve the
// detector's existing coordinate contract.
cudaError_t launchLetterboxBgrToRgbChw(
    const std::uint8_t* source_bgr, int source_width, int source_height,
    std::size_t source_stride_bytes, void* destination_chw,
    bool destination_fp16, int resized_width, int resized_height, float scale,
    cudaStream_t stream);

}  // namespace autoaim_research
