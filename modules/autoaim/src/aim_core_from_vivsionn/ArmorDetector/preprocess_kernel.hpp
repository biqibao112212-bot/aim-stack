#pragma once
#include <cuda_runtime.h>
#include <cstdint>

extern "C" {
void launch_preprocess_kernel(
    const uint8_t* hwc_bgr_u8_in,
    uint16_t* chw_rgb_fp16_out,
    const int width,
    const int height,
    cudaStream_t stream
);
}