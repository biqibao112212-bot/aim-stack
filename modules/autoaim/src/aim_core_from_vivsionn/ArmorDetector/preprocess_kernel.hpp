#pragma once
#include <cuda_runtime.h>
#include <cstdint>

extern "C" {
void launch_preprocess_kernel(
    const uint8_t* hwc_bgr_u8_in,
    void* chw_rgb_out,
    const int width,
    const int height,
    bool output_fp16,
    cudaStream_t stream
);
}
