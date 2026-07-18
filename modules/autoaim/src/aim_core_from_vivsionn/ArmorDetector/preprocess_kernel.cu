#include "preprocess_kernel.hpp" 
#include <cuda_fp16.h>        
#include <stdint.h>           

__global__ void preprocess_kernel_impl(
    const uint8_t* hwc_bgr_u8_in,  
    uint16_t* chw_rgb_fp16_out, 
    const int width,
    const int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pixels = width * height;

    if (idx < num_pixels)
    {
        const uint8_t* pSrc = hwc_bgr_u8_in + idx * 3;
        uint16_t* pDst_R = chw_rgb_fp16_out;
        uint16_t* pDst_G = chw_rgb_fp16_out + num_pixels;
        uint16_t* pDst_B = chw_rgb_fp16_out + num_pixels * 2;

        half r_fp16 = __float2half((float)pSrc[2] / 255.0f);
        half g_fp16 = __float2half((float)pSrc[1] / 255.0f);
        half b_fp16 = __float2half((float)pSrc[0] / 255.0f);

        pDst_R[idx] = *reinterpret_cast<uint16_t*>(&r_fp16);
        pDst_G[idx] = *reinterpret_cast<uint16_t*>(&g_fp16);
        pDst_B[idx] = *reinterpret_cast<uint16_t*>(&b_fp16);
    }
}

extern "C" void launch_preprocess_kernel(
    const uint8_t* hwc_bgr_u8_in,
    uint16_t* chw_rgb_fp16_out,
    const int width,
    const int height,
    cudaStream_t stream)
{   
    const int num_pixels = width * height;
    const int threads_per_block = 256; 
    const int blocks = (num_pixels + threads_per_block - 1) / threads_per_block;

    preprocess_kernel_impl<<<blocks, threads_per_block, 0, stream>>>(
        hwc_bgr_u8_in, chw_rgb_fp16_out, width, height
    );
}