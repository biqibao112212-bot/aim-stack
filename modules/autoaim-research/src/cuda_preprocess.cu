#include "autoaim_research/cuda_preprocess.hpp"

#include <cuda_fp16.h>

namespace autoaim_research {
namespace {

template <typename Output>
__device__ void storeValue(Output* output, int index, float value);

template <>
__device__ void storeValue<float>(float* output, int index, float value) {
  output[index] = value;
}

template <>
__device__ void storeValue<half>(half* output, int index, float value) {
  output[index] = __float2half_rn(value);
}

template <typename Output>
__global__ void letterboxBgrToRgbChwKernel(
    const std::uint8_t* source, int source_width, int source_height,
    std::size_t source_stride, Output* destination, int resized_width,
    int resized_height, float scale) {
  constexpr int destination_width = 640;
  constexpr int destination_height = 640;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixel_count = destination_width * destination_height;
  if (index >= pixel_count) return;

  const int x = index % destination_width;
  const int y = index / destination_width;
  float red = 0.0F;
  float green = 0.0F;
  float blue = 0.0F;
  if (x < resized_width && y < resized_height) {
    float source_x = (static_cast<float>(x) + 0.5F) / scale - 0.5F;
    float source_y = (static_cast<float>(y) + 0.5F) / scale - 0.5F;
    source_x = fminf(fmaxf(source_x, 0.0F), source_width - 1.0F);
    source_y = fminf(fmaxf(source_y, 0.0F), source_height - 1.0F);
    const int x0 = static_cast<int>(floorf(source_x));
    const int y0 = static_cast<int>(floorf(source_y));
    const int x1 = min(x0 + 1, source_width - 1);
    const int y1 = min(y0 + 1, source_height - 1);
    const float dx = source_x - static_cast<float>(x0);
    const float dy = source_y - static_cast<float>(y0);

    const std::uint8_t* row0 = source + y0 * source_stride;
    const std::uint8_t* row1 = source + y1 * source_stride;
    const std::uint8_t* p00 = row0 + x0 * 3;
    const std::uint8_t* p01 = row0 + x1 * 3;
    const std::uint8_t* p10 = row1 + x0 * 3;
    const std::uint8_t* p11 = row1 + x1 * 3;
    const auto interpolate = [&](int channel) {
      const float top = static_cast<float>(p00[channel]) * (1.0F - dx) +
                        static_cast<float>(p01[channel]) * dx;
      const float bottom = static_cast<float>(p10[channel]) * (1.0F - dx) +
                           static_cast<float>(p11[channel]) * dx;
      return (top * (1.0F - dy) + bottom * dy) / 255.0F;
    };
    blue = interpolate(0);
    green = interpolate(1);
    red = interpolate(2);
  }

  storeValue(destination, index, red);
  storeValue(destination, pixel_count + index, green);
  storeValue(destination, 2 * pixel_count + index, blue);
}

}  // namespace

cudaError_t launchLetterboxBgrToRgbChw(
    const std::uint8_t* source_bgr, int source_width, int source_height,
    std::size_t source_stride_bytes, void* destination_chw,
    bool destination_fp16, int resized_width, int resized_height, float scale,
    cudaStream_t stream) {
  constexpr int pixel_count = 640 * 640;
  constexpr int threads = 256;
  constexpr int blocks = (pixel_count + threads - 1) / threads;
  if (destination_fp16) {
    letterboxBgrToRgbChwKernel<<<blocks, threads, 0, stream>>>(
        source_bgr, source_width, source_height, source_stride_bytes,
        static_cast<half*>(destination_chw), resized_width, resized_height,
        scale);
  } else {
    letterboxBgrToRgbChwKernel<<<blocks, threads, 0, stream>>>(
        source_bgr, source_width, source_height, source_stride_bytes,
        static_cast<float*>(destination_chw), resized_width, resized_height,
        scale);
  }
  return cudaGetLastError();
}

}  // namespace autoaim_research
