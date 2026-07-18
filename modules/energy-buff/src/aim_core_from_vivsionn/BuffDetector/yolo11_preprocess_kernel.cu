#include "yolo11_preprocess_kernel.hpp"

#include <algorithm>

namespace auto_buff
{
namespace
{

__device__ __forceinline__ float clamp_float(float value, float lo, float hi)
{
  return fminf(fmaxf(value, lo), hi);
}

__device__ __forceinline__ float sample_bilinear_channel(
  const uint8_t * src,
  int src_w,
  int src_h,
  size_t src_step,
  float src_x,
  float src_y,
  int channel)
{
  src_x = clamp_float(src_x, 0.0f, static_cast<float>(src_w - 1));
  src_y = clamp_float(src_y, 0.0f, static_cast<float>(src_h - 1));

  const int x0 = static_cast<int>(floorf(src_x));
  const int y0 = static_cast<int>(floorf(src_y));
  const int x1 = min(x0 + 1, src_w - 1);
  const int y1 = min(y0 + 1, src_h - 1);
  const float dx = src_x - static_cast<float>(x0);
  const float dy = src_y - static_cast<float>(y0);

  const uint8_t * row0 = src + static_cast<size_t>(y0) * src_step;
  const uint8_t * row1 = src + static_cast<size_t>(y1) * src_step;
  const float v00 = static_cast<float>(row0[x0 * 3 + channel]);
  const float v01 = static_cast<float>(row0[x1 * 3 + channel]);
  const float v10 = static_cast<float>(row1[x0 * 3 + channel]);
  const float v11 = static_cast<float>(row1[x1 * 3 + channel]);

  const float top = v00 + (v01 - v00) * dx;
  const float bottom = v10 + (v11 - v10) * dx;
  return top + (bottom - top) * dy;
}

template <typename T>
__device__ __forceinline__ void store_value(T * dst, int idx, float value);

template <>
__device__ __forceinline__ void store_value<float>(float * dst, int idx, float value)
{
  dst[idx] = value;
}

template <>
__device__ __forceinline__ void store_value<__half>(__half * dst, int idx, float value)
{
  dst[idx] = __float2half(value);
}

template <typename T>
__global__ void yolo11_preprocess_kernel(
  const uint8_t * src_bgr,
  int src_w,
  int src_h,
  size_t src_step,
  T * dst_chw_rgb,
  int dst_w,
  int dst_h,
  int resized_w,
  int resized_h,
  int pad_left,
  int pad_top)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixel_count = dst_w * dst_h;
  if (idx >= pixel_count) {
    return;
  }

  const int x = idx % dst_w;
  const int y = idx / dst_w;
  float r = 114.0f;
  float g = 114.0f;
  float b = 114.0f;

  const int local_x = x - pad_left;
  const int local_y = y - pad_top;
  if (
    local_x >= 0 && local_x < resized_w &&
    local_y >= 0 && local_y < resized_h &&
    src_w > 0 && src_h > 0) {
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(resized_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(resized_h);
    const float src_x = (static_cast<float>(local_x) + 0.5f) * scale_x - 0.5f;
    const float src_y = (static_cast<float>(local_y) + 0.5f) * scale_y - 0.5f;

    b = sample_bilinear_channel(src_bgr, src_w, src_h, src_step, src_x, src_y, 0);
    g = sample_bilinear_channel(src_bgr, src_w, src_h, src_step, src_x, src_y, 1);
    r = sample_bilinear_channel(src_bgr, src_w, src_h, src_step, src_x, src_y, 2);
  }

  const float inv_255 = 1.0f / 255.0f;
  store_value(dst_chw_rgb, idx, r * inv_255);
  store_value(dst_chw_rgb, idx + pixel_count, g * inv_255);
  store_value(dst_chw_rgb, idx + pixel_count * 2, b * inv_255);
}

template <typename T>
void launch_yolo11_preprocess_impl(
  const uint8_t * src_bgr,
  int src_w,
  int src_h,
  size_t src_step,
  T * dst_chw_rgb,
  int dst_w,
  int dst_h,
  int resized_w,
  int resized_h,
  int pad_left,
  int pad_top,
  cudaStream_t stream)
{
  const int pixel_count = dst_w * dst_h;
  constexpr int threads_per_block = 256;
  const int blocks = (pixel_count + threads_per_block - 1) / threads_per_block;
  yolo11_preprocess_kernel<<<blocks, threads_per_block, 0, stream>>>(
    src_bgr,
    src_w,
    src_h,
    src_step,
    dst_chw_rgb,
    dst_w,
    dst_h,
    resized_w,
    resized_h,
    pad_left,
    pad_top);
}

}  // namespace

void launch_yolo11_preprocess_float(
  const uint8_t * src_bgr,
  int src_w,
  int src_h,
  size_t src_step,
  float * dst_chw_rgb,
  int dst_w,
  int dst_h,
  int resized_w,
  int resized_h,
  int pad_left,
  int pad_top,
  cudaStream_t stream)
{
  launch_yolo11_preprocess_impl(
    src_bgr,
    src_w,
    src_h,
    src_step,
    dst_chw_rgb,
    dst_w,
    dst_h,
    resized_w,
    resized_h,
    pad_left,
    pad_top,
    stream);
}

void launch_yolo11_preprocess_half(
  const uint8_t * src_bgr,
  int src_w,
  int src_h,
  size_t src_step,
  __half * dst_chw_rgb,
  int dst_w,
  int dst_h,
  int resized_w,
  int resized_h,
  int pad_left,
  int pad_top,
  cudaStream_t stream)
{
  launch_yolo11_preprocess_impl(
    src_bgr,
    src_w,
    src_h,
    src_step,
    dst_chw_rgb,
    dst_w,
    dst_h,
    resized_w,
    resized_h,
    pad_left,
    pad_top,
    stream);
}

}  // namespace auto_buff
