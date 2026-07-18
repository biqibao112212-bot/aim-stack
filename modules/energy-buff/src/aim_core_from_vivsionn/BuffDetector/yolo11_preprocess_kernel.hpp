#ifndef AUTO_BUFF__YOLO11_PREPROCESS_KERNEL_HPP
#define AUTO_BUFF__YOLO11_PREPROCESS_KERNEL_HPP

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace auto_buff
{

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
  cudaStream_t stream);

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
  cudaStream_t stream);

}  // namespace auto_buff

#endif  // AUTO_BUFF__YOLO11_PREPROCESS_KERNEL_HPP
