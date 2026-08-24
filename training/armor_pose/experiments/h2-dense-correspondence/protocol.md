# H2 Protocol: Dense Planar Correspondence PnP

Status: locked before implementation results.

## Prediction

A same-frame network that predicts a foreground mask, canonical planar UV field, per-pixel uncertainty, and spatially covered 2-D--3-D samples will reduce exploratory-development P95 position and depth error by at least 30% versus raw YOLO IPPE without worsening any represented motion mode by more than 3%.

## Offline label construction

The existing nominal-PnP-equivalent target quadrilateral defines a homography from the fixed 135x55-mm nominal armor plane into the detector-conditioned ROI. It generates mask and UV labels offline. No new simulator field or simulator repository change is required.

## Online contract

Inference receives only the same-frame RGB ROI, raw detector corners/geometry, intrinsics, and ROI transform. It predicts mask, UV, uncertainty, weighted correspondences, and pose. Exact corners and labelled pose never enter `forward()` or exported artifacts.

## Confirmatory measurements

- Aggregate and per-mode P95 `position_mm`, `depth_abs_mm`, and `ray_mrad`.
- Mask IoU, UV error, correspondence coverage, effective point count, Hessian conditioning, invalid/fallback rate, and CUDA latency.
- Predeclared correspondence counts: 32, 64, 128. Compare stratified coverage sampling against confidence-only top-k.
