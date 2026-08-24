# H1 Protocol: Probabilistic Four-Corner PnP

Status: locked before implementation results.

## Prediction

A same-frame network that predicts four spatial heatmaps and calibrated 2-D covariances, trained through a truth-free-inference GPU probabilistic PnP layer, will reduce exploratory-development P95 position and depth error by at least 25% versus raw YOLO IPPE without worsening any represented motion mode by more than 3%.

## Inputs and forbidden fields

Inputs are RGB `[3,64,128]`, raw detector corners `[4,2]`, intrinsics `[4]`, detector geometry `[15]`, and the raw ROI inverse transform `[3,3]`. Exact corners, labelled pose, range, motion, identity, temporal history, tracker state, and future fields are forbidden from `forward()` and inference artifacts.

## Confirmatory measurements

- Aggregate and per-mode P95 `position_mm`, `depth_abs_mm`, and `ray_mrad`.
- Heatmap NLL, covariance coverage at 50/80/90/95%, invalid-solve rate, fallback rate, and CUDA batch-1 latency.
- Three predeclared seeds. Exploratory development uses non-test packs only; a later fresh session-disjoint validation is required before any sealed test.
