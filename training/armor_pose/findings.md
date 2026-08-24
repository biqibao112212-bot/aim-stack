# Armor Pose Research Findings

## Research Question

Can same-frame probabilistic corners and dense planar correspondences reduce raw-YOLO PnP P95 position, depth, and ray error by at least 30%, with a 50% position/depth stretch target, without online truth?

## Current Understanding

The accepted V12/V13 corner-repair candidate improves sealed-test P95 position/depth/ray by 15.12%/15.29%/12.37%, but it remains a four-corner deterministic correction followed by production IPPE. A V17 truth-only oracle analysis showed that the correction direction has substantially more sample-dependent headroom; this is diagnostic evidence only and is not an inference design. The new work removes the deterministic four-coordinate bottleneck rather than learning an oracle control signal.

## Key Results

- A clean CUDA-only weighted planar PnP MVP now performs observable-only planar DLT initialization, deterministic multi-start LM, covariance whitening, top-two result retention, local Hessian covariance, and fail-closed validity. Synthetic translation recovery is below 0.1 mm in the locked test.
- The sparse probability head combines a calibrated in-ROI grid with a continuous Gaussian tail. Zero initialization preserves raw detector corners, including near ROI boundaries; all predicted 2-D covariances pass CUDA Cholesky checks.
- The dense branch produces nominal projected-support, canonical UV, uncertainty, edge distance, and stratified correspondences. Its first immediate pose-loss smoke was numerically invalid; the failure showed that support/UV curriculum and online-MAP consistency are prerequisites for EPro training.
- V19 has now been measured on the 387-row V17 exploratory validation. Sparse probability prediction improved paired median position by 24.8% but regressed paired P95 position by 149.9%; dense and direct fusion produced multi-metre tails. Neither checkpoint qualifies.
- A gate using only the candidate's observable GPU-PnP reprojection residual reduced raw-GPU P95 position from about 795.9 mm to about 665.8 mm (16.3%) and ray P95 from about 32.6 to 24.0 mrad. This confirms that uncertainty-aware selection is useful, but the current representation still lacks the 30--50% headroom requested.
- The dense global-corner head used global average pooling, which discards where evidence occurs in the ROI. V20 replaces it with a calibrated spatial decoder and removes the non-projective per-pixel UV warp that violated planar consistency.

## Patterns and Insights

- Exact corners and reference pose are offline loss/evaluation targets only.
- Online inputs are restricted to same-frame RGB ROI, raw detector corners, camera intrinsics, and the raw-corner ROI transform.
- Planar ambiguity must be represented as uncertainty or multiple pose modes, not hidden by a single forced coordinate.
- A pure 128x64 absolute heatmap cannot cover the tail: 42.9% of V17 exploratory samples have at least one target corner outside the ROI. The continuous tail is therefore part of H1, not an optional ablation.
- Release labels do not contain occlusion-tested visibility; the dense mask is named nominal projected support and must not be interpreted as visible foreground truth.

## Lessons and Constraints

- V15 and V18 sealed-test evidence must never influence future model, threshold, sample, or hyperparameter selection.
- The existing differentiable loss initializes local Gauss--Newton from the labelled pose; this is acceptable as a training diagnostic but is not a truth-free inference solver.
- Simulator source, SDK, Release, production PnP, tracker, predictor, and fire control remain unchanged.
- GPU requests fail closed; no silent CPU fallback is allowed for neural or differentiable-geometry compute.
- Dense EPro loss is disabled during the initial support/UV curriculum. A sample contributes a pose-distribution loss only when its online MAP objective is no worse than the labelled-pose objective within tolerance.

## Open Questions

- Can spatial-bin tail decoding preserve the sparse median gain without the planar-mode P95 jump?
- Does a strictly projective dense representation add stable evidence, or merely resample the same four-point homography?
- Can an observable utility gate reach the V19 truth-only multi-scale oracle ceiling without importing truth into inference?

## Optimization Trajectory

The frozen development evaluator reports raw/sparse/dense/fusion, fail-closed and explicit-raw-fallback policies, and per-session/per-mode P50/P95. V19 is a decisive negative result. V20 changes representation and initialization rather than tuning the failed checkpoints.
