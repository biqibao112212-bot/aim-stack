# Armor Pose Research Findings

## Research Question

Can same-frame probabilistic corners and dense planar correspondences reduce raw-YOLO PnP P95 position, depth, and ray error by at least 30%, with a 50% position/depth stretch target, without online truth?

## Current Understanding

The accepted V12/V13 corner-repair candidate improves sealed-test P95 position/depth/ray by 15.12%/15.29%/12.37%, but it remains a four-corner deterministic correction followed by production IPPE. A V17 truth-only oracle analysis showed that the correction direction has substantially more sample-dependent headroom; this is diagnostic evidence only and is not an inference design. The new work removes the deterministic four-coordinate bottleneck rather than learning an oracle control signal.

## Key Results

- A clean CUDA-only weighted planar PnP MVP now performs observable-only planar DLT initialization, deterministic multi-start LM, covariance whitening, top-two result retention, local Hessian covariance, and fail-closed validity. Synthetic translation recovery is below 0.1 mm in the locked test.
- The sparse probability head combines a calibrated in-ROI grid with a continuous Gaussian tail. Zero initialization preserves raw detector corners, including near ROI boundaries; all predicted 2-D covariances pass CUDA Cholesky checks.
- The dense branch produces nominal projected-support, canonical UV, uncertainty, edge distance, and stratified correspondences. Its first immediate pose-loss smoke was numerically invalid; the failure showed that support/UV curriculum and online-MAP consistency are prerequisites for EPro training.
- These are implementation/smoke findings only. No V19 candidate has yet been measured against the pre-registered P95 gates.

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

- How much gain comes from calibrated sparse uncertainty alone?
- Does dense planar surface supervision add information beyond a homography implied by four labels?
- Is correspondence-level fusion better than product-of-experts pose fusion for planar ambiguity?

## Optimization Trajectory

Baseline is raw production-parity YOLO IPPE. V19 smoke runs validate implementation only; the optimization trajectory remains unmeasured until a development P95 evaluator is frozen.
