# Armor Pose Research Findings

## Research Question

Can same-frame probabilistic corners and dense planar correspondences reduce raw-YOLO PnP P95 position, depth, and ray error by at least 30%, with a 50% position/depth stretch target, without online truth?

## Current Understanding

The accepted V12/V13 corner-repair candidate improves sealed-test P95 position/depth/ray by 15.12%/15.29%/12.37%, but it remains a four-corner deterministic correction followed by production IPPE. A V17 truth-only oracle analysis showed that the correction direction has substantially more sample-dependent headroom; this is diagnostic evidence only and is not an inference design. The new work removes the deterministic four-coordinate bottleneck rather than learning an oracle control signal.

## Key Results

No V19+ experiment has completed yet.

## Patterns and Insights

- Exact corners and reference pose are offline loss/evaluation targets only.
- Online inputs are restricted to same-frame RGB ROI, raw detector corners, camera intrinsics, and the raw-corner ROI transform.
- Planar ambiguity must be represented as uncertainty or multiple pose modes, not hidden by a single forced coordinate.

## Lessons and Constraints

- V15 and V18 sealed-test evidence must never influence future model, threshold, sample, or hyperparameter selection.
- The existing differentiable loss initializes local Gauss--Newton from the labelled pose; this is acceptable as a training diagnostic but is not a truth-free inference solver.
- Simulator source, SDK, Release, production PnP, tracker, predictor, and fire control remain unchanged.
- GPU requests fail closed; no silent CPU fallback is allowed for neural or differentiable-geometry compute.

## Open Questions

- How much gain comes from calibrated sparse uncertainty alone?
- Does dense planar surface supervision add information beyond a homography implied by four labels?
- Is correspondence-level fusion better than product-of-experts pose fusion for planar ambiguity?

## Optimization Trajectory

Baseline is raw production-parity YOLO IPPE. No V19+ run has been measured.
