# Armor Pose Research Log

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-24 | bootstrap | Locked a truth-free online contract and three hypotheses: probabilistic sparse corners, dense planar correspondences, and calibrated fusion. Existing V15/V18 sealed tests remain forbidden for selection. All neural and differentiable-geometry compute is CUDA fail-closed. |
| 2 | 2026-08-24 | inner-loop | V19 implementation smoke: clean GPU weighted planar DLT + deterministic multi-start LM recovered a synthetic 4 m pose within 0.1 mm; covariance inflation correctly downweighted a corrupted corner. Nine contract/CUDA tests passed. |
| 3 | 2026-08-24 | inner-loop | H1 sparse grid+continuous-tail model completed a 16-sample CUDA Laplace-EPro smoke with finite forward/backward and no test access. This is an implementation check, not an accuracy result. |
| 4 | 2026-08-24 | inner-loop | H2 immediate pose-loss smoke exposed an invalid Laplace normalizer when random UV correspondences had not produced a trustworthy MAP mode. Added a solver-consistency gate and bounded Hessian log determinant; the repeated smoke stayed finite. Dense curriculum remains mandatory. |
