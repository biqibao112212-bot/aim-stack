# Armor Pose Research Log

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-24 | bootstrap | Locked a truth-free online contract and three hypotheses: probabilistic sparse corners, dense planar correspondences, and calibrated fusion. Existing V15/V18 sealed tests remain forbidden for selection. All neural and differentiable-geometry compute is CUDA fail-closed. |
| 2 | 2026-08-24 | inner-loop | V19 implementation smoke: clean GPU weighted planar DLT + deterministic multi-start LM recovered a synthetic 4 m pose within 0.1 mm; covariance inflation correctly downweighted a corrupted corner. Nine contract/CUDA tests passed. |
| 3 | 2026-08-24 | inner-loop | H1 sparse grid+continuous-tail model completed a 16-sample CUDA Laplace-EPro smoke with finite forward/backward and no test access. This is an implementation check, not an accuracy result. |
| 4 | 2026-08-24 | inner-loop | H2 immediate pose-loss smoke exposed an invalid Laplace normalizer when random UV correspondences had not produced a trustworthy MAP mode. Added a solver-consistency gate and bounded Hessian log determinant; the repeated smoke stayed finite. Dense curriculum remains mandatory. |
| 5 | 2026-08-24 | negative-result | Full-session V19 sparse training selected epoch 10. On V17 exploratory validation it reached 77.0% candidate validity and improved paired median position by 24.8%, but regressed paired P95 position by 149.9%. It is disqualified as an unconditional candidate. |
| 6 | 2026-08-24 | negative-result | Full-session V19 dense training selected epoch 11. Candidate validity was 30.8% and P95 position exceeded 4 m. Direct sparse+dense fusion also failed. These checkpoints are retained as negative evidence and are not deployment candidates. |
| 7 | 2026-08-24 | diagnosis | V17 truth-only stratification showed sparse continuous-tail behaviour helped the target-outside-ROI subset while the in-ROI grid caused tail damage. A truth-free candidate reprojection-residual gate recovered about 16% P95 position improvement, still below the 30% gate. |
| 8 | 2026-08-24 | outer-loop | Pre-registered V20: spatial-bin rather than global-average corner decoding, strict homography-constrained dense UV with no local warp, observable raw-mode initialization for dense GPU PnP, and a separately trained observable risk gate. V15/V18 remain sealed and forbidden. |
