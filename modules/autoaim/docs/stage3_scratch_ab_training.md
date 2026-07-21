# Stage 3 scratch future-observation A/B training

Date: 2026-07-21

## Experiment contract

Both models were trained from the same random parameter state.  No v3 or v4
checkpoint was loaded.  They consumed the same shuffled batches, event
augmentation and dropout random states from the full 111,527-sample training
split.  The 36,297-sample validation split was used for checkpoint selection;
the test split was not accessed.

The direct observation head predicts four future PnP positions at arbitrary
query times.  An exact future physical target selects one geometry-only armor
permutation, and that same permutation is used by the masked observation loss
and evaluator.  Missing or ambiguous future frames and unobserved armor slots
do not update the position loss.

- Model A: masked direct-observation Huber loss only.
- Model B: the same observation loss plus `0.2 * physical_position_huber`.
- Both: random initialization, 64 channels, 30 epochs, batch 256, AdamW,
  initial learning rate `3e-4`, cosine decay, Huber beta `0.1 m`.
- Selection score: `P-O median + 0.25 * P-O P95`.

The previous residual NLL, residual addition, visibility BCE and joint
observation/physical permutation selection are not part of this experiment.

## Full validation result

Both accepted checkpoints are from epoch 29 and cover 222,848 usable exact
future-observation queries.

| model | P-O mean | P-O median | P-O P90 | P-O P95 | P-O P99 |
|---|---:|---:|---:|---:|---:|
| old v3 physical predictor | 0.490670 m | 0.223675 m | 0.882900 m | 1.345864 m | 3.239398 m |
| scratch A, observation only | **0.449615 m** | **0.198660 m** | **0.754586 m** | 1.090279 m | 3.289847 m |
| scratch B, +0.2 physical auxiliary | 0.450063 m | 0.201418 m | 0.800218 m | **1.069940 m** | 3.283804 m |

Relative to v3, A improves mean/median/P95 by 8.37%/11.18%/18.99%.
B improves them by 8.28%/9.95%/20.50%.  A is the median winner; B is the
P95 and predefined composite-score winner.  Both slightly worsen P99, so the
rarest observation outliers remain unresolved.

Model B's auxiliary physical head reaches P-G median/P95
`0.213581/0.736705 m`, worse than the dedicated v3 physical model
`0.175675/0.569594 m`.  It is a regularizer, not a replacement physical
predictor.  Model A's unused physical head is intentionally untrained.

## Acceptance

This experiment proves that direct future-observation learning from random
initialization is feasible and improves validation P-O without inheriting the
v3 checkpoint.  Model B is the current A/B winner under the predefined
median-plus-P95 score, while model A remains useful when median error is the
primary priority.

Neither checkpoint is an online release yet.  Test evaluation, multi-seed
confirmation, export and runtime integration remain separate acceptance
gates.

Artifacts are retained under:

`D:\仿真\models\engines\stage3-training\20260721-v5-scratch-ab-full-seed0-r3`
