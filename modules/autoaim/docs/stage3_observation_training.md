# Stage 3 future-observation training

> Superseded on 2026-07-21 by the controlled scratch A/B experiment in
> [stage3_scratch_ab_training.md](stage3_scratch_ab_training.md). The r5 residual fine-tune result remains
> reproducible evidence, but its negative learnability conclusion is no longer
> accepted: the residual scale and permutation contract were misconfigured.

Date: 2026-07-21  
Dataset: `stage3-dataset-v4-observation-20260721-r4`  
Training: full train split, validation split, seed 0, initialized from the
v3 physical checkpoint

## Implemented contract

The v4 shard keeps the original physical target and adds exact future
observation labels:

- `future_observation_position[8,4,3]`
- `future_observation_mask[8,4]`
- `future_observation_frame_available[8]`
- `future_observation_ambiguous[8]`
- two history quality channels (reprojection is zero-filled because the
  historical v1 recorder did not provide it; candidate fraction is retained)

Missing exact frames mask the observation branch completely. Exact zero-
candidate frames are explicit visibility negatives. A >4-candidate frame is
ambiguous and masks observation position/visibility losses.

The model has a physical position head, a future PnP residual head, and a
visibility head. Its observation output is
`O_hat = G_hat + e_hat`. The residual branch is zero-initialized and the
physical branch is initialized from the accepted v3 checkpoint. The first
five warm-up epochs train only the new observation heads; subsequent epochs
jointly train the heads with physical loss weight 2.0.

## Result

The accepted r5 run stopped after 8 epochs on validation early stopping. On
222,848 exact usable future queries:

| comparison | median | P95 |
|---|---:|---:|
| future observation `O-G` | 0.101900 m | 1.191637 m |
| new model `P-O` | 0.226535 m | 1.353787 m |
| old v3 model `P-O` | 0.223675 m | 1.345863 m |

The physical result is unchanged from v3: `P-G` is `0.175675/0.569594 m`.
Thus the first observation-target run is feasible and physically safe, but it
does not materially improve prediction of the future PnP output.

The visibility head is not ready for online use: validation accuracy is
70.715%, with predicted positive rate 12.065% versus true 27.852%.

## Interpretation

The v1 raw history contains no reprojection quality values. More importantly,
the matched PnP residual vector has near-zero global mean and large spread
(validation component standard deviations are approximately 1.31, 0.40 and
0.19 m). The future residual is therefore mostly a new-frame stochastic error
that cannot be inferred from past xyz/yaw history. A deterministic residual
head learns a small bias but cannot predict the random future outlier.

The observation-target route should therefore remain an experiment, not a
production replacement. Improving it requires information correlated with the
future frame error (image/detector quality or a causal uncertainty signal), or
an explicitly probabilistic output. The physical truth head remains the
stable prediction path; `P-O` stays a diagnostic/business metric until the
input contract supplies such information.

Artifacts:

- `models/engines/stage3-training/20260721-v4-observation-full-seed0-r5/stage3-observation-seed0-best.pt`
- `models/engines/stage3-training/20260721-v4-observation-full-seed0-r5/validation-observation-report.json`
