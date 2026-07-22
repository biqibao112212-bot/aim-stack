# Stage 3 factorized motion experts

## Why this experiment exists

The completed v11 shared-state model learned q0 pose and rotation more readily
than translation. On the qualified r4 validation split, its best A checkpoint
had q0 P95 0.1108 m and q3 motion P95 1.0163 m. Linear and combined-motion
median speed ratios were only 0.59 and 0.70. Because 53.6% of eligible training
windows have zero translational velocity, adding epochs to the same aggregate
state objective cannot distinguish insufficient optimization from attraction
to the majority static solution.

This experiment changes that causal mechanism rather than merely increasing
the training budget.

## Frozen v12 architecture

One 32-event fixed-slot causal TCN feeds three heads:

- q0 head: center0 and normalized planar phase;
- translation expert: velocity vector and a moving logit;
- rotation expert: yaw rate and a rotating logit.

At inference, logits are thresholded at 0.5. Closed experts emit exactly zero.
The resulting state is propagated by the frozen constant-twist rigid decoder.
No exact state, motion class, gate label, future truth, finite difference, or
analytic fit is an inference input.

## Six-term training objective

The group-mean objective is:

1. q0 center Huber;
2. q0 phase Huber in geometry-radius metres;
3. moving gate balanced BCE, weight 0.10;
4. rotating gate balanced BCE, weight 0.10;
5. velocity expert Huber on moving-positive samples only, weight 1.0;
6. yaw-rate expert Huber on rotating-positive samples only, weight 1.0.

Huber beta is 5 mm. State errors use a 0.5 s reference horizon. Moving labels
are negative at at most 0.01 m/s and positive from 0.10 m/s. Rotating labels are
negative at at most 0.05 rad/s and positive from 0.20 rad/s. Dead-band samples
do not train the corresponding gate. Static samples do not regress velocity to
zero, and non-rotating samples do not regress yaw rate to zero. This prevents
the majority static group from silencing a positive expert.

## Paired 300-epoch experiment

Both arms share exact initialization, batches, dropout RNG, AdamW optimizer,
cosine schedule, BF16 policy, and validation data.

- `E_factorized_original`: qualified r4 data without spatial augmentation.
- `E_factorized_rot_aug`: the same sample rotated uniformly around the center
  recovered from its latest causal history event and translated in xy by at
  most 0.25 m. History and future labels are transformed together; validation
  is never augmented. Future truth never determines a forward input.

The pair isolates whether coordinate diversity improves generalization. It is
not a clean causal comparison against the historical v11 architecture because
v11 used a different objective.

## Checkpoint selection and acceptance

Selection is lexicographic: worst dynamic-class q3 motion P95, overall eligible
q3 motion P95, eligible q0 P95, then worst dynamic-class q3 median. It never
selects on aggregate loss alone. Initial, best, last, and epochs 20, 50, 100,
150, 200, 250, and 300 are retained.

The registered held-out gates are: q0 P95 no worse than 0.111 m, q3 motion P95
at most 0.864 m, moving speed ratio within [0.85, 1.15], rotation sign accuracy
at least 99%, gate positive recall at least 98%, and gate negative false-open
rate at most 2%. Test, export, and online integration remain sealed.
