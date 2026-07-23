# Stage-3 moving-only route refinement (v15)

## Purpose

V14 established a reliable moving/non-moving split, but its remaining hard-route
errors are concentrated between translation and combined motion. V15 does not
retrain or replace any motion expert. It freezes the complete v14 epoch-53
system and adds one binary refinement model that is evaluated only after the
frozen move gate selects a moving target.

## Hierarchical route

1. The frozen v14 move probability selects moving versus non-moving.
2. A non-moving target retains the frozen v14 stationary-versus-rotation
   decision.
3. A moving target is classified as translation or combined by the new binary
   refinement model.
4. The selected frozen specialist produces velocity and/or yaw rate; the
   existing hard-rigid constant-twist decoder produces all future positions.

The v14 last checkpoint is retained as the safe fallback. Its registered SHA-256
is `3fdfb7d0f3ddf3062dc078060caf5d35913eaf8265fecbc66b2d3c9e217029bb`.

## Refinement input and objective

For each of the 32 causal history events, visible armor positions are converted
back to metres and their same-event visible-slot mean is removed. The remaining
relative rigid shape is divided by the fixed geometry radius and concatenated
with the existing cyclic slot encoding. This removes common translation while
retaining rotation-induced relative shape motion. It does not compute a temporal
finite difference and does not use truth motion state, future data, PnP noise,
motion class, or expert output as an inference input.

Only factor-qualified moving rows train the binary head. Translation is the
negative class and combined motion is the positive class. Their two BCE group
means are averaged, so their dataset counts cannot dominate each other. Future
truth is detached supervision and evaluation data only.

## Freeze and provenance contract

- Source: completed v14 run, last checkpoint at epoch 53.
- Frozen: all 2,132,628 v14 parameters, including all experts and the v14
  router.
- Trainable: only the relative-shape encoder and binary head, 184,065
  parameters.
- Validation: the full qualified r4 validation split; test remains sealed.
- Selection: reject q3 regressions beyond the registered tolerance, then
  maximize the worse of translation/combined recall, then macro recall and q3.
- Checkpoints: initial, best, last, epoch 20/40/final milestones.

The one-epoch smoke reproduced the registered v14 q3 P95 exactly, verified the
frozen-base hash after optimization, completed with finite loss and gradients,
and recorded `test_accessed=false`. The official run must start from a clean
commit in a new non-overwriting protected directory.
