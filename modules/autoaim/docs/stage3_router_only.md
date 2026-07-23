# Stage 3 v14: router-only factor-aware fine-tuning

## Purpose

The completed v13 run learned useful frozen translation, pure-rotation and
joint-combined specialists, but the final hard router remained the integrated
bottleneck.  At the best epoch 297 checkpoint, translation and combined recall
were 91.68% and 85.61%; 423 of 2,996 combined validation samples were routed to
translation.  Training router CE was already 0.00165 while validation macro
recall was 94.32%, so repeating the same joint objective would continue an
already-generalized gap rather than address it.

V14 starts from the registered v13 best checkpoint and trains only its existing
`router_encoder` and `router_head`.  Pose/q0, translation, rotation, combined
motion and the constant-twist rigid decoder are frozen and hash-verified after
every epoch and at completion.  The route order and hard inference behavior
remain stationary, translation, rotation and combined.

## Factor-aware objective

The four logits retain the v13 interface.  The training objective is the sum of
three group-balanced, interpretable classification losses:

1. Four-class cross entropy.
2. Moving-factor BCE using
   `P(move) = P(translation) + P(combined)`.
3. Rotating-factor BCE using
   `P(rotate) = P(rotation) + P(combined)`.

The accepted v13 truth-derived thresholds are unchanged: speed at most
0.01 m/s or at least 0.10 m/s, and absolute yaw rate at most 0.05 rad/s or at
least 0.20 rad/s.  Dead-band samples are excluded.  Future truth constructs
detached training/evaluation labels only and is not a predictor input.

Training applies the qualified v12 planar rigid transform to 75% of samples:
one yaw rotation around a center recovered from the latest causal history and
an xy translation within +/-0.25 m.  History and future labels are transformed
together, validation remains untouched, and no PnP noise is added.

## Training and acceptance

The maximum budget is 120 epochs with batch size 256, router-encoder/head
learning rates 5e-5/2e-4, five warm-up epochs, cosine decay, label smoothing
0.02 and patience 20 after epoch 30.  Checkpoint selection first refuses q3
regression beyond 2 mm, then improves the minimum translation/combined recall,
rotate FPR, macro recall and q3 tails.  Epoch zero remains a valid safe fallback.

The registered goals are translation and combined recall at least 97%, macro
recall at least 98%, rotate FPR at most 2%, and no regression from the v13 best
overall/factor q3 P95 values.  Goals are acceptance criteria, not claims.

Formal protected output:

`D:\仿真\models\engines\stage3-training\20260723-v14-router-only-120ep-seed0-r1`

Runtime evidence:

`D:\仿真\runtime\stage3-training-20260723-v14-router-only-120ep-seed0-r1`

The test split, export, TensorRT and online integration remain sealed.
