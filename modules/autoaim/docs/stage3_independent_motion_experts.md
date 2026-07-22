# Stage 3 v13: independent rotation and combined-motion experts

## Purpose

V12 showed that the causal TCN can learn q0 pose and translation, but its
shared encoder and two binary gates do not cleanly separate pure rotation from
combined translation-and-rotation.  V13 keeps the accepted rigid-body decoder
and replaces that shared learned state with physically separate specialists.

This run is a train/validation experiment.  The test split, export, TensorRT,
tracker, MPC, fire control and online integration remain sealed.

## Inference contract

The predictor receives only the latest 32 fixed-slot observations, masks, real
relative timestamps and query `tau`.  It never receives motion class, route
truth, center, velocity, yaw rate, rule-query flags or future truth.

The integrated system contains five independent causal encoders:

1. Frozen pose/q0 foundation from the v12 augmented best checkpoint.
2. Frozen translation foundation from the v12 original best checkpoint.
3. Trainable pure-rotation expert, supervised only on rotation without
   translation.
4. Trainable combined-motion expert, jointly predicting its own velocity and
   yaw rate only on simultaneous translation-and-rotation.
5. Trainable four-class router for stationary, translation, rotation and
   combined motion.

The combined expert is not a sum of translation and rotation trajectories.  A
single combined encoder jointly produces `(v, omega)`, which is then propagated
by the same frozen constant-twist four-armor decoder.  Hard routing is:

| Route | Velocity | Yaw rate |
| --- | --- | --- |
| stationary | zero | zero |
| translation | frozen translation expert | zero |
| rotation | zero | independent rotation expert |
| combined | independent combined expert | independent combined expert |

## Frozen sources

- q0 pose: `E_factorized_rot_aug`, epoch 283, SHA-256
  `e3b93c708d174e476af2badc99620a9904dfb669da6545a9610a75507b896b49`.
- translation: `E_factorized_original`, epoch 266, SHA-256
  `5437d6c48fb57805a4e2a02a8814a7d5681c1821a6137c78d02c7326b954aafa`.

The formal launcher verifies the completed v12 manifest, source commit,
dataset hash, checkpoint roles/epochs/hashes, clean provenance and test seal.
Frozen modules remain `eval()` even while the new branches train, and are
excluded from the optimizer.  Their state hashes are checked after epoch 1 and
again at completion.

## Truth-derived factor labels

Dataset `motion_class` is retained for the historical report but is not a
valid router target: some declared moving sessions contain stationary eligible
windows.  V13 derives factors from the accepted future constant-motion labels:

- moving negative/positive: speed at most `0.01 m/s` / at least `0.10 m/s`;
- rotating negative/positive: absolute yaw rate at most `0.05 rad/s` / at
  least `0.20 rad/s`;
- samples in either dead band are excluded from router and expert objectives.

On the full qualified r4 data the eligible factor counts are:

| Split | stationary | translation | rotation | combined |
| --- | ---: | ---: | ---: | ---: |
| train | 8,604 | 5,520 | 8,007 | 8,865 |
| validation | 2,698 | 1,971 | 2,815 | 2,996 |

Router cross entropy is averaged over the four present factor groups.  The
pure-rotation omega loss is masked to factor 2.  The combined velocity and
omega losses are masked to factor 3.  Expert state errors are expressed in
meter-equivalent scale over the 0.5 s reference horizon.

## Diagnostics and checkpoint selection

Validation reports both the final hard-routed trajectory and each raw expert
before routing.  The router report includes the four-by-four confusion matrix,
per-class precision/recall/FPR, macro recall, hard-route moving/rotating
recall/FPR and a factor-versus-dataset-class cross-tab.  Raw reports include:

- frozen translation velocity error, direction and speed ratio;
- pure-rotation omega error, sign accuracy and magnitude ratio;
- combined velocity and omega errors, direction, speed ratio and sign.

Best-checkpoint selection is lexicographic: worst rotation/combined q3 P95,
worst dynamic-factor q3 P95, overall eligible q3 P95, router macro-recall
penalty, q0 P95, then worst rotation/combined q3 median.  Performance goals do
not stop the 300-epoch run; they are evaluated after completion.

## Formal run

The registered output is non-overwriting:

`D:\仿真\models\engines\stage3-training\20260722-v13-independent-motion-experts-300ep-seed0-r1`

Runtime stdout/stderr and launch metadata use:

`D:\仿真\runtime\stage3-training-20260722-v13-independent-motion-experts-300ep-seed0-r1`

The run uses seed 0, all 32,904 train and 11,189 validation samples, 300
epochs, batch size 128, AdamW `3e-4`, cosine decay, bfloat16, no clipping and no
early stopping.  It retains initial, best, last and epochs
20/50/100/150/200/250/300 checkpoints.  `test_accessed` must remain false.
