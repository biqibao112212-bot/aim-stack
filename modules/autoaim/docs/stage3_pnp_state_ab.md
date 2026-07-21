# Stage 3 PnP-history state A/B

## Question

This experiment asks whether a shared explicit rigid-motion state helps when
the only inference input is the latest variable-rate PnP observation window.
It does not evaluate PnP correction or online armor identity, and test remains
sealed.

Both arms consume the same unordered candidate sets, masks, real event times,
quality features and query horizons. Both produce the same fixed four-armor
geometry and use the same position/state supervision.

## What the two arms do

Arm A encodes the complete causal window once and predicts
`center0`, `velocity`, `yaw0` and `omega`. A frozen FP32 physical layer then
evaluates

```text
center(tau) = center0 + velocity * tau
yaw(tau)    = yaw0 + omega * tau
armor(tau)  = center(tau) + Rz(yaw(tau)) * geometry
```

Arm B uses the same encoder, but combines the encoded window with every `tau`
and predicts that query's center/yaw independently. It has no shared velocity
or yaw-rate state and no hard constant-twist propagation.

Neither arm has a persistent recursive state. Every new exposure forms a new
latest-N window and recomputes its output. A's state is therefore a learned
summary of the current window, not an EKF predict/update state. No finite
difference, truth state or previous network state is supplied at inference.

During training only, exact future positions are converted into physical
center, displacement, velocity, relative rotation and yaw-rate labels. The
same state extractor is applied again to each arm's final decoded positions,
so A has no latent-only shortcut. This is ordinary supervised learning: labels
provide gradients during training but are absent from `forward` and export.

The real geometry is not C4-symmetric. Its minimum unordered-set distance to a
non-trivial 90-degree rotation is 15.89 mm, above the enforced 5 mm gate. The
objective therefore supervises full relative rotation, not a modulo-90-degree
phase. Absolute truth-slot phase is not a predictor input or auxiliary target.

## Evidence

The first held-out pilot used only indirect unordered-set position loss. Both
arms collapsed toward near-zero translation; A did not improve any dynamic
class. This candidate was rejected before full training.

V2 added equal decoded-trajectory state supervision and excluded truth windows
that were not one constant twist across every query within 1 mm and 1 mrad.
The dynamic-only held-out micro run still failed: best validation velocity
error median/P95 was 0.834/2.948 m/s for A and 0.948/2.821 m/s for B. Replaying
the checkpoints on their train sessions was also poor, so this run was not
accepted.

The bounded capacity gate then reused one 561-sample combined-motion train
session for diagnostic validation. It completed 160 epochs from clean commit
`a0e4f61`, kept test sealed, and correctly recorded
`diagnostic_only=true` and `qualified_training_candidate=false`.

| metric | A explicit state | B independent query pose |
| --- | ---: | ---: |
| best epoch | 151 | 156 |
| q0 absolute median / P95 | 51.20 / 128.33 mm | 67.14 / 137.49 mm |
| 0.5 s absolute median / P95 | 54.83 / 128.68 mm | 88.12 / 145.24 mm |
| 0.5 s motion median / P95 | 9.48 / 25.58 mm | 15.86 / 41.03 mm |
| velocity error median / P95 | 0.0136 / 0.0427 m/s | 0.0294 / 0.0796 m/s |
| yaw-rate error median / P95 | 0.0375 / 0.1184 rad/s | 0.0711 / 0.2250 rad/s |
| center constant-twist P95 | 0.00048 mm | 5.25 mm |
| yaw constant-twist P95 | 0.00000021 rad | 0.0789 rad |

The capacity result is not a generalization claim. It proves that both paths
can learn the same-session motion and that the hard shared state is more
sample/optimization efficient and physically consistent. The failed held-out
run identifies cross-session representation/generalization as the next
bottleneck.

## Next gate

Do not start a full run from the current encoder. The next encoder should make
relative temporal evidence explicit without analytically computing deployment
velocity: encode each frame relative to the latest valid frame, include real
time deltas and relative phase evidence, and learn a robust temporal pooling
over the complete variable-rate window. A and B must retain the same encoder,
decoded-position state supervision and frozen geometry.

The next held-out pilot should use substantially more train sessions per speed,
direction, yaw-rate sign/range, distance and visibility condition. It must
again demonstrate non-zero velocity/yaw-rate learning on unseen sessions before
any full training, test access, export or online integration.
