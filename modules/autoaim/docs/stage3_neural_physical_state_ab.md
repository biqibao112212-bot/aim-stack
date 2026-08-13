# Stage 3 fixed-slot neural physical state A/B

## Superseded r2 diagnosis

The `stage3-causal-physical-v1-20260721-r2` capacity run is retained as failure
evidence, not as an expressivity verdict. It certified constant motion over
only the last four events while the model consumed 32, initialized the final
head to exactly zero, allowed the untrained epoch-zero prior to win checkpoint
selection, and used a decoded-position loss whose motion gradient was much
smaller than its q0 gradient. Re-running that contract is forbidden.

Test, PnP, export and online integration remain sealed. The historical
analytic least-squares physical core remains a retained upper-bound baseline
and is not part of either neural forward path.

## Active full-training contract

The input contains the most recent 32 real-timestamp events for four persistent
`relative_slot` histories. A missing plate remains masked in its own slot. The
model does not enumerate candidate permutations or implement armor-switch
compensation. Reacquisition starts a new causal segment in the dataset contract.
Every one of those 32 events must be certified as belonging to the same
constant-velocity and constant-yaw-rate interval through t0. The trainer
rejects a dataset whose manifest does not prove this.

The qualified non-overwriting derivative is
`stage3-causal-physical-v1-20260722-r4` (manifest SHA-256
`8121dc8096952052ca9f9bfe3f5ed951c103a05a1ef7be4d65e2b40c731e113e`).
It contains 32,904 train and 11,189 validation samples, records four short
sessions that became zero-sample under the 32-event rule, and keeps test
unopened.

Before creating a run directory, training rechecks every row's 32 complete
events, strictly increasing timestamps and constant-twist fit. It also requires
at least 85% q0--q3 supervision coverage overall and in every present motion
class. r4 passes at 94.20% train and 93.66% validation overall; its lowest
class coverage is 89.51%. Checkpoint selection uses this same eligible subset.

Allowed predictor inputs are normalized clean armor xyz, cyclic slot features,
event/candidate masks, real event time and query `tau`. Exact center, velocity,
yaw, yaw rate, motion class, rule-query flags, future truth and all PnP values
are forbidden forward inputs.

## Frozen A/B definition

- A encodes the four histories, estimates one
  `center0/velocity/phase0/omega` state, and applies the frozen constant-twist
  rigid decoder at every query.
- B uses the same history encoder class, receives each query `tau`, and predicts
  `center(tau)/phase(tau)` directly. B has no shared velocity or omega output.
- Both use the same fixed geometry decoder and preserve all four armor pairwise
  distances at every query.
- Both decode q0--q3 positions and pass them through the same training-only
  physical state extractor. The loss is a sum of meter-equivalent Smooth-L1
  terms for center0, velocity over a 0.5 s reference horizon, phase at the
  geometry radius, yaw rate over the same horizon, and constant-twist temporal
  consistency. Future truth constructs labels only and never enters forward.
- Neither arm uses least-squares pose fitting, finite-difference velocity,
  history reconstruction, or an arm-specific loss.

The final prediction heads use a small random initialization so the encoder
receives gradient on the first update. Gradient clipping and early stopping
are disabled by default for the full run; epoch zero is saved only as an
initial checkpoint and cannot be selected as the trained best checkpoint.

Velocity and yaw-rate diagnostics are reparsed identically from each arm's
final decoded q0--q3 positions. They are evaluation outputs, not predictor
inputs or arm-specific supervision.

## Held-out pilot

The hash-bound selection is
[causal_physical_state_ab_pilot_v1.json](../../../training/stage3/selections/causal_physical_state_ab_pilot_v1.json). It contains
16 train sessions and 8 disjoint validation sessions, covers stationary,
linear, spin and combined motion, and includes positive/negative rotation,
low/mid/high speed and yaw rate, and near/mid/far distance. Its test list is
empty.

Checkpoint selection is lexicographic: worst q1--q3 rule-motion P95, q0 P95,
worst q1--q3 rule absolute P95, then worst motion median. The pre-registered
clean-physics target is q0 P95 at most 5 mm, q3 motion median at most 3 mm,
q3 motion P95 at most 10 mm, every dynamic-class q3 P95 at most 15 mm, and
rotating-sign accuracy at least 99 percent. Failure stops the experiment for
joint review; it does not authorize an unreviewed architecture or loss change.
