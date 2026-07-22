# Stage 3 fixed-slot neural physical state A/B

## Frozen scope

This experiment reuses the qualified
`stage3-causal-physical-v1-20260721-r2` train/validation dataset. Test, PnP,
export and online integration remain sealed. The historical analytic
least-squares physical core remains a retained upper-bound baseline, but it is
not part of either neural forward path in this experiment.

The input contains the most recent 32 real-timestamp events for four persistent
`relative_slot` histories. A missing plate remains masked in its own slot. The
model does not enumerate candidate permutations or implement armor-switch
compensation. Reacquisition starts a new causal segment in the dataset contract.

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
- Both use the identical fixed-slot decoded-position objective:
  `2*q0 + absolute + 2*motion_delta` with 5 mm Smooth-L1 transition.
- Neither arm uses least-squares pose fitting, finite-difference velocity,
  arm-specific state labels, history reconstruction, or an arm-specific loss.

Velocity and yaw-rate diagnostics are reparsed identically from each arm's
final decoded q0--q3 positions. They are evaluation outputs, not predictor
inputs or arm-specific supervision.

## Held-out pilot

The hash-bound selection is
`training/stage3/selections/causal_physical_state_ab_pilot_v1.json`. It contains
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
