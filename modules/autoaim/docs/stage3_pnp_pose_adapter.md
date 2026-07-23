# Stage 3 Module A: PnP history to current physical pose

## Boundary

Module A restores the current physical rigid pose. It does not predict future
motion:

```text
unordered PnP observations up to t
  -> Module A(center_t, canonical phase_t)
  -> frozen four-slot geometry
  -> append one clean fixed-slot event to an online cache
  -> after 32 qualified events, frozen v15 predicts future motion
```

Each online result may use only observations whose exposure time is no later
than its own anchor. A later window must not rewrite earlier cached poses. A
target/producer/geometry discontinuity or invalid time gap resets the cache.
If the ego frame moves, cached poses require the exposure-matched SE(3)
transform into the current anchor frame before v15 is called.

The modules can be frozen and evaluated separately, but their errors are not
statistically independent: residual, temporally correlated adapter jitter is
still an input to the nonlinear v15 router and specialists.

## V16 training contract

V16 uses the qualified v4 observation train/validation splits. Test remains
sealed. A row is admitted only when its latest valid PnP event is at `q0`
within 1 microsecond. This removes stale-observation rows which would mix
short-horizon extrapolation into observation recovery.

Predictor inputs are:

- unordered normalized PnP xyz and PnP sin/cos yaw;
- candidate/event masks;
- causal real event times.

The encoder receives per-frame candidate shape about the visible centroid,
the frame-centroid displacement from the latest visible centroid, absolute PnP
yaw, mask-derived count, absolute anchor summary, and real time/delta-time.
It is invariant to candidate row order. The head predicts only a correction to
the latest PnP centroid and a unit-complex phase correction. A frozen FP32
geometry decoder always produces the same four canonical `relative_slot`s.

The old reprojection and candidate-fraction channels are deliberately ignored:
the v4 reprojection values are zero-filled and the candidate-fraction join is
not qualified. Motion class is used only for validation strata. Future PnP,
future physical q1--q7, exact velocity/yaw rate, tracked identity, and test are
forbidden inputs.

The complete objective is intentionally small:

```text
L = SmoothL1(predicted q0 center, true q0 center)
  + SmoothL1(radius * predicted unit phase,
             radius * true unit phase)
```

The current center/phase labels are derived from the same-time q0 fixed-slot
truth and the immutable geometry template. No v15 gradient or future-trajectory
loss is used.

## Checkpoint selection and reporting

Checkpoints are selected lexicographically by:

1. fixed-slot q0 position P95;
2. center P95;
3. full canonical phase P95;
4. fixed-slot q0 position P99.

Validation also reports unordered-set error only as an auxiliary diagnostic,
modulo-90-degree phase error, quarter-turn and 180-degree alias fractions,
the raw latest-PnP nearest-truth baseline, rigid residual, and strata by motion
class, distance, and latest visible count. A low unordered-set error cannot
override a failed fixed-slot/full-phase gate.
The first formal run remains `qualified_training_candidate=false` until these
registered metrics and the later causal rolling-cache replay are manually
accepted; successful optimization alone is not a deployment qualification.

## Staged acceptance

V16 A0 is the deterministic current-pose recovery stage and can be trained
from v4 immediately. A later A1 temporal-stability stage requires a
train/validation-only continuous-anchor sidecar so that 4--8 causal prefixes
receive their own same-time labels. It will supervise center/phase increments
without supplying velocity differences as inputs. Real corner/reprojection
quality and calibrated uncertainty require a separately qualified derivative;
v4 cannot support those claims.

After A0/A1 are independently accepted, cascade validation must compare the
same validation cases in paired form:

1. clean fixed-slot history -> frozen v15;
2. PnP history -> Module A current-pose error;
3. causally rolled Module A 32-event cache -> frozen v15;
4. paired difference between cases 3 and 1.

Export, test access, and online fire-control integration remain separate gates.
