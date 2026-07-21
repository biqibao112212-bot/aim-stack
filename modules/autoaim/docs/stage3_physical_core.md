# Stage-3 physical-core isolation and acceptance

## Purpose

This experiment answers one narrow question before any further PnP-target
training: can the constant-velocity rigid-body part predict exact future armor
positions to negligible error when its initial physical state is exact?

The answer is yes. The accepted physical operator is deterministic and has no
learnable parameters. A neural estimator may later infer its internal initial
conditions from PnP history, but it must not relearn or replace the propagation
equation.

## Reused data and leakage boundary

No new simulator capture was performed. The qualified v3 source dataset was
reused. A derived truth-history dataset was built from the existing exact
exposure truth stream:

- source: `stage3-dataset-v3-20260721-r1`;
- derived: `stage3-truth-history-v1-20260721-r5`;
- train/validation: 111,527 / 36,297 samples from 212 / 70 sessions;
- exact q0 label reproduction maximum error: 0 m;
- reconstructed history timestamp maximum error: 953 ns;
- test shards opened: no.

Every history position and physical state is expressed in the anchor tracker
frame. The derived dataset also marks each query as `rule` only when exact
linear velocity and yaw rate remain unchanged over the full interval from q0
to that query. The validation rule-query fraction is 97.2484%.

## Four progressively stricter checks

1. The old neural physical head was rejected as a physics proof because its
   q0 state reconstruction and future propagation errors were entangled.
2. A new direct anchored network passed fixed-32-sample memorization
   (q0 P95 about 6 mm and 0.5 s motion P95 about 4 mm), proving that the new
   loss and output contract are learnable.
3. A parameter-free two-frame position derivative achieved q0 P95 0.080 mm,
   but rule-query motion P95 grew to 8.6/26.9/50.9 mm at nominal
   0.1/0.2/0.5 s. Numerical differentiation and a wrong rotation-center
   assumption were therefore not accepted.
4. The accepted operator uses exact target center, exact translational
   velocity, exact yaw rate, and q0 armor geometry. It translates the true
   target center and rotates every q0 armor offset around that center. Its
   public output remains only `[batch, query, 4, 3]` future positions.

The important geometric repair is the rotation center. The four armor points
are not exactly centered on their arithmetic centroid. Rotating around that
centroid creates a fictitious translation and a centimetre-scale long-horizon
error even when velocity and yaw rate are exact.

## Full validation result

All values below are position error in metres on `rule` queries.

| Query | Median motion | P95 motion | Median absolute P-G | P95 absolute P-G |
| --- | ---: | ---: | ---: | ---: |
| q0 / 0.0 s | 0 | 0 | 0 | 1.86e-9 |
| q1 / about 0.1 s | 6.70e-7 | 4.45e-6 | 6.70e-7 | 4.45e-6 |
| q2 / about 0.2 s | 8.38e-7 | 8.19e-6 | 8.38e-7 | 8.19e-6 |
| q3 / about 0.5 s | 1.31e-6 | 1.76e-5 | 1.31e-6 | 1.76e-5 |

The 0.5 s rule-query motion P95 by motion mode is 0 for stationary,
4.74e-6 m for spin, 1.43e-5 m for linear, and 3.33e-5 m for combined linear
and spin. By distance it is 1.66e-5 m below 3 m, 1.41e-5 m from 3 to 5 m,
and 2.07e-5 m at 5 m or farther.

All predefined acceptance gates are below 1 mm and pass. Future-event queries
remain separate: a velocity or yaw-rate change after q0 is not causally
predictable under the constant-state assumption and must not be mixed into the
physics gate.

## Artifacts and next interface

The protected accepted artifact is:

`D:\仿真\models\engines\stage3-training\20260721-v7-physical-exact-state-core-r5`

The next learned component is an observation adapter:

`PnP history -> internal q0 center/armor geometry/motion estimate -> frozen physical core -> future positions`

Velocity and yaw rate are internal implementation details. They are not added
to the external predictor API. The observation adapter must be evaluated
against both the exact-state core and the already measured PnP input ceiling;
future-observation residual learning comes only after this interface passes.
