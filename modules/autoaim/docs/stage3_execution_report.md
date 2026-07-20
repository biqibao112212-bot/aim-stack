# Stage 3 execution report

Date: 2026-07-20

## Delivered implementation

- Added a dedicated `stage3-observation-v1` pre-tracker JSONL stream. It is
  emitted immediately after `solveArmors()` and before `trackerUpdate()` and
  contains the complete unordered solved-armor set for the exposure frame.
- Added a separate `stage3-truth-v1` exact-exposure JSONL stream. It records
  the full simulator truth batch, exposure chassis/gimbal/camera poses,
  selected target, and a fixed-geometry fingerprint. Missing exact truth,
  target loss, geometry drift, queue overflow, or I/O failure is fail-closed.
- Added the strict target-3 Scene Control stage-three command path and a
  Windows session runner using the locked Daedalus 1.0.1 / SDK 1.0.0 release.
  Each invocation writes an isolated `run-*` directory, so retries never
  append unrelated epochs to one JSONL pair.
- Added the Windows `yolov8` schema/join/tensorization pipeline, causal TCN,
  permutation-invariant loss, static and constant-twist baselines, evaluation,
  and dynamic ONNX export/parity checks.

## Validation completed

- `D:\Anaconda\envs\yolov8\python.exe -m pytest -q training/stage3/tests`: **10 passed**.
- Python bytecode compilation: passed.
- C++ bridge and Scene Control CLI build: passed.
- Ground-truth layout self-test: passed.
- Synthetic five-session conversion, one-epoch CPU training, evaluation, and
  ONNX Runtime dynamic-shape parity: passed.
- Stage-three session runner PowerShell parse: passed.

## Real SDK qualification smoke

The first smoke reused a simulator session id and also enabled per-frame debug
JSONL; it exposed two operational hazards (transient Scene Control errors and
low throughput) and is retained as diagnostic evidence. The runner now uses an
isolated `run-*` output directory, a unique Scene Control control-session id for
each invocation, and disables heavy per-frame debug telemetry by default.

A clean 30-second stationary target-3 run with those settings completed all
three Scene Control acknowledgements. It produced 833 pre-tracker frames, 834
exact truth records (plus one explicitly unavailable startup record), and 106
valid tensor samples. The geometry fingerprint was stable; every visible frame
in this view contained one armor. Evidence is retained under
`runtime/stage3-smoke/real-session15`, with the raw files referenced by its
`session_result.json`.

This is a successful end-to-end smoke, not the formal qualification gate. The
approved 24-session qualification run must still be executed before the 360
session collection and three-seed training. No online tracker integration is
enabled, and the simulator repository remains read-only.

## Formal 360-session collection result

The corrected single-instance runner completed all 360 manifest sessions under
`runtime/stage3-formal-20260720-v2`. Every session produced a non-empty
`session_result.json`, observation JSONL, and truth JSONL. Across 1,389,655
observation frames, 960,533 contained at least one solved armor (69.12% overall)
and 393,958 contained multiple armors. By mode, detection was 70.43% for
stationary, 66.60% for linear, 73.26% for spin, and 67.76% for linear-plus-spin.
Zero-detection records remain in the raw stream for missingness accounting; the
offline tensorizer only emits windows meeting its valid-history and exact-truth
gates, so failed/empty sessions are not training samples.

## Training preparation implementation

The formal-data builder now emits `stage3-dataset-v2` from an exact 360-session
whitelist. It validates captured manifests, direct accepted run paths, complete
observation/truth keys, fixed-ego stable suffixes and cross-session geometry;
then writes session-disjoint train/validation/test shards, train-only
normalization, source/artifact/shard SHA-256 values and a qualification report.
First-article captures and unreferenced retries are not recursively discovered.

P0 tensor fixes are implemented: effective matched tau, exact-key anchors,
history-wide `>4` rejection, latest-valid freshness, strict truth geometry and
camera-origin/AngleSolver label coordinates. Each anchor has four fixed and
four deterministic random queries, so tensors are `[200,4,5] -> [8,4,3]`.

The v2 trainer reads only requested train/validation shards, verifies their
hashes and exact selection membership, rejects legacy/failing datasets by
default, records source/environment/selection provenance and saves best plus
recoverable last checkpoints. Best selection uses permutation-invariant
position L2 in metres. Evaluation is validation-first, requires matching
checkpoint provenance, treats test access as explicit, and compares against
fixed-geometry rigid static and constant-linear/constant-yaw-rate baselines.

Current regression evidence:

- `D:\Anaconda\envs\yolov8\python.exe -m pytest training/stage3/tests -q`:
  **10 passed**.
- Five-session optimized tensorization reproduced the pre-optimization sample
  counts `360/73/0/538/504` and identical shard hashes.
- A one-epoch, one-session entry smoke produced both best and last checkpoints
  and completed logical validation from the train shard without opening test.
  It is an interface smoke only, not an overfit or model-quality result.

## Formal dataset qualification and round-one feasibility

The accepted dataset is
`D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-dataset-v2-20260720-r5`
with manifest SHA-256
`026cbab209884f51150f2650ab25765b095738df3196d4d398bdbc5e54e72a3c`.
It qualifies all 360 raw sessions into 181,426 tensors: 109,159 train,
35,609 validation and 36,658 unopened test samples. Six raw sessions have no
eligible tensor windows; their 1.67% fraction is reported and below the fixed
10% limit. The split remains 216/72/72 sessions and geometry drift is
`1e-5`, below the `1e-4` gate.

At the user's direction, the immediate goal became one complete feasibility
round rather than metric optimization. The frozen pilot selection used 16
train and 8 independent validation sessions, with no test sessions. A
five-epoch GPU run completed and wrote best/last checkpoints, history and a
final run manifest. All 4,520 selected validation samples then completed
provenance-bound neural evaluation plus static and rigid constant-linear/
constant-yaw-rate baselines; both baseline-valid coverages are 100% and both
training and evaluation record `test_accessed=false`.

The protected evidence root is
`D:\仿真\models\engines\stage3-training\20260720-pilot24-seed0-round1`.
`feasibility-report.json` binds the dataset, selection, source, checkpoint and
validation-report hashes. Diagnostic learned median error is 0.975 m versus
0.423 m for the rigid baseline after only five epochs; this number is not an
acceptance gate and does not support a model-quality claim. The result proves
the offline data-to-training-to-validation workflow is executable end to end.
The worktree was dirty, so the checkpoint is exploratory, not a release
candidate. Test, formal three-seed training, ONNX publication and online
integration remain frozen.

## Calibrated-extrinsic v3 full training (2026-07-21)

The previous `H=0.07 m` path is retired. Simulator production coordinates now
compose the exact exposure optical pose with the calibrated camera-to-gimbal
rigid transform and use the exposure gimbal pivot as tracker origin. The
verified simulator calibration is:

```text
R_CAMERA2GIMBAL = [[ 0,  0,  1],
                   [-1,  0,  0],
                   [ 0, -1,  0]]
T_CAMERA2GIMBAL = [0.25631080, 0.00183094, 0.09543117] m
```

An independent exact-truth check covered 72 of the 360 formal sessions and
5,760 exposure poses. Maximum rotation error was `2.77e-5 deg`; maximum
translation error was `3.51e-7 m`. A separate blind signed-axis-permutation
check ranked the selected OpenCV-to-gimbal rotation first. The old synthetic
25-degree asset rotation had a median position residual of about `3.17 m` and
is no longer allowed to update production configuration. The report is
`D:\仿真\runtime\stage3-v3-extrinsic-validation-20260721.json`.

The temporal contract is also upgraded to `stage3-dataset-v3`. Each sample
uses the latest at most 200 valid observation events, left-padded only when
necessary, together with each event's real timestamp relative to the anchor.
No 5 ms grid or index-derived time remains in the model, data loader,
augmentation, physical baseline, evaluator or ONNX interface. Historical
`stage3-observation-v1` positions are reversibly migrated through the frozen
old contract and then transformed by the verified R/T; new v2 observations
must carry an R/T audit record matching the dataset calibration hash.

The qualified derived dataset is
`D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-dataset-v3-20260721-r1`
with manifest SHA-256
`8448ebe788b4a4bb5bd3803e4e64841bf39f3867f711d3198de31f1fb283ada0`.
It contains 185,292 samples: 111,527 train, 36,297 validation and 37,468
unopened test, with the original 216/72/72 session split. Six zero-sample
sessions remain explicit (1.67%), below the 10% qualification limit. The
dataset binds the active simulator YAML with SHA-256
`a4ba60bcd4c9baaaf113e3355ae1819f85cb97dfd5b23223ab67fc044dd6630b`.

A full single-seed run consumed all train and validation shards for 30 epochs
with batch size 128. It completed normally; epoch 28 was selected as best at
validation position-set L2 `0.232881 m`. Independent evaluation then covered
all 36,297 validation samples and 290,376 query results with 100% valid static
and constant-linear/constant-yaw-rate baselines. Overall learned median/P95
set error was `0.175675/0.569595 m`, versus `0.417854/1.336396 m` for the
rigid CV/yaw-rate baseline. Learned median error was lower in every reported
horizon and motion mode; this is a single-seed offline result, and the
stationary-mode learned P95 remained worse than the rigid baseline.

Protected evidence is retained under
`D:\仿真\models\engines\stage3-training\20260721-v3-full-seed0`.
The best checkpoint SHA-256 is
`f3e501eac136b985260a5dcafcb1f5c08814a001eaee18ad4c8e8aef24dc81e2`.
Its opset-17 ONNX passed dynamic batch/time/query parity with maximum absolute
error `9.54e-7`, below the `1e-4` gate. Training and evaluation both record
`test_accessed=false`. This completes the requested full offline feasibility
round; it does not authorize test evaluation, multi-seed metric acceptance,
TensorRT publication, tracker/MPC/fire-control integration or live firing.
