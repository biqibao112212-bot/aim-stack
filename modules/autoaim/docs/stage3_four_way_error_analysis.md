# Stage 3 four-way error analysis

Date: 2026-07-21  
Dataset: `stage3-dataset-v3-20260721-r1`  
Split: validation only (70 sessions, 36,297 samples, 290,376 queries)  
Checkpoint: `stage3-seed0-best.pt` (epoch 28)

## Definitions

For an anchor exposure `t0`, the model predicts `P(t1)`. The four reported
comparisons are:

1. `O(t0) - G(t0)`: current observation versus exact anchor truth.
2. `O(t1) - G(t1)`: the actual future observation at the exact future exposure
   versus exact future truth.
3. `P(t1) - G(t1)`: end-to-end model prediction versus physical truth.
4. `P(t1) - O(t1)`: model prediction versus what the observation system would
   output at that future exposure.

`O(t1)` is joined only through the shard's `future_timestamp_ns`. Missing
timestamps are not replaced by nearest frames or interpolation. Since armor
identity is not assumed, valid future observations are matched to the four
truth points with a minimum-cost injective assignment; frames with zero valid
armors or more than four valid candidates remain coverage failures.

Raw observation-v1 positions are migrated through the same calibrated
camera-to-gimbal R/T path used by v3 tensorization. The historical 0.07 m value
is used only inside the immutable v1 migration and is not a runtime offset.

## Overall result (metres)

| comparison | count | median | P95 | interpretation |
|---|---:|---:|---:|---|
| `O(t0) - G(t0)` | 28,496 | 0.1024 | 1.2305 | current input quality |
| `O(t1) - G(t1)` | 222,848 | 0.1019 | 1.1916 | future perception/PnP quality |
| `P(t1) - G(t1)` | 290,376 | 0.1757 | 0.5696 | end-to-end prediction quality |
| `P(t1) - O(t1)` | 222,848 | 0.2237 | 1.3459 | prediction/observation consistency |

The network result exactly reproduces the existing validation report. The
constant-velocity/yaw-rate baseline remains median/P95 `0.4179/1.3364 m`, so
the learned model is substantially better on the physical-truth comparison.

## Coverage

- Exact future timestamp exists for 289,877/290,376 queries: **99.828%**.
- A usable exact future observation exists for 222,848/290,376 queries:
  **76.745%**.
- The 67,016 unusable queries are almost entirely zero-valid-candidate
  frames; only 499 are missing exact records and 13 contain more than four
  valid candidates.
- Anchor observation quality is available for 28,496/36,297 samples:
  **78.506%**. This is a missingness property of the detector stream, not a
  filtering of the model's truth error.

## Quantitative attribution

On the 222,848 paired future queries:

- `P-G < O-G` in **33.926%** of cases; mean `(O-G) - (P-G)` is **+0.1449 m**.
  Thus the model improves over the future observation in a minority of paired
  cases, but the improvement is large where it occurs.
- The future observation has lower median error than the network
  (`0.1019 m` vs `0.1670 m`), which is expected because it is a direct
  measurement at `t1`, whereas the network forecasts without seeing `t1`.
- `P-O < P-G` in **36.551%** of paired queries. The model is therefore not
  simply copying the future observation.
- Pearson/Spearman correlation of `O(t1)-G(t1)` with `P(t1)-G(t1)` is
  `0.052/0.264`; future observation error is not the dominant explanation for
  the model's ordinary prediction error. Correlation with `P(t1)-O(t1)` is
  `0.990/0.664`, showing that poor future observations directly create large
  prediction-to-observation discrepancies.

The median-split quadrants are:

| future observation | model truth error | fraction |
|---|---|---:|
| low | low | 30.154% |
| high | low | 19.846% |
| low | high | 19.846% |
| high | high | 30.154% |

The balanced split is a consequence of defining each threshold at its own
median; it is not a claim that the two errors are independent.

## Slices

### Motion mode (future paired queries)

| mode | coverage | `O-G` median/P95 | `P-G` median/P95 | `P-O` median/P95 | `P-G < O-G` |
|---|---:|---:|---:|---:|---:|
| stationary | 83.652% | 0.0800/0.6912 | 0.1163/0.6775 | 0.0640/0.9523 | 34.6% |
| linear | 74.447% | 0.1279/1.3877 | 0.1775/0.6259 | 0.2488/1.4733 | 37.3% |
| spin | 77.771% | 0.0815/1.1874 | 0.1453/0.3361 | 0.1974/1.2937 | 34.4% |
| linear + spin | 76.118% | 0.1079/1.0947 | 0.2019/0.5682 | 0.2642/1.3413 | 31.1% |

The largest physical prediction tail is linear motion and combined motion;
pure spin has the smallest model P95. This does not support the earlier claim
that EKF spin behavior alone explains the neural error.

### Distance

| distance group | coverage | `O-G` median/P95 | `P-G` median/P95 | `P-O` median/P95 | `P-G < O-G` |
|---|---:|---:|---:|---:|---:|
| near (<3 m) | 93.043% | 0.0711/0.3306 | 0.1437/0.4289 | 0.1606/0.6601 | 20.0% |
| mid (3–5 m) | 74.587% | 0.1108/1.2511 | 0.1667/0.5252 | 0.2228/1.3405 | 33.4% |
| far (>=5 m) | 62.329% | 0.2688/1.5731 | 0.2214/0.6954 | 0.4926/1.7146 | 54.9% |

Distance is the clearest observation-side issue: far targets have both lower
coverage and much larger observation error. The network corrects more often in
the far slice, but its absolute physical P95 is still worse than near/mid.

### Forecast horizon (`P-G`)

| effective horizon | count | median | P95 |
|---|---:|---:|---:|
| 0–50 ms | 50,927 | 0.1646 | 0.3997 |
| 50–100 ms | 32,673 | 0.1426 | 0.3831 |
| 100–200 ms | 65,319 | 0.1479 | 0.4322 |
| 200–350 ms | 61,691 | 0.1780 | 0.5583 |
| 350–525 ms | 79,766 | 0.2459 | 0.8050 |

The model degradation is horizon-driven: the P95 roughly doubles from the
shortest to the longest bin. This is a prediction problem, not a timestamp
matching problem, because `P-G` is evaluated on all 290,376 queries.

### Visible candidate count

The 1-visible and 2-visible groups have `O-G` median/P95 `0.0868/1.3834 m`
and `0.1098/0.7159 m`, respectively. The rare 3/4-visible groups are highly
abnormal: `O-G` is `1.6576/5.5137 m` and `1.2599/4.0641 m`, while their
corresponding `P-G` medians remain `0.1602/0.1663 m`. The model beats the
future observation in 99.4–99.8% of these cases. This points to bad or
extraneous multi-candidate PnP records, not a neural forecasting failure.

## Decision

The first repair priority is the observation stream: zero-candidate coverage,
far-distance PnP quality, and the rare multi-candidate outliers. The model's
main independent limitation is long-horizon motion prediction, especially
high-speed/linear-plus-spin slices. A future training iteration should keep
`P-G` against exact truth as the primary loss/evaluation target and use
`P-O` only as a diagnostic, never as the training target.

The machine-readable full report is
`models/engines/stage3-training/20260721-v3-full-seed0/triangle-error-analysis-r3.json`.
