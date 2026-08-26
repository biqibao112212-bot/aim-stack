# Fixed-input filter research report

## Data quality and provenance

- Scenarios: 3; particle count: 2048.
- Numeric truth is excluded from all filter inputs. Saved physical slot is used only for oracle association.
- Future truth is read only for post-hoc scoring at 100/200/300/500 ms.

| scenario | exposures | matched PnP | matched fraction | LOS abs p50/p95 | tangent abs p50/p95 | LOS lag-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spin 8 rad/s | 3495 | 3492 | 0.999 | 206.6/589.0 mm | 2.7/69.5 mm | 0.591 |
| Translate 1.5 m/s | 3657 | 3470 | 0.949 | 144.9/599.5 mm | 3.8/77.8 mm | 0.562 |
| Translate 1 m/s + spin 6 rad/s | 3852 | 3740 | 0.971 | 260.9/719.6 mm | 3.3/22.4 mm | 0.489 |

## Filter-family replay

The following values are 3D future-position p95 in centimeters.

| scenario | horizon | EKF | UKF | PF |
| --- | ---: | ---: | ---: | ---: |
| Spin 8 rad/s | 100 ms | 31.40 | 31.54 | 169.97 |
| Spin 8 rad/s | 200 ms | 35.47 | 35.59 | 180.56 |
| Spin 8 rad/s | 300 ms | 39.25 | 39.36 | 197.33 |
| Spin 8 rad/s | 500 ms | 39.37 | 39.50 | 237.58 |
| Translate 1.5 m/s | 100 ms | 58.25 | 58.36 | 173.69 |
| Translate 1.5 m/s | 200 ms | 65.34 | 65.45 | 185.54 |
| Translate 1.5 m/s | 300 ms | 74.33 | 74.43 | 193.87 |
| Translate 1.5 m/s | 500 ms | 105.04 | 105.14 | 224.84 |
| Translate 1 m/s + spin 6 rad/s | 100 ms | 47.10 | 47.29 | 76.86 |
| Translate 1 m/s + spin 6 rad/s | 200 ms | 49.63 | 49.78 | 89.94 |
| Translate 1 m/s + spin 6 rad/s | 300 ms | 54.50 | 54.76 | 97.07 |
| Translate 1 m/s + spin 6 rad/s | 500 ms | 61.17 | 61.31 | 127.14 |

## Missingness and limitations

- The replay preserves every exposure timestamp and every missing PnP event in the locked JSONL.
- Oracle physical-slot association isolates the continuous estimator; it is not an online association result.
- PF is a finite 2048-particle bootstrap implementation with a causal 20-update EKF warm start and the same 11D model and Q/R. The result does not rank all possible particle filters.
- The structure-aware figure comes from the separately sealed combined-04 contract and must not be numerically subtracted from this 1.4.0 replay.
