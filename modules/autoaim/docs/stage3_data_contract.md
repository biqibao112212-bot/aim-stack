# Stage 3 data contract

## Raw streams

Observation and exact-exposure truth are independent append-only JSONL streams,
joined only by `session_id/producer_epoch/frame_seq/timestamp_ns`. Detector
number, tracker id and `jump_flag` are never identities.

`stage3-observation-v2` records every pre-tracker solved armor, including raw
camera `tvec`, calibrated position, R/T audit values and the position contract.
Zero-candidate frames remain in raw evidence. Immutable v1 captures are accepted
only through the named reversible migration: undo the historical 0.07 m
camera-y subtraction, recover camera tvec, then apply the calibrated SE(3).
The historical constant is not a runtime parameter.

Truth records contain exact exposure chassis, gimbal and camera poses plus all
four target armor geometries. Labels use the exact gimbal pivot as origin and
the anchor chassis forward/left/up axes. No empirical height offset is added.

## `stage3-dataset-v3`

Each sample contains the most recent `N=200` valid observation events, ordered
by real timestamp and right-aligned. A valid event contains at least one finite
armor candidate. Frames without a valid candidate do not consume an event, but
elapsed missing time remains visible in the next event timestamp. This choice
adapts naturally to different rates: roughly one second at 200 Hz and a longer
physical history at the current 30--50 Hz.

```text
obs          [200,4,5]  xyz, sin(yaw), cos(yaw)
obs_mask     [200,4]
event_mask   [200]      exactly obs_mask.any(-1)
event_time_s [200]      observation timestamp minus anchor timestamp
tau          [8]        effective matched future time
targets      [8,4,3]    future position and outward normal
```

Padding is left-only with masks false. Valid times are ordered and no later
than the anchor. There is no 5 ms grid, no event collision and no index-derived
time. A contributing span is rejected if any raw frame contains more than four
candidates. The existing real-time gates remain: at least eight valid events in
the last 0.2 s and latest valid age at most 50 ms.

The model receives `event_time_s` and derives inter-event delta internally.
Baselines also fit motion using these real timestamps. Train-only augmentation
drops individual armors and at most one contiguous block of 2--10 events,
compacts remaining events to the right, and is reverted if it violates the
real-time gates.

The 360-session split remains session-disjoint 60/20/20. Normalization is
train-only. Raw inputs, R/T YAML, builder sources, artifacts and shards are
SHA-256 bound. v2 shards/checkpoints are archived and rejected by v3; they are
never overwritten, renamed as compatible, or silently assigned 5 ms times.
