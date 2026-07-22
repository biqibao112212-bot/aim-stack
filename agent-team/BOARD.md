# Aim Stack 任务板

上下文版本：`CTX-AIM-STACK-2026.07-v3`

## 2026-07-21 PnP state-adapter A/B (in progress)

- Goal: quantify whether a single explicit constant-rigid-motion state helps
  predict physical future positions from the existing noisy PnP history.
- Shared input is the qualified v4 train/validation history: up to 200 real
  PnP events, masks, real timestamps and reprojection/count quality. Test stays
  sealed; future observation labels are not predictor inputs or objectives.
- Main A predicts one `center0/velocity/unit-phase/omega` state and passes it
  through a frozen arbitrary-tau rigid decoder. Main B predicts one rigid pose
  independently for each tau, with no shared velocity/omega propagation. Both
  use the same set/temporal encoder, geometry, position target and validation
  metrics; head parameter counts are matched within one percent.
- Existing PnP rows are unordered per-frame candidates, not persistent armor
  IDs. Both encoders and the main physical-set objective are candidate-order
  invariant; the objective uses symmetric nearest-set distance without q0/
  per-query identity, truth-slot input or 24-permutation enumeration. Online
  persistent identity remains a separate deployment gate and cannot be claimed
  from this A/B.
- Data audit: train/validation have 111,527/36,297 samples. Active PnP events
  overwhelmingly contain one or two candidates, and no sample has four fully
  visible slots in each of its last four events. The clean four-slot analytic
  LS core is therefore not a valid PnP adapter baseline.
- Paired implementation and 57 Stage-3 tests pass. A bounded stationary
  tiny-fit learned in both arms after the explicit-state zero-motion prior was
  added; this is optimization evidence only, not a generalization result.
- The hash-bound seed-0 dynamic pilot completed on 8 train and 8 disjoint
  validation sessions (4,584/4,615 samples) without opening test. It is
  rejected as a full-run candidate: A/B both collapse toward almost-static
  translation, and A does not improve any dynamic motion-class q3 P95.
- The v2 common trajectory supervision is implemented and covered by 65
  Stage-3 tests. It re-parses both arms' final decoded positions, masks truth
  windows that are not constant twist across every query within 1 mm/1 mrad, and adds
  common center/delta/velocity/relative-yaw/omega losses. Future truth remains
  loss/evaluator-only and is absent from predictor forward/export inputs.
- The v2 dynamic-only held-out micro gate completed but failed: stronger state
  supervision reduced neither velocity nor yaw-rate tails to a useful level,
  and training-session replay remains similarly poor. No full run is authorized.
- The bounded 561-sample combined-motion capacity gate passed. A and B both
  learned, while A converged to materially lower velocity/yaw-rate and 0.5 s
  motion error with numerical constant-twist consistency. It remains a
  train-sourced diagnostic and is not a generalization candidate.
- Current step: design a shared relative-time/relative-motion encoder and a
  broader held-out session pilot. No full run, test access, export or online
  integration is authorized from the current absolute-frame encoder.

## 2026-07-21 causal clean-physics A/B (completed)

- Scope is physical truth only: causal clean visible armor positions and real
  timestamps in, future physical armor positions out. PnP coordinates and
  exact center/velocity/yaw-rate are forbidden predictor inputs; test remains
  sealed.
- Both arms now share a causal last-four fixed-slot rigid least-squares core
  plus a zero-initialized learned residual. A uses q0/future/delta supervision;
  B adds reconstruction and shared constant-motion consistency only on the
  last four events qualified as one motion segment. The network never receives
  exact center/velocity/yaw-rate.
- Armor identity is a causal cyclic preprocessing contract. No 24-permutation
  training or evaluation alignment is permitted; a discontinuity starts a new
  segment instead of stitching identities across loss of view.
- The strict 8-session pilot retained 2,690/3,509 windows after rejecting 812
  under-observed reacquisition windows and 7 history-to-t0 motion changes.
  All 173,449 admitted history events passed producer/target/geometry identity.
- Validation input-sufficiency P95 is 0.00082 mm at q0 and at most 0.0364 mm
  at the 0.5 s rule query in every motion class; all samples are below 1 mm.
  A/B training selected their identical epoch-0 physical solution and rejected
  degrading neural residual updates. The full train/validation derivative,
  paired run, protected packaging and repository synchronization are complete.

## 当前状态

### 2026-07-21 阶段三 v3 全量离线轮次已完成

- `H=0.07 m` 已从生产坐标链废除；模拟器配置使用 exact-exposure 真值独立验证通过的 camera→gimbal R/T。72/360 会话、5,760 姿态的最大旋转/平移误差为 `2.77e-5 deg / 3.51e-7 m`。
- `stage3-dataset-v3-20260721-r1` 使用最近最多 200 个有效真实观测事件和真实时间戳，共 185,292 样本；split 为 111,527/36,297/37,468，6/360 零样本会话（1.67%）显式保留，资格通过。
- 全量 seed-0 训练完成 30 epochs，epoch 28 为 best；全部 validation 上 learned median/P95 为 `0.175675/0.569595 m`，刚体 CV/yaw-rate baseline 为 `0.417854/1.336396 m`，baseline coverage 100%。
- best checkpoint、last checkpoint、validation report、feasibility report 和动态 opset-17 ONNX 均保存在 `D:\仿真\models\engines\stage3-training\20260721-v3-full-seed0`；ONNX 最大 parity 误差 `9.54e-7`。训练和评估均记录 `test_accessed=false`。
- 当前唯一进行中工作：最终仓库回归、上下文/公开文档同步和本地提交。模拟器仓库、SDK、Release、原始数据、旧 v2 数据集与旧 checkpoint 均保持只读且不覆盖。
- 后续若进入正式 metric release gate，先让 validation report 绑定 `evaluate_v2.py` 与 `baselines.py` 源码哈希；该 P2 取证增强不阻塞当前离线可行性结论。

### 2026-07-20 阶段三采集实现（历史，已完成）

- 用户已批准执行第三阶段计划；正式仓库仍为 `main`，模拟器 Release/SDK 锁不变。
- 当前唯一进行中的工作：实现无泄漏 pre-tracker observation sink、独立 exact-exposure truth stream 和阶段三数据契约。
- 训练环境固定为 `D:\Anaconda\envs\yolov8\python.exe`；该状态已由后续正式采集和 round-1 训练记录取代。
- 模拟器仓库、SDK、Release 和受保护模型目录保持只读；任何接口缺口先停在消费者侧提案。

### 阶段三执行顺序

1. 观测/真值公共 schema 与 writer 接口
2. Scene Control 采集器与资格采集入口
3. Windows 数据转换、TCN、评测和 ONNX
4. 单元/集成测试与资格采集
5. 受保护数据、模型和 manifest 清单

- 自瞄与打符已从模拟器仓库物理分离。
- 自瞄 B 已导入 `modules/autoaim` 并使用 SDK v1。
- 打符源码已导入且模型全部保留，但其旧 SHM v6 消费端尚未升级，因此保持暂停，不声称兼容。
- 已锁定模拟器 Release 1.0.1（SDK 仍为 DaedalusSimSdk 1.0.0、SHM v7 ABI r1）；1.0.1 与消费者所需 SDK/SHM/TCP/UDP/Scene Control 公共契约兼容，架构与边界检查通过。
- 自瞄 B 后续只通过 Release SDK 的元数据、TCP 图像、UDP 云台和 Scene Control v1 访问模拟器；不再维护本地 IPC/场景协议。
- 2026-07-18 已完成两轮相同的 `-DynamicRange -DurationSeconds 30` shooting_range 基线。A：TCP sent 3611、main/capture 150.644/148.648 Hz、bind_fail/drops/GPU map errors=0，bridge completed vision 3579、1440×1080、TensorRT `vivsionn_trt`、exposure/ground-truth exact match=true；B：TCP sent 3252、main/capture 150.668/143.684 Hz、bind_fail/drops/GPU map errors=0，bridge completed vision 3230、1440×1080、TensorRT `vivsionn_trt`、exposure/ground-truth exact match=true。两轮 Scene Control 均收到 create/set_scene/target 1/target 3 ACK；每轮退出后无 daedalus、无精确 token bridge、5602/5603 无占用，第二轮可立即重复启动。
- 已建立所有未来消费者分支继承的 `SIMULATOR_CONSUMER_GUIDE.md`、机器锁和 GitHub 边界检查；三张原生地图、两种运行模式、SDK 用法与模拟器变更审批门禁均已固化。

## 阶段二完成状态

- 自瞄 B 阶段二已完成，G2 已通过：普通装甲板 `+15°` 倾角已在 tracker/chassis 坐标系建模，并通过曝光时刻云台姿态投影到 camera 完成约束 PnP；生产 tracker 消费修正后的底盘系 yaw。非零姿态合成回归与 target 3 的 3/5/7 m 慢速自转连续曲线构成验收证据。
- 2026-07-19 已完成 target-3-only slow-spin 测量：shooting_range 原生靶场、线速度/线性行程均为 0、`spin_deg_s=30`，距离 3/5/7 m 各 30 s，离屏高性能模式，`-PipelineOnly` 保留逐帧 pipeline telemetry 而关闭 bridge JSONL。三轮均收到 create/set_scene/set_target_3_spin ACK，TensorRT `vivsionn_trt`、1440×1080、exposure/GT exact match=true；TCP sent=4024/4026/4008，capture/main=129.25/131.238、162.427/167.409、135.564/136.561 Hz，bind/drop/map errors=0。
- 当前观测曲线和 smoke 结果只作测量证据：tracked yaw 样本分别为 1333/1553/1370，约 45.70/50.58/43.98 Hz；tracked_id 均在 `{0,1,2,3}`，30 s transitions=158/287/178，且存在 detector number 非 3 的少量样本和大量 `jump_flag`。曲线暂不能用于宣称 PnP 正常/异常，也不据此建立 G2 数值门槛。
- 2026-07-19 的分段/相位分析显示：在 active tracker 且 number=3 的主样本中，3/5/7 m 有效样本为 1079/1349/806（占 tracked yaw 的 80.9%/86.9%/58.8%）；`tracked_id` 切换间隔中位数约 18.6/16.7/17.6 ms，远短于 30°/s 下物理 90°切换的 3 s。基于固定 30°/s、每 ID 一个常量偏置的描述性正弦拟合，RMSE=28.5/25.2/27.7°、P95 残差=50.5/48.0/51.1°；这说明当前输出流不能直接当作干净的单正弦/90°跳变序列，但尚不能区分 tracked_id 语义、tracker 切换、检测误分类与 PnP 误差。
- 同一主样本的 selected PnP reprojection error 中位数/P95 为 0.41/1.31、0.49/1.45、0.42/1.52 px，未随距离单调恶化；该结果只用于定位后续实验，不构成 PnP 正常性结论或门槛。
- 对“17 ms 切板”的审计已校正口径：`tracked_armor` 在 `current_match_ids=[]` 或 `primary_observation_index=-1` 时可能只是 `_trackedArmor` 的旧值，不能作为新鲜 PnP 观测。只保留 `detected=true`、当前有匹配且 number=3 的 live 样本后，3/5/7 m 的 `tracked_id` 切换中位间隔为 21.6/26.1/23.1 ms，切换数为 149/266/145；其中单观测→单观测切换为 55/126/104，槽位差 2（0↔2/1↔3）占 53.7%/48.1%/40.7%。代码路径显示 `tracked_id` 是每帧 primary observation 匹配到的预测槽位，不是持久物理装甲板 ID；在 primary 选择和槽位语义明确前，禁止用它做 90°相位补偿。

## 门禁

- G2 已通过，阶段二关闭；通过结论只说明 PnP 输入坐标语义满足后续预测器前置要求，本轮回放不作为训练样本。
- 阶段三 v3 全量单 seed 离线训练和 ONNX parity 已完成；test、三 seed 指标验收、TensorRT 发布和任何在线 tracker/MPC/火控接入仍需后续明确门禁。
- 打符 SDK v1 适配完成前禁止闭环实测。
- 未获得用户针对具体提案的明确批准前，任何自瞄任务禁止修改模拟器仓库、SDK、发布脚本和正式 Release。

### 2026-07-19 observation-set rerun (user-approved recording change)

- Re-ran the same target-3 slow-spin experiment in native `shooting_range`: zero linear motion, `Spin=30 deg/s`, 30 s, 3/5/7 m, offscreen DX12 high-performance mode, `-PipelineOnly`.
- The first d3 startup failed closed at Scene Control `create_session` (`recvfrom failed: Resource temporarily unavailable`); owning PID/WSL/port checks after cleanup found no residual daedalus, exact-token bridge, TCP 5602 or UDP 5603. A fresh d3 run then succeeded; d5 and d7 succeeded on their first runs.
- Every successful run recorded Scene Control `create_session`, `set_scene`, `set_target_3_spin` ACKs; TensorRT backend was `vivsionn_trt`, frames were 1440x1080, TCP sent >0, bind_fail/drops/map errors were 0. The final per-run simulator stats are retained beside the observation artifacts.
- New artifacts preserve every `solved_armors` item per exposure in `observations.jsonl`; no `tracked_id`, `tracked_armor` or `jump_flag` is used for identity. The raw set retains detector metadata, four PnP corners, both PnP candidates, position, yaw, distance and reprojection error.
- Target-3 observation counts are 1312/1499/859 at 3/5/7 m; frames with target observations are 889/1093/685; frames with two or more target observations are 410/403/138. Detector number outliers remain in the raw set and are not silently relabeled.
- A separate replay artifact applies the agreed convention: first visible target plate is canonical ID 0, then temporal 30 deg/s continuity and cyclic allocation 0→1→2→3. It reports matched/new events and cost, while leaving raw observations unchanged. This is a derived analysis stream, not GT and not a G2 pass.

### 2026-07-19 diagnostic joint-PnP A/B

- Implemented a sidecar-only +15 degree joint yaw+tvec reprojection optimizer,
  legacy constrained residual recomputation, raw/refined corner branches,
  per-candidate convergence/conditioning, and marginalized
  `yaw_sensitivity_deg_per_px`. The tracker consumes legacy output only.
- Synthetic known-pose tests cover 3/5/7 m, yaw -70..70 degrees, nonzero
  distortion and invalid four-corner input. The focused candidate/coordinate
  regression passes; architecture and consumer-boundary checks pass.
- The performance launcher no longer forces Win32 `WindowStyle Hidden`; it now
  follows the Release performance contract using `DAEDALUS_PERF_DISABLE_UI=1`.
  It retains bridge/simulator logs and `DAEDALUS_STATS_JSON`, and fails closed
  on TCP bind, capture, GPU-map or surface errors. This fixed a reproduced
  consumer-launcher `ResizeBuffers / Invalid surface` run; no simulator files
  were changed.
- Valid A/B runs: 3 m r3, 5 m r1, 7 m r1. All received create/set_scene/spin
  ACKs, used `vivsionn_trt`, 1440x1080 TCP frames, and had bind/map/drop counts
  of zero. Final simulator snapshots report main/capture 104.94/104.94,
  119.31/119.31 and 118.28/118.28 Hz; TCP sent totals 3448/3247/3442.
- Result: refined joint reprojection improves p50 only 7.8%/2.8%/1.6%, while
  temporal p50 changes 2.74->2.96, 6.14->5.86 and 6.37->7.01 deg. Raw corners
  lower residual further but materially worsen temporal tails. Production PnP
  is unchanged; the current joint solver is evidence, not a selected fix.
- Physical/conditioning evidence: refined yaw sensitivity p50 is
  3.73/5.33/6.58 deg per pixel and p95 is 8.93/12.50/12.92 deg per pixel as
  distance increases. The next repair hypothesis must separate known absolute
  armor orientation transformed through the exposure pose from this measured
  pixel-conditioning floor; do not tune a G2 threshold before that review.
- Retention: `runtime/pnp-joint-ab-*` JSONL/logs/plots are reproducible current
  task evidence and are retained for the joint review. No model, engine,
  annotation, Release or non-reproducible data was deleted.

### 2026-07-19 chassis-frame repair replay

- Corrected production PnP yaw now models the ordinary armor's +15 degree tilt
  in tracker/chassis coordinates and applies the exposure-matched gimbal pose
  before camera projection. The old camera-fixed calculation is retained only
  as the diagnostic baseline; tracker input uses the corrected chassis yaw.
- Synthetic regression passed at nonzero pose: gimbal pitch +7 degrees, yaw
  -11 degrees, known armor yaw recovered within 0.1 degree, exact-pose RMS and
  max reprojection below 1e-4 px. Architecture and consumer-boundary checks
  pass; no simulator repository was modified.
- Successful replay artifacts: `runtime/pnp-chassis-pose-d3-20260719-r3`,
  `...-d5-20260719-r1`, `...-d7-20260719-r1`. A transient d3 retry
  `...-d3-20260719-r2` failed closed on simulator DX12 ResizeBuffers/Invalid
  surface; its stderr and bridge logs are retained. The next clean d3 run
  succeeded, so no process-wide kill or simulator edit was used.
- Successful runs all received Scene Control create/set_scene/set_target-3-spin
  ACKs, used `vivsionn_trt`, 1440x1080 TCP frames, and had bind_fail, queue_drop
  and GPU-map-error totals of zero. Simulator main/capture Hz were
  106.643/106.643, 124.370/124.370 and 130.819/129.820; TCP sent totals were
  3451/3397/3634; bridge completed vision counts were 1655/1475/1723.
- Continuous connected per-canonical-plate plots and metrics are in
  `runtime/pnp-chassis-pose-continuous-yaw-20260719.png` and
  `runtime/pnp-chassis-pose-continuous-metrics-20260719.png`. Production
  chassis-yaw adjacent increment p50/p95 (legacy in parentheses) are
  2.59/14.82 (2.65/15.78), 5.26/20.33 (5.66/39.24), and 7.67/28.88
  (8.76/69.40) degrees for 3/5/7 m. These are descriptive replay results;
  they do not establish a G2 threshold or authorize stage-three collection.

### 2026-07-19 G2 acceptance / stage-two closure

- User review clarified that the 3/5/7 m replay is PnP repair evidence, not a
  training dataset. The repaired production yaw is accepted as a geometrically
  valid per-exposure input for the later neural predictor.
- G2 passes on the combined nonzero-exposure synthetic regression, production
  tracker wiring, native-range continuous curves and complete runtime-chain
  evidence. No post-hoc scalar error threshold is introduced.
- The durable module report and plot are
  `modules/autoaim/docs/pnp_yaw_stage2.md` and
  `modules/autoaim/docs/pnp_yaw_stage2_target3_3_5_7m.png`.
- Stage two is complete. The only next item is joint discussion of the stage-three
  collection/training design; collection and training remain locked until then.

### 2026-07-20 Stage 3 implementation status

- User authorized execution. The consumer now contains the pre-tracker
  `stage3-observation-v1` recorder, independent exact-exposure
  `stage3-truth-v1` recorder, geometry-drift guard, and the public data contract.
- The Scene Control CLI has a strict `--stage3` path for stationary, linear,
  spin, and linear-and-spin motions with the agreed 3 m/s and 15 rad/s limits.
  `scripts/run-stage3-session.ps1` is fail-closed and uses one target-3 session
  token; the simulator repository remains read-only.
- `training/stage3` now provides deterministic manifest generation, raw join and
  tensorization, permutation-invariant causal TCN training, static/constant-
  twist baselines, evaluation, and dynamic ONNX export/parity checks.
- Validation completed: yolov8 pytest (2 tests), Python compileall, C++ bridge
  and Scene Control build, ground-truth self-test, five-session synthetic
  conversion/train/evaluate smoke, and ONNX Runtime dynamic-shape parity.
- This implementation-status paragraph is historical. Formal 360-session
  capture and the first 16/8 offline round are now complete; three-seed training
  and online integration remain unstarted.

### 2026-07-20 formal collection completed

- The corrected single-instance runner completed all 360 manifest sessions in
  `runtime/stage3-formal-20260720-v2`; every session has non-empty observation
  and truth JSONL plus `session_result.json`.
- Aggregate raw observation detection is 69.12% (1,389,655 frames; 960,533
  with at least one solved armor). Zero-detection frames remain raw for
  missingness accounting and are gated by the offline tensorizer.
- The canonical operating procedure is
  `modules/autoaim/docs/stage3_operations.md`; the first offline training round
  has since completed, while formal metric optimization remains deferred.

### 2026-07-20 qualification result

- A clean 30-second real-SDK stationary target-3 smoke completed with all
  three Scene Control acknowledgements and isolated `run-*` raw files.
- It produced 475 observation frames and 476 exact truth records at roughly
  15.8 Hz. The fixed geometry fingerprint was stable, but the approved
  8-observations/0.2-second tensorization gate produced zero samples.
- Formal qualification, 360-session capture, and training remain paused until
  the consumer-side capture/bridge throughput issue is resolved or the gate is
  explicitly reviewed. The simulator repository remains read-only.

### 2026-07-20 clean-smoke follow-up

- The runner now gives every invocation a unique Scene Control control-session
  id and disables expensive per-frame debug JSONL unless `-DebugTelemetry` is
  requested. Raw observation/truth paths remain isolated by `run-*` directory.
- A clean 30-second target-3 smoke completed all ACKs and produced 833
  observations, 834 exact truth records plus one startup-unavailable record,
  and 106 valid tensor samples. The previous zero-sample result was a retry/
  debug-telemetry artifact. The formal 24-session qualification gate remains
  the next required step.

### 2026-07-21 four-way validation error analysis

- Exact-exposure validation is complete for the four required comparisons:
  `O(t0)-G(t0)`, `O(t1)-G(t1)`, `P(t1)-G(t1)`, and `P(t1)-O(t1)`.
- Overall medians/P95 are `0.1024/1.2305`, `0.1019/1.1916`,
  `0.1757/0.5696`, and `0.2237/1.3459 m`, respectively. The model-vs-truth
  values reproduce the existing validation report exactly.
- Exact future-frame coverage is 99.828%; usable future-observation coverage
  is 76.745%, dominated by 67,016 zero-valid-candidate queries. Far targets
  have 62.329% usable coverage and observation median/P95 `0.2688/1.5731 m`.
- The machine-readable report is protected at
  `models/engines/stage3-training/20260721-v3-full-seed0/triangle-error-analysis-r3.json`;
  the method and slice tables are in
  `modules/autoaim/docs/stage3_four_way_error_analysis.md`.
- Current repair priority is the observation stream (zero-candidate coverage,
  far-distance PnP, and rare multi-candidate outliers). The independent model
  limitation is long-horizon/high-speed prediction. Future training continues
  to use physical truth as target; future observation is diagnostic only.

### 2026-07-21 future-observation target experiment

- Built `stage3-dataset-v4-observation-20260721-r4` from the immutable v3
  shards and exact raw timestamps. Missing future frames mask the observation
  branch; zero-candidate frames are explicit visibility negatives.
- Trained a full seed-0 dual-head model initialized from the v3 checkpoint.
  The physical head remains `P-G=0.175675/0.569594 m`; the observation head
  reaches `P-O=0.226535/1.353787 m` versus the old `0.223675/1.345863 m`.
- The first observation-target run is therefore feasible but not an
  improvement. Historical v1 inputs contain no reprojection quality and the
  future PnP residual has near-zero mean with large random spread. Keep this
  branch experimental until causal detector/image-quality features or a
  probabilistic output are available.

### 2026-07-21 scratch future-observation A/B

- Replaced the residual fine-tune with two controlled random-initialization
  models that consume identical batches and random masks. A uses masked direct
  future-observation Huber only; B adds a 0.2 physical-position auxiliary.
- Full training completed for 30 epochs on 111,527 train samples; test was not
  accessed. Both best checkpoints are epoch 29 on 222,848 usable validation
  queries.
- A reaches P-O median/P95 `0.198660/1.090279 m`; B reaches
  `0.201418/1.069940 m`, versus v3 `0.223675/1.345864 m`. B wins the predefined
  median-plus-0.25-P95 score; A wins median alone.
- The result supersedes the earlier claim that past observations contain no
  useful future-PnP signal. The old r5 residual implementation was
  miscalibrated and permutation-inconsistent; direct scratch training learns a
  real validation improvement. P99 remains slightly worse and is still open.

### 2026-07-21 physical-core acceptance

- [x] Cancel unnecessary recapture and bind the experiment to existing exact
  truth without opening the held-out test split.
- [x] Derive causal truth-history train/validation shards with exact target
  center, velocity, yaw rate, rule-query labels and distance strata.
- [x] Add q0-fixed assignment metrics, anchored-direct and rigid-latent A/B,
  masked rule-query loss, and physical-core unit tests.
- [x] Diagnose FP16 gradient overflow and invalid moving-sample tiny-fit;
  rerun a fixed-sample full-precision memorization gate.
- [x] Reject two-frame differentiation and armor-centroid rotation as the
  physical acceptance implementation.
- [x] Package and validate the exact-state rigid operator. All 1 mm gates pass;
  accepted report is under `20260721-v7-physical-exact-state-core-r5`.
- [x] Build the stricter causal last-N exact-position experiment with fixed
  cyclic slots, reset-on-gap, eight-event warmup, real timestamps and no
  24-permutation search.
- [x] Complete the paired full A/B run on 77,725 train and 25,695 validation
  samples. Both models select the identical epoch-0 causal physical solution;
  the 0.5 s validation motion P95/max are `2.15e-5/7.32e-5 m` and every
  1 mm gate passes. Test remains unopened.
- [x] Preserve the qualified dataset, oracle, A/B checkpoints, history and
  feasibility report under
  `models/engines/stage3-training/20260721-causal-physical-full-seed0-r2`.
- [ ] Jointly design the PnP-history adapter/noise layer around the frozen
  physical propagation. Test/export/online integration remain frozen until
  that separate layer has its own acceptance contract.

### 2026-07-22 fixed-slot neural physical state A/B (capacity gate failed)

- Frozen scope: existing clean fixed-slot physical train/validation only;
  test, PnP, export and online integration remain sealed.
- Implemented distinct neural arms: explicit shared state A versus implicit
  per-query pose B. Both use the same per-slot history encoder class, frozen
  geometry decoder and `2*q0 + absolute + 2*motion_delta` objective.
- Removed the unapproved unordered-PnP switching workaround before starting
  this experiment. No analytic LS or hand-derived velocity exists in the new
  forward paths.
- Unit/regression status: 79 Stage-3 tests pass; protected dirty-tree smoke
  runs prove paired training, checkpointing and common state diagnostics execute.
- The pre-registered train-sourced combined-motion capacity gate completed from
  clean commit `a8b3f89` for 31 epochs and stopped by paired early stopping.
  Test remained sealed. A selected the untrained epoch-0 checkpoint and B
  selected epoch 1. Their 0.5 s motion median/P95 errors are respectively
  `1.42535/1.44519 m` and `1.42548/1.44412 m`; A/B velocity-error medians are
  `2.84747/2.84569 m/s`. This is a hard learnability failure, not a threshold
  near miss.
- The frozen 16-train/8-validation held-out pilot has therefore not started.
  No architecture, loss, input, schedule, or selection rule will be changed
  until the failure is jointly diagnosed with the user.
- Protected evidence is retained under
  `models/engines/stage3-training/20260722-v10-causal-neural-state-ab-capacity-seed0-r2`.
