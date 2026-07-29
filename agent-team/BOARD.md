# Aim Stack 任务板

上下文版本：`CTX-AIM-STACK-2026.07-v3`

## 2026-07-25 strict single-handle F redesign (active handoff)

- No trainer is active. Repository `main` is at
  `464605c46f496836897c1db9b8e76e2376376bf7`. The worktree has four preserved
  uncommitted V23 draft files: `cyclic_rotation_ab_loss.py`,
  `cyclic_rotation_ab_model.py`, `tests/test_cyclic_rotation_ab.py`, and
  `train_cyclic_future_expert.py`.
- V23 formal training was never launched. Its two aborted smoke runs are
  invalid because source changed while they ran. The draft adds ordered local
  multi-track relations and tail losses, conflicts with the newly clarified
  strict single-handle F contract, and is paused pending an explicit
  discard/rework decision.
- V19-r2 epoch 110 remains the accepted frozen S/q0 foundation. PnP, router,
  export, online integration and test remain sealed. All datasets and model
  checkpoints are protected and retained.
- Read-only validation audit: the last-32 primary switches in 4.4%/66.2%/59.6%
  of translation/rotation/combined windows. Current-handle observed-span
  P10/P50 is 0.250/0.296 s for translation, 0.041/0.244 s for rotation and
  0.044/0.249 s for combined. At the nominal 0.5-second target, only
  95.35%/43.99%/57.83% of q0-current tracks remain virtually visible, but
  their same-handle future truth is still present; leaving view is supervised
  extrapolation, while cold/opposite never-seen tracks remain excluded.
- Current in-progress item for the next conversation: define and validate a
  flattened per-handle dataset and strict `F_i(history_i, time, tau)` forward
  contract. First stratify clean error by observed time span/arc and forecast-
  to-history ratio; then decide architecture and training budget. Do not start
  another formal run before this interface review.

## 2026-07-23 cyclic-track clean physical experts (historical; superseded)

- User explicitly returned the active scope to a no-PnP physical predictor and
  retained the specialist model: stationary, translation, rotation and a
  genuinely independent combined expert.
- Implemented a non-overwriting virtual observation contract over qualified r4:
  one/two adjacent visible tracks, cyclic primary and 0/+1/-1 switch history;
  no slot feature, center, phase, radius, height or geometry template enters
  inference.
- Implemented C4-equivariant shared-track temporal encoding, circular message
  passing, four independent direct trajectory decoders and a C4-invariant
  four-class router. Added local-label position/motion/self-rigid loss and role-
  split validation for visible and adjacent-hidden tracks.
- Gate status: 124 Stage-3 tests pass. The 100-epoch 1,024/2,048 smoke proved
  optimization capacity but was invalidated as metric evidence when preflight
  found missing `rule_query` masking and an incorrect random-q7 selection; it
  remains diagnostic-only. Both blockers are fixed. A corrected one-epoch
  end-to-end smoke verifies the 0.5 s query-3 selection, per-query eligibility,
  all-class coverage and C4 error below `6e-8`.
- Current step: clean commit, then protected seed-0 300-epoch full train/
  validation. Test, PnP, export and online integration remain sealed.

## 2026-07-23 Module A PnP pose recovery (superseded by clean-physics reset)

- Historical v16 evidence is retained, but this is no longer the active run.
- User had approved the split `PnP history -> current physical pose -> rolling
  32-event clean history -> frozen v15` and authorized v16 training.
- V16 A0 is q0-only current-pose recovery. It uses the qualified v4
  train/validation splits, accepts only rows whose latest PnP event is at q0,
  ignores the unqualified quality channels, and keeps test sealed.
- The permutation-invariant causal encoder consumes relative per-frame shape,
  relative centroid history, PnP yaw, masks/count and real time. The only
  learned outputs are current center and full canonical unit phase; frozen FP32
  geometry decodes the four fixed relative slots.
- The objective is center plus geometry-radius-scaled full-phase SmoothL1.
  Fixed-slot error and full-phase alias rate are primary gates; unordered-set
  error is auxiliary and cannot hide a 90-degree slot slip.
- Frozen v15 is hash-recorded but is not loaded, optimized, or used for
  checkpoint selection. The current step is smoke/full regression, clean
  commit, then the protected seed-0 full train/validation run.

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
  `2.84747/2.84569 m/s`. This is a failed training/selection run, not a
  threshold near miss; the later joint audit does not accept it as proof of a
  TCN expressivity limit.
- The frozen 16-train/8-validation held-out pilot has therefore not started.
  No architecture, loss, input, schedule, or selection rule will be changed
  until the failure is jointly diagnosed with the user.
- Protected evidence is retained under
  `models/engines/stage3-training/20260722-v10-causal-neural-state-ab-capacity-seed0-r2`.

### 2026-07-22 neural state full-training repair (implemented; clean-commit gate)

- The current five-block, two-convolution causal TCN has a 125-event receptive
  field, covers the configured 32 events, receives real anchor-relative times,
  and can represent the bounded session state. The hard decoder correctly
  preserves the four-slot geometry; its validation rigid residual P95 is about
  `5.4e-7 m`. No test access, future-label leakage, mask pollution, coordinate
  swap or time-unit error was found.
- The dataset guarantees constant velocity/yaw rate only from the most recent
  four observation events to `t0`, while the encoder consumes 32 events. The
  selected 319-sample session has a median 25 valid events over `0.2395 s`, so
  the active input contract can include earlier motion regimes not certified by
  the constant-motion gate.
- The planned 120-epoch capacity run performed only about five optimizer steps
  per epoch and stopped at epoch 31 because the untrained epoch-0 static prior
  participated in lexicographic motion-P95 selection. Its training objective
  was still decreasing (`2.54` at epoch 1 to `0.94` at epoch 31); this does not
  establish insufficient capacity.
- The common loss is structurally simple, but not gradient-balanced. A read-only
  autograd probe on the exact seed-0 initialization measured the weighted motion
  term's last-head gradient norm at about `0.042`, versus `0.392` for weighted
  q0 and `0.222` for absolute position. The zero-initialized last head also gives
  the encoder exactly zero gradient on the first update, and later total norms
  are repeatedly clipped at 1.
- Current evidence therefore ranks the likely causes as: epoch-0-controlled
  early stopping/selection, loss-gradient imbalance, 32-event versus last-four
  motion-contract mismatch, zero-head cold start, then raw-time/absolute-frame
  optimization bias. It does not justify changing to another network family.
- The user rejected another minimal diagnostic and requested that the four
  confirmed training defects be repaired before starting the full run. The
  active implementation now requires all 32 consumed events to be certified
  constant-motion history, uses small random final-head initialization, keeps
  epoch zero as an initial checkpoint only, and disables clipping/early
  stopping by default.
- The common loss is now an interpretable sum of meter-equivalent state terms:
  center0, velocity over a 0.5 s reference horizon, phase at geometry radius,
  omega over the same horizon, plus decoded constant-twist consistency. Both
  arms are reparsed from decoded positions by the same training-only extractor;
  future truth remains label-only and absent from forward inference.
- All 86 Stage-3 tests pass. The non-overwriting qualified derivative
  `stage3-causal-physical-v1-20260722-r4` has manifest SHA-256
  `8121dc8096952052ca9f9bfe3f5ed951c103a05a1ef7be4d65e2b40c731e113e`,
  32,904 train and 11,189 validation samples, 278 admitted sessions, four
  explicitly recorded zero-sample sessions, and `test_accessed=false`. All
  278 shard hashes verify.
- The training preflight now rechecks every admitted row rather than trusting
  manifest counts alone. Train/validation q0--q3 supervision coverage is
  `94.20%/93.66%`; the lowest present motion-class coverage is `89.51%`, above
  the fail-closed 85% floor. Every consumed row has 32 complete events; maximum
  fitted-history center/yaw residual is `6.45e-6 m / 1.36e-5 rad`.
- The user authorized committing the repaired source and launching the formal
  seed-0 full run for 300 epochs. The run uses all qualified train/validation
  samples, patience 0, gradient clipping 0, and a new non-overwriting protected
  output directory. Test, export and online integration remain sealed.

### 2026-07-22 factorized motion experts (launch authorized; clean-commit gate)

- The v11 300-epoch run completed with test sealed. Best A selected epoch 272:
  q0 P95 is `0.1108 m`, q3 motion P95 is `1.0163 m`, and linear/combined
  median speed ratios are `0.59/0.70`. B selected epoch 169 and is worse on the
  registered headline motion metric.
- The active v12 contract retains the causal 32-event TCN and hard rigid
  constant-twist decoder, but factorizes q0 pose, translation, and rotation
  heads. Translation and rotation each have a directly supervised positive
  expert plus a separately supervised activity gate.
- The objective has six group-mean terms: q0 center, q0 phase, balanced moving
  BCE, balanced rotating BCE, moving-positive velocity, and rotating-positive
  yaw rate. Static samples do not regress velocity to zero and non-rotating
  samples do not regress yaw rate to zero.
- The paired arms differ only by training-side spatial augmentation. The second
  arm rotates a complete sample around its latest-history center and translates xy within
  `+/-0.25 m`; validation remains untouched. Both arms share initialization,
  batches, dropout RNG, optimizer, scheduler, AMP, and 300-epoch budget.
- Unit and smoke validation must pass before commit. Formal training must start
  from a clean commit in a new protected directory and retain initial, best,
  last, and milestone checkpoints. Test, export and online integration remain
  sealed.
- Registered protected output:
  `D:\仿真\models\engines\stage3-training\20260722-v12-factorized-motion-experts-300ep-seed0-r1`.
  Runtime logs use the matching
  `D:\仿真\runtime\stage3-training-20260722-v12-factorized-motion-experts-300ep-seed0-r1`
  directory.

### 2026-07-22 independent rotation/combined experts (v13 launch in progress)

- V12 completed all 300 epochs with clean provenance and test sealed. Its best
  original arm is epoch 266; its best augmented arm is epoch 283. Translation
  is strong enough to freeze, while the rotation and combined-motion tails and
  route behavior require a structurally independent experiment.
- V13 has five non-sharing causal encoders: frozen augmented-v12 q0 pose,
  frozen original-v12 translation, trainable pure rotation, trainable joint
  combined `(v, omega)`, and a trainable four-class hard router. Combined
  motion is not constructed by adding specialist trajectories.
- Router/expert targets use truth-derived speed/yaw-rate factors rather than
  dataset motion class. Full eligible counts are 8,604/5,520/8,007/8,865 on
  train and 2,698/1,971/2,815/2,996 on validation for
  stationary/translation/rotation/combined.
- The dedicated five-test isolation suite and all 99 Stage-3 tests pass. A
  4,096-sample one-epoch smoke completed with finite loss/gradients, unchanged
  frozen foundations, rigid geometry and test sealed. The failed r1 smoke is
  retained temporarily as reproducible diagnostic evidence; r2 is the passing
  smoke.
- Current in-progress item: commit the reviewed source and start the formal
  seed-0 300-epoch run from that clean commit in the registered non-overwriting
  v13 protected directory. Test, export and online integration remain sealed.

### 2026-07-23 router-only factor-aware fine-tuning (v14 launch authorized)

- V13 completed all 300 epochs at clean commit `f7aa56c8c317f002dc3b698e87af76e57338d348`
  with test sealed and unchanged frozen foundations. Epoch 297 is registered as
  best. Raw specialists are retained, while the hard router remains the final
  bottleneck: translation/combined recall is 91.68%/85.61% and 423 of 2,996
  combined samples route to translation.
- V14 loads that exact best checkpoint and freezes every parameter except the
  existing `router_encoder` and `router_head`. The four-class interface and
  hard expert mapping do not change.
- The router-only objective adds group-balanced moving and rotating factor BCE
  to group-balanced four-class CE. A qualified planar rigid transform is used
  on training only; validation is untouched and no PnP noise is introduced.
- Three dedicated tests and all 102 Stage-3 tests pass. A one-epoch 512-train /
  full-validation smoke exactly reproduced the v13 baseline q3 P95, kept the
  non-router hash unchanged and retained `test_accessed=false`.
- Current in-progress item: commit the reviewed source and launch the formal
  seed-0 maximum-120-epoch run from that clean commit in a new protected v14
  directory. Test, export and online integration remain sealed.

### 2026-07-23 moving-only translation/combined refinement (v15 launch authorized)

- V14 stopped normally by early stopping after epoch 53 with test sealed and
  its frozen foundation unchanged. The last checkpoint is selected as the v15
  source so the accepted moving gate is retained; the complete v14 checkpoint
  remains a safe fallback.
- The remaining routing problem is now hierarchical. Frozen v14 decides
  moving/non-moving and stationary/rotation. A new independent binary model is
  consulted only for moving rows and decides translation versus combined.
- Its 32-event inference input is per-event center-free rigid shape plus the
  existing cyclic slot encoding. There is no temporal finite-difference input,
  future truth input, PnP noise, or expert-output leakage.
- All 2,132,628 source parameters are frozen and hash-verified. Only 184,065
  refinement parameters are optimized with group-balanced moving-only BCE.
- All 106 Stage-3 tests pass. A one-epoch 512-train/full-validation smoke
  reproduced the v14 baseline, completed with finite loss and gradients,
  verified the frozen base unchanged, and kept `test_accessed=false`.
- Current in-progress item: commit the reviewed v15 source and start the formal
  seed-0 maximum-60-epoch run from that clean commit. Protected model output and
  runtime stdout/stderr must be non-overwriting; test, export and online
  integration remain sealed.

### 2026-07-24 cyclic q0 state restorer (v18-S launch authorized)

- V17 has reached epoch 300 and remains immutable baseline evidence. Its
  latest validation still mixes unobservable cold hidden geometry with future
  propagation: stationary/translation adjacent-hidden tracks are cold in
  100%/96.8% of the validation support, while free per-track future heads retain
  decimetre motion and rigid-drift tails.
- The user clarified the S-layer task. Every currently observed one or two
  tracks is updated. Observations exactly at q0 are identity-passed; an
  observation before q0 is propagated to q0. Stationary/translation hidden
  tracks and every cold track are excluded from deterministic position loss
  and final position metrics. Rotation/combined additionally supervise only
  causally warm adjacent-hidden tracks.
- The new state restorer is separate from future motion experts and router. It
  uses one shared per-track causal encoder, sample-specific directed adjacent
  edge memory, shared cyclic message passing, no slot embedding and no fixed
  center/phase/radius/height/template. Motion class is loss/evaluation metadata
  only and is absent from forward inputs.
- Checkpoint selection uses rotation/combined warm-adjacent q0 P95 plus dynamic
  observed-edge P95. Cold contributes counts/coverage only. Validation also
  separates exact q0 observations from stale visible tracks propagated to q0.
- Ten dedicated S-layer tests and all 134 Stage-3 tests pass. A bounded
  4,096/4,096 one-epoch smoke completed with finite loss/gradients, exact q0
  identity to 4.77e-7 m, C4 audit, immutable checkpoint files and test sealed.
- A separate three-epoch 4,096/4,096 smoke was terminated after immutable
  epoch 1, then resumed from its hash- and provenance-verified checkpoint to
  epochs 2--3. Optimizer, scheduler, scaler and RNG state restored; epoch files
  were not overwritten and the final manifest remained test-sealed.
- Current in-progress item: commit the reviewed v18-S source and start the
  formal seed-0 180-epoch full train/validation run from that clean commit.
  Router, future propagation experts, PnP, export and test remain sealed.

### 2026-07-24 anchor-relative cyclic q0 restorer (v19 launch authorized)

- V18-S completed all 180 epochs at clean commit
  `9b4498aa0c8450952237569109c05103cc1b8af9`, with
  `stop_reason=epoch_limit`, test sealed and epoch 180 selected as best. It
  reached 6.17 cm rotation warm-adjacent P95 and 29.84 cm combined
  warm-adjacent P95. The combined tail is concentrated in stale/self-warm
  tracks rather than visible or recent pair-supported tracks.
- V19 keeps every current visible q0 update from v18 but replaces hidden
  absolute-position regression with an exact construction from the current
  primary q0 anchor plus a directed relative edge. An edge is supported when
  both temporary endpoint handles have been causally seen; simultaneous
  co-visibility is no longer required. There is still no center, phase, fixed
  radius/height, geometry template or physical slot identity.
- The exact v18 epoch-180 best checkpoint is the required foundation. New
  asynchronous edge/sigma heads train while the entire v18 foundation is
  frozen and hash-verified unchanged. The objective has only visible q0,
  supported-edge/anchor-composed q0, and uncertainty terms, balanced by motion,
  support kind and recent/stale age. The asynchronous residual is gated off for
  every pair-seen edge, so it cannot overwrite accepted v18 co-visible memory.
  Motion class remains absent from forward inputs.
- Nine dedicated v19 tests and all 143 Stage-3 tests pass. A frozen-foundation
  20-epoch 8,192-train/4,096-validation smoke reduced combined warm-adjacent
  P95 from 35.81 cm to a 14.42 cm best (15.30 cm at epoch 20) and rotation to
  6.40 cm. Pair-supported combined/rotation P95 stayed bit-exact at 9.04/5.96
  cm, translation propagated-visible P95 stayed 3.82 cm, and the frozen hash
  was verified unchanged. Exact q0 identity and C4 audits remain below 1e-6 m;
  interrupted-run resume and test sealing were also verified.
- Current in-progress item: commit the reviewed v19 source and launch a new
  non-overwriting seed-0 full train/validation run from the clean commit.
  Future motion propagation, router, PnP, export and test remain sealed.

### 2026-07-24 center-free future motion experts (v20 preflight)

- V19-r2 completed and epoch 110 is frozen as the sole q0 foundation. The v20
  future layer contains three fully independent trainable experts:
  translation (120 epochs), rotation (180 epochs), and combined (240 epochs).
  Stationary is a deterministic zero-motion path.
- The accepted combined decoder uses primary total 3-D velocity, primary
  planar acceleration and yaw rate. This removes the unobservable split
  between center translation and anchor tangent velocity while remaining
  exactly equivalent to constant translating-center yaw motion. It contains no
  center, phase, radius, height template or physical slot identity.
- Omega auxiliary labels now use only task-eligible adjacent edges. Cold and
  opposite future truth cannot change loss or gradient. Qualified-tail bounds
  are 7 m/s and 20 rad/s, above the audited 5.95 m/s and 14.96 rad/s maxima.
- Twelve dedicated tests and the complete Stage-3 suite pass. Three independent
  GPU smokes completed with finite gradients, exact frozen V19 hashes, sealed
  test, C4 error below 2e-6 m and rigid drift below 1e-6 m. Normal wall-stop
  resume, history-ahead recovery and atomic orphan-checkpoint adoption were
  exercised in separate runtime diagnostics.
- Current in-progress item: obtain final independent GO review, commit the
  exact source and launch the three protected formal runs with distinct model,
  runtime and log directories. GPU parallelism is limited by the single 8 GB
  device and must not displace unrelated user workloads. Router, PnP, export
  and test remain sealed.

### 2026-07-25 deterministic-direction rotation F A/B (v21)

- V20 rotation completed 180 epochs with epoch 135 best. Its 0.5-second
  current-visible/warm-adjacent cascade P95 is about 20.1/20.7 cm; replacing S
  q0 with truth leaves about 20.0/18.8 cm, so future F rather than S dominates.
- V21-A retains the center-free rigid decoder but consumes deterministic
  direction and learns only primary planar velocity plus yaw-rate magnitude.
  V21-B directly predicts continuous q0-relative trajectory candidates and
  applies a per-query, per-sample center-free rigid projection. Neither arm
  learns, classifies or applies a loss to rotation direction.
- The causal direction detector uses visible-history adjacent-edge rotation,
  with single-track curvature fallback and no future truth. Full clean-physics
  audit gives 100% accuracy on direction-qualified train/validation samples;
  coverage is 84.2%/87.3%. Unqualified early histories fail closed and the
  online state will retain a previously acquired direction for the target
  lifetime.
- Six dedicated V21 tests and all 161 Stage-3 tests pass. One-epoch
  512/512 smokes for both arms completed with finite gradients, exact frozen
  foundation hashes, test sealed, deterministic direction loss weight zero,
  direction accuracy 100%, q0 identity and C4 error below 2e-6 m.
- Current in-progress item: complete independent diff review, commit the exact
  clean V21 source, then launch non-overwriting seed-0 A/B formal runs. PnP,
  router, export and test remain sealed.

### 2026-07-25 unsigned relational rotation evidence (v22)

- V21 completed both 180-epoch arms. Direct B is better at 0.5 seconds, but its
  current-visible/warm-adjacent P95 remains 17.17/18.03 cm. Both arms plateaued,
  so further epochs are not justified.
- A read-only causal audit proved that the last 32 events already contain an
  essentially exact motion signal: adjacent-edge yaw-rate P95 error is 0.00018
  rad/s on 83.2% of validation. A truth-q0 physical diagnostic gives 0.0145 mm
  trajectory P95; with the accepted frozen S output it gives 3.46/7.68 cm
  current/warm P95. The production model's extra error is therefore in F.
- The shared V21 encoder temporally compresses each track before cyclic tracks
  exchange messages, so it cannot directly preserve synchronized edge motion.
  V22 adds a pre-compression relational stream with direction-free edge and
  curve invariants. Direction remains deterministic, external and loss-free.
- A2/B2 have 194,595/196,110 trainable parameters, a 0.8% difference. Nine
  dedicated tests and all 164 Stage-3 tests pass. Parallel one-epoch 2,048/
  2,048 smokes completed with finite gradients, empty stderr, C4 below 2e-6 m,
  rigid drift below 1e-6 m, direction accuracy 100%, edge evidence coverage
  82.9%, curve coverage 100% and union coverage 100%.
- Current in-progress item: commit the exact clean source and launch distinct
  30-epoch seed-0 A2/B2 controlled runs. PnP, router, export and test remain
  sealed.

### 2026-07-26 v24 anonymous observable-target F execution

- [x] Supersede same-handle future labels with future-visible target labels.
- [x] Implement dense signed switch unwrapping, q0 history inheritance,
  continuity-preserving ties and query-order-independent label extraction.
- [x] Build and independently audit the qualified r6 train/validation
  derivative: 38,306 windows, 299,190 eligible queries, zero uncovered,
  candidate steps -6..+6, test sealed.
- [x] Implement independent translation/rotation/combined visible-stream TCN
  experts, shared permutation-equivariant candidate heads, exact tau-zero
  identity and parameter-free stationary route.
- [x] Implement per-signed-step balanced switch loss, true-branch-only position
  loss, optional query-order-independent trend loss and mergeable validation
  metrics.
- [x] Pass 16 dedicated Torch behavior tests, 9 legacy cyclic regressions,
  full derivative audit, checkpoint write smoke and 20-update descent smoke.
- [ ] Run balanced 512-window / 5,000-update truth-S tiny-fit independently for
  translation, rotation and combined in Windows `yolov8` when the RTX 4060 is
  not occupied by the active NIGHTREIGN process. Do not displace that workload.
- [ ] After all three truth-S gates pass, add and run frozen V19-S A/B without
  changing the accepted F checkpoint. Report missing-candidate coverage outside
  F error.
- [ ] Decide whether S needs retraining only from the truth-S versus frozen-S
  A/B evidence. Until then V19-r2 epoch 110 remains immutable.
- [ ] Keep PnP, router, export, online integration and test sealed.
# 2026-07-26 observable F full-training handoff

- User decision: current tiny-fit precision is sufficient; stop refinement.
- Completed capacity evidence: translation/rotation accepted; combined
  accepted by explicit user judgment despite the earlier diagnostic 1 mm gate.
- Active: from-scratch full train/validation for the three independent v9
  experts.  Do not initialize formal runs from tiny-fit checkpoints.
- Still pending after truth-S validation: safe frozen-S V19-r2 A/B pairing and
  the evidence-based decision on whether S needs retraining.  Test/PnP remain
  sealed.

## 2026-07-26 observable F full-training closure

- [x] Complete from-scratch truth-S full train/validation for translation,
  rotation and combined in Windows `D:\Anaconda\envs\yolov8` on CUDA.
- [x] Preserve all protected checkpoints/manifests and keep test/PnP/router/
  export/online integration sealed (`test_accessed=false` for every run).
- [x] Stop all tiny-fit and precision refinement under the user's explicit
  acceptance decision; no continuation is authorized from a numeric gate.
- [x] Record that formal held-out validation does not satisfy the historical
  millimetre gates: final conditional P95 is 6.337/31.281/79.265 mm and final
  hard-routed P99 is 299.814/287.889/332.064 mm for translation/rotation/
  combined.
- [x] Do not retrain S. The residual already appears with truth-S, so the
  completed Phase-A evidence does not implicate frozen V19-r2.
- [x] Defer frozen-S A/B: Phase A did not isolate a frozen-S regression, and
  the current derivative lacks a proven hash-bound sample pairing adapter.
  This is a deliberate evidence boundary, not an unfinished training run.
- [ ] Any later work must begin as a new user-authorized definition/data audit,
  not as more epochs or tail tuning on these runs.

## 2026-07-26 real-PnP frozen-F upper-bound result

- [x] Let the dedicated selector finish its fixed 10,000-update budget without
  supervision or continuation. It stopped normally at `max_updates`; best
  switch accuracy is 90.86%, but hard P95 remains 299.52 mm. The gate remains
  failed and selector refinement is closed as a structural plateau.
- [x] Build a qualified clean/PnP paired derivative from real observation-v4,
  causal physical r4, truth-history r5 and observable r6. All 38,306 clean rows
  replay bit-exact; test remains sealed. Strict complete-history PnP coverage is
  22,806/38,306 (59.54%), with q0 coverage 90.13%.
- [x] Freeze and evaluate the accepted translation epoch 205, rotation epoch
  231 and combined epoch 180 checkpoints on CUDA. Paired conditional P95 is
  1.350/0.368/0.937 m after real PnP, versus 0.0056/0.0254/0.0810 m on the same
  clean queries. Every full model state hash remains unchanged.
- [x] Keep hard routing diagnostic-only. The continuous true-branch failure is
  already decisive, so the roughly 10% clean selector error is not the active
  bottleneck at this stage.
- [ ] Current decision point: agree on an observation-domain robustness layer.
  The recommended next experiment is a causal PnP history denoiser/adapter
  trained with paired clean targets and an explicit clean anti-forgetting gate;
  no new training is active. A deployable result additionally needs a
  permutation-invariant unordered-PnP association/S contract.

## 2026-07-26 paired PnP robustness training

- [x] Freeze a fair two-arm combined-motion contract on the identical 5,424
  train / 1,985 validation common-coverage rows. Conditional true-branch error
  is primary; hard routing is diagnostic only.
- [x] Build and hash the qualified r4 S/F sidecar with window-local C4/reversal
  anonymization, no fixed armor ID, no truth-filled failures and
  `test_accessed=false`.
- [x] Add the default-compatible differentiable F observation boundary, the A
  causal adapter, and the B four-handle S-to-F composition. Pass both CUDA
  two-update smokes and all 217 Stage-3 tests.
- [ ] Active: run A and B concurrently for the fixed 10,000-update budget in
  Windows `yolov8`; monitor only the first three validation rounds to estimate
  ETA, then let both finish naturally without extra epochs or tuning.
- [ ] After both finish, compare paired conditional errors, clean replay,
  current-anchor error and supported/unsupported S roles. Do not claim raw-PnP
  deployment; unordered association remains oracle-only in this experiment.

## 2026-07-27 true-A PnP-to-physical mapper closure

- [x] Correct the earlier A definition. The old selected-stream adapter bypassed
  S and could not change non-current candidate absolute positions. The true A
  now maps full sparse `[32,4,3]` PnP observations to paired clean physical XYZ,
  then runs bit-exact frozen V19 S and frozen combined F epoch 180.
- [x] Use a 74,451-parameter causal mapper with shared handle weights, invariant
  pooling, no physical/session/pair ID, no primary/switch/motion/future input,
  mask preservation, zero-init identity, strict causality and C4 equivariance.
- [x] Train on the qualified r4 train/validation pairs in Windows `yolov8`/CUDA.
  The full view supplies 28,322/9,688 windows and more than one million masked
  train occurrences; truth remains loss-only and test remains sealed.
- [x] Retain the downstream-selected checkpoint
  `20260726-v27-pnp-to-clean-observation-full-r2/epoch-0016-update-003552.pt`
  (SHA-256 `7d1cf2b0dcc5e358d43aab7d15a7a0c48546469d9263db47b4f9f4e59c5824db`).
  On combined/common validation, direct point P95 is 115.84 mm, current P95 is
  120.40 mm, conditional P95 is 536.82 mm, hard P95 is 619.01 mm and switch
  accuracy is 62.54%. Frozen S/F state hashes are unchanged.
- [x] Reject the robust-only, combined-only and primary-history-weighted runs:
  none improves the frozen-S/F conditional result. Preserve all run evidence;
  do not delete protected checkpoints.
- [x] Complete all 224 Stage-3 tests in 13.14 s. No training process remains.
- [ ] Next decision is architectural, not more epochs: either revise S support
  for invalid/unseen candidate q0 or build the deployable unordered-PnP/quality
  input contract. B remains the better current PnP predictor at 350.42 mm
  conditional P95, but it is not clean-preserving.

## 2026-07-27 A3 frozen-backbone q0 hypothesis adaptation

- User authorized execution of A3. The accepted PnP observation mapper,
  V19 S backbone and combined F motion backbone remain frozen; mapper-only
  refinement is closed.
- Active: freeze the no-leakage data/interface contract, then implement a small
  C4-equivariant q0 hypothesis adapter H after frozen S. H may correct observed
  and causally seen anonymous handles, predict support and uncertainty, and
  compose adjacent/opposite roles; a never-seen cold handle receives no
  deterministic coordinate loss.
- Next: pass unit/integration gates, run a bounded train-only capacity check in
  Windows `D:\Anaconda\envs\yolov8` on CUDA, then run the fixed full H budget
  only if capacity is demonstrated.
- After H acceptance: freeze mapper/S/H and the F motion/history/trajectory
  paths, train only the anonymous switch selector heads, and verify conditional
  trajectory output is bit-exact before/after selector training.
- Required evidence: C4 equivariance, no reversal augmentation, optimizer owns
  only H (then only router heads), mapper/S/F state hashes unchanged, combined
  tau>0 conditional/hard metrics, support/role/horizon slices, test sealed.
- Deployment remains out of scope: r4 uses oracle association, primary and
  switch labels and is a non-deployable upper bound.

## 2026-07-27 dual-domain PnP F closure

- [x] Reject mapper/window distillation and post-H history adaptation after
  fixed controlled runs: v41 517.81 mm, clean-teacher 478.04 mm,
  physical-target 469.85 mm and history-adapter 487.74 mm conditional P95.
- [x] Implement fail-closed external `CLEAN`/`PNP_V41` routing with independent
  storage. Keep clean F, mapper, V19 S and H frozen and forbid domain, motion
  class and physical identity as F inputs.
- [x] Complete the trajectory-only run. Retain v50 epoch 30/update 2550 at
  215.85 mm conditional P95; automatic plateau stop occurred at update 4250.
  The frozen selector partition and every upstream/clean hash are unchanged.
- [x] Complete the selector-only run. Retain v52 epoch 25/update 2125 at
  363.54 mm hard P95, 83.25% switch accuracy and 69.02% minimum-step recall.
  Trajectory state is bit-exact to the v50 parent, and v52 conditional outputs
  are bit-exact between its own pre-training baseline and every validation.
- [x] Pass all 261 Stage3 tests and checkpoint round-trip/partition audits in
  Windows `D:\Anaconda\envs\yolov8` with CUDA. Test split remains sealed.
- [ ] No training is active. Before any formal/deployable claim, rebuild or
  retrain the mapper/H chain with matching clean provenance and replace the
  oracle association with an accepted causal unordered-PnP interface. Reserve
  a new untouched acceptance split because validation has been used
  adaptively. Do not continue LR/epoch/loss sweeps on v50/v52.

## 2026-07-27 provenance-clean formalization

- [x] Synchronize the complete Stage3 source chain to local Git as separate
  commits: archived cyclic evidence `7df60cd`, observable F `51089af`, paired
  PnP A/B `41458e3`, mapper/H `94d4810`, rejected adapter controls `87275b9`,
  and dual-domain F `050a2cf`. Protected model/data/runtime assets remain
  outside Git. No remote push has been performed.
- [x] Add a formal-oracle training mode that requires one clean unchanged
  commit, canonical protocol/source/environment hashes, locked CUDA
  determinism, immutable asset hashes, independent split audits, matching
  mapper/H provenance and one fixed-final validation checkpoint. All 271
  Stage3 tests pass before the source commit.
- [x] Replay the v41 mapper architecture at fixed update 264 on commit
  `17e54ae`; all four mapper gates passed. The corresponding matched-H r1 run
  was externally interrupted before its first final-only checkpoint and left
  no recovery state, so it is incomplete evidence rather than a failed model.
- [x] Add formal-H full-epoch recovery states that preserve model,
  optimizer, RNG and DataLoader-generator state without accessing validation
  or selecting a checkpoint. Add an output-level OS process lock so duplicate
  launches cannot race recovery/final pointers. All 275 Stage3 tests and two
  independent read-only reviews pass.
- [ ] Active: commit and clean-validate the recovery mechanism, then replay the
  mapper on that exact commit because formal parent/child source contracts must
  match.
- [ ] Restart matched H with recovery enabled and preserve the interrupted empty
  r1 directory. After H completes, train fixed-final trajectory (2550 updates)
  and selector (2125 updates) PnP-F stages. Do not reuse v35/v50/v52 weights as
  formal outputs.
- [ ] Predeclare and validation-rehearse a single-use sealed holdout builder and
  evaluator. Test remains unopened until the entire candidate lock is frozen
  and the user explicitly authorizes one-time consumption.

## 2026-07-28 formal-H throughput correction

- [x] Stop the inefficient server H only after preserving its epoch-6/update-792
  recovery checkpoint; stop its watchdog first and preserve every old asset/log.
- [x] Diagnose the measured 3090 bottleneck: about 900 s/epoch, 30--32% GPU,
  597 MiB VRAM and one CPU core while every update recomputed frozen mapper/S.
- [x] Implement a train-only float32 CUDA cache for the 14 S fields consumed by
  H plus q0 truth. Keep batch 128, DataLoader shuffle/generator, optimizer/LR,
  three H forwards, loss and the partial tail batch unchanged.
- [x] Bind cache schema/content digest/shapes/dtypes/bytes and mapper/S/dataset
  hashes into provenance and recovery. Validation remains uncached and sealed.
- [x] Pass 277 Stage3 tests locally. Two design reviews accepted cache-only as
  the lowest-risk first stage; a final diff review and real 3090 benchmark are
  still required before a new formal launch.
- [ ] Commit the new source/protocol, replay mapper on that exact commit, then
  require cache field equality and at least 2.5x amortized speedup before
  restarting formal H from scratch. Never resume v59 under the new contract.

## 2026-07-29 joint future-visible selection diagnostic

- [x] Train the independent history-dynamics selector for 500 updates, then
  jointly update it with the PnP-domain trajectory path through update 3000.
  The Windows RTX 4060 run completed in 293 s with no stderr and retained the
  immutable update-3000 checkpoint under the protected Stage-3 model store.
- [x] Re-evaluate all 1,985 combined-motion validation windows (15,461 future
  queries) from the final checkpoint. The conditional tensor SHA-256 and
  conditional/hard P95 match the training record exactly; test remains sealed.
- [x] Export reproducible PDF/PNG figures for training dynamics, conditional
  versus hard distributions, coverage thresholds, time/yaw-rate/distance/
  switch/upstream-error trends, signed-step confusion and selector calibration.
  Exact physical distance and yaw rate are joined only after inference.
- [x] Compare on the same diagnostic validation. V64 ends at 217.95/413.83 mm
  conditional/hard P95, 78.16% selection accuracy and 61.96% one-step recall;
  V52 remains better at 215.85/363.54 mm, 83.25% and 69.02% respectively.
- [ ] Active: user review of the retained `figures-r2` evidence. Do not promote
  V64 or start another training run before deciding whether to keep the staged
  selector boundary or redesign the selector objective/representation.
