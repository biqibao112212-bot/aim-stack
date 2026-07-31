# Aim Stack 任务板

上下文版本：`CTX-AIM-STACK-2026.07-v3`

## 2026-07-31 profiled anonymous center/twist V14 (rejected; V15 active)

- V14 replaces residual/alternating state heads with a constrained current-state
  estimator. For supplied omega it profiles the anonymous chassis center and
  four window-local tracklet endpoints from
  `p=(I-R)c+t*v+R*q`, while the common velocity has no ridge or clamp. Center
  truth is joined independently and exists only as
  `anchor_center_position_m - frozen_H_current_position_m` in the loss.
- The state boundary is the original six causal history fields plus only
  `q0_relation_m` and `q0_supported`. It has no physical ID, confidence,
  session, motion class, absolute range, truth or future field. The center
  carrier uses all four finite anonymous H hypotheses because validation shows
  inferred unsupported roles improve Mean/P50; support calibrates uncertainty
  and gates the all-unsupported case instead of deleting geometry.
- The velocity solve now uses a Schur complement and explicitly reports minimum
  velocity information, condition, time span, profile support, fallback support
  and final state support. A short-arc counterexample that previously returned
  2.6 to 194 m/s while claiming success is now rejected. Fallback is a
  translation-only fixed-effect regression, requires at least 1 ms of visible
  time support, and is only claimed equivariant on its supported rows.
- H-informed profiling weakly anchors the four current endpoints to q0 as well
  as anchoring the center. This gives shuffled/corrupted H an observable closure
  diagnostics. A learned invariant component gate reads support, center
  uncertainty, XY/Z profile energies and Schur information, while retaining an
  always-available history-only branch. Its loss-only responsibility target asks
  which component is closer to truth velocity for intact and corrupted train
  batches; truth is never a gate input. Both component velocities and the gate
  features remain exact under a common ramp.
- The complete 750-window validation-only zero-update audit keeps test and all
  future modules unopened. The all-four center error is
  0.134/0.104/0.311 m Mean/P50/P95. With truth omega, q0 soft-center velocity is
  0.894/0.338/3.561 m/s versus history-wide
  1.498/0.299/6.499 and truth-center oracle 0.707/0.123/3.232. Q0 profile and
  final-state coverage are both 100%; history-only uses explicit fallback on
  1.47% of rows. Audit SHA-256 is
  `3aadb3ddf624b4ee57fb7d127a10a19ecdcfecb08b64aca02fbfb59ce974b6db`.
- The implementation was committed at `12ca34e`; 25-update interruption,
  75-update interruption and uninterrupted seed-20260730 paths all finish with
  identical model state SHA-256
  `ba9c489c73615a7e1a2a68615e04488b65eec656f523b21c28377d07189cfaf9`
  and identical metrics/gates. V14 is therefore deterministically rejected,
  not an interrupted or unlucky run. It improves center Mean/P50 by 3.83/2.58%
  and velocity P50 by 5.66%, but velocity Mean improves only 1.22%. The learned
  gate remains nearly constant: intact/shuffled q0 weights are 0.7959/0.7912,
  while their loss-only oracle responsibilities are 0.583/0.271. Shuffled H
  raises P50 from history-only 0.299 to 0.819 m/s, so this is a body failure,
  not tail-only rejection. Do not run the second V14 seed or add updates.

## 2026-07-31 frozen-expert continuous reliability fusion V15 (active)

- V14's two velocity components are retained only as frozen diagnostic experts.
  On the sealed 750-window validation split, a loss-only continuous projection
  oracle reaches intact Mean/P50 0.641/0.152 m/s and shuffled-H 0.808/0.176,
  versus the trained mixture's 0.850/0.286 and 1.549/0.819. Existing observable
  compatibility features predict component preference with grouped linear-probe
  AUC about 0.813 and distinguish intact/shuffled inputs with AUC about 0.958.
  The components and forward information have headroom; the joint gate training
  definition is the rejected part.
- V15 separates the stages. It freezes the center prior and both profiled
  experts, trains only an anonymous reliability fusion head, replaces minibatch
  roll corruption with precomputed hash-bound global and matched-hard
  derangements, and supervises the actual continuous convex coefficient
  `clip(((truth-vh) dot (vq-vh))/||vq-vh||^2,0,1)`. Intact and corruption arms
  receive equal weight. The head must use deployable O(2)/permutation/common-ramp
  invariant compatibility evidence and make q0 weight monotonically decrease as
  explicit incompatibility increases. Truth remains loss-only; truth omega is
  still the declared B0 mechanism slice. Test, free omega, future-position and
  scatter plots remain frozen.
- First run a reliability-only diagnostic parented to the fixed V14 checkpoint.
  It must recover at least 30% of the intact oracle gap (approximately Mean <=
  0.787 and P50 <= 0.246 m/s), keep both global and matched-hard shuffled
  Mean/P50 within 1.02x history-only, separate intact/shuffled mean q0 weight by
  at least 0.15, and reach component-preference AUC at least 0.75. Only a passing
  probe authorizes a clean two-stage, two-seed formal rerun.
- V15-A0 completed locally at
  `D:\仿真\models\engines\stage3-training\20260731-v89-v15-a0-frozen-reliability-oracle-omega-r1`
  from clean commit `d2992d7`. The train-only two-fold screen ended
  `cv_failed`; `validation_accessed=false`, no validation ledger was created,
  and test/future stayed unopened. `screen_result.json` SHA-256 is
  `0e161fee2fde0b4a4c4368f13871a46b120484336935ab6e4777b8ad68968e5c`.
- Integrity correction: this artifact is archived evidence, not an
  authorization source. Its paired validation dataset was never constructed
  and no ledger was claimed, but the legacy truth-index constructors eagerly
  hashed and loaded both train and validation truth shards before attaching
  only train records. The train-CV numbers remain useful because validation
  records were never joined or used, but `validation_accessed=false` was too
  strong. A split-scoped truth index now opens only one declared split; A0 must
  be rerun from clean corrected source before A1 can execute.
- Corrected V15-A0 reran from clean commit `2f68c23` at
  `D:\仿真\models\engines\stage3-training\20260731-v90-v15-a0-split-scoped-reliability-r1`.
  It reproduced both held-out-fold intact metric groups exactly and ended
  `cv_failed`, while `validation_accessed=false`,
  `validation_claimed=false`, no ledger file, and future/test remained
  unopened. The corrected `screen_result.json` SHA-256 is
  `9c86a10041a66baf93588b1e0cb32ce9c99dd36ad0008bc2d6d5a7b7602d7853`;
  only this corrected result may authorize A1.
- The failure affects the distribution body in both held-out session folds.
  Intact overall fused/parent/oracle Mean/P50 is
  `0.767/0.381`, `0.749/0.373`, `0.518/0.192` m/s in fold 0 and
  `0.950/0.682`, `0.912/0.624`, `0.661/0.397` m/s in fold 1. Oracle-gap
  Mean recovery is `-7.9%/-15.1%`. Intact component AUC changes from
  `0.739` to `0.552`; fold-1 rotation is `0.491`. Global/hard corruption
  AUC is stronger in fold 0 (`0.802/0.819`) but falls in fold 1
  (`0.602/0.624`), and intact-minus-corrupt q0-weight separation remains only
  `0.03-0.06` instead of `0.15`. More updates or a wider 13D MLP are rejected.
- A train-fold-only nearest-neighbour audit confirms this is missing
  information rather than ordinary extrapolation: kNN-20 oracle-weight
  correlation is `0.426/0.220` with MAE `0.309/0.305`, while fold 1 has the
  smaller normalized nearest-neighbour distance (`0.687` versus `0.858`).
  Absolute q0 XY energy is not a valid monotone unreliability variable: its
  held-out correlation with oracle q0 weight is `+0.277/+0.035`. The A0 hard
  negative slope on absolute energy is therefore removed with the 13D head,
  not tuned.
- A1 is a train-only endpoint-information probe, not a formal model. It will
  preserve paired local evidence before symmetric set pooling: per-role causal
  history residuals, q0-versus-history fitted endpoint contrast, visibility/
  age/span, and unordered pair geometry. All event/role/pair encoders are
  shared and have no role or physical-ID embedding; XY enters only through
  O(2)-invariant norms/dot products, and expert velocities enter only through
  their difference. Mapper/S/H/V14 experts remain frozen and truth omega stays
  an explicitly diagnostic profiler input. A1 may use only the same two
  session-disjoint train folds. Validation, test, free omega, future-position
  and plots remain frozen.
- The first A1 launch at
  `D:\仿真\models\engines\stage3-training\20260731-v91-v15-a1-p0-endpoint-token-r1`
  failed closed before head training because the exact hard-map builder treated
  each imbalanced `(motion, q0-support-count)` stratum as all-or-none. Fold-0
  held-out rotation coverage was only `544/752=72.34%`. This is a map
  construction defect, not a model metric; the incomplete artifact remains
  archived with validation/test/future unopened.
- A1 now constructs the deterministic maximal balanced subset inside every
  exact stratum, preserving cross-session bijection, no fixed point, no donor
  reuse and no relaxed matching. Full-family denominators and the 80% gates are
  unchanged. Real preflight hard coverage is `98.25%/95.25%` overall,
  `98.33%/93.09%` rotation and `98.13%/99.07%` combined across the two
  complementary fold populations. Commit `0110740` has 686 passing Stage-3
  tests and both consumer-boundary gates pass.
- A1-R2 completed at
  `D:\仿真\models\engines\stage3-training\20260731-v92-v15-a1-p0-endpoint-token-r2`
  with status `failed`; validation/test/future remained unopened and no fold
  checkpoint was authorized. `screen_result.json` SHA-256 is
  `c60beb32377d9cf767207268851dae6a47579a5ec03fd4d308a240bd227670d5`.
  Intact fused Mean/P50 improved over A0 from `0.767/0.381` to
  `0.691/0.311` in fold 0 and from `0.950/0.682` to `0.870/0.601` in fold 1,
  but oracle-gap recovery is only `25.0%/34.1%` and `16.8%/10.2%`.
  Fold-1 combined is weakest at `8.9%/10.4%` recovery.
- The decisive A1 rejection is local-information insufficiency after pooling.
  Intact local-token versus local-ablated AUC gain is only `0.005-0.044`, and
  coefficient-MAE improvement is only `0.8%-2.3%`, far below the declared
  `0.03` and `10%` gates. Global/hard corruption AUC is nevertheless
  `0.75-0.90`, and corruption Mean/P50 are materially better than blind
  selection. Do not add epochs: the next structural probe must preserve
  anonymous per-endpoint temporal evolution until the final decision instead
  of collapsing it through unordered mean/max summaries.

## 2026-07-31 equivariant alternating twist V13 (rejected; next structure active)

- V12 is complete and validly rejected, not an unfinished run. Its two fixed
  seeds pass only 44/94 and 45/94 gates. Relative to V8, overall velocity and
  yaw body metrics regress, pair0/1/2 yaw correspondence often improves when
  broken, and pure-rotation velocity is contaminated. The mechanism evidence
  shows why: the visible-factor apparent-rate mean was used as a translation
  gauge before omega and absorbed rotation; strict omega-first ordering then
  had no way to recover it. More V12 updates or small threshold tuning are not
  authorized. Protected V12 checkpoints and the failed aggregate remain intact.
- V13 is a structural replacement with four typed, detached stages:
  `omega0 -> velocity0 -> omega1 -> velocity1`. Omega is no longer an analytic
  carrier multiplied by a positive gain. Reflection-even temporal networks
  emit unrestricted scalar coefficients over several explicit signed
  pseudoscalar proposals, so a zero or wrong-sign angle carrier remains
  learnable. Handle curvature needs at least two consecutive chords/three
  positions; pair and handle evidence are precision-weighted softly, never
  switched by a physical armor ID or a hard any-pair rule. Omega1 is explicitly
  `detach(omega0) + delta_omega(detach(velocity0), history)`.
- Velocity uses analytic de-rotation followed by exact supported-row WLS. Its
  learned correction is an O(2)-equivariant combination of WLS-residual vector
  bases, so a common planar velocity ramp transfers exactly and physical-Y
  reflection flips only the Y component. Unsupported WLS rows are explicit and
  excluded from state/equivariance losses. Irregular chord acceleration uses
  chord-midpoint time, and the fixed 0.105-s lag/yaw-scale alias envelope is
  asserted. The observed-history closure diagnostic now uses the same analytic
  decoder as forward.
- Training is a fixed 35/20/25/20 update structural screen with a reset LR
  phase per typed stage. The sampler is balanced by motion, session, history,
  active/stationary state and exact observable pair-scale support 0/1/2/3; the
  training loss also macro-balances motion x pair-support x history. Direct
  state supervision, typed observed-history closure, common-ramp equivariance
  and physical reflection are trained together. Real-loss tests require a
  finite nonzero gradient in exactly one typed module at every boundary.
- The six-field anonymous causal inference API is unchanged. No armor ID,
  session, motion class, truth, future, absolute range or PnP-only quality is a
  forward input. The learned future-position decoder remains frozen and
  hash-protected. V13 has 1,359,040 state parameters versus V8's 1,487,688,
  within the fixed capacity ceiling. Focused tests pass 35/35 and the complete
  Stage3 suite passes 596/596; all three independent reviews are READY.
- The formal result validator reconstructs the model strictly, binds the clean
  commit, source hashes, checkpoint payload, 35/20/25/20 transitions, per-stage
  branch hashes, validation, diagnostics, frozen-future hashes and source
  dataset. It reloads only train/validation truth, verifies every truth shard,
  reattaches exact join keys and recomputes the yaw-alias report, so synchronized
  contract/checkpoint report tampering cannot pass. Same-seed V12 result paths
  and SHA-256 values are constants, not
  report-selected inputs. Numeric schema validation precedes all gates;
  pair1/2/3 causal breaks, absolute-or-relative closure worsening and P95
  disaster guards supplement mean/P50 body metrics. The aggregate reruns the
  complete validator for both seeds.
- The clean local RTX 4060 screens completed normally at commit `e3dba305`.
  Seed 20260730 passed 9/38 gates and seed 20260731 passed 8/38; the independent
  aggregate is `failed` and does not authorize continuation. Both checkpoints,
  manifests, screen results and the aggregate are protected evidence. Future
  modules remained hash-identical and test stayed sealed.
- This is a body failure, not a tail-only rejection. Two-seed V13 overall
  velocity Mean/P50/P95 is 1.773/1.759/3.618 m/s versus V12
  0.616/0.455/1.853 and V8 0.511/0.363/1.453. Yaw is
  7.053/6.114/17.553 rad/s versus V12 3.958/2.185/13.173 and V8
  1.861/1.188/5.574. Update 0 to 100 improves overall body Mean by only 0.9%
  velocity and 2.5% yaw, so more V13 updates are forbidden.
- Mechanism review found a concrete pair contract error. The upstream pair bank
  already flips the prior vector when the current primary changes; V13 flips
  current again and repeats the sign in closure. On all supported validation
  rows, single-owner alignment gives yaw carrier Mean/P50/P95
  4.863/2.472/17.261 rad/s and correlation 0.725, while the double flip gives
  8.725/7.361/20.740 and correlation 0.165. For combined pair3 the aligned
  carrier is 2.004/1.325/5.919 with correlation 0.943, versus
  6.742/4.693/17.031 and 0.290 after the erroneous second flip.
- Pair repair alone cannot fix velocity. With truth omega, V13's visible-gauge
  WLS remains 1.759/1.754/3.533 m/s and even a truth-velocity gauge remains
  1.644/1.671/3.349. A zero-training joint rigid profile over anonymous center,
  velocity and per-window tracklets reaches useful body medians but unstable
  means on short arcs. Supplying truth center only as an oracle mechanism bound
  gives overall/rotation/combined velocity Mean/P50 of
  0.708/0.123, 0.894/0.229 and 0.428/0.086 m/s. This identifies center--velocity
  ambiguity as the next structural target, not a request to expose truth at
  inference.
- The interface decision is complete: the state estimator adds only
  `q0_relation_m [B,4,3]` and `q0_supported [B,4]` from frozen S/H. A simple
  anonymous q0 mean already reduces the truth-omega fixed-center validation
  velocity to overall/rotation/combined Mean/P50
  0.902/0.332, 1.082/0.484 and 0.633/0.239 m/s, recovering most of the oracle
  center-prior body improvement without physical IDs, truth or future input.
- Superseded by the active V14 profiled center/twist section above.

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

## 2026-07-29 ordered crossing-time selector diagnostic

- [x] Audit V52/V64 errors: V52 wrong choices are 96.14% adjacent and truth
  switch sequences are monotone in all 5,424 train and 1,978 audited
  validation windows. Preserve the staged boundary and freeze Mapper/S/H/V50.
- [x] Reject the first single-progress curve after an exact capacity audit:
  positive monotone bases force a concave progress curve and can express every
  label in only 194/256 train and 187/256 validation windows. More epochs cannot
  remove this structural limit.
- [x] Implement sample-conditioned T1<...<T6 crossing times, a separate shared
  direction, stratified capacity sampling and immutable per-epoch recovery.
  There is no candidate-wise head, physical ID, future path, global fixed switch
  time or position decoder.
- [x] Pass 30 related Stage3 regression tests and an independent P0 review. The
  fixed 400-update capacity result reaches 94.04% train accuracy, 90.68%
  one-step recall and 254.83 mm hard P95 with zero reversals and bit-exact
  trajectories. The predeclared 95% gate is not marked passed; its 0.96-point
  miss is explicitly manually released for one full diagnostic run.
- [x] Complete the full run at the only fixed endpoint, update 2125, in 148.8 s
  of training time. V66 reaches 358.12 mm hard P95, 554.58 mm hard P99,
  83.565% accuracy and 73.095% one-step recall versus V52 at 363.54/560.30 mm,
  83.255% and 69.021%. Conditional P95/SHA and the V50 state stay bit-exact;
  predicted sequence reversals are zero.
- [x] Replay all 1,985 validation windows and export PDF/PNG training,
  distribution/CDF/coverage, scatter-and-quantile trends, signed-step confusion,
  calibration and V50/V52/V64/V66 comparison figures under `figures-r2`.
- [ ] Do not tune this validation further. V66 improves every V52 relative
  selector metric but misses the stronger predeclared absolute gate at 350 mm
  hard P95, 550 mm hard P99 and 84% accuracy, so `gate_passed=false`. The next
  stage must target confidence/rejection or a new evaluation boundary rather
  than silently selecting epoch 35 or adding epochs.

## 2026-07-29 final future-position residual diagnostic

- [x] Define a zero-initialized final-position residual after the complete
  Mapper/S/H/V50/V66 chain. Mapper, S, H, trajectory and armor selection remain
  bit-exact frozen; only the bounded XYZ residual is trainable.
- [x] Pass six dedicated CPU/CUDA tests for exact zero-init equivalence,
  frozen-state isolation, query/candidate permutation and finite gradients.
  Complete a 256-window CUDA smoke run with recovery, metrics and query export.
- [x] Replace the previous figure suite for this stage with one plain scatter:
  future query time on x and final position error on y. Each dot is one query,
  so the vertical spread at a shared query time is the requested distribution.
- [x] Commit the reproducible trainer at `acee996`, then complete the fixed
  10-epoch/430-update full combined-motion diagnostic locally in the Windows
  `yolov8` CUDA environment. The final endpoint passes its predeclared gate:
  Mean/P50/P95 improve from 109.67/46.00/358.12 mm to
  108.75/45.44/351.96 mm; P99 moves from 554.58 to 560.57 mm within the allowed
  10 mm tail tolerance. The frozen stack is bit-exact and selection accuracy
  remains 83.565%. No intermediate checkpoint is selected and no epochs are
  added.

## 2026-07-29 disjoint simulator generalization evaluation

- [x] Generate a new-seed manifest with six random spin and six random
  linear-plus-spin sessions. Collect all 12 through Simulator 1.0.1/SDK 1.0.0
  for 20 seconds each; every accepted session completes on its first attempt.
  Failed pre-build attempts are retained but have no `session_result.json` and
  are excluded by canonical-source discovery.
- [x] Build a qualified independent v3/PnP/S/F evaluation chain without
  touching the three newly reserved test sessions. Original V67 and new
  train/validation session overlap is exactly zero. V67 and all Mapper/S/H/
  V50/V66 states remain hash-identical and receive no updates.
- [x] Evaluate 477 common-usable windows (3,741 future queries). On new
  combined motion, V67 reaches 79.06/38.92/310.41/385.92 mm
  Mean/P50/P95/P99 and 86.24% selection accuracy; the old combined validation
  result was 108.75/45.44/351.96/560.57 mm and 83.57%. New rotation reaches
  41.92/13.51/265.62/347.85 mm, but only 68 windows from two sessions qualify.
- [ ] Active conclusion: do not tune V67 from these results. End-to-end input
  availability falls from 59.54% on the old paired dataset to 38.01% on the new
  one; spin is only 12.45% usable and two far combined sessions are almost
  empty. Diagnose PnP/history continuity and strict 32-event availability
  before changing trajectory or selector structure.

## 2026-07-29 fixed-6-mm observation recovery

- [x] Separate the apparent observation loss by camera profile on the nine
  unsealed disjoint sessions. Below 7 m, `wide_6mm` has 14,686 frames and
  98.19% valid solved observations; `precision_16mm` has 4,280 frames and only
  23.74%. Sampled bridge diagnostics show missing precision frames at
  `det=0, solved=0`, so this is not an F/V67 error or a PnP-only rejection.
- [x] Make single-focal native 6 mm the default in the shared autoaim config,
  Talos CLI, WSL launcher and ROS 2 bridge. Stage-3 capture also exports the
  setting explicitly and records it in `session_result.json`; 16 mm remains an
  explicit diagnostic opt-in, not a default path.
- [x] Add fail-closed random-trajectory admission: nominal distance at most
  6.5 m (0.5 m camera reserve), full reciprocal path plus the existing 0.10 s
  collection-command lead within the range envelope, at least 0.75 m forward,
  and at most 75 degrees horizontal yaw. Unsafe session 0276 is rejected;
  a safe combined manifest passes.
- [x] Complete a new independent 4.4 m / 9 rad/s / 8 s fixed-6-mm smoke
  capture. All 1,050 frames are `wide_6mm`; valid solved observation coverage
  is 100%, target-3 coverage is 99.90%, and empty coverage is 0%. All four WSL
  CTest targets pass. Do not add this smoke-only session to training data.

## 2026-07-29 fixed-6-mm generalization recollection

- [x] Generate a new-seed (`2026072911`) fixed-6-mm manifest with six spin and
  six linear-plus-spin sessions, 20 seconds each. Reject 18 unsafe combined
  draws before capture; every admitted complete path plus the 0.10 s
  collection-only gimbal lead stays inside the declared range/yaw/forward
  envelope.
- [x] Collect all 12 sessions locally through the locked Simulator Release/SDK.
  Every session succeeds on attempt one, the batch exits normally, and no
  simulator/bridge process remains. Preserve all 332.42 MiB of raw observation
  and exact-exposure truth assets.
- [x] Join and audit all 35,554 observation frames against exact exposure truth.
  Every frame is native `wide_6mm` with dual focal disabled. Overall valid-any,
  target-3, empty and valid-wrong-only rates are 98.82%, 97.83%, 1.18% and
  0.98%; target-3 coverage is 97.33% for spin and 98.36% for combined motion.
- [x] Rebuild the qualified v3/truth-history/observation/causal/observable/PnP
  S/F chain. The standard 60/20/20 split keeps three sessions out of this pass;
  all downstream builders keep `test_accessed=false`.
- [x] Evaluate the complete Mapper/S/H/V50/V66/V67 chain at one causal
  ballistic query per eligible window. Use the frozen upstream current visible
  armor range divided by the configured 22 m/s bullet speed as continuous
  query time. Produce separate distance/error scatter plots and 1 m-bin tables
  for rotation and combined motion; no weight, threshold or checkpoint may
  change.
- [x] Retain the clean-commit r4 result over 910 common-usable windows. Rotation
  reaches 65.72/25.98/180.03/189.51 mm Mean/P50/P95/P99 with 99.62% selection
  accuracy. Combined reaches 87.33/33.18/330.24/865.45 mm with 79.75%
  selection accuracy. All Mapper/S/H/V50/V66/V67 hashes remain unchanged.
- [ ] Await user review of the two distance/error plots and tables. The next
  decision must separate the low 25.90% PnP/S/F admission rate from the
  combined selector's 131 wrong choices; do not train the V67 residual or tune
  a distance threshold from this inspected diagnostic.

## 2026-07-29 observed-primary history admission

- [x] Audit all 112,448 history events. Every previously rejected event still
  has at least one PnP candidate; the 25.90% gate was rejecting a different
  actually observed plate because it did not match the truth rule's prechosen
  primary, rather than measuring detector availability.
- [x] Implement a new non-overwriting v2 paired dataset. Anchor q0 on the
  PnP-range-nearest member of the actual candidate set, associate the maximum
  coherent historical suffix using same/adjacent anonymous-handle transitions,
  one rotation direction and 20 mm switching hysteresis, mask older
  incompatible history, require at least eight active events, and rebuild all
  clean/PnP labels from the same q0 handle.
- [x] Feed S the actually associated observed handle set, with one primary and
  at most one adjacent secondary. Keep the window-local C4 shift and optional
  reversal; no physical slot or armor ID enters the loader/model schema.
  Inactive history is zeroed before composition so it cannot become a
  `-current` pseudo-observation.
- [x] Pass the focused compatibility/association tests and diagnostic chain.
  A two-session build retains 738/850 combined windows (86.82%); all 27,200
  events contain a PnP candidate, every usable history has at least eight
  events, and no usable window reverses switch direction. Old v1 loaders remain
  operational. No network weight has changed.
- [x] Build the complete nine-session v2 parent and S/F datasets from committed
  source. Common end-to-end admission rises from 910/3,514 (25.90%) to
  2,627/3,514 (74.76%): rotation is 994/1,460 (68.08%) and combined is
  1,633/2,054 (79.50%). Every retained window has a coherent active suffix and
  the same observed q0 source for history, candidates and future labels.
- [x] Close the final partial-history review finding: both frozen PnP mappers
  now form `dt` only when the current and previous events are active, so the
  first event after an inactive prefix receives zero rather than a spurious
  negative delta. The inactive-prefix regression and all 309 Stage3 tests pass;
  architecture and consumer-boundary checks pass.
- [x] Run the complete frozen ballistic-time evaluation from clean commit
  `ae20bb7`. All Mapper/S/H/V50/V66/V67 hashes remain unchanged. The expanded
  stream gives rotation 290.57/272.26/681.71 mm Mean/P50/P95 with 45.17%
  selection accuracy, and combined 235.54/126.84/804.15 mm with 48.74%.
- [ ] Discuss retraining scope before changing weights. The old strict subset
  still gives rotation 65.72/25.98/180.06 mm and 99.62% selection, while the
  newly admitted samples give 371.46/358.47/715.57 mm and 25.58%. Combined has
  the same split: old strict 81.84/27.90/336.10 mm and 78.21%, new-only
  336.40/281.62/895.04 mm and 29.41%. Partial histories and changed observed-q0
  roles are an unseen input/label distribution; selector-only or V67-only
  tuning cannot repair it.

## 2026-07-29 observation-only primary correction

- [x] Supersede the first v2 build/evaluation as an association diagnostic.
  Although every candidate came from actual PnP observations, the q0 and DP
  range costs still read truth-reprojected distance and therefore could not be
  reproduced by the inference observation stream.
- [x] Change q0 selection, ties, transition costs and switching hysteresis to
  use exposure-local PnP horizontal range only. Truth remains limited to
  offline candidate-to-label association and future supervision. Add a direct
  conflict test where truth range and PnP range select different plates;
  312 Stage3 tests and both repository boundary checks pass.
- [x] Remove truth-nearest future role selection from the observed-q0 path.
  Each exact future query now chooses only among its actual PnP candidates by
  exposure-local range; dense truth supplies the selected role's clean XYZ and
  signed unwrap/gate only. Sort by tau before unwrapping, mask one incoherent
  query without dropping the window, and never expose future PnP to forward.
  A two-session diagnostic keeps 733/850 windows, 89.19% of their positive-time
  queries and zero signed-step reversals.
- [x] Rebuild the complete parent/SF chain from clean commit `1cc6d1e` without
  overwriting r1. The accepted r2 chain retains 2,559/3,514 windows (72.82%):
  992/1,460 rotation (67.95%) and 1,567/2,054 combined (76.29%). Across the
  retained stream, positive-query retention is 79.70% for rotation and 90.25%
  for combined, with zero unmasked signed-step reversals. All 955 rejected
  windows are explained by a coherent suffix shorter than eight events (808)
  or an observed q0 that cannot seed any adjacent-only positive-time future
  label (147); q0 observation failure is zero.
- [x] Run the complete frozen exact-query evaluation on all r2 common-usable
  windows. Mapper/S/H/V50/V66/V67 hashes remain unchanged, no old training
  session overlaps the new captures, and test remains unopened. Rotation gives
  328.92/199.00/1124.99 mm Mean/P50/P95 with 55.80% switch accuracy; combined
  gives 204.63/97.79/743.76 mm with 68.79%. V67 changes the frozen V66 result
  by only about 1--2 mm and its residual magnitude is about 14 mm mean, so
  another final-position refinement is not justified.
- [x] The structural training boundary was approved under decision 159.
  The current F encoder compresses the available anonymous four-handle history
  into one identity-switching primary stream and invalidates same-handle local
  velocity at switches. The candidate next experiment is an explicit anonymous
  C4-equivariant vehicle MotionContext shared by trajectory and selection,
  while S and all accepted checkpoints remain frozen baselines. Do not launch
  training, reuse a stale truth-nearest ballistic label, or continue selector/
  V67 epoch tuning until this interface is accepted.

## 2026-07-30 anonymous multi-handle MotionContext training

- [x] User approved the structural change. Keep V50/V66/V67 as immutable
  single-stream baselines and keep Mapper/S/H frozen for the first causal PnP
  pilot. The new component is an explicit vehicle-level MotionContext, not
  hidden cross-handle transfer inside the old single-handle F definition.
- [x] Implement the fixed four-handle causal interface, C4/reflection
  structural tests, true-branch trajectory loss, ordered selector loss and
  short-suffix augmentation. Inputs may contain mapped anonymous history and
  frozen H relative q0 support only; physical ID, motion class, truth state,
  future PnP and session identity are forbidden from forward. Ten focused
  structural tests pass locally and on the server.
- [x] Run one non-overwriting CUDA r2 pilot on the user-selected RTX 3090
  server with fixed
  trajectory, selector and controlled-joint update budgets. Report rotation,
  combined, 8--15-event and 3+-history-switch slices; never select an
  intermediate checkpoint from validation. The fixed update-2100 endpoint
  completes in 416.2 s with frozen Mapper/S/H hashes unchanged and test
  unopened. Overall conditional/hard Mean is 207.63/212.74 mm; rotation is
  280.41/280.67 mm and combined is 178.17/185.26 mm. The 8--15-event slice
  remains 377.82/395.33 mm.
- [x] Do not interpret the final 61.44% exact signed-step accuracy as
  physical armor-role accuracy. `k` and `k+/-4` are one physical role in this
  model, final hard exceeds conditional by only 5.12 mm overall, and the
  3+-history-switch hard Mean is paradoxically 4.51 mm below conditional.
  Complete the no-training ballistic-time evaluation with exact-step and
  modulo-4 role metrics, one distance/error scatter and one table per motion
  state before changing another weight. The clean update-2100 evaluation keeps
  581/587 windows; six opposite-source jumps fail closed. Rotation hard/
  conditional Mean is 267.18/255.70 mm with 56.25% modulo-4 role accuracy;
  combined is 166.10/159.56 mm with 71.36% role accuracy. The frozen states
  remain unchanged.
- [x] Implement the independent v2 structure without changing the immutable
  v1 model or runner. Each anonymous handle is compacted to its actually
  visible event stream and same-handle velocity uses the full timestamp gap.
  A history-inferred three-way latent mixture predicts query-independent
  trajectory coefficients, evaluated by one shared learned continuous-time
  basis. Duplicate candidate rows are reconstructed from the same modulo-four
  role state, so `k` and `k+/-4` are exactly one trajectory.
- [x] Make modulo-four role the primary firing output. Exact signed crossing
  probability is normalized within each role and is auxiliary only; final XYZ
  is gathered directly from the selected role. Add equal-within-window loss,
  the existing equal motion/history-bin sampler, and fixed trajectory,
  selector, short-joint and frozen-trajectory recalibration stages. Twenty-one
  v2 contract/gradient/recovery tests and all 349 Stage3 tests pass; architecture and
  consumer-boundary checks pass.
- [x] Pass the fixed r2 CUDA smoke and corrected recovery exercise. Deliberate
  termination in recalibration leaves immutable update 150; resuming repeats
  update-175 and update-200 losses exactly and completes update 300 with
  `gradient_isolation_verified=true`, frozen Mapper/S/H hashes unchanged and
  test unopened. The 48-window capacity subset reaches 35.96 mm overall
  conditional/hard Mean and 100% modulo-four role accuracy; this is only a
  training-closure check, not a generalization result.
- [x] Complete the one fixed full r2 v2 pilot at update 2,400 in 372.1 seconds.
  The clean endpoint preserves Mapper/S/H hashes, verifies joint gradient
  isolation and keeps test unopened, but fails the accuracy gate: overall
  conditional/hard Mean is 217.38/229.04 mm and 8--15-event history is
  395.96/418.37 mm. Final selector recalibration raises hard Mean from the
  joint endpoint's 227.64 to 229.04 mm, so the fixed endpoint is retained for
  evidence but not promoted.
- [x] Run the v2 ballistic-time diagnostic and copy the two distance/error
  plots plus tables back through the official GitHub channel. Rotation
  conditional/hard Mean is 303.63/310.65 mm; combined is 154.33/173.66 mm.
  Frozen states remain unchanged and 581/587 windows are labeled, with the
  same six opposite-source fail-closed cases as v1.
- [x] Implement the bounded multiple-choice routing diagnostic. It hash-locks
  the rejected v70 endpoint, freezes Mapper/S/H, MotionContext and all three
  trajectory experts, derives one detached best-expert label per training
  window from true modulo-four-role future error, and optimizes only the
  history gate. Primary validation is hard argmax routing; soft mixture and
  oracle best-of-three remain explicitly secondary/noncausal. The fixed
  baseline expert is selected from train overall and then held fixed on
  validation. All 357 Stage3 tests and both repository boundary checks pass.
- [x] Complete the CUDA interruption/resume smoke and fixed 600-update full
  routing diagnostic. Recovery is deterministic and every frozen hash is
  unchanged, but the statistical gate fails: train/validation macro recall is
  95.52%/48.70%, overall oracle-gap closure is 5.08%, short-rotation closure is
  9.47%, and the trained router is worse than the train-selected fixed expert.
  The result rejects more gate epochs and post-hoc routing of the frozen v70
  experts; test remains unopened.
- [x] Evaluate the hard history router at range/22 m/s and copy the requested
  rotation/combined distance-error plots and tables through GitHub to
  `D:\仿真\runtime\stage3-evaluations\20260730-v71-history-router-ballistic-r2`.
  Rotation conditional/hard Mean is 306.39/305.71 mm and combined is
  153.46/173.92 mm, materially unchanged from v70. The same 581/587 windows
  are eligible and all frozen hashes remain unchanged.
- [x] Define and implement the independent v3 F interface as one continuous,
  translation-equivariant anonymous role field. Absolute current position is
  used only by the final `current + delta`; future-best expert IDs, latent
  experts, exact crossing heads and soft-role position mixtures are absent.
  The offline sampler balances motion, then session, then history bin, while
  session identity remains metadata and prefix dropout cannot cross bins.
  Twenty-one focused v3 tests and all 379 Stage3 tests pass.
- [x] Complete v72 local CUDA capacity, exact interruption/resume and the one
  fixed 1,200/600/300 endpoint. Resume from update 150 is bit-exact through
  update 200. The full endpoint finishes in 405.1 s with frozen hashes and
  joint isolation intact, but is rejected: train/heldout conditional Mean is
  65.01/228.52 mm and session-macro role accuracy is 85.07%/53.96%.
- [x] Generate the requested v72 ballistic distance-error plots locally at
  `D:\仿真\runtime\stage3-evaluations\20260730-v72-continuous-invariant-ballistic-r1`.
  Rotation conditional/hard Mean is 250.51/287.09 mm; combined is
  166.97/201.22 mm. The broad bodies and distance clusters reject a simple
  monotone range explanation; 581/587 labels are eligible and all hashes pass.
- [x] Complete and reject v73 v4-A as another cross-session failure. The fixed
  endpoint reaches 69.34/89.27 mm train conditional/hard Mean but
  230.85/247.35 mm heldout; rotation regresses to 301.88/320.07 mm while
  combined is 202.11/217.92 mm. Removing raw origins alone therefore does not
  create a stable motion law, and deleting phase information wholesale hurts
  rotation. More v4 epochs are not authorized.
- [x] Complete the fixed v5 endpoint and ballistic evaluation locally. The
  state/decoder isolation, frozen hashes and recovery contract pass, but the
  endpoint is rejected as a cross-session state-estimation failure: train vs
  heldout velocity error is 0.155/0.950 m/s and yaw-rate error is 0.46/6.984
  rad/s. Heldout conditional future Mean is 267.53 mm, while the same decoder
  supplied with truth state reaches 141.74 mm. The decoder is reusable enough
  for the next A/B; the history-to-4D state path is the current bottleneck.
- [x] Collect a fixed-6-mm multistate dataset that breaks the
  session-to-motion shortcut. Each session keeps distance/camera/environment
  fixed but applies 12 ACK-bound continuous random motion segments, including
  one stationary block. The consumer reuses one Scene Control session: the
  first command initializes it and later commands update only target motion.
  Every admitted sample must keep all retained history and all future queries
  inside one ACK half-open interval and pass constant velocity/yaw truth over
  the complete window. Motion class is taken from the active segment, not the
  session family; stationary blocks therefore remain stationary and are not
  silently relabelled as rotation/combined. A 2 us guard protects both segment
  boundaries against float timestamp reconstruction. Segment epoch/start/end,
  full-window truth and rule-query audit fields are preserved and checked
  exactly through observable-clean, paired PnP and PnP/SF derivatives. The
  2-session end-to-end smoke passed and the strict
  builder retained 60 rotation and 16 combined samples from 1-second blocks;
  formal blocks use 3 seconds. Scene Control updates are never retried inside
  one control session because v1 has no idempotency token; any failure discards
  that run and restarts the whole session under a new run/control identity.
  Before capture, freeze the exact 24-session/12-segment/3-second manifest SHA
  and 14/5/5 split hash with the formal capture contract. All 421 Stage3 tests
  and both boundary checks passed before the first formal capture. Formal v1
  was stopped after four accepted sessions and one interrupted fifth session:
  the still-running bridge could make `Get-Content -Tail 1` yield two complete
  truth records while the result was sealed. The complete v1 root is retained
  as protected diagnostic evidence and is not eligible for training. The
  consumer runner now stops the bridge writer before sealing and parses only
  the file-order last LF-committed truth record from a fixed-length snapshot.
  An unterminated append fragment is excluded; malformed committed JSON or a
  non-integer timestamp fails closed. The repaired source commit is `96461ea`.
  Formal v2 completed 24/24 sessions and all 288 segments; 22 sessions passed
  on attempt one and one session used two whole-session retries before its
  accepted third run. Validation found zero plan/hash/ACK/camera/process/port
  failures. The 26 raw runs are protected; only 24 canonical results are
  selected. Manifest/contract SHA-256 are `70f354e0ea75eb2...d4f50e` and
  `2dee2e1dff1fee3...f24259`.
- [x] Rebuild the formal v2 derivative chain under the deployed latest-32
  history contract and pass the predeclared segment-survival audit. Every
  validation session now retains common-usable samples in 11/11 active ACK
  segments; raw base rows checked=8,980, test shards/raw opened=0. Qualified
  derivatives are the non-overwriting base r3 plus truth/causal/clean/
  observation/PnP-SF r2 roots. All captures and earlier derivatives remain
  protected.
- [x] Complete the unchanged v5 control as v77. Heldout velocity/yaw error is
  0.492 m/s and 1.893 rad/s, conditional future Mean is 208.64 mm and the same
  learned decoder with truth state is 156.24 mm. The requested ballistic plots
  give rotation conditional/hard Mean 153.55/164.58 mm and combined
  180.89/201.58 mm. State and future gates fail, although both materially beat
  the prior v5 baseline; role accuracy is not the primary bottleneck.
- [x] Complete the sampler-qualified v6 r2 state-only gate. It is bit-exact
  with diagnostic r1 and fails six of the eight predeclared gates: overall
  velocity/yaw/normalized error is 0.3994 m/s, 1.6206 rad/s and 0.0650;
  combined is 0.6105 m/s and 2.3662 rad/s; speed>1.7-m/s combined velocity is
  1.1318 m/s; combined-11 normalized MAE is 0.1795. Only normalized overall
  error and overall yaw-sign accuracy pass. The protected r2 checkpoint SHA-256
  is `e30759ab...8782a5`, source commit is `eae8ef1`, sampler/frozen-future
  hashes match v77, and test remains unopened.
- [x] Reject further v6 tuning. The clean-observation counterfactual changes
  speed>1.7-m/s combined velocity/yaw error only from 1.132/4.141 to
  1.108/4.174, and every single v6 scale is already bad in that stratum. The
  failure is translation/rotation entanglement and magnitude contraction, not
  tanh saturation, one bad scale, PnP noise, pair-only evidence or insufficient
  external state dimension.
- [x] Complete and reject the qualified V7 r2 state gate. The exact fixed
  update-800 checkpoint is protected at
  `20260730-v79-v7-factorized-state-gate-r2`; SHA-256 is
  `ecec8cf915b507ddc5b84ee99a9a37c92dddd42eac5a0a328c79cc9fe9ff2ba7`,
  source commit is `256c1a3`, and test remains unopened. All eight gates fail:
  overall velocity/yaw/normalized error is 0.458 m/s, 4.127 rad/s and 0.1040;
  combined velocity/yaw is 0.649 m/s and 3.592 rad/s; high-speed combined
  velocity is 1.098 m/s; combined-11 normalized MAE is 0.2648; yaw-sign
  accuracy is 0.8187. Clean-observation and truth-yaw-conditioned
  interventions do not remove the failure. More V7 epochs or fine tuning are
  not authorized.
- [x] Complete and reject the bounded V8 rigid-flow structural screen. The
  checkpoint-bound six-arm aggregate is protected at
  `20260730-v80-v8-probe-aggregate-r1`; source commit is `3c5c704`, test is
  unopened and no full candidate is authorized. Joint is clearly strongest:
  its two-seed overall/combined/high-speed-combined yaw means are
  1.861/2.425/3.945 rad/s and combined velocity is 0.772 m/s. It passes every
  gate for seed 20260731, but seed 20260730 combined yaw improves 29.42%
  against a required 30%. Separated fails both seeds. The miss is not a bad
  high-speed tail: joint improves high-speed yaw by 40.52%; it is concentrated
  in the low-speed, history-32, all-three-pair-scales-available body.
- [x] Implement and run the V9 anonymous geometry--velocity paired-set
  structural screen. Train exactly two 200-update state-only candidates under
  seeds 20260730/20260731 and compare them to the existing same-seed V8 joint
  update-200 checkpoints. Every local handle edge must retain its endpoint
  geometry and velocity in one token; same-set pair edges remain local
  10/30/70-ms tokens. A permutation-invariant latent-query pool provides local,
  steady full-history and handle-only fallback experts; an observation-only
  router uses one scalar mixture per expert for the complete 4D twist, never a
  separate scale choice per coordinate. No ID, class, session, truth, future,
  q0 quality or analytic future decoder enters forward. Gradient-reachable
  state capacity must remain within 5% of V8 joint; permanently dead legacy
  V8 modules do not count toward the comparison.
- [x] The V9 r2 two-seed aggregate is valid and rejected at
  `20260730-v81-v9-paired-twist-aggregate-r2`; SHA-256 is
  `a163e0a51d15cb16babf5c36ac51ffdf138a9ce00937b07c372c8b9a9c3d3c00`,
  source commit is `640cb0d`, test is unopened and no full V9 is authorized.
  Seed 20260730 improves overall/combined/high-speed velocity by only
  1.10/10.48/9.65%; seed 20260731 regresses overall velocity 6.74% and overall
  yaw 11.35%. Broken pairing improves rather than worsens high-speed velocity
  in both seeds. The V9 200-update gate required both seeds: overall velocity
  improves at
  least 10%; combined and high-speed-combined velocity improve at least 15%;
  overall/combined/high-speed yaw regress no more than 5%; yaw-sign regresses
  no more than 0.01; and the low-speed<=1.2 m/s, history-32, pair3 combined
  core improves yaw by at least 10%. A validation-only intervention that breaks
  geometry--velocity pairing while retaining token support/type/scale and each
  within-stream, within-scale geometry marginal must worsen high-speed combined
  velocity by at least 10% or 0.15 m/s. Failure ends the hypothesis without
  extra updates; success alone authorizes a full V9 run.
- [x] V9 implementation preflight passes 513 Stage3 tests and both repository
  boundary checks. Its 1,536,217 gradient-reachable state parameters are 3.26%
  above the audited V8-joint reachable count of 1,487,688 (V8 optimizer total
  1,898,569). All 3,016 train/validation samples have recent local support;
  all 637 pair-free samples have handle fallback support. A formal BF16
  channels=96/batch=64 CUDA backward has finite loss, gradients on every V9
  state parameter, no frozen-future gradients and 859.3 MiB peak allocation.
  The full 750-sample diagnostic exactly reproduces the protected V8 control
  metrics and retains the 300 combined, 82 high-speed-combined and 149 core
  supports. Test remains unopened.
- [x] The first two V9 r1 runs reached their fixed update-200 checkpoints, but
  aggregation correctly stopped before any gate report because a diagnostic
  callable launched through `python -m` was recorded under runtime module
  `__main__`; the normal imported finalizer saw its canonical package module.
  Source/qualname hashes match, but r1 remains protected diagnostic evidence
  and is not promoted. `_callable_contract` now canonicalizes a `python -m`
  entrypoint through `__main__.__spec__.name`; after its clean commit, rerun
  both fixed seeds under new r2 roots and aggregate only those r2 artifacts.
- [x] Replace V9's mutually exclusive complete-4D experts with a V10
  paired-residual state estimator. One full-history anonymous handle set emits
  a single 3D velocity baseline. Every available pair1/pair2/pair3 event-scale
  bundle contributes an angular vote and a learned multiplicative
  geometry--velocity interaction; their aggregate emits yaw plus a planar
  rotation-compensation residual. There is no local/steady/fallback router,
  no per-coordinate scale choice and no complete-4D expert mixture. Pair-free
  histories use the handle baseline and handle angular fallback by deterministic
  support, not by class or learned identity. The six-field causal API and all
  frozen upstream/future boundaries remain unchanged. The final strict
  zero-preserving implementation has no paired learned query, affine
  normalization or linear bias: independently zeroing any geometry or
  kinematics projection makes the paired yaw vote and planar residual exactly
  zero even when common motion changes. It has 1,498,568 gradient-reachable
  state parameters, 0.731% above V8 joint. A formal CUDA BF16 batch-64 backward
  reaches every state parameter while all future parameters remain frozen.
  The full diagnostic retains 750/300/82/149/48 samples in the
  overall/combined/high-speed/core/pair1-2 groups and exactly replays the V8
  controls. Three independent read-only reviews are READY; 526 Stage3 tests
  and both repository boundary checks pass. Test remains unopened.
- [x] The fixed V10 two-seed screen completed and was validly rejected at
  `20260730-v82-v10-paired-residual-aggregate-r1`; both candidates reached
  update 200 from clean commit `d58f9e6`, all artifacts are hash-bound and test
  remains unopened. Each seed passes exactly 4/14 gates. Pair1/2 velocity does
  improve by 11.14/13.13%, and removing the paired planar residual worsens
  combined velocity by 20.74/30.91%, so the paired evidence is real. However,
  pair1/2 yaw regresses 64.78/70.44%, combined yaw regresses 37.15/49.24%, and
  high-speed yaw regresses 43.50/46.17%. Splitting support exposes the semantic
  failure: broken pairing actually improves pair1 yaw, while pair2 carries the
  misleading merged intervention pass. Independent local yaw votes are
  rejected; no extra updates or full V10 run are authorized. The screen used
  the protected same-seed V8-joint controls and original V9 gates, plus causal
  use requirements for high-speed broken pairing, pair1/2 performance and
  broken pairing, and the zero-planar-residual intervention.
- [x] Implement and preflight the replacement for independent local yaw votes:
  one learned global rigid-flow state constrained by the complete observed
  history. Exact event/scale/handle edges are factor messages, not local state
  predictions. A typed block-sparse causal decoder reconstructs already
  observed handle and pair displacements from the single inferred twist; two
  shared residual refinements update that same state. Pair messages are strict
  zero-preserving geometry--motion--time interactions, pair-free histories have
  an exactly zero pair branch, and the decoder cannot read the observed target
  displacement or current endpoint as a shortcut. The formal model has
  1,467,004 gradient-reachable state parameters versus V8's 1,487,688, within
  5%. Validation preflight retains 750 overall, 300 combined, 82 high-speed,
  149 core and 127/25/23/229 pair0/1/2/3 samples. The donor-time crossed-factor
  diagnostic selects 267/300 combined samples from 52 non-self donors and
  preserves both displacement identities to numerical precision. A real CUDA
  BF16 batch-64 backward is finite and reaches every state tensor while future
  modules remain frozen. Fourteen focused and 540 full Stage3 tests pass; all
  read-only design/finalizer reviews are READY. Test remains unopened.
- [x] The fixed V11 two-seed screen completed and was validly rejected at
  `20260730-v83-v11-global-flow-closure-aggregate-r1`, bound to clean commit
  `a11bdda`; both runs reached update 200, checkpoints and controls are
  hash-bound, and test remains unopened. Seeds pass 15/24 and 16/24 gates.
  V11 genuinely improves high-speed combined velocity by 14.65/23.21% and yaw
  by 22.54/24.59%, and pair1 velocity/yaw by about 20--24%. It nevertheless
  regresses overall velocity by 7.42/9.78%, pair3 yaw by 10.04/20.72%, and the
  low-speed full-history core yaw by 39.83/53.70%. Overall P50 velocity and yaw
  both regress while P95 often improves, so most aggregate benefit is tail
  trimming rather than a better body. No extra updates or future-position
  training are authorized. Both checkpoints and the failed aggregate are
  retained as protected evidence.
- [ ] Active: replace V11's shared four-dimensional refinement with a strict
  omega-first, typed, event-ordered closure estimator. Centered-handle and pair
  relative factors alone infer one signed q0 angular rate. That angular state
  and prior geometry predict rotational handle displacement; only the
  de-rotated common residual may infer translation velocity. Relative residuals
  may update omega only, common residuals may update velocity only, and the
  sole cross-state edge is omega-to-velocity. Preserve event order and
  per-support precision instead of averaging every correlated edge equally.
  Preflight must prove fixed-state pairing closure, separate omega/v refinement
  isolation, 2x2 common/relative source crossing, ramp/reversal equivariance,
  and nonzero intervention touch counts for pair0/1/2/3. Only then run the same
  fixed two-seed 200-update local screen; do not tune V11 or add epochs.
- [x] V7 implementation and preflight are complete. The model preserves the
  exact six-field API, common-ramp/C4/reflection invariance and the one-way
  angular-to-planar conditioning boundary. The formal finalizer reconstructs
  the full model strictly from the fixed update-800 checkpoint, binds its file,
  contract, validation, diagnostic, substage and module hashes to the manifest,
  rejects non-finite metrics, requires all four intervention groups and uses
  planar speed for the >1.7-m/s combined stratum. All 482 Stage3 tests and both
  repository boundary checks pass. Formal-capacity CUDA smoke completed 150
  updates at channels=96/batch=64 without OOM. Diagnostic cross-250 and an
  abrupt-stop cross-600 recovery both reproduced model, optimizer, scaler, RNG,
  validation, substage and branch-hash state exactly; the next item records the
  required current-verifier clean-commit rerun.
- [x] The clean-commit cross-250 recovery rerun passed at protected root
  `20260730-v79-v7-factorized-recovery-cross250-r4`: continuous and
  150-update-interrupted paths reached update 300 with exact model, optimizer,
  scaler, RNG, validation, substage, branch-hash and isolation state. The first
  formal V7 r1 then completed all 800 updates but finalization correctly refused
  it because the intervention path omitted the CUDA autocast context used by
  the main validation path. R1 is protected diagnostic evidence, not a formal
  result. Recomputing the fixed r1 checkpoint after the repair makes PnP speed
  and yaw distributions exactly equal in overall, rotation, combined and
  planar-speed>1.7 combined groups. Commit the precision-contract repair and
  rerun the complete deterministic gate at a new r2 root.
- [x] Historical v6 plan (superseded by the completed/rejected r2 gate): V77
  failure is strongest at high combined translation speed; replacing PnP by
  matched clean observations improves only about 10--15%, so the new state
  estimator uses causal non-overlapping 10/30/70/150/280-ms same-handle time
  bands, an FP32 unordered two-visible-armor relative-motion branch, bounded
  learned robust consensus, effective-support-weighted handle pooling and
  availability-aware scale fusion. Its only deployed output is
  the unchanged four-vector `[vx,vy,vz,yaw_rate]`; no ID, session, class,
  absolute range/current position, q0 geometry/quality, future or truth enters
  forward; the state API accepts exactly six causal observation fields and its
  loss never executes the decoder or reads future labels. The first gate
  freezes Mapper/S/H and loads the exact v77-update-800 decoder/selector before
  comparing equal-budget validation state metrics. It is one complete
  800-update state-only artifact, not an interrupted 2,900-update run. Only a
  passing state gate may start the predeclared trajectory and
  selector stages, followed by the two simple rotation/combined distance-error
  scatter plots.
- [x] The historical V6 implementation had 455 passing Stage3 tests, both
  repository boundary checks and a bit-exact 150->200 interruption/resume
  smoke; it is superseded by the completed/rejected V6 r2 result and the V7
  implementation above. Mapper/S/H, v77 decoder/selector, test, export and
  online fire control remain frozen. Use
  `D:\Anaconda\envs\yolov8\python.exe`; the rented RTX 3090 stays powered off
  but retained and must not be released.
- [x] V8 implementation preflight passes 501 Stage3 tests. A real RTX 4060
  CUDA backward at formal channels=96/dropout=0.05 is finite for all three
  arms. The ramp draw is isolated from architecture-dependent dropout RNG and
  depends only on run seed plus update; its dependency source is hash-bound.
  The six-run aggregate restores the intentionally omitted interruption field,
  revalidates checkpoint/manifest/source identity, requires the exact 3x2 arm
  matrix, and reports total state parameters without mislabelling unreachable
  legacy context parameters as active capacity. The rented RTX 3090 remains
  powered off and retained.
- [x] The first v6 r1 state run reached all 800 updates before the parent could
  terminate it and is retained as diagnostic-only because the final review
  found that "same sampler" was not yet machine-bound. It failed six of eight
  gates but improved v77 overall velocity/yaw/normalized error from
  0.492/1.893/0.0791 to 0.399/1.621/0.0650; combined high-speed velocity is
  still 1.132 m/s and combined-11 normalized MAE is 0.1795. The sampler class,
  state cells, history bins and prefix dropout are now semantic-hash-bound to
  v77 source commit `39a2328`, and finalization compares exact sampler strategy
  and support. After the new clean commit, rerun the same fixed gate only at a
  new protected r2 root; do not promote or overwrite r1.
