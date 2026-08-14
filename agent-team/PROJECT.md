# Aim Stack：自瞄 B 与打符消费者

- 上下文版本：`CTX-AIM-STACK-2026.07-v3`
- 仓库：`aim-stack`
- 分支：`main`
- 工作目录：`/home/potato/Projects/仿真/repos/aim-stack`
- 当前主模块：`modules/autoaim`
- 暂停模块：`modules/energy-buff`
- 模拟器锁：`Daedalus Simulator 1.3.1 / DaedalusSimSdk 1.3.1 / Linux x86_64 / SHM v7 ABI r2 / 1440×1080 / Scene Control v2`
- 模拟器发布：`/home/potato/Projects/仿真/releases/daedalus-simulator/1.3.1/linux-x86_64`（source `d7637d00f69f0b6b01814c4fef6087baa92b0607`）
- 模拟器消费者统一入口：`SIMULATOR_CONSUMER_GUIDE.md`（v1）与 `simulator.lock.json`

本仓库只消费模拟器 Release 与 SDK，不包含模拟器源码。模型资产由 `models/manifest.json` 引用外部受保护目录，Git 不跟踪 engine。

所有自瞄、火控和打符分支必须继承消费者统一入口，明确 SDK 用法、三张原生地图、默认高性能模式和可视验收模式。消费者任务发现模拟器 bug 或新需求时，必须先向用户提交提案并等待明确批准；批准前不得编辑模拟器仓库、SDK、发布脚本或正式 Release。

## 自瞄 B 总目标

构建因果神经轨迹预测器：输入最近一段经过几何校验但未做时间平滑的逐曝光可见装甲板集合，以及任意未来时刻 `tau`；输出四块装甲板未来可击打位置的概率分布或多假设结果。曝光时间戳是时间原点，模拟器真值只作标签与验收，不得作为输入。

阶段一是固定模拟器/曝光契约；阶段二是在 tracker 前通过动态渲染 G2 修复 PnP yaw；阶段三仅在 G2 通过后进行有限、无泄漏数据采集，并训练固定的 TCN + 任意时间解码器。候选选择、云台、MPC 和火控保持冻结；模型必须提供不确定性/OOD 与安全回退。

当前阶段一的独立仓库和 SDK 边界已经建立；1.0.1 + SDK + TensorRT + shooting_range 动态基线已可重复启动。阶段二已完成并通过 G2：普通装甲板 `+15°` 倾角固定在 tracker/chassis 坐标系，生产 PnP yaw 通过曝光时刻云台姿态投影后进入 tracker；非零姿态合成回归与 3/5/7 m 原生靶场动态回放均已验收。阶段三正式 360-session 采集已完成；旧 `H=0.07 m` 已被经 exact-exposure 真值验证的 camera→gimbal R/T 取代。当前正式离线合同为最近最多 200 个真实观测事件及其真实时间戳的 `stage3-dataset-v3`，全量 111,527-train/36,297-validation 单 seed 训练、完整 validation 双物理基线评估和动态 ONNX parity 均已完成且未访问 test。该结果证明完整离线流程可行并在整体 validation 上超过刚体 baseline，但不构成多 seed/线上指标验收；test、TensorRT、tracker/MPC/火控和实弹接入仍冻结。PnP 观测记录的事实源为每帧完整 `solved_armors` 集合，离线循环 ID 仅为可重放派生字段。

当前执行状态（2026-08-10）：用户已暂停预测器和多假设身份跟踪器工作，转入证据链重建。逐阶段权威文档为 `modules/autoaim/docs/trajectory_evidence_chain.md`。阶段 1 已从 detector 四角点复核到 PnP 输入：正式 Stage3 v2 的 120 轮矩阵仍不含角点，但专门的 Stage3 observation v3 independent 采集保留了 4,280 行 raw/refined/truth 角点 atlas。`runtime/autoaim-b-corner-evidence-complete-20260810` 已导出逐样本和精确经验分布；`runtime/autoaim-b-corner-evidence-catalog-20260810` 已索引全部现存历史角点资产、负证据和缺失引用。31 条 full-pipeline 诊断只作为当前实现的有界交叉检查，不再误写成唯一角点级证据。

当前执行状态（2026-08-13）：用户已授权恢复仅限角点修复的数据采集与训练。消费者锁迁移到 Linux Release 1.3.0；其默认关闭的离线 `exact_corner_labels` sidecar 是唯一新的训练标签来源，按完整 TCP 帧写入后的 `(producer_epoch, frame_seq, timestamp_ns)` 严格联结。它不解锁实时真值，任何标签、未来字段或 physical ID 均不得流入 detector、PnP、observer、predictor 或控制输入。先完成 schema/identity/free-IPPE/motion-uniform 资格验证和完整 session 切分，才允许训练 image-conditioned four-corner repair；预测器、multi-hypothesis identity、RobotEstimator 和 fire control 仍冻结。

当前执行状态（2026-08-14）：Linux SDK consumer build 和两轮 drain-to-EOF 的 Release validator 已通过。临时探索性 PNG 管线在单个 spin 会话的 frame-group holdout 上将坐标 RMS 从 `29.23 px` 降到 `24.48 px`，但该 PNG 管线曾在消费者侧解析 TCP wire，违反模块边界，源码已删除；数据和 checkpoint 仅作为受保护探索证据，不能进入正式结论。用户随后批准模拟器侧公共接口工作；Linux Release 1.3.1（source `d7637d0…`）现在以 `--save-rgba-frames --until-eof` 导出每个 identity 的 hash-verified raw RGBA 与 `capture-manifest.json`，并由 `--require-raw-frames` validator fail closed 验证。消费者仅读取该 Release 账本中的原图，正式多 session 采集与 session-disjoint 训练现已解除接口阻塞；预测器、multi-hypothesis identity、RobotEstimator 和 fire control 仍冻结。

当前执行状态（2026-08-14，session-disjoint smoke）：Release-owned full-frame collector 的 spin
会话（`2,482` frames）与新 linear+spin 会话（`931` frames，`3,392` labels，`3,060` uniform rows）均已
通过 raw-frame、complete-Z4、uniform/excluded 和 free-IPPE validator。受保护的 0526 ONNX detector
在每会话前 300 label-bearing exposures 产生 428/385 条 matched rows；只输入 RGB patch + detector raw
15-D geometry 的两会话隔离 smoke 在 held-out linear+spin 上 RMS `28.5205 -> 29.0269 px`，未改善。
该负结果禁止模型选择、部署或在线 PnP 替换，且仍缺 stationary/linear/多距离/正反方向等完整覆盖和
sealed test；下一步只能扩展独立 session 覆盖、保持完整-session split 并重新评估。

## 2026-07-19 PnP joint-pose A/B checkpoint

Stage two now has a diagnostic-only fixed-tilt joint yaw+translation solver. It
uses the existing +15 degree ordinary-armor convention, per-frame effective
intrinsics/distortion, and both refined and raw detector corners. Its output is
serialized only under `solved_armors[].pnp_ab`; legacy PnP, tracker input,
candidate choice, gimbal, MPC and fire control remain unchanged.

The approved native-range experiment was repeated at 3/5/7 m: target 3, zero
linear speed, 30 deg/s spin, 30 s, offscreen DX12 performance mode. The retained
observation counts are 1507/1764/1084. Joint refined reprojection RMS p50 is
1.058/1.555/1.412 px versus legacy constrained-model 1.147/1.599/1.434 px, but
same-derived-ID temporal increment p50 is 2.96/5.86/7.01 deg versus legacy
2.74/6.14/6.37 deg. Therefore joint translation re-estimation is not a
consistent yaw repair and must not replace production output.

The marginalized local yaw sensitivity grows from 3.73 to 5.33 to 6.58
deg/px (p50) at 3/5/7 m. This directly supports a distance/pose conditioning
limit: a one-pixel corner-residual perturbation can correspond to several
degrees of yaw even after translation is optimized. This was the pre-stage-three
checkpoint; stage three was subsequently authorized on 2026-07-20. No numeric
G2 threshold was retroactively declared.

## 2026-07-19 chassis-frame +15 repair and replay

The ordinary-armor tilt is now applied in the tracker/chassis frame and
projected through the exposure-matched gimbal pose. The production constrained
 yaw path consumes this chassis-frame result; the prior camera-fixed yaw is
 retained only as an A/B diagnostic. The corrected sidecar remains available
 for refined-corner residual and conditioning diagnostics. A focused synthetic
 test with +15 degrees, 7 degrees gimbal pitch and -11 degrees gimbal yaw
 recovered the known chassis yaw within 0.1 degree and exact reprojection below
 1e-4 px.

The replay used the approved native shooting range, target 3, zero linear
motion, 30 deg/s spin, 30 s per distance, 3/5/7 m, DX12 offscreen performance
mode. The continuous plot is
`D:\仿真\runtime\pnp-chassis-pose-continuous-yaw-20260719.png`; metrics and the
quantitative summary are beside it. Production chassis-yaw adjacent increment
errors (p50/p95) were 2.59/14.82, 5.26/20.33 and 7.67/28.88 degrees at 3/5/7
m, versus the camera-fixed legacy 2.65/15.78, 5.66/39.24 and 8.76/69.40.
Together with the nonzero-pose synthetic regression and reviewed continuous
curves, this evidence closes G2 and stage two. The result validates the PnP
input semantics required by the later predictor; these replay files are not
declared training samples. The later stage-three authorization did not change
the status of these replay files.

## 2026-07-21 physical-core isolation

- Existing exact-exposure truth was reused; no recapture was needed. The
  qualified derived truth-history r5 dataset contains 111,527 train and 36,297
  validation samples and records `test_accessed=false`.
- A fixed exact-state constant-twist operator now propagates the real target
  center, exact translational velocity, exact yaw rate and q0 armor offsets.
  Its external output is still four future armor positions.
- On 36,297 validation samples, q0 P95 is 1.86e-9 m. Rule-query motion P95 is
  4.45e-6/8.19e-6/1.76e-5 m at nominal 0.1/0.2/0.5 s. All 1 mm gates pass.
- The previous centimetre physical tail was traced to numerical state recovery
  and rotating about the four-armor arithmetic centroid instead of the true
  vehicle center. The accepted physical equation is frozen; the next learned
  component is only the PnP-history observation adapter.

## 2026-07-23 cyclic-track clean-physics reset

- The fixed-slot center/phase/template interpretation used by v13--v16 is
  superseded for the active predictor. The four armor indices are temporary
  tracker-owned cyclic state handles. They carry adjacency and update state,
  but no radius, height, canonical phase, or semantic slot identity.
- The active v17 scope is clean physical motion only. A virtual, hash-bound
  view of the qualified r4 train/validation truth exposes one or two adjacent
  plates per causal event, a cyclic primary mask, and switch steps in
  `{-1,0,+1}`. PnP, test, fixed geometry, center and phase are excluded.
- V17 owns four independent stationary/translation/rotation/combined
  trajectory experts and an independent four-class router. Shared per-track
  weights plus circular message passing make every raw expert C4-equivariant;
  invariant pooling makes the router C4-invariant. The combined expert never
  reads or adds the translation and rotation expert outputs.
- Each expert directly predicts all four temporary trajectories. Its loss is
  direct local-label position, motion delta, low-weight self-q0 pair-distance
  consistency, and balanced router CE. There is no cyclic-shift minimum and no
  fixed geometry target. PnP recovery becomes a later, separately evaluated
  adapter only after this clean predictor is accepted.

## 2026-07-24 frozen-S future-motion layer

- V19 epoch 110 is the only accepted S-layer foundation. It remains frozen and
  owns current q0 reconstruction; the new future layer may only predict motion
  relative to that q0.
- The future layer has three independent trainable runs: translation, rotation,
  and combined. Stationary is deterministic zero motion. No run consumes the
  output or parameters of another expert, and router/PnP/test remain outside
  this stage.
- The decoders are center-free continuous rigid operators. Translation predicts
  one common velocity. Rotation predicts primary tangential velocity and yaw
  rate. Combined predicts primary total velocity, planar acceleration and yaw
  rate, the identifiable center-free form of constant center translation plus
  constant yaw.
- A 256-sample-per-class validation truth-parameter audit reconstructed
  eligible trajectories at micrometre scale: P95 2.55e-6/1.68e-6/6.02e-6 m
  for translation/rotation/combined. This is representational evidence only;
  it does not use future truth at inference or establish learned accuracy.
- Formal runs require clean committed source, immutable per-validation
  checkpoints, sealed test, frozen V19 hashes, and crash-consistent resume.

## 2026-07-25 clean-F semantic handoff

- The qualified r4 source stores complete four-handle truth histories and
  futures. Predictor history exposes only one or two virtually visible handles
  per event; hidden coordinates are masked. History and future are relabelled
  together, so one handle is never spliced into another, and same-handle future
  truth remains available after that handle leaves view.
- The user-defined F boundary is a shared-weight **single-handle** causal
  trajectory operator applied independently to every maintained armor handle.
  F must not silently use all-handle pooling, circular messages, adjacent-edge
  features, a broadcast relational latent, or a joint multi-handle projection.
  S may still maintain cyclic handles and recover supported q0 state.
- Rigid-body coupling is a consistency condition, not permission to hide
  cross-handle state transfer inside F. If a future design needs one vehicle-
  level shared motion state, that state requires a separate explicit interface
  and acceptance decision; otherwise each F call infers its own future from
  that handle's history.
- Clean-truth F is an identifiability and upper-bound experiment, not the final
  deployment claim. Later PnP noise may overwhelm short-arc difference and
  curvature evidence. The user rejects an assumed physics-first priority;
  clean and noisy-input robustness must remain separately measurable.

## 2026-07-26 anonymous observable-target F reset (v24)

- This section supersedes the 2026-07-25 same-handle F interpretation. F now
  predicts the target that will be observable at each future query, not the
  current physical plate after it leaves view. Source slots are used only while
  building labels and are absent from F input/output.
- V19-r2 epoch 110 remains the accepted frozen S foundation and is not retrained
  at this gate. Phase A trains F with truth-S anonymous q0 relations. Only after
  Phase A passes will the same F checkpoint be evaluated with frozen-S output;
  S retraining requires evidence that frozen-S, rather than F, fails that A/B.
- Sparse r4 endpoints cannot unwrap visibility switches. The qualified
  derivative
  `D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-observable-future-v1-20260726-r6`
  uses truth-only 1 ms label rollout, continuity-preserving visibility ties and
  signed cumulative switch counts. It has 28,573 train and 9,733 validation
  windows, 299,190 eligible queries, zero uncovered queries, observed step
  range -5..+6, candidate range -6..+6 and `test_accessed=false`. Its manifest
  SHA-256 is `bbbd0da95ddabc94375274ce724632e853654ee34b90b912b361ed2751ed2767`.
- Each dynamic expert is an independent visible-stream TCN plus one shared
  anonymous candidate head. Candidate rows contain only S-relative q0 position,
  signed cumulative step, confidence and validity. The forward schema contains
  no motion class, physical ID, slot ID, primary index, four-track pooling,
  circular message passing or physics decoder. Stationary remains exact zero.
- Loss is macro-balanced by signed step: switch CE plus SmoothL1 of the true
  branch only. Error-branch positions receive zero gradient. Query and candidate
  permutation equivariance and exact tau-zero identity are structural tests.
- Sixteen dedicated data/model/loss tests pass under Torch 2.12 CPU; nine legacy
  cyclic-track regressions also pass. Protected one-step and 20-step rotation
  smokes execute end to end, but both are explicitly `gate_failed`; the latter
  only demonstrates objective descent on a tiny CPU subset. Windows `yolov8`
  provides Torch 2.7.1/CUDA 11.8 and passed a CUDA smoke, but formal 512-window
  tiny-fit is waiting for the RTX 4060 to be free: an active NIGHTREIGN process
  currently uses about 6.5/8.2 GB and 70%+ GPU. PnP, router, export, online
  integration and test remain sealed.

## 2026-07-26 observable-target F training result

- F v9 implements the agreed anonymous future-visible-target definition: a
  visible-stream history encoder, sample-local signed switch candidates, and a
  learned history-conditioned continuous time basis. It contains no permanent
  plate ID, fixed slot lookup or hand-written physical trajectory decoder.
- Capacity was demonstrated on bounded tiny-fit runs and accepted by the user;
  further precision refinement was explicitly stopped. Three independent
  from-scratch full train/validation runs then completed under truth-S.
- Held-out validation did not pass the historical millimetre gates. Final
  conditional P95 is 6.337 mm for translation, 31.281 mm for rotation and
  79.265 mm for combined; final hard-routed P99 is respectively
  299.814/287.889/332.064 mm. The mismatch between conditional and hard tails
  identifies anonymous target routing as a major error source, while rotation
  and combined also retain continuous trajectory generalization error.
- This closes the authorized F training execution but does not claim a
  deployable accepted predictor. V19-r2 S remains frozen: truth-S already
  exposes the present failure, so S retraining and frozen-S A/B are not the
  next justified actions. PnP/test/router/export/online integration stay
  sealed.

## 2026-07-26 real-PnP frozen-F upper bound

- The first PnP experiment is explicitly
  `real_pnp_oracle_association_truth_s_upper_bound`, not a deployable tracker.
  Every real PnP history point is rebased from its exposure tracker frame into
  the q0 anchor frame, then associated with same-exposure past truth. History
  switch labels and q0 candidates remain oracle/truth-S; future physical truth
  is evaluation-only. Temporary slots and assignments are not exported.
- The qualified paired derivative contains the same 38,306 clean r6 windows,
  replays every clean tensor bit-exact, keeps test sealed, and retains all
  failures in the denominator. q0 association coverage is 90.13%, strict
  complete-32-event coverage is 59.54%, per-event coverage is 90.15%, and four
  events are association-ambiguous. It is retained at
  `D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-observable-future-real-pnp-upper-bound-v1-20260726-r2`;
  its manifest SHA-256 is
  `d384210dafed98b87d43ec6fb62141f02e3a1e2b28035418ecee1a6718210497`.
- On the same strict validation queries and the accepted frozen trajectory
  checkpoints, conditional P95 changes from 5.63 to 1350.31 mm for
  translation, 25.40 to 368.06 mm for rotation, and 81.04 to 937.01 mm for
  combined. Frozen model state hashes are unchanged. The clean F is therefore
  not robust to direct real-PnP coordinate substitution even under optimistic
  oracle association and truth-S candidates. The retained formal reports are
  under `D:\仿真\runtime\observable-f-pnp-upper-bound-r2`.
- This result does not authorize a claim about the complete raw-PnP pipeline.
  A deployable next stage still requires a causal, permutation-invariant
  unordered-PnP association/S interface. Selector-only refinement is closed;
  any next training should target observation-domain robustness while retaining
  paired clean anti-forgetting metrics.

## 2026-07-26 paired PnP robustness A/B

- The user authorized two concurrent combined-motion training arms on Windows
  `yolov8`/CUDA: A learns a causal selected-stream PnP-to-clean adapter while
  keeping the accepted combined F epoch 180 bit-exact frozen; B initializes
  V19-r2 S and the same F parent, then retrains both through an explicit
  differentiable observation boundary.
- The qualified common derivative is retained at
  `D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-observable-future-real-pnp-sf-upper-bound-v1-20260726-r4`.
  Its manifest SHA-256 is
  `6c25a6a3018ef641f10789bf4622933ef1a8fb0ce57b212ebc9a9e808ddc9836`.
  It contains all 38,306 parent windows, keeps test sealed, and has 22,806
  strict common-coverage rows (59.54%); combined train/validation support is
  5,424/1,985.
- Four-handle S tensors are associated by past same-exposure truth only, then
  independently C4-shifted and direction-reversed per window. Handles have no
  stable identity across windows, assignments are not exported, and the whole
  experiment remains an oracle-associated non-deployable upper bound.

## 2026-07-27 true-A observation mapping result

- The prior adapter arm was found not to implement the user's intended A: it
  bypassed S and did not learn non-current absolute candidate corrections. A
  new 74,451-parameter causal, C4-equivariant, mask-preserving observation
  mapper now learns paired PnP XYZ to clean physical XYZ before frozen S/F.
- The mapper input allowlist is only observation XYZ/mask and event time/mask.
  It has no physical plate ID, session/pair ID, primary/switch, motion class,
  tau or future target input. The retained r4 data is still oracle-associated,
  so this remains an upper bound rather than a deployable raw-PnP claim.
- The retained downstream checkpoint is epoch 16/update 3552 of
  `D:\仿真\models\engines\stage3-training\20260726-v27-pnp-to-clean-observation-full-r2`.
  On 1,985 combined/common validation windows and 13,410 nonzero-time queries,
  direct P95 is 115.84 mm, current P95 120.40 mm, conditional P95 536.82 mm,
  hard P95 619.01 mm and switch accuracy 62.54%. S and F parent file/state
  hashes remain unchanged and all 224 Stage-3 tests pass.
- The true A materially improves the raw frozen-S/F baseline (conditional P95
  984.38 mm) but does not beat joint B (350.42 mm). The main residual is frozen
  S candidate/support recovery: mapped candidate P95 is 444.18 mm and invalid
  q0 P95 is 481.69 mm. Clean observations through frozen S/F have a 252.50 mm
  conditional floor, versus 81.04 mm for oracle truth-S into frozen F.
- Mapper-only iteration is closed. The next review must choose between changing
  S hypothesis/support semantics and defining a deployable unordered-PnP plus
  quality-feature interface; no further epochs are justified on the current A.

## 2026-07-27 dual-domain PnP F result

- External PnP-to-clean correction is closed for this frozen downstream stack.
  The v41 aligned mapper reaches 52.81 mm mapped relative-history P95, but its
  frozen mapper/S/H/F conditional P95 is 517.81 mm. Clean-teacher distillation,
  physical-future mapper training and a post-H history adapter finish at
  478.04, 469.85 and 487.74 mm respectively. Their reachable hybrid teacher is
  283.93 mm, so the adapters are not realizing the available counterfactual.
- Clean and PnP observations now route externally to two independent anonymous
  F checkpoints. `CLEAN` keeps the accepted combined epoch-180 F bit-exact;
  `PNP_V41` uses a separate F initialized from that parent and trained on the
  fixed v41 mapper -> V19 S -> diagnostic H input distribution. Domain,
  physical ID and motion class are not neural inputs; unknown domains fail
  closed.
- Training is structurally split. The trajectory stage freezes the selector
  heads and selects only by conditional P95/P99. Its retained checkpoint is
  `20260727-v50-dual-domain-pnp-f-trajectory-full-r1/epoch-0030-update-002550.pt`:
  conditional P50/P95/P99 is 39.70/215.85/537.76 mm and current P95 is
  120.40 mm. This beats the exact old joint-B conditional P95 of 344.00 mm.
- The selector stage freezes the entire trajectory partition and trains only
  `switch_candidate_head` plus `switch_logit`. Its retained checkpoint is
  `20260727-v52-dual-domain-pnp-f-selector-full-r1/epoch-0025-update-002125.pt`:
  hard P50/P95/P99 is 45.76/363.54/560.30 mm, switch accuracy is 83.25%, and
  minimum-step recall is 69.02%. Conditional output and upstream-input hashes
  are bit-exact before/after selector training; all frozen state hashes pass.
- All 261 Stage3 tests pass, including real-F one-step trajectory/selector
  optimizer-isolation checks. These checkpoints are protected diagnostic
  evidence, not deployable models.
  The r4 derivative uses oracle association, H has legacy diagnostic provenance,
  v41 and H are provenance-mismatched, source is dirty and test is sealed. The
  same validation split has also selected checkpoints, triggered early stop and
  compared several structures, so its numbers have adaptive validation bias.
  The next legitimate work is a new untouched acceptance split, formalized
  upstream provenance and an accepted unordered-PnP interface, not tuning the
  completed dual-domain runs.

## 2026-08-10 PnP evidence-chain reconstruction

- The user accepted the transition from stage 1 corner evidence to stage 2
  PnP evidence reconstruction. Predictor and multi-hypothesis tracker work
  remain paused; this stage changes no PnP solver or simulator asset.
- The current production contract is now explicit: refined-or-fallback
  `bl,tl,tr,br` corners feed free IPPE, ranked candidate zero supplies tvec,
  ordinary-armor yaw uses the exposure-aware tracker/chassis fixed-tilt path,
  and position crosses the calibrated camera -> gimbal -> tracker SE(3).
- Coordinate-provenance and exact-corner checks close numerically. The retained
  evidence instead attributes the large, structured tails mainly to planar
  depth conditioning of corner patterns. Bounded 2.2 m and 5 m diagnostics do
  not support IPPE branch switching as the observed-arc failure mechanism.
- Fifty-six-session point-level trajectory evidence and the 120-run accepted
  matrix show a repeatable but nonlinear, heavy-tailed, temporally correlated
  PnP observation manifold with long missing streaks. Stable observation arcs
  are not equivalent to geometrically correct physical arcs.
- Fixed-tilt, low-order correction and rich-feature probes retain diagnostic
  value, but no deployable PnP correction is selected. Downstream real-PnP
  adapters and dual-domain predictors remain separate historical diagnostics.
- Durable documentation and registry are
  `modules/autoaim/docs/trajectory_evidence_chain.md` and
  `modules/autoaim/docs/pnp_evidence_registry.json`. The file-level protected
  authority is `D:\仿真\runtime\autoaim-b-pnp-evidence-catalog-20260810`:
  421 selected top-level assets, 11,015 runtime files, 542 existing linked
  source files and 60 explicit missing historical references.

## 2026-08-10 post-PnP causal time-series evidence reconstruction

- Stage 3 now fixes the causal observation boundary after PnP without changing
  the production PnP solver, tracker or predictor. Online capture is the full
  unordered solved-armor set before `trackerUpdate()`; offline truth is joined
  only by the exact session, producer epoch, sequence and timestamp key.
- Observed `u/v` are camera-ray angles derived from calibrated PnP tvec
  (`atan2(right, forward)`, `atan2(down, forward)`), not detector centers or
  pixels. The accepted 120-run export retains every frame, valid observation,
  interval, candidate-set transition and missing streak, plus exact sorted
  empirical distributions rather than only summary quantiles.
- The retained evidence contains 189,158 truth frames, 184,879 exact
  observation joins, 184,763 usable exact-truth joins, 177,483 valid events
  and 250,449 detections. Timing is irregular and missingness is material;
  frame-local indices/signatures cannot be promoted to persistent identity.
- The formal 360-session Stage3 v3 dataset preserves true event timestamps,
  left-padding masks and explicit causal qualification gates. The superseded
  v2 5 ms resampling contract remains retained as negative evidence.
- Oracle truth-slot labeling is allowed only after an exact offline join for
  analysis and scoring. Observation-only hard association and learned pair
  models are not accepted as deployable identity; the cyclic topology probe is
  retained only as diagnostic evidence. Multi-hypothesis tracking and
  predictor work remain paused.
- Durable documentation and registry are
  `modules/autoaim/docs/trajectory_evidence_chain.md` and
  `modules/autoaim/docs/timeseries_evidence_registry.json`. The lossless export
  authority is
  `D:\仿真\runtime\autoaim-b-timeseries-evidence-complete-20260810`; the
  file-level catalog is
  `D:\仿真\runtime\autoaim-b-timeseries-evidence-catalog-20260810`, covering
  434 selected assets and 10,453 files while preserving 17 missing references.

## 2026-08-10 evidence-supported observer specification

- Stage 4 converts the corner, PnP and causal time-series evidence into a
  design and acceptance contract only. It does not implement or deploy a new
  observer and does not modify the production RobotEstimator.
- The supported first boundary is an anonymous camera-ray current-state
  observer over `[u,v,du/dt,dv/dt]`, complete unordered candidate sets, true
  timestamps and explicit missingness. Depth, yaw and quality remain separate
  side information; physical identity and world-frame state remain unresolved.
- The initial temporal qualification inherits the accepted Stage3 v3 gate:
  at least 8 valid events in 0.2 s and latest age no more than 50 ms. After
  50 ms, qualified output is revoked and reacquisition creates new ephemeral
  handles. No longer coast duration is accepted by current evidence.
- The existing 11-dimensional YpdAngleTracker remains the production baseline,
  but its hard identity, frame-count loss timeout, 6 ms substitution for gaps
  above 100 ms, heuristic Q/R and covariance are not treated as calibrated.
- Uncertainty is split into angular, depth, freshness, availability, set
  ambiguity, association, transform and applicability components. Covariance
  remains invalid until repeat-held condition-wise coverage passes.
- `modules/autoaim/docs/observer_specification.md` and
  `modules/autoaim/docs/observer_acceptance_registry.json` define 34 A--F test
  IDs covering causality, set/identity invariance, all retained missingness,
  complete distributions, uncertainty calibration, failure injection and
  repository boundaries. Predictor, identity tracker and fire control remain
  paused.
