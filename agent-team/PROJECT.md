# Aim Stack：自瞄 B 与打符消费者

- 上下文版本：`CTX-AIM-STACK-2026.07-v3`
- 仓库：`aim-stack`
- 分支：`main`
- 工作目录：`D:\仿真\repos\aim-stack`
- 当前主模块：`modules/autoaim`
- 暂停模块：`modules/energy-buff`
- 模拟器锁：`Daedalus Simulator 1.0.1 / DaedalusSimSdk 1.0.0 / SHM v7 ABI r1 / 1440×1080 / Scene Control v1`
- 模拟器发布：`D:\仿真\releases\daedalus-simulator\1.0.1`
- 模拟器消费者统一入口：`SIMULATOR_CONSUMER_GUIDE.md`（v1）与 `simulator.lock.json`

本仓库只消费模拟器 Release 与 SDK，不包含模拟器源码。模型资产由 `models/manifest.json` 引用外部受保护目录，Git 不跟踪 engine。

所有自瞄、火控和打符分支必须继承消费者统一入口，明确 SDK 用法、三张原生地图、默认高性能模式和可视验收模式。消费者任务发现模拟器 bug 或新需求时，必须先向用户提交提案并等待明确批准；批准前不得编辑模拟器仓库、SDK、发布脚本或正式 Release。

## 自瞄 B 总目标

构建因果神经轨迹预测器：输入最近一段经过几何校验但未做时间平滑的逐曝光可见装甲板集合，以及任意未来时刻 `tau`；输出四块装甲板未来可击打位置的概率分布或多假设结果。曝光时间戳是时间原点，模拟器真值只作标签与验收，不得作为输入。

阶段一是固定模拟器/曝光契约；阶段二是在 tracker 前通过动态渲染 G2 修复 PnP yaw；阶段三仅在 G2 通过后进行有限、无泄漏数据采集，并训练固定的 TCN + 任意时间解码器。候选选择、云台、MPC 和火控保持冻结；模型必须提供不确定性/OOD 与安全回退。

当前阶段一的独立仓库和 SDK 边界已经建立；1.0.1 + SDK + TensorRT + shooting_range 动态基线已可重复启动。阶段二已完成并通过 G2：普通装甲板 `+15°` 倾角固定在 tracker/chassis 坐标系，生产 PnP yaw 通过曝光时刻云台姿态投影后进入 tracker；非零姿态合成回归与 3/5/7 m 原生靶场动态回放均已验收。阶段三正式 360-session 采集已完成；旧 `H=0.07 m` 已被经 exact-exposure 真值验证的 camera→gimbal R/T 取代。当前正式离线合同为最近最多 200 个真实观测事件及其真实时间戳的 `stage3-dataset-v3`，全量 111,527-train/36,297-validation 单 seed 训练、完整 validation 双物理基线评估和动态 ONNX parity 均已完成且未访问 test。该结果证明完整离线流程可行并在整体 validation 上超过刚体 baseline，但不构成多 seed/线上指标验收；test、TensorRT、tracker/MPC/火控和实弹接入仍冻结。PnP 观测记录的事实源为每帧完整 `solved_armors` 集合，离线循环 ID 仅为可重放派生字段。

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
