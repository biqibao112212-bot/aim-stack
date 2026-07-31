# Aim Stack 关键决策

上下文版本：`CTX-AIM-STACK-2026.07-v3`

1. 模拟器与算法分为两个独立 Git 仓库；算法只能消费版本化 Release/SDK。
2. 自瞄 B 是当前主研究模块；装甲板主线、火控和打符保持暂停，按需独立迁移。
3. 模拟器版本由 `simulator.lock.json` 固定，禁止从旧工作树或 dirty 二进制运行。
4. 所有模型 engine、ONNX、训练权重、标注数据和不可再生原始数据默认受保护，Agent Team 不得自动删除。
5. 任务收尾时，只有稳定边界、当前状态和关键决策进入三个上下文；跨模块接口进入所有者公共契约；可复现缓存和临时日志才可清理。
6. 打符旧代码仍为 SHM v6，导入不等于兼容；完成 SDK v1 适配和测试前必须失败关闭。
7. 所有消费者分支必须继承 `SIMULATOR_CONSUMER_GUIDE.md` 和 `simulator.lock.json`；指南统一说明 SDK、三张原生地图、性能/可视模式和当前性能基线，公共契约仍由模拟器 Release 所有。
8. 自瞄调试发现模拟器 bug 或需求不构成模拟器写授权。必须先向用户提交具体提案并获得明确批准，再切换到模拟器仓库独立修改和发布；消费者分支永远不得携带模拟器实现。
9. 消费者锁升级到 Daedalus Simulator 1.0.1；经 release manifest、SDK contract、SHM v7/ABI r1、1440×1080 RGB24、TCP 5602、UDP 5601/5603 兼容性检查后，消费者不修改模拟器仓库。
10. 研究默认使用 DX12 高性能离屏模式；人工可视验收使用 Vulkan + 预览上限 60 Hz。TensorRT 由消费者桥接器显式启用，模型路径仍为受保护的 `D:\仿真\models\engines\armor.engine`。
11. shooting_range 动态场景只通过 SDK SceneControlClient 控制；消费者启动器负责 WSL host/bind、端口 fail-closed 预检、单层有界重试和每次运行的唯一 bridge token 精确清理，禁止 broad kill。
12. 两轮动态基线证明启动、Scene Control、TCP 传输、TensorRT 和曝光匹配链路可重复；普通装甲 PnP yaw 仍保持原实现。G2 修复候选、GT 关联、误差门槛及后续采集方案必须先共同确定，未获用户批准前不得实施 PnP 修复或阶段三采集/训练。
13. PnP yaw 的首轮测量固定为单车实验：只激活 shooting_range 的 3 号车，`Spin`、线速度和线性行程为 0，在 3/5/7 m 使用同一 30°/s 角速度；只绘制现有 tracker 的 yaw 与 ID 切换，不引入四板中心/半径反推或复杂 GT 关联。2026-07-19 的曲线作为讨论材料，不把 ID 跳变、误分类或 detector 丢检预先解释成 PnP 缺陷；在共同讨论前不设 G2 门槛、不改 PnP、不启动阶段三采集。
14. 分析时必须把 `tracked_id` 当作待验证的观测字段，而不是先验正确的四板相位标签：本轮实测其切换间隔约 17–19 ms，不能直接解释为 30°/s 下的物理 90°切板。后续若要做 ID 相位补偿，先用小 smoke 明确 `tracked_id`、`match_ids`、`relative_slot` 的语义和连续性；在此之前，固定频率拟合残差只能作为诊断量，禁止据此修 PnP 或设 G2 门槛。
15. `tracked_armor` telemetry 允许在无当前匹配时复用 `_trackedArmor`，因此 live 分析必须同时要求 `current_match_ids` 非空且 `primary_observation_index >= 0`。源码中 primary observation 由图像中心距离排序后的第 0 项承担，`tracked_id_` 在每次 batch update 后写成该 primary 的匹配槽位；这解释了单观测下仍可能快速换槽。该现象首先是字段/选择策略语义问题，尚不足以判定 PnP bug；后续应记录明确的观测物理身份或匹配槽位诊断，再做动态误差分析。
16. `jump_flag` 不是物理换板事件：当前实现仅因本帧 primary 的匹配槽位不等于 0 就置位，因此它可以在 `tracked_id` 连续时保持 1，也可以在槽位切换时清零。后续分析必须将 `jump_flag`、tracker 槽位变化和 GT physical `relative_slot` 分开统计。
17. 本轮源码审计确认普通四板 `computeMatchCost()` 对 armor yaw 使用 `min(|obs-pred|, |obs+pi-pred|)`，使相差 180° 的槽位（如 0/2）在单板观测下不可辨识；结合无滞回的每帧 argmin，微小 PnP/角点扰动即可造成槽位抖动。该行为解释 20 ms 级 `tracked_id` 变化，但不等于物理换板或 PnP 已被证明错误；若需求是持久物理板 ID，需单独提出 tracker identity policy 修复方案。
18. 按用户确认，后续记录以每帧全部 `solved_armors` 为原始事实源；`observation_id` 只作帧内索引，禁止跨帧使用 `tracked_id`/`tracked_armor`/`jump_flag` 作为物理板身份。原始观测集包含 detector、四角点、两组 PnP 候选、位姿/yaw/距离/reprojection error。
19. 本轮重采集固定为 shooting_range 原生靶场、3 号车、零线速度/线性行程、30 deg/s、3/5/7 m、离屏高性能模式。d3 首次 scene-control transient 失败后，按 fail-closed 规则核验 owning PID、WSL ps、TCP 5602/UDP 5603 清空，再新 token 重跑成功；不得以 broad kill 掩盖失败。
20. 离线派生 ID 采用共同确认的简单规则：第一块可见装甲板为 0，沿已知自转方向按时间连续性推断下一块，循环 0→1→2→3。派生 ID 必须与原始观测分离，并记录 `initialized`/`matched`/`new_cyclic` 与角度代价；这不是 simulator GT、不是 PnP 修复，也不解除 G2/阶段三门禁。
21. The first repair candidate is evaluated in parallel, never injected into
    tracker state. `pnp_ab` fixes ordinary-armor tilt at +15 degrees and jointly
    optimizes camera-frame yaw+tvec against the same per-frame camera model.
    Candidate rank is frame-local diagnostic metadata, not a physical plate ID.
22. The 3/5/7 m A/B disproves joint tvec re-estimation as a sufficient repair:
    it slightly reduces constrained reprojection residual but does not
    consistently improve temporal yaw. Production `yaw`, `yaw_absolute`, rvec,
    tvec and candidate zero therefore remain legacy until a later reviewed
    hypothesis has stronger evidence.
23. `yaw_sensitivity_deg_per_px` after translation marginalization is the
    accepted diagnostic for the physical/angle-pixel precision floor. It is not
    itself an error estimate or G2 threshold; infinite/invalid values describe
    locally unobservable poses and must not be coerced to a finite score.
24. Performance mode must not pass a Win32 forced-hidden startup state to the
    simulator. Visibility is controlled by the Release contract
    (`DAEDALUS_PERF_DISABLE_UI=1`, DX12). Consumer runs retain
    `DAEDALUS_STATS_JSON` and fail closed on bind, map, capture or surface
    errors. This is a consumer-launcher correction and does not authorize or
    require simulator repository changes.
25. Uniform rigid-body yaw is linear in time modulo angular wrapping; a sine is
    expected for a normal-vector component or apparent width, not for the yaw
    angle itself. Plots preserve the wrapped per-plate yaw ramps and 90-degree
    plate transitions; a sine-fit residual may be reported only as a diagnostic,
    never used as the PnP correctness criterion.
26. The +15 degree ordinary-armor tilt is fixed in tracker/chassis coordinates,
    not in the camera frame. Production yaw now projects this pose through the
    exposure-matched gimbal rotation. The previous camera-fixed result remains
    a labeled A/B baseline; tracker input uses the corrected chassis yaw. The
    nonzero-pose synthetic regression and 3/5/7 m replay support this repair,
    while corner-limited residuals and distance-dependent conditioning remain
    allowed. No G2 numeric gate is declared and stage-three data/training stays
    locked.
27. G2 and stage two are accepted as complete on the combined evidence of the
    nonzero-exposure synthetic regression, corrected production tracker wiring,
    and reviewed target-3 slow-spin curves at 3/5/7 m. Acceptance means the
    per-exposure PnP yaw coordinate semantics are suitable as a future predictor
    input; it does not designate the replay as training data or claim zero
    distance-dependent corner noise. Stage three remains separately gated and
    unauthorized until the user and project agree on its collection/training
    design.

28. On 2026-07-20 the user authorized execution of stage three using the
    approved collection/training plan. The first delivery is offline-only:
    dataset, reproducible TCN checkpoint, evaluation report and ONNX. The
    online tracker, candidate selection, gimbal, MPC and fire control remain
    frozen.

29. Stage-three raw observations are captured immediately after
    `solveArmors()` and before `trackerUpdate()` as a deep-copied unordered set.
    The raw source is keyed by `session_id/epoch/frame_seq/timestamp_ns` and
    never uses `tracked_id`, detector number, `tracked_armor` or `jump_flag` as
    identity. Ground truth is a separate exact-exposure stream joined offline.

30. Stage-three collection is target-3-only, single-vehicle, fixed-geometry and
    uses the existing Daedalus Simulator 1.0.1 / SDK 1.0.0. Formal data is 360
    successful 30-second sessions (3 hours): stationary 10%, linear 25%, spin
    25%, linear+spin 40%, with 1-8 m stratified distance, continuous speeds up
    to 3 m/s and 15 rad/s, and eight planar translation direction sectors.

31. Windows training and conversion run through the existing
    `D:\Anaconda\envs\yolov8\python.exe` environment. Torch/CUDA are preserved;
    only the missing data/ONNX/test packages may be added and then frozen in an
    environment manifest.

32. The raw recorder's first inspectable implementation is dedicated JSONL,
    not Stage 2 telemetry JSONL. It is append-only, asynchronously written via
    bounded queues, preserves zero-candidate and >4-candidate frames, and is
    converted offline to compressed tensor shards after CRC/SHA checks.

33. PnP `armorPosition` is in the corrected tracker/chassis convention. Truth
    labels are therefore transformed with the anchor exact-exposure chassis
    pose; gimbal and camera transforms remain in the truth record for audit and
    are never silently substituted as the training frame.

34. Target selection is run-local: the truth writer chooses the target-3
    vehicle from the first exact frame using expected distance and armor label,
    retains its simulator-run target id, and rejects missing target or geometry
    hash drift. `relative_slot` is an offline truth alignment field only.

35. ONNX export uses opset 17 with dynamic batch/time/query axes and verifies
    PyTorch/ONNX Runtime parity at batch/time/query shapes (2/200/8, 1/64/3,
    and 3/200/5) below 1e-4. The online estimator remains frozen in stage 3.

36. The first clean real-SDK stage-three smoke (2026-07-20, stationary target
    3, 30 seconds) produced 475 pre-tracker frames and 476 exact truth records
    at approximately 15.8 Hz, with one visible armor per frame. The fixed
    geometry fingerprint was stable. Because the approved tensorizer requires
    at least eight valid observations in the latest 0.2 seconds, this run
    produced zero valid samples. Formal 24-session qualification and the 360
    session collection are therefore blocked pending consumer-side throughput
    diagnosis or an explicitly reviewed sampling-gate change; no simulator
    repository change is authorized.

37. A follow-up clean smoke used a unique control-session id and disabled
    per-frame debug JSONL. It completed all Scene Control ACKs and produced 833
    observations, 834 exact truth records plus one unavailable startup record,
    and 106 valid tensor samples in 30 seconds. The earlier zero-sample result
    was therefore an operational/retry configuration artifact, not evidence
    that the approved tensorizer is impossible on the SDK path. The 24-session
    qualification gate is still required before formal 360-session capture.

38. The attempted formal batch on 2026-07-20 was invalid because two
    manifest runners were started concurrently. They contended for fixed TCP
    5602/UDP 5603 endpoints, and one WSL Stage-3 bridge plus a simulator
    process survived an interrupted run. The stale bridge consumed later TCP
    image connections, so all 32 attempted sessions failed before writing
    `session_result.json`. Batch and session runners now have independent
    exclusive locks, per-invocation raw and IPC paths, scoped Windows/WSL
    cleanup, and fail-closed data-readiness gates. The invalid runtime
    directories are not training data; `stage3_operations.md` is the canonical
    Stage-3 operating procedure.

39. Stage-three training consumes only the 360-record master manifest and the
    direct `runtime/stage3-formal-20260720-v2/<session>/session_result.json`
    mapping. Recursive evidence/raw discovery is forbidden. Captured hidden
    manifests must equal their master record, and every accepted observation
    and truth file is bound by path, size, record count and SHA-256.

40. Formal tensors use `stage3-dataset-v2`: session-disjoint stratified
    216/72/72 train/validation/test splits, split-specific compressed shards,
    and train-only xyz normalization. Test shards stay unopened during data
    selection, overfit and pilot work; test evaluation requires an explicit
    later authorization flag.

41. `Armor::armorPosition` is a camera-origin vector after AngleSolver gimbal
    stabilization and its 0.07 m vertical-offset transform. Stage-three labels
    reproduce that convention at the anchor exposure using anchor camera
    origin and chassis axes, without a y flip. The previous chassis-origin
    statement is superseded: formal A/B evidence measured approximately
    0.2--0.3 m nearest-plate bias from that origin choice.

42. Query time is the effective matched truth delta, while requested tau and
    matched future timestamp remain audit fields. Anchor truth requires an
    exact full-key match; any history frame with more than four candidates
    rejects its window; freshness is determined by the latest finite valid
    armor observation rather than by zero/invalid frames.

43. Fixed-ego qualification may exclude an explicitly reported startup
    settling prefix, but every emitted 0.995 s history must begin within the
    stable suffix. A raw session with no eligible tensors is retained and
    reported, never deleted or silently substituted; dataset qualification
    limits the total zero-sample-session fraction to 10%.

44. The accepted formal derived dataset is
    `D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-dataset-v2-20260720-r5`
    (manifest SHA-256
    `026cbab209884f51150f2650ab25765b095738df3196d4d398bdbc5e54e72a3c`).
    It contains 181,426 samples, split 109,159/35,609/36,658 across
    216/72/72 sessions. Six zero-sample raw sessions remain reported; the
    1.67% fraction passes the fixed 10% qualification limit.

45. On 2026-07-20 the user narrowed the immediate objective to completing one
    full offline round and proving feasibility, without pursuing metric
    acceptance. Capacity tuning was stopped and retained. Round one therefore
    uses the frozen 16-train/8-validation pilot selection for five epochs and
    treats metrics as diagnostic only.

46. Round-one feasibility is accepted at the pipeline level: qualified shards
    were hash-verified, GPU training produced provenance-bound best and last
    checkpoints, and 4,520 samples from all eight requested validation sessions
    completed neural, static and rigid-CV/yaw-rate evaluation with 100%
    baseline validity. Both training and evaluation record
    `test_accessed=false`. This does not claim that the learned model beats the
    physical baseline or that the architecture/hyperparameters are accepted.

47. The canonical round-one evidence is under
    `D:\仿真\models\engines\stage3-training\20260720-pilot24-seed0-round1`.
    It is a protected exploratory asset from a dirty worktree, not a release
    candidate. Test evaluation, three-seed training, ONNX publication and any
    online tracker/MPC/fire-control integration remain frozen.

48. Decision 41's legacy `H=0.07 m` coordinate contract is superseded. The
    simulator path uses the calibrated camera-to-gimbal transform
    `R=[[0,0,1],[-1,0,0],[0,-1,0]]`,
    `T=[0.25631080,0.00183094,0.09543117] m`, composed once with the exact
    exposure optical pose. Tracker position origin is the exposure gimbal
    pivot. Missing or invalid enabled calibration fails closed; the real-device
    template keeps the transform disabled until a device-specific R/T exists.

49. The calibration is accepted for simulator use on independent exact-truth
    evidence: 72 formal sessions and 5,760 exposure poses have maximum
    rotation/translation errors of `2.77e-5 deg / 3.51e-7 m`. The former
    25-degree calibration-board script was a synthetic self-consistency loop,
    not an external calibration; it is archived and cannot update production
    YAML. Camera/gimbal entity quaternions are not OpenCV optical-axis
    quaternions.

50. Decision 40's fixed 5 ms `stage3-dataset-v2` tensor contract is
    superseded. `stage3-dataset-v3` selects the latest at most 200 valid raw
    observation events, preserves their real timestamp relative to anchor,
    and pads only on the left. Model, augmentation, baselines, evaluation and
    ONNX consume `event_mask` and `event_time_s`; index-derived time is
    forbidden. v2 shards and checkpoints are incompatible archival assets.

51. Existing formal raw data may be reused without recollection because the
    v1 position transform is exactly invertible: undo the frozen legacy
    camera/tracker convention and H, then apply the verified R/T. New
    `stage3-observation-v2` capture records raw camera tvec and an R/T audit;
    the dataset builder rejects any audit that differs from its bound
    calibration hash.

52. The accepted v3 derived dataset is
    `D:\仿真\dataset\autoaim-stage3-v1\derived\stage3-dataset-v3-20260721-r1`
    (manifest SHA-256
    `8448ebe788b4a4bb5bd3803e4e64841bf39f3867f711d3198de31f1fb283ada0`).
    It has 185,292 samples split 111,527/36,297/37,468 over 216/72/72
    sessions. Six zero-sample sessions remain explicit (1.67%); the dataset
    qualification passes and test stays unopened.

53. The first formal v3 single-seed full run is retained under
    `D:\仿真\models\engines\stage3-training\20260721-v3-full-seed0`.
    Thirty epochs completed normally; epoch 28 is best. All 36,297 validation
    samples completed paired evaluation with 100% baseline validity: learned
    median/P95 set error is `0.175675/0.569595 m`, versus
    `0.417854/1.336396 m` for rigid CV/yaw-rate. Both training and evaluation
    record `test_accessed=false`.

54. The epoch-28 checkpoint exports to dynamic opset-17 ONNX with maximum
    PyTorch/ONNX Runtime parity error `9.54e-7` below `1e-4`. These artifacts
    prove a complete offline single-seed pipeline and an overall validation
    advantage over the rigid baseline. They do not authorize test evaluation,
    three-seed metric acceptance, TensorRT publication, online tracker/MPC/
    fire-control integration or live firing.

55. The validation error contract is four-way, not a three-way triangle:
    (a) current observation vs anchor truth, (b) exact future observation vs
    future truth, (c) network prediction vs future truth, and (d) network
    prediction vs exact future observation. Future observation joins use only
    `future_timestamp_ns`; no nearest-frame or interpolation substitute is
    permitted. Comparison (c) is retained for every query even when the raw
    future observation is unavailable; (b)/(d) are coverage-qualified
    diagnostics. The full r3 report and slice-level attribution are retained
    under the Stage-3 model directory and the consumer docs.

56. A v4 observation-target experiment is retained as an A/B branch. It adds
    exact future observation positions, per-slot masks and frame-availability
    masks, with no update for missing future frames. A zero-initialized PnP
    residual head plus visibility head is jointly trained with the existing
    physical head. On the full validation split, physical `P-G` remains
    `0.175675/0.569594 m`, while `P-O` is `0.226535/1.353787 m` versus the
    v3 baseline `0.223675/1.345863 m`. The branch is not a production
    replacement: past xyz/yaw history does not contain enough information to
    predict the random future-frame PnP error. Do not optimize away this
    conclusion by treating missing observations as zero labels.

57. Supersede decision 56's negative learnability conclusion with the v5
    controlled scratch result. Future-observation training uses a direct
    observation head, geometry-only permutation shared by training and
    evaluation, and masked Huber labels. Model A is observation-only; model B
    adds a 0.2 physical-position auxiliary. Both start from one identical
    random state and share batches/dropout masks. On validation, A reaches
    P-O median/P95 `0.198660/1.090279 m` and B reaches
    `0.201418/1.069940 m`, both better than v3 `0.223675/1.345864 m`. Retain B
    as the composite-score winner and A as the median winner. Neither is an
    online release until test, multi-seed, export and runtime gates pass.

58. Stage-three physical-core isolation reuses the existing v3 exact-exposure
    truth; a new simulator capture is neither required nor authorized. The r5
    truth-history derivative contains train and validation only, reproduces q0
    labels with zero maximum error, and records `test_accessed=false`.

59. Physical evaluation selects one four-slot permutation from all q0 truth
    plates and holds it fixed for every future query. Queries are separately
    labeled `rule` only when exact velocity and yaw rate remain unchanged over
    the entire q0-to-query interval. Future-event tails are reported but may
    not fail a constant-state physics gate.

60. The accepted physical core is the parameter-free exact-state rigid
    operator, not a network trained from scratch to imitate mechanics. It
    translates the exact target center and rotates q0 armor offsets around that
    center using exact yaw rate. Rotating around the arithmetic armor centroid
    is rejected because the template is not exactly centroid-symmetric and the
    resulting fictitious translation creates centimetre errors.

61. On the full validation split, the accepted operator has q0 P95
    `1.86e-9 m` and rule-query motion P95
    `4.45e-6/8.19e-6/1.76e-5 m` at nominal 0.1/0.2/0.5 seconds. The protected
    r5 artifact passes all 1 mm gates. Its public API exposes positions only;
    center/velocity/yaw-rate are internal inputs. The next trainable boundary
    is PnP history to these internal initial conditions, while propagation is
    frozen.

62. Decisions 59--61 remain an exact-state oracle but do not define the new
    causal-input A/B experiment. The causal experiment admits only exact
    physical armor positions, real event times, masks and query tau from train
    and validation. It uses fixed cyclic geometry slots with no 24-permutation
    association in either loss or metrics. The truth slot phase is an explicit
    assumed-correct external causal-slotter contract, so this experiment is an
    oracle physical upper bound and not yet an online PnP release.

63. A reacquired segment must accumulate at least eight continuous observation
    events before it may train or predict. The last four admitted events and
    t0 must belong to one producer epoch, target id, geometry hash and constant
    translation/yaw-rate interval. Windows failing those causal identifiability
    checks are reported and excluded; they are not zero-filled labels. Future
    rule queries retain the existing t0-to-future constant-motion gate. Test is
    neither constructed nor opened.

64. The common A/B architecture is a causal last-four fixed-slot rigid-pose
    least-squares main path plus a zero-initialized hard-rigid neural residual
    for arbitrary tau. It derives pose only from admitted past positions and
    times; exact center, velocity, yaw and yaw rate remain forbidden predictor
    inputs. A trains q0/future/motion terms. B adds history reconstruction and
    shared constant-motion consistency only on the last four events qualified
    as one motion segment. The common position
    output API and paired initialization/batch/RNG contract are unchanged.

65. The strict 8-session pilot retained 2,690 of 3,509 windows: 812 were below
    eight post-reset events and 7 changed motion between the last-four history
    and t0. All 173,449 retained history events passed identity continuity.
    On 1,335 validation windows the causal input-sufficiency oracle has q0 P95
    8.20e-7 m and 0.5 s motion P95 of 0/2.29e-5/1.27e-5/3.64e-5 m for
    stationary/linear/spin/linear-and-spin; maximum q0 and 0.5 s errors are
    2.81e-6 and 7.48e-5 m. Both A and B selected the identical epoch-0 physical
    solution (headline motion P95 2.74e-5 m); learned residual updates degraded
    validation and were rejected by early stopping. This passes the pilot and
    authorizes a full train/validation run, but not test or online integration.

66. A first full A/B attempt was stopped after epoch 1 and retained as failed
    evidence because B reconstructed the entire visibility segment while only
    the last four events were certified as one motion segment. Earlier motion
    changes made that auxiliary incompatible with the current physical fit;
    validation motion P95 moved from 2.15e-5 m to 3.91e-4 m for B. B history
    and shared-motion losses are therefore restricted to the qualified last
    four events. A new non-overwriting full run is required.

67. The corrected full causal physical A/B run is accepted under
    `20260721-causal-physical-full-seed0-r2`. Its qualified dataset retains
    77,725 train and 25,695 validation samples from 147,824 candidates, rejects
    43,994 short post-reset histories and 410 nonconstant recent histories,
    verifies 4,495,070 history identities, and records `test_accessed=false`.
    Independent checkpoint reload reproduces q0 P95 `9.08e-7 m` and 0.1/0.2/
    0.5 s rule-motion P95 `4.73e-6/9.03e-6/2.15e-5 m`; the 0.5 s maximum is
    `7.32e-5 m` and every motion-class P95 is below 1 mm.

68. A and B both select epoch 0 and their best model-state SHA-256 is the same
    (`dfcf3af7ea33663536e8458eea3ad62737c55fd8695df66c849ade34debb1c3a`).
    Paired learned-residual updates degrade the already numerical-precision
    physical solution, so no trained residual is enabled in the accepted pure
    physical model. This establishes that exact recent fixed-slot positions
    plus real timestamps are sufficient for the agreed constant-rigid-motion
    layer; it does not establish causal slot association, PnP robustness,
    acceleration handling, test performance, export parity or online release.

69. The PnP state-adapter experiment reuses only train/validation shards from
    qualified `stage3-dataset-v4-observation-20260721-r4`. Both arms receive
    the same normalized PnP xyz/yaw, reprojection/count quality, masks, real
    event times and tau; future observation labels, exact motion state, motion
    class and test shards are forbidden predictor inputs. Exact physical future
    positions remain the common supervision target.

70. Existing PnP tensor rows are unordered per-frame candidates and contain no
    persistent armor identity. The main state A/B therefore predicts an
    unordered hard-rigid four-armor set and uses a symmetric nearest-set loss
    and metric. It does not enumerate 24 permutations, select a q0 truth slot,
    or rematch identity per query. This isolates the value of a shared motion
    state without pretending that online cyclic identity is solved; continuous
    local-slot tracking remains a separate required deployment experiment.

71. Main A maps the shared causal set/temporal encoding to one
    `center0/velocity/yaw0/omega` state and uses a frozen constant-twist rigid
    decoder for every tau. Main B maps the same encoding and each tau directly
    to that query's center/yaw and uses the identical rigid decoder; it has no
    shared velocity/omega propagation. Both are rolling-window estimators, not
    recursive EKFs: every new exposure recomputes the result from the latest N
    real-timestamp events. Exact state is never fed to either forward path or
    used as an arm-specific training label.

72. The first PnP dynamic pilot is fixed by
    `training/stage3/selections/pnp_state_dynamic_pilot_v1.json`, whose dataset
    manifest hash is verified before any shard is opened. It contains two
    train and two disjoint validation sessions per motion class, with 4,584
    and 4,615 samples respectively; the test list is exactly empty. Selection
    balances near/far distance and the dominant one/two-visible-candidate
    conditions. Any different sessions, source hash or hyperparameters define
    a new non-overwriting run, not a continuation of this pilot.

73. Seed-0 dynamic pilot r2 completed 30 epochs from clean commit `5e5a42a`,
    with `test_accessed=false`. A/B selected epochs 28/22. A versus B q0 P95
    is 0.533/0.573 m, q3 absolute-set P95 is 1.466/1.478 m, and q3
    centroid-motion P95 is 1.384/1.401 m. The apparent aggregate A advantage
    is not a dynamic-state win: A q3 motion P95 is worse than B by 3.1%, 5.9%
    and 2.7% in linear, spin and linear-and-spin respectively.

74. The pilot is rejected before any full run. A's validation median speed is
    only 0.082 m/s for linear truth near 0.956 m/s and 0.138 m/s for combined
    truth near 2.535 m/s; median direction cosines are 0.118/0.183. A simple
    diagnostic one-second least-squares slope of the same PnP history reaches
    direction-cosine medians 0.970/0.964, so motion direction is present in the
    input but the indirect set-position objective did not extract it. The
    diagnostic is not a deployment candidate. The next A/B must give both
    arms equivalent truth-derived state/delta supervision during training only
    and retain raw PnP history as the sole inference input.

75. The frozen rigid decoder and all state propagation run in FP32 even when
    the learned encoder/head uses AMP. Pilot r2 exposed a B-only bfloat16 phase
    normalization artifact (rigid-shape P95 0.663 mm); this numerical artifact
    is separate from the rejected meter-scale motion result and must not be
    carried into the next experiment.

76. PnP state A/B v2 uses the same trajectory-derived supervision on both
    final decoded position outputs. It fits all query centers and alias-safe q0-to-q1
    yaw rate, then applies center, center-delta, velocity, relative-yaw and
    omega losses in addition to the unordered position-set loss. A receives no
    latent-only state loss; B therefore has soft trajectory-state supervision
    but still no shared constant-twist state in its forward graph. Training and
    validation exclude future truth windows whose complete query trajectory departs
    from one constant twist by more than 1 mm or 1 mrad. This implements the
    user's constant-motion scope and prevents simulator span reversals from
    being mislabeled as learnable constant velocity. Exact future truth is
    detached label/evaluator data only and remains forbidden predictor input.

77. Full relative yaw, not a modulo-90-degree label, is used because the real
    geometry template is measurably non-C4-symmetric: its minimum symmetric set
    distance to a non-trivial quarter turn is 15.89 mm. A 5 mm asymmetry gate
    is enforced before training. Only relative phase/yaw rate are supervised;
    no absolute truth slot phase is a predictor input or auxiliary target. The
    q0-to-q1 alias guard uses the actual time delta and the 15 rad/s bound.
    Validation retains an all-input position report while checkpoint selection
    uses the query-constant-twist subset; overall and per-motion-class coverage
    must each remain at least 75%. The frozen pilot validation coverage is
    4,348/4,615 (94.21%), with class coverage 100.0/93.59/98.59/84.20%.

78. V2 dynamic-only micro r1 completed 30 epochs from clean commit `c39557d`
    on 3,377 train and 3,405 disjoint validation samples; test stayed sealed.
    It is rejected. Best A/B validation velocity error median/P95 is
    0.834/2.948 and 0.948/2.821 m/s; yaw-rate error is 4.90/15.12 and
    1.03/14.00 rad/s. Replaying best checkpoints on their training sessions
    remains poor (A velocity 0.621/2.664 m/s and yaw rate 2.92/13.66 rad/s),
    so held-out session shift is not the sole failure. A bounded same-session
    combined-motion overfit gate is required before deciding whether the
    current encoder lacks capacity or merely needs a different optimization
    budget. This diagnostic must be marked train-sourced, bounded, unqualified,
    and must not be confused with validation evidence.

79. The clean same-session capacity reproduction r2 completed 160 epochs from
    commit `a0e4f61`; it is correctly marked `diagnostic_only=true`,
    `qualified_training_candidate=false`, and `test_accessed=false`. A/B best
    epochs are 151/156. A versus B velocity error median/P95 is
    0.0136/0.0427 versus 0.0294/0.0796 m/s; yaw-rate error is 0.0375/0.1184
    versus 0.0711/0.2250 rad/s; 0.5 s motion error is 9.48/25.58 versus
    15.86/41.03 mm. A's decoded center/yaw constant-twist P95 is numerical
    precision, whereas B is 5.25 mm/0.0789 rad. This proves capacity and the
    utility of the hard shared state on one motion distribution, but the prior
    held-out failure still forbids a full run. The next experiment must change
    the common encoder to a learned relative-time/relative-motion
    representation and increase train-session motion diversity.

80. The user froze a new clean-physics neural A/B contract on 2026-07-22 and
    explicitly superseded the analytic-LS-residual A/B as the active training
    experiment. Input is the most recent 32 real-timestamp events for four
    persistent cyclic slots from the existing qualified causal physical
    train/validation dataset; no PnP, permutation search, analytic pose fit,
    displacement/time derivative, exact motion state, or future label enters
    either forward path. A estimates one neural center0/velocity/phase0/omega
    state and uses frozen constant-twist propagation. B estimates center/phase
    independently for each tau and exposes no velocity or omega. Both use the
    identical fixed-slot decoded-position objective
    `2*q0 + absolute + 2*motion_delta`; velocity and omega are reparsed from
    final decoded positions for symmetric diagnostics only. Any future change
    to this input, architecture, or loss contract requires user discussion
    before implementation.

81. The fixed-slot neural physical A/B capacity gate is rejected before the
    held-out pilot. The clean run from commit `a8b3f89` used the frozen common
    loss, identical batches/RNG/optimizer, one train-sourced combined-motion
    session for both fitting and evaluation, and never accessed test. A chose
    epoch 0; B chose epoch 1. A/B q0 median/P95 is `1.7528/3.5261` and
    `1.6284/3.3051 m`; 0.5 s motion median/P95 is `1.42535/1.44519` and
    `1.42548/1.44412 m`. Both reconstruct nearly zero speed instead of the
    session's approximately `2.85 m/s` motion. Because the model cannot even
    fit its own training motion distribution, more held-out data or epochs are
    not presently accepted as evidence of feasibility. Per the user's frozen
    process, the held-out pilot remains blocked and no change to representation,
    initialization, objective, training schedule, or checkpoint selection is
    authorized until a joint failure analysis chooses it explicitly.

82. The 2026-07-22 joint code/data/loss/literature audit supersedes only the
    interpretation of decision 81, not its freeze. The run proves that the
    registered training and selection procedure failed; it does not prove that
    a causal TCN cannot infer constant rigid motion. The implemented TCN has a
    125-event receptive field for 32 inputs, the state bounds cover the selected
    session, and the fixed decoder enforces the accepted planar constant-twist
    geometry to sub-micrometre residual. Conversely, the dataset certifies only
    the latest four history events as one constant-motion interval, the run made
    about 155 optimizer updates before an untrained epoch-0 static prior exhausted
    patience, and the weighted motion term supplies roughly one tenth of the q0
    last-head gradient at initialization. These are unresolved data-contract and
    optimization confounders. No held-out pilot, test access, model-family change,
    loss/input change, export or online integration is authorized by this audit;
    the next bounded diagnostic must be chosen with the user and must separate
    last-four versus 32-event history, gradient scaling, initialization and
    fixed-update training before revisiting architectural feasibility.

83. The user rejected the proposed minimal diagnostic and requested direct
    repair followed by a full train/validation run. The new active contract
    therefore fixes all four audited confounders together: the dataset must
    certify constant motion across every one of the 32 consumed events; final
    heads use a small random initialization so the encoder receives first-step
    gradient; epoch zero is initial evidence only and cannot become the trained
    best checkpoint; default clipping and early stopping are disabled. The
    prior r2 experiment remains immutable failure evidence.

84. Both arms now receive one common training-only, meter-equivalent state
    objective reparsed from their decoded q0--q3 positions. It supervises
    center0, velocity over a 0.5 s reference horizon, unit phase scaled by the
    geometry radius, yaw rate over the same horizon, and temporal
    constant-twist consistency. Future truth constructs detached labels only;
    exact state, finite differences and future data remain forbidden forward
    inputs. The fixed geometry decoder continues to enforce per-query rigidity.

85. The qualified non-overwriting 32-event derivative is
    `stage3-causal-physical-v1-20260722-r4`, manifest SHA-256
    `8121dc8096952052ca9f9bfe3f5ed951c103a05a1ef7be4d65e2b40c731e113e`.
    It contains 32,904 train and 11,189 validation samples across 278 admitted
    sessions, records four zero-sample short sessions, verifies every shard,
    and keeps test sealed. The failed partial r3 build is retained as
    reproducible failure evidence and is not a training input. An official
    full run must start from a clean repository commit so its provenance is
    not downgraded to exploratory evidence.

86. Formal training performs a row-level preflight in addition to manifest and
    shard hash verification. Every consumed row must contain 32 complete,
    strictly increasing fixed-slot events and fit constant translation/yaw
    within `1e-4 m/rad`. Overall and every present motion class must retain at
    least 85% q0--q3 trajectory supervision. The r4 audit passes at 94.20%
    train and 93.66% validation overall; the lowest class coverage is 89.51%,
    and maximum history residuals are `6.45e-6 m` and `1.36e-5 rad`.
    Checkpoint selection is restricted to this same trajectory-eligible subset
    so ignored future-transition rows cannot control the selected model. Full
    training also requires all four registered motion classes and rejects any
    history interval that could alias the allowed 15 rad/s yaw-rate range.

87. The user explicitly authorized committing the completed repairs and
    starting formal training for 300 epochs. The registered run is seed 0 on
    the complete qualified r4 train/validation splits, with early stopping and
    gradient clipping disabled. It must use a new protected output directory,
    retain stdout/stderr logs, and keep test, export and online integration
    sealed.

88. The completed v11 run is evidence against continuing the same shared
    aggregate state objective for more epochs, not evidence against a causal
    TCN. Best A reaches q0 P95 `0.1108 m` and q3 motion P95 `1.0163 m`, but
    linear and combined-motion median speed ratios remain `0.59/0.70` after
    300 epochs. With 53.6% zero-translation eligible training windows, the next
    experiment must separate positive motion regression from activity
    classification instead of letting static samples dominate one state loss.

89. V12 retains the 32-event causal TCN and frozen hard-rigid constant-twist
    decoder while factorizing q0 pose, translation and rotation heads. Moving
    and rotating gates receive balanced BCE with truth-derived detached labels;
    velocity and yaw-rate experts receive positive-only direct supervision.
    Static/non-rotating samples do not impose zero expert regression. The paired
    augmentation arm rotates the complete history/future sample around a center
    recovered only from the latest causal history and translates xy by at most
    0.25 m. Checkpoint selection prioritizes the worst
    dynamic-class q3 motion P95, and the test split remains sealed.

90. V13 replaces v12's shared learned encoder/gates with physically separate
    specialists while preserving the accepted 32-event causal input and frozen
    constant-twist decoder. The augmented-v12 epoch-283 encoder/q0 head and
    original-v12 epoch-266 encoder/translation head are frozen. Pure rotation,
    joint combined motion and four-class routing each own a different trainable
    TCN encoder. The combined expert jointly predicts its own velocity and yaw
    rate; it is not an addition of translation- and rotation-expert
    trajectories. One integrated checkpoint packages these independent
    parameter blocks for atomic inference and provenance.

91. Router semantics are defined by truth-derived motion factors, not dataset
    session class. A sample is stationary/translation/rotation/combined only
    when its trajectory-eligible speed and absolute yaw rate lie outside both
    configured dead bands; ambiguous samples are excluded. Pure-rotation omega
    supervision applies only to factor 2, combined velocity and omega only to
    factor 3, and the router receives group-balanced four-class CE. Validation
    must report both hard-routed final metrics and raw specialist metrics so a
    routing failure can be distinguished from an expert regression failure.
    Test remains sealed throughout the 300-epoch train/validation run.

92. V14 isolates the trained v13 router without changing expert or inference
    architecture. The exact v13 epoch-297 integrated checkpoint is the source;
    every non-router parameter is frozen and hash-verified. Because the v13
    train router CE is already near zero while validation macro recall plateaus
    near 94%, merely extending its original objective is rejected. The retained
    four logits instead receive group-balanced four-class, moving-factor and
    rotating-factor classification losses. This shares physical rotation
    evidence between rotation and combined classes while preserving the hard
    four-route API. Training-only planar rigid augmentation is permitted;
    validation, test and PnP noise remain untouched.

93. V15 does not replace the v14 four-class router with another four-class
    retrain. The exact completed v14 epoch-53 last checkpoint is frozen as a
    hierarchical coarse system. Its moving probability decides moving versus
    non-moving, and its stationary/rotation conditional decision is retained.
    Only moving samples enter a new translation-versus-combined binary
    refinement model. This preserves all specialists and the already reliable
    coarse decisions while assigning the one unresolved boundary its own
    capacity and balanced objective.

94. The v15 refinement receives the 32 causal armor histories only after each
    event's common visible-slot translation is removed and the relative rigid
    shape is scaled by the fixed geometry radius. Cyclic slot encoding remains
    available. No temporal finite difference, expert prediction, future truth,
    exact motion state, PnP noise, or class label enters inference. Training
    uses equal-weight group-mean BCE for qualified translation and combined
    rows; all v14 parameters are frozen and hash-verified. Selection first
    prevents registered q3 regression, then improves the worse of translation
    and combined recall. The v14 source checkpoint is retained as an explicit
    safe fallback and test remains sealed.

95. Module A is separated from the frozen motion system. On every exposure it
    consumes only the causal unordered PnP history up to that exposure and
    restores one current physical center and full canonical phase. Its frozen
    geometry decoder emits four fixed `relative_slot`s. Online inference must
    append each result when it is produced and may call v15 only after a
    qualified 32-event cache exists; a later window cannot rewrite older cache
    entries. This makes A and B independently trainable/evaluable, but does not
    make their residual errors statistically independent.

96. V16 A0 training admits only v4 rows whose latest valid PnP event is exactly
    q0, so PnP recovery is not confounded with missing-frame extrapolation.
    Only q0 fixed-slot truth derives the center/phase label; q1--q7 stay off the
    training device. The predictor receives xyz/yaw, masks and causal real
    time, but not the zero-filled reprojection channel, broken candidate-
    fraction channel, future labels, motion state, tracked identity, motion
    class or test. Frozen v15 is provenance-only and supplies no gradient.

97. V16 uses a two-term, meter-equivalent objective: current-center SmoothL1
    plus geometry-radius-scaled full unit-phase SmoothL1. Frozen geometry makes
    additional rigidity losses redundant. Checkpoint selection prioritizes
    fixed-slot q0 P95, center P95, full-phase P95 and fixed-slot P99. Unordered
    set error is auxiliary; modulo-90 error and quarter-turn/180-degree alias
    rates are mandatory, because a visually good set with the wrong canonical
    phase is not compatible with v15.

98. The user explicitly superseded decisions 95--97 as the active modeling
    contract. Four armor indices are temporary cyclic tracker handles only.
    They encode adjacency and persistent state bookkeeping, not canonical
    physical identities. A visible armor must not be fitted to a fixed four-
    slot center/phase/radius/height template, and neither a fixed cyclic-shift
    minimum nor fixed-geometry decoder is permitted in the new predictor.

99. V17 first solves clean physical prediction without PnP. The qualified r4
    train/validation truth stays immutable; a virtual hash-bound loader creates
    predictor observations by exposing the nearest horizontal-range plate and,
    near a range-order boundary, its second-nearest adjacent plate. Every valid
    event has one or two visible tracks, the primary is visible, and changes are
    restricted to 0/+1/-1 modulo four. This construction uses only same-event
    truth to synthesize availability; hidden truth and future truth are loss/
    evaluation labels and never predictor inputs. Test remains sealed.

100. V17 is exactly C4 roll-equivariant by construction. One causal temporal
     encoder is applied identically to each track within a specialist, circular
     shared-weight message blocks exchange clockwise/counterclockwise context,
     and a shared per-track query head directly decodes trajectories. Invariant
     mean/max pooling feeds a separate router. Slot embeddings, order-specific
     flattening and per-slot parameters are forbidden. Training applies random
     cyclic origins and direction reversal; validation keeps a deterministic
     local convention.

101. Stationary, translation, rotation and combined are four parameter-
     independent trajectory specialists. A sample supervises only its matching
     raw expert; balanced four-class CE supervises the independent router. The
     combined expert directly predicts its own four trajectories and does not
     read, add or freeze the translation/rotation expert outputs. Validation
     must separate oracle correct-expert error from hard-routed error and report
     visible, clockwise/counterclockwise adjacent-hidden, opposite, one/two-
     visible, switch/no-switch and motion-class strata.

102. The v17 objective is intentionally compact: local-label direct-position
     SmoothL1, q0-relative motion-delta SmoothL1, a low-weight pair-distance
     drift term relative only to each prediction's own q0, and balanced router
     CE. The rigid term contains no template, radius, height, center or phase.
     A fixed 1,024/2,048 bounded run provided optimization diagnostics but was
     completed before decisions 103--104 corrected query masking/selection; it
     is not metric evidence and its weights are forbidden as a formal warm
     start. The authorized formal run uses all
     32,904 train and 11,189 validation rows for 300 epochs from a clean commit.

103. Only source queries with `rule_query=true` may supervise or select v17.
     The mask removes future points that have left the qualified constant-
     motion segment; q0 must always remain eligible. Position, motion-delta,
     self-rigid validation strata and checkpoint selection use the same mask.
     Dataset preflight records eligible counts for every query and requires all
     four motion classes at the selected fixed horizon.

104. V17 checkpoint selection is fixed to source query index 3, whose nominal
     horizon is 0.5 seconds. Query indices 4--7 are sample-specific random tau
     probes and remain useful diagnostic rows, but they must never be called a
     final/longest horizon or control best-checkpoint ordering. User-facing
     reports name the selected horizon in seconds instead of relying on `q3`.

105. V18 separates current-state recovery from future motion propagation. The
     first formal component is S, a q0-only cyclic state restorer. Every one or
     two currently observed armor tracks receives an observation update. A
     clean observation is an exact identity bypass only when its causal event
     time is q0; a latest observation before q0 must be propagated to q0.
     Previously observed hidden tracks continue causally, while a never-seen
     track remains invalid. `primary` defines local cyclic order and does not
     select the only updated track or any physical template.

106. S-layer deterministic supervision follows observability and task need.
     Stationary and translation samples supervise/evaluate current visible
     tracks only. Rotation and combined samples additionally supervise warm
     adjacent-hidden tracks. A warm track must have appeared inside the exact
     history consumed by the model; cold targets never affect objective,
     gradient, checkpoint selection or final position metrics. Cold is reported
     only as count/coverage/confidence. Evaluation masks are derived from causal
     visibility, never model confidence.

107. V18-S remains exactly C4 roll-equivariant and contains no slot embedding,
     center, phase, fixed radius, fixed height or geometry template. A shared
     per-track encoder is augmented by shared directed adjacent-edge temporal
     memory and circular messages. The loss is group-balanced q0 SmoothL1 for
     stale visible tracks and rotation/combined self/edge-warm adjacent tracks,
     plus low-weight observed-edge and uncertainty-calibration terms. Future
     tau, router, PnP and motion class are forbidden forward inputs. Formal
     training uses 180 epochs, five-epoch warmup then cosine decay, immutable
     validation checkpoints, full train/validation and sealed test.

108. V18-S completed at epoch 180 and is retained as immutable evidence rather
     than extended. Its combined warm-adjacent P95 plateaued at 29.84 cm while
     rotation reached 6.17 cm. At epoch 175, combined recent/stale P95 was
     17.42/49.93 cm and pair-supported/self-warm P95 was 23.98/54.28 cm. This
     identifies stale absolute hidden-state propagation as the next structural
     bottleneck; more v18 epochs or a second unconstrained absolute-position
     expert are rejected.

109. V19 parameterizes every dynamic warm adjacent q0 state as a current
     visible primary anchor plus a directed relative edge. The clockwise target
     is `anchor + edge(primary -> primary+1)` and the counterclockwise target is
     `anchor - edge(primary-1 -> primary)`. Both endpoint handles being causally
     seen is sufficient support, even if they were never simultaneously
     visible. Current one/two-visible observations remain exact at q0 or are
     propagated from their latest causal event. Cold and non-adjacent hidden
     handles remain invalid. This is dynamic relative-state learning, not a
     center/phase/radius/height template or persistent physical slot mapping.

110. V19 initializes only from the sealed clean v18 epoch-180 best checkpoint.
     Only new asynchronous edge and edge-uncertainty heads are trainable; every
     inherited parameter is frozen and hash-verified so accepted visible and
     pair-supported behavior cannot drift. The asynchronous residual applies
     only when both endpoints were seen but the pair was never co-visible;
     pair-seen edges
     remain the v18 foundation output. Supervision is group-balanced across rotation/combined,
     co-visible/asynchronous support and recent/stale age. Checkpoint selection
     uses the worse rotation/combined anchor-composed P95, then all supported
     relevant-edge P95. Validation must additionally report per-motion visible
     propagation, pair-seen versus asynchronous edges, and self/pair/recent/
     stale warm subsets. Test, PnP and future truth remain unavailable to the
     predictor.

111. V20 freezes the accepted V19-r2 epoch-110 checkpoint as the entire S
     layer. S alone owns q0. Every future expert must satisfy
     `p(0) = q0_S` exactly and may optimize only q0-relative future motion.
     Truth-q0 decoding is evaluation-only and is rejected in model train mode.
     No future expert may repair S, access motion class in forward, consume
     future truth, or introduce center/phase/radius/height/slot semantics.

112. V20 uses one deterministic stationary path and three parameter-independent
     trainable runs. Translation predicts a common 3-D velocity. Rotation
     predicts primary planar velocity and yaw rate. Combined independently
     predicts primary total 3-D velocity, primary planar acceleration and yaw
     rate. For track-relative planar offset `r_i`, combined constructs
     `w_i=w_a+omega*J*r_i` and `a_i=a_a-omega^2*r_i`, then applies the stable
     closed-form constant-yaw integral. This parameterization has no center-
     translation gauge, preserves unequal track heights and pair distances,
     and is not a sum of the translation and rotation networks.

113. V20 supervision follows the S task mask. Translation uses current visible
     tracks only. Rotation and combined add causally anchor-composed warm
     adjacent tracks. The primary future-delta SmoothL1 and omega auxiliary
     labels use the same mask. An omega edge is valid only when both endpoints
     are eligible; cold and opposite future truth cannot affect objective,
     gradient, validation or selection. Confidence never controls eligibility.
     The expert bounds are 7 m/s, 100 m/s^2 and 20 rad/s, which cover the
     qualified train/validation dynamic tails without clipping at the observed
     maxima.

114. Every v20 formal run is its own crash-consistent transaction stream.
     Checkpoints are written through a unique pending file and atomically
     renamed before history and manifest commit. The manifest defines the last
     committed epoch. Resume validates status, test sealing, configuration,
     source/dataset/foundation hashes, history, best/latest hashes, embedded
     provenance, optimizer/scheduler/scaler and CPU/CUDA RNG. An ahead-only
     history is truncated to the committed prefix; one valid next orphan
     checkpoint may be adopted without overwrite and is recorded in the resume
     chain. Formal runs require clean committed source and distinct protected
     model/runtime paths.

115. V21 compares two rotation-only F layers behind the same frozen V19-r2
     epoch-110 S state: an improved center-free parametric decoder and a
     continuous direct-delta trajectory decoder. Rotation direction is shared
     deterministic causal state, not a neural output or a loss target. It is
     derived from short-time signed rotation of a co-visible directed adjacent
     edge, with single-track three-observation signed curvature as fallback;
     insufficient histories remain direction-invalid rather than guessed. Once
     acquired in streaming inference, direction is locked for the target
     lifetime under the accepted no-reversal motion assumption. The parametric
     arm learns only nonnegative yaw-rate magnitude plus primary planar
     velocity and applies the deterministic sign. The direct arm outputs only
     query-conditioned q0-relative trajectory candidates and exposes no
     velocity, omega, acceleration, center, radius or phase. Each query is
     projected by a center-free 2-D Procrustes layer onto the closest proper
     rigid transform of that sample's own valid q0 tracks; this keeps arbitrary
     per-sample radius and height and does not integrate a motion state. Both
     arms exclude direction-invalid,
     cold and opposite tracks from deterministic position supervision and use
     identical frozen-S, data, query, seed and validation contracts. Future
     truth may supervise trajectory and parametric magnitude only; it cannot
     determine inference-time direction. A/B truth-q0 validation freezes the
     first-forward F delta and changes only the additive q0 anchor, so the
     diagnostic has identical semantics for both architectures. Direction
     coverage and rejection are mandatory validation results and are gated;
     a low-coverage model cannot improve its reported P95 by silently refusing
     difficult histories.

116. V22 treats the V21 rotation plateau as a pre-compression relational
     evidence failure, not an optimization-budget failure. On the same last 32
     causal events, a read-only adjacent-edge diagnostic recovers yaw-rate with
     0.00018 rad/s P95 error on 83.2% of validation samples, while V21-A has
     1.77 rad/s P95 error. With truth q0 the diagnostic's 0.5-second trajectory
     P95 is 0.0145 mm; with the frozen V19 S output it is 3.46 cm for current
     visible tracks and 7.68 cm for warm adjacent tracks. V22 therefore adds a
     shared pre-compression relational stream to both A and B. It encodes only
     unsigned scalar invariants of consecutive adjacent edges and single-track
     curvature: vector norms, cosine, absolute sine/angle and causal time gaps.
     Signed direction is structurally absent, remains owned by the deterministic
     direction state and keeps zero loss weight. The legacy per-track stream,
     frozen S, task mask, trajectory objectives, rigid projection and test seal
     remain unchanged. V22-A retains a nonnegative magnitude/primary-velocity
     parametric decoder; V22-B remains a direct trajectory decoder without an
     omega output. The first controlled comparison is limited to 30 epochs and
     must report relation coverage and the same q3 role metrics before any
     larger budget is authorized.

117. The word "per-track" in v17--v22 described shared per-track output heads,
     not a strict single-track F. V17 jointly encoded four handles with pooled
     context and circular messages. V21-B still used all-track pooled context,
     q0 adjacent edges, circular messages and a joint per-query Procrustes
     projection. V22 additionally broadcast a multi-track relational latent.
     These architectures therefore do not implement the user's clarified
     `one maintained handle history -> the same handle future` interface.

118. V17 motivated the S/F split: its 0.5-second oracle current-visible P95 was
     about 0.12/49.18/15.39/45.61 cm for stationary/translation/rotation/
     combined. V18 exposed stale hidden-q0 propagation, and V19-r2 epoch 110
     became the accepted frozen S foundation, with about 0.97 cm current-visible
     P95 and 7.02/9.33 cm rotation/combined warm-adjacent P95. V20 then reached
     about 6.42 cm translation current-visible cascade P95, 20.12/20.69 cm
     rotation current/warm P95, and 47.97/57.26 cm combined current/warm P95.
     V21 deterministic-direction direct B improved rotation to 17.17/18.03 cm,
     but remained far above the desired tail.

119. Both v22 30-epoch runs completed cleanly at commit
     `464605c46f496836897c1db9b8e76e2376376bf7`, with test sealed and V19
     unchanged. Best A2 current/warm 0.5-second cascade P95 was 41.78/36.44 cm;
     best B2 was 30.22/43.07 cm. Zeroing the learned relation vector changed
     these metrics only at sub-mm/mm scale, and truth-q0 reruns remained about
     41.93/34.86 and 30.39/41.84 cm. Thus S is not the dominant tail and v22 did
     not learn to use its relation branch. Its loss/selection diverged, while
     its 30-epoch schedule reached zero LR after only 8,940 steps versus v21's
     53,640; this rejects the implemented v22 coupling, objective and schedule,
     not neural single-track forecasting in principle.

120. The r4 source holds full four-handle truth, while predictor input exposes
     one or two visible handles and keeps each handle in a separate lane.
     Same-handle future truth remains available after visibility is lost. At
     nominal 0.5 seconds, q0-current tracks remain virtually visible in only
     95.35%/43.99%/57.83% of translation/rotation/combined validation cases,
     yet they retain labels. Leaving view is therefore supervised temporal
     extrapolation, not zero-shot prediction. Cold and opposite never-seen
     handles remain outside the accepted task and loss.

121. The user-defined F contract supersedes multi-track relational fusion inside
     F. One shared-weight F is invoked separately for every visible or warm
     handle; it reads only that handle's ordered position/time/mask history and
     predicts only the same handle. Identity mixing, all-track pooling,
     circular messages, adjacent-edge features, broadcast relation state and
     joint multi-handle projection are forbidden in F forward. A rigid-body
     loss or evaluation may compare independently produced trajectories, but
     may not transmit another handle's features into F.

122. A computationally shared vehicle motion state cannot exist without an
     information-transfer mechanism. If handles are strictly independent,
     common motion is only a property of the truth and a consistency constraint;
     each F call must infer equivalent latent dynamics from its own history. If
     future requirements demand immediate transfer from one handle to another,
     a separate, explicit tracker-level MotionContext (physical parameters or a
     declared neural latent) must be designed and evaluated. Hidden multi-track
     fusion inside a component still called "single-track F" is rejected.

123. Clean single-track rotation is learnable in principle from a sufficiently
     long non-degenerate arc without exposing center or yaw rate as outputs.
     Combined motion is `p(t)=c0+v*t+R(omega*t)*r`; on a short arc, common
     translation and rotational tangent velocity are weakly identifiable and
     require small higher-order curvature evidence. In current validation,
     current-track observed-span P10 is only 0.041 s for rotation and 0.044 s
     for combined, while the 0.5-second forecast/history ratio P90 is 10.46/9.60.
     This conditioning, plus the non-single-track representation and mixed
     supervision path, is a plausible structural cause of the large tails; it
     is not evidence that neural networks cannot learn regular arcs.

124. Clean-truth physics remains an isolation/upper-bound tool only. Exact
     adjacent-edge and single-track curvature signals can be overwhelmed by PnP
     noise, especially through differencing. No final physics-first priority is
     assumed. Clean and noisy-input evaluations must stay separate, and PnP
     robustness, adapter boundaries and anti-forgetting gates require a later
     explicit decision before PnP or end-to-end training resumes.

125. The user's observable-target clarification supersedes Decisions 120--123
     wherever they require same-handle future prediction. A training sample is
     anchored at the currently selected visible plate, but every future query
     re-applies the visibility rule and supervises that query's selected target.
     Permanent physical plate identity is forbidden from F. A source slot may
     exist transiently inside the offline builder only long enough to unwrap an
     adjacent switch, after which it is replaced by a signed sample-local count.

126. Sparse r4 future endpoints are insufficient supervision for this task:
     they cannot distinguish no switch from a full revolution and can miss
     intermediate transitions. V24 therefore constructs a 1 ms label-only
     physical-truth stream from each already-qualified constant-motion anchor,
     checks its sparse endpoints against exact r4 future truth, and accumulates
     visibility switches without modulo reduction. This calculus is permitted
     only in offline label generation; no physical rollout exists at inference.
     Visibility ties preserve the preceding selection and q0 inherits the last
     history selection. The qualified observed range is -5..+6, so the symmetric
     candidate contract is -6..+6; overflow is never clipped or bucketed.

127. One v24 dynamic F forward consumes a single selected-target relative
     history, real time/dt/switch increments, detached current q0 position,
     anonymous detached S candidate relations, signed candidate steps,
     confidence/validity and arbitrary tau. Current absolute q0 is an allowed
     observable feature because nearest-range visibility is not translation
     invariant; withholding it makes translation-induced target selection
     unidentifiable. It is not a plate identity. Candidate encoding and the
     decoder are row-shared and permutation-equivariant. Translation, rotation
     and combined instantiate separate parameters/optimizers/checkpoints;
     stationary is a deterministic identity path and motion class stays outside
     forward.

128. V24 learns branch-conditioned trajectories rather than a probability-
     averaged coordinate. Switch CE and position SmoothL1 are macro-balanced by
     signed step. Position is gathered from the true candidate branch before
     loss, so every wrong branch receives exactly zero position gradient.
     Candidate/query permutations may only permute their matching output axes.
     Every tau equal to zero, regardless of query order, has exact step-zero
     probability one and bit-exact zero displacement. Missing true candidates,
     out-of-horizon tau and invalid masks fail closed.

129. V19-r2 epoch 110 is not retrained for v24 Phase A. The acceptance order is
     fixed: (1) invariant/unit gates, (2) truth-S 512-window tiny-fit for each
     dynamic expert, (3) truth-S train/validation run, (4) identical-F frozen-S
     A/B with candidate coverage reported separately. Only a failure introduced
     specifically at step 4 is evidence to revisit S. The current CPU smokes are
     executable-path evidence only and remain `gate_failed`; they do not satisfy
     any learned-accuracy gate.

130. Windows `D:\Anaconda\envs\yolov8` is the required v24 training runtime;
     it exposes Torch 2.7.1+cu118 and a CUDA-capable RTX 4060. WSL is retained
     only as prior test evidence and must not run formal training. A Windows
     CUDA one-step smoke passed. Formal training must not run concurrently with
     the user's active NIGHTREIGN GPU workload, which was observed at about
     6.5/8.2 GB and 70%+ utilization after the training process stopped. The
     attempted no-checkpoint r2 process was terminated and its exact child PID
     verified dead; restart must use a new non-overwriting output directory.
# 2026-07-26 user acceptance: stop tiny-fit precision chasing

- The user explicitly accepted the current observable-F precision and ordered
  the team to stop further tiny-fit refinement.  The previous 1 mm capacity
  gate is retained as a diagnostic metric, not a blocker that authorizes more
  tuning.
- The active F definition is v9: an anonymous visible-stream encoder, signed
  sample-local switch routing, one tau-independent coefficient tensor per
  sample/candidate, and a learned history-conditioned time basis.  It contains
  no physical plate ID, fixed slot embedding, hand-written circle/ellipse
  decoder, or translation-plus-rotation expert composition.
- Accepted tiny-fit evidence: translation completed at conditional P95
  0.896 mm; rotation completed at 0.946 mm; combined completed at 2.208 mm
  with every observed signed-step routed correctly and hard P99 4.468 mm.
- The interrupted combined tail-only continuation was stopped on request and
  is diagnostic-only.  No further tiny-fit or tail-threshold tuning is allowed
  under this decision.  Work proceeds to from-scratch full train/validation.

## 2026-07-26 decision 131: close observable-F training without refinement

- The three independent v9 experts completed their fixed from-scratch budgets
  in Windows `yolov8`/CUDA: translation 10,000 updates, rotation 15,000, and
  combined 15,000. All stopped normally at `max_updates`, used the same r6
  manifest SHA-256, and kept test sealed. No formal run was initialized from a
  tiny-fit checkpoint.
- Final held-out truth-S validation is diagnostic rather than an acceptance
  claim. Translation has switch macro/minimum-step recall 0.8811/0.6923,
  conditional P50/P95/P99 1.109/6.337/17.100 mm and hard P95/P99
  10.487/299.814 mm. Rotation has 0.9488/0.8679,
  6.748/31.281/87.411 mm and 38.520/287.889 mm. Combined has
  0.8509/0.7516, 17.755/79.265/128.392 mm and 304.358/332.064 mm.
- The manifests correctly retain `status=gate_failed`: the user stopped
  precision chasing, but that instruction does not retroactively turn the old
  millimetre gates into passed evidence. Training completion and model
  acceptance are distinct claims.
- No more epochs, tiny-fit work, CVaR/tail tuning or threshold search will be
  launched. The large hard-routed tails are retained as evidence of held-out
  route/generalization error, not as permission for further optimization.
- S remains frozen and is not retrained. Because the held-out failure is
  already measurable with truth-S relations, a frozen-S A/B cannot establish
  S as the current cause. It is therefore deferred until a future task first
  supplies a provenance-safe paired adapter and a reason to revisit S.
- The completed artifacts are protected evidence. PnP, test, motion router,
  export and online fire-control integration remain outside this closure.

## 2026-07-26 decision 132: stop selector iteration and accept its error as a baseline

- The dedicated selector was allowed to finish its already-fixed 10,000-update
  budget naturally, but no continuation, supervision loop or extra capacity is
  authorized. Its best validation accuracy is 90.86% and hard P95 remains
  299.52 mm while the frozen conditional trajectory is bit-exact. This confirms
  that more selector epochs do not address the wrong-board structural tail.
- The remaining roughly 9--10% clean route error is retained as a diagnostic
  baseline. It is not optimized during the first PnP stage, and hard metrics
  cannot select a PnP robustness model at this gate.

## 2026-07-26 decision 133: define the first real-PnP experiment as an oracle upper bound

- The first PnP arm uses real observation-v4 xyz, but same-exposure past truth
  performs the injective association and supplies signed history switch labels.
  Truth-S supplies all q0 candidate roles; the current role is replaced by the
  q0 PnP measurement, so every candidate with `step mod 4 == 0` is bit-exact
  zero. This is deliberately optimistic and non-deployable.
- Each exposure-local PnP point is transformed through world into the q0 anchor
  tracker frame before history differencing. A sample enters the strict primary
  metric only if all selected 32 history events and q0 are uniquely associated.
  Missing and ambiguous events are reported against the full clean denominator,
  never truth-filled, interpolated or silently removed.
- The paired derivative must replay the existing r6 arrays bit-exact and remain
  hash-bound to observation-v4, truth-history r5, causal r4 and r6. Only train
  and validation shards are opened; test remains sealed. Physical slots and
  oracle assignments are transient builder state and are absent from F input.

## 2026-07-26 decision 134: direct clean-F PnP substitution is rejected

- Under the strict oracle upper bound, real PnP increases paired conditional
  P95 from 5.63 to 1350.31 mm for translation, 25.40 to 368.06 mm for rotation,
  and 81.04 to 937.01 mm for combined. Paired mean degradation is respectively
  364.49, 170.18 and 250.73 mm, with bootstrap 95% intervals excluding zero.
  All frozen state hashes remain unchanged.
- The result is not explained by the clean selector error: it is measured on
  the labelled true branch. PnP q0 anchor P95 is 885.24/158.26/179.51 mm on the
  strict translation/rotation/combined subsets, while conditional response
  drift P95 is 1351.11/368.89/934.77 mm. The frozen network amplifies the
  observation-domain change rather than merely inheriting the anchor offset.
- Therefore the next justified learned component is observation-domain
  robustness: a causal PnP denoiser/adapter or paired noisy-input F training
  with frozen clean targets and an explicit clean anti-forgetting gate. A raw
  PnP deployment claim additionally requires a separately accepted anonymous,
  permutation-invariant association/S interface. More clean selector training
  or direct PnP substitution is rejected.

## 2026-07-26 decision 135: run paired adapter versus joint S+F robustness arms

- Arm A is a learned causal PnP-to-clean selected-stream adapter followed by
  the bit-exact frozen accepted combined F epoch 180. It has no geometry
  decoder or physical-ID input and is supervised by paired clean current,
  history, candidates and future absolute position.
- Arm B is literal joint retraining, not a relabelled F-only experiment. V19-r2
  S and combined F epoch 180 initialize independent trainable copies, and F's
  observation boundary is differentiable only when the new explicit opt-in is
  used; the legacy default remains detached and bit-exact compatible.
- B consumes one/two virtual-visible oracle-associated PnP handles. All four S
  q0 outputs remain hypothesis rows so F keeps four residue classes;
  unsupported cold roles are represented by zero confidence and are reported
  separately, never truth-filled or silently removed.
- Fairness is fixed to the same combined-motion common subset, seed, 10,000
  updates, F objective and validation split. Conditional true-branch P95 is the
  selection metric, hard routing is diagnostic, and paired clean replay is an
  anti-forgetting constraint. The experiment remains non-deployable because
  same-exposure past truth supplies association, primary and switch labels.

## 2026-07-27 decision 136: replace the misdefined A with a true observation mapper

- Decision 135's A implementation is retained as historical evidence but is
  not considered a faithful test of `PnP -> physical observation -> frozen S/F`.
  It read a selected stream plus truth-S candidate relations, bypassed S, and
  preserved every non-current candidate absolute position.
- The replacement mapper reads only PnP XYZ, observation mask, event time and
  event mask. Primary, switch, motion class, future truth, session/pair IDs and
  permanent armor identity are forbidden model inputs. Primary/common flags may
  select or weight loss rows but are not passed to `forward`.
- Corrected observations retain the PnP mask and enter frozen V19 S. Frozen F
  history is rebuilt from the corrected per-event selected absolute positions;
  falling back to stored raw-PnP history is forbidden. Candidate relations and
  confidence come only from frozen S.
- Direct mapping is learnable: on the shared combined/common validation points,
  P95 falls from 184.82 to 115.84 mm and current P95 from 179.51 to 120.40 mm.
  The complete frozen-S/F conditional P95 falls from 984.38 to 536.82 mm, but
  remains worse than joint B's 350.42 mm.
- Controlled replacements locate the residual: clean current alone gives
  481.87 mm, clean selected history 321.49 mm, truth candidates 471.48 mm, and
  clean history plus truth candidates 156.74 mm. Mapped-S candidate P95 is
  444.18 mm and invalid q0 P95 is 481.69 mm. A mask-preserving mapper cannot
  invent an unobserved candidate, so further mapper epochs are not authorized.
- The clean-observation frozen-S/F conditional floor is 252.50 mm, while the
  oracle truth-S frozen-F floor is 81.04 mm. The remaining decision therefore
  concerns S hypothesis/support semantics and deployable association/quality
  inputs, not mapper size or training duration.

## 2026-07-27 decision 137: close external correction of a frozen clean F

- A C4-aligned v4 window mapper preserves the accepted q0 mapper bit-exact and
  reduces mapped relative-history P95 to 52.81 mm, but the frozen downstream
  chain still gives 517.81 mm conditional P95. This disproves the assumption
  that a lower pointwise mapper loss is sufficient for the frozen clean F.
- Task-aware mapper distillation against the clean teacher (478.04 mm), direct
  physical future supervision (469.85 mm), and a post-H anonymous history
  adapter (487.74 mm) all fail their fixed gates. The history-adapter hybrid
  teacher reaches 283.93 mm, showing a reachable counterfactual that the small
  external residuals do not learn.
- No further mapper, adapter, LR, loss-weight or epoch sweep is authorized.
  External correction remains useful only as frozen upstream preprocessing for
  the next structural control.

## 2026-07-27 decision 138: split clean F and PnP F by trusted observation domain

- `CLEAN` and `PNP_V41` are explicit external enum routes. Unknown strings fail
  closed. Clean F and PnP F are independently allocated models/checkpoints with
  no shared tensor storage. The observation domain is not inferred and is not
  passed into the network; physical ID and motion class remain forbidden.
- Clean F stays bit-exact frozen. PnP F is initialized from it once, then learns
  the actual fixed v41 mapper -> V19 S -> diagnostic H distribution without
  clean replay, parent-weight regularization or simultaneous S/H updates. This
  is a domain-expert architecture split, not a last-layer fine-tune.
- Trajectory and selection are separate optimization stages. Trajectory freezes
  `switch_candidate_head`/`switch_logit`, sets switch loss to zero and selects
  only by conditional P95/P99. Selector freezes every trajectory parameter,
  trains only those two selector modules, and must preserve the full validation
  conditional-output hash bit-exact.

## 2026-07-27 decision 139: accept diagnostic dual-domain F and stop tuning

- The v50 trajectory run stopped automatically after four stagnant validation
  checks. Its retained update-2550 checkpoint has conditional P50/P95/P99
  39.70/215.85/537.76 mm, improving P95 by 58.3% from v41's 517.81 mm and by
  37.3% from old joint B's exact 344.00 mm. It passes the predefined 250 mm
  strong gate without changing selector or frozen upstream state.
- The v52 selector run retains update 2125. Relative to its trajectory parent,
  hard P95 falls from 440.70 to 363.54 mm, switch accuracy rises from 72.67% to
  83.25%, and minimum-step recall rises from 37.35% to 69.02%. Conditional P95
  stays 215.85 mm and the complete conditional tensor stream is bit-exact.
- v52 also improves the old joint-B hard P95 of 391.39 mm, although its
  363.54 mm tail remains materially larger than the conditional error. This is
  accepted as evidence that explicit stage isolation works, not as permission
  for another selector tuning campaign.
- Tail risk remains unresolved: PnP conditional P99 is 537.76 mm, hard P99 is
  560.30 mm and the maximum error is 5.99 m. The frozen clean branch on the
  same validation rows has 76.17 mm conditional P95, leaving roughly 140 mm of
  P95 domain gap even after the structural improvement.
- All 261 Stage3 tests pass and loader/parameter-partition audits pass. The
  result remains `diagnostic_only`: oracle association, legacy H provenance,
  v41/H mismatch and dirty source prevent formal or deployable status. Test was
  not accessed. Because the same validation split selected checkpoints, drove
  early stopping and compared multiple structures, it is adaptively reused and
  cannot serve as an untouched final acceptance set. The next work must fix
  those provenance/interface/evaluation boundaries; v50/v52 tuning is closed.

## 2026-07-27 decision 140: formalize by replay, not by relabelling diagnostics

- The previous source chain is now preserved in local Git, but this does not
  convert v41/v35/v50/v52 into clean-commit artifacts. Their manifests and
  diagnostic limitations remain immutable historical evidence.
- The minimum honest next chain is: clean-source replay of the v41 architecture,
  H trained against that exact mapper, then fixed-final dual-domain trajectory
  and selector stages. Formal trainers must reject dirty worktrees, source
  changes during a run, mapper/H mismatch, diagnostic H, and adaptive best-
  checkpoint promotion.
- Replaying v41 from immutable v27/v39 checkpoints makes the new training run
  reproducible but retains those parents as explicit legacy input assets. A
  claim of fully provenance-clean lineage would additionally require rebuilding
  those initial assets from their own clean sources. Until then the precise
  claim is `formal_oracle_evaluation`, never `deployable_pipeline`.
- Fixed endpoints are inherited once from the accepted diagnostic evidence and
  cannot be tuned: mapper 264 updates, H 4224, trajectory 2550 and selector
  2125. Validation uses fixed-final checkpoints and predeclared gates; it no
  longer selects among epochs. Decision 141 separately locks the original
  cosine schedule horizons so an endpoint does not redefine the LR curve.
- The current validation split is adaptively exhausted. A release-candidate
  lock must bind commit, environment, dataset, mapper/S/H/F hashes, evaluator
  and gates before a single-use holdout can be opened. Any holdout failure
  consumes that holdout; it cannot authorize post-test tuning.

## 2026-07-27 decision 141: canonical fixed-final formal-oracle protocol

- Formal identity is bound to the one tracked canonical protocol, one clean Git
  commit, a complete per-stage source bundle, the Windows `yolov8` CUDA
  environment and the exact GPU identity. Deterministic algorithms, cuDNN
  determinism, disabled TF32 and pre-launch
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` are fail-closed requirements.
- Dataset manifest, V19 S, clean F and the legacy v27/v39 mapper initialization
  assets are bound by file/state SHA-256. Every train/validation view must have
  zero duplicate sample keys, zero sample overlap and zero session overlap.
- Mapper, H, trajectory and selector preserve the diagnostic cosine schedule
  horizons of 400/5000/10000/3000 updates but stop at the preselected fixed
  endpoints 264/4224/2550/2125. Each formal stage evaluates exactly once at
  that endpoint; there is no adaptive checkpoint selection or early-stop
  promotion. A downstream stage accepts only the manifest-declared,
  gate-passed fixed final from the same commit, protocol and environment.
- The strict final loader additionally requires selector conditional/upstream
  hashes to remain bit-exact. These controls establish a reproducible
  `formal_oracle_evaluation` chain only. Oracle association and legacy v27/v39
  parents remain explicit, so `full_chain_provenance_clean=false` and
  `deployable_pipeline=false` are mandatory.

## 2026-07-28 decision 142: recovery state is continuity, not model selection

- Formal H writes one recovery checkpoint after every fully completed training
  epoch. It contains the exact H and AdamW states, cumulative update/elapsed
  counters, Python/NumPy/Torch CPU/CUDA RNG states and the dedicated shuffled
  DataLoader generator state. Resume starts at the following epoch, so an
  interruption loses at most the current incomplete epoch.
- A recovery checkpoint is forbidden from reading validation or test and is
  never entered into validation history, `best`, gate evaluation or checkpoint
  selection. The single fixed-final validation at update 4224 remains the only
  formal evaluation and promotion point.
- Resume fails closed unless source commit/protocol/environment, complete
  training arguments, dataset manifest, frozen mapper/S/F state hashes and H
  model configuration all match. Recovery files are immutable protected model
  assets; unique names prevent overwrite, while an atomically replaced hash-
  bound pointer identifies the latest committed recovery state. A process-held
  OS file lock makes each output directory single-writer; owner metadata records
  PID/start/command, live contention is rejected and a dead owner's metadata is
  archived before takeover.
- Adding recovery changes the canonical protocol/source contract. Therefore
  the passed mapper from commit `17e54ae` remains valid historical evidence but
  cannot parent the new H. The mapper must be replayed on the recovery commit
  before H is restarted; no old artifact is relabelled or deleted.

## 2026-07-28 decision 143: cache frozen train features, preserve H semantics

- The server v59 H run demonstrated an implementation bottleneck rather than a
  useful 3090 workload: epochs 3--6 each took 884--906 seconds while GPU use was
  30--32% and VRAM use was below 0.6 GiB. Its watchdog and process were stopped
  after the immutable epoch-6/update-792 recovery point was confirmed.
- Mapper and V19 S are frozen deterministic functions of each train sample, yet
  the old loop recomputed mapper once and S twice per update for all 4224
  updates. The accepted first acceleration stage computes only the 14 S fields
  consumed by H once, keeps them float32 and CUDA-resident, and gathers them by
  the original shuffled row index. Validation is never cached.
- The formal DataLoader structure, generator consumption, batch size, optimizer
  updates, LR schedule, dropout call order, C4 shift, losses and three H forwards
  remain unchanged. A final partial batch continues through the online frozen
  path to preserve its CUDA matrix shape. Cache probes must be `torch.equal`.
- Cache identity is part of the formal protocol and recovery contract. It binds
  content digest, field schema, shapes/dtypes/bytes, dataset manifest and frozen
  mapper/S state hashes. A new source commit requires a fresh mapper replay and
  H restart; the v59 recovery cannot cross this source contract.
- GPU utilization is not the gate. The cache-only implementation must include
  its one-time build cost and deliver at least 2.5x projected end-to-end speedup
  (target 3x or better) without metric/state divergence. H-forward fusion,
  larger batches, AMP, TF32 and `torch.compile` remain deferred because they
  alter numeric execution or training semantics.

## 2026-07-29 decision 144: joint selector/trajectory training does not replace the staged baseline

- V64 tests a materially different selector: it owns a temporal history encoder
  and scores anonymous candidate future paths conditioned on observed dynamics
  and query time. It contains no permanent armor ID, physical slot embedding,
  motion-class input or hand-authored switch time. After a 500-update selector
  warmup, both the selector and PnP trajectory parameters train through a
  probabilistic mixture objective; the obsolete trajectory selector stays
  frozen and hash-unchanged.
- The fixed 3000-update combined-motion diagnostic completed normally, but it
  does not beat the same-validation staged V52 selector. Conditional P95 moves
  from 215.85 to 217.95 mm, hard P95 from 363.54 to 413.83 mm, selection
  accuracy from 83.25% to 78.16%, and one-step recall from 69.02% to 61.96%.
  Therefore V64 is retained as negative structural evidence and is not the new
  baseline; V52 remains the best current diagnostic selector result.
- The full distribution confirms two simultaneous effects. Correctly selected
  queries have hard P95 220.61 mm, while incorrectly selected queries have
  604.82 mm and constitute 3,377/15,461 queries. Error also rises strongly with
  future time, absolute switch count and large Mapper/S/H or raw-PnP input
  error. Physical yaw rate alone has a substantially flatter conditional trend
  than these horizon/input-quality variables.
- The analysis is not independent acceptance evidence: it covers only
  `motion_class=3`, uses oracle association, and reuses an adaptively observed
  validation split. Exact truth distance/yaw rate are post-inference analysis
  labels only. No translation/rotation/stationary comparison, deployable claim
  or untouched-test claim may be inferred from these figures.

## 2026-07-29 decision 145: replace flat selection with ordered crossing times

- Keep V50 trajectories and Mapper/S/H bit-exact frozen. V64 showed that a new
  history encoder plus candidate-future compatibility and joint trajectory
  updates degrade the accepted staged baseline. The new selector reuses frozen
  V50 history/candidate summaries and never reads future candidate paths.
- Reject the single signed progress curve before full training. Although truth
  sequences are ordered, its positive linear/saturating basis forced concavity;
  exact feasibility showed roughly one quarter of windows require locally
  accelerating crossing intervals and cannot satisfy that structure.
- Predict six positive sample-conditioned intervals and cumulatively form
  T1<...<T6. Query time is compared with those learned times; direction is a
  separate shared sign. Windows may therefore have different angular speed,
  phase and successive intervals. These are learned outputs, not a global
  schedule, permanent plate identity or physics decoder.
- Balance capacity data by direction, maximum switch count and session inside
  each existing split. Labels/session used for sampling never enter forward.
  The fixed capacity result misses the 95% accuracy gate at 94.04% but passes
  90% one-step recall and has zero reversals; record a manual diagnostic release,
  never an automatic pass or deployment claim.
- The full run has one fixed endpoint at update 2125. It must compare against
  V52 at 363.54 mm hard P95, 83.25% accuracy and 69.02% one-step recall and
  prove conditional/state hashes unchanged. No checkpoint promotion or test
  access is authorized.

## 2026-07-29 decision 146: retain the fixed V66 result without validation tuning

- V66 completes update 2125 and improves the same-validation V52 baseline on
  hard P95 (358.12 versus 363.54 mm), hard P99 (554.58 versus 560.30 mm),
  accuracy (83.565% versus 83.255%) and one-step recall (73.095% versus
  69.021%). Conditional P95 remains 215.85 mm with bit-exact conditional and
  frozen-trajectory hashes; all 1,978 evaluated query sequences are monotone.
- The stronger absolute gate remains failed: hard P95 is above 350 mm, hard P99
  above 550 mm and accuracy below 84%. Epoch 35 happens to score slightly better
  on some metrics, but it is not promoted because update 2125 was the only
  predeclared endpoint. No extra epochs, learning-rate sweep or checkpoint
  selection is justified.
- Distribution evidence matters more than one tail number: final hard coverage
  is 52.06/67.98/75.78/79.90/88.26% at 50/100/150/200/300 mm. Correct selections
  have 212.69 mm hard P95; wrong selections are 2,541/15,461 queries and have
  412.87 mm hard P95. Error increases mainly with future time, switch count and
  upstream error; physical yaw-rate trend is flatter.
- This remains an oracle-associated, adaptively validated diagnostic with
  legacy Mapper/H mismatch. V66 is useful structural evidence and the best
  fixed-endpoint selector result measured here, not formal acceptance or a
  deployable model. The next responsible work is confidence-based fire gating
  and/or a new untouched evaluation boundary, not further fitting this split.

## 2026-07-29 decision 147: fine-tune only the complete final position

- The requested next target is the actual future XYZ emitted by the full
  pipeline, not another local trajectory or selector proxy. Add one causal
  residual head after frozen V66. It reads frozen history/candidate context,
  query time and the frozen selected route/confidence, but no future truth,
  physical plate ID, motion-class label, candidate-wise identity head or
  hand-written physics decoder.
- Initialize the final XYZ layer to zero so the untrained system is bit-exact
  V66. Train against clean absolute future position while the forward path sees
  only the cached mapped PnP-domain inputs. The residual is bounded to 0.75 m;
  all upstream states and armor decisions must remain hash-identical.
- Use a fixed ten full-data epochs rather than extending training until a
  favorable validation checkpoint appears. A 256-window smoke run showed that
  the loss and gradient path improve all central/tail metrics early, while
  continuing on the tiny subset begins to overfit; this is engineering evidence
  for the fixed short schedule, not a checkpoint-selection rule.
- For user review, generate only one scatter plot: future time versus final XYZ
  error, one point per query. Rare errors above the dense 99.5% body are placed
  on the top plot boundary and counted in the title. Numerical Mean/P50/P95/P99
  and coverage remain in the run manifest, but no additional plot family is
  produced.

## 2026-07-29 decision 148: retain the fixed V67 final-position endpoint

- V67 completes the only declared endpoint, epoch 10/update 430, in 32.1 s of
  measured training time. Against its bit-exact V66 initialization on all
  15,461 validation queries, Mean improves by 0.92 mm, P50 by 0.56 mm and P95
  by 6.16 mm. P99 regresses by 5.99 mm, below the declared 10 mm tail tolerance;
  all gate checks pass and 54.98% of individual queries improve.
- This is a modest body-distribution gain, not a solved selector. Coverage at
  50/100/300 mm moves from 52.06/67.98/88.26% to 52.41/68.37/88.78%, while
  frozen armor selection remains exactly 83.565%. The residual itself has an
  8.22 mm median and 29.98 mm P95, so it is correcting final XYZ locally rather
  than replacing the learned trajectory/selection structure.
- Retain only the fixed final checkpoint as the V67 endpoint; do not promote
  epoch 2 despite its favorable visible metrics and do not add epochs. The run
  remains an oracle-associated combined-motion diagnostic on the adaptively
  observed validation split, with the test split still sealed.

## 2026-07-29 decision 149: new simulator data validates the eligible model, not the whole input pipeline

- The disjoint capture uses seed `2026072902`, new session IDs and new random
  distance/phase/direction/speed/yaw-rate tuples. Six spin and six combined
  sessions complete through the locked public Release/SDK; the original V67
  dataset has zero session overlap. The new dataset's three test sessions stay
  unopened, while seven internally named train and two validation sessions are
  combined strictly for frozen evaluation.
- Among PnP/S/F-common-usable inputs, generalization is not the current model
  failure. New combined V67 Mean/P50/P95/P99 is
  79.06/38.92/310.41/385.92 mm with 86.24% correct selection, and V67 improves
  the same new queries over V66 at Mean/P50/P99. Pure rotation is also accurate
  on its small eligible subset. No weight, threshold or checkpoint is selected
  from the new data.
- The decisive weakness is admission coverage. Only 477/1,255 physical windows
  are common usable (38.01%) versus 59.54% in the old paired dataset. Spin is
  68/546 (12.45%); combined is 409/709 (57.69%). The 5.18 m and 7.56 m combined
  sessions have only 1/85 and 0/50 usable windows, and two of four evaluated
  spin sessions have zero. This strict subset selection makes the favorable
  network metrics conditional rather than an end-to-end success claim.
- Therefore freeze V67 and investigate observation availability, detector/PnP
  continuity, association and the complete-32-event gate on the retained raw
  sessions. Do not respond by adding F epochs or changing the selector until
  the missing-input mechanism is separated from true prediction error.

## 2026-07-29 decision 150: default to native 6 mm and constrain capture paths

- The apparent missing-input problem is primarily a preprocessing policy
  failure, not evidence that the 6 mm armor detector or final trajectory model
  cannot generalize. On the same unsealed frames below 7 m, native `wide_6mm`
  solves 98.19% while the virtual `precision_16mm` crop solves only 23.74%.
  The crop repeatedly loses the target and falls back to wide acquisition,
  producing the long discontinuities that defeat the complete-history gate.
- All autoaim B entry points now default to one native 6 mm focal profile.
  Dual focal remains callable only as an explicit diagnostic override. Stage-3
  capture makes the choice explicit and records it in the durable result; no
  truth, focal-profile flag or crop state is added to any neural input.
- Random capture parameters must describe an admissible full reciprocal path,
  not merely an admissible initial distance. Keep a 0.5 m reserve under the
  requested 7 m camera range, include the collection-only 0.10 s command lead,
  and reject paths below 0.75 m forward or above 75 degrees horizontal yaw.
  Rejection causes resampling; it never changes labels or network inference.
- A new 1,050-frame fixed-6-mm smoke run at 4.4 m and 9 rad/s has no empty
  observation frames and 99.90% target-3 coverage. This is a collection-path
  acceptance test only and is excluded from training. The next dataset should
  be recollected under these defaults before judging end-to-end availability
  or changing Mapper/S/F/V66/V67.

## 2026-07-29 decision 151: accept the fixed-6-mm recollection as raw evaluation evidence

- The new dataset `stage3-generalization-fixed6mm-20260729-v1` is generated
  from seed `2026072911` and contains six spin plus six combined-motion sessions
  with disjoint IDs and parameter tuples. All 12 complete on their first
  attempt; the durable manifest, per-session results, raw observations and
  exact-exposure truth are retained outside Git as protected data.
- The audit exact-joins all 35,554 observation frames. All are native
  `wide_6mm`, and every durable result records `dual_focal=false`. Overall
  target-3 coverage is 97.83% versus the previous pipeline's severe admission
  loss; combined motion reaches 98.36% and spin reaches 97.33%. The residual
  observation loss is concentrated at the long-range boundary: `spin-02` at
  about 6.60 m has 86.35% target-3 coverage, while four near/mid spin sessions
  are at least 99.56%.
- This validates recollection and the removal of the 16 mm crop as a major
  recovery, but it is not yet an end-to-end F/V67 metric. Rebuild PnP/history
  windows and evaluate the complete frozen chain next. Do not train or select
  a checkpoint on this raw-coverage result, and do not call these inspected
  sessions an untouched final test split.

## 2026-07-29 decision 152: evaluate one causal range-derived flight-time query

- Replace the earlier presentation-only future-time scatter with a true
  fire-timing diagnostic. For every common-usable window, take the current
  visible-armor range produced by the frozen Mapper/S/H chain and compute
  `tau = range / 22 m/s`. Pass only this continuous `tau` through the existing
  F/V66/V67 query interface. Bullet speed and ballistic metadata are not added
  to history, candidate or refiner features.
- Build the target at that exact `tau` from the qualified constant-motion truth
  state and the existing 1 ms anonymous visibility rollout. Truth determines
  only the offline label and the plot's exact q0 distance axis; it never
  determines the causal query time and never enters model forward.
- Report rotation and combined motion separately. Each output owns one
  distance/error scatter and one fixed 1 m-bin CSV table over `[1,7)` with
  central/tail metrics and coverage. Bins below 100 windows remain visible but
  are explicitly descriptive. Retain three standard-split sessions unopened
  by the derived/evaluation chain and do not update or select any model.

## 2026-07-29 decision 153: retain the fixed-6-mm ballistic r4 diagnostic

- The r4 evaluation runs from clean commit `f36acc4` and evaluates exactly one
  query per common-usable window at `tau = frozen upstream current range /
  22 m/s`. All 910 label q0 positions reconstruct bit-exactly, test remains
  unopened, and Mapper/S/H/V50/V66/V67 state hashes are unchanged before and
  after evaluation.
- Rotation has 263 windows from four sessions and reaches
  65.72/25.98/180.03/189.51 mm Mean/P50/P95/P99. Selection is correct in
  262/263 windows; the eligible 3--4, 5--6 and 6--7 m bins have P50 error
  21.40, 49.09 and 174.74 mm. The range trend is therefore real on this
  eligible subset and is not mainly a selector failure.
- Combined motion has 647 windows from five sessions and reaches
  87.33/33.18/330.24/865.45 mm. Selection is correct in 516/647 windows:
  correct-selection Mean/P50/P95 is 52.12/27.21/138.98 mm, while wrong-selection
  Mean/P50/P95 is 226.03/197.50/660.12 mm. Combined hard error remains strongly
  selection-dominated, and distance alone is not a monotone error predictor.
- V67 does not improve this ballistic-query endpoint over frozen V66 overall:
  Mean/P50/P95/P99 is 81.09/31.12/312.75/605.93 mm versus
  80.96/27.20/312.34/598.95 mm. Retain V67 as the earlier fixed validation
  endpoint, but do not train or promote another final-position residual from
  this diagnostic.
- Raw target-3 observation coverage was 97.83%, yet only 910/3,514 physical
  windows are PnP/S/F common usable: 25.90% overall, 18.01% spin and 31.50%
  combined. Fixed 6 mm repaired detector continuity but did not by itself
  repair strict association/history admission. This admission boundary and the
  combined selector are separate next-stage problems.

## 2026-07-29 decision 154: define admission around the actual observed stream

- The old PnP gate answered the wrong question: it required all 32 exposures
  to contain the plate preselected by a clean-truth nearest-range rule. All
  22,857 rejected history events still contain one or more PnP candidates, so
  those failures are semantic mismatches, not missing observations.
- The v2 history primary is chosen from actual PnP candidates. q0 uses the
  PnP-range-nearest actually observed candidate; past events are smoothed backward
  through temporary oracle-associated handles and permit only same or adjacent
  cyclic transitions in one rotation direction. A 20 mm switching hysteresis
  prevents PnP/range jitter from becoming a false cut. When older observations
  are incompatible, the coherent recent suffix is retained and the older
  prefix is masked. At least eight active events are required, matching the
  existing upstream minimum rather than inventing a new count threshold.
- Temporary truth handles exist only during offline construction. The saved
  neural contract remains anonymous relative positions, masks and signed
  steps; the S sidecar retains the per-window C4 shift/direction reversal and
  does not export a physical armor ID. All q0, history, candidate and future
  labels are rebuilt from the same actually observed q0 handle.
- This is an input/label definition change, not authorization to retrain.
  Mapper, S, H, trajectory, selector and final refiner remain frozen until the
  complete v2 dataset is evaluated. Retraining will be considered only if the
  larger coherent coverage exposes a systematic frozen-model degradation, and
  then only the responsible partition will be changed.

## 2026-07-29 decision 155: accept observed-stream admission and require distribution adaptation

- The complete v2 build retains 2,627/3,514 windows (74.76%), compared with
  910/3,514 (25.90%) under the old gate. Rotation rises from 263 to 994 usable
  windows and combined from 647 to 1,633. This is the intended recovery: actual
  observed candidates are retained, an incompatible older prefix is masked,
  and no physical armor ID becomes a model input or saved label.
- A final reviewer found that the two PnP mappers formed a delta between a
  zeroed inactive time and the first active event. Commit `ae20bb7` fixes both
  paths by requiring adjacent active events and adds a prefix regression. All
  309 Stage3 tests plus architecture and consumer-boundary checks pass. A clean
  rerun changes the aggregate mean by only 0.10 mm, proving this bug was real
  but not the source of the large frozen-model error. The final sidecar audit
  covers 1,232 partial-history windows and finds zero non-contiguous suffixes,
  zero first-active/valid-delta anomalies and zero S/F mask mismatches.
- The clean frozen rerun at one range-derived ballistic query per window keeps
  all Mapper/S/H/V50/V66/V67 state hashes unchanged. Rotation reaches
  290.57/272.26/681.71 mm Mean/P50/P95 with 45.17% selection; combined reaches
  235.54/126.84/804.15 mm with 48.74%. V67's position residual remains small
  (13.30 mm rotation mean, 13.03 mm combined mean), so final-position residual
  tuning is not the responsible intervention.
- The degradation is distributional, not evidence that the previously trained
  trajectory suddenly forgot the old task. On the old strict subset, rotation
  remains 65.72/25.98/180.06 mm and 99.62% selection; combined remains
  81.84/27.90/336.10 mm and 78.21%. On new-only windows those become
  371.46/358.47/715.57 mm with 25.58% selection and
  336.40/281.62/895.04 mm with 29.41%, respectively.
- Partial suffixes are the largest directly observed split. Full 32-event
  histories give rotation/combined means of 185.76/112.38 mm and selection of
  70.64%/67.09%; 8--15-event suffixes give 377.41/455.92 mm and
  21.75%/21.98%. Changed observed-q0 roles are also poor: 408.26/401.62 mm mean
  and 12.71%/28.04% selection. Because correct-selection errors also rise to
  179.96 mm rotation and 108.52 mm combined, selector-only training is
  insufficient. The next stage must explicitly adapt the history/trajectory
  representation and selector to the v2 stream; keep V67 frozen until that
  upstream contract is repaired.

## 2026-07-29 decision 156: primary choice must be reproducible from PnP observations

- Decision 155's r1 metrics are retained as a superseded diagnostic, not an
  accepted training baseline. The r1 candidate set was observation-derived,
  but q0 ordering and the backward DP range cost used truth coordinates
  reprojected into each exposure. That rule cannot run when truth is absent.
- The primary stream now uses only each associated PnP observation's
  exposure-local horizontal range for q0 choice, ties, historical DP costs and
  switching hysteresis. Oracle truth is still permitted offline to associate
  unordered PnP rows with anonymous supervision handles and to construct
  future labels; it no longer chooses the forward history role.
- This correction does not add physical identity, a physics decoder or truth
  input to the model. Rebuild to a new non-overwriting dataset and rerun frozen
  acceptance before deciding what to retrain; do not reuse r1 coverage or
  frozen-error numbers as final evidence.

## 2026-07-29 decision 157: future role comes from exact-query PnP, not truth visibility

- The first observed-q0 label path was invalid when PnP and clean visibility
  chose different q0 roles. All 425 such r1 windows switched back to the clean
  nearest role at the first 1 ms dense frame, creating a synthetic boundary
  unrelated to the actual observation stream. Those labels and dependent r1
  metrics remain superseded.
- For every exact future query, require an available, usable, non-ambiguous
  observation frame with at least one actual PnP candidate. PnP horizontal
  range alone selects the supervised role. The temporary truth slot then
  supplies that role's physical XYZ target; noisy future PnP coordinates never
  become a regression target or forward input.
- Query arrays are sorted by tau before anonymous signed-step unwrapping and
  restored to their original order afterward. Dense truth may determine only
  whether the observed role has a unique reachable integer turn count within
  one adjacent step of the clean reference. It may not replace the PnP-selected
  role. Direction reversal, opposite/incoherent reachability, missing frames,
  ties and ambiguous queries mask that query only; a valid history window is
  retained whenever at least one positive-time query remains.
- Direct independent future-PnP selection creates direction reversals in
  42.7% of windows, while the coherent query rule retains 87.3% in the full r1
  audit. The rebuilt two-session diagnostic retains 733/850 windows (86.24%),
  89.19% of positive-time queries and zero unmasked reversals. Proceed to the
  full non-overwriting rebuild only from a clean commit; raw future PnP streams
  remain a later consistency audit, not a blocker for this label definition.

## 2026-07-29 decision 158: accept r2 semantics and retire the old frozen-error conclusion

- The full non-overwriting r2 parent/SF chain is built from clean commit
  `1cc6d1e`. It retains 2,559/3,514 common-usable windows (72.82%):
  992/1,460 rotation and 1,567/2,054 combined. Parent `pnp_forward_usable` and
  SF `pnp_sf_common_usable` agree on every sample; S alone can admit 85 more
  windows, which are rejected only by the parent history/future contract.
- Positive-query retention inside accepted windows is 79.70% for rotation and
  90.25% for combined. Every unmasked target sequence is monotone with zero
  direction reversals. The remaining window failures are 808 short coherent
  suffixes and 147 cases where observed q0 cannot seed an adjacent-only future
  label; there are no q0 observation failures. Validation rotation remains a
  difficult, imbalanced boundary: only 181/412 windows are usable, mainly
  because 228 histories have fewer than eight coherent events.
- The frozen exact-query evaluation is complete over 17,279 queries with all
  Mapper/S/H/V50/V66/V67 hashes unchanged and test unopened. Overall final
  Mean/P50/P95/P99 is 251.43/136.25/943.74/1391.56 mm with 63.90% switch
  accuracy. Rotation is 328.92/199.00/1124.99/1463.31 mm and 55.80%; combined
  is 204.63/97.79/743.76/1307.20 mm and 68.79%.
- Therefore the earlier statement that rotation is already good applies only
  to the superseded strict complete-history subset. Under the accepted actual
  observation stream, rotation is worse than combined. V67 changes V66 by only
  about 1--2 mm while its residual magnitude is about 14 mm mean; neither
  final-position refinement nor more selector epochs can recover motion
  evidence discarded by the single selected-stream encoder.
- The old ballistic evaluator still constructs a future role from dense
  truth-nearest visibility and must not be used with decision 157. Its failed
  r3 attempt is not a model result. Until an arbitrary-time label is rebuilt
  from a nearby actual observation, the accepted r2 comparison is exact-query
  only. No new training architecture is approved by this decision.

## 2026-07-30 decision 159: authorize an explicit anonymous vehicle MotionContext

- The user authorizes the structural experiment proposed after decision 158.
  The old single-handle F and its V50/V66/V67 descendants remain frozen
  baselines. They are not widened with hidden cross-handle pooling. Instead a
  separately named vehicle-level MotionContext consumes the existing causal
  four-handle anonymous observation memory and provides shared evidence to new
  trajectory and selection heads, as anticipated by decision 122.
- MotionContext receives only mapped `[T,4,3]` window-local handle observations,
  masks, primary/event time and signed history transitions, plus frozen H q0
  relations, uncertainty, support and age. It must be invariant/equivariant to
  C4 origin changes and consistent under direction reflection. A handle is a
  temporary relative cyclic memory location, never a persistent physical armor
  identity. Motion class, pair/session IDs, truth state and future PnP are
  forbidden forward inputs.
- The trajectory head is a learned continuous-time candidate operator. It may
  enforce exact tau-zero identity through its parameterization, but may not
  contain a circle/ellipse equation, constant-twist rollout, hand-written
  switch schedule or physical-ID lookup. The selector learns sample-specific
  direction and ordered boundary intervals from the same MotionContext.
- The r2 run is a bounded structure pilot, not a deployment or checkpoint
  promotion. Mapper, V19-r2 S and H remain frozen. Train in three fixed stages:
  true-branch trajectory, selector with detached candidate positions, then a
  low-rate joint stage that must not regress conditional trajectory metrics by
  more than 5%. Validation uses natural suffixes; random 8--31-event suffix
  shortening is training-only.
- The current SF data remains explicitly oracle-associated and
  `deployable_pipeline=false`; the pilot tests whether retained multi-handle
  evidence fixes the representation bottleneck, not whether online association
  is solved. Formal training requires a larger decision-157-compatible native
  6 mm corpus and preserves test sealing.
- The user subsequently moved this pilot from the local Windows GPU to the
  rented RTX 3090 server and explicitly removed the environment-name
  restriction. The runtime may therefore use any Windows or Linux Python
  environment with a working CUDA PyTorch stack; this does not weaken any
  data, hash, diagnostic or test-sealing gate.

## 2026-07-30 decision 160: selection count is not the current position bottleneck

- The fixed 1,200 trajectory + 600 selector + 300 joint run completes normally
  at update 2,100 in 416.2 seconds. It is an oracle-associated diagnostic, not
  a deployable pipeline; Mapper/S/H remain hash-identical, test is unopened and
  no intermediate validation checkpoint is selected.
- At the final endpoint, overall conditional/hard Mean/P50/P95 is
  207.63/212.74, 141.15/153.51 and 606.51/630.37 mm. Rotation conditional/hard
  Mean is 280.41/280.67 mm; combined is 178.17/185.26 mm. Short 8--15-event
  histories remain the dominant observed failure at 377.82/395.33 mm Mean.
- The reported 61.44% selector score is exact signed crossing-count accuracy,
  not physical armor-role accuracy. Candidate counts separated by four address
  the same anonymous physical role and deliberately share one trajectory, so
  an exact-count error can have zero XYZ consequence. A modulo-4 role score and
  paired hard-minus-conditional error are required before any selector claim.
- The selector stage is effective in its own metric: update 1,200 to 1,800
  changes overall hard Mean from 243.16 to 210.52 mm and exact-count accuracy
  from 24.87% to 62.50% while conditional output stays fixed. The final joint
  stage gives no net validation benefit: hard Mean becomes 212.74 mm and exact
  accuracy 61.44%. It therefore must not be extended by adding epochs.
- Trajectory learning is imbalanced across the two motions even without a
  motion-class input. From initialization to update 1,200, conditional Mean
  improves from 318.22 to 179.44 mm for combined motion but regresses from
  219.23 to 270.49 mm for rotation. This is evidence for shared-head negative
  transfer or insufficient observable state, not a reason to expose the truth
  motion class to the network.
- The next action is evaluation-only. At one causal query per validation
  window, use frozen-upstream range divided by 22 m/s, reconstruct the exact
  future-visible truth label from the sealed truth history, and publish one
  distance/error scatter plus one table for rotation and combined motion. The
  evaluator must report exact-count and modulo-4 role accuracy, q0 error and
  anchor-relative displacement error without changing any weight.
- Do not compare this 587-window validation result directly to the prior
  all-r2 V67 aggregate. Recompute every retained baseline on the identical
  window/query set before claiming an improvement. A later structural run must
  prioritize short-history trajectory evidence and motion-regime separation;
  selector recalibration is secondary.

## 2026-07-30 decision 161: ballistic-time evaluation confirms a trajectory bottleneck

- The frozen update-2,100 model is evaluated once per validation window at
  `norm(frozen Mapper/S/H current position) / 22 m/s`. Exact dense truth builds
  labels only; `_forward_only` proves that truth, session, pair and motion class
  do not enter the network. Six of 587 windows jump directly from the observed
  q0 source to its opposite source under the adjacent visible-stream contract;
  they fail closed with full audit records. The accepted metric denominator is
  581/587 (98.98%), split 176/181 rotation and 405/406 combined.
- Rotation hard Mean/P50/P95 is 267.18/225.54/667.62 mm, versus conditional
  255.70/208.24/616.61 mm. Exact signed-count accuracy is 50.00%, while
  modulo-4 role accuracy is 56.25%; 11 of 88 exact-count errors are the same
  physical role. Combined hard Mean/P50/P95 is 166.10/156.08/405.79 mm versus
  conditional 159.56/134.85/395.70 mm; both selection definitions are 71.36%.
- Selection is not the main XYZ error source. Hard minus conditional Mean is
  only 11.48 mm for rotation and 6.53 mm for combined. Even among wrong-role
  samples the mean excess is 26.24/22.81 mm. In the scatter plots, hard and
  conditional points mostly overlap while a broad vertical spread remains at
  the same distance.
- Upstream q0 Mean/P50/P95 is 94.79/38.75/317.94 mm for rotation and
  35.70/18.27/122.27 mm for combined. Anchor-relative conditional displacement
  Mean is still 267.17/157.49 mm, so neither perfect role choice nor q0 repair
  alone can remove the trajectory error. Rotation and combined occupy different
  distance bands in this validation, so the plots diagnose each motion but do
  not establish a cross-motion distance law.
- The next model change is therefore structural and trajectory-first. For each
  anonymous handle, temporal state must update only on its truly visible
  events and use the real elapsed time across gaps. A history-conditioned
  latent mixture may separate rotation-like and translation-plus-rotation-like
  regimes, but no truth motion-class input is allowed. The decoder predicts a
  query-independent trajectory state once and evaluates a shared learned
  continuous-time basis at arbitrary tau, rather than independently regressing
  every query.
- The firing selector becomes a two-level task: modulo-4 role is primary and
  exact signed crossing count is auxiliary. Training is equal-weighted by
  window, motion and history bin, followed by a short joint stage and a final
  frozen-trajectory selector recalibration. More epochs on the current head and
  another final-position residual remain closed.

## 2026-07-30 decision 162: v2 is a separate visibility-driven role model

- Preserve `AnonymousVehicleFutureModel` and its v1 training schema unchanged
  so the fixed update-2,100 artifact remains strictly loadable and its recovery
  source contract stays meaningful. The structural successor is separately
  named `VisibilityAwareAnonymousVehicleFutureModel` with an independent v2
  run schema; a v1 state dictionary must fail strict loading into v2.
- Each handle's history is compacted to visible events before the shared causal
  encoder. Inactive or active-but-invisible coordinates are sanitized and do
  not occupy a temporal slot. Same-handle velocity and elapsed features use the
  timestamps of the two most recent visible observations, including arbitrary
  gaps. Q0 support remains the only fallback for a handle with no visible
  history.
- The trajectory head produces per-role, per-latent-regime coefficients once
  from the causal history. A shared learned basis reads only continuous query
  time; explicit multiplication by normalized tau makes q0 identity exact.
  The regime gate is inferred from the anonymous vehicle history and never
  receives the offline rotation/combined label. This is a learned operator,
  not a circle, ellipse, constant-twist or switch-time physics decoder.
- Candidate-row metadata is not allowed to split a physical role. Every
  candidate trajectory is gathered from `step mod 4`, and the complete unique
  signed range is validated. The primary role head predicts four relative
  roles. The ordered signed-crossing distribution is normalized within those
  roles, so its modulo-four aggregate exactly equals the primary role
  probability, while final firing XYZ depends only on selected role.
- Losses first average valid queries inside each window, then average windows;
  the existing sampler balances motion class and history-length strata only in
  the offline loader, never in forward. The immutable four-stage pilot is
  trajectory, selector, short joint with selector context detached, and final
  frozen-trajectory selector recalibration. Exact crossing CE has weight 0.15
  relative to role CE 1.0.
- Acceptance is structural before statistical. The focused suite covers
  visibility isolation, sparse same-handle elapsed time, complete candidate
  range, query/candidate permutations, C4/reflection, tau zero, same-role
  identity, role aggregation, forbidden inputs, four actual optimizer steps
  and joint gradient isolation. Twenty-one focused tests, all 349 Stage3 tests and
  both repository boundary checks pass. A small r2 CUDA run and resume test are
  required before the one full diagnostic pilot; neither run may open test or
  promote a checkpoint by validation.
- The first deliberate interruption at update 575 resumes deterministically
  from immutable update 525: repeated update-550 and update-575 losses are
  identical. It also exposes a recovery-metadata omission: the verified joint
  gradient-isolation flag was not stored in the checkpoint and became false
  after resuming directly into recalibration. The v2 recovery payload now
  persists and restores this gate, with a focused regression test. The affected
  smoke is diagnostic-only and cannot pass the final recovery acceptance.
- After the fix, a fresh CUDA recovery run is terminated during recalibration
  with immutable update 150 as its latest checkpoint. Resume repeats both
  logged update-175 and update-200 objectives exactly, finishes the fixed
  update-300 endpoint, preserves all frozen Mapper/S/H hashes and records the
  isolation gate as true. The checkpoint SHA-256 is
  `acc136b8f169f13f792aee889c7747456a04c959348bd3619c74a2799c956911`.
- On its deliberately tiny 24+24-window validation capacity slice, the
  corrected smoke reaches 35.96 mm conditional/hard Mean, 100% modulo-four
  role accuracy and 75.22% auxiliary exact-count accuracy. This proves the v2
  optimization and role factorization can close a small task; it is not a
  generalization estimate and does not authorize hyperparameter selection.
  The one full r2 pilot now uses the predeclared 1,200/600/300/300 schedule.

## 2026-07-30 decision 163: reject the v2 endpoint and isolate routing supervision

- The single full v2 run completes at fixed update 2,400 in 372.1 seconds from
  clean commit `ef75d8a`. Checkpoint SHA-256 is
  `2b1af57f0cac21ed564b4a9031634dc6dcd7e04ebcf6b3e9ad197d3e497e68cb`.
  Mapper/S/H hashes remain unchanged, joint gradient isolation is verified,
  validation never selects a checkpoint and test remains unopened.
- The accuracy gate fails. At the exact validation queries, v2 overall
  conditional/hard Mean/P50/P95 is 217.38/229.04, 159.13/187.51 and
  682.39/693.36 mm, versus v1's 207.63/212.74, 141.15/153.51 and
  606.51/630.37 mm. The 8--15-event slice is 395.96/418.37 mm versus
  377.82/395.33 mm. Final recalibration improves modulo-four role accuracy only
  from 50.94% to 51.33% while regressing hard Mean from 227.64 to 229.04 mm.
- The same-window trajectory stage repeats the old negative-transfer pattern:
  rotation conditional Mean changes from 196.22 mm at initialization to
  299.78 mm at update 1,200, while combined improves from 310.60 to 188.14 mm.
  The latent expert count alone therefore does not solve regime separation.
- Ballistic-time evaluation from clean commit `1235b03` keeps 581/587 windows
  and all frozen hashes. Rotation conditional/hard Mean is 303.63/310.65 mm
  with 56.82% modulo-four role accuracy, compared with v1's 255.70/267.18 mm
  and 56.25%. Combined conditional improves slightly from 159.56 to 154.33 mm,
  but role accuracy falls from 71.36% to 34.07% and hard Mean rises from
  166.10 to 173.66 mm. Distance/error plots show the same broad vertical body,
  not a monotone range failure.
- The three latent experts did not numerically collapse: gates are nearly
  one-hot and coefficient pairs remain distinct. They instead specialize and
  route incorrectly. Among short rotation windows, 80.61% route to expert 1,
  whose conditional window Mean is about 533 mm; counterfactually choosing the
  best already-trained expert lowers that group to about 209 mm. Long rotation
  routed to expert 2 reaches about 9 mm on 64/66 windows. Across motion/history
  slices, oracle best-of-three window Mean is roughly 15--237 mm, substantially
  below the learned gate's 23--553 mm.
- Future truth may supervise a training target but may never enter forward.
  The next bounded experiment therefore freezes all expert trajectories,
  derives a detached per-window best-expert label from training targets, and
  trains only the history gate to predict it. This is a routing-identifiability
  test, not extra epochs on the failed objective. If the gate generalizes, a
  later v3 will use multiple-choice expert training followed by history-only
  routing; if it cannot, expert selection is not identifiable from the current
  causal state and the MotionContext representation must change again.

## 2026-07-30 decision 164: reject post-hoc discrete routing as the next F basis

- The hash-locked v70 trajectory bank was frozen and only its history gate was
  trained for 600 updates from clean commit `e32397e`. Future truth generated
  one detached best-expert label per window using the true modulo-four role;
  forward still consumed anonymous causal history only. A real interruption at
  update 150 resumed with update-175/200 objectives identical to an
  uninterrupted control. Mapper/S/H, MotionContext and all trajectory-expert
  hashes remain unchanged; test stayed sealed.
- The fixed endpoint
  `20260730-v71-history-router-full-r1/checkpoints/checkpoint-update-000600.pt`
  has SHA-256
  `38118a8ad88a290ea8e107d5feb551a89416b7be5cc8b7cd66fdba88db94631e`.
  It fails the predeclared gate. Train/validation macro recall is
  95.52%/48.70%; validation closes only 5.08% of the overall oracle gap and
  9.47% for 8--15-event rotation. It does not beat the expert selected from
  train overall. Of 185 validation windows whose best expert is expert 1, only
  one is classified correctly.
- Label ambiguity is not the explanation: 90.29% of validation windows have a
  best/second-best margin of at least 20 mm. Combined validation needs expert 1
  on 183/406 windows while the learned gate chooses it zero times; the same
  gate classifies expert 1 accurately on the training combined sessions. The
  arbitrary expert identity therefore does not have a stable cross-session
  observable meaning in the current MotionContext.
- A session-level read-only audit confirms that the aggregate train score hides
  the same instability. Three combined training sessions reach 97.91--99.21%
  gate accuracy, while held-out combined is 49.75%; one rotation training
  session is already only 44.54% despite the other two reaching about 98--99%.
  Held-out combined requires expert 1 on 182/406 windows but predicts it zero
  times. This rules out interpreting the failure as a single unlucky aggregate
  threshold and strengthens the cross-session semantic-shift diagnosis.
- The mismatch is also explicit in the v2 interface: each expert's trajectory
  coefficient reads absolute `current_position_m`, but the history gate does
  not. The future-best expert label may therefore change with absolute pose,
  anchor error or session-specific PnP error direction that the gate cannot
  observe. The next trajectory latent and selector must be translation
  equivariant: absolute current position may be added back only after a
  history-relative delta has been predicted.
- Hard-router ballistic evaluation from clean commit `ba767ce` retains 581/587
  windows and all frozen hashes. Rotation conditional/hard Mean is
  306.39/305.71 mm; combined is 153.46/173.92 mm. This is materially unchanged
  from the rejected v70 ballistic result, so neither more gate epochs nor
  post-hoc relabelling of the same three experts is authorized.
- The next F redesign must remove the assumption that one globally named
  discrete expert can be recovered from current MotionContext. It must learn a
  session-invariant continuous motion representation from normalized causal
  increments and predict the future trajectory directly. Any multi-hypothesis
  outputs must be trained jointly with stable semantics or evaluated as an
  uncertainty set; a future-derived expert ID may not become a deployment
  target again.

## 2026-07-30 decision 165: v3 is one translation-equivariant continuous role field

- V3 is an independent model and training schema; v1/v2 remain loadable and a
  v2 state dictionary fails strict loading into v3. It reuses the verified
  per-handle visible-event MotionContext, but removes the latent expert bank,
  regime gate, future-best-expert supervision, exact signed-crossing head and
  soft-role averaged-position loss.
- Every learned module consumes only relative anonymous history, relative q0
  support and continuous query time. `current_position_m` is used once, after
  prediction, as `role_position = current + role_delta`. Common translation
  leaves coefficients, deltas and role logits bit-exact and translates output
  positions only; the delta has zero gradient with respect to current position.
- The firing target is the q0-primary-relative modulo-four role. Physical ID,
  motion class, session identity, truth fields, future PnP and legacy candidate
  metadata are not forward inputs. Training truth gathers the correct role for
  direct future-position Smooth-L1; the selector uses role CE plus distance
  risk computed from detached role trajectories.
- The fixed schedule has three stages only: 1,200 trajectory updates, 600
  frozen-trajectory selector updates and 300 gradient-isolated joint updates.
  There is no fourth recalibration stage and validation never selects a
  checkpoint. The sampler first balances motion, then session within motion,
  then history bin within session; prefix dropout stays inside the chosen bin.
- Recovery retains all global RNG states, hierarchical sampler state, prefix
  generator, optimizer/scaler state and source/upstream hashes. Mapper/S/H and
  inactive-stage hashes are checked before every checkpoint and at completion.
  Twenty-one focused v3 tests and all 379 Stage3 tests pass. Full training is
  blocked until a local CUDA capacity smoke and deliberate resume check pass.
- Subsequent compute uses the local Windows `yolov8` CUDA environment. The
  rented RTX 3090 instance remains powered off and retained; it is not released.

## 2026-07-30 decision 166: v72 rejects raw-position temporal shortcuts

- V72 passed structure, capacity and recovery gates. A forced stop at update
  150 resumed with updates 160--200 losses and final state SHA bit-exact to an
  uninterrupted control. The one fixed 2,100-update endpoint completed in
  405.1 seconds from clean commit `6405148`; checkpoint SHA-256 is
  `d7de1bf6daf9da142a011f48d1f9e1c65d3931f945b21581cf50899107d2239f`.
  Mapper/S/H hashes stayed unchanged, joint isolation passed, validation did
  not select a checkpoint and test remained unopened.
- The endpoint fails cross-session accuracy rather than optimization closure.
  Train overall conditional/hard Mean is 65.01/92.17 mm with 85.07%
  session-macro role accuracy; heldout is 228.52/234.24 mm and 53.96%.
  Rotation heldout conditional/hard Mean is 265.30/284.72 mm and combined is
  213.64/213.82 mm. Hierarchical session sampling therefore does not itself
  prevent the temporal representation from memorizing session fingerprints.
- Ballistic range/22 m/s evaluation from clean commit `c6e257c` retains
  581/587 windows. Rotation conditional/hard Mean is 250.51/287.09 mm with
  52.84% role accuracy; combined is 166.97/201.22 mm with 46.42%. The scatter
  bodies are broad within distance clusters, so another range calibration or
  selector-only epoch run is rejected.
- Root cause is narrowed to the reused v2 MotionContext: its TCN still consumes
  raw historical relative XYZ and q0 quality fingerprints, despite the v3
  final head being translation equivariant. V4-A replaces this with tokens
  built from same-handle adjacent displacement, true visible-event gap,
  velocity, time/switch masks and first-visible offset relative to q0 geometry.
  Q0 relation remains the S-owned static armor geometry; raw history origins,
  sigma/confidence/age/support-class are excluded from learned motion features.
- Naive synchronized time compression is explicitly deferred. It changes
  camera cadence and scales linear and angular speed together, so it could
  introduce a new shortcut. A future augmentation must preserve the captured
  dt distribution through causal resampling and must beat an identical v4-A
  no-augmentation control. V4-A first isolates the representation change.

## 2026-07-30 decision 167: supervise a stable physical state and hard-isolate it from F

- V73 completed the v4-A structure test rather than merely undertraining it.
  Its fixed endpoint preserves all frozen hashes and test sealing, but train
  conditional/hard Mean of 69.34/89.27 mm becomes 230.85/247.35 mm on unseen
  sessions. Rotation is 301.88/320.07 mm and combined 202.11/217.92 mm.
  Hence raw-position deletion slightly helps combined motion while removing
  necessary rotation phase; the unrestricted high-dimensional state still
  memorizes sessions. Additional v4 epochs and small feature edits are rejected.
- V5 gives the temporal representation stable cross-session semantics. A
  pure-increment context, before q0 geometry or first-origin injection,
  predicts four physical values in the tracker/chassis frame: target-center
  velocity XYZ and physical yaw rate. These labels already exist in the
  qualified truth-history dataset and are strictly joined to paired windows by
  `(split, session_id, exposure t0_ns)`. All 2,559 paired rows match uniquely;
  missing, duplicate and split/motion mismatch counts are zero. Extra truth
  rows are allowed and audited. Test is never opened.
- Anonymous slot reflection only applies `[0,3,2,1]` role reordering and
  reverses signed switch metadata. It does not reflect XYZ coordinates, so all
  four physical motion labels, including yaw rate, remain unchanged. Truth
  manifest, key-set and label hashes, exact join counts and normalization are
  part of the immutable training contract.
- Truth is loss/diagnostic data only. Forward receives no motion target,
  session, timestamp, motion class, physical ID or future truth. The explicit
  learned decoder consumes predicted normalized 4D state, S-owned ordered q0
  relative geometry/support and continuous query time. Current absolute
  position is added only at the end. Zero 4D state yields exact zero dynamic
  displacement for arbitrary geometry and time; no analytic physics rollout
  or teacher-forced truth state is used for formal prediction.
- Four fixed stages prevent the 4D floats from becoming another hidden latent:
  state pretraining updates only pure-increment context and state head;
  trajectory training freezes them and reads a detached prediction; selector
  training freezes both state and trajectory; decoder-joint training keeps
  the state encoder frozen and detaches selector gradients from trajectory
  modules. Recovery saves all three optimizers, scaler, global RNG,
  hierarchical motion/session/history/stationary-active sampler and prefix RNG.
- Validation is heldout-session primary because each capture session contains
  only a stationary state plus one or two nearly constant active states. Train
  state loss alone is not evidence. Report physical velocity/yaw error,
  active yaw sign, combined velocity cosine, truth-state decoder headroom,
  conditional/hard future errors and session macro. Ten focused v5 tests and
  all 406 Stage3 tests pass before CUDA work begins. The rented RTX 3090 stays
  shut down but retained; v5 runs locally in the Windows `yolov8` environment.

## 2026-07-30 decision 168: break the session shortcut with ACK-bound multistate data

- The completed v5 endpoint isolates the remaining failure. Train versus
  heldout physical-state error is 0.155 versus 0.950 m/s for velocity and 0.46
  versus 6.984 rad/s for yaw rate. Heldout conditional future Mean is 267.53
  mm, but the identical learned decoder reaches 141.74 mm when evaluated with
  truth state. More v5 epochs, loss-weight tuning and another small encoder
  edit are rejected; the next controlled variable is the data distribution.
- Existing captures bind almost every session to one active `(v,omega)` state
  plus startup stationary frames. V5 can therefore map session-specific PnP,
  range and phase fingerprints to the label without learning a reusable
  motion estimator. The new formal capture keeps distance, 6 mm camera and
  environment fixed within a session while applying 12 independently sampled
  continuous motion blocks. There are 12 rotation-family and 12 combined-
  family sessions, one stationary block per session, continuous signed omega,
  and continuous direction/speed/span values inside the existing 6 mm safety
  envelope.
- This is a consumer-orchestration change using the locked Scene Control v1
  SDK, not a simulator change. One control session is created once. Segment
  zero uses `--stage3`; later segments use `--stage3-update` and only call
  `setRangeTargetMotion`. The simulator repository, SDK and Release remain
  read-only, so `SIMULATOR_CHANGE_APPROVAL_REQUIRED` is not triggered.
- Every successful motion ACK creates a consumer-owned monotonically numbered
  `motion_command_epoch`. `applied_frame_seq` and `applied_timestamp_ns` must
  increase; SDK `command_id` is recorded but is client-local and may restart
  because every CLI invocation creates a new SDK client. Segment intervals are
  half-open `[ack_timestamp,next_ack_timestamp)`, with the final end bound by
  the last exact truth frame. Results are reusable only when the captured
  manifest hash, segment count and complete ACK plan match exactly.
- Scene Control v1 has no idempotency token, so an update is attempted exactly
  once inside a control session. A command/ACK failure invalidates the entire
  run; only the outer runner may recapture the session from a fresh simulator,
  raw run directory and control-session identity. The formal capture is frozen
  before execution: exactly 24 sessions (12 rotation, 12 combined), 12 segments
  per session (1 stationary, 11 active), 3 seconds per segment, wide 6 mm only,
  plus immutable manifest SHA and 14/5/5 session split hash. The dataset builder
  must bind that capture-contract SHA and exact split.
- The v2 dataset builder admits a row only when its entire retained observation
  history and every matched future query lie inside one epoch. Exact truth must
  also satisfy constant velocity, constant yaw rate, constant-twist position/
  yaw residual and producer/target/geometry identity across the full history-
  to-future interval. This is row rejection, not query masking. It rejects a
  boundary even when adjacent commands have identical numeric parameters.
  Segment metadata is audit-only and is explicitly excluded from model forward.
- Motion class belongs to the active ACK segment, not the session's rotation or
  combined family. The stationary control block is class 0 and is intentionally
  excluded by the unchanged first v5 A/B loader, while the eleven active
  continuous states per session provide the anti-shortcut supervision. A 2 us
  history/future boundary guard protects later float event-time reconstruction.
  Segment epoch/start/end, full-window bounds, constant-motion flag and complete
  rule-query mask are hash-bound and exact-joined through observable-clean,
  paired PnP and PnP/SF datasets; the S/F last 32 events must also remain inside
  the same segment. Derived manifests record zero join mismatches and the source
  manifest hash. These audit tensors remain outside every model forward input.
- Planned segment count is not acceptance evidence. Before the v5 A/B, produce
  actual per-segment survival counts through base, truth-history, observable,
  paired PnP and PnP/SF data; each heldout session must retain at least 8 of 11
  active segments. Raw records must be scanned for wide_6mm-only provenance.
  Report state and final-position metrics separately for rotation and combined;
  include a session/nuisance-only shortcut baseline and session-bootstrap
  uncertainty so aggregate window counts cannot hide a remaining shortcut.
- A 2-session, 4-segment native smoke verified init/update ordering, stream
  growth, strictly increasing ACK frame/time, crash-safe result hashing and
  deterministic rerun skipping. Its strict tensorization admitted 60 rotation
  and 16 combined samples from intentionally short 1-second blocks; the
  formal 3-second blocks provide larger stable interiors. The smoke dataset is
  diagnostic and not a formal training source. All 412 Stage3 tests, the WSL
  consumer CLI build, architecture check and consumer-boundary check pass.
- The first retraining A/B freezes the exact v5 architecture, loss, update
  counts, initialization policy and Mapper/S/H checkpoints. Predeclared
  heldout targets are velocity error <=0.35 m/s, yaw-rate error <=1.5 rad/s,
  normalized state error <=0.08, conditional future Mean <=190 mm, predicted-
  state versus truth-state conditional gap <=40 mm, and combined modulo-four
  role accuracy >=55%. Failure of state metrics redirects work to explicit
  multi-scale robust state estimation; success with a remaining decoder gap
  redirects work to the decoder/role selector.
- All subsequent compute remains local in the Windows `yolov8` CUDA
  environment. The rented RTX 3090 instance stays powered off but retained and
  must not be released.

## 2026-07-30 decision 169: formal capture source changes require a full recapture

- The first source-bound formal capture (`stage3-multistate-fixed6mm-20260730-v1`)
  accepted four sessions before the batch was stopped. While the bridge was
  appending the exact-truth JSONL, Windows PowerShell could return two pipeline
  objects for `Get-Content -Tail 1`; piping both through `ConvertFrom-Json`
  produced an object array and made scalar timestamp validation fail. The raw
  JSONL records themselves remained individually valid and ordered.
- Final result sealing must stop and join the bridge writer, snapshot the truth
  file length and parse only the file-order last LF-committed record. Bytes
  after the last LF are an uncommitted append fragment; malformed committed
  UTF-8/JSON or a `timestamp_ns` that is absent, non-scalar, non-integer or
  outside Int64 remains a hard error with no fallback to an older record. The
  committed timestamp must still extend beyond the final motion ACK. This
  changes only consumer capture orchestration and does not modify the
  simulator, SDK or Release.
- The v1 capture root, contract, logs, four accepted session results and all
  failed raw attempts are protected diagnostic evidence. They are neither
  deleted nor admitted to training. Because the formal contract binds the
  source commit, the repaired runner requires a new v2 contract and a complete
  24-session recapture; no pre-fix result is carried across the source boundary.
- The rented RTX 3090 remains powered off and retained. Capture, derivation and
  training continue locally in the Windows `yolov8` environment.

## 2026-07-30 decision 170: derived observation data keeps formal test sealed

- Formal v2 capture completed 24/24 source-bound sessions and 288/288 motion
  segments from clean commit `96461ea`. It produced 26 immutable raw runs: 24
  canonical accepted results plus two rejected whole-session attempts. The
  manifest SHA-256 is `70f354e0ea75eb212fcd09e93452b74fc89142cc831c1709f25f7ba050d4f50e`
  and the capture-contract SHA-256 is
  `2dee2e1dff1fee35ac37a2a17a1c03c9bbba02f16add66388e718236d7f24259`.
- The qualified base r2 dataset contains 9,562 samples split as
  5,303 train, 2,072 validation and 2,187 sealed test. Base construction owns
  initial split materialization; every later truth, clean, observation, PnP,
  S/F, audit and training stage must report `test_accessed=false` and must not
  load a test shard.
- The observation-v4 builder previously iterated train, validation and test
  shards while documenting only that training would not use test. That is too
  weak for the formal contract. Observation derivation now enumerates only
  train/validation, records zero test shards opened, and reports counts for the
  actual derivative rather than copying base totals. The paired PnP builder
  rejects an observation manifest that lacks the fail-closed test-access flag.
- The timed-out base r1 directory has `build_state=in_progress` and no dataset
  manifest. It is retained as incomplete diagnostic evidence and is forbidden
  as a downstream source; the independently completed r2 directory is the only
  qualified base for this run.

## 2026-07-30 decision 171: rule-query truth stays out of observation derivation

- `rule_query` is computed only after exact truth checks constant velocity,
  yaw rate, center rollout, identity and geometry over every future endpoint.
  The observation-v4 branch derives from base observations and must not
  reconstruct, assume or copy this truth-owned label.
- Exact PnP pairing therefore requires observation-v4 to match the six
  source-owned ACK fields: motion epoch, segment start/end, history start,
  future end and constant-motion admission. Truth-history, causal-physical and
  observable-clean must additionally contain and exactly match `rule_query`.
  The source row still requires every rule query true and both 2 us guards.
- The first formal PnP-parent r1 attempt failed closed on the former overstrict
  join before producing a manifest. It is retained as incomplete diagnostic
  evidence and will not be used by PnP/SF or training; a new r2 derivative is
  required after clean-commit validation.

## 2026-07-30 decision 172: formal history matches the deployed 32-event estimator

- The first complete v2 derivative chain is diagnostic-only. All three
  rotation validation sessions retain samples in 11/11 active ACK segments,
  while both combined validation sessions retain only 5/11 and fail the
  predeclared 8/11 gate. Raw observation scans show valid wide-6-mm detections
  throughout nearly every segment, so the missing combined coverage is not a
  detector, camera-range or model-visibility failure.
- The base tensorizer retained up to 200 valid events, roughly two seconds at
  the recorded cadence, while every deployed S/F path consumes only the final
  32. Combined linear motion can reverse at its span boundary inside one
  3-second command block. A pre-reversal event that the model never consumes
  therefore made the full 200-event truth window nonconstant and rejected
  otherwise valid post-reversal supervision until almost the segment end.
- Formal rows with ACK segment metadata now select the latest at most 32 valid
  events inside that segment and right-align them in the unchanged `[200,...]`
  base tensors. Legacy rows without segment metadata retain the previous
  latest-200 contract. Segment discovery is moved before the ego-stability
  precheck so both stages use the same segment-local selected history.
- This is a data-contract correction, not a relaxation of truth supervision.
  `history_start_ns` is the first actually retained event; the complete
  retained-history-to-future interval must still have constant velocity,
  yaw-rate, constant-twist position/yaw, producer, target and geometry. The ACK
  interval remains half-open and both 2-us boundary guards remain mandatory.
- A predeclared train/validation-only audit owns the next gate. It left-joins
  every planned ACK epoch, hash-checks opened shards, checks core audit fields
  and history subsets through base/truth/causal/clean/PnP/PnP-SF, recomputes
  base latest-32 timestamps from raw observations, scans every opened frame for
  `wide_6mm`, and records zero test shard/raw opens. Every validation session
  must retain common-usable samples in at least 8/11 active epochs before the
  unchanged v5 architecture, losses and update schedule may train.
- All prior base/derivative attempts and raw captures remain protected and are
  never overwritten or deleted. Rebuilds use new r3/r2 artifact roots. Compute
  remains local in the Windows `yolov8` CUDA environment; the rented RTX 3090
  stays powered off but retained and must not be released.

## 2026-07-30 decision 173: v6 replaces adjacent differences with robust physical-time scales

- The latest-32 rebuild passed its train/validation-only survival audit with
  11/11 active ACK segments in every validation session and zero test access.
  The unchanged v5 control (v77) improves heldout velocity/yaw error to
  0.492 m/s and 1.893 rad/s, but still misses the state gates; conditional
  future Mean is 208.64 mm versus 156.24 mm when the same learned decoder is
  supplied with truth state. The state magnitude, not additional selector or
  decoder epochs, is the next authorized target.
- Read-only stratification shows the strongest within-session correlate is
  combined translation speed. Matched clean-observation substitution improves
  validation state error only about 10--15%; high-speed combined windows still
  fail. Distance, short history and PnP increment noise amplify the error but
  do not independently explain it. The v5 adjacent same-handle velocity divides
  PnP displacement noise by roughly 10-ms intervals, then uses lane last/mean
  and cross-handle mean/max without explicit physical-time scales or robust
  reliability.
- V6 keeps the exact four-dimensional state contract and existing learned
  decoder/selector. It replaces only the state estimator with causal,
  non-overlapping 10/30/70/150/280-ms same-handle displacement bands, an
  unordered two-visible-armor relative-motion edge based on temporal changes
  of `(a-b)(a-b)^T`, a two-pass bounded learned consensus, symmetric cyclic
  interaction and availability-aware per-coordinate scale fusion. The pair
  vector is subtracted before its outer product in autocast-disabled FP32, so
  large common translation cannot numerically leak into the rotation branch.
  Learned weight mass and effective sample size enter handle pooling and scale
  fusion; pair-only evidence may support yaw but not translational coordinates.
  Missing long scales are masked and never reject an otherwise eligible history.
- State forward is a separate six-field API and may read only anonymous
  relative observations, observation/primary/event masks, relative q0 history
  times and switch steps. Its dedicated loss does not instantiate the decoder
  path or read q0/future labels. Physical plate ID, session/epoch, motion
  class, absolute timestamp, current position/range, q0 relation or quality,
  future observations, truth state and labels are forbidden. Truth supervises
  the fused and available per-scale four-vectors only. Scale uncertainty is a
  training/diagnostic output and never reaches decoder or selector. This is a
  learned temporal estimator, not analytic finite-difference rollout or a
  physics decoder.
- The first formal comparison is one complete equal-budget 800-update
  state-only artifact on the same frozen Mapper/S/H, dataset, sampler and seed
  as v77. The exact v77 update-800 learned decoder/selector modules are loaded
  by hash and remain frozen; their before/after hashes must match. The formal
  wrapper cannot silently continue into later stages and binds control
  checkpoint SHA-256 `f823f147...a91b99662` plus contract
  `c36717df...fec53871`. Primary targets are overall velocity <=0.35 m/s, yaw <=1.5
  rad/s and normalized MAE <=0.08. Because the stated objective is specifically
  combined motion, the same pre-run hard gate also requires combined velocity
  <=0.57 m/s, combined yaw <=2.10 rad/s, speed>1.7-m/s combined velocity
  <=0.80 m/s, combined-11 normalized MAE <=0.12 and overall yaw-sign accuracy
  >=0.963. Session macro, all five validation sessions and history bins remain
  mandatory diagnostics. A passing state gate may resume the fixed trajectory,
  selector and joint stages; a failed gate ends the run without extra epochs.
- The implementation gate is 455 Stage3 tests, both repository boundary checks,
  a complete state-only CUDA smoke and a bit-exact 150->200 interruption/resume
  comparison. Compute remains local in Windows `yolov8`. The rented RTX 3090 is powered off
  but retained and must not be released. Test, export, online integration and
  fire control remain sealed.

## 2026-07-30 decision 174: equal-budget A/B binds sampler semantics, not only seed

- The first v6 r1 state gate completed all 800 updates while the parent was
  acting on a final read-only review. It is diagnostic-only even though its
  model and metrics are intact: the run bound the same data, parameters and
  seed as v77 but did not machine-bind the sampler implementation itself. It
  reports overall velocity/yaw/normalized error 0.399/1.621/0.0650 versus v77
  0.492/1.893/0.0791, while combined high-speed velocity remains 1.132 m/s and
  combined-11 normalized MAE remains 0.1795. Six of eight predeclared gates
  fail, so the result would not authorize downstream training in any case.
- Equal-budget comparisons now hash the exact semantic source of
  `motion_state_cells`, `HierarchicalSessionHistorySampler`, `_history_label`
  and `apply_bin_preserving_prefix_dropout` against v77 source commit
  `39a23282160f158f6dfd3278aeb8c0d5e60b14fb`. The control checkpoint and
  manifest must agree on their recomputed contract, run ID, dataset, truth,
  frozen upstream and sampler provenance before training starts. Finalization
  additionally requires the actual sampler strategy and support to equal the
  control. The sampler implementation file is part of source provenance.
- R1 and all checkpoints remain protected diagnostic evidence and are never
  overwritten. The qualified rerun uses a new r2 root from a clean commit.
  Test and later trajectory/selector stages remain sealed regardless of r1.

## 2026-07-30 decision 175: v7 factorizes angular, planar-common and vertical motion

- The sampler-qualified v6 r2 run is complete and protected but fails six of
  eight hard gates. It improves v77 overall velocity/yaw error from
  0.492/1.893 to 0.399/1.621, yet speed>1.7-m/s combined error remains
  1.132 m/s and 4.141 rad/s and combined-11 remains 1.154 m/s,
  4.808 rad/s and 0.1795 normalized MAE. The qualified checkpoint SHA-256 is
  `e30759ab441b9113f07c9bab6c525c55814905a06fcb9443310454ad586782a5`;
  sampler, lineage and frozen-future hashes match the v77 control and test was
  not accessed.
- Read-only interventions reject incremental v6 tuning. Clean observations
  leave high-speed combined error at 1.108 m/s and 4.174 rad/s; all single
  scales fail there, activations are not saturated, and pair-only samples are
  easier rather than harder. The dominant failure is systematic magnitude
  contraction caused by mixing common translation and orbital rotation inside
  one shared latent/four-coordinate projection.
- V7 keeps the external state definition, normalization and exact six-field
  causal API. The angular branch primarily reads unordered normalized
  relative-shape evidence: current/prior `rrT`, delta/rate, matrix commutator,
  dot/norm, causal elapsed time, q0-relative time, switch and effective support.
  A scale without pair evidence may use a same-handle higher-order curvature
  fallback, but that fallback may read only consecutive velocity differences,
  acceleration differences and their normalized cross/dot terms; absolute
  velocity direction/magnitude are forbidden and common constant-velocity ramp
  invariance is tested. It emits signed yaw and reliability through a yaw-only
  multi-scale head. A same-handle plus
  same-unordered-set centroid branch emits planar common motion. Its only
  rotational input is detached predicted yaw/reliability used by a learned
  bounded residual; pair latent never enters directly. Vertical velocity uses
  a separate robust head. Translation cannot enter yaw, no shared head emits
  the four-vector, and uncertainty remains loss/diagnostic-only.
- A training-only common ramp adds `u*t` to 50% of histories and adds `u` to
  the planar velocity label. It supervises yaw invariance and planar
  equivariance; it is absent from deployed forward and never supplies truth yaw
  to the model. Fixed updates 1--250 train angular evidence/head only; 251--600
  freeze angular and train planar/z using detached predicted yaw; 601--800
  jointly calibrate the separated branches while preserving the one-way
  boundary. Mapper/S/H and exact v77-update-800 future modules remain frozen.
- The original eight v6 gates remain mandatory. Additional diagnostics report
  per-session/history bins, PnP-to-clean and predicted-to-truth-yaw translation
  interventions, but do not weaken the gate. If every high-speed combined
  single scale still fails, v7 is rejected without extra epochs. External state
  expansion is not authorized unless truth 4D plus truth q0/current and a fixed
  decoder later show a systematic residual explained by a missing physical
  quantity.
- Compute is local in `D:\Anaconda\envs\yolov8`; the rented RTX 3090 is powered
  off but retained. No cloud resource release or protected-asset deletion is
  authorized.

## 2026-07-30 decision 176: v7 acceptance is checkpoint-bound and crash-consistent

- The formal v7 result is accepted only from the exact fixed
  `checkpoint-update-000800.pt` inside its run root. The finalizer recomputes
  the checkpoint file SHA, contract hash and full state hash, strictly
  reconstructs the declared v7 model/config, verifies every trainable module,
  and requires checkpoint/manifest equality for provenance, validation
  history, final validation, diagnostics, gradient isolation, substage counts,
  transitions and branch hashes. This protects the formal workflow against
  stale JSON, wrong endpoints, partial writes and accidental one-sided edits;
  unsigned local artifacts do not claim authenticity against an attacker able
  to rewrite every file and recompute every digest. Finalization also re-reads
  Git and requires the checkout to remain clean at the same commit recorded at
  training start, so source edits or commit changes during the run fail closed.
- Final validation-only interventions are computed before the fixed checkpoint
  is written and are stored in that checkpoint. They must declare
  `validation_only=true`, `test_accessed=false`, match the validation audit and
  provide exactly overall, rotation, combined and planar-speed>1.7-m/s combined
  groups. Every required distribution must be non-empty, finite and
  nonnegative; accuracy must be finite and within `[0,1]`; PnP distributions
  must exactly reproduce the checkpoint-bound final validation baseline.
- The implementation has 482 passing Stage3 tests and both consumer boundary
  checks pass. A channels=96/batch=64 CUDA capacity smoke completed 150 updates
  without OOM. The recovery harness demonstrated exact model, optimizer,
  scaler, RNG, validation, substage and branch-hash recovery across both
  internal boundaries; the current committed verifier is rerun once before the
  formal 800-update gate. All smoke and recovery outputs remain protected.

## 2026-07-30 decision 177: intervention and gate baseline share one precision contract

- V7 r1 completed its fixed 800 updates, but the checkpoint-bound finalizer
  rejected it before producing a state gate. Main validation executed frozen
  upstream plus the state model under the trainer's CUDA autocast context;
  validation-only interventions executed the same operations in FP32. This
  created small but real PnP distribution differences and demonstrated that
  exact baseline equality was correctly fail-closed.
- The intervention now wraps `_prepare_batch`, PnP state, clean state and the
  truth-yaw-conditioned diagnostic in the exact `_cuda_amp_dtype()` context
  used by main validation. A read-only recomputation from the protected r1
  update-800 checkpoint makes all overall/rotation/combined/high-speed PnP
  velocity and yaw distributions exactly equal to final validation. R1 remains
  `state_gate_invalid` and is never overwritten or promoted; the repaired
  clean commit runs from scratch at a new protected r2 root.

## 2026-07-30 decision 178: reject long-axis yaw and screen local rigid flow before another full run

- The qualified V7 r2 fixed endpoint fails all eight predeclared gates. Its
  overall velocity/yaw errors are 0.458 m/s and 4.127 rad/s, combined errors
  are 0.649 m/s and 3.592 rad/s, high-speed combined velocity error is
  1.098 m/s, combined-11 normalized MAE is 0.2648 and yaw-sign accuracy is
  0.8187. PnP-to-clean and truth-yaw-conditioned interventions do not explain
  the residual. Update history shows translation learned substantially while
  yaw stayed near 4.1 rad/s, so more epochs on the same V7 structure are not an
  authorized remedy.
- Validation evidence localizes the yaw failure to the projective-axis path:
  exact pair evidence is much better than curvature fallback; 30-ms pairs are
  best, while 150/280-ms axes degrade sharply; histories with the full latest
  32 observations are far better than short histories. V8 therefore deletes
  the long-lag `rrT` yaw route and curvature fallback. It uses only anonymous
  exact-same-visible-set 10/30/70-ms pair vectors, orienting a vector reversal
  only when the primary handle swaps. No physical ID, class, session, truth or
  future field enters the unchanged six-field state API.
- Before another 800-update run, V8 performs a fixed three-arm, two-seed,
  200-update state-only screen: a newly trained V7 control, a separated
  handle-common-translation/pair-yaw model, and a joint geometry-conditioned
  4D twist model. The latter two have total state parameter counts within 5%;
  comparison to the smaller V7 control selects a practical candidate but does
  not causally isolate capacity. The report therefore labels total parameters
  explicitly and does not call unreachable legacy context modules active.
- The same V6 state loss and common velocity-ramp constraint apply to every
  arm. Ramp samples are generated by an independent CPU generator keyed only
  by the run seed and update number, so variant-specific dropout consumption
  cannot change the augmentation. Runner, probe model/step, ramp dependency,
  sampler and frozen artifacts are source/hash bound. Aggregation requires the
  exact 3x2 matrix at one clean commit and reconstructs each completed fixed
  checkpoint rather than trusting summary JSON.
- A separated or joint candidate advances only if, under both seeds, overall
  and combined yaw improve by at least 30% versus the same-seed trained V7
  control, high-speed combined yaw also improves by at least 30%, yaw-sign
  accuracy regresses by no more than 0.01, and overall, combined plus
  high-speed-combined velocity regress by no more than 0.03 m/s. Failure ends
  the screen; success authorizes one full 800-update V8 state run.
  Future-position fine tuning and the user-requested simple distance/error
  scatter plots remain frozen until a state candidate passes. Compute remains
  local in Windows `yolov8`; the rented RTX 3090 is powered off, retained and
  not released.

## 2026-07-30 decision 179: V9 preserves geometry--velocity pairing before learned pooling

- The protected V8 six-arm screen is `failed`, so no 800-update V8 run is
  authorized. Joint nevertheless identifies the correct direction: across two
  seeds it reaches overall/combined/high-speed-combined yaw means of
  1.861/2.425/3.945 rad/s, combined velocity 0.772 m/s and yaw-sign accuracy
  0.9667. Seed 20260731 passes every V8 gate; seed 20260730 misses only combined
  yaw, improving 29.42% rather than the required 30%. The aggregate is retained
  at `20260730-v80-v8-probe-aggregate-r1`; all six checkpoints remain protected
  and test is unopened.
- Exact checkpoint forward stratification rejects the hypothesis that the miss
  is caused by the difficult tail. Seed-20260730 high-speed combined yaw
  improves 40.52%. Combined-11 improves 44.76%, while the 231-sample
  combined-02 session improves only 6.60%. Combined history-32 regresses 3.03%
  while histories 8--31 improve 35--70%. The concentrated core is combined,
  speed<=1.2 m/s, all three local pair scales available and history=32
  (149 samples): V7/joint yaw is 1.584/1.565 rad/s. This is a body-calibration
  and evidence-fusion failure, not a few bad tail samples.
- V8 joins evidence only after handle flow, pair differential and current
  geometry have already been pooled separately. It therefore discards which
  observed velocity occurred at which anonymous geometry and lets different
  state coordinates select different scale hypotheses. V9 moves the join to
  the causal edge: each handle token carries matched endpoint geometry,
  displacement/velocity, elapsed time, primary/switch/support and local scale;
  exact-same-visible-set pair tokens carry their matched geometry change.
- V9 uses permutation-invariant latent queries over all local tokens. Separate
  learned local, steady-full-history and handle-only fallback queries provide
  proposals, but an observation-only router applies one common expert weight
  to the complete `[vx,vy,vz,yaw_rate]` twist. Thus the model can average many
  phase-safe local increments in the low-speed full-history body without
  restoring V7's aliased 150/280-ms projective axes, while retaining a local
  expert for short/high-speed histories and a pair-missing fallback. Temporary
  handles establish within-window causal edges only; there is no handle
  embedding, physical ID, class, session, truth/future input, q0 quality or
  hand-written future decoder.
- The V9 state loss supervises only the final unified twist. Planar velocity is
  a two-dimensional robust term, vertical velocity and yaw are separate, and
  no local scale is forced to predict a complete state independently. The same
  seed/update-determined common-ramp augmentation still enforces yaw invariance
  and planar translation equivariance. Gradient-reachable state capacity must
  be within 5% of V8 joint; V8's permanently unreachable legacy pair/vehicle
  modules cannot be used to inflate the control capacity denominator.
- The fixed screen is two 200-update V9 candidates compared to the existing
  same-seed V8 joint update-200 checkpoints. Both seeds must improve overall
  velocity by >=10%, combined and high-speed-combined velocity by >=15%, keep
  overall/combined/high-speed yaw within +5%, keep yaw-sign within -0.01, and
  improve yaw in the speed<=1.2/history32/pair3 combined core by >=10%. A
  validation-only within-sample intervention rolls geometry independently
  within each anonymous stream and local time scale while preserving token
  support/type/scale and each group's geometry marginal. It must worsen
  high-speed combined velocity by >=10% or >=0.15 m/s; otherwise the added
  structure was not used. Failure stops without more updates. Future-position
  training and simple distance/error scatter plots remain frozen until the
  state screen passes. Compute remains local; the retained RTX 3090 server
  stays powered off.

## 2026-07-30 decision 180: callable provenance uses the canonical `python -m` module

- Both first V9 r1 runs completed their fixed 200 updates and individually
  validated in-process, but the independent aggregate refused them before
  evaluating gates. A hook defined in the executable runner was recorded as
  module `__main__` during `python -m`; importing that same runner from the
  aggregate exposed its canonical package name. The qualname and semantic
  source SHA-256 were identical, so this is a runtime alias bug in provenance,
  not changed training code or corrupted checkpoints.
- Callable contracts now replace `__main__` with
  `sys.modules["__main__"].__spec__.name` when that canonical name is available.
  Direct script execution without a module spec remains `__main__`; ordinary
  imported functions are unchanged. A dedicated regression simulates the
  `python -m` entrypoint. The fix is shared by Stage3 runners because the base
  contract helper owns this serialization rule.
- The two r1 checkpoints remain protected diagnostic evidence and are never
  rewritten or manually promoted. Formal V9 evidence must be regenerated from
  a clean commit into new r2 roots so checkpoint, manifest, callable contract
  and aggregate all bind the same canonical source identity. Test remains
  unopened; no gate is weakened.

## 2026-07-30 decision 181: V9 fails because evidence availability and 4D calibration are coupled

- The clean V9 r2 aggregate is a valid `failed` result at
  `20260730-v81-v9-paired-twist-aggregate-r2`, bound to source commit
  `640cb0d`; test remains unopened. Seed 20260730 improves overall, combined
  and high-speed-combined velocity by 1.10%, 10.48% and 9.65%, while seed
  20260731 regresses overall velocity by 6.74%. Core yaw improves 10.22% in the
  first seed but regresses 36.57% in the second. Both seeds fail the pairing
  intervention because broken pairing slightly improves high-speed velocity.
- This does not mean every pair token is unused. On full-history pair3 samples,
  breaking pairing worsens velocity by 0.138/0.086 m/s and yaw by
  0.357/0.587 rad/s. The structural hole is partial support: 143 pair1/pair2
  samples cannot enter the steady expert, and 24.8% of all validation samples
  are forced to local. Combined pair1/pair2 error is about 1.15 m/s and
  3.84 rad/s. A per-sample available-expert oracle recovers only 7% velocity
  and 9--12% yaw overall and almost nothing in high-speed combined motion, so
  router tuning alone is insufficient.
- The complete observation latent remains identifiable: nearest nonlocal
  combined neighbours do not show strong same-input/different-truth conflicts.
  Instead V9 contracts planar magnitude and couples its calibration to yaw.
  The two seed yaw predictions correlate at 0.997 but differ by 0.739 rad/s in
  mean. Combined signed y-velocity error and yaw error correlate at 0.87 even
  though the corresponding truth correlation is only -0.17. One expert scalar
  applied to the complete 4D twist is therefore rejected as the next state
  definition.
- The frozen learned future decoder is also nearly insensitive to the supplied
  4D state: on combined validation, replacing candidate yaw, planar velocity
  or the complete state by truth changes conditional future error by less than
  0.02 mm around 409.7 mm. State learning remains independently measurable,
  but a later accepted state must be followed by a learned decoder redesign
  with counterfactual state-sensitivity tests. The user-rejected analytic
  future decoder is not reintroduced.

## 2026-07-30 decision 182: V10 uses paired residual subspaces without a 4D router

- V10 retains the anonymous local-edge encoders but deletes the complete-4D
  local/steady/fallback proposals and their scalar router. A single handle-set
  latent emits one coherent 3D velocity baseline. Every available 10/30/70-ms
  pair scale, including pair1 and pair2 support, forms an event-scale bundle
  with the matching handle summary. A multiplicative learned interaction
  prevents geometry and kinematics from bypassing their pairing on the yaw and
  rotation-compensation paths.
- Pair bundles emit local angular votes with direct yaw-only supervision and
  learned reliability. Their aggregate emits yaw plus a planar correction to
  the handle velocity baseline; z velocity is never corrected by planar
  rotation. With no pair bundle, the correction is structurally zero and yaw
  uses a handle-only fallback. This is an observation-support rule, not a
  physical ID, class, session, truth, future or quality input. The two
  subspaces share the causal token trunk and train jointly; there is no V7-style
  detached one-way boundary and no per-coordinate expert selection.
- The paired path is structurally zero-preserving, not merely encouraged by a
  loss. Geometry and kinematics are projected separately by bias-free layers;
  their products form the handle and pair interactions, and the resulting
  event-scale bundle is the only source of paired yaw and planar correction.
  The bundle, yaw-vote and correction normalizations have no affine terms, the
  downstream paired linears have no bias, and no learned query reads pair
  evidence. Thus zeroing any one paired marginal makes both paired outputs
  exactly zero, and the common handle latent cannot synthesize a residual by
  itself. The handle-only fallback is used only when pair support is absent.
- The first screen is two fixed 200-update seeds against the existing V8-joint
  controls, with gradient-reachable capacity within 5%. All V9 gates remain.
  In addition, pair1/pair2 combined velocity and yaw must each improve at least
  10%; broken pairing must worsen high-speed-combined velocity by at least 10%
  or 0.15 m/s; and zeroing the planar paired residual must worsen combined or
  high-speed-combined velocity by at least 8% or 0.10 m/s. Broken pairing in
  the pair1/pair2 group must also worsen velocity by at least 5% or 0.05 m/s
  and yaw by at least 10% or 0.30 rad/s. Both seeds pass separately or the
  structure ends without extra epochs. Future fine tuning, scatter plots,
  export and online integration remain frozen.

## 2026-07-30 decision 183: local yaw votes are rejected in favour of global history closure

- The clean fixed V10 aggregate at
  `20260730-v82-v10-paired-residual-aggregate-r1` is a valid `failed` result
  from commit `d58f9e6`; both seeds reached update 200 and test remains
  unopened. Each seed passes exactly 4/14 gates. The paired planar branch is
  causally useful: pair1/2 velocity improves 11.14/13.13% over V8, and removing
  the planar residual worsens combined velocity by 20.74/30.91%. This rules out
  the claim that relative evidence is absent or wholly ignored.
- The angular definition fails in the body of the distribution, not only the
  tail. Pair1/2 yaw regresses 64.78/70.44%, combined yaw regresses
  37.15/49.24% and high-speed-combined yaw regresses 43.50/46.17%. The yaw
  median also worsens. Pair1 and pair2 cannot remain one diagnostic group:
  broken pairing improves pair1 yaw by 20.22/10.14%, while it worsens pair2 yaw
  by about 47% in both seeds. The merged pair1/2 intervention therefore hides
  an invalid pair1 shortcut. The base local pair-vote yaw is worse than final
  yaw by roughly 2 rad/s; the common-conditioned correction rescues part of the
  error but cannot repair the local-vote definition.
- The next estimator treats each exact event/scale/handle edge as an evidence
  factor for one global twist rather than asking each local bundle to predict
  global yaw. A learned causal history decoder must reconstruct already
  observed displacements from that twist and prior geometry, and aggregated
  closure residuals refine the same state for a fixed small number of shared
  iterations. This is learned analysis-by-synthesis over causal history, not a
  hand-written physical future decoder. It still consumes exactly the six
  observation fields and no ID, class, session, truth, future or q0 quality.
- The next screen reports pair1, pair2 and pair3 separately. Its decisive
  counterfactual crosses translation evidence from one validation sample with
  intact relative-rotation factors from another compatible sample: planar
  velocity must follow the former and yaw/sign the latter. Breaking the donor
  factor's internal event/scale geometry--differential correspondence must
  worsen state and history-closure error. Separate handle-geometry and
  pair-geometry interventions are also retained. Failure ends the structure
  without extra updates; future-position training stays frozen.
- The implemented closure decoder is typed and block-sparse. Handle translation
  reads only candidate velocity and time. Handle rotation reads yaw, centered
  prior geometry and time. Pair rotation reads yaw, prior pair geometry and
  time and cannot read velocity. The initial pair and pair-residual messages
  are bias-free multiplicative geometry--motion--time gates, so zeroing any
  factor makes that branch exactly zero. Closure loss is normalized per sample
  and per factor type, preventing full pair3 histories from dominating pair1
  and pair2 solely by token count. The formal capacity is 1,467,004 reachable
  state parameters, within 5% of the protected V8 control.
- The crossed counterfactual is a donor-timeline resynthesis rather than tensor
  splicing. The translation source contributes the arithmetic masked mean of
  all valid handle rates and a q0 center estimated from every valid source
  edge. The rotation donor contributes centered handle endpoints, pair
  geometry, times and support. Absolute endpoints are reconstructed on the
  donor timeline so both `delta = current - prior` and
  `delta = rate * elapsed / history_scale` remain true. The broken version
  independently rolls the donor's centered handle and pair geometry while
  retaining common translation, kinematics and support. Formal acceptance
  requires that this break worsen both hybrid-state error and observed-history
  closure error; it is not enough for one tail metric to move.
- The independent aggregate reconstructs each model from its immutable
  manifest, restores only the runtime `stop_after_update=0` default omitted
  from the contract hash, reruns the complete locked argument validator,
  recomputes reachable capacity, replays the final diagnostics from the fixed
  checkpoint, and binds the protected V8 path/SHA and recorded control metrics.
  Candidate self-reported capacity or controls cannot satisfy the gate.

## 2026-07-31 decision 184: V11 closes history but couples the wrong state updates

- The clean fixed V11 aggregate at
  `20260730-v83-v11-global-flow-closure-aggregate-r1` is a valid `failed`
  result from commit `a11bdda`; both seeds reached update 200 and test remains
  unopened. Seed checkpoints are retained with SHA-256
  `0a8b1555be9e092c5791ddc4e27ce9a7995904f313170e0a970d9372e5569e37`
  and `33b2c2eb13c04cdd18963db88adab6d636b97c5398f7450b587860e7e943e1e8`.
  The aggregate SHA-256 is
  `0768ee1034b7a14215781c0691fc0d5d6f572ff3364a948ca6bb682ebf3e5964`.
- V11 is not a uniform failure. Relative to the protected V8 controls,
  high-speed combined velocity improves 14.65/23.21% and yaw improves
  22.54/24.59%; pair1 velocity/yaw improve about 20--24%, and pair2 yaw
  improves about 20--22%. In contrast, overall velocity regresses 7.42/9.78%,
  pair3 yaw regresses 10.04/20.72%, and low-speed full-history core yaw
  regresses 39.83/53.70%. Overall velocity P50 becomes 0.452/0.441 m/s versus
  V8's 0.373/0.353, and yaw P50 becomes 1.504/1.493 rad/s versus 1.275/1.100,
  even though several P95 tails improve. The model therefore trims difficult
  tails while damaging the dense long-history body; extra epochs are not the
  justified action.
- Closure refinement is active but its state incidence is wrong. Removing both
  refinements makes combined yaw 56.5/80.8% worse relative to the final model,
  while combined velocity changes only 4.9/4.3%. The shared handle update can
  write all four state coordinates, so an angular correction can perturb
  velocity; on high-speed and pair1/2 strata this refinement in fact worsens
  velocity. Breaking handle geometry changes overall velocity by only
  0.50/0.60%, showing that the handle path can bypass geometry through raw
  kinematics. Pair correspondence has measurable effect mainly under pair3
  support; a within-group roll cannot touch a single pair1 factor and must not
  be treated as evidence that pair1 used correspondence.
- The crossed diagnostic shows a coarse but uncalibrated decomposition:
  velocity is much closer to its common-flow source than to the rotation donor,
  and yaw is closer to the rotation donor, but donor yaw-sign accuracy is only
  0.805/0.835. Re-estimating state after a broken cross lets the network absorb
  the break: yaw worsens while history closure changes only 2.81/3.58% and
  velocity can improve. Future pairing diagnostics must hold the intact state
  fixed and recompute only decoder residuals; a small re-optimized closure is
  not physical-consistency evidence.
- The q0 constant-twist target remains valid because every joined window is
  bounded inside one constant-motion segment. The replacement is therefore a
  strict typed estimator, not a new target: event-ordered relative factors
  alone estimate signed omega; omega and prior geometry predict rotation;
  de-rotated common handle residuals alone estimate velocity. Relative closure
  may update only omega, common closure only velocity, and omega-to-velocity is
  the only cross-state direction. Acceptance requires fixed-state pairing
  closure, isolated omega/v refinement effects, a 2x2 common/relative source
  cross, common-ramp and relative-reversal equivariance, and explicit touched
  counts for pair0/1/2/3. Future-position decoding remains frozen.

## 2026-07-31 decision 185: V12 makes omega-to-velocity the only state dependency

- Decision 184's statement that the original V11 crossed diagnostic showed a
  coarse translation/rotation decomposition is superseded. Its translation
  source was the masked arithmetic mean of visible-plate apparent rates, which
  includes rotational motion and is not vehicle-center velocity. A dedicated
  mechanism audit still confirms the useful parts of the conclusion: V11's
  state-conditioned behaviour is real, but exact factor correspondence is weak,
  yaw magnitude is contracted, and the shared four-dimensional refinement lets
  angular evidence perturb velocity. The q0 constant-twist target remains valid.
- V12 implements a typed, ordered estimator. A bias-free causal GRU aggregates
  anonymous relative event factors and completes all angular refinements first.
  A learned history decoder combines the resulting omega with centered prior
  geometry and elapsed time to predict already-observed rotational displacement.
  Only the de-rotated common handle residual enters the separate velocity branch.
  Pair evidence cannot directly update velocity and velocity never updates
  omega; `omega -> velocity` is the sole cross-state edge. Every state path is
  bias-free and a zero-motion/support input produces an exactly zero state.
- The deployment contract remains exactly six raw causal fields and excludes
  physical armor IDs, slots as identity, session, motion class, truth, future,
  absolute range and PnP-only quality. The learned future-position modules stay
  frozen and hash-checked. Reachable state capacity is 1,547,652 parameters,
  4.03% above the protected V8-joint control and within the fixed 5% envelope.
- The structural screen is two independent fixed-200-update seeds against their
  same-seed V8 controls. In addition to body, combined, high-speed and
  pair0/1/2/3 state metrics, acceptance requires fixed-state handle/pair closure
  degradation under equal-support geometry derangement and changed-only
  re-estimated yaw degradation. Pair1 and pair2 are never merged for this gate.
  A full-validation AA/AB/BA/BB resynthesis keeps target support, masks and time,
  switches truth velocity and omega independently, and uses a fixed prior gauge
  satisfying zero weighted relative displacement and rate. Translation transfer
  must pass separately on x and y with all four truth-delta quadrants present;
  relative yaw transfer, common-ramp invariance and relative-reflection
  antisymmetry must also pass.
- Diagnostics and aggregation are fail-closed: exact container schemas, native
  numeric types, finite values, fixed sample counts, ordered percentiles and
  bounded probabilities/correlations are independently recomputed from each
  fixed checkpoint. A boolean cannot masquerade as exact-zero write isolation.
  Both seeds must pass every gate; otherwise V12 ends without extra updates.
  Only a pass can start learned future-position redesign/fine-tuning and the
  requested per-motion distance/flight-time versus error scatter plots.

## 2026-07-31 decision 186: V12 is rejected; V13 alternates signed equivariant state updates

- V12 completed two clean fixed-200-update seeds and is validly rejected. The
  runs pass 44/94 and 45/94 gates. Across seeds, overall velocity changes from
  initial mean/P50/P95 0.543/0.116/2.086 m/s to
  0.616/0.455/1.853 m/s, versus V8 0.511/0.363/1.453 m/s. Overall yaw changes
  from 8.578/9.005/14.192 rad/s to 3.958/2.185/13.173 rad/s, versus V8
  1.861/1.188/5.574 rad/s. Combined/high-speed body and tail metrics likewise
  remain worse than V8. Breaking pair correspondence improves pair0/1/2 yaw in
  the dominant evidence, while pair3 is only weakly causal. Pure-rotation
  velocity grows from about 0.082 to 0.375 m/s. These are structural failures,
  not a request for more epochs.
- The mechanism diagnosis supersedes V12's strict ordering. Its masked mean of
  visible-factor apparent rates is a gauge, not vehicle-center velocity, and
  absorbs rotation before omega. The velocity objective can also backpropagate
  through omega-conditioned de-rotation unless the closure is typed at the loss
  boundary. Pooling handle/scale evidence before recurrence and hard reliance
  on sparse pair support further weaken pair0/1/2. The replacement therefore
  alternates estimates and detaches every cross-stage state rather than making
  omega-to-velocity the only one-way dependency.
- V13 fixes the computation as
  `omega0 -> velocity0 -> omega1 -> velocity1`, with fixed screen lengths
  35/20/25/20 and an independently reset LR phase for each. Omega stages use
  reflection-even ordered handle/pair encoders plus explicit signed angle,
  orbital and curvature pseudoscalars. Learned coefficients are unrestricted,
  so the estimator can reverse or replace a wrong carrier and still receives a
  real-loss gradient when the analytic angle carrier is zero. Handle evidence
  requires two midpoint-ordered consecutive chords; pair and handle estimates
  use soft support/dispersion precision weights. Omega1 is exactly
  `detach(omega0) + delta_omega` after recomputing the gauge from detached
  velocity0.
- V13 velocity analytically de-rotates observed handle displacement, solves a
  supported-row WLS carrier and learns only scalar combinations of
  carrier-centred residual vectors. The correction is common-ramp invariant and
  O(2)-equivariant by construction; it cannot emit a free XYZ bias. WLS rows
  below the denominator threshold are explicitly unsupported, return high
  uncertainty and do not enter state/equivariance losses. Irregular acceleration
  divides chord-rate differences by chord-midpoint time. A dataset preflight
  bounds physical train/validation yaw at 15 rad/s; over the maximum 0.105-s lag
  its 1.575-rad phase remains below pi and prevents atan2 alias.
- The typed loss macro-balances motion class x exact pair-scale support x
  history length; the hierarchical sampler additionally balances motion,
  session, history, stationary/active state and pair0/1/2/3 support without
  exposing any of those labels to forward. Each substage trains its state
  coordinate, the matching observed-history closure, common-ramp equivariance
  and physical reflection. Real training-loss autograd tests require finite
  nonzero gradients only under the active module prefix at updates
  1/36/56/81. Future labels are unread and future modules remain frozen.
- Formal V13 acceptance is a two-seed, fixed-100-update local RTX 4060 screen,
  not full continuation. Each result is reconstructed from its checkpoint and
  bound to a clean non-unknown commit, semantic callable contracts, current
  source hashes, exact update counts/transitions, branch-hash changes,
  validation history, diagnostics, source dataset and unchanged future tensor
  hashes. Protected same-seed V12 result paths, result SHA-256 and checkpoint
  SHA-256 are constants. Numeric domains and sample counts are validated before
  evaluating mean/P50 improvements, P95 catastrophe guards, pair0 handle and
  pair1/2/3 pair causality, common-ramp/reflection and exact write isolation.
  The finalizer reopens only train/validation, verifies truth-shard hashes,
  repeats the exact truth joins and recomputes the complete yaw-alias summary;
  phase, sample-count or maximum-yaw report forgery is rejected even if contract,
  provenance and checkpoint payload are rewritten together. The reviewed source
  passes 35/35 focused and 596/596 complete Stage3 tests plus both repository
  boundary checks.
  The aggregate independently reruns this validator for both seeds. A failure
  ends V13 without update piling; future-position fine-tuning and user-facing
  distance/flight-time scatter plots remain frozen until a state gate passes.

## 2026-07-31 decision 187: reject V13 and move the nuisance center into the state definition

- Both V13 screens completed normally from clean commit
  `e3dba30503f431290a1f0f14c487e5d9003be0f9`. Seed 20260730 passes 9/38
  checks and seed 20260731 passes 8/38. Their checkpoints have SHA-256
  `f5ffe725b0c9b2daa9c2cbb8e52b4870116095a3ece3183d091f8cf881859224`
  and `67ea204f744c0ac3a0754300186a00d6339cab3541da31a690e5e08907437933`;
  the failed aggregate SHA-256 is
  `75c1e492dbc66a16489382518d5e2e158ab0be3c4d80e69ac6c8665b589f6e0c`.
  The complete validator reconstructs both results, future hashes are unchanged
  and test is sealed. These assets are retained as protected negative evidence.
- Rejection is based on the body distribution. Averaged across seeds, V13
  overall velocity Mean/P50/P95 is 1.773/1.759/3.618 m/s and yaw is
  7.053/6.114/17.553 rad/s. V12 is 0.616/0.455/1.853 and
  3.958/2.185/13.173; V8 is 0.511/0.363/1.453 and
  1.861/1.188/5.574. V13's own update-0 to update-100 Mean changes are only
  -0.9% velocity and -2.5% yaw. Two seeds agree on 37/38 gates and final body
  metrics differ by about 0.1%, so neither random variance nor more updates is
  a credible remedy.
- V13 double-canonicalizes pair direction. The paired context already negates
  the prior vector when primary changes, placing prior and current in the same
  current-primary frame. V13 then negates current again and repeats that sign in
  analytic closure, introducing an artificial pi rotation. On supported
  validation rows, removing only the second flip improves the raw yaw carrier
  Mean/P50/P95 from 8.725/7.361/20.740 to 4.863/2.472/17.261 rad/s and raises
  truth correlation from 0.165 to 0.725. On combined pair3 it improves from
  6.742/4.693/17.031 to 2.004/1.325/5.919 and correlation from 0.290 to 0.943.
  Therefore the V13 pair intervention cannot be used to reject pair evidence;
  all future pair geometry has exactly one canonicalization owner.
- Velocity remains independently misdefined. With truth omega, the old WLS is
  still 1.759/1.754/3.533 m/s overall; using a truth translation gauge only
  reaches 1.644/1.671/3.349. The visible-factor mean and a window-wide mean
  center are not the chassis center, so their rotation error is absorbed as
  translation. A validation-only profiled rigid equation jointly treating
  center and local tracklet positions as nuisance obtains useful medians but is
  ill-conditioned on short arcs. Giving the same equation truth center and
  truth omega only as an oracle mechanism bound reaches overall, rotation and
  combined Mean/P50 of 0.708/0.123, 0.894/0.229 and 0.428/0.086 m/s. Thus a
  learned anonymous center prior has measurable headroom, while the remaining
  long tail requires robust likelihood and explicit uncertainty rather than
  sample deletion.
- The next state definition learns a distribution for the current-primary to
  chassis-center offset from causal anonymous relative geometry. Center truth
  is a loss-only label derived from `anchor_center_position_m` and the frozen H
  current-primary estimate in the same forward pass; it is forbidden from
  forward, export and inference.
  The offset is a per-window vector, not a physical armor identity. For fixed
  omega, a profiled solver jointly eliminates center and per-tracklet geometry
  while estimating unregularized common velocity. Learned precision/bias is
  restricted to ramp-invariant O(2)-legal features so PnP ellipse/noise is an
  observation likelihood, not a hard-coded future decoder. The learned future
  and selector stay frozen until zero-update mechanics, center-prior recovery
  and a fixed double-seed state screen all pass.
- The forward-interface decision is to expose only frozen S/H
  `q0_relation_m [B,4,3]` and `q0_supported [B,4]` to this state estimator.
  There is no physical ID or fixed-slot embedding. Validation-only zero-training
  diagnostics show the simple anonymous q0 mean reaches overall, rotation and
  combined velocity Mean/P50 of 0.902/0.332, 1.082/0.484 and 0.633/0.239 m/s
  under truth omega. This recovers most of the body headroom between the zero
  center prior and the correctly framed truth-center oracle. Confidence, sigma,
  age and support-class stay outside the first learned center-prior input so an
  uncalibrated S/H metadata fingerprint cannot replace relative geometry.

## 2026-07-31 decision 188: use a Schur-profiled two-component center mechanism

- The initial V14 implementation exposed a structural numerical bug before any
  formal training. A direct 12x12 normal solve treated any invertible short arc
  as valid; spans from 1e-4 to 1e-6 seconds produced velocity norms from 2.6 to
  194 m/s with near-zero residual energy. V14 therefore profiles center and
  tracklet nuisance variables through a velocity Schur complement and gates the
  minimum eigenvalue, condition and time span. Velocity remains completely
  unregularized. Failed profile and failed translation fallback are separate,
  observable states and may not be silently scored as a q0 profile. The
  fallback additionally requires at least 1 ms between its earliest and latest
  visible event, even when many handles make its scalar information appear
  numerically large.
- The anonymous center carrier is the mean of all four finite H hypotheses, not
  a supported-only mean. On 750 validation rows, all-four center Mean/P50/P95 is
  0.134/0.104/0.311 m relative to the exact frozen-H current origin. Independent
  diagnostics found supported-only averaging sacrifices the body median to
  improve part of P95, especially at support two and three. Support is retained
  as reliability metadata; it no longer deletes a finite inferred role.
- For supplied truth omega as a mechanism bound, the Schur implementation gives
  q0 soft-center velocity Mean/P50/P95 0.894/0.338/3.561 m/s, history-wide
  1.498/0.299/6.499 and truth-center oracle 0.707/0.123/3.232. Q0 and oracle
  profiles cover 100% of validation; history-wide profiles 98.53% and explicitly
  falls back on 1.47%, with final state support 100%. The validation-only audit
  `20260731-v87-v14-profiled-center-zero-update-r4.json` has SHA-256
  `3aadb3ddf624b4ee57fb7d127a10a19ecdcfecb08b64aca02fbfb59ce974b6db`;
  test and future modules were not accessed.
- A center prior alone cannot detect a cross-sample but geometrically plausible
  H center because the short history can trade center error against velocity.
  The informed component therefore also uses q0 as a weak current-endpoint
  prior. Because MAP energies under a strong informed prior and a nearly free
  history prior are not comparable evidence, they do not directly set the
  mixture weight. A learned invariant gate instead consumes support, predicted
  center uncertainty, separate XY/Z profile energies and Schur information.
  It is supervised by a loss-only soft responsibility derived from which
  component is closer to truth velocity, on intact and corrupted train batches.
  This is not a physical-ID lookup: all endpoint parameters are anonymous and
  permutation shared, and truth is never a gate input. Formal
  B0 validation must report intact, blind and shuffled-H body distributions;
  shuffled H may not exceed blind Mean/P50 by more than 2%.
- The B0 runner is fixed to 100 updates, checkpoints every 25 updates, supports
  exact latest-checkpoint resume, binds dataset/truth/source/frozen checkpoint
  and state hashes, and refuses dirty or unknown git. It learns the center
  mean/variance and invariant component gate from center loss, truth-omega
  profiled velocity loss and gate responsibility loss. Truth
  yaw is explicitly declared as a diagnostic forward input; truth velocity and
  center remain loss-only. No future module is loaded.

## 2026-07-31 decision 189: reject joint gating and isolate reliability fusion

- V14-B0 seed 20260730 is a deterministic structural rejection. Runs resumed at
  update 25 and 75 and an uninterrupted control produce the exact same final
  model SHA-256
  `ba9c489c73615a7e1a2a68615e04488b65eec656f523b21c28377d07189cfaf9`
  and identical metrics. Center Mean/P50 improve, velocity P50 improves 5.66%,
  but velocity Mean improves only 1.22%. Intact/shuffled q0 weights differ by
  only 0.0047 although their oracle responsibilities differ by 0.312. The
  shuffled P50 regression from 0.299 to 0.819 m/s affects the distribution body,
  so it cannot be excused as a small tail.
- The two frozen experts are not the blocker. Their loss-only continuous convex
  oracle reaches intact Mean/P50 0.641/0.152 m/s and shuffled Mean/P50
  0.808/0.176. Existing forward-visible compatibility features have grouped
  probe AUC 0.813 for component preference and 0.958 for intact versus shuffled
  H. V14 failed because a biased gate, moving experts, minibatch-roll negatives,
  unequal intact/corrupt velocity weights and weak BCE supervision were trained
  jointly; extra epochs do not repair that definition.
- V15 makes reliability a separate frozen-expert stage. Its target is the
  actual loss-only optimal projection coefficient
  `w*=clip(((truth-vh) dot (vq-vh))/||vq-vh||^2,0,1)` rather than a probability
  that one endpoint is better. Counterfactual donors are generated over the
  whole split, have no fixed points, are hash-bound and include both global and
  motion/support-matched hard corruption. The forward head is anonymous and
  invariant, consumes explicit compatibility evidence, and structurally makes
  increased incompatibility reduce q0 reliance. A diagnostic frozen-parent
  screen precedes any clean formal two-stage rerun; test, future and free omega
  remain unopened until it passes.

## 2026-07-31 decision 190: reject window-pooled reliability and test paired endpoint evidence

- V15-A0 is a valid train-only structural rejection. It ran from clean commit
  `d2992d7` with fixed V14/Mapper/S/H hashes and stopped after both grouped
  train-CV folds failed. It did not claim or load validation, did not access
  test and did not load future modules. The immutable result is
  `20260731-v89-v15-a0-frozen-reliability-oracle-omega-r1/screen_result.json`,
  SHA-256
  `0e161fee2fde0b4a4c4368f13871a46b120484336935ab6e4777b8ad68968e5c`.
- This is not an optimization-length failure. The two folds' intact overall
  component-preference AUC is `0.739/0.552`, while oracle-gap Mean recovery is
  `-0.079/-0.151`. The learned intact mean q0 weights are `0.523/0.548` even
  though the loss-only oracle means move from `0.522` to `0.636`. Corrupted
  arms can often be ranked, especially in fold 0, but paired q0-weight
  separation never exceeds about `0.06`. The scalar head collapses toward a
  conservative half mixture and does not generalize which expert is right.
- The rejected 13D representation pools the four anonymous endpoints before
  reliability inference. It retains support count, center uncertainty, total
  XY/Z profile energies and global Schur information, but discards which
  endpoint conflicts with which observed history, whether that endpoint is
  current or recently switched, and whether one severe local mismatch is
  diluted by three compatible roles. Increasing MLP width, adding updates or
  changing thresholds does not restore discarded information.
- A train-fold-only non-parametric audit rules out a simple capacity or
  normalization explanation. kNN-20 oracle-weight correlation is only
  `0.426/0.220`, although fold 1 is closer to the fit-fold feature manifold
  (`0.687` normalized nearest-neighbour distance versus `0.858`). Similar 13D
  features map to oracle-weight neighbourhoods with standard deviation about
  `0.35`. Moreover absolute q0 XY energy correlates positively, not negatively,
  with oracle q0 weight (`+0.277/+0.035`). It mixes motion magnitude,
  information and fit error and is not a cross-session monotone reliability
  measure. The absolute-energy hard slope is deleted rather than retuned;
  only an independently tested, information-normalized local innovation may
  later receive a structural monotonic constraint.
- The authorized A1 mechanism probe therefore moves the architectural boundary
  before pooling. It forms shared event-role tokens, shared anonymous endpoint
  tokens and six shared unordered-pair tokens. Each token compares q0-informed
  and history-only explanations of the same local evidence; joint role
  relabeling only permutes tokens and symmetric aggregation removes all fixed
  slot identity. Only O(2)-invariant scalar contrasts reach the reliability
  head, and common velocity ramps cancel because absolute expert velocity is
  forbidden. The final state remains the continuous convex combination of the
  same two frozen experts with deterministic single-expert/fallback states.
- A1 is allowed exactly one fixed-budget, two-fold train-only endpoint probe.
  It must prove that local paired evidence improves intact AUC and positive
  oracle-gap recovery in each motion family and each fold, not merely detect
  synthetic corruptions. Counterfactual tests must distinguish local
  compatibility from session membership, verify dose response when one, two
  and four endpoints are damaged, and show that breaking only endpoint pairing
  changes the result while synchronous S4/C4 relabeling does not. Any failure
  keeps formal two-seed training, validation, free omega and future position
  unauthorized.

## 2026-07-31 decision 191: invalidate eager truth access and enforce split-scoped joins

- Review found that the legacy motion and center truth indexes iterate every
  manifest shard and load both train and validation truth into memory. A0
  attached and trained only on split-keyed train records, never constructed
  the paired validation dataset, never selected a checkpoint from validation
  and never claimed its ledger. Its negative train-CV metrics are therefore
  not numerically contaminated, but the declared `validation_accessed=false`
  boundary is procedurally false. The original A0 artifact with SHA-256
  `0e161fee2fde0b4a4c4368f13871a46b120484336935ab6e4777b8ad68968e5c`
  is archived negative evidence only and cannot authorize A1.
- New screens use `SplitScopedTruthIndex(root, split=...)`. Reading the shared
  manifest is allowed, but only shards for the declared split are hashed or
  opened. A regression fixture proves train construction succeeds while the
  declared validation shard is deliberately absent. End-of-run integrity
  checks likewise rehash only the split actually authorized. In A0, a
  validation-scoped truth index can be constructed only after both train folds
  pass and the global validation ledger has been atomically claimed.
- The fixed A0 train-CV must be rerun from corrected clean source before A1.
  The original validation scope remains unclaimed, so the correction consumes
  no validation opportunity. A reproduced rejection authorizes A1 only through
  the new result hash; an unexpected pass follows the existing one-shot ledger
  path and blocks A1 until A0 is resolved.

## 2026-07-31 decision 192: corrected A0 reproduces rejection and authorizes A1

- Corrected V15-A0 ran from clean commit `2f68c23` at
  `D:\仿真\models\engines\stage3-training\20260731-v90-v15-a0-split-scoped-reliability-r1`.
  Both held-out-fold intact metric groups are exactly equal to the archived
  run: fold 0 fused Mean/P50/AUC is `0.767297/0.381225/0.739313`, and fold 1
  is `0.949819/0.682133/0.551743`. The screen therefore again ends
  `cv_failed` and preserves the structural conclusion rather than attributing
  it to truth-index behavior.
- The corrected run reports `validation_accessed=false` and its run state
  reports `validation_claimed=false`; no ledger exists, future modules were
  not loaded and test remained unopened. `screen_result.json` SHA-256 is
  `9c86a10041a66baf93588b1e0cb32ce9c99dd36ad0008bc2d6d5a7b7602d7853`
  and `run_state.json` SHA-256 is
  `9d073f523bf0cef2ce56c3d4cedf4cde488f5491006911c861aedc4c87421f8f`.
  This corrected result, and not the archived eager-access artifact, is the
  sole authorization input for the fixed A1 endpoint-information probe.

## 2026-07-31 decision 193: use maximal balanced exact hard maps without relaxing gates

- The original A1 hard map discarded an entire exact stratum when its largest
  session contained more than half the rows. In the fold-0 held-out rotation
  population this discarded all 188 `(rotation, support=3)` rows and all 20
  `(rotation, support=1)` rows, reducing family coverage to `72.34%` before
  head training. Exhaustive session-fold analysis shows no partition can make
  the old all-or-none definition satisfy both family gates.
- For stratum size `n` with maximum session count `m`, the largest subset that
  admits a cross-session bijection has size `n` when `m <= n/2`, otherwise
  `2(n-m)`. A1 deterministically keeps every minority-session row and selects
  `n-m` majority rows by a domain/stratum/sample-key SHA-256 order. The exact
  motion/support stratum, cross-session constraint, recipient/donor set
  equality, no fixed point and no relaxed matching remain unchanged.
- Metadata feasibility is now checked before any V14 expert profiling or head
  training, and the same sealed maps are reused by training. A failed domain is
  persisted with its fold, fit/held-out identity, maps, full-family coverage
  and unavoidable exclusions under `preflight_failed`. The hard artifact
  validator binds this policy and all selected/excluded counts. The full
  denominator and 80% family gates are deliberately not weakened.

## 2026-07-31 decision 194: reject pooled endpoint A1 despite body improvement

- A1-R2 completed train-only at
  `D:\仿真\models\engines\stage3-training\20260731-v92-v15-a1-p0-endpoint-token-r2`
  with `status=failed`, `validation_accessed=false`, `test_accessed=false` and
  `future_modules_loaded=false`. Result SHA-256 is
  `c60beb32377d9cf767207268851dae6a47579a5ec03fd4d308a240bd227670d5`;
  run-state SHA-256 is
  `6990f38a8457e22a2822af1863ccb538441e5fb6071230f9353f542d52f90c1f`.
- The result improves the intact distribution body, not merely the tail:
  fold-0 Mean/P50 changes from A0 `0.767/0.381` to `0.691/0.311`, and fold 1
  from `0.950/0.682` to `0.870/0.601`. This recovers only `25.0%/34.1%` and
  `16.8%/10.2%` of oracle Mean/P50 headroom; fold-1 combined recovers just
  `8.9%/10.4%`. The result is useful evidence but cannot authorize a formal
  model or validation access.
- Synthetic corruption discrimination is adequate: global/hard AUC spans
  approximately `0.75-0.90`, and fused corruption Mean/P50 remain well below
  blind output. The missing mechanism is not coarse bad-input detection.
  Removing the local endpoint tokens changes intact AUC by only `0.005-0.044`
  and coefficient MAE by only `0.8%-2.3%`. Therefore the event/role/pair
  mean/max boundary still erases most useful local temporal information.
  Additional updates, width or threshold tuning are rejected. Any successor
  must keep anonymous endpoint temporal evolution available at the final
  reliability decision and must independently prove that this temporal path,
  rather than a global/session shortcut, supplies the gain.
