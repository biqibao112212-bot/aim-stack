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
