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
