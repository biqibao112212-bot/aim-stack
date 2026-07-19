# Aim Stack 关键决策

上下文版本：`CTX-AIM-STACK-2026.07-v2`

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
