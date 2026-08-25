# Aim Stack 关键决策

本文件只保留仍约束实现和研究的决定。被取代的路线、逐轮指标和制作过程由 Git 历史、实验
manifest、证据 registry 与 `training/*/findings.md` 保存，不再复制到项目上下文。

## D1. 模拟器是独立的上游依赖

消费者只通过版本锁、公开合同和版本化 SDK/Release 使用模拟器。模拟器问题先走
`SIMULATOR_CHANGE_APPROVAL_REQUIRED`，获批后由模拟器仓库实现和发布；消费者不得复制实现、
手写 IPC 或读取临时 build。

## D2. 完整曝光键和 Release 采集物是数据事实源

原始帧、标签和 detector 输出必须按
`(session_id, producer_epoch, frame_seq, timestamp_ns)` 精确连接。正式采集只使用当前锁定
Release 的 collector 和 validator；最近帧匹配、插值补齐、消费者自写 TCP collector 以及旧
Windows 性能记录都不能替代该合同。

## D3. Truth 只存在于离线训练与验收

Exact corners、目标姿态和运动标签可以作为离线 target 或 metric，但不得成为在线特征。
train、validation 和 test 按完整 session 隔离；sealed test 一旦开启，不得再用于选择模型、
阈值或样本。

## D4. PnP 与坐标语义不能被下游补偿改写

生产四角顺序为 `bl, tl, tr, br`；每个 IPPE 候选从一开始就是完整的 `(rvec, tvec)`，选中的
tvec 表示装甲板中心在 OpenCV 相机坐标系中的位置。位置随后按同一曝光的姿态完成
camera → gimbal → tracker 变换。所有改动必须通过重投影、单位、坐标轴、曝光姿态和
round-trip 检查，并遵守 `COORDINATE_CONTRACT.md`。

## D5. 观测集合、身份和时间各自独立

在线输入是每次曝光的完整无序 `solved_armors` 集合。帧内编号、signature 和现有 tracker
字段都不能升级为永久物理身份。单帧 PnP 不依赖 `dt`；曝光时间用于选择匹配的云台姿态，连续
timestamp 与 `dt` 只在跟踪、预测、缺失和超时处理中使用。

## D6. 离线研究通过不等于生产替换

当前学习型 PnP 研究只允许使用同帧图像、检测几何和相机参数，并以现有生产路径作基线。
离线候选必须先经过 fresh validation 和一次 sealed test；生产接入还需独立批准、raw/candidate
A/B、数值一致性、时延预算和 fail-closed 回退。未满足这些条件时，生产 detector、PnP、
tracker、predictor 和 fire control 保持不变。

## D7. 权威信息单点保存

- 当前工作：`agent-team/BOARD.md`
- 概率 PnP 研究：`training/armor_pose/research-state.yaml`、`training/armor_pose/findings.md`
- 角点修复实验：`training/corner_pnp/` 下的 manifest 与结果
- 坐标合同：`modules/autoaim/src/aim_core_from_vivsionn/AngleSolver/COORDINATE_CONTRACT.md`
- 证据链与 registry：`modules/autoaim/docs/trajectory_evidence_chain.md` 及相邻 JSON registry

模型、checkpoint、原始数据、正式或失败的验收证据和 `deletion_allowed=false` 目录均受保护；
上下文文件只做索引，不再承载完整实验报告。
