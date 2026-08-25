# Aim Stack 项目上下文

- 仓库：`/home/potato/Projects/仿真/repos/aim-stack`
- 正式分支：`main`
- 当前主模块：`modules/autoaim`
- 暂停模块：`modules/energy-buff`
- 模拟器锁：Daedalus Simulator / SDK `1.3.1`，Ubuntu 24.04，Linux x86_64

本文件只保存长期有效的边界和约束。当前研究进度见 `BOARD.md`，已确认的设计选择见
`DECISIONS.md`；逐轮指标、失败过程和复现实物由实验 manifest、registry 与 Git 历史保存。

## 仓库边界

- 本仓库拥有自瞄、轨迹预测、火控和打符的消费者代码，不拥有模拟器、SDK 或 Release。
- 消费者只通过 `SIMULATOR_CONSUMER_GUIDE.md`、`simulator.lock.json` 和版本化 SDK 使用
  模拟器，禁止复制模拟器源码、手写 SHM/TCP 协议或依赖临时构建产物。
- 模拟器改动受 `SIMULATOR_CHANGE_APPROVAL_REQUIRED` 约束：先提交复现、影响、公共接口和
  版本方案，获得用户明确批准后，再到模拟器仓库独立实现、测试和发布。
- `modules/energy-buff` 在完成 SDK v1 适配与验收前不得声称兼容。

## 数据与在线边界

- 跨会话曝光事件以
  `(session_id, producer_epoch, frame_seq, timestamp_ns)` 唯一标识；离线数据只能按完整键连接。
- 每次曝光在 `trackerUpdate()` 前的完整无序 `solved_armors` 集合是观测事实源。
- 帧内索引、detector number、signature、`tracked_id`、`tracked_armor` 和 `jump_flag` 都不是
  可跨帧继承的物理装甲身份。
- `u/v` 表示由 PnP tvec 得到的相机射线角，不是像素坐标或世界坐标。
- 模拟器真值只能用于离线标签和验收，禁止进入 detector、PnP、tracker、predictor、MPC
  或火控的在线输入。

## 几何合同

- 装甲板四角顺序为 `bl, tl, tr, br`；生产位置来自 free-IPPE 选中候选的 tvec。
- 相机、云台和 tracker 坐标变换必须遵守
  `modules/autoaim/src/aim_core_from_vivsionn/AngleSolver/COORDINATE_CONTRACT.md`。
- 不得用经验偏置、弹道补偿或符号翻转掩盖坐标系、单位、曝光姿态或时间戳错误。

## 受保护资产

- `/home/potato/Projects/仿真/models/engines`
- `/home/potato/Projects/仿真/dataset/autoaim-stage3-v1`
- `runtime` 中的原始帧、标签、identity ledger、采集清单、checkpoint 和 accepted/failed evidence
- 正式 Release、ONNX、TensorRT engine、训练权重、标注与不可再生数据

这些资产不得自动删除或覆盖。只有构建缓存和已由 manifest 证明可再生的临时结果可以清理。

## 稳定入口

- 消费者指南：`SIMULATOR_CONSUMER_GUIDE.md`
- 模拟器锁：`simulator.lock.json`
- 角点/PnP 研究：`training/corner_pnp/`、`training/armor_pose/`
- PnP 与时序证据：`modules/autoaim/docs/trajectory_evidence_chain.md`
- 消费者边界检查：`.github/workflows/consumer-boundary.yml`
