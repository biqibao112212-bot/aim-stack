# Aim Stack 长期上下文

- 仓库：`/home/potato/Projects/仿真/repos/aim-stack`
- 正式分支：`main`
- 内部 PnP/EKF 唯一实现：`modules/autoaim-research`
- 模拟器唯一锁：Daedalus `1.4.0-learning-r1`，Ubuntu 24.04 / Linux x86_64
- 旧 `modules/autoaim` 只保留历史证据，不作新实验运行入口。

## 边界

- 自动瞄准研究链只包含四角点检测、PnP、11 维 EKF 和评估日志；不含开火、弹道、
  命中事件或云台控制。
- 模拟器、SDK 和 Release 由 `daedalus-simulator` 仓库拥有；本仓库只通过版本化 SDK 和
  `simulator.lock.json` 使用它们。
- 目标真值只能在估计器更新后用于记录、匹配和评估；不得进入 detector、PnP 或 EKF。
- 图像、曝光位姿和真值必须以
  `(producer_epoch, frame_seq, timestamp_ns)` 精确联结。
- 正式数据采集只使用无前端的 Release 高性能模式；必须保存曝光时间戳，
  并分别报告处理 FPS 和源序列推进速率。

## 受保护资产

`models/engines`、原始帧、标注、checkpoint、实验 manifest 和正式 Release 不得自动删除或
覆盖。临时构建缓存只有在可再生且不含证据时才可清理。

## 权威入口

- 实现与运行：`modules/autoaim-research/README.md`
- 自动瞄准、模型、推理运行时和坐标锁：`modules/autoaim-research/implementation.lock.json`
- 模拟器锁：`simulator.lock.json`
- 当前任务：`agent-team/BOARD.md`
- 长期决策：`agent-team/DECISIONS.md`
- 已接受的 EKF 异常基线：`modules/autoaim-research/experiments/ekf11-baseline`
