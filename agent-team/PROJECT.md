# Aim Stack：火控与云台控制研究

- 上下文版本：`CTX-FIRE-CONTROL-GIMBAL-2026.07-v1`
- 仓库：`aim-stack`
- 分支：`research/fire-control-gimbal`
- 工作目录：`D:\仿真\active-worktrees\aim-stack-fire-control-gimbal`
- 所属模块：`modules/autoaim`
- 基线提交：`464605c46f496836897c1db9b8e76e2376376bf7`
- 模拟器锁：`Daedalus Simulator 1.0.1 / DaedalusSimSdk 1.0.0`
- 公共依赖入口：`SIMULATOR_CONSUMER_GUIDE.md` 与 `simulator.lock.json`

## 目标与所有权

本分支专门研究预测结果之后的火控与云台控制，包括目标选择后的规划、
弹道/延迟补偿、云台命令生成与执行、发射建议和安全门禁。研究应形成清晰的
输入输出合同、可解释的控制时序、可重复测试以及分阶段验收指标。

核心研究问题分为两个层次：

1. 冻结使用干净、可重复的预测器输入，只优化云台控制和火控方案。在最大允许
   射频 20 Hz 下，测量方案能够达到的实际有效射频、命中率及二者之间的前沿。
2. 冻结同一套已验收火控方案和全部非预测器条件，仅改变预测器输出质量，测量
   不同误差、延迟、不确定性和失效模式对最终实际射频与命中率的影响。

第一层隔离控制器自身能力，第二层量化预测器质量到最终作战指标的传递函数。
二者不得通过同时调整预测器与火控器来混合归因。

预测器与 tracker 在本分支中默认视为上游输入提供者。除非后续明确批准一个
最小接口适配，本分支不得继续 Stage 3 训练、修改预测器结构或混入预测器实验。
`modules/energy-buff` 保持暂停且不属于本分支。

## 边界

- 允许修改 `modules/autoaim` 内的 FireControl、planner、MPC、弹道、云台命令环、
  发射门禁、消费者侧适配、测试和本分支文档。
- 修改坐标、姿态、瞄准点或云台转换前，必须读取并遵守
  `modules/autoaim/src/aim_core_from_vivsionn/AngleSolver/COORDINATE_CONTRACT.md`。
- 不修改 `D:\仿真\repos\daedalus-simulator`、正式 Release、SDK 或发布脚本；
  不复制模拟器源码，不手写或镜像 SHM/TCP 协议。
- Simulator 1.0.1 / SDK 1.0.0 是固定的集成和验收依赖，不是当前研究对象。
  在纯算法、控制器和离线单元研究阶段不要求启动模拟器；测量实际发射事件、
  命中率和闭环云台行为时，使用该锁定版本作为统一验收环境，但不修改它。
- 发现模拟器缺陷或新增公共接口需求时，执行
  `SIMULATOR_CHANGE_APPROVAL_REQUIRED`，先提交证据和提案，等待明确批准。
- 模型、ONNX、TensorRT engine、checkpoint、数据集、标注和正式 Release 均为
  受保护资产，不得自动删除或覆盖。

## 起始实现

当前 `main` 已包含旧研究中的 FireControl、二阶位置 MPC、fixed-rate command
loop 和 SDK v1 消费者适配。旧共享仓库的 `feature/auto-aim-fire-control` 仅作
历史取证，不作为工作分支，也不整体合并或 cherry-pick。任何旧设计约束都需在
本分支重新审查后才能成为当前合同。

## 稳定验证入口

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-consumer-boundary.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-architecture.ps1
```

具体构建、单元测试和控制性能门禁将在完成现状审计与任务定义后写入本文件。

## 实验不变量与指标族

- 射频硬上限固定为 `20 Hz`；必须区分请求射频、实际发射事件射频、有效交战
  时间内射频以及因安全门禁抑制的发射。
- 命中率的分母使用实际发生的发射事件，不得使用 fire advice、请求命令或理论
  槽位替代；命中事件与发射事件必须可追踪关联。
- 干净输入基线必须版本化并可完全重放，且不得把模拟器真值绕过预测器接口直接
  注入火控内部。
- 第二层实验必须锁定火控代码、参数、随机种子、目标轨迹、弹道环境和发射上限，
  只允许改变版本化的预测器输出质量配置。
- 除总体射频与命中率外，后续至少按距离、目标运动状态、预测时域、云台误差、
  预测误差、延迟和门禁原因分层报告；正式口径在基线审计后冻结。
