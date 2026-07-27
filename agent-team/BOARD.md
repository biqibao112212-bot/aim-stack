# 火控与云台控制任务板

上下文版本：`CTX-FIRE-CONTROL-GIMBAL-2026.07-v1`

## 当前状态

- 分支 `research/fire-control-gimbal` 已从 `aim-stack/main` 的 `464605c` 创建。
- 独立 worktree 已建立；主 worktree 的预测器草稿和上下文改动保持原样。
- Simulator 1.0.3 / SDK 1.0.0 已锁定；高性能模式纯后台运行，模拟器物理射频硬上限为 20 Hz。
- 总目标已确定：在冻结干净预测器输入下优化控制/火控，测量 20 Hz 上限下的
  实际射频—命中率前沿；随后冻结同一火控方案，量化预测器质量变化的最终影响。
- FireControl、MPC、planner、fixed-rate command loop 和桥接器 8 项测试均通过；
  控制效果和逐发中心命中仍需后续静态、多距离验收。
- 已以 Release 1.0.3 / SDK 1.0.0 默认高性能模式进入原生 Shooting Range；
  动态靶的 `set_target_3_spin` 与静止靶的 `set_target_3_motion` SDK ACK 均成功。
- 已从默认火控运行路径移除预测器研究的 pipeline JSON/JSONL、Stage3、FOV 和
  profiler I/O。无 I/O 正式窗口达到 105.768 Hz 完整视觉、7.187 ms 流水线、
  6.194 Hz 实际射频；250.008 Hz 物理、检测覆盖、采集丢帧和 GPU map 错误均正常。
- 模拟器变更已在所有者仓库独立发布为 Release 1.0.3；消费者分支只更新版本锁和
  SDK 依赖，不携带模拟器实现。SDK、模型和数据集未修改；消费者侧修复了
  PowerShell→WSL CRLF 启动故障、补充自测注册，并增加可选遥测隔离开关。

## 进行中

冻结静态多距离验证合同；先保留 22 m/s 控制侧与 25 m/s 模拟器侧差异作为无补偿
基线，不以经验角度、距离或中心偏置修正结果。

## 冻结与限制

- Stage 3 预测器训练、模型导出、tracker 重构和打符保持冻结。
- 未完成坐标合同审查前，不调整 pitch/yaw 符号、经验偏置或瞄准点补偿。
- 未定义安全门禁和可重复测试前，不进行自动实弹或等价高风险控制验证。
- 未经模拟器变更审批，不修改模拟器、SDK 或正式 Release。
- 未完成性能与静态弹道门禁前，不实现物理真值注入。
- 未经用户审批，不新增任何经验弹道、距离、角度或中心偏置补偿。

## 下一步

1. 冻结静态多距离矩阵，审计 22 m/s 控制参数与 25 m/s 模拟器弹速
   契约差异，再验证装甲板中心误差；不得用经验补偿掩盖差异。
2. 梳理当前链路：预测/tracker 输出 → planner/MPC → 云台命令 → fire advice →
   实际发射/命中事件，并冻结射频/命中统计口径。
3. 设计三臂真值诊断：现有预测基线、PnP 当前锚点+真值未来、全真值当前/未来。
4. 只有门禁和注入合同通过评审后，才实现真值 A/B 与后续预测器质量退化实验。

## 待验证

- 消费者边界与仓库架构检查。
- 当前 FireControl/MPC/fixed-rate command loop 的已有测试与实际覆盖范围。
- 输入输出坐标、单位、时间戳、延迟和饱和语义。
- 发射建议到实际命令的幂等性、时序和失败关闭行为。
- 20 Hz 是命令槽上限还是实际发射器可实现上限，以及两者的可观测事件来源。
- 干净预测器输入的定义、来源、重放格式和禁止泄漏边界。
- 命中事件与发射事件的关联键、有效交战时间和统计置信区间。
- 预测器质量维度：位置/速度/时序误差、延迟、抖动、漏检、离群和不确定性校准。
- 当前 Release 是否公开了足够的发射—命中关联字段；不足时只能提交 SDK 提案，
  不能从模拟器内部旁路取数。
- 多距离“装甲板正中心”的命中点真值与统计口径是否已由 SDK v1 暴露。

## 当前证据

- 正式无 I/O 原生靶场窗口：
  `runtime/fire-control-gimbal/native-range-dynamic-20260727-r9-no-io-formal/`
- 正式 1.0.3 静止靶窗口：
  `runtime/fire-control-gimbal/native-range-stationary-20260727-r2-sim103/`。5 m、3 号靶、SDK `stationary` ACK；45 秒内 151 发/151 命中，全窗口 3.356 Hz，首发至末发活跃段 6.708 Hz；exact-exposure 有效，完整视觉约 64 Hz。后半段检测器持续失去目标，因此该轮不能宣称火控达到 20 Hz，也不能把 `accurate` 当作装甲板正中心命中证明。
- `native-range-stationary-20260727-r1-sim102/` 为废弃证据：1.0.2 零等待无窗口循环导致图像源年龄增至 13.49 s、完整视觉为 0；不得计入自瞄性能。
- 逐帧遥测回归对照：`native-range-dynamic-20260726-r3-formal/`、
  `native-range-dynamic-20260726-r4-detector-profile/`、
  `native-range-dynamic-20260726-r5-summary-profile/`。
- 外参精确曝光复核：72 个会话、5760 个样本通过；静态 PnP 几何在
  2/3/4.5/5/7 m 的 small/large armor 合成矩阵均通过。该证据不等价于逐发中心命中。
