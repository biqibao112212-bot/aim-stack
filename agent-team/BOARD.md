# Aim Stack 当前任务板

## 已完成

- 锁定 Daedalus `1.4.0-learning-r1` 为内部唯一模拟器。
- 建立 `modules/autoaim-research`，固定同济 YOLO/IPPE/11D EKF 上游提交。
- 完成受保护 ONNX 模型路径、ONNX Runtime 1.22.1、ROS odom 坐标和同曝光真值日志适配。
- 通过构建、核心单测、模型形状冒烟和 1.4.0 真实帧端到端冒烟。

## 已完成：11 维 EKF 异常起点

使用 Daedalus `1.4.0-learning-r1` Release 高性能无前端模式和同一版本的
`modules/autoaim-research` 完成三组 20 秒采集：

1. 原地旋转：真值角速度 `8 rad/s`。
2. 平移：真值线速度约 `1.5 m/s`。
3. 平移 + 旋转：真值线速度 `1 m/s`，角速度 `6 rad/s`。

接受运行为 `20260825-ekf11-baseline-r2`。原始 JSONL、Release 日志、SDK 运动 ACK 和
真值云台日志保留在工作区 `runtime/autoaim-research`；锁、指标、图和可复现脚本保留在
`modules/autoaim-research/experiments/ekf11-baseline`。

## 下一步

先将这三组图组织成教程的“观察到异常”章节，再以对照实验区分 PnP 观测质量、
关联错误、滤波器模型和参数的影响，不预设唯一原因。

## 保留

旧教程图、旧数据、旧代码和历史实验暂不删除；它们仅供重写时取证，不构成当前结论。
