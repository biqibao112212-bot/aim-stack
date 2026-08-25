# Aim Stack 长期决策

## D1. 唯一内部实现

PnP 与 11 维 EKF 研究只使用 `modules/autoaim-research`。旧 `modules/autoaim` 不删除，但只能
用于证据回溯，不得成为新实验的依赖、链接目标或隐式回退。

## D2. 同济基线直接固定

固定使用 `TongjiSuperPower/sp_vision_25@bd9f5e798fa3c6dd3b483ae6627796afb41c608d`
的 `Armor` / `Target` / `ExtendedKalmanFilter` 核心。本地适配只负责 ONNX 模型加载、Daedalus
同曝光位姿、坐标变换和研究日志。

## D3. 坐标与时间

PnP 在 OpenCV optical 系求解，再转到 ROS odom 轴方向，研究原点为同曝光云台轴心。
单帧 PnP 不使用 `dt`；曝光时间用于匹配位姿，连续时间才进入 EKF。

## D4. 真值与控制隔离

Daedalus 1.4.0 的同曝光目标真值只用于 EKF 后验评估。采集用真值云台可由模拟器
或独立采集器实现，但不得经由被测 estimator，也不得将目标真值写回它的观测。

## D5. 研究从异常出发

教程研究选读先展示现有 11 维 EKF 状态与真值的差异，再讨论原因和改进路线。
在正式数据出现前不预写结论，也不把更换滤波器或提升观测质量当作预设答案。

## D6. 正式采集运行模式

所有后续模拟器实验固定使用已锁定的 Release `--performance` 无前端模式。禁止使用
visible/debug 源码运行生成正式数据。采集必须记录曝光时间戳和帧序列，最终同时给出
算法处理 FPS 与源序列速率。靶车运动必须由 Release 公开 SDK 的 ACK 确认，不依赖会被
Release 清理的开发环境变量。
