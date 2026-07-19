# PnP yaw 阶段二（G2）收口记录

- 状态：`PASS`，阶段二完成
- 日期：2026-07-19
- 仓库/分支：`aim-stack/main`
- 基线 HEAD：`f681a467b45f34d88afda19cebe3c1f6cad8b250`（改动尚未提交）
- 模拟器：Daedalus Simulator 1.0.1 / DaedalusSimSdk 1.0.0
- 后续门禁：阶段三采集与训练仍未授权

## 结论

修正后的生产 PnP yaw 已满足后续神经轨迹预测器的前置输入要求。这里的
“满足”指逐曝光装甲板 yaw 的坐标语义和几何模型已经正确，并不表示本轮
3/5/7 m 回放是训练样本，也不表示远距离 PnP 不再受角点精度限制。

G2 采用证据组合验收，不设置事后调参得到的单一数值门槛：

1. 非零云台姿态的合成回归证明坐标模型可恢复已知底盘 yaw；
2. 生产路径确实把修正后的底盘系 yaw 送入 tracker；
3. 原生靶场 target 3 在 3/5/7 m 的慢速自转回放呈现连续的分板 yaw 斜坡；
4. 运行链路、曝光匹配、TensorRT、分辨率和场景 ACK 均完整；
5. 7 m 的较大抖动和观测间断被归为角点/检测精度上限，作为后续模型需要
   面对的观测噪声保留，不再通过修改 tracker、MPC 或经验偏置掩盖。

## 已接受的修复

- 普通装甲板的固定 `+15°` pitch 倾角定义在 tracker/chassis 坐标系。
- 装甲板姿态先在底盘系由 yaw 与固定倾角构造，再通过曝光时刻的云台姿态
  投影到 OpenCV camera 坐标系做约束重投影。
- `Armor::yaw_absolute` 与生产 `Armor::yaw` 表示修正后的 tracker/chassis yaw；
  tracker 消费该值。
- 原先把倾角固定在 camera 坐标系的结果只保留为显式 A/B 诊断字段，不再
  进入 tracker。
- 联合 yaw+tvec 求解器继续作为诊断 sidecar；它没有被选为生产修复。
- 候选选择、tracker 身份策略、云台、MPC、火控和模拟器实现均未修改。

## 合成回归证据

聚焦测试构造普通装甲板 `+15°` 倾角、底盘 yaw `37°`、曝光云台 pitch
`+7°`、yaw `-11°`、深度 `5 m`。修正模型恢复底盘 yaw 的误差小于
`0.1°`，使用精确位姿时 RMS 与最大重投影误差均小于 `1e-4 px`。

验证命令：

```powershell
wsl.exe -d Ubuntu-OSTEP -- bash -lc "cd /mnt/d/仿真/repos/aim-stack/modules/autoaim && cmake --build build/ros2_trt --target aim_angle_solver_pnp_candidates_test aim_sim_talos_auto_aim_bridge -j2 && ./build/ros2_trt/aim_angle_solver_pnp_candidates_test"
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-architecture.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-consumer-boundary.ps1
```

结果：聚焦测试通过；架构检查和消费者边界检查通过。本阶段没有对模拟器
仓库执行写操作；其现有、范围外的工作树改动保持未触碰。

## 动态渲染证据

实验固定为原生 `shooting_range`、3 号靶车、零线速度、`30 deg/s` 慢速
自转、3/5/7 m 各 30 s、DX12 离屏高性能模式、TensorRT `vivsionn_trt`、
1440×1080。三轮均收到 Scene Control `create_session`、`set_scene`、
`set_target_3_spin` ACK。

成功运行目录：

- `D:\仿真\runtime\pnp-chassis-pose-d3-20260719-r3`
- `D:\仿真\runtime\pnp-chassis-pose-d5-20260719-r1`
- `D:\仿真\runtime\pnp-chassis-pose-d7-20260719-r1`

三轮模拟器 main/capture 分别为 106.643/106.643、124.370/124.370、
130.819/129.820 Hz；TCP sent 总数为 3451/3397/3634；bridge completed
vision 为 1655/1475/1723。`tcp_bind_failed`、queue drop 和 GPU map error
均为 0，曝光位姿/真值严格匹配。

连续曲线见 [pnp_yaw_stage2_target3_3_5_7m.png](pnp_yaw_stage2_target3_3_5_7m.png)。
该图 SHA-256 为
`F9D84DD2F01DCAF99C82113236747BF7911BE6C0B261350064CCBD2376B63622`。
相邻观测间隔超过 0.2 s 的位置故意断开，四种颜色仅表示离线循环板 ID。
3 m 曲线最清晰，5 m 保留可辨识的连续旋转趋势，7 m 噪声和间断明显增加；
这一距离退化符合角点分辨率的物理限制，并未破坏修复后的坐标语义。

绘图可由仓库脚本 `scripts/plot-pnp-chassis-yaw.py` 从三个
`observations.cyclic.jsonl` 文件重建。

## 保留与边界

- 源码、聚焦测试、分析/绘图脚本、本文和连续曲线保存在消费者仓库工作区。
- 三轮原始 pipeline、观测、ACK、bridge 与 simulator stats 日志保留在上述
  `D:\仿真\runtime` 目录，未删除。
- d3 的一次 DX12 `ResizeBuffers / Invalid surface` 失败运行保留在
  `pnp-chassis-pose-d3-20260719-r2`；启动器失败关闭，随后使用新 token 重跑
  成功，没有 broad kill，也没有修改模拟器。
- 模型、engine、标注、训练数据和正式 Release 均未删除或覆盖。
- G2 通过不授权阶段三；下一步仅讨论采集与训练方案。
