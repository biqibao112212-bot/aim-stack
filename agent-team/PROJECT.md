# Aim Stack：自瞄 B 与打符消费者

- 上下文版本：`CTX-AIM-STACK-2026.07-v3`
- 仓库：`aim-stack`
- 分支：`main`
- 工作目录：`D:\仿真\repos\aim-stack`
- 当前主模块：`modules/autoaim`
- 暂停模块：`modules/energy-buff`
- 模拟器锁：`Daedalus Simulator 1.0.1 / DaedalusSimSdk 1.0.0 / SHM v7 ABI r1 / 1440×1080 / Scene Control v1`
- 模拟器发布：`D:\仿真\releases\daedalus-simulator\1.0.1`
- 模拟器消费者统一入口：`SIMULATOR_CONSUMER_GUIDE.md`（v1）与 `simulator.lock.json`

本仓库只消费模拟器 Release 与 SDK，不包含模拟器源码。模型资产由 `models/manifest.json` 引用外部受保护目录，Git 不跟踪 engine。

所有自瞄、火控和打符分支必须继承消费者统一入口，明确 SDK 用法、三张原生地图、默认高性能模式和可视验收模式。消费者任务发现模拟器 bug 或新需求时，必须先向用户提交提案并等待明确批准；批准前不得编辑模拟器仓库、SDK、发布脚本或正式 Release。

## 自瞄 B 总目标

构建因果神经轨迹预测器：输入最近一段经过几何校验但未做时间平滑的逐曝光可见装甲板集合，以及任意未来时刻 `tau`；输出四块装甲板未来可击打位置的概率分布或多假设结果。曝光时间戳是时间原点，模拟器真值只作标签与验收，不得作为输入。

阶段一是固定模拟器/曝光契约；阶段二是在 tracker 前通过动态渲染 G2 修复 PnP yaw；阶段三仅在 G2 通过后进行有限、无泄漏数据采集，并训练固定的 TCN + 任意时间解码器。候选选择、云台、MPC 和火控保持冻结；模型必须提供不确定性/OOD 与安全回退。

当前阶段一的独立仓库和 SDK 边界已经建立；1.0.1 + SDK + TensorRT + shooting_range 动态基线已可重复启动。阶段二已完成并通过 G2：普通装甲板 `+15°` 倾角固定在 tracker/chassis 坐标系，生产 PnP yaw 通过曝光时刻云台姿态投影后进入 tracker；非零姿态合成回归与 3/5/7 m 原生靶场动态回放均已验收。阶段三正式 360-session 采集已完成；旧 `H=0.07 m` 已被经 exact-exposure 真值验证的 camera→gimbal R/T 取代。当前正式离线合同为最近最多 200 个真实观测事件及其真实时间戳的 `stage3-dataset-v3`，全量 111,527-train/36,297-validation 单 seed 训练、完整 validation 双物理基线评估和动态 ONNX parity 均已完成且未访问 test。该结果证明完整离线流程可行并在整体 validation 上超过刚体 baseline，但不构成多 seed/线上指标验收；test、TensorRT、tracker/MPC/火控和实弹接入仍冻结。PnP 观测记录的事实源为每帧完整 `solved_armors` 集合，离线循环 ID 仅为可重放派生字段。

## 2026-07-19 PnP joint-pose A/B checkpoint

Stage two now has a diagnostic-only fixed-tilt joint yaw+translation solver. It
uses the existing +15 degree ordinary-armor convention, per-frame effective
intrinsics/distortion, and both refined and raw detector corners. Its output is
serialized only under `solved_armors[].pnp_ab`; legacy PnP, tracker input,
candidate choice, gimbal, MPC and fire control remain unchanged.

The approved native-range experiment was repeated at 3/5/7 m: target 3, zero
linear speed, 30 deg/s spin, 30 s, offscreen DX12 performance mode. The retained
observation counts are 1507/1764/1084. Joint refined reprojection RMS p50 is
1.058/1.555/1.412 px versus legacy constrained-model 1.147/1.599/1.434 px, but
same-derived-ID temporal increment p50 is 2.96/5.86/7.01 deg versus legacy
2.74/6.14/6.37 deg. Therefore joint translation re-estimation is not a
consistent yaw repair and must not replace production output.

The marginalized local yaw sensitivity grows from 3.73 to 5.33 to 6.58
deg/px (p50) at 3/5/7 m. This directly supports a distance/pose conditioning
limit: a one-pixel corner-residual perturbation can correspond to several
degrees of yaw even after translation is optimized. This was the pre-stage-three
checkpoint; stage three was subsequently authorized on 2026-07-20. No numeric
G2 threshold was retroactively declared.

## 2026-07-19 chassis-frame +15 repair and replay

The ordinary-armor tilt is now applied in the tracker/chassis frame and
projected through the exposure-matched gimbal pose. The production constrained
 yaw path consumes this chassis-frame result; the prior camera-fixed yaw is
 retained only as an A/B diagnostic. The corrected sidecar remains available
 for refined-corner residual and conditioning diagnostics. A focused synthetic
 test with +15 degrees, 7 degrees gimbal pitch and -11 degrees gimbal yaw
 recovered the known chassis yaw within 0.1 degree and exact reprojection below
 1e-4 px.

The replay used the approved native shooting range, target 3, zero linear
motion, 30 deg/s spin, 30 s per distance, 3/5/7 m, DX12 offscreen performance
mode. The continuous plot is
`D:\仿真\runtime\pnp-chassis-pose-continuous-yaw-20260719.png`; metrics and the
quantitative summary are beside it. Production chassis-yaw adjacent increment
errors (p50/p95) were 2.59/14.82, 5.26/20.33 and 7.67/28.88 degrees at 3/5/7
m, versus the camera-fixed legacy 2.65/15.78, 5.66/39.24 and 8.76/69.40.
Together with the nonzero-pose synthetic regression and reviewed continuous
curves, this evidence closes G2 and stage two. The result validates the PnP
input semantics required by the later predictor; these replay files are not
declared training samples. The later stage-three authorization did not change
the status of these replay files.

## 2026-07-21 physical-core isolation

- Existing exact-exposure truth was reused; no recapture was needed. The
  qualified derived truth-history r5 dataset contains 111,527 train and 36,297
  validation samples and records `test_accessed=false`.
- A fixed exact-state constant-twist operator now propagates the real target
  center, exact translational velocity, exact yaw rate and q0 armor offsets.
  Its external output is still four future armor positions.
- On 36,297 validation samples, q0 P95 is 1.86e-9 m. Rule-query motion P95 is
  4.45e-6/8.19e-6/1.76e-5 m at nominal 0.1/0.2/0.5 s. All 1 mm gates pass.
- The previous centimetre physical tail was traced to numerical state recovery
  and rotating about the four-armor arithmetic centroid instead of the true
  vehicle center. The accepted physical equation is frozen; the next learned
  component is only the PnP-history observation adapter.
