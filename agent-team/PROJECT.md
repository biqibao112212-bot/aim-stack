# Aim Stack：自瞄 B 与打符消费者

- 上下文版本：`CTX-AIM-STACK-2026.07-v1`
- 仓库：`aim-stack`
- 分支：`main`
- 工作目录：`D:\仿真\repos\aim-stack`
- 当前主模块：`modules/autoaim`
- 暂停模块：`modules/energy-buff`
- 模拟器锁：`Daedalus Simulator 1.0.0 / DaedalusSimSdk 1.0.0 / SHM v7 ABI r1 / 1440×1080 / Scene Control v1`
- 模拟器发布：`D:\仿真\releases\daedalus-simulator\1.0.0`

本仓库只消费模拟器 Release 与 SDK，不包含模拟器源码。模型资产由 `models/manifest.json` 引用外部受保护目录，Git 不跟踪 engine。

## 自瞄 B 总目标

构建因果神经轨迹预测器：输入最近一段经过几何校验但未做时间平滑的逐曝光可见装甲板集合，以及任意未来时刻 `tau`；输出四块装甲板未来可击打位置的概率分布或多假设结果。曝光时间戳是时间原点，模拟器真值只作标签与验收，不得作为输入。

阶段一是固定模拟器/曝光契约；阶段二是在 tracker 前通过动态渲染 G2 修复 PnP yaw；阶段三仅在 G2 通过后进行有限、无泄漏数据采集，并训练固定的 TCN + 任意时间解码器。候选选择、云台、MPC 和火控保持冻结；模型必须提供不确定性/OOD 与安全回退。

当前阶段一的独立仓库和 SDK 边界已经建立。下一项研究工作仍是阶段二动态 G2；阶段三未授权。
