# Aim Stack 消费者仓库指令

本仓库固定为 `D:\仿真\repos\aim-stack`，正式分支为 `main`。它只维护自瞄和打符算法，不包含模拟器源码。

开始任务前核对仓库、分支、HEAD 和 dirty 状态，只读取本仓库的 `agent-team/PROJECT.md`、`BOARD.md`、`DECISIONS.md`，并读取 `simulator.lock.json` 指向的模拟器 Release 契约。

模拟器依赖只能来自 `D:\仿真\repos\daedalus-simulator` 发布的 `DaedalusSimSdk`。禁止复制 Rust 模拟器、手写 SHM/TCP 协议、从旧工作树链接二进制或修改 Release 包。

模型二进制位于 `D:\仿真\models\engines`，由 `models/manifest.json` 记录。模型、ONNX、训练权重、标注数据和正式发布包属于受保护资产，任何自动清理都不得删除；缺少用途判断时默认保留并报告。

任务结束时按 Agent Team skill 的收尾分类处理：长期边界写 PROJECT，当前状态写 BOARD，关键决策写 DECISIONS；跨模块接口进入所有者的公共契约；构建目录、缓存和可复现实验日志可删除；模型和不可再生数据不得自动删除。
