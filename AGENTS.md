# Aim Stack 消费者仓库指令

本仓库固定为 `D:\仿真\repos\aim-stack`，正式分支为 `main`。它只维护自瞄和打符算法，不包含模拟器源码。

开始任务前核对仓库、分支、HEAD 和 dirty 状态，只读取本仓库的 `agent-team/PROJECT.md`、`BOARD.md`、`DECISIONS.md`，并读取 `simulator.lock.json` 与 `SIMULATOR_CONSUMER_GUIDE.md`。任何新自瞄、火控或打符分支必须从包含这两个文件的 `main` 创建，不得删除、绕过或私有复制消费者指南。

模拟器依赖只能来自 `D:\仿真\repos\daedalus-simulator` 发布的 `DaedalusSimSdk`。禁止复制 Rust 模拟器、手写 SHM/TCP 协议、从旧工作树链接二进制或修改 Release 包。自瞄仓库及其任务无权编辑 `D:\仿真\repos\daedalus-simulator` 中的模拟器源码、SDK 或发布脚本。

审批门禁标识：`SIMULATOR_CHANGE_APPROVAL_REQUIRED`。调试自瞄时发现模拟器 bug 或新需求，必须立即停止任何模拟器侧写操作，向用户提交复现、证据、影响、建议接口和预期版本变化，并等待用户在该提案之后明确批准。批准前只能修改消费者侧诊断、适配或文档；不得先改后报、不得把消费者需求伪装成兼容修复、不得直接修改正式 Release。批准后也必须切换到模拟器仓库独立实现和发布，不能在自瞄分支携带模拟器代码。

模型二进制位于 `D:\仿真\models\engines`，由 `models/manifest.json` 记录。模型、ONNX、训练权重、标注数据和正式发布包属于受保护资产，任何自动清理都不得删除；缺少用途判断时默认保留并报告。

任务结束时按 Agent Team skill 的收尾分类处理：长期边界写 PROJECT，当前状态写 BOARD，关键决策写 DECISIONS；跨模块接口进入所有者的公共契约；构建目录、缓存和可复现实验日志可删除；模型和不可再生数据不得自动删除。
