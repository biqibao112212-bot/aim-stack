# Energy Buff（暂停）

Energy Buff（小符/大符）模块已迁入本仓库，但**尚未完成 Daedalus SDK v1 适配，也没有当前版本锁下的可验收运行入口**。请不要把本目录的历史脚本、旧 WSL 路径或示例命令当成可运行教程。

## 正确入口

- 当前可用的学习与开发入口是 [自瞄 B 教程](../autoaim/README.md)。
- 模拟器版本、资产边界和平台限制见[消费者统一指南](../../SIMULATOR_CONSUMER_GUIDE.md)与[版本锁](../../simulator.lock.json)。
- Energy Buff 的受限实现范围见 [BRANCH_SCOPE.md](BRANCH_SCOPE.md)。
- 历史导入来源和提交记录见[迁移记录](../../MIGRATION_SOURCES.md)。

## 传统视觉代码来源

本目录的 `src/aim_core_from_vivsionn` 是已导入、随本仓库版本控制的传统视觉基线代码；可直接从[本地源码目录](src/aim_core_from_vivsionn)阅读。过去引用的 Gitee 上游已不提供匿名可复现的克隆入口，因此已移除，不能作为读者的示例仓库或安装依赖。

装甲板模型的历史合同来源为 [RobotDetectionModel](https://github.com/broalantaps/RobotDetectionModel)。模型、TensorRT engine 和训练权重属于受保护资产，不会由文档脚本下载；实际使用时必须以项目所有者提供的版本锁与资产清单为准。

## Linux 说明

本仓库可在 Linux 上阅读、修改源码并运行文档检查；当前锁定的正式模拟器 Release 是 Windows 平台，因此不得把 Linux 上的编译或旧 Release 当成完整 Energy Buff 链路已验证。安装 Git 与基础工具的 Linux/Windows 命令见仓库根目录的[获取源码说明](../../README.md#获取源码)。
