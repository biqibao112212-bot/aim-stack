# 角点修复网络方向否决记录

- 决策日期：2026-08-19
- 状态：**终止并否决，不训练、不部署、不再保留数据资产**
- 生产行为：继续使用现有 detector/legacy refinement 与原始 PnP 链路

## 决策

停止以 CNN、MLP、热图网络或二级 benefit gate 修正 detector 四角点的路线。该路线不得接入生产 PnP、tracker、预测器或火控，也不得由自动任务继续采集、训练或调参。只有用户以后明确撤销本决定并重新给出资源预算、实车部署目标和验收门，才允许重启。

## 否决依据

1. Linux 1.3.1 的 session-disjoint sealed test 中，最佳候选总体角点 RMS 只改善 `2.55%`，低于预先声明的 `5%` 门。
2. 部署域复核中，stationary 的改善不能迁移到 spin/combined；修正后的 transverse P95 分别由 `11.06/13.82 mm` 恶化到 `25.48/38.06 mm`。这不是可接受的尾部风险。
3. 图中装甲常只有约 `20 px` 宽。raw 与 exact 常只差 `1--2 px`，网络却会系统性改变宽度和面积；平面 IPPE 会把这种小像素偏差放大成数百毫米的深度或分支跳变。
4. motion/view/visibility 强共线，多个预声明 train cell 的 detector 有效匹配接近零。角点网络既无法修复 detector 没有给出的目标候选，也无法凭单帧输入补回不可见的灯条/数字特征。
5. 更保守的 U-Net 热图先验版本避免了明显的 per-mode 回退，但总体只改善 `2.25%`；线性 gate、小 MLP 和 8,000 组确定性 veto 均只能选择 `reject-all`。
6. 即使离线指标继续提高，实车仍需证明 detector 域迁移、预处理与 TensorRT/ONNX 数值一致、native latency、异常回退和端到端尾部安全。当前收益不足以支持这些额外存储和工程成本。

## 清理与边界

2026-08-19 已停止所有相关训练，并删除该方向的 raw RGBA 采集、exact labels 副本、训练/验证/测试会话、checkpoint、gate 模型、PnP/observer sidecar、可视化和离线评测产物。根盘与 Data 盘合计已释放约 `119 GiB`。删除不可恢复；仓库 Git 历史只用于审计曾经做过的工作，不构成可复现实验资产。

Data 盘是 NTFS3。清理最后两个目录树时内核停在不可中断 `vfs_unlink`，无 I/O error，但无法从用户态安全完成；若下次重新挂载后它们仍存在，应删除 `linux-observer-session-macro-v1-20260819` 和 `linux-observer-validation-linear-recap-v3-20260819`。这两个残留不得作为保留证据或继续训练输入。

本次否决只针对“学习式角点修复”。模拟器 Release 1.3.1、公共 SDK、现有 detector、legacy refinement、PnP、通用因果匿名观测器、预测器及其非角点修复数据不在删除范围。
