# Linux 1.3.1 角点可靠性修复与预测器闭环

> 2026-08-19 后续部署域复核：本文的 sealed 图像域角点结论仍作为原始证据保留，但新 Linux 完整 detector/repair/IPPE/observer 会话已证明 v3 在 spin/combined 上过修正，生产 PnP 接入门失败。当前有效授权边界以 [因果观测器与角点修复部署域复核](causal_observer_and_corner_repair_validation.md) 为准。

## 结论

本轮把“角点像素指标通过”和“预测器部署指标通过”分开验收：

- `v3-context-spatial-reliability` 角点修复器通过全新 session-disjoint sealed test；
- 同一修复器经 unchanged nominal-small-armor IPPE 后，能降低冻结 400 ms LOS 预测器的平均误差并提高 55 mm 覆盖；
- 50/100/200 ms 的部署 P95 仍约为 293.7/282.1/211.0 mm，未达到 55 mm；因此不得声称生产预测器完成或接入 C++；
- exact-corner 同可用性上界的 P95 仅为 2.15/3.26/5.83 mm，确认冻结物理预测核在该 constant-direction combined session 上成立，剩余主瓶颈是 detector candidate identity、角点/PnP 重尾和观测可用性。

## 为什么增加可靠性头

上一版 `v2-context-spatial` 只输出 8 维修正。它把两个不同问题混在一起：raw 错了多少、应该怎样移动四角；以及当前 raw 是否已经足够准确、根本不该动。

姿态等价目标加 `4 px` 最小预测量门在 validation 上通过，但新 sealed test 整体恶化 17.81%；linear 的 raw 原本只有 1.90 px RMS，却被修到 5.74 px。预测量大只表示模型“想改很多”，不是可靠性。

v3 共享相同的 RGB context CNN 和 15D raw geometry trunk，分成两个头：

```text
RGB 1.5x ROI -- CNN --+
                      +-- shared 128D -- 8D corner correction
raw geometry 15D -----+                \
                                        +-- reliability logit
detector score -------------------------/
```

修正头使用 context-normalized、nominal-PnP-pose-equivalent 监督；可靠性头的离线标签是 `raw visual coordinate RMS >= 4 px`，使用带训练集正类权重的 BCE。总损失固定为 `normalized correction MSE + 0.25 * reliability BCE`。

线上只在 `sigmoid(reliability_logit) >= 0.5` 时应用修正。exact corners、运动模式、距离、identity、range 和 future 均不是模型输入。

## 姿态等价角点目标

Release 标签给出 marker asset 的实测宽高，而生产 IPPE 固定使用 nominal 135x55 mm。训练时用固定 `bl,tl,tr,br` 规范矩形建立同一图像 homography，把视觉 exact corners 转成“送入 nominal IPPE 后恢复实测板姿态”的监督角点。

在 1029 个 validation 样本上，该目标相对视觉 exact corners 仅移动约 0.23 px P95；重新运行 nominal IPPE 后与实测模板姿态的闭环误差约 0.012 mm P95。不能按 physical vertex 符号直接映射，因为装甲板转到背面时物理顶点符号会反转，而 detector/PnP 仍要求图像语义顺序。

## 正式角点结果

三种子 validation：

| seed | aggregate RMS improvement | reliability balanced accuracy | gate |
| --- | ---: | ---: | --- |
| 1801 | 6.64% | 80.49% | pass |
| 2801 | 6.77% | 77.95% | pass |
| 3801 | 5.74% | 79.26% | pass |

按预注册规则选择 seed 2801。checkpoint：

```text
/home/potato/Projects/仿真/models/engines/corner-repair-reliability-linux-1.3.1-20260819-r2-seed2801/corner-repair.pt
sha256 46068e714899dd612e9de7ca3642db75c4022d15770793325e714bb5eac67f91
```

该冻结 checkpoint 的旧 metadata 把 `context_scale` 误记为 `1.0`，但训练、validation、sealed test 和下游适配器实际都调用固定 `context_patch`，其数值路径是 `1.5x`。源码已修正未来 checkpoint 的记录条件；在生成绑定正确 metadata 的新 artifact 并完成数值等价证明前，不得仅依赖旧 checkpoint 字段导出生产 ONNX。

全新 sealed test 共 494 行：

| mode | raw RMS px | repaired RMS px | improvement |
| --- | ---: | ---: | ---: |
| aggregate | 7.391 | 6.915 | 6.43% |
| linear | 1.572 | 1.572 | 0.00% |
| linear_and_spin | 9.449 | 8.593 | 9.06% |
| spin | 11.730 | 11.493 | 2.01% |
| stationary | 2.844 | 1.588 | 44.16% |

494 行只应用 164 行。linear raw 已很准，可靠性头全部拒绝修正，避免重演上一版的新域过修。

## detector applicability 必须单列

新增采集中两个 session 在不变的 25 px detector-to-truth 关联门下为零覆盖。数据构建器现在保留 header-only rows、qualified exposures、matched exposures=0 和 coverage=0；训练器跳过其角点样本，但不删除 applicability 证据。

这不是把零样本计入角点 RMS，而是区分 conditional repair quality 和 detector applicability。修复器不能恢复 detector 未检出的装甲板。

## 下游同锚点 A/B

pose-aware sealed session：

```text
/home/potato/Projects/仿真/runtime/corner-repair-reliability-prediction-test-v2-20260819
```

该 session 使用 SDK-only 单 TCP client，同时保存完整 RGBA 和 `readExposureStateForFrame` 同曝光相机/云台/底盘位姿。276 个带标签曝光中，271 个曝光得到 379 行合格候选；uniform 部分 307 行，raw/repaired 均有 304 个相同 observation events，49 行实际应用修复。

冻结合同：400 ms history、50/100/200 ms horizon、depth weight 0.1、Huber delta 20 mm、omega memory 31、相同 anchor/availability/truth，只改变 raw 或 repaired corners。

| horizon | raw mean mm | repaired mean mm | raw P95 mm | repaired P95 mm | raw 55 mm | repaired 55 mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 ms | 138.7 | 131.2 | 287.4 | 293.7 | 23.8% | 38.1% |
| 100 ms | 142.8 | 134.3 | 267.7 | 282.1 | 20.0% | 30.0% |
| 200 ms | 123.3 | 109.1 | 235.2 | 211.0 | 26.7% | 40.0% |

paired mean delta 为 -7.47/-8.53/-14.17 mm；但 worsened fraction 仍为 42.9%/50.0%/46.7%，且 50/100 ms P95 回退。因此只能说角点修复带来下游平均收益，不能授权生产预测器。

exact-matched 上界在同一可用性 mask 下 55 mm 覆盖为 100%，P95 为 2.15/3.26/5.83 mm。这排除了“运动模式本身让 400 ms 模型必然失败”的解释。

## 失败方法与原因

1. exact-centered PnP Jacobian：跨 detector 常见十几像素误差时，线性近似中位误差约 55--60 mm，不能作为正式损失。
2. raw-centered Jacobian：中位数改善，但 applied 样本近似误差 P95 仍约 57 mm，尾部不足以支撑部署目标。
3. correction-magnitude gate：validation 有效、首次新 test 严重过修；“模型想改多少”不等于“模型知道 raw 错了”。
4. 三种子 ensemble disagreement：共享分布偏差仍让 linear 失败，不能靠模型间方差修复共同错误。
5. v4 U-Net heatmap seed 1901：validation aggregate 仅改善 1.61%，linear 恶化 8.90%；全分辨率 8192-way 空间监督在当前数据量和 CPU 训练合同下未学稳，停止其余种子。
6. post-hoc 10 px truth association：只剔除 1 行，预测 P95 仍很高；邻板错配是真问题但不是全部重尾，且 truth gate 不是线上方案。

## 当前授权边界与下一步

- 角点修复 v3：正式离线通过，可继续做 ONNX/Python-C++ preprocessing parity 和 shadow integration；尚未授权改变生产 PnP 输入。
- 400 ms 物理预测核：exact-observation 条件下成立。
- 部署观察链：未通过。下一步必须实现候选集合、ambiguous/missing、真实时间和 causal identity/innovation observer，再做大规模 session-macro 复核；不能继续只用 truth slot 或通过调角点 test 阈值掩盖身份问题。
- C++ 接入点仍应位于 `AngleSolver::solveArmors` 的 raw vertices 保存之后、legacy refine/PnP 之前；模型接受时跳过二次 refine，拒绝/异常时完整 fallback legacy。生产接入还需要静态 ONNX、OpenCV-DNN parity、模型 manifest/hash、延迟与 fallback 诊断。
