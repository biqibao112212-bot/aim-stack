# 仿真四角点联合修复网络：完整实验记录

## 1. 本轮问题和结论

本轮只回答一个冻结的问题：在不改 detector、不改 free-IPPE、不改 tracker 和预测器的条件下，能否从 detector 的四个 raw 角点本身预测一个联合 8 维修正量，并把改善传到 PnP？

答案分两层：

1. **同采集域、完整 segment 外推成立。** 12 折 leave-complete-segment-out 中，网络把联合角点 coordinate RMS 的 P50/P90/P95 从 `0.962/1.779/2.139 px` 降到 `0.586/1.205/1.446 px`；同一批角点经未改动 free-IPPE 后，3D PnP 从 `142.9/369.5/458.3 mm` 降到 `59.9/203.3/277.9 mm`，camera 横向从 `7.90/17.55/21.28 mm` 降到 `3.45/11.05/14.59 mm`。
2. **跨 session 只部分成立，不能声称普遍泛化。** 两个 session 合并后仍改善，但方向不对称。用 combined 训练、外推 spin 时明显改善；只用 spin 训练、外推 combined 时，角点 P50/P95 从 raw 的 `0.969/2.186 px` 恶化到 `1.177/2.317 px`。该条件下 PnP 横向仍由 `6.82/22.39 mm` 改善到 `5.31/18.80 mm`，但角点本身的负迁移说明模型尚未覆盖组合运动观测域。

所以本轮结果是：**仿真角点偏差包含可学习的系统结构，联合修复值得继续；当前 coordinate-only 网络不是生产模型，也不能替代实车标定或 sim-to-real 验证。**

## 2. 数据与冻结输入合同

权威输入是：

```text
D:\仿真\runtime\stage3-corner-observation-atlas-v1-observational-20260805-r3\rows.csv
SHA-256 dd4e35cd5e67971fa996776466817ae7c521349b6e2223328f7267e1ede6bc88
```

数据包含 4,280 个检测、2 个 session、12 个完整运动 segment，`wide_6mm` 相机，距离 `1.679--5.690 m`。四角点顺序固定为 `bl/tl/tr/br`。标签是：

```text
exact_projected_corner_px - raw_detector_corner_px
```

网络允许输入的只有 raw 四边形的可部署几何：

- 四角点相对中心、按可见尺度归一化后的 8 个坐标；
- 图像中的中心位置 2 维；
- log 尺度、log 长宽比、可见宽度和方向的 sin/cos；
- 合计 15 维。

以下字段明确禁止进入模型：truth corner/pose、`range_m`、斜视角、运动模式、session/segment、物理板身份、PnP 输出、未来状态。它们只用于离线分层和评分。完整预注册合同见 `sim_corner_residual_network_contract.json`。

## 3. 网络与训练方式

模型是一个低容量、联合四角点 MLP：

```text
15 -> 64 -> 64 -> 8
LayerNorm + SiLU + dropout 0.05
```

输出层零初始化，预测四个角点的有符号 `(dx,dy)`，每个坐标的最终修正限制在 `[-6,+6] px`。训练使用 SmoothL1、AdamW、最多 240 epoch、patience 30。每个外层折训练 seed `17/29/43` 三个模型，均值作为角点修正，seed 间离散度完整保留但尚未校准成置信度。

所有输入和标签标准化参数只用外层训练数据拟合，并写进每个 Lightning checkpoint。环境固定为：

- `D:\Anaconda\envs\yolov8\python.exe`；
- Python 3.10.18；
- PyTorch 2.7.1+cu118；
- Lightning 2.5.6；
- NVIDIA GeForce RTX 4060 Laptop GPU。

## 4. 数据隔离与过拟合检查

使用两套外层合同：

1. 主结果：12 折 leave-one-complete-`(session,segment)`-out；
2. 压力结果：2 折 leave-one-complete-session-out。

内验证从训练域再留一个完整 segment，禁止随机拆帧。外层测试折不参与标准化、早停、超参数或 checkpoint 选择，只在训练完成后生成 OOF 预测。

第一次正式运行在进入跨 session 折时触发“训练集为空”门禁。原因是旧实现试图把唯一剩余训练 session 整体再作为验证集。失败目录和 36 个 checkpoint 均保留，进度标记为 `failed`；修正后改为从剩余训练 session 留一个完整 segment，再从新目录完整重跑，没有复用失败运行的主折输出。

## 5. 对照方法

每个 OOF 检测都保留六个角点臂：

- `raw`：不修正；
- `mean`：只用训练折的每坐标平均偏差；
- `ridge`：用相同 15 维可观察特征的线性 Ridge；
- `current_refined`：当前继承的 raw-to-refined 图像启发式；
- `network`：三 seed 联合角点 MLP；
- `exact`：精确投影角点，只作 oracle 闭环。

没有按 PnP 结果挑选网络。角点训练全部结束后，独立脚本逐检测读取原始 observation 的曝光时刻相机内参，再运行 `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE)`；物点为 `135x55 mm`，`Point2f/Point3f`，选择重投影 RMS 最小候选。

4,280 条 raw 重算结果与历史 production-matched 证据的最大偏差为：位置分量 `<1.1e-12 mm`，重投影 `<1e-16 px`。这证明新评估没有悄悄更换 PnP 合同。

## 6. 主 12 折：完整中心与尾部分布

### 6.1 联合角点 coordinate RMS（px）

| 方法 | mean | P25 | P50 | P75 | P90 | P95 | P99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 1.070 | 0.679 | 0.962 | 1.324 | 1.779 | 2.139 | 2.891 | 4.895 |
| mean | 1.009 | 0.604 | 0.881 | 1.243 | 1.707 | 2.075 | 2.835 | 4.925 |
| ridge | 0.797 | 0.523 | 0.693 | 0.961 | 1.312 | 1.629 | 2.286 | 4.645 |
| current refined | 1.406 | 0.851 | 1.345 | 1.893 | 2.214 | 2.378 | 3.039 | 4.567 |
| network | **0.688** | **0.424** | **0.586** | **0.857** | **1.205** | **1.446** | **2.144** | 4.675 |
| exact | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

网络不仅降低 P95；P25/P50/P75/P90/P95/P99 都低于 raw。相对于同一检测的 raw，`74.65%` 的 3D PnP、`77.08%` 的 camera 横向 PnP 变好；仍有 `25.35%/22.92%` 变差，不能描述成逐帧单调改善。

### 6.2 四个角点分别的误差范数（px）

| 角点 | raw P50/P90/P95/P99 | network P50/P90/P95/P99 |
| --- | --- | --- |
| bl | 1.226 / 2.816 / 3.461 / 5.008 | **0.748 / 1.827 / 2.274 / 3.577** |
| tl | 1.116 / 2.575 / 3.095 / 4.550 | **0.747 / 1.645 / 2.045 / 3.276** |
| tr | 1.080 / 2.472 / 3.122 / 4.261 | **0.708 / 1.635 / 2.098 / 3.029** |
| br | 1.173 / 2.861 / 3.470 / 5.017 | **0.758 / 1.840 / 2.262 / 3.686** |

四个角点都改善，因此收益不是只把某一个点的常量偏差抵消掉。`bl/br` 的原始尾部更大，网络后的尾部仍略大，说明短边端点/下边缘相关的局部图像信息仍可能是下一阶段的重要输入。

### 6.3 未改动 free-IPPE（mm）

| 指标 / 方法 | mean | P25 | P50 | P75 | P90 | P95 | P99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3D raw | 181.0 | 78.6 | 142.9 | 239.6 | 369.5 | 458.3 | 605.2 | 840.9 |
| 3D network | **90.5** | **28.9** | **59.9** | **120.1** | **203.3** | **277.9** | **432.7** | 837.3 |
| 横向 raw | 9.15 | 4.96 | 7.90 | 12.20 | 17.55 | 21.28 | 29.00 | 44.43 |
| 横向 network | **4.99** | **1.92** | **3.45** | **6.44** | **11.05** | **14.59** | **23.00** | **44.31** |
| 深度绝对值 raw | 180.7 | 78.3 | 142.6 | 239.3 | 369.1 | 457.9 | 604.7 | 840.2 |
| 深度绝对值 network | **90.3** | **28.8** | **59.8** | **120.0** | **203.2** | **277.7** | **432.2** | 836.2 |
| 3D exact | 0.0036 | 0.0013 | 0.0031 | 0.0052 | 0.0069 | 0.0089 | 0.0123 | 0.0204 |

3D 误差几乎完全由 camera depth 主导；命中相关的 camera 横向定义为 `sqrt(error_x^2+error_y^2)`，没有把 depth 混入。网络同时改善深度和横向，但最大值几乎不变，说明极端异常仍存在。

当前 inherited refinement 在这套 atlas 上没有通过：角点 P50/P95 为 `1.345/2.378 px`，3D P50/P95 为 `140.8/500.3 mm`。它作为历史负证据继续保留，不据此删除生产代码。

## 7. 跨 session 压力测试与不对称性

合并两个 held session 时：

| 指标 | raw P50/P90/P95 | network P50/P90/P95 |
| --- | --- | --- |
| 角点 RMS (px) | 0.962 / 1.779 / 2.139 | **0.868 / 1.745 / 2.066** |
| 3D PnP (mm) | 142.9 / 369.5 / 458.3 | **111.4 / 288.9 / 350.0** |
| camera 横向 (mm) | 7.90 / 17.55 / 21.28 | **4.71 / 12.61 / 15.78** |

但按 held session 拆开后：

| held session | 方法 | 角点 P50/P95 (px) | 3D P50/P95 (mm) | 横向 P50/P95 (mm) |
| --- | --- | --- | --- | --- |
| combined-00 | raw | 0.969 / 2.186 | 140.8 / 467.6 | 6.82 / 22.39 |
| combined-00 | mean | **0.906 / 2.169** | **83.9 / 382.5** | 5.54 / 20.58 |
| combined-00 | network | 1.177 / 2.317 | 126.3 / 403.4 | **5.31 / 18.80** |
| spin-00 | raw | 0.952 / 2.082 | 199.4 / 453.9 | 8.66 / 20.23 |
| spin-00 | mean | 0.866 / 2.048 | 118.7 / 473.5 | 5.04 / 20.61 |
| spin-00 | network | **0.635 / 1.343** | **78.2 / 285.2** | **4.12 / 11.38** |

这表明：

- combined 采集域包含的几何变化可以覆盖 spin；
- spin 单域不足以覆盖 combined，网络发生负迁移；
- 简单 mean 在 held-combined 的角点与 3D 上反而优于网络，但网络的横向仍更好，说明不同角点误差模式对 PnP depth/横向的作用不同；
- 两个 session 远不足以证明 sim-to-real 或广条件泛化。

三 seed 离散度不能安全解决这个问题。held-combined 的平均 ensemble uncertainty 为 `0.342 px`，但 uncertainty 与“网络相对 raw 是否变差”的 Spearman 相关仅约 `0.044`，不能把它当成可靠拒绝 gate。

## 8. 当前能回答的误差来源

1. exact corners 将 PnP 关闭到微米以下，说明物点、坐标语义和 free-IPPE 数值链不是当前百毫米误差源。
2. train-fold mean 已带来小幅改善，证明四角点存在稳定的有符号系统偏差。
3. 主折 Ridge/MLP 继续改善，证明偏差还依赖 raw 四边形的中心、尺度、形状和方向，不是纯常量。
4. 跨 session 不对称说明一部分规律依赖未覆盖的图像/运动/姿态域；coordinate-only 输入无法分辨所有成因。
5. PnP 的 3D 长尾主要是 depth 放大；但最终瞄准更敏感的 camera 横向也真实改善，不能只看 3D 或角度。
6. 网络仍使约四分之一主折样本、约三分之一跨 session 样本变差；当前没有已校准的不确定性把这些样本安全筛出。

## 9. 下一步与边界

本 pilot 通过了预注册的主折门禁，因此下一步可以在仿真中采集与同一检测严格对齐的装甲板图像 patch、detector heatmap/feature 和 exact corner label，训练 image-conditioned 联合修复器。采集必须增加更多独立 session，覆盖旋转方向、角速度、线速度、组合运动、距离、斜视角、亮度和背景；仍按完整 session 留出，禁止随机帧切分。

在此之前：

- 不接入生产 detector/PnP/tracker/fire control；
- 不声称实车可直接学习，因为实车没有 exact label；
- 不声称仿真规律一定迁移，实车只能通过无真值一致性、少量弱标定、重投影/时序约束和安全 A/B 验证建立证据；
- 不用当前 ensemble spread 冒充 calibrated covariance；
- 不删除旧 refinement、失败运行或历史负证据。

## 10. 证据入口

训练与模型：

```text
D:\仿真\runtime\corner-residual-network-sim-20260811-r3
D:\仿真\models\engines\stage3-training\corner-residual-network-sim-20260811-r3
```

PnP 传播、全分布与图：

```text
D:\仿真\runtime\corner-residual-network-pnp-evidence-20260811-r4
```

其中 `oof_corner_predictions.csv.gz`、`oof_pnp_rows.csv.gz` 和 `paired_deltas_vs_raw.csv.gz` 是逐样本权威；CSV summary、Markdown 和 PNG/PDF 只用于导航，不替代完整分布。精确哈希和失败目录见 `sim_corner_residual_network_registry.json`。
