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

以下字段明确禁止进入模型：truth corner/pose、`range_m`、斜视角、运动模式、session/segment、物理板身份、PnP 输出、未来状态。它们只用于离线分层和评分。完整预注册合同见[机器合同](sim_corner_residual_network_contract.json)。

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

## 10. 为什么按运动场景分组，以及组合运动下游传播

角点网络本身**不使用时序，也不按运动场景选择模型**。每个检测帧独立输入 15 维 raw 四边形几何，独立输出四个角点的 8 维修正量；motion mode、session、segment、角速度、线速度和未来状态均未进入网络。完整 segment/session 分组只用于训练外评估，原因有二：

1. 相邻视频帧高度相关，随机拆帧会让几乎相同的相邻图像同时进入训练集和测试集，夸大泛化结果；
2. 不同运动配置会改变距离、斜视角、板面朝向、可见边缘和图像位置的覆盖。它们不是网络输入，但会改变输入分布，因此必须检查修复规律是否跨采集域成立。

这一区分由两个 OOF 口径同时保留：`network_segment_oof` 表示同采集域内完整 segment 外留；`network_session_oof` 表示整个 combined session 外留、网络只从 spin session 学习。前者回答“同域新轨迹能否改善”，后者回答“未见过的组合运动观测域能否泛化”。部署时若未来通过验证，只会是一个逐帧角点网络，不会先分类平移/旋转/组合运动。

为回答角点改善最终映射到多少组合运动误差，保持 PnP 和阶段 9 冻结的 400 ms 局部 LOS 刚体专家不变，将 `current_refined/raw/network_segment_oof/network_session_oof/exact` 五种同帧角点输入分别传播到 50/100/200 ms 后真值。预测只使用 anchor 及其过去 400 ms；未来真值只在预测生成后评分。主指标为用户定义的 tracker 横向合误差 `sqrt(error_y^2+error_z^2)`，不含 depth。该 atlas 只有 6 个相互独立、每个约 2.2 s 的 combined 配置，因此只能验证局部专家，不能伪造 4 s 反转相位专家结果。

冻结局部专家的全分布摘要如下；`55 mm` 列是诊断覆盖率，不是实弹命中率：

| 角点/PnP 输入 | 时域 | n | mean | P50 | P75 | P90 | <=55 mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current refined | 50 ms | 51 | 41.0 | 11.7 | 43.5 | 110.2 | 80.4% |
| segment OOF network | 50 ms | 51 | 39.4 | 10.7 | 32.1 | 96.4 | 86.3% |
| session OOF network | 50 ms | 51 | 50.5 | 17.8 | 51.3 | 125.8 | 74.5% |
| exact | 50 ms | 51 | 24.6 | 0.0 | 0.0 | 26.4 | 92.2% |
| current refined | 100 ms | 48 | 60.4 | 13.9 | 51.3 | 141.8 | 77.1% |
| segment OOF network | 100 ms | 48 | 56.0 | 11.1 | 34.9 | 106.6 | 85.4% |
| session OOF network | 100 ms | 48 | 67.2 | 17.9 | 58.3 | 147.7 | 70.8% |
| exact | 100 ms | 48 | 43.0 | 0.0 | 0.0 | 47.7 | 89.6% |
| current refined | 200 ms | 47 | 96.4 | 44.2 | 87.8 | 187.8 | 55.3% |
| segment OOF network | 200 ms | 47 | 87.5 | 36.0 | 76.4 | 146.0 | 70.2% |
| session OOF network | 200 ms | 47 | 92.8 | 42.7 | 71.7 | 177.2 | 68.1% |
| exact | 200 ms | 47 | 57.7 | 0.0 | 0.01 | 64.6 | 87.2% |

相对 current refined 的同 anchor 配对结果，segment OOF 网络在 50/100/200 ms 分别有 `64.7/68.8/66.0%` 样本改善，平均横向误差变化为 `-1.6/-4.4/-8.9 mm`。但严格 session OOF 在 50/100 ms 仅有 `43.1/45.8%` 样本改善，平均反而增加 `9.6/6.8 mm`；200 ms 平均减少 `3.6 mm`，仍不足以抵消前两时域的负迁移。因此当前网络只证明了**同采集域内可学习、可传播的收益**，尚未证明对未见 combined 域稳定有效。

exact 输入在大多数 anchor 上接近零，却仍在高角速或换向附近出现巨大尾部；精确角点的 PnP 世界位置 P95 只有 `0.0083 mm`，所以这些未来误差属于局部组合运动模型/相位覆盖，而不是角点或 PnP。结论不能写成“优化角点即可解决组合运动”，也不能把 segment OOF 的较好数字当作部署预期。

## 11. 证据入口

训练与模型：

```text
D:\仿真\runtime\corner-residual-network-sim-20260811-r3
D:\仿真\models\engines\stage3-training\corner-residual-network-sim-20260811-r3
```

PnP 传播、全分布与图：

```text
D:\仿真\runtime\corner-residual-network-pnp-evidence-20260811-r4
```

冻结局部组合运动预测传播、逐预测误差与 ECDF：

```text
D:\仿真\runtime\corner-repair-combined-prediction-20260811-r2
```

其中 `oof_corner_predictions.csv.gz`、`oof_pnp_rows.csv.gz`、`paired_deltas_vs_raw.csv.gz` 和下游的 `prediction_rows.csv.gz` 是逐样本权威；CSV summary、Markdown 和 PNG/PDF 只用于导航，不替代完整分布。精确哈希和失败目录见[机器登记](sim_corner_residual_network_registry.json)。

## 12. 只考虑匀速段后的继续训练审计（2026-08-11）

本阶段按最新研究边界排除平移端点、反向瞬间及其邻域。角点修复器仍是逐帧模型，运动模式、时间、session、速度和未来真值均不进入网络；“匀速段”只约束数据纳入与下游评分。

为了判断能否在不增加数据的情况下抑制跨 session 负迁移，本轮对已有三 seed MLP 增加 nested shrinkage 审计。对每个完整 held session，只使用训练 session 内一个完整 validation segment 选择

```text
mean_correction + alpha * (network_correction - mean_correction)
```

中的 `alpha`，选择目标是 validation 全分布均值而非 P95；outer session 不参与选择。结果为：

| held session | validation 选择的 alpha | raw mean/P50/P90/P95/P99 px | nested mean/P50/P90/P95/P99 px |
| --- | ---: | --- | --- |
| combined-00 | 1.00 | 1.089/0.969/1.798/2.186/3.188 | 1.295/1.177/2.033/2.317/3.074 |
| spin-00 | 0.95 | 1.049/0.952/1.772/2.082/2.676 | 0.688/0.636/1.157/1.334/1.936 |

因此训练 session 内部 validation 无法识别未见 combined 域上的负迁移，简单缩放、ensemble spread 或内部 early stopping 不能把当前网络变成可部署修复器。完整逐检测/逐角点有符号残差、ECDF、直方图和 alpha 曲线保存在：

```text
D:\仿真\runtime\corner-repair-generalization-shrinkage-20260811-r1
```

同时，Windows Release 原图证据钩子已完成三次 smoke，共写入 308 张图和 1,541 条严格同曝光相机姿态。正式 `1.2.1` Release 为 `distribution_locked=true`，1,541 条 SDK 行的 `has_exact_exposure_truth` 均为 true，但目标批次均为空；所以新图无法获得 exact projected corners。不能用截图手工角点、运动指令近似重建或开发模拟器替代正式 Release 来伪造约 1 px 精度的监督。

detector 侧还发现并闭合了一个独立问题：`0526.onnx` 在同一保存原图上正确输出数字 3 候选；原始 FP16 ONNX 直接生成的 `0526` FP16/FP32 TensorRT engine 都发生明显数值退化，不能靠降低阈值收集伪角点。将输入、120 个 FP16 initializer 和中间张量合同显式转换为 FP32 后再生成 FP32 engine，同一原图上 ONNX/TensorRT 选择相同的 row 2441、数字类别 3 和四角点，置信度分别为 `0.868835/0.868896`。转换由[转换脚本](../../../scripts/convert-armor-onnx-fp16-to-fp32.py)无覆盖生成并执行 ONNX Runtime 对照。新的匀速旋转 smoke 在 829 条 Stage3 行中有 817 条包含 detector/PnP 观测，共 1,099 个 armor observation，并保存 212 张严格曝光身份图像；因此消费者原图采集与 detector/PnP 链已经可用。证据分别位于 `D:\仿真\runtime\corner-detector-engine-audit-20260811-r1` 和 `D:\仿真\runtime\stage3-corner-engine-fixed-smoke-20260811-r1`。这只闭合检测输入，仍不提供训练所需的 exact projected corners。

需要模拟器侧新增离线 exact-corner 标签导出后，才能采集多独立 session 并训练图像条件修复器。具体复现、建议接口、版本变化和验收门禁见 [corner_repair_image_training_data_proposal.md](corner_repair_image_training_data_proposal.md)；在用户明确批准该提案前，模拟器仓库、SDK、Release 和版本锁保持不变。
