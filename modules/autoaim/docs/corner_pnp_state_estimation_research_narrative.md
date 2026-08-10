# 自瞄 B：从角点、PnP 到运动状态估计的完整研究叙事

- 日期：2026-08-10
- 当前状态：观测证据、处理链与离线方法消融已经收敛；生产预测器仍暂停
- 本轮新增：真值角点介入、PnP 残差缩放、径向/横向误差分解、四运动简单滤波基线、观测/预测/真值叠加
- 权威证据目录：`D:\仿真\runtime\autoaim-b-hit-oriented-ablation-20260810-r1`
- 机器可读登记：`modules/autoaim/docs/hit_oriented_ablation_registry.json`

这份文档不再按“做过哪些脚本”罗列，而是按研究问题的因果顺序叙述：我们先看到了什么，为什么继续做下一步，尝试了哪些优化，哪些成立、哪些失败，误差怎样从角点传到 PnP、状态估计和未来瞄点，以及当前证据支持什么方法、不支持什么结论。

## 0. 先统一最终评价问题

此前的角度误差和三维位置误差都不够直观，原因正如本轮复核所指出的：

1. 相同角度误差在不同距离对应不同横向脱靶量。
2. 相同三维位置误差若主要沿视线深度方向，对图像瞄准线的影响可以很小；若垂直于视线，则可能直接造成脱靶。
3. “打中装甲板”还取决于板宽高、板面斜角、弹道、系统延迟、枪口散布和控制误差，不能由一个角度或一个位置范数代表。

因此本轮固定报告以下互不替代的量：

| 指标 | 定义 | 直接回答的问题 |
| --- | --- | --- |
| 3D position error | 估计点与真值点的欧氏距离 | 三维状态总体相差多远 |
| LOS radial error | 误差在真值视线方向的投影 | 主要是不是深度错 |
| LOS transverse error | 去掉径向分量后的范数 | 有多少误差真正偏离瞄准线 |
| horizontal/pitch miss | 把估计射线投到真值深度平面后的水平/垂直偏移 | 在当前距离上相当于偏了多少毫米 |
| 55 mm yaw gate | 当前 `FIRE_YAW_MISS_TOLERANCE_M=0.055` | 是否满足现有 yaw 火控门限 |
| small-armor rectangle proxy | 水平不超过 67.5 mm、垂直不超过 27.5 mm | 在乐观正对相机平面上是否落入小装甲板中心矩形 |

最后一个量只是几何代理，不是实弹命中率。它没有计入装甲板斜置后有效面积缩小、弹道、延迟、散布和控制误差。本阶段用它把结果转换为易理解的单位，但不据此宣称“可以开火”。

## 1. 第一阶段：角点观测到底是什么分布

### 1.1 实际生产链

当前生产输入不是一个中心点，而是 detector 输出并规范为 `bl, tl, tr, br` 的四个角点。启用角点精修时，现有链路会在左右灯条局部做亮点提取、PCA 主轴/分位端点估计、`cornerSubPix`，再施加移动上限和软平行约束；失败时回退到原始 detector 角点。随后四点进入 `SOLVEPNP_IPPE`，候选按重投影 RMS 排序并选第一个。

这条 refinement 路径是仓库继承的生产启发式，并不是近期数据重新选出的最优方案。

### 1.2 完整分布与四角不对称

独立角点 atlas 有 4,280 条观测、3,149 个唯一 exposure、2 个 session、12 个运动段；逐角点保留 17,120 条 raw→refined 样本和 34,240 条 raw/refined→truth 残差。完整经验分布保存在：

```text
D:\仿真\runtime\autoaim-b-corner-evidence-complete-20260810
```

四角的 refinement 位移确实不同：

| 角点 | 平均 dx/dy (px) | 位移均值 (px) | 位移 P95 (px) |
| --- | ---: | ---: | ---: |
| `bl` | -0.0887 / -0.2579 | 0.7533 | 3.8971 |
| `tl` | -0.4938 / +0.4776 | 1.0574 | 4.0525 |
| `tr` | -0.3969 / +0.1085 | 1.0781 | 4.4997 |
| `br` | -0.2590 / -0.1919 | 0.8913 | 4.0369 |

这不是四个同分布的噪声点。上、下边的平均 `dy` 方向不同，左右和上下角点的位移幅值也不同，因而会改变板的短边、高度、面积和透视，而不只是把整块板平移。

相对 exact-projection truth，raw/refined 的角点模长残差为：

| 角点 | raw P50/P95 (px) | refined P50/P95 (px) |
| --- | ---: | ---: |
| `bl` | 1.2258 / 3.4606 | 1.1910 / 4.4927 |
| `tl` | 1.1163 / 3.0949 | 1.2356 / 4.2131 |
| `tr` | 1.0796 / 3.1219 | 1.3179 / 4.4009 |
| `br` | 1.1730 / 3.4699 | 1.2374 / 4.5931 |

因此现有 refinement 的真实结论是：`bl` 的中位数略好，但四角 P95 全部变差，`tl/tr/br` 中位数也变差。它对 52.1% 的样本改善、11.2% 基本持平、36.7% 变差；不能用“平均误差略降”或“重投影 RMS 更小”概括。

### 1.3 为什么角点问题自然进入 PnP

对平面目标，四点既决定图像中心，也决定板的透视、高度和短边。受控 1 px 扰动表明：

- common translation 往往只造成毫米级位姿变化；
- short-edge/height 变形会造成数十厘米到米级深度变化；
- 固定 1 px 扰动产生的位置误差与距离 Spearman 相关 `0.952`，与投影尺寸相关 `-0.897`；
- 角点 RMS 与三维位置误差相关性弱，误差模式比单一 RMS 更重要。

这解释了为什么四角必须分别保存，为什么不能只优化中心点，也解释了后续 PnP 长尾为何主要出现在深度方向。

## 2. 角点优化做过什么，效果怎样

### 2.1 当前 PCA/`cornerSubPix` refinement

优点是保留亚像素局部信息，并有移动上限和回退；缺点是对灯条亮度结构、上下端点和短边高度存在方向性偏置。本数据没有证明它优于 raw，尤其尾部变重。因此它目前是“保留的生产基线”，不是“已接受的最优优化”。

### 2.2 exact-corner 真值介入：当前 PnP 到底能不能正确

本轮把同一批 4,280 个 exposure 的角点直接替换为 exact projected corners，仍调用同一 IPPE solver、候选排序和坐标契约。结果为：

| PnP 输入 | 3D P50/P95/P99 | camera-lateral P95 |
| --- | ---: | ---: |
| exact corners | 0.003 / 0.009 / 0.012 mm | <0.001 mm |
| actual raw | 142.9 / 458.3 / 605.2 mm | 21.3 mm |
| actual refined | 140.8 / 500.3 / 708.5 mm | 25.0 mm |

这项消融回答了一个关键问题：如果角点完全正确，当前 PnP 与真值能够闭合到微米级。因此大误差不是“当前 IPPE/坐标接口必然错误”，而是测量角点形状经过平面深度病态放大的结果。它也说明角点仍有很高的优化价值，但优化目标必须是方向敏感的 pose/hit-plane tail，而不是单一重投影 RMS。

### 2.3 固定倾角约束

已尝试利用目标固定倾角减少自由度。raw corners 在限定工作域内约 61.7% 样本改善，refined 只有约 47.0% 改善。它证明“物理约束可以帮助”，但帮助依赖角点误差方向和几何条件，不能直接替换自由 IPPE。

### 2.4 joint PnP、候选切换与单帧修正

- joint yaw/tvec 优化可降低重投影残差，但没有稳定改善时序增量；raw arm 甚至加重尾部，因此未采用。
- 2.2 m 的 31/31 帧和 5 m 的 767/767 帧都选择 solver index 0，没有候选切换；候选间距只有毫米，而真值误差是厘米到米。因此“IPPE 分支跳变导致轨迹翻转”被否定。
- 单帧 constant/linear/quadratic、跨 session affine 和更丰富特征修正都做过。六 session 线性映射曾将 raw P95 从 0.441 m 降到 0.216 m，但条件混杂；leave-one-session-out 丰富特征仍有约 0.517 m P95，没有达到可部署尾部。

这些失败方案保留为证据：以后不能因为重投影残差下降或同 session 拟合很好，就重复宣布 PnP 已修复。

## 3. 第二阶段：PnP 观测实际呈现什么数据特征

### 3.1 先确认坐标契约

相机 `tvec` 使用 OpenCV `right, down, forward` 轴，经过曝光时刻匹配的 camera→gimbal→tracker 链转换。坐标契约审计的 P50/P99 残差约为 `1.77e-10/8.55e-8 m`。因此后续大误差不能继续归因于遗漏刚体变换。

### 3.2 全局三维分布与轨迹结构

56 session 的全轨迹配对有 180,289 行。直接把 selected PnP 当真值位置时，RMSE/P95 为 `0.3649/0.8306 m`；cross-session affine 可到 `0.2203/0.4222 m`；使用真值 phase/slot/rate 的 oracle-conditioned 映射可到 `0.0199/0.0392 m`，但它不可在线使用。

真实匀速旋转在 PnP 中变成相位相关的非均匀速度，最大/最小表观相位速度比约 `1.61–2.45`。不同 session 的曲线相关性 `0.691–0.981`，说明畸变并非纯白噪声，而是既可重复又受条件影响的观测流形。

### 3.3 径向与横向分解改变了结论的可读性

本轮在 77,518 条“已经配对且 truth-visible”的 PnP 行上重新计算方向敏感指标：

| 运动 | 3D/径向误差特征 | LOS transverse P50/P95 | truth-depth 水平偏移 P95 |
| --- | --- | ---: | ---: |
| stationary | 径向 P50/P95 90.1/1005.5 mm | 3.3/5.5 mm | 6.1 mm |
| translation | 径向 P50/P95 84.4/825.3 mm | 2.5/7.3 mm | 6.9 mm |
| rotation | 径向 P50/P95 95.3/812.0 mm | 2.5/9.1 mm | 7.5 mm |
| combined | 径向 P50/P95 108.7/868.3 mm | 2.6/8.3 mm | 7.5 mm |

这就是“深度和横向对命中影响不同”的直接证据：当前配对成功的 PnP 三维长尾几乎完全被深度主导，而射线横向误差小一个到两个数量级。反过来也意味着：以后即使三维范数很大，也不能立刻判定瞄准线不可用；但也不能因此说深度无关，因为深度会影响弹道、目标尺度、关联门限、中心/半径状态和未来运动建模。

上述 6–8 mm 水平 P95 只描述“已配对、可见板”的当前测量，不包含零检测、错关联和不可见板，不是端到端命中结果。

### 3.4 观测轨迹与真实轨迹贴合程度

逐 session/slot 的 ray 轨迹指标为：

| 运动 | yaw-ray correlation 中位数 | PnP/truth ray path length 中位比 | ray-plane 点对点 RMSE 中位数 |
| --- | ---: | ---: | ---: |
| stationary | 真值方差为零，不定义 | 0.00x | 3.7 mm |
| translation | 0.995 | 1.31x | 3.7 mm |
| rotation | 0.999 | 1.03x | 3.3 mm |
| combined | 0.999 | 1.04x | 3.7 mm |

高相关不等于三维位置正确：它只说明配对可见弧的射线形状贴合。相机 `x-depth` 叠加图直观看到 truth 曲线与 PnP 深度点云大幅分离，而投到 truth-depth 瞄准平面后又明显接近。两种图都必须保留，否则只看其中一张会得到相反且片面的印象。

## 4. PnP 能优化到什么程度，才进入状态估计允许范围

### 4.1 真值替换与残差缩放设计

对每个历史观测时刻构造：

```text
p_alpha = truth + alpha * (current_pnp - truth)
```

- `alpha=1`：当前 PnP；
- `alpha=0`：相同观测时刻、相同缺失、相同可见弧下的 exact PnP 上界；
- 中间保留 `0.75/0.5/0.25/0.1`，分别模拟不同程度的 PnP 改善。

缩放不改变时间戳、可见性、历史长度和 offline slot，只改变 PnP 残差幅度。因此 `alpha=1` 与 `alpha=0` 的差是测量误差贡献，`alpha=0` 仍剩下的误差是运动模型、历史窗口、未来时域和可见弧贡献。

### 4.2 简单滤波器基线

为了检验“平移是否根本不需要复杂预测器”，使用了一个故意简单的基线：最近 16 个曝光匹配样本、真实事件时间、camera XYZ 上的普通最小二乘恒速模型；历史跨度不超过 0.75 s、相邻 gap 不超过 120 ms，评估 0/50/100/200 ms。物理 slot 来自 offline truth，只用于隔离测量/模型误差，不能部署。

### 4.3 四类运动的核心结果

下面是 simple CV 的 truth-depth 水平偏移 P95：

| 运动 | 50 ms 当前/真值 PnP | 100 ms 当前/真值 PnP | 200 ms 当前/真值 PnP |
| --- | ---: | ---: | ---: |
| stationary | 6.1 / 0.0 mm | 6.1 / 0.0 mm | 6.1 / 0.0 mm |
| translation | 11.9 / 9.8 mm | 15.4 / 12.9 mm | 25.2 / 23.5 mm |
| rotation | 12.5 / 8.4 mm | 20.3 / 10.0 mm | 37.2 / 14.2 mm |
| combined | 56.1 / 49.2 mm (n=203) | 110.6 / 110.5 mm (n=34) | 183.8 / 97.7 mm (n=13) |

与当前 55 mm yaw 门限对照：

1. stationary、translation 和当前受支持可见弧内的 rotation，在这项 oracle-identity P95 实验中当前 PnP 已过门限。
2. combined 50 ms 略超门限；离散缩放网格表明至少需要约 25% 的 PnP 残差下降才进入 P95 门限。
3. combined 100/200 ms 即使 `alpha=0` 仍失败，因此继续优化角点/PnP 不能单独解决，需要改变运动模型、历史/相位表达或目标假设。
4. combined 长时域样本只有 34/13 条，说明高角速、可见窗口和未来覆盖本身就是问题；这些数值是明确的失败信号，但不能被当作精确总体通过率。

### 4.4 “简单滤波就能预测平移”是否成立

在当前 PnP 输入下，translation 的 hold P95 为 `11.7/14.6/22.3 mm`，simple CV 为 `11.9/15.4/25.2 mm`。两者都在 55 mm 内，CV 并没有在 P95 上稳定优于 hold；边界往返运动的反向点会使局部恒速外推吃亏。

因此准确结论不是“CV 一定优于所有方法”，而是“平移在这批速度 0.4/0.8/1.2 m/s、受支持时域和 oracle 身份下，简单基线已经达到当前横向门限，不是当前最主要困难”。这正适合作为后续方法的最低基线。

rotation 则不同：hold P95 为 `59.0/83.1/156.7 mm`，simple CV 降到 `12.5/20.3/37.2 mm`。只要身份和可见弧已知，局部速度信息对旋转有显著价值。但 4 rad/s、200 ms 只有 52 条，6 rad/s 没有合格的 200 ms 样本；不能把总体 P95 解读成全速域已经解决。

combined 是真正的难点：同时存在中心平移、相位旋转、弧段可见性、历史曲率和身份切换，单一局部直线模型会把两种运动混在一起。

### 4.5 深度进入状态估计后仍是长尾

虽然横向瞄准误差较小，simple CV 在当前 PnP 上的径向 P95 仍然很大：translation 约 `0.78/0.88/1.14 m`，rotation 约 `0.68/0.85/1.45 m`，combined 在少量 100/200 ms 样本上超过 `1.2/1.8 m`。因此当前证据支持用 camera-ray `u/v` 做第一版可观测状态，却不支持把 raw depth 直接融合成高置信车辆中心/半径状态。

## 5. 时序数据为何要求先做观测器，再做预测器

120 轮接受矩阵中：

- truth frame 189,158；observation frame 184,879；有效事件 177,483；候选 250,449；
- 典型事件间隔约 8 ms，但 P95 约 23–24 ms，存在数百毫秒 missing streak；
- combined 有两轮完全零 observation，最长失败段约 15.02 s；
- candidate count 每帧可为 0、1、2、3、4，formal 360 中还有 38 帧超过 4；
- `observation_index`、detector number 和当前 `tracked_id` 都不是可靠物理身份；
- 角向误差 lag-1 自相关约 0.831（spin）和 0.741（combined），不满足固定独立高斯噪声假设。

因此第一版观测器合同使用匿名 camera-ray 状态 `[u,v,du/dt,dv/dt]`，按真实事件时间更新，depth 单独保留并降权，缺失/歧义/过期显式 fail-closed。先把 physical identity、时间和不确定性做对，再恢复预测器。

## 6. 当前数据特征分别支持哪些方法

### 6.1 平移：CV、α-β 或标准 CV-KF

适用特征：短时局部近线性、简单 hold/CV 已进入横向门限。应使用真实 `dt`，在往返反向点增加加速度/变点检测，不要把 6 ms 固定帧周期写死。深度 R 必须与角向 R 分离。

### 6.2 旋转：局部 CV 是下界，周期/协调转弯是候选而非既定答案

旋转在已知身份和可见弧内由速度信息显著改善，但 PnP 会把恒定真值角速映成相位非均匀的观测速度。可尝试：

- `u/v` 局部 Ridge/CV 作为保守短时基线；
- 显式 phase 的正弦/谐波或 coordinated-turn 模型；
- 对旋转方向、角速和当前相位保持多假设，而不是一次硬选。

已有 periodic EKF/UKF 在 120 轮分组测试中明显差于 CV+Ridge；原因包括错误身份、观测流形畸变和参数失配。因此“物理上周期”不等于“现有周期 EKF 可直接使用”。

### 6.3 组合运动：因子化或 IMM/多模型

适用特征：中心平移与板相位旋转同时存在，exact PnP 仍不能救 100/200 ms simple CV。优先方向是：

1. 将车辆中心平移状态与装甲板相位/半径状态因子化；
2. 维护 CV/CA 与旋转/协调转弯的 IMM 或多假设；
3. 把 physical identity 当独立离散变量，不让错误 slot 更新连续状态；
4. 对可见弧结束和 reacquisition 显式降级，不能跨 gap 继承旧身份；
5. 用横向毫米、板面代理和覆盖率评价，不再只看角度或三维 L2。

### 6.4 学习方法的边界

oracle identity 下，CV+`u/v` Ridge 的分组 P95 在 50/100/200 ms 约 `0.214/0.247/0.345 deg`（leave-distance-out），优于简单 CV Kalman；MLP motion-transfer 更差。学习残差可作为局部观测流形补偿，但必须 complete-run/repeat holdout，且不能用 truth phase、slot 或未来真值作为在线特征。

## 7. 当前最可信的结论与下一步缺口

### 已经能说

1. exact corners 可使当前 IPPE PnP 闭合，角点是三维深度长尾的主要上游来源之一。
2. 当前 refinement 不是已证最优；它略改善中位数但恶化尾部，四角必须分开建模。
3. 当前配对可见板的 PnP 三维误差主要沿深度，横向射线误差小得多。
4. 平移在当前数据条件下由简单基线即可达到 55 mm 横向门限；rotation 需要速度信息，但支持弧内 local CV 已明显改善。
5. combined 100/200 ms 的主要限制已经越过 PnP：即使 exact PnP，简单恒速模型也失败。
6. physical identity、可见弧覆盖和 missingness 仍未解决；任何 oracle-identity 预测指标都不是部署结果。

### 仍不能说

1. 不能说当前横向代理通过就能实弹命中。
2. 不能说 rotation 已解决；高角速长时域覆盖不足，非 oracle association 仍会产生灾难性尾部。
3. 不能说 depth 无用；它对弹道、中心/半径、关联和三维状态仍重要，只是不能与横向误差混成一个指标。
4. 不能说 25% PnP 改善是普适阈值；它只针对 combined 4 rad/s、50 ms、当前离散缩放网格和 P95 yaw gate。
5. 不能把 rectangle proxy 当命中率，也不能把 55 mm yaw gate 当完整二维容差。

### 后续若恢复预测器，优先补的消融

1. 真实板面坐标命中代理：利用未来装甲板 normal/orientation，把误差投到装甲板自身宽高轴，而不是正对相机平面。
2. 弹道与系统延迟：将观测/状态误差传到真正 impact time，加入枪口散布与控制误差预算。
3. 非 oracle association：同一套 alpha 消融分别在 oracle、多假设、当前 hard association 下重放，拆出身份错误贡献。
4. 高角速/长时域补采：尤其 combined 100/200 ms 和 spin 6 rad/s 200 ms，先解决样本覆盖再选模型。
5. 按距离、速度、入射角、候选数、gap 和角点质量做完整 coverage 校准，不能只报总体 P95。

## 8. 图与完整分布索引

所有图同时保存 PNG/SVG/PDF，所有判断可回读逐样本表：

```text
D:\仿真\runtime\autoaim-b-hit-oriented-ablation-20260810-r1\corner_exact_raw_refined_ecdf.*
D:\仿真\runtime\autoaim-b-hit-oriented-ablation-20260810-r1\pnp_radial_transverse_ecdf_by_motion_distance.*
D:\仿真\runtime\autoaim-b-hit-oriented-ablation-20260810-r1\trajectory_overlay_observed_truth_camera_xz.*
D:\仿真\runtime\autoaim-b-hit-oriented-ablation-20260810-r1\trajectory_overlay_prediction_observation_truth.*
D:\仿真\runtime\autoaim-b-hit-oriented-ablation-20260810-r1\prediction_p95_vs_pnp_residual_scale.*
D:\仿真\runtime\autoaim-b-hit-oriented-ablation-20260810-r1\prediction_motion_current_vs_exact_pnp.*
```

逐样本完整分布：

```text
corner_to_pnp_samples.csv.gz               12,840 rows
pnp_directional_samples.csv.gz            180,289 rows
prediction_ablation_samples.csv.gz        204,768 rows
trajectory_fit_metrics.csv                150 session/slot groups
representative_prediction_trajectories.csv
```

`*_metrics.csv`、`summary.json` 和本文中的分位数仅用于导航，不替代逐样本分布。`retention_manifest.json` 对 29 个产物和两项源证据保存了大小与 SHA-256。
