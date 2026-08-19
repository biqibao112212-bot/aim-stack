# Linux 1.3.1 图像角点修复正式实验

## 1. 当前结论

本轮完成了 Linux `daedalus-simulator 1.3.1` 原始 RGBA 全帧采集、冻结检测器推理、完整 session 切分、图像角点修复训练和两轮封存测试。最重要的结论不是某个单一网络分数，而是把“检测覆盖、检测到真值的离线关联、同一目标的角点修复”三个问题分开了。

最终候选 `v8-match25-score055` 使用：

- 检测器原始四边形向外扩展 `1.5x` 后透视到 `128x64` RGB ROI；
- 保留 `4x8` 空间特征图的 CNN，而不是全局平均池化；
- CNN 特征与 15 维 raw-corner 几何特征经 MLP 输出四角归一化残差；
- 输出先在 ROI 坐标中表示，再用每帧逆透视矩阵还原到全图像素；
- detector-to-truth 离线训练关联必须满足角点坐标 RMS `<=25 px`；
- 在线 detector score `<0.55` 时不应用网络，直接返回 raw corners；
- exact corners 只用于离线监督和验收，不进入模型输入、PnP 或任何在线字段。

它在第一次干净 session-disjoint validation 上总体从 `8.0742 px` 降到 `7.6187 px`，改善 `5.64%`，四种运动模式均无退化，因此通过预声明 validation gate。第二轮全新封存测试总体从 `7.4827 px` 降到 `7.2917 px`，改善 `2.55%`；四种模式仍均无退化，但未达到预声明的 `5%` test gate。故当前状态是：**安全、可回退的研究候选，未取得 5% 强门的正式晋级资格**。

受保护 checkpoint：

```text
/home/potato/Projects/仿真/models/engines/corner-repair-formal-linux-1.3.1-20260819-v8-match25-score055/corner-repair.pt
SHA-256 53e0cba9e2375a206b56eb16d1120edfe7e163763256828da7f51e5fe693ca27
```

## 2. 图像是怎样处理的

每个输入样本仍是一张独立图像，不含相邻帧或运动状态。冻结 detector 给出 `bl,tl,tr,br` 四点后：

1. 按 `tl,bl,br,tr` 构造透视四边形；
2. 以四边形中心为基准扩大 `1.5x`，保留灯条端点外侧上下文；
3. 从 Release 的字面 `RGBA32` 原始帧读取像素，使用 `RGBA -> BGR -> RGB`，不能按 OpenCV PNG 的 `BGRA` 解释；
4. 透视为 `128x64`、三通道、`[0,1]` 浮点张量；
5. CNN 逐层提取局部边缘、亮条和端点组合，最后保留 `4x8` 空间布局；
6. MLP 将空间视觉特征和 raw quadrilateral 的 15 维几何量组合为 8 个数，即四角各自的二维修正；
7. 8 个数表示 ROI 宽高比例，而非固定全图 px，最后逆透视回原图。

这里 CNN 负责“看见什么、在什么位置”，MLP 负责“把这些视觉和几何证据组合成 8 个连续输出”。每层卷积的具体语义不是设计者预先手写的；训练通过 exact-vs-raw 残差损失反向传播，自动更新滤镜权重。

## 3. 为什么运动 session 会影响单帧网络

网络输入虽不含时序，但运动决定采集到哪些单帧视角。短 session 的慢旋转只覆盖部分 yaw；若随机初始 yaw 恰好侧对相机，整段可能几乎没有可靠检测。第一批 session 的 detector exposure coverage 从 `0.4%` 到约 `100%`，不是运动变量进入了 CNN，而是运动轨迹改变了图像分布。

补充高速整圈旋转后，四个 session 的覆盖仍为约 `3.2%/92.0%/92.2%/8.8%`，进一步说明初始姿态、旋转方向、距离和可见装甲面共同决定 detector applicability。报告训练误差时必须同时报告覆盖率，不能只在成功检测的少数帧上宣称泛化。

## 4. 失败方法与原因

### 4.1 RGBA 被当成 BGRA

Release 导出的是字面 RGBA32。错误的 BGRA 转换会交换红蓝通道。修正后 100 帧 A/B 中 detector mean corner RMS 从约 `3.19 px` 降到 `1.93 px`，置信度从约 `0.696` 升到 `0.771`。该修复已在提交 `c33479d` 中保留。

### 4.2 无上下文 ROI + 全局平均池化

早期 ROI 把 raw 四点正好拉到图像边缘，向外的真实端点可能被裁掉；全局平均池化又弱化了“端点在左上还是右下”的空间信息。扩大 ROI 并保留空间网格后，验证总体改善从约 `10.5%` 提升到 `17.4%`，静止退化也明显缩小。

### 4.3 训练分布缺少低误差样本

第一版训练中 stationary 只有 1 行可用，网络几乎只见过“必须大修”的样本，导致验证 stationary 从 `7.67 px` 被修坏到 `25.44 px`。补采三组覆盖率 `90.6%--95.7%`、raw RMS `7.32--8.65 px` 的 stationary train session 后，退化大幅收缩。这证明失败来自训练分布缺口，而非“运动模式必须作为输入”。

### 4.4 detector score 不是角点误差

在错误关联数据上，简单 `score>=0.70` 门控仍会修坏 stationary。objectness 只说明“像不像装甲板”，不能直接冒充角点位置不确定性。最终 `0.55` 只用于拒绝干净 match-25 数据中的低质量检测，不能解释为校准后的像素概率。

### 4.5 80 px 离线关联门制造了不可能监督

最严重的根因是早期 builder 使用 `--match-rms-px 80`。当 3 号车漏检时，邻车检测框会被分给 3 号车 exact label；最大失败样本的 raw 在画面左车、exact 在右车，相距约 `90 px`。网络被要求从错误车辆 ROI 输出另一辆车角点，当然不能泛化。

关联门收紧到 `25 px` 后，同一个 combined validation session 仍有 349 行，但关联 RMS 整体从约 `45.34 px` 降到 `9.30 px`。这说明数据行数不等于监督质量。旧 CSV、checkpoint 和失败 test 均保留，未覆盖或删除。

## 5. 最终验证与测试

### Validation（通过）

| 模式 | raw RMS px | repaired RMS px | 相对变化 |
| --- | ---: | ---: | ---: |
| aggregate | 8.0742 | 7.6187 | +5.64% |
| stationary | 3.7942 | 3.7648 | +0.78% |
| linear | 18.3622 | 18.3622 | 0.00%（低置信度回退） |
| spin | 10.2553 | 9.8109 | +4.33% |
| linear_and_spin | 10.1313 | 9.2968 | +8.24% |

### 第二轮 sealed test（未通过 5% 强门）

| 模式 | raw RMS px | repaired RMS px | 相对变化 |
| --- | ---: | ---: | ---: |
| aggregate | 7.4827 | 7.2917 | +2.55% |
| stationary | 2.8631 | 2.7988 | +2.24% |
| linear | 15.1120 | 15.1120 | 0.00%（仅 2 行，回退） |
| spin | 10.2872 | 9.9293 | +3.48% |
| linear_and_spin | 8.6193 | 8.5188 | +1.17% |

测试 p95 从 `26.5310 px` 降到 `25.9755 px`，但 p50 从 `2.4094 px` 升到 `2.6977 px`。当前网络主要压低长尾，仍会给部分本来很准的样本带来无益小改动，这是下一轮研究应解决的重点。

## 6. 复现入口和受保护证据

源码入口：

- `scripts/collect-linux-corner-repair-matrix.py`：只编排 Release simulator、Release collector、公共 SDK CLI 和 Release validator；不手写协议。
- `scripts/detect-linux-corner-repair-rows.py`：冻结 detector 推理和 truth-only 离线关联。
- `scripts/build-linux-corner-repair-dataset.py`：哈希绑定、match gate 和 test authorization。
- `training/stage3/train_image_corner_repair_formal.py`：session-disjoint 训练与 validation gate。
- `training/stage3/evaluate_image_corner_repair_formal.py`：冻结 checkpoint 的一次性 test 评估。

最终 validation/test：

```text
/home/potato/Projects/仿真/models/engines/corner-repair-formal-linux-1.3.1-20260819-v8-match25-score055/validation-result.json
/home/potato/Projects/仿真/models/engines/corner-repair-formal-linux-1.3.1-20260819-v8-match25-score055/test-result.json
```

最终 test 原始 session：

```text
/home/potato/Projects/仿真/runtime/corner-repair-final-test-v1-20260819
```

这些 raw frames、labels、CSV、checkpoint 和测试结果均为受保护资产。不得自动删除、覆盖或混入 Git。

## 7. 当前工程边界

自瞄 B 已有生产 PCA/`cornerSubPix` 精修和 5 px 回退路径。v8 尚未通过 5% 强门，也没有 Linux C++ 推理依赖与延迟验收，因此本轮不直接替换生产精修器。可将它作为离线候选和后续 PnP/hit-oriented 消融输入；任何在线接入都必须保留 raw/PCA/v8 三臂诊断、严格回退和坐标契约测试。
