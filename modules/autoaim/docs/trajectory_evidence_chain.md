# 自瞄 B 轨迹研究证据链

- 当前阶段：`1 / 四角点观测 -> PnP 输入`
- 状态：`已复核，有明确证据边界`
- 日期：2026-08-09
- 研究状态：预测器暂停；本文件只整理已有数据、代码与证据，不授权新的预测器训练或在线接入。
- 数据保留：原始采集和派生报告位于 `D:\仿真\runtime`，属于受保护证据；Git 只保存处理代码、字段契约、证据索引和哈希，不复制大体积原始数据。

本文按数据真正经过的顺序逐阶段收口。被后续证据推翻的分析不删除，而是在对应阶段标为“被取代”；最终结论必须能追到原始输入、处理脚本、输出报告和哈希。

## 阶段 1：四角点观测到 PnP 输入

### 1.1 当前实际方案

当前生产链路不是直接把网络角点送入 PnP，而是：

1. TensorRT detector 为每个候选输出四个二维关键点、目标分数、数字类别和颜色类别。
2. 后处理把四点规范为 `bl, tl, tr, br`，即左下、左上、右上、右下；同时生成 detector 中心点和帧内 `observation_id`。
3. `AngleSolver` 在进入 PnP 前保留一份原始 detector 四角点，并在灰度图可用且配置开启时执行角点精修。
4. 精修先在左右灯条局部区域提取亮点，以 PCA 主轴和 5%/95% 分位重估端点，再执行 OpenCV `cornerSubPix`。精修点相对本轮中间估计移动小于 3 px才接受；左右灯条近似平行时施加 0.2 权重的软平行约束；相对原始 detector 角点移动超过配置上限 5 px时回退原角点。
5. 平面 PnP 使用精修后的四点和每帧有效相机内参/畸变。`solvePnPGeneric(..., SOLVEPNP_IPPE)` 产生候选，按重投影 RMS 排序，当前生产结果取排序后的第一个候选。
6. detector 中心点没有在精修后重算；它仍是网络候选中心。后续轨迹研究使用的是 PnP 位置派生的观测 `u/v`，不是把 detector 中心点当成三维位置。

对应实现：

- detector 关键点解码与规范顺序：`ArmorDetector/mt_detector_tensorrt.cpp`
- 精修和回退：`AngleSolver/AngleSolver.cpp::refineKeypoints`
- PnP 候选与选择：`AngleSolver/AngleSolver.cpp::solvePlanarPnPCandidates`
- full-pipeline 诊断字段：`aim_core_bridge/vivsionn_pipeline.cpp`
- Stage3 正式观测字段：`aim_core_bridge/stage3_capture.cpp`

当前模拟器参数启用精修：

```text
SUBPIXEL_REFINE_KEYPOINTS = 1
SUBPIXEL_REFINE_THRESHOLD = 140
SUBPIXEL_REFINE_MAX_MOVE_PX = 5.0
SUBPIXEL_REFINE_ROI_WIDTH_RATIO = 0.55
SUBPIXEL_REFINE_PARALLEL_LIGHTBAR_CONSTRAINT = 1
```

### 1.2 字段与保存边界

| 层级 | 字段 | 语义 | 是否进入正式 Stage3 原始流 |
| --- | --- | --- | --- |
| detector | `detector.candidates[].raw_corners_px` | 网络后处理得到的四角点，顺序 `bl,tl,tr,br` | 否，仅 full-pipeline 诊断流 |
| detector | `center_px` | detector 候选中心，精修后不重算 | 否，仅 full-pipeline 诊断流 |
| detector | `observation_id` | 帧内索引，只用于把 detector、PnP 结果连接起来 | 间接保存为 `observation_index`；禁止跨帧当身份 |
| PnP 输入 | `solved_armors[].pnp_vertices_px` | 经过精修/回退、真正送入 PnP 的四角点 | 否，仅 full-pipeline 诊断流 |
| PnP 诊断 | `pnp_candidates[]` | IPPE 候选、重投影误差、rvec/tvec 和选择标记 | 否，仅 full-pipeline 诊断流 |
| Stage3 | `armors[].position_m/camera_tvec_m/yaw_*` | PnP 后观测结果 | 是，`stage3-observation-v2` |
| 不确定性 | `corner_covariance_px2` | 角点协方差 | 当前为 `null/unavailable`，正式流也没有 |

因此，120 轮正式 method-selection 矩阵能证明 PnP 后观测轨迹的重复性和误差结构，但不能回溯每个样本的原始/精修角点。角点级证据只能由专门保留的 full-pipeline 诊断采集支持，不能把两者混称为同一层证据。

### 1.3 已接受证据

#### A. 距离相关的角点条件数

2026-07-19 的 3/5/7 m 联合 PnP A/B 表明，平移边缘化后的局部 yaw 灵敏度 P50 为 `3.73/5.33/6.58 deg/px`，P95 为 `8.93/12.50/12.92 deg/px`。这说明远距离、斜视平面位姿下，一像素角点扰动足以对应数度 yaw 变化。

联合重估 yaw+tvec 只小幅降低重投影误差，时间增量没有一致改善；原始角点分支虽然能进一步压低残差，却使时间尾部更差。因此，“重投影误差更小”不能单独作为角点精修或 PnP 修复的验收依据，联合求解器也没有替换生产路径。

这组记录使用历史 Simulator 1.0.1/SDK 1.0.0 和旧 WSL 复现命令。它仍是已提交的历史条件数证据，但其采集命令已被当前 Windows-native 消费者契约取代，不能作为新采集模板。

#### B. 当前 full-pipeline 角点诊断

受保护输入：

```text
D:\仿真\runtime\autoaim-b-arcflip-diag-20260809T013519Z\r0p75-d2p2-rep01\pipeline.jsonl
D:\仿真\runtime\autoaim-b-arcflip-diag-20260809T013519Z\r0p75-d2p2-rep01\truth.jsonl
```

当前可复现输出：

```text
D:\仿真\runtime\autoaim-b-arcflip-diag-20260809T013519Z\arc-flip-analysis-v3-corner-audit
```

该诊断在 `31` 个 truth 匹配的 target-3 观测上得到：

- 原始角点到 PnP 输入角点的平均移动 P50/P90/最大值为 `1.072/2.327/2.453 px`；单角点最大移动的 P50/P90/最大值为 `3.464/4.324/4.760 px`。
- 原始和 PnP 输入角点的协方差均未提供。
- 修正后的 truth 相机投影与 detector 中心误差中位数为 `0.978 px`，错误镜像相机基为 `91.733 px`；这验证了相机基修正。
- IPPE 选择索引在 31 个匹配观测中始终为 0，顺序切换为 0；两个候选 tvec 相距中位数 `3.012 mm`、最大 `11.205 mm`。
- 选中 tvec 相对 truth 的位置误差中位数/P90 为 `0.054/0.242 m`。候选间距离远小于选中位姿的 truth 误差，因此该条件下的轨迹翻转不能归因于 IPPE 候选切换。

这是一组 `r=0.75, d=2.2 m` 的有界诊断，不代表 120 轮矩阵的角点统计，也不能证明当前精修必然优于原始角点。

#### C. 5 m 前端边界

5 m full-pipeline A/B 中，非空帧的 detector 数和 PnP solved 数一致，PnP 拒绝为 0；把研究阈值从 0.50 降到 0.25主要提高了 detector admission coverage。后续 5 m 证据仍显示明显的 PnP 深度偏差和观测/truth 曲率不一致。

所以：

- “PnP 没有拒绝”只说明求解得到有限解，不等于解准确。
- 0.25 是这批 manifest-bound 研究数据的采集阈值，生产默认仍是 0.50。
- 低置信度会加重已有误差，但没有证据表明它制造了 IPPE 分支切换。

### 1.4 本阶段结论

1. 四角点顺序、帧内关联和 PnP 输入链路已经明确；没有发现角点顺序随机翻转或 IPPE 候选跳变证据。
2. 当前主要不确定性边界是 detector 角点定位、启发式精修、平面深度条件数，以及缺失的逐角点协方差。
3. 已有证据不足以把误差唯一归因于 detector、精修或 PnP 中的某一个；因此本轮不改 detector/PnP，也不宣称现有精修最优。
4. 后续数据处理把 PnP `u/v` 视为带系统偏差、重尾、会缺失的观测，而不是精确物理圆弧；这正是后续选择因果残差处理和安全回退的依据之一。

### 1.5 仍有的优化空间

以下是根据现有文档确认的缺口，不是重新发起预测器研究：

1. `stage3-observation-v2` 未保存 raw/refined corners、每角点移动、置信度和角点协方差。现有 120 轮数据不能离线补回这些字段。若未来确实要重采，建议升级观测 schema 并同时保存两套角点；当前不需要为此重采。
2. full-pipeline 诊断只有一个受控条件的 31 个 truth 匹配样本。可以用现有 full-pipeline 数据继续做有限复核，但不能外推到全部距离、半径和运动类型。
3. 精修算法包含亮度阈值、PCA 分位端点、`cornerSubPix`、软平行约束和最大移动回退，尚无逐模块消融证据。以后若研究观测器前端，应使用完整开关 A/B 和 truth-only 评分，不能只比较重投影残差。
4. `corner_covariance_px2` 当前明确为 unavailable。若未来要做异方差滤波或观测置信度，必须先定义可校准的不确定性来源；不能把 detector objectness 直接冒充位置概率。
5. 5 m 阈值 0.25提高可用率但不是生产优化结论。若部署需要改阈值，必须另做吞吐、误检、重捕获和真实数据域验收。

### 1.6 本阶段证据哈希

`arc-flip-analysis-v3-corner-audit/retention_manifest.json` 固化了以下对象：

| 对象 | SHA-256 |
| --- | --- |
| `pipeline.jsonl` | `8bce3f68da7c13b5a5340051a86225bfed832732f3d914e42694b9db8c5a55b8` |
| `truth.jsonl` | `d1afd26bc8e96abcab81728ee55baed1450f18e81867abbdd9ef7d3c57237545` |
| `scripts/analyze-pnp-arc-flip.py` | `e4102dd07b94f56c2e48ad309466d1d816d97b5533067066a83b9d14adbb5b95` |
| `scripts/analyze-stage3-truth-grid.py` | `d4457e85ceb5cd9020eca42753fd27892de64f064d7ee73624d29d9a9f65e90b` |
| `pnp_arc_flip_diagnostics.jsonl` | `f1875484caac6a3a23b256e8fe5135a7f6a76924b64e6a7f3f5f4371ed6885cd` |
| `pnp_arc_flip_summary.json` | `fd7bf4637ff841a7685367eb6f6c2f5209eb978e5b8551f3f39433fdea1efc0a` |
| `PNP_ARC_FLIP_REPORT.md` | `dd4b2c760ee37db4bd222b86d1d5fdea0742e477b2a20e2d495297c4e5e569bb` |

## 下一步

下一阶段从“PnP 输入四角点”继续到“相机 `tvec`、camera->gimbal/tracker 坐标、观测 `u/v` 和 exact-exposure truth join”，复核坐标语义、候选选择、时间键和哪些 truth 字段只允许用于离线标签/评分。
