# 自瞄 B 轨迹研究证据链

- 当前阶段：`1 / 四角点观测 -> PnP 输入`
- 状态：`已补全逐样本分布和历史负证据，等待阶段验收`
- 日期：2026-08-10
- 研究状态：预测器暂停；本文件只整理已有数据、处理过程、方案依据、负证据和优化边界，不授权新的预测器训练或在线接入。
- 数据保留：原始采集和大体积派生证据位于 `D:\仿真\dataset`、`D:\仿真\runtime`，均按受保护资产处理；Git 保存处理代码、字段契约、证据登记和哈希，不复制大体积数据。

本文按数据真正经过的顺序逐阶段收口。均值、中位数和分位数只能用于导航，不能代替完整分布；被后续证据推翻、未采用或失败的方案不删除，而是保留并标明结论边界。机器可读登记见 `modules/autoaim/docs/corner_evidence_registry.json`。

## 阶段 1：四角点观测到 PnP 输入

### 1.1 当前实际方案

当前生产链路不是直接把网络角点送入 PnP，而是：

1. TensorRT detector 为每个候选输出四个二维关键点、目标分数、数字类别和颜色类别。
2. 后处理把四点规范为 `bl, tl, tr, br`，即左下、左上、右上、右下；同时生成 detector 中心点和帧内 `observation_id`。
3. `AngleSolver` 在进入 PnP 前保留原始 detector 四角点，并在灰度图可用且配置开启时执行角点精修。
4. 精修在左右灯条局部区域提取亮点，以 PCA 主轴和 5%/95% 分位重估端点，再执行 OpenCV `cornerSubPix`。精修点相对本轮中间估计移动小于 3 px 才接受；左右灯条近似平行时施加 0.2 权重的软平行约束；相对原始 detector 角点移动超过配置上限 5 px 时回退原角点。
5. 平面 PnP 使用精修/回退后的四点和每帧有效相机内参、畸变。`solvePnPGeneric(..., SOLVEPNP_IPPE)` 产生候选，按重投影 RMS 排序，当前生产结果取排序后的第一个候选。
6. detector 中心点没有在精修后重算；后续轨迹研究使用 PnP 位置派生的观测 `u/v`，不是把 detector 中心点当作三维位置。

对应实现：

- detector 解码和规范顺序：`ArmorDetector/mt_detector_tensorrt.cpp`
- 精修和回退：`AngleSolver/AngleSolver.cpp::refineKeypoints`
- PnP 候选与选择：`AngleSolver/AngleSolver.cpp::solvePlanarPnPCandidates`
- full-pipeline 诊断字段：`aim_core_bridge/vivsionn_pipeline.cpp`
- Stage3 观测字段：`aim_core_bridge/stage3_capture.cpp`

当前模拟器消费者配置启用精修：

```text
SUBPIXEL_REFINE_KEYPOINTS = 1
SUBPIXEL_REFINE_THRESHOLD = 140
SUBPIXEL_REFINE_MAX_MOVE_PX = 5.0
SUBPIXEL_REFINE_ROI_WIDTH_RATIO = 0.55
SUBPIXEL_REFINE_PARALLEL_LIGHTBAR_CONSTRAINT = 1
```

这条 PCA/`cornerSubPix` 路径是初始消费者仓库已经继承的生产启发式，不是近期 A/B 重新选出的最优方案。保留它不等于已有证据证明它优于 raw corners。

### 1.2 字段、数据层级和处理过程

| 层级 | 字段 | 语义 | 保存边界 |
| --- | --- | --- | --- |
| detector | `raw_corners_px` | 网络后处理四角点，顺序 `bl,tl,tr,br` | full-pipeline 和专门 Stage3 v3 采集 |
| detector | `center_px` | detector 候选中心，精修后不重算 | full-pipeline 诊断 |
| detector | `observation_id` | 帧内索引 | 禁止跨帧当物理身份 |
| PnP 输入 | `pnp_vertices_px` / refined corners | 精修/回退后真正送入 PnP 的四点 | full-pipeline 和专门 Stage3 v3 采集 |
| PnP 诊断 | `pnp_candidates[]` | IPPE 候选、误差、rvec/tvec、选择标记 | full-pipeline 诊断 |
| Stage3 v2 | `position_m/camera_tvec_m/yaw_*` | PnP 后正式观测 | 120 轮 method-selection 数据；不含角点 |
| 不确定性 | `corner_covariance_px2` | 逐角点协方差 | 当前为 `null/unavailable` |

需要同时保留两条不同证据链：

1. 120 轮正式 Stage3 v2 矩阵能证明 PnP 后轨迹的重复性和误差结构，但没有 raw/refined corners，不能离线补回逐角点分布。
2. 专门的 Stage3 observation v3 independent 采集确实保存了 raw/refined corners。两次会话的 `observations.jsonl` 和 exact-exposure `truth.jsonl` 经过离线 truth 关联、精确角点投影和屏幕坐标规范化，生成 4,280 行 atlas；truth 只用于关联、投影和评分，不进入 detector、精修或 PnP。

v3 输入是：

```text
D:\仿真\dataset\autoaim-stage3-v1\stage3-observation-v3-independent-20260803-r1-spin-00\run-20260803T063052053Z\observations.jsonl
D:\仿真\dataset\autoaim-stage3-v1\stage3-observation-v3-independent-20260803-r1-spin-00\run-20260803T063052053Z\truth.jsonl
D:\仿真\dataset\autoaim-stage3-v1\stage3-observation-v3-independent-20260803-r1-combined-00\run-20260803T063155798Z\observations.jsonl
D:\仿真\dataset\autoaim-stage3-v1\stage3-observation-v3-independent-20260803-r1-combined-00\run-20260803T063155798Z\truth.jsonl
```

atlas 权威输出是：

```text
D:\仿真\runtime\stage3-corner-observation-atlas-v1-observational-20260805-r3\rows.csv
```

`rows.csv` 有 4,280 个观测、3,149 个唯一 exposure frame、2 个 session、12 个运动段。每行保留 truth/raw/refined 四角点坐标、raw/refined 相对 truth 的 `dx/dy/norm`、raw 到 refined 的每角点 `dx/dy/norm`，以及距离、投影尺寸、入射角和关联上下文。行键是 `session_id/producer_epoch/frame_seq/timestamp_ns/armor_index`。

为避免再次把完整分布压缩成少量统计量，2026-08-10 又从 atlas 无损导出：

```text
D:\仿真\runtime\autoaim-b-corner-evidence-complete-20260810
```

- `corner_refinement_samples.csv`：17,120 条逐观测、逐角点 raw/refined 坐标和 `dx/dy/norm`。
- `corner_refinement_empirical_distribution.csv`：51,360 个按角点和指标精确排序的值，含样本键、rank、经验 CDF 和 survival。
- `corner_truth_residual_samples.csv`：34,240 条 raw/refined 相对 truth 的逐角点残差。
- `corner_truth_residual_empirical_distribution.csv`：102,720 个按 source、角点和指标精确排序的值。
- `summary.json` 只作导航，任何判断都应回读上述逐样本或经验分布文件。

### 1.3 四个角点确实不相同

位移约定为 `refined - raw`。下表是完整分布的快速索引，不是分布替代：

| 角点 | dx 均值 px | dy 均值 px | 位移模长均值 px | P50 px | P95 px | 最大 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bl` | -0.0887 | -0.2579 | 0.7533 | 0.0000 | 3.8971 | 4.9939 |
| `tl` | -0.4938 | +0.4776 | 1.0574 | 0.0000 | 4.0525 | 4.9974 |
| `tr` | -0.3969 | +0.1085 | 1.0781 | 0.2945 | 4.4997 | 4.9920 |
| `br` | -0.2590 | -0.1919 | 0.8913 | 0.5412 | 4.0369 | 4.9916 |

这不仅是幅值不同，方向偏置也不同：上侧两个角点的平均 `dy` 为正，下侧两个为负；`tl/tr` 的平均向左移动也明显大于 `bl`。把四点混成一个“平均移动量”会丢掉与高度、短边变形和深度条件数直接相关的结构。

相对 exact-projection truth，refined 残差也不是一致优于 raw：

| 角点 | raw norm P50/P95 px | refined norm P50/P95 px |
| --- | ---: | ---: |
| `bl` | 1.2258 / 3.4606 | 1.1910 / 4.4927 |
| `tl` | 1.1163 / 3.0949 | 1.2356 / 4.2131 |
| `tr` | 1.0796 / 3.1219 | 1.3179 / 4.4009 |
| `br` | 1.1730 / 3.4699 | 1.2374 / 4.5931 |

`bl` 中位数略有改善，但四个角点的 P95 都变差，`tl/tr/br` 的中位数也变差。因此现有精修有时帮助、有时保持不动、有时加重尾部；不能只看重投影 RMS 或全局均值宣称“精修有效”。

### 1.4 角点误差如何传到 PnP

现存 `stage3-corner-to-pose-propagation-findings-independent-20260805-r1` 给出以下机制证据：

- ideal corners 的位置误差最大仅约 `0.020 mm`，说明几何定义、坐标接口和评估投影可以闭合。
- 实际 raw corners 的三维误差 P50/P95/P99 为 `0.143/0.458/0.605 m`；refined 为 `0.141/0.500/0.708 m`。
- refined 对 52.1% 样本改善、11.2% 基本持平、36.7% 变差：中心略好但尾部更重。
- 16 个固定 1 px 正交扰动方向的平均位置误差随距离上升、随投影尺寸下降；与距离的 Spearman 为 `0.952`，与投影尺寸为 `-0.897`。
- common translation 通常只带来毫米级误差；short-edge/height 变形可造成数十厘米到米级深度误差。角点总 RMS 与三维误差相关性弱，角点误差模式比单一 RMS 更重要。
- PnP 候选选择 regret 远小于总三维误差，说明错误角点会把整组候选共同推向错误深度；不能把全部误差简单归为 IPPE 候选切换。

这组结论支持后续把 PnP 观测视为带结构化偏差和重尾的量，但它是离线机制证据，不是新的在线算法。

### 1.5 历史角点方案和负证据保留

完整文件级目录位于：

```text
D:\仿真\runtime\autoaim-b-corner-evidence-catalog-20260810
```

目录索引了 74 个顶层资产、423 个 runtime 文件（2,612,677,720 bytes），并追踪 431 个当前仍存在的外部源文件（4,621,119,722 bytes）。`file_inventory.csv` 保存现存文件逐一 SHA-256；`absolute_path_references.csv` 同时保留存在和缺失的引用。

| 方案/时间 | 做了什么 | 当前判定 | 为什么保留 |
| --- | --- | --- | --- |
| 生产 PCA + `cornerSubPix` | 灯条 ROI、PCA 端点、亚像素、软平行和 5 px 回退 | 仍在生产；继承的启发式，不是已证明最优 | 当前行为基线，后续任何改动都必须与它 A/B |
| 2026-07-19 joint-PnP | 3/5/7 m 比较 legacy/corrected/joint-refined/joint-raw，联合重估 yaw+tvec | 未采用；残差小幅下降但时间误差没有一致改善，raw 分支尾部更差 | 证明“重投影更小”不是充分验收条件 |
| 2026-08-01 fixed-tilt 候选族 | 比较 `fixed_tilt_raw/fixed_tilt_refined/free_ippe_refined` | 未替换生产；pilot 中 refined P50 略好但 P95/P99 明显更差 | 保留约束姿态和 raw/refined 候选族的边界 |
| 2026-08-01~02 rigid-corner window 系列 | window、fixed-tilt-corrected、canonical-order、winding marginal、Student-t likelihood、switch-pilot、event-local-edge、continuous-polish | 没有形成当前生产观测器；多处只有极小 deployable support 或明确 `rejected.json` | 这是直接用四角点做时间窗/换板推理的历史试错，不能与 `cornerSubPix` 混称，但必须防止重复探索 |
| 2026-08-05 atlas/relationship/propagation | 单曝光分布、关系模型、角点到位姿传播、fixed-tilt 传播 | 接受为观测与机制证据，不是在线方案 | 当前最广的角点分布和灵敏度依据 |
| 2026-08-09 bounded arc-flip | 当前 full-pipeline 下 raw/PnP corners、IPPE 候选与 truth 交叉检查 | 只接受在 `r=0.75 m, d=2.2 m, 31` 个匹配观测范围内 | 验证当前代码路径和候选切换边界，不替代 4,280 行 atlas |

其中 rigid-corner 系列完整保留了 `window-pilot-eval r1-r5`、`switch-pilot-eval r1-r10`、`canonical-order r3-r4`、`winding-marginal`、likelihood calibration、`event-local-edge r6-r7` 和 `continuous-polish r9-r10`。未采用、被取代、失败和 tiny-support 输出均未删除。

### 1.6 当前 full-pipeline 有界交叉检查

输入和输出：

```text
D:\仿真\runtime\autoaim-b-arcflip-diag-20260809T013519Z\r0p75-d2p2-rep01\pipeline.jsonl
D:\仿真\runtime\autoaim-b-arcflip-diag-20260809T013519Z\r0p75-d2p2-rep01\truth.jsonl
D:\仿真\runtime\autoaim-b-arcflip-diag-20260809T013519Z\arc-flip-analysis-v3-corner-audit
```

31 个 truth 匹配的 target-3 观测表明：

- 原始到 PnP 输入的单观测平均角点移动 P50/P90/最大为 `1.072/2.327/2.453 px`；单角点最大移动为 `3.464/4.324/4.760 px`。
- 修正相机基后的 truth 投影与 detector 中心误差中位数为 `0.978 px`，错误镜像相机基为 `91.733 px`。
- IPPE 选择索引始终为 0，顺序切换为 0；两个候选 tvec 相距中位数/最大值为 `3.012/11.205 mm`。
- 选中 tvec 相对 truth 的位置误差 P50/P90 为 `0.054/0.242 m`。候选间距离远小于 truth 误差，因此该条件下的轨迹翻转不能归因于 IPPE 候选切换。

这组小样本只用于核对当前实现字段和候选路径。逐角点分布判断以 4,280 行 atlas 和完整经验分布为主。

### 1.7 已知证据缺口

文件目录明确发现 19 个缺失引用，不能声称所有历史试验都已完全复现：

1. 2026-08-05 分析链的 9 个 Python 脚本和 1 个 source spec 已不在当前仓库，包括 atlas、relationship、corner-to-pose、fixed-tilt propagation 和 descriptive 的生成/汇总脚本。现存 manifest 保存了当时脚本路径、大小和 SHA-256，输入与输出也仍在，但不能从当前 Git 直接重新运行原程序。
2. 2026-07-19 的 3/5/7 m joint-PnP 每个距离都缺少 `observations.jsonl`、`observations.cyclic.jsonl` 和 `pipeline.jsonl`，共 9 个原始/处理中间流；目前只能保留汇总 JSON、图、少量 run metadata 和现存 `scripts/analyze-pnp-joint-ab.py`。
3. 逐角点协方差从未保存，不能从现有坐标样本反推出当时 detector 的校准概率。

本轮新写的无损导出脚本不伪装成缺失旧脚本的复刻；它只读取已保留、哈希固定的 atlas `rows.csv`，把其中已有字段重新整理为逐角点长表和精确经验分布。

### 1.8 本阶段结论和方案依据

1. 四角点顺序、帧内关联、精修/回退和 PnP 输入链路已经明确；没有证据表明角点顺序随机翻转。
2. 四个角点存在不同方向偏置和不同尾部分布，不能再合并成一个平均移动量作为唯一证据。
3. 当前 PCA/`cornerSubPix` 精修不是一致改善：个别中心统计略好，但各角点 truth 残差 P95 均变差；因此本轮保留生产实现，不宣称其最优，也不凭现有离线结果直接删除它。
4. 平面 PnP 的主要风险不仅是候选选择，还包括 short-edge/height 角点变形引起的共同深度偏差。后续把 PnP `u/v` 视为带系统偏差、重尾和缺失的观测，有明确角点传播证据。
5. 当前不继续预测器；本阶段先以完整分布、处理脚本、历史目录、负证据和缺口登记收口。

### 1.9 文档已有依据下的优化空间

以下不是重新发起优化，而是对现有证据所显示缺口的如实登记：

1. 若以后重新研究前端，最低要求是 raw/refined 双分支、逐角点和逐误差模式评分、完整经验分布、truth-only 评估及运动/距离/视角分层；只看平均值或重投影 RMS 不足以验收。
2. 当前精修可研究按角点、短边/高度模式、距离和入射角做接受门控，而不是统一 5 px 门限；现有数据只说明有空间，不足以选定具体门控公式。
3. `corner_covariance_px2` 仍不可用。未来若做异方差观测器，应先定义可校准不确定性；detector objectness 不能直接冒充位置概率。
4. 120 轮 Stage3 v2 无法补回角点。若未来确需重采，schema 应原生保存 raw/refined coordinates、每角点位移、refinement accepted/fallback 原因、PnP 全候选和可校准协方差；当前阶段不要求重采。
5. 历史生成脚本缺失是可复现性风险。若未来需要复做对应结论，应基于 manifest 契约重新实现并用现存输入/输出做 hash-bound parity，而不是猜测旧实现。

### 1.10 本阶段主要哈希

| 对象 | SHA-256 |
| --- | --- |
| atlas `rows.csv` | `dd4e35cd5e67971fa996776466817ae7c521349b6e2223328f7267e1ede6bc88` |
| atlas `manifest.json` | `acd5600e584d90c7e78e60a29dba44f0ecd7aac59ed6b9fb9d4e6d37c11fdb1a` |
| 完整分布 `retention_manifest.json` | `1d41b4302a710174bd4fcf8702662c20a78bf3eb8adf3c8b3e5c59557bb7ee15` |
| 历史目录 `retention_manifest.json` | `0ded2d96857da0ae322345fdd9794db4c37d165a3247518fc693d6974a4689fb` |
| 2026-07-19 joint-PnP summary | `395fbcc9d302a71624eefc91cc8282dd21ed4b5bba132207a47b846fce55b8ab` |
| fixed-tilt candidate audit | `d7c2a55636bf9d4f4d90b044605d443db5e691a1883fd27d97375b1961c53766` |
| corner likelihood calibration | `6da6e9b8c5095f251a7c13de2e5e65d765a7336371284308539857e369ad45f1` |
| corner-to-pose findings manifest | `78d338742318bddfb13461a08d977794614636e54059027471099f49f1a50c69` |

## 下一步

本阶段先等待用户验收。验收后再从“PnP 输入四角点”继续到“相机 `tvec`、camera->gimbal/tracker 坐标、观测 `u/v` 和 exact-exposure truth join”，逐步复核坐标语义、候选选择、时间键，以及哪些 truth 字段只允许用于离线标签/评分。
