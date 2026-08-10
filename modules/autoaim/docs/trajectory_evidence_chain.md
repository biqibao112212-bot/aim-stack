# 自瞄 B 轨迹研究证据链

- 当前阶段：`2 / PnP 求解、坐标语义与整体观测轨迹`
- 状态：`阶段 1 已验收；阶段 2 已完成证据盘点，等待阶段验收`
- 日期：2026-08-10
- 研究状态：预测器暂停；本文件只整理已有数据、处理过程、方案依据、负证据和优化边界，不授权新的预测器训练或在线接入。
- 数据保留：原始采集和大体积派生证据位于 `D:\仿真\dataset`、`D:\仿真\runtime`，均按受保护资产处理；Git 保存处理代码、字段契约、证据登记和哈希，不复制大体积数据。

本文按数据真正经过的顺序逐阶段收口。均值、中位数和分位数只能用于导航，不能代替完整分布；被后续证据推翻、未采用或失败的方案不删除，而是保留并标明结论边界。机器可读登记见 `modules/autoaim/docs/corner_evidence_registry.json` 和 `modules/autoaim/docs/pnp_evidence_registry.json`。

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

## 阶段 2：PnP 求解、坐标语义与整体观测轨迹

### 2.1 当前生产数据处理链

四角点进入当前生产 PnP 后，实际经历以下步骤：

1. 根据小/大装甲板选择固定三维物点模板，二维输入使用阶段 1 的 refined-or-fallback `bl,tl,tr,br` 四点。
2. `solvePlanarPnPCandidates` 先验证角点、物点、内参和畸变均为有限值，再把物点轴顺序转换给 OpenCV `solvePnPGeneric(..., SOLVEPNP_IPPE)`。
3. 每个有限候选转换回装甲板 rvec 语义，重新投影全部角点并计算 RMS。候选先按 RMS 排序，再用 tvec、rvec 和原始 solver index 确定稳定次序；排序第 0 项标记为 selected。
4. 当前生产平移就是 selected free-IPPE candidate 的 `tvec`。`camera_tvec_m` 保存 OpenCV camera 坐标米值；`position_m` 经曝光时刻 camera->gimbal->tracker 刚体变换后进入 tracker/轨迹数据。
5. 普通装甲板 yaw 不是直接读取 IPPE rvec。生产 `yaw_absolute/yaw` 使用 tracker/chassis 系固定 `+15°` 倾角，并通过该曝光的光学姿态投影到 camera 后做约束重投影优化。
6. Stage3 写出 `position_m`、`camera_tvec_m`、`yaw_rad`、`yaw_absolute_rad` 和 reprojection 字段；truth 不进入这条在线求解链。

对应实现：

- IPPE 枚举、过滤、排序和选择：`AngleSolver.cpp::solvePlanarPnPCandidates`
- tvec 到 tracker 位置：`AngleSolver.cpp::cameraPointToTracker`
- exposure-aware yaw：`AngleSolver.cpp::optimizeYaw` 调用链
- Stage3 字段：`aim_core_bridge/stage3_capture.cpp`
- 帧和坐标契约：`AngleSolver/COORDINATE_CONTRACT.md`

一个容易混淆的字段语义是：Stage3 的位置来自 selected free-IPPE tvec；`reprojection_rms_px` 在 parallel joint diagnostic 可用时优先保存 exposure-constrained yaw+tvec 诊断残差，否则回退 legacy constrained residual。它不应被无条件解释成 selected free-IPPE candidate 自己的 RMS。需要候选级 RMS 时必须使用保留 `pnp_candidates[]` 的 full-pipeline/v3 诊断流。

### 2.2 camera、gimbal 和 tracker 坐标契约

当前固定坐标为：

- OpenCV camera `C`：`+x right, +y down, +z forward`；PnP tvec 单位为 mm。
- calibrated gimbal `G`：`+x forward, +y left, +z up`。
- tracker/chassis `T`：`+x forward, +y left, +z up`。
- YAML 的 `R_CAMERA2GIMBAL = ^G R_C`，`T_CAMERA2GIMBAL = ^G t_C`，T 以 m 为单位。
- `FrameMeta.poseEuler` 表示严格曝光匹配的 `^T R_C` 光学姿态，不是可随意替换的相邻帧 joint pose。

每个曝光按以下同一 SE(3) 对转换：

```text
^T R_G = ^T R_C (^G R_C)^T
p_T = ^T R_G (^G R_C p_C + ^G t_C)
p_C = (^G R_C)^T ((^T R_G)^T p_T - ^G t_C)
```

当前契约没有经验高度 `H`。图像缺少精确曝光光学姿态时必须在 PnP 前拒绝，不能用邻帧补齐。

2026-08-01 的 15,676 行 provenance 审计得到：位置合同残差 P50/P95/P99 为约 `1.77e-10/3.83e-8/8.55e-8 m`。历史大 PnP 百分位可由 truth label 漏掉 `R_gimbal_pose_to_tracker` 复现；因此那部分历史“大误差”首先是评估坐标错误，不是生产 PnP 本身突然变坏。

### 2.3 PnP 实验时间线和当前判定

| 时间/实验族 | 研究问题 | 当前判定 |
| --- | --- | --- |
| 2026-07-19 初始 yaw/tracker 字段分析 | `tracked_id/jump_flag` 是否能代表物理板；原 yaw 坐标是否正确 | ID 字段解释被推翻；不能作为物理真值 |
| 2026-07-19 PnP yaw G2 | `+15°` 倾角应在 camera 还是 tracker/chassis 系定义 | tracker/chassis 定义通过合成和动态验收，进入生产 |
| 2026-07-19 joint-PnP A/B | 联合重估 yaw+tvec、raw/refined corners 是否更好 | 未采用；残差略降但时间增量没有一致改善，raw 分支尾部更差 |
| 2026-07-21~23 PnP-history state / pose adapter | 从 PnP 历史恢复当前刚体状态是否可行 | held-out 失败；同会话 capacity 只作诊断，后续被 clean-physics 路线取代 |
| 2026-07-26 real-PnP frozen-F upper bound | 把真实 PnP 直接送入已有 clean-F 是否足够 | 明确拒绝，冻结网络会放大 observation-domain shift |
| 2026-07-26~27 mapper、adapter、joint robustness、dual-domain F | 外部修正或域专用模型能否追回误差 | 外部 mapper/adapter 收口失败；dual-domain F 仅保留诊断，不是部署结果 |
| 2026-08-01 coordinate provenance | camera/tvec/tracker/truth 的坐标链是否闭合 | 通过；确认历史漏旋转标签错误 |
| 2026-08-01 PnP feature/manifold audit | 当前帧可部署特征能否校正 PnP | 有可学结构但跨会话尾部仍大；未选择修正器 |
| 2026-08-01~06 free/fixed tilt、角点传播 | 候选、约束倾角和角点误差如何传到位姿 | exact 关闭数值误差；fixed tilt 条件性有效，不统一替代 free IPPE |
| 2026-08-06 constant/linear/quadratic correction | 单帧偏差能否用简单模型纠正 | 常量偏差存在；条件模型跨会话失效或过拟合，未部署 |
| 2026-08-06 distance-yaw geometry grid | 距离、斜视角、可见性和深度误差关系 | 接受为条件分布证据；不是简单 yaw 单调函数 |
| 2026-08-05~06 PnP trajectory distribution/stability | PnP 整体轨迹是否稳定、可重复、可逆 | 轨迹有重复结构但非线性、重尾且不点对点可逆 |
| 2026-08-09 arc/outlier/candidate diagnostics | 观测弧翻转是否由 IPPE 分支切换 | 两个有界采集均否定 branch-switch 假设 |
| 2026-08-09 120-run truth-gated matrix | 半径、距离、spin/combined 下整体观测误差如何 | 接受为当前最广 PnP 后观测过程证据；不等于部署验收 |
| 2026-08-09 CV+Ridge/association | PnP 后 u/v 能否做因果离线处理 | CV+Ridge 仅条件接受为离线组件；身份和预测器继续暂停 |

### 2.4 yaw、joint-PnP 和条件数证据

PnP yaw G2 已接受的内容只有坐标语义修复：普通装甲板固定倾角在 tracker/chassis 系构造，再通过曝光姿态投影到 camera。非零云台合成回归恢复已知 `37°` chassis yaw 的误差小于 `0.1°`，精确姿态重投影小于 `1e-4 px`；3/5/7 m 动态曲线保持连续趋势，但 7 m 噪声与缺失增加。

joint-PnP sidecar 的 refined yaw 灵敏度 P50 随距离从 `3.734` 增至 `5.331/6.585 deg/px`，P95 为 `8.934/12.496/12.916 deg/px`。这说明像素误差经平面几何会被放大，但该灵敏度是局部条件数，不是实际误差概率。

joint refined 只小幅改善重投影残差，时间增量没有跨距离一致改善；joint raw 虽进一步降低残差，却显著加重时间尾部。因此 joint 求解器保留为历史诊断 sidecar，没有替换生产 free-IPPE translation + exposure-aware yaw 路线。

### 2.5 单帧 tvec、候选和修正试验

角点到位姿传播给出以下关键结果：

- exact corners 使 production free-IPPE 位姿误差收敛到数值零，证明物点、轴交换、投影和评分接口闭合。
- 4,280 行独立证据中，actual raw P50/P95/P99 为 `0.143/0.458/0.605 m`，actual refined 为 `0.141/0.500/0.708 m`；精修中心略好但尾部更重。
- 更广 distance-yaw grid 中 raw/refined P50/P95 为 `0.112/0.695 m` 和 `0.090/0.724 m`；1 px 固定正交扰动也能形成 `0.621 m` P95，证明平面短边/高度模式会放大成深度长尾。
- known-tilt solver 在工作区内对 raw 的逐行改善率为 61.7%，但 refined 只有 47.0%；不能宣布统一替代。

候选分支诊断分别覆盖 2.2 m 的 31 个和 5 m 的 767 个 truth 匹配观测：selected solver index 始终为 0，连续切换为 0。5 m 两候选 tvec 中位间距仅 `3.095 mm`，selected pose truth error 中位数却为 `0.417 m`。所以错误角点/平面条件会把整组候选共同推向错误深度，当前没有证据把整体轨迹异常归因于 IPPE 候选跳变。

单帧修正试验的结论是“存在结构，但没有选出可部署修正器”：

- 两会话审计中，segment 留出的 linear/quadratic 模型改善中位数和 P95，但整会话留出可恶化到 `2.6~3.2 m` P95，属于条件过拟合。
- 六会话 control 中 raw linear correction 从 `0.112/0.441 m` 改到 `0.077/0.216 m` P50/P95，但每会话只有一个运动段，session 和 segment 留出数值重合，证据仍不足以宣称跨运动泛化。
- PnP feature audit 的严格 leave-one-session-out 从 raw PnP `304/1780/3069 mm` 改善到包含 corners、双候选和 reprojection 特征的 `146/517/943 mm` P50/P95/P99；尾部仍过大。固定中心的 same-session 结果只有数十毫米，说明随机分帧会严重夸大性能。

### 2.6 56-session PnP 测量流形

`stage3-pnp-trajectory-distribution-v1-full-r4` 保存 56 个 session、180,289 条 paired PnP/truth 行和 188,797 条完整 visible-truth 行。点级 `paired_trajectory_rows.csv` 是分布权威，图和汇总不替代原始行。

整体映射结果为：

| 映射 | overall RMSE/P95 m | 边界 |
| --- | ---: | --- |
| identity | 0.3649 / 0.8306 | 当前 selected PnP 直接当 truth |
| cross-session affine | 0.2203 / 0.4222 | 可部署字段的低容量跨 session probe |
| cross-session conditioned | 0.0199 / 0.0392 | 使用 truth phase 等 oracle 条件，仅作描述上限 |

跨 session collision audit 进一步显示，selected-PnP XYZ 相距 5 mm 的配对中，truth 相距超过 20/50/100 mm 的比例仍为 `26.61%/6.79%/0.97%`。这不能证明严格数学不可逆，但足以否定“selected PnP XYZ = truth + 小的独立高斯噪声”这一工程假设。

### 2.7 整体观测轨迹证据

观测轨迹稳定性审计以 session 为独立单位，而不是把帧当重复实验：

- 56 个 source sessions、16 个空间重复 pair、8 个 phase-curve pair、61,629 条纯旋转区间行。
- 真实匀速旋转的 phase-bin speed ratio 最大仅 `1.0014`，PnP apparent speed 的 phase max/min 却为 `1.61~2.45`；跨 session PnP speed curve correlation 为 `0.691~0.981`。
- 因此 PnP 位置曲线可以稳定、重复，但逐帧差分速度仍会因非线性和跳点强烈放大。

120-run truth-gated matrix 是当前最广的 PnP 后观测过程证据：

| motion | 样本 | angular P50/P95 | depth abs P50/P95 | lag-1 autocorrelation | median P90 missing streak |
| --- | ---: | ---: | ---: | ---: | ---: |
| spin | 117,558 | 0.0705/0.2047° | 0.0974/0.8285 m | 0.831 | 797.7 ms |
| combined | 112,240 | 0.1544/0.3346° | 0.1082/0.8630 m | 0.741 | 519.2 ms |

这解释了为什么“角度看起来很好”与“三维深度误差很大”可以同时成立。distance-yaw hit-area audit 中，raw/refined 的三维位置 P95 达到 `0.695/0.724 m`，但估计中心投影仍有 100% 落在可见 truth plate 四边形内；所以后续评价不能只使用三维欧氏距离，也不能只使用图像角度。

5 m 观测开弧在五次重复中可以稳定拟合，但 12 个物理槽位里 11 个与 truth 曲率方向不一致。这是可学习的系统观测模式，不是物理轨迹正确性的证明。combined motion 相比 spin-only 使 phase-bin P90 band 扩宽约 `1.41~1.56x`，且可改变部分槽位的 cup/cap 形状；camera/gimbal 观测空间中的平移与自转不可简单解耦。

### 2.8 已被推翻或限制的解释

以下结论继续保留为负证据，禁止在后续重新当作前提：

1. `tracked_id`、`jump_flag` 或 detector digit 不是物理装甲板身份。
2. 旧 truth camera 横向符号是镜像错误；正确 OpenCV 基为 `[-sim_local_y, -sim_local_z, sim_local_x]`。
3. 全局四阶多项式连接空白区会制造端点翘曲和假 `U/cup`，不能证明真实弧翻转。
4. 旧 angular-only 物理槽位分配在修正审计中大面积改变，历史 observation-only per-slot 预测数字不能冒充真实四板结果。
5. 2.2 m 和 5 m 候选级数据都否定 IPPE index switching 是所见翻转的原因。
6. 5 m 空 Stage3 行主要来自 detector admission；非空帧 raw detector 数与 solved PnP 数一致，PnP rejection 为 0。
7. 早期 nominal 0.75/1.0/1.25 radius grid 实际都使用 stock radius；相关“半径效应”结论被 truth-gated 重采取代。
8. 更低 reprojection residual 不保证更好的 truth pose 或时间连续性。
9. 单帧条件模型在同 session 或相邻 segment 的改善不能当作跨 session 泛化。
10. PnP yaw 对第一版 Ridge 帮助有限且损害 observation-only identity heuristic，不能把 yaw 当安全身份 cue。

### 2.9 下游 PnP 消费实验的边界

真实 PnP 直接替换 clean observation 会被冻结预测网络放大：translation/rotation/combined conditional P95 从 `5.63/25.40/81.04 mm` 增至 `1350.31/368.06/937.01 mm`。后续 mapper 把 combined current P95 从 `179.51` 降至 `120.40 mm`，但完整 frozen chain 仍为 `536.82 mm`。dual-domain PnP F 最终诊断 conditional P50/P95/P99 为 `39.70/215.85/537.76 mm`，仍有 oracle association、provenance 和 validation reuse 限制。

这些实验说明 PnP observation domain 与 clean physics domain 不能直接互换，但它们不是 PnP 求解器修复。2026-08-09 接受的 u/v CV+Ridge 也只是 PnP 后的离线因果处理组件；它在身份正确条件下表现良好，却不能修复错误物理板关联。用户已经暂停预测器和多假设 tracker，本阶段不恢复这些工作。

### 2.10 PnP 证据文件目录和缺口

完整文件级目录：

```text
D:\仿真\runtime\autoaim-b-pnp-evidence-catalog-20260810
```

- 421 个 PnP/pose/observation/trajectory 顶层资产。
- 11,015 个 runtime 文件，共 23,124,946,682 bytes。
- 542 个由 JSON/manifest 引用且当前存在的外部源文件，共 3,878,007,922 bytes。
- 目录保留 accepted、rejected、superseded、invalid 和 failed 运行，不以最终报告覆盖旧资产。

规范历史 `D:\浠跨湡` 路径乱码并排除整条命令行后，仍有 60 个真实缺失引用：

1. 31 个历史分析 Python、测试或 source spec 已不在当前 Git，包括 PnP trajectory、stability、distortion recoverability、motion ceiling、geometry grid、temporal observation 和部分 corner propagation 生成程序。现存 manifest 仍保留旧脚本哈希和输出。
2. 29 个早期 runtime 原始/中间文件缺失，主要是 2026-07-19 的 chassis-pose、joint-PnP、yaw observation-set/pipeline 流，以及一个早期 screen-corner result。对应小型 summary、图片或 run metadata 仍保留，但不能声称完整重放。

机器可读逐项清单为 `path_references.csv`；不在本段省略或美化缺失项。

### 2.11 本阶段结论

1. 当前生产 PnP 的 translation、yaw 和坐标链已经明确：free-IPPE selected tvec、exposure-aware chassis yaw、严格曝光 camera->gimbal->tracker SE(3)。
2. 坐标合同和 exact-corner 求解都能数值闭合；当前大误差主要不是接口无法闭合，而是 detector/refinement 角点误差经平面深度条件放大形成的结构化观测偏差。
3. IPPE 候选切换没有得到证据支持；错误角点可使两个候选共同偏向错误深度。
4. PnP 误差依赖角点模式、距离、斜视角、画面位置、运动状态和可见性，并具有重尾、时间相关和长缺失段；固定独立高斯噪声模型不成立。
5. PnP 整体轨迹具有跨重复的稳定结构，但它是非线性的“观测轨迹”，不等于真实物理圆弧。稳定可学和几何正确必须分别评价。
6. fixed tilt、常量/线性 correction、rich feature mapping 都显示一定空间，但现有文档没有选择可部署 PnP 修正方案；因此本阶段不改生产求解器。

### 2.12 依据现有证据保留的优化空间

以下只登记文档已经显示的空间，不重新启动优化：

1. 若以后研究 PnP correction，必须使用完整候选、raw/refined corners、距离/斜视/画面位置等可部署字段，并按完整 session 和受控几何条件留出；随机分帧或单 session 结果无效。
2. fixed-tilt 可保留为有可观测工作区和安全回退的候选臂，而不是统一替换 free IPPE；当前证据不足以定义最终 gating。
3. 评价至少同时报告 camera-ray angular、image hit-area、tracker transverse、depth 和 3D Euclidean error；任何单项都可能掩盖另一项风险。
4. 应明确分离 free-IPPE candidate RMS、exposure-constrained yaw residual 和最终 tracker position error，避免继续复用一个 `reprojection_rms_px` 名称混淆不同求解层。
5. `corner_covariance_px2` 和 pose covariance 仍不可用。若未来使用 EKF/UKF/factor graph，应先建立按角点模式和几何条件校准的不确定性，不能使用固定 R 矩阵冒充已校准噪声。
6. 未来重采 schema 应原生保存候选 rank、原始 solver index、候选间隔、candidate tvec/rvec、free-IPPE RMS、yaw-constrained residual、raw/refined corners 和 accepted/fallback 原因；120-run v2 无法离线补回这些字段。
7. 历史生成脚本缺失需通过 manifest-contract parity 重新实现，不能把现存报告反推成旧代码的精确复刻。

### 2.13 本阶段主要哈希

| 对象 | SHA-256 |
| --- | --- |
| `pnp_yaw_stage2.md` | `22141bbfaf4b086081ff851615f2217ff91f328618abae6c7b8b55d2298c5683` |
| `COORDINATE_CONTRACT.md` | `20524740ba659b5c755806be9f1033cce34156f1a04f022489616de0593eb841` |
| `pnp-chassis-pose-summary-20260719.json` | `ee17c0bcb2e27e73fa3a801074f4e89076260454bc9896d01ca2900687f0184a` |
| `pnp-joint-ab-summary-20260719.json` | `395fbcc9d302a71624eefc91cc8282dd21ed4b5bba132207a47b846fce55b8ab` |
| PnP coordinate provenance summary | `07f1e11e9ff2c416651d5d3ce886956ff62ad5e1da17805954fff4e7850b77fc` |
| 56-session PnP trajectory manifest | `7b7c1d3f83ac6435041bcc37de0aa847b5699c2a08f1f56f379f93f7d6f4759a` |
| observation stability manifest | `6b7c97ba5596f9c9ac5254485a0b54b943614620493cc906b74a5c0ac188b8b1` |
| distortion recoverability manifest | `3288b30b2f87490ad01e764cded21f28b9a828f3d1dbbdc7d98f31b1b21bd3ef` |
| distance-yaw grid summary manifest | `07165f782d28d6b68dfb6f49390d5539f3e44bf0ddf7908ff22e494d733278b9` |
| 120-run observation-error retention manifest | `55e6e926c255d363715cddea8e21649312f22c85f33f91317de243d5885eb8fe` |
| PnP 历史目录 retention manifest | `56abca29bf3399aad3be31442bd9a261d60361b27c081290adf78c1837336742` |

## 下一步

阶段 2 先等待用户验收。验收后再进入阶段 3：只梳理 PnP 后观测如何形成因果时间序列，包括 exact-exposure join、`u/v` 定义、缺失/间断、相邻集合变化、离线物理槽位与部署关联边界；预测器继续保持暂停。
