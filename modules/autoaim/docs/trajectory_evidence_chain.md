# 自瞄 B 轨迹研究证据链

- 当前阶段：`5 / 命中相关误差传递与真值替换消融`
- 状态：`阶段 1--4 已完成；阶段 5 已完成离线证据收敛，等待阶段验收`
- 日期：2026-08-10
- 研究状态：生产预测器暂停；阶段 5 只增加离线真值替换和 simple-CV 基线，不授权预测器训练、在线接入或火控放行。
- 数据保留：原始采集和大体积派生证据位于 `D:\仿真\dataset`、`D:\仿真\runtime`，均按受保护资产处理；Git 保存处理代码、字段契约、证据登记和哈希，不复制大体积数据。

本文按数据真正经过的顺序逐阶段收口。均值、中位数和分位数只能用于导航，不能代替完整分布；被后续证据推翻、未采用或失败的方案不删除，而是保留并标明结论边界。机器可读登记见 `modules/autoaim/docs/corner_evidence_registry.json`、`modules/autoaim/docs/pnp_evidence_registry.json`、`modules/autoaim/docs/timeseries_evidence_registry.json`、`modules/autoaim/docs/observer_acceptance_registry.json` 和 `modules/autoaim/docs/hit_oriented_ablation_registry.json`。阶段 4 的完整设计合同见 `modules/autoaim/docs/observer_specification.md`；阶段 1--5 的连续研究叙事、径向/横向口径和真值替换结论见 `modules/autoaim/docs/corner_pnp_state_estimation_research_narrative.md`。

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

## 阶段 3：PnP 后因果时间序列、缺失与身份边界

### 3.1 在线采集与离线 truth 连接

当前在线数据和离线标签是两条不同的流：

1. TCP 图像头提供 `producer_epoch`、`source_sequence/frame_seq` 和 `capture_timestamp_ns`。Windows bridge 以该 frame sequence 读取对应的云台状态和 ground-truth exposure snapshot。
2. 图像经 detector、角点处理和 PnP 后，在 `solveArmors()` 之后、`trackerUpdate()` 之前把整帧 `solved_armors` 深拷贝给 Stage3 sink。这里没有 tracker 选择、持久 ID 或预测器输出。
3. `stage3-observation-v2` 是追加写 JSONL；异步队列上限为 8,192 行。队列溢出或第一次持久 I/O 失败后 sink 失败关闭，不静默覆盖旧行。
4. truth 另写 `stage3-truth-v1`。只有 ground truth 和 exposure state 的 frame sequence、timestamp 都与图像头相等时，`has_exact_exposure_truth=true`；否则仍保留同键不可用记录，而不是拿相邻 truth 代替。
5. 离线只允许用 `session_id + producer_epoch + frame_seq + timestamp_ns` 四字段完全相等连接。禁止 nearest frame、插值、仅 timestamp、仅 frame sequence 或跨 producer epoch 连接。

对应实现：

- observation sink：`modules/autoaim/src/aim_core_bridge/stage3_capture.cpp`
- 图像头、云台和 truth 读取：`modules/autoaim/src/sim_adapter/windows_talos_bridge_node.cpp`
- 字段契约：`modules/autoaim/docs/stage3_data_contract.md`
- 120-run 逐样本导出：`scripts/export-timeseries-evidence-distributions.py`

正式 120-run 证据有 189,158 个 truth 记录，其中 189,041 个 `has_exact_exposure_truth=true`。184,879 个 observation 记录全部能找到相同四字段键，但只有 184,763 个同时具有可用 exact truth。历史 `analysis_summary.json` 中的 `exact_join_rows` 实际是四字段键相交数，不能自动解释成可用 exact-truth 数；本轮已把两者拆为 `full_key_join_frames` 和 `usable_exact_truth_join_frames`。

两条流都没有重复键。原分析也记录 truth/observation 的 sequence 和 timestamp regression 均为 0。仍需保留的质量标志是：184,763 个 observation 标记 `gimbal_pose_exposure_matched=true`，116 个为 false；全部 184,879 行都使用配置内 camera-gimbal extrinsic 和 `calibrated-camera-gimbal-extrinsic-v1` position contract，但 `tracker_world_transform_exposure_matched` 全为 false。因此这批数据足以定义 camera-frame `u/v`，却不能单靠该字段证明 world-transform exposure match；tracker-frame位置合同的独立验证仍应引用阶段 2 的 coordinate provenance。

### 3.2 `u/v` 的唯一含义

轨迹处理中的 `u/v` 不是 detector center，也不是像素坐标：

```text
camera_tvec_m = [x_right, y_down, z_forward]
u = atan2(x_right, z_forward)
v = atan2(y_down, z_forward)
```

离线文件以 degree 保存，处理器内部可以使用 degree 或 radian，但必须显式标单位。由此得到的射线为 `[tan(u), tan(v), 1]` 归一化值；最终角误差是两条三维单位射线的夹角，不是简单把 `du/dv` 欧氏距离永远当球面角。

`u/v` 选择 camera tvec 而不是 tracker position 有两个依据：

1. 它直接来自当前 PnP 的 camera-frame 平移，不依赖尚未被该帧 audit flag 证明的 world transform。
2. 它保留图像射线方向，使阶段 2 已观察到的“角度较准、深度长尾很大”可以被拆开处理；若直接压成 tracker 3D 欧氏误差会混淆角向和深度风险。

完整 `u/v` 权威位于 `detection_uv_samples.csv`：250,449 条 raw candidate 均逐条保留 frame key、帧内 observation index、detector 字段、camera XYZ、u/v、yaw 和 reprojection 字段。本轮没有发现 invalid tvec，但这不允许未来 schema 删除 valid/null 检查。

### 3.3 帧、有效事件和真实时间

必须区分三个层级：

| 层级 | 定义 | 是否进入 causal history |
| --- | --- | --- |
| truth exposure frame | bridge 收到并写出的图像曝光/truth 记录 | 不直接作为在线输入 |
| observation frame | Stage3 sink 写出的一整帧 armors，可为空 | 空帧不形成有效事件 |
| valid event | 至少一个 `valid=true` 且 finite `camera_tvec_m` 的 observation frame | 进入事件历史 |

在 120-run 中：

- truth frame：189,158；observation frame：184,879；valid event：177,483。
- spin 有 404 个 truth 帧没有 observation 记录，另有 3,782 个 observation 空候选帧。
- combined 有 3,875 个 truth 帧没有 observation 记录，另有 3,614 个 observation 空候选帧；其中两轮完全没有 observation，必须继续计入 availability 分母。
- frame sequence 经 latest-only 图像链可以跳号。相邻 observation transition 中，spin/combined 分别有 16,213/16,085 次 `frame_seq_delta>1`；因此禁止按 frame index 假设固定采样周期。

当前接受的 `stage3-dataset-v3` 事件合同为：

- 最近最多 200 个 valid events，按真实 timestamp 排序并右对齐；左侧 padding 的 mask 为 false。
- `event_time_s = observation_timestamp - anchor_timestamp`，不构造 5 ms 网格，不用 event index 推时间。
- 空候选或 invalid frame 不占事件槽，但到下一有效事件的真实时间差仍保留，所以缺失不能被压缩掉。
- 任一 contributing span 出现超过 4 candidates 时拒绝整段；实时门禁仍要求最近 0.2 s 至少 8 个有效事件、最新有效事件年龄不超过 50 ms。
- train-only event dropout 是下游历史增强，不改变 raw evidence；违反实时门禁时必须回退该增强。

旧 `stage3-dataset-v2` 的固定 5 ms 合同已被明确取代，不能把 v2 shard/checkpoint 伪装成 v3 兼容数据。v2 资产继续保留用于解释历史结果。

### 3.4 360-session 正式数据对事件合同的证据

2026-07-20 的正式数据包含 360 个成功的 30 s session，按 stationary/linear/spin/combined、距离、速度和方向分层。无效的并发 runner 尝试、首轮低吞吐 smoke 和其修复过程仍在 runtime 历史目录中，但不进入 360-record master manifest。

360-session raw 流的完整规模为：

- 1,389,655 条 observation records，1,621,444,269 bytes。
- 1,393,235 条 exact truth records，另有 1,669 条 unavailable truth records；truth 共 11,269,063,685 bytes。
- 429,122 个 zero-candidate records、393,958 个 multiple-candidate records、38 个 `>4` candidate records、0 个 invalid-armor records。
- v3 最终产生 185,292 个 samples，按 session 独立分成 111,527 train、36,297 validation、37,468 test；360 个 session 中 6 个保持 zero-sample，1.67% 低于当时预注册的 10% 门槛。

主要 tensorization exclusion 也完整保留：`insufficient_recent_valid_observations=158,932`、`missing_future_truth=32,808`、`ego_unstable_history=20,302`、`no_valid_observation_events=14,745`、`history_more_than_four_candidates=8,869`、`latest_valid_observation_too_old=2,344`、`anchor_more_than_four_candidates=6`。这些是窗口拒绝计数，不是可拿零填充的训练标签。

权威不是上述总数，而是 360 个 raw JSONL、每 session qualification record、185,292 个 shard sample 和 manifest 哈希。`qualification_report.json` 保留每个 session 的记录数、空候选、候选数、exact truth、排除原因和 tau 误差。

### 3.5 完整的时间间隔与缺失分布

本轮新增无损导出：

```text
D:\仿真\runtime\autoaim-b-timeseries-evidence-complete-20260810
```

- `frame_availability_samples.csv`：189,158 个 truth-frame 基准行，逐行记录 observation 是否存在、candidate 数、valid-event 状态、时间间隔和曝光匹配标志。
- `valid_event_interval_samples.csv`：177,483 个有效事件及其前序有效事件真实间隔。
- `missing_streak_samples.csv`：2,974 个连续缺失段，分别标记 `observation_frame` 与 `valid_event` 两层、起止键、帧数、时长和右截断。
- `empirical_distributions.csv`：554,138 个精确排序值，含 sample key、rank、CDF 和 survival；任何分位数只作导航。
- `run_summary_samples.csv`：120 轮逐轮 availability，不删除两轮零观测失败。

有效事件间隔的 P50/P95/P99/max 为：spin `7.95/23.48/32.75/592.19 ms`，combined `7.93/24.05/33.13/339.43 ms`。但间隔分位数不能替代缺失段：valid-event missing streak 的 P95/P99/max 为 spin `144.09/275.18/558.00 ms`，combined `79.74/191.78/15021.28 ms`；combined 最大值由零观测运行形成。

这说明典型帧间隔接近 8 ms，与“会出现数百毫秒甚至整轮缺失”并不矛盾。只用平均 FPS 或 P50 interval 无法定义观测器的 timeout、reacquisition 和安全回退。

### 3.6 相邻候选集合变化

原始事实源是每帧完整候选集合，而不是“一个已经跟好的物理板”。120-run 中 observation frame 的 valid candidate 数完整频数为：

| motion | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spin | 3,782 | 58,166 | 26,901 | 5,094 | 93 | 0 |
| combined | 3,614 | 57,592 | 24,072 | 5,187 | 360 | 18 |

combined 的 18 个五候选帧是 raw 证据；它们不能进入最大四候选的 v3 contributing span，但不能从历史目录删除。

`candidate_set_transition_samples.csv` 保存 184,761 个相邻 observation transitions：

- spin 93,976 次 transition 中，7,311 次 valid count 改变，15,916 次 detector number/color/type 多重集合签名改变。
- combined 90,785 次中，15,258 次 valid count 改变，28,106 次 detector 签名改变。

这些只能称为集合基数或 detector-signature 变化。`observation_index` 每帧从 0 重新分配，detector number 可能误分类，集合签名相同也不证明是同一物理板；因此本阶段没有从这些行伪造 birth/death 或 physical-ID switch。

### 3.7 truth 物理槽位只用于离线标签和评分

离线 truth 的 `relative_slot=0..3` 是 target-local 物理槽位。使用它之前必须：

1. 四字段 exact join；
2. 在本 run/frame 内选择目标 3 的完整四板目标；`target_id` 只在 simulator run 内有效；
3. 将 truth 板中心转换到 exposure camera frame；
4. 在 truth-only 标签侧执行一对一 assignment。

历史 assignment 方案的边界为：

- angular-only 最近射线曾用于早期复现，但对面板可能落在接近相同的 camera ray，不能作为安全物理身份。
- PnP 3D 最近邻使用完整深度，但阶段 2 已证明平面深度误差可大于板间距，也不能天然保证身份。
- 当前 120-run accepted analysis 使用“所有正 facing truth plates 中的最近 image ray + 0.75° gate + 一对一约束”，并保留 second-best margin。它是 oracle-labeling 方法，不是部署 associator。
- 旧 angular-only 物理槽位和 observation-only polynomial 结果被修正版覆盖，但原输出继续保留。

truth phase、relative slot、target pose、future truth 都禁止作为在线 observation processor 或 associator 输入。即使一个分析只在分组阶段用了 truth slot，其结果也只能称为 oracle-identity upper bound。

### 3.8 observation-only 关联实验

部署关联实验在 runtime 输入侧去掉 truth，再在整轮结束后用 truth 做评分：

| 方案 | 结果 | 当前判定 |
| --- | ---: | --- |
| 简单 u/v CV hard tracks | mean mapping accuracy 0.535 | 基线，身份尾部不可接受 |
| learned independent pair cost | best accuracy 0.462 | 拒绝；高 purity 主要来自丢检测 |
| nested cyclic segment rule | accuracy 0.683，associated fraction 0.974 | 接受 cyclic topology 为结构证据，不接受 hard ID 部署 |
| cyclic hard ID + CV/Ridge | identity-correct P95 0.287°，全链 P95 19.62° | 灾难性错身份尾部，拒绝部署 |
| confidence >=0.95 | coverage 0.466，identity accuracy 0.692，全链 P95 10.29° | confidence threshold 不能修复 hard-ID tail |

因此“PnP 后 u/v 可以做良好的局部时间处理”和“系统知道它是哪块物理板”是两个不同命题。离线 CV+Ridge 在身份正确时的 P95/P99 为 `0.310/0.717°`，但旧 hard association 的全链 P95 为 `14.50°`。多假设 C4 belief tracker 是历史上提出的下一结构，却已按用户要求暂停，本阶段不实现。

### 3.9 v4 future-observation 与缺失标签负证据

历史 `stage3-dataset-v4-observation` 增加 future observation position、per-slot mask、frame-availability 和 ambiguous mask。其缺失语义仍值得保留：

- future timestamp 必须精确命中原始 observation；不允许 nearest/interpolation。
- 缺失 exact future frame 时 observation loss 整体 mask。
- exact frame 存在但 zero candidate 时，这是显式 visibility negative，不是位置 `[0,0,0]`。
- `>4` candidate future frame 标记 ambiguous，不能强行配成四个 truth slots。

v4/v5 的网络 learnability 争论属于下游预测器历史；本阶段只继承上述数据语义，不恢复训练，也不把某次 future-observation 网络结果当成当前观测器方案。

### 3.10 历史文件目录和复现缺口

文件级目录：

```text
D:\仿真\runtime\autoaim-b-timeseries-evidence-catalog-20260810
```

目录覆盖 434 个资产、10,453 个直接文件、29,071,372,161 bytes，并追踪 192 个当前存在的外部源文件、5,130,854,778 bytes。范围包括：

- 360-session formal raw observation/truth 与 master evidence；
- 被取代的 v2、接受的 v3 和 v4 observation datasets/shards；
- 独立 observation-v3 采集；
- 采集吞吐、temporal/future-query/trajectory 实验；
- 120-run accepted/invalid raw roots、完整分布、离线 processor 和 association 尝试。

仍有 17 个真实缺失引用：16 个历史 `training/stage3` 分析/构建/测试源和 1 个早期 screen-corner result。它们逐项记录在 `path_references.csv`；相应 manifest/output 仍可作为历史证据，但不得声称从当前 Git 完整重放。

### 3.11 本阶段结论

1. PnP 后在线事实源是无序、逐曝光、可为空且候选数可变的 frame-local 集合；不是四条已经有物理 ID 的等频轨迹。
2. exact exposure 必须拆成“键相等”和“truth 可用”两层。本轮 184,879 个键连接中有 184,763 个可用 exact truth；相邻帧补齐仍被禁止。
3. `u/v` 是 camera tvec 射线角，显式分离角向轨迹与深度长尾。detector center、tracker 位置和 image pixel 不能无说明混用为同一 `u/v`。
4. 真实事件时间不均匀且 frame sequence 经常跳号；v3 的 timestamp/event-mask 合同正确取代固定 5 ms 网格。
5. 空 observation、zero-candidate frame、invalid candidate、`>4` ambiguous frame、长缺失段和整轮零观测是不同 failure modes，必须分别保留和处理。
6. 候选集合变化频繁，但 frame-local index/signature 不能定义物理 birth/death。truth slot assignment 只允许离线标签/评分。
7. cyclic topology 提升 observation-only association，但单一 hard identity 仍产生十几度尾部；当前没有可部署身份方案，预测器和 belief tracker 继续暂停。

### 3.12 依据现有证据保留的优化空间

以下是证据指出的缺口，不是本轮开始实施的新方案：

1. 观测器接口应原生接收真实 timestamp、可变集合和 validity/missing reason；禁止在输入边界先重采成固定帧率并丢弃 gap。
2. timeout/reacquisition 不能只按平均 FPS 设定，应在逐 run、逐 motion 的完整 missing-streak 分布上验收，并显式覆盖整轮无观测。
3. 若未来需要 tracker/world frame 精确审计，采集端应正确填充并验收 `tracker_world_transform_exposure_matched`；当前 120-run 该标志全 false，不能用 camera-frame证据冒充 world-transform 证据。
4. `>4` candidates 不应静默裁成 4 个；可保留全集合并标 ambiguous，或在上游定义可解释 admission。当前 v3 的窗口拒绝是安全基线。
5. 未来 identity 模块必须输出 permutation/phase 概率或 top-k，而不是单一未经校准的 ID；验收应包含 top-k coverage、NLL/Brier、reacquisition、拒绝覆盖率和全链尾部。
6. candidate/pose covariance 仍未校准。观测器若使用自适应 R，应以角点模式、候选间距、几何条件、缺失和 empirical residual 校准，不能把 detector confidence 直接当位置概率。
7. 16 个历史 time-series 生成源缺失。需要复做相应结论时，应按 manifest 输入/输出/hash contract 重写并做 parity，而不是猜测旧代码。

### 3.13 本阶段主要哈希

| 对象 | SHA-256 |
| --- | --- |
| `stage3_data_contract.md` | `ae06ac53054a1e101eed8952388af2ac84e020637474101163a31e913cf9ae05` |
| `stage3_capture.cpp` | `0c4f8982792209b80f817ce9807e73df3fc319ae540b983f785207eced7fefec` |
| `windows_talos_bridge_node.cpp` | `f45055d50832a725953d8d63945a6c7ffec79871d37bd07b87c7598dc9558cbc` |
| `stage3-dataset-v2-20260720-r5/dataset_manifest.json` | `026cbab209884f51150f2650ab25765b095738df3196d4d398bdbc5e54e72a3c` |
| `stage3-dataset-v3-20260721-r1/dataset_manifest.json` | `8448ebe788b4a4bb5bd3803e4e64841bf39f3867f711d3198de31f1fb283ada0` |
| v3 `qualification_report.json` | `f839cb42f9e9abbe8f5682b015f1b984cdc418e80deba8205b715e3e46367f18` |
| v4 observation `dataset_manifest.json` | `bbf1c18b3bfad8a184e9b2c03725e34320a56ac3dbb0077b9317b5783aab157e` |
| 120-run 完整分布 `retention_manifest.json` | `e0b644fcc4b816f571c7e23304fb206336be32eeaabc661297c63e2a32be3815` |
| time-series 历史目录 `retention_manifest.json` | `7542068c4981062418e8b9709e9799397abf59d2906ff50a625f1d2a76e9d93b` |
| observation-error retention manifest | `55e6e926c255d363715cddea8e21649312f22c85f33f91317de243d5885eb8fe` |
| CV+Ridge replay retention manifest | `422d847db042aeea4d31a33d7ff369efb819aca3817099ec8272111d214f764d` |
| processor/association decision retention manifest | `1c20d4bdb11dcad46424e63ea9a9e48df3ab3c046e84ad21c1a35b0b2366ee8f` |

## 阶段 4：当前状态观测器规格与验收矩阵

### 4.1 证据允许的观测器边界

前三阶段只支持先做“相机射线域、匿名、因果、显式缺失”的当前状态观测器：

- 输入是每个曝光时刻的完整无序 PnP candidate set 和真实 timestamp。
- 连续状态先限制为 `[u,v,du/dt,dv/dt]`；depth、PnP yaw 和 quality 旁路保留。
- handle 只在短时连续段内匿名有效，不是物理 `relative_slot`。
- 输出显式分开 angular、depth、freshness、availability、set ambiguity、association、transform 和 applicability uncertainty。
- physical identity、world state、future prediction 和 fire-control eligibility 在本阶段固定为 false/unresolved。

完整字段和禁用字段见 `observer_specification.md`。这不是对当前 11 维 `YpdAngleTracker` 的直接修改。

### 4.2 当前 tracker 为什么不能直接称为证据接受的观测器

当前实现保留 11 维车辆中心/速度/yaw/半径状态、NIS/物理 gate、几何恢复和 covariance telemetry，具有生产基线价值；但它还存在与现有证据不一致的合同：

- `dt>100 ms` 会被替换成 `6 ms`，真实长 gap 被隐藏。
- tracking/lost 按帧计数，当前 `LOST_THRESHOLD=50` 无法跨不稳定 latest-only 帧率解释为固定时间。
- primary observation 和 slot selection 依赖 hard identity，而 observation-only association 尚未验收。
- Q/R、0.711 NIS reference 和 covariance 没有在 retained repeat-held 完整分布上做 coverage calibration。
- 米制 match/jump gate 会受到阶段 2 已证 depth long tail 影响。

所以本阶段把它登记为“保留的实现基线”，没有删除，也没有把现有 telemetry 升格为校准概率。

### 4.3 初版状态机和 timeout

规格状态为 `NO_DATA -> ACQUIRING -> OBSERVED_ANONYMOUS`，并显式包含 `AMBIGUOUS_SET`、`STALE`、`REACQUIRING` 和 `INVALID_STREAM`。

- 继承 Stage3 v3：最近 `0.2 s` 至少 8 个 valid events，latest age 不超过 `50 ms`。
- `50 ms` 是初版观测资格边界，不是已经证明的火控阈值。
- age 超过 50 ms 立即撤销 qualified output；旧值只保留 telemetry。
- stale 后的新事件建立新 ephemeral handle，并从头满足 8 events/0.2 s，禁止硬接旧 physical ID。
- `>4` candidate、close assignment/crossing、epoch change、timestamp regression 和 sink failure 分别进入 ambiguity 或 fail-closed。
- 历史 cyclic selection 的 continuity gate 一致选到 2 deg，但 timeout 在 0.25--2 s 之间变化；这只支持 cyclic topology，不支持拿任一 timeout 部署。

### 4.4 不确定性结论

当前不能宣称 calibrated covariance。初版必须先输出可审计分量；只有在独立 held-run 上对 50/80/90/95/99% 区间逐 motion、distance、radius、facing、candidate count、gap 和 quality 验证 coverage 后，才允许 `covariance_valid=true`。安全区间的一侧 coverage 置信下界必须达到 nominal level，否则扩大区间或声明不适用。

### 4.5 验收覆盖

`observer_specification.md` 和 `observer_acceptance_registry.json` 定义 A--F 六组门禁：

1. exact-key、truth stripping、future perturbation 和 irregular-time causality；
2. candidate permutation、zero candidate、`>4` ambiguity 和 hard-identity guard；
3. 50 ms freshness、全部 2,974 个 missing streak、两轮零 observation 和重新获取；
4. condition-wise paired current-state non-inferiority、angular/depth 分离、full tail/worst-run/worst-slot；
5. nominal/condition-wise uncertainty coverage 和 covariance fail-closed；
6. deterministic replay、native latency完整分布、failure injection、资产保留和 simulator boundary。

性能 tolerance 和 native latency budget 仍需在阶段 5 实现前预注册。历史 oracle-identity CV+Ridge 的 `0.310/0.717 deg` P95/P99 只作离线 future-horizon 参考，不能直接当当前观测器上线门槛。

### 4.6 本阶段非声明

- 没有实现新观测器。
- 没有接受 physical armor ID、车辆中心/半径状态或 world-frame history。
- 没有接受 detector confidence、PnP residual、NIS 或现有 covariance 为概率。
- 没有恢复 predictor 或 multi-hypothesis tracker。
- 没有修改生产 `RobotEstimator`、模拟器、SDK、Release 或 fire control。

## 阶段 5：命中相关误差传递与真值替换消融

阶段 5 没有实现生产预测器，而是补齐此前缺失的因果归因：

1. 用 exact projected corners 替换 actual corners，但保持 IPPE 和坐标链不变；exact PnP 的 3D P95 为 `0.009 mm`，证明当前 solver 在理想角点下可以闭合。
2. 将 PnP 误差拆成 LOS radial、LOS transverse，以及投到真值深度平面的水平/垂直毫米偏移；在 77,518 条已配对、truth-visible 行上，四类运动的径向 P95 为 `0.81--1.01 m`，水平 P95 只有 `6.1--7.5 mm`。
3. 固定观测时间、缺失、可见弧和 oracle slot，构造 `p_alpha=truth+alpha*(PnP-truth)`，完整扫描 0/10/25/50/75/100% 剩余 PnP 残差。
4. 用 16 点、真实时间的 simple-CV 基线比较 stationary/translation/rotation/combined 在 0/50/100/200 ms 的当前状态和未来横向误差。
5. translation 的当前 PnP P95 为 `11.9/15.4/25.2 mm`；rotation 为 `12.5/20.3/37.2 mm`；combined 为 `56.1/110.6/183.8 mm`。combined 在 100/200 ms 即使 exact PnP 仍失败，说明剩余限制属于运动模型/可见弧，而不能由角点优化单独解决。

完整叙事、样本数、55 mm yaw gate、小装甲板宽高代理和边界见 `corner_pnp_state_estimation_research_narrative.md`；204,768 条逐预测样本、180,289 条 PnP 方向样本和全部 PNG/SVG/PDF 位于 `D:\仿真\runtime\autoaim-b-hit-oriented-ablation-20260810-r1`，由 `hit_oriented_ablation_registry.json` 和 retention manifest 锁定。

## 下一步

阶段 5 先等待用户验收。若明确继续，优先补真实板面坐标、弹道/系统延迟、非 oracle association 和高角速长时域覆盖；随后才选择 combined 的因子化/IMM/多假设模型。阶段 4 的 truth-stripped reference observer 和 A--F acceptance harness 仍未实现；在它通过前，不接入在线 `RobotEstimator`，不恢复生产预测器或 fire control。
