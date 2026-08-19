# 自瞄 B 当前状态观测器规格（证据收敛稿）

- 规格版本：`autoaim-observer-spec-v1`
- 日期：2026-08-10
- 状态：`设计与验收合同；Python 因果匿名参考实现已落地，尚未部署`
- 范围：角点、PnP 和 PnP 后因果时序证据收敛为当前状态观测器
- 暂停项：轨迹预测器、多假设 C4 身份 tracker、火控接入

## 1. 结论先行

现有证据只支持先建立一个**相机射线域、匿名、因果、显式缺失的当前状态观测器**。它可以报告当前观测方向、局部角速度、时间新鲜度和分层质量状态，但不能声称已经恢复车辆中心、稳定物理装甲板 ID、校准的三维协方差或未来轨迹。

初版结构固定为四层：

1. 完整保存每个曝光时刻的无序 PnP 候选集合。
2. 只在 observation domain 建立短时匿名 handle，不把帧内 index 当身份。
3. 以真实 timestamp 估计当前 `u/v` 与局部变化率；深度单独保留，不让深度长尾主导角向关联。
4. 对缺失、歧义和过期显式降级；不满足条件时输出“不可用/身份未解析”，而不是生成看似连续的物理轨迹。

这是一份设计和验收合同，不授权修改当前 `RobotEstimator`、恢复预测器训练或接入火控。

Python 参考实现、历史 120-run 结构回放和 Linux 1.3.1 完整 detector/repair/PnP sidecar 结果见 [因果观测器与角点修复部署域复核](causal_observer_and_corner_repair_validation.md)。

## 2. 证据边界

### 2.1 已接受事实

- 角点阶段证明四角点误差和 refinement 变化不对称，当前 refinement 不是已证最优方案。
- PnP 阶段证明 planar depth conditioning 形成结构化长尾；角向误差小不等于三维位置正确。
- PnP 后时序阶段证明在线输入是无序、可为空、候选数可变的 frame-local 集合，且真实事件时间不均匀。
- 120 轮中有 189,158 个 truth frame、184,879 个 observation 精确键连接、184,763 个可用 exact-truth 连接、177,483 个有效事件和 250,449 个候选。
- 有效事件典型间隔约 8 ms，但缺失段可达数百毫秒；combined 还保留 15.02 s 的整轮无有效观测失败。
- simple hard association、learned pair cost 和 single cyclic hard identity 都没有达到部署身份要求。
- oracle identity 条件下，因果 CV + `u/v` Ridge/hold replay 的 P95/P99 为 `0.310/0.717 deg`；这只是离线处理上界，不是完整部署结果。

### 2.2 本规格不使用的捷径

- 不把 truth、`relative_slot`、target phase、future truth 或 simulator motion command 作为在线输入。
- 不用 nearest frame、插值或固定 5 ms 网格补齐丢失曝光。
- 不把 `observation_index`、detector number、类型/颜色签名或当前 `tracked_id` 当作可信物理身份。
- 不把 detector confidence、PnP reprojection residual 或当前 EKF covariance 直接解释成概率。
- 不因候选数超过 4 就静默裁剪，也不把空 observation 填成零位置。
- 不把 camera-frame `u/v` 证据冒充 exposure-matched world-frame 证据。

## 3. 当前生产 tracker 的取证审计

当前 `RobotEstimator/YpdAngleTracker` 是必须保留的历史和生产基线，但不是本规格自动接受的观测器：

| 项目 | 当前实现 | 证据判定 |
| --- | --- | --- |
| 状态 | 11 维中心/速度/高度/yaw/角速度/半径几何状态 | 有实现，不等于已由现有 PnP 数据可辨识 |
| 输入选择 | 按装甲板类型/number 过滤并选 primary observation | 依赖 hard identity，和阶段 3 结论冲突 |
| 时间 | `dt<=0` 或 `dt>100 ms` 时替换成 `6 ms` | 会隐藏真实长 gap；不能作为新合同 |
| 状态机 | `LOST/DETECTING/TRACKING/TEMP_LOST`，检测和丢失按帧计数 | latest-only 且帧率不固定，帧数 timeout 不可迁移 |
| 当前阈值 | `_trackingThreshold=2`、`LOST_THRESHOLD=50` | 尚未在完整 missing-streak 分布上校准 |
| 匹配/跳变 | 0.2 m match、0.8 rad yaw、0.15 m observation jump 等 | 历史启发式；阶段 2 深度长尾会影响米制 gate |
| 测量噪声 | yaw/pitch 固定项，distance/yaw 用对数启发式 | 没有 retained repeat-held coverage calibration |
| NIS | 参考阈值固定为 0.711，未随维度变化 | 可作 telemetry，不可声称统计校准 |
| divergence | 半径、物理 jump、近期 NIS failure 和几何恢复 | 有 fail-closed 价值，但还没有正式数据集 acceptance |
| covariance | 暴露 `P` 和诊断字段 | 存在数值不等于 coverage 正确 |

因此阶段 4 不删除或重写当前 tracker，而是把“当前实现”“证据支持的合同”和“未来需要校准的项目”分开登记。

## 4. 观测器输入合同

### 4.1 帧级必需字段

每次调用接收一个完整 frame event：

```text
session_id / runtime_instance_id
producer_epoch
frame_seq
capture_timestamp_ns
observation_sink_status
gimbal_pose + gimbal_pose_exposure_match_status
camera_to_gimbal_extrinsic_id
candidates[]
```

规则：

- `producer_epoch + frame_seq + timestamp` 必须单调且不跨 epoch 拼接。
- source sequence 跳号不是异常修补条件，必须记录为 gap evidence。
- sink queue overflow、I/O failure 或 schema mismatch 进入 fail-closed 状态。
- world/tracker transform 若没有 exposure-match 证明，只能输出 camera-frame 观测。

### 4.2 候选级必需字段

每个 candidate 至少保留：

```text
valid
camera_tvec_m = [right, down, forward]
u_deg = degrees(atan2(right, forward))
v_deg = degrees(atan2(down, forward))
observed_yaw_rad
raw/refined-corner provenance or fallback status
PnP candidate/residual diagnostics when present
detector color / number / type / confidence as non-probabilistic metadata
frame_local_observation_index
```

候选入场最低条件是 `valid=true`、三维 tvec finite 且 `forward>0`。其余 quality 字段用于分组、诊断和以后校准；在没有 held-out coverage 证据前不得自动组合成“置信度”。

### 4.3 明确禁止的在线字段

```text
truth target_id
truth relative_slot
truth phase / center / velocity
future truth or future observation
simulator scene motion command
offline assignment result
repeat id used to select a model
```

离线评分工具必须在调用观测器前剥离这些字段，并在观测器返回后才重新连接 truth。

## 5. 输出和内部状态

### 5.1 输出必须分层

观测器每帧输出：

```text
observer_status
status_reason[]
observation_timestamp_ns
age_ns
frame_availability
candidate_count / valid_candidate_count
anonymous_handles[]
set_ambiguity_status
camera_frame_only
physical_identity_status = unresolved
uncertainty_status
downstream_eligibility
```

每个匿名 handle 输出：

```text
ephemeral_handle_id
u_deg, v_deg
du_dt_deg_s, dv_dt_deg_s when qualified
raw_camera_tvec_m
depth_status
history_event_count
history_span_s
max_gap_s
last_gap_s
local_continuity_residual_deg
measurement_quality_features
angular_uncertainty_interval when calibrated
```

`ephemeral_handle_id` 只在连续短历史内定位同一局部轨迹，不具有物理 slot 含义，也不得跨 reacquisition 复用。

### 5.2 初版状态只包含可观测部分

初版允许的连续状态：

```text
x_ray = [u, v, du/dt, dv/dt]
```

其中 `u/v` 来自最新有效 PnP ray，变化率只能用过去和当前事件的真实 timestamp 估计。允许保留 raw depth、observed yaw 和 PnP quality 作为旁路字段，但初版不把它们并入物理中心/半径状态，理由是：

- depth P95 误差约 `0.83--0.86 m`，明显不是小高斯噪声。
- observed PnP yaw 对历史 Ridge 增益很小，却会伤害简单身份关联。
- 当前候选 identity 不稳定，车辆中心/半径状态对错误 slot 极敏感。

### 5.3 历史缓存

- 按真实 timestamp 保存最近最多 200 个 valid events，与 Stage3 v3 合同一致。
- 空 frame 不占 valid-event 槽，但 availability 和 elapsed gap 必须另行累计。
- history 不能跨 session、producer epoch、sink failure 或重新获取边界。
- 不构造固定采样网格；需要等间隔算法时只能在内部显式建模不规则时间，不能改写原始事件。

## 6. 不确定性合同

### 6.1 在校准完成前输出分量，不输出伪概率

初版不确定性由以下可审计分量组成：

1. `angular_empirical`: 基于 held-out run 的角向误差经验分布或区间。
2. `depth_empirical`: 与角向分开记录的深度误差分布。
3. `freshness`: 最新有效事件年龄、history span、max gap。
4. `availability`: 当前 frame 是否存在、是否 zero candidate、missing streak 状态。
5. `set_ambiguity`: candidate count、`>4`、多项 continuity cost 接近等。
6. `association`: 初版固定为 `physical_identity_status=unresolved`。
7. `transform`: camera/gimbal/world exposure-match 状态。
8. `model_applicability`: motion/distance/facing/quality 条件是否在证据支持范围内。

只有在独立 repeat-held 条件上验证 nominal coverage 后，才允许设置 `covariance_valid=true`。在此之前可以输出 covariance 数值用于 debug，但必须标 `uncalibrated`，下游不得据此计算发射概率。

### 6.2 校准验收原则

- 50/80/90/95/99% 区间分别报告 empirical coverage 和一侧置信下界。
- 按 motion、distance、radius、facing、candidate count、gap band 和 corner/PnP quality 分层。
- 同时报完整 per-sample interval/score 分布，不能只报平均 coverage。
- 安全区间的置信下界必须达到 nominal coverage；否则扩大区间或声明不适用。
- depth 和 angular 分开校准；不能用一个三维 covariance 掩盖 radial heavy tail。
- detector confidence 和 current NIS 必须经过单独 calibration 才能进入概率融合。

## 7. 门禁顺序

门禁必须按顺序执行，并保存 rejection reason；后级不得覆盖前级失败。

### G0：流和 schema

- schema/version 可识别。
- session/epoch/sequence/timestamp 合法且单调。
- sink 没有 overflow/I/O fail-closed。
- frame key 不重复。

失败结果：`INVALID_STREAM`，清空匿名动态状态。

### G1：候选完整性

- 保留完整 candidate set。
- candidate tvec finite、forward positive。
- zero candidate 显式记为 availability negative。
- `>4` 标记 `AMBIGUOUS_SET`，不得裁剪后继续进入四槽几何。

失败结果：无当前 measurement update，但 gap 继续计时。

### G2：时间新鲜度

- 用真实 timestamp 计算 age 和 gap。
- 当前接受的资格基线是最近 `0.2 s` 至少 8 个 valid events，latest valid age 不超过 `50 ms`。
- 这两个值继承自 Stage3 v3 causal qualification，是**初版资格边界**，不是已经证明的火控安全阈值。

失败结果：`ACQUIRING` 或 `STALE`，不得输出 qualified dynamic state。

### G3：匿名局部连续性

- cost 只使用 observation-domain 过去状态、真实 `dt` 和当前 `u/v`。
- 禁止 truth slot、PnP depth-only identity 和 detector number hard identity。
- 历史 cyclic sweep 的 `2 deg` continuity gate 在各 repeat 中相对稳定，但 reacquisition timeout 从 `0.25--2 s` 不稳定；两者都只作为 diagnostic evidence，不直接成为部署常量。
- 若多个 assignment 代价接近、集合基数变化或出现 crossing，输出 ambiguity，不强行选择一个物理 ID。

### G4：观测器适用性

- history count/span/gap 满足条件。
- 当前方法在对应 motion/distance/facing/quality support 内。
- 经验 uncertainty 有效；否则只能输出 raw observation/hold diagnostic。

### G5：下游资格

当前阶段恒定规则：

- 匿名 current-ray 状态可以标记 `anonymous_current_state_valid=true`。
- `physical_identity_resolved=false`。
- `world_state_valid=false`，除非以后有 exposure-matched transform 独立验收。
- `prediction_valid=false`、`fire_control_valid=false`。

## 8. 状态机、timeout 和重捕获

### 8.1 状态定义

| 状态 | 含义 | 允许输出 |
| --- | --- | --- |
| `NO_DATA` | 尚无有效候选或整轮零观测 | availability telemetry |
| `ACQUIRING` | 有事件但不足 8 events/0.2 s | raw anonymous rays；无 qualified rate |
| `OBSERVED_ANONYMOUS` | 通过 G0--G4，最新事件年龄不超过 50 ms | 当前 anonymous ray state |
| `AMBIGUOUS_SET` | `>4`、assignment 接近、集合 crossing 或身份不确定 | 完整候选集和 ambiguity；无 hard identity |
| `STALE` | 最新有效事件年龄超过 50 ms | hold diagnostic only；downstream invalid |
| `REACQUIRING` | stale/ambiguity/epoch 后出现新候选 | 新 handle，从头积累资格历史 |
| `INVALID_STREAM` | epoch/key/schema/sink failure | fail closed，清空状态 |

### 8.2 初版 timeout 政策

- `T_fresh = 50 ms`：继承 Stage3 v3 资格门。超过后立刻撤销 qualified output，进入 `STALE`。
- 初版不允许 stale history 继续生成可用速度或身份；只能保留最后值做 telemetry。
- stale 后第一个有效事件进入 `REACQUIRING`，生成新的 ephemeral handle，不与旧 handle 硬连接。
- 重新满足“最近 0.2 s 至少 8 个有效事件”后才回到 `OBSERVED_ANONYMOUS`。
- session/producer epoch 改变、timestamp regression、sink fail 或 schema mismatch 直接进入 `INVALID_STREAM`，不等待 50 ms。

这比历史 `LOST_THRESHOLD=50 frames` 更保守，但不会把 latest-only 的帧率变化误当时间。未来若要允许 50 ms 以上 coast，必须在完整 2,974 个 missing streak、两轮零观测和独立 holdout 上预注册时长、误接率与全链尾部；当前没有证据接受该扩展。

### 8.3 重捕获原则

- 重捕获是新局部轨迹的建立，不是旧 physical ID 的自动恢复。
- 新旧 handle 可以在离线 truth 下评分，但 runtime 不共享 truth 结果。
- 必须报告 reacquisition latency、false stitch、handle fragmentation 和拒绝率。
- 在多假设身份模块恢复并验收前，重捕获后的物理身份始终为 unresolved。

## 9. 验收矩阵

### 9.1 数据和因果性

| ID | 测试 | 通过条件 |
| --- | --- | --- |
| A01 | exact-key replay | 连接结果与四字段 exact join 完全一致；nearest/interpolation 调用数为 0 |
| A02 | duplicate/regression | duplicate、timestamp/sequence regression 和 epoch change 全部 fail closed |
| A03 | truth stripping | truth/slot/future 字段随机化或删除不改变 runtime 输出 |
| A04 | future perturbation | 修改 anchor 后数据不改变 anchor 时刻输出 |
| A05 | irregular time | 改变 event 间隔但保持值相同会按真实 `dt` 改变 rate；不得退化成 index time |
| A06 | empty frame | 不产生 valid event，但 gap、availability 和 state transition 正确更新 |

### 9.2 集合和身份边界

| ID | 测试 | 通过条件 |
| --- | --- | --- |
| B01 | candidate permutation | 同帧候选顺序任意置换，输出集合等价 |
| B02 | frame-local index | 每帧重编号不改变 observation-domain 结果 |
| B03 | zero candidate | 输出 visibility/availability negative，不生成零位置 |
| B04 | `>4` candidate | 120-run 的 18 帧和 formal 360 的 38 帧全部标 ambiguous，不静默裁剪 |
| B05 | hard identity guard | 初版所有输出 `physical_identity_resolved=false` |
| B06 | crossing/close cost | 无唯一局部 assignment 时拒绝或保留集合，不强制单 ID |

### 9.3 timeout 和重捕获

| ID | 测试 | 通过条件 |
| --- | --- | --- |
| C01 | 50 ms freshness | age 从 `<=50 ms` 跨到 `>50 ms` 时 qualified output 单调撤销 |
| C02 | full missing distribution | 全部 2,974 个 missing streak 都能重放且 state transition 与定义一致 |
| C03 | zero-observation runs | 两轮零 observation 全程保持无 qualified output |
| C04 | long gap | 100/200/500 ms 和 15.02 s gap 后不能沿用旧 ephemeral identity |
| C05 | reacquisition | 只有重新达到 8 events/0.2 s 才恢复 qualified anonymous state |
| C06 | epoch reset | producer epoch 改变立即清空 history 和 handle |

### 9.4 精度和完整分布

| ID | 测试 | 通过条件 |
| --- | --- | --- |
| D01 | raw measurement reference | 原始 angular/depth 完整经验分布与阶段 2/3 hash authority 一致 |
| D02 | current-state non-inferiority | 每个 motion/distance/radius/facing 条件分别对 raw baseline 做 paired held-run 比较；不得用总体均值掩盖退化 |
| D03 | tail reporting | P50/P90/P95/P99/max、worst-run、worst-slot 和完整逐样本分布同时保留 |
| D04 | depth separation | angular 和 depth 分开报告；不得只报三维平均距离 |
| D05 | identity upper bound | oracle identity 结果明确标 upper bound，不能进入部署结论 |

D02 的具体数值容差必须在实现前预注册。现有 `0.310/0.717 deg` 是未来时域、oracle identity、离线 Python replay 的 P95/P99，只能作参考，不能直接作为当前观测器上线门槛。

### 9.5 不确定性

| ID | 测试 | 通过条件 |
| --- | --- | --- |
| E01 | nominal coverage | 每个 nominal level 的一侧置信下界达到 nominal coverage，否则标 uncalibrated |
| E02 | condition coverage | motion/distance/facing/gap/candidate-count 分层均报告，不允许只报 pooled coverage |
| E03 | monotonicity | gap/age 增大或 quality 降低时 uncertainty 不得反向缩小而无证据解释 |
| E04 | covariance validity | 未通过 E01/E02 时 `covariance_valid=false` |
| E05 | detector/NIS calibration | 使用 detector confidence 或 NIS 前必须有独立 reliability/coverage 证据 |

### 9.6 鲁棒性、延迟和回归

| ID | 测试 | 通过条件 |
| --- | --- | --- |
| F01 | finite/PSD | 所有 qualified 连续输出 finite；若输出 covariance 则 symmetric PSD |
| F02 | deterministic replay | 相同输入、配置和 source hash 得到 bitwise 或声明容差内一致结果 |
| F03 | latency distribution | native runtime 报 P50/P95/P99/max 和丢帧；离线 Python `272.65 us P99` 不冒充生产结果 |
| F04 | failure injection | queue overflow、I/O fail、schema mismatch、timestamp regression 均 fail closed |
| F05 | protected assets | 不删除 raw、失败、invalid、superseded、模型或 Release 资产 |
| F06 | simulator boundary | 不修改 simulator、SDK 或正式 Release；若发现问题先走审批门禁 |

## 10. 当前可接受方案与未决参数

### 10.1 当前接受为设计基线

- camera-frame `u/v` 和真实 timestamp。
- 完整无序 candidate set 与显式 missing/ambiguity。
- 匿名 ephemeral handle，而非物理 slot。
- 8 events/0.2 s 和 50 ms freshness 作为初版 qualification。
- stale 后 fail closed 并从头 reacquire。
- angular/depth/association/availability/transform 分层 uncertainty。
- full distribution、worst condition、zero-observation runs 和失败资产共同验收。

### 10.2 仍未接受的数值或模块

- continuity/assignment gate 的部署阈值。
- 50 ms 以上 coast 的最大时长。
- 经验 uncertainty 的条件分箱和最终 interval width。
- detector confidence、PnP residual、candidate separation 到概率的映射。
- world/tracker frame exposure-match 合同。
- physical identity、车辆中心/半径状态和 C4 belief。
- native latency预算和任何 fire-control threshold。

这些项目不是遗漏，而是现有文档没有足够证据支持固定值。实现前必须预注册并用保留数据/新独立数据验收。

## 11. 阶段完成条件和下一步边界

阶段 4 完成意味着：输入、输出、状态、门禁、timeout、重捕获、uncertainty 和验收矩阵已经有可审计合同；不意味着观测器已经实现或可部署。

若用户继续批准阶段 5，只允许先做：

1. 离线、只读、truth-stripped reference observer。
2. 上述 A--F deterministic acceptance harness。
3. 完整 120-run 和 formal 360 replay 报告。

在阶段 5 单独验收前，仍不得接入在线 `RobotEstimator`、恢复预测器、实现多假设身份 tracker 或进入火控。
