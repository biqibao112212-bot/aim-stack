# Linux 1.3.1 因果观测器与角点修复部署域复核

## 当前结论

本轮完成了三件事：

1. 实现了一个保留完整无序候选集、显式 missing/ambiguous/stale、只输出匿名当前射线的因果观测器参考实现。
2. 打通 Linux 1.3.1 `raw RGBA -> offline ONNX detector -> frozen v3 repair -> unchanged IPPE -> anonymous observer -> truth-after-return scoring` 完整旁路。
3. 证明既有 v3 repairer 的图像域 sealed 通过不能直接换成部署 PnP 通过；在新会话中 stationary 有效，spin/combined 存在系统性过修正。

因此当前授权边界是：

- 因果匿名观测器可继续做 offline/shadow 研究；
- v3 repairer 保留为冻结 proposer，不得改变生产 PnP 输入；
- 二级 benefit gate 的线性模型和小型 MLP 均未达到开启新验证的条件，安全 fallback 为 reject-all；
- sealed test 保持关闭，预测器和火控仍未授权。

## 为什么单帧网络会在不同运动会话上表现不同

repair CNN 的输入确实只是当前图像和 raw 四边形，不使用时序、motion mode 或未来真值。理论上，只要“装甲正面特征仍可见，detector 已经给出对应 raw 四角”，运动类型只是产生不同视角，充分覆盖后应当能够泛化。

当前数据的问题是 motion 与 view 强共线：

- validation stationary 主要是窄、侧视四边形，v3 倾向扩宽，横向误差改善；
- spin/combined 出现更正面、更宽的四边形，v3 仍按旧域改变宽度/面积，导致 IPPE 深度和分支跳变；
- linear d165 会话中目标正面数字/双灯条几乎不可见，detector 只检出背景悬挂装甲板，repairer 没有目标 raw corners 可修。

所以“运动影响结果”不表示 CNN 偷看了时序，而是当前采集设计把运动、视角、尺度、姿态和可观测性绑在了一起。下一轮数据必须要做 view-motion 交叉覆盖，不能把 mode 名字当网络输入来走捷径。

## 因果匿名观测器

参考实现在 `scripts/causal_ray_observer.py`，主要合同是：

- 输入为每帧完整候选集，不用 truth slot 和 future；
- 候选数 `>4` 直接 `AMBIGUOUS_SET`，不静默丢弃；
- 空帧推进 freshness，但不伪造观测 event；
- 初版资格为最近 `0.2 s` 至少 8 events，latest age `<=50 ms`；
- 候选排列不改变集合结果，close-cost 不强行输出单一身份；
- 输出恒为 `physical_identity_resolved=false`、`prediction_valid=false`、`fire_control_valid=false`。

9 个单元测试覆盖候选排列、0/5 候选、close cost、50 ms 过期、重获、epoch/sequence/timestamp/sink fail-closed、future perturbation 和 truth stripping。

在历史 120-run 开发数据上：

- 189,158 frames，184,879 observation frames；
- zero-candidate frames 11,675，`>4` frames 18；
- 2,974 missing streaks，最长缺失 15.014 s；
- `OBSERVED_ANONYMOUS` fraction 87.79%，ambiguous 0.63%，stale 1.41%。

这些只是结构合同和历史 Windows/Simulator 1.2.1 回放证据，不是 Linux 生产延迟或部署验收。

## Linux 1.3.1 完整旁路

本轮 sidecar 每个收到的 frame 都保存：

```text
exact frame identity + image/hash
  -> complete post-NMS detector candidates (including zero-candidate frames)
  -> raw/model-proposed/selected corners and reason
  -> raw and selected IPPE candidates
  -> anonymous observer status
  -> observer return 后才连接 exact truth 评分
```

offline detector 旁路记录 model/script hash、score/NMS threshold、decode logits、candidate rank 和 runtime version。repair/PnP 旁路记录 frozen checkpoint hash、`raw/proposed/selected`、reliability、全部 IPPE 解与 failure reason。这足以做确定性 offline shadow，但仍没有证明 ONNX Runtime 与部署 TensorRT detector 逐候选等价，也不代表 native async drop/latency。

8 个有效开发/诊断验证会话的评分结果中，预注册验证门失败：

| validation mode | raw/repaired observer availability | angular P95 deg | radial P95 mm | transverse P95 mm |
| --- | ---: | ---: | ---: | ---: |
| combined | 58.2% / 56.0% | 0.151 / 0.389 | 2024.5 / 1758.5 | 13.82 / 38.06 |
| spin | 93.6% / 93.6% | 0.126 / 0.268 | 604.6 / 607.3 | 11.06 / 25.48 |
| stationary | 98.1% / 98.1% | 0.155 / 0.102 | 358.5 / 344.8 | 14.60 / 9.01 |
| linear d165 | 0% / 0% | no match | no match | no match |

linear d165 的 358 帧可与 1,432 条 exact labels 精确连接，但 327 帧零候选，32 个候选全部落在背景悬挂板，到目标 truth 的最小 ordered RMS 为 134.8 px。这条会话是真实端到端 availability 失败，但不可作为角点 repair conditional quality 的负样本。

## 为什么原 v3 reliability 失败

v3 reliability 的标签是 `raw visual corner RMS >= 4 px`。它回答的是：

> 原始角点是否可能很差？

部署真正需要的问题却是：

> 当前这一个具体修正提议，经 PnP 后是否会更好？

“raw 很差”不推出“模型的修正方向正确”。因此 v3 能在 spin/combined 上高置信地过修正；单看 reliability 或四点 PnP reprojection error 都不能证明真实姿态改善。

## 二级 benefit gate 实验

二级 gate 只能 veto，不能 rescue：

```text
final_apply = frozen_v3_apply AND benefit_probability >= threshold
```

runtime 特征共 67 维，包括 raw 15D 四边形、归一化 8D proposal、detector/reliability、raw/proposal 反事实 PnP 变化、IPPE 双解间隔、宽高面积变化，以及严格来自过去 raw history 的连续性残差。不含 truth、mode、session、frame identity 或 future。

离线标签为 `BENEFIT/HARM/UNCERTAIN`，临时使用 0.5 mm transverse margin；在完成同姿态重复测量噪声标定前，该 margin 不是部署常数。`UNCERTAIN` 训练权重为 0，runtime 默认回退 raw。

四个 train session 按整会话 leave-one-session-out，三种子 ensemble 结果：

- 线性 gate 找不到同时满足尾部和错误放行门的非空解；
- MLP 的旧版只放行 11 次，session-macro transverse P95 改善 0.31%，但 spin 放行 5 次全部为有害修正；
- 增加 `benefit precision >=80%`、`harm apply <=5%`、至少 10 次 OOF 放行、session-macro transverse P95 至少改善 1% 后，linear 和 MLP 均被拒绝；
- 当前决策为 `selected_by_development_oof=reject_all`，不得开启 sealed test。

这个负面结果还说明：只看 P95 可能忽略少量但真实有害的放行，所以必须同时报 benefit precision、harm-apply fraction、coverage 和 session macro 结果。

## 证据与下一步

主要保留证据：

- 历史观测器合同：`runtime/causal-ray-observer-contract-v1-r2-20260819`；
- Linux 会话旁路：`runtime/linux-observer-dev-validation-analysis-v1-20260819`；
- 部署域评分：`runtime/linux-repair-observer-validation-score-v3-20260819`；
- benefit gate 最终开发证据：`runtime/corner-repair-benefit-gate-development-v4-20260819`；
- 候选 gate 模型（仅保留，未授权）：`models/engines/corner-repair-benefit-gate-development-v4-20260819`。

下一轮不是继续在当前 validation 上调阈值，而是：

1. 采集 motion/view/scale 交叉覆盖的多会话 train 数据，并单列 detector visibility failure；
2. 用同姿态重复采集确定 BENEFIT/HARM 噪声 margin；
3. 先比较确定性 veto、正则线性 gate 和小 MLP，不训练更大网络；
4. 方法和阈值冻结后再采全新 validation；
5. 只有新 validation 通过才能开启 sealed test，然后再做 repaired-corners -> unchanged IPPE -> frozen 400 ms predictor 同锚点 A/B。
