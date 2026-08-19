# Linux 1.3.1 角点修复到局部预测的同曝光 A/B

## 1. 这次实验要回答什么

本实验不重新选择预测网络，而是验证一条更基础的因果链：

```text
detector raw corners -> [raw / frozen repair] -> same IPPE -> same coordinate chain
                     -> same frozen 400 ms LOS rigid expert -> 50/100/200 ms error
```

两条主支路只允许角点不同；IPPE 模板、candidate 选择、曝光身份、坐标变换、历史窗口、运动模型和评分时刻均保持一致。因此 repaired 与 raw 的差异才能归因于角点修复。

评分主指标是 anchor tracker 坐标的 cross-depth 误差
`sqrt(error_y^2 + error_z^2)`；`55 mm` 只是诊断线，不是真实命中率。

## 2. 先修正评价链，再解释模型

第一版适配器有两个问题，它们会让“完美角点”也得到数百毫米误差：

1. 对所有装甲板强行使用理想 `135 x 55 mm` 模板。Linux 1.3.1 标签已给出资产测得的逐板 `object_corners_armor_m`，exact 闭合必须使用它；生产 raw/repaired 支路仍保持当前 nominal-small-armor IPPE，不改算法。
2. 只做了 OpenCV camera `[right, down, forward]` 到 `[forward, left, up]` 的固定换轴，没有应用曝光时刻的相机世界位姿。采集时云台会转动，漏旋转会将相机运动误认为目标高度和速度变化。

修正后的变换为：

```text
p_camera_opencv = [right, down, forward]
p_camera_bevy   = [forward, left, up]
p_world         = camera_position_world + R(camera_quaternion_world) * p_camera_bevy
p_tracker       = R(chassis_quaternion_world)^T * (p_world - gimbal_position_world)
```

修正原则是：如果 exact/oracle 基线不合理，先审计标签几何、身份、坐标和可用性，不用换网络来遮盖评价错误。

## 3. Linux 同曝光联合采集

Release 1.3.1 的 TCP 图像流在实测中是单消费者：Release Python collector 与旧 bridge 同时连接时，bridge 持续收帧，collector 超时且得到零帧。失败证据保留在：

`/home/potato/Projects/仿真/runtime/pose-parallel-smoke-20260819-v1`

因此消费者侧新增 SDK-only 单客户端采集器：

- `TcpImageClient` 只接收完整、校验过的 RGBA32 latest-only 帧；
- 收帧后立即用 `TalosMetadataReader::readExposureStateForFrame(frame_seq)` 取同曝光底盘/云台/相机世界位姿；
- 严格使用 `(producer_epoch, frame_seq, timestamp_ns)` 连接，不用邻帧补缺；
- 资格化脚本重算 raw payload SHA-256，生成 Release validator 可读的 identity ledger/capture manifest 及独立 exposure manifest；
- 不读取在线 target truth，不包含 future truth。

smoke 验证得到 `51/51` 同曝光位姿，Release validator 通过。随后的开发会话得到：

- `430` 个完整 RGBA 帧；
- `429` 个严格同曝光位姿，覆盖 `99.77%`；
- `402` 个标签曝光，`1608` 行exact-corner；
- 冻结 detector 在 `396` 个曝光上生成 `651` 条 `25 px` 关联门内记录。

该会话是 sealed test 之后的开发复核，不用于重新选择 `v8` 角点模型。

## 4. 冻结对象和实验条件

角点修复 checkpoint：

`/home/potato/Projects/仿真/models/engines/corner-repair-formal-linux-1.3.1-20260819-v8-match25-score055/corner-repair.pt`

SHA-256：`53e0cba9e2375a206b56eb16d1120edfe7e163763256828da7f51e5fe693ca27`。

结构是 `1.5x ROI -> RGB 128x64 -> context-spatial CNN + 15D geometry MLP -> 8D corner residual`，detector score 低于 `0.55` 时返回 raw。冻结预测器是：

- `400 ms` 局部 LOS 各向异性 Huber 拟合；
- depth weight `0.1`，Huber delta `20 mm`；
- `31` 次角速度慢状态中位数；
- `50/100/200 ms` 三个预测时域。

往返运动的 `motion_uniform=false` 保护区不作观测。根据同曝光线速度反向划分 segment，`400 ms` 历史不得跨越换向。这个门禁用于分离“观测误差”和“未观测端点导致的相位不可辨识”，不表示部署时已知换向真值。

## 5. 主结果

下表是不跨换向历史的同锚点结果，单位为 mm。

| 输入 | 时域 | n | mean | P50 | P95 | `<=55 mm` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact matched | 50 | 44 | 3.1 | 2.8 | 6.2 | 100.0% |
| exact matched | 100 | 38 | 4.3 | 4.2 | 7.7 | 100.0% |
| exact matched | 200 | 29 | 7.0 | 7.0 | 12.5 | 100.0% |
| raw | 50 | 44 | 111.1 | 88.3 | 300.9 | 31.8% |
| raw | 100 | 38 | 119.5 | 98.8 | 272.9 | 23.7% |
| raw | 200 | 29 | 147.8 | 130.5 | 325.4 | 24.1% |
| repaired | 50 | 44 | 59.4 | 38.4 | 152.0 | 61.4% |
| repaired | 100 | 38 | 68.2 | 55.7 | 166.0 | 50.0% |
| repaired | 200 | 29 | 84.5 | 71.7 | 189.5 | 41.4% |

repaired 相对 raw 的 paired mean 变化为 `-51.7/-51.3/-63.3 mm`，改善锚点比例为 `81.8%/81.6%/82.8%`，变差比例为 `18.2%/18.4%/17.2%`。

exact matched 的结果说明，在正确观测、不跨换向且物理身份已知的条件下，冻结局部预测器可闭合。raw/repaired 与 exact 之间仍有大幅差距，当前主瓶颈仍是 PnP 观测而不是新的时序网络。

## 6. 为什么像素改善小，PnP/预测改善大

在该会话的 `532` 条可用观测上：

- corner coordinate RMS：`9.86 -> 9.48 px`，改善 `3.83%`；
- PnP cross-depth observation RMS：`103.7 -> 77.8 mm`，改善 `24.95%`；
- PnP depth observation RMS：`414.5 -> 280.2 mm`，改善 `32.41%`。

`23.5%` 的记录出现“像素 RMS 变差，但 PnP cross-depth 变好”；反过来的比例是 `10.2%`，两种 delta 的相关系数只有约 `0.27`。

这不是矛盾。角点的独立欧氏像素距离没有表示四边形对透视姿态的联合约束。一组修正可能让某个角离标签稍远，却让四边形的透视形状更接近真实刚体，因而改善 IPPE 的 translation。这也说明下一代损失函数应同时关注 corner 和 pose，而不是只优化平均像素误差。

## 7. 失败的修补/门控方法

### 7.1 固定 shrinkage

将网络修正缩放为 `25%/50%/75%` 都不如完整 repaired。例如 50 ms mean/P95：

- `25%`: `98.4/268.7 mm`；
- `50%`: `84.9/228.1 mm`；
- `75%`: `71.8/188.6 mm`；
- `100% repaired`: `59.4/152.0 mm`。

因此当前不支持“网络普遍过度修正”这个解释。

### 7.2 单帧连续性门控

开发性 causal gate 用过去同一 slot 的两次选中位置做 CV 外推；repaired 的加权创新量比 raw 大 `20 mm` 时退回 raw。它选了 `204` 次 raw 和 `328` 次 repaired，但 50/100/200 ms mean 变为 `86.8/95.1/119.6 mm`，明显差于连续 repaired 流。

失败原因是逐帧择优目标与 `400 ms` 历史拟合目标不同。raw/repaired 切换本身会产生新的时间不连续；当前帧创新量小也不代表整段刚体拟合更好。

### 7.3 单帧 oracle-best

即使离线使用当前帧真值选择 raw/repaired 中 cross-depth 更小的一个，50/100 ms 也没有超过连续 repaired；它在 200 ms 略好（`80.0` vs `84.5 mm` mean）。这进一步说明局部单帧目标不足以定义时序最优门控。

上述三个方法均仅作研究对照，不进入生产。

## 8. 结论与部署边界

1. `v8` 角点修复在这条同曝光开发会话上能稳定传递到 PnP 和局部预测结果，不是“只有像素指标的假改善”。
2. 它仍不能上线：独立 sealed test 的 aggregate corner RMS 只改善 `2.55%`，未通过预声明 `5%` 强门；本次只有一条 post-test development 组合运动会话，不能替代 session-disjoint 正式验收。
3. 冻结 `400 ms` 预测器在 exact 恒向条件下通过局部闭合，但生产 observer 仍缺少无 truth 物理身份解析、missing/ambiguous 候选合同、换向相位可用性和真实 impact-time 时域。
4. 因此当前状态是：“角点到预测的离线因果链已闭合，角点修复和预测器部署验收均未完成”。

下一个有证据支持的研究方向是 pose-aware/multi-task 修复：保留 corner residual 监督，同时加入可微或数值稳定的 PnP cross-depth/depth 损失与时序一致性指标。新方法必须使用新的 session-disjoint train/validation，然后用未见 sealed test 验收，不得继续在已打开的测试上调参。

## 9. 受保护证据

- 同曝光采集：`/home/potato/Projects/仿真/runtime/corner-repair-pose-prediction-dev-v1-20260819`
- 最小主对照：`/home/potato/Projects/仿真/runtime/linux-corner-local-prediction-pose-aware-v1-20260819-r2`
- 含 shrinkage/causal/oracle 失败对照：`/home/potato/Projects/仿真/runtime/linux-corner-local-prediction-pose-aware-v1-20260819-r4`
- 首次错误坐标适配的保留证据：`/home/potato/Projects/仿真/runtime/linux-corner-local-prediction-v1-20260819-r2` 至 `r4`

所有采集、模型、manifest 和评价目录均为受保护资产，不自动覆盖或删除。
