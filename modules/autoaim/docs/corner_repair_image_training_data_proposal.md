# 图像角点修复器：精确标签采集缺口与模拟器接口提案

状态：等待 `SIMULATOR_CHANGE_APPROVAL_REQUIRED` 明确批准。本文只记录消费者侧复现与建议；在批准前不得修改模拟器仓库、SDK、正式 Release 或消费者版本锁。

## 1. 目标与范围

目标是训练只依赖单帧图像 patch 与同帧 YOLO 四角的联合修复器，输出四个角点的 8 维修正量。运动模式、时间、session、目标真值、PnP 和未来状态都不得作为网络输入。

本阶段只需要匀速运动曝光。平移端点、反向瞬间及其邻域不进入训练、验证和下游传播；它们也不属于本阶段性能目标。

## 2. 消费者侧复现

正式消费者锁定为 `daedalus-simulator 1.2.1`。Windows Release、TCP 图像、Release SDK、YOLO/PnP 消费链均未被模拟器源码替代。

本轮新增了环境变量门控的消费者原图证据钩子，并完成三次 smoke：

| 证据 | 数量/结果 |
| --- | ---: |
| 精确曝光身份与相机姿态行 | 1,541 |
| 保存原图 | 308 |
| `has_exact_exposure_truth=true` | 1,541/1,541 |
| `ground_truth.targets` 非空 | 0/1,541 |

原图、相机曝光姿态、pipeline 和 Stage3 行可以按 `(producer_epoch, frame_seq, timestamp_ns)` 严格联结；但锁定 Release 的目标批次固定为空，不能恢复装甲板世界位姿和 exact projected corners。

这不是消费者读取错误。Release 文档和 `release.json` 明确声明 `distribution_locked=true`，锁定版不发布 ground truth。消费者不得绕过锁定语义、读取模拟器私有状态或改用开发二进制冒充正式采集。

## 3. 影响

旧 atlas 有 4,280 个 exact-corner 样本，但只有两个独立 session，且没有原图。现有坐标网络在留出 spin session 时显著改善，在留出 combined session 时发生负迁移；训练 session 内部的验证段也无法预警该负迁移。

因此当前可以继续做两会话方法审计，但不能诚实地声称图像修复器已完成泛化验证或可部署。若用像素分割近似角点、从截图手工量角点或用运动指令重建目标位姿，会把约 1 px 量级的标签污染混入监督，不能作为正式训练证据。

## 4. 建议的公共接口

建议由模拟器仓库提供“离线研究标签导出”，而不是解锁在线 SDK ground truth：

1. 新增显式 opt-in 的离线输出，例如 `DAEDALUS_CORNER_LABELS_JSONL=<path>`；默认关闭。
2. 正式实时 SDK 继续保持锁定版 `target_count=0`，消费者在线算法不能读取标签。
3. 输出仅包含训练所需的同曝光二维几何，不发布未来状态：
   - `schema_version`、`producer_epoch`、`frame_seq`、`timestamp_ns`；
   - 相机 profile、有效内参、畸变合同和图像尺寸；
   - `target_id`、`relative_slot`、可见性；
   - 固定物理板规格 `135×55 mm`、固定俯角 `15°`；
   - 屏幕规范顺序 `bl,tl,tr,br` 的 `exact_corners_px[4][2]`；
   - `motion_uniform` 及仅用于离线筛选的 `distance_m`、当前线速度和当前角速度；
   - 明确 `future_truth_included=false`。
4. 输出和消费者保存的原图只通过三元曝光身份联结；标签不得反馈到同次 detector/PnP/预测器输入。
5. `sdk/contract.json` 和 Release 文档增加该离线文件 schema；无需让消费者手写 SHM/TCP 协议。

建议版本：若该 opt-in 离线接口进入正式公共 Release，按新增向后兼容能力发布 `1.3.0`；消费者完成数据采集验证后再单独更新 `simulator.lock.json`。若只允许内部研究构建，则应发布可追溯的独立 research Release，不能用开发目录二进制替代版本锁。

## 5. 验收条件

1. 100% 标签行与实际 TCP 图像严格同曝光；身份缺失或不一致时 fail closed。
2. exact corners 使用既有 4,280 样本证据相同的物理板、坐标和屏幕排序合同。
3. exact corners 送入不变 free-IPPE 后仍达到既有微米级数值闭合。
4. 匀速标签由当前真实速度/角速度判定；端点和反向邻域可被确定性排除。
5. 至少覆盖多个距离、旋转方向、角速度、线速度、平移方向、组合运动与半径配置的独立 session。
6. Release manifest、SDK contract、哈希和复现命令完整；受保护数据和模型不被自动删除。

## 6. 批准后的消费者工作

批准并发布新模拟器版本后，消费者侧将：

1. 更新锁并采集多独立 session 的原图、0526 YOLO 四角和 exact corners；
2. 按完整 session 划分训练/验证/测试，禁止随机帧切分；
3. 训练图像 patch + 15 维四边形几何的联合修复器；
4. 保存每个样本和四个角点的完整误差分布、ECDF、直方图、失败样本和 checkpoint；
5. 只在匀速段把修复角点传播到不变 PnP 和组合运动预测器，明确排除端点/反向瞬间。
