# 图像角点修复器：精确标签采集缺口与模拟器接口提案

状态：用户已于 2026-08-13 授权 Linux 采集与训练；1.3.1 Release 已将 exact-label 与 full-frame
采集接口发布为 `daedalus.offline-exact-corners/1` 和 `daedalus.offline-frame-capture/1`。消费者仍不得修改模拟器仓库、SDK 或正式 Release。

## 1. 目标与范围

目标是训练只依赖单帧图像 patch 与同帧 YOLO 四角的联合修复器，输出四个角点的 8 维修正量。运动模式、时间、session、目标真值、PnP 和未来状态都不得作为网络输入。

本阶段只需要匀速运动曝光。平移端点、反向瞬间及其邻域不进入训练、验证和下游传播；它们也不属于本阶段性能目标。

## 2. 消费者侧复现

正式消费者锁定为 `daedalus-simulator 1.3.1/linux-x86_64`，source 为
`d7637d00f69f0b6b01814c4fef6087baa92b0607`。Linux Release、TCP 图像、Release SDK 和
default-off label sidecar 均未被模拟器源码替代。

本轮新增了环境变量门控的消费者原图证据钩子，并完成三次 smoke：

| 证据 | 数量/结果 |
| --- | ---: |
| 精确曝光身份与相机姿态行 | 1,541 |
| 保存原图 | 308 |
| `has_exact_exposure_truth=true` | 1,541/1,541 |
| `ground_truth.targets` 非空 | 0/1,541 |

1.3.0 仍保持 `distribution_locked=true` 与空在线目标批次，但增加写入式
`DAEDALUS_CORNER_LABELS_JSONL`。它在完整 TCP RGBA32 帧发送后才写出同曝光 exact corners，
并不解锁实时 truth。原图、identity ledger、labels 和后续 detector 输出只能按
`(producer_epoch, frame_seq, timestamp_ns)` 严格联结；消费者不得读取私有状态或让标签进入
在线输入。

## 3. 影响

旧 atlas 有 4,280 个 exact-corner 样本，但只有两个独立 session，且没有原图。现有坐标网络在留出 spin session 时显著改善，在留出 combined session 时发生负迁移；训练 session 内部的验证段也无法预警该负迁移。

因此先采集多独立 full sessions 并按完整 session 切分，才可开始正式 image-conditioned
repair 训练；当前尚不能声称已完成泛化验证或可部署。若用像素分割近似角点、从截图手工量
角点或用运动指令重建目标位姿，会把约 1 px 量级的标签污染混入监督，不能作为正式训练证据。


2026-08-14 的 Linux exploratory pilot 以 `925` 个 spin 曝光帧的 `1,274` 条 matched uniform
rows 训练 RGB patch + raw-geometry 修复器；frame-group holdout 的坐标 RMS 为
`29.23 -> 24.48 px`。该结果不改变本节结论：它没有独立 session split，而且临时全 PNG
collector 自行解析 TCP wire，违反消费者边界，已从源码删除。相关数据和 checkpoint 只作受保护
探索证据，不能参与正式模型选择。

该历史探索性结果不再代表正式导出路径：Linux 1.3.1 Release 已发布 `--save-rgba-frames --until-eof`，为每个 identity 写入 raw RGBA、hash 与 capture manifest。正式采集必须使用该路径和 validator 的 `--require-raw-frames`；消费者不得手写协议、修改 Release 或把标签送入线上链路。

已完成一个受限的 1.3.1 两会话隔离 smoke：官方 spin full-frame 会话训练，独立 linear+spin
full-frame 会话验证；两者均通过 Release raw-frame/Z4/uniform/IPPE 闭合。0526 ONNX detector 的前
300 个 label-bearing exposures 形成 407 train / 336 validation 条 uniform matched rows。图像 patch +
raw 15-D geometry 网络在 held-out session 的 coordinate RMS 为 `28.5205 -> 29.0269 px`，没有改善。
checkpoint、metrics 和原始会话均为受保护负面证据；它既不构成正式数据集，也不授权模型选择、部署、
PnP 替换或在线接入。必须先补齐 stationary/linear/旋转方向/距离/半径等独立 session 覆盖，并保留未访问
test session，再开展正式训练结论。

## 4. 已发布公共接口

Simulator 1.3.1 提供“离线研究标签与全帧导出”，而不解锁在线 SDK ground truth：

1. 显式 opt-in 的 `DAEDALUS_CORNER_LABELS_JSONL=<path>`，默认关闭且不得覆盖既有文件。
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
6. `--save-rgba-frames` 仅在 `--until-eof` 下启用，逐 identity 创建 `frames/*.rgba`、写入 payload/raw SHA-256 与 `capture-manifest.json`；validator 用 `--require-raw-frames` fail closed 检查所有原图。

exact-corner 接口先以 1.3.0 发布；full-frame 采集器以向后兼容的 1.3.1 发布。消费者现已更新[版本锁](../../../simulator.lock.json)。其 schema、使用与 validator 是 Release 的公共契约；采集只使用该 Release，不能用开发目录二进制替代。

## 5. 验收条件

1. 100% 标签行与实际 TCP 图像严格同曝光；身份缺失或不一致时 fail closed。
2. exact corners 使用既有 4,280 样本证据相同的物理板、坐标和屏幕排序合同。
3. exact corners 送入不变 free-IPPE 后仍达到既有微米级数值闭合。
4. 匀速标签由当前真实速度/角速度判定；端点和反向邻域可被确定性排除。
5. 至少覆盖多个距离、旋转方向、角速度、线速度、平移方向、组合运动与半径配置的独立 session。
6. Release manifest、SDK contract、哈希和复现命令完整；受保护数据和模型不被自动删除。

## 6. 已授权消费者工作

消费者侧现在将：

1. 更新锁并采集多独立 session 的原图、0526 YOLO 四角和 exact corners；
2. 按完整 session 划分训练/验证/测试，禁止随机帧切分；
3. 训练图像 patch + 15 维四边形几何的联合修复器；
4. 保存每个样本和四个角点的完整误差分布、ECDF、直方图、失败样本和 checkpoint；
5. 只在匀速段把修复角点传播到不变 PnP 和组合运动预测器，明确排除端点/反向瞬间。
