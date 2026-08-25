# Aim Stack 当前任务板

## 当前状态

- 生产 detector、PnP、`RobotEstimator`、tracker、predictor 和 fire control 均未因离线研究而改变。
- V12 角点修复器与 V13 同帧 benefit gate 已通过 V17 fresh validation 和 V18 sealed test，
  但只获得“离线候选”资格；线上接入仍需单独的 C++/ONNX A/B、时延和部署审批。
- `training/armor_pose` 正在研究同帧概率角点与图像证据能否把 raw-YOLO PnP 的 position、depth
  和 ray P95 至少降低 30%。H1 的无条件稀疏概率角点和 H2 的同一单应重采样 dense PnP 已否定。
- 当前最佳 V20 observable gate 仍是开发集结果，未达到 position/depth 30% 目标，也不能作为
  泛化或部署结论。权威状态与数字只保存在 `training/armor_pose/research-state.yaml` 和
  `training/armor_pose/findings.md`。

## 进行中

- V21：高分辨率双尺度图像输入、dense pixel-to-corner voting、四角概率分布与
  probabilistic PnP。目标是增加独立图像证据，而不是从同一四角单应重复生成伪对应点。
- 在线输入限于同一帧 RGB、原始检测角点、相机内参和 ROI 几何；truth、motion label、
  physical identity、future 和 tracker history 均不得进入推理。

## 下一步

1. 完成 V21 的可复现训练与开发集检查；失败时保留负证据，不继续围绕同一表示调阈值。
2. 候选冻结后采集新的 session-disjoint validation；达到预注册门槛后才能开启一次 sealed test。
3. 只有离线验收通过且用户另行批准，才提出生产集成；必须保留 raw/repaired 双路对照和
   fail-closed 回退。

## 不可越过的门禁

- V15/V18 sealed evidence 不得用于后续模型、阈值、样本或超参数选择。
- train/validation/test 按完整 session 隔离，并验证 raw frame、label、identity 与 SHA-256 闭合。
- GPU 研究请求 fail closed，不允许静默回退 CPU；生产候选还需 ONNX/C++ 数值一致性与时延预算。
- 未获单独批准前，不修改生产 PnP、tracker、predictor、fire control、模拟器或 SDK。
