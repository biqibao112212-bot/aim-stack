# Aim Stack 任务板

上下文版本：`CTX-AIM-STACK-2026.07-v2`

## 当前状态

- 自瞄与打符已从模拟器仓库物理分离。
- 自瞄 B 已导入 `modules/autoaim` 并使用 SDK v1。
- 打符源码已导入且模型全部保留，但其旧 SHM v6 消费端尚未升级，因此保持暂停，不声称兼容。
- 已锁定模拟器 Release 1.0.1（SDK 仍为 DaedalusSimSdk 1.0.0、SHM v7 ABI r1）；1.0.1 与消费者所需 SDK/SHM/TCP/UDP/Scene Control 公共契约兼容，架构与边界检查通过。
- 自瞄 B 后续只通过 Release SDK 的元数据、TCP 图像、UDP 云台和 Scene Control v1 访问模拟器；不再维护本地 IPC/场景协议。
- 2026-07-18 已完成两轮相同的 `-DynamicRange -DurationSeconds 30` shooting_range 基线。A：TCP sent 3611、main/capture 150.644/148.648 Hz、bind_fail/drops/GPU map errors=0，bridge completed vision 3579、1440×1080、TensorRT `vivsionn_trt`、exposure/ground-truth exact match=true；B：TCP sent 3252、main/capture 150.668/143.684 Hz、bind_fail/drops/GPU map errors=0，bridge completed vision 3230、1440×1080、TensorRT `vivsionn_trt`、exposure/ground-truth exact match=true。两轮 Scene Control 均收到 create/set_scene/target 1/target 3 ACK；每轮退出后无 daedalus、无精确 token bridge、5602/5603 无占用，第二轮可立即重复启动。
- 已建立所有未来消费者分支继承的 `SIMULATOR_CONSUMER_GUIDE.md`、机器锁和 GitHub 边界检查；三张原生地图、两种运行模式、SDK 用法与模拟器变更审批门禁均已固化。

## 阶段二完成状态

- 自瞄 B 阶段二已完成，G2 已通过：普通装甲板 `+15°` 倾角已在 tracker/chassis 坐标系建模，并通过曝光时刻云台姿态投影到 camera 完成约束 PnP；生产 tracker 消费修正后的底盘系 yaw。非零姿态合成回归与 target 3 的 3/5/7 m 慢速自转连续曲线构成验收证据。
- 2026-07-19 已完成 target-3-only slow-spin 测量：shooting_range 原生靶场、线速度/线性行程均为 0、`spin_deg_s=30`，距离 3/5/7 m 各 30 s，离屏高性能模式，`-PipelineOnly` 保留逐帧 pipeline telemetry 而关闭 bridge JSONL。三轮均收到 create/set_scene/set_target_3_spin ACK，TensorRT `vivsionn_trt`、1440×1080、exposure/GT exact match=true；TCP sent=4024/4026/4008，capture/main=129.25/131.238、162.427/167.409、135.564/136.561 Hz，bind/drop/map errors=0。
- 当前观测曲线和 smoke 结果只作测量证据：tracked yaw 样本分别为 1333/1553/1370，约 45.70/50.58/43.98 Hz；tracked_id 均在 `{0,1,2,3}`，30 s transitions=158/287/178，且存在 detector number 非 3 的少量样本和大量 `jump_flag`。曲线暂不能用于宣称 PnP 正常/异常，也不据此建立 G2 数值门槛。
- 2026-07-19 的分段/相位分析显示：在 active tracker 且 number=3 的主样本中，3/5/7 m 有效样本为 1079/1349/806（占 tracked yaw 的 80.9%/86.9%/58.8%）；`tracked_id` 切换间隔中位数约 18.6/16.7/17.6 ms，远短于 30°/s 下物理 90°切换的 3 s。基于固定 30°/s、每 ID 一个常量偏置的描述性正弦拟合，RMSE=28.5/25.2/27.7°、P95 残差=50.5/48.0/51.1°；这说明当前输出流不能直接当作干净的单正弦/90°跳变序列，但尚不能区分 tracked_id 语义、tracker 切换、检测误分类与 PnP 误差。
- 同一主样本的 selected PnP reprojection error 中位数/P95 为 0.41/1.31、0.49/1.45、0.42/1.52 px，未随距离单调恶化；该结果只用于定位后续实验，不构成 PnP 正常性结论或门槛。
- 对“17 ms 切板”的审计已校正口径：`tracked_armor` 在 `current_match_ids=[]` 或 `primary_observation_index=-1` 时可能只是 `_trackedArmor` 的旧值，不能作为新鲜 PnP 观测。只保留 `detected=true`、当前有匹配且 number=3 的 live 样本后，3/5/7 m 的 `tracked_id` 切换中位间隔为 21.6/26.1/23.1 ms，切换数为 149/266/145；其中单观测→单观测切换为 55/126/104，槽位差 2（0↔2/1↔3）占 53.7%/48.1%/40.7%。代码路径显示 `tracked_id` 是每帧 primary observation 匹配到的预测槽位，不是持久物理装甲板 ID；在 primary 选择和槽位语义明确前，禁止用它做 90°相位补偿。

## 门禁

- G2 已通过，阶段二关闭；通过结论只说明 PnP 输入坐标语义满足后续预测器前置要求，本轮回放不作为训练样本。
- 阶段三仍未授权。在与用户共同确定采集和训练方案前，禁止启动阶段三数据采集或训练。
- 打符 SDK v1 适配完成前禁止闭环实测。
- 未获得用户针对具体提案的明确批准前，任何自瞄任务禁止修改模拟器仓库、SDK、发布脚本和正式 Release。

### 2026-07-19 observation-set rerun (user-approved recording change)

- Re-ran the same target-3 slow-spin experiment in native `shooting_range`: zero linear motion, `Spin=30 deg/s`, 30 s, 3/5/7 m, offscreen DX12 high-performance mode, `-PipelineOnly`.
- The first d3 startup failed closed at Scene Control `create_session` (`recvfrom failed: Resource temporarily unavailable`); owning PID/WSL/port checks after cleanup found no residual daedalus, exact-token bridge, TCP 5602 or UDP 5603. A fresh d3 run then succeeded; d5 and d7 succeeded on their first runs.
- Every successful run recorded Scene Control `create_session`, `set_scene`, `set_target_3_spin` ACKs; TensorRT backend was `vivsionn_trt`, frames were 1440x1080, TCP sent >0, bind_fail/drops/map errors were 0. The final per-run simulator stats are retained beside the observation artifacts.
- New artifacts preserve every `solved_armors` item per exposure in `observations.jsonl`; no `tracked_id`, `tracked_armor` or `jump_flag` is used for identity. The raw set retains detector metadata, four PnP corners, both PnP candidates, position, yaw, distance and reprojection error.
- Target-3 observation counts are 1312/1499/859 at 3/5/7 m; frames with target observations are 889/1093/685; frames with two or more target observations are 410/403/138. Detector number outliers remain in the raw set and are not silently relabeled.
- A separate replay artifact applies the agreed convention: first visible target plate is canonical ID 0, then temporal 30 deg/s continuity and cyclic allocation 0→1→2→3. It reports matched/new events and cost, while leaving raw observations unchanged. This is a derived analysis stream, not GT and not a G2 pass.

### 2026-07-19 diagnostic joint-PnP A/B

- Implemented a sidecar-only +15 degree joint yaw+tvec reprojection optimizer,
  legacy constrained residual recomputation, raw/refined corner branches,
  per-candidate convergence/conditioning, and marginalized
  `yaw_sensitivity_deg_per_px`. The tracker consumes legacy output only.
- Synthetic known-pose tests cover 3/5/7 m, yaw -70..70 degrees, nonzero
  distortion and invalid four-corner input. The focused candidate/coordinate
  regression passes; architecture and consumer-boundary checks pass.
- The performance launcher no longer forces Win32 `WindowStyle Hidden`; it now
  follows the Release performance contract using `DAEDALUS_PERF_DISABLE_UI=1`.
  It retains bridge/simulator logs and `DAEDALUS_STATS_JSON`, and fails closed
  on TCP bind, capture, GPU-map or surface errors. This fixed a reproduced
  consumer-launcher `ResizeBuffers / Invalid surface` run; no simulator files
  were changed.
- Valid A/B runs: 3 m r3, 5 m r1, 7 m r1. All received create/set_scene/spin
  ACKs, used `vivsionn_trt`, 1440x1080 TCP frames, and had bind/map/drop counts
  of zero. Final simulator snapshots report main/capture 104.94/104.94,
  119.31/119.31 and 118.28/118.28 Hz; TCP sent totals 3448/3247/3442.
- Result: refined joint reprojection improves p50 only 7.8%/2.8%/1.6%, while
  temporal p50 changes 2.74->2.96, 6.14->5.86 and 6.37->7.01 deg. Raw corners
  lower residual further but materially worsen temporal tails. Production PnP
  is unchanged; the current joint solver is evidence, not a selected fix.
- Physical/conditioning evidence: refined yaw sensitivity p50 is
  3.73/5.33/6.58 deg per pixel and p95 is 8.93/12.50/12.92 deg per pixel as
  distance increases. The next repair hypothesis must separate known absolute
  armor orientation transformed through the exposure pose from this measured
  pixel-conditioning floor; do not tune a G2 threshold before that review.
- Retention: `runtime/pnp-joint-ab-*` JSONL/logs/plots are reproducible current
  task evidence and are retained for the joint review. No model, engine,
  annotation, Release or non-reproducible data was deleted.

### 2026-07-19 chassis-frame repair replay

- Corrected production PnP yaw now models the ordinary armor's +15 degree tilt
  in tracker/chassis coordinates and applies the exposure-matched gimbal pose
  before camera projection. The old camera-fixed calculation is retained only
  as the diagnostic baseline; tracker input uses the corrected chassis yaw.
- Synthetic regression passed at nonzero pose: gimbal pitch +7 degrees, yaw
  -11 degrees, known armor yaw recovered within 0.1 degree, exact-pose RMS and
  max reprojection below 1e-4 px. Architecture and consumer-boundary checks
  pass; no simulator repository was modified.
- Successful replay artifacts: `runtime/pnp-chassis-pose-d3-20260719-r3`,
  `...-d5-20260719-r1`, `...-d7-20260719-r1`. A transient d3 retry
  `...-d3-20260719-r2` failed closed on simulator DX12 ResizeBuffers/Invalid
  surface; its stderr and bridge logs are retained. The next clean d3 run
  succeeded, so no process-wide kill or simulator edit was used.
- Successful runs all received Scene Control create/set_scene/set_target-3-spin
  ACKs, used `vivsionn_trt`, 1440x1080 TCP frames, and had bind_fail, queue_drop
  and GPU-map-error totals of zero. Simulator main/capture Hz were
  106.643/106.643, 124.370/124.370 and 130.819/129.820; TCP sent totals were
  3451/3397/3634; bridge completed vision counts were 1655/1475/1723.
- Continuous connected per-canonical-plate plots and metrics are in
  `runtime/pnp-chassis-pose-continuous-yaw-20260719.png` and
  `runtime/pnp-chassis-pose-continuous-metrics-20260719.png`. Production
  chassis-yaw adjacent increment p50/p95 (legacy in parentheses) are
  2.59/14.82 (2.65/15.78), 5.26/20.33 (5.66/39.24), and 7.67/28.88
  (8.76/69.40) degrees for 3/5/7 m. These are descriptive replay results;
  they do not establish a G2 threshold or authorize stage-three collection.

### 2026-07-19 G2 acceptance / stage-two closure

- User review clarified that the 3/5/7 m replay is PnP repair evidence, not a
  training dataset. The repaired production yaw is accepted as a geometrically
  valid per-exposure input for the later neural predictor.
- G2 passes on the combined nonzero-exposure synthetic regression, production
  tracker wiring, native-range continuous curves and complete runtime-chain
  evidence. No post-hoc scalar error threshold is introduced.
- The durable module report and plot are
  `modules/autoaim/docs/pnp_yaw_stage2.md` and
  `modules/autoaim/docs/pnp_yaw_stage2_target3_3_5_7m.png`.
- Stage two is complete. The only next item is joint discussion of the stage-three
  collection/training design; collection and training remain locked until then.
