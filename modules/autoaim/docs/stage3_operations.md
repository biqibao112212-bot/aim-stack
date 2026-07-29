# 第三阶段数据采集操作手册

本文是第三阶段装甲板集合预测数据采集的唯一操作入口。模拟器与 SDK
的公共约束仍以仓库根目录 `SIMULATOR_CONSUMER_GUIDE.md`、
`simulator.lock.json` 和锁定 Release 内文档为准；本文只补充 Stage 3
的采集步骤。禁止手动并行启动第二个模拟器或第二个 Stage 3 runner。

## 1. 固定入口与所有权

- 仓库：`D:\仿真\repos\aim-stack`
- 单 session：`scripts\run-stage3-session.ps1`
- manifest 串行采集：`scripts\run-stage3-manifest.ps1`
- 正式 manifest：
  `D:\仿真\dataset\autoaim-stage3-v1\stage3-20260719-v1\session_manifest.jsonl`
- 模拟器：只使用 `simulator.lock.json` 锁定的正式 Release；消费者任务
  不修改模拟器仓库、SDK 或 Release。

runner 独占管理以下资源：Windows `daedalus.exe`、WSL
`aim_sim_talos_auto_aim_bridge_stage3_*`、TCP 5602、UDP 5603、每次调用
唯一的 `TALOS_IPC_DIR`、原始 observation/truth JSONL 和证据目录。

## 2. 启动前检查

```powershell
Set-Location D:\仿真\repos\aim-stack
git branch --show-current
git status --short
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-architecture.ps1
```

不要预先手动启动模拟器或自瞄桥接器。runner 会清理仅属于 Stage 3 的
残留 WSL 桥接器，并拒绝非 Stage 3 进程占用 5602/5603 的情况。

## 3. 单 session 首件验收

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\run-stage3-session.ps1 `
  -Manifest <单个-session-json> `
  -EvidenceRoot D:\仿真\runtime\stage3-first-article\<session-id> `
  -DurationSeconds 30
```

成功标准：命令退出码为 0，证据目录存在 `session_result.json`，其中
引用的 `observations.jsonl` 和 `truth.jsonl` 均非空。runner 在正式计时
前等待两条数据流就绪；模拟器、桥接器、Scene Control ACK 或写盘任一
失败都会 fail-closed，失败 session 不得进入训练。

Stage 3 与自瞄 B 默认固定使用原生 `wide_6mm` 全图，关闭 16 mm 虚拟
裁切。`session_result.json` 必须记录 `camera_profile=wide_6mm` 和
`dual_focal=false`。随机 manifest 可先加 `-ValidateManifestOnly` 做无副作用
校验；runner 会拒绝初始距离大于 6.5 m，以及整段往返轨迹超出 7 m
相机距离预留、进入云台后方或超过 75 度安全水平转角的样本。被拒绝的
随机参数应重新采样，不得缩短采集时间来绕过门禁。

## 4. 正式 manifest 串行采集

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\run-stage3-manifest.ps1 `
  -Manifest D:\仿真\dataset\autoaim-stage3-v1\stage3-20260719-v1\session_manifest.jsonl `
  -EvidenceRoot D:\仿真\runtime\stage3-formal-20260720-v2 `
  -DurationSeconds 30
```

该入口严格串行：一个 session 成功并产生 `session_result.json` 后才启动
下一个；失败 session 最多重试三次，仍失败则停止整个 batch。证据根目录
存在独占 `.stage3-manifest.lock`；单 session runner 另有系统互斥锁，重复
启动会立即失败，不会争用端口。

## 5. 采集期间检查

```powershell
Get-Process daedalus -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort 5602 -ErrorAction SilentlyContinue
Get-NetUDPEndpoint -LocalPort 5603 -ErrorAction SilentlyContinue
Get-ChildItem D:\仿真\runtime\stage3-formal-20260720-v2 `
  -Recurse -Filter session_result.json | Measure-Object
```

原始零检测帧保留用于缺测统计，但转换器不会将不满足有效观测历史门槛
的窗口作为训练样本。采集频率不人为限制为 40 Hz；模拟器采集上限保持
200 Hz，实际吞吐由完整链路决定，训练按时间戳构造 0.2 秒窗口。

## 6. 失败恢复

1. 先停止 batch，不要启动第二个 batch 抢救。
2. 阅读失败 session 的 `bridge.log`、`simulator.stderr.log`、
   `simulator.stats.json` 和 batch 标准错误日志。
3. 确认没有 `daedalus.exe`，5602/5603 空闲；Stage 3 runner 下一次启动
   会清理遗留的 `aim_sim_talos_auto_aim_bridge_stage3_*`。
4. 使用同一 evidence root 重启 batch。已有 `session_result.json` 的 session
   会跳过；失败重试会使用新的 raw `run-*` 和新的 IPC 目录。
5. 不删除旧 raw、模型、标注或 Release。失败目录保留为证据，但不进入
   数据转换清单。

## 7. 禁止事项

- 禁止同时运行两个 Stage 3 manifest runner。
- 禁止手动复用 `TALOS_IPC_DIR`。
- 禁止把失败目录、空 JSONL 或没有 `session_result.json` 的 session 输入训练。
- 禁止在消费者任务中修改模拟器仓库、SDK、Release 或手写协议。
- 禁止用预览 FPS、物理 250 Hz 或配置上限 200 Hz 冒充实际观测帧率。
