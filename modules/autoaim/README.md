# aim_sim_bridge

> 当前权威入口：本模块属于 `D:\仿真\repos\aim-stack` 的自瞄 B；模拟器只能来自 `D:\仿真\releases\daedalus-simulator\1.2.1\windows-x86_64`，并以仓库根目录 `simulator.lock.json` 为准。下方 WSL、ROS2 和旧路径说明仅用于迁移溯源，不得作为新采集、性能测试或默认运行入口。

## 当前 Windows 原生模拟器链路（自瞄 B）

自瞄 B 的正式采集已从 Windows+WSL 切换为完全 Windows：Release 模拟器、原生 `aim_sim_windows_auto_aim_bridge.exe`、localhost TCP RGBA32 1440×1080、Windows TensorRT engine 和 Stage3 JSONL 均在 Windows 进程中运行。不要通过 `wsl.exe`、`/mnt/d`、文件三缓冲或旧 `run_talos_bridge_wsl.sh` 开展新的靶场采集。

目标始终在视野内的标准复现使用 Shooting Range、目标 3 与连续真值云台锁定；目标可自转：

```powershell
Set-Location D:\仿真\repos\aim-stack
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bench-windows-autoaim-e2e.ps1 `
  -WarmupSeconds 5 -DurationSeconds 30 -EnableStage3 `
  -InitialScene shooting_range -TruthGimbalTarget 3
```

Windows Release 1.2.1 的该条件端到端 Stage3 吞吐为 140.967 FPS；目标原始帧、逐帧 JSONL、分布图和结论边界在 `D:\仿真\runtime\SHOOTING_RANGE_TARGET_IN_VIEW_PERFORMANCE_1.2.1_20260808.md`。当前 engine 对该渲染目标仍可能报零接受检测，这不改变采集吞吐结论，也不得被表述为检测准确率验收。

## 当前构建与运行

推荐从消费者仓库根目录执行 Windows 原生采集：

```powershell
Set-Location D:\仿真\repos\aim-stack
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bench-windows-autoaim-e2e.ps1 `
  -WarmupSeconds 5 -DurationSeconds 30 -EnableStage3 `
  -InitialScene shooting_range -TruthGimbalTarget 3
```

启动器固定使用：

- Release SDK：`D:\仿真\releases\daedalus-simulator\1.2.1\windows-x86_64\sdk`；
- 模型：`D:\仿真\models\engines\windows\armor-0708-trt861-win-rtx4060-fp16.engine`；
- 图像：`1440×1080 RGBA32 / localhost TCP 5602 / latest-only`；
- 云台命令：`UDP 5601`，无需 F5 或按住空格授权；
- 场景管理：`daedalus.scene-control/2 / UDP 5603`。

Windows 原生桥的正式二进制和 engine 由上述端到端脚本校验；模型及训练资产不在 Git 中，由仓库根目录 `models/manifest.json` 校验并保护。`run_talos_bridge_wsl.sh` 只服务旧迁移路径，不能替代 Windows 原生采集。

普通装甲板 PnP yaw 的阶段二（G2）修复、坐标语义、动态靶场证据和复现命令
记录在 [docs/pnp_yaw_stage2.md](docs/pnp_yaw_stage2.md)。G2 已通过，但阶段三
采集与训练仍需单独讨论和授权。

## 当前联合性能基线

2026-07-18 使用模拟器 Release 1.0.0、1440×1080 RGB24/TCP、RTX 4060 Laptop GPU、`armor.engine` 和 `vivsionn_trt` 后端实测。桥接器处理 100 帧后预热 15 秒，再采样 30 秒：完整视觉结果均值 121.233 Hz（中位数 122 Hz），模拟器 TCP 发送均值 121.366 Hz，累计流水线均值 6.032 ms；最终处理 6133 帧并完成 6122 个视觉结果，曝光真值严格匹配，采集丢帧和 GPU map 错误均为 0。

这组数字只适用于记录的机器和后台负载。硬件、驱动、提交、模型/二进制哈希、每秒范围、异常样本和复现口径以模拟器仓库 `SIMULATOR_PERFORMANCE.md` 及 `benchmarks/1.0.0/performance-2026-07-18.json` 为准，消费者不另建一份公共性能契约。

Independent simulator adapter for the `vivsionn` C++ auto-aim code.

This project does not modify `D:\仿真\upstream\daedalus` and does not modify the clean Gitee snapshot under `third_party/vivsionn_snapshot`.

## Layout

- `third_party/vivsionn_snapshot`: clean clone of `https://gitee.com/SEU-3SE/vivsionn.git`.
- `src/aim_core_from_vivsionn`: copied auto-aim code used as the adaptation base.
- `src/sim_adapter`: simulator-facing ROS2 adapter.
- `src/aim_core_bridge`: narrow wrapper between simulator frames and copied vivsionn pipeline.
- `config`: simulator-specific camera and buff configs.

## Build On WSL

```bash
cd /mnt/d/仿真/aim_sim_bridge
bash scripts/build_wsl.sh
```

The default build sets `AIM_SIM_WITH_VIVSIONN_TRT=OFF`, so it builds the ROS2 interface and copied MPC smoke target without requiring TensorRT engines. It publishes no-target commands and is meant to validate the simulator bridge.

For Jetson/TensorRT or a WSL machine with TensorRT installed:

```bash
cd /mnt/d/仿真/aim_sim_bridge
AIM_SIM_WITH_VIVSIONN_TRT=ON bash scripts/build_wsl.sh
```

Put model engines outside git, for example:

```text
models/armor.engine
models/buff.engine
```

## Armor Model

The current armor detector model comes from `https://github.com/broalantaps/RobotDetectionModel.git`.
The checked model is:

```text
third_party/RobotDetectionModel/Model/0526.onnx
```

Convert it to the default FP16 TensorRT engine on WSL with:

```bash
cd /mnt/d/仿真/aim_sim_bridge
bash scripts/convert_armor_onnx_to_trt_wsl.sh
```

Default output:

```text
models/armor.engine
```

Validated binding contract for `0526.onnx`:

```text
input:  images 1x3x640x640
output: output 1x25200x22
```

The current engine is hardware/TensorRT-version specific. Re-run the conversion script if the target GPU, TensorRT version, or model file changes.

Armor and outpost modes only require `models/armor.engine`. The buff TensorRT engine is loaded lazily and is only required when running `small_buff` or `big_buff`.

The run script passes `models/armor.engine` as `armor_detector_config` by default. Override it with:

```bash
AIM_SIM_ARMOR_ENGINE=/abs/path/to/armor.engine bash scripts/run_ros2_bridge_wsl.sh armor
```

## Energy Buff Model

The current simulator-trained energy-mechanism detector is a YOLO pose model exported at:

```text
../agent-team/models/energy_buff_yolo_pose_v5_neg/yolov8n_pose_640_e12_v5_neg/weights/best.onnx
```

Convert it to the default FP16 TensorRT engine on WSL with:

```bash
cd /mnt/d/仿真/aim_sim_bridge
bash scripts/convert_buff_onnx_to_trt_wsl.sh
```

Default output:

```text
models/buff.engine
```

Validated ONNX contract for the v4 model:

```text
input:  images 1x3x640x640
output: output0 1x20x8400
classes: 6
keypoints: 5 x (x, y)
```

## Run

Start Daedalus in ROS2 mode, press `F5` in the simulator to enable external command receive, then run one target mode:

```bash
cd /mnt/d/仿真/aim_sim_bridge
bash scripts/run_ros2_bridge_wsl.sh armor
bash scripts/run_ros2_bridge_wsl.sh outpost
bash scripts/run_ros2_bridge_wsl.sh small_buff
bash scripts/run_ros2_bridge_wsl.sh big_buff
```

The adapter subscribes to `/image_raw`, `/camera_info`, and `/tf`, and publishes `/rm_gimbal/cmd`.

## Integrated Windows Rendering Mode

This historical integrated mode is superseded for normal autoaim-B acquisition. New work must use the Windows-native command at the top of this README; this section documents the prior Windows-rendering + WSL-bridge arrangement only.

Desktop shortcut:

```text
C:\Users\Administrator\Desktop\Daedalus Integrated AutoAim.lnk
```

Workspace launcher:

```cmd
D:\仿真\agent-team\launch\Start-Daedalus-AutoAim-Integrated.cmd
D:\仿真\agent-team\launch\Start-Daedalus-AutoAim-Integrated.cmd outpost
```

What the launcher does:

- starts the Windows Talos-enabled Daedalus executable from `upstream\daedalus\target-integrated-talos`;
- sets `TALOS_IPC_DIR` to `D:\仿真\talos-ipc`, shared with WSL as `/mnt/d/仿真/talos-ipc`;
- lets Daedalus start and stop the hidden WSL `aim_sim_talos_bridge` with `models/armor.engine`;
- reads Talos images in WSL with ordinary file polling because persistent WSL `mmap` views of Windows-written `/mnt/d` files can stay stale;
- sends gimbal/fire commands to Daedalus over UDP `5601` and also writes the Talos command slot for compatibility;
- starts with auto-aim receive off. Press `F5` in the simulator to enable auto-aim mode. When auto-aim mode is off, `Space` manually fires a projectile. When auto-aim mode is on, hold `Space` to authorize auto-aim takeover; Daedalus applies yaw/pitch commands only while `Space` is held, and fires only when the bridge reports a target plus `fire_advice`.

Bridge log:

```text
D:\仿真\aim_sim_bridge\build\integrated_talos_bridge_armor.log
```

## Pitch Convention

`vivsionn` control output uses real gimbal-style pitch degrees. Daedalus ROS2 currently expects neutral pitch as `90.0`, so the adapter publishes:

```text
sim_pitch = vivsionn_pitch + sim_pitch_neutral_deg
```

The default `sim_pitch_neutral_deg` is `90.0`.
