# Aim Stack 模拟器消费者统一指南

- 规则版本：`AIM_SIMULATOR_CONSUMER_GUIDE_V1`
- 审批门禁：`SIMULATOR_CHANGE_APPROVAL_REQUIRED`

本文件由 `aim-stack/main` 跟踪，所有新自瞄、火控和打符分支必须从包含本文件与 `simulator.lock.json` 的 `main` 创建。分支可以不知道模拟器 Rust 实现，但必须按本文理解并使用发布版 SDK、场景和运行模式。

## 1. 唯一依赖入口

- 模拟器所有者：私有仓库 `biqibao112212-bot/daedalus-simulator`；
- 本机只读源码位置：`D:\仿真\repos\daedalus-simulator`；
- 当前正式 Release：`D:\仿真\releases\daedalus-simulator\1.0.0`；
- 消费者版本锁：`simulator.lock.json`；
- SDK 安装目录：`D:\仿真\releases\daedalus-simulator\1.0.0\sdk`；
- 正式契约：Release 根目录 `release.json` 与 `docs/sdk-contract.json`；
- SDK 用法：Release 内 `docs/README.md`；
- 接口语义：Release 内 `docs/SIMULATOR_INTERFACE.md`；
- 场景控制：Release 内 `docs/SCENARIO_CONTROL.md`；
- 性能基线：Release 内 `docs/SIMULATOR_PERFORMANCE.md`，以及模拟器仓库当前同名文档与 `benchmarks/`。

消费者不得从模拟器工作区的临时 build 目录取依赖，不得使用旧工作树二进制，不得把模拟器源码或协议头复制进本仓库。

## 2. 当前锁定配置

| 项目 | 固定值 |
| --- | --- |
| 模拟器 / SDK | Daedalus Simulator 1.0.0 / DaedalusSimSdk 1.0.0 |
| IPC | SHM v7，ABI revision 1，元数据 76992 字节 |
| 图像 | RGB24，1440×1080 |
| 物理 | 250 Hz，单 substep |
| 高性能采集上限 | 200 Hz；这是配置上限，不是承诺帧率 |
| 图像数据面 | TCP 5602，latest-only |
| 云台/发射命令 | UDP 5601 |
| 场景控制 | `daedalus.scene-control/1`，UDP 5603 |
| 运行时 IPC 根 | `D:\仿真\runtime` 下的任务专用目录 |

跨 Windows/WSL 的 1440×1080 图像必须使用 TCP。文件三缓冲图像只用于同系统兼容调试，不能作为自瞄默认配置或性能结论。

## 3. 三张原生地图与 SDK 场景名

| 原生地图 | 人工快捷键 | SDK `set_scene` 值 | 用途 |
| --- | --- | --- | --- |
| Normal Map | F7 | `armor` | 默认装甲板/竞技场地图 |
| Shooting Range | F8 | `shooting_range` | 1 号和 3 号靶车；支持静止、直线、自转、直线+自转 |
| Energy Mechanism | F9 | `energy` | 能量机关地图；小符/大符和叶片状态由 `set_rune_state` 控制 |

`outpost` 是 SDK v1 支持的前哨站任务/目标场景模式，但没有独立的 F7–F9 原生地图快捷键，因此不得称为第四张原生地图。程序化切换必须使用 `SceneControlClient`，先 `createSession()`，再 `setScene()`；禁止修改模拟器实体或内部资源来切图。

## 4. 两种运行模式

### 默认高性能模式

```powershell
Set-Location D:\仿真\releases\daedalus-simulator\1.0.0
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-simulator.ps1
```

该模式关闭可见预览，但离屏相机仍持续渲染、GPU readback 和采集。隐藏窗口或黑屏不代表没有图像；应检查 `capture_copy_submit_hz`、TCP 发送率和消费者输入计数。自瞄 B 默认使用此模式，启动器会在消费者侧自动启用 TensorRT。

### 正常可视渲染/验收模式

```powershell
Set-Location D:\仿真\releases\daedalus-simulator\1.0.0
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-simulator.ps1 -Visible
```

该模式显示最高 60 Hz 的预览，用于人工验收画面、相机和场景；离屏 1440×1080 采集仍独立运行。不得用预览帧率代替采集或自瞄吞吐。

当前 2026-07-18 工作基线：高性能纯模拟器主更新/采集均值 177.199/160.009 Hz；可视复测 158.298/153.015 Hz，预览 60.010 Hz；高性能 + 自瞄 B/TensorRT 完整视觉均值 121.233 Hz，流水线累计均值 6.032 ms，采集丢帧和 GPU map 错误均为 0。数字只适用于性能文档记录的提交、机器和后台负载；未来版本必须读取所有者发布的新基线，不得把旧数字当永久承诺。

## 5. SDK 构建和链接

正式消费者只链接 Release 内 SDK：

```cmake
find_package(DaedalusSimSdk 1 REQUIRED CONFIG)
target_link_libraries(my_consumer PRIVATE DaedalusSimSdk::DaedalusSimSdk)
```

配置示例：

```bash
cmake -S <consumer> -B <build> \
  -DCMAKE_BUILD_TYPE=Release \
  -DDaedalusSimSdk_DIR=/mnt/d/仿真/releases/daedalus-simulator/1.0.0/sdk/lib/cmake/DaedalusSimSdk
```

可用头文件：

```cpp
#include <daedalus_sim_sdk/talos_v1.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>
#include <daedalus_sim_sdk/udp_gimbal_client.hpp>
#include <daedalus_sim_sdk/scene_control_client.hpp>
```

标准调用顺序：

1. 从 `simulator.lock.json` 解析 Release 与 SDK 路径；
2. 设置任务专用 `TALOS_IPC_DIR`，启动正式 Release；
3. 用 `TalosMetadataMapping/TalosMetadataReader` 打开元数据，并调用 `isCompatible`；版本、分辨率、元数据大小或 ABI 不一致时立即失败；
4. 用 `TcpImageClient` 接收 5602/TCP 完整 RGB 帧；
5. 按生产者 epoch、`frame_seq` 和 `timestamp_ns` 严格关联图像、曝光位姿和真值，禁止用相邻帧补缺失曝光；
6. 用 `UdpGimbalClient` 向 5601/UDP 发送统一云台/发射命令；
7. 用 `SceneControlClient` 管理地图、靶车和能量机关状态；
8. 模拟器真值只可用于标签和验收，严禁作为神经预测器输入。

当前自瞄 B 一键入口：

```powershell
Set-Location D:\仿真\repos\aim-stack
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-autoaim-b.ps1
```

## 6. 模拟器修改审批门禁

自瞄调试中发现模拟器 bug 或新增需求时：

1. 立即停止模拟器仓库、SDK 和正式 Release 的任何写操作；
2. 在向用户的提案中写明：Release/提交、复现命令、期望与实际、日志/截图/最小证据、对 SDK/场景/性能的影响、建议的公共接口和版本变化；
3. 明确标记状态为“等待用户批准”，在用户针对该提案作出明确批准前不得修改；
4. 批准后切换到 `daedalus-simulator` 仓库独立实现、测试并发布新版本；
5. 自瞄仓库只更新 `simulator.lock.json`、SDK 依赖和消费者适配，不携带模拟器实现。

禁止先改后报、直接修补 Release、复制 SDK 布局、绕过兼容检查，或因为“只是为了调试自瞄”而默认获得模拟器写权限。

## 7. 新分支门禁

新分支从当前 `main` 创建后首先运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-consumer-boundary.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-architecture.ps1
```

第一项只检查 Git 可追踪边界，可在 GitHub Actions 运行；第二项还检查本机 Release、SDK 和受保护模型。任何分支删除本指南、删除审批规则、跟踪模拟器协议副本、模型或编译产物，均视为边界检查失败。
