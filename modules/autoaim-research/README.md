# PnP—11 维 EKF 研究基线

`modules/autoaim-research` 是内部 PnP 与 EKF 实验的唯一实现入口。它从
[TongjiSuperPower/sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)
固定提取 YOLO 四角点解析、IPPE PnP 和原生 11 维整车 EKF，只保留研究所需的
`image -> detector -> PnP -> 11D EKF -> JSONL` 链路。

精确版本、哈希和坐标约定见
[`implementation.lock.json`](implementation.lock.json)。旧 `modules/autoaim` 仅作历史证据和回溯，
不得作为新实验基线。

## 固定组合

| 部分 | 唯一选择 |
| --- | --- |
| 模拟器 | Daedalus `1.4.0-learning-r1` Linux x86_64 |
| 上游自动瞄准 | Tongji `sp_vision_25@bd9f5e798fa3c6dd3b483ae6627796afb41c608d` |
| 检测模型 | `armor-0526-fp32-converted.onnx`，`1x25200x22` |
| 推理运行时 | ONNX Runtime `1.22.1` CPU |
| 位姿 | 同济 IPPE 流程，尺寸适配为 `135/225 x 55 mm` |
| 跟踪 | 同济 `Target` 的 11 维 EKF |

旧 `YpdAngleTracker` 没有被编译、链接或调用。同济 `Target` 自身会在观测方程中将
`xyz` 转成 `yaw/pitch/distance`，这是被研究的上游 11 维 EKF 数学的一部分，不是旧
tracker 实现。

## 研究边界

- 不含开火、弹道、命中事件、云台控制和比赛逻辑。
- 目标真值只在 EKF 更新完成后写入评估日志；不进入 detector、PnP 或 EKF。
- 同曝光相机/云台位姿用于必要的坐标变换，不是目标真值。
- 数据采集时的真值云台控制由模拟器或独立采集器负责，本程序不发送 aim/fire 命令。

## Ubuntu 24.04 构建

先安装常规依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev libopencv-dev curl unzip
```

ONNX Runtime 被固定在工作区 `deps` 中。新机器上可用带哈希校验的安装脚本：

```bash
./modules/autoaim-research/scripts/install-onnxruntime-1.22.1.sh \
  /home/potato/Projects/仿真/deps/onnxruntime-linux-x64-1.22.1
```

构建并测试：

```bash
cmake -S modules/autoaim-research -B build/autoaim-research \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH=/home/potato/Projects/仿真/releases/daedalus-simulator/1.4.0-learning-r1/linux-x86_64/sdk \
  -DBUILD_TESTING=ON
cmake --build build/autoaim-research --parallel
ctest --test-dir build/autoaim-research --output-on-failure
```

CTest 同时校验 vendored 源文件、模型、模拟器 Release 和 ONNX Runtime 哈希；任何一项漂移都
会 fail closed。

## 运行

终端 A：

```bash
/home/potato/Projects/仿真/releases/daedalus-simulator/1.4.0-learning-r1/linux-x86_64/daedalus-learning.sh \
  --runtime-dir /tmp/daedalus-learning-1000 start --performance --scene shooting-range
```

终端 B：

```bash
./build/autoaim-research/autoaim_research_runner \
  --config modules/autoaim-research/config/research.yaml \
  --output /absolute/path/run.jsonl \
  --max-frames 1000
```

`--model` 可以显式覆盖 ONNX 路径。输出文件必须不存在，程序拒绝覆盖。每行保存完整
`producer_epoch + frame_seq + timestamp_ns`、PnP 观测、11 维 EKF 状态、匹配的同曝光真值和
匹配误差。

`ekf_state` 的顺序与同济上游一致：
`[cx, vx, cy, vy, cz, vz, yaw, omega, r_even, r_odd-r_even, h_odd-h_even]`。
`ekf_estimate` 同时提供带名字的中心、速度、角速度和长短半径，用于后续直接作图。

## 坐标约定

PnP 输出先使用 OpenCV optical `(+x 右, +y 下, +z 前)`，再转到 ROS camera-link
`(+x 前, +y 左, +z 上)`，最后由同一曝光的相机世界位姿转到 ROS odom 方向。研究坐标系
保持 ROS odom 轴方向，原点移到该曝光时的云台轴心。

目前只完成构建、模型形状和 1.4.0 真实帧冒烟验证；用于教程的三组 20 秒正式实验尚未采集。
