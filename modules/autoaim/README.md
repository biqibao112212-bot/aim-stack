# 自瞄 B 教程

本模块是 `aim-stack` 当前活跃的装甲板自瞄与轨迹研究模块。它包含检测、PnP、观测器、轨迹研究和模拟器消费者适配，但不包含模拟器源码、正式 Release、TensorRT engine 或训练数据。

> 当前边界：预测器和多假设身份跟踪器仍处于暂停状态。教程中的运行命令用于现有自瞄 B 链路和受控采集，不代表轨迹预测器已经上线。

## 1. 先读这些文档

| 内容 | 文档 |
| --- | --- |
| 模拟器 Release、SDK、地图和运行模式 | [模拟器消费者统一指南](../../SIMULATOR_CONSUMER_GUIDE.md) |
| 当前锁定版本与端口 | [simulator.lock.json](../../simulator.lock.json) |
| 模块所有权边界 | [BRANCH_SCOPE.md](BRANCH_SCOPE.md) |
| 相机、云台、tracker 和 PnP 坐标语义 | [COORDINATE_CONTRACT.md](src/aim_core_from_vivsionn/AngleSolver/COORDINATE_CONTRACT.md) |
| 从四角点到后 PnP 时序的权威证据链 | [trajectory_evidence_chain.md](docs/trajectory_evidence_chain.md) |
| 观测器输入、状态、门控与验收定义 | [observer_specification.md](docs/observer_specification.md) |

仓库里出现的 `D:\仿真\...` 是历史 Windows 项目机部署路径，不是互联网下载地址。当前 Linux 工作区示例是 `/home/potato/Projects/仿真/repos/aim-stack`；其他 Linux 用户不必照抄这个绝对路径。公开克隆只能获得源码和文档；运行所需的私有 Release 和受保护模型必须由项目所有者按版本锁单独提供。

## 2. 获取源码

### Linux（当前 Ubuntu 24.04）

安装 Git、Python、C/C++ 基础工具和本模块常用的非 TensorRT 依赖：

```bash
sudo apt update
sudo apt install -y \
  git python3 python3-venv build-essential cmake ninja-build pkg-config \
  libeigen3-dev libopencv-dev libyaml-cpp-dev
```

Fedora 可使用：

```bash
sudo dnf install \
  git python3 gcc-c++ cmake ninja-build pkgconf-pkg-config \
  eigen3-devel opencv-devel yaml-cpp-devel
```

Arch Linux 可使用：

```bash
sudo pacman -S --needed \
  git python base-devel cmake ninja pkgconf eigen3 opencv yaml-cpp
```

克隆并检查基础环境：

```bash
git clone https://github.com/biqibao112212-bot/aim-stack.git
cd aim-stack
git --version
python3 --version
cmake --version
```

当前这台 Linux 机器已经具备上述基础依赖，不需要重复执行安装命令。

### Windows

确认 Git 已安装：

```powershell
git --version
```

如果命令不存在，请从 [Git 官方 Windows 安装页](https://git-scm.com/downloads/win) 安装，或使用：

```powershell
winget install --id Git.Git -e --source winget
```

克隆并进入仓库：

```powershell
git clone https://github.com/biqibao112212-bot/aim-stack.git
Set-Location .\aim-stack
```

本模块的代码由早期 `SEU-3SE/vivsionn` 基线迁移而来，并使用 [RobotDetectionModel](https://github.com/broalantaps/RobotDetectionModel) 的装甲板模型合同作为历史来源。原 Gitee 地址目前会要求登录，因此不再把它列为安装入口；可复现来源以仓库内的[迁移记录](../../MIGRATION_SOURCES.md)和 Git 历史为准。当前实际维护代码已经纳入本仓库的 [`src/aim_core_from_vivsionn`](src/aim_core_from_vivsionn)；不要再寻找已经移除的 `third_party/vivsionn_snapshot`。

## 3. 准备项目机布局

Linux 上的源码与离线工作区建议采用：

```text
<workspace>/
  repos/aim-stack/                       本仓库
  models/engines/                        可选；受保护模型，不由 Git 下载
  runtime/                               可选；本机离线输出
```

当前机器对应：

```text
/home/potato/Projects/仿真/repos/aim-stack
```

Windows 完整运行需要以下同级布局：

```text
<workspace>/
  repos/aim-stack/                       本仓库
  releases/daedalus-simulator/<version>/ 与版本锁匹配的模拟器 Release
  models/engines/                        受保护模型
  runtime/                               运行证据与 IPC
```

其中 `<version>`、SDK ABI、分辨率和端口必须以 [`simulator.lock.json`](../../simulator.lock.json) 为准，模型文件和哈希必须以 [`models/manifest.json`](../../models/manifest.json) 为准。不要从旧工作树、临时 build 目录或 README 中的历史路径复制二进制。

Windows 项目机可先运行边界检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-consumer-boundary.ps1
```

已安装私有 Release 与模型后，再运行完整架构检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check-architecture.ps1
```

### Linux 当前可执行检查

在 Linux 仓库根目录可直接运行文档检查：

```bash
python3 -B ./scripts/check-doc-links.py
```

当前锁定 Release 已提供 Linux `DaedalusSimSdk 1.3.1`。基础构建应显式指定其 SDK，不要从
模拟器源码 build 目录取依赖：

```bash
cmake -S modules/autoaim -B /tmp/aim-stack-autoaim-1.3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/potato/Projects/仿真/releases/daedalus-simulator/1.3.1/linux-x86_64/sdk \
  -DAIM_SIM_WITH_ROS2=OFF -DAIM_SIM_WITH_VIVSIONN_TRT=OFF
```

Linux Release 不携带 TensorRT engine；缺少匹配的 Linux detector 运行时/模型时，不要把配置或
推理失败解释为模拟器或角点标签错误，也不要将 Windows engine 当作 Linux 资产使用。

## 4. 启动自瞄 B

当前正式入口是 Linux 1.3.1 的离线 exact-corner/full-frame 采集，而非生产预测器。创建一个
新的任务目录并使用 Release 的 collector/validator，完整命令与安全边界见[消费者统一指南](../../SIMULATOR_CONSUMER_GUIDE.md)。`--save-rgba-frames --until-eof` 会收集每曝光 raw RGBA、wire identity ledger、capture manifest 与 exact-corner 标签；它不提供 detector 输出或在线真值。

每次正式训练前必须先完成完整 session 的 train/validation/test 切分，并使用模拟器所有者发布、ledger hash 验证过的逐曝光 raw RGBA 在同一批图像上生成 raw detector corners。修复网络只接受图像 patch 与 raw corners；exact corners 只作监督/验收标签，不能作为网络输入、PnP 输入或任何在线状态字段。消费者不得自行解析 TCP 来实现全帧图像导出。

Linux 1.3.1 图像角点修复已完成 session-disjoint 正式实验。当前 `v8` 候选在第二轮全新测试中四种运动模式均无退化、总体 RMS 改善 `2.55%`，但未达到预声明的 `5%` 强门，因此保留为带回退的离线研究候选，不替换现有生产 PCA/`cornerSubPix`。完整数据关联根因、网络结构、失败方法、指标和受保护证据见[Linux 1.3.1 图像角点修复正式实验](docs/corner_repair_linux_1_3_1_formal.md)。

在不重新选择模型的前提下，又完成了一条 Linux 1.3.1 同曝光位姿的 post-test 开发 A/B：冻结修复器在同 IPPE、同坐标链和同 `400 ms` LOS 预测器下，将 50/100/200 ms mean cross-depth 误差从 `111.1/119.5/147.8 mm` 降到 `59.4/68.2/84.5 mm`；exact 恒向基线 P95 为 `6.2/7.7/12.5 mm`。该结果闭合了“角点修复确实会传递到预测”的离线因果链，但单会话不构成部署验收；采集合同、坐标根因、失败门控和全分布见[Linux 角点到局部预测 A/B](docs/linux_corner_to_local_prediction.md)。

## 5. 调试界面

先运行一次自瞄任务以生成 `bridge.json` 和 `pipeline.json`，再把实际证据目录传给调试服务器。

Linux：

```bash
python3 ./modules/autoaim/scripts/serve_debug_ui.py \
  --bridge-json <evidence-root>/bridge.json \
  --pipeline-json <evidence-root>/pipeline.json
```

Windows：

```powershell
python .\modules\autoaim\scripts\serve_debug_ui.py `
  --bridge-json <evidence-root>\bridge.json `
  --pipeline-json <evidence-root>\pipeline.json
```

服务器启动后打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。这个地址只在本机调试服务器运行期间有效，不是公网链接。

## 6. 研究文档导航

当前优先阅读：

- [PnP yaw 阶段二与 3/5/7 m 回放](docs/pnp_yaw_stage2.md)
- [四角点联合残差网络 pilot](docs/sim_corner_residual_network.md)
- [组合运动大规模验证](docs/combined_motion_large_scale_validation.md)
- [组合运动 PnP 误差缩减](docs/combined_motion_pnp_error_reduction.md)
- [Linux 角点到局部预测 A/B](docs/linux_corner_to_local_prediction.md)
- [EKF11 与 CV 消融](docs/ekf11_cv_ablation.md)

Stage3 合同与复现资料：

- [数据合同](docs/stage3_data_contract.md)
- [操作手册](docs/stage3_operations.md)
- [执行报告](docs/stage3_execution_report.md)
- [物理核心](docs/stage3_physical_core.md)
- [PnP pose adapter](docs/stage3_pnp_pose_adapter.md)
- [PnP state A/B](docs/stage3_pnp_state_ab.md)
- [四路误差分析](docs/stage3_four_way_error_analysis.md)

其余 `docs/` 文件是保留的实验记录。文档中的本机数据路径用于证据溯源；读者没有对应受保护数据时，应阅读结论和合同，不要把这些路径当成下载链接。

## 7. 链接与内容检查

Linux：

```bash
python3 -B ./scripts/check-doc-links.py
```

Windows：

```powershell
python .\scripts\check-doc-links.py
```

模型、Release 和数据路径不由这个检查器伪装成可下载链接；它们的缺失会由架构检查明确报告。
