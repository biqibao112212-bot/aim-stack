# Aim Stack

RoboMaster 自瞄 B、轨迹研究与打符消费者仓库。模拟器源码已经拆分到独立仓库；本仓库只通过发布版 SDK 和版本锁消费模拟器。

## 从这里开始

- [自瞄 B 教程与文档导航](modules/autoaim/README.md)
- [模拟器消费者统一指南](SIMULATOR_CONSUMER_GUIDE.md)
- [模拟器与 SDK 版本锁](simulator.lock.json)
- [模型资产清单](models/manifest.json)
- [迁移来源](MIGRATION_SOURCES.md)
- [打符模块（当前暂停）](modules/energy-buff/README.md)

仓库中的 `D:\仿真\...` 表示历史 Windows 项目机约定的本地部署位置，不是网页链接。当前 Linux 工作区可使用 `/home/potato/Projects/仿真`，其他机器则可把仓库放在任意有写权限的位置。Git 仓库只包含源码、配置和文档；模拟器 Release、TensorRT engine、训练权重和运行证据不会随 `git clone` 下载。

## 获取源码

### Linux（当前系统）

当前开发环境是 Ubuntu 24.04。Ubuntu/Debian 可安装 Git、Python 和基础构建工具：

```bash
sudo apt update
sudo apt install -y git python3 python3-venv build-essential cmake ninja-build pkg-config
```

Fedora 使用：

```bash
sudo dnf install git python3 gcc-c++ cmake ninja-build pkgconf-pkg-config
```

Arch Linux 使用：

```bash
sudo pacman -S --needed git python base-devel cmake ninja pkgconf
```

然后克隆仓库：

```bash
git clone https://github.com/biqibao112212-bot/aim-stack.git
cd aim-stack
git --version
python3 --version
```

当前这台 Ubuntu 机器已经安装 Git `2.43.0`，无需重复安装。只阅读教程、修改源码、运行文档检查和不依赖私有 SDK 的离线 Python 工具，可以直接在 Linux 完成。

### Windows

先确认 Git 可用：

```powershell
git --version
```

Windows 尚未安装 Git 时，可使用 [Git 官方安装页](https://git-scm.com/downloads/win)，或在终端执行：

```powershell
winget install --id Git.Git -e --source winget
```

然后克隆仓库：

```powershell
git clone https://github.com/biqibao112212-bot/aim-stack.git
Set-Location .\aim-stack
```

只阅读教程和源码不需要私有资产。运行完整仿真链路前，还需要项目所有者提供与 [`simulator.lock.json`](simulator.lock.json) 匹配的模拟器 Release 和与 [`models/manifest.json`](models/manifest.json) 匹配的受保护模型；缺少它们时，启动脚本会明确失败，而不会从无效地址下载替代品。

### Linux 与正式运行链路的边界

当前版本锁是 `Daedalus Simulator 1.2.1 / windows-x86_64`，正式模拟器、Windows 原生桥和端到端采集脚本仍只能在 Windows 项目机运行。工作区中出现的旧 Linux Release 不满足当前版本锁，不能用于兼容性、性能或发布结论。

在正式的同版本 Linux Release 和 Linux SDK 发布并更新版本锁之前，Linux 环境支持的范围是：

- 阅读和维护源码、文档及配置；
- 运行 `python3 -B scripts/check-doc-links.py`；
- 运行不依赖私有模型、Windows 二进制和当前 SDK 的离线 Python 分析；
- 准备 CMake 依赖，但不能声称已经通过完整模拟器或 TensorRT 端到端验收。

## 当前模块

| 模块 | 状态 | 入口 |
| --- | --- | --- |
| 自瞄 B / 神经轨迹研究 | 活跃 | [`modules/autoaim`](modules/autoaim/README.md) |
| Energy Buff | 暂停，尚未完成 SDK v1 适配 | [`modules/energy-buff`](modules/energy-buff/README.md) |

Windows 项目机从仓库根目录运行当前正式链路：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bench-windows-autoaim-e2e.ps1 `
  -WarmupSeconds 5 -DurationSeconds 30 -EnableStage3 `
  -InitialScene shooting_range -TruthGimbalTarget 3
```

该命令依赖统一指南所述的固定工作区布局和项目机运行时。第一次使用前请先阅读[自瞄 B 教程](modules/autoaim/README.md)，不要把旧 WSL 路径或历史工作树当成当前入口。

## 检查文档链接

Linux 上检查仓库内链接、图片和标题锚点：

```bash
python3 -B ./scripts/check-doc-links.py
```

发布前或需要核验网页/示例仓库时，再验证仓库内所有受版本控制文本中的公开 HTTP(S) 地址和 HTTPS Git 克隆地址（包括研究登记和源码溯源注释）：

```bash
python3 -B ./scripts/check-doc-links.py --external
```

`127.0.0.1` / `localhost` 调试地址会被识别为本机临时服务：检查器会记录但不会尝试连接，必须先按对应页面的启动步骤运行服务。

Windows 上运行同一检查：

```powershell
python .\scripts\check-doc-links.py
python .\scripts\check-doc-links.py --external
```

GitHub Actions 会同时运行本地与公开链接检查，避免再次提交指向不存在文件的相对链接或失效的示例仓库。
