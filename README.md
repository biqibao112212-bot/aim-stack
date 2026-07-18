# Aim Stack

自瞄与打符算法消费者仓库。模拟器已经拆分到独立仓库，不再作为本仓库的分支或源码目录。

```text
D:\仿真\repos\aim-stack\
  modules\autoaim\          当前自瞄 B / 神经轨迹预测器
  modules\energy-buff\      打符模块，当前暂停
  agent-team\               三个中文活动上下文
  models\manifest.json      外部模型资产与哈希
  simulator.lock.json       固定模拟器/SDK/Release
  scripts\                  启动与轻量架构检查
```

正式模拟器仓库：`D:\仿真\repos\daedalus-simulator`

正式发布目录：`D:\仿真\releases\daedalus-simulator`
受保护模型目录：`D:\仿真\models\engines`

默认启动自瞄 B：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-autoaim-b.ps1
```

可视验收：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-autoaim-b.ps1 -Visible
```
