# 迁移来源

- `modules/autoaim`：旧共享仓库提交 `76626a4045b5`，分支 `research/learned-state-estimator`。
- `modules/energy-buff`：旧共享仓库提交 `57e7e6297676`，分支 `feature/energy-buff`。
- 装甲板主线 `dd71f5ed09c5` 与火控 `c95a7a18ed70` 当前只保留在旧只读仓库，等待专门迁移，不混入当前主线。
- 所有导入均来自 `git archive` 的已跟踪源码；未导入 build、target、实验帧、日志和未跟踪模型。
