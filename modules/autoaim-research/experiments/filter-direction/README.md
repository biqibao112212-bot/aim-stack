# 固定输入的滤波器方向实验

这个目录为教程“输入不变，改进滤波器”章节保留可复现分析。它做三件事：

1. 对 Daedalus `1.4.0-learning-r1` 三组 20 s 数据的 PnP 残差做方向、尾部、时间相关和更新间隔检查；
2. 使用相同的同济 11 维状态、转移方程、`Q/R`、PnP 输入与曝光时间戳，离线回放 EKF、UKF 和 bootstrap PF；
3. 从自瞄 B 封存的 `combined-04` 登记中重画“按视线方向分权、跨窗口维护角速度、联合拟合整车轨迹”的对比；
4. 将 2026-08-09 历史方法筛选中的角度指标连同来源哈希保存下来，避免把它与当前三维位置指标混为一谈。

## 三类数据各自回答什么

| 数据 | 用途 | 指标 | 预测时长 |
| --- | --- | --- | --- |
| `20260825-ekf11-tensorrt-r2` | 在同一输入下比较 EKF、UKF 与粒子滤波 | 未来装甲板三维位置误差 P95 | 100/200/300/500 ms |
| `autoaim-b-method-selection-accepted-radii-20260809T080000Z` | 筛选“恒速外推＋历史窗口线性校正”等候选方法 | 条件等权角误差 P95 | 50/100/200 ms |
| `combined-04` | 检查方向分权、跨窗口角速度和整车轨迹模型 | tracker 横向位置误差 P95 | 50/100/200 ms |

三组结果的实验对象和单位不同。第一组是本章的统一主评价；后两组只说明哪些方法值得迁入第一组继续比较。

## 回放合同

- 三种滤波器只能读取 `timestamp_ns`、`primary_pnp.xyz_m` 和 `primary_pnp.yaw_rad`。
- 保存的 `matched_armor_slot` 只选择同一个非线性观测分支，因此结果是排除在线关联错误后的上界。
- 任何坐标、速度、yaw、角速度或半径真值都不进入滤波器。未来真值只在后验评分时读取。
- PF 在前 20 次更新使用因果 EKF 收窄原始 11 维先验，然后切换为 2048 粒子的 bootstrap PF。评分窗从 2 s 开始。
- 三者使用完全相同的 `100 / 200 / 300 / 500 ms` 起点、未来真值插值和位置误差定义。

## 复现

```bash
python3 -m venv /tmp/autoaim-filter-research
/tmp/autoaim-filter-research/bin/pip install -r \
  modules/autoaim-research/experiments/filter-direction/requirements.txt

/tmp/autoaim-filter-research/bin/python \
  modules/autoaim-research/experiments/filter-direction/analyze_filter_direction.py \
  --raw-root /home/potato/Projects/仿真/runtime/autoaim-research/20260825-ekf11-tensorrt-r2 \
  --combined-registry modules/autoaim/docs/combined_motion_final_registry.json \
  --pnp-evidence-registry modules/autoaim/docs/pnp_evidence_registry.json \
  --method-ranking /home/potato/Projects/仿真/runtime/autoaim-b-method-selection-accepted-radii-20260809T080000Z/method-analysis/fair-core-final/summary/method_ranking.csv \
  --output modules/autoaim-research/experiments/filter-direction/results \
  --particles 2048
```

脚本会检查三种方法的评分锚点数完全相同，并输出 PNG/SVG/PDF、结构化统计、数据源哈希和研究报告。原始 JSONL 仍是受保护资产，不复制进 Git。
