# 11 维 EKF 异常基线

这组实验只回答一个问题：现有 11 维 EKF 维护的角速度、中心速度和长短半径，
与模拟器真值相比呈现什么偏差。它是后续讨论原因与改进方案的起点，不预设结论。

## 固定条件

- Daedalus `1.4.0-learning-r1` Linux x86_64 正式 Release。
- `--performance` 高性能无前端模式，不启动可见窗口。
- `modules/autoaim-research` 内的同济 `sp_vision_25` 11 维 EKF。
- 3 号靶车；独立真值云台只控制未来视角，不向 detector、PnP 或 EKF 提供目标真值。
- 每组以曝光 `timestamp_ns` 截取 20 s，并保存
  `(producer_epoch, frame_seq, timestamp_ns)` 三元身份。

可复现采集：

```bash
python3 modules/autoaim-research/experiments/ekf11-baseline/collect.py \
  --output-root /home/potato/Projects/仿真/runtime/autoaim-research/<new-run-id> \
  --duration-s 20 --settle-s 1
```

绘图使用固定 `matplotlib==3.10.5`，生成脚本保留在
[`figures/gen_fig_ekf11_baseline.py`](figures/gen_fig_ekf11_baseline.py)。原始 JSONL 作为受保护实验证据留在
`runtime/autoaim-research/20260825-ekf11-baseline-r2`；接受锁和哈希见
[`accepted-run.json`](accepted-run.json)。

## 图与初始观察

| 工况 | 时间窗 | 处理 FPS | 源序列速率 | 角速度 RMSE | 中心速率 RMSE | 长/短半径 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原地 `8 rad/s` | 20.013 s | 47.3 | 241.6 Hz | 0.774 rad/s | 0.172 m/s | 2.21 / 1.80 cm |
| 平移 `1.5 m/s` | 20.005 s | 46.2 | 229.7 Hz | 0.650 rad/s | 0.351 m/s | 7.00 / 10.82 cm |
| `1 m/s + 6 rad/s` | 20.003 s | 42.2 | 210.7 Hz | 0.653 rad/s | 0.176 m/s | 4.21 / 3.15 cm |

- [原地旋转四联图](figures/fig_spin_8.png)
- [平移四联图](figures/fig_translate_1p5.png)
- [平移加旋转四联图](figures/fig_translate_1_spin_6.png)

图中的启动过渡、丢检和短时跳变均保留，未插值或裁掉。当前只能说明这套
11 维状态与真值存在可见偏差；偏差是由 PnP 观测、数据关联、滤波器模型还是参数引起，
需要后续对照实验，这里不把任何一种解释写成标准答案。
