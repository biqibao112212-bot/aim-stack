# 11 维车辆中心 EKF 与 simple-CV 对照实验

## 1. 这次到底测了什么

这是一项离线、oracle physical-slot 的效果对照，不是生产预测器实现。未来真值只在预测完成后用于插值和评分，绝不进入滤波器；历史真值只出现在名称明确的 `exact_truth` 干预组。

本实验采用 RoboMaster 开源 `rm_vision/rm_auto_aim` 的车辆中心 EKF 结构。需要严谨说明：上游实际 EKF 是 9 维

```text
[xc,vxc,yc,vyc,zc,vzc,theta,omega,r]
```

而 `another_r` 与 `dz` 保存在 tracker 外部。为了按本项目常见目标状态统一比较，本实验将两项几何量折入滤波器，得到可审计的 11 维状态：

```text
[xc,vxc,yc,vyc,zc,vzc,theta,omega,r_even,r_odd,dz_odd]
```

状态转移、车辆中心到装甲板观测方程以及 `Q/R/P0` 取自上游公开实现；两个半径和高度差沿用上游 radius 随机游走尺度。没有加入本仓库生产 tracker 中的 NIS gate、装甲板跳变恢复、几何恢复、速度拟合或其他工程启发式。因此本文所说的“11 维 EKF”准确含义是“`rm_vision` 9 维 EKF 加上其外置两项几何状态的透明 11 维展开”，不是声称上游原文件本身就是 11 维。

上游依据：

- `tracker_node.cpp`：9 维状态、状态转移、观测函数和 Q/R；
- `tracker.hpp`：外置 `another_r`、`dz`；
- `tracker.cpp`：半径切换与高度差更新。

## 2. 公平性与因果条件

全部方法使用相同 target-slot 评测锚点、真实事件时间以及 0/50/100/200 ms 未来时域。锚点历史至少 16 个样本，跨度不超过 0.75 s，相邻 gap 不超过 120 ms；未来真值插值 bracket 不超过 40 ms；评测锚点至少间隔 100 ms。

方法必须用完整名称区分：

| 方法 | 历史输入 | 坐标系/记忆 |
| --- | --- | --- |
| `hold_camera` | 当前 PnP 或历史真值 | 保持当前相机坐标；复现旧基线 |
| `cv_ols_camera_16` | 同一物理板最近 16 点 | 移动相机坐标 OLS-CV；复现此前 simple-CV |
| `cv_ols_world_16` | 同一物理板最近 16 点 | 世界坐标 OLS-CV；与 EKF 的目标运动坐标一致 |
| `ekf11_window16_single_slot` | 同一物理板最近 16 点 | 与 world-CV 等输入支持，逐锚点重建 |
| `ekf11_persistent_single_slot` | 同一物理板因果历史 | 观测 gap 超过 120 ms 后 LOST-style 重置 |
| `ekf11_persistent_oracle_multislot` | 所有可见板因果历史 | oracle slot；gap 超过 120 ms 后重置 |

camera-CV 会同时继承云台跟踪造成的相机坐标运动；world-CV 才直接拟合目标世界运动。二者不是同一种算法，不能继续简称成一个模糊的“CV”。

## 3. 指标口径纠正

用户指定的主指标是去掉相机深度轴 `z` 后另外两个方向的误差合向量：

```text
camera_lateral_xy_error = sqrt(error_x^2 + error_y^2)
```

逐样本表同时保存 signed `error_x/error_y/error_z`、深度绝对误差、三维误差、truth-depth yaw/pitch plane miss 和射线角误差。

此前引用的平移 `11.9/15.4/25.2 mm` 与旋转 `12.5/20.3/37.2 mm`，准确含义是 camera-CV 的**单轴 truth-depth yaw-plane miss P95**。同一批样本改用本次指定的原始相机 `x/y` 合向量后为：

| 运动 | 50 ms | 100 ms | 200 ms |
| --- | ---: | ---: | ---: |
| translation，camera-CV，二维横向 | 27.0 mm | 34.8 mm | 42.2 mm |
| rotation，camera-CV，二维横向 | 22.5 mm | 31.3 mm | 54.1 mm |

55 mm 线画在二维范数上时，比原单轴 yaw gate 更严格；它仍只是离线诊断线，不是完整命中率或火控容差。

## 4. 当前 PnP 输入下的方法对比

下表为二维横向 P95；`n` 是有效未来比较数，不是原始帧数。

| 运动 | 时域 | camera-CV | world-CV | EKF11 同 16 点 | EKF11 oracle 多板因果段 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stationary | 50/100/200 ms | 27.8/27.8/27.8 | 27.8/27.8/27.8 | 27.8/27.8/27.8 | 27.8/27.8/27.8 |
| translation | 50/100/200 ms | 27.0/34.8/42.2 | 26.6/36.2/52.3 | 469.8/455.6/449.0 | 537.3/592.0/820.9 |
| rotation | 50/100/200 ms | 22.5/31.3/54.1 | 22.5/31.3/54.1 | 3498.1/3871.8/4587.8 | 4239.4/5505.9/7967.8 |
| combined | 50/100/200 ms | 51.7/105.9/108.8 | 96.1/180.8/368.1 | 5025.6/2957.5/5872.9 | 5025.6/2957.5/5872.9 |

对应样本量为：stationary `1780/1747/1745`，translation `1646/1540/1358`，rotation `841/520/283`，combined `203/34/13`。

结论很直接：未经本数据校准的裸 11 维 EKF 不能接收当前 PnP 位置/yaw 后直接部署。它会把深度长尾、yaw 畸变和半径/中心耦合吸收到连续状态中，误差从厘米级放大到米级。oracle 多板在绝大多数锚点与单板结果相同，因为可见板切换通常跨过 120 ms 支持 gap，触发 LOST-style 重置；不能把 oracle 名称误读成实际获得了持续多板几何约束。

## 5. 位置真值与 yaw 真值消融

以 `ekf11_persistent_oracle_multislot` 的 100 ms 二维横向 P95 为例：

| 运动 | 当前位置 + 当前 yaw | 真值位置 + 当前 yaw | 当前位置 + 真值 yaw | 真值位置 + 真值 yaw |
| --- | ---: | ---: | ---: | ---: |
| stationary | 27.8 | 0.0 | 27.8 | 0.0 |
| translation | 592.0 | 175.8 | 408.3 | 178.4 |
| rotation | 5505.9 | 49.5 | 8794.2 | 18.6 |
| combined | 2957.5 | 162.3 | 5485.8 | 133.9 |

这组消融不是简单可加的：在 rotation/combined 中，单独恢复 yaw 反而可能破坏当前 PnP 位置与错误 yaw 之间偶然形成的补偿。能够稳健读出的结论是：

1. 当前 PnP 位置是 11 维中心/半径状态崩坏的首要输入瓶颈；恢复位置真值后，米级错误回到厘米至分米级。
2. yaw 误差仍重要。逐 session 的 observed-vs-truth yaw 绝对误差 P95 中位数约为 translation 0.551 rad、rotation 0.748 rad、combined 0.924 rad；它不是可直接当高置信状态的测量。
3. 位置和 yaw 必须联合校准，不能根据一个单变量 oracle arm 独立承诺收益。

## 6. 真值输入下的纯运动模型上限

以下各方法只接收历史真值；未来真值仍只用于事后评分。

| 运动 | 方法 | 50 ms | 100 ms | 200 ms |
| --- | --- | ---: | ---: | ---: |
| translation | world-CV | 0.0 | 0.0 | 0.0 |
| translation | EKF11 oracle 多板 | 133.2 | 178.4 | 273.9 |
| rotation | world-CV | 7.9 | 9.8 | 13.9 |
| rotation | EKF11 oracle 多板 | 17.5 | 18.6 | 27.4 |
| combined | world-CV | 97.9 | 182.0 | 344.7 |
| combined | EKF11 oracle 多板 | 88.6 | 133.9 | 215.1 |

这正面回答“即使有真实输入，普通 CV 能否处理组合运动”：不能。world-CV 在 combined 的 50/100/200 ms 均超过 55 mm，100/200 ms 达 182.0/344.7 mm。11 维中心+旋转分解相对 world-CV 将 combined P95 分别降低约 9.5%/26.4%/37.6%，说明结构建模有价值；但绝对误差仍为 88.6/133.9/215.1 mm，仍不满足当前二维 55 mm 诊断线。

同一 11 维 EKF 对纯平移和纯旋转反而弱于更简单的 world-CV，说明当前 Q/R、短可见段、重置和几何状态并非通用最优。不能因为 combined 有相对改善就替换全部运动模式。

camera-CV 的 exact combined 为 49.2/110.2/95.0 mm，看上去优于 world-CV，是因为它利用了移动相机/云台坐标中的补偿；这可作为系统级相机射线预测基线，但不能冒充车辆世界运动模型上限。

## 7. 深度仍需单独保留

camera-CV 当前 PnP 的深度 P95 在 translation 为 `0.784/0.876/1.137 m`，rotation 为 `0.682/0.847/1.449 m`，combined 为 `0.673/1.223/1.846 m`。这些深度误差没有混入二维横向主指标，但会破坏车辆中心、半径、弹道飞行时间和三维关联；它正是 11 维 EKF 当前输入下失败的重要原因之一。

## 8. 当前研究判断

1. 平移不是现阶段主要预测难点。camera-CV 和 world-CV 的二维横向 P95 在本批 oracle-identity 数据的 50/100/200 ms 都不超过 55 mm；但这不是实弹命中结论，且 camera-CV 包含云台补偿。
2. rotation 在当前支持可见弧内，16 点 CV 比这版裸 11 维 EKF 更可靠；200 ms 二维 P95 已接近 55 mm，且高角速长时域覆盖仍不足。
3. combined 才是纯运动模型难点。真值历史下 11 维 EKF 比 world-CV 好，但仍明显失败，需要专门的平移中心 + 周期相位/旋转因子化模型、IMM 或多假设，而不是继续调一个通用 CV。
4. 在任何车辆中心/半径 EKF 进入生产前，必须先解决 PnP depth/yaw 联合校准、物理身份、可见弧/reacquisition 和 condition-wise covariance coverage。
5. 本实验不修改生产 `RobotEstimator`，不恢复生产预测器，也不接受 oracle slot 为部署输入。

## 9. 完整证据

正式目录：

```text
D:\仿真\runtime\rmvision-ekf11-ablation-v1-full
```

其中保留：

- `prediction_samples.csv.gz`：307,152 条逐预测样本，含完整状态、输入干预标签、signed x/y/z、二维横向、深度、三维和角度误差；
- `prediction_metrics.csv`：按运动、速度、距离、方法、输入和时域的 P50/P75/P90/P95/P99；
- `lateral_xy_histogram_distribution.csv`：固定分箱完整分布；
- `joined_visible_observations.csv.gz`：77,518 条 PnP/truth 与 measured/exact yaw；
- `primary_p95_lateral_xy_mm.csv`、`anchor_coverage.csv`、`yaw_join_diagnostics.csv`、`coordinate_transform_diagnostics.csv`；
- current/exact P95、100 ms ECDF、输入消融和预测/真值轨迹图，均保存 PNG/SVG/PDF；
- `manifest.json`：源文件、状态/噪声/因果合同及所有产物的大小与 SHA-256。

逐帧 measured yaw 匹配率为 100%；逐帧世界到相机坐标重建的最坏 session P95 残差为 `1.646e-6 m`。完整分布而不是本文分位数摘要是最终证据。
