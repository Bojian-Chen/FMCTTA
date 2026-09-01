# MCC5-THU Gearbox 的 online CTTA 数据协议

## 1. 设计结论

每个原始 CSV 是 60 秒、12.8 kHz 的连续记录。数据构建器保留电机侧和齿轮箱侧的
`motor_vibration_x/y/z`、`gearbox_vibration_x/y/z` 六个振动通道，直接截取连续、
互不重叠的 1024 点窗口（0.08 秒），不重采样、不插值、不归一化。

12 条 circulation 记录中的每一条都生成三个逻辑域：

1. 稳态 1 source 候选：10–20 秒和 40–50 秒；
2. 稳态 2 source 候选：25–35 秒；
3. 0–60 秒完整 target stream。

因此共有 36 个域，其中 24 个可选 source、12 个完整 target 流。JSON 默认保存全部
窗口：稳态 1、稳态 2、完整流每类分别为 250、125、749 个，最终
`data.shape = (269760, 6, 1024)`。完整流末尾不足 1024 点的余数不会填充或插值。
由于各域长度不同，manifest 会写入
`samples_per_class_by_domain`，输出采用流式 HDF5-backed `.mat`。

## 2. 工况的具体含义与制作过程

### 2.1 torque 与 speed 记录表示什么

MCC5-Gearbox 的 circulation 记录分为两种控制方式：

- `torque, 1000 rpm, 10→5 Nm` 表示转速维持在名义 1000 rpm，主要变化量是
  负载转矩。完整过程是 `0→10 Nm→5 Nm→10 Nm→0`。
- `speed, 20 Nm, 1000→500 rpm` 表示负载维持在名义 20 Nm，主要变化量是
  转速。完整过程是 `0→1000 rpm→500 rpm→1000 rpm→0`。

其余 torque/speed 文件按相同方式解释，只需将示例中的固定量和两个平台值替换为
Domain 表中的数值。例如 `torque, 3000 rpm, 20→15 Nm` 是固定 3000 rpm 下的
`0→20→15→20→0 Nm`；`speed, 10 Nm, 3000→2500 rpm` 是固定 10 Nm 下的
`0→3000→2500→3000→0 rpm`。

一条 60 秒记录中的工况时间轴为：

| 时间 | 变化量的状态 | 数据含义 |
| --- | --- | --- |
| 0–10 s | 0 线性变化到平台 1 | 启动或加载过程 |
| 10–20 s | 平台 1 | 第一个稳定工况 |
| 20–25 s | 平台 1 线性变化到平台 2 | 第一次工况漂移 |
| 25–35 s | 平台 2 | 第二个稳定工况 |
| 35–40 s | 平台 2 线性返回平台 1 | 第二次工况漂移 |
| 40–50 s | 平台 1 | 再次回到第一个稳定工况 |
| 50–60 s | 平台 1 线性返回 0 | 卸载或停机过程 |

这里的“平台 1/2”对 torque 文件指两个转矩值，对 speed 文件指两个转速值。因而完整
target 不是把若干固定工况随机混合，而是保留“启动—稳态—漂移—稳态—漂移—稳态—
卸载”的连续工况轨迹。

### 2.2 一条原始记录为什么生成三个域

每个故障类别在同一种 circulation 工况下都有一条独立 CSV。构建器对同一条 CSV
建立三个逻辑视图，并不是进行了三次不同的物理采集：

1. `steady_1` 取 10–20 秒和 40–50 秒，合并为平台 1 的有标签 source 样本池；两个
   区间之间不做拼接插值，source 训练时会打乱样本。
2. `steady_2` 取 25–35 秒，作为平台 2 的有标签 source 候选。
3. `stream` 保存从 0 秒开始的全部完整窗口，按原始时间顺序保留启动、两次平台切换
   和卸载过程，作为无标签在线 target。

因此 source 域用于训练初始故障分类器，完整 stream 用于测试时在线适配。稳态窗口也
包含在同组完整 stream 中，所以实验必须在“同组稳态 source”和“同组完整 target”
之间二选一，不能把同一原始 CSV 同时用于两端。

### 2.3 多类别如何组成一条在线工况流

20 个故障类别来自 20 次独立采集，并不是同一台设备在同一时刻同时具有 20 种故障。
构建器利用相同的控制程序和采样率，将各类别的相同时间窗口对齐；运行时
`interleaved` 形成：

```text
time_0: class_0, class_1, ..., class_19
time_1: class_0, class_1, ..., class_19
...
```

所以每 20 个样本代表控制轨迹向前推进 0.08 秒。类别标签只用于离线对齐和最终评价，
不会传给在线适配算法。`protocol.domains` 则定义更高一层的工况顺序：前一个完整
circulation 流结束后，再进入列表中的下一个流，适配状态在域边界不重置。

## 3. 每个 Domain ID 的具体含义

Domain ID 由 `mcc5_gearbox.json` 中的顺序确定。每条原始 circulation CSV 固定占用
三个连续 ID：

- `3k`：平台 1 的 source 候选，取 10–20 秒和 40–50 秒，每类 250 个窗口；
- `3k+1`：平台 2 的 source 候选，取 25–35 秒，每类 125 个窗口；
- `3k+2`：同一 CSV 的完整 target stream，取约 0–60 秒，每类 749 个窗口。

“source 候选”表示该域允许放入 `source_domains`，并不表示程序永远固定使用它。
实际 source 和 target 仍由实验 YAML 决定。

### 3.1 变转矩域：固定转速，转矩变化

| ID | Manifest 中的域名 | 用途 | 具体工况及保存内容 |
| ---: | --- | --- | --- |
| 0 | `torque_1000rpm_10to5Nm__steady_1_10Nm` | source 候选 | 固定 1000 rpm、10 Nm 平台；取 10–20 s 和 40–50 s |
| 1 | `torque_1000rpm_10to5Nm__steady_2_5Nm` | source 候选 | 固定 1000 rpm、5 Nm 平台；取 25–35 s |
| 2 | `torque_1000rpm_10to5Nm__stream` | target 流 | 固定 1000 rpm，转矩按 `0→10→5→10→0 Nm` 连续变化 |
| 3 | `torque_2000rpm_10to5Nm__steady_1_10Nm` | source 候选 | 固定 2000 rpm、10 Nm 平台；取 10–20 s 和 40–50 s |
| 4 | `torque_2000rpm_10to5Nm__steady_2_5Nm` | source 候选 | 固定 2000 rpm、5 Nm 平台；取 25–35 s |
| 5 | `torque_2000rpm_10to5Nm__stream` | target 流 | 固定 2000 rpm，转矩按 `0→10→5→10→0 Nm` 连续变化 |
| 6 | `torque_3000rpm_10to5Nm__steady_1_10Nm` | source 候选 | 固定 3000 rpm、10 Nm 平台；取 10–20 s 和 40–50 s |
| 7 | `torque_3000rpm_10to5Nm__steady_2_5Nm` | source 候选 | 固定 3000 rpm、5 Nm 平台；取 25–35 s |
| 8 | `torque_3000rpm_10to5Nm__stream` | target 流 | 固定 3000 rpm，转矩按 `0→10→5→10→0 Nm` 连续变化 |
| 9 | `torque_3000rpm_20to15Nm__steady_1_20Nm` | source 候选 | 固定 3000 rpm、20 Nm 平台；取 10–20 s 和 40–50 s |
| 10 | `torque_3000rpm_20to15Nm__steady_2_15Nm` | source 候选 | 固定 3000 rpm、15 Nm 平台；取 25–35 s |
| 11 | `torque_3000rpm_20to15Nm__stream` | target 流 | 固定 3000 rpm，转矩按 `0→20→15→20→0 Nm` 连续变化 |
| 12 | `torque_2000rpm_20to15Nm__steady_1_20Nm` | source 候选 | 固定 2000 rpm、20 Nm 平台；取 10–20 s 和 40–50 s |
| 13 | `torque_2000rpm_20to15Nm__steady_2_15Nm` | source 候选 | 固定 2000 rpm、15 Nm 平台；取 25–35 s |
| 14 | `torque_2000rpm_20to15Nm__stream` | target 流 | 固定 2000 rpm，转矩按 `0→20→15→20→0 Nm` 连续变化 |
| 15 | `torque_1000rpm_20to15Nm__steady_1_20Nm` | source 候选 | 固定 1000 rpm、20 Nm 平台；取 10–20 s 和 40–50 s |
| 16 | `torque_1000rpm_20to15Nm__steady_2_15Nm` | source 候选 | 固定 1000 rpm、15 Nm 平台；取 25–35 s |
| 17 | `torque_1000rpm_20to15Nm__stream` | target 流 | 固定 1000 rpm，转矩按 `0→20→15→20→0 Nm` 连续变化 |

### 3.2 变转速域：固定转矩，转速变化

| ID | Manifest 中的域名 | 用途 | 具体工况及保存内容 |
| ---: | --- | --- | --- |
| 18 | `speed_20Nm_1000to500rpm__steady_1_1000rpm` | source 候选 | 固定 20 Nm、1000 rpm 平台；取 10–20 s 和 40–50 s |
| 19 | `speed_20Nm_1000to500rpm__steady_2_500rpm` | source 候选 | 固定 20 Nm、500 rpm 平台；取 25–35 s |
| 20 | `speed_20Nm_1000to500rpm__stream` | target 流 | 固定 20 Nm，转速按 `0→1000→500→1000→0 rpm` 连续变化 |
| 21 | `speed_20Nm_2000to1500rpm__steady_1_2000rpm` | source 候选 | 固定 20 Nm、2000 rpm 平台；取 10–20 s 和 40–50 s |
| 22 | `speed_20Nm_2000to1500rpm__steady_2_1500rpm` | source 候选 | 固定 20 Nm、1500 rpm 平台；取 25–35 s |
| 23 | `speed_20Nm_2000to1500rpm__stream` | target 流 | 固定 20 Nm，转速按 `0→2000→1500→2000→0 rpm` 连续变化 |
| 24 | `speed_20Nm_3000to2500rpm__steady_1_3000rpm` | source 候选 | 固定 20 Nm、3000 rpm 平台；取 10–20 s 和 40–50 s |
| 25 | `speed_20Nm_3000to2500rpm__steady_2_2500rpm` | source 候选 | 固定 20 Nm、2500 rpm 平台；取 25–35 s |
| 26 | `speed_20Nm_3000to2500rpm__stream` | target 流 | 固定 20 Nm，转速按 `0→3000→2500→3000→0 rpm` 连续变化 |
| 27 | `speed_10Nm_3000to2500rpm__steady_1_3000rpm` | source 候选 | 固定 10 Nm、3000 rpm 平台；取 10–20 s 和 40–50 s |
| 28 | `speed_10Nm_3000to2500rpm__steady_2_2500rpm` | source 候选 | 固定 10 Nm、2500 rpm 平台；取 25–35 s |
| 29 | `speed_10Nm_3000to2500rpm__stream` | target 流 | 固定 10 Nm，转速按 `0→3000→2500→3000→0 rpm` 连续变化 |
| 30 | `speed_10Nm_2000to1500rpm__steady_1_2000rpm` | source 候选 | 固定 10 Nm、2000 rpm 平台；取 10–20 s 和 40–50 s |
| 31 | `speed_10Nm_2000to1500rpm__steady_2_1500rpm` | source 候选 | 固定 10 Nm、1500 rpm 平台；取 25–35 s |
| 32 | `speed_10Nm_2000to1500rpm__stream` | target 流 | 固定 10 Nm，转速按 `0→2000→1500→2000→0 rpm` 连续变化 |
| 33 | `speed_10Nm_1000to500rpm__steady_1_1000rpm` | source 候选 | 固定 10 Nm、1000 rpm 平台；取 10–20 s 和 40–50 s |
| 34 | `speed_10Nm_1000to500rpm__steady_2_500rpm` | source 候选 | 固定 10 Nm、500 rpm 平台；取 25–35 s |
| 35 | `speed_10Nm_1000to500rpm__stream` | target 流 | 固定 10 Nm，转速按 `0→1000→500→1000→0 rpm` 连续变化 |

例如，`source_domains: [0, 1]` 表示使用第一条 1000 rpm、10/5 Nm 记录的两个稳态平台
训练 source；`domains: [5, 8]` 表示随后依次输入 2000 rpm 和 3000 rpm 下的完整
10→5 Nm 变转矩流。domain 0/1 只是默认示例，不是固定源域。

## 4. 防止 source/target 原始记录泄漏

选中某三元组内任一稳态作为 source 后，不能再把同组完整流作为 target。例如选择
`source_domains: [3, 4]` 时，target 必须排除 domain 5。若 source 来自多个三元组，
应排除每个对应的完整流。

manifest 为每个域写入 `recording_group`，FMCTTA 在加载时会检查 source/target 是否来自
同一原始 CSV，冲突时直接报错。默认组合是：

```yaml
protocol:
  source_domains: [0, 1]
  domains: [5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
```

例如可改用高转速、高负载平台：

```yaml
protocol:
  source_domains: [9, 10]
  domains: [2, 5, 8, 14, 17, 20, 23, 26, 29, 32, 35]
```

## 5. 建议的实验设计

### 5.1 实验目标

建议围绕以下四个问题组织实验：

1. CTTA 方法相对不适配的 source-only 模型，能否改善连续变工况下的故障识别？
2. 方法对“固定转速、转矩变化”和“固定转矩、转速变化”是否同样有效？
3. source 稳态工况的选择是否会显著影响跨工况泛化？
4. 保持真实时间顺序是否重要，适配 batch 覆盖多少个工况时刻最合适？

### 5.2 主实验的统一设置

为保证方法之间可比，主实验建议固定以下设置：

| 项目 | 建议设置 | 含义 |
| --- | --- | --- |
| source | `source_domains: [0, 1]` | 使用 1000 rpm、10 Nm 与 5 Nm 两个平台 |
| source 样本 | `source_samples_per_class: ""` | 使用两个 source 域中每类全部 250+125 个窗口 |
| target 样本 | `target_samples_per_class: ""` | 每个完整流使用每类全部 749 个窗口 |
| target 顺序 | `stream_order: interleaved` | 按原始工况时间推进，不打乱未来和过去 |
| target batch | `stream_batch_size: 20` | 一个 batch 恰好对应一个 0.08 秒工况时刻的 20 类样本 |
| 模型 | `resnet32` + `linear` | 所有方法使用相同分类模型 |
| 输入 | `frequency` + `zscore` | 归一化统计量只由 source-train 拟合 |
| 在线协议 | 单遍、域间不重置 | target 标签不参与适配或超参数选择 |
| 随机种子 | 至少 3 个 | 建议 2023、2024、2025，报告均值与标准差 |

当前若只设置 `domains: [17]`，只能得到单一 target 流的调试结果，不建议把它作为完整
基准结论。正式主实验可拆成三个互补协议。

### 5.3 按工况类型拆分主实验

三个协议都使用 source `[0, 1]`，因此必须排除同组 stream 2。

| 协议 | `domains` | 要回答的问题 |
| --- | --- | --- |
| P-Torque | `[5, 8, 11, 14, 17]` | 从 1000 rpm、10/5 Nm 稳态出发，适应不同转速和不同转矩幅值的变转矩流 |
| P-Speed | `[20, 23, 26, 29, 32, 35]` | 从变转矩记录的稳态出发，跨控制模式适应固定负载下的变转速流 |
| P-Mixed | `[5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]` | 先经历全部变转矩流，再切换到变转速流，评价长序列持续适配和遗忘 |

P-Torque 中的顺序先将转速从 2000 rpm 提高到 3000 rpm，再切换到更高转矩幅值并把
转速降回 1000 rpm。P-Speed 先在固定 20 Nm 下提高转速范围，再切换到固定 10 Nm 并
反向降低转速范围。这个“蛇形”顺序减少无意义的大跳变，同时仍包含多次明显分布漂移。
P-Mixed 应作为主要 CTTA 结果，另外两个协议用于解释性能来自哪一类工况。

### 5.4 12 组留一原始记录实验

仅使用 `[0, 1]` 作为 source 可能让结论依赖某一个起始工况。建议增加 12-fold
leave-one-circulation-out 实验。第 `k` 折定义为：

```text
source_domains = [3k, 3k+1]
target_domains = [2, 5, 8, ..., 35] 去掉 3k+2
```

例如：

- 第 0 折：source `[0, 1]`，target 排除 2；
- 第 3 折：source `[9, 10]`，target 排除 11；
- 第 11 折：source `[33, 34]`，target 排除 35。

每折都从一条 circulation 的两个稳态平台训练，再测试其余 11 条完整流。最终先计算
每折内部的 target 指标，再对 12 折做宏平均。这样可以评价方法是否只对某个特定
转速、转矩或控制模式有效。每折必须使用独立的 source checkpoint 和 run name。

### 5.5 方法对比

建议至少包含以下配置：

| 类别 | 方法 | 作用 |
| --- | --- | --- |
| 下界 | Source | 参数固定，不进行在线适配 |
| 非参数基线 | kNN | 固定源域特征库，不更新模型 |
| 熵最小化 | TENT、EATA、SAR | 比较基础、样本筛选和 SAM 稳定化策略 |
| 教师/记忆方法 | CoTTA、NOTE、RoTTA | 比较教师模型、样本记忆和类别平衡机制 |
| 辅助任务 | TTT、Continual-MAE | 比较变换预测和重建式在线学习 |
| 拓扑方法 | TCA | 比较类别结构保持与对齐策略 |

除 TTT 外，其余方法应复用同一折、同一种子的 source checkpoint。TTT 的 checkpoint
包含额外的变换预测头，必须独立完成 source 联合训练。不要使用 target 标签为每个方法
分别挑选学习率或停止时刻。

### 5.6 评价指标与工况曲线

建议同时报告：

- 每个 target domain 的 online accuracy 和 balanced accuracy；
- 所有样本加权的总体准确率，以及各 domain 等权的 domain-average accuracy；
- 每经过一个 target domain 后的 source-retention accuracy 和 source forgetting；
- P-Mixed 最终时刻的 source retention，用于衡量长序列灾难性遗忘；
- 至少 3 个随机种子的均值与标准差。

除汇总表外，建议绘制“准确率—工况时间”曲线。batch size 为 20 时，第 `b` 个 batch
对应约 `b × 0.08 s`。在 10、20、25、35、40、50 秒处画竖线，就能观察启动、稳态、
两次工况漂移和卸载阶段分别造成多大性能变化。不同 domain 之间再用分隔线标明完整
circulation 的切换。这类曲线比单一总体准确率更能说明方法是否真正跟踪了工况变化。

### 5.7 必要消融实验

| 消融因素 | 设置 | 目的 |
| --- | --- | --- |
| 流顺序 | `interleaved` / `shuffled` / `class_block` | 验证真实时间顺序的重要性；`shuffled` 只作为负对照 |
| batch size | 20 / 40 / 80 | 分别让一次更新观察 1 / 2 / 4 个相邻工况时刻 |
| source 平台 | `[0]` / `[1]` / `[0, 1]` | 比较单平台与双平台监督训练；三者都必须排除 target 2 |
| source 数量 | 全部 / 每类固定 100 | 区分 source 数据量增加与 source 工况覆盖增加的作用 |
| 输入表示 | `time` / `frequency` | 判断频谱表示是否更适合变转速或变转矩漂移 |

主表应始终使用 `interleaved` 和 batch size 20；其他顺序与 batch size 只进入消融表，
避免把被打乱的 target 当作持续变工况主结果。

### 5.8 推荐执行顺序

1. 只构建一次完整 MCC5-Gearbox 数据集；不同实验复用同一 MAT 和 manifest。
2. 对每个 source 设置和随机种子先运行 `source.yaml`，保存唯一 checkpoint。
3. 用该 checkpoint 依次运行 Source、kNN、TENT、EATA、SAR、CoTTA、NOTE、RoTTA、
   Continual-MAE 和 TCA；TTT 单独训练并运行。
4. 先完成 P-Torque、P-Speed、P-Mixed，再执行 12-fold 和消融实验。
5. 保存完整 YAML、seed、target 域顺序和 checkpoint 元数据，避免只记录最终准确率。

建议为 P-Torque、P-Speed、P-Mixed 和每个 fold 分别创建独立 YAML，而不是反复手工
修改 `base.yaml`，这样更容易复现实验并避免 checkpoint 路径相互覆盖。

## 6. 生成与运行

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/mcc5_gearbox.json \
  --overwrite

python main.py --config configs/MCC5_gearbox/tent.yaml
```

运行时序遵循 2.3 节的 `interleaved` 工况流，推荐 batch size 为 20 或其整数倍。

数据来源：[MCC5-THU-Gearbox-Benchmark-Datasets](https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets)。
