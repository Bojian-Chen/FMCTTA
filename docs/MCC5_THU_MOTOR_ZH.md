# MCC5-THU Motor 的 online CTTA 数据协议

## 1. 设计结论

每个有效原始 CSV 是 90 秒、12.8 kHz 的连续记录。数据构建器只保留
`motor_vibration_x/y/z`，直接截取连续、互不重叠的 1024 点窗口（0.08 秒），不重
采样、不插值、不归一化。

每条有效 circulation 记录都生成三个逻辑域：

1. 稳态 1 source 候选：10–30 秒和 60–80 秒；
2. 稳态 2 source 候选：35–55 秒；
3. 0–90 秒完整 target stream。

本地可用的 11 条完整记录因此生成 33 个域，其中 22 个可选 source、11 个完整 target
流。JSON 默认保存全部窗口：稳态 1、稳态 2、完整流每类分别为 500、250、1125 个，
最终 `data.shape = (495000, 3, 1024)`。由于各域长度不同，manifest 会写入
`samples_per_class_by_domain`，输出采用流式 HDF5-backed `.mat`。

## 2. 工况的具体含义与制作过程

### 2.1 torque 与 speed 记录表示什么

MCC5-Motor 的 circulation 记录同样分为两种控制方式：

- `torque, 1000 rpm, 20→15 Nm` 表示转速维持在名义 1000 rpm，主要变化量是
  负载转矩。完整过程是 `0→20 Nm→15 Nm→20 Nm→0`。
- `speed, 40 Nm, 1000→500 rpm` 表示负载维持在名义 40 Nm，主要变化量是
  转速。完整过程是 `0→1000 rpm→500 rpm→1000 rpm→0`。

其余记录按同一规则解释。例如 `torque, 3000 rpm, 40→30 Nm` 是固定 3000 rpm 下的
`0→40→30→40→0 Nm`；`speed, 20 Nm, 2000→1500 rpm` 是固定 20 Nm 下的
`0→2000→1500→2000→0 rpm`。

一条 90 秒记录中的工况时间轴为：

| 时间 | 变化量的状态 | 数据含义 |
| --- | --- | --- |
| 0–10 s | 0 线性变化到平台 1 | 启动或加载过程 |
| 10–30 s | 平台 1 | 第一个稳定工况 |
| 30–35 s | 平台 1 线性变化到平台 2 | 第一次工况漂移 |
| 35–55 s | 平台 2 | 第二个稳定工况 |
| 55–60 s | 平台 2 线性返回平台 1 | 第二次工况漂移 |
| 60–80 s | 平台 1 | 再次回到第一个稳定工况 |
| 80–90 s | 平台 1 线性返回 0 | 卸载或停机过程 |

这里的“平台 1/2”对 torque 文件指两个转矩值，对 speed 文件指两个转速值。完整
target 因而保留真实的连续变化过程，而不是把 20 Nm、15 Nm 等固定工况样本随机混合。

### 2.2 一条原始记录为什么生成三个域

每个故障类别在同一种 circulation 工况下都有一条独立 CSV。构建器从同一条 CSV
建立三个逻辑视图：

1. `steady_1` 取 10–30 秒和 60–80 秒，合并为平台 1 的有标签 source 样本池；两个
   区间之间不做拼接插值，source 训练时会打乱样本。
2. `steady_2` 取 35–55 秒，作为平台 2 的有标签 source 候选。
3. `stream` 保存 0–90 秒的全部窗口及其原始时间顺序，包含启动、两次平台切换和
   卸载过程，作为无标签在线 target。

这三个域不是三次不同采集。source 域用于训练初始故障分类器，完整 stream 用于测试
时在线适配。由于稳态片段本身也包含在同组完整 stream 内，选择某组稳态作为 source
后，必须排除该组的完整 target，防止原始记录泄漏。

### 2.3 多类别如何组成一条在线工况流

24 个故障类别来自 24 次独立采集，不表示一台电机同时存在 24 种故障。由于这些采集
采用相同的 90 秒控制程序，构建器可按相同时间位置对齐；运行时 `interleaved` 形成：

```text
time_0: class_0, class_1, ..., class_23
time_1: class_0, class_1, ..., class_23
...
```

每 24 个样本代表控制轨迹向前推进 0.08 秒。类别标签只参与离线流构造和评价，不会
提供给在线适配算法。`protocol.domains` 定义完整 circulation 记录之间的宏观工况
顺序：当前 90 秒流结束后进入列表中的下一条流，模型的适配状态不会在边界重置。

## 3. 每个 Domain ID 的具体含义

Domain ID 由 `mcc5_motor.json` 中的顺序确定。每条原始 circulation CSV 固定占用
三个连续 ID：

- `3k`：平台 1 的 source 候选，取 10–30 秒和 60–80 秒，每类 500 个窗口；
- `3k+1`：平台 2 的 source 候选，取 35–55 秒，每类 250 个窗口；
- `3k+2`：同一 CSV 的完整 target stream，取 0–90 秒，每类 1125 个窗口。

“source 候选”表示该域允许放入 `source_domains`，并不表示程序永远固定使用它。
实际 source 和 target 仍由实验 YAML 决定。

### 3.1 变转矩域：固定转速，转矩变化

| ID | Manifest 中的域名 | 用途 | 具体工况及保存内容 |
| ---: | --- | --- | --- |
| 0 | `torque_1000rpm_20to15Nm__steady_1_20Nm` | source 候选 | 固定 1000 rpm、20 Nm 平台；取 10–30 s 和 60–80 s |
| 1 | `torque_1000rpm_20to15Nm__steady_2_15Nm` | source 候选 | 固定 1000 rpm、15 Nm 平台；取 35–55 s |
| 2 | `torque_1000rpm_20to15Nm__stream` | target 流 | 固定 1000 rpm，转矩按 `0→20→15→20→0 Nm` 连续变化 |
| 3 | `torque_2000rpm_20to15Nm__steady_1_20Nm` | source 候选 | 固定 2000 rpm、20 Nm 平台；取 10–30 s 和 60–80 s |
| 4 | `torque_2000rpm_20to15Nm__steady_2_15Nm` | source 候选 | 固定 2000 rpm、15 Nm 平台；取 35–55 s |
| 5 | `torque_2000rpm_20to15Nm__stream` | target 流 | 固定 2000 rpm，转矩按 `0→20→15→20→0 Nm` 连续变化 |
| 6 | `torque_3000rpm_20to15Nm__steady_1_20Nm` | source 候选 | 固定 3000 rpm、20 Nm 平台；取 10–30 s 和 60–80 s |
| 7 | `torque_3000rpm_20to15Nm__steady_2_15Nm` | source 候选 | 固定 3000 rpm、15 Nm 平台；取 35–55 s |
| 8 | `torque_3000rpm_20to15Nm__stream` | target 流 | 固定 3000 rpm，转矩按 `0→20→15→20→0 Nm` 连续变化 |
| 9 | `torque_3000rpm_40to30Nm__steady_1_40Nm` | source 候选 | 固定 3000 rpm、40 Nm 平台；取 10–30 s 和 60–80 s |
| 10 | `torque_3000rpm_40to30Nm__steady_2_30Nm` | source 候选 | 固定 3000 rpm、30 Nm 平台；取 35–55 s |
| 11 | `torque_3000rpm_40to30Nm__stream` | target 流 | 固定 3000 rpm，转矩按 `0→40→30→40→0 Nm` 连续变化 |
| 12 | `torque_2000rpm_40to30Nm__steady_1_40Nm` | source 候选 | 固定 2000 rpm、40 Nm 平台；取 10–30 s 和 60–80 s |
| 13 | `torque_2000rpm_40to30Nm__steady_2_30Nm` | source 候选 | 固定 2000 rpm、30 Nm 平台；取 35–55 s |
| 14 | `torque_2000rpm_40to30Nm__stream` | target 流 | 固定 2000 rpm，转矩按 `0→40→30→40→0 Nm` 连续变化 |
| 15 | `torque_1000rpm_40to30Nm__steady_1_40Nm` | source 候选 | 固定 1000 rpm、40 Nm 平台；取 10–30 s 和 60–80 s |
| 16 | `torque_1000rpm_40to30Nm__steady_2_30Nm` | source 候选 | 固定 1000 rpm、30 Nm 平台；取 35–55 s |
| 17 | `torque_1000rpm_40to30Nm__stream` | target 流 | 固定 1000 rpm，转矩按 `0→40→30→40→0 Nm` 连续变化 |

### 3.2 变转速域：固定转矩，转速变化

| ID | Manifest 中的域名 | 用途 | 具体工况及保存内容 |
| ---: | --- | --- | --- |
| 18 | `speed_40Nm_1000to500rpm__steady_1_1000rpm` | source 候选 | 固定 40 Nm、1000 rpm 平台；取 10–30 s 和 60–80 s |
| 19 | `speed_40Nm_1000to500rpm__steady_2_500rpm` | source 候选 | 固定 40 Nm、500 rpm 平台；取 35–55 s |
| 20 | `speed_40Nm_1000to500rpm__stream` | target 流 | 固定 40 Nm，转速按 `0→1000→500→1000→0 rpm` 连续变化 |
| 21 | `speed_40Nm_2000to1500rpm__steady_1_2000rpm` | source 候选 | 固定 40 Nm、2000 rpm 平台；取 10–30 s 和 60–80 s |
| 22 | `speed_40Nm_2000to1500rpm__steady_2_1500rpm` | source 候选 | 固定 40 Nm、1500 rpm 平台；取 35–55 s |
| 23 | `speed_40Nm_2000to1500rpm__stream` | target 流 | 固定 40 Nm，转速按 `0→2000→1500→2000→0 rpm` 连续变化 |
| 24 | `speed_40Nm_3000to2500rpm__steady_1_3000rpm` | source 候选 | 固定 40 Nm、3000 rpm 平台；取 10–30 s 和 60–80 s |
| 25 | `speed_40Nm_3000to2500rpm__steady_2_2500rpm` | source 候选 | 固定 40 Nm、2500 rpm 平台；取 35–55 s |
| 26 | `speed_40Nm_3000to2500rpm__stream` | target 流 | 固定 40 Nm，转速按 `0→3000→2500→3000→0 rpm` 连续变化 |
| 27 | `speed_20Nm_2000to1500rpm__steady_1_2000rpm` | source 候选 | 固定 20 Nm、2000 rpm 平台；取 10–30 s 和 60–80 s |
| 28 | `speed_20Nm_2000to1500rpm__steady_2_1500rpm` | source 候选 | 固定 20 Nm、1500 rpm 平台；取 35–55 s |
| 29 | `speed_20Nm_2000to1500rpm__stream` | target 流 | 固定 20 Nm，转速按 `0→2000→1500→2000→0 rpm` 连续变化 |
| 30 | `speed_20Nm_1000to500rpm__steady_1_1000rpm` | source 候选 | 固定 20 Nm、1000 rpm 平台；取 10–30 s 和 60–80 s |
| 31 | `speed_20Nm_1000to500rpm__steady_2_500rpm` | source 候选 | 固定 20 Nm、500 rpm 平台；取 35–55 s |
| 32 | `speed_20Nm_1000to500rpm__stream` | target 流 | 固定 20 Nm，转速按 `0→1000→500→1000→0 rpm` 连续变化 |

例如，`source_domains: [0, 1]` 表示使用第一条 1000 rpm、20/15 Nm 记录的两个稳态
平台训练 source；`domains: [5, 8]` 表示随后依次输入 2000 rpm 和 3000 rpm 下的完整
20→15 Nm 变转矩流。domain 0/1 只是默认示例，不是固定源域。

## 4. 防止 source/target 原始记录泄漏

选中某三元组内任一稳态作为 source 后，不能再把同组完整流作为 target。例如选择
`source_domains: [12, 13]` 时，target 必须排除 domain 14。manifest 为每个域写入
`recording_group`，FMCTTA 会在加载时拒绝同一原始 CSV 同时出现在 source 与 target。

默认组合是：

```yaml
protocol:
  source_domains: [0, 1]
  domains: [5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
```

例如可改用 3000 rpm、40/30 Nm 两个平台：

```yaml
protocol:
  source_domains: [9, 10]
  domains: [2, 5, 8, 14, 17, 20, 23, 26, 29, 32]
```

## 5. 本地原始数据异常

`speed_circulation_20Nm_3000rpm` 缺少 `bearing_inner_H`，同时有两份
`bearing_outer_H`，无法从文件名可靠恢复标签。该整条工况被排除，未复制样本或猜测
标签；其余 11 条记录均已验证 24 类完整。

## 6. 生成与运行

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/mcc5_motor.json \
  --overwrite

python main.py --config configs/MCC5_motor/unifreq/tent.yaml
```

运行时序遵循 2.3 节的 `interleaved` 工况流，推荐 batch size 为 24 或其整数倍。

数据来源：[MCC5-THU-Motor-Benchmark-Datasets](https://github.com/liuzy0708/MCC5-THU-Motor-Benchmark-Datasets)。
