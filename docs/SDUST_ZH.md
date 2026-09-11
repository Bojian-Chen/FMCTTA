# SDUST-Bearing 与 SDUST-Gear 的 online CTTA 数据协议

## 1. 当前数据制作结论

本项目中的 SDUST 包含两个独立子数据集：

- `SDUST-Bearing`：轴承故障诊断，工况由转速与径向负载共同定义；
- `SDUST-Gear`：行星齿轮箱故障诊断，包含固定转速、连续变负载以及转速/负载共同波动工况。

它们与 MCC5 的制作方式不同。MCC5 会从同一条连续记录中切出稳态 source 候选和完整
target stream；SDUST 则把原始文件中已经定义好的某个工况直接映射成一个 domain，不再
从同一条记录派生 `steady_1/steady_2/stream` 三个逻辑域。

当前配置的实际输出为：

| 数据集 | 类别数 | 域数 | 当前输出通道 | 每类每域窗口数 | 最终 `data` 形状 |
| --- | ---: | ---: | --- | ---: | --- |
| SDUST-Bearing | 10 | 36 | `upper_x, upper_y, upper_z` | 1024 | `(368640, 3, 1024)` |
| SDUST-Gear | 7 | 19 | `left_z` | 512 | `(68096, 1, 1024)` |

这里“每类每域窗口数”是经过 JSON 中 `samples_per_class` 限制后的保存数量，并不等同于
保存全部原始窗口。

## 2. 两个数据集共同的数据制作含义

### 2.1 原始信号、采样率与取窗

两个适配器都从原始 MAT 文件的 `Signal.y_values.values` 读取振动信号：

- 原始采样率：25,600 Hz；
- 当前 `window_size_points`：1024 点；
- 一个窗口对应：`1024 / 25600 = 0.04 s`；
- 步长：1024 点，因此窗口互不重叠；
- 制作时不重采样、不插值、不归一化；
- 不能组成完整 1024 点窗口的末尾余数会被舍弃；
- `frequency` 变换和 `zscore` 归一化发生在训练/适配阶段，而不是数据制作阶段。

更改 `window_size_points` 后，输入张量的最后一维会随 manifest 自动变化，数据加载部分
不需要再手动填写长度；但所选 backbone 必须能接受新的长度，并且需要重新构建数据集、
重新训练 source checkpoint。

### 2.2 `samples_per_class` 与“保存全部数据”

Bearing 的配置是 `"samples_per_class": 1024`，Gear 的配置是
`"samples_per_class": 512`；两者都使用 `"selection": "uniform"`。

含义是：对每个“domain × class”的原始窗口池，按时间范围均匀选择固定数量的窗口。
因此：

- Bearing 的固定工况通常有 1025 个可用窗口，但当前只保存 1024 个；部分变转速组合因
  原始采集条数不同，有 1025 或 2050 个可用窗口，也只保存 1024 个；
- Gear 的固定工况有 525 个可用窗口，只保存 512 个；变负载工况有 1050 个、共同波动
  工况有 2050 个可用窗口，也都只保存 512 个。

把 `samples_per_class` 改为 `null` 才表示保存每个组合的全部非重叠窗口。此模式为了形成
类别对齐的 `interleaved` 流，要求同一 domain 内各类别的窗口数相同。SDUST-Bearing 的
部分变转速组合在不同类别中可能有不同的原始记录数，所以直接设为 `null` 可能触发等长
检查；当前固定数量的均匀采样方案兼顾了类别平衡和时间范围覆盖。

### 2.3 `target_stream_order` 与 YAML `stream_order`

JSON 中的 `"target_stream_order": "interleaved"` 只是写入 manifest 的默认顺序。
运行实验时，如果 YAML 显式设置了 `protocol.stream_order`，YAML 会覆盖 manifest。
当前两份 SDUST `base.yaml` 都设置为 `stream_order: shuffled`，所以实际运行会把每个
target domain 内的样本随机打乱。

| 设置 | 顺序 | 是否适合研究连续变工况 |
| --- | --- | --- |
| `shuffled` | domain 内所有类别与时间随机混合 | 否；工况的先后关系被破坏 |
| `class_block` | 先输入某一类的全部样本，再输入下一类 | 否；会制造不自然的标签块漂移 |
| `interleaved` | 第 0 个时间位置的各类别依次输入，再进入第 1 个位置 | 是；保留各类别内部的采集次序 |

这里的“时间对齐”是一种评测协议：不同故障类别来自不同的独立采集，不代表设备在同一
时刻同时具有多种故障。`interleaved` 只是利用相同工况程序，把各类别的对应位置组织成
平衡的在线流；target 标签仅用于离线评价，不会提供给适配算法。

## 3. SDUST-Bearing

### 3.1 通道含义

轴承原始数据有 6 个振动通道，原始列顺序为：

```text
upper_z, upper_y, upper_x, lower_z, lower_y, lower_x
```

当前 [sdust_bearing.json](../ctta_data_builder/configs/sdust_bearing.json) 选择：

```json
"channels": ["upper_x", "upper_y", "upper_z"]
```

即使用上测点的 x/y/z 三轴振动，输出形状中的通道数为 3。通道选择只做列筛选，不会
自动融合三轴信号。

### 3.2 类别标号

类别 label 由 JSON 中的顺序重新编号：

| Label | 类别名 | 含义 |
| ---: | --- | --- |
| 0 | `NC` | 正常状态 |
| 1 | `IF0.2` | 内圈故障，等级 0.2 |
| 2 | `OF0.2` | 外圈故障，等级 0.2 |
| 3 | `RF0.2` | 滚动体故障，等级 0.2 |
| 4 | `IF0.4` | 内圈故障，等级 0.4 |
| 5 | `OF0.4` | 外圈故障，等级 0.4 |
| 6 | `RF0.4` | 滚动体故障，等级 0.4 |
| 7 | `IF0.6` | 内圈故障，等级 0.6 |
| 8 | `OF0.6` | 外圈故障，等级 0.6 |
| 9 | `RF0.6` | 滚动体故障，等级 0.6 |

`IF/OF/RF` 分别表示 inner race、outer race 和 rolling element fault。文档把
`0.2/0.4/0.6` 视为原始数据命名中的故障等级，不额外假设其物理单位。

### 3.3 Domain 0–23：固定转速、固定负载

这 24 个域是 6 个固定转速与 4 个径向负载的笛卡尔积。在一个 domain 内，转速和负载
在采集期间保持为该域名所示的固定值。

| ID | Manifest 域名 | 转速 | 负载 | 工况类型 |
| ---: | --- | ---: | ---: | --- |
| 0 | `1000rpm_0N` | 1000 rpm | 0 N | 固定工况 |
| 1 | `1000rpm_20N` | 1000 rpm | 20 N | 固定工况 |
| 2 | `1000rpm_40N` | 1000 rpm | 40 N | 固定工况 |
| 3 | `1000rpm_60N` | 1000 rpm | 60 N | 固定工况 |
| 4 | `1500rpm_0N` | 1500 rpm | 0 N | 固定工况 |
| 5 | `1500rpm_20N` | 1500 rpm | 20 N | 固定工况 |
| 6 | `1500rpm_40N` | 1500 rpm | 40 N | 固定工况 |
| 7 | `1500rpm_60N` | 1500 rpm | 60 N | 固定工况 |
| 8 | `1800rpm_0N` | 1800 rpm | 0 N | 固定工况 |
| 9 | `1800rpm_20N` | 1800 rpm | 20 N | 固定工况 |
| 10 | `1800rpm_40N` | 1800 rpm | 40 N | 固定工况 |
| 11 | `1800rpm_60N` | 1800 rpm | 60 N | 固定工况 |
| 12 | `2000rpm_0N` | 2000 rpm | 0 N | 固定工况 |
| 13 | `2000rpm_20N` | 2000 rpm | 20 N | 固定工况 |
| 14 | `2000rpm_40N` | 2000 rpm | 40 N | 固定工况 |
| 15 | `2000rpm_60N` | 2000 rpm | 60 N | 固定工况 |
| 16 | `2500rpm_0N` | 2500 rpm | 0 N | 固定工况 |
| 17 | `2500rpm_20N` | 2500 rpm | 20 N | 固定工况 |
| 18 | `2500rpm_40N` | 2500 rpm | 40 N | 固定工况 |
| 19 | `2500rpm_60N` | 2500 rpm | 60 N | 固定工况 |
| 20 | `3000rpm_0N` | 3000 rpm | 0 N | 固定工况 |
| 21 | `3000rpm_20N` | 3000 rpm | 20 N | 固定工况 |
| 22 | `3000rpm_40N` | 3000 rpm | 40 N | 固定工况 |
| 23 | `3000rpm_60N` | 3000 rpm | 60 N | 固定工况 |

### 3.4 Domain 24–35：连续变转速、固定负载

这 12 个域中，负载保持固定，转速沿原始采集轨迹从区间起点变化到终点。域名只说明
起止范围，构建器不重新生成或线性插值转速曲线，而是保留被选窗口在原始记录中的顺序。

| ID | Manifest 域名 | 转速变化范围 | 固定负载 | 工况类型 |
| ---: | --- | --- | ---: | --- |
| 24 | `speed_800to1500rpm_0N` | 800→1500 rpm | 0 N | 连续变转速 |
| 25 | `speed_800to1500rpm_20N` | 800→1500 rpm | 20 N | 连续变转速 |
| 26 | `speed_800to1500rpm_40N` | 800→1500 rpm | 40 N | 连续变转速 |
| 27 | `speed_800to1500rpm_60N` | 800→1500 rpm | 60 N | 连续变转速 |
| 28 | `speed_1000to2000rpm_0N` | 1000→2000 rpm | 0 N | 连续变转速 |
| 29 | `speed_1000to2000rpm_20N` | 1000→2000 rpm | 20 N | 连续变转速 |
| 30 | `speed_1000to2000rpm_40N` | 1000→2000 rpm | 40 N | 连续变转速 |
| 31 | `speed_1000to2000rpm_60N` | 1000→2000 rpm | 60 N | 连续变转速 |
| 32 | `speed_1500to2500rpm_0N` | 1500→2500 rpm | 0 N | 连续变转速 |
| 33 | `speed_1500to2500rpm_20N` | 1500→2500 rpm | 20 N | 连续变转速 |
| 34 | `speed_1500to2500rpm_40N` | 1500→2500 rpm | 40 N | 连续变转速 |
| 35 | `speed_1500to2500rpm_60N` | 1500→2500 rpm | 60 N | 连续变转速 |

当前 [SDUST_Bearing/base.yaml](../configs/SDUST_Bearing/unifreq/base.yaml) 使用：

```yaml
source_domains: [0, 1, 2, 3]
domains: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
stream_order: shuffled
```

它表示用 1000 rpm 下的四种固定负载训练 source，再依次测试 1500–3000 rpm 的固定
工况。它没有使用 domain 24–35，因此当前默认协议是“跨固定工况泛化”，还不是连续变
转速 CTTA 实验。

## 4. SDUST-Gear

### 4.1 通道含义

齿轮箱原始数据有 6 个振动通道，原始列顺序为：

```text
left_z, left_y, left_x, right_z, right_y, right_x
```

当前 [sdust_gear.json](../ctta_data_builder/configs/sdust_gear.json) 选择
`"channels": ["left_z"]`，即只使用左侧测点 z 方向振动，输出通道数为 1。

### 4.2 类别标号

| Label | 类别名 | 含义 |
| ---: | --- | --- |
| 0 | `NC` | 正常状态 |
| 1 | `太阳断裂` | 太阳轮断裂 |
| 2 | `太阳点蚀` | 太阳轮点蚀 |
| 3 | `太阳磨损` | 太阳轮磨损 |
| 4 | `行星断裂` | 行星轮断裂 |
| 5 | `行星点蚀` | 行星轮点蚀 |
| 6 | `行星磨损` | 行星轮磨损 |

### 4.3 Domain 0–18 的具体含义

当前 JSON 只为所有 7 类原始文件都完整的 1500、1800、2000 rpm 建立固定 0.5 A 域。
1000 和 2500 rpm 的部分固定工况文件在某些故障类中缺失，因此没有进入当前自定义输出。

| ID | Manifest 域名 | 固定量 | 变化量 | 工况类型 |
| ---: | --- | --- | --- | --- |
| 0 | `1500rpm_0.5A` | 1500 rpm | 负载 0.5 A | 固定工况 |
| 1 | `1800rpm_0.5A` | 1800 rpm | 负载 0.5 A | 固定工况 |
| 2 | `2000rpm_0.5A` | 2000 rpm | 负载 0.5 A | 固定工况 |
| 3 | `1000rpm_load_0to0.2A` | 1000 rpm | 负载 0→0.2 A | 连续变负载 |
| 4 | `1000rpm_load_0to0.35A` | 1000 rpm | 负载 0→0.35 A | 连续变负载 |
| 5 | `1000rpm_load_0to0.5A` | 1000 rpm | 负载 0→0.5 A | 连续变负载 |
| 6 | `1500rpm_load_0to0.2A` | 1500 rpm | 负载 0→0.2 A | 连续变负载 |
| 7 | `1500rpm_load_0to0.35A` | 1500 rpm | 负载 0→0.35 A | 连续变负载 |
| 8 | `1500rpm_load_0to0.5A` | 1500 rpm | 负载 0→0.5 A | 连续变负载 |
| 9 | `1800rpm_load_0to0.2A` | 1800 rpm | 负载 0→0.2 A | 连续变负载 |
| 10 | `1800rpm_load_0to0.35A` | 1800 rpm | 负载 0→0.35 A | 连续变负载 |
| 11 | `1800rpm_load_0to0.5A` | 1800 rpm | 负载 0→0.5 A | 连续变负载 |
| 12 | `2000rpm_load_0to0.2A` | 2000 rpm | 负载 0→0.2 A | 连续变负载 |
| 13 | `2000rpm_load_0to0.35A` | 2000 rpm | 负载 0→0.35 A | 连续变负载 |
| 14 | `2000rpm_load_0to0.5A` | 2000 rpm | 负载 0→0.5 A | 连续变负载 |
| 15 | `2500rpm_load_0to0.2A` | 2500 rpm | 负载 0→0.2 A | 连续变负载 |
| 16 | `2500rpm_load_0to0.35A` | 2500 rpm | 负载 0→0.35 A | 连续变负载 |
| 17 | `2500rpm_load_0to0.5A` | 2500 rpm | 负载 0→0.5 A | 连续变负载 |
| 18 | `fluctuating_speed_and_load` | 无单一固定量 | 转速与负载共同波动 | 复合连续变工况 |

`0→x A` 和 domain 18 的变化过程完全由原始记录决定。仅凭文件名不能确定变化是否严格
线性、每一阶段持续多久或 domain 18 的精确转速/负载轨迹，因此实验中不应虚构这些
时间参数。

当前 [SDUST_Gear/base.yaml](../configs/SDUST_Gear/unifreq/base.yaml) 使用 domain 0 作为 source，
并把 domain 1–18 全部作为 target。域的输入顺序是列表顺序，但由于
`stream_order: shuffled`，每个域内部的窗口仍被随机打乱。

## 5. 建议实验设计

### 5.1 统一实验原则

所有方法应使用相同的数据、source checkpoint、target 域顺序和随机种子。target 流只
遍历一次，域切换时不重置适配状态，不使用 target 标签做适配、早停或超参数选择。建议
至少报告 3 个随机种子的均值和标准差。

若研究“持续变工况”，主实验应设置 `stream_order: interleaved`。为使一个 batch 对应
整数个对齐位置，可优先采用：

- Bearing：`stream_batch_size: 10`、20 或 40；类别数为 10；
- Gear：`stream_batch_size: 7`、14 或 28；类别数为 7。

当前的 batch size 64 仍能运行，但会跨越不完整的类别对齐位置，不便于把一次适配解释为
若干完整工况时刻。

### 5.2 Bearing 的三个互补协议

固定使用 `source_domains: [0, 1, 2, 3]`，即 1000 rpm 下的全部四种固定负载：

| 协议 | `domains` | 研究问题 |
| --- | --- | --- |
| B-Fixed | `[4, 5, ..., 23]` | 从 1000 rpm 泛化到 1500–3000 rpm 的固定工况；对应当前默认 target 集合 |
| B-Variable | `[24, 25, ..., 35]` | 在固定负载下经历连续升速，是主要的时变工况 CTTA 协议 |
| B-Mixed | `[4, 5, ..., 35]` | 先经历固定工况，再进入连续变转速工况，考察长期适配和遗忘 |

B-Variable 的建议 YAML 核心配置为：

```yaml
protocol:
  source_domains: [0, 1, 2, 3]
  domains: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
  stream_order: interleaved
  stream_batch_size: 10
```

补充实验可以固定负载做跨转速比较。例如以所有 0 N 固定域
`[0, 4, 8, 12, 16, 20]` 为 source，把其他固定负载域作为 target，用于区分“转速迁移”
和“负载迁移”带来的影响。每种 source 组合都应单独训练 checkpoint。

### 5.3 Gear 的四个互补协议

固定使用 `source_domains: [0]`，即 1500 rpm、0.5 A 固定工况：

| 协议 | `domains` | 研究问题 |
| --- | --- | --- |
| G-Fixed | `[1, 2]` | 固定负载下的跨转速泛化 |
| G-Load | `[3, 4, ..., 17]` | 不同固定转速、不同幅值的连续变负载适配 |
| G-Fluctuating | `[18]` | 转速和负载共同波动时的复合漂移 |
| G-Mixed | `[1, 2, ..., 18]` | 从固定工况进入变负载，最后进入共同波动域；对应完整长期 CTTA |

G-Mixed 的建议 YAML 核心配置为：

```yaml
protocol:
  source_domains: [0]
  domains: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  stream_order: interleaved
  stream_batch_size: 7
```

### 5.4 方法与指标

建议至少比较 Source、kNN、TENT、EATA、SAR、CoTTA、NOTE、RoTTA、TTT、
Continual-MAE、TCA、GOLD 与 GOLD-TCA。除需要额外 source 训练头的方法外，其余方法应复用相同的 source
checkpoint。

建议同时报告：

- 每个 target domain 的 accuracy 和 balanced accuracy；
- 所有 target 样本加权的总体准确率；
- 各 domain 等权的 domain-average accuracy；
- 按 `sequence_index` 绘制的滑动窗口准确率，用于观察连续工况中的性能变化；
- 每经过一个 target domain 后的 source-retention accuracy 与 source forgetting；
- 3 个或更多随机种子的均值与标准差。

为验证时间顺序的影响，可把 `interleaved` 与 `shuffled` 作为消融实验。`shuffled` 结果
只能解释为随机混合 target 上的适配能力，不应称为真实连续工况结果。

## 6. 构建与运行

从仓库根目录构建：

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/sdust_bearing.json

python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/sdust_gear.json
```

先检查原始目录与组合是否完整，可以使用：

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/sdust_bearing.json \
  --validate_only

python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/sdust_gear.json \
  --validate_only
```

构建后以生成的 manifest 为准。source 与 target 域由实验 YAML 决定；JSON 中 domain 的
排列只决定编号，不会自动指定哪个域必须是 source 或 target。同一个 domain ID 可以
同时出现在 `source_domains` 和 `domains` 中；此时 target 只使用未进入 source train
或 validation 的 sample ID。同一 raw recording group 中的其他域仍受防泄漏约束。

### GOLD

复用同模型的 source checkpoint 后运行：

```bash
python main.py --config configs/SDUST_Bearing/unifault/gold.yaml
```

`SDUST_Gear` 以及 `unifreq`、`tf_profm` 目录同样提供 `gold.yaml`。
GOLD 会从 source train 提取类别原型，目标流不读取标签；参数与移植差异见
[GOLD_ZH.md](GOLD_ZH.md)。

### GOLD-TCA

先训练该 foundation model 对应的 cosine source classifier，再让默认模式和 prototype
变体复用同一个 checkpoint：

```bash
python main.py --config configs/SDUST_Bearing/unifault/source_cosine.yaml
python main.py --config configs/SDUST_Bearing/unifault/gold_tca.yaml
python main.py --config configs/SDUST_Bearing/unifault/gold_tca_prototype.yaml
```

默认模式不计算 source prototypes，使用冻结 classifier weights 作为 semantic anchor、
source topology 和 AGOP 初始化；prototype 变体维护 target-updated EMA memory。详见
[GOLD_TCA_ZH.md](GOLD_TCA_ZH.md)。

## Source 各域表现与 TTA 相对提升

Source 训练结束、选定最佳 checkpoint 后，默认冻结评估所有域；源域只使用 held-out
validation，非源域使用完整域。该诊断不会把测试域加入训练，也不会根据测试域指标
选择 checkpoint，并且不受 `target_samples_per_class` 或 `max_target_per_domain` 限制。
`method: source` 在诊断后直接结束，不构建 adapter，也不进入 online target stream。

```yaml
protocol:
  source_eval_domains: all  # 默认；也可写 sources、targets 或 [4, 8, 12]
```

- `sources`：仅评估各源域的 held-out validation 样本。
- `all`：所有域，包括源域验证集。
- `targets`：当前 `domains` 中的目标域，去重后评估完整域。
- 域 ID 列表：只做指定域的 source 诊断；不改变在线目标序列。

Source 实验目录生成 `source_evaluation.csv` 和 `source_evaluation.json`，包含
`domain`、`domain_name`、`evaluation_split`、样本数、Accuracy、Balanced Accuracy、
Macro-F1 等。JSON 按域 ID 建索引，新训练的 checkpoint 目录也保存一份该 JSON。
只有 source 模式或本次确实训练了源模型才执行此诊断；加载 checkpoint 的 TTA
实验只额外计算本次实际目标样本的 source 基线。

例如需要完整域报告时显式评估所有域，再按选择的域顺序执行 GOLD：

```bash
python main.py --config configs/SDUST_Bearing/unifault/source.yaml --source_eval_domains all
python main.py --config configs/SDUST_Bearing/unifault/gold.yaml --domains 12,4,8
```

`source_domains` 仍代表训练域，`domains` 在显式设置 `source_domains` 时仅代表目标流。
若训练域再次出现在 target stream，必须为 source 选择保留足够的未见样本；仅做全域
诊断时使用 `source_eval_domains`，无需把训练域塞进 `domains`。
TTA 可删减、重排目标域，也可按现有重复域采样规则重复某域；不同出现次数分别记录。

### 精确匹配与输出

TTA 使用本次加载的未适应 source classifier，在每个目标 batch 的**同一批样本**上
重新计算基线。不会把旧 source 报告按行拼接，也不会拿完整域分数与抽样后的 TTA 分数
直接相减。即使旧 checkpoint 没有 `source_evaluation.json`，也可以计算提升。
因此会增加一次冻结 source 模型的目标前向计算。

- `source_baseline.csv/json`：本次实际选中的每个目标域出现次数的 source 表现，记录
  `domain`、`position`、样本数和样本 ID 摘要。
- `domains.csv`：新增 `source_accuracy`、`source_balanced_accuracy`、`source_macro_f1`，
  以及在线、更新后两组提升指标。
- `predictions.csv`：新增 `source_prediction`、`source_correct`、`source_confidence`；
  可用 `domain + position + sample_id` 核对对应样本。
- `summary.json`：`source_baseline` 为本次目标流基线，`online` / `post_adapt` 各自记录
  相对提升。整体分数使用本次目标流全部样本累计混淆矩阵计算；域平均则对本次每个
  域出现次数等权平均。未选中的域不进入任何 TTA 汇总。

### 指标定义

Accuracy、Balanced Accuracy、Macro-F1 均使用 0–100 分数尺度：

| 字段 | 公式 | 例：source 50%，TTA 60% |
| --- | --- | --- |
| `accuracy_gain_pp` | TTA Accuracy − Source Accuracy | +10 个百分点 |
| `accuracy_relative_gain_pct` | 100 × (TTA − Source) / Source | +20% |
| `error_reduction_pct` | 100 × (TTA Accuracy − Source Accuracy) / (100 − Source Accuracy) | +20% |

Balanced Accuracy 和 Macro-F1 也有对应的 `_gain_pp` 与 `_relative_gain_pct`。
`domains.csv` 中更新后指标带 `post_adapt_` 前缀；汇总 JSON 中分别放在 `online`
和 `post_adapt` 下。`domain_average_accuracy_gain_pp` 和
`domain_average_accuracy_relative_gain_pct` 比较相同域出现次数的等权均值。
下降时保留负值；基线为 0 时相对提升记为 JSON `null` / CSV 空白，基线准确率为
100 时错误率降低比例同样为空，避免除零或产生误导值。

仍需保证 source/TTA 的训练域、分类头、预处理和 source 划分一致。新 checkpoint
额外保存并校验 `source_validation_ratio`，避免源域归一化统计因划分变化而不同；
旧 checkpoint 缺少该字段时保留兼容，此时需自行保持原训练配置一致。
