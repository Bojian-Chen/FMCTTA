# 可配置 CTTA 数据集构建器

本目录提供 SK、PU、WT、SDUST-Bearing、SDUST-Gear、MCC5-Gearbox、MCC5-Motor 和 HUST-Gearbox 的
原始数据读取与 FMCTTA MAT 构建
代码。数据制作阶段只分配类别和数值域标签；源域、目标域及漂移顺序由运行实验时的
`main.py --domains` 决定。

## 切窗约束

- 每条样本包含原始采样率下连续的 `window_size_points` 个点，默认 1024。
- `stride_points` 控制相邻窗口起点的间隔；省略时等于窗口点数，默认不重叠。
- 步长小于窗口长度时产生重叠窗口，大于窗口长度时窗口间留有间隔。
- 不重采样、不插值、不归一化。
- `window_seconds` 和 `stride_seconds` 分别按窗口点数、步长除以采样率写入元数据。
- MCC5-Motor 使用原生 90 秒轨迹：稳态 1 为 10–30 秒和 60–80 秒，稳态 2
  为 35–55 秒。
- MCC5-Gearbox 保持原生 60 秒轨迹：稳态 1 为 10–20 秒和 40–50 秒，稳态 2
  为 25–35 秒；不会补造 60–90 秒数据。
- 每条有效 MCC5 circulation 记录都制作成连续三个域：稳态 1 候选、稳态 2 候选、
  完整时序 target。Gearbox 因此有 24 个 source 候选和 12 个 target 流；Motor
  有 22 个 source 候选和 11 个 target 流。

| 数据集 | 原采样率 | 默认 1024 点时长 | 通道 |
|---|---:|---:|---|
| SK | 25,600 Hz | 0.04 s | 3 |
| PU | 64,000 Hz | 0.016 s | 1 |
| WT | 48,000 Hz | 约 0.02133 s | 2（x/y） |
| SDUST-Bearing | 25,600 Hz | 0.04 s | 6 |
| SDUST-Gear | 25,600 Hz | 0.04 s | 6 |
| MCC5-Gearbox | 12,800 Hz | 0.08 s | 6（电机 x/y/z、齿轮箱 x/y/z） |
| MCC5-Motor | 12,800 Hz | 0.08 s | 3（电机 x/y/z） |
| HUST-Gearbox | 25,600 Hz | 0.04 s | 3（最后三个振动通道） |

UniFreq 的处理器支持不同长度的一维信号，因此采样点数不要求为完全平方数；切换点数后
需使用相同数据配置重新训练源域分类器。

## 查看目录与验证配置

```bash
python -m ctta_data_builder.build --catalog SK
python -m ctta_data_builder.build --catalog PU
python -m ctta_data_builder.build --catalog WT --window_size_points 2048
python -m ctta_data_builder.build --catalog SDUST-Bearing
python -m ctta_data_builder.build --catalog SDUST-Gear
python -m ctta_data_builder.build --catalog MCC5-Gearbox
python -m ctta_data_builder.build --catalog MCC5-Motor

python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/pu.json \
  --validate_only
```

`validate_only` 只检查配置和原始路径，不读取大型信号；每个域×类别组合实际可用的窗口
数会在正式构建时检查。

## JSON 配置

- `dataset`：`SK`、`PU`、`WT`、`SDUST-Bearing`、`SDUST-Gear`、`MCC5-Gearbox`、`MCC5-Motor` 或
  `HUST-Gearbox`。
- `name`：动态 FMCTTA 数据集名称。
- `raw_root`、`output`：原始数据根目录与输出 MAT；相对路径相对于 JSON 文件解析。
- `classes`：类别及顺序，输出标签重新编号为 `0..K-1`。
- `channels`：按 `--catalog` 显示的名字选择输出通道及顺序；省略时保留全部通道。
- `domains`：连续数值标签到原始工况的映射，输出域编号为 `0..D-1`。
- `samples_per_class`：每个域、每个类别选取的相同样本数，至少为 2；设为
  `null` 时保存每个域中按配置步长得到的全部完整窗口。完整模式允许不同域具有不同长度，
  但同一域内各类别必须等长，才能保持工况时间对齐。
- `window_size_points`：每条样本的采样点数，正整数，默认 1024。
- `stride_points`：窗口起点间隔的采样点数，正整数；省略或设为 `null` 时等于窗口长度。
- `selection`：`head` 取最早窗口，`uniform` 在可用时间序列中均匀选择。
- `target_stream_order`：运行时的默认 `class_block`、`interleaved`、`shuffled` 或 `dirichlet`。
  Dirichlet 的参数在运行 YAML 的 `protocol.dirichlet_beta` 和
  `protocol.dirichlet_num_chunks` 中设置，无需重新制作 MAT。

CLI 可临时覆盖常用字段：

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/pu.json \
  --classes K001,KA04,KB23,KI14 \
  --channels vibration \
  --domains N15_M07_F10,N15_M01_F10,N09_M07_F10 \
  --samples_per_class 500 \
  --window_size_points 1024 \
  --output data/ctta_custom/PU_demo.mat
```

也可以用 `--all_windows` 临时覆盖 JSON 中的固定样本数。完整数据通常超过 MATLAB v5
单矩阵 2 GiB 限制，构建器会边读取边写入 HDF5-backed `.mat`，不会在内存中复制整个
输出。

输出已存在时默认拒绝覆盖；确认替换时添加 `--overwrite`。

## 从当前 OCTTA 同步的配置

`configs/octta/` 保存当前 OCTTA 的八套正式模板，保留其类别、通道、域顺序、样本数和
切窗设置；相对路径已适配 FMCTTA 目录。原有 `configs/*.json` 仍保留原来的配置。
新构建器输出 `fmctta_spec`，生成的 manifest 可直接用于 FMCTTA 的 `--dataset_manifest`。

例如，先检查配置与原始文件，再构建 SDUST-Bearing：

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/octta/sdust_bearing.json --validate_only
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/octta/sdust_bearing.json
```

OCTTA 模板中的步长是显式值。覆盖窗口长度时，如需非重叠窗口，也需将
`--stride_points` 设为相同长度。`--raw_root` 和 `--output` 可覆盖模板中的路径。

SK（山科）支持 10 类、24 个固定转速/载荷域，按转速优先排列，domain 6 对应
`1000rpm_40N`。读取 `Signal.y_values.values` 的三个通道：`point2_z`、`point3_y`、
`another_y`；精确匹配文件名 `<speed> <load>.mat`，排除 `flu` 和额外重复记录。
模板 `configs/octta/sk.json` 默认选择 `point2_z`，使用 1024 点窗口、512 点步长，
保存全部完整窗口。

HUST-Gearbox 增加 domain 25–29，对应 `0~40~0Hz_L0` 至 `0~40~0Hz_L4`，
读取 `<class>_L<load>_VS_0_40_0.txt` 的最后三个振动通道。
`configs/octta/hust_gearbox.json` 包含全部 30 个域，默认 1024 点窗口、512 点步长。
原有 `configs/hust_gearbox.json` 保留 25 个固定转速域和 0.1 秒切窗设置。

## 已迁移的 OCTTA 1024 点数据

`data/ctta_custom/` 已包含从 OCTTA 迁移并转换为 `fmctta_spec` 的六套成品数据：

| 数据集 | 成品名称 | 单条时长 |
| --- | --- | ---: |
| PU | `PU_1024point_custom` | 0.016 s |
| WT | `WT_1024point_custom` | 约 0.0213 s |
| SDUST-Bearing | `SDUST_Bearing_1024point_custom` | 0.04 s |
| SDUST-Gear | `SDUST_Gear_1024point_custom` | 0.04 s |
| MCC5-Gearbox | `MCC5_Gearbox_1024point_custom` | 0.08 s |
| MCC5-Motor | `MCC5_Motor_1024point_custom` | 0.08 s |

这些名称中的 `1024point` 表示每条样本固定为 1024 个原始采样点；由于各数据集采样率
不同，它不等于统一的秒数。PU、WT、SDUST、MCC5 的实验基础配置均可直接加载对应
manifest。已有的 `*_0p1s_custom` 文件继续保留，与 1024 点版本并存。

## 各数据集构建 0.1 秒版本

不同数据集必须按各自原始采样率换算点数。下面的样本数是所有所选域、所有类别都可用的
最大均衡数量；每条命令均采用原始采样率、不重叠切窗和 `uniform` 选择：

| 数据集 | 采样率 | 0.1 秒点数 | 每域每类样本数 |
| --- | ---: | ---: | ---: |
| PU | 64,000 Hz | 6,400 | 780 |
| WT | 48,000 Hz | 4,800 | 200 |
| SDUST-Bearing | 25,600 Hz | 2,560 | 410 |
| SDUST-Gear | 25,600 Hz | 2,560 | 210 |
| MCC5-Gearbox（36 域） | 12,800 Hz | 1,280 | 100 |
| MCC5-Motor（33 域） | 12,800 Hz | 1,280 | 200 |
| HUST-Gearbox | 25,600 Hz | 2,560 | 102 |

```bash
# PU
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/pu_1024.json \
  --window_size_points 6400 --samples_per_class 780 \
  --name PU_0p1s_custom \
  --output data/ctta_custom/PU_0p1s_custom.mat

# WT
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/wt_1024.json \
  --window_size_points 4800 --samples_per_class 200 \
  --name WT_0p1s_custom \
  --output data/ctta_custom/WT_0p1s_custom.mat

# SDUST-Bearing
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/sdust_bearing.json \
  --window_size_points 2560 --samples_per_class 410 \
  --name SDUST_Bearing_0p1s_custom \
  --output data/ctta_custom/SDUST_Bearing_0p1s_custom.mat

# SDUST-Gear
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/sdust_gear.json \
  --window_size_points 2560 --samples_per_class 210 \
  --name SDUST_Gear_0p1s_custom \
  --output data/ctta_custom/SDUST_Gear_0p1s_custom.mat

# MCC5-Gearbox：使用当前完整的 36 域协议
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/mcc5_gearbox.json \
  --window_size_points 1280 --samples_per_class 100 \
  --name MCC5_Gearbox_36domains_0p1s_custom \
  --output data/ctta_custom/MCC5_Gearbox_36domains_0p1s_custom.mat

# MCC5-Motor：使用当前完整的 33 域协议；名称避免与旧 11 域文件混淆
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/mcc5_motor.json \
  --window_size_points 1280 --samples_per_class 200 \
  --name MCC5_Motor_33domains_0p1s_custom \
  --output data/ctta_custom/MCC5_Motor_33domains_0p1s_custom.mat

# HUST-Gearbox 的 JSON 已经是 0.1 秒配置
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/hust_gearbox.json
```

若同名输出已存在，构建器会安全退出；确认需要替换时才在对应命令末尾添加
`--overwrite`。MCC5-Motor 的旧 11 域 0.1 秒协议仍可用
`ctta_data_builder/configs/mcc5_motor_0p1s_legacy.json` 单独构建。

## SDUST 类别与工况

两个 SDUST 适配器都从原始 MAT 的 `Signal.y_values.values` 读取 6 个通道，并根据 MAT
内的 `function_record.name` 命名。Bearing 为 `upper_z/upper_y/upper_x` 和
`lower_z/lower_y/lower_x`；Gear 为 `left_z/left_y/left_x` 和
`right_z/right_y/right_x`。当前 JSON 分别输出上测点三轴
`upper_x/upper_y/upper_z` 和单通道 `left_z`；可将
`channels` 改为例如 `["upper_x", "lower_x"]`，也可临时传
`--channels upper_x,lower_x`。通道选择只改变输出，不做通道融合。

- `SDUST-Bearing`：10 类；包含 1000、1500、1800、2000、2500、3000 rpm 六个
  固定转速与 0、20、40、60 N 四个负载的完整组合，共 24 个固定域；另有
  `800~1500`、`1000~2000`、`1500~2500` rpm 三个变转速区间与四个负载的组合，
  共 12 个时变域。合计 36 域。原始文件对区间分隔符同时使用 `~` 和 `-`，读取器会
  统一识别。
- `SDUST-Gear`：7 类，适配器支持负载 0.5 A、采集序号 1 下的 1000、1500、1800、
  2000、2500 rpm。当前原始目录中部分故障缺少 1000/2500 rpm 的 0.5 A 文件，因此
  默认配置选取所有 7 类都完整的 1500、1800、2000 rpm 三个固定域；此外直接读取
  5 个转速下 `0→0.2A`、`0→0.35A`、`0→0.5A` 的 15 个变负载域，以及 `flu flu`
  转速/负载共同波动域，共 19 域。时变域的两次原始采集都会保留到可选窗口池中。

完整的类别、通道、Domain ID、时变工况与建议实验协议见
[SDUST online CTTA 数据协议](../docs/SDUST_ZH.md)。

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/sdust_bearing.json \
  --validate_only

python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/sdust_bearing.json

python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/sdust_gear.json \
  --channels left_x,right_x
```

## MCC5 可选稳态源域协议

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/mcc5_gearbox.json
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/mcc5_motor.json
```

域按每条原始 circulation 记录三元排列：`3k` 是稳态 1 候选，`3k+1` 是稳态 2
候选，`3k+2` 是该记录的完整时序 target。domain 0/1 只是默认示例，不是固定源域。
Gearbox 可从 24 个候选域中选择，Motor 可从 22 个候选域中选择，也支持多个候选联合
监督训练。

为防止原始记录泄漏，若选了 `3k` 或 `3k+1` 作为 source，就不能再把 `3k+2` 放入
target。manifest 会写入 `source_candidate_domains`、`target_stream_domains`、
`recording_group` 和推荐默认组合；`FaultDataModule` 也会在运行前检查同组冲突。
完整 ID 表见 [Gearbox 协议](../docs/MCC5_THU_GEARBOX_ZH.md) 与
[Motor 协议](../docs/MCC5_THU_MOTOR_ZH.md)。

## 输出

每次构建同时生成：

- `*.mat`：`data`、`label`、`domain`、原始类别/域编号、窗口起始秒、序列索引、记录
  编号、点数/秒数/采样率等元数据。顺序固定为“域 → 类别 → 采集时间”。
- `*.json`：完整映射、采样约束、每个组合的可用/选中窗口数，以及可被 FMCTTA 动态
  加载的 `fmctta_spec`。

当任意数组超过 MATLAB v5 的 2 GiB 单矩阵上限时，构建器会自动将 `.mat` 写成 HDF5
容器，并在 manifest 中标注 `storage_format: hdf5`。FMCTTA 会自动读取这种大文件；它需要
`h5py`（已列入 `requirements.txt`）。

运行生成的数据：

```bash
python main.py \
  --dataset_manifest data/ctta_custom/PU_1024point_custom.json \
  --domains 0,1,2,3 \
  --method tent \
  --feature_branch frequency
```

manifest 只描述有哪些域。未设置 `source_domains` 时，`domains` 的第一个域是有标签源域，
后续位置为无标签目标流；设置 `source_domains` 后，它独立指定一个或多个有标签源域，
此时 `domains` 只表示目标流及其在线顺序。
