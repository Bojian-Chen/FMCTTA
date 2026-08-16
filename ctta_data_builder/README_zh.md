# 可配置 CTTA 数据集构建器

本目录为 `PU`、`WT`、`MCC5-Motor` 和 `HUST-Gearbox` 单独提供原始数据到
FMCTTA `MAT` 文件的构建代码。读取通道与
`PHMFD-data-prepocessing/data_scripts` 保持一致。数据制作阶段只给每个工况分配
`0, 1, 2, ...` 域标签，不区分源域、目标域，也不决定 CTTA 漂移顺序；运行实验时
再通过 `main.py --domains` 指定顺序，该参数的第一个编号才是有标签源域。

## 固定的数据约束

所有样本都是原采样率下连续、互不重叠的 `0.1 s` 窗口。构建器没有重采样代码，
也不做归一化；需要归一化时仍由 FMCTTA 使用源域训练集统计量完成。

| 数据集 | 原采样率 | 每条样本点数 | 通道 |
|---|---:|---:|---|
| PU | 64,000 Hz | 6,400 | 1 |
| WT | 48,000 Hz | 4,800 | 2（x/y） |
| MCC5-Motor | 12,800 Hz | 1,280 | 3（电机 x/y/z） |
| HUST-Gearbox | 25,600 Hz | 2,560 | 3（最后三个振动通道） |

`MCC5-Motor` 的 `steady_1000rpm_20Nm` 域沿用原代码中的两个稳态平台
`15-25 s` 和 `65-75 s`，其他循环工况域使用相应记录的完整时间顺序。已知
原始文件缺类/重复的
`speed_3000rpm_20Nm` 不进入安全目录，避免猜测标签。

## 查看可用类别与域

在 FMCTTA 根目录执行：

```bash
python -m ctta_data_builder.build --catalog PU
python -m ctta_data_builder.build --catalog WT
python -m ctta_data_builder.build --catalog MCC5-Motor
python -m ctta_data_builder.build --catalog HUST-Gearbox
```

## 定制配置

复制 `configs/` 中对应的 JSON，然后修改：

- `classes`：所选类别及其顺序；生成标签按列表顺序重新编号为 `0..K-1`。
- `domains`：显式的“数值域标签 → 原始工况”映射。标签必须连续使用
  `0..D-1`，但这些编号本身不表示源域或目标域。
- `samples_per_class`：每个“域 × 类别”的样本数，可设置为任意不超过所选组合
  可用窗口数的整数。为匹配 FMCTTA 当前的平衡域块读取方式，所有组合使用同一个
  数值，不能给不同域或类别分别设置不同数量。
- `selection`：`uniform` 按时间均匀选取，`head` 取最早的窗口；两者都不会打乱
  采集顺序，也不会制造重叠窗口。
- `target_stream_order`：FMCTTA 读取目标域时使用 `interleaved`、
  `class_block` 或 `shuffled`。推荐 `interleaved`，它按时间位置轮流送入各类别，
  同时保持每类内部的采集顺序。
- `name`：FMCTTA 运行时显示的数据集名。
- `raw_root`、`output`：相对路径会相对于 JSON 文件所在目录解析。

例如只用 PU 的 4 类，并给三个工况分配域标签：

```json
{
  "dataset": "PU",
  "name": "PU_4class_N15_to_N09",
  "raw_root": "/home/xyc3090/data1/PHMFD-data-prepocessing/Raw_data/Finetune",
  "output": "../../data/ctta_custom/PU_4class_N15_to_N09.mat",
  "classes": ["K001", "KA04", "KB23", "KI14"],
  "domains": {
    "0": "N15_M07_F10",
    "1": "N15_M01_F10",
    "2": "N09_M07_F10"
  },
  "samples_per_class": 200,
  "selection": "uniform",
  "target_stream_order": "interleaved"
}
```

先做快速路径检查（不读取大型信号文件）：

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/pu.json \
  --validate-only
```

正式构建：

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/pu.json
```

CLI 可以临时覆盖常用字段，无需修改 JSON：

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/pu.json \
  --classes K001,KA04,KB23,KI14 \
  --domains N15_M07_F10,N15_M01_F10,N09_M07_F10 \
  --samples-per-class 500 \
  --output data/ctta_custom/PU_demo.mat
```

这里 CLI 的 `--domains A,B,C` 只是 `0=A, 1=B, 2=C` 的简写，不代表运行时
顺序。建议正式实验配置直接使用 JSON 对象，使标签映射一目了然。

### 样本数量上限

在当前原始数据和固定的 `0.1 s` 非重叠窗口协议下，默认全域配置的平衡上限为：

| 数据集 | 每个域、每个类最多可取 | 示例配置默认值 |
|---|---:|---:|
| PU | 780 | 200 |
| WT | 200 | 200 |
| MCC5-Motor | 200 | 200 |
| HUST-Gearbox | 102 | 102 |

MCC5-Motor 的 200 上限来自 `steady_1000rpm_20Nm` 域；如果配置中不选择该域，
其他完整循环记录通常可提供 900 个窗口。构建时会逐组合检查真实可用数量，超过
任一所选组合上限会直接报错，不会使用重叠窗口补足。

例如 PU 可以扩大为每域每类 500 条：

```bash
python -m ctta_data_builder.build \
  --config ctta_data_builder/configs/pu.json \
  --samples-per-class 500 \
  --output data/ctta_custom/PU_500.mat
```

实际数量会同时写入 MAT 的 `samples_per_class` 和 JSON 的 `fmctta_spec`。FMCTTA
通过 `--dataset-manifest` 读取该值，因此不需要同步手改 `fmctta/config.py`。

输出已存在时默认拒绝覆盖；确认替换时显式加 `--overwrite`。

## 输出内容

每次构建同时生成：

- `*.mat`：`data`、`label`、`domain`、原始类别/域编号、窗口起始时间、原始记录
  编号及名称等数组。存储顺序固定为“域 → 类别 → 采集时间”，符合
  `FaultDataModule` 的平衡域块要求。
- `*.json`：完整类别/域标签映射、采样约束、每个组合的可用/选中窗口数和可直接
  加载的 `fmctta_spec`；其中不预设源域和目标域角色。

直接运行生成的自定义数据集：

```bash
python main.py \
  --dataset-manifest data/ctta_custom/PU_0p1s_custom.json \
  --domains 3,2,0,1 \
  --method tent \
  --model-id bojian1/UniFreq-tiny \
  --run-name pu-custom-tent
```

`--dataset-manifest` 会自动读取类别数、通道数和域标签映射，因此无需再改
`fmctta/config.py`。实验的源域及目标漂移序列只由 `--domains` 决定：例如
`--domains 3,2,0,1` 表示域 `3` 是有标签源域，目标流依次经过域 `2、0、1`。
