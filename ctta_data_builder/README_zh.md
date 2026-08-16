# 可配置 CTTA 数据集构建器

本目录为 `PU`、`WT`、`MCC5-Motor` 和 `HUST-Gearbox` 单独提供原始数据到
FMCTTA `MAT` 文件的构建代码。读取通道与
`PHMFD-data-prepocessing/data_scripts` 保持一致，但协议专门面向 CTTA：配置中
域列表的第一项是有标签源域，后续项依次组成无标签目标域流。

## 固定的数据约束

所有样本都是原采样率下连续、互不重叠的 `0.1 s` 窗口。构建器没有重采样代码，
也不做归一化；需要归一化时仍由 FMCTTA 使用源域训练集统计量完成。

| 数据集 | 原采样率 | 每条样本点数 | 通道 |
|---|---:|---:|---|
| PU | 64,000 Hz | 6,400 | 1 |
| WT | 48,000 Hz | 4,800 | 2（x/y） |
| MCC5-Motor | 12,800 Hz | 1,280 | 3（电机 x/y/z） |
| HUST-Gearbox | 25,600 Hz | 2,560 | 3（最后三个振动通道） |

`MCC5-Motor` 的源域沿用原代码中的两个稳态平台 `15-25 s` 和 `65-75 s`。
目标域使用相应循环工况记录的完整时间顺序。已知原始文件缺类/重复的
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
- `domains`：所选域及 CTTA 顺序；首项重新编号为源域 `0`，其余为目标域
  `1..D-1`。
- `samples_per_class`：每个“域 × 类别”的样本数，所有组合保持平衡。
- `selection`：`uniform` 按时间均匀选取，`head` 取最早的窗口；两者都不会打乱
  采集顺序，也不会制造重叠窗口。
- `target_stream_order`：FMCTTA 读取目标域时使用 `interleaved`、
  `class_block` 或 `shuffled`。推荐 `interleaved`，它按时间位置轮流送入各类别，
  同时保持每类内部的采集顺序。
- `name`：FMCTTA 运行时显示的数据集名。
- `raw_root`、`output`：相对路径会相对于 JSON 文件所在目录解析。

例如只用 PU 的 4 类、按自定义域漂移顺序生成：

```json
{
  "dataset": "PU",
  "name": "PU_4class_N15_to_N09",
  "raw_root": "/home/xyc3090/data1/PHMFD-data-prepocessing/Raw_data/Finetune",
  "output": "../../data/ctta_custom/PU_4class_N15_to_N09.mat",
  "classes": ["K001", "KA04", "KB23", "KI14"],
  "domains": ["N15_M07_F10", "N15_M01_F10", "N09_M07_F10"],
  "samples_per_class": 100,
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
  --samples-per-class 80 \
  --output data/ctta_custom/PU_demo.mat
```

输出已存在时默认拒绝覆盖；确认替换时显式加 `--overwrite`。

## 输出内容

每次构建同时生成：

- `*.mat`：`data`、`label`、`domain`、原始类别/域编号、窗口起始时间、原始记录
  编号及名称等数组。存储顺序固定为“域 → 类别 → 采集时间”，符合
  `FaultDataModule` 的平衡域块要求。
- `*.json`：完整类别/域映射、采样约束、每个组合的可用/选中窗口数、CTTA
  序列和可直接加载的 `fmctta_spec`。

直接运行生成的自定义数据集：

```bash
python main.py \
  --dataset-manifest data/ctta_custom/PU_0p1s_custom.json \
  --method tent \
  --model-id bojian1/UniFreq-tiny \
  --run-name pu-custom-tent
```

`--dataset-manifest` 会自动读取类别数、通道数、域名及默认域顺序，因此无需再改
`fmctta/config.py`。如需对生成后的域编号再次重排，仍可传
`--domains 2,0,1,...`；该列表中的第一项始终作为源域。
