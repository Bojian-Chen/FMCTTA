# FMCTTA

> 面向预训练 UniFreq、UniFault 与 TF-ProFM 故障诊断模型的
> 持续测试时自适应（Continual Test-Time Adaptation, CTTA）框架。

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Hugging%20Face-UniFreq-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/bojian1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md) | 简体中文

FMCTTA 首先用一个有标签源域为预训练故障基础模型训练下游分类器，随后在
无标签目标域流上进行有状态的在线自适应。模型状态会跨 batch、跨域延续，不在
目标域之间重置。

## 主要特性

- 可从 Hugging Face 加载 UniFreq，也可直接读取 PHMFD `log` 中的 UniFault、
  TF-ProFM 预训练 checkpoint；无需重新预训练。
- 支持 Source-only、源域 kNN、TENT、NOTE、EATA、SAR、CoTTA、ViDA、Continual-MAE、RoTTA、TCA、GOLD 和经典 TTT。
- 目标标签不会传入任何适配器，只在预测完成后用于计算评估指标。
- 严格采用单遍目标流，并记录跨域源域遗忘。
- 源域模型使用 `safetensors` 保存，同时输出逐域、逐样本和在线/适配后指标。
- 构建器覆盖 PU、WT、HUST、MCC5 Gearbox/Motor 与 SDUST Bearing/Gear，并支持 1024 点默认切窗、任意点数和 HDF5 大文件。

## 模型

| `foundation_model` | 权重来源 | 输入约定 |
|---|---|---|
| `unifreq` | `--model_id` 指定的 Hugging Face / 本地目录 | 使用模型自带 UniFreqProcessor，支持时域、频域或融合分支 |
| `unifault` | `--pretrained_checkpoint` 指定的 PHMFD `.pt` | 默认重采样至 1024 点并逐样本 min-max |
| `tf_profm` | `--pretrained_checkpoint` 指定的 PHMFD `.pt` | 默认短序列插值、长序列截断至 2048 点，再逐样本 z-score |

`--pretrained_checkpoint auto` 会直接选择
`/data1/PHM-Vibench/PHMFD/log` 中 seed 42 的已训练参数：UniFault 支持
`--foundation_model_size tiny|small`，TF-ProFM 使用其已训练的 2048 点配置。
也可以传入任意 `best_pretrain.pt` 文件或包含 `pretrain/best_pretrain.pt` 的运行目录。
加载时要求骨干张量 100% 覆盖，并输出实际加载张量数和参数数；预训练用 projection/head
不会迁入，当前数据集的分类头仍在 source 阶段训练。

仓库提供可直接运行或复制其 `model` 段的模型配置：
`configs/foundation_models/unifault_tiny.yaml`、`unifault_small.yaml` 和
`tf_profm.yaml`。

### UniFreq 型号

| CLI 参数 | Hugging Face 模型 | 建议用途 |
|---|---|---|
| `bojian1/UniFreq-tiny` | [UniFreq-tiny](https://huggingface.co/bojian1/UniFreq-tiny) | 快速实验与流程验证 |
| `bojian1/UniFreq-small` | [UniFreq-small](https://huggingface.co/bojian1/UniFreq-small) | 精度与计算量平衡 |
| `bojian1/UniFreq-base` | [UniFreq-base](https://huggingface.co/bojian1/UniFreq-base) | 最大容量编码器 |

`fmctta/unifreq/` 中保留了与公开权重一致的纯推理架构；迁移的 PHMFD 架构位于
`fmctta/phmfd_models.py`。大体积权重不会复制进本仓库，而是从原 `log` 路径读取。

## CTTA 方法

| 方法 | 在线目标 | 默认更新参数 |
|---|---|---|
| `source` | 不进行目标域更新 | 无 |
| `knn` | 固定源域训练集特征库上的余弦 kNN | 无 |
| `tent` | 熵最小化 | LayerNorm 仿射参数 |
| `note` | 伪标签平衡回放（PBRS）上的熵最小化 | LayerNorm 仿射参数 |
| `eata` | 可靠性/多样性筛选 + Fisher 约束 | LayerNorm 仿射参数 |
| `sar` | 可靠熵最小化 + SAM | LayerNorm 仿射参数 |
| `cotta` | EMA 教师一致性 + 随机恢复 | 全部参数 |
| `vida` | EMA 一致性 + 高/低秩 ViDA + HKA | ViDA 分支 + 全部基础参数 |
| `continual_mae` | 频域点分布感知掩码 + 重建 + 教师一致性 | 全部基础参数 + MAE 辅助参数 |
| `rotta` | CSTU 时效/不确定性内存 + EMA 一致性 | LayerNorm 仿射参数 |
| `tca` | 持续类别中心、拓扑一致性和双视图对齐 | 全部参数 |
| `gold` | AGOP 低秩子空间、自训练与源原型对比损失 | 全部参数 + 缩放向量和投影头 |
| `ttt` | 四路循环移位预测；先适应再做主任务预测 | 基础模型主干 + 辅助头 |

除依赖 UniFreq global-FFT token 接口的 `continual_mae` 外，其余方法均可使用三种
基础模型。UniFault 与 TF-ProFM 也包含 LayerNorm，因此 TENT、NOTE、EATA、SAR、
RoTTA 的默认 LayerNorm 适配范围保持有效。

[NOTE](https://github.com/TaesikGong/NOTE) 保留原方法的伪标签类别平衡 reservoir
和周期性内存熵最小化。原图像实现可选的 instance-aware BatchNorm 不适用于采用
Transformer/LayerNorm 的 UniFreq；这里通过统一的参数范围配置更新 LayerNorm 仿射参数。

[ViDA](https://github.com/Yangsenqiao/vida) 会在基础模型主干的线性层中
注入并联的高秩、低秩分支。HKA 根据增强后的教师预测不确定性，在域偏移较大时
提高高秩分支权重，否则提高低秩分支权重。ViDA 分支在加载源域 checkpoint 后
进行零输出初始化，因此已有 FMCTTA 源域 checkpoint 可以直接复用。

UniFreq 使用 Transformer/LayerNorm，因此不加入仅适用于 BatchNorm 的 AdaBN。RoTTA
保留 CSTU、EMA teacher 与时效加权一致性目标，但明确采用 LayerNorm 仿射更新，不伪造
Robust BN。TTT 会在 source 阶段联合训练四路循环移位辅助头，并把辅助头单独保存为
`safetensors`；普通 source checkpoint 不能冒充 TTT checkpoint。

## 安装

```bash
cd /data1/PHMT/FMCTTA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

公开模型无需 Hugging Face token。数据默认读取 `data/ctta_custom`；原始数据预处理说明见
[PHMFD-data-preprocessing](https://github.com/Bojian-Chen/PHMFD-data-preprocessing)。

需要从原始数据自定义 PU、WT、SDUST-Bearing/Gear、MCC5-Gearbox/Motor 或 HUST-Gearbox 时，使用
[ctta_data_builder/README_zh.md](ctta_data_builder/README_zh.md)。
该构建器默认生成原采样率下的 1024 点非重叠窗口，也可用 `window_size_points` 覆盖；不做重采样，并输出可通过
`--dataset_manifest` 直接加载的动态数据集描述。

## 快速开始

先训练源域分类器并运行 Source-only：

```bash
python main.py \
  --dataset PU \
  --method source \
  --model_id bojian1/UniFreq-tiny \
  --domains 3,2,0,1 \
  --run_name pu-source
```

使用 PHMFD 的 UniFault-small：

```bash
python main.py \
  --dataset PU --method source --domains 3,2,0,1 \
  --foundation_model unifault \
  --foundation_model_size small \
  --pretrained_checkpoint auto \
  --normalization none \
  --run_name pu-unifault-small-source
```

使用 TF-ProFM：

```bash
python main.py \
  --dataset PU --method source --domains 3,2,0,1 \
  --foundation_model tf_profm \
  --pretrained_checkpoint auto \
  --normalization none \
  --run_name pu-tf-profm-source
```

这里的 `normalization: none` 表示不再施加 FMCTTA 的源域统计量归一化；模型包装器仍会
执行与 PHMFD 预训练一致的逐样本归一化。训练出的 source checkpoint 之后可照常用于
TENT、NOTE、EATA、SAR、CoTTA、ViDA、RoTTA、TCA 或 TTT，但加载时必须保持同一
`foundation_model` 和模型结构。

源域 checkpoint 会保存在：

```text
outputs/PU/source/pu-source-seed2025/source_checkpoint/
```

### 分支选择

`--feature_branch fused` 为默认值，保持原有的时频融合。做消融实验时可指定
`--feature_branch time` 只运行时域分支，或指定 `--feature_branch frequency`
只运行频域分支。单分支的特征维度是 `embed_dim`，与融合后的拼接特征不同，因此每种
选择都需要重新训练源域分类器并使用对应 checkpoint；checkpoint 会记录该选项并拒绝
加载分支不一致的模型。

例如，仅用频域分支训练源域模型：

```bash
python main.py \
  --dataset PU --domains 3,2,0,1 --method source \
  --feature_branch frequency --run_name pu-frequency-source
```

### 每类采样数与重复目标域

`--source_samples_per_class` 可填一个数（所有源域类别相同），或以逗号分隔的每类
样本数。`--target_samples_per_class` 的每一项也可填一个数或每类样本数；单项会应用于
所有目标流位置，多个目标流位置则用分号分隔。目标域允许重复，同一原始样本会按出现
位置无放回分配，因此同一目标域同一类别的累计请求数不能超过可用样本数。
需要让每次重复都从该域的完整样本池重新抽取、允许不同位置重用样本时，添加
`--allow_repeated_target_samples`。
加载源域 checkpoint 时，须传入训练该 checkpoint 时相同的源域采样配置；该配置会被
记录在 checkpoint 元数据中。

```bash
python main.py \
  --dataset PU --method tent \
  --domains 0,1,2,3,1,3,2,3 \
  --source_samples_per_class 60 \
  --target_samples_per_class 20 \
  --run_name repeated-targets
```

上例有 7 个目标流位置，每个位置的每类各取 20 个样本；因此域 1、域 2 各使用每类
40 个互不重复的样本，域 3 使用每类 60 个。需要按目标流位置设置不同数目时，例如可用
`--target_samples_per_class '20;10;15;20;10;15;20'`。

完整实验 YAML 位于 `configs/PU`、`configs/WT`、`configs/HUST_Gearbox`、
`configs/MCC5_gearbox`、`configs/MCC5_motor`、`configs/SDUST_Bearing` 和
`configs/SDUST_Gear`。每个数据集目录都分为 `unifreq/`、`unifault/` 和 `tf_profm/`：
三者分别提供 UniFreq、UniFault-small 与 TF-ProFM 配置。UniFault-small 和 TF-ProFM
包含 source、kNN、TENT、NOTE、EATA、SAR、CoTTA、ViDA、RoTTA、TCA、GOLD 和 TTT；
Continual-MAE 只位于 `unifreq/`。
详细的目录、运行顺序和参数说明见 [`configs/README_zh.md`](configs/README_zh.md)。
例如，先训练 PU 的源模型，再运行 TENT：

```bash
python main.py --config configs/PU/unifreq/source.yaml
python main.py --config configs/PU/unifreq/tent.yaml

python main.py --config configs/PU/unifault/source.yaml
python main.py --config configs/PU/unifault/tent.yaml

python main.py --config configs/PU/tf_profm/source.yaml
python main.py --config configs/PU/tf_profm/tent.yaml
```

命令行参数会覆盖 YAML 中的同名配置。

然后复用该 checkpoint 运行 TENT：

```bash
python main.py --config configs/PU/unifreq/tent.yaml
```

把 `tent` 替换为 `note`、`eata`、`sar`、`cotta`、`vida`、`continual_mae`、`rotta`
或 `tca` 即可复用普通 source checkpoint。TTT 必须运行独立的 `ttt.yaml` 联合训练辅助头。
ViDA 示例：

```bash
python main.py \
  --dataset PU --domains 3,2,0,1 --method vida \
  --source_checkpoint outputs/PU/source/pu-source-seed2025/source_checkpoint \
  --vida_low_rank 1 --vida_high_rank 128 \
  --run_name pu-vida
```

可用
`--adaptation_parameter_scope` 选择 `layernorm`、`classifier`、`backbone`
或 `all` 做受控消融。ViDA 的高/低秩分支始终参与更新，该参数只控制额外更新
哪些基础模型参数。

快速端到端检查：

```bash
python main.py \
  --dataset PU --domains 3,2 --method tent \
  --source_epochs 1 --source_backbone_lr_scale 0 \
  --max_source_per_class 4 --max_target_per_domain 8 \
  --stream_batch_size 4 --device cpu --run_name smoke
```

## 评估协议

1. 将序列中的第一个域按类别分层拆分为源域训练集和验证集。
2. 可选的归一化统计量仅使用源域训练集拟合。
3. 训练或加载源域分类器。
4. 按指定顺序处理目标域，每个目标样本只经过一次。
5. 先记录在线预测，再执行不使用标签的适配，并单独记录同 batch 的适配后诊断。
6. 更新后的状态延续到下一个 batch 和下一个域。
7. 每个目标域结束后在源域验证集上检查遗忘，此过程不会更新适配器。
   kNN 的源域验证集不会进入其 reference bank。

`online` 是主要指标，表示从当前 batch 学习之前可获得的预测；`post_adapt` 是
同 batch 更新后的补充诊断，两者不能当作同一预测阶段直接混用。

## 输出文件

| 文件 | 内容 |
|---|---|
| `config.json` | 完整运行配置 |
| `source_history.json` | 源域训练曲线 |
| `source_checkpoint/` | 安全权重与元数据 |
| `domains.csv` | 各域在线/适配后指标和源域保持率 |
| `predictions.csv` | 样本 ID、真实标签、预测与置信度 |
| `summary.json` | 协议声明与汇总结果 |

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

## 许可证

本项目使用 [MIT License](LICENSE)。

GOLD 的运行方式、参数和移植差异见 [docs/GOLD_ZH.md](docs/GOLD_ZH.md)。
