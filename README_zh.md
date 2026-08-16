# FMCTTA

> 面向预训练 [UniFreq](https://github.com/Bojian-Chen/UniFreq) 故障诊断模型的
> 持续测试时自适应（Continual Test-Time Adaptation, CTTA）框架。

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Hugging%20Face-UniFreq-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/bojian1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md) | 简体中文

FMCTTA 首先用一个有标签源域为预训练 UniFreq 编码器训练下游分类器，随后在
无标签目标域流上进行有状态的在线自适应。模型状态会跨 batch、跨域延续，不在
目标域之间重置。

## 主要特性

- 直接从 Hugging Face 加载 `UniFreq-tiny`、`UniFreq-small` 或
  `UniFreq-base`，项目中不包含预训练流程。
- 支持 Source-only、TENT、EATA、SAR、CoTTA 和 ViDA。
- 目标标签不会传入任何适配器，只在预测完成后用于计算评估指标。
- 严格采用单遍目标流，并记录跨域源域遗忘。
- 源域模型使用 `safetensors` 保存，同时输出逐域、逐样本和在线/适配后指标。
- 默认复用同层级 `OCTTA/data` 中的数据，不重复存储大型 MAT 文件。

## 模型

| CLI 参数 | Hugging Face 模型 | 建议用途 |
|---|---|---|
| `bojian1/UniFreq-tiny` | [UniFreq-tiny](https://huggingface.co/bojian1/UniFreq-tiny) | 快速实验与流程验证 |
| `bojian1/UniFreq-small` | [UniFreq-small](https://huggingface.co/bojian1/UniFreq-small) | 精度与计算量平衡 |
| `bojian1/UniFreq-base` | [UniFreq-base](https://huggingface.co/bojian1/UniFreq-base) | 最大容量编码器 |

`fmctta/unifreq/` 中保留了与公开权重一致的纯推理架构，权重仍按需从 Hugging
Face 下载，不会提交到本项目。

## CTTA 方法

| 方法 | 在线目标 | 默认更新参数 |
|---|---|---|
| `source` | 不进行目标域更新 | 无 |
| `tent` | 熵最小化 | LayerNorm 仿射参数 |
| `eata` | 可靠性/多样性筛选 + Fisher 约束 | LayerNorm 仿射参数 |
| `sar` | 可靠熵最小化 + SAM | LayerNorm 仿射参数 |
| `cotta` | EMA 教师一致性 + 随机恢复 | 全部参数 |
| `vida` | EMA 一致性 + 高/低秩 ViDA + HKA | ViDA 分支 + 全部基础参数 |

[ViDA](https://github.com/Yangsenqiao/vida) 会在 UniFreq 主干的线性层中
注入并联的高秩、低秩分支。HKA 根据增强后的教师预测不确定性，在域偏移较大时
提高高秩分支权重，否则提高低秩分支权重。ViDA 分支在加载源域 checkpoint 后
进行零输出初始化，因此已有 FMCTTA 源域 checkpoint 可以直接复用。

UniFreq 使用 Transformer/LayerNorm，因此不加入仅适用于 BatchNorm 的 AdaBN。
经典 TTT 需要在源域阶段联合训练额外的自监督任务头，而公开 UniFreq 权重不包含
该任务，所以本项目没有加入不严谨的事后 TTT 实现。

## 安装

```bash
cd /data1/PHMT/FMCTTA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

公开模型无需 Hugging Face token。数据默认读取 `../OCTTA/data`；数据预处理与
说明见 [PHMFD-data-preprocessing](https://github.com/Bojian-Chen/PHMFD-data-preprocessing)。

需要从原始数据自定义 PU、WT、MCC5-Motor 或 HUST-Gearbox 的类别和数值域标签
时，使用 [ctta_data_builder/README_zh.md](ctta_data_builder/README_zh.md)。
该构建器固定生成原采样率下的 0.1 秒非重叠窗口，不做重采样，并输出可通过
`--dataset-manifest` 直接加载的动态数据集描述。

## 快速开始

先训练源域分类器并运行 Source-only：

```bash
python main.py \
  --dataset PU_Real \
  --method source \
  --model-id bojian1/UniFreq-tiny \
  --domains 3,2,0,1 \
  --run-name pu-source
```

源域 checkpoint 会保存在：

```text
outputs/PU_Real/source/pu-source-seed2025/source_checkpoint/
```

### 分支选择

`--feature-branch fused` 为默认值，保持原有的时频融合。做消融实验时可指定
`--feature-branch time` 只运行时域分支，或指定 `--feature-branch frequency`
只运行频域分支。单分支的特征维度是 `embed_dim`，与融合后的拼接特征不同，因此每种
选择都需要重新训练源域分类器并使用对应 checkpoint；checkpoint 会记录该选项并拒绝
加载分支不一致的模型。

例如，仅用频域分支训练源域模型：

```bash
python main.py \
  --dataset PU_Real --domains 3,2,0,1 --method source \
  --feature-branch frequency --run-name pu-frequency-source
```

### 每类采样数与重复目标域

`--source-samples-per-class` 可填一个数（所有源域类别相同），或以逗号分隔的每类
样本数。`--target-samples-per-class` 的每一项也可填一个数或每类样本数；单项会应用于
所有目标流位置，多个目标流位置则用分号分隔。目标域允许重复，同一原始样本会按出现
位置无放回分配，因此同一目标域同一类别的累计请求数不能超过可用样本数。
加载源域 checkpoint 时，须传入训练该 checkpoint 时相同的源域采样配置；该配置会被
记录在 checkpoint 元数据中。

```bash
python main.py \
  --dataset PU_Real --method tent \
  --domains 0,1,2,3,1,3,2,3 \
  --source-samples-per-class 60 \
  --target-samples-per-class 20 \
  --run-name repeated-targets
```

上例有 7 个目标流位置，每个位置的每类各取 20 个样本；因此域 1、域 2 各使用每类
40 个互不重复的样本，域 3 使用每类 60 个。需要按目标流位置设置不同数目时，例如可用
`--target-samples-per-class '20;10;15;20;10;15;20'`。

同样的实验已写为可直接运行的
[`configs/experiments/tent_frequency_repeated.yaml`](configs/experiments/tent_frequency_repeated.yaml)。
它通过 [`configs/base.yaml`](configs/base.yaml) 继承公共配置，因此实验文件中只保留
该次实验需要调整的参数：

```bash
python main.py --config configs/experiments/tent_frequency_repeated.yaml
```

命令行参数会覆盖 YAML 中的同名配置。

然后复用该 checkpoint 运行 TENT：

```bash
python main.py \
  --dataset PU_Real \
  --method tent \
  --model-id bojian1/UniFreq-tiny \
  --domains 3,2,0,1 \
  --source-checkpoint outputs/PU_Real/source/pu-source-seed2025/source_checkpoint \
  --run-name pu-tent
```

把 `tent` 替换为 `eata`、`sar`、`cotta` 或 `vida` 即可运行其他方法。
ViDA 示例：

```bash
python main.py \
  --dataset PU_Real --domains 3,2,0,1 --method vida \
  --source-checkpoint outputs/PU_Real/source/pu-source-seed2025/source_checkpoint \
  --vida-low-rank 1 --vida-high-rank 128 \
  --run-name pu-vida
```

可用
`--adaptation-parameter-scope` 选择 `layernorm`、`classifier`、`backbone`
或 `all` 做受控消融。ViDA 的高/低秩分支始终参与更新，该参数只控制额外更新
哪些基础模型参数。

快速端到端检查：

```bash
python main.py \
  --dataset PU_Real --domains 3,2 --method tent \
  --source-epochs 1 --source-backbone-lr-scale 0 \
  --max-source-per-class 4 --max-target-per-domain 8 \
  --stream-batch-size 4 --device cpu --run-name smoke
```

## 评估协议

1. 将序列中的第一个域按类别分层拆分为源域训练集和验证集。
2. 可选的归一化统计量仅使用源域训练集拟合。
3. 训练或加载源域分类器。
4. 按指定顺序处理目标域，每个目标样本只经过一次。
5. 先记录在线预测，再执行不使用标签的适配，并单独记录同 batch 的适配后诊断。
6. 更新后的状态延续到下一个 batch 和下一个域。
7. 每个目标域结束后在源域验证集上检查遗忘，此过程不会更新适配器。

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
