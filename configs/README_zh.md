# 实验 YAML 配置说明

本目录的 YAML 文件由 `main.py --config` 读取，用于配置 FMCTTA 的一次完整实验。
本文档说明的是**运行实验**的配置；数据构建器使用的 JSON 配置请参阅
[`ctta_data_builder/README_zh.md`](../ctta_data_builder/README_zh.md)。

## 最小示例

```yaml
# configs/PU/my_tent.yaml
base: base.yaml

protocol:
  method: tent

data:
  target_samples_per_class: 20

runtime:
  run_name: pu-my-tent
```

运行：

```bash
python main.py --config configs/PU/my_tent.yaml
```

## 读取、继承与覆盖规则

YAML 顶层可使用 `protocol`、`data`、`model`、`adaptation`、`runtime` 五个区块。
这些区块只用于组织内容；运行时所有键会合并为同一个参数空间。例如，
`protocol.stream_order` 和命令行的 `--stream-order` 是同一个设置。

`base` 是可选的、相对于当前 YAML 文件的父配置路径。父配置先加载，子配置中出现的
同名键覆盖父配置；可以多层继承，但不能循环引用。有效优先级从低到高为：

```text
main.py 的程序默认值 < base 配置 < 当前 YAML < 命令行显式参数
```

YAML 键名使用 Python/`argparse` 属性名，即下划线形式；命令行参数使用连字符形式。
两者一一对应：

```yaml
source_learning_rate: 0.01
adaptation_parameter_scope: layernorm
```

```bash
--source-learning-rate 0.01 --adaptation-parameter-scope layernorm
```

注意：这条转换只针对**键名**，不转换参数值。因此 `class_block` 必须保留为
`class_block`，不能写成 `class-block`。未知键会直接报错。

除非另有说明，下面的“程序默认值”指不加载 YAML 时 `main.py` 使用的值；
[`base.yaml`](base.yaml) 显式固定跨方法共享的训练、适应和运行时参数；每个方法 YAML
显式固定自己的专用超参数及更新范围；数据集 YAML 固定 manifest、域序列、样本数、特征
分支和 checkpoint。三层继承后构成完整实验配置。只有 `dataset`/`data_root`（由
`dataset_manifest` 动态决定）、`hf_token`（不应写入版本库）与 `config`（YAML 文件入口）
不写进 YAML。

## PU / WT 的目录与运行顺序

PU 与 WT 的配置彼此独立，并都继承共享的 [`base.yaml`](base.yaml)：

```text
configs/
├── base.yaml                 # 两个数据集共用的训练、适应和运行时默认值
├── PU/
│   ├── base.yaml             # PU manifest、域序列与流顺序
│   ├── source.yaml
│   ├── knn.yaml
│   ├── tent.yaml
│   ├── eata.yaml
│   ├── sar.yaml
│   ├── cotta.yaml
│   ├── vida.yaml
│   └── continual_mae.yaml
└── WT/
    ├── base.yaml             # WT manifest、域序列与流顺序
    ├── source.yaml
    ├── knn.yaml
    ├── tent.yaml
    ├── eata.yaml
    ├── sar.yaml
    ├── cotta.yaml
    ├── vida.yaml
    └── continual_mae.yaml
```

请从仓库根目录运行。先运行对应数据集的 `source.yaml`，它会保存 source checkpoint；
再运行任意适应方法的 YAML。例如：

```bash
python main.py --config configs/PU/source.yaml
python main.py --config configs/PU/knn.yaml

python main.py --config configs/WT/source.yaml
python main.py --config configs/WT/vida.yaml
```

数据集配置使用 `data/ctta_custom` 中的 manifest，而不是旧的内置数据集条目。PU 是
14 类、4 域（`[0, 1, 2, 3]`），WT 是 5 类、8 域（`[0, 1, 2, 3, 4, 5, 6, 7]`）；
两者每类每域有 200 个样本；数据集 `base.yaml` 也显式设定
`source_samples_per_class: 200` 和 `target_samples_per_class: 200`。PU 显式采用 `shuffled`，
WT 显式采用 `interleaved`，均与各自 manifest 的默认流顺序一致。
每个数据集 `base.yaml` 的 `source_checkpoint` 精确指向同目录 `source.yaml` 在继承默认
`output_dir`、`seed` 和 `run_name` 时生成的位置，因此所有 target-domain 方法会自动继承；
`source.yaml` 显式将该项覆盖为空字符串，以确保它训练并保存 source checkpoint。若修改
这三项或源模型设置，必须相应修改数据集 `base.yaml` 的 checkpoint 路径，或在命令行以
`--source-checkpoint` 覆盖它。

## `protocol`：实验协议与目标流

| YAML 键 | 程序默认值 / 可选值 | 含义 |
| --- | --- | --- |
| `dataset` | `PU`；内置数据集名 | 使用的内置数据集规格。可选值由程序内注册表决定，例如 `PU_Real`、`WT`、`HUST`、`MCC5_THU_Motor`。设置 `dataset_manifest` 后会由清单中的数据集规格覆盖。 |
| `domains` | 空，即使用数据集默认序列 | 域序列。第一个域是有标签源域，后续每个位置都是无标签目标流；至少要有两个域。可写字符串 `"3,2,0,1"` 或 YAML 列表 `[3, 2, 0, 1]`。目标域允许重复，表示同一域在流中再次出现。域编号必须在所选数据集范围内。 |
| `method` | `source`；`source`、`knn`、`tent`、`eata`、`sar`、`cotta`、`vida`、`continual_mae` | 实验方法。`source` 与 `knn` 都不在目标流上更新；`knn` 用一次性提取的带标签 source-train 特征库分类。其余方法会在每个目标 batch 的在线预测后进行无标签更新，并将状态延续到后续 batch 和域。 |
| `stream_order` | `null`，即使用数据集的 `default_stream_order` | 每一个目标域内部的样本送入顺序：`class_block` 先完成一个类别再处理下一类；`interleaved` 按类别轮流取样；`shuffled` 将本目标流已选的全部样本随机混排。它只改变顺序，不改变各类被选样本数。PU/WT 的数据集基础配置显式设为 `interleaved`，以固定实验协议。 |
| `stream_batch_size` | `16`，正整数 | 在线目标流的 batch 大小。适应方法通常每个 batch 更新一次（`adaptation_steps` 可增加同一 batch 的更新次数）；改变它会改变在线适应轨迹，不只是显存占用。 |
| `seed` | `2025` | 随机种子，用于源域划分、样本选择/排序、模型随机性及数据加载等。相同环境和配置下有助于复现；不同硬件或底层库仍可能带来微小差异。 |

### `stream_order` 示例

若已为每类各选出两个样本，记为 `A1 A2`、`B1 B2`、`C1 C2`，则顺序分别为：

```text
class_block: A1 A2 B1 B2 C1 C2
interleaved: A1 B1 C1 A2 B2 C2
shuffled:    六个样本的确定性随机混排
```

若各类样本数不同，`interleaved` 会在某一类耗尽后继续交织其余类别。样本在该排序前
已经以 `seed` 进行类别内随机选择；因此这里的“顺序”不表示保留原始采集时序。

## `data`：数据位置、划分与采样

| YAML 键 | 程序默认值 / 可选值 | 含义 |
| --- | --- | --- |
| `data_root` | 相邻 `OCTTA/data` 目录 | 内置数据集 MAT 文件所在目录。设置 `dataset_manifest` 后，程序改用该清单所在目录，不再采用此项。 |
| `dataset_manifest` | 空字符串 | `ctta_data_builder` 生成的 JSON 清单路径。加载后，其中的动态数据集规格、MAT 文件位置和默认域序列生效，并覆盖 `dataset`、`data_root`。若 `stream_order` 未显式设置，清单中的默认流顺序会生效。 |
| `normalization` | `none`；`none`、`zscore`、`minmax` | 输入归一化方式。`zscore` 使用源域训练集均值/标准差；`minmax` 使用源域训练集最小/最大值；`none` 不变换。统计量绝不会从验证或目标域估计。 |
| `source_validation_ratio` | `0.2`，开区间 `(0, 1)` | 源域中每个类别划作验证集的比例，源域分类器按验证集 macro-F1 选取最佳 epoch。每类至少保留一个训练样本和一个验证样本。 |
| `strict_data` | `false` | 是否禁止已知数据集的标签长度修复。通常数据的 `data` 和 `label` 长度不一致会报错；仅对声明可修复标签布局的数据集，默认可按元数据重建标签并记录警告。设为 `true` 时也一律报错。 |
| `max_source_per_class` | `null` | 快速调试用：每个源域类别最多使用的样本数，正整数。与 `source_samples_per_class` 互斥。源域仍会在该数目内按 `source_validation_ratio` 分层划分。 |
| `max_target_per_domain` | `null` | 快速调试用：每个目标域最多读取的**总**样本数，正整数，不保证类别平衡。与 `target_samples_per_class` 互斥；使用它时目标域不能在 `domains` 中重复。 |
| `source_samples_per_class` | 空，即使用数据集每类全部样本 | 精确指定源域每类样本数。标量如 `60` 表示全部类别各取 60；列表如 `[60, 50, 60, 50, 60]` 表示按类别指定。每类必须在 `[2, 数据集每类样本数]` 内，且与 `max_source_per_class` 互斥。 |
| `target_samples_per_class` | 空，即每个目标流位置每类使用数据集全部样本 | 精确指定每个**目标流位置**每类的样本数。标量 `20` 表示每个目标流位置、每个类别各取 20。它和 `stream_order` 不冲突：本项先决定选哪些以及选多少样本，`stream_order` 再排列这些样本。与 `max_target_per_domain` 互斥。 |

### `target_samples_per_class` 的 YAML 写法

设当前有 5 类、3 个目标流位置。以下写法均有效：

```yaml
# 每个目标流位置、每个类别各取 20 个。
target_samples_per_class: 20

# 三个目标流位置分别让每个类别取 20、10、15 个。
target_samples_per_class: [20, 10, 15]

# 每个目标流位置都采用同一份“按类别”数目。
target_samples_per_class: [[20, 10, 15, 20, 10]]

# 每个目标流位置分别给出 5 个类别的样本数。
target_samples_per_class:
  - [20, 10, 15, 20, 10]
  - [10, 10, 10, 10, 10]
  - [15, 20, 15, 20, 15]
```

对于重复目标域，样本按出现位置**无放回**分配。例如目标序列为
`[1, 2, 1]` 且设为 `20` 时，域 1 的每个类别会共使用 40 个互不重复的样本。
同一域同一类别的累计请求量超过可用样本数会报错，而不会重用样本。

## `model`：UniFreq、分类头与源域训练

| YAML 键 | 程序默认值 / 可选值 | 含义 |
| --- | --- | --- |
| `model_id` | `bojian1/UniFreq-tiny` | Hugging Face 上的 UniFreq 模型 ID，或本地模型目录。目录/仓库需要包含 UniFreq 的 `config.json` 和 `model.safetensors`。 |
| `model_revision` | 空字符串 | Hugging Face 模型的 revision、分支、tag 或 commit。留空时由 Hugging Face 使用默认 revision。使用 checkpoint 时应与训练该 checkpoint 的 revision 相同。 |
| `hf_token` | 空字符串 | 访问私有或受限 Hugging Face 模型的 token；公开模型不需要。不要把真实 token 提交到 YAML；运行输出的 `config.json` 会脱敏，但 YAML 文件本身不会。 |
| `cache_dir` | 空字符串 | Hugging Face 下载/缓存模型文件的位置。空值使用 Hugging Face 的默认缓存目录。 |
| `local_files_only` | `false` | 为 `true` 时禁止联网下载，只从本地模型目录或缓存中读取；所需文件不存在会失败。 |
| `classifier` | `mlp`；`mlp`、`linear`、`cosine` | 源域监督训练和目标预测使用的下游分类头。`mlp` 为带 LayerNorm、隐藏层和 dropout 的 MLP；`linear` 是线性分类器；`cosine` 是归一化特征/权重的余弦分类器。更换该项不能复用不同分类头的 checkpoint。 |
| `feature_branch` | `fused`；`fused`、`time`、`frequency` | 分类器所用 UniFreq 特征。`fused` 使用时频融合特征；`time` 或 `frequency` 仅使用单个分支。三者的特征维度可能不同，切换后必须重新训练并使用对应 checkpoint。 |
| `classifier_dropout` | `0.1`，范围 `[0, 1)` | 仅 `mlp` 分类头的 dropout 概率；对 `linear`、`cosine` 无影响。 |
| `source_epochs` | `30`，正整数 | 从头训练源域分类头/可选主干的 epoch 数。提供 `source_checkpoint` 时跳过源域训练，因此本项不生效。 |
| `source_batch_size` | `64`，正整数 | 源域训练的 batch 大小。 |
| `eval_batch_size` | `128`，正整数 | 源域验证、目标域评估和适配后诊断的推理 batch 大小。不会改变 `stream_batch_size` 所定义的在线适应分组。 |
| `source_learning_rate` | `0.01` | 源域分类头的 AdamW 初始学习率，并经余弦退火调度。若主干可训练，其学习率还会乘以 `source_backbone_lr_scale`。 |
| `source_backbone_lr_scale` | `0.01`，非负数 | 源域训练时主干学习率相对分类头学习率的倍率。设为 `0` 会冻结 UniFreq 主干，只训练分类头；正数会微调主干。 |
| `source_weight_decay` | `0.0001` | 源域 AdamW 的 weight decay。 |
| `source_gradient_clip` | `1.0` | 源域训练的梯度范数裁剪阈值。正数时启用裁剪；`0` 或负数时不裁剪。 |
| `source_checkpoint` | 空字符串 | 已训练源域 checkpoint 目录，必须包含 `source_model.safetensors` 与 `source_model.json`。提供后跳过源域训练。程序会校验数据集、源域、模型/revision、分类头、特征分支、源域采样、归一化、输入通道和 seed 等元数据；不一致时拒绝加载。 |

## `adaptation`：在线适应的公共设置

`source` 与 `knn` 不更新目标数据：前者使用分类头，后者使用固定的 source-train 参考库。
其余方法遵循“先记录本 batch 的 online 预测，再根据无标签输入更新”的协议。因此
`stream_batch_size`、`adaptation_steps` 和学习率都会改变这些方法的适应过程。

| YAML 键 | 程序默认值 / 合法范围 | 含义 |
| --- | --- | --- |
| `adaptation_learning_rate` | `0.0005` | TENT、EATA、SAR、CoTTA、ViDA、Continual-MAE 中基础模型参数的适应学习率。`base.yaml` 将其设为 `0.0001`。SAR 使用 SGD+SAM；其余梯度适应方法使用 AdamW。 |
| `adaptation_weight_decay` | `0.0` | 在线适应优化器的 weight decay。 |
| `adaptation_steps` | `1`，正整数 | 对同一个在线目标 batch 连续进行的更新次数。TENT、SAR、ViDA、Continual-MAE 会按此数目重复更新；EATA 和 CoTTA 的实现固定每 batch 一次更新，设置该项不会增加它们的次数。 |
| `adaptation_parameter_scope` | `default`；见下表 | 指定哪些**基础模型**参数允许被在线更新。`source`、`knn` 只能保持 `default`。ViDA 的注入高/低秩分支、以及 Continual-MAE 的 mask token 与重建解码器始终更新；此项只额外控制基础模型参数。 |
| `knn_neighbors` | `5`，正整数 | 仅 kNN：每个目标特征投票的 source-train 邻居数。实际不能大于 source-train reference 样本总数。 |

`adaptation_parameter_scope` 的取值如下：

| 值 | 更新范围 |
| --- | --- |
| `default` | TENT/EATA/SAR 等同 `layernorm`；CoTTA/ViDA/Continual-MAE 等同 `all`。 |
| `layernorm` | 所有 `LayerNorm` 层的仿射参数。 |
| `classifier` | 仅下游分类头。 |
| `backbone` | 仅 UniFreq 编码器主干。 |
| `all` | 分类头和主干的全部基础模型参数。 |

### kNN 基线专用参数

`method: knn` 在加载（或训练）source checkpoint 后，用**源域训练集**全部已选样本一次性建立
reference bank；源域验证集不进入 reference bank，因此仍可做无泄漏的 source-validation
诊断。每个样本经过与 source checkpoint 一致的 UniFreq `feature_branch` 提取特征，L2
归一化后以余弦相似度寻找最近的 `knn_neighbors` 个源样本，并按类别做等权投票。它不会
读取目标标签、不会改变模型、不会随 batch 或域累积状态，因此 `online` 和 `post_adapt`
预测相同；target 域的 `accuracy` 即为所需的跨域 kNN 分类准确率。

PU/WT 默认 `source_samples_per_class: 200`、`source_validation_ratio: 0.2`，所以 bank 为
每类 160 个 source-train 特征（PU 共 2240，WT 共 800）。若更改源域采样、划分、特征分支
或 checkpoint，reference bank 会随该次运行重新提取。

### EATA 专用参数

下列参数仅在 `method: eata` 时使用，其他方法会忽略它们。

| YAML 键 | 程序默认值 / 合法范围 | 含义 |
| --- | --- | --- |
| `eata_fisher_alpha` | `1.0`，非负数 | Fisher 正则项权重，用源域参数附近的约束来降低遗忘。设为 `0` 时不计算 Fisher 信息，也不添加该正则项。 |
| `eata_fisher_batches` | `16`，正整数 | 估计 Fisher 信息时最多使用的源域训练 batch 数；仅在 `eata_fisher_alpha > 0` 时读取。更大通常更稳定，但源域准备更慢。 |
| `eata_entropy_margin` | `null` | 可靠样本的熵上限：仅熵严格小于该值的样本可参与更新。`null` 自动设为 `0.4 × ln(类别数)`；增大阈值通常会接纳更多样本。 |
| `eata_diversity_margin` | `0.95` | 当前预测分布与历史预测 EMA 的余弦相似度必须小于该值，样本才被视为不冗余。减小会更严格地过滤相似样本；增大则更容易接受样本。 |

### SAR 专用参数

下列参数仅在 `method: sar` 时使用。

| YAML 键 | 程序默认值 / 合法范围 | 含义 |
| --- | --- | --- |
| `sar_rho` | `0.05`，非负数 | SAM 扰动半径。`0` 退化为无半径扰动的更新；更大通常使优化更保守，但也可能不稳定。 |
| `sar_entropy_margin` | `null` | SAR 两阶段更新的可靠样本熵上限。`null` 自动设为 `0.4 × ln(类别数)`；仅低于该值的样本能参与对应阶段。 |
| `sar_reset_threshold` | `0.2`，非负数 | 源码中，当可靠样本第二阶段熵的 EMA **低于**该阈值时，模型与优化器状态恢复到源域初始状态。设为 `0` 基本可关闭此恢复条件；调整前应结合输出中的 `entropy_ema` 和 `resets` 检查实际触发情况。 |
| `sar_ema_momentum` | `0.9`，范围 `[0, 1)` | 上述熵 EMA 的动量。越接近 1，历史权重越大、触发恢复越平滑但越慢。 |

### CoTTA 专用参数

下列参数仅在 `method: cotta` 时使用。CoTTA 用教师模型的 EMA 预测作为学生目标，
并可以随机将学生参数元素恢复到源域参数。

| YAML 键 | 程序默认值 / 合法范围 | 含义 |
| --- | --- | --- |
| `cotta_teacher_momentum` | `0.999`，范围 `[0, 1)` | 教师 EMA 动量。越大，教师变化越慢、越平滑。 |
| `cotta_restore_probability` | `0.01`，范围 `[0, 1]` | 每次更新后，每个允许更新的学生参数元素恢复为源域初始值的独立概率。设为 `0` 可关闭随机恢复。 |
| `cotta_anchor_threshold` | `0.92` | 锚点模型平均最大类别概率的阈值。锚点置信度低于该值时，教师会使用多个增强视图的平均输出；达到或超过该值时使用未增强教师输出。提高该值会使增强路径更容易触发。 |
| `cotta_augmentations` | `8`，非负整数 | 触发增强教师路径时的增强视图数。设为 `0` 时不使用增强教师路径。 |
| `cotta_augmentation_noise_std` | `0.005`，非负数 | 每个样本按其当前信号标准差缩放的高斯噪声系数。`0` 关闭该增强。 |
| `cotta_augmentation_scale_std` | `0.05`，非负数 | 每个样本乘以 `1 + N(0, std)` 的幅度缩放标准差。`0` 关闭该增强。 |
| `cotta_augmentation_mask_probability` | `0.0`，范围 `[0, 1]` | 信号中每个元素被置零的独立概率。`0` 关闭掩码。 |
| `cotta_augmentation_shift_ratio` | `0.0`，范围 `[0, 1]` | 循环时移的最大幅度与信号长度的比例。实际位移在正负该上限内随机采样；`0` 关闭时移。 |

### ViDA 专用参数

下列参数仅在 `method: vida` 时使用。ViDA 在 UniFreq 主干的线性层中注入并联的低秩和
高秩分支，以 EMA 教师的一致性为目标；HKA 会根据多增强教师输出的方差，在两类分支间
调整权重。

| YAML 键 | 程序默认值 / 合法范围 | 含义 |
| --- | --- | --- |
| `vida_learning_rate` | `0.00001`，非负数 | 仅 ViDA 注入分支的学习率；基础模型参数仍使用 `adaptation_learning_rate`。设为 `0` 可使分支不学习，但由 `adaptation_parameter_scope` 选中的基础参数仍可能更新。 |
| `vida_low_rank` | `1`，正整数 | 低秩分支的内部秩。较小参数量更低，适合较小/稳定的偏移。 |
| `vida_high_rank` | `128`，正整数 | 高秩分支的内部秩。较大表达能力和显存/计算开销也更高，通常用于较大的偏移。 |
| `vida_teacher_momentum` | `0.999`，范围 `[0, 1)` | 基础模型参数写入 EMA 教师时的动量。 |
| `vida_adapter_momentum` | `0.999`，范围 `[0, 1)` | ViDA 高/低秩分支参数写入 EMA 教师时的动量。 |
| `vida_uncertainty_threshold` | `0.2`，范围 `[0, 1]` | HKA 的不确定性分界。教师在多增强视图上的 logit 方差（经 `vida_uncertainty_scale` 缩放并截断到 `[0,1]`）达到该值时，提高高秩分支权重、降低低秩分支权重；低于该值时相反。 |
| `vida_uncertainty_scale` | `0.1`，非负数 | 多增强预测方差到不确定性分数的倍率。增大后更容易越过 `vida_uncertainty_threshold`。 |
| `vida_augmentations` | `10`，正整数 | 每次估计不确定性所用的教师增强视图数。更大通常使方差估计更稳定，但线性增加教师前向计算。 |
| `vida_augmentation_noise_std` | `0.005`，非负数 | ViDA 不确定性估计所用增强的噪声系数，计算方式与 CoTTA 同名项相同。 |
| `vida_augmentation_scale_std` | `0.05`，非负数 | ViDA 增强的随机幅度缩放标准差。 |
| `vida_augmentation_mask_probability` | `0.0`，范围 `[0, 1]` | ViDA 增强中的元素置零概率。 |
| `vida_augmentation_shift_ratio` | `0.0`，范围 `[0, 1]` | ViDA 增强中的最大循环时移比例。 |

### Continual-MAE 专用参数

`continual_mae` 参考 Continual-MAE 的分布感知掩码、掩码/原始预测一致性、重建和随机
恢复机制。它以 UniFreq 的 **global FFT 频域分支** 为主体：去掉 DC 后的每个频域点
（FFT bin）就是一个 patch/token。原论文代码的 HOG 重建只适用于图像 ViT；这里改为
重建每个频域点的归一化幅值。因此这是针对振动信号和 UniFreq 的频域实现，不应与原仓库
的 CIFAR 数值结果直接比较。它要求 `feature_branch` 为 `frequency` 或 `fused`，并要求
预训练 UniFreq 使用 `freq_tokenizer: global_fft`；`patch_fft` 与 `fft_patch_conv` 会将多个
频域点或时间窗合并，不能满足“一点一 patch”的实验定义。PU/WT 配置默认使用
`feature_branch: frequency`；若改用 `fused`，掩码和重建仍只施加在频域 token，上游时域
分支保持未掩码地参与融合分类。

| YAML 键 | 程序默认值 / 合法范围 | 含义 |
| --- | --- | --- |
| `continual_mae_mask_ratio` | `0.5`，范围 `(0, 1]` | 每个样本中被掩码的频域 token（FFT bin）比例。掩码位置不是均匀随机选择：先对增强后教师 token 的方差排序，再选择不确定性最高的位置。 |
| `continual_mae_augmentations` | `10`，整数且至少为 2 | 每个目标 batch 用于估计 token 方差的增强教师前向次数。增大通常稳定掩码选择，但线性增加计算量。 |
| `continual_mae_reconstruction_weight` | `0.5`，非负数 | 归一化频域幅值重建 MSE 的权重；总损失为掩码学生对原始教师的软交叉熵加上该权重乘以重建损失。设为 `0` 可关闭重建项。 |
| `continual_mae_restore_probability` | `0.1`，范围 `[0, 1]` | 每步更新后，将每个允许适应的基础模型参数元素恢复为源域初始值的独立概率。设为 `0` 关闭随机恢复。 |
| `continual_mae_augmentation_noise_std` | `0.005`，非负数 | 方差估计增强中的相对高斯噪声系数。 |
| `continual_mae_augmentation_scale_std` | `0.05`，非负数 | 方差估计增强中的随机幅度缩放标准差。 |
| `continual_mae_augmentation_mask_probability` | `0.0`，范围 `[0, 1]` | 方差估计增强中信号元素置零的概率。 |
| `continual_mae_augmentation_shift_ratio` | `0.0`，范围 `[0, 1]` | 方差估计增强中的最大循环时移比例。 |

## `runtime`：设备与输出

| YAML 键 | 程序默认值 | 含义 |
| --- | --- | --- |
| `device` | `auto` | 运行设备。`auto` 自动选择可用 CUDA，否则 CPU；也可写 `cpu`、`cuda` 或 `cuda:0` 等 PyTorch 设备字符串。 |
| `workers` | `0`，非负整数 | PyTorch 数据加载工作进程数。`0` 表示在主进程加载，最易调试；增加后可能提高吞吐，但也会增加进程和内存开销。 |
| `output_dir` | 项目下的 `outputs` | 实验输出根目录。每次运行在其中按 `数据集/方法/运行名-seed` 创建目录，写入配置、checkpoint、预测和指标文件。`base.yaml` 写的是相对路径 `outputs`，其解析相对于启动命令的当前工作目录。 |
| `run_name` | 当前时间戳 | 本次运行名称。建议显式设置为能表达数据集、方法和关键消融项的名称，避免多次运行难以区分。 |

## 常见组合与限制

- `target_samples_per_class` 与 `stream_order` 可以一起设置，前者控制数量，后者控制顺序。
- `max_target_per_domain` 与 `target_samples_per_class` 不能一起设置；`max_source_per_class` 与 `source_samples_per_class` 也不能一起设置。
- 目标域若在 `domains` 中重复，应使用 `target_samples_per_class`，以便程序按流位置无放回分配样本；不能使用 `max_target_per_domain`。
- `method: source` 与 `method: knn` 都不执行目标自适应，`adaptation_parameter_scope` 必须保留 `default`。kNN 必须有可用的 source checkpoint 或在当前运行中先训练 source 模型。
- EATA、SAR、CoTTA、ViDA、Continual-MAE 和 kNN 各自专用的参数在选择其他方法时不会生效。为避免实验记录产生歧义，建议只在相应方法的 YAML 中保留这些参数。
- 加载 `source_checkpoint` 时，源域有关的配置必须与训练时一致。目标域序列、目标采样数、流顺序和在线适应参数可以改变。
- `hf_token` 不应放进会提交的 YAML。优先通过安全的运行环境注入，再以命令行传入。

## 配置模板的约定

- 两个数据集的 `source.yaml` 均从完整源域数据训练，目标域也默认使用各类全部 200 个样本；若需要小规模实验，可在数据集 `base.yaml` 或某个方法 YAML 中添加 `source_samples_per_class`、`target_samples_per_class`，也可使用互斥的 `max_*` 调试参数。
- `knn.yaml` 不设置更新范围；`tent.yaml`、`eata.yaml`、`sar.yaml` 选择 `layernorm`；`cotta.yaml`、`vida.yaml`、`continual_mae.yaml` 选择 `all`，与各方法的默认更新范围一致。
- 每个方法 YAML 显式列出该方法的专用超参数及其当前默认值，便于对单一方法做消融；两个数据集使用相同的算法超参数，唯一的数据集差异保留在各自的 `base.yaml` 中。
