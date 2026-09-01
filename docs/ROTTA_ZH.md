# RoTTA 在 FMCTTA 中的实现

本实现依据 [RoTTA 官方仓库](https://github.com/BIT-DA/RoTTA) 和 CVPR 2023 论文
“Robust Test-Time Adaptation in Dynamic Scenarios”，将图像 PTTA 方法的 CSTU、EMA
teacher、时效加权一致性目标移植到 UniFreq 在线协议。目标真实标签只进入指标统计器。

> 兼容性说明：UniFreq 是 Transformer/LayerNorm 主干，没有 BatchNorm running
> statistics。因此本仓库的 `rotta` 是明确标注的 LayerNorm 端口：保留 RoTTA 的动态
> 内存与 teacher 核心，在线更新 LayerNorm 仿射参数；不会伪造 Robust BatchNorm。

## 核心流程

对每个 target batch，EMA teacher 先在更新前输出 logits。每个样本的 teacher 伪标签和
预测熵随信号写入 CSTU（category-balanced sampling by timeliness and uncertainty）：

1. 优先保持类别占用平衡；
2. 同一类别或满内存候选中，依据时效与不确定性联合分数淘汰；
3. 每写入一个流样本，所有保留项的 age 加一；
4. 每累计 `rotta_update_frequency` 个目标样本，用当前完整内存更新一次。

更新时，teacher 在原始内存信号上产生软目标，student 在增强信号上预测。逐样本 soft
cross-entropy 乘以 `sigmoid(-age / memory_size)` 后取平均，Adam 更新 student，再执行：

```text
teacher = (1 - nu) * teacher + nu * student
```

`adaptation_steps` 表示每个触发事件对同一内存执行的 optimizer step 数，不会把同一流
样本重复写入 CSTU。

## LayerNorm 适配边界

默认 `adaptation_parameter_scope: layernorm`，只更新 UniFreq 中 LayerNorm 的 weight/bias。
`rotta_bn_alpha` 为与 OCTTA 配置对齐而保留，但在此 LayerNorm 端口中不参与前向计算；
输出统计会明确记录 `normalization_backend: layernorm_affine` 和
`robust_batch_norm: false`。需要论文原版 Robust BN 时应使用含 BatchNorm 的 OCTTA 主干。

## 信号增强

官方实现使用 CIFAR 图像强增强。FMCTTA 将其替换为已有的故障信号增强：幅值缩放、小幅
高斯噪声、可选逐点 mask 和末轴平移。默认只启用保守的缩放与噪声；mask/平移可能改变
冲击或谱线特征，因此保持为 0，需按数据集显式消融。

## YAML 与默认值

四个数据配置均提供可直接运行的 `rotta.yaml`：

- `configs/PU/rotta.yaml`
- `configs/WT/rotta.yaml`
- `configs/HUST_Gearbox/rotta.yaml`
- `configs/MCC5_motor/rotta.yaml`

先运行同目录 `source.yaml`，再运行 RoTTA。例如：

```bash
python main.py --config configs/PU/source.yaml
python main.py --config configs/PU/rotta.yaml
```

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `adaptation_learning_rate` | `0.001` | student Adam 学习率 |
| `rotta_memory_size` | `64` | CSTU 容量 |
| `rotta_update_frequency` | `64` | 按目标样本计的更新间隔 |
| `rotta_nu` | `0.001` | EMA teacher 的 student 更新率 |
| `rotta_bn_alpha` | `0.05` | 兼容保留项；LayerNorm 端口不使用该融合率 |
| `rotta_lambda_t` | `1.0` | CSTU 时效分数权重 |
| `rotta_lambda_u` | `1.0` | CSTU 不确定性分数权重 |

`summary.json` 的 `adapter` 字段会记录内存容量/类别占用、接收与丢弃样本数、更新事件、
待更新样本数、EMA/BN 参数及实际适配参数量。
