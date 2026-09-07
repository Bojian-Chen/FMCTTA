# GOLD：低秩子空间持续测试时适应

本项目提供 [AIGNLAI/GOLD](https://github.com/AIGNLAI/GOLD) 的分类方法移植，
对应 CVPR 2026 论文 *The Golden Subspace: Where Efficiency Meets Generalization
in Continual Test-Time Adaptation*。算法参考官方
[`classification/methods/gold.py`](https://github.com/AIGNLAI/GOLD/blob/main/classification/methods/gold.py)
和 [`classification/conf.py`](https://github.com/AIGNLAI/GOLD/blob/main/classification/conf.py)。
本实现使用 FMCTTA 的模型、数据与 online 评估接口，不需要安装官方图像数据集依赖。

## 运行

先运行同一基础模型的 `source.yaml` 生成 checkpoint，再运行 GOLD：

```bash
python main.py --config configs/SDUST_Bearing/unifault/source.yaml
python main.py --config configs/SDUST_Bearing/unifault/gold.yaml
```

已有兼容 source checkpoint 时直接运行第二条。`gold.yaml` 继承本模型的 `base.yaml`，
因此复用其中的 normalization、分类头、source checkpoint 和域顺序。
GOLD 与 `source.yaml` 从同一份 `base.yaml` 继承源样本数和验证比例，保证 checkpoint
元数据匹配，并从相同的 source train 划分提取原型。
七个数据集的 `unifreq`、`unifault`、`tf_profm` 子目录均提供 `gold.yaml`。
`source_checkpoint` 路径支持 checkpoint 目录或其上一级实验目录，均保留元数据校验。

## 适应过程

1. 加载/训练 source model 后，仅用 **source train** 样本累计各类别特征均值。
   每类必须至少有一个样本；不使用 source validation 或 target 标签计算原型。
   原型在进入目标流前生成，在线期间只保留这些均值，不回放源域样本。
2. 线性分类头用 `W.T @ W` 初始化 AGOP 矩阵，取最大特征值对应的 `r` 个方向 `V`。
   缩放向量 `s` 从零初始化，使用 `f + ((f @ V) * s) @ V.T` 校准特征。
3. 原始信号和增强信号通过学生模型；EMA 教师提供停止梯度的软目标。
   最近源原型、原始特征和增强特征组成三视图，使用可训练投影头计算对比损失。
   总损失为对称交叉熵自训练项与对比项的加权和。
4. 对置信度达标的样本，计算**未适配特征上最大分类 logit 的梯度**外积平均值，
   按 `(1-alpha) * M + alpha * batch_agop` 更新 AGOP。少于两个可靠样本时跳过
   AGOP 更新，但仍可进行损失优化。每 `gold_update_every` 个适应迭代刷新子空间。
5. `forward()` 返回本 batch **第一次更新前**的学生与教师 logits 之和，
   `predict()` 使用当前模型和子空间，不更新参数或统计计数。跨域保留状态。

## 参数

公共参数位于 `configs/methods/gold.yaml`，可以通过命令行覆盖。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `adaptation_learning_rate` | `0.001` | Adam 学习率 |
| `adaptation_parameter_scope` | `all` | 基础模型更新范围；也可选现有的 `layernorm`、`classifier`、`backbone` |
| `gold_top_r` | `64` | 子空间维度，实际取与特征维度的较小值 |
| `gold_alpha` | `0.1` | 新 batch AGOP 的权重 |
| `gold_conf_thresh` | `0.8` | AGOP 样本置信度阈值，不是损失筛选阈值 |
| `gold_update_every` | `50` | 子空间刷新间隔，以适应迭代计数 |
| `gold_adapter_lr_scale` | `1.0` | 缩放向量相对基础学习率的倍率 |
| `gold_lambda_ce_trg` / `gold_lambda_cont` | `1.0` / `1.0` | 自训练 / 三视图对比损失权重 |
| `gold_temperature` | `0.1` | 对比学习温度 |
| `gold_projection_dim` | `128` | 两层投影头维度 |
| `gold_teacher_momentum` | `0.999` | 基础模型 EMA 教师动量 |
| `gold_warmup_steps` | `0` | 进入目标流之前额外执行的源域无标签 warm-up 步数 |
| `gold_augmentation_noise_std` / `gold_augmentation_scale_std` | `0.005` / `0.05` | 信号噪声和幅值增强 |
| `gold_augmentation_mask_probability` / `gold_augmentation_shift_ratio` | `0.0` / `0.0` | 信号遮罩和循环平移增强 |

基础模型参数范围不影响缩放向量、投影头的训练；二者始终加入优化器。
适应统计包含实际训练参数量、子空间维度、AGOP / 子空间刷新次数及 warm-up 步数。

## 与官方图像实现的区别

- 官方分类代码在 CIFAR 上还训练基础模型全部参数，在其他图像设置上选择归一化参数，
  并训练投影头。这里默认 `all`，因此本配置并非“只训练 r 个参数”的消融。
  基础模型保持 eval 模式，避免 dropout 和小 batch 的 BatchNorm 统计改变。
- 官方初始化假设线性分类头。这里对 MLP 和余弦分类头使用源原型处分类器 Jacobian
  的 `J.T @ J` 平均值初始化，这是适配 FMCTTA 分类头的扩展；线性头保持官方公式。
- 图像增强改为一维/多通道信号增强，沿用项目已有实现。
- 官方默认额外 warm-up 样本数为 50000。为复用统一 source checkpoint、保持现有
  基线的源模型起点，本项目默认关闭额外 warm-up；需要时显式设置 `gold_warmup_steps`。
  warm-up 只发生在目标流之前，不计入 online 更新数。
- 子空间刷新放在当前反向传播完成后，避免修改参与 autograd 的矩阵；新方向从下次
  适应迭代生效。秩会按特征维度裁剪，单样本 batch 直接适应，且增加非有限值检查。

这是一份故障诊断信号任务上的方法移植，不代表已复现官方图像 benchmark 或论文指标。
