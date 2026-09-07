# TCA 在 FMCTTA 中的实现说明

TCA（Topological Consistency Adaptation）通过持续维护类别中心和类别间拓扑，降低
continual test-time adaptation 中的类别混淆。本实现面向 UniFreq 一维故障信号，
复用普通 source checkpoint，不读取 target 标签。

## 1. 上游状态与复现口径

- 论文：[Maintaining Consistent Inter-Class Topology in Continual Test-Time
  Adaptation](https://openaccess.thecvf.com/content/CVPR2025/html/Ni_Maintaining_Consistent_Inter-Class_Topology_in_Continual_Test-Time_Adaptation_CVPR_2025_paper.html)，
  CVPR 2025。
- 官方仓库：[Successybbdwm/TCA](https://github.com/Successybbdwm/TCA)。
- 截至本实现核对时，官方 `main` 分支的 `classification/methods/tca.py` 只有
  `coming soon!`，没有可直接运行的 TCA 或配置文件。
- 官方仓库的[历史提交
  `1de3ad7`](https://github.com/Successybbdwm/TCA/blob/1de3ad71861891b8384cc1b458eaeebfbfd6048f/classification/methods/tca.py)
  曾包含一份未完成草稿，但存在未定义变量和缺失函数。

因此，FMCTTA 的 TCA 是依据论文公式、算法流程和该历史草稿进行的可运行复原，并针对
缺失类别、单样本伪类别及数值稳定性做了加固；它不是官方完整代码的逐行移植，也不
声称与尚未公开的最终实现逐比特一致。

## 2. 初始化

从同一 source 模型创建 student 与冻结梯度的 EMA teacher。student 默认开放全部参数，
使用 Adam 更新所选 UniFreq 参数；teacher 保存完整 EMA 状态。另维护：

- 每个伪类别的历史中心 `historical_class_centers`；
- 类别中心是否曾出现的 `seen_class_mask`；
- 历史 batch topology weighting（BTW）边权矩阵
  `historical_topology_weights`。

三项状态由 `TCAStateManager` 集中维护；`reset()` 会同时清空历史中心、seen mask、历史
边权及边权初始化状态。

普通 source checkpoint 即可，不需要额外训练辅助头。

## 3. 每个目标 batch

1. student 与 teacher 在原始信号上分别产生更新前 logits。
2. 生成两个独立信号增强视图：默认只做小幅高斯噪声和幅值缩放。
3. student 在第一个视图上的预测给出伪标签，并按伪类别计算归一化特征中心。
4. 当前中心与历史中心按 `tca_centroid_momentum` 融合；当前 batch 未出现的类别保留
   历史中心。
5. 用类别计数的平方根频率构造 BTW 边权，并从第二个 batch 起按
   `tca_topology_weight_momentum` 跨 batch 平滑。
6. 计算三类无标签目标：
   - student/teacher 双向 soft cross-entropy；默认严格按 Eq. 11 将两个方向直接相加，
     teacher logits 始终 stop-gradient；
   - 两个增强视图的特征 alignment；
   - 类间与类内 Gaussian-potential topology uniformity。
7. 总损失为：

       L = L_sce + lambda_1 * L_align + lambda_2 * L_topology

8. 更新 student，再用 `teacher = m * teacher + (1-m) * student` 更新 teacher。
9. 标准 online 输出为该次更新前 student 与 teacher logits 的等权平均；更新后的
   student/teacher 等权输出只用于 `post_adapt_*` 诊断指标。

target 标签只在 adapter 返回预测后交给指标模块，不参与伪标签、loss 或更新。

## 4. 数值与公式解释

- 论文给出 `lambda_1=0.025`、`lambda_2=0.15`、Gaussian temperature `t=2`，YAML
  直接采用这些值。
- 论文 Alignment 段落要求“最小化两视图距离”，但印刷公式在平方距离前带负号；若
  直接最小化负距离会把两视图推远。这里按文字目标及标准 alignment 定义实现为正的
  squared distance，并在配置与结果解释中明确这一点。
- 论文未公开类别中心 momentum 的默认值，本项目声明式地设为 `0.9`，可通过 YAML/CLI
  调整。
- BTW 历史草稿使用 `0.8 * previous + 0.2 * current`，所以
  `tca_topology_weight_momentum=0.8`。
- 缺失伪类别沿用历史中心；从未出现的类别不参与类间 loss。单样本伪类别不参与类内
  pairwise loss，从而避免空均值和 NaN。
- 类间加权以 log-sum-exp 计算，并对 BTW 权重停止梯度。

## 5. YAML 与参数

四套可直接继承对应 source 配置的文件：

```bash
python main.py --config configs/PU/unifreq/tca.yaml
python main.py --config configs/WT/unifreq/tca.yaml
python main.py --config configs/HUST_Gearbox/unifreq/tca.yaml
python main.py --config configs/MCC5_motor/unifreq/tca.yaml
```

核心默认值：

| 参数 | 默认值 | 来源/含义 |
| --- | ---: | --- |
| `adaptation_learning_rate` | `0.001` | Adam 学习率 |
| `adaptation_parameter_scope` | `all` | student 全参数 |
| `tca_teacher_momentum` | `0.999` | EMA teacher |
| `tca_centroid_momentum` | `0.9` | 声明的复原默认值 |
| `tca_topology_weight_momentum` | `0.8` | 历史草稿的 previous/current 比例 |
| `tca_alignment_weight` | `0.025` | 论文 `lambda_1` |
| `tca_topology_weight` | `0.15` | 论文 `lambda_2` |
| `tca_uniformity_temperature` | `2.0` | 论文 Gaussian temperature |
| `tca_symmetric_alpha` | `0.5` | 两方向等权，此时严格得到 Eq. 11 的直接和而非平均值 |

故障信号增强参数均以 `tca_augmentation_*` 暴露。默认关闭逐点 mask 和轴平移，避免
人为制造冲击或移动频谱故障线。

## 6. 持续状态与结果字段

student、Adam、EMA teacher、类别中心和 BTW 边权跨 batch、跨 domain 保留，不在域边界
重置。`summary.json` 的 `adapter` 字段记录参数范围、更新次数、已观察中心数、增强视图
数以及最近一次 SCE/alignment/inter/intra/total loss，便于核查方法确实发生了更新。
