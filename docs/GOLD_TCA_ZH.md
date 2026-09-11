# GOLD-TCA

FMCTTA 的 `gold_tca` 对齐 OCTTA 当前算法，同时保留 UniFreq / UniFault /
TF-ProFM 的 preprocessing、`forward_features()` 接口、LayerNorm 更新范围和
safetensors checkpoint。

## 默认模式：classifier weights

默认 `gold_tca_agop_init: classifier_weights` 不计算 source prototypes。训练好的
cosine classifier 权重 `W` 在 CTTA 全程冻结，并同时承担三种角色：

1. 伪标签对应的归一化分类器权重是 semantic contrastive anchor；
2. `normalize(W).T @ normalize(W)` 初始化 GOLD Golden Subspace 的 AGOP；
3. `T_source = normalize(W) @ normalize(W).T` 是固定 source class topology。

之后的动态 AGOP 仍由高置信 target samples 计算。每个 target batch 的 GOLD-adapted
feature 先 L2 normalization，再按伪标签计算类中心；跨 batch/domain 的中心以 EMA
持续更新，缺失类别保持不变。目标 topology 为：

```text
T_target = normalize(C) @ normalize(C).T
```

topology consistency 对当前已见类别的 source/target 非对角 cosine 关系计算 MSE，
并保留 normalized feature 的 intra-class compactness：

```text
L = L_gold
  + gold_tca_lambda_topology * (L_topology_consistency + L_intra_compactness)
```

source topology 始终来自冻结 classifier weights，不会随 target center 或 prototype
memory 改变。

## Prototype-memory 变体

`gold_tca_agop_init: source_prototypes` 只在初始化时从 source train 计算 class means：

- `source_prototypes` 作为 immutable buffer 保留；
- 独立 `prototype_memory` 从它初始化；
- 每个适应 batch 用未归一化的 adapted target features 和伪标签按类别做 EMA；
- batch 中缺失的类别不更新，memory 跨 batch/domain 保留；
- 更新后的 memory 同时用于 semantic anchors 和 dynamic prototype AGOP；
- `predict()` 是只读路径，不更新 center、prototype、AGOP、basis、teacher 或计数器。

该变体只替换 anchor/AGOP 来源；source topology 仍固定为 classifier-weight Gram。

## 配置与运行

每个数据集/基础模型目录提供三份配套配置：

```bash
# 1. 先训练唯一的 cosine source checkpoint
python main.py --config configs/SDUST_Bearing/unifault/source_cosine.yaml

# 2a. 默认 classifier-weight 模式
python main.py --config configs/SDUST_Bearing/unifault/gold_tca.yaml

# 2b. prototype-memory 变体；复用完全相同的 cosine checkpoint
python main.py --config configs/SDUST_Bearing/unifault/gold_tca_prototype.yaml
```

参数：

- `gold_tca_agop_init`：`classifier_weights`（默认）或 `source_prototypes`；
- `gold_tca_lambda_topology`：结构损失权重，默认 `0.15`；
- `gold_tca_centroid_momentum`：target center 和 prototype memory 的 EMA 动量，默认 `0.9`。

GOLD-TCA 强制使用 `classifier: cosine`，且不能选择仅适应 classifier 的参数范围。
共享配置位于 `configs/methods/gold_tca.yaml`、`source_cosine.yaml` 和
`gold_tca_prototype.yaml`。
