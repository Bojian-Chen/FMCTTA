# GOLD-TCA

`gold_tca` 将 GOLD 的低秩判别子空间适应、TCA 类别拓扑约束和冻结的
Cosine Classifier 组合在同一个角度特征空间中。

在线步骤如下：

1. backbone 产生特征 `z`；
2. GOLD 使用 `z' = z + ((z @ V) * s) @ V.T` 进行低秩适应；
3. 冻结的 Cosine Classifier 使用 `z'` 预测伪标签；
4. 对 `z'` 做 L2 normalization，并按伪标签计算 batch 类中心；
5. 用 EMA 更新历史类中心，当前 batch 缺失的类别保留已有中心；
6. 在归一化类中心上计算类间 uniformity loss，在归一化样本特征上计算
   类内 compactness loss；
7. 将拓扑损失加入 GOLD 原有 self-training 和 source-prototype contrastive
   目标。

总损失为：

```text
L = lambda_ce_trg * L_self_training
  + lambda_cont * L_source_contrastive
  + gold_tca_lambda_topology * (L_inter + L_intra)
```

该实现不引入第二个 TCA teacher、额外 self-training、feature alignment 或
第二套 augmentation。GOLD 的 EMA teacher 与 augmentation 被原样复用。

## 配置

共享配置位于 `configs/methods/gold_tca.yaml`。例如：

```bash
python main.py --config configs/SDUST_Bearing/unifault/gold_tca.yaml
```

主要新增参数：

- `gold_tca_lambda_topology`：拓扑总损失权重，默认 `0.15`；
- `gold_tca_centroid_momentum`：历史类别中心 EMA 动量，默认 `0.9`；
- `gold_tca_uniformity_temperature`：类间 uniformity 温度，默认 `2.0`。

`gold_tca` 强制使用 `classifier: cosine`，并在 CTTA 阶段冻结 classifier 的
weight 和 scale。现有公共 source checkpoint 使用 linear classifier，因此各
`gold_tca.yaml` 默认清空 `source_checkpoint` 并训练匹配的 cosine source model。
如已有兼容的 cosine checkpoint，可在实验配置中显式指定它。
