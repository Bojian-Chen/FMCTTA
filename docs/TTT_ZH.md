# TTT 在故障诊断在线流中的实现

本实现对应 Sun 等人的经典
[Test-Time Training with Self-Supervision](https://github.com/yueatsprograms/ttt_cifar_release)，
核心时序是“先使用当前测试输入完成自监督适应，再进行主任务预测”。它不是 TENT 的
另一个名字，也不使用预测熵作为损失。

## 1. 与经典 TTT 的关系

经典图像 TTT 在 source 阶段联合训练分类任务和旋转预测任务；测试时根据当前图像的
旋转标签更新共享 feature extractor，然后预测原始图像。官方测试脚本默认每个样本
恢复 source 参数，使用 `--online` 才持续保留更新。

FMCTTA 面向 continual test-time adaptation，因此固定采用 online 方式：模型、辅助头
和 optimizer 状态跨 batch、跨 domain 保留，不在 domain 边界重置。

## 2. 故障信号辅助任务

辅助任务一共有四类，标签由变换编号自动产生，不使用 fault label：

- UniFreq 一维输入沿信号轴循环移位 0、1/4、1/2、3/4 周期。

每个 source 或 target batch 会生成四组视图，所以辅助 batch 大小为原 batch 的四倍。
二维任务保持经典 TTT 的旋转预测；一维循环移位是针对振动/声学输入的替代任务。

## 3. Source 联合训练

TTT 不能直接使用普通 source checkpoint，因为普通 checkpoint 没有 jointly-trained
auxiliary head。source 训练目标为：

```text
L_source = L_fault_classification
           + ttt_source_auxiliary_weight * L_transform_prediction
```

默认：

```bash
--ttt-source-auxiliary-weight 1.0
```

两项损失共同更新 backbone；故障分类损失更新主 classifier，变换预测损失更新
auxiliary head。保存的 checkpoint 目录包含：

```text
source_model.safetensors
ttt_auxiliary_head.safetensors
metadata.ttt_auxiliary_task
metadata.ttt_source_auxiliary_weight
```

使用普通 source checkpoint 启动 TTT 会明确报错，而不是随机初始化辅助头后静默运行。

## 4. 每个目标 batch 的严格时序

对当前 batch `x_b`：

1. 额外保存更新前故障预测，写入 `online_acc`，仅用于统一诊断；
2. 为 `x_b` 生成四类变换及自动辅助标签；
3. 计算 auxiliary transform-classification loss；
4. 更新 feature extractor 和 auxiliary head；
5. 对原始 `x_b` 使用更新后的 feature extractor 和冻结的主 classifier；
6. 将第 5 步结果写入 `post_adapt_acc`，它是 TTT 的原生主结果；
7. 保留更新后的参数和 optimizer，继续处理下一个 batch。

`adaptation-steps=N` 表示第 2 至 4 步在同一个 batch 上执行 N 次，第 5 步只预测、
不再更新。

## 5. 参数更新范围

默认：

```bash
--adaptation_parameter_scope backbone
```

即更新完整 UniFreq backbone 和 auxiliary head，冻结主 classifier。也可显式选择
`layernorm` 做轻量消融；`classifier` 不受辅助任务计算图连接，因此会被明确拒绝。

## 6. 运行命令

第一次运行必须联合训练并生成 TTT checkpoint。例如 PU：

```bash
python main.py --config configs/PU/unifreq/ttt.yaml
```

后续复现必须加载上一步产生的 TTT checkpoint：

```bash
python main.py --config configs/PU/unifreq/ttt.yaml \
  --source_checkpoint outputs/PU_0p1s_custom/ttt/pu-ttt-seed2025/source_checkpoint
```

报告 TTT 时应以 `post_adapt.accuracy` 为主，同时保留 `online.accuracy` 说明当前 batch
自监督更新带来的即时变化。
