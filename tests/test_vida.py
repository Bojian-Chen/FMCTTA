import copy

import torch
import torch.nn as nn

from fmctta.adapters import build_adapter
from fmctta.vida import (
    ViDAInjectedLinear,
    inject_vida,
    is_vida_parameter,
    set_vida_scales,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 8),
            nn.LayerNorm(8),
            nn.GELU(),
        )
        self.head = nn.Linear(8, 3)

    @property
    def backbone(self) -> nn.Module:
        return self.encoder

    def enable_encoder_finetuning(self) -> None:
        self.encoder.requires_grad_(True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs.squeeze(1)))


def test_injection_preserves_source_outputs_and_weights() -> None:
    torch.manual_seed(7)
    backbone = nn.Sequential(nn.Linear(8, 6), nn.GELU(), nn.Linear(6, 4))
    source = copy.deepcopy(backbone)
    inputs = torch.randn(5, 8)
    expected = source(inputs)

    names = inject_vida(backbone, low_rank=1, high_rank=8)

    assert names == ["0", "2"]
    assert isinstance(backbone[0], ViDAInjectedLinear)
    assert torch.equal(backbone[0].weight, source[0].weight)
    assert torch.equal(backbone[0].bias, source[0].bias)
    assert torch.allclose(backbone(inputs), expected)


def test_hka_scales_update_both_branches() -> None:
    layer = ViDAInjectedLinear(nn.Linear(4, 3), low_rank=1, high_rank=4)
    assert set_vida_scales(layer, low=1.25, high=0.75) == 1
    assert layer.vida_low_scale.item() == 1.25
    assert layer.vida_high_scale.item() == 0.75


def test_vida_builds_separate_adapter_group_and_ema_teacher() -> None:
    torch.manual_seed(11)
    adapter = build_adapter(
        "vida",
        TinyModel(),
        torch.device("cpu"),
        learning_rate=1e-3,
        num_classes=3,
        parameter_scope="classifier",
        vida_learning_rate=2e-3,
        vida_low_rank=1,
        vida_high_rank=8,
        vida_augmentations=2,
        vida_augmentation_noise_std=0.1,
    )
    inputs = torch.randn(6, 1, 8)

    online = adapter(inputs)
    post = adapter.predict(inputs)
    stats = adapter.statistics()

    assert online.shape == post.shape == (6, 3)
    assert len(adapter.optimizer.param_groups) == 2
    assert adapter.optimizer.param_groups[0]["lr"] == 1e-3
    assert adapter.optimizer.param_groups[1]["lr"] == 2e-3
    assert adapter.vida_parameter_names
    assert all(is_vida_parameter(name) for name in adapter.vida_parameter_names)
    assert all(name.startswith("head.") for name in adapter.base_parameter_names)
    assert stats["injected_linear_layers"] == 1
    assert stats["updates"] == 1
    assert stats["low_rank_batches"] + stats["high_rank_batches"] == 1
