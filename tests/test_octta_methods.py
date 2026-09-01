from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fmctta.adapters import build_adapter, resolve_adaptation_parameter_scope
from fmctta.source import load_checkpoint, save_checkpoint
from fmctta.tca import (
    batch_class_centers,
    batch_topology_weights,
    symmetric_cross_entropy,
)
from fmctta.ttt import build_ttt_head, make_ttt_batch


class TinyUniFreqLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.GELU())
        self.head = nn.Linear(8, 3)

    @property
    def backbone(self):
        return self.encoder

    @property
    def feature_dim(self) -> int:
        return 8

    def enable_encoder_finetuning(self) -> None:
        self.encoder.requires_grad_(True)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs.squeeze(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))


@pytest.mark.parametrize(
    ("method", "scope"),
    [("rotta", "layernorm"), ("tca", "all"), ("ttt", "backbone")],
)
def test_transferred_methods_have_unifreq_scopes(method: str, scope: str) -> None:
    assert resolve_adaptation_parameter_scope(method, "default") == scope


def test_ttt_transforms_checkpoint_and_online_update(tmp_path: Path) -> None:
    model = TinyUniFreqLikeModel()
    head = build_ttt_head(model.feature_dim)
    transformed, labels = make_ttt_batch(torch.randn(2, 1, 8))
    assert transformed.shape == (8, 1, 8)
    assert labels.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]

    save_checkpoint(
        tmp_path,
        model,
        {"method": "ttt"},
        extra_modules={"ttt_auxiliary_head": head},
    )
    restored_model = TinyUniFreqLikeModel()
    restored_head = build_ttt_head(restored_model.feature_dim)
    load_checkpoint(
        tmp_path,
        restored_model,
        {"method": "ttt"},
        extra_modules={"ttt_auxiliary_head": restored_head},
    )
    adapter = build_adapter(
        "ttt",
        restored_model,
        torch.device("cpu"),
        learning_rate=1e-3,
        num_classes=3,
        ttt_auxiliary_head=restored_head,
    )
    inputs = torch.randn(4, 1, 8)
    assert adapter(inputs).shape == adapter.predict(inputs).shape == (4, 3)
    assert adapter.updates == 1


def test_tca_helpers_and_adapter_are_finite() -> None:
    features = torch.randn(5, 8)
    pseudo_labels = torch.tensor([0, 0, 1, 1, 1])
    centers, counts, present = batch_class_centers(features, pseudo_labels, 3)
    assert centers.shape == (3, 8)
    assert present.tolist() == [True, True, False]
    assert torch.isfinite(batch_topology_weights(counts)).all()
    assert torch.isfinite(
        symmetric_cross_entropy(torch.randn(5, 3), torch.randn(5, 3))
    ).all()

    adapter = build_adapter(
        "tca",
        TinyUniFreqLikeModel(),
        torch.device("cpu"),
        learning_rate=1e-3,
        num_classes=3,
    )
    inputs = torch.randn(5, 1, 8)
    assert adapter(inputs).shape == adapter.predict(inputs).shape == (5, 3)
    assert adapter.updates == 1


def test_rotta_layernorm_port_uses_cstu_memory_and_ema_teacher() -> None:
    adapter = build_adapter(
        "rotta",
        TinyUniFreqLikeModel(),
        torch.device("cpu"),
        learning_rate=1e-3,
        num_classes=3,
        rotta_memory_size=6,
        rotta_update_frequency=3,
    )
    inputs = torch.randn(6, 1, 8)
    assert adapter(inputs).shape == adapter.predict(inputs).shape == (6, 3)
    statistics = adapter.statistics()
    assert statistics["memory_occupancy"] <= 6
    assert statistics["update_events"] == 2
    assert statistics["normalization_backend"] == "layernorm_affine"
