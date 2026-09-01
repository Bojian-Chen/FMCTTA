import copy

import pytest
import torch
import torch.nn as nn

from fmctta.adapters import (
    adaptation_parameter_names,
    build_adapter,
    resolve_adaptation_parameter_scope,
)
from fmctta.model import UniFreqFaultModel
from fmctta.unifreq import UniFreqConfig, UniFreqModel, UniFreqProcessor


class TinyFoundationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 8),
            nn.LayerNorm(8),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 3))

    @property
    def backbone(self):
        return self.encoder

    def enable_encoder_finetuning(self) -> None:
        self.encoder.requires_grad_(True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs.squeeze(1)))


class TinyFeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs.squeeze(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))


def test_parameter_scopes_select_only_requested_modules() -> None:
    model = TinyFoundationModel()
    layernorm = adaptation_parameter_names(model, "layernorm")
    classifier = adaptation_parameter_names(model, "classifier")
    backbone = adaptation_parameter_names(model, "backbone")
    assert layernorm
    assert all("encoder.1" in name or "head.0" in name for name in layernorm)
    assert classifier and all(name.startswith("head.") for name in classifier)
    assert backbone and all(name.startswith("encoder.") for name in backbone)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("knn", "none"),
        ("tent", "layernorm"),
        ("note", "layernorm"),
        ("eata", "layernorm"),
        ("sar", "layernorm"),
        ("cotta", "all"),
        ("vida", "all"),
        ("continual_mae", "all"),
    ],
)
def test_method_specific_default_scope(method: str, expected: str) -> None:
    assert resolve_adaptation_parameter_scope(method, "default") == expected


@pytest.mark.parametrize(
    "method", ["source", "tent", "note", "eata", "sar", "cotta", "vida"]
)
def test_adapters_are_label_free_and_stateful(method: str) -> None:
    torch.manual_seed(3)
    model = TinyFoundationModel()
    inputs = torch.randn(6, 1, 8)
    kwargs = {}
    if method == "eata":
        kwargs["eata_entropy_margin"] = 10.0
    if method == "note":
        kwargs["note_memory_size"] = 6
        kwargs["note_update_interval"] = 6
    if method == "sar":
        kwargs["sar_entropy_margin"] = 10.0
        kwargs["sar_reset_threshold"] = 0.0
    if method == "cotta":
        kwargs["cotta_augmentations"] = 1
        kwargs["cotta_anchor_threshold"] = 1.1
    if method == "vida":
        kwargs["vida_augmentations"] = 2
        kwargs["vida_high_rank"] = 8
    adapter = build_adapter(
        method,
        model,
        torch.device("cpu"),
        learning_rate=1e-3,
        num_classes=3,
        **kwargs,
    )
    before = copy.deepcopy(adapter.model.state_dict())
    online = adapter(inputs)
    post = adapter.predict(inputs)
    assert online.shape == (6, 3)
    assert post.shape == (6, 3)
    assert adapter.batches_seen == 1
    assert adapter.samples_seen == 6
    if method == "source":
        assert adapter.updates == 0
        assert all(torch.equal(before[name], value) for name, value in adapter.model.state_dict().items())
    else:
        assert adapter.updates == 1


def test_source_rejects_adaptation_scope() -> None:
    with pytest.raises(ValueError, match="does not apply"):
        resolve_adaptation_parameter_scope("source", "all")


def test_knn_uses_only_the_labeled_source_reference_bank() -> None:
    model = TinyFeatureModel()
    reference_features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
    )
    reference_labels = torch.tensor([0, 0, 1, 1])
    adapter = build_adapter(
        "knn",
        model,
        torch.device("cpu"),
        learning_rate=1e-3,
        num_classes=2,
        knn_neighbors=3,
        knn_reference_features=reference_features,
        knn_reference_labels=reference_labels,
    )
    inputs = torch.tensor([[[0.9, 0.1]], [[0.1, 0.9]]])

    online = adapter(inputs)
    post = adapter.predict(inputs)
    stats = adapter.statistics()

    assert online.argmax(dim=1).tolist() == [0, 1]
    assert torch.equal(online, post)
    assert stats["updates"] == 0
    assert stats["reference_samples"] == 4
    assert stats["neighbors"] == 3


@pytest.mark.parametrize("feature_branch", ["frequency", "fused"])
def test_continual_mae_adapts_unifreq_frequency_bins_without_labels(
    feature_branch: str,
) -> None:
    config = UniFreqConfig(
        num_classes=0,
        num_channels=1,
        embed_dim=8,
        heads=2,
        depth=1,
        dropout=0.0,
        window_seconds=0.1,
        time_patch_seconds=0.025,
        time_transformer_layers=1,
        freq_transformer_layers=1,
        freq_max_bins=8,
    )
    encoder = UniFreqModel(config)
    model = UniFreqFaultModel(
        encoder,
        config,
        UniFreqProcessor(config),
        nn.Linear(8 if feature_branch == "frequency" else 16, 3),
        "test-unifreq",
        feature_branch=feature_branch,
    )
    adapter = build_adapter(
        "continual_mae",
        model,
        torch.device("cpu"),
        learning_rate=1e-3,
        num_classes=3,
        continual_mae_augmentations=2,
        continual_mae_augmentation_noise_std=0.1,
    )
    inputs = torch.randn(4, 1, 32)

    token_batch = adapter.model.frequency_token_batch(inputs)
    assert token_batch["tokens"].shape == (4, 8, 8)
    assert token_batch["frequency_values"].shape == (4, 8)
    unmasked_outputs, _ = adapter.model.classify_frequency_tokens(
        token_batch,
        token_batch["tokens"],
    )
    assert torch.allclose(unmasked_outputs, adapter.model(inputs))

    online = adapter(inputs)
    post = adapter.predict(inputs)
    stats = adapter.statistics()

    assert online.shape == post.shape == (4, 3)
    assert stats["updates"] == 1
    assert stats["masked_tokens"] == stats["reconstructed_tokens"] == 16
    assert stats["last_consistency_loss"] is not None
    assert stats["last_reconstruction_loss"] is not None
