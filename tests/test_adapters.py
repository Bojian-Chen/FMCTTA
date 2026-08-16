import copy

import pytest
import torch
import torch.nn as nn

from fmctta.adapters import (
    adaptation_parameter_names,
    build_adapter,
    resolve_adaptation_parameter_scope,
)


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
        ("tent", "layernorm"),
        ("eata", "layernorm"),
        ("sar", "layernorm"),
        ("cotta", "all"),
        ("vida", "all"),
    ],
)
def test_method_specific_default_scope(method: str, expected: str) -> None:
    assert resolve_adaptation_parameter_scope(method, "default") == expected


@pytest.mark.parametrize(
    "method", ["source", "tent", "eata", "sar", "cotta", "vida"]
)
def test_adapters_are_label_free_and_stateful(method: str) -> None:
    torch.manual_seed(3)
    model = TinyFoundationModel()
    inputs = torch.randn(6, 1, 8)
    kwargs = {}
    if method == "eata":
        kwargs["eata_entropy_margin"] = 10.0
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
