from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from fmctta.model import UniFreqFaultModel, build_model


class SpyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(1))
        self.calls = []

    def forward(self, batch):
        self.calls.append("fused")
        return torch.cat([batch["time"], batch["frequency"]], dim=1)

    def time_features(self, batch):
        self.calls.append("time")
        return batch["time"]

    def frequency_features(self, batch):
        self.calls.append("frequency")
        return batch["frequency"]


class SpyProcessor:
    def __call__(self, inputs: torch.Tensor):
        return {
            "time": inputs[:, :4],
            "frequency": inputs[:, 4:],
        }


@pytest.mark.parametrize(
    ("feature_branch", "expected", "expected_dim"),
    [
        ("fused", "fused", 8),
        ("time", "time", 4),
        ("frequency", "frequency", 4),
    ],
)
def test_feature_branch_selects_only_the_requested_encoder_path(
    feature_branch: str, expected: str, expected_dim: int
) -> None:
    encoder = SpyEncoder()
    config = SimpleNamespace(embed_dim=4, timefreq_fusion="concat")
    model = UniFreqFaultModel(
        encoder,
        config,
        SpyProcessor(),
        nn.Identity(),
        "test-model",
        feature_branch=feature_branch,
    )

    outputs = model(torch.arange(16, dtype=torch.float32).view(2, 8))

    assert outputs.shape == (2, expected_dim)
    assert encoder.calls == [expected]
    assert model.feature_dim == expected_dim


def test_invalid_feature_branch_is_rejected() -> None:
    with pytest.raises(ValueError, match="feature_branch"):
        UniFreqFaultModel(
            SpyEncoder(),
            SimpleNamespace(embed_dim=4, timefreq_fusion="concat"),
            SpyProcessor(),
            nn.Identity(),
            "test-model",
            feature_branch="invalid",
        )


def test_build_model_sizes_single_branch_classifier(monkeypatch) -> None:
    config = SimpleNamespace(embed_dim=4, timefreq_fusion="concat")
    monkeypatch.setattr(
        "fmctta.model.load_model",
        lambda *args, **kwargs: (SpyEncoder(), config),
    )
    monkeypatch.setattr("fmctta.model.UniFreqProcessor", lambda _: SpyProcessor())

    model, _ = build_model(
        "test-model",
        num_classes=3,
        classifier="linear",
        feature_branch="frequency",
    )

    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == 4
    assert model.feature_dim == 4
