from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from fmctta.model import PHMFDFaultModel, UniFreqFaultModel, build_model
from fmctta.phmfd_models import TFProFMBackbone


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


class CaptureBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(1))
        self.prepared = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.prepared = inputs.detach().clone()
        return inputs.mean(dim=(1, 2), keepdim=False).unsqueeze(1)


@pytest.mark.parametrize(
    ("family", "normalization", "expected_mean"),
    [
        ("unifault", "sample_minmax", 0.5),
        ("tf_profm", "sample_zscore", 0.0),
    ],
)
def test_phmfd_wrapper_reproduces_length_and_sample_normalization(
    family: str, normalization: str, expected_mean: float
) -> None:
    backbone = CaptureBackbone()
    model = PHMFDFaultModel(
        backbone,
        nn.Linear(1, 2),
        family=family,
        feature_dim=1,
        input_length=8,
        input_channels=1,
        input_normalization=normalization,
        length_mode=(
            "resample" if family == "unifault" else "resample_short_truncate_long"
        ),
        pretrained_checkpoint=None,
    )

    output = model(torch.arange(4, dtype=torch.float32).view(1, 4))

    assert output.shape == (1, 2)
    assert backbone.prepared is not None
    assert backbone.prepared.shape == (1, 1, 8)
    assert float(backbone.prepared.mean()) == pytest.approx(expected_mean, abs=1e-6)


def test_unifault_phmfd_checkpoint_roundtrip(tmp_path) -> None:
    original, _ = build_model(
        "unused",
        num_classes=3,
        classifier="linear",
        foundation_model="unifault",
        foundation_model_size="tiny",
    )
    with torch.no_grad():
        for parameter in original.encoder.parameters():
            parameter.fill_(0.125)
    checkpoint = tmp_path / "best_pretrain.pt"
    torch.save(
        {
            "model": original.encoder.state_dict(),
            "args": {
                "model_arch": "unifault",
                "model_size": "tiny",
                "seq_len": 1024,
                "unifault_patch_size": 64,
                "embed_dim": 128,
                "heads": 4,
                "depth": 4,
                "dropout": 0.1,
            },
        },
        checkpoint,
    )

    restored, info = build_model(
        "unused",
        num_classes=3,
        classifier="linear",
        foundation_model="unifault",
        pretrained_checkpoint=str(checkpoint),
        foundation_model_size="tiny",
    )

    assert info.loaded_tensors == info.backbone_tensors
    assert info.loaded_parameters == info.backbone_parameters
    assert all(
        torch.allclose(parameter, torch.full_like(parameter, 0.125))
        for parameter in restored.encoder.parameters()
    )


def test_tf_profm_backbone_forward_shape() -> None:
    backbone = TFProFMBackbone(
        input_len=32,
        patch_len=8,
        num_patches=4,
        d_model=8,
        d_ffn=16,
        layers=2,
        heads=2,
        feature_dim=12,
        dropout=0.0,
    )

    features = backbone(torch.randn(3, 2, 32))

    assert features.shape == (3, 12)
