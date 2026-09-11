import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fmctta.adapters import build_adapter, resolve_adaptation_parameter_scope
from fmctta.gold_tca import GOLDTCAAdapter, intra_class_compactness_loss
from fmctta.model import CosineClassifier
from main import parse_args, validate_args


class FeatureModel(nn.Module):
    def __init__(self, classifier: str = "cosine") -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.head = (
            CosineClassifier(4, 3)
            if classifier == "cosine"
            else nn.Linear(4, 3)
        )

    @property
    def feature_dim(self) -> int:
        return 4

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs.squeeze(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))


def make_adapter(model=None, **kwargs) -> GOLDTCAAdapter:
    return build_adapter(
        "gold_tca",
        model or FeatureModel(),
        torch.device("cpu"),
        1e-3,
        3,
        gold_source_prototypes=torch.randn(3, 4),
        **kwargs,
    )


def test_gold_tca_updates_topology_on_normalized_adapted_features() -> None:
    torch.manual_seed(12)
    adapter = make_adapter(gold_conf_thresh=0.0, gold_update_every=1)
    head_before = copy.deepcopy(adapter.model.head.state_dict())
    inputs = torch.randn(12, 1, 4)
    with torch.no_grad():
        adapted = adapter.apply_adapter(adapter.model.forward_features(inputs))
        normalized = F.normalize(adapted, dim=1)
        pseudo_labels = adapter.model.head(adapted).argmax(dim=1)
        expected_centers = torch.stack(
            [
                normalized[pseudo_labels == label].mean(dim=0)
                for label in pseudo_labels.unique()
            ]
        )

    outputs = adapter(inputs)

    assert outputs.shape == (12, 3)
    assert torch.isfinite(outputs).all()
    assert adapter.updates == adapter.topology_updates == 1
    assert adapter.statistics()["classifier_frozen"] is True
    assert all(
        not parameter.requires_grad
        for parameter in adapter.model.head.parameters()
    )
    assert all(
        torch.equal(value, adapter.model.head.state_dict()[name])
        for name, value in head_before.items()
    )
    seen_centers = adapter.topology_state.historical_class_centers[
        adapter.topology_state.seen_class_mask
    ]
    assert torch.allclose(seen_centers, expected_centers, atol=1e-5)
    assert adapter.last_inter_class_loss is not None
    assert adapter.last_intra_class_loss is not None


def test_missing_class_retains_historical_center() -> None:
    adapter = make_adapter(gold_tca_centroid_momentum=0.5)
    first_features = F.normalize(torch.eye(4)[:3], dim=1)
    adapter._update_class_centers(first_features, torch.tensor([0, 1, 2]))
    class_two = adapter.topology_state.historical_class_centers[2].clone()

    second_features = F.normalize(torch.tensor([[1.0, 1.0, 0.0, 0.0]]), dim=1)
    centers, seen = adapter._update_class_centers(
        second_features,
        torch.tensor([0]),
    )

    assert seen.tolist() == [True, True, True]
    assert torch.equal(centers[2], class_two)
    assert torch.equal(adapter.topology_state.historical_class_centers[2], class_two)


def test_intra_compactness_is_zero_for_compact_classes_and_singletons() -> None:
    compact = F.normalize(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        dim=1,
    )
    spread = F.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        dim=1,
    )
    labels = torch.tensor([0, 0, 1])
    assert intra_class_compactness_loss(compact, labels) == 0
    assert intra_class_compactness_loss(spread, labels) > 0


def test_gold_tca_requires_cosine_and_never_adapts_classifier_scope() -> None:
    with pytest.raises(ValueError, match="cosine classifier"):
        make_adapter(FeatureModel("linear"))
    with pytest.raises(ValueError, match="freezes the classifier"):
        make_adapter(parameter_scope="classifier")
    with pytest.raises(ValueError, match="freezes the classifier"):
        validate_args(
            parse_args(
                [
                    "--method",
                    "gold_tca",
                    "--classifier",
                    "cosine",
                    "--adaptation_parameter_scope",
                    "classifier",
                ]
            )
        )
    assert resolve_adaptation_parameter_scope("gold_tca", "default") == "all"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("gold_tca_lambda_topology", "-1"),
        ("gold_tca_centroid_momentum", "1"),
        ("gold_tca_uniformity_temperature", "0"),
    ],
)
def test_invalid_gold_tca_config_is_rejected(option: str, value: str) -> None:
    with pytest.raises(ValueError):
        validate_args(
            parse_args(
                [
                    "--method",
                    "gold_tca",
                    "--classifier",
                    "cosine",
                    "--" + option,
                    value,
                ]
            )
        )


def test_gold_tca_profiles_are_complete_and_use_cosine() -> None:
    profiles = list(Path("configs").glob("*/*/gold_tca.yaml"))
    assert len(profiles) == 21
    for profile in profiles:
        args = parse_args(["--config", str(profile)])
        validate_args(args)
        assert args.method == "gold_tca"
        assert args.classifier == "cosine"
        assert args.source_checkpoint == ""
        assert args.adaptation_parameter_scope == "all"
