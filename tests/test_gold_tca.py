import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

from fmctta import runner
from fmctta.adapters import build_adapter, resolve_adaptation_parameter_scope
from fmctta.config import DATASETS, DatasetSpec
from fmctta.gold_tca import GOLDTCAAdapter
from fmctta.model import CosineClassifier
from fmctta.tca import (
    batch_class_centers,
    intra_class_compactness_loss,
    topology_consistency_loss,
)
from main import parse_args, validate_args


class FeatureModel(nn.Module):
    feature_branch = "fused"

    def __init__(self, cosine: bool = True) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.head = CosineClassifier(4, 3) if cosine else nn.Linear(4, 3)

    @property
    def backbone(self) -> nn.Module:
        return self.encoder

    @property
    def feature_dim(self) -> int:
        return 4

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs.squeeze(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))

    def disable_encoder_finetuning(self) -> None:
        self.backbone.requires_grad_(False)

    def enable_encoder_finetuning(self) -> None:
        self.backbone.requires_grad_(True)


def make_adapter(model=None, **options) -> GOLDTCAAdapter:
    source_prototypes = options.pop("gold_source_prototypes", None)
    return build_adapter(
        "gold_tca",
        model or FeatureModel(),
        torch.device("cpu"),
        1e-3,
        3,
        gold_source_prototypes=source_prototypes,
        gold_conf_thresh=0.0,
        gold_augmentation_noise_std=0.0,
        gold_augmentation_scale_std=0.0,
        **options,
    )


def test_gold_tca_uses_frozen_classifier_topology_and_adapted_centers() -> None:
    torch.manual_seed(12)
    adapter = make_adapter(gold_update_every=1)
    classifier_before = copy.deepcopy(adapter.model.head.state_dict())
    normalized_weights = F.normalize(classifier_before["weight"], dim=1)
    torch.testing.assert_close(
        adapter.source_topology, normalized_weights @ normalized_weights.T
    )
    torch.testing.assert_close(adapter.agop, normalized_weights.T @ normalized_weights)

    classifier_parameter_ids = {
        id(parameter) for parameter in adapter.model.head.parameters()
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in adapter.optimizer.param_groups
        for parameter in group["params"]
    }
    inputs = torch.randn(12, 1, 4)
    with torch.no_grad():
        adapter.adapter_scale.copy_(torch.tensor([0.5, -0.25, 0.75, -0.5]))
        features = adapter.model.forward_features(inputs)
        adapted = adapter.apply_adapter(features)
        normalized = F.normalize(adapted, dim=1)
        assert not torch.allclose(normalized, F.normalize(features, dim=1))
        pseudo_labels = adapter.model.head(adapted).argmax(dim=1)
        expected_centers, _, present = batch_class_centers(
            normalized, pseudo_labels, 3
        )

    outputs = adapter(inputs)
    assert outputs.shape == (12, 3)
    assert torch.isfinite(outputs).all()
    assert classifier_parameter_ids.isdisjoint(optimizer_parameter_ids)
    assert not any(p.requires_grad for p in adapter.model.head.parameters())
    for name, value in classifier_before.items():
        torch.testing.assert_close(adapter.model.head.state_dict()[name], value)
    torch.testing.assert_close(
        adapter.topology_state.historical_class_centers[present],
        expected_centers[present],
    )

    statistics = adapter.statistics()
    assert statistics["classifier_frozen"] is True
    assert statistics["agop_initialization"] == "classifier_weights"
    assert statistics["source_prototype_classes"] == 0
    assert statistics["topology_updates"] == 1
    assert statistics["semantic_anchor"] == "frozen_cosine_classifier_weight"
    assert statistics["dynamic_agop_source"] == "reliable_target_samples"
    assert statistics["source_topology_reference"] == (
        "frozen_cosine_classifier_weight_gram"
    )
    assert statistics["last_total_loss"] == pytest.approx(
        statistics["last_gold_loss"]
        + statistics["topology_weight"]
        * (
            statistics["last_topology_consistency_loss"]
            + statistics["last_intra_class_loss"]
        )
    )

    state_before = copy.deepcopy(adapter.state_dict())
    statistics_before = adapter.statistics()
    assert torch.isfinite(adapter.predict(inputs)).all()
    assert adapter.statistics() == statistics_before
    for name, value in state_before.items():
        torch.testing.assert_close(adapter.state_dict()[name], value)


def test_missing_class_retains_historical_center() -> None:
    adapter = make_adapter(gold_tca_centroid_momentum=0.5)
    first_features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        requires_grad=True,
    )
    first_logits = torch.tensor(
        [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
    )
    adapter.adaptation_regularizer(first_features, first_logits, first_features.sum())
    class_one = adapter.topology_state.historical_class_centers[1].clone()

    second_features = torch.tensor(
        [[1.0, 0.2, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]], requires_grad=True
    )
    second_logits = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    regularizer = adapter.adaptation_regularizer(
        second_features, second_logits, second_features.sum()
    )
    torch.testing.assert_close(
        adapter.topology_state.historical_class_centers[1], class_one
    )
    assert adapter.topology_state.seen_class_mask.tolist() == [True, True, False]
    regularizer.backward()
    assert torch.isfinite(second_features.grad).all()


def test_compactness_and_topology_consistency_losses() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    collapsed = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    spread = collapsed.clone()
    spread[1] = torch.tensor([0.0, 1.0])
    assert intra_class_compactness_loss(collapsed, labels, 2) == 0
    assert intra_class_compactness_loss(spread, labels, 2) > 0

    source = F.normalize(
        torch.tensor([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.25, 0.5, 1.0]]),
        dim=1,
    )
    source_topology = source @ source.T
    valid = torch.tensor([True, True, True])
    torch.testing.assert_close(
        topology_consistency_loss(source, source_topology, valid), torch.tensor(0.0)
    )
    distorted = source.clone()
    distorted[2] = torch.tensor([1.0, 0.0, 0.0])
    distorted.requires_grad_(True)
    loss = topology_consistency_loss(distorted, source_topology, valid)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(distorted.grad).all()


def test_classifier_weights_define_anchors_and_dynamic_target_agop() -> None:
    adapter = make_adapter()
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 4.0]])
    anchors = adapter.contrastive_anchors(torch.randn(3, 4), logits)
    weights = F.normalize(adapter.model.head.weight, dim=1)
    torch.testing.assert_close(anchors, weights)

    features = torch.randn(5, 4)
    feature_logits = adapter.model.head(features)
    selected = features.detach().requires_grad_(True)
    gradients = torch.autograd.grad(
        adapter.model.head(selected).amax(1).sum(), selected
    )[0]
    torch.testing.assert_close(
        adapter.batch_agop(features, feature_logits),
        gradients.T @ gradients / selected.shape[0],
    )
    assert adapter.prototype_state is None


def test_gold_tca_rejects_invalid_model_or_mode() -> None:
    with pytest.raises(ValueError, match="cosine classifier"):
        make_adapter(FeatureModel(cosine=False))
    with pytest.raises(ValueError, match="prototypes"):
        make_adapter(gold_tca_agop_init="source_prototypes")
    with pytest.raises(ValueError, match="classifier scope"):
        make_adapter(parameter_scope="classifier")
    assert resolve_adaptation_parameter_scope("gold_tca", "default") == "all"

    for arguments in (
        ["--method", "gold_tca", "--classifier", "linear"],
        [
            "--method", "gold_tca", "--classifier", "cosine",
            "--gold_tca_lambda_topology", "-1",
        ],
        [
            "--method", "gold_tca", "--classifier", "cosine",
            "--gold_tca_centroid_momentum", "1",
        ],
    ):
        with pytest.raises(ValueError):
            validate_args(parse_args(arguments))


def test_all_profiles_share_matching_cosine_source_checkpoint() -> None:
    profiles = list(Path("configs").glob("*/*/gold_tca.yaml"))
    assert len(profiles) == 21
    for profile in profiles:
        args = parse_args(["--config", str(profile)])
        source = parse_args(["--config", str(profile.with_name("source_cosine.yaml"))])
        validate_args(args)
        validate_args(source)
        assert args.method == "gold_tca"
        assert args.gold_tca_agop_init == "classifier_weights"
        assert args.classifier == source.classifier == "cosine"
        assert args.adaptation_parameter_scope == "all"
        assert args.source_checkpoint.endswith(
            f"/{source.run_name}-seed{source.seed}/source_checkpoint"
        )

    prototype_profiles = list(Path("configs").glob("*/*/gold_tca_prototype.yaml"))
    assert len(prototype_profiles) == 21
    for profile in prototype_profiles:
        args = parse_args(["--config", str(profile)])
        default = parse_args(["--config", str(profile.with_name("gold_tca.yaml"))])
        validate_args(args)
        assert args.gold_tca_agop_init == "source_prototypes"
        assert args.source_checkpoint == default.source_checkpoint


def test_prototype_mode_uses_separate_persistent_ema_memory() -> None:
    source_prototypes = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]
    )
    adapter = make_adapter(
        gold_tca_agop_init="source_prototypes",
        gold_source_prototypes=source_prototypes,
        gold_tca_centroid_momentum=0.5,
    )
    assert adapter.prototype_memory.data_ptr() != adapter.source_prototypes.data_ptr()
    assert adapter.prototype_state.seen_class_mask.tolist() == [True, True, True]

    first_features = torch.tensor(
        [[4.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]],
        requires_grad=True,
    )
    first_logits = torch.tensor(
        [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
    )
    adapter.adaptation_regularizer(first_features, first_logits, first_features.sum())
    expected = torch.tensor(
        [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]
    )
    torch.testing.assert_close(adapter.prototype_memory, expected)

    second_features = torch.tensor(
        [[6.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    second_logits = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    adapter.adaptation_regularizer(
        second_features, second_logits, second_features.sum()
    )
    expected[0, 0] = 4.5
    torch.testing.assert_close(adapter.prototype_memory, expected)
    torch.testing.assert_close(adapter.source_prototypes, source_prototypes)
    assert adapter.prototype_updates == 2

    anchors = adapter.contrastive_anchors(second_features, second_logits)
    torch.testing.assert_close(anchors, adapter.prototype_memory[[0, 0]])
    torch.testing.assert_close(
        adapter.batch_agop(second_features, second_logits),
        adapter.prototype_agop(adapter.prototype_memory),
    )
    assert adapter.statistics()["dynamic_agop_source"] == "persistent_prototype_memory"


def test_predict_does_not_modify_prototype_mode_state() -> None:
    adapter = make_adapter(
        gold_tca_agop_init="source_prototypes",
        gold_source_prototypes=torch.randn(3, 4),
    )
    inputs = torch.randn(5, 1, 4)
    adapter(inputs)
    state_before = copy.deepcopy(adapter.state_dict())
    statistics_before = adapter.statistics()
    assert torch.isfinite(adapter.predict(inputs)).all()
    assert adapter.statistics() == statistics_before
    for name, value in state_before.items():
        torch.testing.assert_close(adapter.state_dict()[name], value)


def test_default_runner_does_not_build_source_prototypes(tmp_path, monkeypatch) -> None:
    spec = DatasetSpec(
        "gold_tca_toy",
        "gold_tca_toy.mat",
        num_classes=3,
        num_domains=2,
        samples_per_class=3,
        default_sequence=(0, 1),
        description="GOLD-TCA runner smoke test",
    )
    signals = np.random.RandomState(7).randn(spec.expected_samples, 4).astype(
        "float32"
    )
    labels = np.tile(np.repeat(np.arange(3), 3), 2)
    sio.savemat(tmp_path / spec.filename, {"data": signals, "label": labels})
    monkeypatch.setitem(DATASETS, spec.name, spec)
    foundation_config = SimpleNamespace(
        architecture={},
        input_length=4,
        input_normalization="none",
        to_dict=lambda: {},
    )
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda *args, **options: (FeatureModel(), foundation_config),
    )

    def fail_prototype_build(*args, **options):
        raise AssertionError(
            "classifier-weight GOLD-TCA must not build source prototypes"
        )

    monkeypatch.setattr("fmctta.gold.build_source_prototypes", fail_prototype_build)
    summary = runner.run_experiment(
        parse_args(
            [
                "--dataset", spec.name,
                "--data_root", str(tmp_path),
                "--output_dir", str(tmp_path / "outputs"),
                "--method", "gold_tca",
                "--classifier", "cosine",
                "--normalization", "none",
                "--source_domains", "0",
                "--domains", "1",
                "--source_epochs", "1",
                "--source_batch_size", "3",
                "--eval_batch_size", "3",
                "--stream_batch_size", "3",
                "--gold_top_r", "4",
                "--gold_projection_dim", "4",
                "--gold_conf_thresh", "0",
                "--device", "cpu",
                "--run_name", "smoke",
            ]
        )
    )
    assert summary["adapter"]["classifier_frozen"] is True
    assert summary["adapter"]["agop_initialization"] == "classifier_weights"
    assert summary["adapter"]["source_prototype_classes"] == 0
    assert summary["adapter"]["topology_updates"] == 3
