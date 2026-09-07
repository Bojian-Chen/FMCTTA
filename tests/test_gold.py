import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fmctta.adapters import build_adapter, resolve_adaptation_parameter_scope
from fmctta.gold import build_source_prototypes, multiview_contrastive_loss
from fmctta.model import build_classifier
from main import parse_args, validate_args


class FeatureModel(nn.Module):
    def __init__(self, classifier="linear"):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.head = build_classifier(classifier, 4, 3, 0.0)

    def forward_features(self, inputs):
        return self.backbone(inputs.squeeze(1))

    def forward(self, inputs):
        return self.head(self.forward_features(inputs))


def make_adapter(model=None, **kwargs):
    return build_adapter(
        "gold",
        model or FeatureModel(),
        torch.device("cpu"),
        1e-3,
        3,
        gold_source_prototypes=torch.randn(3, 4),
        **kwargs,
    )


def test_source_prototypes_are_class_means_and_require_all_classes():
    model = FeatureModel()
    inputs = torch.randn(6, 1, 4)
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    loader = DataLoader(TensorDataset(inputs, labels, torch.arange(6)), batch_size=2)
    prototypes = build_source_prototypes(model, loader, torch.device("cpu"), 3)
    features = model.forward_features(inputs).detach()
    assert torch.allclose(
        prototypes,
        torch.stack([features[labels == label].mean(0) for label in range(3)]),
        atol=1e-6,
    )
    assert not prototypes.requires_grad
    with pytest.raises(ValueError, match="every class"):
        build_source_prototypes(model, loader, torch.device("cpu"), 4)


def test_linear_agop_matches_classifier_gradients_and_confidence_filter():
    adapter = make_adapter(gold_conf_thresh=0.0)
    weight = adapter.model.head.weight.detach()
    assert torch.allclose(adapter.agop, weight.T @ weight)
    features = torch.randn(7, 4)
    logits = adapter.model.head(features)
    gradients = weight[logits.argmax(1)]
    expected = gradients.T @ gradients / 7
    assert torch.allclose(adapter.batch_agop(features, logits), expected)
    adapter.conf_thresh = 1.0
    assert adapter.batch_agop(features, logits) is None
    adapter.conf_thresh = 0.0
    assert adapter.batch_agop(features[:1], logits[:1]) is None


@pytest.mark.parametrize("classifier", ["linear", "mlp", "cosine"])
def test_gold_updates_and_predict_does_not_mutate_state(classifier):
    torch.manual_seed(4)
    model = FeatureModel(classifier)
    original = copy.deepcopy(model.state_dict())
    adapter = make_adapter(
        model, gold_top_r=64, gold_update_every=1, gold_conf_thresh=0.0, steps=2
    )
    assert resolve_adaptation_parameter_scope("gold", "default") == "all"
    assert adapter.adapter_scale.shape == (4,)
    assert torch.allclose(adapter.basis.T @ adapter.basis, torch.eye(4), atol=1e-5)
    assert not any(p.requires_grad for p in adapter.teacher.parameters())
    inputs = torch.randn(5, 1, 4)
    with torch.no_grad():
        expected = 2 * model(inputs)
        outputs = adapter(inputs)
    assert torch.allclose(outputs, expected)
    assert not outputs.requires_grad
    assert adapter.updates == adapter.agop_updates == adapter.basis_updates == 2
    assert adapter.adapter_scale.abs().sum() > 0
    assert any(
        not torch.equal(adapter.model.state_dict()[key], value)
        for key, value in original.items()
    )
    assert all(
        torch.equal(model.state_dict()[key], value) for key, value in original.items()
    )
    state = copy.deepcopy(adapter.state_dict())
    stats = adapter.statistics()
    prediction = adapter.predict(inputs)
    assert torch.isfinite(prediction).all()
    assert adapter.statistics() == stats
    assert all(
        torch.equal(value, adapter.state_dict()[key]) for key, value in state.items()
    )
    # A trailing single-sample batch still adapts; it cannot estimate AGOP.
    assert torch.isfinite(adapter(inputs[:1])).all()
    assert adapter.agop_updates == 2


def test_gold_classifier_scope_freezes_backbone_and_teacher_tracks_student():
    torch.manual_seed(0)
    adapter = make_adapter(parameter_scope="classifier", gold_teacher_momentum=0.0)
    before = copy.deepcopy(adapter.model.backbone.state_dict())
    adapter(torch.randn(6, 1, 4))
    assert all(
        torch.equal(value, adapter.model.backbone.state_dict()[key])
        for key, value in before.items()
    )
    assert all(
        torch.equal(p, q)
        for p, q in zip(adapter.teacher.parameters(), adapter.model.parameters())
    )


def test_gold_agop_ema_and_refresh_interval():
    adapter = make_adapter(gold_conf_thresh=0.0, gold_update_every=2, gold_alpha=0.25)
    inputs = torch.randn(5, 1, 4)
    features = adapter.model.forward_features(inputs)
    matrix = adapter.batch_agop(features, adapter.model.head(features))
    expected = 0.75 * adapter.agop.clone() + 0.25 * matrix
    basis = adapter.basis.clone()
    adapter(inputs)
    assert torch.allclose(adapter.agop, expected)
    assert torch.equal(adapter.basis, basis)
    adapter(inputs)
    assert adapter.basis_updates == 1


def test_contrastive_loss_matches_uniform_logits_and_has_finite_gradients():
    views = torch.ones(2, 3, 4, requires_grad=True)
    loss = multiview_contrastive_loss(views, 0.1)
    assert torch.allclose(loss, torch.tensor(5.0).log())
    loss.backward()
    assert torch.isfinite(views.grad).all()


def test_source_warmup_keeps_online_counters_and_subspace_unchanged():
    adapter = make_adapter()
    loader = DataLoader(TensorDataset(torch.randn(3, 1, 4)), batch_size=2)
    before = adapter.basis.clone()
    adapter.warmup(loader, 3)
    assert adapter.warmup_updates == 3
    assert adapter.updates == adapter.batches_seen == adapter.samples_seen == 0
    assert torch.equal(adapter.basis, before)
    assert torch.equal(adapter.adapter_scale, torch.zeros_like(adapter.adapter_scale))


@pytest.mark.parametrize(
    "option,value",
    [
        ("gold_top_r", "0"),
        ("gold_alpha", "1.1"),
        ("gold_conf_thresh", "-1"),
        ("gold_update_every", "0"),
        ("gold_temperature", "0"),
        ("gold_warmup_steps", "-1"),
        ("gold_lambda_cont", "-1"),
        ("gold_teacher_momentum", "1"),
    ],
)
def test_invalid_gold_config_is_rejected(option, value):
    with pytest.raises(ValueError):
        validate_args(parse_args(["--method", "gold", "--" + option, value]))


def test_gold_requires_source_prototypes():
    with pytest.raises(ValueError, match="prototypes"):
        build_adapter("gold", FeatureModel(), torch.device("cpu"), 1e-3, 3)


def test_all_gold_profiles_parse_and_match_source_preparation():
    profiles = list(Path("configs").glob("*/*/gold.yaml"))
    assert len(profiles) == 21
    for profile in profiles:
        args = parse_args(["--config", str(profile)])
        source = parse_args(["--config", str(profile.with_name("source.yaml"))])
        validate_args(args)
        assert args.method == "gold"
        assert args.adaptation_parameter_scope == "all"
        for name in (
            "source_samples_per_class",
            "source_validation_ratio",
            "normalization",
            "classifier",
        ):
            assert getattr(args, name) == getattr(source, name)
