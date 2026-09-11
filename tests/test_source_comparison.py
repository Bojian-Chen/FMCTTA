import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fmctta import runner
from fmctta.config import DATASETS, DatasetSpec
from fmctta.data import FaultDataModule
from fmctta.metrics import relative_improvement
from fmctta.tca import TCAAdapter
from main import parse_args


class SmallModel(nn.Module):
    feature_branch = "fused"
    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.head = nn.Linear(4, 2)

    @property
    def backbone(self):
        return self.encoder

    def forward_features(self, inputs):
        return self.encoder(inputs.squeeze(1))

    def forward(self, inputs):
        return self.head(self.forward_features(inputs))


@pytest.fixture
def toy_setup(tmp_path, monkeypatch):
    spec = DatasetSpec(
        "comparison_toy",
        "toy.mat",
        2,
        4,
        6,
        (0, 1, 2, 3),
        "test",
        domain_names=("source", "target_a", "target_b", "target_c"),
    )
    signals = np.random.RandomState(5).randn(spec.expected_samples, 4).astype("float32")
    labels = np.tile(np.repeat(np.arange(2), 6), 4)
    sio.savemat(tmp_path / spec.filename, {"data": signals, "label": labels})
    monkeypatch.setitem(DATASETS, spec.name, spec)
    config = SimpleNamespace(
        architecture={},
        input_length=4,
        input_normalization="none",
        to_dict=lambda: {},
    )
    monkeypatch.setattr(runner, "build_model", lambda *a, **kw: (SmallModel(), config))
    common = [
        "--dataset",
        spec.name,
        "--data_root",
        str(tmp_path),
        "--output_dir",
        str(tmp_path / "outputs"),
        "--device",
        "cpu",
        "--source_domains",
        "0",
        "--source_epochs",
        "1",
        "--source_batch_size",
        "4",
        "--source_backbone_lr_scale",
        "0",
        "--stream_batch_size",
        "3",
        "--eval_batch_size",
        "4",
    ]
    # Exercise actual training, checkpoint I/O and adaptation with a small model.
    monkeypatch.setattr(
        SmallModel,
        "disable_encoder_finetuning",
        lambda self: self.encoder.requires_grad_(False),
        raising=False,
    )
    return spec, common, tmp_path


def test_relative_improvement_units_and_zero_denominators():
    baseline = {"accuracy": 50.0, "balanced_accuracy": 40.0, "macro_f1": 0.0}
    current = {"accuracy": 60.0, "balanced_accuracy": 30.0, "macro_f1": 10.0}
    gains = relative_improvement(current, baseline)
    assert gains["accuracy_gain_pp"] == 10
    assert gains["accuracy_relative_gain_pct"] == 20
    assert gains["balanced_accuracy_relative_gain_pct"] == -25
    assert gains["macro_f1_relative_gain_pct"] is None
    assert gains["error_reduction_pct"] == 20
    assert (
        relative_improvement(current, dict(baseline, accuracy=100))[
            "error_reduction_pct"
        ]
        is None
    )


def test_source_domain_diagnostics_do_not_include_training_samples(toy_setup):
    spec, _, root = toy_setup
    data = FaultDataModule(
        spec,
        root,
        source_domains=(0, 1),
        target_domains=(3,),
        max_target_per_domain=2,
        normalization="none",
    )
    for domain in (0, 1):
        dataset = data.source_evaluation_dataset(domain)
        assert set(dataset.sample_ids.tolist()).isdisjoint(
            data.source.train.sample_ids.tolist()
        )
        assert set(dataset.sample_ids.tolist()).issubset(
            data.source.validation.sample_ids.tolist()
        )
        assert all(
            spec.domain_offset(domain) <= index < spec.domain_offset(domain) + 12
            for index in dataset.sample_ids.tolist()
        )
    assert len(data.source_evaluation_dataset(2)) == 12
    assert len(data.source_evaluation_dataset(3)) == 12
    assert len(data.target_dataset(3)) == 2
    with pytest.raises(ValueError):
        data.source_evaluation_dataset(4)


def test_source_domains_then_tta_subset_reorder_repetition(toy_setup):
    spec, common, root = toy_setup
    source = runner.run_experiment(
        parse_args(
            common
            + [
                "--method",
                "source",
                "--domains",
                "1,2,3",
                "--run_name",
                "source",
            ]
        )
    )
    source_dir = Path(source["run_dir"])
    report = json.loads((source_dir / "source_evaluation.json").read_text())
    assert set(report["domains"]) == {"0", "1", "2", "3"}
    assert report["domains"]["0"]["evaluation_split"] == "source_validation"
    assert (source_dir / "source_checkpoint" / "source_evaluation.json").is_file()
    assert source["protocol"]["source_evaluation_only"] is True
    assert source["protocol"]["target_stream_processed"] is False
    assert source["protocol"]["configured_target_domains"] == [1, 2, 3]
    assert source["online"] is None
    assert source["source_baseline"] is None
    assert source["adapter"] is None
    assert source["domains"] == []
    assert not (source_dir / "predictions.csv").exists()

    tta = runner.run_experiment(
        parse_args(
            common
            + [
                "--method",
                "tent",
                "--domains",
                "3,1,3",
                "--source_checkpoint",
                str(source_dir),
                "--run_name",
                "tta",
                "--target_samples_per_class",
                "1;2;2",
            ]
        )
    )
    assert tta["protocol"]["target_domains"] == [3, 1, 3]
    assert tta["protocol"]["mode"] == "standard"
    assert tta["protocol"]["single_pass"] is True
    assert tta["target_forgetting"] == {"enabled": False}
    assert Path(tta["run_dir"]).name == "tta-standard-seed2025"
    baseline = tta["source_baseline"]
    assert [row["domain"] for row in baseline["domains"]] == [3, 1, 3]
    assert [row["position"] for row in baseline["domains"]] == [1, 2, 3]
    assert [row["samples"] for row in baseline["domains"]] == [2, 4, 4]
    assert baseline["samples"] == 10
    assert tta["source"]["evaluation"] is None  # Only the selected stream is run.
    output = Path(tta["run_dir"])
    rows = list(csv.DictReader((output / "predictions.csv").open()))
    baseline_rows = list(csv.DictReader((output / "source_baseline.csv").open()))
    assert len(baseline_rows) == 3
    expected_accuracy = (
        100 * sum(int(row["source_correct"]) for row in rows) / len(rows)
    )
    assert baseline["accuracy"] == pytest.approx(expected_accuracy)
    assert tta["online"]["accuracy_gain_pp"] == pytest.approx(
        tta["online"]["accuracy"] - expected_accuracy
    )
    assert tta["online"]["domain_average_accuracy_gain_pp"] == pytest.approx(
        np.mean([row["accuracy_gain_pp"] for row in tta["domains"]])
    )
    assert (
        baseline["domains"][0]["sample_ids_sha256"]
        != baseline["domains"][2]["sample_ids_sha256"]
    )
    for index, domain_row in enumerate(tta["domains"], 1):
        selected = [row for row in rows if int(row["position"]) == index]
        expected = (
            100 * sum(int(row["source_correct"]) for row in selected) / len(selected)
        )
        assert domain_row["source_accuracy"] == pytest.approx(expected)
        assert domain_row["post_adapt_accuracy_gain_pp"] == pytest.approx(
            domain_row["post_adapt_accuracy"] - expected
        )


def test_source_mode_does_not_build_adapter_or_process_target_stream(
    toy_setup, monkeypatch
):
    _, common, _ = toy_setup

    def fail_adapter_build(*args, **kwargs):
        raise AssertionError("source mode must not build an online adapter")

    monkeypatch.setattr(runner, "build_adapter", fail_adapter_build)
    summary = runner.run_experiment(
        parse_args(
            common
            + [
                "--method",
                "source",
                "--domains",
                "1,2",
                "--protocol_mode",
                "target_forgetting",
                "--target_domain_repeats",
                "0",
                "--allow_target_sample_reuse",
                "--run_name",
                "source-only",
            ]
        )
    )

    assert summary["protocol"]["target_stream_processed"] is False
    assert summary["target_forgetting"] == {
        "enabled": False,
        "reason": "source_evaluation_only",
    }
    assert summary["online"] is None
    assert summary["domains"] == []
    assert not (Path(summary["run_dir"]) / "predictions.csv").exists()


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_target_forgetting_probe_is_side_effect_free():
    torch.manual_seed(9)
    adapter = TCAAdapter(
        SmallModel(),
        torch.device("cpu"),
        learning_rate=1e-3,
        num_classes=2,
        parameter_scope="all",
    )
    inputs = torch.randn(6, 1, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    loader = DataLoader(
        TensorDataset(inputs, labels, torch.arange(6)),
        batch_size=2,
        shuffle=False,
    )
    adapter(inputs)
    adapter.predict(inputs)
    state_before = copy.deepcopy(adapter.state_dict())
    optimizer_before = copy.deepcopy(adapter.optimizer.state_dict())
    statistics_before = copy.deepcopy(adapter.statistics())
    modes_before = [module.training for module in adapter.modules()]

    metrics = runner.evaluate_adapter(
        adapter,
        loader,
        torch.device("cpu"),
        num_classes=2,
    )

    assert metrics["samples"] == 6
    _assert_nested_equal(state_before, adapter.state_dict())
    _assert_nested_equal(optimizer_before, adapter.optimizer.state_dict())
    assert statistics_before == adapter.statistics()
    assert modes_before == [module.training for module in adapter.modules()]


def test_target_forgetting_mode_reports_each_revisit(toy_setup):
    _, common, _ = toy_setup
    summary = runner.run_experiment(
        parse_args(
            common
            + [
                "--method",
                "tent",
                "--domains",
                "1,2",
                "--protocol_mode",
                "target_forgetting",
                "--target_domain_repeats",
                "2",
                "--allow_target_sample_reuse",
                "--run_name",
                "forgetting",
            ]
        )
    )

    forgetting = summary["target_forgetting"]
    assert summary["protocol"]["mode"] == "target_forgetting"
    assert summary["protocol"]["single_pass"] is False
    assert summary["protocol"]["configured_target_domains"] == [1, 2]
    assert summary["protocol"]["target_domains"] == [1, 2, 1, 2]
    assert summary["protocol"]["target_domain_repeats"] == 2
    assert summary["protocol"]["allow_target_sample_reuse"] is True
    assert Path(summary["run_dir"]).name == "forgetting-target_forgetting-seed2025"
    assert forgetting["aggregate"]["repeat_measurements"] == 2
    assert forgetting["sample_policy"] == "same_sample_ids"
    assert [row["occurrence"] for row in forgetting["measurements"]] == [1, 1, 2, 2]
    repeated = [row for row in forgetting["measurements"] if row["is_repeat"]]
    assert [row["domain"] for row in repeated] == [1, 2]
    for row in repeated:
        assert row["accuracy_drop_from_previous_pp"] == pytest.approx(
            row["previous_post_visit_accuracy"] - row["pre_visit_accuracy"]
        )
        assert row["accuracy_drop_from_first_pp"] == pytest.approx(
            row["first_post_visit_accuracy"] - row["pre_visit_accuracy"]
        )
        assert row["accuracy_forgetting_from_best_pp"] == pytest.approx(
            row["best_previous_post_visit_accuracy"] - row["pre_visit_accuracy"]
        )
        first = next(
            item
            for item in forgetting["measurements"]
            if item["domain"] == row["domain"]
        )
        assert row["accuracy_change_from_first_occurrence_pp"] == pytest.approx(
            row["pre_visit_accuracy"] - first["pre_visit_accuracy"]
        )
        assert row["sample_set_sha256"] == first["sample_set_sha256"]
    assert [row["occurrence"] for row in forgetting["occurrence_means"]] == [1, 2]
    output = Path(summary["run_dir"])
    assert (output / "target_forgetting.json").is_file()
    assert len(list(csv.DictReader((output / "target_forgetting.csv").open()))) == 4
    assert len(
        list(
            csv.DictReader(
                (output / "target_forgetting_occurrences.csv").open()
            )
        )
    ) == 2


def test_target_forgetting_can_allocate_distinct_samples(toy_setup):
    _, common, _ = toy_setup
    summary = runner.run_experiment(
        parse_args(
            common
            + [
                "--method",
                "tent",
                "--domains",
                "1",
                "--protocol_mode",
                "target_forgetting",
                "--target_domain_repeats",
                "2",
                "--target_samples_per_class",
                "2",
                "--run_name",
                "forgetting-no-reuse",
            ]
        )
    )

    forgetting = summary["target_forgetting"]
    first, second = forgetting["measurements"]
    assert forgetting["sample_policy"] == "distinct_samples_without_replacement"
    assert forgetting["same_samples_reused_across_occurrences"] is False
    assert first["sample_set_sha256"] != second["sample_set_sha256"]


def test_source_eval_selection_and_rng_isolation(toy_setup):
    spec, common, root = toy_setup
    args = parse_args(common + ["--domains", "1,2"])
    data = FaultDataModule(
        spec, root, source_domain=0, target_domains=(1, 2), normalization="none"
    )
    model = SmallModel().eval()
    before = torch.get_rng_state().clone()
    result = runner.evaluate_source_domains(model, data, [3], args, torch.device("cpu"))
    assert torch.equal(before, torch.get_rng_state())
    assert result[0]["domain"] == 3
    assert runner.resolve_source_eval_domains(
        "sources", spec, (0, 2), (3, 1)
    ) == (0, 2)
    assert runner.resolve_source_eval_domains(
        "all", spec, (0,), (3, 1)
    ) == (0, 1, 2, 3)
    assert runner.resolve_source_eval_domains(
        "targets", spec, (0,), (3, 1, 3)
    ) == (3, 1)
    assert runner.resolve_source_eval_domains(
        "3,1", spec, (0,), (2,)
    ) == (3, 1)
    for invalid in ("4", "1,1", "bad"):
        with pytest.raises(ValueError):
            runner.resolve_source_eval_domains(invalid, spec, (0,), (1,))


def test_source_eval_yaml_list(tmp_path):
    config = tmp_path / "evaluation.yaml"
    config.write_text("protocol:\n  source_eval_domains: [3, 1]\n")
    assert parse_args(["--config", str(config)]).source_eval_domains == "3,1"
