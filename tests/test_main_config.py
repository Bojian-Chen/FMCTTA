from pathlib import Path

import pytest

from main import build_parser, parse_args, validate_args


def test_static_dataset_choices_cover_pu_wt_hust_and_mcc5() -> None:
    parser = build_parser()
    dataset_action = next(
        action for action in parser._actions if action.dest == "dataset"
    )

    assert set(dataset_action.choices) == {
        "PU",
        "WT",
        "HUST",
        "HUST_gear",
        "MCC5_THU_Gearbox",
        "MCC5_THU_Motor",
    }
    assert parse_args([]).dataset == "PU"


def test_cli_uses_underscore_options_and_accepts_legacy_dash_aliases() -> None:
    canonical = parse_args(
        ["--source_epochs", "7", "--source_backbone_lr_scale", "0"]
    )
    legacy = parse_args(["--source-epochs", "8", "--source-backbone-lr-scale=0"])

    assert canonical.source_epochs == 7
    assert canonical.source_backbone_lr_scale == 0
    assert legacy.source_epochs == 8
    assert legacy.source_backbone_lr_scale == 0
    assert "--source_epochs" in build_parser().format_help()
    assert "--source-epochs" not in build_parser().format_help()


def test_cli_supports_phmfd_foundation_models() -> None:
    args = parse_args(
        [
            "--foundation_model",
            "tf_profm",
            "--pretrained_checkpoint",
            "auto",
            "--model_input_normalization",
            "sample_zscore",
        ]
    )

    validate_args(args)
    assert args.foundation_model == "tf_profm"
    assert args.pretrained_checkpoint == "auto"


def test_continual_mae_rejects_non_unifreq_backbones() -> None:
    args = parse_args(
        ["--foundation_model", "unifault", "--method", "continual_mae"]
    )

    with pytest.raises(ValueError, match="requires UniFreq"):
        validate_args(args)


def test_yaml_config_sets_defaults_and_cli_overrides_them(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        """
protocol:
  dataset: PU
  method: tent
model:
  feature_branch: fused
runtime:
  run_name: base-run
""".strip(),
        encoding="utf-8",
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
base: base.yaml
protocol:
  domains: [0, 1, 2, 3, 1, 3, 2, 3]
data:
  source_samples_per_class: [60, 50, 60, 50, 60]
  target_samples_per_class: [20, 10, 15, 20, 10, 15, 20]
  allow_repeated_target_samples: true
model:
  feature_branch: frequency
runtime:
  run_name: yaml-run
""".strip(),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config), "--method", "source"])

    assert args.method == "source"
    assert args.domains == "0,1,2,3,1,3,2,3"
    assert args.source_samples_per_class == "60,50,60,50,60"
    assert args.target_samples_per_class == "20;10;15;20;10;15;20"
    assert args.allow_repeated_target_samples is True
    assert args.feature_branch == "frequency"
    assert args.run_name == "yaml-run"


def test_yaml_source_domains_are_independent_from_target_domains(
    tmp_path: Path,
) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "protocol:\n  source_domains: [0, 2]\n  domains: [1]\n",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.source_domains == "0,2"
    assert args.domains == "1"


def test_yaml_config_rejects_unknown_options(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("model:\n  not_an_option: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown YAML"):
        parse_args(["--config", str(config)])


def test_yaml_config_rejects_base_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base: second.yaml\n", encoding="utf-8")
    second.write_text("base: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        parse_args(["--config", str(first)])


def test_yaml_config_supports_ordered_multiple_bases(tmp_path: Path) -> None:
    dataset_base = tmp_path / "dataset.yaml"
    dataset_base.write_text(
        """
protocol:
  dataset: WT
  method: source
model:
  classifier: mlp
runtime:
  run_name: dataset-base
""".strip(),
        encoding="utf-8",
    )
    method_base = tmp_path / "method.yaml"
    method_base.write_text(
        """
protocol:
  method: tent
model:
  classifier: linear
adaptation:
  adaptation_parameter_scope: layernorm
""".strip(),
        encoding="utf-8",
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
base:
  - dataset.yaml
  - method.yaml
runtime:
  run_name: multi-base
""".strip(),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.dataset == "WT"
    assert args.method == "tent"
    assert args.classifier == "linear"
    assert args.adaptation_parameter_scope == "layernorm"
    assert args.run_name == "multi-base"


@pytest.mark.parametrize(
    "path",
    [
        "configs/HUST_Gearbox/unifreq/note.yaml",
        "configs/PU/unifreq/note.yaml",
        "configs/WT/unifreq/note.yaml",
    ],
)
def test_note_yaml_configs_are_complete(path: str) -> None:
    args = parse_args(["--config", path])

    assert args.method == "note"
    assert args.adaptation_parameter_scope == "layernorm"
    assert args.adaptation_learning_rate == 1e-4
    assert args.note_memory_size == 64
    assert args.note_update_interval == 64
    assert args.note_temperature == 1.0


def test_dataset_yaml_directories_are_grouped_by_foundation_model() -> None:
    datasets = {
        "HUST_Gearbox",
        "MCC5_gearbox",
        "MCC5_motor",
        "PU",
        "SDUST_Bearing",
        "SDUST_Gear",
        "WT",
    }
    unifreq_configs = {
        "base",
        "source",
        "knn",
        "tent",
        "note",
        "eata",
        "sar",
        "cotta",
        "vida",
        "continual_mae",
        "rotta",
        "tca",
        "gold",
        "ttt",
    }

    for dataset in datasets:
        dataset_root = Path("configs") / dataset
        assert {path.name for path in dataset_root.iterdir() if path.is_dir()} == {
            "unifault",
            "tf_profm",
            "unifreq",
        }
        assert not list(dataset_root.glob("*.yaml"))
        assert {
            path.stem for path in (dataset_root / "unifreq").glob("*.yaml")
        } == unifreq_configs


@pytest.mark.parametrize(
    ("family_dir", "foundation_model", "run_tag", "model_size"),
    [
        ("unifault", "unifault", "unifault-small", "small"),
        ("tf_profm", "tf_profm", "tf-profm", "auto"),
    ],
)
def test_phmfd_yaml_matrix_is_complete_and_valid(
    family_dir: str,
    foundation_model: str,
    run_tag: str,
    model_size: str,
) -> None:
    datasets = {
        "HUST_Gearbox": "hust-gearbox",
        "MCC5_gearbox": "mcc5-gearbox",
        "MCC5_motor": "mcc5-motor",
        "PU": "pu",
        "SDUST_Bearing": "sdust-bearing",
        "SDUST_Gear": "sdust-gear",
        "WT": "wt",
    }
    methods = {
        "source",
        "knn",
        "tent",
        "note",
        "eata",
        "sar",
        "cotta",
        "vida",
        "rotta",
        "tca",
        "gold",
        "ttt",
    }

    for dataset_dir, dataset_tag in datasets.items():
        family_root = Path("configs") / dataset_dir / family_dir
        base_args = parse_args(["--config", str(family_root / "base.yaml")])
        assert (family_root / "base.yaml").is_file()
        assert not (family_root / "continual_mae.yaml").exists()
        assert {path.stem for path in family_root.glob("*.yaml")} == {
            "base",
            *methods,
        }

        for method in methods:
            args = parse_args(["--config", str(family_root / f"{method}.yaml")])
            validate_args(args)

            assert args.foundation_model == foundation_model
            assert args.foundation_model_size == model_size
            assert args.pretrained_checkpoint == "auto"
            assert args.model_input_length == 0
            assert args.model_input_normalization == "auto"
            assert args.normalization == base_args.normalization
            assert args.feature_branch == "fused"
            assert args.method == method
            assert args.run_name == f"{dataset_tag}-{run_tag}-{method}"
            if method in {"source", "ttt"}:
                assert args.source_checkpoint == ""
            else:
                assert f"{dataset_tag}-{run_tag}-source-seed2025" in Path(
                    args.source_checkpoint
                ).parts
