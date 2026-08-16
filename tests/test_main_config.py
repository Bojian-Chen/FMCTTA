from pathlib import Path

import pytest

from main import parse_args


def test_yaml_config_sets_defaults_and_cli_overrides_them(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        """
protocol:
  dataset: PU_Real
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
    assert args.feature_branch == "frequency"
    assert args.run_name == "yaml-run"


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
