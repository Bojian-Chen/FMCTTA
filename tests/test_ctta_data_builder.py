import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ctta_data_builder import core
from ctta_data_builder.build import build_parser, _normalize_legacy_dashed_options
from ctta_data_builder.core import (
    BuildConfig,
    _selection_indices,
    build_dataset,
    config_from_mapping,
)
from ctta_data_builder.datasets import (
    DATASETS,
    DatasetAdapter,
    DomainDefinition,
    WindowBatch,
    _window_signal,
)
from fmctta.config import dataset_spec_from_manifest
from fmctta.data import FaultDataModule


def test_builder_cli_uses_underscore_options_and_accepts_legacy_dash_aliases() -> None:
    parser = build_parser()
    canonical = parser.parse_args(["--samples_per_class", "7", "--validate_only"])
    legacy = parser.parse_args(
        _normalize_legacy_dashed_options(
            ["--samples-per-class", "8", "--validate-only"], parser
        )
    )

    assert canonical.samples_per_class == 7
    assert canonical.validate_only is True
    assert legacy.samples_per_class == 8
    assert legacy.validate_only is True
    assert "--samples_per_class" in parser.format_help()
    assert "--samples-per-class" not in parser.format_help()


def test_builder_supports_pu_wt_hust_and_mcc5() -> None:
    assert {
        "PU",
        "WT",
        "HUST-Gearbox",
        "MCC5-Gearbox",
        "MCC5-Motor",
        "SDUST-Bearing",
        "SDUST-Gear",
    }.issubset(DATASETS)


def test_native_rate_windowing_has_no_overlap_or_resampling() -> None:
    signal = np.arange(2 * 25, dtype=np.float32).reshape(2, 25)
    batch = _window_signal(signal, 100, "recording", window_size_points=10)

    assert batch.values.shape == (2, 2, 10)
    assert np.array_equal(batch.values[0], signal[:, :10])
    assert np.array_equal(batch.values[1], signal[:, 10:20])
    assert batch.start_seconds.tolist() == [0.0, 0.1]


def test_configured_class_and_domain_order_is_retained(tmp_path: Path) -> None:
    config = config_from_mapping(
        {
            "dataset": "PU",
            "raw_root": str(tmp_path),
            "output": "custom.mat",
            "classes": ["KI14", "K001"],
            "domains": {
                "0": "N15_M01_F10",
                "1": "N09_M07_F10",
            },
            "samples_per_class": 5,
        },
        tmp_path,
    )

    assert config.classes == ("KI14", "K001")
    assert config.domains == ("N15_M01_F10", "N09_M07_F10")
    assert config.output == tmp_path / "custom.mat"
    assert _selection_indices(11, 5, "uniform").tolist() == [0, 2, 5, 7, 10]


def test_generated_manifest_loads_as_dynamic_fmctta_spec(tmp_path: Path) -> None:
    manifest = tmp_path / "custom.json"
    manifest.write_text(
        json.dumps(
            {
                "fmctta_spec": {
                    "name": "custom",
                    "filename": "custom.mat",
                    "num_classes": 2,
                    "num_domains": 3,
                    "samples_per_class": 5,
                    "default_sequence": [0, 1, 2],
                    "description": "test",
                    "num_channels": 3,
                    "default_stream_order": "interleaved",
                    "domain_names": ["source", "target-a", "target-b"],
                }
            }
        ),
        encoding="utf-8",
    )

    spec = dataset_spec_from_manifest(manifest)
    assert spec.name == "custom"
    assert spec.expected_samples == 30
    assert spec.num_channels == 3
    assert spec.domain_name(2) == "target-b"


def test_build_reindexes_in_configured_order(tmp_path: Path, monkeypatch) -> None:
    class FakeAdapter(DatasetAdapter):
        name = "fake"
        sampling_frequency = 100
        channel_names = ("x",)
        class_names = ("a", "b")
        domains = (
            DomainDefinition("d0", "d0"),
            DomainDefinition("d1", "d1"),
        )

        def validate_selection(self, raw_root, classes, domains):
            return {}

        def load_windows(
            self, raw_root, class_name, domain_name, window_size_points
        ):
            value = self.domain_names.index(domain_name) * 10
            value += self.class_names.index(class_name)
            values = np.full(
                (128, 1, window_size_points), value, dtype=np.float32
            )
            return WindowBatch(
                values,
                np.arange(128, dtype=np.float64) / 10,
                [f"{domain_name}-{class_name}"] * 128,
            )

    monkeypatch.setitem(core.DATASETS, "fake", FakeAdapter())
    config = BuildConfig(
        dataset="fake",
        raw_root=tmp_path,
        output=tmp_path / "ordered.mat",
        classes=("b", "a"),
        domains=("d1", "d0"),
        samples_per_class=120,
        window_size_points=10,
        name="ordered",
    )

    result = build_dataset(config)
    content = loadmat(result.mat_path)
    assert content["data"].shape == (480, 1, 10)
    assert content["label"][:120].reshape(-1).tolist() == [0] * 120
    assert content["label"][120:240].reshape(-1).tolist() == [1] * 120
    assert content["domain"][:240].reshape(-1).tolist() == [0] * 240
    assert content["domain"][240:].reshape(-1).tolist() == [1] * 240
    assert content["original_label"][:120].reshape(-1).tolist() == [1] * 120
    assert content["original_domain"][:240].reshape(-1).tolist() == [1] * 240
    assert result.manifest["domain_labels"] == {"0": "d1", "1": "d0"}
    assert result.manifest["domain_sequence_defined_at_runtime"] is True
    assert "source_domain" not in result.manifest

    spec = dataset_spec_from_manifest(result.manifest_path)
    data = FaultDataModule(spec, tmp_path, 0, (1,), normalization="none")
    assert spec.samples_per_class == 120
    assert len(data.source.train) == 192
    assert len(data.source.validation) == 48
    assert len(data.target_dataset(1)) == 240
