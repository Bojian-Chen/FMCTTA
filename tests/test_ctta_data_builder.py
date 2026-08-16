import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ctta_data_builder import core
from ctta_data_builder.core import (
    BuildConfig,
    _selection_indices,
    build_dataset,
    config_from_mapping,
)
from ctta_data_builder.datasets import (
    DatasetAdapter,
    DomainDefinition,
    WindowBatch,
    _window_signal,
)
from fmctta.config import dataset_spec_from_manifest


def test_native_rate_windowing_has_no_overlap_or_resampling() -> None:
    signal = np.arange(2 * 25, dtype=np.float32).reshape(2, 25)
    batch = _window_signal(signal, 100, "recording")

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
            "domains": ["N15_M01_F10", "N09_M07_F10"],
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

        def load_windows(self, raw_root, class_name, domain_name):
            value = self.domain_names.index(domain_name) * 10
            value += self.class_names.index(class_name)
            values = np.full((4, 1, 10), value, dtype=np.float32)
            return WindowBatch(
                values,
                np.arange(4, dtype=np.float64) / 10,
                [f"{domain_name}-{class_name}"] * 4,
            )

    monkeypatch.setitem(core.DATASETS, "fake", FakeAdapter())
    config = BuildConfig(
        dataset="fake",
        raw_root=tmp_path,
        output=tmp_path / "ordered.mat",
        classes=("b", "a"),
        domains=("d1", "d0"),
        samples_per_class=2,
        name="ordered",
    )

    result = build_dataset(config)
    content = loadmat(result.mat_path)
    assert content["data"].shape == (8, 1, 10)
    assert content["label"].reshape(-1).tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert content["domain"].reshape(-1).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert content["original_label"].reshape(-1).tolist() == [1, 1, 0, 0] * 2
    assert content["original_domain"].reshape(-1).tolist() == [1] * 4 + [0] * 4
