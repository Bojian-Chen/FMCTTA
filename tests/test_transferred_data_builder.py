import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from ctta_data_builder import core
from ctta_data_builder.build import (
    _normalize_legacy_dashed_options,
    build_parser,
    main as build_main,
)
from ctta_data_builder.core import BuildConfig, build_dataset, config_from_mapping
from ctta_data_builder.datasets import (
    DEFAULT_WINDOW_SIZE_POINTS,
    DatasetAdapter,
    DomainDefinition,
    MCC5GearboxAdapter,
    MCC5MotorAdapter,
    SDUSTBearingAdapter,
    SDUSTGearAdapter,
    WindowBatch,
    _window_signal,
)
from fmctta.config import dataset_spec_from_manifest
from fmctta.data import FaultDataModule


def test_builder_cli_uses_underscore_and_accepts_legacy_dash() -> None:
    parser = build_parser()
    canonical = parser.parse_args(["--window_size_points", "2048", "--validate_only"])
    legacy = parser.parse_args(
        _normalize_legacy_dashed_options(
            ["--window-size-points", "512", "--validate-only"], parser
        )
    )
    assert canonical.window_size_points == 2048
    assert canonical.validate_only is True
    assert legacy.window_size_points == 512
    assert legacy.validate_only is True
    assert "--window_size_points" in parser.format_help()


def test_builder_catalog_cli_derives_seconds_from_points(capsys) -> None:
    build_main(["--catalog", "PU", "--window-size-points", "2048"])
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["window_size_points"] == 2048
    assert catalog["window_seconds"] == pytest.approx(2048 / 64_000)


def test_mcc5_catalog_marks_all_selectable_source_domains(capsys) -> None:
    build_main(["--catalog", "MCC5-Motor"])
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["source_candidate_domains"] == [
        domain for domain in range(33) if domain % 3 in (0, 1)
    ]
    assert catalog["target_stream_domains"] == list(range(2, 33, 3))
    assert catalog["recommended_source_domains"] == [0, 1]
    assert catalog["domain_catalog"][2]["recording_group"] == catalog[
        "domain_catalog"
    ][0]["recording_group"]


def test_point_windowing_has_no_overlap_or_resampling() -> None:
    signal = np.arange(2 * 25, dtype=np.float32).reshape(2, 25)
    batch = _window_signal(signal, 100, "recording", window_size_points=6)
    assert batch.values.shape == (4, 2, 6)
    assert np.array_equal(batch.values[0], signal[:, :6])
    assert np.array_equal(batch.values[1], signal[:, 6:12])
    assert batch.start_seconds.tolist() == [0.0, 0.06, 0.12, 0.18]


def test_mcc5_adapters_expose_two_source_candidates_per_recording() -> None:
    gearbox = MCC5GearboxAdapter()
    motor = MCC5MotorAdapter()

    assert gearbox.channel_names == (
        "motor_vibration_x",
        "motor_vibration_y",
        "motor_vibration_z",
        "gearbox_vibration_x",
        "gearbox_vibration_y",
        "gearbox_vibration_z",
    )
    assert gearbox.vibration_columns == (2, 3, 4, 5, 6, 7)
    assert gearbox.recommended_source_domains == gearbox.domain_names[:2]
    assert len(gearbox.domains) == 36
    assert gearbox.domains[0].segments_sec == ((10.0, 20.0), (40.0, 50.0))
    assert gearbox.domains[1].segments_sec == ((25.0, 35.0),)
    assert gearbox.condition_profile[-1]["stop_sec"] == 60.0

    assert motor.recommended_source_domains == motor.domain_names[:2]
    assert len(motor.domains) == 33
    assert motor.domains[0].segments_sec == ((10.0, 30.0), (60.0, 80.0))
    assert motor.domains[1].segments_sec == ((35.0, 55.0),)
    assert motor.condition_profile[-1]["stop_sec"] == 90.0

    for adapter in (gearbox, motor):
        assert len(adapter.domains) % 3 == 0
        for offset in range(0, len(adapter.domains), 3):
            steady_1, steady_2, stream = adapter.domains[offset : offset + 3]
            assert steady_1.role == steady_2.role == "source_candidate"
            assert stream.role == "target_stream"
            assert not stream.segments_sec
            assert steady_1.raw_key == steady_2.raw_key == stream.raw_key
            assert (
                steady_1.recording_group
                == steady_2.recording_group
                == stream.recording_group
            )
            assert steady_1.segments_sec == adapter.domains[0].segments_sec
            assert steady_2.segments_sec == adapter.domains[1].segments_sec


@pytest.mark.parametrize(
    ("adapter", "class_name", "domain_name", "relative_path"),
    [
        (
            SDUSTBearingAdapter(),
            "NC",
            "1000rpm_60N",
            Path("轴承数据集/NC/NC 1000 60.mat"),
        ),
        (
            SDUSTGearAdapter(),
            "太阳断裂",
            "1500rpm_0.5A",
            Path("齿轮数据集/太阳断裂/sunfracture 1500 0.5A 1.mat"),
        ),
    ],
)
def test_sdust_adapters_read_all_six_channels_without_resampling(
    tmp_path: Path,
    adapter: DatasetAdapter,
    class_name: str,
    domain_name: str,
    relative_path: Path,
) -> None:
    raw_root = tmp_path / "SUDST"
    recording = raw_root / relative_path
    recording.parent.mkdir(parents=True)
    values = np.arange(8 * 6, dtype=np.float64).reshape(8, 6)
    savemat(recording, {"Signal": {"y_values": {"values": values}}})

    batch = adapter.load_windows(tmp_path, class_name, domain_name, 4)

    assert batch.values.shape == (2, 6, 4)
    assert batch.values[0].tolist() == [
        [0.0, 6.0, 12.0, 18.0],
        [1.0, 7.0, 13.0, 19.0],
        [2.0, 8.0, 14.0, 20.0],
        [3.0, 9.0, 15.0, 21.0],
        [4.0, 10.0, 16.0, 22.0],
        [5.0, 11.0, 17.0, 23.0],
    ]
    assert batch.start_seconds.tolist() == [0.0, 4 / 25_600]
    assert batch.recording_names == [
        f"SUDST/{relative_path}",
        f"SUDST/{relative_path}",
    ]


def test_builder_default_is_1024_points(tmp_path: Path) -> None:
    config = config_from_mapping(
        {
            "dataset": "PU",
            "raw_root": str(tmp_path),
            "output": "custom.mat",
            "classes": ["K001", "KI14"],
            "domains": ["N15_M07_F10", "N09_M07_F10"],
            "samples_per_class": 5,
        },
        tmp_path,
    )
    assert config.window_size_points == DEFAULT_WINDOW_SIZE_POINTS == 1024


def test_large_arrays_use_hdf5_backed_mat_storage(tmp_path: Path, monkeypatch) -> None:
    h5py = pytest.importorskip("h5py")
    monkeypatch.setattr(core, "MATLAB_V5_MAX_MATRIX_BYTES", 1)
    output, manifest = core._save_outputs(
        tmp_path / "large.mat",
        {
            "data": np.arange(12, dtype=np.float32).reshape(2, 1, 6),
            "label": np.asarray([[0], [1]], dtype=np.int64),
            "channels": np.asarray([["sensor"]], dtype=object),
        },
        {"storage_format": "hdf5"},
        overwrite=False,
    )

    assert h5py.is_hdf5(output)
    with h5py.File(output, "r") as handle:
        assert handle["data"].shape == (2, 1, 6)
        assert handle["channels"].asstr()[()].tolist() == [["sensor"]]
    assert json.loads(manifest.read_text(encoding="utf-8"))["storage_format"] == "hdf5"


def test_builder_rejects_nonpositive_point_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="window_size_points"):
        config_from_mapping(
            {
                "dataset": "PU",
                "raw_root": str(tmp_path),
                "output": "custom.mat",
                "window_size_points": 0,
            },
            tmp_path,
        )


def test_builder_rejects_unknown_channel(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid channels"):
        config_from_mapping(
            {
                "dataset": "PU",
                "raw_root": str(tmp_path),
                "output": "custom.mat",
                "channels": ["missing"],
            },
            tmp_path,
        )


def test_build_outputs_octta_manifest_and_point_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeAdapter(DatasetAdapter):
        name = "fake"
        sampling_frequency = 100
        channel_names = ("x", "y")
        class_names = ("a", "b")
        domains = (DomainDefinition("d0", "d0"), DomainDefinition("d1", "d1"))

        def validate_selection(self, raw_root, classes, domains):
            return {}

        def load_windows(
            self, raw_root, class_name, domain_name, window_size_points,
            stride_points=None,
        ):
            value = self.domain_names.index(domain_name) * 10
            value += self.class_names.index(class_name)
            values = np.empty((8, 2, window_size_points), dtype=np.float32)
            values[:, 0] = value
            values[:, 1] = value + 100
            return WindowBatch(
                values,
                np.arange(8, dtype=np.float64) * window_size_points / 100,
                [f"{domain_name}-{class_name}"] * 8,
            )

    monkeypatch.setitem(core.DATASETS, "fake", FakeAdapter())
    config = BuildConfig(
        dataset="fake",
        raw_root=tmp_path,
        output=tmp_path / "ordered.mat",
        classes=("b", "a"),
        domains=("d1", "d0"),
        channels=("y",),
        samples_per_class=5,
        window_size_points=12,
        name="ordered",
    )
    result = build_dataset(config)
    content = loadmat(result.mat_path)
    assert content["data"].shape == (20, 1, 12)
    assert content["label"][:5].reshape(-1).tolist() == [0] * 5
    assert content["label"][5:10].reshape(-1).tolist() == [1] * 5
    assert content["domain"][:10].reshape(-1).tolist() == [0] * 10
    assert content["domain"][10:].reshape(-1).tolist() == [1] * 10
    assert content["original_label"][:5].reshape(-1).tolist() == [1] * 5
    assert content["original_domain"][:10].reshape(-1).tolist() == [1] * 10
    assert content["window_size"].item() == 12
    assert content["window_seconds"].item() == pytest.approx(0.12)
    assert content["sampling_frequency"].item() == 100
    assert content["stride_points"].item() == 12
    assert content["data"][0, 0, 0] == 111
    assert content["window_start_sec"].shape == (20, 1)
    assert result.manifest["window_size_points"] == 12
    assert result.manifest["stride_points"] == 12
    assert result.manifest["channels"] == ["y"]
    assert result.manifest["num_channels"] == 1
    assert result.manifest["window_seconds"] == pytest.approx(0.12)
    assert result.manifest["domain_labels"] == {"0": "d1", "1": "d0"}
    assert result.manifest["fmctta_spec"]["domain_groups"] == ["d1", "d0"]
    assert result.manifest["domain_sequence_defined_at_runtime"] is True
    assert result.manifest["recordings"] == [
        "d1-b",
        "d1-a",
        "d0-b",
        "d0-a",
    ]
    assert "fmctta_spec" in result.manifest

    spec = dataset_spec_from_manifest(result.manifest_path)
    data = FaultDataModule(
        spec,
        tmp_path,
        source_domain=0,
        target_domains=(1,),
        representation="time",
        normalization="none",
    )
    assert len(data.source.train) == 8
    assert len(data.source.validation) == 2
    assert len(data.target_dataset(1)) == 10


def test_full_window_build_preserves_variable_domain_lengths_and_time_order(
    tmp_path: Path, monkeypatch
) -> None:
    h5py = pytest.importorskip("h5py")

    class VariableAdapter(DatasetAdapter):
        name = "variable"
        sampling_frequency = 10
        channel_names = ("x",)
        class_names = ("a", "b")
        domains = (
            DomainDefinition("source", "source"),
            DomainDefinition("stream", "stream"),
        )

        def validate_selection(self, raw_root, classes, domains):
            return {}

        def load_windows(
            self, raw_root, class_name, domain_name, window_size_points,
            stride_points=None,
        ):
            count = 3 if domain_name == "source" else 5
            class_id = self.class_names.index(class_name)
            values = np.arange(count, dtype=np.float32).reshape(-1, 1, 1)
            values = np.repeat(values + 10 * class_id, window_size_points, axis=2)
            return WindowBatch(
                values,
                np.arange(count, dtype=np.float64),
                [f"{domain_name}-{class_name}"] * count,
            )

    monkeypatch.setitem(core.DATASETS, "variable", VariableAdapter())
    config = BuildConfig(
        dataset="variable",
        raw_root=tmp_path,
        output=tmp_path / "all.mat",
        classes=("a", "b"),
        domains=("source", "stream"),
        samples_per_class=None,
        window_size_points=4,
        target_stream_order="interleaved",
        name="all-windows",
    )

    result = build_dataset(config)

    assert h5py.is_hdf5(result.mat_path)
    assert result.data_shape == (16, 1, 4)
    assert result.manifest["selection"] == "all"
    assert result.manifest["samples_per_class_per_domain"] == [3, 5]
    assert result.manifest["fmctta_spec"][
        "samples_per_class_by_domain"
    ] == [3, 5]

    spec = dataset_spec_from_manifest(result.manifest_path)
    data = FaultDataModule(
        spec,
        tmp_path,
        source_domain=0,
        target_domains=(1,),
        normalization="none",
        stream_order="interleaved",
    )
    target = data.target_dataset(1)
    assert target.sample_ids.tolist() == [6, 11, 7, 12, 8, 13, 9, 14, 10, 15]
    assert target.signals[:, 0, 0].tolist() == [
        0,
        10,
        1,
        11,
        2,
        12,
        3,
        13,
        4,
        14,
    ]

