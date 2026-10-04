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
    HUSTGearboxAdapter,
    MCC5GearboxAdapter,
    MCC5MotorAdapter,
    SDUSTBearingAdapter,
    SDUSTGearAdapter,
    SKAdapter,
    WindowBatch,
    _window_signal,
)
from fmctta.config import dataset_spec_from_manifest
from fmctta.data import FaultDataModule


@pytest.mark.parametrize("samples_per_class", [None, 3])
def test_sk_build_preserves_overlap_time_and_loads_in_fmctta(
    tmp_path: Path, samples_per_class
) -> None:
    import h5py

    adapter = SKAdapter()
    raw_root = tmp_path / "SK"
    for class_name in ("NC", "IF0.2"):
        for load in (0, 20):
            path = raw_root / class_name / f"500 {load}.mat"
            path.parent.mkdir(parents=True, exist_ok=True)
            values = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
            savemat(path, {"Signal": {"y_values": {"values": values}}})
    result = build_dataset(
        BuildConfig(
            dataset="SK",
            raw_root=raw_root,
            output=tmp_path / "sk.mat",
            classes=("NC", "IF0.2"),
            domains=adapter.domain_names[:2],
            channels=("point3_y",),
            samples_per_class=samples_per_class,
            window_size_points=4,
            stride_points=2,
            target_stream_order="interleaved",
        )
    )
    count = 5 if samples_per_class is None else 3
    offsets = np.arange(0, 10, 2) if samples_per_class is None else np.array([0, 4, 8])
    if samples_per_class is None:
        assert h5py.is_hdf5(result.mat_path)
        with h5py.File(result.mat_path) as handle:
            assert handle.attrs["fmctta_storage_format"] == "hdf5"
            starts = handle["window_start_sec"][:].reshape(-1)
    else:
        starts = loadmat(result.mat_path)["window_start_sec"].reshape(-1)
    np.testing.assert_allclose(starts, np.tile(offsets / 25_600, 4))
    assert result.manifest["stride_points"] == 2
    assert result.manifest["overlap"] is True
    spec = dataset_spec_from_manifest(result.manifest_path)
    data = FaultDataModule(
        spec,
        tmp_path,
        source_domain=0,
        target_domains=(1,),
        normalization="none",
        stream_order=spec.default_stream_order,
    )
    target = data.target_dataset(1)
    assert target.labels.tolist() == [0, 1] * count
    assert target.signals[:, 0, 0].tolist() == np.repeat(offsets * 3 + 1, 2).tolist()


def test_builder_cli_uses_underscore_and_accepts_legacy_dash() -> None:
    parser = build_parser()
    canonical = parser.parse_args(
        [
            "--window_size_points",
            "2048",
            "--stride_points",
            "512",
            "--validate_only",
        ]
    )
    legacy = parser.parse_args(
        _normalize_legacy_dashed_options(
            [
                "--window-size-points",
                "512",
                "--stride-points",
                "128",
                "--validate-only",
            ],
            parser,
        )
    )
    assert canonical.window_size_points == 2048
    assert canonical.stride_points == 512
    assert canonical.validate_only is True
    assert legacy.window_size_points == 512
    assert legacy.stride_points == 128
    assert legacy.validate_only is True
    assert "--window_size_points" in parser.format_help()
    assert "--stride_points" in parser.format_help()


def test_builder_catalog_cli_derives_seconds_from_points(capsys) -> None:
    build_main(
        [
            "--catalog",
            "PU",
            "--window-size-points",
            "2048",
            "--stride-points",
            "512",
        ]
    )
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["window_size_points"] == 2048
    assert catalog["window_seconds"] == pytest.approx(2048 / 64_000)
    assert catalog["stride_points"] == 512
    assert catalog["stride_seconds"] == pytest.approx(512 / 64_000)
    assert catalog["overlap"] is True


@pytest.mark.parametrize("nested_root", [False, True])
def test_hust_variable_speed_recordings_preserve_channels_and_time(
    tmp_path: Path, nested_root: bool
) -> None:
    adapter = HUSTGearboxAdapter()
    root = tmp_path / "HUST_Gearbox" if nested_root else tmp_path
    root.mkdir(exist_ok=True)
    # The fixed-speed marker also exercises both supported raw-root layouts.
    (root / "H_20_0.txt").touch()
    data = np.column_stack(
        (
            np.arange(12) / 25_600,
            np.linspace(0, 40, 12),
            np.arange(12),
            np.arange(12) + 100,
            np.arange(12) + 200,
        )
    )
    for load in range(5):
        for class_name in adapter.class_names:
            path = root / f"{class_name}_L{load}_VS_0_40_0.txt"
            np.savetxt(
                path,
                data,
                header="Title: variable speed\nTime Speed X Y Z",
                comments="",
            )
            domain_name = f"0~40~0Hz_L{load}"
            assert adapter.recording_paths(tmp_path, class_name, domain_name) == [path]
            batch = adapter.load_windows(tmp_path, class_name, domain_name, 6, 3)
            assert batch.values.shape == (3, 3, 6)
            np.testing.assert_array_equal(batch.values[0], data[:6, -3:].T)
            np.testing.assert_array_equal(batch.values[-1], data[6:12, -3:].T)
            np.testing.assert_allclose(
                batch.start_seconds, np.array([0, 3, 6]) / 25_600
            )
            assert batch.recording_names == [str(path.relative_to(tmp_path))] * 3

    config_path = (
        Path(__file__).resolve().parents[1]
        / "ctta_data_builder/configs/octta/hust_gearbox.json"
    )
    config = config_from_mapping(
        json.loads(config_path.read_text()), config_path.parent
    )
    assert config.domains == adapter.domain_names
    assert len(config.domains) == 30
    assert config.domains[0] == "20Hz_L0"
    assert config.domains[24] == "40Hz_L4"
    counts = adapter.validate_selection(
        tmp_path, adapter.class_names, adapter.domain_names[25:]
    )
    assert len(counts) == 15


def test_point_windowing_supports_configurable_overlap_stride() -> None:
    signal = np.arange(2 * 15, dtype=np.float32).reshape(2, 15)
    batch = _window_signal(
        signal,
        100,
        "recording",
        window_size_points=6,
        stride_points=3,
    )

    assert batch.values.shape == (4, 2, 6)
    assert np.array_equal(batch.values[0], signal[:, :6])
    assert np.array_equal(batch.values[1], signal[:, 3:9])
    assert np.array_equal(batch.values[-1], signal[:, 9:15])
    assert batch.start_seconds.tolist() == [0.0, 0.03, 0.06, 0.09]


def test_sk_adapter_reads_three_channels_and_excludes_repeat_files(
    tmp_path: Path,
) -> None:
    adapter = SKAdapter()
    raw_root = tmp_path / "SK"
    recording = raw_root / "IF0.4" / "3000 0.mat"
    recording.parent.mkdir(parents=True)
    values = np.arange(8 * 3, dtype=np.float64).reshape(8, 3)
    savemat(recording, {"Signal": {"y_values": {"values": values}}})
    savemat(
        recording.with_name("3000 0 2.mat"),
        {"Signal": {"y_values": {"values": values + 100}}},
    )

    paths = adapter.recording_paths(tmp_path, "IF0.4", "3000rpm_0N")
    batch = adapter.load_windows(
        tmp_path,
        "IF0.4",
        "3000rpm_0N",
        window_size_points=4,
        stride_points=2,
    )

    assert adapter.sampling_frequency == 25_600
    assert adapter.channel_names == ("point2_z", "point3_y", "another_y")
    assert len(adapter.class_names) == 10
    assert len(adapter.domain_names) == 24
    assert adapter.domain_names[6] == "1000rpm_40N"
    assert paths == [recording]
    assert batch.values.shape == (3, 3, 4)
    assert batch.values[0].tolist() == [
        [0.0, 3.0, 6.0, 9.0],
        [1.0, 4.0, 7.0, 10.0],
        [2.0, 5.0, 8.0, 11.0],
    ]
    assert batch.values[1, 0].tolist() == [6.0, 9.0, 12.0, 15.0]
    assert batch.start_seconds.tolist() == [0.0, 2 / 25_600, 4 / 25_600]


@pytest.mark.parametrize("target_stream_order", ["interleaved", "dirichlet"])
def test_builder_default_is_1024_points(
    tmp_path: Path, target_stream_order: str
) -> None:
    config = config_from_mapping(
        {
            "dataset": "PU",
            "raw_root": str(tmp_path),
            "output": "custom.mat",
            "classes": ["K001", "KI14"],
            "domains": ["N15_M07_F10", "N09_M07_F10"],
            "samples_per_class": 5,
            "target_stream_order": target_stream_order,
        },
        tmp_path,
    )
    assert config.window_size_points == DEFAULT_WINDOW_SIZE_POINTS == 1024
    assert config.resolved_stride_points == config.window_size_points
    assert config.target_stream_order == target_stream_order


def test_official_builder_configs_explicitly_list_all_build_fields(tmp_path: Path) -> None:
    required_fields = {
        "dataset",
        "name",
        "raw_root",
        "output",
        "classes",
        "channels",
        "domains",
        "samples_per_class",
        "window_size_points",
        "stride_points",
        "selection",
        "target_stream_order",
    }
    config_paths = [
        path
        for path in Path("ctta_data_builder/configs/octta").glob("*.json")
        if " copy" not in path.name
    ]

    assert (
        len(config_paths) == len(core.DATASETS) - 1
    )  # FMCTTA also retains Motor legacy.
    for path in config_paths:
        content = json.loads(path.read_text(encoding="utf-8"))
        assert required_fields.issubset(content), path
        adapter = core.DATASETS[content["dataset"]]
        assert content["classes"]
        assert content["channels"]
        assert content["domains"]
        assert set(content["classes"]).issubset(adapter.class_names)
        assert set(content["channels"]).issubset(adapter.channel_names)
        config = config_from_mapping({**content, "raw_root": str(tmp_path)}, path.parent)
        core.validate_config(config, adapter)
        assert config.output.parent == Path("data/ctta_custom").resolve()


def test_builder_rejects_nonpositive_stride(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stride_points"):
        config_from_mapping(
            {
                "dataset": "PU",
                "raw_root": str(tmp_path),
                "output": "custom.mat",
                "stride_points": 0,
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
            self,
            raw_root,
            class_name,
            domain_name,
            window_size_points,
            stride_points=None,
        ):
            value = self.domain_names.index(domain_name) * 10
            value += self.class_names.index(class_name)
            values = np.empty((8, 2, window_size_points), dtype=np.float32)
            values[:, 0] = value
            values[:, 1] = value + 100
            return WindowBatch(
                values,
                np.arange(8, dtype=np.float64) * stride_points / 100,
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
        stride_points=3,
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
    assert content["stride_points"].item() == 3
    assert content["stride_seconds"].item() == pytest.approx(0.03)
    assert content["overlap"].item() == 1
    assert content["data"][0, 0, 0] == 111
    assert content["window_start_sec"].shape == (20, 1)
    assert result.manifest["window_size_points"] == 12
    assert result.manifest["stride_points"] == 3
    assert result.manifest["stride_seconds"] == pytest.approx(0.03)
    assert result.manifest["overlap"] is True
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
    assert "octta_spec" not in result.manifest

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
