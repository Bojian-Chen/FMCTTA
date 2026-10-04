"""Configuration, aggregation, and MAT/manifest output for CTTA datasets."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio

from .datasets import (
    DATASETS,
    DEFAULT_WINDOW_SIZE_POINTS,
    DatasetAdapter,
    resolve_dataset_name,
)


# MATLAB v5 stores the size of an individual matrix in a signed 32-bit field.
# Larger sensor tensors need the HDF5-backed output handled by fmctta.data.
MATLAB_V5_MAX_MATRIX_BYTES = 2**31 - 1


@dataclass(frozen=True)
class BuildConfig:
    dataset: str
    raw_root: Path
    output: Path
    classes: Tuple[str, ...]
    domains: Tuple[str, ...]
    channels: Tuple[str, ...] = ()
    # None preserves every complete window at the configured stride.
    samples_per_class: Optional[int] = 100
    window_size_points: int = DEFAULT_WINDOW_SIZE_POINTS
    # None preserves the historical behavior: stride equals window size.
    stride_points: Optional[int] = None
    selection: str = "uniform"
    target_stream_order: str = "interleaved"
    name: str = ""

    @property
    def dataset_name(self) -> str:
        return self.name or f"{self.dataset}_custom"

    @property
    def resolved_stride_points(self) -> int:
        return (
            self.window_size_points
            if self.stride_points is None
            else self.stride_points
        )


@dataclass
class BuildResult:
    mat_path: Path
    manifest_path: Path
    data_shape: Tuple[int, ...]
    manifest: Dict[str, Any]


def _resolve_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _string_list(value: object, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty JSON list")
    items = tuple(str(item) for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{field} contains duplicates: {items}")
    return items


def _domain_labels(value: object) -> Tuple[str, ...]:
    if isinstance(value, list):
        return _string_list(value, "domains")
    if not isinstance(value, dict) or not value:
        raise ValueError('domains must map numeric labels to raw domain names, for example {"0": "condition_a", "1": "condition_b"}')
    try:
        indexed = {int(label): str(name) for label, name in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("domains keys must be integer labels starting at 0") from exc
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise ValueError(f"domains labels must be contiguous 0..{len(indexed) - 1}; got {sorted(indexed)}")
    names = tuple(indexed[label] for label in expected)
    if len(set(names)) != len(names):
        raise ValueError(f"domains maps multiple labels to the same domain: {names}")
    return names


def config_from_mapping(
    content: Mapping[str, object],
    base_dir: Optional[Path] = None,
) -> BuildConfig:
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    if "dataset" not in content:
        raise ValueError("Config is missing 'dataset'")
    dataset = resolve_dataset_name(str(content["dataset"]))
    adapter = DATASETS[dataset]
    if "raw_root" not in content or "output" not in content:
        raise ValueError("Config must contain 'raw_root' and 'output'")

    classes = (
        _string_list(content["classes"], "classes")
        if "classes" in content
        else adapter.class_names
    )
    domains = (
        _domain_labels(content["domains"])
        if "domains" in content
        else adapter.domain_names
    )
    channels = (
        _string_list(content["channels"], "channels")
        if "channels" in content
        else adapter.channel_names
    )
    samples_value = content.get("samples_per_class", 100)
    config = BuildConfig(
        dataset=dataset,
        raw_root=_resolve_path(content["raw_root"], base_dir),
        output=_resolve_path(content["output"], base_dir),
        classes=classes,
        domains=domains,
        channels=channels,
        samples_per_class=(
            None if samples_value is None else int(samples_value)
        ),
        window_size_points=int(
            content.get("window_size_points", DEFAULT_WINDOW_SIZE_POINTS)
        ),
        stride_points=(
            None
            if content.get("stride_points") is None
            else int(content["stride_points"])
        ),
        selection=str(content.get("selection", "uniform")),
        target_stream_order=str(
            content.get("target_stream_order", "interleaved")
        ),
        name=str(content.get("name", "")),
    )
    validate_config(config, adapter)
    return config


def load_build_config(path: Path) -> BuildConfig:
    path = Path(path).resolve()
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("The build config must be a JSON object")
    return config_from_mapping(content, path.parent)


def validate_config(
    config: BuildConfig,
    adapter: Optional[DatasetAdapter] = None,
) -> None:
    adapter = DATASETS[config.dataset] if adapter is None else adapter
    if not config.raw_root.is_dir():
        raise FileNotFoundError(f"Raw data root does not exist: {config.raw_root}")
    if config.output.suffix.lower() != ".mat":
        raise ValueError("output must use the .mat extension")
    if config.samples_per_class is not None and config.samples_per_class <= 1:
        raise ValueError("samples_per_class must be at least 2 for source train/validation splitting")
    if config.window_size_points < 1:
        raise ValueError("window_size_points must be positive")
    if config.resolved_stride_points < 1:
        raise ValueError("stride_points must be positive")
    if config.selection not in {"head", "uniform"}:
        raise ValueError("selection must be 'head' or 'uniform'")
    if config.target_stream_order not in {
        "class_block",
        "interleaved",
        "shuffled",
        "dirichlet",
    }:
        raise ValueError(
            "target_stream_order must be class_block, interleaved, shuffled, or dirichlet"
        )
    if len(config.classes) < 2:
        raise ValueError("At least two classes are required")
    if len(config.domains) < 2:
        raise ValueError("At least two labeled domains are required for CTTA")
    invalid_classes = [
        name for name in config.classes if name not in adapter.class_names
    ]
    invalid_domains = [
        name for name in config.domains if name not in adapter.domain_names
    ]
    if invalid_classes:
        raise ValueError(f"Invalid classes for {adapter.name}: {invalid_classes}; available={list(adapter.class_names)}")
    if invalid_domains:
        raise ValueError(f"Invalid domains for {adapter.name}: {invalid_domains}; available={list(adapter.domain_names)}")
    channels = config.channels or adapter.channel_names
    if len(set(channels)) != len(channels):
        raise ValueError(f"channels contains duplicates: {channels}")
    invalid_channels = [
        name for name in channels if name not in adapter.channel_names
    ]
    if invalid_channels:
        raise ValueError(
            f"Invalid channels for {adapter.name}: {invalid_channels}; "
            f"available={list(adapter.channel_names)}"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.dataset_name):
        raise ValueError(
            "name may contain only letters, digits, underscore, dash, and dot"
        )


def validate_raw_layout(config: BuildConfig) -> Dict[str, int]:
    adapter = DATASETS[config.dataset]
    validate_config(config, adapter)
    return adapter.validate_selection(config.raw_root, config.classes, config.domains)


def _selection_indices(available: int, requested: int, method: str) -> np.ndarray:
    if available < requested:
        raise ValueError(
            f"Only {available} point windows are available, but {requested} "
            "were requested"
        )
    if method == "head":
        return np.arange(requested, dtype=np.int64)
    indices = np.linspace(0, available - 1, requested, dtype=np.int64)
    if len(np.unique(indices)) != requested:
        raise RuntimeError("Uniform selection unexpectedly produced duplicate windows")
    return indices


def _mat_string_row(values: Sequence[str]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=object).reshape(1, -1)


def _allocate_arrays(
    config: BuildConfig,
    adapter: DatasetAdapter,
) -> Dict[str, np.ndarray]:
    if config.samples_per_class is None:
        raise ValueError("Full-window builds use streaming HDF5 allocation")
    total = len(config.domains) * len(config.classes) * config.samples_per_class
    channels = config.channels or adapter.channel_names
    return {
        "data": np.empty(
            (total, len(channels), config.window_size_points),
            dtype=np.float32,
        ),
        "label": np.empty((total, 1), dtype=np.int64),
        "domain": np.empty((total, 1), dtype=np.int64),
        "original_label": np.empty((total, 1), dtype=np.int64),
        "original_domain": np.empty((total, 1), dtype=np.int64),
        "window_start_sec": np.empty((total, 1), dtype=np.float64),
        "sequence_index": np.empty((total, 1), dtype=np.int64),
        "recording_id": np.empty((total, 1), dtype=np.int64),
    }


def _build_arrays(
    config: BuildConfig,
    adapter: DatasetAdapter,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    if config.samples_per_class is None:
        raise ValueError("Full-window builds use streaming HDF5 allocation")
    arrays = _allocate_arrays(config, adapter)
    recordings: List[str] = []
    recording_ids: Dict[str, int] = {}
    combination_counts: List[Dict[str, object]] = []
    block_size = config.samples_per_class
    selected_channels = config.channels or adapter.channel_names
    channel_indices = [
        adapter.channel_names.index(name) for name in selected_channels
    ]

    for new_domain, domain_name in enumerate(config.domains):
        for new_label, class_name in enumerate(config.classes):
            batch = adapter.load_windows(
                config.raw_root,
                class_name,
                domain_name,
                config.window_size_points,
                config.resolved_stride_points,
            )
            if batch.values.shape[1:] != (
                len(adapter.channel_names),
                config.window_size_points,
            ):
                raise ValueError(f"Unexpected sample shape for {domain_name}/{class_name}: {batch.values.shape[1:]}")
            indices = _selection_indices(
                len(batch.values), config.samples_per_class, config.selection
            )
            begin = (new_domain * len(config.classes) + new_label) * block_size
            stop = begin + block_size
            arrays["data"][begin:stop] = batch.values[indices][:, channel_indices]
            arrays["label"][begin:stop, 0] = new_label
            arrays["domain"][begin:stop, 0] = new_domain
            arrays["original_label"][begin:stop, 0] = adapter.class_names.index(
                class_name
            )
            arrays["original_domain"][begin:stop, 0] = adapter.domain_names.index(
                domain_name
            )
            arrays["window_start_sec"][begin:stop, 0] = batch.start_seconds[indices]
            arrays["sequence_index"][begin:stop, 0] = np.arange(block_size)

            selected_names = [batch.recording_names[int(index)] for index in indices]
            block_recording_ids = []
            for name in selected_names:
                if name not in recording_ids:
                    recording_ids[name] = len(recordings)
                    recordings.append(name)
                block_recording_ids.append(recording_ids[name])
            arrays["recording_id"][begin:stop, 0] = block_recording_ids
            combination_counts.append(
                {
                    "domain": domain_name,
                    "class": class_name,
                    "available_windows": int(len(batch.values)),
                    "selected_windows": block_size,
                }
            )
            print(f"[{new_domain + 1:02d}/{len(config.domains):02d}] {domain_name} / {class_name}: selected {block_size} of {len(batch.values)}", flush=True)

    return arrays, {
        "recordings": recordings,
        "combination_counts": combination_counts,
    }


def _manifest(
    config: BuildConfig,
    adapter: DatasetAdapter,
    total_samples: int,
    samples_per_class_by_domain: Sequence[int],
    build_metadata: Mapping[str, object],
) -> Dict[str, Any]:
    total = int(total_samples)
    domain_counts = [int(count) for count in samples_per_class_by_domain]
    window_seconds = config.window_size_points / adapter.sampling_frequency
    stride_seconds = config.resolved_stride_points / adapter.sampling_frequency
    channels = config.channels or adapter.channel_names
    class_mapping = [
        {
            "label": new_id,
            "name": name,
            "original_label": adapter.class_names.index(name),
        }
        for new_id, name in enumerate(config.classes)
    ]
    domain_mapping = [
        {
            "domain_label": new_id,
            "name": name,
            "original_domain": adapter.domain_names.index(name),
            "raw_key": adapter.domain(name).raw_key,
            "segments_sec": [
                list(segment) for segment in adapter.domain(name).segments_sec
            ],
            "role": adapter.domain(name).role,
            "recording_group": (
                adapter.domain(name).recording_group
                or adapter.domain(name).raw_key
            ),
        }
        for new_id, name in enumerate(config.domains)
    ]
    recommended_source_domains = [
        new_id
        for new_id, name in enumerate(config.domains)
        if name in adapter.recommended_source_domains
    ]
    source_candidate_domains = [
        domain["domain_label"]
        for domain in domain_mapping
        if domain["role"] == "source_candidate"
    ]
    target_stream_domains = [
        domain["domain_label"]
        for domain in domain_mapping
        if domain["role"] == "target_stream"
    ]
    recommended_source_groups = {
        domain_mapping[domain]["recording_group"]
        for domain in recommended_source_domains
    }
    recommended_target_domains = [
        domain["domain_label"]
        for domain in domain_mapping
        if domain["role"] == "target_stream"
        and domain["recording_group"] not in recommended_source_groups
    ]
    default_sequence = (
        recommended_source_domains[:1] + recommended_target_domains
        if recommended_source_domains and recommended_target_domains
        else list(range(len(config.domains)))
    )
    return {
        "format_version": 1,
        "dataset": config.dataset,
        "name": config.dataset_name,
        "raw_root": str(config.raw_root.resolve()),
        "mat_file": str(config.output.resolve()),
        "sampling_frequency_hz": adapter.sampling_frequency,
        "window_size_points": config.window_size_points,
        "window_seconds": window_seconds,
        "stride_points": config.resolved_stride_points,
        "stride_seconds": stride_seconds,
        "overlap": config.resolved_stride_points < config.window_size_points,
        "resampling": "none",
        "normalization": "none",
        "channels": list(channels),
        "num_channels": len(channels),
        "classes": class_mapping,
        "domains": domain_mapping,
        "domain_labels": {
            str(label): name for label, name in enumerate(config.domains)
        },
        "domain_sequence_defined_at_runtime": True,
        "available_domain_labels": list(range(len(config.domains))),
        "source_candidate_domains": source_candidate_domains,
        "target_stream_domains": target_stream_domains,
        "transition_segment_domains": [
            domain["domain_label"] for domain in domain_mapping
            if domain["role"] == "transition_segment"
        ],
        "recommended_source_domains": recommended_source_domains,
        "recommended_target_domains": recommended_target_domains,
        "condition_profile": [
            dict(phase) for phase in adapter.condition_profile
        ],
        "samples_per_class_per_domain": domain_counts,
        "num_classes": len(config.classes),
        "num_domains": len(config.domains),
        "total_samples": total,
        "storage_order": "domain -> class -> acquisition time",
        "selection": (
            "all" if config.samples_per_class is None else config.selection
        ),
        "target_stream_order": config.target_stream_order,
        "recordings": build_metadata["recordings"],
        "combination_counts": build_metadata["combination_counts"],
        "fmctta_spec": {
            "name": config.dataset_name,
            "filename": config.output.name,
            "num_classes": len(config.classes),
            "num_domains": len(config.domains),
            # Retained for compatibility with uniform legacy manifests. Runtime
            # indexing uses samples_per_class_by_domain when it is present.
            "samples_per_class": max(domain_counts),
            "samples_per_class_by_domain": domain_counts,
            "default_sequence": default_sequence,
            "description": f"Custom native-rate {config.window_size_points}-point {config.dataset} CTTA dataset",
            "num_channels": len(channels),
            "default_stream_order": config.target_stream_order,
            "domain_names": list(config.domains),
            "domain_groups": [
                str(domain["recording_group"]) for domain in domain_mapping
            ],
        },
    }


def _add_mat_metadata(
    arrays: Dict[str, np.ndarray],
    config: BuildConfig,
    adapter: DatasetAdapter,
    recordings: Sequence[str],
    samples_per_class_by_domain: Sequence[int],
) -> None:
    window_seconds = config.window_size_points / adapter.sampling_frequency
    stride_seconds = config.resolved_stride_points / adapter.sampling_frequency
    channels = config.channels or adapter.channel_names
    selected_domains = [adapter.domain(name) for name in config.domains]
    arrays.update(
        {
            "class_names": _mat_string_row(config.classes),
            "domain_names": _mat_string_row(config.domains),
            "domain_roles": _mat_string_row(
                [domain.role for domain in selected_domains]
            ),
            "domain_groups": _mat_string_row(
                [
                    domain.recording_group or domain.raw_key
                    for domain in selected_domains
                ]
            ),
            "channels": _mat_string_row(channels),
            "recording_names": _mat_string_row(recordings),
            "sampling_frequency": np.asarray(
                [[adapter.sampling_frequency]], dtype=np.int64
            ),
            "window_size": np.asarray(
                [[config.window_size_points]], dtype=np.int64
            ),
            "window_seconds": np.asarray([[window_seconds]], dtype=np.float64),
            "stride_points": np.asarray(
                [[config.resolved_stride_points]], dtype=np.int64
            ),
            "stride_seconds": np.asarray([[stride_seconds]], dtype=np.float64),
            "overlap": np.asarray(
                [[config.resolved_stride_points < config.window_size_points]],
                dtype=np.bool_,
            ),
            "samples_per_class": np.asarray(
                [[max(samples_per_class_by_domain)]], dtype=np.int64
            ),
            "samples_per_class_by_domain": np.asarray(
                samples_per_class_by_domain, dtype=np.int64
            ).reshape(1, -1),
            "resampling": np.asarray([["none"]], dtype=object),
        }
    )


def _save_outputs(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, object],
    overwrite: bool,
) -> Tuple[Path, Path]:
    output = output.resolve()
    manifest_path = output.with_suffix(".json")
    existing = [path for path in (output, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    mat_temp = output.with_name(f".{output.name}.tmp")
    json_temp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        if _requires_hdf5_storage(arrays):
            _save_hdf5(mat_temp, arrays)
        else:
            sio.savemat(mat_temp, dict(arrays), appendmat=False, do_compression=False)
        json_temp.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        mat_temp.replace(output)
        json_temp.replace(manifest_path)
    finally:
        if mat_temp.exists():
            mat_temp.unlink()
        if json_temp.exists():
            json_temp.unlink()
    return output, manifest_path


def _requires_hdf5_storage(arrays: Mapping[str, np.ndarray]) -> bool:
    return any(array.nbytes > MATLAB_V5_MAX_MATRIX_BYTES for array in arrays.values())


def _save_hdf5(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Save a large FMCTTA tensor in an HDF5-backed ``.mat`` container.

    SciPy cannot write MATLAB v5 matrices larger than 2 GiB. The builder and
    FMCTTA loader use the same dataset names and array shapes in this container.
    """

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Large datasets require h5py; install the project requirements"
        ) from exc

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["fmctta_storage_format"] = "hdf5"
        for name, array in arrays.items():
            if array.dtype.kind == "O":
                handle.create_dataset(
                    name,
                    data=np.asarray(array, dtype=object),
                    dtype=string_dtype,
                )
            else:
                handle.create_dataset(name, data=array)


def _append_rows(dataset: Any, values: np.ndarray) -> None:
    begin = int(dataset.shape[0])
    stop = begin + int(values.shape[0])
    dataset.resize(stop, axis=0)
    dataset[begin:stop] = values


def _build_all_windows_dataset(
    config: BuildConfig,
    adapter: DatasetAdapter,
) -> BuildResult:
    """Stream every available window into HDF5 without a multi-GB RAM copy."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Full-window datasets require h5py; install project requirements"
        ) from exc

    output = config.output.resolve()
    manifest_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    mat_temp = output.with_name(f".{output.name}.tmp")
    json_temp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    recordings: List[str] = []
    recording_ids: Dict[str, int] = {}
    combination_counts: List[Dict[str, object]] = []
    domain_counts: List[int] = []
    selected_channels = config.channels or adapter.channel_names
    channel_indices = tuple(
        adapter.channel_names.index(name) for name in selected_channels
    )
    all_channels = channel_indices == tuple(range(len(adapter.channel_names)))

    try:
        with h5py.File(mat_temp, "w") as handle:
            handle.attrs["fmctta_storage_format"] = "hdf5"
            datasets = {
                "data": handle.create_dataset(
                    "data",
                    shape=(0, len(selected_channels), config.window_size_points),
                    maxshape=(None, len(selected_channels), config.window_size_points),
                    chunks=(
                        64,
                        len(selected_channels),
                        config.window_size_points,
                    ),
                    dtype=np.float32,
                )
            }
            for name, dtype in (
                ("label", np.int64),
                ("domain", np.int64),
                ("original_label", np.int64),
                ("original_domain", np.int64),
                ("window_start_sec", np.float64),
                ("sequence_index", np.int64),
                ("recording_id", np.int64),
            ):
                datasets[name] = handle.create_dataset(
                    name,
                    shape=(0, 1),
                    maxshape=(None, 1),
                    chunks=(4096, 1),
                    dtype=dtype,
                )

            for new_domain, domain_name in enumerate(config.domains):
                domain_count: Optional[int] = None
                for new_label, class_name in enumerate(config.classes):
                    batch = adapter.load_windows(
                        config.raw_root,
                        class_name,
                        domain_name,
                        config.window_size_points,
                        config.resolved_stride_points,
                    )
                    if batch.values.shape[1:] != (
                        len(adapter.channel_names),
                        config.window_size_points,
                    ):
                        raise ValueError(
                            f"Unexpected sample shape for {domain_name}/"
                            f"{class_name}: {batch.values.shape[1:]}"
                        )
                    block_size = int(len(batch.values))
                    if block_size < 2:
                        raise ValueError(
                            f"{domain_name}/{class_name} has only {block_size} "
                            "complete windows"
                        )
                    if domain_count is None:
                        domain_count = block_size
                    elif block_size != domain_count:
                        raise ValueError(
                            "Full-window time-aligned streams require equal "
                            f"class lengths within {domain_name}; expected "
                            f"{domain_count}, but {class_name} has {block_size}"
                        )

                    values = (
                        batch.values
                        if all_channels
                        else batch.values[:, channel_indices]
                    )
                    _append_rows(datasets["data"], values)
                    _append_rows(
                        datasets["label"],
                        np.full((block_size, 1), new_label, dtype=np.int64),
                    )
                    _append_rows(
                        datasets["domain"],
                        np.full((block_size, 1), new_domain, dtype=np.int64),
                    )
                    _append_rows(
                        datasets["original_label"],
                        np.full(
                            (block_size, 1),
                            adapter.class_names.index(class_name),
                            dtype=np.int64,
                        ),
                    )
                    _append_rows(
                        datasets["original_domain"],
                        np.full(
                            (block_size, 1),
                            adapter.domain_names.index(domain_name),
                            dtype=np.int64,
                        ),
                    )
                    _append_rows(
                        datasets["window_start_sec"],
                        batch.start_seconds.reshape(-1, 1),
                    )
                    _append_rows(
                        datasets["sequence_index"],
                        np.arange(block_size, dtype=np.int64).reshape(-1, 1),
                    )

                    block_recording_ids = np.empty(
                        (block_size, 1), dtype=np.int64
                    )
                    for index, recording_name in enumerate(
                        batch.recording_names
                    ):
                        if recording_name not in recording_ids:
                            recording_ids[recording_name] = len(recordings)
                            recordings.append(recording_name)
                        block_recording_ids[index, 0] = recording_ids[
                            recording_name
                        ]
                    _append_rows(
                        datasets["recording_id"], block_recording_ids
                    )
                    combination_counts.append(
                        {
                            "domain": domain_name,
                            "class": class_name,
                            "available_windows": block_size,
                            "selected_windows": block_size,
                        }
                    )
                    print(
                        f"[{new_domain + 1:02d}/{len(config.domains):02d}] "
                        f"{domain_name} / {class_name}: saved all "
                        f"{block_size} windows",
                        flush=True,
                    )
                if domain_count is None:
                    raise RuntimeError(f"No classes were built for {domain_name}")
                domain_counts.append(domain_count)

            metadata_arrays: Dict[str, np.ndarray] = {}
            _add_mat_metadata(
                metadata_arrays,
                config,
                adapter,
                recordings,
                domain_counts,
            )
            string_dtype = h5py.string_dtype(encoding="utf-8")
            for name, array in metadata_arrays.items():
                if array.dtype.kind == "O":
                    handle.create_dataset(
                        name,
                        data=np.asarray(array, dtype=object),
                        dtype=string_dtype,
                    )
                else:
                    handle.create_dataset(name, data=array)
            total_samples = int(datasets["data"].shape[0])

        build_metadata = {
            "recordings": recordings,
            "combination_counts": combination_counts,
        }
        manifest = _manifest(
            config,
            adapter,
            total_samples,
            domain_counts,
            build_metadata,
        )
        manifest["storage_format"] = "hdf5"
        json_temp.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        mat_temp.replace(output)
        json_temp.replace(manifest_path)
    finally:
        if mat_temp.exists():
            mat_temp.unlink()
        if json_temp.exists():
            json_temp.unlink()

    return BuildResult(
        mat_path=output,
        manifest_path=manifest_path,
        data_shape=(
            total_samples,
            len(selected_channels),
            config.window_size_points,
        ),
        manifest=manifest,
    )


def _ensure_outputs_available(output: Path, overwrite: bool) -> None:
    if overwrite:
        return
    output = output.resolve()
    existing = [path for path in (output, output.with_suffix(".json")) if path.exists()]
    if existing:
        raise FileExistsError(
            f"Output already exists: {existing[0]}; pass --overwrite to replace it"
        )


def build_dataset(config: BuildConfig, overwrite: bool = False) -> BuildResult:
    adapter = DATASETS[config.dataset]
    validate_config(config, adapter)
    _ensure_outputs_available(config.output, overwrite)
    validate_raw_layout(config)
    if config.samples_per_class is None:
        return _build_all_windows_dataset(config, adapter)
    arrays, build_metadata = _build_arrays(config, adapter)
    domain_counts = [config.samples_per_class] * len(config.domains)
    _add_mat_metadata(
        arrays,
        config,
        adapter,
        build_metadata["recordings"],
        domain_counts,
    )
    manifest = _manifest(
        config,
        adapter,
        int(arrays["data"].shape[0]),
        domain_counts,
        build_metadata,
    )
    manifest["storage_format"] = (
        "hdf5" if _requires_hdf5_storage(arrays) else "matlab_v5"
    )
    mat_path, manifest_path = _save_outputs(
        config.output, arrays, manifest, overwrite
    )
    return BuildResult(
        mat_path=mat_path,
        manifest_path=manifest_path,
        data_shape=tuple(arrays["data"].shape),
        manifest=manifest,
    )
