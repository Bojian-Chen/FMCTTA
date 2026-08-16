"""Configuration, aggregation, and MAT/manifest output for CTTA datasets."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio

from .datasets import DATASETS, WINDOW_SECONDS, DatasetAdapter, resolve_dataset_name


@dataclass(frozen=True)
class BuildConfig:
    dataset: str
    raw_root: Path
    output: Path
    classes: Tuple[str, ...]
    domains: Tuple[str, ...]
    samples_per_class: int = 100
    selection: str = "uniform"
    target_stream_order: str = "interleaved"
    name: str = ""

    @property
    def dataset_name(self) -> str:
        return self.name or f"{self.dataset}_custom"


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
        raise ValueError(
            "domains must map numeric labels to raw domain names, for example "
            "{\"0\": \"condition_a\", \"1\": \"condition_b\"}"
        )
    try:
        indexed = {int(label): str(name) for label, name in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("domains keys must be integer labels starting at 0") from exc
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise ValueError(
            f"domains labels must be contiguous 0..{len(indexed) - 1}; "
            f"got {sorted(indexed)}"
        )
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
    config = BuildConfig(
        dataset=dataset,
        raw_root=_resolve_path(content["raw_root"], base_dir),
        output=_resolve_path(content["output"], base_dir),
        classes=classes,
        domains=domains,
        samples_per_class=int(content.get("samples_per_class", 100)),
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
    if config.samples_per_class <= 1:
        raise ValueError(
            "samples_per_class must be at least 2 for FMCTTA train/validation splitting"
        )
    if config.selection not in {"head", "uniform"}:
        raise ValueError("selection must be 'head' or 'uniform'")
    if config.target_stream_order not in {
        "class_block",
        "interleaved",
        "shuffled",
    }:
        raise ValueError(
            "target_stream_order must be class_block, interleaved, or shuffled"
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
        raise ValueError(
            f"Invalid classes for {adapter.name}: {invalid_classes}; "
            f"available={list(adapter.class_names)}"
        )
    if invalid_domains:
        raise ValueError(
            f"Invalid domains for {adapter.name}: {invalid_domains}; "
            f"available={list(adapter.domain_names)}"
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
            f"Only {available} non-overlapping 0.1-second windows are available, "
            f"but {requested} were requested"
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
    total = len(config.domains) * len(config.classes) * config.samples_per_class
    return {
        "data": np.empty(
            (total, len(adapter.channel_names), adapter.window_size),
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
    arrays = _allocate_arrays(config, adapter)
    recordings: List[str] = []
    recording_ids: Dict[str, int] = {}
    combination_counts: List[Dict[str, object]] = []
    block_size = config.samples_per_class

    for new_domain, domain_name in enumerate(config.domains):
        for new_label, class_name in enumerate(config.classes):
            batch = adapter.load_windows(config.raw_root, class_name, domain_name)
            if batch.values.shape[1:] != (
                len(adapter.channel_names),
                adapter.window_size,
            ):
                raise ValueError(
                    f"Unexpected sample shape for {domain_name}/{class_name}: "
                    f"{batch.values.shape[1:]}"
                )
            indices = _selection_indices(
                len(batch.values), config.samples_per_class, config.selection
            )
            begin = (new_domain * len(config.classes) + new_label) * block_size
            stop = begin + block_size
            arrays["data"][begin:stop] = batch.values[indices]
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
                    "available_non_overlapping_windows": int(len(batch.values)),
                    "selected_windows": block_size,
                }
            )
            print(
                f"[{new_domain + 1:02d}/{len(config.domains):02d}] "
                f"{domain_name} / {class_name}: "
                f"selected {block_size} of {len(batch.values)}",
                flush=True,
            )

    metadata = {
        "recordings": recordings,
        "combination_counts": combination_counts,
    }
    return arrays, metadata


def _manifest(
    config: BuildConfig,
    adapter: DatasetAdapter,
    arrays: Mapping[str, np.ndarray],
    build_metadata: Mapping[str, object],
) -> Dict[str, Any]:
    total = int(arrays["data"].shape[0])
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
        }
        for new_id, name in enumerate(config.domains)
    ]
    return {
        "format_version": 1,
        "dataset": config.dataset,
        "name": config.dataset_name,
        "raw_root": str(config.raw_root.resolve()),
        "mat_file": str(config.output.resolve()),
        "sampling_frequency_hz": adapter.sampling_frequency,
        "window_seconds": WINDOW_SECONDS,
        "window_size_points": adapter.window_size,
        "stride_seconds": WINDOW_SECONDS,
        "overlap": False,
        "resampling": "none",
        "normalization": "none",
        "channels": list(adapter.channel_names),
        "num_channels": len(adapter.channel_names),
        "classes": class_mapping,
        "domains": domain_mapping,
        "domain_labels": {
            str(label): name for label, name in enumerate(config.domains)
        },
        "domain_sequence_defined_at_runtime": True,
        "available_domain_labels": list(range(len(config.domains))),
        "samples_per_class_per_domain": config.samples_per_class,
        "num_classes": len(config.classes),
        "num_domains": len(config.domains),
        "total_samples": total,
        "storage_order": "domain -> class -> acquisition time",
        "selection": config.selection,
        "target_stream_order": config.target_stream_order,
        "recordings": build_metadata["recordings"],
        "combination_counts": build_metadata["combination_counts"],
        "fmctta_spec": {
            "name": config.dataset_name,
            "filename": config.output.name,
            "num_classes": len(config.classes),
            "num_domains": len(config.domains),
            "samples_per_class": config.samples_per_class,
            "default_sequence": list(range(len(config.domains))),
            "description": (
                f"Custom native-rate 0.1-second {config.dataset} CTTA dataset"
            ),
            "num_channels": len(adapter.channel_names),
            "default_stream_order": config.target_stream_order,
            "domain_names": list(config.domains),
        },
    }


def _add_mat_metadata(
    arrays: Dict[str, np.ndarray],
    config: BuildConfig,
    adapter: DatasetAdapter,
    recordings: Sequence[str],
) -> None:
    arrays.update(
        {
            "class_names": _mat_string_row(config.classes),
            "domain_names": _mat_string_row(config.domains),
            "channels": _mat_string_row(adapter.channel_names),
            "recording_names": _mat_string_row(recordings),
            "sampling_frequency": np.asarray(
                [[adapter.sampling_frequency]], dtype=np.int64
            ),
            "window_size": np.asarray([[adapter.window_size]], dtype=np.int64),
            "window_seconds": np.asarray([[WINDOW_SECONDS]], dtype=np.float64),
            "samples_per_class": np.asarray(
                [[config.samples_per_class]], dtype=np.int64
            ),
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
    arrays, build_metadata = _build_arrays(config, adapter)
    manifest = _manifest(config, adapter, arrays, build_metadata)
    _add_mat_metadata(
        arrays,
        config,
        adapter,
        build_metadata["recordings"],
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
