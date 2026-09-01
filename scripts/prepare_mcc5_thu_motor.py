#!/usr/bin/env python3
"""Build all MCC5-THU Motor steady candidates and online streams.

Each valid raw circulation recording contributes two steady source candidates
and one complete 90-second target stream.
"""

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = (
    REPO_ROOT.parents[1]
    / "PHMFD-data-prepocessing"
    / "Raw_data"
    / "Finetune"
    / "MCC5-THU_Motor"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "MCC5_THU_Motor_online_24classes_33domains.mat"
)

SAMPLING_FREQUENCY = 12_800
WINDOW_SIZE = 1024
# The local CSV files contain time, key phase, torque, vibration x/y/z, and current A/B/C. Only the motor vibration axes are model inputs.
MOTOR_VIBRATION_COLUMNS = (3, 4, 5)
CHANNEL_NAMES = (
    "motor_vibration_x",
    "motor_vibration_y",
    "motor_vibration_z",
)
CLASS_NAMES = (
    "health",
    "bearing_ball_H",
    "bearing_ball_L",
    "bearing_inner_H",
    "bearing_inner_L",
    "bearing_outer_H",
    "bearing_outer_L",
    "bearing_outer_H_and_inner_H",
    "bend",
    "broken_bar",
    "broken_bar_and_bearing_inner_H",
    "broken_bar_and_bearing_outer_H",
    "dynamic_eccentricity",
    "dynamic_eccentricity_and_bearing_inner_H",
    "dynamic_eccentricity_and_bearing_outer_H",
    "static_eccentricity_H",
    "static_eccentricity_L",
    "static_eccentricity_H_and_bearing_inner_H",
    "static_eccentricity_H_and_bearing_outer_H",
    "voltage_unbalance_L",
    "winding_H",
    "winding_L",
    "winding_H_and_bearing_inner_H",
    "winding_H_and_bearing_outer_H",
)


@dataclass(frozen=True)
class DomainDefinition:
    name: str
    mode: str
    torque_nm: int
    rpm: int
    source_segments_sec: Tuple[Tuple[float, float], ...] = ()

    @property
    def is_source(self) -> bool:
        return bool(self.source_segments_sec)

    @property
    def file_condition(self) -> str:
        return f"{self.mode}_circulation_{self.torque_nm}Nm_{self.rpm}rpm"


# speed/20Nm/3000rpm is intentionally omitted. The local copy is missing
# bearing_inner_H and contains two bearing_outer_H recordings for that profile.
STEADY_1_SEGMENTS = ((10.0, 30.0), (60.0, 80.0))
STEADY_2_SEGMENTS = ((35.0, 55.0),)
PROFILES = (
    ("torque_1000rpm_20to15Nm", "torque", 20, 1000, "20Nm", "15Nm"),
    ("torque_2000rpm_20to15Nm", "torque", 20, 2000, "20Nm", "15Nm"),
    ("torque_3000rpm_20to15Nm", "torque", 20, 3000, "20Nm", "15Nm"),
    ("torque_3000rpm_40to30Nm", "torque", 40, 3000, "40Nm", "30Nm"),
    ("torque_2000rpm_40to30Nm", "torque", 40, 2000, "40Nm", "30Nm"),
    ("torque_1000rpm_40to30Nm", "torque", 40, 1000, "40Nm", "30Nm"),
    ("speed_40Nm_1000to500rpm", "speed", 40, 1000, "1000rpm", "500rpm"),
    ("speed_40Nm_2000to1500rpm", "speed", 40, 2000, "2000rpm", "1500rpm"),
    ("speed_40Nm_3000to2500rpm", "speed", 40, 3000, "3000rpm", "2500rpm"),
    ("speed_20Nm_2000to1500rpm", "speed", 20, 2000, "2000rpm", "1500rpm"),
    ("speed_20Nm_1000to500rpm", "speed", 20, 1000, "1000rpm", "500rpm"),
)


def _domain_triplet(profile: Tuple[str, str, int, int, str, str]) -> Tuple[DomainDefinition, ...]:
    name, mode, torque, rpm, steady_1, steady_2 = profile
    return (
        DomainDefinition(f"{name}__steady_1_{steady_1}", mode, torque, rpm, STEADY_1_SEGMENTS),
        DomainDefinition(f"{name}__steady_2_{steady_2}", mode, torque, rpm, STEADY_2_SEGMENTS),
        DomainDefinition(f"{name}__stream", mode, torque, rpm),
    )


DOMAINS = tuple(domain for profile in PROFILES for domain in _domain_triplet(profile))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the MCC5-THU Motor online CTTA MAT dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Must match DatasetSpec.samples_per_class in fmctta/config.py for benchmark loading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selected raw files without reading signal values",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing MAT file and manifest",
    )
    return parser


def condition_directory(raw_dir: Path, mode: str) -> Path:
    return raw_dir / f"MCC5-THU Motor_{mode}_circulation"


def resolve_recording(
    raw_dir: Path,
    class_name: str,
    domain: DomainDefinition,
) -> Path:
    folder = condition_directory(raw_dir, domain.mode)
    pattern = f"{class_name}_{domain.file_condition}*.csv"
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise ValueError(f"Expected exactly one file for {class_name}/{domain.name}, found {len(matches)}: {names}")
    return matches[0]


def validate_layout(raw_dir: Path) -> None:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"MCC5-THU Motor directory not found: {raw_dir}")
    for domain in DOMAINS:
        for class_name in CLASS_NAMES:
            resolve_recording(raw_dir, class_name, domain)


def local_layout_audit(raw_dir: Path) -> Dict[str, object]:
    all_files = sorted(raw_dir.glob("MCC5-THU Motor_*_circulation/*.csv"))
    excluded = condition_directory(raw_dir, "speed")
    missing_pattern = "bearing_inner_H_speed_circulation_20Nm_3000rpm*.csv"
    duplicate_pattern = "bearing_outer_H_speed_circulation_20Nm_3000rpm*.csv"
    missing_matches = sorted(excluded.glob(missing_pattern))
    duplicate_matches = sorted(excluded.glob(duplicate_pattern))
    return {
        "official_readme_recording_count": 282,
        "local_csv_count": len(all_files),
        "excluded_condition": "speed_circulation_20Nm_3000rpm",
        "reason": "The local condition has no bearing_inner_H recording and has two bearing_outer_H recordings; it is excluded instead of guessing a replacement label.",
        "bearing_inner_H_matches": [path.name for path in missing_matches],
        "bearing_outer_H_matches": [path.name for path in duplicate_matches],
    }


@lru_cache(maxsize=len(CLASS_NAMES))
def read_motor_vibration(path: Path) -> np.ndarray:
    frame = pd.read_csv(
        path,
        header=None,
        usecols=list(MOTOR_VIBRATION_COLUMNS),
        dtype=np.float32,
    )
    signal = frame.to_numpy(dtype=np.float32, copy=True)
    if signal.ndim != 2 or signal.shape[1] != len(MOTOR_VIBRATION_COLUMNS):
        raise ValueError(f"Unexpected vibration shape in {path}: {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite vibration values found in {path}")
    return signal


def source_starts(
    segments_sec: Sequence[Tuple[float, float]],
    sample_count: int,
    signal_length: int,
) -> np.ndarray:
    starts: List[int] = []
    for start_sec, end_sec in segments_sec:
        segment_start = int(round(start_sec * SAMPLING_FREQUENCY))
        segment_stop = min(
            int(round(end_sec * SAMPLING_FREQUENCY)), signal_length
        )
        starts.extend(
            range(segment_start, segment_stop - WINDOW_SIZE + 1, WINDOW_SIZE)
        )
    if len(starts) < sample_count:
        raise ValueError(f"Steady source segments provide {len(starts)} non-overlapping windows, fewer than requested {sample_count}")
    selection = np.linspace(0, len(starts) - 1, sample_count, dtype=np.int64)
    return np.asarray(starts, dtype=np.int64)[selection]


def target_starts(
    signal_length: int,
    sample_count: int,
) -> np.ndarray:
    last_start = signal_length - WINDOW_SIZE
    if last_start < 0:
        raise ValueError(
            f"Signal length {signal_length} is shorter than window {WINDOW_SIZE}"
        )
    starts = np.linspace(0, last_start, sample_count, dtype=np.int64)
    if len(np.unique(starts)) != sample_count:
        raise ValueError("Requested more windows than distinct start positions")
    if sample_count > 1 and int(np.diff(starts).min()) < WINDOW_SIZE:
        raise ValueError(
            "Requested target windows overlap; reduce --samples-per-class"
        )
    return starts


def extract_windows(
    signal: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [signal[start : start + WINDOW_SIZE].T for start in starts], axis=0
    ).astype(np.float32, copy=False)


def build_dataset(
    raw_dir: Path,
    samples_per_class: int,
) -> Dict[str, np.ndarray]:
    if samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")

    total = len(DOMAINS) * len(CLASS_NAMES) * samples_per_class
    data = np.empty((total, len(CHANNEL_NAMES), WINDOW_SIZE), dtype=np.float32)
    labels = np.empty(total, dtype=np.int64)
    domain_ids = np.empty(total, dtype=np.int64)
    start_seconds = np.empty(total, dtype=np.float32)

    for domain_id, domain in enumerate(DOMAINS):
        for class_id, class_name in enumerate(CLASS_NAMES):
            path = resolve_recording(raw_dir, class_name, domain)
            signal = read_motor_vibration(path)
            if domain.is_source:
                starts = source_starts(
                    domain.source_segments_sec,
                    samples_per_class,
                    len(signal),
                )
            else:
                starts = target_starts(len(signal), samples_per_class)
            windows = extract_windows(signal, starts)
            begin = (domain_id * len(CLASS_NAMES) + class_id) * samples_per_class
            stop = begin + samples_per_class
            data[begin:stop] = windows
            labels[begin:stop] = class_id
            domain_ids[begin:stop] = domain_id
            start_seconds[begin:stop] = starts / SAMPLING_FREQUENCY
        print(f"[{domain_id + 1:02d}/{len(DOMAINS):02d}] {domain.name}: {len(CLASS_NAMES) * samples_per_class} samples", flush=True)

    return {
        "data": data,
        "label": labels.reshape(-1, 1),
        "domain": domain_ids.reshape(-1, 1),
        "window_start_sec": start_seconds.reshape(-1, 1),
        "class_names": np.asarray(CLASS_NAMES, dtype=object).reshape(1, -1),
        "domain_names": np.asarray(
            [domain.name for domain in DOMAINS], dtype=object
        ).reshape(1, -1),
        "channels": np.asarray(CHANNEL_NAMES, dtype=object).reshape(1, -1),
        "sampling_frequency": np.asarray([[SAMPLING_FREQUENCY]], dtype=np.int64),
        "window_size": np.asarray([[WINDOW_SIZE]], dtype=np.int64),
        "resampling": np.asarray([["none"]], dtype=object),
        "samples_per_class": np.asarray([[samples_per_class]], dtype=np.int64),
    }


def manifest(
    raw_dir: Path,
    output: Path,
    samples_per_class: int,
) -> Dict[str, object]:
    return {
        "dataset": "MCC5_THU_Motor",
        "raw_dir": str(raw_dir.resolve()),
        "mat_file": str(output.resolve()),
        "sampling_frequency_hz": SAMPLING_FREQUENCY,
        "recording_seconds": 90,
        "window_size_points": WINDOW_SIZE,
        "window_seconds": WINDOW_SIZE / SAMPLING_FREQUENCY,
        "resampling": "none",
        "channels": list(CHANNEL_NAMES),
        "class_names": list(CLASS_NAMES),
        "samples_per_class_per_domain": samples_per_class,
        "num_classes": len(CLASS_NAMES),
        "num_domains": len(DOMAINS),
        "total_samples": len(DOMAINS) * len(CLASS_NAMES) * samples_per_class,
        "storage_order": "domain -> class -> acquisition time",
        "recommended_stream_order": "interleaved",
        "recommended_stream_batch_size": len(CLASS_NAMES),
        "sources": [
            {
                "domain": domain_id,
                "name": domain.name,
                "raw_condition": domain.file_condition,
                "steady_segments_sec": [
                    list(item) for item in domain.source_segments_sec
                ],
                "recording_group": domain.file_condition,
            }
            for domain_id, domain in enumerate(DOMAINS)
            if domain.is_source
        ],
        "targets": [
            {
                "domain": domain_id,
                "name": domain.name,
                "raw_condition": domain.file_condition,
                "time_order_preserved": True,
                "recording_group": domain.file_condition,
            }
            for domain_id, domain in enumerate(DOMAINS)
            if not domain.is_source
        ],
        "recommended_source_domains": [0, 1],
        "recommended_target_domains": list(range(5, len(DOMAINS), 3)),
        "raw_layout_audit": local_layout_audit(raw_dir),
        "target_labels_used_for_stream_construction_only": True,
        "target_labels_used_for_adaptation": False,
    }


def save_dataset(output: Path, arrays: Dict[str, np.ndarray], overwrite: bool) -> None:
    output = output.resolve()
    manifest_path = output.with_suffix(".json")
    if not overwrite:
        existing = [path for path in (output, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"Output already exists: {existing[0]}; pass --overwrite to replace it"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        sio.savemat(temporary, arrays, appendmat=False, do_compression=False)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = args.raw_dir.resolve()
    output = args.output.resolve()
    validate_layout(raw_dir)
    details = manifest(
        raw_dir,
        output,
        args.samples_per_class,
    )
    if args.dry_run:
        print(json.dumps(details, indent=2, ensure_ascii=False))
        return

    arrays = build_dataset(
        raw_dir,
        args.samples_per_class,
    )
    save_dataset(output, arrays, args.overwrite)
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(details, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {arrays['data'].shape} float32 signals to {output}")
    print(f"Saved protocol manifest to {manifest_path}")


if __name__ == "__main__":
    main()

