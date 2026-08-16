"""Raw-data readers for the four CTTA datasets.

The readers follow the channel and condition choices in
``PHMFD-data-prepocessing``.  They deliberately expose only native-rate,
non-overlapping 0.1-second windows; there is no resampling path in this module.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.io import loadmat


WINDOW_SECONDS = 0.1


@dataclass(frozen=True)
class DomainDefinition:
    name: str
    raw_key: str
    segments_sec: Tuple[Tuple[float, float], ...] = ()


@dataclass
class WindowBatch:
    values: np.ndarray
    start_seconds: np.ndarray
    recording_names: List[str]


def _natural_key(path: Path) -> Tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def _first_matching_root(
    raw_root: Path,
    candidates: Sequence[str],
    marker: str,
) -> Path:
    options = [raw_root / name for name in candidates] + [raw_root]
    for option in options:
        if (option / marker).exists():
            return option
    return options[0]


def _window_signal(
    signal: np.ndarray,
    sampling_frequency: int,
    recording_name: str,
    segments_sec: Sequence[Tuple[float, float]] = (),
) -> WindowBatch:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.ndim != 2:
        raise ValueError(f"Expected [channels, points], got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite values found in {recording_name}")

    window_size = int(round(sampling_frequency * WINDOW_SECONDS))
    intervals = segments_sec or ((0.0, signal.shape[1] / sampling_frequency),)
    values: List[np.ndarray] = []
    starts: List[np.ndarray] = []
    names: List[str] = []
    for start_sec, stop_sec in intervals:
        start = max(0, int(round(start_sec * sampling_frequency)))
        stop = min(signal.shape[1], int(round(stop_sec * sampling_frequency)))
        count = (stop - start) // window_size
        if count <= 0:
            continue
        trimmed = signal[:, start : start + count * window_size]
        windows = trimmed.reshape(
            signal.shape[0], count, window_size
        ).transpose(1, 0, 2)
        values.append(np.ascontiguousarray(windows, dtype=np.float32))
        point_starts = start + np.arange(count, dtype=np.int64) * window_size
        starts.append(point_starts.astype(np.float64) / sampling_frequency)
        names.extend([recording_name] * count)
    if not values:
        return WindowBatch(
            np.empty((0, signal.shape[0], window_size), dtype=np.float32),
            np.empty(0, dtype=np.float64),
            [],
        )
    return WindowBatch(
        np.concatenate(values, axis=0),
        np.concatenate(starts),
        names,
    )


def _concat_batches(
    batches: Sequence[WindowBatch],
    channels: int,
    length: int,
) -> WindowBatch:
    populated = [batch for batch in batches if len(batch.values)]
    if not populated:
        return WindowBatch(
            np.empty((0, channels, length), dtype=np.float32),
            np.empty(0, dtype=np.float64),
            [],
        )
    return WindowBatch(
        np.concatenate([batch.values for batch in populated], axis=0),
        np.concatenate([batch.start_seconds for batch in populated]),
        [name for batch in populated for name in batch.recording_names],
    )


class DatasetAdapter:
    name = ""
    sampling_frequency = 0
    channel_names: Tuple[str, ...] = ()
    class_names: Tuple[str, ...] = ()
    domains: Tuple[DomainDefinition, ...] = ()

    @property
    def window_size(self) -> int:
        return int(round(self.sampling_frequency * WINDOW_SECONDS))

    @property
    def domain_names(self) -> Tuple[str, ...]:
        return tuple(domain.name for domain in self.domains)

    def domain(self, name: str) -> DomainDefinition:
        for domain in self.domains:
            if domain.name == name:
                return domain
        raise KeyError(name)

    def recording_paths(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> List[Path]:
        raise NotImplementedError

    def load_windows(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> WindowBatch:
        raise NotImplementedError

    def relative_recording_name(self, raw_root: Path, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(raw_root.resolve()))
        except ValueError:
            return str(path.resolve())

    def validate_selection(
        self,
        raw_root: Path,
        classes: Sequence[str],
        domains: Sequence[str],
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for domain_name in domains:
            for class_name in classes:
                paths = self.recording_paths(raw_root, class_name, domain_name)
                if not paths:
                    raise FileNotFoundError(
                        f"No raw recording for {self.name}/{domain_name}/{class_name}"
                    )
                missing = [str(path) for path in paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing raw recording: {missing[0]}")
                counts[f"{domain_name}/{class_name}"] = len(paths)
        return counts


PU_CLASSES = (
    "K001",
    "KA04",
    "KA15",
    "KA16",
    "KA22",
    "KA30",
    "KB23",
    "KB24",
    "KB27",
    "KI14",
    "KI16",
    "KI17",
    "KI18",
    "KI21",
)
PU_DOMAINS = (
    DomainDefinition("N15_M07_F10", "N15_M07_F10"),
    DomainDefinition("N09_M07_F10", "N09_M07_F10"),
    DomainDefinition("N15_M01_F10", "N15_M01_F10"),
    DomainDefinition("N15_M07_F04", "N15_M07_F04"),
)


class PUAdapter(DatasetAdapter):
    name = "PU"
    sampling_frequency = 64_000
    channel_names = ("vibration",)
    class_names = PU_CLASSES
    domains = PU_DOMAINS
    raw_duration_seconds = 3.9

    def _root(self, raw_root: Path) -> Path:
        return _first_matching_root(
            raw_root,
            ("PU/RM_027_PU", "PU", "RM_027_PU"),
            self.class_names[0],
        )

    def recording_paths(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> List[Path]:
        root = self._root(raw_root)
        return sorted(
            (root / class_name).glob(f"{domain_name}_{class_name}_*.mat"),
            key=_natural_key,
        )

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        content = loadmat(path)
        raw = content[path.stem]["Y"][0][0][0][6][2]
        signal = np.asarray(raw, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
        return signal

    def load_windows(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> WindowBatch:
        max_points = int(round(self.sampling_frequency * self.raw_duration_seconds))
        batches = []
        for path in self.recording_paths(raw_root, class_name, domain_name):
            signal = self._read(path)[:, :max_points]
            batches.append(
                _window_signal(
                    signal,
                    self.sampling_frequency,
                    self.relative_recording_name(raw_root, path),
                )
            )
        return _concat_batches(batches, len(self.channel_names), self.window_size)


WT_CLASSES = ("healthy", "broken", "missing_tooth", "root_crack", "wear")
WT_DOMAINS = tuple(
    DomainDefinition(str(condition), str(condition))
    for condition in (20, 25, 30, 35, 40, 45, 50, 55)
)


class WTAdapter(DatasetAdapter):
    name = "WT"
    sampling_frequency = 48_000
    channel_names = ("vibration_x", "vibration_y")
    class_names = WT_CLASSES
    domains = WT_DOMAINS
    slice_seconds = (70.0, 90.0)

    def _root(self, raw_root: Path) -> Path:
        return _first_matching_root(raw_root, ("WT",), self.class_names[0])

    def recording_paths(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> List[Path]:
        root = self._root(raw_root)
        paths = sorted((root / class_name / "1").glob(f"*_{domain_name}.MAT"))
        if len(paths) > 1:
            raise ValueError(
                f"Expected one WT file for {class_name}/{domain_name}, got {paths}"
            )
        return paths

    def load_windows(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> WindowBatch:
        paths = self.recording_paths(raw_root, class_name, domain_name)
        if not paths:
            return _concat_batches([], len(self.channel_names), self.window_size)
        path = paths[0]
        data = np.asarray(loadmat(path)["Data"], dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Expected WT Data with at least 2 channels in {path}")
        signal = data[:, :2].T
        return _window_signal(
            signal,
            self.sampling_frequency,
            self.relative_recording_name(raw_root, path),
            (self.slice_seconds,),
        )


HUST_CLASSES = ("H", "B", "M")
HUST_DOMAINS = tuple(
    DomainDefinition(f"{speed}Hz_L{load}", f"{speed}_{load}")
    for speed in (20, 25, 30, 35, 40)
    for load in range(5)
)


def _first_numeric_line(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                float(parts[0])
            except ValueError:
                continue
            return line_number
    raise ValueError(f"Cannot find numeric data rows in {path}")


class HUSTGearboxAdapter(DatasetAdapter):
    name = "HUST-Gearbox"
    sampling_frequency = 25_600
    channel_names = ("vibration_1", "vibration_2", "vibration_3")
    class_names = HUST_CLASSES
    domains = HUST_DOMAINS

    def _root(self, raw_root: Path) -> Path:
        return _first_matching_root(raw_root, ("HUST_Gearbox",), "H_20_0.txt")

    def recording_paths(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> List[Path]:
        domain = self.domain(domain_name)
        path = self._root(raw_root) / f"{class_name}_{domain.raw_key}.txt"
        return [path]

    def load_windows(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> WindowBatch:
        path = self.recording_paths(raw_root, class_name, domain_name)[0]
        data = np.loadtxt(
            path,
            skiprows=_first_numeric_line(path),
            dtype=np.float32,
        )
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(
                f"Expected time/speed plus at least 3 channels in {path}"
            )
        return _window_signal(
            data[:, -3:].T,
            self.sampling_frequency,
            self.relative_recording_name(raw_root, path),
        )


MOTOR_CLASSES = (
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
MOTOR_DOMAINS = (
    DomainDefinition(
        "steady_1000rpm_20Nm",
        "torque_circulation_20Nm_1000rpm",
        ((15.0, 25.0), (65.0, 75.0)),
    ),
    DomainDefinition("torque_2000rpm_20Nm", "torque_circulation_20Nm_2000rpm"),
    DomainDefinition("torque_3000rpm_20Nm", "torque_circulation_20Nm_3000rpm"),
    DomainDefinition("torque_3000rpm_40Nm", "torque_circulation_40Nm_3000rpm"),
    DomainDefinition("torque_2000rpm_40Nm", "torque_circulation_40Nm_2000rpm"),
    DomainDefinition("torque_1000rpm_40Nm", "torque_circulation_40Nm_1000rpm"),
    DomainDefinition("speed_1000rpm_40Nm", "speed_circulation_40Nm_1000rpm"),
    DomainDefinition("speed_2000rpm_40Nm", "speed_circulation_40Nm_2000rpm"),
    DomainDefinition("speed_3000rpm_40Nm", "speed_circulation_40Nm_3000rpm"),
    DomainDefinition("speed_2000rpm_20Nm", "speed_circulation_20Nm_2000rpm"),
    DomainDefinition("speed_1000rpm_20Nm", "speed_circulation_20Nm_1000rpm"),
)


class MCC5MotorAdapter(DatasetAdapter):
    name = "MCC5-Motor"
    sampling_frequency = 12_800
    channel_names = (
        "motor_vibration_x",
        "motor_vibration_y",
        "motor_vibration_z",
    )
    class_names = MOTOR_CLASSES
    domains = MOTOR_DOMAINS
    vibration_columns = (3, 4, 5)

    def _root(self, raw_root: Path) -> Path:
        return _first_matching_root(
            raw_root,
            ("MCC5-THU_Motor", "MCC5-THU-Motor"),
            "MCC5-THU Motor_torque_circulation",
        )

    def recording_paths(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> List[Path]:
        domain = self.domain(domain_name)
        mode = domain.raw_key.split("_", 1)[0]
        folder = self._root(raw_root) / f"MCC5-THU Motor_{mode}_circulation"
        paths = sorted(folder.glob(f"{class_name}_{domain.raw_key}*.csv"))
        if len(paths) > 1:
            raise ValueError(
                f"Expected one MCC5-Motor file for {class_name}/{domain_name}, "
                f"got {[path.name for path in paths]}"
            )
        return paths

    def load_windows(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> WindowBatch:
        path = self.recording_paths(raw_root, class_name, domain_name)[0]
        data = np.loadtxt(
            path,
            delimiter=",",
            usecols=self.vibration_columns,
            dtype=np.float32,
        )
        if data.ndim != 2 or data.shape[1] != len(self.channel_names):
            raise ValueError(f"Unexpected MCC5-Motor shape in {path}: {data.shape}")
        domain = self.domain(domain_name)
        return _window_signal(
            data.T,
            self.sampling_frequency,
            self.relative_recording_name(raw_root, path),
            domain.segments_sec,
        )


DATASETS: Dict[str, DatasetAdapter] = {
    adapter.name: adapter
    for adapter in (PUAdapter(), WTAdapter(), MCC5MotorAdapter(), HUSTGearboxAdapter())
}


ALIASES = {
    "pu": "PU",
    "wt": "WT",
    "mcc5-motor": "MCC5-Motor",
    "mcc5_thu_motor": "MCC5-Motor",
    "mcc5-thu-motor": "MCC5-Motor",
    "hust-gearbox": "HUST-Gearbox",
    "hust_gearbox": "HUST-Gearbox",
}


def resolve_dataset_name(name: str) -> str:
    if name in DATASETS:
        return name
    normalized = ALIASES.get(name.lower())
    if normalized is None:
        raise ValueError(
            f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}"
        )
    return normalized
