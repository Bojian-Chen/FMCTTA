"""Raw-data readers for configurable, native-rate CTTA signal windows."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat


DEFAULT_WINDOW_SIZE_POINTS = 1024


@dataclass(frozen=True)
class DomainDefinition:
    name: str
    raw_key: str
    segments_sec: Tuple[Tuple[float, float], ...] = ()
    role: str = "runtime"
    recording_group: str = ""


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
    window_size_points: int = DEFAULT_WINDOW_SIZE_POINTS,
    segments_sec: Sequence[Tuple[float, float]] = (),
    stride_points: Optional[int] = None,
) -> WindowBatch:
    """Split a recording into fixed-size windows at the requested stride."""

    if window_size_points < 1:
        raise ValueError("window_size_points must be positive")
    stride_points = window_size_points if stride_points is None else stride_points
    if stride_points < 1:
        raise ValueError("stride_points must be positive")
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.ndim != 2:
        raise ValueError(f"Expected [channels, points], got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite values found in {recording_name}")

    intervals = segments_sec or ((0.0, signal.shape[1] / sampling_frequency),)
    values: List[np.ndarray] = []
    starts: List[np.ndarray] = []
    names: List[str] = []
    for start_sec, stop_sec in intervals:
        start = max(0, int(round(start_sec * sampling_frequency)))
        stop = min(signal.shape[1], int(round(stop_sec * sampling_frequency)))
        available_points = stop - start
        count = (
            0
            if available_points < window_size_points
            else 1 + (available_points - window_size_points) // stride_points
        )
        if count <= 0:
            continue
        point_starts = start + np.arange(count, dtype=np.int64) * stride_points
        relative_starts = point_starts - start
        interval = signal[:, start:stop]
        windows = np.lib.stride_tricks.sliding_window_view(
            interval,
            window_size_points,
            axis=1,
        )[:, relative_starts, :].transpose(1, 0, 2)
        values.append(np.ascontiguousarray(windows, dtype=np.float32))
        starts.append(point_starts.astype(np.float64) / sampling_frequency)
        names.extend([recording_name] * count)
    if not values:
        return WindowBatch(
            np.empty((0, signal.shape[0], window_size_points), dtype=np.float32),
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
    condition_profile: Tuple[Dict[str, object], ...] = ()
    recommended_source_domains: Tuple[str, ...] = ()

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
        self,
        raw_root: Path,
        class_name: str,
        domain_name: str,
        window_size_points: int,
        stride_points: Optional[int] = None,
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
        self,
        raw_root: Path,
        class_name: str,
        domain_name: str,
        window_size_points: int,
        stride_points: Optional[int] = None,
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
                    window_size_points,
                    stride_points=stride_points,
                )
            )
        return _concat_batches(batches, len(self.channel_names), window_size_points)


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
        self,
        raw_root: Path,
        class_name: str,
        domain_name: str,
        window_size_points: int,
        stride_points: Optional[int] = None,
    ) -> WindowBatch:
        paths = self.recording_paths(raw_root, class_name, domain_name)
        if not paths:
            return _concat_batches([], len(self.channel_names), window_size_points)
        path = paths[0]
        data = np.asarray(loadmat(path)["Data"], dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Expected WT Data with at least 2 channels in {path}")
        return _window_signal(
            data[:, :2].T,
            self.sampling_frequency,
            self.relative_recording_name(raw_root, path),
            window_size_points,
            (self.slice_seconds,),
            stride_points,
        )


SK_CLASSES = (
    "NC",
    "IF0.2",
    "IF0.4",
    "IF0.6",
    "OF0.2",
    "OF0.4",
    "OF0.6",
    "RF0.2",
    "RF0.4",
    "RF0.6",
)
SK_DOMAINS = tuple(
    DomainDefinition(f"{speed}rpm_{load}N", f"{speed}|{load}")
    for speed in (500, 1000, 1500, 2000, 2500, 3000)
    for load in (0, 20, 40, 60)
)


class SKAdapter(DatasetAdapter):
    """Reader for the three-channel Shanke bearing recordings."""

    name = "SK"
    sampling_frequency = 25_600
    channel_names = ("point2_z", "point3_y", "another_y")
    class_names = SK_CLASSES
    domains = SK_DOMAINS
    # Preserve the source domain used by the legacy SK protocol (domain 6).
    recommended_source_domains = (SK_DOMAINS[6].name,)

    def _root(self, raw_root: Path) -> Path:
        return _first_matching_root(
            raw_root,
            ("SK", "山科数据集"),
            "NC/500 0.mat",
        )

    def recording_paths(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> List[Path]:
        speed, load = self.domain(domain_name).raw_key.split("|")
        # Exact construction deliberately excludes flu recordings and extra
        # repeat files such as ``3000 0 2.mat`` from the 24 fixed domains.
        return [self._root(raw_root) / class_name / f"{speed} {load}.mat"]

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        content = loadmat(path, squeeze_me=True, struct_as_record=False)
        try:
            values = content["Signal"].y_values.values
        except (AttributeError, KeyError) as exc:
            raise KeyError(
                f"Cannot find Signal.y_values.values in {path}"
            ) from exc
        data = np.asarray(values, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f"Expected SK signal with 3 channels in {path}, got {data.shape}"
            )
        return data.T

    def load_windows(
        self,
        raw_root: Path,
        class_name: str,
        domain_name: str,
        window_size_points: int,
        stride_points: Optional[int] = None,
    ) -> WindowBatch:
        path = self.recording_paths(raw_root, class_name, domain_name)[0]
        return _window_signal(
            self._read(path),
            self.sampling_frequency,
            self.relative_recording_name(raw_root, path),
            window_size_points,
            stride_points=stride_points,
        )


SDUST_CHANNELS = tuple(f"signal_{index}" for index in range(1, 7))
SDUST_BEARING_CHANNELS = (
    "upper_z",
    "upper_y",
    "upper_x",
    "lower_z",
    "lower_y",
    "lower_x",
)
SDUST_GEAR_CHANNELS = (
    "left_z",
    "left_y",
    "left_x",
    "right_z",
    "right_y",
    "right_x",
)
SDUST_BEARING_CLASSES = (
    "NC",
    "IF0.2",
    "IF0.4",
    "IF0.6",
    "OF0.2",
    "OF0.4",
    "OF0.6",
    "RF0.2",
    "RF0.4",
    "RF0.6",
)
SDUST_BEARING_DOMAINS = tuple(
    DomainDefinition(f"{speed}rpm_{load}N", f"{speed}|{load}")
    for speed in (1000, 1500, 1800, 2000, 2500, 3000)
    for load in (0, 20, 40, 60)
) + tuple(
    DomainDefinition(
        f"speed_{start}to{stop}rpm_{load}N",
        f"{start}~{stop}|{load}",
    )
    for start, stop in ((800, 1500), (1000, 2000), (1500, 2500))
    for load in (0, 20, 40, 60)
)
SDUST_GEAR_CLASSES = (
    "NC",
    "太阳断裂",
    "太阳点蚀",
    "太阳磨损",
    "行星断裂",
    "行星点蚀",
    "行星磨损",
)
SDUST_GEAR_DOMAINS = tuple(
    DomainDefinition(f"{speed}rpm_0.5A", f"{speed}|0.5A|1")
    for speed in (1000, 1500, 1800, 2000, 2500)
) + tuple(
    DomainDefinition(
        f"{speed}rpm_load_0to{load}",
        f"{speed}|0-{load}|*",
    )
    for speed in (1000, 1500, 1800, 2000, 2500)
    for load in ("0.2A", "0.35A", "0.5A")
) + (
    DomainDefinition("fluctuating_speed_and_load", "flu|flu|*"),
)
SDUST_GEAR_STEMS = {
    "NC": "normal",
    "太阳断裂": "sunfracture",
    "太阳点蚀": "sunpitting",
    "太阳磨损": "sunwear",
    "行星断裂": "planetrayfracture",
    "行星点蚀": "planetraypitting",
    "行星磨损": "planetraywear",
}


def _read_sdust_signal(path: Path) -> np.ndarray:
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    try:
        values = content["Signal"].y_values.values
    except (AttributeError, KeyError) as exc:
        raise KeyError(f"Cannot find Signal.y_values.values in {path}") from exc
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != len(SDUST_CHANNELS):
        raise ValueError(
            f"Expected SDUST signal with 6 channels in {path}, got {data.shape}"
        )
    return data.T


class SDUSTAdapter(DatasetAdapter):
    sampling_frequency = 25_600
    channel_names = SDUST_CHANNELS
    subset_dir_name = ""

    def _root(self, raw_root: Path) -> Path:
        return _first_matching_root(
            raw_root,
            ("SUDST", "SDUST"),
            f"{self.subset_dir_name}/NC",
        )

    def load_windows(
        self,
        raw_root: Path,
        class_name: str,
        domain_name: str,
        window_size_points: int,
        stride_points: Optional[int] = None,
    ) -> WindowBatch:
        paths = self.recording_paths(raw_root, class_name, domain_name)
        batches = [
            _window_signal(
                _read_sdust_signal(path),
                self.sampling_frequency,
                self.relative_recording_name(raw_root, path),
                window_size_points,
                stride_points=stride_points,
            )
            for path in paths
        ]
        return _concat_batches(
            batches,
            len(self.channel_names),
            window_size_points,
        )


class SDUSTBearingAdapter(SDUSTAdapter):
    name = "SDUST-Bearing"
    subset_dir_name = "轴承数据集"
    channel_names = SDUST_BEARING_CHANNELS
    class_names = SDUST_BEARING_CLASSES
    domains = SDUST_BEARING_DOMAINS

    def recording_paths(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> List[Path]:
        speed, load = self.domain(domain_name).raw_key.split("|")
        label_dir = self._root(raw_root) / self.subset_dir_name / class_name
        if "~" not in speed:
            return [label_dir / f"{class_name} {speed} {load}.mat"]
        paths = []
        for path in label_dir.glob("*.mat"):
            parts = path.stem.split()
            if len(parts) < 3:
                continue
            recorded_speed = parts[1].replace("-", "~")
            if (
                parts[0] == class_name
                and recorded_speed == speed
                and parts[2] == load
            ):
                paths.append(path)
        return sorted(paths, key=_natural_key)


class SDUSTGearAdapter(SDUSTAdapter):
    name = "SDUST-Gear"
    subset_dir_name = "齿轮数据集"
    channel_names = SDUST_GEAR_CHANNELS
    class_names = SDUST_GEAR_CLASSES
    domains = SDUST_GEAR_DOMAINS

    def recording_paths(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> List[Path]:
        speed, load, repeat = self.domain(domain_name).raw_key.split("|")
        stem = SDUST_GEAR_STEMS[class_name]
        label_dir = self._root(raw_root) / self.subset_dir_name / class_name
        if repeat != "*":
            return [label_dir / f"{stem} {speed} {load} {repeat}.mat"]
        return sorted(
            label_dir.glob(f"{stem} {speed} {load} *.mat"),
            key=_natural_key,
        )


HUST_CLASSES = ("H", "B", "M")
HUST_DOMAINS = tuple(
    DomainDefinition(f"{speed}Hz_L{load}", f"{speed}_{load}")
    for speed in (20, 25, 30, 35, 40)
    for load in range(5)
) + tuple(
    DomainDefinition(f"0~40~0Hz_L{load}", f"L{load}_VS_0_40_0")
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
        return [self._root(raw_root) / f"{class_name}_{domain.raw_key}.txt"]

    def load_windows(
        self,
        raw_root: Path,
        class_name: str,
        domain_name: str,
        window_size_points: int,
        stride_points: Optional[int] = None,
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
            window_size_points,
            stride_points=stride_points,
        )


GEARBOX_CLASSES = (
    "health",
    "gear_pitting_H",
    "gear_pitting_M",
    "gear_pitting_L",
    "gear_wear_H",
    "gear_wear_M",
    "gear_wear_L",
    "miss_teeth",
    "teeth_break_H",
    "teeth_break_M",
    "teeth_break_L",
    "teeth_crack_H",
    "teeth_crack_M",
    "teeth_crack_L",
    "teeth_break_and_bearing_inner_H",
    "teeth_break_and_bearing_inner_M",
    "teeth_break_and_bearing_inner_L",
    "teeth_break_and_bearing_outer_H",
    "teeth_break_and_bearing_outer_M",
    "teeth_break_and_bearing_outer_L",
)


def _mcc5_chronological_domains(profiles, condition_profile):
    """Seven temporal segments per recording, followed by every full stream."""
    segments = []
    streams = []
    for profile_name, raw_key, steady_1, steady_2 in profiles:
        for index, phase in enumerate(condition_profile):
            start, stop = float(phase['start_sec']), float(phase['stop_sec'])
            phase_name = phase['phase']
            if phase_name == 'steady_1':
                phase_name += f"_{steady_1}"
            elif phase_name == 'steady_2':
                phase_name += f"_{steady_2}"
            # The two steady-1 intervals are now separate domains.
            name = f"{profile_name}__{phase_name}__{start:g}to{stop:g}s"
            segments.append(DomainDefinition(
                name, raw_key, ((start, stop),),
                'transition_segment' if phase['phase'].startswith('linear_') else 'source_candidate',
                raw_key,
            ))
        streams.append(DomainDefinition(
            f"{profile_name}__stream", raw_key, (), 'target_stream', raw_key,
        ))
    return tuple(segments + streams)


GEARBOX_STEADY_1_SEGMENTS = ((10.0, 20.0), (40.0, 50.0))
GEARBOX_STEADY_2_SEGMENTS = ((25.0, 35.0),)
GEARBOX_PROFILES = (
    ("torque_1000rpm_10to5Nm", "torque_circulation_1000rpm_10Nm", "10Nm", "5Nm"),
    ("torque_2000rpm_10to5Nm", "torque_circulation_2000rpm_10Nm", "10Nm", "5Nm"),
    ("torque_3000rpm_10to5Nm", "torque_circulation_3000rpm_10Nm", "10Nm", "5Nm"),
    ("torque_3000rpm_20to15Nm", "torque_circulation_3000rpm_20Nm", "20Nm", "15Nm"),
    ("torque_2000rpm_20to15Nm", "torque_circulation_2000rpm_20Nm", "20Nm", "15Nm"),
    ("torque_1000rpm_20to15Nm", "torque_circulation_1000rpm_20Nm", "20Nm", "15Nm"),
    ("speed_20Nm_1000to500rpm", "speed_circulation_20Nm-1000rpm", "1000rpm", "500rpm"),
    ("speed_20Nm_2000to1500rpm", "speed_circulation_20Nm-2000rpm", "2000rpm", "1500rpm"),
    ("speed_20Nm_3000to2500rpm", "speed_circulation_20Nm-3000rpm", "3000rpm", "2500rpm"),
    ("speed_10Nm_3000to2500rpm", "speed_circulation_10Nm-3000rpm", "3000rpm", "2500rpm"),
    ("speed_10Nm_2000to1500rpm", "speed_circulation_10Nm-2000rpm", "2000rpm", "1500rpm"),
    ("speed_10Nm_1000to500rpm", "speed_circulation_10Nm-1000rpm", "1000rpm", "500rpm"),
)
GEARBOX_CONDITION_PROFILE = (
    {"phase": "linear_zero_to_steady_1", "start_sec": 0.0, "stop_sec": 10.0},
    {"phase": "steady_1", "start_sec": 10.0, "stop_sec": 20.0},
    {
        "phase": "linear_steady_1_to_steady_2",
        "start_sec": 20.0,
        "stop_sec": 25.0,
    },
    {"phase": "steady_2", "start_sec": 25.0, "stop_sec": 35.0},
    {
        "phase": "linear_steady_2_to_steady_1",
        "start_sec": 35.0,
        "stop_sec": 40.0,
    },
    {"phase": "steady_1", "start_sec": 40.0, "stop_sec": 50.0},
    {"phase": "linear_steady_1_to_zero", "start_sec": 50.0, "stop_sec": 60.0},
)


GEARBOX_DOMAINS = _mcc5_chronological_domains(GEARBOX_PROFILES, GEARBOX_CONDITION_PROFILE)


class MCC5GearboxAdapter(DatasetAdapter):
    name = "MCC5-Gearbox"
    sampling_frequency = 12_800
    channel_names = (
        "motor_vibration_x",
        "motor_vibration_y",
        "motor_vibration_z",
        "gearbox_vibration_x",
        "gearbox_vibration_y",
        "gearbox_vibration_z",
    )
    class_names = GEARBOX_CLASSES
    domains = GEARBOX_DOMAINS
    condition_profile = GEARBOX_CONDITION_PROFILE
    recommended_source_domains = tuple(GEARBOX_DOMAINS[i].name for i in (1, 3, 5))
    vibration_columns = (2, 3, 4, 5, 6, 7)

    @staticmethod
    @lru_cache(maxsize=len(GEARBOX_CLASSES))
    def _read_recording(path: Path) -> np.ndarray:
        return np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=MCC5GearboxAdapter.vibration_columns,
            dtype=np.float32,
        )

    def _root(self, raw_root: Path) -> Path:
        return _first_matching_root(
            raw_root,
            ("MCC5-THU_Gearbox", "MCC5"),
            "health_torque_circulation_1000rpm_10Nm.csv",
        )

    def recording_paths(
        self, raw_root: Path, class_name: str, domain_name: str
    ) -> List[Path]:
        domain = self.domain(domain_name)
        return [self._root(raw_root) / f"{class_name}_{domain.raw_key}.csv"]

    def load_windows(
        self,
        raw_root: Path,
        class_name: str,
        domain_name: str,
        window_size_points: int,
        stride_points: Optional[int] = None,
    ) -> WindowBatch:
        path = self.recording_paths(raw_root, class_name, domain_name)[0]
        data = self._read_recording(path)
        if data.ndim != 2 or data.shape[1] != len(self.channel_names):
            raise ValueError(f"Unexpected MCC5-Gearbox shape in {path}: {data.shape}")
        domain = self.domain(domain_name)
        return _window_signal(
            data.T,
            self.sampling_frequency,
            self.relative_recording_name(raw_root, path),
            window_size_points,
            domain.segments_sec,
            stride_points,
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
MOTOR_STEADY_1_SEGMENTS = ((10.0, 30.0), (60.0, 80.0))
MOTOR_STEADY_2_SEGMENTS = ((35.0, 55.0),)
MOTOR_PROFILES = (
    ("torque_1000rpm_20to15Nm", "torque_circulation_20Nm_1000rpm", "20Nm", "15Nm"),
    ("torque_2000rpm_20to15Nm", "torque_circulation_20Nm_2000rpm", "20Nm", "15Nm"),
    ("torque_3000rpm_20to15Nm", "torque_circulation_20Nm_3000rpm", "20Nm", "15Nm"),
    ("torque_3000rpm_40to30Nm", "torque_circulation_40Nm_3000rpm", "40Nm", "30Nm"),
    ("torque_2000rpm_40to30Nm", "torque_circulation_40Nm_2000rpm", "40Nm", "30Nm"),
    ("torque_1000rpm_40to30Nm", "torque_circulation_40Nm_1000rpm", "40Nm", "30Nm"),
    ("speed_40Nm_1000to500rpm", "speed_circulation_40Nm_1000rpm", "1000rpm", "500rpm"),
    ("speed_40Nm_2000to1500rpm", "speed_circulation_40Nm_2000rpm", "2000rpm", "1500rpm"),
    ("speed_40Nm_3000to2500rpm", "speed_circulation_40Nm_3000rpm", "3000rpm", "2500rpm"),
    ("speed_20Nm_2000to1500rpm", "speed_circulation_20Nm_2000rpm", "2000rpm", "1500rpm"),
    ("speed_20Nm_1000to500rpm", "speed_circulation_20Nm_1000rpm", "1000rpm", "500rpm"),
)
MOTOR_CONDITION_PROFILE = (
    {"phase": "linear_zero_to_steady_1", "start_sec": 0.0, "stop_sec": 10.0},
    {"phase": "steady_1", "start_sec": 10.0, "stop_sec": 30.0},
    {
        "phase": "linear_steady_1_to_steady_2",
        "start_sec": 30.0,
        "stop_sec": 35.0,
    },
    {"phase": "steady_2", "start_sec": 35.0, "stop_sec": 55.0},
    {
        "phase": "linear_steady_2_to_steady_1",
        "start_sec": 55.0,
        "stop_sec": 60.0,
    },
    {"phase": "steady_1", "start_sec": 60.0, "stop_sec": 80.0},
    {"phase": "linear_steady_1_to_zero", "start_sec": 80.0, "stop_sec": 90.0},
)


MOTOR_DOMAINS = _mcc5_chronological_domains(MOTOR_PROFILES, MOTOR_CONDITION_PROFILE)


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
    condition_profile = MOTOR_CONDITION_PROFILE
    recommended_source_domains = tuple(MOTOR_DOMAINS[i].name for i in (1, 3, 5))
    vibration_columns = (3, 4, 5)

    @staticmethod
    @lru_cache(maxsize=len(MOTOR_CLASSES))
    def _read_recording(path: Path) -> np.ndarray:
        return np.loadtxt(
            path,
            delimiter=",",
            usecols=MCC5MotorAdapter.vibration_columns,
            dtype=np.float32,
        )

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
            raise ValueError(f"Expected one MCC5-Motor file for {class_name}/{domain_name}, got {[path.name for path in paths]}")
        return paths

    def load_windows(
        self,
        raw_root: Path,
        class_name: str,
        domain_name: str,
        window_size_points: int,
        stride_points: Optional[int] = None,
    ) -> WindowBatch:
        path = self.recording_paths(raw_root, class_name, domain_name)[0]
        data = self._read_recording(path)
        if data.ndim != 2 or data.shape[1] != len(self.channel_names):
            raise ValueError(f"Unexpected MCC5-Motor shape in {path}: {data.shape}")
        domain = self.domain(domain_name)
        return _window_signal(
            data.T,
            self.sampling_frequency,
            self.relative_recording_name(raw_root, path),
            window_size_points,
            domain.segments_sec,
            stride_points,
        )


MOTOR_LEGACY_DOMAINS = (
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


class MCC5MotorLegacyAdapter(MCC5MotorAdapter):
    """Compatibility adapter for FMCTTA's earlier 0.1-second motor protocol."""

    name = "MCC5-Motor-Legacy"
    domains = MOTOR_LEGACY_DOMAINS
    recommended_source_domains = (MOTOR_LEGACY_DOMAINS[0].name,)


DATASETS: Dict[str, DatasetAdapter] = {
    adapter.name: adapter
    for adapter in (
        PUAdapter(),
        WTAdapter(),
        SDUSTBearingAdapter(),
        SDUSTGearAdapter(),
        MCC5GearboxAdapter(),
        MCC5MotorAdapter(),
        MCC5MotorLegacyAdapter(),
        HUSTGearboxAdapter(),
        SKAdapter(),
    )
}


ALIASES = {
    "pu": "PU",
    "wt": "WT",
    "sk": "SK",
    "sdust-bearing": "SDUST-Bearing",
    "sdust_bearing": "SDUST-Bearing",
    "sudst-bearing": "SDUST-Bearing",
    "sudst_bearing": "SDUST-Bearing",
    "sdust-gear": "SDUST-Gear",
    "sdust_gear": "SDUST-Gear",
    "sudst-gear": "SDUST-Gear",
    "sudst_gear": "SDUST-Gear",
    "mcc5-motor": "MCC5-Motor",
    "mcc5-motor-legacy": "MCC5-Motor-Legacy",
    "mcc5_motor_legacy": "MCC5-Motor-Legacy",
    "mcc5_thu_motor": "MCC5-Motor",
    "mcc5-thu-motor": "MCC5-Motor",
    "mcc5-gearbox": "MCC5-Gearbox",
    "mcc5_thu_gearbox": "MCC5-Gearbox",
    "mcc5-thu-gearbox": "MCC5-Gearbox",
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
