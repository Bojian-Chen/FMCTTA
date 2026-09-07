import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DatasetSpec


class FaultDomainDataset(Dataset):
    """A single machine operating domain.

    Labels are returned for evaluation by the runner. Adaptation methods only
    receive the first element (the signal tensor).
    """

    def __init__(
        self,
        signals: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: torch.Tensor,
        domain: int,
    ) -> None:
        self.signals = signals
        self.labels = labels
        self.sample_ids = sample_ids
        self.domain = int(domain)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, index: int):
        return self.signals[index], self.labels[index], self.sample_ids[index]


@dataclass
class SourceSplit:
    train: FaultDomainDataset
    validation: FaultDomainDataset


class FaultDataModule:
    """Load domain-blocked MAT files without target-domain statistic leakage."""

    def __init__(
        self,
        spec: DatasetSpec,
        data_root: Path,
        source_domain: Optional[int] = None,
        target_domains: Sequence[int] = (),
        representation: str = "time",
        input_dimension: str = "1d",
        normalization: str = "zscore",
        source_validation_ratio: float = 0.2,
        stream_order: str = "shuffled",
        seed: int = 2025,
        strict_data: bool = False,
        allow_repeated_target_samples: bool = False,
        max_source_per_class: Optional[int] = None,
        max_target_per_domain: Optional[int] = None,
        source_samples_per_class: Optional[Sequence[int]] = None,
        target_samples_per_class: Optional[Sequence[Sequence[int]]] = None,
        source_domains: Optional[Sequence[int]] = None,
    ) -> None:
        self.spec = spec
        if source_domains is None:
            if source_domain is None:
                raise ValueError("At least one source domain must be specified")
            self.source_domains = (int(source_domain),)
        else:
            self.source_domains = tuple(int(domain) for domain in source_domains)
            if not self.source_domains:
                raise ValueError("At least one source domain must be specified")
            if source_domain is not None and int(source_domain) != self.source_domains[0]:
                raise ValueError("source_domain must match the first source_domains entry")
        # Retained for callers that use the legacy single-source attribute.
        self.source_domain = self.source_domains[0]
        self.target_domains = tuple(int(domain) for domain in target_domains)
        self.representation = representation
        self.input_dimension = input_dimension
        self.normalization = normalization
        self.source_validation_ratio = source_validation_ratio
        self.stream_order = stream_order
        self.seed = seed
        self.allow_repeated_target_samples = allow_repeated_target_samples
        self.max_source_per_class = max_source_per_class
        self.max_target_per_domain = max_target_per_domain
        self.source_samples_per_class = (
            None
            if source_samples_per_class is None
            else tuple(int(count) for count in source_samples_per_class)
        )
        self.target_samples_per_class = (
            None
            if target_samples_per_class is None
            else tuple(
                tuple(int(count) for count in occurrence)
                for occurrence in target_samples_per_class
            )
        )
        self.data_warnings: List[str] = []

        self._validate_options()
        raw_signals, labels = self._load_mat(spec.path(Path(data_root)), strict_data)
        self.raw_signals = raw_signals
        self.labels = labels
        self._validate_domain_layout()

        train_ids, validation_ids = self._source_split_indices()
        self._normalizer = self._fit_normalizer(train_ids)
        self.signals = self._transform(self.raw_signals)
        self.source = SourceSplit(
            self._make_dataset(train_ids, self.source_domain),
            self._make_dataset(validation_ids, self.source_domain),
        )
        self._target_datasets = self._build_target_datasets()

    def _validate_options(self) -> None:
        if len(set(self.source_domains)) != len(self.source_domains):
            raise ValueError("source_domains cannot contain duplicates")
        overlapping = sorted(set(self.source_domains).intersection(self.target_domains))
        if overlapping:
            raise ValueError(
                "Source domains cannot also appear in the target stream: "
                f"{overlapping}"
            )
        all_domains = self.source_domains + self.target_domains
        invalid = [d for d in all_domains if not 0 <= d < self.spec.num_domains]
        if invalid:
            raise ValueError(f"Invalid domains for {self.spec.name}: {invalid}")
        if self.spec.domain_groups:
            source_groups = {
                self.spec.domain_groups[domain]
                for domain in self.source_domains
                if self.spec.domain_groups[domain]
            }
            conflicting_targets = [
                domain
                for domain in self.target_domains
                if self.spec.domain_groups[domain] in source_groups
            ]
            if conflicting_targets:
                raise ValueError(
                    "Source and target domains cannot come from the same raw "
                    "recording group; conflicting target domains: "
                    f"{conflicting_targets}"
                )
        if not 0.0 < self.source_validation_ratio < 1.0:
            raise ValueError("source_validation_ratio must be between 0 and 1")
        if self.representation not in {"time", "frequency"}:
            raise ValueError("representation must be 'time' or 'frequency'")
        if self.input_dimension not in {"1d", "2d"}:
            raise ValueError("input_dimension must be '1d' or '2d'")
        if self.normalization not in {"zscore", "minmax", "none"}:
            raise ValueError("normalization must be zscore, minmax, or none")
        if self.stream_order not in {"shuffled", "class_block", "interleaved"}:
            raise ValueError(
                "stream_order must be shuffled, class_block, or interleaved"
            )
        if self.max_source_per_class is not None:
            if self.max_source_per_class < 1:
                raise ValueError("max_source_per_class must be positive")
            if self.source_samples_per_class is not None:
                raise ValueError("max_source_per_class and source_samples_per_class are mutually exclusive")
        if self.max_target_per_domain is not None:
            if self.max_target_per_domain < 1:
                raise ValueError("max_target_per_domain must be positive")
            if self.target_samples_per_class is not None:
                raise ValueError("max_target_per_domain and target_samples_per_class are mutually exclusive")
            if (
                not self.allow_repeated_target_samples
                and len(set(self.target_domains)) != len(self.target_domains)
            ):
                raise ValueError("Repeated target domains require target_samples_per_class so their samples can be allocated without reuse; set allow_repeated_target_samples to allow reuse")
        if self.source_samples_per_class is not None:
            if len(self.source_samples_per_class) != self.spec.num_classes:
                raise ValueError(
                    "source_samples_per_class must contain one count per class"
                )
            source_limit = min(
                self.spec.samples_per_class_for_domain(domain)
                for domain in self.source_domains
            )
            if any(
                count < 2 or count > source_limit
                for count in self.source_samples_per_class
            ):
                raise ValueError(
                    "Each source class count must be between 2 and "
                    f"{source_limit} for the selected source domains"
                )
        if self.target_samples_per_class is not None:
            if len(self.target_samples_per_class) != len(self.target_domains):
                raise ValueError("target_samples_per_class must contain one entry for each target-stream occurrence")
            for domain, counts in zip(
                self.target_domains, self.target_samples_per_class
            ):
                if len(counts) != self.spec.num_classes:
                    raise ValueError(
                        "Each target sample entry must contain one count per class"
                    )
                target_limit = self.spec.samples_per_class_for_domain(domain)
                if any(count < 1 or count > target_limit for count in counts):
                    raise ValueError(
                        "Each target class count must be between 1 and "
                        f"{target_limit} for target domain {domain}"
                    )

    def _load_mat(self, path: Path, strict_data: bool) -> Tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        content = self._load_arrays(path)
        if "data" not in content or "label" not in content:
            raise ValueError(f"{path} must contain 'data' and 'label' arrays")

        signals = np.asarray(content["data"], dtype=np.float32)
        signals = np.squeeze(signals)
        labels = np.asarray(content["label"]).reshape(-1).astype(np.int64)
        if signals.ndim not in {2, 3}:
            raise ValueError(f"Expected [samples, signal_length] or [samples, channels, signal_length], got {signals.shape}")

        channels = 1 if signals.ndim == 2 else signals.shape[1]
        if channels != self.spec.num_channels:
            raise ValueError(f"{self.spec.name}: metadata declares {self.spec.num_channels} channels, but data contains {channels}")

        if labels.shape[0] != signals.shape[0]:
            if not self.spec.repair_labels or strict_data:
                raise ValueError(f"{self.spec.name}: {signals.shape[0]} signals but {labels.shape[0]} labels")
            labels = np.concatenate(
                [
                    np.repeat(
                        np.arange(self.spec.num_classes),
                        self.spec.samples_per_class_for_domain(domain),
                    )
                    for domain in range(self.spec.num_domains)
                ]
            )
            message = f"{self.spec.name}: reconstructed {labels.shape[0]} labels from the declared domain/class block layout because the MAT label length does not match the signal count"
            warnings.warn(message)
            self.data_warnings.append(message)
        return signals, labels

    @staticmethod
    def _load_arrays(path: Path) -> Dict[str, np.ndarray]:
        """Load either a MATLAB v5 file or the builder's large HDF5 container."""

        try:
            import h5py
        except ImportError:
            h5py = None  # type: ignore[assignment]

        if h5py is not None and h5py.is_hdf5(path):
            with h5py.File(path, "r") as handle:
                return {
                    name: np.asarray(handle[name])
                    for name in ("data", "label")
                    if name in handle
                }
        return sio.loadmat(path)

    def _validate_domain_layout(self) -> None:
        if self.raw_signals.shape[0] != self.spec.expected_samples:
            raise ValueError(f"{self.spec.name}: expected {self.spec.expected_samples} samples from metadata, found {self.raw_signals.shape[0]}")
        expected_classes = np.arange(self.spec.num_classes)
        for domain in range(self.spec.num_domains):
            ids = self._domain_indices(domain)
            labels, counts = np.unique(self.labels[ids], return_counts=True)
            expected_count = self.spec.samples_per_class_for_domain(domain)
            if not np.array_equal(labels, expected_classes) or not np.all(
                counts == expected_count
            ):
                raise ValueError(f"{self.spec.name} domain {domain} is not a balanced class block: labels={labels.tolist()}, counts={counts.tolist()}")

    def _domain_indices(self, domain: int) -> np.ndarray:
        start = self.spec.domain_offset(domain)
        stop = start + self.spec.samples_for_domain(domain)
        return np.arange(start, stop, dtype=np.int64)

    def _source_split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        train: List[np.ndarray] = []
        validation: List[np.ndarray] = []
        for domain in self.source_domains:
            # Keep the legacy single-source split exactly reproducible.
            rng_seed = (
                self.seed
                if len(self.source_domains) == 1
                else self.seed + 1009 * (domain + 1)
            )
            rng = np.random.RandomState(rng_seed)
            domain_ids = self._domain_indices(domain)
            for class_id in range(self.spec.num_classes):
                class_ids = domain_ids[self.labels[domain_ids] == class_id].copy()
                rng.shuffle(class_ids)
                class_limit = (
                    self.source_samples_per_class[class_id]
                    if self.source_samples_per_class is not None
                    else self.max_source_per_class
                )
                if class_limit is not None:
                    class_ids = class_ids[:class_limit]
                if class_ids.shape[0] < 2:
                    raise ValueError(
                        "Each source domain/class needs at least two samples for train/validation"
                    )
                validation_size = max(
                    1, int(round(class_ids.shape[0] * self.source_validation_ratio))
                )
                validation_size = min(validation_size, class_ids.shape[0] - 1)
                validation.append(class_ids[:validation_size])
                train.append(class_ids[validation_size:])
        train_ids = np.concatenate(train)
        validation_ids = np.concatenate(validation)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(train_ids)
        rng.shuffle(validation_ids)
        return train_ids, validation_ids

    def _represent(self, signals: np.ndarray) -> np.ndarray:
        if self.representation == "frequency":
            signals = np.abs(np.fft.fft(signals, axis=-1)).astype(np.float32)
        return signals.astype(np.float32, copy=False)

    def _fit_normalizer(self, source_train_ids: np.ndarray) -> Dict[str, np.ndarray]:
        source = self._represent(self.raw_signals[source_train_ids])
        if self.normalization == "zscore":
            return {
                "offset": source.mean(axis=0, keepdims=True),
                "scale": source.std(axis=0, keepdims=True),
            }
        if self.normalization == "minmax":
            minimum = source.min(axis=0, keepdims=True)
            return {
                "offset": minimum,
                "scale": source.max(axis=0, keepdims=True) - minimum,
            }
        return {
            "offset": np.zeros((1,) + source.shape[1:], dtype=np.float32),
            "scale": np.ones((1,) + source.shape[1:], dtype=np.float32),
        }

    def _transform(self, signals: np.ndarray) -> torch.Tensor:
        transformed = self._represent(signals)
        scale = np.maximum(self._normalizer["scale"], 1e-6)
        transformed = (transformed - self._normalizer["offset"]) / scale
        tensor = torch.from_numpy(np.ascontiguousarray(transformed)).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        if self.input_dimension == "1d":
            return tensor
        signal_length = tensor.shape[-1]
        side = int(round(signal_length ** 0.5))
        if side * side != signal_length:
            raise ValueError(
                f"Signal length {signal_length} cannot be reshaped to a square"
            )
        return tensor.reshape(tensor.shape[0], tensor.shape[1], side, side)

    def _make_dataset(
        self, sample_ids: Iterable[int], domain: int
    ) -> FaultDomainDataset:
        ids = torch.as_tensor(np.asarray(list(sample_ids)), dtype=torch.long)
        return FaultDomainDataset(
            self.signals[ids],
            torch.from_numpy(self.labels[ids.numpy()]).long(),
            ids,
            domain,
        )

    def _target_counts(self) -> Tuple[Tuple[int, ...], ...]:
        if self.target_samples_per_class is not None:
            return self.target_samples_per_class
        return tuple(
            (self.spec.samples_per_class_for_domain(domain),)
            * self.spec.num_classes
            for domain in self.target_domains
        )

    def _target_class_ids(
        self, domain: int, stream_position: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        ids = self._domain_indices(domain)
        rng = None
        if self.stream_order == "shuffled":
            seed = self.seed + 1009 * (domain + 1)
            if stream_position is not None:
                seed += 7919 * (stream_position + 1)
            rng = np.random.RandomState(seed)
        by_class = []
        for class_id in range(self.spec.num_classes):
            class_ids = ids[self.labels[ids] == class_id].copy()
            if rng is not None:
                rng.shuffle(class_ids)
            by_class.append(class_ids)
        return tuple(by_class)

    def _order_target_ids(
        self,
        by_class: Sequence[np.ndarray],
        domain: int,
        stream_position: int,
    ) -> np.ndarray:
        if self.stream_order == "class_block":
            return np.concatenate(by_class)
        if self.stream_order == "interleaved":
            return np.asarray(
                [
                    class_ids[offset]
                    for offset in range(max(len(ids) for ids in by_class))
                    for class_ids in by_class
                    if offset < len(class_ids)
                ],
                dtype=np.int64,
            )
        ids = np.concatenate(by_class)
        rng = np.random.RandomState(
            self.seed + 1009 * (domain + 1) + 7919 * (stream_position + 1)
        )
        rng.shuffle(ids)
        return ids

    def _build_target_datasets(self) -> Dict[int, FaultDomainDataset]:
        if self.max_target_per_domain is not None:
            return {}

        cursors: Dict[int, List[int]] = {}
        available: Dict[int, Tuple[np.ndarray, ...]] = {}
        datasets: Dict[int, FaultDomainDataset] = {}
        for position, (domain, counts) in enumerate(
            zip(self.target_domains, self._target_counts())
        ):
            if self.allow_repeated_target_samples:
                occurrence_ids = self._target_class_ids(domain, position)
            else:
                if domain not in available:
                    available[domain] = self._target_class_ids(domain)
                    cursors[domain] = [0] * self.spec.num_classes
                occurrence_ids = available[domain]
            selected = []
            for class_id, count in enumerate(counts):
                start = (
                    0
                    if self.allow_repeated_target_samples
                    else cursors[domain][class_id]
                )
                stop = start + count
                class_ids = occurrence_ids[class_id]
                if stop > len(class_ids):
                    request = (
                        f"{count} samples for this occurrence"
                        if self.allow_repeated_target_samples
                        else f"{stop} distinct samples across repeated occurrences"
                    )
                    raise ValueError(f"Target domain {domain}, class {class_id} requests {request}, but only {len(class_ids)} are available")
                selected.append(class_ids[start:stop])
                if not self.allow_repeated_target_samples:
                    cursors[domain][class_id] = stop
            ids = self._order_target_ids(selected, domain, position)
            datasets[position] = self._make_dataset(ids, domain)
        return datasets

    @property
    def sample_selection(self) -> Dict[str, object]:
        source_counts = (
            self.source_samples_per_class
            if self.source_samples_per_class is not None
            else (self.max_source_per_class,) * self.spec.num_classes
            if self.max_source_per_class is not None
            else (
                self.spec.samples_per_class_for_domain(self.source_domain),
            )
            * self.spec.num_classes
        )
        return {
            "source_domains": list(self.source_domains),
            "source_samples_per_class": list(source_counts),
            "source_samples_per_class_by_domain": {
                str(domain): [
                    (
                        self.source_samples_per_class[class_id]
                        if self.source_samples_per_class is not None
                        else min(
                            self.max_source_per_class,
                            self.spec.samples_per_class_for_domain(domain),
                        )
                        if self.max_source_per_class is not None
                        else self.spec.samples_per_class_for_domain(domain)
                    )
                    for class_id in range(self.spec.num_classes)
                ]
                for domain in self.source_domains
            },
            "target_samples_per_class": (
                None
                if self.max_target_per_domain is not None
                else [list(counts) for counts in self._target_counts()]
            ),
            "max_target_per_domain": self.max_target_per_domain,
            "allow_repeated_target_samples": self.allow_repeated_target_samples,
        }

    def target_dataset(
        self, domain: int, stream_position: Optional[int] = None
    ) -> FaultDomainDataset:
        if self.max_target_per_domain is None:
            if stream_position is None:
                stream_position = self.target_domains.index(domain)
            if not 0 <= stream_position < len(self.target_domains):
                raise ValueError(f"Invalid target stream position: {stream_position}")
            if self.target_domains[stream_position] != domain:
                raise ValueError(f"Target stream position {stream_position} is domain {self.target_domains[stream_position]}, not {domain}")
            return self._target_datasets[stream_position]

        ids = self._domain_indices(domain)
        rng = np.random.RandomState(self.seed + 1009 * (domain + 1))
        if self.stream_order == "shuffled":
            ids = ids.copy()
            rng.shuffle(ids)
        elif self.stream_order == "interleaved":
            by_class = [
                ids[self.labels[ids] == class_id]
                for class_id in range(self.spec.num_classes)
            ]
            ids = np.asarray(
                [
                    by_class[class_id][offset]
                    for offset in range(
                        self.spec.samples_per_class_for_domain(domain)
                    )
                    for class_id in range(self.spec.num_classes)
                ],
                dtype=np.int64,
            )
        if self.max_target_per_domain is not None:
            ids = ids[: self.max_target_per_domain]
        return self._make_dataset(ids, domain)

    def source_evaluation_dataset(self, domain: int) -> FaultDomainDataset:
        """Evaluate source domains on held-out data and other domains in full.

        This diagnostic is independent of target-stream ordering and sampling.
        It never changes training data or fits target normalization statistics.
        """
        if not 0 <= domain < self.spec.num_domains:
            raise ValueError(f"Invalid source evaluation domain: {domain}")
        ids = self._domain_indices(domain)
        if domain in self.source_domains:
            validation_ids = self.source.validation.sample_ids.numpy()
            ids = ids[np.isin(ids, validation_ids)]
        return self._make_dataset(ids, domain)

    @staticmethod
    def loader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        workers: int,
        device: torch.device,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
