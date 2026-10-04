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
        reuse_repeated_target_samples: bool = False,
        dirichlet_beta: float = 0.1,
        dirichlet_num_chunks: Optional[int] = None,
        allow_source_target_recording_overlap: bool = False,
        source_seed: int = 2025,
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
        self.dirichlet_beta = dirichlet_beta
        self.dirichlet_num_chunks = dirichlet_num_chunks
        self.seed = seed
        self.source_seed = source_seed
        self.allow_source_target_recording_overlap = allow_source_target_recording_overlap
        self.allow_repeated_target_samples = allow_repeated_target_samples
        self.reuse_repeated_target_samples = reuse_repeated_target_samples
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
        # Source domains may reappear in the online target stream, but samples
        # used by either supervised split must remain permanently excluded.
        self._source_sample_ids = np.concatenate((train_ids, validation_ids))
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
                if domain not in self.source_domains
                and self.spec.domain_groups[domain] in source_groups
            ]
            if conflicting_targets and not self.allow_source_target_recording_overlap:
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
        normalizations = {
            "zscore",
            "minmax",
            "sample_zscore",
            "sample_minmax",
            "none",
        }
        if self.normalization not in normalizations:
            raise ValueError(
                "normalization must be zscore, minmax, sample_zscore, "
                "sample_minmax, or none"
            )
        if self.stream_order not in {
            "shuffled", "class_block", "interleaved", "dirichlet"
        }:
            raise ValueError(
                "stream_order must be shuffled, class_block, interleaved, or dirichlet"
            )
        if not np.isfinite(self.dirichlet_beta) or self.dirichlet_beta <= 0:
            raise ValueError("dirichlet_beta must be finite and positive")
        if self.dirichlet_num_chunks is not None and (
            not isinstance(self.dirichlet_num_chunks, (int, np.integer))
            or isinstance(self.dirichlet_num_chunks, bool)
            or self.dirichlet_num_chunks < 1
        ):
            raise ValueError("dirichlet_num_chunks must be a positive integer")
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
        else:
            empty_overlaps = [
                domain
                for domain in dict.fromkeys(self.target_domains)
                if domain in self.source_domains
                and any(
                    self._remaining_target_count(domain, class_id) == 0
                    for class_id in range(self.spec.num_classes)
                )
            ]
            if empty_overlaps:
                raise ValueError(
                    "Source/target domains have no unseen samples left after "
                    "source selection; reduce source_samples_per_class or "
                    f"max_source_per_class for domains: {empty_overlaps}"
                )
        if self.reuse_repeated_target_samples:
            if not self.allow_repeated_target_samples:
                raise ValueError(
                    "reuse_repeated_target_samples requires "
                    "allow_repeated_target_samples"
                )
            counts_by_domain: Dict[int, Tuple[int, ...]] = {}
            for domain, counts in zip(self.target_domains, self._target_counts()):
                previous = counts_by_domain.setdefault(domain, counts)
                if counts != previous:
                    raise ValueError(
                        "Target-forgetting probes require identical per-class "
                        f"sample counts whenever domain {domain} repeats"
                    )

    def _load_mat(self, path: Path, strict_data: bool) -> Tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        content = self._load_arrays(path)
        if "data" not in content or "label" not in content:
            raise ValueError(f"{path} must contain 'data' and 'label' arrays")

        self.window_start_sec = (
            np.asarray(content["window_start_sec"], dtype=np.float64).reshape(-1)
            if "window_start_sec" in content else None
        )
        if self.window_start_sec is not None and len(self.window_start_sec) != np.asarray(content["label"]).size:
            raise ValueError("window_start_sec must have one timestamp per sample")
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
                    for name in ("data", "label", "window_start_sec")
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
                self.source_seed
                if len(self.source_domains) == 1
                else self.source_seed + 1009 * (domain + 1)
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
        rng = np.random.RandomState(self.source_seed)
        rng.shuffle(train_ids)
        rng.shuffle(validation_ids)
        return train_ids, validation_ids

    def _represent(self, signals: np.ndarray) -> np.ndarray:
        if self.representation == "frequency":
            signals = np.abs(np.fft.fft(signals, axis=-1)).astype(np.float32)
        return signals.astype(np.float32, copy=False)

    def _fit_normalizer(self, source_train_ids: np.ndarray) -> Dict[str, np.ndarray]:
        if self.normalization in {"sample_zscore", "sample_minmax"}:
            return {}
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
        if self.normalization == "sample_zscore":
            offset = transformed.mean(axis=-1, keepdims=True)
            scale = transformed.std(axis=-1, keepdims=True)
        elif self.normalization == "sample_minmax":
            offset = transformed.min(axis=-1, keepdims=True)
            scale = transformed.max(axis=-1, keepdims=True) - offset
        else:
            offset = self._normalizer["offset"]
            scale = self._normalizer["scale"]
        transformed = (transformed - offset) / np.maximum(scale, 1e-6)
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
            tuple(
                self._remaining_target_count(domain, class_id)
                for class_id in range(self.spec.num_classes)
            )
            for domain in self.target_domains
        )

    def _remaining_target_count(self, domain: int, class_id: int) -> int:
        total = self.spec.samples_per_class_for_domain(domain)
        if domain not in self.source_domains:
            return total
        if self.source_samples_per_class is not None:
            source_count = self.source_samples_per_class[class_id]
        elif self.max_source_per_class is not None:
            source_count = min(self.max_source_per_class, total)
        else:
            source_count = total
        return total - source_count

    def _target_class_ids(
        self, domain: int, stream_position: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        ids = self._domain_indices(domain)
        if domain in self.source_domains:
            ids = ids[~np.isin(ids, self._source_sample_ids)]
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
        if self.stream_order == "dirichlet":
            return self._dirichlet_target_ids(by_class, domain, stream_position)
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

    def _dirichlet_target_ids(
        self, by_class: Sequence[np.ndarray], domain: int, stream_position: int
    ) -> np.ndarray:
        """NOTE's class-wise Dirichlet partition and class blocks within tokens.

        Matches learner/dnn.py::target_data_processing in TaesikGong/NOTE:
        shuffle each class, balance token sizes during partitioning, shuffle the
        class order inside each token, then concatenate tokens. Only already
        selected target IDs are reordered; labels never enter adaptation.
        """
        total = sum(len(ids) for ids in by_class)
        if total == 0:
            return np.empty(0, dtype=np.int64)
        chunks = self.dirichlet_num_chunks or min(self.spec.num_classes, total)
        if chunks > total:
            raise ValueError(
                "dirichlet_num_chunks cannot exceed selected target samples"
            )
        # NOTE uses 10; reduce it for small/debug streams to avoid impossible
        # rejection sampling. Bound retries instead of NOTE's unbounded loop.
        minimum = min(10, total // chunks)
        rng = np.random.RandomState(
            self.seed + 1009 * (domain + 1) + 7919 * (stream_position + 1)
        )
        for _ in range(1000):
            sizes = np.zeros(chunks, dtype=np.int64)
            partitions = [[] for _ in range(chunks)]
            for class_ids in by_class:
                ids = np.asarray(class_ids, dtype=np.int64).copy()
                if len(ids) == 0:
                    for token in partitions:
                        token.append(ids)
                    continue
                rng.shuffle(ids)
                if 1e-3 <= self.dirichlet_beta <= 1e100:
                    proportions = rng.dirichlet(np.full(chunks, self.dirichlet_beta))
                else:
                    # Avoid underflow/hangs in numpy's small-alpha sampler,
                    # and overflow for very large finite concentrations.
                    beta = self.dirichlet_beta
                    gamma = rng.standard_gamma(
                        beta + 1 if beta < 1 else beta, chunks
                    )
                    log_weights = np.log(gamma).astype(np.longdouble)
                    if beta < 1:
                        uniform = np.maximum(
                            rng.random_sample(chunks), np.finfo(float).tiny
                        )
                        log_weights += (
                            np.log(uniform).astype(np.longdouble) / beta
                        )
                    proportions = np.asarray(
                        np.exp(log_weights - log_weights.max()), dtype=float
                    )
                proportions *= sizes < total / chunks
                if proportions.sum() == 0:
                    # Underflow can leave all mass on a full token.
                    proportions = (sizes < total / chunks).astype(float)
                proportions /= proportions.sum()
                cuts = (np.cumsum(proportions) * len(ids)).astype(int)[:-1]
                for token, part in enumerate(np.split(ids, cuts)):
                    partitions[token].append(part)
                    sizes[token] += len(part)
            if sizes.min() >= minimum:
                ordered = []
                for token in partitions:
                    for class_id in rng.permutation(len(by_class)):
                        ordered.append(token[class_id])
                return np.concatenate(ordered)
        raise ValueError(
            "Could not generate a NOTE Dirichlet stream with nonempty chunks "
            "after 1000 attempts; increase dirichlet_beta or reduce dirichlet_num_chunks"
        )

    def _build_target_datasets(self) -> Dict[int, FaultDomainDataset]:
        if self.max_target_per_domain is not None:
            return {}

        cursors: Dict[int, List[int]] = {}
        available: Dict[int, Tuple[np.ndarray, ...]] = {}
        repeated_samples: Dict[int, Tuple[np.ndarray, ...]] = {}
        datasets: Dict[int, FaultDomainDataset] = {}
        for position, (domain, counts) in enumerate(
            zip(self.target_domains, self._target_counts())
        ):
            if self.reuse_repeated_target_samples and domain in repeated_samples:
                occurrence_ids = repeated_samples[domain]
            elif self.allow_repeated_target_samples:
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
            if self.reuse_repeated_target_samples and domain not in repeated_samples:
                repeated_samples[domain] = tuple(item.copy() for item in selected)
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
            "allow_source_target_recording_overlap": self.allow_source_target_recording_overlap,
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
            "reuse_repeated_target_samples": self.reuse_repeated_target_samples,
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
        if domain in self.source_domains:
            ids = ids[~np.isin(ids, self._source_sample_ids)]
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
                    class_ids[offset]
                    for offset in range(max(len(item) for item in by_class))
                    for class_ids in by_class
                    if offset < len(class_ids)
                ],
                dtype=np.int64,
            )
        elif self.stream_order == "dirichlet":
            # Select the capped sample set before ordering so changing beta
            # compares the same samples, rather than different stream prefixes.
            ids = ids[: self.max_target_per_domain]
            by_class = [
                ids[self.labels[ids] == c] for c in range(self.spec.num_classes)
            ]
            ids = self._order_target_ids(
                by_class,
                domain,
                self.target_domains.index(domain)
                if stream_position is None else stream_position,
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

    def source_output_dataset(self, domain: int) -> FaultDomainDataset:
        """Full-domain cache coverage, separate from held-out source evaluation."""
        if not 0 <= domain < self.spec.num_domains:
            raise ValueError(f"Invalid source output domain: {domain}")
        return self._make_dataset(self._domain_indices(domain), domain)

    @staticmethod
    def loader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        workers: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            generator=generator,
        )
