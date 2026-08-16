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
        source_domain: int,
        target_domains: Sequence[int],
        normalization: str = "zscore",
        source_validation_ratio: float = 0.2,
        stream_order: str = "shuffled",
        seed: int = 2025,
        strict_data: bool = False,
        max_source_per_class: Optional[int] = None,
        max_target_per_domain: Optional[int] = None,
    ) -> None:
        self.spec = spec
        self.source_domain = int(source_domain)
        self.target_domains = tuple(int(domain) for domain in target_domains)
        self.normalization = normalization
        self.source_validation_ratio = source_validation_ratio
        self.stream_order = stream_order
        self.seed = seed
        self.max_source_per_class = max_source_per_class
        self.max_target_per_domain = max_target_per_domain
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

    def _validate_options(self) -> None:
        if self.source_domain in self.target_domains:
            raise ValueError(
                "The source domain cannot also appear in the target stream"
            )
        all_domains = (self.source_domain,) + self.target_domains
        invalid = [d for d in all_domains if not 0 <= d < self.spec.num_domains]
        if invalid:
            raise ValueError(f"Invalid domains for {self.spec.name}: {invalid}")
        if not 0.0 < self.source_validation_ratio < 1.0:
            raise ValueError("source_validation_ratio must be between 0 and 1")
        if self.normalization not in {"zscore", "minmax", "none"}:
            raise ValueError("normalization must be zscore, minmax, or none")
        if self.stream_order not in {"shuffled", "class_block", "interleaved"}:
            raise ValueError(
                "stream_order must be shuffled, class_block, or interleaved"
            )

    def _load_mat(self, path: Path, strict_data: bool) -> Tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        content = sio.loadmat(path)
        if "data" not in content or "label" not in content:
            raise ValueError(f"{path} must contain 'data' and 'label' arrays")

        signals = np.asarray(content["data"], dtype=np.float32)
        signals = np.squeeze(signals)
        labels = np.asarray(content["label"]).reshape(-1).astype(np.int64)
        if signals.ndim not in {2, 3}:
            raise ValueError(
                "Expected [samples, signal_length] or "
                f"[samples, channels, signal_length], got {signals.shape}"
            )

        channels = 1 if signals.ndim == 2 else signals.shape[1]
        if channels != self.spec.num_channels:
            raise ValueError(
                f"{self.spec.name}: metadata declares {self.spec.num_channels} "
                f"channels, but data contains {channels}"
            )

        if labels.shape[0] != signals.shape[0]:
            if not self.spec.repair_labels or strict_data:
                raise ValueError(
                    f"{self.spec.name}: {signals.shape[0]} signals but "
                    f"{labels.shape[0]} labels"
                )
            labels = np.tile(
                np.repeat(
                    np.arange(self.spec.num_classes),
                    self.spec.samples_per_class,
                ),
                self.spec.num_domains,
            )
            message = (
                f"{self.spec.name}: reconstructed {labels.shape[0]} labels from the "
                "declared domain/class block layout because the MAT label length "
                "does not match the signal count"
            )
            warnings.warn(message)
            self.data_warnings.append(message)
        return signals, labels

    def _validate_domain_layout(self) -> None:
        if self.raw_signals.shape[0] != self.spec.expected_samples:
            raise ValueError(
                f"{self.spec.name}: expected {self.spec.expected_samples} samples "
                f"from metadata, found {self.raw_signals.shape[0]}"
            )
        expected_classes = np.arange(self.spec.num_classes)
        for domain in range(self.spec.num_domains):
            ids = self._domain_indices(domain)
            labels, counts = np.unique(self.labels[ids], return_counts=True)
            if not np.array_equal(labels, expected_classes) or not np.all(
                counts == self.spec.samples_per_class
            ):
                raise ValueError(
                    f"{self.spec.name} domain {domain} is not a balanced class block: "
                    f"labels={labels.tolist()}, counts={counts.tolist()}"
                )

    def _domain_indices(self, domain: int) -> np.ndarray:
        start = domain * self.spec.samples_per_domain
        stop = start + self.spec.samples_per_domain
        return np.arange(start, stop, dtype=np.int64)

    def _source_split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.seed)
        domain_ids = self._domain_indices(self.source_domain)
        train: List[np.ndarray] = []
        validation: List[np.ndarray] = []
        for class_id in range(self.spec.num_classes):
            class_ids = domain_ids[self.labels[domain_ids] == class_id].copy()
            rng.shuffle(class_ids)
            if self.max_source_per_class is not None:
                class_ids = class_ids[: self.max_source_per_class]
            if class_ids.shape[0] < 2:
                raise ValueError(
                    "Each source class needs at least two samples for train/validation"
                )
            validation_size = max(
                1, int(round(class_ids.shape[0] * self.source_validation_ratio))
            )
            validation_size = min(validation_size, class_ids.shape[0] - 1)
            validation.append(class_ids[:validation_size])
            train.append(class_ids[validation_size:])
        train_ids = np.concatenate(train)
        validation_ids = np.concatenate(validation)
        rng.shuffle(train_ids)
        rng.shuffle(validation_ids)
        return train_ids, validation_ids

    def _fit_normalizer(self, source_train_ids: np.ndarray) -> Dict[str, np.ndarray]:
        source = self.raw_signals[source_train_ids]
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
        scale = np.maximum(self._normalizer["scale"], 1e-6)
        transformed = (signals - self._normalizer["offset"]) / scale
        tensor = torch.from_numpy(np.ascontiguousarray(transformed)).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        return tensor

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

    def target_dataset(self, domain: int) -> FaultDomainDataset:
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
                    for offset in range(self.spec.samples_per_class)
                    for class_id in range(self.spec.num_classes)
                ],
                dtype=np.int64,
            )
        if self.max_target_per_domain is not None:
            ids = ids[: self.max_target_per_domain]
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
