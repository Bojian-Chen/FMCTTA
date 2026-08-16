from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from fmctta.config import DatasetSpec
from fmctta.data import FaultDataModule


def _make_dataset(path: Path) -> DatasetSpec:
    spec = DatasetSpec(
        "toy",
        "toy.mat",
        num_classes=2,
        num_domains=3,
        samples_per_class=5,
        default_sequence=(0, 1, 2),
        description="test",
    )
    rng = np.random.RandomState(4)
    signals = rng.randn(spec.expected_samples, 32).astype(np.float32)
    labels = np.tile(np.repeat(np.arange(2), 5), 3)
    sio.savemat(path / spec.filename, {"data": signals, "label": labels})
    return spec


def test_source_only_normalization_and_single_pass_order(tmp_path: Path) -> None:
    spec = _make_dataset(tmp_path)
    data = FaultDataModule(
        spec,
        tmp_path,
        source_domain=0,
        target_domains=(1, 2),
        normalization="zscore",
        source_validation_ratio=0.2,
        seed=7,
    )
    assert data.signals.shape == (30, 1, 32)
    assert len(data.source.train) == 8
    assert len(data.source.validation) == 2
    first = data.target_dataset(1)
    second = data.target_dataset(1)
    assert torch.equal(first.sample_ids, second.sample_ids)
    assert len(torch.unique(first.sample_ids)) == len(first)


def test_target_cap_does_not_change_source_split(tmp_path: Path) -> None:
    spec = _make_dataset(tmp_path)
    data = FaultDataModule(
        spec,
        tmp_path,
        source_domain=0,
        target_domains=(1,),
        max_source_per_class=4,
        max_target_per_domain=3,
    )
    assert len(data.source.train) == 6
    assert len(data.source.validation) == 2
    assert len(data.target_dataset(1)) == 3


def test_per_class_sampling_partitions_repeated_target_domains(tmp_path: Path) -> None:
    spec = _make_dataset(tmp_path)
    data = FaultDataModule(
        spec,
        tmp_path,
        source_domain=0,
        target_domains=(1, 1, 1),
        source_samples_per_class=(3, 4),
        target_samples_per_class=((2, 1), (1, 2), (2, 2)),
        stream_order="class_block",
        seed=7,
    )

    assert len(data.source.train) == 5
    assert len(data.source.validation) == 2
    targets = [data.target_dataset(1, position) for position in range(3)]
    assert [len(target) for target in targets] == [3, 3, 4]
    assert [
        torch.bincount(target.labels, minlength=2).tolist() for target in targets
    ] == [[2, 1], [1, 2], [2, 2]]
    all_ids = torch.cat([target.sample_ids for target in targets])
    assert len(torch.unique(all_ids)) == len(all_ids)


def test_repeated_target_request_cannot_reuse_samples(tmp_path: Path) -> None:
    spec = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match="distinct samples"):
        FaultDataModule(
            spec,
            tmp_path,
            source_domain=0,
            target_domains=(1, 1),
            target_samples_per_class=((3, 3), (3, 3)),
        )
