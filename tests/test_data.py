from pathlib import Path

import numpy as np
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
