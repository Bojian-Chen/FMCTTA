"""Target-stream protocol regressions, using synthetic data only."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

from fmctta.config import DatasetSpec
from fmctta.data import FaultDataModule


def _write_dataset(path: Path, samples_per_class: int = 4) -> DatasetSpec:
    spec = DatasetSpec(
        "synthetic",
        path.name,
        num_classes=2,
        num_domains=3,
        samples_per_class=samples_per_class,
        default_sequence=(0, 1, 2),
        description="Dirichlet stream test",
    )
    signals = np.random.RandomState(4).randn(spec.expected_samples, 32)
    labels = np.tile(np.repeat(np.arange(2), samples_per_class), 3)
    sio.savemat(path, {"data": signals.astype(np.float32), "label": labels})
    return spec


def test_dirichlet_preserves_selected_samples_and_is_reproducible(tmp_path):
    spec = _write_dataset(tmp_path / "synthetic.mat", samples_per_class=100)
    options = dict(
        source_domain=0,
        target_domains=(1, 2),
        target_samples_per_class=((80, 40), (30, 70)),
        seed=7,
    )
    baseline = FaultDataModule(spec, tmp_path, stream_order="class_block", **options)
    first = FaultDataModule(spec, tmp_path, stream_order="dirichlet", **options)
    second = FaultDataModule(spec, tmp_path, stream_order="dirichlet", **options)
    different = FaultDataModule(
        spec, tmp_path, stream_order="dirichlet", **{**options, "seed": 8}
    )
    for domain in (1, 2):
        target = first.target_dataset(domain)
        assert (
            sorted(target.sample_ids.tolist())
            == baseline.target_dataset(domain).sample_ids.tolist()
        )
        assert torch.equal(target.sample_ids, second.target_dataset(domain).sample_ids)
        assert not torch.equal(
            target.sample_ids, different.target_dataset(domain).sample_ids
        )
        assert len(torch.unique(target.sample_ids)) == len(target)
        assert not set(target.sample_ids.tolist()) & set(
            first.source.train.sample_ids.tolist()
        )
        batches = list(
            first.loader(
                target,
                batch_size=17,
                shuffle=False,
                workers=0,
                device=torch.device("cpu"),
            )
        )
        assert torch.equal(
            torch.cat([batch[2] for batch in batches]), target.sample_ids
        )


def test_dirichlet_beta_controls_local_class_concentration(tmp_path):
    spec = _write_dataset(tmp_path / "synthetic.mat", samples_per_class=200)
    concentrations = {0.01: [], 100.0: []}
    for beta, values in concentrations.items():
        for seed in range(4):
            data = FaultDataModule(
                spec,
                tmp_path,
                source_domain=0,
                target_domains=(1,),
                stream_order="dirichlet",
                dirichlet_beta=beta,
                seed=seed,
            )
            labels = data.target_dataset(1).labels.numpy()
            values.append(np.bincount(labels[:200], minlength=2).max() / 200)
            assert np.bincount(labels).tolist() == [200, 200]
    assert np.mean(concentrations[0.01]) > 1.5 * np.mean(concentrations[100.0])


@pytest.mark.parametrize("beta", [0.1, 1e-300, 1e300])
def test_dirichlet_small_stream_and_zero_class_counts(tmp_path, beta):
    spec = _write_dataset(tmp_path / "synthetic.mat")
    data = FaultDataModule(
        spec,
        tmp_path,
        source_domain=0,
        target_domains=(1,),
        max_target_per_domain=3,
        stream_order="dirichlet",
        dirichlet_beta=beta,
        dirichlet_num_chunks=1,
    )
    target = data.target_dataset(1)
    assert target.labels.tolist() == [0, 0, 0]
    assert sorted(target.sample_ids.tolist()) == [8, 9, 10]


def test_dirichlet_repeated_domains_use_disjoint_sample_sets(tmp_path):
    spec = _write_dataset(tmp_path / "synthetic.mat")
    data = FaultDataModule(
        spec,
        tmp_path,
        source_domain=0,
        target_domains=(1, 1),
        target_samples_per_class=((2, 2), (2, 2)),
        stream_order="dirichlet",
    )
    first, second = [data.target_dataset(1, position) for position in (0, 1)]
    assert len(first) == len(second) == 4
    assert not set(first.sample_ids.tolist()) & set(second.sample_ids.tolist())


def test_dirichlet_debug_cap_keeps_sample_set_fixed_across_beta(tmp_path):
    spec = _write_dataset(tmp_path / "synthetic.mat")
    targets = []
    for beta in (0.01, 100.0):
        data = FaultDataModule(
            spec,
            tmp_path,
            source_domain=0,
            target_domains=(1,),
            max_target_per_domain=6,
            stream_order="dirichlet",
            dirichlet_beta=beta,
        )
        targets.append(data.target_dataset(1))
    assert [sorted(target.sample_ids.tolist()) for target in targets] == [
        list(range(8, 14))
    ] * 2


@pytest.mark.parametrize(
    "option,value",
    [
        ("dirichlet_beta", 0),
        ("dirichlet_beta", -1),
        ("dirichlet_beta", float("nan")),
        ("dirichlet_beta", float("inf")),
        ("dirichlet_num_chunks", 0),
        ("dirichlet_num_chunks", 1.5),
        ("dirichlet_num_chunks", True),
        ("dirichlet_num_chunks", 9),
    ],
)
def test_dirichlet_rejects_invalid_options(tmp_path, option, value):
    spec = _write_dataset(tmp_path / "synthetic.mat")
    with pytest.raises(ValueError, match="dirichlet"):
        FaultDataModule(
            spec,
            tmp_path,
            source_domain=0,
            target_domains=(1,),
            stream_order="dirichlet",
            **{option: value},
        )


@pytest.mark.parametrize("reuse", [False, True])
def test_dirichlet_repeated_reuse_preserves_selected_sets(tmp_path, reuse):
    spec = _write_dataset(tmp_path / "synthetic.mat", samples_per_class=100)
    options = dict(
        source_domain=0,
        target_domains=(1, 1),
        target_samples_per_class=((30, 70), (30, 70)),
        allow_repeated_target_samples=True,
        reuse_repeated_target_samples=reuse,
        seed=7,
    )
    baseline = FaultDataModule(spec, tmp_path, stream_order="class_block", **options)
    data = FaultDataModule(spec, tmp_path, stream_order="dirichlet", **options)
    targets = [data.target_dataset(1, position) for position in (0, 1)]
    for position, target in enumerate(targets):
        assert (
            sorted(target.sample_ids.tolist())
            == baseline.target_dataset(1, position).sample_ids.tolist()
        )
    assert sorted(targets[0].sample_ids.tolist()) == sorted(
        targets[1].sample_ids.tolist()
    )
    assert not torch.equal(targets[0].sample_ids, targets[1].sample_ids)
    assert data.sample_selection == baseline.sample_selection


@pytest.mark.parametrize("cap", [None, 6])
def test_dirichlet_source_overlap_excludes_both_source_splits(tmp_path, cap):
    spec = _write_dataset(tmp_path / "synthetic.mat", samples_per_class=10)
    options = dict(
        source_domain=0,
        target_domains=(0,),
        source_samples_per_class=(3, 4),
        max_target_per_domain=cap,
        seed=7,
    )
    baseline = FaultDataModule(spec, tmp_path, stream_order="class_block", **options)
    data = FaultDataModule(spec, tmp_path, stream_order="dirichlet", **options)
    target = data.target_dataset(0)
    source_ids = torch.cat(
        [data.source.train.sample_ids, data.source.validation.sample_ids]
    )
    assert set(source_ids.tolist()).isdisjoint(target.sample_ids.tolist())
    assert (
        sorted(target.sample_ids.tolist())
        == baseline.target_dataset(0).sample_ids.tolist()
    )
    assert data.sample_selection == baseline.sample_selection


def test_dirichlet_cap_repeats_require_explicit_reuse(tmp_path):
    spec = _write_dataset(tmp_path / "synthetic.mat", samples_per_class=100)
    options = dict(
        source_domain=0,
        target_domains=(1, 1),
        max_target_per_domain=30,
        stream_order="dirichlet",
        seed=7,
    )
    with pytest.raises(ValueError, match="Repeated target domains"):
        FaultDataModule(spec, tmp_path, **options)
    data = FaultDataModule(
        spec, tmp_path, allow_repeated_target_samples=True, **options
    )
    first, second = [data.target_dataset(1, position) for position in (0, 1)]
    assert sorted(first.sample_ids.tolist()) == sorted(second.sample_ids.tolist())
    assert not torch.equal(first.sample_ids, second.sample_ids)
