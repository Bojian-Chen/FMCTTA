import pytest

from fmctta.config import (
    DatasetSpec,
    parse_source_samples_per_class,
    parse_target_samples_per_class,
    resolve_sequence,
)


def test_repeated_target_domains_are_valid_in_sequence() -> None:
    spec = DatasetSpec(
        "toy",
        "toy.mat",
        num_classes=2,
        num_domains=4,
        samples_per_class=5,
        default_sequence=(0, 1),
        description="test",
    )
    assert resolve_sequence(spec, "0,1,2,3,1,3,2,3") == (
        0,
        1,
        2,
        3,
        1,
        3,
        2,
        3,
    )


def test_per_class_sample_cli_parsing() -> None:
    assert parse_source_samples_per_class("3,4", 2) == (3, 4)
    assert parse_target_samples_per_class("2;1,3", 2, 2) == ((2, 2), (1, 3))


def test_target_sample_cli_requires_an_entry_for_each_stream_position() -> None:
    with pytest.raises(ValueError, match="target entries"):
        parse_target_samples_per_class("2;3", 2, 3)
