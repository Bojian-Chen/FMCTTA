from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fmctta.source import load_checkpoint, save_checkpoint


@pytest.mark.parametrize("from_run_directory", [False, True])
def test_safe_checkpoint_roundtrip(
    tmp_path: Path, from_run_directory: bool
) -> None:
    model = nn.Linear(4, 2)
    expected = {"dataset": "toy", "model_id": "local/tiny"}
    destination = tmp_path / "source_checkpoint" if from_run_directory else tmp_path
    save_checkpoint(destination, model, expected)
    restored = nn.Linear(4, 2)
    metadata = load_checkpoint(tmp_path, restored, expected)
    assert metadata == expected
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)


def test_checkpoint_metadata_mismatch_is_rejected(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    save_checkpoint(tmp_path, model, {"dataset": "one"})
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(tmp_path, model, {"dataset": "two"})
