import random

import torch
import torch.nn as nn

from fmctta.adapters import build_adapter
from fmctta.note import PrioritizedBalancedReplay


class TinyNOTEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.head = nn.Linear(4, 3)

    @property
    def backbone(self):
        return self.encoder

    def enable_encoder_finetuning(self) -> None:
        self.encoder.requires_grad_(True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs.squeeze(1)))


def test_pbrs_evicts_from_the_largest_pseudo_class() -> None:
    random.seed(7)
    memory = PrioritizedBalancedReplay(capacity=4, num_classes=3)
    for value in range(4):
        memory.add(torch.tensor([float(value)]), 0)

    memory.add(torch.tensor([10.0]), 1)
    memory.add(torch.tensor([20.0]), 2)

    assert memory.occupancy == 4
    assert memory.occupancy_per_class == [2, 1, 1]
    assert memory.seen_per_class == [4, 1, 1]
    assert memory.replacements == 2


def test_note_updates_only_at_sample_interval_and_keeps_balanced_memory() -> None:
    torch.manual_seed(4)
    random.seed(4)
    adapter = build_adapter(
        "note",
        TinyNOTEModel(),
        torch.device("cpu"),
        learning_rate=1e-3,
        num_classes=3,
        note_memory_size=4,
        note_update_interval=4,
    )
    inputs = torch.randn(6, 1, 4)

    first = adapter(inputs[:2])
    assert first.shape == (2, 3)
    assert adapter.updates == 0

    second = adapter(inputs[2:])
    stats = adapter.statistics()
    assert second.shape == (4, 3)
    assert adapter.updates == 1
    assert stats["adaptation_events"] == 1
    assert stats["memory_type"] == "PBRS"
    assert stats["memory_occupancy"] == 4
    assert stats["samples_until_update"] == 2
    assert sum(stats["memory_occupancy_per_class"]) == 4
    assert stats["last_entropy_loss"] is not None
