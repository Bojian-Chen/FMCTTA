"""NOTE adaptation with prioritized balanced replay for target streams.

This module ports the class-balanced replay core of NOTE to FMCTTA's
Transformer/LayerNorm models.  The original image implementation optionally
uses instance-aware BatchNorm; UniFreq has no BatchNorm layers, so its
normalization affine parameters are selected through the standard adaptation
parameter scope instead.
"""

import copy
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .adapters import (
    OnlineAdapter,
    configure_gradient_adaptation,
    softmax_entropy,
)


class PrioritizedBalancedReplay:
    """Pseudo-label balanced reservoir from the official NOTE implementation."""

    def __init__(self, capacity: int, num_classes: int) -> None:
        if capacity < 1:
            raise ValueError("NOTE memory capacity must be positive")
        if num_classes < 1:
            raise ValueError("NOTE num_classes must be positive")
        self.capacity = int(capacity)
        self.num_classes = int(num_classes)
        self._data: List[List[torch.Tensor]] = [
            [] for _ in range(self.num_classes)
        ]
        self.seen_per_class = [0 for _ in range(self.num_classes)]
        self.accepted = 0
        self.rejected = 0
        self.replacements = 0

    @property
    def occupancy(self) -> int:
        return sum(len(samples) for samples in self._data)

    @property
    def occupancy_per_class(self) -> List[int]:
        return [len(samples) for samples in self._data]

    def add(self, sample: torch.Tensor, pseudo_label: int) -> bool:
        """Add one sample, evicting from an overrepresented class if needed."""

        label = int(pseudo_label)
        if not 0 <= label < self.num_classes:
            raise ValueError(
                f"NOTE pseudo-label {label} is outside [0, {self.num_classes})"
            )
        self.seen_per_class[label] += 1
        stored_sample = sample.detach().to(device="cpu").clone()

        if self.occupancy < self.capacity:
            self._data[label].append(stored_sample)
            self.accepted += 1
            return True

        occupancies = self.occupancy_per_class
        largest_value = max(occupancies)
        largest_classes = [
            index
            for index, occupancy in enumerate(occupancies)
            if occupancy == largest_value
        ]
        if label not in largest_classes:
            victim_class = random.choice(largest_classes)
        else:
            # Preserve a time-uniform reservoir within an already dominant
            # pseudo-class, exactly as PBRS does in NOTE.
            probability = occupancies[label] / self.seen_per_class[label]
            if random.random() > probability:
                self.rejected += 1
                return False
            victim_class = label

        victim_index = random.randrange(len(self._data[victim_class]))
        self._data[victim_class].pop(victim_index)
        self._data[label].append(stored_sample)
        self.accepted += 1
        self.replacements += 1
        return True

    def add_batch(
        self,
        samples: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> None:
        if samples.shape[0] != pseudo_labels.shape[0]:
            raise ValueError("NOTE samples and pseudo-labels have different sizes")
        for sample, pseudo_label in zip(samples, pseudo_labels):
            self.add(sample, int(pseudo_label.item()))

    def as_tensor(self, device: torch.device) -> torch.Tensor:
        samples = [
            sample
            for class_samples in self._data
            for sample in class_samples
        ]
        if not samples:
            raise RuntimeError("Cannot train NOTE from an empty replay memory")
        return torch.stack(samples).to(device)


class NOTEAdapter(OnlineAdapter):
    """Entropy minimization over NOTE's pseudo-balanced replay memory."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        num_classes: int,
        steps: int,
        weight_decay: float,
        parameter_scope: str,
        memory_size: int,
        update_interval: int,
        temperature: float,
    ) -> None:
        if steps < 1:
            raise ValueError("NOTE adaptation steps must be positive")
        if update_interval < 1:
            raise ValueError("NOTE update interval must be positive")
        if temperature <= 0:
            raise ValueError("NOTE temperature must be positive")

        super().__init__(copy.deepcopy(model), device)
        parameters, names = configure_gradient_adaptation(
            self.model,
            parameter_scope,
        )
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.memory = PrioritizedBalancedReplay(memory_size, num_classes)
        self.steps = int(steps)
        self.parameter_scope = parameter_scope
        self.parameter_names = tuple(names)
        self.update_interval = int(update_interval)
        self.temperature = float(temperature)
        self.samples_since_update = 0
        self.adaptation_events = 0
        self.last_entropy_loss: Optional[float] = None

    def _adapt_from_memory(self) -> None:
        memory_inputs = self.memory.as_tensor(self.device)
        self.adaptation_events += 1
        self.model.eval()
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(memory_inputs)
            loss = softmax_entropy(logits / self.temperature).mean()
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                self.skipped_updates += 1
                continue
            loss.backward()
            self.optimizer.step()
            self.updates += 1
            self.last_entropy_loss = float(loss.detach().item())

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        online_outputs: List[torch.Tensor] = []
        offset = 0

        # Splitting only at update boundaries preserves NOTE's sample-count
        # schedule even when a stream batch crosses one of those boundaries.
        while offset < inputs.shape[0]:
            remaining = self.update_interval - self.samples_since_update
            end = min(offset + remaining, inputs.shape[0])
            current_inputs = inputs[offset:end]
            self.model.eval()
            with torch.no_grad():
                current_outputs = self.model(current_inputs).detach()
            online_outputs.append(current_outputs)
            self.memory.add_batch(
                current_inputs,
                current_outputs.argmax(dim=1),
            )
            consumed = end - offset
            self.samples_since_update += consumed
            offset = end

            if self.samples_since_update == self.update_interval:
                self._adapt_from_memory()
                self.samples_since_update = 0

        return torch.cat(online_outputs, dim=0)

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        values.update(
            {
                "parameter_scope": self.parameter_scope,
                "adapted_parameter_tensors": len(self.parameter_names),
                "adapted_parameters": sum(
                    parameter.numel()
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ),
                "memory_type": "PBRS",
                "memory_size": self.memory.capacity,
                "memory_occupancy": self.memory.occupancy,
                "memory_occupancy_per_class": self.memory.occupancy_per_class,
                "memory_seen_per_class": list(self.memory.seen_per_class),
                "memory_accepted_samples": self.memory.accepted,
                "memory_rejected_samples": self.memory.rejected,
                "memory_replacements": self.memory.replacements,
                "update_interval": self.update_interval,
                "samples_until_update": (
                    self.update_interval - self.samples_since_update
                ),
                "adaptation_events": self.adaptation_events,
                "temperature": self.temperature,
                "last_entropy_loss": self.last_entropy_loss,
            }
        )
        return values
