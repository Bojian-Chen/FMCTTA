"""RoTTA memory and robust BatchNorm building blocks.

Adapted from the official MIT-licensed implementation at
https://github.com/BIT-DA/RoTTA; FMCTTA uses the memory helpers with UniFreq.
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Type

import torch
import torch.nn as nn

from .adapters import (
    OnlineAdapter,
    SignalAugment,
    configure_gradient_adaptation,
    soft_cross_entropy,
)


@dataclass
class CSTUMemoryItem:
    """One sample retained by category-balanced sampling by timeliness and uncertainty."""

    data: torch.Tensor
    uncertainty: float
    age: int = 0


class CSTUMemory:
    """RoTTA's category-balanced timeliness/uncertainty memory (CSTU)."""

    def __init__(
        self,
        capacity: int,
        num_classes: int,
        lambda_t: float = 1.0,
        lambda_u: float = 1.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("RoTTA memory capacity must be positive")
        if num_classes < 2:
            raise ValueError("RoTTA CSTU requires at least two classes")
        if lambda_t < 0 or lambda_u < 0:
            raise ValueError("RoTTA CSTU weights must be non-negative")
        self.capacity = capacity
        self.num_classes = num_classes
        self.per_class = capacity / num_classes
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.storage: List[List[CSTUMemoryItem]] = [[] for _ in range(num_classes)]
        self.accepted_samples = 0
        self.discarded_samples = 0

    @property
    def occupancy(self) -> int:
        return sum(len(items) for items in self.storage)

    @property
    def occupancy_per_class(self) -> List[int]:
        return [len(items) for items in self.storage]

    def heuristic_score(self, age: int, uncertainty: float) -> float:
        timeliness = 1.0 / (1.0 + math.exp(-age / self.capacity))
        normalized_uncertainty = uncertainty / math.log(self.num_classes)
        return self.lambda_t * timeliness + self.lambda_u * normalized_uncertainty

    def _majority_classes(self) -> List[int]:
        occupancies = self.occupancy_per_class
        largest = max(occupancies)
        return [
            class_index
            for class_index, occupancy in enumerate(occupancies)
            if occupancy == largest
        ]

    def _replace_from_classes(
        self,
        classes: Sequence[int],
        new_score: float,
    ) -> bool:
        worst_class = None
        worst_index = None
        worst_score = None
        for class_index in classes:
            for item_index, item in enumerate(self.storage[class_index]):
                score = self.heuristic_score(item.age, item.uncertainty)
                if worst_score is None or score >= worst_score:
                    worst_class = class_index
                    worst_index = item_index
                    worst_score = score

        if worst_class is None:
            return True
        if worst_score is not None and worst_score > new_score:
            assert worst_index is not None
            self.storage[worst_class].pop(worst_index)
            return True
        return False

    def _make_room(self, pseudo_label: int, new_score: float) -> bool:
        class_occupancy = len(self.storage[pseudo_label])
        if class_occupancy < self.per_class:
            if self.occupancy < self.capacity:
                return True
            return self._replace_from_classes(
                self._majority_classes(),
                new_score,
            )
        return self._replace_from_classes([pseudo_label], new_score)

    def add(
        self,
        sample: torch.Tensor,
        pseudo_label: int,
        uncertainty: float,
    ) -> bool:
        if not 0 <= pseudo_label < self.num_classes:
            raise ValueError(f"Invalid RoTTA pseudo-label: {pseudo_label}")
        new_score = self.heuristic_score(0, uncertainty)
        accepted = self._make_room(pseudo_label, new_score)
        if accepted:
            self.storage[pseudo_label].append(
                CSTUMemoryItem(
                    data=sample.detach().clone(),
                    uncertainty=float(uncertainty),
                )
            )
            self.accepted_samples += 1
        else:
            self.discarded_samples += 1

        # The official implementation ages all retained items, including the
        # newly inserted sample, after every stream instance.
        for class_items in self.storage:
            for item in class_items:
                item.age += 1
        return accepted

    def as_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        items = [item for class_items in self.storage for item in class_items]
        if not items:
            raise RuntimeError("RoTTA cannot adapt with an empty memory")
        samples = torch.stack([item.data for item in items])
        normalized_ages = torch.tensor(
            [item.age / self.capacity for item in items],
            device=samples.device,
            dtype=samples.dtype,
        )
        return samples, normalized_ages


class _RobustBatchNorm(nn.Module):
    """Source-anchored running statistics used by RoTTA."""

    batch_norm_type: Type[nn.Module]
    expected_dimensions: Tuple[int, ...]

    def __init__(self, batch_norm: nn.Module, alpha: float) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("RoTTA robust-BN alpha must be in [0, 1]")
        if batch_norm.running_mean is None or batch_norm.running_var is None:
            raise ValueError(
                "RoTTA requires source BatchNorm layers with running statistics"
            )
        self.num_features = batch_norm.num_features
        self.alpha = alpha
        self.eps = batch_norm.eps
        self.affine = batch_norm.affine
        self.register_buffer("source_mean", copy.deepcopy(batch_norm.running_mean))
        self.register_buffer("source_var", copy.deepcopy(batch_norm.running_var))
        self.register_buffer(
            "source_num_batches_tracked",
            copy.deepcopy(batch_norm.num_batches_tracked),
        )
        if self.affine:
            self.weight = copy.deepcopy(batch_norm.weight)
            self.bias = copy.deepcopy(batch_norm.bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.train(batch_norm.training)

    @classmethod
    def from_batch_norm(
        cls,
        batch_norm: nn.Module,
        alpha: float,
    ) -> "_RobustBatchNorm":
        return cls(batch_norm, alpha)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim not in self.expected_dimensions:
            expected = ", ".join(str(value) for value in self.expected_dimensions)
            raise ValueError(
                f"{type(self).__name__} expects {expected}-D input, "
                f"got {inputs.ndim}-D"
            )
        spatial_dimensions = tuple(range(2, inputs.ndim))
        reduce_dimensions = (0, *spatial_dimensions)
        if self.training:
            batch_variance, batch_mean = torch.var_mean(
                inputs,
                dim=reduce_dimensions,
                unbiased=False,
            )
            mean = (1.0 - self.alpha) * self.source_mean + self.alpha * batch_mean
            variance = (
                1.0 - self.alpha
            ) * self.source_var + self.alpha * batch_variance
            self.source_mean.copy_(mean.detach())
            self.source_var.copy_(variance.detach())
        else:
            mean = self.source_mean
            variance = self.source_var

        shape = (1, self.num_features) + (1,) * len(spatial_dimensions)
        normalized = (inputs - mean.view(shape)) * torch.rsqrt(
            variance.view(shape) + self.eps
        )
        if self.affine:
            normalized = normalized * self.weight.view(shape) + self.bias.view(shape)
        return normalized

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, alpha={self.alpha}, "
            f"eps={self.eps}, affine={self.affine}"
        )


class RobustBatchNorm1d(_RobustBatchNorm):
    batch_norm_type = nn.BatchNorm1d
    expected_dimensions = (2, 3)


class RobustBatchNorm2d(_RobustBatchNorm):
    batch_norm_type = nn.BatchNorm2d
    expected_dimensions = (4,)


ROBUST_BATCH_NORM_TYPES = (RobustBatchNorm1d, RobustBatchNorm2d)


def convert_robust_batch_norm(
    module: nn.Module,
    alpha: float = 0.05,
) -> nn.Module:
    """Replace 1-D/2-D BatchNorm children while preserving source state."""

    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            replacement: nn.Module = RobustBatchNorm1d.from_batch_norm(
                child,
                alpha,
            )
        elif isinstance(child, nn.BatchNorm2d):
            replacement = RobustBatchNorm2d.from_batch_norm(child, alpha)
        else:
            replacement = convert_robust_batch_norm(child, alpha)
        module.add_module(name, replacement)
    return module


def timeliness_reweighting(ages: torch.Tensor) -> torch.Tensor:
    """Official RoTTA sigmoid decay for memory-sample ages."""

    return torch.sigmoid(-ages)

class RoTTAAdapter(OnlineAdapter):
    """Robust TTA with CSTU memory, robust BN, and an EMA teacher."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        num_classes: int,
        memory_size: int = 64,
        update_frequency: int = 64,
        nu: float = 0.001,
        bn_alpha: float = 0.05,
        lambda_t: float = 1.0,
        lambda_u: float = 1.0,
        augmentation_noise_std: float = 0.005,
        augmentation_scale_std: float = 0.05,
        augmentation_mask_probability: float = 0.0,
        augmentation_shift_ratio: float = 0.0,
        weight_decay: float = 0.0,
        steps: int = 1,
        parameter_scope: str = "bn",
    ) -> None:
        if update_frequency < 1:
            raise ValueError("RoTTA update frequency must be positive")
        if not 0.0 <= nu <= 1.0:
            raise ValueError("RoTTA EMA update rate nu must be in [0, 1]")
        if not 0.0 <= bn_alpha <= 1.0:
            raise ValueError("RoTTA robust-BN alpha must be in [0, 1]")

        super().__init__(copy.deepcopy(model), device)
        parameters, parameter_names = configure_gradient_adaptation(
            self.model,
            parameter_scope,
        )

        self.optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.teacher = copy.deepcopy(self.model).to(device)
        self.teacher.requires_grad_(False)
        self.memory = CSTUMemory(
            capacity=memory_size,
            num_classes=num_classes,
            lambda_t=lambda_t,
            lambda_u=lambda_u,
        )
        self.augment = SignalAugment(
            noise_std=augmentation_noise_std,
            scale_std=augmentation_scale_std,
            mask_probability=augmentation_mask_probability,
            shift_ratio=augmentation_shift_ratio,
        )
        self.steps = steps
        self.update_frequency = update_frequency
        self.nu = nu
        self.bn_alpha = bn_alpha
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.parameter_scope = parameter_scope
        self.parameter_names = tuple(parameter_names)
        self.current_instance = 0
        self.update_events = 0
        self.augmentation_batches = 0
        self.model.eval()
        self.teacher.eval()

    @torch.no_grad()
    def _update_teacher(self) -> None:
        for teacher_parameter, student_parameter in zip(
            self.teacher.parameters(),
            self.model.parameters(),
        ):
            teacher_parameter.mul_(1.0 - self.nu).add_(
                student_parameter,
                alpha=self.nu,
            )

    def _update_model(self) -> None:
        memory_inputs, ages = self.memory.as_batch()
        for _ in range(self.steps):
            self.model.eval()
            self.teacher.eval()
            strong_inputs = self.augment(memory_inputs)
            self.augmentation_batches += 1
            with torch.no_grad():
                teacher_outputs = self.teacher(memory_inputs)
            self.optimizer.zero_grad()
            student_outputs = self.model(strong_inputs)
            weights = timeliness_reweighting(ages)
            loss = (
                soft_cross_entropy(student_outputs, teacher_outputs) * weights
            ).mean()
            loss.backward()
            self.optimizer.step()
            self._update_teacher()
            self.updates += 1
        self.update_events += 1

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        online_outputs: List[torch.Tensor] = []
        offset = 0
        while offset < inputs.shape[0]:
            pending = self.current_instance % self.update_frequency
            until_update = self.update_frequency - pending
            chunk = inputs[offset : offset + until_update]
            self.model.eval()
            self.teacher.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher(chunk)
                probabilities = teacher_outputs.softmax(dim=1)
                pseudo_labels = probabilities.argmax(dim=1)
                uncertainties = -(probabilities * (probabilities + 1e-6).log()).sum(
                    dim=1
                )
            online_outputs.append(teacher_outputs.detach())

            for sample, pseudo_label, uncertainty in zip(
                chunk,
                pseudo_labels,
                uncertainties,
            ):
                self.memory.add(
                    sample,
                    int(pseudo_label.item()),
                    float(uncertainty.item()),
                )
                self.current_instance += 1
            offset += chunk.shape[0]
            if self.current_instance % self.update_frequency == 0:
                self._update_model()
        return torch.cat(online_outputs, dim=0)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        return self.teacher(inputs.to(self.device))

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
                "memory_capacity": self.memory.capacity,
                "memory_occupancy": self.memory.occupancy,
                "memory_occupancy_per_class": self.memory.occupancy_per_class,
                "memory_accepted_samples": self.memory.accepted_samples,
                "memory_discarded_samples": self.memory.discarded_samples,
                "update_frequency": self.update_frequency,
                "update_events": self.update_events,
                "samples_pending_update": (
                    self.current_instance % self.update_frequency
                ),
                "nu": self.nu,
                "bn_alpha": self.bn_alpha,
                "normalization_backend": "layernorm_affine",
                "robust_batch_norm": False,
                "lambda_t": self.lambda_t,
                "lambda_u": self.lambda_u,
                "augmentation_batches": self.augmentation_batches,
            }
        )
        return values
