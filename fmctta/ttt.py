"""Classic test-time training for vibration foundation-model features."""

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import OnlineAdapter, adaptation_parameter_names


TTT_TRANSFORM_CLASSES = 4


def build_ttt_head(feature_dimension: int) -> nn.Module:
    """Build the four-way transform classifier used in source and target stages."""

    if feature_dimension < 1:
        raise ValueError("TTT feature dimension must be positive")
    return nn.Linear(feature_dimension, TTT_TRANSFORM_CLASSES)


def make_ttt_batch(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create four label-preserving circular shifts of a 1-D signal batch."""

    if inputs.ndim not in {2, 3}:
        raise ValueError(
            "TTT expects [batch, length] or [batch, channels, length]"
        )
    transformed = []
    labels = []
    for transform_id in range(TTT_TRANSFORM_CLASSES):
        shift = transform_id * inputs.shape[-1] // TTT_TRANSFORM_CLASSES
        transformed.append(torch.roll(inputs, shifts=shift, dims=-1))
        labels.append(
            torch.full(
                (inputs.shape[0],),
                transform_id,
                dtype=torch.long,
                device=inputs.device,
            )
        )
    return torch.cat(transformed), torch.cat(labels)


def ttt_auxiliary_logits(
    model: nn.Module,
    auxiliary_head: nn.Module,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    transformed, labels = make_ttt_batch(inputs)
    return auxiliary_head(model.forward_features(transformed)), labels


class TTTAdapter(OnlineAdapter):
    """Adapt the source-trained auxiliary task before each task prediction."""

    def __init__(
        self,
        model: nn.Module,
        auxiliary_head: nn.Module,
        device: torch.device,
        learning_rate: float,
        steps: int = 1,
        weight_decay: float = 0.0,
        parameter_scope: str = "backbone",
    ) -> None:
        super().__init__(copy.deepcopy(model), device)
        self.auxiliary_head = copy.deepcopy(auxiliary_head).to(device)
        if hasattr(self.model, "enable_encoder_finetuning"):
            self.model.enable_encoder_finetuning()
        self.model.requires_grad_(False)
        names = adaptation_parameter_names(self.model, parameter_scope)
        selected = set(names)
        parameters = []
        for name, parameter in self.model.named_parameters():
            if name in selected:
                parameter.requires_grad_(True)
                parameters.append(parameter)
        self.auxiliary_head.requires_grad_(True)
        parameters.extend(self.auxiliary_head.parameters())
        self.optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.steps = steps
        self.parameter_scope = parameter_scope
        self.parameter_names = tuple(names)

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        self.model.eval()
        with torch.no_grad():
            pre_adapt_outputs = self.model(inputs).detach()

        for _ in range(self.steps):
            self.model.train()
            self.auxiliary_head.train()
            self.optimizer.zero_grad(set_to_none=True)
            auxiliary_logits, transform_labels = ttt_auxiliary_logits(
                self.model,
                self.auxiliary_head,
                inputs,
            )
            F.cross_entropy(auxiliary_logits, transform_labels).backward()
            self.optimizer.step()
            self.updates += 1
        return pre_adapt_outputs

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
                "auxiliary_parameters": sum(
                    parameter.numel() for parameter in self.auxiliary_head.parameters()
                ),
                "native_prediction_stage": "post_adapt",
            }
        )
        return values
