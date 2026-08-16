"""Minimal sharpness-aware optimizer used by SAR."""

from copy import deepcopy
from typing import Iterable, Type

import torch
import torch.nn as nn


class SAM:
    """Two-step SAM wrapper with an explicit perturbation rollback path."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        base_optimizer: Type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0:
            raise ValueError("SAM rho must be non-negative")
        self.base_optimizer = base_optimizer(parameters, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        self.adaptive = adaptive
        self._old_parameters = {}

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> bool:
        gradients = [
            (parameter, group)
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not gradients:
            return False
        shared_device = gradients[0][0].device
        gradient_norm = torch.norm(
            torch.stack(
                [
                    (
                        (parameter.abs() if self.adaptive else 1.0)
                        * parameter.grad
                    )
                    .norm(p=2)
                    .to(shared_device)
                    for parameter, _ in gradients
                ]
            ),
            p=2,
        )
        scale = self.rho / (gradient_norm + 1e-12)
        self._old_parameters = {}
        for parameter, _ in gradients:
            self._old_parameters[parameter] = parameter.detach().clone()
            multiplier = parameter.pow(2) if self.adaptive else 1.0
            parameter.add_(multiplier * parameter.grad * scale.to(parameter))
        if zero_grad:
            self.zero_grad()
        return True

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False, update: bool = True) -> None:
        for parameter, old_value in self._old_parameters.items():
            parameter.copy_(old_value)
        if update:
            self.base_optimizer.step()
        self._old_parameters = {}
        if zero_grad:
            self.zero_grad()

    def state_dict(self):
        return {
            "base_optimizer": deepcopy(self.base_optimizer.state_dict()),
            "rho": self.rho,
            "adaptive": self.adaptive,
        }

    def load_state_dict(self, state_dict) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self.base_optimizer.param_groups
        self.rho = state_dict["rho"]
        self.adaptive = state_dict["adaptive"]
        self._old_parameters = {}
