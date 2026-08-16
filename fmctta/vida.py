"""High/low-rank ViDA branches for linear layers.

The implementation follows the adapter structure from ViDA: every selected
linear layer keeps its original branch and gains parallel low- and high-rank
linear bottlenecks.  Both up projections are zero-initialized, so injection
does not change the source model's predictions.
"""

from typing import List

import torch
import torch.nn as nn


class ViDAInjectedLinear(nn.Module):
    """A linear layer with parallel low- and high-rank ViDA branches."""

    def __init__(
        self,
        base: nn.Linear,
        low_rank: int,
        high_rank: int,
    ) -> None:
        super().__init__()
        if low_rank < 1 or high_rank < 1:
            raise ValueError("ViDA ranks must be positive")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.vida_low_down = nn.Linear(
            self.in_features,
            low_rank,
            bias=False,
            **factory_kwargs,
        )
        self.vida_low_up = nn.Linear(
            low_rank,
            self.out_features,
            bias=False,
            **factory_kwargs,
        )
        self.vida_high_down = nn.Linear(
            self.in_features,
            high_rank,
            bias=False,
            **factory_kwargs,
        )
        self.vida_high_up = nn.Linear(
            high_rank,
            self.out_features,
            bias=False,
            **factory_kwargs,
        )
        nn.init.normal_(self.vida_low_down.weight, std=1.0 / low_rank**2)
        nn.init.zeros_(self.vida_low_up.weight)
        nn.init.normal_(self.vida_high_down.weight, std=1.0 / high_rank**2)
        nn.init.zeros_(self.vida_high_up.weight)
        self.register_buffer("vida_low_scale", torch.tensor(1.0))
        self.register_buffer("vida_high_scale", torch.tensor(1.0))

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter:
        return self.base.bias

    def set_scales(self, low: float, high: float) -> None:
        self.vida_low_scale.fill_(low)
        self.vida_high_scale.fill_(high)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        low = self.vida_low_up(self.vida_low_down(inputs))
        high = self.vida_high_up(self.vida_high_down(inputs))
        return (
            self.base(inputs)
            + self.vida_low_scale * low
            + self.vida_high_scale * high
        )


def inject_vida(
    module: nn.Module,
    low_rank: int,
    high_rank: int,
) -> List[str]:
    """Inject ViDA into ordinary linear layers below ``module``.

    UniFreq uses explicit q/k/v projections in its tokenizers and harmonic
    attention, plus standard Transformer feed-forward projections.  Visiting
    the whole backbone covers those projections while intentionally skipping
    PyTorch's packed ``MultiheadAttention`` output projection subclass, whose
    forward path bypasses the child module call.
    """

    if low_rank < 1 or high_rank < 1:
        raise ValueError("ViDA ranks must be positive")
    injected: List[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            qualified_name = f"{prefix}.{name}" if prefix else name
            if type(child) is nn.Linear:
                parent._modules[name] = ViDAInjectedLinear(
                    child,
                    low_rank,
                    high_rank,
                )
                injected.append(qualified_name)
            else:
                visit(child, qualified_name)

    visit(module, "")
    if not injected:
        raise ValueError("ViDA could not find a linear layer in the backbone")
    return injected


def set_vida_scales(module: nn.Module, low: float, high: float) -> int:
    """Set HKA scales on every injected ViDA layer."""

    count = 0
    for child in module.modules():
        if isinstance(child, ViDAInjectedLinear):
            child.set_scales(low, high)
            count += 1
    return count


def is_vida_parameter(name: str) -> bool:
    """Return whether a state/parameter name belongs to a ViDA branch."""

    return ".vida_low_" in name or ".vida_high_" in name
