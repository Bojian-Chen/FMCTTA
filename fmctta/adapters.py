"""Label-free online adapters for fault foundation-model classifiers."""

import copy
import math
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .sar import SAR
from .vida import inject_vida, is_vida_parameter, set_vida_scales


ADAPTATION_PARAMETER_SCOPES = (
    "default",
    "layernorm",
    "classifier",
    "backbone",
    "all",
)
GRADIENT_ADAPTATION_METHODS = {
    "tent",
    "note",
    "eata",
    "sar",
    "cotta",
    "vida",
    "continual_mae",
    "rotta",
    "tca",
    "gold",
    "gold_tca",
    "ttt",
}


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    return -(probabilities * logits.log_softmax(dim=1)).sum(dim=1)


def soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    targets = teacher_logits.softmax(dim=1)
    return -(targets * student_logits.log_softmax(dim=1)).sum(dim=1)


def resolve_adaptation_parameter_scope(method: str, scope: str) -> str:
    method = method.lower()
    if scope not in ADAPTATION_PARAMETER_SCOPES:
        raise ValueError(
            f"Unknown adaptation parameter scope {scope!r}; expected one of "
            f"{', '.join(ADAPTATION_PARAMETER_SCOPES)}"
        )
    if method not in GRADIENT_ADAPTATION_METHODS:
        if scope != "default":
            raise ValueError(
                f"--adaptation_parameter_scope does not apply to {method!r}"
            )
        return "none"
    if scope == "default":
        return (
            "all"
            if method in {
                "cotta",
                "vida",
                "continual_mae",
                "tca",
                "gold",
                "gold_tca",
            }
            else "backbone"
            if method == "ttt"
            else "layernorm"
        )
    return scope


def adaptation_parameter_names(model: nn.Module, scope: str) -> List[str]:
    """Return the exact parameters selected by an already-resolved scope."""

    if scope not in ADAPTATION_PARAMETER_SCOPES or scope == "default":
        raise ValueError(f"Parameter scope must be resolved first: {scope}")
    named_parameters = list(model.named_parameters())
    if scope == "all":
        names = [name for name, _ in named_parameters]
    elif scope == "layernorm":
        selected_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, nn.LayerNorm)
            for parameter in module.parameters(recurse=False)
        }
        names = [
            name
            for name, parameter in named_parameters
            if id(parameter) in selected_ids
        ]
    elif scope == "classifier":
        head = getattr(model, "head", None)
        if head is None:
            raise ValueError("classifier scope requires model.head")
        selected_ids = {id(parameter) for parameter in head.parameters()}
        names = [
            name
            for name, parameter in named_parameters
            if id(parameter) in selected_ids
        ]
    else:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise ValueError("backbone scope requires model.backbone")
        selected_ids = {id(parameter) for parameter in backbone.parameters()}
        names = [
            name
            for name, parameter in named_parameters
            if id(parameter) in selected_ids
        ]
    if not names:
        raise ValueError(f"No parameters were selected for scope {scope!r}")
    return names


def configure_gradient_adaptation(
    model: nn.Module,
    scope: str,
) -> Tuple[List[nn.Parameter], List[str]]:
    """Freeze stochastic layers and enable only the selected parameters."""

    if hasattr(model, "enable_encoder_finetuning"):
        model.enable_encoder_finetuning()
    model.eval()
    model.requires_grad_(False)
    names = adaptation_parameter_names(model, scope)
    selected = set(names)
    parameters: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name in selected:
            parameter.requires_grad_(True)
            parameters.append(parameter)
    return parameters, names


class OnlineAdapter(nn.Module):
    """Stateful CTTA interface. Its forward method never receives labels."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.batches_seen = 0
        self.samples_seen = 0
        self.updates = 0
        self.skipped_updates = 0

    def _record_batch(self, inputs: torch.Tensor) -> None:
        self.batches_seen += 1
        self.samples_seen += int(inputs.shape[0])

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(inputs.to(self.device))

    def statistics(self) -> Dict[str, object]:
        return {
            "batches_seen": self.batches_seen,
            "samples_seen": self.samples_seen,
            "updates": self.updates,
            "skipped_updates": self.skipped_updates,
        }


class SourceAdapter(OnlineAdapter):
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        # Source-only inference never mutates the model, so sharing avoids a
        # large and unnecessary copy of the foundation backbone.
        super().__init__(model, device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        return self.model(inputs.to(self.device))


class GradientAdapter(OnlineAdapter):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        steps: int,
        weight_decay: float,
        parameter_scope: str,
    ) -> None:
        super().__init__(copy.deepcopy(model), device)
        parameters, names = configure_gradient_adaptation(
            self.model, parameter_scope
        )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.steps = steps
        self.parameter_scope = parameter_scope
        self.parameter_names = tuple(names)

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
            }
        )
        return values


class TentAdapter(GradientAdapter):
    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        online_outputs: Optional[torch.Tensor] = None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            current = self.model(inputs)
            if online_outputs is None:
                online_outputs = current.detach()
            softmax_entropy(current).mean().backward()
            self.optimizer.step()
            self.updates += 1
        assert online_outputs is not None
        return online_outputs


class EATAAdapter(GradientAdapter):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        num_classes: int,
        fisher: Optional[Mapping[str, torch.Tensor]],
        fisher_alpha: float,
        entropy_margin: Optional[float],
        diversity_margin: float,
        probability_momentum: float,
        weight_decay: float,
        parameter_scope: str,
    ) -> None:
        super().__init__(
            model,
            device,
            learning_rate,
            1,
            weight_decay,
            parameter_scope,
        )
        self.source_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in set(self.parameter_names)
        }
        self.fisher = {
            name: value.detach().to(device)
            for name, value in (fisher or {}).items()
            if name in self.source_parameters
        }
        self.fisher_alpha = fisher_alpha
        self.entropy_margin = (
            entropy_margin
            if entropy_margin is not None
            else 0.4 * math.log(num_classes)
        )
        self.diversity_margin = diversity_margin
        self.probability_momentum = probability_momentum
        self.probability_ema: Optional[torch.Tensor] = None
        self.reliable_samples = 0
        self.nonredundant_samples = 0

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        online_outputs = outputs.detach()
        probabilities = outputs.softmax(dim=1)
        entropies = softmax_entropy(outputs)
        reliable = entropies < self.entropy_margin
        self.reliable_samples += int(reliable.sum().item())

        if self.probability_ema is None:
            diverse = torch.ones_like(reliable)
        else:
            similarity = torch.nn.functional.cosine_similarity(
                probabilities.detach(),
                self.probability_ema.unsqueeze(0),
                dim=1,
            )
            diverse = similarity < self.diversity_margin
        selected = reliable & diverse
        self.nonredundant_samples += int(selected.sum().item())
        if int(selected.sum().item()) == 0:
            self.skipped_updates += 1
        else:
            loss = entropies[selected].mean()
            if self.fisher_alpha > 0 and self.fisher:
                penalty = outputs.new_zeros(())
                for name, parameter in self.model.named_parameters():
                    if name in self.fisher:
                        penalty = penalty + (
                            self.fisher[name]
                            * (parameter - self.source_parameters[name]).pow(2)
                        ).sum()
                loss = loss + self.fisher_alpha * penalty
            loss.backward()
            self.optimizer.step()
            self.updates += 1

        batch_mean = probabilities.detach().mean(dim=0)
        self.probability_ema = (
            batch_mean
            if self.probability_ema is None
            else self.probability_momentum * self.probability_ema
            + (1.0 - self.probability_momentum) * batch_mean
        )
        return online_outputs

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        values.update(
            {
                "reliable_samples": self.reliable_samples,
                "nonredundant_samples": self.nonredundant_samples,
                "entropy_margin": self.entropy_margin,
                "fisher_parameter_tensors": len(self.fisher),
            }
        )
        return values


class SARAdapter(OnlineAdapter):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        num_classes: int,
        steps: int,
        weight_decay: float,
        parameter_scope: str,
        rho: float,
        entropy_margin: Optional[float],
        reset_threshold: float,
        ema_momentum: float,
    ) -> None:
        super().__init__(copy.deepcopy(model), device)
        parameters, names = configure_gradient_adaptation(
            self.model, parameter_scope
        )
        self.optimizer = SAR(
            parameters,
            torch.optim.SGD,
            rho=rho,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        self.steps = steps
        self.parameter_scope = parameter_scope
        self.parameter_names = tuple(names)
        self.entropy_margin = (
            entropy_margin
            if entropy_margin is not None
            else 0.4 * math.log(num_classes)
        )
        self.reset_threshold = reset_threshold
        self.ema_momentum = ema_momentum
        self.entropy_ema: Optional[float] = None
        self.source_state = copy.deepcopy(self.model.state_dict())
        self.source_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.reliable_samples_first = 0
        self.reliable_samples_second = 0
        self.resets = 0

    def _reset(self) -> None:
        self.model.load_state_dict(self.source_state, strict=True)
        self.optimizer.load_state_dict(self.source_optimizer_state)
        self.entropy_ema = None
        self.resets += 1

    def _adapt_once(self, inputs: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        entropies = softmax_entropy(outputs)
        first = entropies < self.entropy_margin
        self.reliable_samples_first += int(first.sum().item())
        if not first.any():
            self.skipped_updates += 1
            return outputs.detach()
        entropies[first].mean().backward()
        if not self.optimizer.first_step(zero_grad=True):
            self.skipped_updates += 1
            return outputs.detach()

        perturbed = softmax_entropy(self.model(inputs))
        second = first & (perturbed < self.entropy_margin)
        self.reliable_samples_second += int(second.sum().item())
        if not second.any():
            self.optimizer.second_step(zero_grad=True, update=False)
            self.skipped_updates += 1
            return outputs.detach()
        loss = perturbed[second].mean()
        value = float(loss.detach().item())
        if math.isfinite(value):
            self.entropy_ema = (
                value
                if self.entropy_ema is None
                else self.ema_momentum * self.entropy_ema
                + (1.0 - self.ema_momentum) * value
            )
        loss.backward()
        self.optimizer.second_step(zero_grad=True, update=True)
        self.updates += 1
        if (
            self.entropy_ema is not None
            and self.entropy_ema < self.reset_threshold
        ):
            self._reset()
        return outputs.detach()

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        online_outputs: Optional[torch.Tensor] = None
        for _ in range(self.steps):
            outputs = self._adapt_once(inputs)
            if online_outputs is None:
                online_outputs = outputs
        assert online_outputs is not None
        return online_outputs

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        values.update(
            {
                "parameter_scope": self.parameter_scope,
                "adapted_parameter_tensors": len(self.parameter_names),
                "adapted_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                "reliable_samples_first": self.reliable_samples_first,
                "reliable_samples_second": self.reliable_samples_second,
                "resets": self.resets,
                "entropy_ema": self.entropy_ema,
                "entropy_margin": self.entropy_margin,
            }
        )
        return values


class SignalAugment(nn.Module):
    def __init__(
        self,
        noise_std: float,
        scale_std: float,
        mask_probability: float,
        shift_ratio: float,
    ) -> None:
        super().__init__()
        self.noise_std = noise_std
        self.scale_std = scale_std
        self.mask_probability = mask_probability
        self.shift_ratio = shift_ratio

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        if self.scale_std > 0:
            shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
            scale = 1.0 + torch.randn(shape, device=inputs.device) * self.scale_std
            outputs = outputs * scale
        if self.noise_std > 0:
            signal_std = outputs.flatten(1).std(dim=1, keepdim=True)
            shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
            outputs = outputs + torch.randn_like(outputs) * signal_std.view(shape) * self.noise_std
        if self.mask_probability > 0:
            outputs = outputs * (
                torch.rand_like(outputs) >= self.mask_probability
            ).to(outputs.dtype)
        max_shift = int(round(inputs.shape[-1] * self.shift_ratio))
        if max_shift > 0:
            shifts = torch.randint(
                -max_shift,
                max_shift + 1,
                (inputs.shape[0],),
                device=inputs.device,
            )
            outputs = torch.stack(
                [torch.roll(item, int(shift.item()), dims=-1) for item, shift in zip(outputs, shifts)]
            )
        return outputs


class CoTTAAdapter(GradientAdapter):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        parameter_scope: str,
        teacher_momentum: float,
        restore_probability: float,
        anchor_threshold: float,
        augmentations: int,
        augmentation: SignalAugment,
    ) -> None:
        super().__init__(
            model,
            device,
            learning_rate,
            1,
            weight_decay,
            parameter_scope,
        )
        self.teacher = copy.deepcopy(self.model).to(device).eval()
        self.anchor = copy.deepcopy(self.model).to(device).eval()
        self.teacher.requires_grad_(False)
        self.anchor.requires_grad_(False)
        self.source_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in set(self.parameter_names)
        }
        self.teacher_momentum = teacher_momentum
        self.restore_probability = restore_probability
        self.anchor_threshold = anchor_threshold
        self.augmentations = augmentations
        self.augmentation = augmentation
        self.augmented_teacher_batches = 0
        self.restored_values = 0

    @torch.no_grad()
    def _teacher_targets(self, inputs: torch.Tensor) -> torch.Tensor:
        plain = self.teacher(inputs)
        anchor_confidence = self.anchor(inputs).softmax(dim=1).amax(dim=1).mean()
        if self.augmentations < 1 or anchor_confidence >= self.anchor_threshold:
            return plain
        augmented = [self.teacher(self.augmentation(inputs)) for _ in range(self.augmentations)]
        self.augmented_teacher_batches += 1
        return torch.stack(augmented).mean(dim=0)

    @torch.no_grad()
    def _update_teacher_and_restore(self) -> None:
        for teacher_parameter, student_parameter in zip(
            self.teacher.parameters(), self.model.parameters()
        ):
            teacher_parameter.mul_(self.teacher_momentum).add_(
                student_parameter,
                alpha=1.0 - self.teacher_momentum,
            )
        if self.restore_probability <= 0:
            return
        selected = set(self.parameter_names)
        for name, parameter in self.model.named_parameters():
            if name not in selected:
                continue
            mask = torch.rand_like(parameter) < self.restore_probability
            self.restored_values += int(mask.sum().item())
            parameter.copy_(
                torch.where(mask, self.source_parameters[name], parameter)
            )

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            online_outputs = self.teacher(inputs).detach()
            targets = self._teacher_targets(inputs).detach()
        self.optimizer.zero_grad(set_to_none=True)
        student_outputs = self.model(inputs)
        soft_cross_entropy(student_outputs, targets).mean().backward()
        self.optimizer.step()
        self._update_teacher_and_restore()
        self.updates += 1
        return online_outputs

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        return self.teacher(inputs.to(self.device))

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        values.update(
            {
                "teacher_momentum": self.teacher_momentum,
                "restore_probability": self.restore_probability,
                "augmented_teacher_batches": self.augmented_teacher_batches,
                "restored_parameter_values": self.restored_values,
            }
        )
        return values


class ViDAAdapter(OnlineAdapter):
    """Homeostatic high/low-rank adaptation with an EMA teacher."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        vida_learning_rate: float,
        steps: int,
        weight_decay: float,
        parameter_scope: str,
        low_rank: int,
        high_rank: int,
        teacher_momentum: float,
        adapter_momentum: float,
        uncertainty_threshold: float,
        uncertainty_scale: float,
        augmentations: int,
        augmentation: SignalAugment,
    ) -> None:
        super().__init__(copy.deepcopy(model), device)
        if hasattr(self.model, "enable_encoder_finetuning"):
            self.model.enable_encoder_finetuning()
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            raise ValueError("ViDA requires model.backbone")
        self.injected_layers = tuple(
            inject_vida(backbone, low_rank, high_rank)
        )
        self.model.requires_grad_(False)
        scoped_names = adaptation_parameter_names(self.model, parameter_scope)
        base_names = [
            name for name in scoped_names if not is_vida_parameter(name)
        ]
        vida_names = [
            name
            for name, _ in self.model.named_parameters()
            if is_vida_parameter(name)
        ]
        selected = set(base_names + vida_names)
        named_parameters = dict(self.model.named_parameters())
        for name in selected:
            named_parameters[name].requires_grad_(True)
        parameter_groups = []
        if base_names:
            parameter_groups.append(
                {
                    "params": [named_parameters[name] for name in base_names],
                    "lr": learning_rate,
                }
            )
        parameter_groups.append(
            {
                "params": [named_parameters[name] for name in vida_names],
                "lr": vida_learning_rate,
            }
        )
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=weight_decay,
        )
        self.teacher = copy.deepcopy(self.model).to(device).eval()
        self.teacher.requires_grad_(False)
        self.steps = steps
        self.parameter_scope = parameter_scope
        self.base_parameter_names = tuple(base_names)
        self.vida_parameter_names = tuple(vida_names)
        self.low_rank = low_rank
        self.high_rank = high_rank
        self.teacher_momentum = teacher_momentum
        self.adapter_momentum = adapter_momentum
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_scale = uncertainty_scale
        self.augmentations = augmentations
        self.augmentation = augmentation
        self.last_uncertainty = 0.0
        self.uncertainty_sum = 0.0
        self.high_rank_batches = 0
        self.low_rank_batches = 0
        self.last_low_scale = 1.0
        self.last_high_scale = 1.0

    @torch.no_grad()
    def _measure_uncertainty(self, inputs: torch.Tensor) -> float:
        self.teacher.eval()
        predictions = [
            self.teacher(self.augmentation(inputs))
            for _ in range(self.augmentations)
        ]
        variance = torch.stack(predictions).var(dim=0, unbiased=False)
        uncertainty = variance.mean() * self.uncertainty_scale
        value = float(uncertainty.clamp(min=0.0, max=1.0).item())
        self.last_uncertainty = value
        self.uncertainty_sum += value
        return value

    @torch.no_grad()
    def _apply_hka(self, uncertainty: float) -> None:
        if uncertainty >= self.uncertainty_threshold:
            high = 1.0 + uncertainty
            low = 1.0 - uncertainty
            self.high_rank_batches += 1
        else:
            high = 1.0 - uncertainty
            low = 1.0 + uncertainty
            self.low_rank_batches += 1
        set_vida_scales(self.model, low, high)
        set_vida_scales(self.teacher, low, high)
        self.last_low_scale = low
        self.last_high_scale = high

    @torch.no_grad()
    def _update_teacher(self) -> None:
        student_parameters = dict(self.model.named_parameters())
        for name, teacher_parameter in self.teacher.named_parameters():
            momentum = (
                self.adapter_momentum
                if is_vida_parameter(name)
                else self.teacher_momentum
            )
            teacher_parameter.mul_(momentum).add_(
                student_parameters[name],
                alpha=1.0 - momentum,
            )

    def _adapt_once(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            uncertainty = self._measure_uncertainty(inputs)
            self._apply_hka(uncertainty)
            teacher_outputs = self.teacher(inputs).detach()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        student_outputs = self.model(inputs)
        loss = soft_cross_entropy(student_outputs, teacher_outputs).mean()
        if not torch.isfinite(loss):
            self.skipped_updates += 1
            self.optimizer.zero_grad(set_to_none=True)
            return teacher_outputs
        loss.backward()
        self.optimizer.step()
        self._update_teacher()
        self.updates += 1
        return teacher_outputs

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        online_outputs: Optional[torch.Tensor] = None
        for _ in range(self.steps):
            outputs = self._adapt_once(inputs)
            if online_outputs is None:
                online_outputs = outputs
        assert online_outputs is not None
        return online_outputs

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        return self.teacher(inputs.to(self.device))

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        named_parameters = dict(self.model.named_parameters())
        values.update(
            {
                "parameter_scope": self.parameter_scope,
                "injected_linear_layers": len(self.injected_layers),
                "low_rank": self.low_rank,
                "high_rank": self.high_rank,
                "base_parameter_tensors": len(self.base_parameter_names),
                "vida_parameter_tensors": len(self.vida_parameter_names),
                "adapted_parameters": sum(
                    named_parameters[name].numel()
                    for name in (
                        self.base_parameter_names + self.vida_parameter_names
                    )
                ),
                "teacher_momentum": self.teacher_momentum,
                "adapter_momentum": self.adapter_momentum,
                "uncertainty_threshold": self.uncertainty_threshold,
                "last_uncertainty": self.last_uncertainty,
                "mean_uncertainty": self.uncertainty_sum
                / max(self.updates + self.skipped_updates, 1),
                "last_low_scale": self.last_low_scale,
                "last_high_scale": self.last_high_scale,
                "low_rank_batches": self.low_rank_batches,
                "high_rank_batches": self.high_rank_batches,
            }
        )
        return values


def build_adapter(
    method: str,
    model: nn.Module,
    device: torch.device,
    learning_rate: float,
    num_classes: int,
    fisher: Optional[Mapping[str, torch.Tensor]] = None,
    steps: int = 1,
    weight_decay: float = 0.0,
    parameter_scope: str = "default",
    eata_fisher_alpha: float = 1.0,
    eata_entropy_margin: Optional[float] = None,
    eata_diversity_margin: float = 0.95,
    note_memory_size: int = 64,
    note_update_interval: int = 64,
    note_temperature: float = 1.0,
    rotta_memory_size: int = 64,
    rotta_update_frequency: int = 64,
    rotta_nu: float = 0.001,
    rotta_bn_alpha: float = 0.05,
    rotta_lambda_t: float = 1.0,
    rotta_lambda_u: float = 1.0,
    rotta_augmentation_noise_std: float = 0.005,
    rotta_augmentation_scale_std: float = 0.05,
    rotta_augmentation_mask_probability: float = 0.0,
    rotta_augmentation_shift_ratio: float = 0.0,
    tca_teacher_momentum: float = 0.999,
    tca_centroid_momentum: float = 0.9,
    tca_topology_weight_momentum: float = 0.8,
    tca_alignment_weight: float = 0.025,
    tca_topology_weight: float = 0.15,
    tca_uniformity_temperature: float = 2.0,
    tca_symmetric_alpha: float = 0.5,
    tca_augmentation_noise_std: float = 0.005,
    tca_augmentation_scale_std: float = 0.05,
    tca_augmentation_mask_probability: float = 0.0,
    tca_augmentation_shift_ratio: float = 0.0,
    gold_source_prototypes: Optional[torch.Tensor] = None,
    gold_top_r: int = 64,
    gold_alpha: float = 0.1,
    gold_conf_thresh: float = 0.8,
    gold_update_every: int = 50,
    gold_adapter_lr_scale: float = 1.0,
    gold_lambda_ce_trg: float = 1.0,
    gold_lambda_cont: float = 1.0,
    gold_temperature: float = 0.1,
    gold_projection_dim: int = 128,
    gold_teacher_momentum: float = 0.999,
    gold_augmentation_noise_std: float = 0.005,
    gold_augmentation_scale_std: float = 0.05,
    gold_augmentation_mask_probability: float = 0.0,
    gold_augmentation_shift_ratio: float = 0.0,
    gold_tca_lambda_topology: float = 0.15,
    gold_tca_centroid_momentum: float = 0.9,
    gold_tca_uniformity_temperature: float = 2.0,
    ttt_auxiliary_head: Optional[nn.Module] = None,
    sar_rho: float = 0.05,
    sar_entropy_margin: Optional[float] = None,
    sar_reset_threshold: float = 0.2,
    sar_ema_momentum: float = 0.9,
    cotta_teacher_momentum: float = 0.999,
    cotta_restore_probability: float = 0.01,
    cotta_anchor_threshold: float = 0.92,
    cotta_augmentations: int = 8,
    cotta_augmentation_noise_std: float = 0.005,
    cotta_augmentation_scale_std: float = 0.05,
    cotta_augmentation_mask_probability: float = 0.0,
    cotta_augmentation_shift_ratio: float = 0.0,
    vida_learning_rate: float = 1e-5,
    vida_low_rank: int = 1,
    vida_high_rank: int = 128,
    vida_teacher_momentum: float = 0.999,
    vida_adapter_momentum: float = 0.999,
    vida_uncertainty_threshold: float = 0.2,
    vida_uncertainty_scale: float = 0.1,
    vida_augmentations: int = 10,
    vida_augmentation_noise_std: float = 0.005,
    vida_augmentation_scale_std: float = 0.05,
    vida_augmentation_mask_probability: float = 0.0,
    vida_augmentation_shift_ratio: float = 0.0,
    continual_mae_mask_ratio: float = 0.5,
    continual_mae_augmentations: int = 10,
    continual_mae_reconstruction_weight: float = 0.5,
    continual_mae_restore_probability: float = 0.1,
    continual_mae_augmentation_noise_std: float = 0.005,
    continual_mae_augmentation_scale_std: float = 0.05,
    continual_mae_augmentation_mask_probability: float = 0.0,
    continual_mae_augmentation_shift_ratio: float = 0.0,
    knn_neighbors: int = 5,
    knn_reference_features: Optional[torch.Tensor] = None,
    knn_reference_labels: Optional[torch.Tensor] = None,
) -> OnlineAdapter:
    method = method.lower()
    scope = resolve_adaptation_parameter_scope(method, parameter_scope)
    if method == "source":
        return SourceAdapter(model, device)
    if method == "knn":
        # The runner creates the labeled source feature bank after loading (or fitting) the source model, then hands it to this non-adaptive adapter.
        from .knn import KNNAdapter

        if knn_reference_features is None or knn_reference_labels is None:
            raise ValueError("kNN requires labeled source reference features")
        return KNNAdapter(
            model,
            device,
            knn_reference_features,
            knn_reference_labels,
            num_classes,
            knn_neighbors,
        )
    if method == "tent":
        return TentAdapter(
            model, device, learning_rate, steps, weight_decay, scope
        )
    if method == "note":
        # Imported lazily because note.py reuses the public adapter base and
        # parameter-selection helpers from this module.
        from .note import NOTEAdapter

        return NOTEAdapter(
            model,
            device,
            learning_rate,
            num_classes,
            steps,
            weight_decay,
            scope,
            note_memory_size,
            note_update_interval,
            note_temperature,
        )
    if method == "ttt":
        from .ttt import TTTAdapter

        if ttt_auxiliary_head is None:
            raise ValueError(
                "TTT requires an auxiliary head trained jointly on the source domain"
            )
        return TTTAdapter(
            model,
            ttt_auxiliary_head,
            device,
            learning_rate,
            steps,
            weight_decay,
            scope,
        )
    if method == "rotta":
        from .rotta import RoTTAAdapter

        return RoTTAAdapter(
            model,
            device,
            learning_rate,
            num_classes,
            rotta_memory_size,
            rotta_update_frequency,
            rotta_nu,
            rotta_bn_alpha,
            rotta_lambda_t,
            rotta_lambda_u,
            rotta_augmentation_noise_std,
            rotta_augmentation_scale_std,
            rotta_augmentation_mask_probability,
            rotta_augmentation_shift_ratio,
            weight_decay,
            steps,
            scope,
        )
    if method == "gold":
        from .gold import GOLDAdapter

        if gold_source_prototypes is None:
            raise ValueError("GOLD requires class prototypes from source training data")
        return GOLDAdapter(
            model,
            device,
            learning_rate,
            num_classes,
            gold_source_prototypes,
            steps=steps,
            weight_decay=weight_decay,
            parameter_scope=scope,
            top_r=gold_top_r,
            alpha=gold_alpha,
            conf_thresh=gold_conf_thresh,
            update_every=gold_update_every,
            adapter_lr_scale=gold_adapter_lr_scale,
            lambda_ce_trg=gold_lambda_ce_trg,
            lambda_cont=gold_lambda_cont,
            temperature=gold_temperature,
            projection_dim=gold_projection_dim,
            teacher_momentum=gold_teacher_momentum,
            augmentation_noise_std=gold_augmentation_noise_std,
            augmentation_scale_std=gold_augmentation_scale_std,
            augmentation_mask_probability=gold_augmentation_mask_probability,
            augmentation_shift_ratio=gold_augmentation_shift_ratio,
        )
    if method == "gold_tca":
        from .gold_tca import GOLDTCAAdapter

        if gold_source_prototypes is None:
            raise ValueError(
                "gold_tca requires class prototypes from source training data"
            )
        return GOLDTCAAdapter(
            model,
            device,
            learning_rate,
            num_classes,
            gold_source_prototypes,
            steps=steps,
            weight_decay=weight_decay,
            parameter_scope=scope,
            top_r=gold_top_r,
            alpha=gold_alpha,
            conf_thresh=gold_conf_thresh,
            update_every=gold_update_every,
            adapter_lr_scale=gold_adapter_lr_scale,
            lambda_ce_trg=gold_lambda_ce_trg,
            lambda_cont=gold_lambda_cont,
            temperature=gold_temperature,
            projection_dim=gold_projection_dim,
            teacher_momentum=gold_teacher_momentum,
            augmentation_noise_std=gold_augmentation_noise_std,
            augmentation_scale_std=gold_augmentation_scale_std,
            augmentation_mask_probability=gold_augmentation_mask_probability,
            augmentation_shift_ratio=gold_augmentation_shift_ratio,
            lambda_topology=gold_tca_lambda_topology,
            centroid_momentum=gold_tca_centroid_momentum,
            uniformity_temperature=gold_tca_uniformity_temperature,
        )
    if method == "tca":
        from .tca import TCAAdapter

        return TCAAdapter(
            model,
            device,
            learning_rate,
            num_classes,
            tca_teacher_momentum,
            tca_centroid_momentum,
            tca_topology_weight_momentum,
            tca_alignment_weight,
            tca_topology_weight,
            tca_uniformity_temperature,
            tca_symmetric_alpha,
            tca_augmentation_noise_std,
            tca_augmentation_scale_std,
            tca_augmentation_mask_probability,
            tca_augmentation_shift_ratio,
            weight_decay,
            steps,
            scope,
        )
    if method == "eata":
        return EATAAdapter(
            model,
            device,
            learning_rate,
            num_classes,
            fisher,
            eata_fisher_alpha,
            eata_entropy_margin,
            eata_diversity_margin,
            0.9,
            weight_decay,
            scope,
        )
    if method == "sar":
        return SARAdapter(
            model,
            device,
            learning_rate,
            num_classes,
            steps,
            weight_decay,
            scope,
            sar_rho,
            sar_entropy_margin,
            sar_reset_threshold,
            sar_ema_momentum,
        )
    if method == "cotta":
        augmentation = SignalAugment(
            cotta_augmentation_noise_std,
            cotta_augmentation_scale_std,
            cotta_augmentation_mask_probability,
            cotta_augmentation_shift_ratio,
        )
        return CoTTAAdapter(
            model,
            device,
            learning_rate,
            weight_decay,
            scope,
            cotta_teacher_momentum,
            cotta_restore_probability,
            cotta_anchor_threshold,
            cotta_augmentations,
            augmentation,
        )
    if method == "vida":
        augmentation = SignalAugment(
            vida_augmentation_noise_std,
            vida_augmentation_scale_std,
            vida_augmentation_mask_probability,
            vida_augmentation_shift_ratio,
        )
        return ViDAAdapter(
            model,
            device,
            learning_rate,
            vida_learning_rate,
            steps,
            weight_decay,
            scope,
            vida_low_rank,
            vida_high_rank,
            vida_teacher_momentum,
            vida_adapter_momentum,
            vida_uncertainty_threshold,
            vida_uncertainty_scale,
            vida_augmentations,
            augmentation,
        )
    if method == "continual_mae":
        # Import here so continual_mae.py can reuse the public adapter base and helpers above without creating a module-initialization cycle.
        from .continual_mae import ContinualMAEAdapter

        augmentation = SignalAugment(
            continual_mae_augmentation_noise_std,
            continual_mae_augmentation_scale_std,
            continual_mae_augmentation_mask_probability,
            continual_mae_augmentation_shift_ratio,
        )
        return ContinualMAEAdapter(
            model,
            device,
            learning_rate,
            steps,
            weight_decay,
            scope,
            continual_mae_mask_ratio,
            continual_mae_augmentations,
            continual_mae_reconstruction_weight,
            continual_mae_restore_probability,
            augmentation,
        )
    raise ValueError(f"Unknown method: {method}")
