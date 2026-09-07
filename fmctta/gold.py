"""GOLD classification adapted to the fault-model feature/head interface.

Reference: https://github.com/AIGNLAI/GOLD (CVPR 2026).
See docs/GOLD_ZH.md for source preparation and differences from image CTTA.
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adapters import OnlineAdapter, SignalAugment, configure_gradient_adaptation


@torch.no_grad()
def build_source_prototypes(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> torch.Tensor:
    """Accumulate class means from source training data, without retaining it."""
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, device=device)
    for inputs, labels, _ in loader:
        features = model.forward_features(inputs.to(device)).float()
        labels = labels.to(device=device, dtype=torch.long)
        if sums is None:
            sums = features.new_zeros(num_classes, features.shape[1])
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
    if sums is None or (counts == 0).any():
        raise ValueError("GOLD requires source training samples for every class")
    return sums / counts[:, None]


def symmetric_cross_entropy(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    teacher = teacher.detach()
    return -0.5 * (
        (teacher.softmax(1) * student.log_softmax(1)).sum(1)
        + (student.softmax(1) * teacher.log_softmax(1)).sum(1)
    )


def multiview_contrastive_loss(views: torch.Tensor, temperature: float) -> torch.Tensor:
    """All views of each instance are positives; exclude self-comparisons."""
    batch, count, dimension = views.shape
    features = F.normalize(views.transpose(0, 1).reshape(-1, dimension), dim=1)
    scores = features @ features.T / temperature
    diagonal = torch.eye(batch * count, device=views.device, dtype=torch.bool)
    ids = torch.arange(batch, device=views.device).repeat(count)
    positives = (ids[:, None] == ids[None, :]) & ~diagonal
    log_denominator = scores.masked_fill(diagonal, -torch.inf).logsumexp(1)
    log_prob = scores - log_denominator[:, None]
    return -(log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()


class GOLDAdapter(OnlineAdapter):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        num_classes: int,
        source_prototypes: torch.Tensor,
        steps: int = 1,
        weight_decay: float = 0.0,
        parameter_scope: str = "all",
        top_r: int = 64,
        alpha: float = 0.1,
        conf_thresh: float = 0.8,
        update_every: int = 50,
        adapter_lr_scale: float = 1.0,
        lambda_ce_trg: float = 1.0,
        lambda_cont: float = 1.0,
        temperature: float = 0.1,
        projection_dim: int = 128,
        teacher_momentum: float = 0.999,
        augmentation_noise_std: float = 0.005,
        augmentation_scale_std: float = 0.05,
        augmentation_mask_probability: float = 0.0,
        augmentation_shift_ratio: float = 0.0,
    ) -> None:
        if source_prototypes.ndim != 2 or source_prototypes.shape[0] != num_classes:
            raise ValueError("GOLD prototypes must have shape [classes, features]")
        if not torch.isfinite(source_prototypes).all():
            raise ValueError("GOLD source prototypes must be finite")
        if min(top_r, update_every, projection_dim, steps) < 1:
            raise ValueError(
                "GOLD rank, update interval, projection and steps must be positive"
            )
        if not 0 <= alpha <= 1 or not 0 <= conf_thresh <= 1:
            raise ValueError("GOLD alpha and confidence threshold must be in [0, 1]")
        if not 0 <= teacher_momentum < 1:
            raise ValueError("GOLD teacher momentum must be in [0, 1)")
        if temperature <= 0 or learning_rate <= 0 or adapter_lr_scale <= 0:
            raise ValueError("GOLD temperature and learning rates must be positive")
        if min(lambda_ce_trg, lambda_cont, weight_decay) < 0:
            raise ValueError("GOLD loss weights and weight decay cannot be negative")
        if lambda_ce_trg + lambda_cont == 0:
            raise ValueError("GOLD requires at least one nonzero loss weight")
        super().__init__(copy.deepcopy(model), device)
        parameters, names = configure_gradient_adaptation(self.model, parameter_scope)
        self.parameter_scope = parameter_scope
        self.parameter_names = tuple(names)
        self.teacher = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.register_buffer(
            "source_prototypes",
            source_prototypes.detach().to(device=device, dtype=torch.float32).clone(),
        )
        self.feature_dim = source_prototypes.shape[1]
        self.top_r = min(top_r, self.feature_dim)
        self.alpha = alpha
        self.conf_thresh = conf_thresh
        self.update_every = update_every
        self.steps = steps
        self.lambda_ce_trg = lambda_ce_trg
        self.lambda_cont = lambda_cont
        self.temperature = temperature
        self.teacher_momentum = teacher_momentum
        self.augmentation = SignalAugment(
            augmentation_noise_std,
            augmentation_scale_std,
            augmentation_mask_probability,
            augmentation_shift_ratio,
        )
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        ).to(device)
        self.adapter_scale = nn.Parameter(torch.zeros(self.top_r, device=device))
        self.register_buffer("agop", self._initial_agop())
        self.register_buffer("basis", self._eigenbasis())
        self.optimizer = torch.optim.Adam(
            [
                {"params": parameters},
                {"params": self.projector.parameters()},
                {
                    "params": [self.adapter_scale],
                    "lr": learning_rate * adapter_lr_scale,
                },
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.agop_updates = 0
        self.basis_updates = 0
        self.adaptation_iterations = 0
        self.warmup_updates = 0

    @torch.enable_grad()
    def _initial_agop(self) -> torch.Tensor:
        head = self.model.head
        if isinstance(head, nn.Linear):
            weights = head.weight.detach().float()
            return weights.T @ weights
        # Extension for nonlinear heads: average J(f)^T J(f) over source means.
        features = self.source_prototypes.detach().float().requires_grad_(True)
        logits = head(features)
        result = features.new_zeros(self.feature_dim, self.feature_dim)
        for index in range(logits.shape[1]):
            gradients = torch.autograd.grad(
                logits[:, index].sum(),
                features,
                retain_graph=index + 1 < logits.shape[1],
            )[0]
            result.add_(gradients.T @ gradients / features.shape[0])
        return result.detach()

    @torch.no_grad()
    def _eigenbasis(self) -> torch.Tensor:
        _, vectors = torch.linalg.eigh(self.agop)
        return vectors[:, -self.top_r :].flip(1).contiguous()

    def apply_adapter(self, features: torch.Tensor) -> torch.Tensor:
        return features + ((features @ self.basis) * self.adapter_scale) @ self.basis.T

    @torch.enable_grad()
    def batch_agop(
        self, features: torch.Tensor, logits: torch.Tensor
    ) -> Optional[torch.Tensor]:
        selected = logits.detach().softmax(1).amax(1) >= self.conf_thresh
        if selected.sum() < 2:
            return None
        inputs = features.detach()[selected].requires_grad_(True)
        gradients = torch.autograd.grad(self.model.head(inputs).amax(1).sum(), inputs)[
            0
        ]
        return (gradients.T @ gradients / inputs.shape[0]).detach()

    @torch.no_grad()
    def _update_teacher(self) -> None:
        for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
            teacher.lerp_(student, 1.0 - self.teacher_momentum)
        for teacher, student in zip(self.teacher.buffers(), self.model.buffers()):
            teacher.copy_(student)

    @torch.enable_grad()
    def warmup(self, loader: DataLoader, steps: int) -> None:
        """Optional source-only teacher warm-up, before the target stream."""
        iterator = iter(loader)
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            inputs = batch[0].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                targets = self.teacher(inputs)
            loss = symmetric_cross_entropy(self.model(inputs), targets).mean()
            if not torch.isfinite(loss):
                raise ValueError("GOLD source warm-up produced a nonfinite loss")
            loss.backward()
            self.optimizer.step()
            self._update_teacher()
            self.warmup_updates += 1
        self.optimizer.zero_grad(set_to_none=True)

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        self.model.eval()
        self.teacher.eval()
        online_outputs = None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            features = self.model.forward_features(inputs)
            adapted = self.apply_adapter(features)
            outputs = self.model.head(adapted)
            augmented = self.apply_adapter(
                self.model.forward_features(self.augmentation(inputs))
            )
            with torch.no_grad():
                targets = self.teacher(inputs)
                nearest = (
                    F.normalize(adapted, dim=1)
                    @ F.normalize(self.source_prototypes, dim=1).T
                ).argmax(1)
            if online_outputs is None:
                online_outputs = (outputs + targets).detach()
            loss = (
                self.lambda_ce_trg
                * 0.5
                * (
                    symmetric_cross_entropy(outputs, targets)
                    + symmetric_cross_entropy(self.model.head(augmented), targets)
                ).mean()
            )
            if self.lambda_cont > 0:
                views = torch.stack(
                    (self.source_prototypes[nearest], adapted, augmented), dim=1
                )
                loss = loss + self.lambda_cont * multiview_contrastive_loss(
                    self.projector(views),
                    self.temperature,
                )
            batch_matrix = self.batch_agop(features, outputs)
            self.adaptation_iterations += 1
            if torch.isfinite(loss):
                loss.backward()
                if all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for group in self.optimizer.param_groups
                    for p in group["params"]
                ):
                    self.optimizer.step()
                    self._update_teacher()
                    self.updates += 1
                else:
                    self.skipped_updates += 1
            else:
                self.skipped_updates += 1
            self.optimizer.zero_grad(set_to_none=True)
            # Change the basis only after backward has consumed the old basis.
            with torch.no_grad():
                if batch_matrix is not None and torch.isfinite(batch_matrix).all():
                    self.agop.lerp_(batch_matrix, self.alpha)
                    self.agop_updates += 1
                if self.adaptation_iterations % self.update_every == 0:
                    self.basis.copy_(self._eigenbasis())
                    self.basis_updates += 1
        assert online_outputs is not None
        return online_outputs

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.teacher.eval()
        inputs = inputs.to(self.device)
        return self.model.head(
            self.apply_adapter(self.model.forward_features(inputs))
        ) + self.teacher(inputs)

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        values.update(
            parameter_scope=self.parameter_scope,
            adapted_parameters=sum(
                p.numel()
                for group in self.optimizer.param_groups
                for p in group["params"]
            ),
            base_parameter_tensors=len(self.parameter_names),
            subspace_rank=self.top_r,
            feature_dim=self.feature_dim,
            agop_updates=self.agop_updates,
            basis_updates=self.basis_updates,
            warmup_updates=self.warmup_updates,
            source_prototype_classes=self.source_prototypes.shape[0],
        )
        return values
