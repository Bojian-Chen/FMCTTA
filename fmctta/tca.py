"""Numerically stable building blocks for Topological Consistency Adaptation.

The implementation follows the TCA equations from the CVPR 2025 paper and the
authors' historical MIT-licensed implementation at commit
1de3ad71861891b8384cc1b458eaeebfbfd6048f of
https://github.com/Successybbdwm/TCA. The repository's current ``tca.py`` is a
placeholder, so the helpers below also handle missing classes and small batches
that made the historical draft produce NaNs.
"""

import math
import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import OnlineAdapter, SignalAugment, configure_gradient_adaptation


def symmetric_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Return TCA Eq. 11 with a stop-gradient teacher branch.

    ``alpha=0.5`` gives the exact unscaled sum of teacher-to-student and
    student-to-teacher cross-entropies. Other values retain the same total
    coefficient while allowing an explicit directional-weighting ablation.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("TCA symmetric cross-entropy alpha must be in [0, 1]")
    teacher_logits = teacher_logits.detach()
    student_log_probabilities = student_logits.log_softmax(dim=1)
    teacher_log_probabilities = teacher_logits.log_softmax(dim=1)
    student_probabilities = student_logits.softmax(dim=1)
    teacher_probabilities = teacher_logits.softmax(dim=1)
    forward = -(teacher_probabilities * student_log_probabilities).sum(dim=1)
    reverse = -(student_probabilities * teacher_log_probabilities).sum(dim=1)
    return 2.0 * ((1.0 - alpha) * forward + alpha * reverse)


class TCAStateManager(nn.Module):
    """Persistent, detached class-centroid and topology-weight state."""

    def __init__(
        self,
        num_classes: int,
        feature_dimension: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("TCA state requires at least one class")
        if feature_dimension < 1:
            raise ValueError("TCA state requires a positive feature dimension")
        self.num_classes = num_classes
        self.feature_dimension = feature_dimension
        self.register_buffer(
            "historical_class_centers",
            torch.zeros(num_classes, feature_dimension, device=device),
        )
        self.register_buffer(
            "seen_class_mask",
            torch.zeros(num_classes, dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "historical_topology_weights",
            torch.zeros(num_classes, num_classes, device=device),
        )
        self.register_buffer(
            "topology_weights_initialized",
            torch.tensor(False, dtype=torch.bool, device=device),
        )

    @torch.no_grad()
    def reset(self) -> None:
        self.historical_class_centers.zero_()
        self.seen_class_mask.zero_()
        self.historical_topology_weights.zero_()
        self.topology_weights_initialized.fill_(False)

    def update_class_centers(
        self,
        batch_centers: torch.Tensor,
        present_classes: torch.Tensor,
        momentum: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update detached history while returning differentiable EMA centers."""

        if not 0.0 <= momentum < 1.0:
            raise ValueError("TCA centroid momentum must be in [0, 1)")
        expected_centers = (self.num_classes, self.feature_dimension)
        if batch_centers.shape != expected_centers:
            raise ValueError(
                "TCA batch center shape must be "
                f"{expected_centers}, got {tuple(batch_centers.shape)}"
            )
        if present_classes.shape != (self.num_classes,):
            raise ValueError("TCA present-class mask has an invalid shape")

        historical = self.historical_class_centers.detach().clone()
        previously_seen = self.seen_class_mask.detach().clone()
        ema_centers = (
            momentum * historical + (1.0 - momentum) * batch_centers
        )
        initialized_centers = torch.where(
            previously_seen.unsqueeze(1),
            ema_centers,
            batch_centers,
        )
        centers_for_loss = torch.where(
            present_classes.unsqueeze(1),
            initialized_centers,
            historical,
        )
        seen_for_loss = previously_seen | present_classes

        with torch.no_grad():
            self.historical_class_centers.copy_(centers_for_loss.detach())
            self.seen_class_mask.copy_(seen_for_loss)
        return centers_for_loss, seen_for_loss

    def update_topology_weights(
        self,
        batch_weights: torch.Tensor,
        momentum: float,
    ) -> Optional[torch.Tensor]:
        """Apply TCA Eq. 9 and return detached historical edge weights."""

        if not 0.0 <= momentum < 1.0:
            raise ValueError("TCA topology-weight momentum must be in [0, 1)")
        expected = (self.num_classes, self.num_classes)
        if batch_weights.shape != expected:
            raise ValueError(
                f"TCA topology weight shape must be {expected}, "
                f"got {tuple(batch_weights.shape)}"
            )

        was_initialized = bool(self.topology_weights_initialized.item())
        historical = self.historical_topology_weights.detach()
        updated = (
            momentum * historical + (1.0 - momentum) * batch_weights.detach()
            if was_initialized
            else batch_weights.detach()
        )
        with torch.no_grad():
            self.historical_topology_weights.copy_(updated)
            self.topology_weights_initialized.fill_(True)

        # Algorithm 1 uses the unweighted Eq. 4 for the first batch, then the
        # historical Eq. 9 matrix in Eq. 10 for subsequent batches.
        if not was_initialized:
            return None
        return self.historical_topology_weights.detach()


def batch_class_centers(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute differentiable pseudo-class centers and detached class counts."""

    if features.ndim != 2:
        raise ValueError("TCA expects a [batch, feature] embedding tensor")
    if pseudo_labels.shape != (features.shape[0],):
        raise ValueError("TCA pseudo-label shape must match the feature batch")
    assignments = F.one_hot(pseudo_labels, num_classes=num_classes).to(features.dtype)
    counts = assignments.sum(dim=0)
    centers = assignments.transpose(0, 1) @ features
    centers = centers / counts.clamp_min(1.0).unsqueeze(1)
    present = counts > 0
    return centers, counts.detach(), present.detach()


def batch_topology_weights(
    class_counts: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Build the batch-imbalance topology-weighting matrix from Eqs. 7-8."""

    if class_counts.ndim != 1:
        raise ValueError("TCA class counts must be one-dimensional")
    if epsilon <= 0:
        raise ValueError("TCA topology epsilon must be positive")
    roots = class_counts.clamp_min(0).sqrt()
    frequencies = roots / (roots + epsilon).sum().clamp_min(epsilon)
    return (frequencies.unsqueeze(0) + frequencies.unsqueeze(1)) / 2.0


def inter_class_uniformity_loss(
    centers: torch.Tensor,
    valid_classes: torch.Tensor,
    temperature: float = 2.0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the (optionally BTW-weighted) Gaussian-potential loss."""

    if temperature <= 0:
        raise ValueError("TCA uniformity temperature must be positive")
    selected = centers[valid_classes]
    if selected.shape[0] < 2:
        return centers.sum() * 0.0
    distances = torch.pdist(selected, p=2).square()
    potentials = -temperature * distances
    if edge_weights is None:
        return torch.logsumexp(potentials, dim=0) - math.log(distances.numel())

    selected_weights = edge_weights[valid_classes][:, valid_classes]
    pair_indices = torch.triu_indices(
        selected.shape[0],
        selected.shape[0],
        offset=1,
        device=centers.device,
    )
    pair_weights = selected_weights[pair_indices[0], pair_indices[1]].detach()
    positive = pair_weights > 0
    if not bool(positive.any()):
        return torch.logsumexp(potentials, dim=0) - math.log(distances.numel())
    log_weights = pair_weights[positive].log()
    return torch.logsumexp(log_weights + potentials[positive], dim=0) - torch.logsumexp(
        log_weights,
        dim=0,
    )


def intra_class_uniformity_loss(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Compute TCA Eq. 5 while safely skipping singleton pseudo-classes."""

    if temperature <= 0:
        raise ValueError("TCA uniformity temperature must be positive")
    class_potentials = []
    for pseudo_class in pseudo_labels.unique():
        class_features = features[pseudo_labels == pseudo_class]
        if class_features.shape[0] < 2:
            continue
        mean_squared_distance = torch.pdist(class_features, p=2).square().mean()
        class_potentials.append(-temperature * mean_squared_distance)
    if not class_potentials:
        return features.sum() * 0.0
    stacked = torch.stack(class_potentials)
    return torch.logsumexp(stacked, dim=0) - math.log(stacked.numel())


def intra_class_compactness_loss(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Pull normalized features toward their differentiable pseudo-class mean."""

    centers, _, _ = batch_class_centers(features, pseudo_labels, num_classes)
    assigned_centers = centers[pseudo_labels]
    return (features - assigned_centers).square().sum(dim=1).mean()


def topology_consistency_loss(
    target_centers: torch.Tensor,
    source_topology: torch.Tensor,
    valid_classes: torch.Tensor,
) -> torch.Tensor:
    """Match target/source pairwise cosine topology for currently seen classes."""

    if target_centers.ndim != 2:
        raise ValueError("Target class centers must have shape [classes, features]")
    num_classes = target_centers.shape[0]
    if source_topology.shape != (num_classes, num_classes):
        raise ValueError(
            f"Source topology must have shape [{num_classes}, {num_classes}]"
        )
    if valid_classes.shape != (num_classes,):
        raise ValueError("Valid-class mask must have one entry per class")

    selected_centers = F.normalize(target_centers[valid_classes], dim=1)
    selected_source = source_topology[valid_classes][:, valid_classes]
    if selected_centers.shape[0] < 2:
        return target_centers.sum() * 0.0
    target_topology = selected_centers @ selected_centers.T
    pair_indices = torch.triu_indices(
        selected_centers.shape[0],
        selected_centers.shape[0],
        offset=1,
        device=target_centers.device,
    )
    target_pairs = target_topology[pair_indices[0], pair_indices[1]]
    source_pairs = selected_source[pair_indices[0], pair_indices[1]].detach()
    return F.mse_loss(target_pairs, source_pairs)


def feature_alignment_loss(
    first_view: torch.Tensor,
    second_view: torch.Tensor,
) -> torch.Tensor:
    """Align two augmented views with the standard positive squared distance."""

    if first_view.shape != second_view.shape:
        raise ValueError("TCA alignment views must have identical shapes")
    return (first_view - second_view).square().sum(dim=1).mean()

class TCAAdapter(OnlineAdapter):
    """Mean-teacher adaptation with class-topology consistency constraints."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        num_classes: int,
        teacher_momentum: float = 0.999,
        centroid_momentum: float = 0.9,
        topology_weight_momentum: float = 0.8,
        alignment_weight: float = 0.025,
        topology_weight: float = 0.15,
        uniformity_temperature: float = 2.0,
        symmetric_alpha: float = 0.5,
        augmentation_noise_std: float = 0.005,
        augmentation_scale_std: float = 0.05,
        augmentation_mask_probability: float = 0.0,
        augmentation_shift_ratio: float = 0.0,
        weight_decay: float = 0.0,
        steps: int = 1,
        parameter_scope: str = "all",
    ) -> None:
        if num_classes < 2:
            raise ValueError("TCA requires at least two classes")
        for name, value in {
            "teacher momentum": teacher_momentum,
            "centroid momentum": centroid_momentum,
            "topology-weight momentum": topology_weight_momentum,
        }.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"TCA {name} must be in [0, 1)")
        if alignment_weight < 0 or topology_weight < 0:
            raise ValueError("TCA loss weights cannot be negative")
        if uniformity_temperature <= 0:
            raise ValueError("TCA uniformity temperature must be positive")
        if not 0.0 <= symmetric_alpha <= 1.0:
            raise ValueError("TCA symmetric alpha must be in [0, 1]")

        super().__init__(copy.deepcopy(model), device)
        parameters, parameter_names = configure_gradient_adaptation(
            self.model,
            parameter_scope,
        )
        feature_dimension = int(getattr(self.model, "feature_dim", 0))
        if feature_dimension < 1:
            raise ValueError("TCA requires model.feature_dim")

        self.teacher = copy.deepcopy(self.model).to(device)
        self.teacher.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.augment = SignalAugment(
            noise_std=augmentation_noise_std,
            scale_std=augmentation_scale_std,
            mask_probability=augmentation_mask_probability,
            shift_ratio=augmentation_shift_ratio,
        )
        self.topology_state = TCAStateManager(
            num_classes,
            feature_dimension,
            device=device,
        )
        self.num_classes = num_classes
        self.teacher_momentum = teacher_momentum
        self.centroid_momentum = centroid_momentum
        self.topology_weight_momentum = topology_weight_momentum
        self.alignment_weight = alignment_weight
        self.topology_weight = topology_weight
        self.uniformity_temperature = uniformity_temperature
        self.symmetric_alpha = symmetric_alpha
        self.steps = steps
        self.parameter_scope = parameter_scope
        self.parameter_names = tuple(parameter_names)
        self.augmentation_views = 0
        self.topology_weight_updates = 0
        self.teacher_updates = 0
        self.last_self_training_loss: Optional[float] = None
        self.last_alignment_loss: Optional[float] = None
        self.last_inter_class_loss: Optional[float] = None
        self.last_intra_class_loss: Optional[float] = None
        self.last_total_loss: Optional[float] = None
        self.model.train()
        self.teacher.eval()

    @staticmethod
    def _features_and_logits(
        model: nn.Module,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = model.forward_features(inputs)
        return features, model.head(features)

    def _update_topology(
        self,
        features: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_centers, class_counts, present = batch_class_centers(
            features,
            pseudo_labels,
            self.num_classes,
        )
        centers_for_loss, seen_classes = self.topology_state.update_class_centers(
            batch_centers,
            present,
            self.centroid_momentum,
        )
        current_weights = batch_topology_weights(class_counts)
        edge_weights = self.topology_state.update_topology_weights(
            current_weights,
            self.topology_weight_momentum,
        )
        self.topology_weight_updates += 1
        return centers_for_loss, seen_classes, edge_weights

    @torch.no_grad()
    def _update_teacher(self) -> None:
        teacher_state = self.teacher.state_dict()
        student_state = self.model.state_dict()
        for name, teacher_value in teacher_state.items():
            student_value = student_state[name]
            if teacher_value.is_floating_point():
                teacher_value.mul_(self.teacher_momentum).add_(
                    student_value,
                    alpha=1.0 - self.teacher_momentum,
                )
            else:
                teacher_value.copy_(student_value)
        self.teacher_updates += 1

    def _adapt_once(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.train()
        self.teacher.eval()
        with torch.no_grad():
            _, teacher_outputs = self._features_and_logits(self.teacher, inputs)

        _, original_outputs = self._features_and_logits(
            self.model,
            inputs,
        )
        topology_inputs = self.augment(inputs)
        alignment_inputs = self.augment(inputs)
        self.augmentation_views += 2
        topology_features, topology_outputs = self._features_and_logits(
            self.model,
            topology_inputs,
        )
        alignment_features, _ = self._features_and_logits(
            self.model,
            alignment_inputs,
        )
        pseudo_labels = topology_outputs.detach().argmax(dim=1)
        normalized_topology_features = F.normalize(topology_features, dim=1)
        normalized_alignment_features = F.normalize(alignment_features, dim=1)
        centers, valid_classes, edge_weights = self._update_topology(
            normalized_topology_features,
            pseudo_labels,
        )
        normalized_centers = F.normalize(centers, dim=1)

        self_training_loss = 0.5 * (
            symmetric_cross_entropy(
                original_outputs,
                teacher_outputs,
                alpha=self.symmetric_alpha,
            ).mean()
            + symmetric_cross_entropy(
                topology_outputs,
                teacher_outputs,
                alpha=self.symmetric_alpha,
            ).mean()
        )
        alignment_loss = feature_alignment_loss(
            normalized_alignment_features,
            normalized_topology_features,
        )
        inter_class_loss = inter_class_uniformity_loss(
            normalized_centers,
            valid_classes,
            temperature=self.uniformity_temperature,
            edge_weights=edge_weights,
        )
        intra_class_loss = intra_class_uniformity_loss(
            normalized_topology_features,
            pseudo_labels,
            temperature=self.uniformity_temperature,
        )
        topology_loss = inter_class_loss + intra_class_loss
        loss = (
            self_training_loss
            + self.alignment_weight * alignment_loss
            + self.topology_weight * topology_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_teacher()
        self.updates += 1
        self.last_self_training_loss = float(self_training_loss.detach().item())
        self.last_alignment_loss = float(alignment_loss.detach().item())
        self.last_inter_class_loss = float(inter_class_loss.detach().item())
        self.last_intra_class_loss = float(intra_class_loss.detach().item())
        self.last_total_loss = float(loss.detach().item())
        return (0.5 * original_outputs + 0.5 * teacher_outputs).detach()

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        online_outputs: Optional[torch.Tensor] = None
        for _ in range(self.steps):
            current_outputs = self._adapt_once(inputs)
            if online_outputs is None:
                online_outputs = current_outputs
        assert online_outputs is not None
        return online_outputs

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        self.model.eval()
        self.teacher.eval()
        _, student_outputs = self._features_and_logits(self.model, inputs)
        _, teacher_outputs = self._features_and_logits(self.teacher, inputs)
        return 0.5 * student_outputs + 0.5 * teacher_outputs

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
                "teacher_momentum": self.teacher_momentum,
                "centroid_momentum": self.centroid_momentum,
                "topology_weight_momentum": self.topology_weight_momentum,
                "alignment_weight": self.alignment_weight,
                "topology_weight": self.topology_weight,
                "uniformity_temperature": self.uniformity_temperature,
                "symmetric_alpha": self.symmetric_alpha,
                "class_centers_seen": int(
                    self.topology_state.seen_class_mask.sum().item()
                ),
                "augmentation_views": self.augmentation_views,
                "topology_weight_updates": self.topology_weight_updates,
                "teacher_updates": self.teacher_updates,
                "last_self_training_loss": self.last_self_training_loss,
                "last_alignment_loss": self.last_alignment_loss,
                "last_inter_class_loss": self.last_inter_class_loss,
                "last_intra_class_loss": self.last_intra_class_loss,
                "last_total_loss": self.last_total_loss,
            }
        )
        return values

