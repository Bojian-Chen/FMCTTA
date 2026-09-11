"""GOLD adaptation with angular class-topology preservation.

This variant keeps GOLD's AGOP subspace, scaling adapter, EMA teacher,
self-training, and source-prototype contrastive objective.  TCA contributes
only persistent pseudo-class centers and topology losses, evaluated on the
normalized GOLD-adapted features that enter a frozen cosine classifier.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gold import GOLDAdapter
from .model import CosineClassifier
from .tca import (
    TCAStateManager,
    batch_class_centers,
    inter_class_uniformity_loss,
)


def intra_class_compactness_loss(
    normalized_features: torch.Tensor,
    pseudo_labels: torch.Tensor,
) -> torch.Tensor:
    """Mean within-class angular dispersion, safely ignoring singletons."""

    class_losses = []
    for pseudo_class in pseudo_labels.unique():
        class_features = normalized_features[pseudo_labels == pseudo_class]
        if class_features.shape[0] < 2:
            continue
        class_losses.append(torch.pdist(class_features, p=2).square().mean())
    if not class_losses:
        return normalized_features.sum() * 0.0
    return torch.stack(class_losses).mean()


class GOLDTCAAdapter(GOLDAdapter):
    """Selective GOLD plasticity regularized by persistent class topology."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        num_classes: int,
        source_prototypes: torch.Tensor,
        *,
        lambda_topology: float = 0.15,
        centroid_momentum: float = 0.9,
        uniformity_temperature: float = 2.0,
        **gold_kwargs,
    ) -> None:
        if not isinstance(getattr(model, "head", None), CosineClassifier):
            raise ValueError("gold_tca requires the cosine classifier")
        if gold_kwargs.get("parameter_scope") == "classifier":
            raise ValueError(
                "gold_tca freezes the classifier; classifier parameter scope is invalid"
            )
        if num_classes < 2:
            raise ValueError("gold_tca requires at least two classes")
        if lambda_topology < 0:
            raise ValueError("gold_tca topology weight cannot be negative")
        if not 0.0 <= centroid_momentum < 1.0:
            raise ValueError("gold_tca centroid momentum must be in [0, 1)")
        if uniformity_temperature <= 0:
            raise ValueError("gold_tca uniformity temperature must be positive")

        super().__init__(
            model,
            device,
            learning_rate,
            num_classes,
            source_prototypes,
            **gold_kwargs,
        )

        # The classifier defines the fixed angular geometry.  GOLD may adapt
        # any requested backbone scope, plus its projector and scaling vector.
        head_parameter_ids = {
            id(parameter) for parameter in self.model.head.parameters()
        }
        self.model.head.requires_grad_(False)
        self.parameter_names = tuple(
            name for name in self.parameter_names if not name.startswith("head.")
        )
        for group in self.optimizer.param_groups:
            group["params"] = [
                parameter
                for parameter in group["params"]
                if id(parameter) not in head_parameter_ids
            ]
        self.optimizer.param_groups[:] = [
            group for group in self.optimizer.param_groups if group["params"]
        ]

        self.num_classes = num_classes
        self.lambda_topology = lambda_topology
        self.centroid_momentum = centroid_momentum
        self.uniformity_temperature = uniformity_temperature
        self.topology_state = TCAStateManager(
            num_classes,
            self.feature_dim,
            device=device,
        )
        self.topology_updates = 0
        self.last_inter_class_loss: Optional[float] = None
        self.last_intra_class_loss: Optional[float] = None
        self.last_topology_loss: Optional[float] = None

    def _update_class_centers(
        self,
        normalized_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_centers, _, present_classes = batch_class_centers(
            normalized_features,
            pseudo_labels,
            self.num_classes,
        )
        return self.topology_state.update_class_centers(
            batch_centers,
            present_classes,
            self.centroid_momentum,
        )

    def additional_regularization(
        self,
        adapted_features: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        normalized_features = F.normalize(adapted_features, dim=1)
        pseudo_labels = student_logits.detach().argmax(dim=1)
        centers, seen_classes = self._update_class_centers(
            normalized_features,
            pseudo_labels,
        )
        normalized_centers = F.normalize(centers, dim=1)
        inter_class_loss = inter_class_uniformity_loss(
            normalized_centers,
            seen_classes,
            temperature=self.uniformity_temperature,
        )
        intra_class_loss = intra_class_compactness_loss(
            normalized_features,
            pseudo_labels,
        )
        topology_loss = inter_class_loss + intra_class_loss

        self.topology_updates += 1
        self.last_inter_class_loss = float(inter_class_loss.detach().item())
        self.last_intra_class_loss = float(intra_class_loss.detach().item())
        self.last_topology_loss = float(topology_loss.detach().item())
        return self.lambda_topology * topology_loss

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        values.update(
            classifier="cosine",
            classifier_frozen=not any(
                parameter.requires_grad for parameter in self.model.head.parameters()
            ),
            lambda_topology=self.lambda_topology,
            centroid_momentum=self.centroid_momentum,
            uniformity_temperature=self.uniformity_temperature,
            class_centers_seen=int(self.topology_state.seen_class_mask.sum().item()),
            topology_updates=self.topology_updates,
            last_inter_class_loss=self.last_inter_class_loss,
            last_intra_class_loss=self.last_intra_class_loss,
            last_topology_loss=self.last_topology_loss,
        )
        return values
