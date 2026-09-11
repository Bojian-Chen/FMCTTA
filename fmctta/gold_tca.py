"""GOLD selective adaptation with TCA topology preservation."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gold import GOLDAdapter
from .model import CosineClassifier
from .tca import (
    TCAStateManager,
    batch_class_centers,
    intra_class_compactness_loss,
    topology_consistency_loss,
)


class GOLDTCAAdapter(GOLDAdapter):
    """Add persistent class topology to GOLD's adapted angular features."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        num_classes: int,
        source_prototypes: Optional[torch.Tensor] = None,
        topology_weight: float = 0.15,
        centroid_momentum: float = 0.9,
        agop_initialization: str = "classifier_weights",
        **gold_options,
    ) -> None:
        if not isinstance(getattr(model, "head", None), CosineClassifier):
            raise ValueError("GOLD-TCA requires the cosine classifier")
        if gold_options.get("parameter_scope", "all") == "classifier":
            raise ValueError(
                "GOLD-TCA freezes the classifier; classifier scope is invalid"
            )
        if num_classes < 2:
            raise ValueError("GOLD-TCA requires at least two classes")
        if topology_weight < 0:
            raise ValueError("GOLD-TCA topology weight cannot be negative")
        if not 0.0 <= centroid_momentum < 1.0:
            raise ValueError("GOLD-TCA centroid momentum must be in [0, 1)")

        super().__init__(
            model,
            device,
            learning_rate,
            num_classes,
            source_prototypes,
            freeze_classifier=True,
            agop_initialization=agop_initialization,
            **gold_options,
        )
        self.topology_state = TCAStateManager(
            num_classes,
            self.feature_dim,
            device=device,
        )
        self.register_buffer(
            "prototype_memory",
            self.source_prototypes.detach().clone(),
        )
        self.prototype_state: Optional[TCAStateManager] = None
        if self.agop_initialization == "source_prototypes":
            self.prototype_state = TCAStateManager(
                num_classes,
                self.feature_dim,
                device=device,
            )
            with torch.no_grad():
                self.prototype_state.historical_class_centers.copy_(
                    self.prototype_memory
                )
                self.prototype_state.seen_class_mask.fill_(True)
        self.num_classes = num_classes
        self.topology_weight = topology_weight
        self.centroid_momentum = centroid_momentum
        normalized_weights = F.normalize(self.model.head.weight.detach().float(), dim=1)
        self.register_buffer(
            "source_topology",
            normalized_weights @ normalized_weights.T,
        )
        self.topology_updates = 0
        self.prototype_updates = 0
        self.last_gold_loss: Optional[float] = None
        self.last_topology_consistency_loss: Optional[float] = None
        self.last_intra_class_loss: Optional[float] = None
        self.last_topology_loss: Optional[float] = None
        self.last_total_loss: Optional[float] = None

    @torch.no_grad()
    def contrastive_anchors(
        self,
        adapted_features: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Select the configured persistent semantic anchor for GOLD contrast."""

        pseudo_labels = logits.argmax(dim=1)
        if self.prototype_state is not None:
            return self.prototype_memory[pseudo_labels]
        normalized_weights = F.normalize(self.model.head.weight, dim=1)
        return normalized_weights[pseudo_labels]

    @torch.enable_grad()
    def batch_agop(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Use evolving prototypes only in the prototype-memory mode."""

        if self.prototype_state is not None:
            return self.prototype_agop(self.prototype_memory)
        return super().batch_agop(features, logits)

    def adaptation_regularizer(
        self,
        adapted_features: torch.Tensor,
        logits: torch.Tensor,
        gold_loss: torch.Tensor,
    ) -> torch.Tensor:
        normalized_features = F.normalize(adapted_features, dim=1)
        pseudo_labels = logits.detach().argmax(dim=1)
        batch_centers, _, present_classes = batch_class_centers(
            normalized_features,
            pseudo_labels,
            self.num_classes,
        )
        centers, valid_classes = self.topology_state.update_class_centers(
            batch_centers,
            present_classes,
            self.centroid_momentum,
        )
        if self.prototype_state is not None:
            prototype_batch_centers, _, prototype_present_classes = batch_class_centers(
                adapted_features.detach(),
                pseudo_labels,
                self.num_classes,
            )
            updated_prototypes, _ = self.prototype_state.update_class_centers(
                prototype_batch_centers,
                prototype_present_classes,
                self.centroid_momentum,
            )
            with torch.no_grad():
                self.prototype_memory.copy_(updated_prototypes.detach())
            self.prototype_updates += 1

        topology_loss = topology_consistency_loss(
            centers,
            self.source_topology,
            valid_classes,
        )
        intra_class_loss = intra_class_compactness_loss(
            normalized_features,
            pseudo_labels,
            self.num_classes,
        )
        structural_loss = topology_loss + intra_class_loss
        weighted_topology_loss = self.topology_weight * structural_loss

        self.topology_updates += 1
        self.last_gold_loss = float(gold_loss.detach().item())
        self.last_topology_consistency_loss = float(topology_loss.detach().item())
        self.last_intra_class_loss = float(intra_class_loss.detach().item())
        self.last_topology_loss = float(structural_loss.detach().item())
        self.last_total_loss = float(
            (gold_loss.detach() + weighted_topology_loss.detach()).item()
        )
        return weighted_topology_loss

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        historical_centers = self.topology_state.historical_class_centers.detach()
        normalized_centers = F.normalize(historical_centers, dim=1)
        target_topology = normalized_centers @ normalized_centers.T
        prototype_mode = self.prototype_state is not None
        values.update(
            method_components=[
                "golden_subspace",
                "persistent_class_topology",
                "cosine_classifier",
            ],
            topology_weight=self.topology_weight,
            centroid_momentum=self.centroid_momentum,
            semantic_anchor=(
                "persistent_prototype_memory"
                if prototype_mode
                else "frozen_cosine_classifier_weight"
            ),
            dynamic_agop_source=(
                "persistent_prototype_memory"
                if prototype_mode
                else "reliable_target_samples"
            ),
            prototype_updates=self.prototype_updates,
            source_topology_reference="frozen_cosine_classifier_weight_gram",
            topology_distance="off_diagonal_mean_squared_error",
            class_centers_seen=int(self.topology_state.seen_class_mask.sum().item()),
            seen_class_mask=self.topology_state.seen_class_mask.detach().cpu().tolist(),
            source_topology=self.source_topology.detach().cpu().tolist(),
            target_topology=target_topology.cpu().tolist(),
            topology_updates=self.topology_updates,
            last_gold_loss=self.last_gold_loss,
            last_topology_consistency_loss=self.last_topology_consistency_loss,
            last_intra_class_loss=self.last_intra_class_loss,
            last_topology_loss=self.last_topology_loss,
            last_total_loss=self.last_total_loss,
        )
        return values
