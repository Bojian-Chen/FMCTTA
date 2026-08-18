"""Source-reference k-nearest-neighbor baseline for UniFreq features."""

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adapters import OnlineAdapter


@torch.no_grad()
def build_reference_bank(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode the labeled source-train split once for kNN lookup."""

    model.eval()
    feature_batches = []
    label_batches = []
    for inputs, labels, _ in loader:
        features = model.forward_features(inputs.to(device))
        feature_batches.append(features.detach())
        label_batches.append(labels.to(device=device, dtype=torch.long))
    if not feature_batches:
        raise ValueError("Cannot build a kNN reference bank from an empty loader")
    return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)


class KNNAdapter(OnlineAdapter):
    """Non-adaptive kNN classifier over source-train UniFreq features."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        reference_features: torch.Tensor,
        reference_labels: torch.Tensor,
        num_classes: int,
        neighbors: int,
    ) -> None:
        if reference_features.ndim != 2:
            raise ValueError(
                "kNN reference_features must have shape [samples, features]"
            )
        if reference_labels.ndim != 1:
            raise ValueError("kNN reference_labels must have shape [samples]")
        if reference_features.shape[0] != reference_labels.shape[0]:
            raise ValueError("kNN reference features and labels have different sizes")
        if reference_features.shape[0] == 0:
            raise ValueError("kNN reference bank must contain at least one sample")
        if neighbors < 1 or neighbors > reference_features.shape[0]:
            raise ValueError(
                "knn_neighbors must be between 1 and the number of source "
                f"reference samples ({reference_features.shape[0]})"
            )
        if (
            reference_labels.min().item() < 0
            or reference_labels.max().item() >= num_classes
        ):
            raise ValueError("kNN reference labels must be valid class indices")

        super().__init__(copy.deepcopy(model), device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.register_buffer(
            "reference_features",
            F.normalize(
                reference_features.to(device=device, dtype=torch.float32),
                dim=1,
            ),
        )
        self.register_buffer(
            "reference_labels",
            reference_labels.to(device=device, dtype=torch.long),
        )
        self.num_classes = int(num_classes)
        self.neighbors = int(neighbors)

    @torch.no_grad()
    def _classify(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(inputs.to(self.device))
        if features.shape[1] != self.reference_features.shape[1]:
            raise RuntimeError(
                "kNN query feature dimension differs from the source reference "
                f"bank: {features.shape[1]} versus {self.reference_features.shape[1]}"
            )
        similarities = F.normalize(features, dim=1) @ self.reference_features.T
        indices = similarities.topk(self.neighbors, dim=1).indices
        labels = self.reference_labels[indices]
        votes = features.new_zeros(features.shape[0], self.num_classes)
        votes.scatter_add_(1, labels, torch.ones_like(labels, dtype=votes.dtype))
        # Log vote counts preserve the kNN argmax while yielding meaningful
        # probabilities for the existing prediction CSV confidence fields.
        return votes.clamp_min(1e-12).log()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        return self._classify(inputs)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._classify(inputs)

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        values.update(
            {
                "reference_samples": int(self.reference_features.shape[0]),
                "reference_feature_dim": int(self.reference_features.shape[1]),
                "neighbors": self.neighbors,
                "distance": "cosine",
                "vote_weighting": "uniform",
            }
        )
        return values
