from typing import Dict, Mapping, Optional

import torch


def relative_improvement(
    current: Mapping[str, float], baseline: Mapping[str, float]
) -> Dict[str, Optional[float]]:
    """Compare percentage-scale classification scores on identical samples."""
    result = {}
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        difference = current[name] - baseline[name]
        result[f"{name}_gain_pp"] = difference
        result[f"{name}_relative_gain_pct"] = (
            100.0 * difference / baseline[name] if baseline[name] != 0 else None
        )
    source_error = 100.0 - baseline["accuracy"]
    result["error_reduction_pct"] = (
        100.0 * (current["accuracy"] - baseline["accuracy"]) / source_error
        if source_error != 0
        else None
    )
    return result


class ClassificationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        self.loss_sum = 0.0
        self.confidence_sum = 0.0
        self.entropy_sum = 0.0
        self.samples = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits = logits.detach().cpu()
        labels = labels.detach().cpu().long()
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        indices = labels * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes * self.num_classes)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)
        self.loss_sum += float(
            torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        )
        self.confidence_sum += float(probabilities.amax(dim=1).sum().item())
        self.entropy_sum += float(
            (-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1))
            .sum()
            .item()
        )
        self.samples += labels.numel()

    def merge(self, other: "ClassificationMeter") -> None:
        if self.num_classes != other.num_classes:
            raise ValueError("Cannot merge meters with different class counts")
        self.confusion += other.confusion
        self.loss_sum += other.loss_sum
        self.confidence_sum += other.confidence_sum
        self.entropy_sum += other.entropy_sum
        self.samples += other.samples

    def compute(self) -> Dict[str, float]:
        if self.samples == 0:
            raise ValueError("Cannot compute metrics without samples")
        matrix = self.confusion.float()
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        recall = true_positive / support.clamp_min(1.0)
        precision = true_positive / predicted.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
        present = support > 0
        return {
            "samples": float(self.samples),
            "accuracy": 100.0 * float(true_positive.sum().item()) / self.samples,
            "balanced_accuracy": 100.0 * float(recall[present].mean().item()),
            "macro_f1": 100.0 * float(f1[present].mean().item()),
            "cross_entropy": self.loss_sum / self.samples,
            "mean_confidence": self.confidence_sum / self.samples,
            "mean_entropy": self.entropy_sum / self.samples,
        }
