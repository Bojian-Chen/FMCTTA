from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unifreq import UniFreqConfig, UniFreqProcessor, load_model


class CosineClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        self.scale = nn.Parameter(torch.tensor(10.0))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(
            F.normalize(features, dim=1),
            F.normalize(self.weight, dim=1),
        )


class MLPClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, feature_dim // 2)
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_classifier(
    name: str,
    feature_dim: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    if name == "mlp":
        return MLPClassifier(feature_dim, num_classes, dropout)
    if name == "linear":
        return nn.Linear(feature_dim, num_classes)
    if name == "cosine":
        return CosineClassifier(feature_dim, num_classes)
    raise ValueError(f"Unknown classifier: {name}")


class UniFreqFaultModel(nn.Module):
    """A pretrained UniFreq encoder with a downstream fault classifier."""

    def __init__(
        self,
        encoder: nn.Module,
        config: UniFreqConfig,
        processor: UniFreqProcessor,
        head: nn.Module,
        model_id: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.processor = processor
        self.head = head
        self.model_id = model_id

    @property
    def backbone(self) -> nn.Module:
        return self.encoder

    @property
    def feature_dim(self) -> int:
        multiplier = 2 if self.config.timefreq_fusion == "concat" else 1
        return int(self.config.embed_dim) * multiplier

    def _prepare(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if inputs.ndim not in {2, 3}:
            raise ValueError(
                "UniFreq expects [B, L] or [B, C, L] raw signals, got "
                f"{tuple(inputs.shape)}"
            )
        device = next(self.encoder.parameters()).device
        inputs = inputs.to(device=device, dtype=torch.float32)
        batch = self.processor(inputs)
        return {name: value.to(device) for name, value in batch.items()}

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._prepare(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))

    def enable_encoder_finetuning(self) -> None:
        if hasattr(self.encoder, "enable_finetuning"):
            self.encoder.enable_finetuning()
        else:
            self.encoder.requires_grad_(True)

    def disable_encoder_finetuning(self) -> None:
        if hasattr(self.encoder, "disable_finetuning"):
            self.encoder.disable_finetuning()
        else:
            self.encoder.requires_grad_(False)
            self.encoder.eval()


def build_model(
    model_id: str,
    num_classes: int,
    classifier: str = "mlp",
    classifier_dropout: float = 0.1,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> Tuple[UniFreqFaultModel, UniFreqConfig]:
    encoder, config = load_model(
        model_id,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    processor = UniFreqProcessor(config)
    multiplier = 2 if config.timefreq_fusion == "concat" else 1
    feature_dim = int(config.embed_dim) * multiplier
    head = build_classifier(
        classifier,
        feature_dim,
        num_classes,
        classifier_dropout,
    )
    model = UniFreqFaultModel(encoder, config, processor, head, model_id)
    model.disable_encoder_finetuning()
    return model, config
