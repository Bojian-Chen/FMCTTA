from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unifreq import UniFreqConfig, UniFreqProcessor, load_model
from .unifreq.time_frequency_common import (
    _flatten_channel_samples,
    _masked_mean,
    _mean_channel_features,
)


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
        feature_branch: str = "fused",
    ) -> None:
        super().__init__()
        if feature_branch not in {"fused", "time", "frequency"}:
            raise ValueError(
                "feature_branch must be fused, time, or frequency; got "
                f"{feature_branch!r}"
            )
        self.encoder = encoder
        self.config = config
        self.processor = processor
        self.head = head
        self.model_id = model_id
        self.feature_branch = feature_branch

    @property
    def backbone(self) -> nn.Module:
        return self.encoder

    @property
    def feature_dim(self) -> int:
        if self.feature_branch != "fused":
            return int(self.config.embed_dim)
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
        batch = self._prepare(inputs)
        if self.feature_branch == "fused":
            return self.encoder(batch)
        if self.feature_branch == "time":
            return self.encoder.time_features(batch)
        return self.encoder.frequency_features(batch)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))

    def frequency_token_batch(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """Prepare one-token-per-frequency-bin inputs for Continual-MAE.

        Continual-MAE treats every non-DC global-FFT bin as one patch.  The
        two alternative UniFreq frequency tokenizers combine several bins (or
        use local time windows), so they do not preserve this definition.
        """

        if getattr(self.encoder, "freq_tokenizer_name", None) != "global_fft":
            raise ValueError(
                "Frequency-bin Continual-MAE requires freq_tokenizer="
                "'global_fft' so each frequency point is one token"
            )
        batch = self._prepare(inputs)
        _, _, freq_x, freq_mask, freq_tail_stats = self.encoder._unpack(batch)
        if freq_x is None:
            raise RuntimeError("global_fft tokenization did not provide freq_x")
        freq_x, freq_mask, batch_size, num_channels = _flatten_channel_samples(
            freq_x,
            freq_mask,
        )
        tokens, token_mask = self.encoder.freq_tokenizer(
            freq_x,
            freq_mask=freq_mask,
        )
        flattened_tail_stats = self.encoder._flatten_freq_tail_stats(
            freq_tail_stats,
            batch_size,
            num_channels,
            tokens.device,
            tokens.dtype,
        )
        return {
            "batch": batch,
            "frequency_values": freq_x.squeeze(1),
            "tokens": tokens,
            "token_mask": token_mask,
            "freq_tail_stats": flattened_tail_stats,
            "batch_size": batch_size,
            "num_channels": num_channels,
        }

    def encode_frequency_tokens(
        self,
        token_batch: Dict[str, Any],
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode frequency-bin tokens prepared by :meth:`frequency_token_batch`."""

        values = token_batch["tokens"] if tokens is None else tokens
        return self.encoder._encode_transformer(
            self.encoder.freq_encoder,
            values,
            token_batch["token_mask"],
        )

    def classify_frequency_tokens(
        self,
        token_batch: Dict[str, Any],
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify a masked frequency-bin token sequence and its encoding."""

        if self.feature_branch == "time":
            raise ValueError(
                "Frequency-bin Continual-MAE requires feature_branch="
                "'frequency' or 'fused'"
            )
        encoded = self.encode_frequency_tokens(token_batch, tokens)
        frequency_feature = _masked_mean(encoded, token_batch["token_mask"])
        frequency_feature = self.encoder._apply_freq_tail_feature(
            frequency_feature,
            token_batch["freq_tail_stats"],
        )
        frequency_feature = _mean_channel_features(
            frequency_feature,
            token_batch["batch_size"],
            token_batch["num_channels"],
        )
        if self.feature_branch == "frequency":
            features = frequency_feature
        else:
            time_feature = self.encoder.time_features(token_batch["batch"])
            if self.encoder.fusion_mode == "concat":
                features = torch.cat([time_feature, frequency_feature], dim=1)
            else:
                if self.encoder.fusion is None:
                    raise RuntimeError("UniFreq learned-gate fusion is unavailable")
                features = self.encoder.fusion(time_feature, frequency_feature)
        return self.head(features), encoded

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
    feature_branch: str = "fused",
) -> Tuple[UniFreqFaultModel, UniFreqConfig]:
    encoder, config = load_model(
        model_id,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    processor = UniFreqProcessor(config)
    if feature_branch not in {"fused", "time", "frequency"}:
        raise ValueError(
            "feature_branch must be fused, time, or frequency; got "
            f"{feature_branch!r}"
        )
    multiplier = (
        2
        if feature_branch == "fused" and config.timefreq_fusion == "concat"
        else 1
    )
    feature_dim = int(config.embed_dim) * multiplier
    head = build_classifier(
        classifier,
        feature_dim,
        num_classes,
        classifier_dropout,
    )
    model = UniFreqFaultModel(
        encoder,
        config,
        processor,
        head,
        model_id,
        feature_branch=feature_branch,
    )
    model.disable_encoder_finetuning()
    return model, config
