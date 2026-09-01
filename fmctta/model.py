from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unifreq import UniFreqConfig, UniFreqProcessor, load_model
from .unifreq.time_frequency_common import (
    _flatten_channel_samples,
    _masked_mean,
    _mean_channel_features,
)
from .phmfd_models import TFProFMBackbone, UniFaultBackbone


PHMFD_ROOT = Path("/data1/PHM-Vibench/PHMFD")
AUTO_PRETRAINED_CHECKPOINTS = {
    ("unifault", "tiny"): PHMFD_ROOT
    / "log/unifault/unifault_default_tiny_seed42-93037bea/pretrain/best_pretrain.pt",
    ("unifault", "small"): PHMFD_ROOT
    / "log/unifault/unifault_default_seed42-9515bd26/pretrain/best_pretrain.pt",
    ("tf_profm", "tiny"): PHMFD_ROOT
    / "log/TF-ProFM/tf_profm_default-2c740181/pretrain/best_pretrain.pt",
}

UNIFAULT_SIZE_CONFIGS = {
    "tiny": {"embed_dim": 128, "heads": 4, "depth": 4},
    "small": {"embed_dim": 256, "heads": 8, "depth": 8},
    "base": {"embed_dim": 512, "heads": 12, "depth": 16},
}

TF_PROFM_DEFAULT_CONFIG = {
    "input_len": 2048,
    "patch_len": 256,
    "num_patches": 16,
    "d_model": 256,
    "d_ffn": 512,
    "layers": 6,
    "heads": 4,
    "feature_dim": 4096,
}


@dataclass(frozen=True)
class FoundationModelConfig:
    """Serializable description of the instantiated foundation model."""

    family: str
    architecture: Dict[str, Any]
    input_length: Optional[int]
    input_normalization: str
    pretrained_checkpoint: Optional[str]
    loaded_tensors: int = 0
    backbone_tensors: int = 0
    loaded_parameters: int = 0
    backbone_parameters: int = 0
    unifreq_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


class PHMFDFaultModel(nn.Module):
    """A PHMFD backbone with model-specific preprocessing and an FMCTTA head."""

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        family: str,
        feature_dim: int,
        input_length: int,
        input_channels: int,
        input_normalization: str,
        length_mode: str,
        pretrained_checkpoint: Optional[str],
    ) -> None:
        super().__init__()
        if family not in {"unifault", "tf_profm"}:
            raise ValueError(f"Unknown PHMFD model family: {family}")
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if input_normalization not in {
            "none",
            "sample_zscore",
            "sample_minmax",
        }:
            raise ValueError(
                "input_normalization must be none, sample_zscore, or "
                "sample_minmax"
            )
        if length_mode not in {"resample", "resample_short_truncate_long"}:
            raise ValueError(f"Unknown length mode: {length_mode}")
        self.encoder = encoder
        self.head = head
        self.family = family
        self._feature_dim = int(feature_dim)
        self.input_length = int(input_length)
        self.input_channels = int(input_channels)
        self.input_normalization = input_normalization
        self.length_mode = length_mode
        self.pretrained_checkpoint = pretrained_checkpoint
        self.feature_branch = "backbone"

    @property
    def backbone(self) -> nn.Module:
        return self.encoder

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _prepare(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3:
            raise ValueError(
                f"{self.family} expects [B, L] or [B, C, L], got "
                f"{tuple(inputs.shape)}"
            )
        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"{self.family} was built for {self.input_channels} channels, "
                f"but received {inputs.shape[1]}"
            )
        device = next(self.encoder.parameters()).device
        inputs = inputs.to(device=device, dtype=torch.float32)
        current_length = int(inputs.shape[-1])
        if current_length != self.input_length:
            if (
                self.length_mode == "resample_short_truncate_long"
                and current_length > self.input_length
            ):
                inputs = inputs[..., : self.input_length].contiguous()
            else:
                inputs = F.interpolate(
                    inputs,
                    size=self.input_length,
                    mode="linear",
                    align_corners=False,
                )
        if self.input_normalization == "sample_zscore":
            mean = inputs.mean(dim=-1, keepdim=True)
            std = inputs.std(dim=-1, keepdim=True, unbiased=False)
            inputs = (inputs - mean) / std.clamp_min(1e-8)
        elif self.input_normalization == "sample_minmax":
            minimum = inputs.amin(dim=-1, keepdim=True)
            maximum = inputs.amax(dim=-1, keepdim=True)
            inputs = (inputs - minimum) / (maximum - minimum).clamp_min(1e-8)
        return inputs

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._prepare(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))

    def enable_encoder_finetuning(self) -> None:
        self.encoder.requires_grad_(True)
        self.encoder._finetuning_enabled = True

    def disable_encoder_finetuning(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder._finetuning_enabled = False
        self.encoder.eval()


def _canonical_family(value: str) -> str:
    family = value.strip().lower().replace("-", "_")
    if family not in {"unifreq", "unifault", "tf_profm"}:
        raise ValueError(
            "foundation_model must be unifreq, unifault, or tf_profm"
        )
    return family


def _resolve_pretrained_checkpoint(
    family: str,
    checkpoint: str,
    model_size: str,
) -> Optional[Path]:
    if not checkpoint:
        return None
    if checkpoint != "auto":
        path = Path(checkpoint).expanduser().resolve()
    else:
        resolved_size = model_size
        if family == "tf_profm":
            resolved_size = "tiny"
        elif resolved_size == "auto":
            resolved_size = "small" if family == "unifault" else "tiny"
        key = (family, resolved_size)
        if key not in AUTO_PRETRAINED_CHECKPOINTS:
            available = ", ".join(
                size
                for candidate_family, size in AUTO_PRETRAINED_CHECKPOINTS
                if candidate_family == family
            )
            raise ValueError(
                f"No automatic {family} checkpoint for size {resolved_size!r}; "
                f"available sizes: {available or 'none'}"
            )
        path = AUTO_PRETRAINED_CHECKPOINTS[key]
    if path.is_dir():
        candidates = (
            path / "pretrain" / "best_pretrain.pt",
            path / "best_pretrain.pt",
        )
        path = next((item for item in candidates if item.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    return path


def _load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(
            str(path), map_location="cpu", weights_only=True
        )
    except TypeError:
        # PyTorch < 2 does not provide the restricted weights-only unpickler.
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint {path} must contain a mapping")
    return checkpoint


def _checkpoint_state(
    checkpoint: Mapping[str, Any], path: Path
) -> Mapping[str, torch.Tensor]:
    state: Any = checkpoint.get(
        "model", checkpoint.get("state_dict", checkpoint)
    )
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint {path} has no model state mapping")
    tensor_state = {
        str(name): value
        for name, value in state.items()
        if torch.is_tensor(value)
    }
    if not tensor_state:
        raise ValueError(f"Checkpoint {path} contains no model tensors")
    return tensor_state


def _candidate_checkpoint_keys(name: str) -> Tuple[str, ...]:
    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "encoder.",
        "model.encoder.",
        "model.model.",
    )
    candidates = [name]
    for prefix in prefixes:
        if name.startswith(prefix):
            candidates.append(name[len(prefix) :])
    return tuple(dict.fromkeys(candidates))


def _load_complete_backbone_state(
    backbone: nn.Module,
    state: Mapping[str, torch.Tensor],
    path: Path,
) -> Tuple[int, int, int, int]:
    target = backbone.state_dict()
    matched: Dict[str, torch.Tensor] = {}
    for old_name, value in state.items():
        for candidate in _candidate_checkpoint_keys(old_name):
            if candidate in target and target[candidate].shape == value.shape:
                matched[candidate] = value
                break
    missing = sorted(set(target).difference(matched))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            f"Checkpoint {path} does not fully cover the {backbone.__class__.__name__} "
            f"backbone; missing {len(missing)} tensors: {preview}{suffix}"
        )
    backbone.load_state_dict(matched, strict=True)
    loaded_parameters = sum(value.numel() for value in matched.values())
    backbone_parameters = sum(value.numel() for value in target.values())
    return (
        len(matched),
        len(target),
        loaded_parameters,
        backbone_parameters,
    )


def _checkpoint_arguments(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    arguments = checkpoint.get("args", {})
    return dict(arguments) if isinstance(arguments, Mapping) else {}


def _checkpoint_family(arguments: Mapping[str, Any]) -> Optional[str]:
    value = str(arguments.get("model_arch", "")).strip()
    if not value:
        return None
    return _canonical_family(value)


def _resolve_input_normalization(
    family: str,
    value: str,
    checkpoint_value: Optional[str] = None,
) -> str:
    if value != "auto":
        return value
    if checkpoint_value in {
        "none",
        "sample_zscore",
        "sample_minmax",
    }:
        return str(checkpoint_value)
    return "sample_minmax" if family == "unifault" else "sample_zscore"


def _build_phmfd_model(
    family: str,
    num_classes: int,
    classifier: str,
    classifier_dropout: float,
    pretrained_checkpoint: str,
    foundation_model_size: str,
    model_input_length: int,
    model_input_normalization: str,
    input_channels: int,
) -> Tuple[PHMFDFaultModel, FoundationModelConfig]:
    path = _resolve_pretrained_checkpoint(
        family, pretrained_checkpoint, foundation_model_size
    )
    checkpoint: Optional[Mapping[str, Any]] = None
    state: Optional[Mapping[str, torch.Tensor]] = None
    checkpoint_args: Dict[str, Any] = {}
    if path is not None:
        checkpoint = _load_torch_checkpoint(path)
        state = _checkpoint_state(checkpoint, path)
        checkpoint_args = _checkpoint_arguments(checkpoint)
        recorded_family = _checkpoint_family(checkpoint_args)
        if recorded_family is not None and recorded_family != family:
            raise ValueError(
                f"Checkpoint {path} is for {recorded_family}, not {family}"
            )

    recorded_size = str(checkpoint_args.get("model_size", "")).lower()
    if (
        family == "unifault"
        and foundation_model_size != "auto"
        and recorded_size
        and foundation_model_size != recorded_size
    ):
        raise ValueError(
            f"Checkpoint model_size is {recorded_size!r}, but "
            f"--foundation_model_size is {foundation_model_size!r}"
        )
    default_size = "small" if family == "unifault" else "tiny"
    selected_size = recorded_size or (
        foundation_model_size
        if foundation_model_size != "auto"
        else default_size
    )

    if family == "unifault":
        if selected_size not in UNIFAULT_SIZE_CONFIGS:
            raise ValueError(f"Unknown UniFault size: {selected_size}")
        size_config = UNIFAULT_SIZE_CONFIGS[selected_size]
        architecture = {
            "model_size": selected_size,
            "input_length": int(
                checkpoint_args.get(
                    "seq_len",
                    checkpoint_args.get("unifault_resample_len", 1024),
                )
            ),
            "patch_size": int(
                checkpoint_args.get("unifault_patch_size", 64)
            ),
            "embed_dim": int(
                checkpoint_args.get("embed_dim", size_config["embed_dim"])
            ),
            "heads": int(checkpoint_args.get("heads", size_config["heads"])),
            "depth": int(checkpoint_args.get("depth", size_config["depth"])),
            "dropout": float(checkpoint_args.get("dropout", 0.1)),
        }
        default_length = architecture["input_length"]
        length_mode = "resample"
        feature_dim = architecture["embed_dim"]
        backbone: nn.Module = UniFaultBackbone(
            seq_len=architecture["input_length"],
            patch_size=architecture["patch_size"],
            embed_dim=architecture["embed_dim"],
            heads=architecture["heads"],
            depth=architecture["depth"],
            dropout=architecture["dropout"],
        )
    else:
        architecture = {
            "model_size": recorded_size or "tiny",
            "input_length": int(
                checkpoint_args.get(
                    "tf_profm_input_len", TF_PROFM_DEFAULT_CONFIG["input_len"]
                )
            ),
            "patch_len": int(
                checkpoint_args.get(
                    "tf_profm_patch_len", TF_PROFM_DEFAULT_CONFIG["patch_len"]
                )
            ),
            "num_patches": int(
                checkpoint_args.get(
                    "tf_profm_num_patches",
                    TF_PROFM_DEFAULT_CONFIG["num_patches"],
                )
            ),
            "d_model": int(
                checkpoint_args.get(
                    "tf_profm_d_model", TF_PROFM_DEFAULT_CONFIG["d_model"]
                )
            ),
            "d_ffn": int(
                checkpoint_args.get(
                    "tf_profm_d_ffn", TF_PROFM_DEFAULT_CONFIG["d_ffn"]
                )
            ),
            "layers": int(
                checkpoint_args.get(
                    "tf_profm_layers", TF_PROFM_DEFAULT_CONFIG["layers"]
                )
            ),
            "heads": int(
                checkpoint_args.get(
                    "tf_profm_heads", TF_PROFM_DEFAULT_CONFIG["heads"]
                )
            ),
            "feature_dim": int(
                checkpoint_args.get(
                    "tf_profm_feature_dim",
                    TF_PROFM_DEFAULT_CONFIG["feature_dim"],
                )
            ),
            "dropout": float(checkpoint_args.get("dropout", 0.1)),
        }
        default_length = architecture["input_length"]
        length_mode = "resample_short_truncate_long"
        feature_dim = architecture["feature_dim"]
        backbone = TFProFMBackbone(
            input_len=architecture["input_length"],
            patch_len=architecture["patch_len"],
            num_patches=architecture["num_patches"],
            d_model=architecture["d_model"],
            d_ffn=architecture["d_ffn"],
            layers=architecture["layers"],
            heads=architecture["heads"],
            feature_dim=architecture["feature_dim"],
            dropout=architecture["dropout"],
        )

    input_length = model_input_length or default_length
    if input_length != default_length:
        raise ValueError(
            f"{family} architecture expects input length {default_length}; "
            f"got --model_input_length {input_length}"
        )
    normalization = _resolve_input_normalization(
        family,
        model_input_normalization,
        checkpoint_value=checkpoint_args.get("time_data_normalize"),
    )
    report = (0, len(backbone.state_dict()), 0, sum(
        value.numel() for value in backbone.state_dict().values()
    ))
    if state is not None and path is not None:
        report = _load_complete_backbone_state(backbone, state, path)
        print(
            f"Loaded {report[0]}/{report[1]} {family} backbone tensors "
            f"({report[2]:,}/{report[3]:,} parameters) from {path}"
        )
    head = build_classifier(
        classifier, feature_dim, num_classes, classifier_dropout
    )
    model = PHMFDFaultModel(
        backbone,
        head,
        family=family,
        feature_dim=feature_dim,
        input_length=input_length,
        input_channels=input_channels,
        input_normalization=normalization,
        length_mode=length_mode,
        pretrained_checkpoint=str(path) if path is not None else None,
    )
    model.disable_encoder_finetuning()
    config = FoundationModelConfig(
        family=family,
        architecture=architecture,
        input_length=input_length,
        input_normalization=normalization,
        pretrained_checkpoint=str(path) if path is not None else None,
        loaded_tensors=report[0],
        backbone_tensors=report[1],
        loaded_parameters=report[2],
        backbone_parameters=report[3],
    )
    return model, config


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
    foundation_model: str = "unifreq",
    pretrained_checkpoint: str = "",
    foundation_model_size: str = "auto",
    model_input_length: int = 0,
    model_input_normalization: str = "auto",
    input_channels: int = 1,
) -> Tuple[nn.Module, FoundationModelConfig]:
    family = _canonical_family(foundation_model)
    if family != "unifreq":
        if feature_branch != "fused":
            raise ValueError(
                "feature_branch is only available for UniFreq; use fused "
                "with UniFault/TF-ProFM"
            )
        return _build_phmfd_model(
            family=family,
            num_classes=num_classes,
            classifier=classifier,
            classifier_dropout=classifier_dropout,
            pretrained_checkpoint=pretrained_checkpoint,
            foundation_model_size=foundation_model_size,
            model_input_length=model_input_length,
            model_input_normalization=model_input_normalization,
            input_channels=input_channels,
        )
    if pretrained_checkpoint:
        raise ValueError(
            "UniFreq loads weights through --model_id; "
            "--pretrained_checkpoint is for UniFault/TF-ProFM"
        )
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
    foundation_config = FoundationModelConfig(
        family="unifreq",
        architecture={
            "model_id": model_id,
            "revision": revision,
            "feature_branch": feature_branch,
        },
        input_length=None,
        input_normalization="unifreq_processor",
        pretrained_checkpoint=None,
        loaded_tensors=len(encoder.state_dict()),
        backbone_tensors=len(encoder.state_dict()),
        loaded_parameters=sum(
            value.numel() for value in encoder.state_dict().values()
        ),
        backbone_parameters=sum(
            value.numel() for value in encoder.state_dict().values()
        ),
        unifreq_config=(
            config.to_dict()
            if hasattr(config, "to_dict")
            else dict(vars(config))
        ),
    )
    return model, foundation_config
