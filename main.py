#!/usr/bin/env python3
"""Run foundation-model continual test-time adaptation for fault diagnosis."""

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from fmctta.config import DATASETS, dataset_spec_from_manifest
from fmctta.runner import run_experiment


CONFIG_SECTIONS = {"protocol", "data", "model", "adaptation", "runtime"}


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Continual test-time adaptation with UniFreq, UniFault, or TF-ProFM"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    protocol = parser.add_argument_group("protocol")
    protocol.add_argument("--dataset", choices=sorted(DATASETS), default="PU")
    protocol.add_argument(
        "--domains",
        default="",
        help=(
            "Comma-separated domain sequence. Without --source_domains, the "
            "first domain is the labeled source and the remainder are unlabeled "
            "online targets. With --source_domains, this contains only the "
            "target stream. Uses the dataset default when omitted."
        ),
    )
    protocol.add_argument(
        "--source_domains",
        default="",
        help=(
            "Comma-separated labeled source domains combined for supervised "
            "training. When set, --domains is the target stream and the two "
            "sets must not overlap."
        ),
    )
    protocol.add_argument(
        "--method",
        choices=[
            "source",
            "knn",
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
        ],
        default="source",
    )
    protocol.add_argument(
        "--stream_order",
        choices=["shuffled", "class_block", "interleaved"],
        default=None,
    )
    protocol.add_argument("--stream_batch_size", type=int, default=16)
    protocol.add_argument("--seed", type=int, default=2025)
    protocol.add_argument(
        "--source_eval_domains",
        default="sources",
        help=(
            "Source-model diagnostics after training (or in source mode): "
            "sources, all, targets, or comma-separated domain IDs. The default "
            "avoids evaluating target domains again before the online stream; "
            "labeled source domains use held-out validation samples."
        ),
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data_root",
        default=str((project_dir / "data" / "ctta_custom").resolve()),
        help="Directory containing the PU/WT MAT files",
    )
    data.add_argument(
        "--dataset_manifest",
        default="",
        help=(
            "JSON manifest emitted by ctta_data_builder; when set, its dynamic "
            "DatasetSpec and colocated MAT file override --dataset/--data_root"
        ),
    )
    data.add_argument(
        "--normalization",
        choices=[
            "zscore",
            "minmax",
            "sample_zscore",
            "sample_minmax",
            "none",
        ],
        default="none",
        help=(
            "Input normalization: zscore/minmax fit source-train statistics; "
            "sample_zscore/sample_minmax normalize each sample and channel "
            "independently along time"
        ),
    )
    data.add_argument("--source_validation_ratio", type=float, default=0.2)
    data.add_argument("--strict_data", action="store_true")
    data.add_argument(
        "--allow_repeated_target_samples",
        action="store_true",
        help=(
            "Allow target samples to be selected again when a target domain "
            "occurs repeatedly in the stream"
        ),
    )
    data.add_argument("--max_source_per_class", type=int, default=None)
    data.add_argument("--max_target_per_domain", type=int, default=None)
    data.add_argument(
        "--source_samples_per_class",
        default="",
        help=(
            "Source samples per class: one count for all classes, or a "
            "comma-separated count for each class"
        ),
    )
    data.add_argument(
        "--target_samples_per_class",
        default="",
        help=(
            "Target samples per class: one count for every target-stream "
            "occurrence, or semicolon-separated occurrences; each occurrence "
            "is one count or comma-separated counts for every class"
        ),
    )

    model = parser.add_argument_group("foundation model and source classifier")
    model.add_argument(
        "--foundation_model",
        choices=["unifreq", "unifault", "tf_profm"],
        default="unifreq",
        help="Foundation-model family used as the fault feature encoder",
    )
    model.add_argument(
        "--model_id",
        default="bojian1/UniFreq-tiny",
        help="UniFreq Hugging Face model ID or local model directory",
    )
    model.add_argument("--model_revision", default="")
    model.add_argument(
        "--hf_token",
        default="",
        help="Optional Hugging Face token; public models need none",
    )
    model.add_argument("--cache_dir", default="")
    model.add_argument("--local_files_only", action="store_true")
    model.add_argument(
        "--pretrained_checkpoint",
        default="",
        help=(
            "UniFault/TF-ProFM PHMFD .pt checkpoint or run directory; "
            "'auto' selects the bundled PHMFD log preset"
        ),
    )
    model.add_argument(
        "--foundation_model_size",
        choices=["auto", "tiny", "small", "base"],
        default="auto",
        help=(
            "UniFault architecture preset when no checkpoint supplies it; "
            "also selects the UniFault 'auto' checkpoint"
        ),
    )
    model.add_argument(
        "--model_input_length",
        type=int,
        default=0,
        help=(
            "UniFault/TF-ProFM fixed input length; 0 infers it from the "
            "checkpoint (1024/2048 in the migrated presets)"
        ),
    )
    model.add_argument(
        "--model_input_normalization",
        choices=["auto", "none", "sample_zscore", "sample_minmax"],
        default="auto",
        help=(
            "Model-side per-sample preprocessing; auto reproduces PHMFD "
            "UniFault min-max and TF-ProFM z-score preprocessing"
        ),
    )
    model.add_argument(
        "--classifier", choices=["mlp", "linear", "cosine"], default="mlp"
    )
    model.add_argument(
        "--feature_branch",
        choices=["fused", "time", "frequency"],
        default="fused",
        help=(
            "UniFreq features for the downstream classifier: fused uses both "
            "branches, while time or frequency runs only that branch"
        ),
    )
    model.add_argument("--classifier_dropout", type=float, default=0.1)
    model.add_argument("--source_epochs", type=int, default=30)
    model.add_argument("--source_batch_size", type=int, default=64)
    model.add_argument("--eval_batch_size", type=int, default=128)
    model.add_argument("--source_learning_rate", type=float, default=1e-2)
    model.add_argument(
        "--source_backbone_lr_scale",
        type=float,
        default=0.01,
        help=(
            "0 freezes the foundation backbone; positive values fine-tune it "
            "at this LR scale"
        ),
    )
    model.add_argument("--source_weight_decay", type=float, default=1e-4)
    model.add_argument("--source_gradient_clip", type=float, default=1.0)
    model.add_argument(
        "--source_checkpoint",
        default="",
        help="Directory with source_model.safetensors and source_model.json",
    )

    adaptation = parser.add_argument_group("online adaptation")
    adaptation.add_argument("--adaptation_learning_rate", type=float, default=5e-4)
    adaptation.add_argument("--adaptation_weight_decay", type=float, default=0.0)
    adaptation.add_argument("--adaptation_steps", type=int, default=1)
    adaptation.add_argument(
        "--knn_neighbors",
        type=int,
        default=5,
        help="Number of labeled source-train neighbors used by the kNN baseline",
    )
    adaptation.add_argument(
        "--adaptation_parameter_scope",
        choices=["default", "layernorm", "classifier", "backbone", "all"],
        default="default",
        help=(
            "default is LayerNorm for TENT/NOTE/EATA/SAR and all base parameters "
            "for CoTTA/ViDA/Continual-MAE/TCA/GOLD/GOLD-TCA; TTT defaults "
            "to the backbone, "
            "while RoTTA defaults to LayerNorm; source and kNN do not adapt"
        ),
    )
    adaptation.add_argument("--note_memory_size", type=int, default=64)
    adaptation.add_argument("--note_update_interval", type=int, default=64)
    adaptation.add_argument("--note_temperature", type=float, default=1.0)
    adaptation.add_argument("--rotta_memory_size", type=int, default=64)
    adaptation.add_argument("--rotta_update_frequency", type=int, default=64)
    adaptation.add_argument("--rotta_nu", type=float, default=0.001)
    adaptation.add_argument("--rotta_bn_alpha", type=float, default=0.05)
    adaptation.add_argument("--rotta_lambda_t", type=float, default=1.0)
    adaptation.add_argument("--rotta_lambda_u", type=float, default=1.0)
    adaptation.add_argument("--rotta_augmentation_noise_std", type=float, default=0.005)
    adaptation.add_argument("--rotta_augmentation_scale_std", type=float, default=0.05)
    adaptation.add_argument(
        "--rotta_augmentation_mask_probability", type=float, default=0.0
    )
    adaptation.add_argument(
        "--rotta_augmentation_shift_ratio", type=float, default=0.0
    )
    adaptation.add_argument("--gold_top_r", type=int, default=64)
    adaptation.add_argument("--gold_alpha", type=float, default=0.1)
    adaptation.add_argument("--gold_conf_thresh", type=float, default=0.8)
    adaptation.add_argument("--gold_update_every", type=int, default=50)
    adaptation.add_argument("--gold_adapter_lr_scale", type=float, default=1.0)
    adaptation.add_argument("--gold_lambda_ce_trg", type=float, default=1.0)
    adaptation.add_argument("--gold_lambda_cont", type=float, default=1.0)
    adaptation.add_argument("--gold_temperature", type=float, default=0.1)
    adaptation.add_argument("--gold_projection_dim", type=int, default=128)
    adaptation.add_argument("--gold_teacher_momentum", type=float, default=0.999)
    adaptation.add_argument("--gold_augmentation_noise_std", type=float, default=0.005)
    adaptation.add_argument("--gold_augmentation_scale_std", type=float, default=0.05)
    adaptation.add_argument(
        "--gold_augmentation_mask_probability", type=float, default=0.0
    )
    adaptation.add_argument("--gold_augmentation_shift_ratio", type=float, default=0.0)
    adaptation.add_argument("--gold_warmup_steps", type=int, default=0)
    adaptation.add_argument("--gold_tca_lambda_topology", type=float, default=0.15)
    adaptation.add_argument("--gold_tca_centroid_momentum", type=float, default=0.9)
    adaptation.add_argument(
        "--gold_tca_uniformity_temperature", type=float, default=2.0
    )
    adaptation.add_argument("--tca_teacher_momentum", type=float, default=0.999)
    adaptation.add_argument("--tca_centroid_momentum", type=float, default=0.9)
    adaptation.add_argument(
        "--tca_topology_weight_momentum", type=float, default=0.8
    )
    adaptation.add_argument("--tca_alignment_weight", type=float, default=0.025)
    adaptation.add_argument("--tca_topology_weight", type=float, default=0.15)
    adaptation.add_argument(
        "--tca_uniformity_temperature", type=float, default=2.0
    )
    adaptation.add_argument("--tca_symmetric_alpha", type=float, default=0.5)
    adaptation.add_argument("--tca_augmentation_noise_std", type=float, default=0.005)
    adaptation.add_argument("--tca_augmentation_scale_std", type=float, default=0.05)
    adaptation.add_argument(
        "--tca_augmentation_mask_probability", type=float, default=0.0
    )
    adaptation.add_argument(
        "--tca_augmentation_shift_ratio", type=float, default=0.0
    )
    adaptation.add_argument(
        "--ttt_source_auxiliary_weight",
        type=float,
        default=1.0,
        help="Weight of the four-way shift-prediction loss during source training",
    )
    adaptation.add_argument("--eata_fisher_alpha", type=float, default=1.0)
    adaptation.add_argument("--eata_fisher_batches", type=int, default=16)
    adaptation.add_argument("--eata_entropy_margin", type=float, default=None)
    adaptation.add_argument("--eata_diversity_margin", type=float, default=0.95)
    adaptation.add_argument("--sar_rho", type=float, default=0.05)
    adaptation.add_argument("--sar_entropy_margin", type=float, default=None)
    adaptation.add_argument("--sar_reset_threshold", type=float, default=0.2)
    adaptation.add_argument("--sar_ema_momentum", type=float, default=0.9)
    adaptation.add_argument("--cotta_teacher_momentum", type=float, default=0.999)
    adaptation.add_argument("--cotta_restore_probability", type=float, default=0.01)
    adaptation.add_argument("--cotta_anchor_threshold", type=float, default=0.92)
    adaptation.add_argument("--cotta_augmentations", type=int, default=8)
    adaptation.add_argument("--cotta_augmentation_noise_std", type=float, default=0.005)
    adaptation.add_argument("--cotta_augmentation_scale_std", type=float, default=0.05)
    adaptation.add_argument(
        "--cotta_augmentation_mask_probability", type=float, default=0.0
    )
    adaptation.add_argument("--cotta_augmentation_shift_ratio", type=float, default=0.0)
    adaptation.add_argument("--vida_learning_rate", type=float, default=1e-5)
    adaptation.add_argument("--vida_low_rank", type=int, default=1)
    adaptation.add_argument("--vida_high_rank", type=int, default=128)
    adaptation.add_argument("--vida_teacher_momentum", type=float, default=0.999)
    adaptation.add_argument("--vida_adapter_momentum", type=float, default=0.999)
    adaptation.add_argument(
        "--vida_uncertainty_threshold", type=float, default=0.2
    )
    adaptation.add_argument("--vida_uncertainty_scale", type=float, default=0.1)
    adaptation.add_argument("--vida_augmentations", type=int, default=10)
    adaptation.add_argument(
        "--vida_augmentation_noise_std", type=float, default=0.005
    )
    adaptation.add_argument(
        "--vida_augmentation_scale_std", type=float, default=0.05
    )
    adaptation.add_argument(
        "--vida_augmentation_mask_probability", type=float, default=0.0
    )
    adaptation.add_argument(
        "--vida_augmentation_shift_ratio", type=float, default=0.0
    )
    adaptation.add_argument(
        "--continual_mae_mask_ratio", type=float, default=0.5
    )
    adaptation.add_argument("--continual_mae_augmentations", type=int, default=10)
    adaptation.add_argument(
        "--continual_mae_reconstruction_weight", type=float, default=0.5
    )
    adaptation.add_argument(
        "--continual_mae_restore_probability", type=float, default=0.1
    )
    adaptation.add_argument(
        "--continual_mae_augmentation_noise_std", type=float, default=0.005
    )
    adaptation.add_argument(
        "--continual_mae_augmentation_scale_std", type=float, default=0.05
    )
    adaptation.add_argument(
        "--continual_mae_augmentation_mask_probability", type=float, default=0.0
    )
    adaptation.add_argument(
        "--continual_mae_augmentation_shift_ratio", type=float, default=0.0
    )

    runtime = parser.add_argument_group("runtime and output")
    runtime.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    runtime.add_argument("--workers", type=int, default=0)
    runtime.add_argument("--output_dir", default=str(project_dir / "outputs"))
    runtime.add_argument("--run_name", default="")
    runtime.add_argument(
        "--config",
        default="",
        help="YAML experiment configuration; command-line options override it",
    )
    return parser


def _normalize_legacy_dashed_options(
    argv: Optional[Sequence[str]], parser: argparse.ArgumentParser
) -> list[str]:
    """Accept old ``--kebab-case`` options while documenting ``--snake_case``."""
    normalized = list(sys.argv[1:] if argv is None else argv)
    known_options = parser._option_string_actions
    for index, item in enumerate(normalized):
        if item == "--":
            break
        if not item.startswith("--"):
            continue
        option, separator, value = item.partition("=")
        canonical_option = "--" + option[2:].replace("-", "_")
        if canonical_option in known_options:
            normalized[index] = canonical_option + (
                separator + value if separator else ""
            )
    return normalized


def _as_cli_value(name: str, value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        return value
    if name in {"domains", "source_domains", "source_eval_domains"} and isinstance(value, Sequence):
        return ",".join(str(item) for item in value)
    if name == "source_samples_per_class":
        if isinstance(value, Sequence):
            return ",".join(str(item) for item in value)
        return str(value)
    if name == "target_samples_per_class":
        if not isinstance(value, Sequence):
            return str(value)
        if all(
            not isinstance(item, Sequence) or isinstance(item, (str, bytes))
            for item in value
        ):
            return ";".join(str(item) for item in value)
        return ";".join(
            ",".join(str(count) for count in occurrence)
            for occurrence in value
        )
    return value


def _load_yaml_defaults(
    path: Path,
    parser: argparse.ArgumentParser,
    inherited_paths: Optional[Sequence[Path]] = None,
) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML experiment configs require PyYAML; install requirements.txt"
        ) from exc

    path = path.resolve()
    inherited_paths = tuple(inherited_paths or ())
    if path in inherited_paths:
        chain = " -> ".join(str(item) for item in (*inherited_paths, path))
        raise ValueError(f"YAML base configuration cycle: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"YAML configuration does not exist: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if content is None:
        return {}
    if not isinstance(content, Mapping):
        raise ValueError("YAML configuration must contain a mapping")

    base = content.pop("base", None)
    if base is None:
        base_paths: Sequence[str] = ()
    elif isinstance(base, str) and base.strip():
        base_paths = (base,)
    elif (
        isinstance(base, Sequence)
        and not isinstance(base, (str, bytes))
        and base
        and all(isinstance(item, str) and item.strip() for item in base)
    ):
        base_paths = tuple(base)
    else:
        raise ValueError(
            "YAML base must be a non-empty relative file path or list of paths"
        )

    inherited_values: dict[str, Any] = {}
    for base_path in base_paths:
        inherited_values.update(
            _load_yaml_defaults(
                path.parent / base_path,
                parser,
                (*inherited_paths, path),
            )
        )

    values = dict(inherited_values)
    for section, section_values in content.items():
        if section in CONFIG_SECTIONS:
            if not isinstance(section_values, Mapping):
                raise ValueError(f"YAML section {section!r} must be a mapping")
            values.update(section_values)
        else:
            values[section] = section_values

    known = {action.dest for action in parser._actions}
    unknown = sorted(set(values).difference(known))
    if unknown:
        raise ValueError(
            "Unknown YAML configuration options: " + ", ".join(unknown)
        )
    return {
        name: _as_cli_value(name, value)
        for name, value in values.items()
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    argv = _normalize_legacy_dashed_options(argv, parser)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        parser.set_defaults(
            **_load_yaml_defaults(Path(config_args.config), parser)
        )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "source_epochs": args.source_epochs,
        "source_batch_size": args.source_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "stream_batch_size": args.stream_batch_size,
        "adaptation_steps": args.adaptation_steps,
        "gold_top_r": args.gold_top_r,
        "gold_update_every": args.gold_update_every,
        "gold_projection_dim": args.gold_projection_dim,
        "rotta_memory_size": args.rotta_memory_size,
        "rotta_update_frequency": args.rotta_update_frequency,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.source_backbone_lr_scale < 0:
        raise ValueError("source_backbone_lr_scale cannot be negative")
    if args.model_input_length < 0:
        raise ValueError("model_input_length cannot be negative")
    if args.foundation_model == "unifreq":
        if args.pretrained_checkpoint:
            raise ValueError(
                "UniFreq uses model_id, not pretrained_checkpoint"
            )
        if args.foundation_model_size != "auto":
            raise ValueError(
                "foundation_model_size is only used by UniFault/TF-ProFM"
            )
        if args.model_input_length:
            raise ValueError(
                "model_input_length is only used by UniFault/TF-ProFM"
            )
        if args.model_input_normalization != "auto":
            raise ValueError(
                "model_input_normalization is only used by UniFault/TF-ProFM"
            )
    elif args.feature_branch != "fused":
        raise ValueError(
            "feature_branch is UniFreq-specific; use fused with "
            "UniFault/TF-ProFM"
        )
    if args.method == "continual_mae" and args.foundation_model != "unifreq":
        raise ValueError(
            "continual_mae currently requires UniFreq global-FFT tokens"
        )
    if args.max_source_per_class is not None and args.source_samples_per_class:
        raise ValueError(
            "Use either max_source_per_class or source_samples_per_class, not both"
        )
    if args.max_target_per_domain is not None and args.target_samples_per_class:
        raise ValueError(
            "Use either max_target_per_domain or target_samples_per_class, not both"
        )
    if not 0 <= args.classifier_dropout < 1:
        raise ValueError("classifier_dropout must be in [0, 1)")
    if (
        args.method in {"source", "knn"}
        and args.adaptation_parameter_scope != "default"
    ):
        raise ValueError(f"{args.method} does not use an adaptation parameter scope")
    if args.knn_neighbors < 1:
        raise ValueError("knn_neighbors must be positive")
    if args.note_memory_size < 1 or args.note_update_interval < 1:
        raise ValueError("NOTE memory size and update interval must be positive")
    if args.note_temperature <= 0:
        raise ValueError("note_temperature must be positive")
    if args.ttt_source_auxiliary_weight < 0:
        raise ValueError("ttt_source_auxiliary_weight cannot be negative")
    if args.method == "ttt" and not args.source_checkpoint:
        if args.source_backbone_lr_scale <= 0:
            raise ValueError(
                "TTT source training requires source_backbone_lr_scale > 0"
            )
    if args.method == "ttt" and args.adaptation_parameter_scope == "classifier":
        raise ValueError("TTT cannot adapt the task classifier scope")
    if args.gold_warmup_steps < 0:
        raise ValueError("gold_warmup_steps cannot be negative")
    for name in ("gold_alpha", "gold_conf_thresh"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for name in ("gold_temperature", "gold_adapter_lr_scale"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if min(args.gold_lambda_ce_trg, args.gold_lambda_cont) < 0:
        raise ValueError("GOLD loss weights cannot be negative")
    if args.gold_lambda_ce_trg + args.gold_lambda_cont == 0:
        raise ValueError("GOLD requires at least one nonzero loss weight")
    if args.method == "gold_tca" and args.classifier != "cosine":
        raise ValueError("gold_tca requires --classifier cosine")
    if (
        args.method == "gold_tca"
        and args.adaptation_parameter_scope == "classifier"
    ):
        raise ValueError("gold_tca freezes the classifier parameter scope")
    if args.gold_tca_lambda_topology < 0:
        raise ValueError("gold_tca_lambda_topology cannot be negative")
    if not 0 <= args.gold_tca_centroid_momentum < 1:
        raise ValueError("gold_tca_centroid_momentum must be in [0, 1)")
    if args.gold_tca_uniformity_temperature <= 0:
        raise ValueError("gold_tca_uniformity_temperature must be positive")
    if not 0 <= args.rotta_nu <= 1:
        raise ValueError("rotta_nu must be in [0, 1]")
    if not 0 <= args.rotta_bn_alpha <= 1:
        raise ValueError("rotta_bn_alpha must be in [0, 1]")
    if args.rotta_lambda_t < 0 or args.rotta_lambda_u < 0:
        raise ValueError("RoTTA memory weights cannot be negative")
    for name in (
        "rotta_augmentation_noise_std",
        "rotta_augmentation_scale_std",
        "gold_augmentation_noise_std",
        "gold_augmentation_scale_std",
        "tca_augmentation_noise_std",
        "tca_augmentation_scale_std",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    for name in (
        "rotta_augmentation_mask_probability",
        "rotta_augmentation_shift_ratio",
        "gold_augmentation_mask_probability",
        "gold_augmentation_shift_ratio",
        "tca_augmentation_mask_probability",
        "tca_augmentation_shift_ratio",
    ):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for name in (
        "gold_teacher_momentum",
        "tca_teacher_momentum",
        "tca_centroid_momentum",
        "tca_topology_weight_momentum",
    ):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    if args.tca_alignment_weight < 0 or args.tca_topology_weight < 0:
        raise ValueError("TCA loss weights cannot be negative")
    if args.tca_uniformity_temperature <= 0:
        raise ValueError("tca_uniformity_temperature must be positive")
    if not 0 <= args.tca_symmetric_alpha <= 1:
        raise ValueError("tca_symmetric_alpha must be in [0, 1]")
    if args.eata_fisher_batches < 1:
        raise ValueError("eata_fisher_batches must be positive")
    if args.sar_rho < 0 or args.sar_reset_threshold < 0:
        raise ValueError("SAR rho and reset threshold cannot be negative")
    if not 0 <= args.sar_ema_momentum < 1:
        raise ValueError("sar_ema_momentum must be in [0, 1)")
    if not 0 <= args.cotta_teacher_momentum < 1:
        raise ValueError("cotta_teacher_momentum must be in [0, 1)")
    if not 0 <= args.cotta_restore_probability <= 1:
        raise ValueError("cotta_restore_probability must be in [0, 1]")
    if args.cotta_augmentations < 0:
        raise ValueError("cotta_augmentations cannot be negative")
    if args.cotta_augmentation_noise_std < 0:
        raise ValueError("cotta_augmentation_noise_std cannot be negative")
    if args.cotta_augmentation_scale_std < 0:
        raise ValueError("cotta_augmentation_scale_std cannot be negative")
    if not 0 <= args.cotta_augmentation_mask_probability <= 1:
        raise ValueError("cotta_augmentation_mask_probability must be in [0, 1]")
    if not 0 <= args.cotta_augmentation_shift_ratio <= 1:
        raise ValueError("cotta_augmentation_shift_ratio must be in [0, 1]")
    if args.vida_learning_rate < 0:
        raise ValueError("vida_learning_rate cannot be negative")
    if args.vida_low_rank < 1 or args.vida_high_rank < 1:
        raise ValueError("ViDA ranks must be positive")
    if not 0 <= args.vida_teacher_momentum < 1:
        raise ValueError("vida_teacher_momentum must be in [0, 1)")
    if not 0 <= args.vida_adapter_momentum < 1:
        raise ValueError("vida_adapter_momentum must be in [0, 1)")
    if not 0 <= args.vida_uncertainty_threshold <= 1:
        raise ValueError("vida_uncertainty_threshold must be in [0, 1]")
    if args.vida_uncertainty_scale < 0:
        raise ValueError("vida_uncertainty_scale cannot be negative")
    if args.vida_augmentations < 1:
        raise ValueError("vida_augmentations must be positive")
    if args.vida_augmentation_noise_std < 0:
        raise ValueError("vida_augmentation_noise_std cannot be negative")
    if args.vida_augmentation_scale_std < 0:
        raise ValueError("vida_augmentation_scale_std cannot be negative")
    if not 0 <= args.vida_augmentation_mask_probability <= 1:
        raise ValueError("vida_augmentation_mask_probability must be in [0, 1]")
    if not 0 <= args.vida_augmentation_shift_ratio <= 1:
        raise ValueError("vida_augmentation_shift_ratio must be in [0, 1]")
    if not 0 < args.continual_mae_mask_ratio <= 1:
        raise ValueError("continual_mae_mask_ratio must be in (0, 1]")
    if args.continual_mae_augmentations < 2:
        raise ValueError("continual_mae_augmentations must be at least 2")
    if args.continual_mae_reconstruction_weight < 0:
        raise ValueError("continual_mae_reconstruction_weight cannot be negative")
    if not 0 <= args.continual_mae_restore_probability <= 1:
        raise ValueError("continual_mae_restore_probability must be in [0, 1]")
    if args.continual_mae_augmentation_noise_std < 0:
        raise ValueError("continual_mae_augmentation_noise_std cannot be negative")
    if args.continual_mae_augmentation_scale_std < 0:
        raise ValueError("continual_mae_augmentation_scale_std cannot be negative")
    if not 0 <= args.continual_mae_augmentation_mask_probability <= 1:
        raise ValueError(
            "continual_mae_augmentation_mask_probability must be in [0, 1]"
        )
    if not 0 <= args.continual_mae_augmentation_shift_ratio <= 1:
        raise ValueError("continual_mae_augmentation_shift_ratio must be in [0, 1]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.dataset_manifest:
        manifest_path = Path(args.dataset_manifest).resolve()
        spec = dataset_spec_from_manifest(manifest_path)
        DATASETS[spec.name] = spec
        args.dataset = spec.name
        args.data_root = str(manifest_path.parent)
    run_experiment(args)


if __name__ == "__main__":
    main()
