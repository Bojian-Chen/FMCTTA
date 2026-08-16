#!/usr/bin/env python3
"""Run UniFreq-based continual test-time adaptation for fault diagnosis."""

import argparse
from pathlib import Path

from fmctta.config import DATASETS, dataset_spec_from_manifest
from fmctta.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Foundation-model continual test-time adaptation with UniFreq",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    protocol = parser.add_argument_group("protocol")
    protocol.add_argument("--dataset", choices=sorted(DATASETS), default="PU_Real")
    protocol.add_argument(
        "--domains",
        default="",
        help=(
            "Comma-separated stream: first domain is labeled source; all "
            "remaining domains are unlabeled online targets"
        ),
    )
    protocol.add_argument(
        "--method",
        choices=["source", "tent", "eata", "sar", "cotta", "vida"],
        default="source",
    )
    protocol.add_argument(
        "--stream-order",
        choices=["shuffled", "class_block", "interleaved"],
        default=None,
    )
    protocol.add_argument("--stream-batch-size", type=int, default=16)
    protocol.add_argument("--seed", type=int, default=2025)

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        default=str((project_dir.parent / "OCTTA" / "data").resolve()),
        help="Directory containing the processed MAT files",
    )
    data.add_argument(
        "--dataset-manifest",
        default="",
        help=(
            "JSON manifest emitted by ctta_data_builder; when set, its dynamic "
            "DatasetSpec and colocated MAT file override --dataset/--data-root"
        ),
    )
    data.add_argument(
        "--normalization",
        choices=["zscore", "minmax", "none"],
        default="none",
        help="Optional statistics fitted only on source-train",
    )
    data.add_argument("--source-validation-ratio", type=float, default=0.2)
    data.add_argument("--strict-data", action="store_true")
    data.add_argument("--max-source-per-class", type=int, default=None)
    data.add_argument("--max-target-per-domain", type=int, default=None)

    model = parser.add_argument_group("UniFreq and source classifier")
    model.add_argument(
        "--model-id",
        default="bojian1/UniFreq-tiny",
        help="Hugging Face model ID or local model directory",
    )
    model.add_argument("--model-revision", default="")
    model.add_argument(
        "--hf-token",
        default="",
        help="Optional Hugging Face token; public models need none",
    )
    model.add_argument("--cache-dir", default="")
    model.add_argument("--local-files-only", action="store_true")
    model.add_argument(
        "--classifier", choices=["mlp", "linear", "cosine"], default="mlp"
    )
    model.add_argument("--classifier-dropout", type=float, default=0.1)
    model.add_argument("--source-epochs", type=int, default=30)
    model.add_argument("--source-batch-size", type=int, default=64)
    model.add_argument("--eval-batch-size", type=int, default=128)
    model.add_argument("--source-learning-rate", type=float, default=1e-2)
    model.add_argument(
        "--source-backbone-lr-scale",
        type=float,
        default=0.01,
        help="0 freezes UniFreq; positive values fine-tune it at this LR scale",
    )
    model.add_argument("--source-weight-decay", type=float, default=1e-4)
    model.add_argument("--source-gradient-clip", type=float, default=1.0)
    model.add_argument(
        "--source-checkpoint",
        default="",
        help="Directory with source_model.safetensors and source_model.json",
    )

    adaptation = parser.add_argument_group("online adaptation")
    adaptation.add_argument("--adaptation-learning-rate", type=float, default=1e-4)
    adaptation.add_argument("--adaptation-weight-decay", type=float, default=0.0)
    adaptation.add_argument("--adaptation-steps", type=int, default=1)
    adaptation.add_argument(
        "--adaptation-parameter-scope",
        choices=["default", "layernorm", "classifier", "backbone", "all"],
        default="default",
        help=(
            "default is LayerNorm for TENT/EATA/SAR and all base parameters "
            "for CoTTA/ViDA; ViDA branches are always updated"
        ),
    )
    adaptation.add_argument("--eata-fisher-alpha", type=float, default=1.0)
    adaptation.add_argument("--eata-fisher-batches", type=int, default=16)
    adaptation.add_argument("--eata-entropy-margin", type=float, default=None)
    adaptation.add_argument("--eata-diversity-margin", type=float, default=0.95)
    adaptation.add_argument("--sar-rho", type=float, default=0.05)
    adaptation.add_argument("--sar-entropy-margin", type=float, default=None)
    adaptation.add_argument("--sar-reset-threshold", type=float, default=0.2)
    adaptation.add_argument("--sar-ema-momentum", type=float, default=0.9)
    adaptation.add_argument("--cotta-teacher-momentum", type=float, default=0.999)
    adaptation.add_argument("--cotta-restore-probability", type=float, default=0.01)
    adaptation.add_argument("--cotta-anchor-threshold", type=float, default=0.92)
    adaptation.add_argument("--cotta-augmentations", type=int, default=8)
    adaptation.add_argument("--cotta-augmentation-noise-std", type=float, default=0.005)
    adaptation.add_argument("--cotta-augmentation-scale-std", type=float, default=0.05)
    adaptation.add_argument(
        "--cotta-augmentation-mask-probability", type=float, default=0.0
    )
    adaptation.add_argument("--cotta-augmentation-shift-ratio", type=float, default=0.0)
    adaptation.add_argument("--vida-learning-rate", type=float, default=1e-5)
    adaptation.add_argument("--vida-low-rank", type=int, default=1)
    adaptation.add_argument("--vida-high-rank", type=int, default=128)
    adaptation.add_argument("--vida-teacher-momentum", type=float, default=0.999)
    adaptation.add_argument("--vida-adapter-momentum", type=float, default=0.999)
    adaptation.add_argument(
        "--vida-uncertainty-threshold", type=float, default=0.2
    )
    adaptation.add_argument("--vida-uncertainty-scale", type=float, default=0.1)
    adaptation.add_argument("--vida-augmentations", type=int, default=10)
    adaptation.add_argument(
        "--vida-augmentation-noise-std", type=float, default=0.005
    )
    adaptation.add_argument(
        "--vida-augmentation-scale-std", type=float, default=0.05
    )
    adaptation.add_argument(
        "--vida-augmentation-mask-probability", type=float, default=0.0
    )
    adaptation.add_argument(
        "--vida-augmentation-shift-ratio", type=float, default=0.0
    )

    runtime = parser.add_argument_group("runtime and output")
    runtime.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    runtime.add_argument("--workers", type=int, default=0)
    runtime.add_argument("--output-dir", default=str(project_dir / "outputs"))
    runtime.add_argument("--run-name", default="")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "source_epochs": args.source_epochs,
        "source_batch_size": args.source_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "stream_batch_size": args.stream_batch_size,
        "adaptation_steps": args.adaptation_steps,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.source_backbone_lr_scale < 0:
        raise ValueError("source_backbone_lr_scale cannot be negative")
    if not 0 <= args.classifier_dropout < 1:
        raise ValueError("classifier_dropout must be in [0, 1)")
    if args.method == "source" and args.adaptation_parameter_scope != "default":
        raise ValueError("source does not use an adaptation parameter scope")
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
