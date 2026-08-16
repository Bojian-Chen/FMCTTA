"""End-to-end source training and continual test-time adaptation runner."""

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapters import (
    OnlineAdapter,
    adaptation_parameter_names,
    build_adapter,
    resolve_adaptation_parameter_scope,
)
from .config import (
    DATASETS,
    parse_source_samples_per_class,
    parse_target_samples_per_class,
    resolve_sequence,
)
from .data import FaultDataModule
from .metrics import ClassificationMeter
from .model import build_model
from .source import (
    compute_fisher,
    evaluate_supervised,
    load_checkpoint,
    save_checkpoint,
    train_source,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _json_ready(value: Any) -> Any:
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(content), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(
    sample_ids: torch.Tensor,
    labels: torch.Tensor,
    online_outputs: torch.Tensor,
    post_adapt_outputs: torch.Tensor,
    position: int,
    domain: int,
    domain_name: str,
    batch: int,
) -> List[Dict[str, Any]]:
    online_probabilities = online_outputs.detach().cpu().softmax(dim=1)
    post_probabilities = post_adapt_outputs.detach().cpu().softmax(dim=1)
    online_predictions = online_probabilities.argmax(dim=1)
    post_predictions = post_probabilities.argmax(dim=1)
    labels = labels.detach().cpu().long()
    return [
        {
            "position": position,
            "domain": domain,
            "domain_name": domain_name,
            "batch": batch,
            "sample_id": int(sample_ids[offset].item()),
            "target": int(labels[offset].item()),
            "online_prediction": int(online_predictions[offset].item()),
            "post_adapt_prediction": int(post_predictions[offset].item()),
            "online_correct": int(
                online_predictions[offset].eq(labels[offset]).item()
            ),
            "post_adapt_correct": int(
                post_predictions[offset].eq(labels[offset]).item()
            ),
            "online_confidence": float(
                online_probabilities[offset].amax().item()
            ),
            "post_adapt_confidence": float(
                post_probabilities[offset].amax().item()
            ),
        }
        for offset in range(labels.shape[0])
    ]


@torch.no_grad()
def evaluate_adapter(
    adapter: OnlineAdapter,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    meter = ClassificationMeter(num_classes)
    for inputs, labels, _ in loader:
        meter.update(adapter.predict(inputs.to(device)), labels)
    return meter.compute()


def run_experiment(args) -> Dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    spec = DATASETS[args.dataset]
    sequence = resolve_sequence(spec, args.domains)
    source_domain, target_domains = sequence[0], sequence[1:]
    source_samples_per_class = parse_source_samples_per_class(
        args.source_samples_per_class, spec.num_classes
    )
    target_samples_per_class = parse_target_samples_per_class(
        args.target_samples_per_class,
        spec.num_classes,
        len(target_domains),
    )
    stream_order = args.stream_order or spec.default_stream_order
    resolved_scope = resolve_adaptation_parameter_scope(
        args.method, args.adaptation_parameter_scope
    )
    args.stream_order = stream_order
    args.resolved_adaptation_parameter_scope = resolved_scope

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        Path(args.output_dir)
        / args.dataset
        / args.method.lower()
        / f"{run_name}-seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    data = FaultDataModule(
        spec=spec,
        data_root=Path(args.data_root),
        source_domain=source_domain,
        target_domains=target_domains,
        normalization=args.normalization,
        source_validation_ratio=args.source_validation_ratio,
        stream_order=stream_order,
        seed=args.seed,
        strict_data=args.strict_data,
        max_source_per_class=args.max_source_per_class,
        max_target_per_domain=args.max_target_per_domain,
        source_samples_per_class=source_samples_per_class,
        target_samples_per_class=target_samples_per_class,
    )
    source_train_loader = data.loader(
        data.source.train,
        args.source_batch_size,
        shuffle=True,
        workers=args.workers,
        device=device,
    )
    source_validation_loader = data.loader(
        data.source.validation,
        args.eval_batch_size,
        shuffle=False,
        workers=args.workers,
        device=device,
    )

    model, unifreq_config = build_model(
        args.model_id,
        spec.num_classes,
        classifier=args.classifier,
        classifier_dropout=args.classifier_dropout,
        revision=args.model_revision or None,
        token=args.hf_token or None,
        cache_dir=args.cache_dir or None,
        local_files_only=args.local_files_only,
        feature_branch=args.feature_branch,
    )
    model.to(device)
    checkpoint_expected = {
        "dataset": args.dataset,
        "source_domain": source_domain,
        "model_id": args.model_id,
        "model_revision": args.model_revision or None,
        "classifier": args.classifier,
        "feature_branch": args.feature_branch,
        "source_samples_per_class": data.sample_selection[
            "source_samples_per_class"
        ],
        "normalization": args.normalization,
        "input_channels": spec.num_channels,
        "seed": args.seed,
    }
    source_history: List[Dict[str, float]] = []
    checkpoint_metadata: Dict[str, object]
    if args.source_checkpoint:
        checkpoint_metadata = load_checkpoint(
            Path(args.source_checkpoint), model, checkpoint_expected
        )
        print(f"Loaded source checkpoint: {args.source_checkpoint}")
    else:
        source_result = train_source(
            model,
            source_train_loader,
            source_validation_loader,
            device,
            spec.num_classes,
            epochs=args.source_epochs,
            learning_rate=args.source_learning_rate,
            backbone_lr_scale=args.source_backbone_lr_scale,
            weight_decay=args.source_weight_decay,
            gradient_clip=args.source_gradient_clip,
        )
        source_history = source_result.history
        checkpoint_metadata = {
            **checkpoint_expected,
            "unifreq_config": unifreq_config.to_dict(),
            "best_validation_macro_f1": source_result.best_macro_f1,
            "best_epoch": source_result.best_epoch,
        }
        checkpoint_dir = run_dir / "source_checkpoint"
        save_checkpoint(checkpoint_dir, model, checkpoint_metadata)
        print(f"Saved source checkpoint: {checkpoint_dir}")

    source_before = evaluate_supervised(
        model, source_validation_loader, device, spec.num_classes
    )
    print(
        f"Source domain {source_domain} validation accuracy: "
        f"{source_before['accuracy']:.2f}%"
    )

    fisher: Optional[Dict[str, torch.Tensor]] = None
    if args.method.lower() == "eata" and args.eata_fisher_alpha > 0:
        fisher = compute_fisher(
            model,
            source_train_loader,
            device,
            adaptation_parameter_names(model, resolved_scope),
            max_batches=args.eata_fisher_batches,
        )

    adapter = build_adapter(
        method=args.method,
        model=model,
        device=device,
        learning_rate=args.adaptation_learning_rate,
        num_classes=spec.num_classes,
        fisher=fisher,
        steps=args.adaptation_steps,
        weight_decay=args.adaptation_weight_decay,
        parameter_scope=args.adaptation_parameter_scope,
        eata_fisher_alpha=args.eata_fisher_alpha,
        eata_entropy_margin=args.eata_entropy_margin,
        eata_diversity_margin=args.eata_diversity_margin,
        sar_rho=args.sar_rho,
        sar_entropy_margin=args.sar_entropy_margin,
        sar_reset_threshold=args.sar_reset_threshold,
        sar_ema_momentum=args.sar_ema_momentum,
        cotta_teacher_momentum=args.cotta_teacher_momentum,
        cotta_restore_probability=args.cotta_restore_probability,
        cotta_anchor_threshold=args.cotta_anchor_threshold,
        cotta_augmentations=args.cotta_augmentations,
        cotta_augmentation_noise_std=args.cotta_augmentation_noise_std,
        cotta_augmentation_scale_std=args.cotta_augmentation_scale_std,
        cotta_augmentation_mask_probability=(
            args.cotta_augmentation_mask_probability
        ),
        cotta_augmentation_shift_ratio=args.cotta_augmentation_shift_ratio,
        vida_learning_rate=args.vida_learning_rate,
        vida_low_rank=args.vida_low_rank,
        vida_high_rank=args.vida_high_rank,
        vida_teacher_momentum=args.vida_teacher_momentum,
        vida_adapter_momentum=args.vida_adapter_momentum,
        vida_uncertainty_threshold=args.vida_uncertainty_threshold,
        vida_uncertainty_scale=args.vida_uncertainty_scale,
        vida_augmentations=args.vida_augmentations,
        vida_augmentation_noise_std=args.vida_augmentation_noise_std,
        vida_augmentation_scale_std=args.vida_augmentation_scale_std,
        vida_augmentation_mask_probability=(
            args.vida_augmentation_mask_probability
        ),
        vida_augmentation_shift_ratio=args.vida_augmentation_shift_ratio,
    )

    overall = ClassificationMeter(spec.num_classes)
    post_adapt_overall = ClassificationMeter(spec.num_classes)
    domain_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    for position, domain in enumerate(target_domains, start=1):
        domain_name = spec.domain_name(domain)
        target_loader = data.loader(
            data.target_dataset(domain, stream_position=position - 1),
            args.stream_batch_size,
            shuffle=False,
            workers=args.workers,
            device=device,
        )
        domain_meter = ClassificationMeter(spec.num_classes)
        post_adapt_meter = ClassificationMeter(spec.num_classes)
        for batch, (inputs, labels, sample_ids) in enumerate(target_loader):
            # Protocol boundary: target labels are used by meters only.
            device_inputs = inputs.to(device)
            outputs = adapter(device_inputs)
            post_outputs = adapter.predict(device_inputs)
            domain_meter.update(outputs, labels)
            post_adapt_meter.update(post_outputs, labels)
            prediction_rows.extend(
                _prediction_rows(
                    sample_ids,
                    labels,
                    outputs,
                    post_outputs,
                    position,
                    domain,
                    domain_name,
                    batch,
                )
            )

        overall.merge(domain_meter)
        post_adapt_overall.merge(post_adapt_meter)
        metrics = domain_meter.compute()
        post_metrics = post_adapt_meter.compute()
        retention = evaluate_adapter(
            adapter, source_validation_loader, device, spec.num_classes
        )
        row: Dict[str, Any] = {
            "position": position,
            "domain": domain,
            "domain_name": domain_name,
            **metrics,
            **{
                f"post_adapt_{key}": value
                for key, value in post_metrics.items()
                if key != "samples"
            },
            "source_retention_accuracy": retention["accuracy"],
            "source_forgetting": (
                source_before["accuracy"] - retention["accuracy"]
            ),
        }
        domain_rows.append(row)
        print(
            f"[target {position:02d}/{len(target_domains):02d}] "
            f"domain={domain} samples={int(metrics['samples'])} "
            f"online_acc={metrics['accuracy']:.2f}% "
            f"post_adapt_acc={post_metrics['accuracy']:.2f}% "
            f"source_retention={retention['accuracy']:.2f}%"
        )

    online_metrics = overall.compute()
    post_metrics = post_adapt_overall.compute()
    domain_average = float(np.mean([row["accuracy"] for row in domain_rows]))
    post_domain_average = float(
        np.mean([row["post_adapt_accuracy"] for row in domain_rows])
    )
    final_retention = domain_rows[-1]["source_retention_accuracy"]
    summary: Dict[str, Any] = {
        "protocol": {
            "name": "online_continual_test_time_adaptation",
            "source_domain": source_domain,
            "source_domain_name": spec.domain_name(source_domain),
            "target_domains": list(target_domains),
            "target_domain_names": [
                spec.domain_name(domain) for domain in target_domains
            ],
            "single_pass": True,
            "reset_between_domains": False,
            "target_labels_used_for_adaptation": False,
            "primary_prediction_stage": "online_pre_adapt",
            "post_adapt_reprediction": True,
            "post_adapt_reprediction_updates_model": False,
            "stream_order": stream_order,
        },
        "dataset": {
            "name": spec.name,
            "description": spec.description,
            "num_classes": spec.num_classes,
            "num_domains": spec.num_domains,
            "num_channels": spec.num_channels,
            "warnings": data.data_warnings,
            "sample_selection": data.sample_selection,
        },
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision or None,
            "classifier": args.classifier,
            "feature_branch": args.feature_branch,
            "feature_dim": model.feature_dim,
        },
        "method": args.method.lower(),
        "source": {
            "validation": source_before,
            "checkpoint_metadata": checkpoint_metadata,
        },
        "online": {
            **online_metrics,
            "domain_average_accuracy": domain_average,
            "final_source_retention_accuracy": final_retention,
            "final_source_forgetting": (
                source_before["accuracy"] - final_retention
            ),
        },
        "post_adapt": {
            **post_metrics,
            "domain_average_accuracy": post_domain_average,
            "definition": (
                "Same-batch prediction after adaptation, without another update"
            ),
        },
        "adapter": adapter.statistics(),
        "domains": domain_rows,
        "run_dir": str(run_dir),
    }

    public_args = dict(vars(args))
    if public_args.get("hf_token"):
        public_args["hf_token"] = "<redacted>"
    _write_json(run_dir / "config.json", public_args)
    _write_json(run_dir / "source_history.json", {"epochs": source_history})
    _write_json(run_dir / "summary.json", summary)
    _write_csv(run_dir / "domains.csv", domain_rows)
    _write_csv(run_dir / "predictions.csv", prediction_rows)
    print(
        f"Online accuracy={online_metrics['accuracy']:.2f}% "
        f"post-adapt accuracy={post_metrics['accuracy']:.2f}% "
        f"domain-average={domain_average:.2f}% results={run_dir}"
    )
    return summary
