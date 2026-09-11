"""End-to-end source training and continual test-time adaptation runner."""

import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    parse_domains,
    parse_source_samples_per_class,
    parse_target_samples_per_class,
    resolve_sequence,
    resolve_target_domains,
)
from .data import FaultDataModule
from .metrics import ClassificationMeter, relative_improvement
from .model import build_model
from .knn import build_reference_bank
from .source import (
    compute_fisher,
    evaluate_supervised,
    load_checkpoint,
    save_checkpoint,
    train_source,
)
from .ttt import build_ttt_head


TARGET_FORGETTING_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


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
    source_outputs: torch.Tensor,
) -> List[Dict[str, Any]]:
    source_probabilities = source_outputs.detach().cpu().softmax(dim=1)
    source_predictions = source_probabilities.argmax(dim=1)
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
            "source_prediction": int(source_predictions[offset].item()),
            "source_correct": int(source_predictions[offset].eq(labels[offset]).item()),
            "source_confidence": float(source_probabilities[offset].amax().item()),
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


def resolve_source_eval_domains(value: str, spec, source_domains, target_domains):
    if value == "sources":
        return tuple(dict.fromkeys(source_domains))
    if value == "all":
        return tuple(range(spec.num_domains))
    if value == "targets":
        return tuple(dict.fromkeys(target_domains))
    domains = parse_domains(value)
    if len(set(domains)) != len(domains):
        raise ValueError("source_eval_domains cannot contain duplicates")
    if any(domain < 0 or domain >= spec.num_domains for domain in domains):
        raise ValueError("source_eval_domains contains an invalid domain ID")
    return domains


def _sample_digest(sample_ids: torch.Tensor) -> str:
    return hashlib.sha256(
        sample_ids.detach().cpu().numpy().astype("<i8").tobytes()
    ).hexdigest()


def _sample_set_digest(sample_ids: torch.Tensor) -> str:
    """Hash a sample set independently of its online presentation order."""

    ordered = torch.sort(sample_ids.detach().cpu().long()).values
    return hashlib.sha256(ordered.numpy().astype("<i8").tobytes()).hexdigest()


def _target_forgetting_summary(
    rows: List[Dict[str, Any]],
    allow_target_sample_reuse: bool,
) -> Dict[str, Any]:
    repeat_rows = [row for row in rows if row["is_repeat"]]
    domains: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row["domain"])
        domain = domains.setdefault(
            key,
            {
                "domain": row["domain"],
                "domain_name": row["domain_name"],
                "occurrences": 0,
                "repeat_measurements": 0,
            },
        )
        domain["occurrences"] += 1
        if row["is_repeat"]:
            domain["repeat_measurements"] += 1
            for metric in TARGET_FORGETTING_METRICS:
                domain[f"final_{metric}_drop_from_previous_pp"] = row[
                    f"{metric}_drop_from_previous_pp"
                ]
                domain[f"final_{metric}_drop_from_first_pp"] = row[
                    f"{metric}_drop_from_first_pp"
                ]
                domain[f"final_{metric}_forgetting_from_best_pp"] = row[
                    f"{metric}_forgetting_from_best_pp"
                ]
                domain[f"final_{metric}_change_from_first_occurrence_pp"] = row[
                    f"{metric}_change_from_first_occurrence_pp"
                ]

    aggregate: Dict[str, Any] = {"repeat_measurements": len(repeat_rows)}
    for metric in TARGET_FORGETTING_METRICS:
        for suffix in (
            "drop_from_previous_pp",
            "drop_from_first_pp",
            "forgetting_from_best_pp",
            "change_from_first_occurrence_pp",
        ):
            values = [float(row[f"{metric}_{suffix}"]) for row in repeat_rows]
            aggregate[f"mean_{metric}_{suffix}"] = (
                float(np.mean(values)) if values else None
            )
            aggregate[f"max_{metric}_{suffix}"] = max(values) if values else None

    occurrence_means: List[Dict[str, Any]] = []
    occurrence_ids = sorted({int(row["occurrence"]) for row in rows})
    for occurrence in occurrence_ids:
        occurrence_rows = [
            row for row in rows if int(row["occurrence"]) == occurrence
        ]
        occurrence_mean: Dict[str, Any] = {
            "occurrence": occurrence,
            "domain_count": len(occurrence_rows),
            "domains": [int(row["domain"]) for row in occurrence_rows],
        }
        for metric in TARGET_FORGETTING_METRICS:
            for stage in ("pre_visit", "post_visit"):
                occurrence_mean[f"mean_{stage}_{metric}"] = float(
                    np.mean(
                        [
                            float(row[f"{stage}_{metric}"])
                            for row in occurrence_rows
                        ]
                    )
                )
            for suffix in (
                "drop_from_previous_pp",
                "drop_from_first_pp",
                "forgetting_from_best_pp",
                "change_from_first_occurrence_pp",
            ):
                values = [
                    float(row[f"{metric}_{suffix}"])
                    for row in occurrence_rows
                    if row[f"{metric}_{suffix}"] is not None
                ]
                occurrence_mean[f"mean_{metric}_{suffix}"] = (
                    float(np.mean(values)) if values else None
                )
        occurrence_means.append(occurrence_mean)

    return {
        "enabled": True,
        "definition": (
            "For each repeated target domain, forgetting is measured before "
            "re-adaptation. drop_from_previous compares the current pre-visit "
            "probe with the previous occurrence's post-visit probe; positive "
            "values mean performance declined. change_from_first_occurrence "
            "compares pre-visit probes at the same stage (current minus first), "
            "so positive values mean improvement. Probes never update the adapter."
        ),
        "score_unit": "percentage_points",
        "probe_updates_adapter": False,
        "same_samples_reused_across_occurrences": allow_target_sample_reuse,
        "sample_policy": (
            "same_sample_ids"
            if allow_target_sample_reuse
            else "distinct_samples_without_replacement"
        ),
        "comparison_warning": (
            None
            if allow_target_sample_reuse
            else (
                "Probe samples differ across occurrences, so measured drops "
                "may include sample-composition effects as well as forgetting."
            )
        ),
        "metrics": list(TARGET_FORGETTING_METRICS),
        "aggregate": aggregate,
        "occurrence_means": occurrence_means,
        "domains": domains,
        "measurements": rows,
    }


def evaluate_source_domains(model, data, domains, args, device):
    """Run diagnostics without advancing the adaptation RNG or target stream."""
    rows = []
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        for domain in domains:
            dataset = data.source_evaluation_dataset(domain)
            loader = data.loader(
                dataset, args.eval_batch_size, shuffle=False,
                workers=args.workers, device=device,
            )
            metrics = evaluate_supervised(model, loader, device, data.spec.num_classes)
            row = {
                "domain": domain,
                "domain_name": data.spec.domain_name(domain),
                "evaluation_split": (
                    "source_validation" if domain in data.source_domains else "full_domain"
                ),
                "sample_ids_sha256": _sample_digest(dataset.sample_ids),
                **metrics,
            }
            rows.append(row)
            print(
                f"[source evaluation] domain={domain} "
                f"samples={int(metrics['samples'])} accuracy={metrics['accuracy']:.2f}%"
            )
    return rows


def run_experiment(args) -> Dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    spec = DATASETS[args.dataset]
    if args.source_domains:
        source_domains = parse_domains(args.source_domains)
        if not args.domains:
            raise ValueError(
                "--domains must explicitly specify at least one target domain "
                "when --source_domains is used"
            )
        target_domains = resolve_target_domains(spec, args.domains)
    else:
        sequence = resolve_sequence(spec, args.domains)
        source_domains, target_domains = (sequence[0],), sequence[1:]
    source_domain = source_domains[0]
    target_forgetting_enabled = args.protocol_mode == "target_forgetting"
    source_evaluation_only = args.method.lower() == "source"
    configured_target_domains = target_domains
    if (
        not source_evaluation_only
        and target_forgetting_enabled
        and args.target_domain_repeats < 1
    ):
        raise ValueError("target_domain_repeats must be positive")
    if target_forgetting_enabled and not source_evaluation_only:
        target_domains = configured_target_domains * args.target_domain_repeats
        repeated_domains = {
            domain for domain in target_domains if target_domains.count(domain) > 1
        }
        if not repeated_domains:
            raise ValueError(
                "target_forgetting protocol mode requires at least one repeated "
                "target domain"
            )
    allow_repeated_target_samples = (
        args.allow_target_sample_reuse
        if target_forgetting_enabled and not source_evaluation_only
        else args.allow_repeated_target_samples
    )
    source_eval_domains = resolve_source_eval_domains(
        args.source_eval_domains, spec, source_domains, target_domains
    )
    source_samples_per_class = parse_source_samples_per_class(
        args.source_samples_per_class, spec.num_classes
    )
    target_samples_per_class = (
        None
        if source_evaluation_only
        else parse_target_samples_per_class(
            args.target_samples_per_class,
            spec.num_classes,
            len(target_domains),
        )
    )
    stream_order = args.stream_order or spec.default_stream_order
    resolved_scope = resolve_adaptation_parameter_scope(
        args.method, args.adaptation_parameter_scope
    )
    args.stream_order = stream_order
    args.resolved_adaptation_parameter_scope = resolved_scope

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_directory_name = (
        f"{run_name}-seed{args.seed}"
        if source_evaluation_only
        else f"{run_name}-{args.protocol_mode}-seed{args.seed}"
    )
    run_dir = (
        Path(args.output_dir)
        / args.dataset
        / args.method.lower()
        / run_directory_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    data = FaultDataModule(
        spec=spec,
        data_root=Path(args.data_root),
        source_domains=source_domains,
        target_domains=() if source_evaluation_only else target_domains,
        normalization=args.normalization,
        source_validation_ratio=args.source_validation_ratio,
        stream_order=stream_order,
        seed=args.seed,
        strict_data=args.strict_data,
        allow_repeated_target_samples=allow_repeated_target_samples,
        max_source_per_class=args.max_source_per_class,
        max_target_per_domain=args.max_target_per_domain,
        source_samples_per_class=source_samples_per_class,
        target_samples_per_class=target_samples_per_class,
        reuse_repeated_target_samples=(
            target_forgetting_enabled
            and not source_evaluation_only
            and args.allow_target_sample_reuse
        ),
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
    model, foundation_config = build_model(
        args.model_id,
        spec.num_classes,
        classifier=args.classifier,
        classifier_dropout=args.classifier_dropout,
        revision=args.model_revision or None,
        token=args.hf_token or None,
        cache_dir=args.cache_dir or None,
        local_files_only=args.local_files_only,
        feature_branch=args.feature_branch,
        foundation_model=args.foundation_model,
        pretrained_checkpoint=args.pretrained_checkpoint,
        foundation_model_size=args.foundation_model_size,
        model_input_length=args.model_input_length,
        model_input_normalization=args.model_input_normalization,
        input_channels=spec.num_channels,
    )
    model.to(device)
    ttt_auxiliary_head = (
        build_ttt_head(model.feature_dim).to(device)
        if args.method.lower() == "ttt"
        else None
    )
    ttt_extra_modules = (
        {"ttt_auxiliary_head": ttt_auxiliary_head}
        if ttt_auxiliary_head is not None
        else None
    )
    effective_model_id = args.model_id if args.foundation_model == "unifreq" else ""
    effective_revision = (
        args.model_revision or None
        if args.foundation_model == "unifreq"
        else None
    )
    effective_feature_branch = model.feature_branch
    checkpoint_expected = {
        "dataset": args.dataset,
        "source_domain": source_domain,
        "source_domains": list(source_domains),
        "foundation_model": args.foundation_model,
        "foundation_architecture": foundation_config.architecture,
        "model_input_length": foundation_config.input_length,
        "model_input_normalization": foundation_config.input_normalization,
        "model_id": effective_model_id,
        "model_revision": effective_revision,
        "classifier": args.classifier,
        "feature_branch": effective_feature_branch,
        "source_samples_per_class": data.sample_selection[
            "source_samples_per_class"
        ],
        "normalization": args.normalization,
        "source_validation_ratio": args.source_validation_ratio,
        "input_channels": spec.num_channels,
        "seed": args.seed,
    }
    source_history: List[Dict[str, float]] = []
    checkpoint_metadata: Dict[str, object]
    if args.source_checkpoint:
        checkpoint_metadata = load_checkpoint(
            Path(args.source_checkpoint),
            model,
            checkpoint_expected,
            extra_modules=ttt_extra_modules,
        )
        if len(source_domains) > 1 and checkpoint_metadata.get(
            "source_domains"
        ) != list(source_domains):
            raise ValueError(
                "Source checkpoint is incompatible: a multi-source run requires "
                f"source_domains={list(source_domains)} in its metadata"
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
            ttt_auxiliary_head=ttt_auxiliary_head,
            ttt_auxiliary_weight=args.ttt_source_auxiliary_weight,
        )
        source_history = source_result.history
        checkpoint_metadata = {
            **checkpoint_expected,
            "foundation_config": foundation_config.to_dict(),
            "best_validation_macro_f1": source_result.best_macro_f1,
            "best_epoch": source_result.best_epoch,
        }
        if ttt_auxiliary_head is not None:
            checkpoint_metadata.update(
                {
                    "ttt_auxiliary_task": "four_way_circular_shift_prediction",
                    "ttt_source_auxiliary_weight": args.ttt_source_auxiliary_weight,
                }
            )
        checkpoint_dir = run_dir / "source_checkpoint"
        save_checkpoint(
            checkpoint_dir,
            model,
            checkpoint_metadata,
            extra_modules=ttt_extra_modules,
        )
        print(f"Saved source checkpoint: {checkpoint_dir}")

    source_model_validation = evaluate_supervised(
        model, source_validation_loader, device, spec.num_classes
    )
    print(
        f"Source domains {list(source_domains)} validation accuracy: "
        f"{source_model_validation['accuracy']:.2f}%"
    )

    source_evaluation = None
    if args.method.lower() == "source" or not args.source_checkpoint:
        if source_evaluation_only:
            model.eval()
            model.requires_grad_(False)
        source_evaluation_rows = evaluate_source_domains(
            model, data, source_eval_domains, args, device
        )
        source_evaluation = {
            "dataset": spec.name,
            "dataset_file": str(spec.path(Path(args.data_root)).resolve()),
            "source_checkpoint": args.source_checkpoint or str(run_dir / "source_checkpoint"),
            "checkpoint_metadata": checkpoint_metadata,
            "normalization": args.normalization,
            "source_validation_ratio": args.source_validation_ratio,
            "domains": {str(row["domain"]): row for row in source_evaluation_rows},
            "definition": "Post-training frozen model; source validation and full non-source domains",
        }
        _write_json(run_dir / "source_evaluation.json", source_evaluation)
        _write_csv(run_dir / "source_evaluation.csv", source_evaluation_rows)
        if not args.source_checkpoint:
            _write_json(run_dir / "source_checkpoint" / "source_evaluation.json", source_evaluation)

    if source_evaluation_only:
        summary: Dict[str, Any] = {
            "protocol": {
                "name": "source_evaluation_only",
                "mode": args.protocol_mode,
                "source_evaluation_only": True,
                "source_domain": source_domain,
                "source_domain_name": spec.domain_name(source_domain),
                "source_domains": list(source_domains),
                "source_domain_names": [
                    spec.domain_name(domain) for domain in source_domains
                ],
                "configured_target_domains": list(configured_target_domains),
                "target_domain_repeats_ignored": args.target_domain_repeats,
                "target_stream_processed": False,
                "target_labels_used_for_adaptation": False,
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
                "foundation_model": args.foundation_model,
                "foundation_config": foundation_config.to_dict(),
                "model_id": effective_model_id or None,
                "revision": effective_revision,
                "classifier": args.classifier,
                "feature_branch": effective_feature_branch,
                "feature_dim": model.feature_dim,
            },
            "method": "source",
            "source": {
                "validation": source_model_validation,
                "classifier_validation": source_model_validation,
                "checkpoint_metadata": checkpoint_metadata,
                "evaluation": source_evaluation,
            },
            "source_baseline": None,
            "online": None,
            "post_adapt": None,
            "adapter": None,
            "target_forgetting": {
                "enabled": False,
                "reason": "source_evaluation_only",
            },
            "domains": [],
            "run_dir": str(run_dir),
        }
        public_args = dict(vars(args))
        if public_args.get("hf_token"):
            public_args["hf_token"] = "<redacted>"
        _write_json(run_dir / "config.json", public_args)
        _write_json(run_dir / "source_history.json", {"epochs": source_history})
        _write_json(run_dir / "summary.json", summary)
        print(f"Source evaluation only; results={run_dir}")
        return summary

    source_reference_loader = data.loader(
        data.source.train,
        args.eval_batch_size,
        shuffle=False,
        workers=args.workers,
        device=device,
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

    knn_reference_features: Optional[torch.Tensor] = None
    knn_reference_labels: Optional[torch.Tensor] = None
    if args.method.lower() == "knn":
        knn_reference_features, knn_reference_labels = build_reference_bank(
            model,
            source_reference_loader,
            device,
        )
        print(
            "Built kNN source-train reference bank: "
            f"{knn_reference_features.shape[0]} samples, "
            f"{knn_reference_features.shape[1]} features"
        )

    gold_source_prototypes = None
    if args.method.lower() == "gold" or (
        args.method.lower() == "gold_tca"
        and args.gold_tca_agop_init == "source_prototypes"
    ):
        from .gold import build_source_prototypes

        gold_source_prototypes = build_source_prototypes(
            model, source_reference_loader, device, spec.num_classes
        )
        print(
            "Built GOLD source-train prototypes: "
            f"{tuple(gold_source_prototypes.shape)}"
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
        note_memory_size=args.note_memory_size,
        note_update_interval=args.note_update_interval,
        note_temperature=args.note_temperature,
        rotta_memory_size=args.rotta_memory_size,
        rotta_update_frequency=args.rotta_update_frequency,
        rotta_nu=args.rotta_nu,
        rotta_bn_alpha=args.rotta_bn_alpha,
        rotta_lambda_t=args.rotta_lambda_t,
        rotta_lambda_u=args.rotta_lambda_u,
        rotta_augmentation_noise_std=args.rotta_augmentation_noise_std,
        rotta_augmentation_scale_std=args.rotta_augmentation_scale_std,
        rotta_augmentation_mask_probability=(
            args.rotta_augmentation_mask_probability
        ),
        rotta_augmentation_shift_ratio=args.rotta_augmentation_shift_ratio,
        gold_source_prototypes=gold_source_prototypes,
        gold_top_r=args.gold_top_r,
        gold_alpha=args.gold_alpha,
        gold_conf_thresh=args.gold_conf_thresh,
        gold_update_every=args.gold_update_every,
        gold_adapter_lr_scale=args.gold_adapter_lr_scale,
        gold_lambda_ce_trg=args.gold_lambda_ce_trg,
        gold_lambda_cont=args.gold_lambda_cont,
        gold_temperature=args.gold_temperature,
        gold_projection_dim=args.gold_projection_dim,
        gold_teacher_momentum=args.gold_teacher_momentum,
        gold_augmentation_noise_std=args.gold_augmentation_noise_std,
        gold_augmentation_scale_std=args.gold_augmentation_scale_std,
        gold_augmentation_mask_probability=args.gold_augmentation_mask_probability,
        gold_augmentation_shift_ratio=args.gold_augmentation_shift_ratio,
        gold_tca_lambda_topology=args.gold_tca_lambda_topology,
        gold_tca_centroid_momentum=args.gold_tca_centroid_momentum,
        gold_tca_agop_init=args.gold_tca_agop_init,
        tca_teacher_momentum=args.tca_teacher_momentum,
        tca_centroid_momentum=args.tca_centroid_momentum,
        tca_topology_weight_momentum=args.tca_topology_weight_momentum,
        tca_alignment_weight=args.tca_alignment_weight,
        tca_topology_weight=args.tca_topology_weight,
        tca_uniformity_temperature=args.tca_uniformity_temperature,
        tca_symmetric_alpha=args.tca_symmetric_alpha,
        tca_augmentation_noise_std=args.tca_augmentation_noise_std,
        tca_augmentation_scale_std=args.tca_augmentation_scale_std,
        tca_augmentation_mask_probability=args.tca_augmentation_mask_probability,
        tca_augmentation_shift_ratio=args.tca_augmentation_shift_ratio,
        ttt_auxiliary_head=ttt_auxiliary_head,
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
        continual_mae_mask_ratio=args.continual_mae_mask_ratio,
        continual_mae_augmentations=args.continual_mae_augmentations,
        continual_mae_reconstruction_weight=(
            args.continual_mae_reconstruction_weight
        ),
        continual_mae_restore_probability=args.continual_mae_restore_probability,
        continual_mae_augmentation_noise_std=(
            args.continual_mae_augmentation_noise_std
        ),
        continual_mae_augmentation_scale_std=(
            args.continual_mae_augmentation_scale_std
        ),
        continual_mae_augmentation_mask_probability=(
            args.continual_mae_augmentation_mask_probability
        ),
        continual_mae_augmentation_shift_ratio=(
            args.continual_mae_augmentation_shift_ratio
        ),
        knn_neighbors=args.knn_neighbors,
        knn_reference_features=knn_reference_features,
        knn_reference_labels=knn_reference_labels,
    )
    if args.method.lower() == "gold_tca":
        print(
            "Initialized GOLD-TCA Golden Subspace: "
            f"agop_init={args.gold_tca_agop_init} "
            f"feature_dim={adapter.feature_dim} rank={adapter.top_r}"
        )
    if args.method.lower() in {"gold", "gold_tca"} and args.gold_warmup_steps > 0:
        adapter.warmup(source_train_loader, args.gold_warmup_steps)

    # Adaptive adapters own a copy. Keep the original checkpoint model as the
    # frozen baseline; SourceAdapter safely shares it because it never updates.
    model.eval()
    model.requires_grad_(False)

    source_before = (
        evaluate_adapter(adapter, source_validation_loader, device, spec.num_classes)
        if args.method.lower() == "knn"
        or (
            args.method.lower() in {"gold", "gold_tca"}
            and args.gold_warmup_steps > 0
        )
        else source_model_validation
    )
    if args.method.lower() == "knn":
        print(
            "kNN source-train-reference validation accuracy: "
            f"{source_before['accuracy']:.2f}%"
        )

    overall = ClassificationMeter(spec.num_classes)
    post_adapt_overall = ClassificationMeter(spec.num_classes)
    source_overall = ClassificationMeter(spec.num_classes)
    source_baseline_rows: List[Dict[str, Any]] = []
    domain_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    target_forgetting_rows: List[Dict[str, Any]] = []
    target_forgetting_history: Dict[int, List[Dict[str, Any]]] = {}
    for position, domain in enumerate(target_domains, start=1):
        domain_name = spec.domain_name(domain)
        target_dataset = data.target_dataset(domain, stream_position=position - 1)
        target_loader = data.loader(
            target_dataset,
            args.stream_batch_size,
            shuffle=False,
            workers=args.workers,
            device=device,
        )
        forgetting_row: Optional[Dict[str, Any]] = None
        if target_forgetting_enabled:
            pre_visit_metrics = evaluate_adapter(
                adapter,
                target_loader,
                device,
                spec.num_classes,
            )
            history = target_forgetting_history.setdefault(domain, [])
            forgetting_row = {
                "position": position,
                "domain": domain,
                "domain_name": domain_name,
                "occurrence": len(history) + 1,
                "is_repeat": bool(history),
                "previous_position": history[-1]["position"] if history else None,
                "intervening_domains": (
                    position - history[-1]["position"] - 1 if history else None
                ),
                "samples": pre_visit_metrics["samples"],
                "sample_set_sha256": _sample_set_digest(target_dataset.sample_ids),
            }
            for metric in TARGET_FORGETTING_METRICS:
                current = pre_visit_metrics[metric]
                first_occurrence = (
                    history[0]["pre_visit"][metric] if history else current
                )
                forgetting_row[f"pre_visit_{metric}"] = current
                forgetting_row[
                    f"first_occurrence_pre_visit_{metric}"
                ] = first_occurrence
                forgetting_row[
                    f"{metric}_change_from_first_occurrence_pp"
                ] = (current - first_occurrence)
                if history:
                    previous = history[-1]["post_visit"][metric]
                    first = history[0]["post_visit"][metric]
                    best = max(item["post_visit"][metric] for item in history)
                    forgetting_row[f"previous_post_visit_{metric}"] = previous
                    forgetting_row[f"first_post_visit_{metric}"] = first
                    forgetting_row[f"best_previous_post_visit_{metric}"] = best
                    forgetting_row[f"{metric}_drop_from_previous_pp"] = (
                        previous - current
                    )
                    forgetting_row[f"{metric}_drop_from_first_pp"] = first - current
                    forgetting_row[f"{metric}_forgetting_from_best_pp"] = (
                        best - current
                    )
                else:
                    forgetting_row[f"previous_post_visit_{metric}"] = None
                    forgetting_row[f"first_post_visit_{metric}"] = None
                    forgetting_row[f"best_previous_post_visit_{metric}"] = None
                    forgetting_row[f"{metric}_drop_from_previous_pp"] = None
                    forgetting_row[f"{metric}_drop_from_first_pp"] = None
                    forgetting_row[f"{metric}_forgetting_from_best_pp"] = None
        domain_meter = ClassificationMeter(spec.num_classes)
        post_adapt_meter = ClassificationMeter(spec.num_classes)
        source_meter = ClassificationMeter(spec.num_classes)
        for batch, (inputs, labels, sample_ids) in enumerate(target_loader):
            # Protocol boundary: target labels are used by meters only.
            device_inputs = inputs.to(device)
            outputs = adapter(device_inputs)
            post_outputs = (
                outputs
                if args.method.lower() == "source"
                else adapter.predict(device_inputs)
            )
            with torch.no_grad():
                source_outputs = (
                    outputs if args.method.lower() == "source" else model(device_inputs)
                )
            source_meter.update(source_outputs, labels)
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
                    source_outputs,
                )
            )

        overall.merge(domain_meter)
        post_adapt_overall.merge(post_adapt_meter)
        source_overall.merge(source_meter)
        metrics = domain_meter.compute()
        post_metrics = post_adapt_meter.compute()
        source_metrics = source_meter.compute()
        source_baseline_rows.append({
            "position": position,
            "domain": domain,
            "domain_name": domain_name,
            "sample_ids_sha256": _sample_digest(target_dataset.sample_ids),
            **source_metrics,
        })
        retention = (
            source_before
            if args.method.lower() == "source"
            else evaluate_adapter(
                adapter, source_validation_loader, device, spec.num_classes
            )
        )
        row: Dict[str, Any] = {
            "position": position,
            "domain": domain,
            "domain_name": domain_name,
            **metrics,
            **{f"source_{key}": value for key, value in source_metrics.items()},
            **relative_improvement(metrics, source_metrics),
            **{
                f"post_adapt_{key}": value
                for key, value in relative_improvement(post_metrics, source_metrics).items()
            },
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
        if forgetting_row is not None:
            post_visit_metrics = evaluate_adapter(
                adapter,
                target_loader,
                device,
                spec.num_classes,
            )
            for metric in TARGET_FORGETTING_METRICS:
                forgetting_row[f"post_visit_{metric}"] = post_visit_metrics[metric]
            target_forgetting_history[domain].append(
                {
                    "position": position,
                    "pre_visit": pre_visit_metrics,
                    "post_visit": post_visit_metrics,
                }
            )
            target_forgetting_rows.append(forgetting_row)
            row.update(
                {
                    "target_domain_occurrence": forgetting_row["occurrence"],
                    "target_forgetting_accuracy_drop_pp": forgetting_row[
                        "accuracy_drop_from_previous_pp"
                    ],
                    "target_forgetting_accuracy_from_best_pp": forgetting_row[
                        "accuracy_forgetting_from_best_pp"
                    ],
                    "target_accuracy_change_from_first_occurrence_pp": forgetting_row[
                        "accuracy_change_from_first_occurrence_pp"
                    ],
                }
            )
        domain_rows.append(row)
        print(
            f"[target {position:02d}/{len(target_domains):02d}] "
            f"domain={domain} samples={int(metrics['samples'])} "
            f"online_acc={metrics['accuracy']:.2f}% "
            f"source_acc={source_metrics['accuracy']:.2f}% "
            f"gain={metrics['accuracy'] - source_metrics['accuracy']:+.2f}pp "
            f"post_adapt_acc={post_metrics['accuracy']:.2f}% "
            f"source_retention={retention['accuracy']:.2f}%"
        )
        if forgetting_row is not None and forgetting_row["is_repeat"]:
            print(
                f"[target forgetting] domain={domain} "
                f"occurrence={forgetting_row['occurrence']} "
                f"pre_visit_acc={forgetting_row['pre_visit_accuracy']:.2f}% "
                f"drop_from_previous="
                f"{forgetting_row['accuracy_drop_from_previous_pp']:+.2f}pp "
                f"from_best="
                f"{forgetting_row['accuracy_forgetting_from_best_pp']:+.2f}pp "
                f"change_from_first="
                f"{forgetting_row['accuracy_change_from_first_occurrence_pp']:+.2f}pp"
            )

    online_metrics = overall.compute()
    post_metrics = post_adapt_overall.compute()
    source_metrics = source_overall.compute()
    source_domain_average = float(np.mean([row["accuracy"] for row in source_baseline_rows]))
    domain_average = float(np.mean([row["accuracy"] for row in domain_rows]))
    post_domain_average = float(
        np.mean([row["post_adapt_accuracy"] for row in domain_rows])
    )
    final_retention = domain_rows[-1]["source_retention_accuracy"]
    target_forgetting = (
        _target_forgetting_summary(
            target_forgetting_rows,
            args.allow_target_sample_reuse,
        )
        if target_forgetting_enabled
        else {"enabled": False}
    )
    if target_forgetting_enabled:
        for occurrence_mean in target_forgetting["occurrence_means"]:
            print(
                "[target forgetting occurrence mean] "
                f"occurrence={occurrence_mean['occurrence']} "
                f"domains={occurrence_mean['domain_count']} "
                f"pre_visit_acc={occurrence_mean['mean_pre_visit_accuracy']:.2f}% "
                f"post_visit_acc={occurrence_mean['mean_post_visit_accuracy']:.2f}% "
                f"change_from_first="
                f"{occurrence_mean['mean_accuracy_change_from_first_occurrence_pp']:+.2f}pp"
            )
    summary: Dict[str, Any] = {
        "protocol": {
            "name": (
                "online_continual_target_forgetting"
                if target_forgetting_enabled
                else "online_continual_test_time_adaptation"
            ),
            "mode": args.protocol_mode,
            "source_domain": source_domain,
            "source_domain_name": spec.domain_name(source_domain),
            "source_domains": list(source_domains),
            "source_domain_names": [
                spec.domain_name(domain) for domain in source_domains
            ],
            "target_domains": list(target_domains),
            "configured_target_domains": list(configured_target_domains),
            "target_domain_repeats": args.target_domain_repeats,
            "allow_target_sample_reuse": args.allow_target_sample_reuse,
            "target_domain_names": [
                spec.domain_name(domain) for domain in target_domains
            ],
            "single_pass": not target_forgetting_enabled,
            "reset_between_domains": False,
            "target_labels_used_for_adaptation": False,
            "primary_prediction_stage": (
                "post_adapt" if args.method.lower() == "ttt" else "online_pre_adapt"
            ),
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
            "foundation_model": args.foundation_model,
            "foundation_config": foundation_config.to_dict(),
            "model_id": effective_model_id or None,
            "revision": effective_revision,
            "classifier": args.classifier,
            "feature_branch": effective_feature_branch,
            "feature_dim": model.feature_dim,
        },
        "method": args.method.lower(),
        "source": {
            "validation": source_before,
            "classifier_validation": source_model_validation,
            "checkpoint_metadata": checkpoint_metadata,
            "evaluation": source_evaluation,
        },
        "source_baseline": {
            **source_metrics,
            "domain_average_accuracy": source_domain_average,
            "definition": "Frozen source checkpoint evaluated on exactly the current target samples",
            "matching_keys": ["domain", "position", "sample_id"],
            "source_checkpoint": args.source_checkpoint or str(run_dir / "source_checkpoint"),
            "domains": source_baseline_rows,
        },
        "online": {
            **online_metrics,
            **relative_improvement(online_metrics, source_metrics),
            "domain_average_accuracy_gain_pp": domain_average - source_domain_average,
            "domain_average_accuracy_relative_gain_pct": (
                100.0 * (domain_average - source_domain_average) / source_domain_average
                if source_domain_average else None
            ),
            "domain_average_accuracy": domain_average,
            "final_source_retention_accuracy": final_retention,
            "final_source_forgetting": (
                source_before["accuracy"] - final_retention
            ),
        },
        "post_adapt": {
            **post_metrics,
            **relative_improvement(post_metrics, source_metrics),
            "domain_average_accuracy_gain_pp": post_domain_average - source_domain_average,
            "domain_average_accuracy_relative_gain_pct": (
                100.0 * (post_domain_average - source_domain_average) / source_domain_average
                if source_domain_average else None
            ),
            "domain_average_accuracy": post_domain_average,
            "definition": (
                "Same-batch prediction after adaptation, without another update"
            ),
        },
        "adapter": adapter.statistics(),
        "target_forgetting": target_forgetting,
        "domains": domain_rows,
        "run_dir": str(run_dir),
    }

    public_args = dict(vars(args))
    if public_args.get("hf_token"):
        public_args["hf_token"] = "<redacted>"
    _write_json(run_dir / "config.json", public_args)
    _write_json(run_dir / "source_history.json", {"epochs": source_history})
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "source_baseline.json", summary["source_baseline"])
    _write_csv(run_dir / "source_baseline.csv", source_baseline_rows)
    _write_csv(run_dir / "domains.csv", domain_rows)
    _write_csv(run_dir / "predictions.csv", prediction_rows)
    if target_forgetting_enabled:
        _write_json(run_dir / "target_forgetting.json", target_forgetting)
        _write_csv(
            run_dir / "target_forgetting.csv",
            target_forgetting_rows,
        )
        _write_csv(
            run_dir / "target_forgetting_occurrences.csv",
            target_forgetting["occurrence_means"],
        )
    print(
        f"Online accuracy={online_metrics['accuracy']:.2f}% "
        f"post-adapt accuracy={post_metrics['accuracy']:.2f}% "
        f"domain-average={domain_average:.2f}% results={run_dir}"
    )
    return summary
