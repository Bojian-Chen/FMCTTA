import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from .metrics import ClassificationMeter
from .ttt import ttt_auxiliary_logits


@dataclass
class SourceTrainingResult:
    best_macro_f1: float
    best_epoch: int
    history: List[Dict[str, float]]


@torch.no_grad()
def evaluate_supervised(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    meter = ClassificationMeter(num_classes)
    for inputs, labels, _ in loader:
        logits = model(inputs.to(device))
        meter.update(logits, labels)
    return meter.compute()


def train_source(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    learning_rate: float,
    backbone_lr_scale: float,
    weight_decay: float,
    gradient_clip: float,
    ttt_auxiliary_head: Optional[nn.Module] = None,
    ttt_auxiliary_weight: float = 1.0,
) -> SourceTrainingResult:
    if epochs < 1:
        raise ValueError("source epochs must be at least 1")
    if backbone_lr_scale < 0:
        raise ValueError("source backbone learning-rate scale cannot be negative")

    model.to(device)
    if ttt_auxiliary_weight < 0:
        raise ValueError("TTT source auxiliary weight cannot be negative")
    if ttt_auxiliary_head is not None:
        ttt_auxiliary_head.to(device)
    parameter_groups = [{"params": model.head.parameters(), "lr": learning_rate}]
    if ttt_auxiliary_head is not None:
        parameter_groups.append(
            {"params": ttt_auxiliary_head.parameters(), "lr": learning_rate}
        )
    if backbone_lr_scale > 0:
        model.enable_encoder_finetuning()
        parameter_groups.insert(
            0,
            {
                "params": model.encoder.parameters(),
                "lr": learning_rate * backbone_lr_scale,
            },
        )
    else:
        model.disable_encoder_finetuning()

    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_state = copy.deepcopy(model.state_dict())
    best_ttt_state = (
        copy.deepcopy(ttt_auxiliary_head.state_dict())
        if ttt_auxiliary_head is not None
        else None
    )
    best_macro_f1 = -1.0
    best_epoch = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        if ttt_auxiliary_head is not None:
            ttt_auxiliary_head.train()
        if backbone_lr_scale == 0:
            model.encoder.eval()
        loss_sum = 0.0
        total = 0
        correct = 0
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            classification_loss = nn.functional.cross_entropy(logits, labels)
            auxiliary_loss = classification_loss.new_zeros(())
            if ttt_auxiliary_head is not None:
                auxiliary_logits, transform_labels = ttt_auxiliary_logits(
                    model,
                    ttt_auxiliary_head,
                    inputs,
                )
                auxiliary_loss = nn.functional.cross_entropy(
                    auxiliary_logits,
                    transform_labels,
                )
            loss = classification_loss + ttt_auxiliary_weight * auxiliary_loss
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ],
                    gradient_clip,
                )
            optimizer.step()
            loss_sum += float(loss.detach().item()) * labels.numel()
            correct += int(logits.detach().argmax(dim=1).eq(labels).sum().item())
            total += labels.numel()
        scheduler.step()

        validation = evaluate_supervised(
            model,
            validation_loader,
            device,
            num_classes,
        )
        record = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(total, 1),
            "train_accuracy": 100.0 * correct / max(total, 1),
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "head_learning_rate": optimizer.param_groups[-1]["lr"],
        }
        history.append(record)
        print(
            f"[source {epoch:03d}/{epochs:03d}] "
            f"loss={record['train_loss']:.4f} "
            f"train_acc={record['train_accuracy']:.2f}% "
            f"val_acc={record['validation_accuracy']:.2f}% "
            f"val_f1={record['validation_macro_f1']:.2f}%"
        )
        if validation["macro_f1"] >= best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if ttt_auxiliary_head is not None:
                best_ttt_state = copy.deepcopy(ttt_auxiliary_head.state_dict())

    model.load_state_dict(best_state, strict=True)
    if ttt_auxiliary_head is not None and best_ttt_state is not None:
        ttt_auxiliary_head.load_state_dict(best_ttt_state, strict=True)
    model.eval()
    return SourceTrainingResult(best_macro_f1, best_epoch, history)


def save_checkpoint(
    directory: Path,
    model: nn.Module,
    metadata: Mapping[str, object],
    extra_modules: Optional[Mapping[str, nn.Module]] = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(state, str(directory / "source_model.safetensors"))
    for name, module in (extra_modules or {}).items():
        extra_state = {
            key: value.detach().cpu().contiguous()
            for key, value in module.state_dict().items()
        }
        save_file(extra_state, str(directory / f"{name}.safetensors"))
    (directory / "source_model.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(
    directory: Path,
    model: nn.Module,
    expected_metadata: Mapping[str, object],
    extra_modules: Optional[Mapping[str, nn.Module]] = None,
) -> Dict[str, object]:
    weights_path = directory / "source_model.safetensors"
    metadata_path = directory / "source_model.json"
    if not weights_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"{directory} must contain source_model.safetensors and "
            "source_model.json"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mismatches = {
        key: (metadata[key], expected)
        for key, expected in expected_metadata.items()
        if key in metadata and metadata[key] != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={actual!r}, run={wanted!r}"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"Source checkpoint is incompatible: {details}")
    model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
    for name, module in (extra_modules or {}).items():
        extra_path = directory / f"{name}.safetensors"
        if not extra_path.is_file():
            raise ValueError(
                f"Checkpoint {directory} has no {extra_path.name}; "
                "TTT requires a checkpoint trained with --method ttt"
            )
        module.load_state_dict(
            load_file(str(extra_path), device="cpu"),
            strict=True,
        )
        module.eval()
    model.eval()
    return metadata


def compute_fisher(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    parameter_names: Iterable[str],
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    selected = set(parameter_names)
    if not selected:
        return {}
    original_flags = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    encoder_was_enabled = bool(
        getattr(model.encoder, "_finetuning_enabled", False)
    )
    model.enable_encoder_finetuning()
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)

    fisher = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in model.named_parameters()
        if name in selected
    }
    model.eval()
    batches = 0
    for inputs, labels, _ in loader:
        if max_batches is not None and batches >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        logits = model(inputs.to(device))
        loss = nn.functional.cross_entropy(logits, labels.to(device))
        loss.backward()
        for name, parameter in model.named_parameters():
            if name in fisher and parameter.grad is not None:
                fisher[name] += parameter.grad.detach().pow(2)
        batches += 1
    for name in fisher:
        fisher[name] /= max(batches, 1)
    model.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_flags[name])
    if not encoder_was_enabled:
        model.disable_encoder_finetuning()
    model.eval()
    return fisher
