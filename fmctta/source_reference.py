"""Persistent, exact-sample frozen-source outputs shared by online runs."""

import fcntl
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Mapping

import torch

from .metrics import ClassificationMeter


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SourceReference:
    """Labels validate identity; only the frozen model supplies cached logits.

    Writers lock, reload, merge by ID, then atomically replace the file. This
    preserves additions from other runs even when their in-memory views differ.
    Readers see either the previous complete file or the next complete file.
    """

    def __init__(
        self, path: Path, metadata: Mapping[str, object], announce=True,
        rebuild_on_checkpoint_change=False,
    ):
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.domains = {}
        self.source_validation = None
        self.samples_reused = 0
        self.samples_forwarded = 0
        self.domains_completed = set()
        self.dirty = False
        if rebuild_on_checkpoint_change and self.path.exists():
            # Use the same lock as persist so an old cache cannot be merged
            # into the new checkpoint's reference during replacement.
            with self.path.with_suffix(self.path.suffix + ".lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if self.path.exists():
                    payload = torch.load(self.path, map_location="cpu")
                    previous_hash = payload.get("metadata", {}).get("checkpoint_sha256")
                    current_hash = self.metadata.get("checkpoint_sha256")
                    if previous_hash and current_hash and previous_hash != current_hash:
                        backup = self.path.with_name(
                            f"{self.path.stem}.stale-{time.time_ns()}{self.path.suffix}"
                        )
                        os.replace(self.path, backup)
                        if announce:
                            print(f"[source reference] checkpoint changed; archived {backup}")
                        # No outputs or validation metrics from the old model
                        # can contribute to the new reference.
                    else:
                        self._read(payload)
                        if announce:
                            print(f"[source reference] loaded {self.path}")
                        return
        if self.path.exists():
            self._read(torch.load(self.path, map_location="cpu"))
            if announce:
                print(f"[source reference] loaded {self.path}")
        elif announce:
            print(f"[source reference] creating {self.path}")

    def _read(self, payload):
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported source reference schema_version")
        actual = payload.get("metadata", {})
        if actual != self.metadata:
            keys = set(actual) | set(self.metadata)
            mismatches = [key for key in sorted(keys)
                          if actual.get(key) != self.metadata.get(key)]
            raise ValueError(
                "Source reference is incompatible with this run: "
                + ", ".join(mismatches)
            )
        for domain, outputs in payload["domains"].items():
            self.domains[int(domain)] = self._unpack(outputs)
        validation = payload.get("source_validation")
        if validation is not None:
            self._unpack(validation)
            accuracy = 100.0 * int(validation["logits"].argmax(1).eq(
                validation["labels"]).sum()) / len(validation["labels"])
            if (validation["accuracy"] != accuracy
                    or validation["metrics"]["accuracy"] != accuracy
                    or validation["metrics"]["samples"] != len(validation["labels"])):
                raise ValueError("Invalid source validation baseline metrics")
        self.source_validation = validation

    def _unpack(self, outputs):
        ids, labels, logits = (outputs[key] for key in
                               ("sample_ids", "labels", "logits"))
        if (ids.ndim != 1 or labels.shape != ids.shape
                or ids.dtype != torch.long or labels.dtype != torch.long
                or logits.shape != (len(ids), self.metadata["num_classes"])
                or not logits.is_floating_point() or not torch.isfinite(logits).all()
                or len(torch.unique(ids)) != len(ids)):
            raise ValueError("Invalid source reference sample IDs, labels, or logits")
        if ((labels < 0) | (labels >= self.metadata["num_classes"])).any():
            raise ValueError("Invalid source reference labels")
        return {int(sid): (int(label), output.clone())
                for sid, label, output in zip(ids.tolist(), labels.tolist(), logits)}

    @staticmethod
    def _pack(rows):
        ids = sorted(rows)
        return {
            "sample_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor([rows[sid][0] for sid in ids], dtype=torch.long),
            "logits": torch.stack([rows[sid][1] for sid in ids]).cpu(),
        }

    @torch.no_grad()
    def outputs(self, model, domain, inputs, labels, sample_ids, device):
        """Forward unique missing IDs only, then restore the input batch order."""
        domain = int(domain)
        rows = self.domains.setdefault(domain, {})
        ids = sample_ids.detach().cpu().long().tolist()
        targets = labels.detach().cpu().long().tolist()
        missing = {}
        for offset, (sid, label) in enumerate(zip(ids, targets)):
            if sid in rows:
                if rows[sid][0] != label:
                    raise ValueError(f"Source reference label mismatch for sample_id={sid}")
                self.samples_reused += 1
            elif sid in missing:
                if targets[missing[sid]] != label:
                    raise ValueError(f"Conflicting labels for sample_id={sid}")
            else:
                missing[sid] = offset
        if missing:
            offsets = torch.tensor(list(missing.values()), device=inputs.device)
            cuda_devices = ([device.index if device.index is not None
                             else torch.cuda.current_device()]
                            if device.type == "cuda" else [])
            # Cache misses must not change adaptation's random-number stream.
            with torch.random.fork_rng(devices=cuda_devices):
                model.eval()
                logits = model(inputs.index_select(0, offsets).to(device)).detach().cpu()
            if (logits.shape != (len(missing), self.metadata["num_classes"])
                    or not torch.isfinite(logits).all()):
                raise ValueError("Invalid frozen-source logits")
            for (sid, offset), output in zip(missing.items(), logits):
                rows[sid] = (targets[offset], output.clone())
            self.samples_forwarded += len(missing)
            self.domains_completed.add(domain)
            self.dirty = True
        return torch.stack([rows[sid][1] for sid in ids])

    def validation_metrics(self, model, data, loader, device):
        """Use the saved baseline only after checking the fixed split exactly."""
        ids = data.source.validation.sample_ids.cpu()
        labels = data.source.validation.labels.cpu()
        if self.source_validation is not None:
            saved = self.source_validation
            if not torch.equal(ids, saved["sample_ids"]):
                raise ValueError("Source validation sample_ids do not exactly match source reference")
            if not torch.equal(labels, saved["labels"]):
                raise ValueError("Source validation labels do not match source reference")
            return dict(saved["metrics"])

        # Old checkpoints without a reference are upgraded once, then reused.
        meter = ClassificationMeter(data.spec.num_classes)
        validation_rows = {}
        for inputs, targets, sample_ids in loader:
            outputs = torch.empty((len(sample_ids), data.spec.num_classes))
            for domain in data.source_domains:
                start = data.spec.domain_offset(domain)
                end = start + data.spec.samples_for_domain(domain)
                mask = (sample_ids >= start) & (sample_ids < end)
                if mask.any():
                    outputs[mask] = self.outputs(
                        model, domain, inputs[mask], targets[mask], sample_ids[mask], device
                    )
            meter.update(outputs, targets)
            validation_rows.update({int(sid): output.clone()
                                    for sid, output in zip(sample_ids.tolist(), outputs)})
        metrics = meter.compute()
        self.source_validation = {
            "sample_ids": ids.clone(),
            "labels": labels.clone(),
            "logits": torch.stack([validation_rows[int(sid)] for sid in ids.tolist()]),
            "accuracy": metrics["accuracy"],
            "metrics": metrics,
        }
        self.dirty = True
        return metrics

    def statistics(self):
        return {
            "cache_file": str(self.path.resolve()),
            "samples_reused": self.samples_reused,
            "samples_forwarded": self.samples_forwarded,
            # Domains receiving new outputs; not a claim of full-domain coverage.
            "domains_completed": sorted(self.domains_completed),
        }

    def reset_statistics(self):
        self.samples_reused = self.samples_forwarded = 0
        self.domains_completed.clear()

    def persist(self):
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            merged = SourceReference(self.path, self.metadata, announce=False)
            for domain, rows in self.domains.items():
                destination = merged.domains.setdefault(domain, {})
                for sid, row in rows.items():
                    if sid in destination:
                        existing = destination[sid]
                        if existing[0] != row[0] or not torch.allclose(
                            existing[1], row[1], rtol=1e-4, atol=1e-5
                        ):
                            raise ValueError(f"Conflicting source reference for sample_id={sid}")
                    else:
                        destination[sid] = row
            if merged.source_validation is None:
                merged.source_validation = self.source_validation
            elif self.source_validation is not None:
                for key in ("sample_ids", "labels"):
                    if not torch.equal(merged.source_validation[key], self.source_validation[key]):
                        raise ValueError("Conflicting source validation reference")
            payload = {
                "schema_version": 1,
                "metadata": self.metadata,
                "domains": {domain: self._pack(rows)
                            for domain, rows in merged.domains.items() if rows},
                "source_validation": merged.source_validation,
            }
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as handle:
                    temporary_path = Path(handle.name)
                    torch.save(payload, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self.path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            self.domains = merged.domains
            self.source_validation = merged.source_validation
            self.dirty = False
