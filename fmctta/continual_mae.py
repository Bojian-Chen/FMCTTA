"""One-dimensional Continual-MAE adaptation for UniFreq fault classifiers.

This adapts the central mechanism of Continual-MAE (CVPR 2024) to UniFreq's
global-FFT frequency-token stream: every non-DC frequency bin is one patch;
distribution-aware masks are selected from teacher-token variance; masked-token
predictions are matched to an unmasked teacher; and a small decoder
reconstructs normalized frequency magnitudes.  The original repository's HOG
reconstruction is image-specific, so it is not applicable to vibration signals.
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

# These common adapter utilities are imported lazily by build_adapter(), after fmctta.adapters has completed module initialization.
from .adapters import (
    OnlineAdapter,
    SignalAugment,
    configure_gradient_adaptation,
    soft_cross_entropy,
)


class ContinualMAEAdapter(OnlineAdapter):
    """Distribution-aware masked frequency-bin adaptation for UniFreq."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        steps: int,
        weight_decay: float,
        parameter_scope: str,
        mask_ratio: float,
        augmentations: int,
        reconstruction_weight: float,
        restore_probability: float,
        augmentation: SignalAugment,
    ) -> None:
        if not 0.0 < mask_ratio <= 1.0:
            raise ValueError("continual_mae_mask_ratio must be in (0, 1]")
        if augmentations < 2:
            raise ValueError("continual_mae_augmentations must be at least 2")
        if reconstruction_weight < 0:
            raise ValueError(
                "continual_mae_reconstruction_weight cannot be negative"
            )
        if not 0.0 <= restore_probability <= 1.0:
            raise ValueError(
                "continual_mae_restore_probability must be in [0, 1]"
            )
        if getattr(model, "feature_branch", None) == "time":
            raise ValueError(
                "Frequency-bin Continual-MAE requires feature_branch="
                "'frequency' or 'fused'"
            )
        if not all(
            hasattr(model, name)
            for name in (
                "frequency_token_batch",
                "encode_frequency_tokens",
                "classify_frequency_tokens",
            )
        ):
            raise TypeError("Continual-MAE requires a UniFreqFaultModel")

        super().__init__(copy.deepcopy(model), device)
        base_parameters, parameter_names = configure_gradient_adaptation(
            self.model,
            parameter_scope,
        )
        embed_dim = int(self.model.config.embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        nn.init.normal_(self.mask_token, std=0.02)
        self.reconstruction_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            [
                *base_parameters,
                *self.reconstruction_decoder.parameters(),
                self.mask_token,
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.teacher = copy.deepcopy(self.model).to(device).eval()
        self.teacher.requires_grad_(False)
        self.parameter_scope = parameter_scope
        self.parameter_names = tuple(parameter_names)
        self.source_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in set(self.parameter_names)
        }
        self.steps = steps
        self.mask_ratio = mask_ratio
        self.augmentations = augmentations
        self.reconstruction_weight = reconstruction_weight
        self.restore_probability = restore_probability
        self.augmentation = augmentation
        self.masked_tokens = 0
        self.reconstructed_tokens = 0
        self.restored_values = 0
        self.last_consistency_loss: Optional[float] = None
        self.last_reconstruction_loss: Optional[float] = None

    @staticmethod
    @torch.no_grad()
    def _frequency_point_descriptors(
        frequency_values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return one normalized magnitude reconstruction target per FFT bin."""

        if frequency_values.ndim != 2:
            raise ValueError(
                "frequency_values must have shape [batch, frequency_bins], got "
                f"{tuple(frequency_values.shape)}"
            )
        values = frequency_values.detach()
        if token_mask is None:
            weights = torch.ones_like(values)
        else:
            weights = token_mask.to(device=values.device, dtype=values.dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (values * weights).sum(dim=1, keepdim=True) / count
        variance = ((values - mean).square() * weights).sum(
            dim=1,
            keepdim=True,
        ) / count
        descriptors = (values - mean) / variance.clamp_min(1e-6).sqrt()
        descriptors = descriptors.masked_fill(~weights.bool(), 0.0)
        return descriptors.unsqueeze(-1)

    def _select_mask(
        self,
        uncertainty: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        selected = torch.zeros_like(uncertainty, dtype=torch.bool)
        if token_mask is None:
            count = max(1, int(uncertainty.shape[1] * self.mask_ratio))
            indices = uncertainty.topk(count, dim=1).indices
            selected.scatter_(1, indices, True)
            return selected
        for batch_id in range(uncertainty.shape[0]):
            valid = torch.nonzero(token_mask[batch_id], as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            count = max(1, int(valid.numel() * self.mask_ratio))
            local = uncertainty[batch_id, valid].topk(count).indices
            selected[batch_id, valid[local]] = True
        return selected

    @torch.no_grad()
    def _distribution_aware_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        token_outputs = []
        token_mask = None
        self.teacher.eval()
        for _ in range(self.augmentations):
            token_batch = self.teacher.frequency_token_batch(
                self.augmentation(inputs)
            )
            token_outputs.append(self.teacher.encode_frequency_tokens(token_batch))
            token_mask = token_batch["token_mask"]
        variance = torch.stack(token_outputs, dim=0).var(dim=0, unbiased=False)
        uncertainty = variance.mean(dim=-1)
        return self._select_mask(uncertainty, token_mask)

    @torch.no_grad()
    def _restore_source_values(self) -> None:
        if self.restore_probability <= 0:
            return
        for name, parameter in self.model.named_parameters():
            source = self.source_parameters.get(name)
            if source is None:
                continue
            restore = torch.rand_like(parameter) < self.restore_probability
            self.restored_values += int(restore.sum().item())
            parameter.copy_(torch.where(restore, source, parameter))

    @torch.no_grad()
    def _update_teacher(self) -> None:
        # The reference implementation uses a one-update-lag teacher rather than an EMA teacher; retain that behavior for this adaptation.
        self.teacher.load_state_dict(self.model.state_dict(), strict=True)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

    @torch.enable_grad()
    def _adapt_once(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            online_outputs = self.teacher(inputs).detach()
            selected_mask = self._distribution_aware_mask(inputs)

        token_batch = self.model.frequency_token_batch(inputs)
        tokens = token_batch["tokens"]
        if selected_mask.shape != tokens.shape[:2]:
            raise RuntimeError(
                "Teacher and student Continual-MAE token shapes differ: "
                f"{tuple(selected_mask.shape)} versus {tuple(tokens.shape[:2])}"
            )
        masked_tokens = torch.where(
            selected_mask.unsqueeze(-1),
            self.mask_token.to(device=tokens.device, dtype=tokens.dtype),
            tokens,
        )
        self.optimizer.zero_grad(set_to_none=True)
        student_outputs, encoded_tokens = self.model.classify_frequency_tokens(
            token_batch,
            masked_tokens,
        )
        consistency_loss = soft_cross_entropy(student_outputs, online_outputs).mean()
        descriptors = self._frequency_point_descriptors(
            token_batch["frequency_values"],
            token_batch["token_mask"],
        )
        reconstruction_loss = torch.nn.functional.mse_loss(
            self.reconstruction_decoder(encoded_tokens[selected_mask]),
            descriptors[selected_mask],
        )
        loss = consistency_loss + self.reconstruction_weight * reconstruction_loss
        if not torch.isfinite(loss):
            self.skipped_updates += 1
            self.optimizer.zero_grad(set_to_none=True)
            return online_outputs
        loss.backward()
        self.optimizer.step()
        self._restore_source_values()
        self._update_teacher()
        self.updates += 1
        self.masked_tokens += int(selected_mask.sum().item())
        self.reconstructed_tokens += int(selected_mask.sum().item())
        self.last_consistency_loss = float(consistency_loss.detach().item())
        self.last_reconstruction_loss = float(reconstruction_loss.detach().item())
        return online_outputs

    @torch.enable_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._record_batch(inputs)
        inputs = inputs.to(self.device)
        online_outputs: Optional[torch.Tensor] = None
        for _ in range(self.steps):
            outputs = self._adapt_once(inputs)
            if online_outputs is None:
                online_outputs = outputs
        assert online_outputs is not None
        return online_outputs

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        return self.teacher(inputs.to(self.device))

    def statistics(self) -> Dict[str, object]:
        values = super().statistics()
        values.update(
            {
                "parameter_scope": self.parameter_scope,
                "adapted_model_parameter_tensors": len(self.parameter_names),
                "adapted_model_parameters": sum(
                    parameter.numel()
                    for name, parameter in self.model.named_parameters()
                    if name in set(self.parameter_names)
                ),
                "auxiliary_parameters": sum(
                    parameter.numel()
                    for parameter in self.reconstruction_decoder.parameters()
                )
                + self.mask_token.numel(),
                "mask_ratio": self.mask_ratio,
                "augmentations": self.augmentations,
                "reconstruction_weight": self.reconstruction_weight,
                "restore_probability": self.restore_probability,
                "masked_tokens": self.masked_tokens,
                "reconstructed_tokens": self.reconstructed_tokens,
                "restored_parameter_values": self.restored_values,
                "last_consistency_loss": self.last_consistency_loss,
                "last_reconstruction_loss": self.last_reconstruction_loss,
            }
        )
        return values
