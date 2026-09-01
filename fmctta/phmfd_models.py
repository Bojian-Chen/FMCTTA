"""PHMFD UniFault and TF-ProFM inference backbones.

The implementations in this module preserve the parameter names used by the
PHMFD pretraining checkpoints.  Pretraining-only projection/classification
heads are deliberately omitted: FMCTTA attaches its own downstream fault
classifier after loading the reusable backbone tensors.
"""

from math import sqrt
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class FullAttention(nn.Module):
    """Full attention used by the PHMFD UniFault reproduction."""

    def __init__(
        self,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        del attn_mask
        _, _, _, head_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(head_dim)
        output_dtype = queries.dtype
        scores = torch.einsum(
            "blhe,bshe->bhls",
            queries.float(),
            keys.float(),
        )
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum(
            "bhls,bshd->blhd",
            attention,
            values.float(),
        )
        return output.to(output_dtype).contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if n_heads < 1 or d_model // n_heads < 1:
            raise ValueError("UniFault heads must be between 1 and embed_dim")
        head_dim = d_model // n_heads
        self.inner_attention = FullAttention(attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, head_dim * n_heads)
        self.key_projection = nn.Linear(d_model, head_dim * n_heads)
        self.value_projection = nn.Linear(d_model, head_dim * n_heads)
        self.out_projection = nn.Linear(head_dim * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        batch_size, query_length, _ = queries.shape
        key_length = keys.shape[1]
        queries = self.query_projection(queries).view(
            batch_size, query_length, self.n_heads, -1
        )
        keys = self.key_projection(keys).view(
            batch_size, key_length, self.n_heads, -1
        )
        values = self.value_projection(values).view(
            batch_size, key_length, self.n_heads, -1
        )
        output, attention = self.inner_attention(
            queries, keys, values, attn_mask
        )
        return self.out_projection(output.view(batch_size, query_length, -1)), attention


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Linear(d_model, 4 * d_model)
        self.conv2 = nn.Linear(4 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Kept even though the PHMFD forward path does not use it.  It is part
        # of the released checkpoints and therefore part of the architecture.
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(
        self,
        inputs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        attended, attention = self.attention(
            inputs, inputs, inputs, attn_mask=attn_mask
        )
        inputs = inputs + self.dropout(attended)
        residual = inputs = self.norm1(inputs)
        residual = self.dropout(self.activation(self.conv1(residual)))
        residual = self.dropout(self.conv2(residual))
        return self.norm2(inputs + residual), attention


class Encoder(nn.Module):
    def __init__(self, layers: List[EncoderLayer], d_model: int) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.attn_layers:
            inputs, _ = layer(inputs)
        return self.norm(inputs)


class PatchEmbed(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError(
                "UniFault input length must be divisible by its patch size"
            )
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.num_patches = self.seq_len // self.patch_size
        self.input_layer = nn.Linear(self.patch_size, embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "UniFault expects [B, C, L], got " f"{tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.seq_len:
            raise ValueError(
                f"UniFault input length is {inputs.shape[-1]}, expected "
                f"{self.seq_len}"
            )
        patches = inputs.unfold(-1, self.patch_size, self.patch_size)
        batch_size, channels, patch_count, patch_size = patches.shape
        patches = patches.contiguous().view(
            batch_size * channels, patch_count, patch_size
        )
        return self.input_layer(patches)


class UniFaultBackbone(nn.Module):
    """Time-domain Transformer compatible with PHMFD UniFault checkpoints."""

    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        embed_dim: int,
        heads: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_embed = PatchEmbed(seq_len, patch_size, self.embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(float(dropout))
        self.encoder = Encoder(
            [
                EncoderLayer(self.embed_dim, int(heads), float(dropout))
                for _ in range(int(depth))
            ],
            self.embed_dim,
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _ = inputs.shape
        tokens = self.patch_embed(inputs)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.encoder(tokens)
        tokens = tokens.reshape(
            batch_size,
            channels * self.patch_embed.num_patches,
            self.embed_dim,
        )
        return tokens.mean(dim=1)


class RandomPatchEmbedding(nn.Module):
    """Random local-FFT patch embedding used by TF-ProFM."""

    def __init__(self, patch_len: int, num_patches: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = int(patch_len)
        self.num_patches = int(num_patches)
        self.projector = nn.Sequential(
            nn.Linear(self.patch_len, int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), int(d_model)),
        )

    def _patch(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, seq_len = inputs.shape
        if self.patch_len > seq_len:
            repeats = (self.patch_len + seq_len - 1) // seq_len
            inputs = inputs.repeat(1, 1, repeats)
            seq_len = inputs.shape[-1]
        max_start = seq_len - self.patch_len
        starts = torch.randint(
            0,
            max_start + 1,
            (batch_size, self.num_patches),
            device=inputs.device,
        )
        offsets = torch.arange(self.patch_len, device=inputs.device)
        indices = (starts.unsqueeze(-1) + offsets).unsqueeze(2)
        patches = inputs.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        patches = patches.gather(
            3, indices.expand(-1, -1, channels, -1)
        )
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return torch.fft.fft(patches, dim=-1).abs()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projector(self._patch(inputs))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_ffn)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_ffn), int(d_model)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(inputs, inputs, inputs, need_weights=False)
        inputs = self.norm1(inputs + self.dropout(attended))
        return self.norm2(inputs + self.dropout(self.ffn(inputs)))


class TFProFMBackbone(nn.Module):
    """Backbone compatible with PHMFD TF-ProFM pretraining checkpoints."""

    def __init__(
        self,
        input_len: int,
        patch_len: int,
        num_patches: int,
        d_model: int,
        d_ffn: int,
        layers: int,
        heads: int,
        feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("TF-ProFM d_model must be divisible by heads")
        if patch_len > input_len:
            raise ValueError("TF-ProFM patch_len cannot exceed input_len")
        self.input_len = int(input_len)
        self.patch_len = int(patch_len)
        self.num_patches = int(num_patches)
        self.d_model = int(d_model)
        self.feature_dim = int(feature_dim)
        self.embedding = RandomPatchEmbedding(
            self.patch_len, self.num_patches, self.d_model
        )
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(d_model, d_ffn, heads, dropout)
                for _ in range(int(layers))
            ]
        )
        self.feature_projector = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.num_patches * self.d_model, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "TF-ProFM expects [B, C, L], got " f"{tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.input_len:
            raise ValueError(
                f"TF-ProFM input length is {inputs.shape[-1]}, expected "
                f"{self.input_len}"
            )
        tokens = self.embedding(inputs)
        batch_size, channels, _, _ = tokens.shape
        tokens = tokens.view(
            batch_size * channels, self.num_patches, self.d_model
        )
        for block in self.encoder:
            tokens = block(tokens)
        features = self.feature_projector(tokens)
        return features.view(batch_size, channels, self.feature_dim).mean(dim=1)
