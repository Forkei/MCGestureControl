"""Transformer encoder model for multi-label gesture classification.

Architecture:
    Linear projection: input_dim (485) -> d_model (128)
    Learnable positional encoding (max_seq_len positions)
    Transformer encoder: 3 layers, 4 heads, dim_feedforward=256, dropout=0.35
    Classification head: masked mean pooling -> Linear -> num_gestures

Input features include raw pose+hand coordinates (260 dims) plus
frame-to-frame velocity of position coordinates (225 dims) = 485 total.

Output: raw logits for each gesture (no sigmoid). During training,
BCEWithLogitsLoss applies sigmoid internally. At inference time, apply
torch.sigmoid() manually and threshold at 0.5.
"""

import torch
import torch.nn as nn


class GestureTransformer(nn.Module):
    """Small Transformer encoder for multi-label classification of pose+hand sequences.

    Each output logit corresponds to an independent gesture. Multiple gestures
    can be active simultaneously (e.g., walking + crouching + mining).

    Args:
        input_dim: Feature dimension per frame (default 260).
        num_gestures: Number of independent gesture labels (sigmoid outputs).
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length (for positional encoding).
    """

    def __init__(
        self,
        input_dim: int = 485,
        num_gestures: int = 9,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_seq_len: int = 120,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head: one logit per gesture (independent sigmoids)
        self.classifier = nn.Linear(d_model, num_gestures)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) -- input features.
            mask: (batch, seq_len) -- 1.0 for real frames, 0.0 for padding.

        Returns:
            logits: (batch, num_gestures) -- raw logits, apply sigmoid for
                    probabilities.
        """
        B, T, _ = x.shape

        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embedding(positions)  # (B, T, d_model)

        x = self.dropout(x)

        # Transformer expects src_key_padding_mask: True = ignore (padded)
        padding_mask = mask == 0  # (B, T), True where padded

        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # (B, T, d_model)

        # Masked mean pooling -- average only non-padded positions
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # (B, d_model)

        logits = self.classifier(x)  # (B, num_gestures)
        return logits
