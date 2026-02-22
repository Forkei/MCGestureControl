"""Transformer encoder model for Minecraft control policy.

Architecture:
    Linear projection: input_dim (559) -> d_model (192)
    Learnable positional encoding (max 120 positions)
    Transformer encoder: 4 layers, 6 heads, dim_feedforward=384, dropout=0.3
    Exponentially-weighted temporal pooling -> (batch, 192)
    Action head: Linear(192->64) -> ReLU -> Dropout(0.2) -> Linear(64->8) -> raw logits
    Look head:   Linear(192->64) -> ReLU -> Dropout(0.2) -> Linear(64->2) -> tanh -> [-1, 1]

Input features (559 dims per frame):
    -- Core pose & hand (260) --
    pose_world (99), left_hand_3d (63), right_hand_3d (63),
    left_hand_present (1), right_hand_present (1), pose_visibility (33)
    -- Velocity (225) --
    Frame-to-frame deltas of position coordinates (indices 0:225)
    -- Game state (24) --
    Item properties, health, hunger, player state flags, combat context
    -- Action history (50) --
    Model's own output from previous 5 frames (10 outputs x 5 = 50)

Outputs:
    action_logits: (batch, 8) -- raw logits for binary controls (BCE loss)
    look: (batch, 2) -- yaw/pitch in [-1, 1] (MSE loss)
"""

import torch
import torch.nn as nn


class ControlTransformer(nn.Module):
    """Transformer encoder for continuous control from pose+hand+game state sequences.

    Dual output heads: binary action logits and analog look direction.
    Uses exponentially-weighted temporal pooling to emphasize recent frames.

    Args:
        input_dim: Feature dimension per frame.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout: Dropout rate for transformer layers.
        max_seq_len: Maximum sequence length (for positional encoding).
        num_binary: Number of binary action outputs.
        num_analog: Number of analog look outputs.
    """

    def __init__(
        self,
        input_dim: int = 559,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.3,
        max_seq_len: int = 120,
        num_binary: int = 8,
        num_analog: int = 2,
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

        self.dropout = nn.Dropout(dropout)

        # Action head: binary controls (raw logits, no sigmoid)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_binary),
        )

        # Look head: analog yaw/pitch in [-1, 1]
        self.look_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_analog),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) -- input features.
            mask: (batch, seq_len) -- 1.0 for real frames, 0.0 for padding.

        Returns:
            action_logits: (batch, num_binary) -- raw logits for binary controls.
            look: (batch, num_analog) -- yaw/pitch values in [-1, 1].
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

        x = self.transformer_encoder(
            x, src_key_padding_mask=padding_mask
        )  # (B, T, d_model)

        # Exponentially-weighted temporal pooling
        # Recent frames get more weight: e^-2 (oldest) to e^0 (newest)
        weights = torch.exp(
            torch.linspace(-2.0, 0.0, T, device=x.device)
        )  # (T,)
        weights = weights.unsqueeze(0) * mask  # (B, T) -- zero out padding
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # normalize
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        action_logits = self.action_head(pooled)  # (B, num_binary)
        look = self.look_head(pooled)  # (B, num_analog)

        return action_logits, look


class ControlTransformerV2(nn.Module):
    """V2 Transformer encoder for Minecraft control with 5 output heads.

    Architecture:
        Linear projection: input_dim (671) -> d_model (256)
        Learnable positional encoding (max 30 positions)
        Transformer encoder: 6 layers, 8 heads, dim_feedforward=512, dropout=0.3
        Exponentially-weighted temporal pooling with learnable decay
        5 output heads sharing the 256-dim backbone output:
            Action:   Linear(256->128)->ReLU->Dropout(0.3)->Linear(128->12) -> raw logits
            Look:     Linear(256->128)->ReLU->Dropout(0.3)->Linear(128->2)  -> tanh
            Hotbar:   Linear(256->128)->ReLU->Dropout(0.3)->Linear(128->9)  -> raw logits
            Cursor:   Linear(256->128)->ReLU->Dropout(0.3)->Linear(128->2)  -> sigmoid
            InvClick: Linear(256->128)->ReLU->Dropout(0.3)->Linear(128->3)  -> raw logits

    Input features (671 dims per frame):
        Core pose & hand (260), velocity (225), game state (46), action history (140)

    Outputs (dict):
        action_logits:    (batch, 12) -- raw logits for binary controls
        look:             (batch, 2)  -- yaw/pitch in [-1, 1]
        hotbar_logits:    (batch, 9)  -- raw logits for hotbar slot
        cursor:           (batch, 2)  -- cursor position in [0, 1]
        inv_click_logits: (batch, 3)  -- raw logits for inventory clicks
    """

    def __init__(
        self,
        input_dim: int = 671,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        max_seq_len: int = 30,
        num_action: int = 12,
        num_look: int = 2,
        num_hotbar: int = 9,
        num_cursor: int = 2,
        num_inv_click: int = 3,
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

        self.dropout = nn.Dropout(dropout)

        # Learnable decay for exponential temporal pooling (init -2.0)
        self.temporal_decay = nn.Parameter(torch.tensor(-2.0))

        # Head 1: Action (12 binary actions) — raw logits
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_action),
        )

        # Head 2: Look (yaw, pitch) — tanh → [-1, 1]
        self.look_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_look),
            nn.Tanh(),
        )

        # Head 3: Hotbar (9 slots) — raw logits for CrossEntropy
        self.hotbar_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_hotbar),
        )

        # Head 4: Cursor (x, y) — sigmoid → [0, 1]
        self.cursor_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_cursor),
            nn.Sigmoid(),
        )

        # Head 5: Inventory clicks (left, right, shift) — raw logits
        self.inv_click_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_inv_click),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: (batch, seq_len, 671) input features.
            mask: (batch, seq_len) — 1.0 for real frames, 0.0 for padding.

        Returns dict with keys:
            'action_logits':    (batch, 12) — raw logits for BCE
            'look':             (batch, 2)  — tanh output [-1, 1]
            'hotbar_logits':    (batch, 9)  — raw logits for CE
            'cursor':           (batch, 2)  — sigmoid output [0, 1]
            'inv_click_logits': (batch, 3)  — raw logits for BCE
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

        x = self.transformer_encoder(
            x, src_key_padding_mask=padding_mask
        )  # (B, T, d_model)

        # Exponentially-weighted temporal pooling with learnable decay
        # decay < 0 means recent frames (higher index) get more weight
        time_steps = torch.linspace(0.0, 1.0, T, device=x.device)  # (T,)
        weights = torch.exp(self.temporal_decay * (1.0 - time_steps))  # (T,)
        weights = weights.unsqueeze(0) * mask  # (B, T) — zero out padding
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        return {
            'action_logits': self.action_head(pooled),       # (B, 12)
            'look': self.look_head(pooled),                  # (B, 2)
            'hotbar_logits': self.hotbar_head(pooled),       # (B, 9)
            'cursor': self.cursor_head(pooled),              # (B, 2)
            'inv_click_logits': self.inv_click_head(pooled), # (B, 3)
        }

    @staticmethod
    def config_from_dict(d: dict) -> 'ControlTransformerV2':
        """Build a ControlTransformerV2 from a JSON config dict (Contract 7)."""
        model_cfg = d.get('model', d)
        return ControlTransformerV2(
            input_dim=model_cfg.get('input_dim', 671),
            d_model=model_cfg.get('d_model', 256),
            nhead=model_cfg.get('nhead', 8),
            num_layers=model_cfg.get('num_layers', 6),
            dim_feedforward=model_cfg.get('dim_feedforward', 512),
            dropout=model_cfg.get('dropout', 0.3),
            max_seq_len=model_cfg.get('max_seq_len', 30),
            num_action=model_cfg.get('num_action', 12),
            num_look=model_cfg.get('num_look', 2),
            num_hotbar=model_cfg.get('num_hotbar', 9),
            num_cursor=model_cfg.get('num_cursor', 2),
            num_inv_click=model_cfg.get('num_inv_click', 3),
        )

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
