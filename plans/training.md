# Training Pipeline v2

## Overview

Train the 5-head ControlTransformer (~3.5M params) on recorded demonstrations
using mode-aware masked losses. Each output head has its own loss function and
evaluation metrics.

---

## Training Script: `train_controls.py`

### Inputs

- Recorded sessions from `client/recordings/` (.npz files)
- Each contains: frames (N, 260), controls (N, 28), game_state (N, 46)

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| batch_size | 64 | Increase to 128 if GPU memory allows |
| learning_rate | 5e-4 | Peak LR after warmup |
| warmup_epochs | 5 | Linear warmup from 0 to peak LR |
| lr_schedule | cosine decay | After warmup, decay to 1e-6 |
| weight_decay | 1e-4 | AdamW regularization |
| max_epochs | 200 | |
| early_stop_patience | 30 | Based on combined validation score |
| gradient_clip | 1.0 | Max gradient norm |
| action_history_dropout | 0.3 | Zero out 30% of history entries during training |

---

## Loss Functions

### Combined Loss

```python
loss = (w_action * action_loss * gameplay_mask.mean()
      + w_look   * look_loss   * gameplay_mask.mean()
      + w_hotbar * hotbar_loss * hotbar_mask.mean()
      + w_cursor * cursor_loss * screen_mask.mean()
      + w_inv    * inv_loss    * screen_mask.mean()
      + w_idle   * idle_loss)
```

### 1. Action Loss (BCEWithLogitsLoss)

- 12 binary outputs: 9 gameplay actions + 3 utility (drop, swap, open_inv)
- Per-control pos_weight = min(total / (2 * positives), 50.0)
- Applied only during gameplay frames (gameplay_mask)

```python
action_loss = F.binary_cross_entropy_with_logits(
    action_logits,           # (batch, 12)
    action_targets,          # (batch, 12)
    pos_weight=pos_weights,  # (12,) per-control
    reduction='none'
)
action_loss = (action_loss * gameplay_mask.unsqueeze(1)).mean()
```

### 2. Look Loss (MSE / Smooth L1)

- 2 analog outputs: yaw, pitch (both [-1, 1])
- Only on frames where look is active (magnitude > deadzone)
- Deadzone threshold: 0.08

```python
look_active = (look_targets.abs() > 0.08).any(dim=1)  # (batch,)
look_mask = gameplay_mask & look_active
look_loss = F.smooth_l1_loss(look_output, look_targets, reduction='none')
look_loss = (look_loss * look_mask.unsqueeze(1)).sum() / look_mask.sum().clamp(min=1)
```

### 3. Hotbar Loss (CrossEntropyLoss)

- 9-class output (one-hot slot selection)
- Only on frames where slot actually changed
- Very sparse — most frames have no hotbar change

```python
# hotbar_target: (batch,) long tensor, -1 = no change
hotbar_changed = hotbar_target >= 0
if hotbar_changed.any():
    hotbar_loss = F.cross_entropy(
        hotbar_logits[hotbar_changed],
        hotbar_target[hotbar_changed]
    )
else:
    hotbar_loss = 0.0
```

### 4. Cursor Loss (MSE)

- 2 continuous outputs: x, y (both [0, 1])
- Only when inventory screen is open

```python
cursor_loss = F.mse_loss(cursor_output, cursor_targets, reduction='none')
cursor_loss = (cursor_loss * screen_mask.unsqueeze(1)).sum() / screen_mask.sum().clamp(min=1)
```

### 5. InvClick Loss (BCEWithLogitsLoss)

- 3 binary outputs: left_click, right_click, shift_held
- Only when inventory screen is open
- Per-control pos_weight for click imbalance

```python
inv_loss = F.binary_cross_entropy_with_logits(
    inv_logits, inv_targets, pos_weight=inv_pos_weights, reduction='none'
)
inv_loss = (inv_loss * screen_mask.unsqueeze(1)).sum() / screen_mask.sum().clamp(min=1)
```

### 6. Idle Penalty (optional)

- Extra penalty for false positives during idle segments
- `idle_mask`: frames where ALL gameplay controls are 0

```python
idle_mask = (action_targets.sum(dim=1) == 0) & gameplay_mask
if idle_mask.any():
    idle_probs = torch.sigmoid(action_logits[idle_mask])
    idle_loss = idle_probs.mean()  # penalize any activation during idle
else:
    idle_loss = 0.0
```

### Loss Weights

| Weight | Initial | Notes |
|---|---|---|
| w_action | 1.0 | Primary objective |
| w_look | 0.5 | Lower to prevent dominating during low-activity periods |
| w_hotbar | 0.3 | Sparse events, CE is already well-scaled |
| w_cursor | 0.5 | Important for inventory usability |
| w_inv | 0.3 | Sparse clicks |
| w_idle | 0.3 | Prevents false positives |

---

## Action History (Teacher Forcing)

During training, feed ground truth control vectors as action history:

```python
# history: (batch, 5, 28) — last 5 ground-truth frames
# Apply dropout: randomly zero entire frames
dropout_mask = torch.rand(batch, 5) > action_history_dropout  # 0.3
history = history * dropout_mask.unsqueeze(2)  # (batch, 5, 28)
history_flat = history.reshape(batch, 140)  # flatten
```

During inference, use model's own previous outputs.

---

## Evaluation Metrics

### Per-Head Metrics

| Head | Metric | Computation |
|---|---|---|
| Action | Per-control F1 + Avg F1 | Compare sigmoid(logits) > threshold vs target |
| Look | MSE + direction accuracy | Mean squared error; sign match for yaw/pitch |
| Hotbar | Accuracy | Top-1 accuracy on changed frames |
| Cursor | MSE + pixel error | Mean squared error; estimated pixel distance |
| InvClick | Per-control F1 | Compare sigmoid(logits) > threshold vs target |

### Idle Accuracy

- What % of idle frames (all gameplay controls = 0) does the model correctly
  predict as idle (all outputs below threshold)?
- Target: >95%

### Combined Score

```python
score = (0.4 * avg_action_f1
       + 0.2 * (1.0 - look_mse)
       + 0.1 * hotbar_accuracy
       + 0.1 * (1.0 - cursor_mse)
       + 0.1 * avg_inv_f1
       + 0.1 * idle_accuracy)
```

This is used for early stopping and model selection.

---

## Threshold Optimization

After training, optimize per-control thresholds on the validation set:

```python
for control_idx in range(num_binary_controls):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = sigmoid(logits[:, control_idx]) > thresh
        f1 = compute_f1(preds, targets[:, control_idx])
        # Also check false positive rate on idle frames
        fpr = compute_fpr_idle(preds, idle_mask)
        # Balance F1 and low FPR
        score = f1 - 0.5 * fpr
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    thresholds[control_name] = best_thresh
```

Save optimized thresholds in `control_config.json`.

---

## Training Output

### Console

```
  Ep  TrLoss  Act    Look  Hbar   Curs   Inv    Idle   VlLoss  VlF1  Score    LR
----------------------------------------------------------------------------------
   1  1.234  0.892  0.003  0.100  0.050  0.080  0.500  1.345  0.150  0.300  1.0e-4 *
   2  0.800  0.702  0.001  0.080  0.040  0.060  0.700  0.811  0.200  0.450  2.0e-4 *
  ...
```

### Saved Files

- `models/control_policy.pt` — Best model weights (by combined score)
- `models/control_config.json` — Config, thresholds, normalization stats
- `models/training_log.json` — Per-epoch metrics for analysis

### Config Format

```json
{
  "version": 2,
  "input_dim": 671,
  "d_model": 256,
  "nhead": 8,
  "num_layers": 6,
  "dim_feedforward": 512,
  "dropout": 0.3,
  "max_seq_len": 30,
  "num_action_controls": 12,
  "num_look_controls": 2,
  "num_hotbar_slots": 9,
  "num_cursor_controls": 2,
  "num_inv_controls": 3,
  "control_names": ["move_forward", "move_backward", "..."],
  "thresholds": {"move_forward": 0.35, "...": "..."},
  "look_deadzone": 0.08,
  "look_smoothing_alpha": 0.35,
  "action_history_length": 5,
  "action_history_dropout": 0.3,
  "game_state_dim": 46,
  "feature_mean": ["..."],
  "feature_std": ["..."],
  "training": {
    "epochs_trained": 150,
    "best_combined_score": 0.82,
    "per_head_metrics": {}
  }
}
```

---

## Curriculum Learning (Phase 10)

Optional staged training approach:

1. **Stage 1**: Movement only (forward, backward, strafe, sprint, sneak, jump, look)
   - Freeze other heads, train with movement-heavy data
2. **Stage 2**: Add combat (attack, use_item, drop, swap)
   - Unfreeze combat controls, fine-tune on combat scenarios
3. **Stage 3**: Add inventory (cursor, clicks, hotbar)
   - Unfreeze all heads, fine-tune on inventory sessions

This can help if training all 28 controls at once is unstable.
