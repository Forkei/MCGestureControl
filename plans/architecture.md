# Architecture — Control Policy Model v2

## Overview

A transformer encoder that takes a temporal sequence of body/hand pose + game context
and outputs a 28-dim control vector for Minecraft. Five output heads: action (binary
gameplay), look (analog camera), hotbar (slot selection), cursor (inventory mouse),
and inventory clicks (inventory buttons).

---

## Input Specification

Each frame in the sequence has the following features concatenated:

### Core Pose & Hand Features (260 dims)

| Feature | Dims | Source |
|---|---|---|
| pose_world | 99 | 33 MediaPipe landmarks × 3 (x, y, z) in meters, hip-centered |
| left_hand_3d | 63 | 21 WiLoR landmarks × 3 |
| right_hand_3d | 63 | 21 WiLoR landmarks × 3 |
| left_hand_present | 1 | Binary: is left hand detected? |
| right_hand_present | 1 | Binary: is right hand detected? |
| pose_visibility | 33 | Per-landmark visibility [0, 1] |

### Velocity Features (225 dims)

Frame-to-frame deltas of position coordinates (indices 0:225 of the raw features).
First frame velocity = 0. Computed by existing `append_velocity()`.

### Game State Features (46 dims) — EXPANDED

#### Item Context (11 dims, indices 0-10)

| Feature | Dim | Description |
|---|---|---|
| can_melee | 1 | Held item is sword, axe, trident |
| can_ranged | 1 | Held item is bow, crossbow |
| can_place | 1 | Held item is a placeable block |
| can_eat | 1 | Held item is food |
| is_tool | 1 | Held item is pickaxe, shovel, hoe, shears |
| is_empty | 1 | Nothing in main hand |
| is_other | 1 | Item doesn't fit above categories |
| offhand_is_shield | 1 | Shield in offhand |
| offhand_is_food | 1 | Food in offhand |
| offhand_is_totem | 1 | Totem of Undying in offhand |
| offhand_is_empty | 1 | Nothing in offhand |

#### Player Vitals (3 dims, indices 11-13)

| Feature | Dim | Description |
|---|---|---|
| health | 1 | Normalized [0, 1] (raw / 20) |
| hunger | 1 | Normalized [0, 1] (raw / 20) |
| armor | 1 | Normalized [0, 1] (raw / 20) |

#### Movement State (10 dims, indices 14-23)

| Feature | Dim | Description |
|---|---|---|
| on_ground | 1 | Binary |
| in_water | 1 | Binary |
| is_swimming | 1 | Binary |
| is_flying | 1 | Binary (creative/elytra) |
| is_climbing | 1 | Binary (ladder/vine) |
| on_fire | 1 | Binary |
| is_sprinting | 1 | Binary (game-reported, not our command) |
| is_sneaking | 1 | Binary (game-reported) |
| fall_distance | 1 | Normalized [0, 1] (clamp at 50 blocks) |
| velocity_y | 1 | Normalized [-1, 1] (clamp at ±3 m/tick) |

#### Combat State (5 dims, indices 24-28)

| Feature | Dim | Description |
|---|---|---|
| attack_cooldown | 1 | [0, 1] — 0 = ready, 1 = just swung |
| is_using_item | 1 | Binary (drawing bow, eating, blocking) |
| is_blocking | 1 | Binary (shield raised) |
| item_use_progress | 1 | [0, 1] — how far through use animation |
| recently_hurt | 1 | Binary (hurt in last 10 ticks) |

#### Crosshair Target (4 dims, indices 29-32)

| Feature | Dim | Description |
|---|---|---|
| target_is_entity | 1 | Binary: crosshair on an entity |
| target_is_block | 1 | Binary: crosshair on a block |
| target_entity_hostile | 1 | Binary: targeted entity is hostile |
| target_distance | 1 | [0, 1] normalized (clamp at 6 blocks) |

#### Nearby Threats (3 dims, indices 33-35)

| Feature | Dim | Description |
|---|---|---|
| nearest_hostile_dist | 1 | [0, 1] (clamp at 16 blocks, 0 = closest) |
| nearest_hostile_yaw | 1 | [-1, 1] relative angle (0 = straight ahead) |
| hostile_count | 1 | [0, 1] normalized (clamp at 5) |

#### Environment (3 dims, indices 36-38)

| Feature | Dim | Description |
|---|---|---|
| time_of_day | 1 | [0, 1] (0 = dawn, 0.5 = dusk) |
| game_mode_survival | 1 | Binary (1 = survival, 0 = creative/other) |
| screen_open_type | 1 | [0, 1] (0 = no screen, >0 = some screen open) |

#### Status Effects (5 dims, indices 39-43)

| Feature | Dim | Description |
|---|---|---|
| has_speed | 1 | Binary: speed potion active |
| has_slowness | 1 | Binary: slowness active |
| has_strength | 1 | Binary: strength active |
| taking_dot | 1 | Binary: poison/wither DOT active |
| has_fire_resist | 1 | Binary: fire resistance active |

#### Extra (2 dims, indices 44-45)

| Feature | Dim | Description |
|---|---|---|
| current_hotbar_slot | 1 | [0, 1] normalized (raw / 8) |
| horizontal_collision | 1 | Binary: player hitting a wall |

### Action History (140 dims) — EXPANDED

The model's own output from the previous 5 inference frames, flattened.
28 outputs × 5 frames = 140 dims. Gives the model memory of what it recently did.

During training: use ground truth labels (teacher forcing) with 30% dropout
(randomly zero out entire history entries to prevent over-reliance).

During inference: use the model's own previous outputs.

### Total Input Dimensions

| Component | Dims |
|---|---|
| Pose + hands | 260 |
| Velocity | 225 |
| Game state | 46 |
| Action history | 140 |
| **Total** | **671** |

---

### Feature Normalization

Compute per-feature mean and std from training data. Apply z-score normalization
at inference time. Store normalization stats in the config file.

```python
# During training:
feature_mean = train_features.mean(axis=0)  # (671,)
feature_std = train_features.std(axis=0)    # (671,)
feature_std[feature_std < 1e-6] = 1.0       # avoid division by zero

# Applied at inference:
normalized = (features - feature_mean) / feature_std
```

Binary features (hand_present, game state flags) will have near-zero std — that's
fine, they become approximately zero-centered.

---

## Sequence Length

- **Window: 30 frames** (~1 second at 30fps)
- Each input tensor: (batch, 30, 671)
- Padding mask: (batch, 30) — binary, 1 = real frame

Can increase to 60 frames later if temporal context proves insufficient.

---

## Model Architecture

```
Input (batch, 30, 671)
    │
    ▼
Linear Projection: 671 → d_model (256)
    │
    ▼
Positional Encoding: learnable, max 120 positions
    │
    ▼
Transformer Encoder:
    layers: 6
    heads: 8
    d_feedforward: 512
    dropout: 0.3
    │
    ▼
Weighted Temporal Pooling → (batch, 256)
    │
    ├──→ Action Head:   Linear(256→128)→ReLU→Drop(0.2)→Linear(128→9)  → raw logits
    │
    ├──→ Look Head:     Linear(256→128)→ReLU→Drop(0.2)→Linear(128→2)  → tanh → [-1,1]
    │
    ├──→ Hotbar Head:   Linear(256→128)→ReLU→Drop(0.2)→Linear(128→9)  → raw logits
    │
    ├──→ Cursor Head:   Linear(256→128)→ReLU→Drop(0.2)→Linear(128→2)  → sigmoid → [0,1]
    │
    └──→ InvClick Head: Linear(256→128)→ReLU→Drop(0.2)→Linear(128→3)  → raw logits
```

### Pooling Strategy: Exponential Temporal Pooling

Recent frames matter more than old frames for control output.

```python
weights = torch.exp(torch.linspace(-2.0, 0.0, seq_len))  # e^-2 to e^0
weights = weights * mask  # zero out padding
weights = weights / weights.sum()  # normalize
pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)  # (batch, d_model)
```

### Parameter Count (estimated)

| Component | Params |
|---|---|
| Projection (671→256) | ~172K |
| Positional encoding | ~31K |
| Transformer (6 layers, 256 dim, 8 heads) | ~3.15M |
| Action head (256→128→9) | ~34K |
| Look head (256→128→2) | ~33K |
| Hotbar head (256→128→9) | ~34K |
| Cursor head (256→128→2) | ~33K |
| InvClick head (256→128→3) | ~33K |
| **Total** | **~3.5M** |

Should run <8ms on GPU, <25ms on CPU.

---

## Output Specification

### Binary Gameplay Controls (9 outputs, Action Head)

| Index | Name | Source | Type | Description |
|---|---|---|---|---|
| 0 | move_forward | player.input.movementForward > 0 | held | Walk forward |
| 1 | move_backward | player.input.movementForward < 0 | held | Walk backward |
| 2 | strafe_left | player.input.movementSideways > 0 | held | Strafe left |
| 3 | strafe_right | player.input.movementSideways < 0 | held | Strafe right |
| 4 | sprint | playerInput.sprint() | held | Sprint |
| 5 | sneak | playerInput.sneak() | held | Sneak/crouch |
| 6 | jump | playerInput.jump() | pulse | Jump |
| 7 | attack | attackKey.isPressed() | pulse | Attack/mine |
| 8 | use_item | useKey.isPressed() | held | Use item (place/bow/eat/block) |

### Analog Look (2 outputs, Look Head)

| Index | Name | Range | Description |
|---|---|---|---|
| 9 | look_yaw | [-1, 1] | Horizontal camera: negative=left, positive=right |
| 10 | look_pitch | [-1, 1] | Vertical camera: negative=up, positive=down |

### Utility Actions (3 binary outputs, part of Action Head or separate)

| Index | Name | Type | Description |
|---|---|---|---|
| 11 | drop_item | pulse | Drop held item (Q key equivalent) |
| 12 | swap_offhand | pulse | Swap main/offhand (F key equivalent) |
| 13 | open_inventory | pulse | Toggle inventory screen (E key equivalent) |

### Hotbar Selection (9 outputs, Hotbar Head)

| Index | Name | Type | Description |
|---|---|---|---|
| 14-22 | hotbar_slot_1..9 | one-hot pulse | Select hotbar slot (on change only) |

### Inventory Cursor (2 outputs, Cursor Head)

| Index | Name | Range | Description |
|---|---|---|---|
| 23 | cursor_x | [0, 1] | Horizontal cursor position (left→right) |
| 24 | cursor_y | [0, 1] | Vertical cursor position (top→bottom) |

### Inventory Clicks (3 outputs, InvClick Head)

| Index | Name | Type | Description |
|---|---|---|---|
| 25 | inv_left_click | pulse | Left click in inventory |
| 26 | inv_right_click | pulse | Right click in inventory |
| 27 | inv_shift_held | held | Shift held during inventory click |

**Total: 28 output dimensions**

---

## Loss Function

### Mode-Aware Masked Loss

```
L = w_action * BCE(action_logits, action_targets) * gameplay_mask
  + w_look   * MSE(look_output, look_targets)     * gameplay_mask
  + w_hotbar * CE(hotbar_logits, hotbar_targets)   * hotbar_change_mask
  + w_cursor * MSE(cursor_output, cursor_targets)  * screen_open_mask
  + w_inv    * BCE(inv_logits, inv_targets)        * screen_open_mask
  + w_idle   * idle_penalty
```

**Masks**:
- `gameplay_mask`: 1 when no screen is open (normal gameplay)
- `screen_open_mask`: 1 when inventory/crafting screen is open
- `hotbar_change_mask`: 1 only on frames where hotbar slot changed
- `idle_mask`: 1 on frames explicitly labeled as idle

### Action Loss (BCE with logits)

- BCEWithLogitsLoss with per-control pos_weight
- Weight = total_frames / (2 × positive_frames), capped at 50.0
- Handles extreme imbalance (jump, attack, drop are rare)

### Look Loss (MSE)

- Only on frames where user was actively looking (magnitude > deadzone)
- Can use Smooth L1 (Huber) for outlier robustness

### Hotbar Loss (Cross-Entropy)

- Only on frames where selected slot changed
- 9-class classification (or "no change" as 10th class)

### Cursor Loss (MSE)

- Only when screen is open
- Normalized [0,1] coordinates

### InvClick Loss (BCE)

- Only when screen is open
- Per-control pos_weight like action loss

### Loss Weights

Start with: `w_action=1.0, w_look=0.5, w_hotbar=0.3, w_cursor=0.5, w_inv=0.3, w_idle=0.3`

---

## Design Rationale

### Why 5 Heads?

Each output type has different characteristics:
- **Action**: Binary held/pulse states → BCE loss
- **Look**: Continuous smooth values → MSE loss with deadzone
- **Hotbar**: Categorical one-of-9 selection → CE loss, very sparse
- **Cursor**: Continuous 2D position → MSE loss, only when screen open
- **InvClick**: Binary clicks → BCE loss, only when screen open

Sharing the transformer backbone means they benefit from the same temporal features.

### Why In-Game Control Capture?

Reading `player.input.movementForward` instead of pynput keyboard events:
- Keybind independent (works with any key mapping)
- Sensitivity independent (look deltas are in game units, not mouse DPI)
- Captures mod interactions (sprint toggling, inventory shortcuts)
- No platform-specific keyboard hooking

### Why ~3.5M Parameters?

- v1 was ~755K with 10 outputs and 16-dim game state → underpowered
- 28 outputs with 46-dim game state and mode switching needs more capacity
- 6 transformer layers can learn richer temporal patterns
- Still real-time on consumer GPU (<8ms inference)
- Comparable to other behavioral cloning models in literature

### Why Not End-to-End from Video?

- Pose/hand landmarks already extracted (~671 dims vs ~300K pixels)
- Can train on modest hardware
- More interpretable (can inspect which landmarks drive which controls)
- Faster inference (no vision backbone)

---

## Future Extensions

1. **Recurrent layer**: Add GRU after transformer for long-term memory
2. **Larger window**: 60-90 frames for longer motion patterns
3. **Analog movement**: Replace binary forward with continuous [0, 1] for walk speed
4. **Causal transformer**: GPT-style with KV cache for streaming inference
5. **Online learning**: Update model weights during gameplay from user corrections
6. **Multi-task**: Train on multiple players' demonstrations
7. **Reinforcement fine-tuning**: Use game rewards to improve beyond human demos
