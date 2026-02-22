# V2 Instance Prompts

Paste each prompt into a separate Claude Code instance.
Open each in the `C:\Users\forke\Documents\CVcoolz` directory.
All 4 can run in parallel — they share no code dependencies, only the contracts in `plans/v2_coordination.md`.

---

## Instance A: MODEL (control_model.py)

```
# Task: Rewrite control_model.py for V2 (Instance A — MODEL)

You are Instance A of a 4-instance parallel implementation. Your job is to rewrite `AndroidCamera/client/control_model.py` with the V2 architecture.

## FIRST: Read the coordination contracts
Read `AndroidCamera/plans/v2_coordination.md` — it contains the exact interfaces all instances must follow. Pay special attention to Contract 4 (model architecture & forward signature).

## THEN: Read the current file
Read `AndroidCamera/client/control_model.py` to understand the v1 implementation.

## What to Build

Rewrite `ControlTransformerV2` with these specs:

### Architecture
- Input projection: Linear(671 → 256)
- Learnable positional encoding: max 30 positions
- Transformer encoder: 6 layers, 8 heads, d_ff=512, dropout=0.3
- Exponential temporal pooling with learnable decay (init -2.0)
- 5 output heads (all sharing the 256-dim backbone output):
  1. **Action**: Linear(256→128)→ReLU→Dropout(0.3)→Linear(128→12) → raw logits
  2. **Look**: Linear(256→128)→ReLU→Dropout(0.3)→Linear(128→2) → tanh
  3. **Hotbar**: Linear(256→128)→ReLU→Dropout(0.3)→Linear(128→9) → raw logits
  4. **Cursor**: Linear(256→128)→ReLU→Dropout(0.3)→Linear(128→2) → sigmoid
  5. **InvClick**: Linear(256→128)→ReLU→Dropout(0.3)→Linear(128→3) → raw logits

### Forward signature
```python
def forward(self, x: Tensor, mask: Tensor) -> dict:
    """
    x: (batch, seq_len, 671)
    mask: (batch, seq_len) — 1.0 real, 0.0 padding
    Returns: {
        'action_logits': (batch, 12),
        'look': (batch, 2),
        'hotbar_logits': (batch, 9),
        'cursor': (batch, 2),
        'inv_click_logits': (batch, 3),
    }
    """
```

### Requirements
- Keep the v1 `ControlTransformer` class intact (rename nothing, don't delete it)
- Add `ControlTransformerV2` as a NEW class below it
- The exponential temporal pooling should weight recent frames higher (same concept as v1 but with learnable decay parameter)
- Use `nn.TransformerEncoder` with batch_first=True
- Generate proper causal or padding mask from the input mask
- Include a `@staticmethod config_from_dict(d)` that builds from the JSON config (Contract 7)
- Include `param_count()` method returning total trainable params
- Target ~3.5M parameters — verify this

### Deliverables
1. Updated `AndroidCamera/client/control_model.py` with ControlTransformerV2
2. Write status to `AndroidCamera/plans/status_MODEL.md`
```

---

## Instance B: DATA (control_dataset.py + control_recorder.py)

```
# Task: Rewrite control_dataset.py and control_recorder.py for V2 (Instance B — DATA)

You are Instance B of a 4-instance parallel implementation. Your job is to rewrite the data pipeline: `AndroidCamera/client/control_dataset.py` and `AndroidCamera/client/control_recorder.py`.

## FIRST: Read the coordination contracts
Read `AndroidCamera/plans/v2_coordination.md` — it contains the exact interfaces all instances must follow. Pay special attention to Contracts 1-3, 5-6, and 8.

## THEN: Read current files
Read `AndroidCamera/client/control_dataset.py` and `AndroidCamera/client/control_recorder.py`.
Also read `AndroidCamera/plans/data_pipeline.md` for detailed specs.

## Part 1: control_recorder.py — In-Game Capture

### What changes
The v1 recorder uses `pynput` to capture keyboard/mouse. V2 must capture controls from the Minecraft game state via MCCTP instead.

### Key changes
1. **Remove `ControlInputCapture` class** (pynput-based) — replace with `MCCTPControlCapture`:
   - Connects to MCCTP via `SyncMCCTPClient`
   - Each tick, reads resolved input state from `client.state.to_control_dict()`
   - Movement: `movementForward > 0` → forward, `< 0` → backward, `movementSideways` → strafe
   - Jump/sneak/sprint: from playerInput flags
   - Look: `currentYaw - prevYaw`, `currentPitch - prevPitch` per tick, normalized to [-1, 1]
   - Attack/use: from options key press state
   - Hotbar: detect `selectedSlot` changes, encode as one-hot in indices 14-22
   - Drop/swap/inventory: from options key press state
   - When `screen_open`: capture cursor position (normalized 0-1) and mouse button state
   - Produces a 28-dim control vector per frame

2. **Remove `HeadLookTracker`** — look now comes from game state deltas, not pose estimation

3. **Update `encode_game_state_v2`**: Import from control_policy.py (Contract 8). For now, define a stub version locally that maps MCCTP state dict to 46-dim vector — the real one will be in control_policy.py but you need it for recording.

4. **Update save format**: Use Contract 6 (v2 .npz with version=2, 28-dim controls, 46-dim game_state)

5. **Keep the UI controls** (R to record, Q to quit, overlay drawing) but update the overlay to show all 28 controls grouped by head

6. **Keep session management** (countdown, discard, file naming)

### MCCTP API Reference
```python
from mcctp import SyncMCCTPClient, Actions
client = SyncMCCTPClient(host="localhost", port=8080)
client.connect()
state = client.state                    # GameState object
d = state.to_control_dict()             # dict with all game state fields
```

The `to_control_dict()` returns a dict with keys matching the game state fields. Check `AndroidCamera/plans/architecture.md` for the full field list.

## Part 2: control_dataset.py — V2 Dataset

### What changes
1. **Dimensions**: INPUT_DIM=671, NUM_CONTROLS=28, GAME_STATE_DIM=46, ACTION_HISTORY_DIM=140
2. **Version detection**: Auto-detect v1 vs v2 recordings on load. For v1 files, pad controls from 10→28 and game_state from 16/24→46 with zeros
3. **Action history**: K=5, each frame is 28-dim → 140-dim history. Teacher forcing with 30% dropout during training
4. **Mode masks**: Generate per-window masks:
   - `gameplay_mask`: game_state[38] == 0 at target frame
   - `screen_open_mask`: game_state[38] > 0 at target frame
   - `hotbar_change_mask`: hotbar slot differs from previous frame
   - `look_active_mask`: look magnitude > 0.08 at target frame
5. **Window stride**: Change from 5 to 1
6. **Return format**: `__getitem__` returns `(features, mask, target, mode_masks)` where mode_masks is a dict:
   ```python
   {
       'gameplay': bool,
       'screen_open': bool,
       'hotbar_changed': bool,
       'look_active': bool,
   }
   ```
7. **Keep all existing augmentation** (temporal crop, time warp, noise, action history dropout) and extend to work with 28-dim controls

### Deliverables
1. Updated `AndroidCamera/client/control_recorder.py`
2. Updated `AndroidCamera/client/control_dataset.py`
3. Write status to `AndroidCamera/plans/status_DATA.md`
```

---

## Instance C: TRAIN (train_controls.py)

```
# Task: Rewrite train_controls.py for V2 (Instance C — TRAIN)

You are Instance C of a 4-instance parallel implementation. Your job is to rewrite `AndroidCamera/client/train_controls.py` for the V2 multi-head model.

## FIRST: Read the coordination contracts
Read `AndroidCamera/plans/v2_coordination.md` — it contains the exact interfaces all instances must follow. Pay special attention to Contracts 1, 2, 4, and 7.

## THEN: Read current files
Read `AndroidCamera/client/train_controls.py`.
Also read `AndroidCamera/plans/training.md` for detailed loss/metric specs.

## What to Build

Rewrite the training pipeline for the 5-head V2 model.

### Model Config
- Import `ControlTransformerV2` from `control_model` (the class will exist — Instance A is building it)
- D_MODEL=256, NHEAD=8, NUM_LAYERS=6, DIM_FEEDFORWARD=512, DROPOUT=0.3
- Expect forward() to return a dict with keys: action_logits, look, hotbar_logits, cursor, inv_click_logits

### Dataset Contract
- Import from `control_dataset` (Instance B is building it)
- `__getitem__` returns `(features, mask, target, mode_masks)`
  - features: (30, 671)
  - mask: (30,)
  - target: (28,)
  - mode_masks: dict with keys 'gameplay', 'screen_open', 'hotbar_changed', 'look_active' (all bool)
- You'll need to collate mode_masks in a custom collate function into batch tensors

### Loss Function (Mode-Aware Masked)
```python
def compute_v2_loss(outputs, targets, mode_masks, pos_weights_action, pos_weights_inv):
    """
    outputs: dict from model forward
    targets: (batch, 28)
    mode_masks: dict of (batch,) bool tensors

    Returns: (total_loss, loss_dict) where loss_dict has per-head losses
    """
    # 1. Action loss: BCE on indices [0:9, 11, 12, 13] — ONLY on gameplay frames
    #    pos_weights_action: (12,) computed from training data
    #    Mask: gameplay_mask

    # 2. Look loss: SmoothL1 on indices [9, 10] — ONLY on gameplay AND look_active frames
    #    Mask: gameplay_mask & look_active_mask

    # 3. Hotbar loss: CrossEntropy on indices [14:23] — ONLY on hotbar_changed frames
    #    Convert one-hot target to class index, handle no-change as ignore_index
    #    Mask: hotbar_change_mask

    # 4. Cursor loss: MSE on indices [23, 24] — ONLY on screen_open frames
    #    Mask: screen_open_mask

    # 5. InvClick loss: BCE on indices [25, 26, 27] — ONLY on screen_open frames
    #    pos_weights_inv: (3,)
    #    Mask: screen_open_mask

    # 6. Idle penalty: on gameplay frames where ALL action indices are 0,
    #    penalize any sigmoid(action_logits) activation

    # Weights: w_action=1.0, w_look=0.5, w_hotbar=0.3, w_cursor=0.5, w_inv=0.3, w_idle=0.3
    # Handle empty masks gracefully (if no frames match a mask, that loss = 0)
```

### Training Loop
- BATCH_SIZE=64, EPOCHS=150, LR=5e-4, warmup 5 epochs → cosine to 1e-6
- WEIGHT_DECAY=1e-4, GRADIENT_CLIP=1.0, EARLY_STOP_PATIENCE=25
- Mixed precision (torch.amp)
- Custom collate function to batch mode_masks

### Evaluation Metrics (per epoch)
- **Per-action F1** (12 actions) + average F1
- **Look MSE** + directional accuracy (sign match %)
- **Hotbar accuracy** (on hotbar_changed frames only)
- **Cursor MSE** (on screen_open frames only)
- **InvClick F1** (on screen_open frames only)
- **Idle accuracy** (% of all-zero gameplay frames predicted as idle)
- **Combined score**: 0.4*action_F1 + 0.2*look_score + 0.1*hotbar + 0.1*cursor + 0.1*inv + 0.1*idle

### Threshold Optimization
- After training, optimize per-action thresholds on validation set (same approach as v1)
- Also optimize hotbar_min_confidence threshold
- Also optimize inv_click thresholds

### Output Files
- `models/control_policy_v2.pt` — best model weights
- `models/control_config_v2.json` — architecture + thresholds (Contract 7 format)
- `models/training_log_v2.json` — per-epoch metrics for all 5 heads

### Deliverables
1. Updated `AndroidCamera/client/train_controls.py`
2. Write status to `AndroidCamera/plans/status_TRAIN.md`
```

---

## Instance D: INFER (control_policy.py + control_bridge.py)

```
# Task: Rewrite control_policy.py and control_bridge.py for V2 (Instance D — INFER)

You are Instance D of a 4-instance parallel implementation. Your job is to rewrite the inference and bridge for V2's 28-control, 5-head output.

## FIRST: Read the coordination contracts
Read `AndroidCamera/plans/v2_coordination.md` — it contains the exact interfaces all instances must follow. Pay special attention to Contracts 2, 3, 5, 7, and 8.

## THEN: Read current files
Read `AndroidCamera/client/control_policy.py` and `AndroidCamera/client/control_bridge.py`.
Also read `AndroidCamera/plans/integration.md` for detailed specs.

## Part 1: control_policy.py — V2 Inference Wrapper

### What changes

1. **`encode_game_state_v2(mcctp_state_dict) → (46,)`**
   This is THE canonical implementation — other files import it from here.
   Map the MCCTP state dict to the 46-dim vector per Contract 3:
   - Item context (11 dims): classify mainhand item (can_melee/ranged/place/eat/tool/empty/other), classify offhand (shield/food/totem/empty)
   - Vitals (3): health/20, hunger/20, armor/20
   - Movement (10): on_ground, in_water, swimming, flying, climbing, on_fire, sprinting, sneaking, fall_distance (clamp 0-20 then /20), velocity_y (clamp -3 to 3 then /3)
   - Combat (5): attack_cooldown [0,1], is_using_item, is_blocking, item_use_progress [0,1], recently_hurt
   - Crosshair (4): target_is_entity, target_is_block, target_entity_hostile, target_distance (clamp 0-6 then /6)
   - Threats (3): nearest_hostile_dist (clamp 0-32 then /32), nearest_hostile_yaw /180 [-1,1], hostile_count (clamp 0-10 then /10)
   - Environment (3): time_of_day /24000 [0,1], game_mode_survival (bool), screen_open_type (0=none, 0.33=inventory, 0.66=chest, 1.0=other)
   - Status effects (5): bools
   - Extra (2): current_hotbar_slot /8 [0,1], horizontal_collision (bool)

2. **`ControlOutputV2` dataclass** — implement exactly per Contract 5

3. **`ControlPolicyV2` class**:
   - Loads `ControlTransformerV2` from `control_policy_v2.pt` + `control_config_v2.json`
   - Rolling buffer: deque(maxlen=30) of 260-dim raw features
   - Game state buffer: deque(maxlen=30) of 46-dim game states
   - Action history: deque(maxlen=5) of 28-dim control vectors
   - `push_frame(pose, hands, game_state_dict)` — extracts 260 raw features, encodes 46-dim game state, appends to buffers
   - `predict() → ControlOutputV2`:
     1. Assemble 671-dim features (260 raw + 225 velocity + 46 game_state + 140 action_history)
     2. Pad/mask to (1, 30, 671)
     3. Forward through model → dict of 5 head outputs
     4. **Mode switch on screen_open flag (game_state[38])**:
        - If gameplay: apply action thresholds, look post-processing, hotbar argmax
        - If screen_open: apply cursor smoothing, inv_click thresholds
        - `open_inventory` (index 13) is always active regardless of mode
     5. Post-process:
        - Actions: threshold + hysteresis (margin 0.05)
        - Look: deadzone (0.08), EMA (α=0.35), rate limit (0.3)
        - Hotbar: argmax with min confidence (0.3), only output if different from current slot
        - Cursor: EMA (α=0.5)
        - InvClick: threshold + hysteresis
     6. Update action history with output vector
     7. Return ControlOutputV2

4. **Keep `encode_game_state` (v1) and `ControlPolicy` (v1)** — don't delete, just add v2 alongside

## Part 2: control_bridge.py — V2 MCCTP Bridge

### What changes

1. **`ControlBridgeV2` class** — handles all 28 controls:
   - **Held actions** (start/stop on transitions): forward, backward, strafe_left, strafe_right, sprint, sneak, use_item, inv_shift_held
   - **Pulse actions** (fire on rising edge, 0.3s cooldown): jump, attack, drop_item, swap_offhand, open_inventory, inv_left_click, inv_right_click
   - **Analog look** (every frame if magnitude > 0.01): yaw, pitch with sensitivity scaling (5.0°, 3.0°)
   - **Hotbar** (on slot change): send hotbar select command
   - **Cursor** (when screen_open, every frame): send cursor position

2. **Mode-aware sending**:
   - `update(output: ControlOutputV2)`:
     - Check `output.screen_open`
     - If gameplay mode: send actions + look + hotbar. Ignore cursor/inv_clicks
     - If screen_open mode: send cursor + inv_clicks. Ignore movement/look
     - Always allow: open_inventory (transitions between modes)

3. **Safety**:
   - Auto-release held actions after 10s
   - `release_all()` emergency stop
   - Edge detection prevents duplicate commands
   - Disconnect releases everything

4. **MCCTP API usage**:
   ```python
   from mcctp import SyncMCCTPClient, Actions
   client.perform(Actions.move(forward=True))
   client.perform(Actions.look(yaw=0.5, pitch=0.0))
   client.perform(Actions.jump())
   client.perform(Actions.hotbar(slot=3))
   client.perform(Actions.cursor(x=0.5, y=0.3))
   client.perform(Actions.click(button='left'))
   client.perform(Actions.drop())
   client.perform(Actions.swap_offhand())
   client.perform(Actions.open_inventory())
   # etc — check the mcctp package for exact API
   ```

5. **Keep `ControlBridge` (v1)** — don't delete

### Deliverables
1. Updated `AndroidCamera/client/control_policy.py`
2. Updated `AndroidCamera/client/control_bridge.py`
3. Write status to `AndroidCamera/plans/status_INFER.md`
```
