# V2 Implementation Coordination

## Instance Assignment

| Instance | Codename  | Files                                      | Phase |
|----------|-----------|-------------------------------------------|-------|
| A        | MODEL     | control_model.py                          | 7     |
| B        | DATA      | control_dataset.py, control_recorder.py   | 6+8d  |
| C        | TRAIN     | train_controls.py                         | 8     |
| D        | INFER     | control_policy.py, control_bridge.py      | 9     |

## Status Tracking

Each instance writes to `plans/status_<codename>.md` on completion.
Format:
```
# Status: MODEL (or DATA, TRAIN, INFER)
## Completed
- [ ] list of completed items
## Interface Changes
- any deviations from the contracts below
## Notes
- anything the next instance needs to know
```

## Shared Contracts (ALL INSTANCES MUST FOLLOW THESE EXACTLY)

### Contract 1: Dimension Constants

```python
# Shared constants — every file must agree on these
RAW_FEATURE_DIM = 260       # pose(99) + hands(126) + presence(2) + visibility(33)
VELOCITY_DIM = 225          # frame-to-frame deltas of first 225 raw features
GAME_STATE_DIM = 46         # v2 game state
ACTION_HISTORY_K = 5        # number of previous frames
NUM_CONTROLS = 28           # v2 control vector
ACTION_HISTORY_DIM = ACTION_HISTORY_K * NUM_CONTROLS  # 140
INPUT_DIM = RAW_FEATURE_DIM + VELOCITY_DIM + GAME_STATE_DIM + ACTION_HISTORY_DIM  # 671

WINDOW_SIZE = 30            # temporal window
WINDOW_STRIDE = 1           # v2 uses stride 1 (was 5 in v1)
```

### Contract 2: Control Vector Layout (28 dims)

```python
# Index mapping
CTRL_MOVE_FORWARD    = 0    # bool
CTRL_MOVE_BACKWARD   = 1    # bool
CTRL_STRAFE_LEFT     = 2    # bool
CTRL_STRAFE_RIGHT    = 3    # bool
CTRL_SPRINT          = 4    # bool
CTRL_SNEAK           = 5    # bool
CTRL_JUMP            = 6    # bool (pulse)
CTRL_ATTACK          = 7    # bool (pulse)
CTRL_USE_ITEM        = 8    # bool
CTRL_LOOK_YAW        = 9    # float [-1, 1]
CTRL_LOOK_PITCH      = 10   # float [-1, 1]
CTRL_DROP_ITEM       = 11   # bool (pulse)
CTRL_SWAP_OFFHAND    = 12   # bool (pulse)
CTRL_OPEN_INVENTORY  = 13   # bool (pulse)
CTRL_HOTBAR_START    = 14   # indices 14-22, one-hot (9 slots)
CTRL_HOTBAR_END      = 22
CTRL_CURSOR_X        = 23   # float [0, 1]
CTRL_CURSOR_Y        = 24   # float [0, 1]
CTRL_INV_LEFT_CLICK  = 25   # bool (pulse)
CTRL_INV_RIGHT_CLICK = 26   # bool (pulse)
CTRL_INV_SHIFT_HELD  = 27   # bool

# Head groupings
ACTION_INDICES = list(range(0, 9)) + [11, 12, 13]  # 12 binary actions
LOOK_INDICES = [9, 10]                               # 2 analog look
HOTBAR_INDICES = list(range(14, 23))                  # 9 one-hot
CURSOR_INDICES = [23, 24]                             # 2 analog cursor
INV_CLICK_INDICES = [25, 26, 27]                      # 3 binary inv
```

### Contract 3: Game State Layout (46 dims)

```python
# Category breakdown with indices
GS_ITEM_CONTEXT   = slice(0, 11)    # can_melee, can_ranged, can_place, can_eat, is_tool, is_empty, is_other, offhand_shield/food/totem/empty
GS_VITALS         = slice(11, 14)   # health, hunger, armor (all /20)
GS_MOVEMENT       = slice(14, 24)   # on_ground, in_water, swimming, flying, climbing, on_fire, sprinting, sneaking, fall_distance, velocity_y
GS_COMBAT         = slice(24, 29)   # attack_cooldown, is_using_item, is_blocking, item_use_progress, recently_hurt
GS_CROSSHAIR      = slice(29, 33)   # target_is_entity, target_is_block, target_entity_hostile, target_distance
GS_THREATS         = slice(33, 36)   # nearest_hostile_dist, nearest_hostile_yaw, hostile_count
GS_ENVIRONMENT    = slice(36, 39)   # time_of_day, game_mode_survival, screen_open_type
GS_STATUS_EFFECTS = slice(39, 44)   # speed, slowness, strength, taking_dot, fire_resist
GS_EXTRA          = slice(44, 46)   # current_hotbar_slot, horizontal_collision

# Key flag for mode switching
GS_SCREEN_OPEN_IDX = 38             # game_state[38] > 0 means a screen/GUI is open
```

### Contract 4: Model Architecture & Forward Signature

```python
class ControlTransformerV2(nn.Module):
    """
    Config dict keys (for JSON serialization):
        input_dim: 671
        d_model: 256
        nhead: 8
        num_layers: 6
        dim_feedforward: 512
        dropout: 0.3
        max_seq_len: 30
        num_action: 12
        num_look: 2
        num_hotbar: 9
        num_cursor: 2
        num_inv_click: 3
    """

    def forward(self, x: Tensor, mask: Tensor) -> dict:
        """
        Args:
            x: (batch, seq_len, 671) input features
            mask: (batch, seq_len) — 1.0 for real frames, 0.0 for padding

        Returns dict with keys:
            'action_logits':  (batch, 12)  — raw logits for BCE
            'look':           (batch, 2)   — tanh output [-1, 1]
            'hotbar_logits':  (batch, 9)   — raw logits for CE
            'cursor':         (batch, 2)   — sigmoid output [0, 1]
            'inv_click_logits': (batch, 3) — raw logits for BCE
        """
```

### Contract 5: ControlOutput Dataclass

```python
@dataclass
class ControlOutputV2:
    # Gameplay actions (binary, thresholded)
    move_forward: bool = False
    move_backward: bool = False
    strafe_left: bool = False
    strafe_right: bool = False
    sprint: bool = False
    sneak: bool = False
    jump: bool = False
    attack: bool = False
    use_item: bool = False

    # Look (analog)
    look_yaw: float = 0.0      # [-1, 1]
    look_pitch: float = 0.0    # [-1, 1]

    # Utility actions (binary, thresholded)
    drop_item: bool = False
    swap_offhand: bool = False
    open_inventory: bool = False

    # Hotbar (one-hot → int or None)
    hotbar_slot: Optional[int] = None   # 0-8, None = no change

    # Cursor (only valid when screen_open)
    cursor_x: float = 0.0     # [0, 1]
    cursor_y: float = 0.0     # [0, 1]

    # Inventory clicks (only valid when screen_open)
    inv_left_click: bool = False
    inv_right_click: bool = False
    inv_shift_held: bool = False

    # Raw model outputs (for debugging/logging)
    raw_action_probs: np.ndarray = None   # (12,)
    raw_look: np.ndarray = None           # (2,)
    raw_hotbar_probs: np.ndarray = None   # (9,)
    raw_cursor: np.ndarray = None         # (2,)
    raw_inv_probs: np.ndarray = None      # (3,)

    # Mode
    screen_open: bool = False

    def to_control_vector(self) -> np.ndarray:
        """Convert back to 28-dim vector (for action history)."""
        v = np.zeros(28, dtype=np.float32)
        v[0] = float(self.move_forward)
        v[1] = float(self.move_backward)
        v[2] = float(self.strafe_left)
        v[3] = float(self.strafe_right)
        v[4] = float(self.sprint)
        v[5] = float(self.sneak)
        v[6] = float(self.jump)
        v[7] = float(self.attack)
        v[8] = float(self.use_item)
        v[9] = self.look_yaw
        v[10] = self.look_pitch
        v[11] = float(self.drop_item)
        v[12] = float(self.swap_offhand)
        v[13] = float(self.open_inventory)
        if self.hotbar_slot is not None:
            v[14 + self.hotbar_slot] = 1.0
        v[23] = self.cursor_x
        v[24] = self.cursor_y
        v[25] = float(self.inv_left_click)
        v[26] = float(self.inv_right_click)
        v[27] = float(self.inv_shift_held)
        return v
```

### Contract 6: Recording Format v2 (.npz)

```python
# Save format
np.savez_compressed(path,
    frames=frames,          # (N, 260) float32 — raw pose+hand features
    controls=controls,      # (N, 28) float32 — control vector
    game_state=game_state,  # (N, 46) float32 — game state
    timestamps=timestamps,  # (N,) float64
    fps=np.float32(fps),
    version=np.int32(2),
)

# Load detection
data = np.load(path, allow_pickle=True)
version = int(data.get('version', 1))
if version == 1:
    # legacy: controls is (N, 10), game_state is (N, 24) or (N, 16)
elif version == 2:
    # new: controls is (N, 28), game_state is (N, 46)
```

### Contract 7: Config JSON Format

```json
{
    "version": 2,
    "model": {
        "input_dim": 671,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 512,
        "dropout": 0.3,
        "max_seq_len": 30,
        "num_action": 12,
        "num_look": 2,
        "num_hotbar": 9,
        "num_cursor": 2,
        "num_inv_click": 3
    },
    "thresholds": {
        "action": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "hotbar_min_confidence": 0.3,
        "inv_click": [0.5, 0.5, 0.5]
    },
    "post_processing": {
        "hysteresis_margin": 0.05,
        "look_deadzone": 0.08,
        "look_ema_alpha": 0.35,
        "look_rate_limit": 0.3,
        "cursor_ema_alpha": 0.5
    },
    "training": {
        "epochs": 150,
        "batch_size": 64,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "combined_score_weights": {
            "action_f1": 0.4,
            "look": 0.2,
            "hotbar": 0.1,
            "cursor": 0.1,
            "inv": 0.1,
            "idle": 0.1
        }
    }
}
```

### Contract 8: encode_game_state_v2 Signature

```python
def encode_game_state_v2(mcctp_state_dict: dict) -> np.ndarray:
    """
    Convert MCCTP GameState.to_control_dict() → (46,) float32.
    Lives in control_policy.py but used by control_recorder.py too.
    Import as: from control_policy import encode_game_state_v2
    """
```

## Backwards Compatibility

- All v2 files must handle loading v1 recordings gracefully
- `control_dataset.py` must auto-detect version and pad v1 data appropriately
- Model checkpoint format: `control_policy_v2.pt` (separate from v1's `control_policy.pt`)
- Config file: `control_config_v2.json`
