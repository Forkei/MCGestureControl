# Data Pipeline v2

## Overview

The data pipeline captures human demonstrations of Minecraft gameplay through body
gestures. The camera records pose/hand landmarks while the MCCTP mod captures the
player's resolved inputs and game state. This produces (observation, action) pairs
for behavioral cloning.

---

## Recording Tool: `control_recorder.py`

### v2 Changes (from v1)

| Aspect | v1 | v2 |
|---|---|---|
| Action capture | pynput (keyboard/mouse) | In-game resolved inputs from mod |
| Game state | 16 dims (basic) | 46 dims (full) |
| Controls | 10 dims (8 binary + 2 analog) | 28 dims (see below) |
| Keybind dependent | Yes | No — reads game state directly |
| Sensitivity dependent | Yes (mouse DPI) | No — yaw/pitch in game units |
| Inventory support | None | Cursor position + clicks + shift |

### In-Game Control Capture

Instead of hooking keyboard/mouse with pynput, the recorder reads the mod's broadcast
of resolved game input state each tick:

```python
# From MCCTP state broadcast:
state = mcctp_client.state
inputs = state.resolved_inputs  # New field from mod

controls = np.zeros(28, dtype=np.float32)

# Movement (from player.input)
controls[0] = float(inputs['movement_forward'] > 0)   # move_forward
controls[1] = float(inputs['movement_forward'] < 0)   # move_backward
controls[2] = float(inputs['movement_sideways'] > 0)   # strafe_left
controls[3] = float(inputs['movement_sideways'] < 0)   # strafe_right
controls[4] = float(inputs['sprint'])                   # sprint
controls[5] = float(inputs['sneak'])                    # sneak
controls[6] = float(inputs['jump'])                     # jump

# Combat
controls[7] = float(inputs['attack'])                   # attack
controls[8] = float(inputs['use_item'])                 # use_item

# Look (yaw/pitch deltas in game units)
controls[9] = inputs['yaw_delta']                       # look_yaw
controls[10] = inputs['pitch_delta']                    # look_pitch

# Utility
controls[11] = float(inputs['drop'])                    # drop_item
controls[12] = float(inputs['swap_offhand'])            # swap_offhand
controls[13] = float(inputs['open_inventory'])          # open_inventory

# Hotbar (one-hot, only on change)
current_slot = inputs['selected_slot']
if current_slot != prev_slot:
    controls[14 + current_slot] = 1.0
    prev_slot = current_slot

# Inventory (only when screen open)
if inputs['screen_open']:
    controls[23] = inputs['cursor_x']                   # normalized [0,1]
    controls[24] = inputs['cursor_y']                   # normalized [0,1]
    controls[25] = float(inputs['mouse_left'])          # inv_left_click
    controls[26] = float(inputs['mouse_right'])         # inv_right_click
    controls[27] = float(inputs['shift_held'])          # inv_shift_held
```

### Game State Encoding (46 dims)

```python
def encode_game_state(state_dict: dict) -> np.ndarray:
    """Convert MCCTP state dict to 46-dim feature vector."""
    gs = np.zeros(46, dtype=np.float32)

    # Item context (0-10)
    item = state_dict.get('held_item', {})
    gs[0] = float(item.get('can_melee', False))
    gs[1] = float(item.get('can_ranged', False))
    gs[2] = float(item.get('can_place', False))
    gs[3] = float(item.get('can_eat', False))
    gs[4] = float(item.get('is_tool', False))
    gs[5] = float(item.get('is_empty', True))
    gs[6] = float(not any(gs[0:6]))  # is_other
    offhand = state_dict.get('offhand_item', {})
    gs[7] = float(offhand.get('is_shield', False))
    gs[8] = float(offhand.get('is_food', False))
    gs[9] = float(offhand.get('is_totem', False))
    gs[10] = float(not any(gs[7:10]))  # offhand_is_empty

    # Player vitals (11-13)
    gs[11] = state_dict.get('health', 20) / 20.0
    gs[12] = state_dict.get('hunger', 20) / 20.0
    gs[13] = state_dict.get('armor', 0) / 20.0

    # Movement (14-23)
    gs[14] = float(state_dict.get('on_ground', True))
    gs[15] = float(state_dict.get('in_water', False))
    gs[16] = float(state_dict.get('is_swimming', False))
    gs[17] = float(state_dict.get('is_flying', False))
    gs[18] = float(state_dict.get('is_climbing', False))
    gs[19] = float(state_dict.get('on_fire', False))
    gs[20] = float(state_dict.get('is_sprinting', False))
    gs[21] = float(state_dict.get('is_sneaking', False))
    gs[22] = min(state_dict.get('fall_distance', 0) / 50.0, 1.0)
    gs[23] = max(-1.0, min(1.0, state_dict.get('velocity_y', 0) / 3.0))

    # Combat (24-28)
    gs[24] = state_dict.get('attack_cooldown', 0.0)
    gs[25] = float(state_dict.get('is_using_item', False))
    gs[26] = float(state_dict.get('is_blocking', False))
    gs[27] = state_dict.get('item_use_progress', 0.0)
    gs[28] = float(state_dict.get('recently_hurt', False))

    # Crosshair (29-32)
    gs[29] = float(state_dict.get('target_is_entity', False))
    gs[30] = float(state_dict.get('target_is_block', False))
    gs[31] = float(state_dict.get('target_entity_hostile', False))
    gs[32] = min(state_dict.get('target_distance', 6.0) / 6.0, 1.0)

    # Nearby threats (33-35)
    gs[33] = min(state_dict.get('nearest_hostile_dist', 16.0) / 16.0, 1.0)
    gs[34] = max(-1.0, min(1.0, state_dict.get('nearest_hostile_yaw', 0) / 180.0))
    gs[35] = min(state_dict.get('hostile_count', 0) / 5.0, 1.0)

    # Environment (36-38)
    gs[36] = state_dict.get('time_of_day', 0) / 24000.0
    gs[37] = float(state_dict.get('game_mode', 0) == 0)  # survival
    gs[38] = float(state_dict.get('screen_open', False))

    # Status effects (39-43)
    effects = state_dict.get('status_effects', {})
    gs[39] = float('speed' in effects)
    gs[40] = float('slowness' in effects)
    gs[41] = float('strength' in effects)
    gs[42] = float('poison' in effects or 'wither' in effects)
    gs[43] = float('fire_resistance' in effects)

    # Extra (44-45)
    gs[44] = state_dict.get('selected_slot', 0) / 8.0
    gs[45] = float(state_dict.get('horizontal_collision', False))

    return gs
```

### Recording Format (.npz)

```python
np.savez_compressed(path,
    frames=frames,          # (N, 260) - pose + hand features
    controls=controls,      # (N, 28)  - 28-dim control vector
    game_state=game_state,  # (N, 46)  - 46-dim game state
    timestamps=timestamps,  # (N,)     - frame timestamps (seconds)
    fps=fps,                # float    - recording FPS
    version=2,              # int      - format version
)
```

### CLI

```bash
# Record with game state from MCCTP
python client/control_recorder.py <phone_ip> --game --game-port 8080

# Record without game (game state = zeros, controls from keyboard fallback)
python client/control_recorder.py <phone_ip>
```

### Recording Session Workflow

1. Start Minecraft with MCCTP mod loaded
2. Start recorder with `--game` flag
3. Stand in front of camera, visible to pose tracker
4. Play Minecraft normally — move, fight, build, use inventory
5. Press R to start/stop recording segments
6. Press Q to quit and save all segments
7. Review recordings with visualization tool

---

## Dataset: `control_dataset.py`

### v2 Changes

| Aspect | v1 | v2 |
|---|---|---|
| Control dims | 10 | 28 |
| Game state dims | 16 | 46 |
| Action history | 10 × 5 = 50 | 28 × 5 = 140 |
| Total input | 551 | 671 |
| Masks | None | gameplay_mask, screen_open_mask, hotbar_change_mask |

### Windowing

- Window size: 30 frames (~1 second at 30fps)
- Stride: 1 frame (maximum data utilization)
- Target: control vector of the LAST frame in the window
- Padding: shorter sequences padded with zeros + mask

### Feature Assembly

For each window of 30 frames:

```python
# Per frame (260 dims):
raw_features = np.concatenate([
    pose_world,           # (99,) - 33 landmarks × 3
    left_hand_3d,         # (63,) - 21 landmarks × 3
    right_hand_3d,        # (63,) - 21 landmarks × 3
    [left_hand_present],  # (1,)
    [right_hand_present], # (1,)
    pose_visibility,      # (33,)
])

# Velocity (225 dims): frame-to-frame deltas of indices 0:225
velocity = raw_features[1:] - raw_features[:-1]  # first frame = 0

# Game state (46 dims): from encode_game_state()
# Action history (140 dims): last 5 control vectors × 28, flattened

# Total per frame: 260 + 225 + 46 + 140 = 671
```

### Mask Generation

```python
screen_open = game_state[:, 38] > 0.5   # screen_open_type > 0
gameplay_mask = ~screen_open              # normal gameplay
hotbar_change_mask = diff(controls[:, 14:23]).any(axis=1)  # slot changed
```

### Train/Val Split

- Split by recording session (not by window) to prevent data leakage
- 80/20 split
- Stratified by control activation rates if possible

---

## Bootstrapping from v1 Data

Existing recordings (309 sessions from gesture system) can be partially converted:

```python
# v1 format: 10-dim controls
# [forward, strafe_left, strafe_right, sprint, sneak, jump, attack, use_item, look_yaw, look_pitch]

# Map to v2 28-dim:
v2_controls = np.zeros(28)
v2_controls[0] = v1[0]    # forward → move_forward
v2_controls[2] = v1[1]    # strafe_left
v2_controls[3] = v1[2]    # strafe_right
v2_controls[4] = v1[3]    # sprint
v2_controls[5] = v1[4]    # sneak
v2_controls[6] = v1[5]    # jump
v2_controls[7] = v1[6]    # attack
v2_controls[8] = v1[7]    # use_item
v2_controls[9] = v1[8]    # look_yaw
v2_controls[10] = v1[9]   # look_pitch
# Indices 1, 11-27 = 0 (no backward, drop, swap, inventory, hotbar data)
# Game state: pad 16→46 with zeros for missing fields
```

This gives a warm start but real live-play data is needed for the expanded controls.

---

## Data Augmentation (Phase 10)

1. **Mirror**: Swap left/right hands, negate strafe + yaw
2. **Temporal jitter**: Random frame skip (1-2 frames) to simulate FPS variation
3. **Noise injection**: Small Gaussian noise on pose coordinates
4. **Partial hand dropout**: Randomly zero one hand (simulate occlusion)
5. **Game state perturbation**: Small noise on continuous game state values
