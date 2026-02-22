# Status: DATA

## Completed
- [x] control_dataset.py — full V2 rewrite
  - [x] V2 dimension constants (Contract 1): INPUT_DIM=671, NUM_CONTROLS=28, GAME_STATE_DIM=46, ACTION_HISTORY_DIM=140
  - [x] Control vector layout constants (Contract 2): all 28 indices + head groupings
  - [x] WINDOW_STRIDE changed from 5 to 1
  - [x] V1 → V2 control mapping (_map_v1_controls_to_v2): maps 10-dim v1 to 28-dim v2
  - [x] V1 → V2 game state remapping (_remap_v1_game_state_to_v2): maps 16/24 dim to 46 dim with correct semantic mapping
  - [x] Auto-detect v2 'frames' key vs v1 individual arrays in extract_control_features
  - [x] _load_session_data: auto-detects version, normalizes to v2 format
  - [x] Mode mask generation (compute_mode_masks): gameplay, screen_open, hotbar_changed, look_active
  - [x] Augmentation extended for 28-dim controls + 46-dim game state (temporal crop, time warp, noise)
  - [x] Action history dropout extended for 28-dim
  - [x] window_session returns (features, target, mode_masks) tuples
  - [x] ControlDataset.__getitem__ returns 4-tuple: (features, mask, target, mode_masks)
  - [x] Statistics updated for all 5 output heads
  - [x] CLI test mode prints mode_masks

- [x] control_recorder.py — full V2 rewrite
  - [x] Removed ControlInputCapture (pynput-based)
  - [x] Removed HeadLookTracker and head_orientation
  - [x] Removed pynput dependency entirely
  - [x] Added MCCTPControlCapture: reads resolved inputs from MCCTP state
    - Handles movement_forward/backward, strafing, sprint, sneak, jump
    - Look via yaw_delta/pitch_delta (preferred) or computed from yaw/pitch changes
    - Hotbar one-hot on slot change detection
    - Inventory cursor + clicks when screen open
    - Robust to missing keys (tries multiple key name variants)
  - [x] Added encode_game_state_v2: full 46-dim implementation matching Contract 3
    - Item context with category + keyword fallback (11 dims)
    - Vitals, movement, combat, crosshair, threats, environment, status effects, extra
  - [x] extract_frame_features: returns 260-dim vector directly (not dict)
  - [x] V2 save format (Contract 6): frames, controls, game_state, timestamps, fps, version=2
  - [x] Updated overlay: 4 rows (movement, actions, look+hotbar, inventory)
  - [x] Removed --head-look CLI flag
  - [x] Window title: "Control Recorder V2"

## Interface Changes
- **ControlDataset.__getitem__** now returns 4-tuple `(features, mask, target, mode_masks)` instead of 3-tuple. Instance C (TRAIN) must use a custom collate function to batch the mode_masks dict.
- **mode_masks** is a dict with string keys and bool values: `{'gameplay': bool, 'screen_open': bool, 'hotbar_changed': bool, 'look_active': bool}`
- **V2 recording format** saves pre-extracted `frames` (N, 260) instead of individual pose/hand arrays. The dataset loader handles both formats.
- **encode_game_state_v2** is implemented locally in control_recorder.py. Instance D should provide the canonical version in control_policy.py. Once available, recorder can import from there instead.

## Notes
- V1 game state remapping: fall_distance normalization differs (v1=/10, v2=/20) — adjusted with 0.5 multiplier
- V1 recordings have no backward, drop, swap, inventory, hotbar, or cursor data — these dimensions are zero-padded
- MCCTPControlCapture tries multiple key name variants (snake_case and camelCase) for MCCTP state dict compatibility
- Look normalization uses LOOK_NORMALIZE_FACTOR=15.0 (game degrees per tick → [-1, 1])
- The `session_type` metadata field is preserved in v2 saves for split_sessions_by_type compatibility
- Game state normalization ranges:
  - fall_distance: /20 (clamp 0-20)
  - velocity_y: /3 (clamp -3 to 3)
  - target_distance: /6 (clamp 0-6)
  - nearest_hostile_dist: /32 (clamp 0-32)
  - hostile_count: /10 (clamp 0-10)
  - time_of_day: /24000
  - Instance D should use the SAME normalization ranges for consistency
