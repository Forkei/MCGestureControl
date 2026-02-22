# Status: INFER

## Completed
- [x] `encode_game_state_v2()` — canonical 46-dim encoder (Contract 8)
- [x] `ControlOutputV2` dataclass with `to_control_vector()` (Contract 5)
- [x] `ControlPolicyV2` class — 5-head inference wrapper with mode-aware post-processing
- [x] `ControlBridgeV2` class — 28-control MCCTP sender with mode-aware logic
- [x] All v1 code preserved intact (ControlOutput, encode_game_state, ControlPolicy, ControlBridge)

## Interface Changes
- None. All contracts followed exactly as specified.

## Notes
- `encode_game_state_v2` handles both named categories and keyword fallback for item classification
- `ControlPolicyV2` stores per-frame game states (not tiled like v1), enabling true temporal game state
- Mode switching uses `game_state[38]` (GS_SCREEN_OPEN_IDX) from the last frame in the buffer
- `open_inventory` (action head index 11) is processed in both gameplay and screen-open modes
- `ControlBridgeV2._update_screen()` auto-releases gameplay held actions when entering screen mode
- Hotbar slot sync: reads from game state index 44 (current_hotbar_slot / 8) to stay in sync with actual game slot
- MCCTP API calls use patterns from existing v1 code (Actions.move("forward","start"), etc.)
- New MCCTP commands (Actions.open_inventory, Actions.cursor, Actions.click, Actions.swap_offhand) assumed from instance prompt — may need verification against actual mcctp package
- `ControlBridgeV2` uses `inv_shift_held` mapped to sneak start/stop (shift = sneak key in MC)
- Feature normalization is optional — applied only if `feature_mean`/`feature_std` present in config JSON
