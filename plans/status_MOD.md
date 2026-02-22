# Phase 5 Status — MCCTP Mod V2 Expansion (Instance E — MOD)

## Status: COMPLETE

## Java Changes (mod side)

### New State Classes (src/client/java/com/mcctp/state/)
1. **PlayerInputInfo.java** — Resolved player input per tick: movementForward/Sideways (float), jump/sprint/sneak/attack/useItem/drop/swapOffhand/openInventory (boolean), yawDelta/pitchDelta (float). Tracks previous yaw/pitch with statics for delta computation.
2. **ScreenStateInfo.java** — GUI state: screenOpen (bool), screenType (class name string), cursorX/cursorY (normalized 0-1, -1 when closed), mouseLeft/mouseRight/shiftHeld (GLFW polling).
3. **StatusEffectInfo.java** — 6 boolean status effect checks: hasSpeed, hasSlowness, hasStrength, hasFireResistance, hasPoison, hasWither. Uses `StatusEffects` registry entries.
4. **ThreatInfo.java** — Threat scanning: targetEntityHostile, targetDistance (from crosshair), nearestHostileDist/nearestHostileYaw (scan 16-block AABB), hostileCount. Relative yaw normalized to [-180, 180].

### Updated State Classes
5. **PlayerStateInfo.java** — Added 4 fields: armor (int, getArmor()), isClimbing (bool), recentlyHurt (hurtTime > 0), horizontalCollision (bool).
6. **GameStatePayload.java** — Added playerInput, screenState, statusEffects, threats, timeOfDay (long), gameMode (String) to constructor and fields.
7. **GameStateCollector.java** — Collects all new info objects. Gets timeOfDay from world (% 24000), gameMode from interactionManager.

### New Action Handlers (src/client/java/com/mcctp/action/handlers/)
8. **CursorHandler.java** — "cursor" action: Sets GLFW cursor position from normalized (x, y) params. Only works when screen is open.
9. **ClickHandler.java** — "click" action: Simulates mouse click on open screen via `screen.mouseClicked()`. Supports "left"/"right" button param.
10. **ActionDispatcher.java** — Registered "cursor" → CursorHandler, "click" → ClickHandler.

## Python Changes (mcctp package)

### state.py
11. **New dataclasses**: PlayerInputInfo, ScreenStateInfo, StatusEffectInfo, ThreatInfo — each with `from_dict()` mapping camelCase JSON keys.
12. **PlayerState** — Added 4 new fields: armor, is_climbing, recently_hurt, horizontal_collision.
13. **GameState** — Added player_input, screen_state, status_effects, threats, time_of_day, game_mode. Updated `from_dict()`.
14. **to_control_dict()** — Returns all ~60 fields expected by `encode_game_state_v2()`:
    - Includes `screen_open_type` (computed string: "none"/"inventory"/"chest"/other) for proper encoder mapping
    - Includes both `has_fire_resistance` and `has_fire_resist` (alias) since encoder uses `has_fire_resist`
    - Resolved input fields prefixed with `input_` (input_jump, input_attack, etc.)
    - Screen interaction fields (cursor_x/y, mouse_left/right, shift_held)

### actions.py
15. Added `CURSOR` and `CLICK` action constants. Added `Actions.cursor(x, y)` and `Actions.click(button)` static methods.

## Key Decisions
- **yaw/pitch delta tracking**: Used static fields in PlayerInputInfo.java (reset on NaN for first tick)
- **Screen cursor**: Normalized 0-1 via `mouse.getX() / window.getWidth()`; set to -1 when no screen open
- **Threat scanning**: Uses `getEntitiesByClass(HostileEntity.class, box, e -> true)` on 16-block expanded AABB
- **Click handling**: Uses `screen.mouseClicked()` with scaled coordinates, not raw GLFW events
- **screen_open_type**: Computed in Python to_control_dict(), maps screen class names to categories the encoder expects

## Build Note
Could not verify build — gradle wrapper jar missing from repo. Code follows existing patterns and APIs.

## Backwards Compatibility
- All existing JSON fields unchanged (same names, same semantics)
- New fields are additive — old Python clients still work
- Python `from_dict()` methods default missing new fields gracefully
- Existing `to_control_dict()` v1 keys preserved
