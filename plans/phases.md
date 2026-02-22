# Implementation Phases v2

## Completed (v1)

### Phase 0: MCCTP Game State ✅
- Basic game state flowing from Minecraft to Python (16→24 dims)
- `encode_game_state()` in control_policy.py

### Phase 1: Control Recorder ✅
- `control_recorder.py` — records pose + hands + keyboard/mouse actions
- pynput-based capture (replaced in Phase 6)

### Phase 2: Dataset & Model ✅
- `control_dataset.py` — windowed dataset with velocity features
- `control_model.py` — ControlTransformer (559 input, 10 output, ~755K params)
- `bootstrap_controls.py` — converted gesture recordings to control format

### Phase 3: Training Pipeline ✅
- `train_controls.py` — training loop with early stopping
- Per-control threshold optimization on validation set
- Combined score metric (F1 + idle accuracy)

### Phase 4: Integration ✅
- `control_policy.py` — live inference wrapper
- `control_bridge.py` — MCCTP command sender
- Basic forward/sprint/sneak/attack/use_item working from bootstrapped data

---

## V2 Python Code — Completed ✅ (Phases 6-9)

### Phase 6: Recorder v2 ✅ (code ready, waiting on Phase 5 mod)
- `control_recorder.py` — MCCTPControlCapture replaces pynput
- 28-dim control vector, 46-dim game state, v2 .npz format
- Imports canonical `encode_game_state_v2` from control_policy.py

### Phase 7: Model v2 ✅
- `control_model.py` — ControlTransformerV2 (671 input, 5 heads, ~3.51M params)
- `control_dataset.py` — v2 dataset with mode masks, v1 auto-compat

### Phase 8: Training v2 ✅
- `train_controls.py` — mode-aware masked loss, 5-head metrics, combined scoring
- Threshold optimization for action + hotbar + inv_click heads

### Phase 9: Inference + Bridge v2 ✅
- `control_policy.py` — ControlPolicyV2, ControlOutputV2, encode_game_state_v2
- `control_bridge.py` — ControlBridgeV2 with mode-aware MCCTP sending

### Integration fixes ✅
- Consolidated encode_game_state_v2 to single canonical source (control_policy.py)
- Unified CTRL_* constants and head groupings via control_dataset.py imports
- All cross-file interfaces verified (model↔training, dataset↔training, model↔policy)

---

## Remaining Phases

### Phase 5: Mod Expansion — CURRENT PRIORITY 🔴

**Goal**: Update MCCTP Fabric mod to broadcast resolved player input + extended game state.

**Status**: The v2 Python code is complete and waiting on this. Phase 5 is the critical path.

**Mod repo**: `C:\Users\forke\Documents\CVcoolz\mcctp\`

**What the mod ALREADY broadcasts** (from GameStateCollector):
- PlayerStateInfo: health, hunger, x/y/z, yaw, pitch, onGround, sprinting, sneaking, swimming, flying, inWater, onFire, fallDistance, velocityY
- HeldItemInfo: name, category, stackCount, durability (main + offhand)
- CombatContextInfo: selectedSlot, isUsingItem, isBlocking, activeHand, crosshairTarget, crosshairEntityType, crosshairBlockPos, attackCooldown, itemUseProgress

**What MUST BE ADDED** (grouped by implementation area):

1. **Resolved player input state** (NEW — for recording):
   - `movementForward` (float): player.input.movementForward
   - `movementSideways` (float): player.input.movementSideways
   - `jump` (bool): player.input.playerInput.jump
   - `sprint` (bool): player.input.playerInput.sprint
   - `sneak` (bool): player.input.playerInput.sneak
   - `attack` (bool): mc.options.attackKey.isPressed()
   - `useItem` (bool): mc.options.useKey.isPressed()
   - `drop` (bool): mc.options.dropKey.isPressed()
   - `swapOffhand` (bool): mc.options.swapHandsKey.isPressed()
   - `openInventory` (bool): mc.options.inventoryKey.isPressed()
   - `yawDelta` / `pitchDelta` (float): per-tick look deltas
   - `screenOpen` (bool): mc.currentScreen != null
   - `screenType` (string): screen class name when open
   - `cursorX` / `cursorY` (float 0-1): normalized mouse position when screen open
   - `mouseLeft` / `mouseRight` (bool): mouse button state when screen open
   - `shiftHeld` (bool): shift key state when screen open

2. **Missing game state fields**:
   - `armor` (int): player.getArmor()
   - `isClimbing` (bool): player.isClimbing()
   - `recentlyHurt` (bool): player.hurtTime > 0
   - `horizontalCollision` (bool): player.horizontalCollision
   - `timeOfDay` (long): world.getTimeOfDay()
   - `gameMode` (string): interactionManager.getCurrentGameMode()
   - `screenOpenType` (float): 0=none, 0.33=inventory, 0.66=chest, 1.0=other

3. **Status effects** (NEW):
   - `hasSpeed`, `hasSlowness`, `hasStrength`, `hasFireResist` (bool)
   - `hasPoison`, `hasWither` (bool) — for taking_dot

4. **Threat scanning** (NEW):
   - `targetEntityHostile` (bool): is crosshair entity hostile
   - `targetDistance` (float): distance to crosshair target
   - `nearestHostileDist` (float): nearest hostile mob distance
   - `nearestHostileYaw` (float): relative yaw to nearest hostile
   - `hostileCount` (int): hostile mobs within 16 blocks

5. **New action handlers**:
   - `cursor` action: set mouse cursor position on open screen
   - `click` action: mouse click with button param on open screen

6. **Field name normalization**:
   - Flatten nested offhandItem.category → offhandCategory at root
   - Flatten nested heldItem.category → heldItemCategory at root
   - Add snake_case aliases for all camelCase fields

**Deliverables**:
- Updated Fabric mod JAR with all fields
- Updated Python mcctp package (if separate)
- Test script: `python -c "from mcctp import SyncMCCTPClient; ..."`

**Milestone**: Run test → see all 46 game state dims + resolved inputs updating live.

---

### Phase 10: Data Collection & Training

**Goal**: Record v2 training data and train the model.

**Tasks**:
1. Record 30-60 min of gameplay sessions using v2 recorder
2. Verify recordings: all 28 control dims populated, game state valid
3. Run `train_controls.py` — all 5 heads should show learning curves
4. Evaluate combined score, per-head metrics
5. Iterate: record more data for weak heads (inventory, hotbar)

**Milestone**: Combined score > 0.6 on validation set.

---

### Phase 11: End-to-End Integration & Polish

**Goal**: Wire everything into stream_client.py and refine for daily use.

**Tasks**:
1. Update `stream_client.py` to use ControlPolicyV2 + ControlBridgeV2
2. End-to-end testing: camera → model → Minecraft
3. Performance profiling (<30ms pipeline latency)
4. Failure recovery: auto-release if tracking lost
5. Debug overlay showing controls, confidence, game state
6. Curriculum learning: movement → combat → inventory

**Milestone**: Play Minecraft for 30+ minutes using only body gestures.

---

## Dependency Graph

```
Phase 5 (Mod) ──→ Phase 10 (Record + Train) ──→ Phase 11 (Integration + Polish)
     🔴                   ⬚                              ⬚
  CURRENT            needs Phase 5                  needs Phase 10

Phases 6-9 (Python code) ✅ DONE — waiting on Phase 5
```

Phase 5 is the ONLY blocker. All Python code is ready.
