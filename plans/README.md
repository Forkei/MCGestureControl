# Control Policy Model v2 — Master Plan

## Vision

Replace the gesture classifier + lookup table with a **regression policy model** that
directly outputs full Minecraft controls from body pose, hand landmarks, and game state.
The model learns the mapping end-to-end from recorded demonstrations (behavioral cloning).

**v2 expands** to full Minecraft control: inventory management, hotbar switching,
cursor control, and mode-aware outputs (gameplay vs inventory screen).

## Why Change

| Current System | Policy Model v2 |
|---|---|
| Classify gesture → label → lookup table → action | Pose + context → controls directly |
| High false positive rate (walking@0.3 threshold) | Idle is dominant in training data — learned naturally |
| No game state awareness | 46-dim context: items, combat, threats, environment |
| Same gesture always fires same action | Context-dependent: bow gesture only fires when holding bow |
| 9 gesture thresholds to hand-tune | Per-control optimized thresholds from validation |
| Fixed gesture vocabulary | Learns whatever you demonstrate |
| No inventory control | Full inventory: cursor, clicks, hotbar, drop, swap |
| Keyboard/mouse dependent recording | In-game resolved input capture (keybind-independent) |

## Architecture at a Glance

```
Every frame:
                                    ┌─────────────┐
  Pose (33x3 world) ──────────────→ │             │──→ Action Head  ──→ [9 binary controls]
  Hands (2x21x3)    ──────────────→ │ Transformer │──→ Look Head    ──→ [yaw, pitch]
  Velocity features ──────────────→ │  Encoder    │──→ Hotbar Head  ──→ [9 slots one-hot]
  Game state (46 dims) ──────────→ │  ~3.5M      │──→ Cursor Head  ──→ [x, y]
  Action history (last 5 × 28) ──→ │             │──→ InvClick Head──→ [3 binary]
                                    └─────────────┘
                                         ↓
                                    Post-processing
                                    (thresholds, hysteresis, smoothing, mode switch)
                                         ↓
                                    Control Bridge → MCCTP → Minecraft
```

**Mode-aware**: When `screen_open` flag is active, cursor/inv_click heads are used.
When screen is closed, action/look/hotbar heads are used. Masked loss ensures each
head only trains on relevant frames.

## Documents

| File | Contents |
|---|---|
| [architecture.md](architecture.md) | Model architecture, input/output spec, loss design |
| [data_pipeline.md](data_pipeline.md) | Recording tool, dataset, in-game control capture |
| [training.md](training.md) | Training loop, mode-aware loss, evaluation metrics |
| [integration.md](integration.md) | How it plugs into stream_client.py |
| [risks.md](risks.md) | Known issues, edge cases, mitigations |
| [phases.md](phases.md) | Implementation roadmap with milestones |
| [reference.md](reference.md) | Quick-lookup: dimensions, APIs, configs |

## Key Decisions

1. **Multi-head output (5 heads)**: Action, Look, Hotbar, Cursor, InvClick — each
   specializes with its own loss function while sharing the transformer backbone.

2. **In-game control capture**: Recording reads resolved game input state from the mod
   (`player.input.movementForward`, `playerInput.jump()`, yaw/pitch deltas) instead of
   keyboard/mouse via pynput. Keybind and sensitivity independent.

3. **46-dim game state**: Item capabilities, player vitals, movement state, combat state,
   crosshair target, nearby threats, environment, status effects. Derived from MCCTP mod.

4. **Mode-aware training**: `screen_open` flag determines which output heads are active.
   Cursor/inv_click only train on inventory-open frames. Action/look only on gameplay frames.

5. **Action history feedback (140 dims)**: Last 5 outputs × 28 dims fed back as input.
   Teacher forcing with 30% dropout during training, own outputs during inference.

6. **Behavioral cloning from live play**: User plays Minecraft normally while camera
   records pose/hands. Mod captures resolved inputs + game state. ~30-60 min sessions.

7. **Post-processing outside the model**: Per-control thresholds with hysteresis, EMA
   smoothing + deadzone for look/cursor, edge detection for pulse actions, mode switching.
