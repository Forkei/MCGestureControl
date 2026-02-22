# Quick Reference v2

Concrete numbers, dimensions, APIs, and configs for implementation.

---

## Feature Dimensions

| Component | Dims | Index Range |
|---|---|---|
| pose_world (33×3) | 99 | 0:99 |
| left_hand_3d (21×3) | 63 | 99:162 |
| right_hand_3d (21×3) | 63 | 162:225 |
| left_hand_present | 1 | 225 |
| right_hand_present | 1 | 226 |
| pose_visibility (33) | 33 | 227:260 |
| **Raw total** | **260** | |
| velocity (deltas of 0:225) | 225 | 260:485 |
| **With velocity** | **485** | |
| game_state | 46 | 485:531 |
| action_history (5×28) | 140 | 531:671 |
| **Total input** | **671** | |

---

## Game State Breakdown (46 dims)

| Category | Dims | Index Range | Fields |
|---|---|---|---|
| Item context | 11 | 0:11 | can_melee, can_ranged, can_place, can_eat, is_tool, is_empty, is_other, offhand_shield/food/totem/empty |
| Player vitals | 3 | 11:14 | health, hunger, armor |
| Movement | 10 | 14:24 | on_ground, in_water, is_swimming, is_flying, is_climbing, on_fire, is_sprinting, is_sneaking, fall_distance, velocity_y |
| Combat | 5 | 24:29 | attack_cooldown, is_using_item, is_blocking, item_use_progress, recently_hurt |
| Crosshair | 4 | 29:33 | target_is_entity, target_is_block, target_entity_hostile, target_distance |
| Nearby threats | 3 | 33:36 | nearest_hostile_dist, nearest_hostile_yaw, hostile_count |
| Environment | 3 | 36:39 | time_of_day, game_mode_survival, screen_open_type |
| Status effects | 5 | 39:44 | has_speed, has_slowness, has_strength, taking_dot, has_fire_resist |
| Extra | 2 | 44:46 | current_hotbar_slot, horizontal_collision |

---

## Control Vector (28 dims)

| Index | Name | Type | Head | Active When |
|---|---|---|---|---|
| 0 | move_forward | bool | Action | gameplay |
| 1 | move_backward | bool | Action | gameplay |
| 2 | strafe_left | bool | Action | gameplay |
| 3 | strafe_right | bool | Action | gameplay |
| 4 | sprint | bool | Action | gameplay |
| 5 | sneak | bool | Action | gameplay |
| 6 | jump | bool (pulse) | Action | gameplay |
| 7 | attack | bool (pulse) | Action | gameplay |
| 8 | use_item | bool | Action | gameplay |
| 9 | look_yaw | float [-1,1] | Look | gameplay |
| 10 | look_pitch | float [-1,1] | Look | gameplay |
| 11 | drop_item | bool (pulse) | Action | gameplay |
| 12 | swap_offhand | bool (pulse) | Action | gameplay |
| 13 | open_inventory | bool (pulse) | Action | always |
| 14-22 | hotbar_slot_1..9 | bool (one-hot) | Hotbar | gameplay |
| 23 | cursor_x | float [0,1] | Cursor | screen_open |
| 24 | cursor_y | float [0,1] | Cursor | screen_open |
| 25 | inv_left_click | bool (pulse) | InvClick | screen_open |
| 26 | inv_right_click | bool (pulse) | InvClick | screen_open |
| 27 | inv_shift_held | bool | InvClick | screen_open |

---

## Model Hyperparameters

| Parameter | Value |
|---|---|
| d_model | 256 |
| nhead | 8 |
| num_layers | 6 |
| dim_feedforward | 512 |
| dropout | 0.3 |
| max_seq_len | 30 |
| pooling | exponential (decay=-2.0) |
| total params | ~3.5M |

---

## Output Heads

| Head | Inputs | Output Dims | Activation | Loss |
|---|---|---|---|---|
| Action | pooled (256) | 9 + 3 utility = 12 | sigmoid (via BCE logits) | BCEWithLogitsLoss + pos_weight |
| Look | pooled (256) | 2 | tanh | MSE (masked by look_active) |
| Hotbar | pooled (256) | 9 | softmax (via CE logits) | CrossEntropyLoss (masked by slot_changed) |
| Cursor | pooled (256) | 2 | sigmoid | MSE (masked by screen_open) |
| InvClick | pooled (256) | 3 | sigmoid (via BCE logits) | BCEWithLogitsLoss (masked by screen_open) |

---

## In-Game Control Capture Sources (from Mod)

| Control | Mod Field / Method |
|---|---|
| move_forward | `player.input.movementForward > 0` |
| move_backward | `player.input.movementForward < 0` |
| strafe_left | `player.input.movementSideways > 0` |
| strafe_right | `player.input.movementSideways < 0` |
| sprint | `player.input.playerInput.sprint()` |
| sneak | `player.input.playerInput.sneak()` |
| jump | `player.input.playerInput.jump()` |
| attack | `mc.options.attackKey.isPressed()` or cooldown reset detection |
| use_item | `mc.options.useKey.isPressed()` |
| look_yaw | `currentYaw - prevYaw` (per tick delta) |
| look_pitch | `currentPitch - prevPitch` (per tick delta) |
| hotbar_slot | `player.getInventory().selectedSlot` change detection |
| drop_item | `mc.options.dropKey.isPressed()` |
| swap_offhand | `mc.options.swapHandsKey.isPressed()` or offhand change detection |
| open_inventory | `mc.options.inventoryKey.isPressed()` or screen state change |
| cursor_x/y | `mc.mouse.getX()/getY()` normalized to screen when GUI open |
| inv_left_click | mouse button 0 when GUI open |
| inv_right_click | mouse button 1 when GUI open |
| inv_shift_held | shift key held when GUI open |

---

## New Mod Fields Needed

Fields NOT in current MCCTP broadcast that must be added:

| Field | Source | Description |
|---|---|---|
| attack_cooldown | `player.getAttackCooldownProgress(0.0f)` | 0-1 cooldown progress |
| velocity_y | `player.getVelocity().y` | Vertical velocity |
| item_use_progress | `player.getItemUseTimeLeft() / maxUseTime` | Item use progress |
| armor | `player.getArmor()` | Armor points |
| screen_open | `mc.currentScreen != null` | Is GUI screen open |
| screen_type | `mc.currentScreen.getClass()` | What type of screen |
| target_entity | `mc.crosshairTarget` type check | Crosshair raytrace |
| target_distance | `mc.crosshairTarget.squaredDistanceTo()` | Distance to target |
| hostile_scan | Entity scan within 16 blocks | Nearest hostile info |
| status_effects | `player.getStatusEffects()` | Active potion effects |
| time_of_day | `world.getTimeOfDay()` | World time |
| horizontal_collision | `player.horizontalCollision` | Wall collision |
| resolved_inputs | `player.input.*` | Resolved movement/jump/sneak/sprint |
| cursor_position | `mc.mouse.getX()/getY()` | Mouse position when GUI open |
| mouse_buttons | Mouse button states when GUI open | Click detection |

---

## Training Config

| Parameter | Value |
|---|---|
| batch_size | 64 |
| learning_rate | 5e-4 (warmup 5 epochs → cosine decay) |
| weight_decay | 1e-4 |
| max_epochs | 200 |
| early_stop_patience | 30 |
| action_history_dropout | 0.3 |
| look_deadzone | 0.08 |
| pos_weight_cap | 50.0 |
| train/val split | 80/20 by session |

---

## MCCTP API Quick Reference

```python
from mcctp import SyncMCCTPClient, Actions

client = SyncMCCTPClient(host="localhost", port=8080)
client.connect()

# Read state
state = client.state              # GameState object
d = state.to_control_dict()       # dict with all fields

# Send actions
client.perform(Actions.move(forward=True))
client.perform(Actions.look(yaw=0.5, pitch=0.0))
client.perform(Actions.jump())
client.perform(Actions.sprint(True))
client.perform(Actions.sneak(True))
client.perform(Actions.attack())
client.perform(Actions.use_item())
client.perform(Actions.hotbar(slot=0))  # 0-8
client.perform(Actions.drop_item())
client.perform(Actions.swap_offhand())
client.perform(Actions.open_inventory())

client.disconnect()
```

---

## File Locations

| File | Purpose |
|---|---|
| `client/control_recorder.py` | Records demonstrations (pose + controls + game state) |
| `client/control_model.py` | ControlTransformer model definition |
| `client/control_dataset.py` | PyTorch dataset with windowing + velocity |
| `client/train_controls.py` | Training script |
| `client/control_policy.py` | Live inference wrapper |
| `client/control_bridge.py` | Sends model outputs to MCCTP |
| `client/models/control_policy.pt` | Trained model weights |
| `client/models/control_config.json` | Model config + thresholds + normalization |
| `client/recordings/` | Recorded demo sessions (.npz) |
