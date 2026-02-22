# Task: Phase 5 — Expand MCCTP Fabric Mod for V2 (Instance E — MOD)

You are expanding the MCCTP Fabric mod to broadcast the full game state and resolved player input needed by the V2 control policy model. This is the critical-path blocker — all V2 Python code is complete and waiting on this.

## Project Overview

MCCTP is a Fabric client mod that runs a WebSocket server inside Minecraft. Every tick, it broadcasts the game state as JSON to connected Python clients. Python clients can also send action commands back to control the player. The Python side uses these broadcasts for a behavioral cloning system (camera → pose estimation → transformer model → Minecraft controls).

## Mod Location & Build Info

- **Mod repo**: `C:\Users\forke\Documents\CVcoolz\mcctp\`
- **Minecraft**: 1.21.11
- **Fabric Loader**: 0.18.4
- **Fabric API**: 0.141.3+1.21.11
- **Yarn Mappings**: 1.21.11+build.4
- **Java**: 21
- **Loom**: 1.15.3

## Current Architecture

### Tick Broadcast (MCCTPClient.java)
Every N ticks, `GameStateCollector.collect(client)` produces a JSON payload that gets broadcast via `ConnectionManager.broadcast()` to all connected WebSocket clients.

### Current State Classes (src/client/java/com/mcctp/state/)

**GameStatePayload.java** — top-level wrapper:
- `type`: "game_state"
- `timestamp`: long
- `selectedSlot`: int (0-8)
- `heldItem`: HeldItemInfo
- `offhandItem`: HeldItemInfo
- `playerState`: PlayerStateInfo
- `combatContext`: CombatContextInfo

**PlayerStateInfo.java** — 18 fields:
- health, maxHealth, hunger, saturation, x, y, z, yaw, pitch
- onGround, sprinting, sneaking, swimming, flying, inWater, onFire
- fallDistance, velocityY

**CombatContextInfo.java** — 8 fields:
- isUsingItem, isBlocking, activeHand, crosshairTarget, crosshairEntityType
- crosshairBlockPos, attackCooldown, itemUseProgress

**HeldItemInfo.java** — 5 fields:
- name, category, stackCount, maxDurability, currentDurability
- Uses ItemCategorizer → ItemCategory enum (SWORD, BOW, CROSSBOW, BLOCK, AXE, PICKAXE, SHOVEL, HOE, FOOD, SHIELD, TRIDENT, FISHING_ROD, THROWABLE, EMPTY, OTHER)

### Current Action Handlers (src/client/java/com/mcctp/action/handlers/)
13 handlers registered in ActionDispatcher: move, look, jump, sneak, sprint, attack, use_item, throw_item, drop_item, select_slot, swap_hands, open_inventory, toggle_wheel

### Python Package (mcctp/python/src/mcctp/)
- `state.py`: GameState dataclass with `to_control_dict()` returning ~19 fields (v1 format)
- `actions.py`: Actions class with 13 static methods matching Java handlers
- `__init__.py`: Exports MCCTPClient, SyncMCCTPClient, GameState, Actions

---

## WHAT MUST BE ADDED

The V2 Python code expects `GameState.to_control_dict()` to return a flat dict with ~60+ fields. Below is the complete gap analysis organized by implementation area.

### 1. Resolved Player Input State (NEW — for recording)

The Python recorder (`control_recorder.py`) needs to know what keys the player is actually pressing each tick, so it can record training data without pynput.

**New Java class**: `PlayerInputInfo.java`
```java
// Fields to capture:
float movementForward;     // player.input.movementForward (-1 to 1)
float movementSideways;    // player.input.movementSideways (-1 to 1)
boolean jump;              // player.input.playerInput.jump()
boolean sprint;            // player.input.playerInput.sprint()
boolean sneak;             // player.input.playerInput.sneak()
boolean attack;            // mc.options.attackKey.isPressed()
boolean useItem;           // mc.options.useKey.isPressed()
boolean drop;              // mc.options.dropKey.isPressed()
boolean swapOffhand;       // mc.options.swapHandsKey.isPressed()
boolean openInventory;     // mc.options.inventoryKey.isPressed()
float yawDelta;            // current yaw - previous yaw (per tick)
float pitchDelta;          // current pitch - previous pitch (per tick)
```

**Implementation notes**:
- `player.input` gives the resolved input state after Minecraft processes key bindings
- `movementForward`/`movementSideways` are float (-1 to 1) — handles analog input
- `yawDelta`/`pitchDelta` require tracking previous yaw/pitch across ticks — store in the collector or in this class as statics
- Use `mc.options.attackKey.isPressed()` etc. for the boolean key states

### 2. Screen/GUI State (NEW — for inventory interaction)

When a screen (inventory, chest, crafting table) is open, the Python code needs cursor position and click state.

**New Java class**: `ScreenStateInfo.java`
```java
boolean screenOpen;        // mc.currentScreen != null
String screenType;         // mc.currentScreen.getClass().getSimpleName() or "none"
float cursorX;             // normalized mouse X (0-1) when screen open, -1 when closed
float cursorY;             // normalized mouse Y (0-1) when screen open, -1 when closed
boolean mouseLeft;         // left mouse button state when screen open
boolean mouseRight;        // right mouse button state when screen open
boolean shiftHeld;         // shift key state when screen open
```

**Implementation notes**:
- Normalize cursor: `(float) mc.mouse.getX() / mc.getWindow().getWidth()`
- When no screen is open, set cursorX/Y to -1, mouseLeft/Right to false
- screenType should be a simple class name: "InventoryScreen", "GenericContainerScreen", etc.
- For the Python side, screen_open_type encodes as: 0=none, 0.33=inventory, 0.66=chest, 1.0=other

### 3. Missing Player State Fields

**Add to PlayerStateInfo.java**:
```java
int armor;                 // player.getArmor()
boolean isClimbing;        // player.isClimbing()
boolean recentlyHurt;      // player.hurtTime > 0
boolean horizontalCollision; // player.horizontalCollision
```

### 4. Status Effects (NEW)

**New Java class**: `StatusEffectInfo.java`
```java
boolean hasSpeed;          // player.hasStatusEffect(StatusEffects.SPEED)
boolean hasSlowness;       // player.hasStatusEffect(StatusEffects.SLOWNESS)
boolean hasStrength;       // player.hasStatusEffect(StatusEffects.STRENGTH)
boolean hasFireResistance; // player.hasStatusEffect(StatusEffects.FIRE_RESISTANCE)
boolean hasPoison;         // player.hasStatusEffect(StatusEffects.POISON)
boolean hasWither;         // player.hasStatusEffect(StatusEffects.WITHER)
```

**Implementation notes**:
- Import `net.minecraft.entity.effect.StatusEffects`
- The Python side maps hasPoison || hasWither → "taking_dot" (damage over time)
- hasFireResistance → "fire_resist" in Python

### 5. Threat Scanning (NEW)

**New Java class**: `ThreatInfo.java`
```java
boolean targetEntityHostile; // is the crosshair entity a hostile mob
float targetDistance;         // distance to crosshair target entity
float nearestHostileDist;    // distance to nearest hostile mob
float nearestHostileYaw;     // relative yaw to nearest hostile (degrees)
int hostileCount;            // hostile mobs within 16 blocks
```

**Implementation notes**:
- Scan entities within 16 blocks: `world.getEntitiesByClass(HostileEntity.class, box, e -> true)`
- Use AABB centered on player: `player.getBoundingBox().expand(16)`
- `targetEntityHostile`: check if crosshair entity is `instanceof HostileEntity`
- `targetDistance`: `player.distanceTo(crosshairEntity)` or 0 if no entity target
- `nearestHostileYaw`: compute relative yaw = `atan2(dx, dz) * 180/PI - playerYaw`, normalize to [-180, 180]
- Clamp values: nearestHostileDist to 32, hostileCount to 255

### 6. Environment Info (NEW)

**Add to GameStatePayload or new class**:
```java
long timeOfDay;            // world.getTimeOfDay() % 24000
String gameMode;           // interactionManager.getCurrentGameMode().getName()
```

### 7. New Action Handlers

**CursorHandler.java** — set mouse cursor position on open screen:
```java
// action: "cursor", params: {"x": 0.5, "y": 0.3}
// Implementation: Use GLFW to set cursor position
// x, y are normalized 0-1, convert to screen pixel coordinates
// Only works when a screen is open (mc.currentScreen != null)
```

**ClickHandler.java** — mouse click on open screen:
```java
// action: "click", params: {"button": "left"} or {"button": "right"}
// Implementation: Simulate mouse click at current cursor position on the open screen
// button: "left" (0) or "right" (1)
// Only works when a screen is open
```

### 8. Updated GameStatePayload

Add new info classes to the payload:
```java
public class GameStatePayload {
    public final String type = "game_state";
    public final long timestamp;
    public final int selectedSlot;
    public final HeldItemInfo heldItem;
    public final HeldItemInfo offhandItem;
    public final PlayerStateInfo playerState;
    public final CombatContextInfo combatContext;
    // NEW:
    public final PlayerInputInfo playerInput;
    public final ScreenStateInfo screenState;
    public final StatusEffectInfo statusEffects;
    public final ThreatInfo threats;
    public final long timeOfDay;
    public final String gameMode;
}
```

### 9. Updated GameStateCollector

The `collect()` method must create all the new info objects. For threat scanning, do the entity scan here (it has access to `client.world`).

---

## PYTHON PACKAGE UPDATES

### state.py — Expanded GameState

Add new dataclasses matching the Java classes:

```python
@dataclass
class PlayerInputInfo:
    movement_forward: float = 0.0
    movement_sideways: float = 0.0
    jump: bool = False
    sprint: bool = False
    sneak: bool = False
    attack: bool = False
    use_item: bool = False
    drop: bool = False
    swap_offhand: bool = False
    open_inventory: bool = False
    yaw_delta: float = 0.0
    pitch_delta: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> PlayerInputInfo:
        return cls(
            movement_forward=data.get("movementForward", 0.0),
            movement_sideways=data.get("movementSideways", 0.0),
            jump=data.get("jump", False),
            sprint=data.get("sprint", False),
            sneak=data.get("sneak", False),
            attack=data.get("attack", False),
            use_item=data.get("useItem", False),
            drop=data.get("drop", False),
            swap_offhand=data.get("swapOffhand", False),
            open_inventory=data.get("openInventory", False),
            yaw_delta=data.get("yawDelta", 0.0),
            pitch_delta=data.get("pitchDelta", 0.0),
        )

@dataclass
class ScreenStateInfo:
    screen_open: bool = False
    screen_type: str = "none"
    cursor_x: float = -1.0
    cursor_y: float = -1.0
    mouse_left: bool = False
    mouse_right: bool = False
    shift_held: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> ScreenStateInfo:
        # ... from_dict mapping camelCase JSON keys

@dataclass
class StatusEffectInfo:
    has_speed: bool = False
    has_slowness: bool = False
    has_strength: bool = False
    has_fire_resistance: bool = False
    has_poison: bool = False
    has_wither: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> StatusEffectInfo:
        # ... from_dict mapping camelCase JSON keys

@dataclass
class ThreatInfo:
    target_entity_hostile: bool = False
    target_distance: float = 0.0
    nearest_hostile_dist: float = 0.0
    nearest_hostile_yaw: float = 0.0
    hostile_count: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> ThreatInfo:
        # ... from_dict mapping camelCase JSON keys
```

### Updated GameState dataclass:
```python
@dataclass
class GameState:
    timestamp: int = 0
    selected_slot: int = 0
    held_item: HeldItemInfo = field(default_factory=HeldItemInfo)
    offhand_item: HeldItemInfo = field(default_factory=HeldItemInfo)
    player_state: PlayerState = field(default_factory=PlayerState)
    combat_context: CombatContext = field(default_factory=CombatContext)
    # NEW:
    player_input: PlayerInputInfo = field(default_factory=PlayerInputInfo)
    screen_state: ScreenStateInfo = field(default_factory=ScreenStateInfo)
    status_effects: StatusEffectInfo = field(default_factory=StatusEffectInfo)
    threats: ThreatInfo = field(default_factory=ThreatInfo)
    time_of_day: int = 0
    game_mode: str = "survival"
```

### Updated PlayerState (add missing fields):
```python
@dataclass
class PlayerState:
    # ... existing 18 fields ...
    # NEW:
    armor: int = 0
    is_climbing: bool = False
    recently_hurt: bool = False
    horizontal_collision: bool = False
```

### Expanded to_control_dict()

This is the critical method. It must return ALL fields the Python `encode_game_state_v2()` function expects. Here's the complete target dict:

```python
def to_control_dict(self) -> dict:
    """Flatten into the dict format expected by encode_game_state_v2().
    Returns a dict with all fields needed for the 46-dim game state vector + resolved inputs.
    """
    return {
        # Item context (existing)
        "held_item": self.held_item.name,
        "held_item_category": self.held_item.category,
        "offhand_category": self.offhand_item.category,

        # Vitals
        "health": self.player_state.health,
        "hunger": float(self.player_state.hunger),
        "armor": self.player_state.armor,           # NEW

        # Movement flags
        "on_ground": self.player_state.on_ground,
        "in_water": self.player_state.in_water,
        "swimming": self.player_state.swimming,
        "flying": self.player_state.flying,
        "is_climbing": self.player_state.is_climbing,   # NEW
        "on_fire": self.player_state.on_fire,
        "is_sprinting": self.player_state.sprinting,
        "is_sneaking": self.player_state.sneaking,
        "fall_distance": self.player_state.fall_distance,
        "velocity_y": self.player_state.velocity_y,

        # Combat
        "attack_cooldown": self.combat_context.attack_cooldown,
        "is_using_item": self.combat_context.is_using_item,
        "is_blocking": self.combat_context.is_blocking,
        "item_use_progress": self.combat_context.item_use_progress,
        "recently_hurt": self.player_state.recently_hurt,   # NEW

        # Crosshair
        "crosshair_target": self.combat_context.crosshair_target,
        "crosshair_entity_type": self.combat_context.crosshair_entity_type,

        # Threats (ALL NEW)
        "target_entity_hostile": self.threats.target_entity_hostile,
        "target_distance": self.threats.target_distance,
        "nearest_hostile_dist": self.threats.nearest_hostile_dist,
        "nearest_hostile_yaw": self.threats.nearest_hostile_yaw,
        "hostile_count": self.threats.hostile_count,

        # Environment (ALL NEW)
        "time_of_day": self.time_of_day,
        "game_mode": self.game_mode,
        "screen_open": self.screen_state.screen_open,
        "screen_type": self.screen_state.screen_type,

        # Status effects (ALL NEW)
        "has_speed": self.status_effects.has_speed,
        "has_slowness": self.status_effects.has_slowness,
        "has_strength": self.status_effects.has_strength,
        "has_fire_resistance": self.status_effects.has_fire_resistance,
        "has_poison": self.status_effects.has_poison,
        "has_wither": self.status_effects.has_wither,

        # Extra
        "selected_slot": self.selected_slot,
        "horizontal_collision": self.player_state.horizontal_collision,  # NEW

        # Position + look (for recorder, not part of 46-dim but used for delta computation)
        "yaw": self.player_state.yaw,
        "pitch": self.player_state.pitch,
        "x": self.player_state.x,
        "y": self.player_state.y,
        "z": self.player_state.z,

        # Resolved player input (ALL NEW — for recording)
        "movement_forward": self.player_input.movement_forward,
        "movement_sideways": self.player_input.movement_sideways,
        "input_jump": self.player_input.jump,
        "input_sprint": self.player_input.sprint,
        "input_sneak": self.player_input.sneak,
        "input_attack": self.player_input.attack,
        "input_use_item": self.player_input.use_item,
        "input_drop": self.player_input.drop,
        "input_swap_offhand": self.player_input.swap_offhand,
        "input_open_inventory": self.player_input.open_inventory,
        "yaw_delta": self.player_input.yaw_delta,
        "pitch_delta": self.player_input.pitch_delta,

        # Screen interaction (ALL NEW — for recording)
        "cursor_x": self.screen_state.cursor_x,
        "cursor_y": self.screen_state.cursor_y,
        "mouse_left": self.screen_state.mouse_left,
        "mouse_right": self.screen_state.mouse_right,
        "shift_held": self.screen_state.shift_held,
    }
```

### actions.py — New Action Methods

Add two new methods:

```python
CURSOR = "cursor"
CLICK = "click"

class Actions:
    # ... existing 13 methods ...

    @staticmethod
    def cursor(x: float, y: float) -> dict:
        """Set cursor position on open screen (x, y normalized 0-1)."""
        return {"action": CURSOR, "params": {"x": x, "y": y}}

    @staticmethod
    def click(button: str = "left") -> dict:
        """Click on open screen. button: 'left' or 'right'."""
        return {"action": CLICK, "params": {"button": button}}
```

---

## BACKWARDS COMPATIBILITY

- All existing JSON fields must keep their exact names and semantics
- New fields are additive — old Python clients that don't read them will still work
- The Python `GameState.from_dict()` must handle missing new fields gracefully (defaults)
- The existing `to_control_dict()` keys that v1 Python code uses must not change

---

## TESTING

After implementation, verify with:

```python
from mcctp import SyncMCCTPClient

client = SyncMCCTPClient(host="localhost", port=8080)
client.connect()

# Wait for a state update
import time
time.sleep(0.5)

state = client.state
d = state.to_control_dict()

# Check new fields exist
assert "armor" in d, "Missing armor"
assert "is_climbing" in d, "Missing is_climbing"
assert "recently_hurt" in d, "Missing recently_hurt"
assert "horizontal_collision" in d, "Missing horizontal_collision"
assert "has_speed" in d, "Missing has_speed"
assert "target_entity_hostile" in d, "Missing target_entity_hostile"
assert "nearest_hostile_dist" in d, "Missing nearest_hostile_dist"
assert "hostile_count" in d, "Missing hostile_count"
assert "time_of_day" in d, "Missing time_of_day"
assert "game_mode" in d, "Missing game_mode"
assert "screen_open" in d, "Missing screen_open"
assert "screen_type" in d, "Missing screen_type"
assert "movement_forward" in d, "Missing movement_forward"
assert "yaw_delta" in d, "Missing yaw_delta"
assert "cursor_x" in d, "Missing cursor_x"
assert "mouse_left" in d, "Missing mouse_left"
assert "shift_held" in d, "Missing shift_held"
assert "input_attack" in d, "Missing input_attack"

print(f"All {len(d)} fields present!")
print(f"Game mode: {d['game_mode']}")
print(f"Health: {d['health']}, Armor: {d['armor']}")
print(f"Screen open: {d['screen_open']}, Type: {d.get('screen_type', 'none')}")
print(f"Movement forward: {d['movement_forward']}")
print(f"Hostile count: {d['hostile_count']}")
print(f"Time of day: {d['time_of_day']}")

client.disconnect()
```

---

## DELIVERABLES

### Java (mod side):
1. `PlayerInputInfo.java` — new class in state/ package
2. `ScreenStateInfo.java` — new class in state/ package
3. `StatusEffectInfo.java` — new class in state/ package
4. `ThreatInfo.java` — new class in state/ package
5. `PlayerStateInfo.java` — add 4 new fields (armor, isClimbing, recentlyHurt, horizontalCollision)
6. `GameStatePayload.java` — add new info classes + timeOfDay + gameMode
7. `GameStateCollector.java` — collect all new data including threat scan
8. `CursorHandler.java` — new action handler
9. `ClickHandler.java` — new action handler
10. `ActionDispatcher.java` — register cursor + click handlers

### Python (mcctp package):
11. `state.py` — new dataclasses + expanded PlayerState + expanded GameState + expanded to_control_dict()
12. `actions.py` — add cursor() and click() methods

### Status:
13. Write status to `C:\Users\forke\Documents\CVcoolz\AndroidCamera\plans\status_MOD.md`

---

## REFERENCE FILES

For the exact 46-dim game state layout the Python side expects, see:
- `C:\Users\forke\Documents\CVcoolz\AndroidCamera\plans\v2_coordination.md` — Contract 3 (game state layout), Contract 8 (encode_game_state_v2 signature)
- `C:\Users\forke\Documents\CVcoolz\AndroidCamera\client\control_policy.py` — the canonical `encode_game_state_v2()` function that will consume the dict your `to_control_dict()` produces

The key names in `to_control_dict()` must match what `encode_game_state_v2()` looks up. Read that function to verify key name alignment.
