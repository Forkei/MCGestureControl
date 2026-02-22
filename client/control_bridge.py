"""Simplified MCCTP bridge for the control policy model.

Takes ControlOutput from ControlPolicy and translates it directly to
MCCTP commands. Much simpler than GameBridge — no gesture-to-action
mapping needed.

Held actions (forward, sprint, sneak, use_item, strafe): send start/stop
on state transitions only (edge detection).

Pulse actions (jump, attack): fire on rising edge with cooldown.

Analog look: send every frame when above deadzone, scaled by speed.

Usage:
    bridge = ControlBridge(port=8765)
    bridge.connect()
    # In main loop:
    game_state = bridge.get_game_state()
    bridge.update(control_output)
    # On exit:
    bridge.disconnect()
"""

import time

from mcctp import SyncMCCTPClient, Actions

from control_policy import ControlOutput, ControlOutputV2


class ControlBridge:
    """Sends ControlOutput to Minecraft via MCCTP."""

    # Look sensitivity: degrees per frame at full deflection
    LOOK_YAW_SPEED = 5.0
    LOOK_PITCH_SPEED = 3.0

    # Minimum time between pulse actions (seconds)
    PULSE_COOLDOWN = 0.3

    # Auto-release held actions after this many seconds (safety)
    HELD_MAX_DURATION = 10.0

    def __init__(self, host: str = "localhost", port: int = 8765):
        self._client = SyncMCCTPClient(host, port)

        # Track held action states for edge detection
        self._held = {
            "forward": False,
            "sprint": False,
            "sneak": False,
            "use_item": False,
            "strafe_left": False,
            "strafe_right": False,
        }

        # Timestamps for held action auto-release
        self._held_since: dict = {}

        # Pulse action state for rising-edge detection
        self._prev_jump = False
        self._prev_attack = False

        # Pulse cooldowns
        self._last_jump = 0.0
        self._last_attack = 0.0

    def connect(self):
        self._client.connect()
        print("[ControlBridge] Connected to MCCTP")

    def disconnect(self):
        self.release_all()
        self._client.disconnect()
        print("[ControlBridge] Disconnected")

    def get_game_state(self) -> dict:
        """Query current game state from MCCTP.

        Returns a flat dict with keys expected by encode_game_state():
        held_item, health, hunger, on_ground, in_water, is_sprinting,
        is_sneaking, is_blocking, is_using_item, fall_distance.
        """
        try:
            state = self._client.state
            if state is None:
                return {}
            return state.to_control_dict()
        except Exception:
            return {}

    def update(self, output: ControlOutput):
        """Send one frame of control output to Minecraft.

        Args:
            output: ControlOutput from ControlPolicy.predict().
        """
        now = time.monotonic()

        # --- Held controls: send start/stop on transitions ---
        self._set_held("forward", output.forward, now,
                       Actions.move("forward", "start"),
                       Actions.move("forward", "stop"))
        self._set_held("sprint", output.sprint, now,
                       Actions.sprint("start"),
                       Actions.sprint("stop"))
        self._set_held("sneak", output.sneak, now,
                       Actions.sneak("start"),
                       Actions.sneak("stop"))
        self._set_held("use_item", output.use_item, now,
                       Actions.use_item("start"),
                       Actions.use_item("stop"))
        self._set_held("strafe_left", output.strafe_left, now,
                       Actions.move("left", "start"),
                       Actions.move("left", "stop"))
        self._set_held("strafe_right", output.strafe_right, now,
                       Actions.move("right", "start"),
                       Actions.move("right", "stop"))

        # --- Pulse controls: fire on rising edge with cooldown ---
        if output.jump and not self._prev_jump:
            if now - self._last_jump >= self.PULSE_COOLDOWN:
                self._send(Actions.jump())
                self._last_jump = now
        self._prev_jump = output.jump

        if output.attack and not self._prev_attack:
            if now - self._last_attack >= self.PULSE_COOLDOWN:
                self._send(Actions.attack())
                self._last_attack = now
        self._prev_attack = output.attack

        # --- Analog look: send every frame when above deadzone ---
        if abs(output.look_yaw) > 0.01 or abs(output.look_pitch) > 0.01:
            yaw = output.look_yaw * self.LOOK_YAW_SPEED
            pitch = output.look_pitch * self.LOOK_PITCH_SPEED
            self._send(Actions.look(yaw=yaw, pitch=pitch, relative=True))

        # --- Auto-release safety ---
        self._check_auto_release(now)

    def release_all(self):
        """Release all held actions."""
        release_map = {
            "forward": Actions.move("forward", "stop"),
            "sprint": Actions.sprint("stop"),
            "sneak": Actions.sneak("stop"),
            "use_item": Actions.use_item("stop"),
            "strafe_left": Actions.move("left", "stop"),
            "strafe_right": Actions.move("right", "stop"),
        }
        for name, action in release_map.items():
            if self._held.get(name, False):
                self._send(action)
                self._held[name] = False
        self._held_since.clear()
        self._prev_jump = False
        self._prev_attack = False

    def _set_held(self, name: str, wanted: bool, now: float,
                  start_action: dict, stop_action: dict):
        """Send start/stop only on state transitions."""
        current = self._held[name]
        if wanted and not current:
            self._send(start_action)
            self._held[name] = True
            self._held_since[name] = now
        elif not wanted and current:
            self._send(stop_action)
            self._held[name] = False
            self._held_since.pop(name, None)

    def _check_auto_release(self, now: float):
        """Auto-release held actions that have been active too long."""
        for name in list(self._held_since.keys()):
            if now - self._held_since[name] > self.HELD_MAX_DURATION:
                release_map = {
                    "forward": Actions.move("forward", "stop"),
                    "sprint": Actions.sprint("stop"),
                    "sneak": Actions.sneak("stop"),
                    "use_item": Actions.use_item("stop"),
                    "strafe_left": Actions.move("left", "stop"),
                    "strafe_right": Actions.move("right", "stop"),
                }
                if name in release_map:
                    self._send(release_map[name])
                    self._held[name] = False
                    del self._held_since[name]

    def _send(self, action: dict):
        """Send an action, silently ignoring errors."""
        try:
            self._client.send(action)
        except Exception:
            pass


# ===========================================================================
# V2 Control Bridge — 28 controls, mode-aware sending
# ===========================================================================

class ControlBridgeV2:
    """Sends ControlOutputV2 to Minecraft via MCCTP with mode-aware logic.

    Gameplay mode: sends action start/stop, pulse, look, hotbar.
    Screen-open mode: sends cursor position, inventory clicks.
    open_inventory is always active (transitions between modes).

    Safety: auto-releases held actions after HELD_MAX_DURATION seconds.
    Edge detection prevents duplicate start/stop commands.
    """

    # Look sensitivity: degrees per frame at full deflection
    LOOK_YAW_SPEED = 5.0
    LOOK_PITCH_SPEED = 3.0

    # Minimum time between pulse actions (seconds)
    PULSE_COOLDOWN = 0.3

    # Auto-release held actions after this many seconds
    HELD_MAX_DURATION = 10.0

    def __init__(self, host: str = "localhost", port: int = 8765):
        self._client = SyncMCCTPClient(host, port)

        # Track held action states for edge detection
        self._held = {
            "forward": False,
            "backward": False,
            "strafe_left": False,
            "strafe_right": False,
            "sprint": False,
            "sneak": False,
            "use_item": False,
            "inv_shift_held": False,
        }

        # Timestamps for held action auto-release
        self._held_since: dict = {}

        # Pulse action state for rising-edge detection
        self._prev_pulse = {
            "jump": False,
            "attack": False,
            "drop_item": False,
            "swap_offhand": False,
            "open_inventory": False,
            "inv_left_click": False,
            "inv_right_click": False,
        }

        # Pulse cooldowns (action_name → last fire timestamp)
        self._last_pulse: dict = {}

    def connect(self):
        self._client.connect()
        print("[ControlBridgeV2] Connected to MCCTP")

    def disconnect(self):
        self.release_all()
        self._client.disconnect()
        print("[ControlBridgeV2] Disconnected")

    def get_game_state(self) -> dict:
        """Query current game state from MCCTP."""
        try:
            state = self._client.state
            if state is None:
                return {}
            return state.to_control_dict()
        except Exception:
            return {}

    def update(self, output: ControlOutputV2):
        """Send one frame of control output to Minecraft.

        Mode-aware: only sends relevant controls based on screen_open flag.
        open_inventory is always processed regardless of mode.
        """
        now = time.monotonic()

        if not output.screen_open:
            self._update_gameplay(output, now)
        else:
            self._update_screen(output, now)

        # open_inventory always active (allows mode transitions)
        self._fire_pulse("open_inventory", output.open_inventory, now,
                         Actions.open_inventory())

        # Auto-release safety
        self._check_auto_release(now)

    def _update_gameplay(self, output: ControlOutputV2, now: float):
        """Handle gameplay controls: movement, actions, look, hotbar."""

        # Held actions: send start/stop on state transitions
        self._set_held("forward", output.move_forward, now,
                       Actions.move("forward", "start"),
                       Actions.move("forward", "stop"))
        self._set_held("backward", output.move_backward, now,
                       Actions.move("backward", "start"),
                       Actions.move("backward", "stop"))
        self._set_held("strafe_left", output.strafe_left, now,
                       Actions.move("left", "start"),
                       Actions.move("left", "stop"))
        self._set_held("strafe_right", output.strafe_right, now,
                       Actions.move("right", "start"),
                       Actions.move("right", "stop"))
        self._set_held("sprint", output.sprint, now,
                       Actions.sprint("start"),
                       Actions.sprint("stop"))
        self._set_held("sneak", output.sneak, now,
                       Actions.sneak("start"),
                       Actions.sneak("stop"))
        self._set_held("use_item", output.use_item, now,
                       Actions.use_item("start"),
                       Actions.use_item("stop"))

        # Pulse actions: fire on rising edge with cooldown
        self._fire_pulse("jump", output.jump, now, Actions.jump())
        self._fire_pulse("attack", output.attack, now, Actions.attack())
        self._fire_pulse("drop_item", output.drop_item, now,
                         Actions.throw_item())
        self._fire_pulse("swap_offhand", output.swap_offhand, now,
                         Actions.swap_offhand())

        # Analog look: send every frame when above threshold
        if abs(output.look_yaw) > 0.01 or abs(output.look_pitch) > 0.01:
            yaw = output.look_yaw * self.LOOK_YAW_SPEED
            pitch = output.look_pitch * self.LOOK_PITCH_SPEED
            self._send(Actions.look(yaw=yaw, pitch=pitch, relative=True))

        # Hotbar: send on slot change
        if output.hotbar_slot is not None:
            self._send(Actions.select_slot(output.hotbar_slot))

    def _update_screen(self, output: ControlOutputV2, now: float):
        """Handle screen-open controls: cursor, inventory clicks."""

        # Release any gameplay held actions when entering screen mode
        for name in ("forward", "backward", "strafe_left", "strafe_right",
                     "sprint", "sneak", "use_item"):
            if self._held[name]:
                self._release_held(name)

        # Cursor: send every frame
        self._send(Actions.cursor(
            x=output.cursor_x, y=output.cursor_y
        ))

        # Inv shift held
        self._set_held("inv_shift_held", output.inv_shift_held, now,
                       Actions.sneak("start"),    # shift = sneak key
                       Actions.sneak("stop"))

        # Inventory clicks: pulse on rising edge
        self._fire_pulse("inv_left_click", output.inv_left_click, now,
                         Actions.click(button="left"))
        self._fire_pulse("inv_right_click", output.inv_right_click, now,
                         Actions.click(button="right"))

    def _set_held(self, name: str, wanted: bool, now: float,
                  start_action: dict, stop_action: dict):
        """Send start/stop only on state transitions."""
        current = self._held[name]
        if wanted and not current:
            self._send(start_action)
            self._held[name] = True
            self._held_since[name] = now
        elif not wanted and current:
            self._send(stop_action)
            self._held[name] = False
            self._held_since.pop(name, None)

    def _release_held(self, name: str):
        """Force-release a held action."""
        release_map = {
            "forward": Actions.move("forward", "stop"),
            "backward": Actions.move("backward", "stop"),
            "strafe_left": Actions.move("left", "stop"),
            "strafe_right": Actions.move("right", "stop"),
            "sprint": Actions.sprint("stop"),
            "sneak": Actions.sneak("stop"),
            "use_item": Actions.use_item("stop"),
            "inv_shift_held": Actions.sneak("stop"),
        }
        if name in release_map:
            self._send(release_map[name])
            self._held[name] = False
            self._held_since.pop(name, None)

    def _fire_pulse(self, name: str, wanted: bool, now: float,
                    action: dict):
        """Fire a pulse action on rising edge with cooldown."""
        prev = self._prev_pulse.get(name, False)
        if wanted and not prev:
            if now - self._last_pulse.get(name, 0) >= self.PULSE_COOLDOWN:
                self._send(action)
                self._last_pulse[name] = now
        self._prev_pulse[name] = wanted

    def _check_auto_release(self, now: float):
        """Auto-release held actions active too long."""
        for name in list(self._held_since.keys()):
            if now - self._held_since[name] > self.HELD_MAX_DURATION:
                self._release_held(name)

    def release_all(self):
        """Release all held actions (emergency stop / disconnect)."""
        for name in list(self._held.keys()):
            if self._held[name]:
                self._release_held(name)
        self._prev_pulse = {k: False for k in self._prev_pulse}

    def _send(self, action: dict):
        """Send an action, silently ignoring errors."""
        try:
            self._client.send(action)
        except Exception:
            pass
