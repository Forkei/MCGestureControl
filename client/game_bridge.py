"""Bridge between gesture recognition and Minecraft via MCCTP WebSocket.

Maps gesture classifier output, look joystick, and strafe detector
to MCCTP action commands sent over WebSocket to the Minecraft mod.

Usage:
    bridge = GameBridge()
    bridge.connect()
    # In main loop:
    bridge.update(active_gestures, look_output, strafe_output)
    # On exit:
    bridge.disconnect()
"""

import time
from typing import List, Optional

from mcctp import SyncMCCTPClient, Actions

from look_joystick import LookOutput
from strafe_detector import StrafeOutput


class GameBridge:
    """Maps gesture/look/strafe outputs to Minecraft actions via MCCTP."""

    # Pulse action cooldown in seconds
    PULSE_COOLDOWN = 0.5

    # Look sensitivity: degrees per frame at full joystick deflection
    LOOK_YAW_SPEED = 5.0
    LOOK_PITCH_SPEED = 3.0

    def __init__(self, host: str = "localhost", port: int = 8765):
        self._client = SyncMCCTPClient(host, port)

        # Track held action states (only send on transitions)
        self._held = {
            "forward": False,
            "sprint": False,
            "sneak": False,
            "use_item": False,
            "strafe_left": False,
            "strafe_right": False,
        }

        # Pulse cooldowns: gesture_name -> last fire timestamp
        self._last_pulse: dict = {}

    def connect(self):
        self._client.connect()
        print("[GameBridge] Connected to MCCTP")

    def disconnect(self):
        self._release_all()
        self._client.disconnect()
        print("[GameBridge] Disconnected")

    @property
    def game_state(self):
        return self._client.state

    def update(
        self,
        active_gestures: List[str],
        look_output: Optional[LookOutput] = None,
        strafe_output: Optional[StrafeOutput] = None,
    ):
        """Process one frame of inputs and send MCCTP commands.

        Only sends commands on state transitions (start/stop) to avoid
        flooding the WebSocket with redundant messages.
        """
        gesture_set = set(active_gestures)
        now = time.monotonic()

        # --- Held states (send on transitions only) ---

        # Forward movement: walking OR sprinting
        wants_forward = "walking" in gesture_set or "sprinting" in gesture_set
        self._set_held("forward", wants_forward,
                       Actions.move("forward", "start"),
                       Actions.move("forward", "stop"))

        # Sprint
        self._set_held("sprint", "sprinting" in gesture_set,
                       Actions.sprint("start"),
                       Actions.sprint("stop"))

        # Sneak (crouch)
        self._set_held("sneak", "crouching" in gesture_set,
                       Actions.sneak("start"),
                       Actions.sneak("stop"))

        # Use item (draw bow = hold right-click)
        self._set_held("use_item", "draw_bow" in gesture_set,
                       Actions.use_item("start"),
                       Actions.use_item("stop"))

        # --- Strafe (from strafe detector) ---
        if strafe_output is not None:
            self._set_held("strafe_left", strafe_output.strafe_left,
                           Actions.move("left", "start"),
                           Actions.move("left", "stop"))
            self._set_held("strafe_right", strafe_output.strafe_right,
                           Actions.move("right", "start"),
                           Actions.move("right", "stop"))

        # --- Pulse actions (fire once with cooldown) ---

        pulse_map = {
            "jump": Actions.jump(),
            "swing": Actions.attack(),
            "throw": Actions.throw_item(),
        }

        for gesture, action in pulse_map.items():
            if gesture in gesture_set:
                if now - self._last_pulse.get(gesture, 0) >= self.PULSE_COOLDOWN:
                    self._send(action)
                    self._last_pulse[gesture] = now

        # Place block: quick right-click (only if draw_bow isn't holding use_item)
        if "place_block" in gesture_set and not self._held["use_item"]:
            if now - self._last_pulse.get("place_block", 0) >= self.PULSE_COOLDOWN:
                self._send(Actions.use_item("start"))
                self._send(Actions.use_item("stop"))
                self._last_pulse["place_block"] = now

        # Sword draw: select slot (TODO: find sword slot from game state)
        if "sword_draw" in gesture_set:
            if now - self._last_pulse.get("sword_draw", 0) >= self.PULSE_COOLDOWN:
                self._send(Actions.select_slot(0))
                self._last_pulse["sword_draw"] = now

        # --- Look (from joystick, every frame when active) ---
        if look_output is not None and look_output.active:
            if abs(look_output.yaw) > 0.01 or abs(look_output.pitch) > 0.01:
                yaw = look_output.yaw * self.LOOK_YAW_SPEED
                pitch = look_output.pitch * self.LOOK_PITCH_SPEED
                self._send(Actions.look(yaw=yaw, pitch=pitch, relative=True))

    def _set_held(self, name: str, wanted: bool, start_action: dict, stop_action: dict):
        """Send start/stop only on state transitions."""
        current = self._held[name]
        if wanted and not current:
            self._send(start_action)
            self._held[name] = True
        elif not wanted and current:
            self._send(stop_action)
            self._held[name] = False

    def _send(self, action: dict):
        """Send an action, silently ignoring errors."""
        try:
            self._client.send(action)
        except Exception:
            pass

    def _release_all(self):
        """Release all held actions on disconnect."""
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
