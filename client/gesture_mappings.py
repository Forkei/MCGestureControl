"""Maps gesture events to camera control commands."""

from gesture_event import GestureEvent, GestureEventSystem
from gesture_recognizer import DynamicGesture, StaticGesture


class GestureCameraMapper:
    """Connects gesture events to CameraControl commands.

    Default mappings:
        Pinch  -> zoom out (0.5x step)
        Spread -> zoom in (0.5x step)
        Swipe Up   -> EV compensation +2
        Swipe Down -> EV compensation -2
    """

    def __init__(self, ctrl, state: dict, event_system: GestureEventSystem):
        self.ctrl = ctrl
        self.state = state
        self.enabled = False

        self._zoom_val = state.get("zoom", 1.0)
        self._ev_val = state.get("exposure_compensation", 0)

        # Register callbacks
        event_system.on(DynamicGesture.PINCH, self._on_pinch)
        event_system.on(DynamicGesture.SPREAD, self._on_spread)
        event_system.on(DynamicGesture.SWIPE_UP, self._on_swipe_up)
        event_system.on(DynamicGesture.SWIPE_DOWN, self._on_swipe_down)

    def toggle(self):
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        print(f"[Gesture Control] {status}")
        return self.enabled

    def _on_pinch(self, event: GestureEvent):
        if not self.enabled:
            return
        self._zoom_val = max(1.0, self._zoom_val - 0.5)
        self.ctrl.set_zoom(self._zoom_val)
        self.state["zoom"] = self._zoom_val
        print(f"[Gesture] Pinch -> Zoom {self._zoom_val:.1f}x")

    def _on_spread(self, event: GestureEvent):
        if not self.enabled:
            return
        self._zoom_val = min(8.0, self._zoom_val + 0.5)
        self.ctrl.set_zoom(self._zoom_val)
        self.state["zoom"] = self._zoom_val
        print(f"[Gesture] Spread -> Zoom {self._zoom_val:.1f}x")

    def _on_swipe_up(self, event: GestureEvent):
        if not self.enabled:
            return
        if self.state.get("ae_mode") != "auto":
            return
        self._ev_val = min(self._ev_val + 2, 20)
        self.ctrl.set_ev(self._ev_val)
        self.state["exposure_compensation"] = self._ev_val
        print(f"[Gesture] Swipe Up -> EV {self._ev_val:+d}")

    def _on_swipe_down(self, event: GestureEvent):
        if not self.enabled:
            return
        if self.state.get("ae_mode") != "auto":
            return
        self._ev_val = max(self._ev_val - 2, -20)
        self.ctrl.set_ev(self._ev_val)
        self.state["exposure_compensation"] = self._ev_val
        print(f"[Gesture] Swipe Down -> EV {self._ev_val:+d}")

    def sync_state(self):
        """Sync internal values from state dict (after manual keyboard changes)."""
        self._zoom_val = self.state.get("zoom", 1.0)
        self._ev_val = self.state.get("exposure_compensation", 0)
