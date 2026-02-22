"""Gesture event/callback system with per-gesture debouncing."""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from gesture_recognizer import (
    DynamicGesture, GestureResult, StaticGesture,
)
from config import GestureConfig


@dataclass
class GestureEvent:
    gesture: Union[StaticGesture, DynamicGesture]
    gesture_type: str  # "static" or "dynamic"
    handedness: str
    confidence: float
    timestamp: float


# Callback type: receives a GestureEvent
GestureCallback = Callable[[GestureEvent], None]


class GestureEventSystem:
    """Register callbacks for gesture events with debouncing."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        # Global callbacks (receive all events)
        self._global_callbacks: List[GestureCallback] = []
        # Per-gesture callbacks
        self._gesture_callbacks: Dict[Union[StaticGesture, DynamicGesture],
                                      List[GestureCallback]] = {}
        # Debounce tracking: (gesture, handedness) -> last_fire_time
        self._last_fired: Dict[Tuple[str, str], float] = {}

    def on(self, gesture: Union[StaticGesture, DynamicGesture],
           callback: GestureCallback):
        """Register a callback for a specific gesture."""
        if gesture not in self._gesture_callbacks:
            self._gesture_callbacks[gesture] = []
        self._gesture_callbacks[gesture].append(callback)

    def on_any(self, callback: GestureCallback):
        """Register a callback for all gesture events."""
        self._global_callbacks.append(callback)

    def process_results(self, results: List[GestureResult]):
        """Process gesture results and fire debounced events."""
        now = time.time()

        for result in results:
            # Static gesture events
            if result.static_gesture != StaticGesture.NONE:
                self._maybe_fire(
                    gesture=result.static_gesture,
                    gesture_type="static",
                    handedness=result.handedness,
                    confidence=result.static_confidence,
                    now=now,
                    debounce=self.config.debounce_static_sec,
                )

            # Dynamic gesture events
            if result.dynamic_gesture != DynamicGesture.NONE:
                self._maybe_fire(
                    gesture=result.dynamic_gesture,
                    gesture_type="dynamic",
                    handedness=result.handedness,
                    confidence=result.dynamic_confidence,
                    now=now,
                    debounce=self.config.debounce_dynamic_sec,
                )

    def _maybe_fire(self, gesture, gesture_type: str, handedness: str,
                    confidence: float, now: float, debounce: float):
        key = (gesture.name, handedness)
        last = self._last_fired.get(key, 0.0)

        if now - last < debounce:
            return

        self._last_fired[key] = now

        event = GestureEvent(
            gesture=gesture,
            gesture_type=gesture_type,
            handedness=handedness,
            confidence=confidence,
            timestamp=now,
        )

        # Fire specific callbacks
        for cb in self._gesture_callbacks.get(gesture, []):
            try:
                cb(event)
            except Exception as e:
                print(f"[GestureEvent] Callback error: {e}")

        # Fire global callbacks
        for cb in self._global_callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"[GestureEvent] Global callback error: {e}")

    def clear_callbacks(self):
        self._global_callbacks.clear()
        self._gesture_callbacks.clear()
        self._last_fired.clear()
