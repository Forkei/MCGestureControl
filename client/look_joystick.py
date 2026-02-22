"""Hand-based analog look controller (virtual joystick).

Uses the left hand as a virtual joystick for camera look control:
  INACTIVE -> (fist detected) -> ARMED -> (palm opens) -> ACTIVE

In ACTIVE state, displacement of the palm from the neutral point
(captured when the palm first opens) produces analog yaw/pitch values
in [-1, 1], suitable for driving in-game camera rotation.
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from hand_tracker import HandLandmarks
from gesture_recognizer import FingerAnalyzer


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class LookOutput:
    """Analog look output from the virtual joystick."""
    yaw: float = 0.0       # [-1, 1] horizontal: negative=left, positive=right
    pitch: float = 0.0     # [-1, 1] vertical: negative=up, positive=down
    active: bool = False
    state: str = "INACTIVE"


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class _State(Enum):
    INACTIVE = auto()
    ARMED = auto()
    ACTIVE = auto()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sign(x: float) -> float:
    """Return -1, 0, or 1 matching the sign of *x*."""
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# LookJoystick
# ---------------------------------------------------------------------------

class LookJoystick:
    """Left-hand virtual joystick for analog look control.

    Parameters
    ----------
    dead_zone : float
        Normalized displacement below which output is clamped to zero.
    max_range : float
        Displacement at which output saturates to +/-1.
    smoothing_alpha : float
        EMA smoothing factor. 1.0 = raw (no smoothing), 0.0 = maximum
        smoothing (very laggy).
    persist_frames : int
        Number of consecutive frames the left hand can be missing before
        the joystick deactivates.  At 30 fps, 15 frames ~ 0.5 s.
    sensitivity : float
        Output multiplier applied after the quadratic curve.
    """

    def __init__(
        self,
        dead_zone: float = 0.03,
        max_range: float = 0.25,
        smoothing_alpha: float = 0.4,
        persist_frames: int = 15,
        sensitivity: float = 1.0,
    ):
        self._dead_zone = dead_zone
        self._max_range = max_range
        self._alpha = smoothing_alpha
        self._persist_frames = persist_frames
        self._sensitivity = sensitivity

        # Internal state
        self._state = _State.INACTIVE
        self._neutral: Optional[Tuple[float, float]] = None  # (x, y) norm
        self._miss_counter: int = 0

        # Smoothed output (EMA state)
        self._smooth_yaw: float = 0.0
        self._smooth_pitch: float = 0.0

        # Last known left-hand palm position (for drawing when hand blinks)
        self._last_palm_norm: Optional[Tuple[float, float]] = None

        # Finger analyzer — reuses the same logic as gesture_recognizer
        self._finger_analyzer = FingerAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, hands: List[HandLandmarks]) -> LookOutput:
        """Process one frame of hand data and return analog look values.

        Parameters
        ----------
        hands : list[HandLandmarks]
            All hands detected this frame (both left and right).

        Returns
        -------
        LookOutput
            Analog yaw/pitch in [-1, 1] plus state information.
        """
        left_hand = self._find_left_hand(hands)

        if left_hand is not None:
            self._miss_counter = 0
            self._last_palm_norm = left_hand.palm_center_norm
        else:
            self._miss_counter += 1

        # --- State transitions ---

        if self._state == _State.INACTIVE:
            if left_hand is not None and self._is_fist(left_hand):
                self._state = _State.ARMED
            return self._inactive_output()

        if self._state == _State.ARMED:
            if self._miss_counter > self._persist_frames:
                self._deactivate()
                return self._inactive_output()
            if left_hand is not None:
                if self._is_open_palm(left_hand):
                    # Transition to ACTIVE: snapshot neutral point
                    self._neutral = left_hand.palm_center_norm
                    self._smooth_yaw = 0.0
                    self._smooth_pitch = 0.0
                    self._state = _State.ACTIVE
                    return LookOutput(
                        yaw=0.0, pitch=0.0, active=True, state="ACTIVE"
                    )
                if not self._is_fist(left_hand):
                    # Hand is neither fist nor open palm -> back to INACTIVE
                    self._deactivate()
                    return self._inactive_output()
            # Still armed (fist held, or brief blink)
            return LookOutput(yaw=0.0, pitch=0.0, active=False, state="ARMED")

        # --- _State.ACTIVE ---
        if self._miss_counter > self._persist_frames:
            self._deactivate()
            return self._inactive_output()

        if left_hand is not None and self._is_fist(left_hand):
            self._deactivate()
            return self._inactive_output()

        # Compute displacement from neutral
        if left_hand is not None:
            palm = left_hand.palm_center_norm
        elif self._last_palm_norm is not None:
            palm = self._last_palm_norm
        else:
            # Should not happen, but be safe
            self._deactivate()
            return self._inactive_output()

        yaw, pitch = self._compute_output(palm)
        return LookOutput(yaw=yaw, pitch=pitch, active=True, state="ACTIVE")

    @property
    def is_active(self) -> bool:
        """Whether the joystick is currently producing look output."""
        return self._state == _State.ACTIVE

    # ------------------------------------------------------------------
    # Drawing / visualization
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, hands: List[HandLandmarks]) -> np.ndarray:
        """Draw joystick visualization overlay onto *frame*.

        Draws different indicators depending on the current state:
        - INACTIVE: nothing
        - ARMED: yellow "ARMED" label near the left hand
        - ACTIVE: crosshair at neutral, circle at current position,
          connecting line, and yaw/pitch readout
        """
        h, w = frame.shape[:2]
        left_hand = self._find_left_hand(hands)

        if self._state == _State.ARMED:
            self._draw_armed(frame, left_hand, w, h)
        elif self._state == _State.ACTIVE:
            self._draw_active(frame, left_hand, w, h)

        return frame

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _find_left_hand(
        hands: List[HandLandmarks],
    ) -> Optional[HandLandmarks]:
        """Return the first hand labelled 'Left', or None."""
        for hand in hands:
            if hand.handedness == "Left":
                return hand
        return None

    def _is_fist(self, hand: HandLandmarks) -> bool:
        """True when at most 1 finger is extended (thumb can be ambiguous)."""
        extended = self._finger_analyzer.fingers_extended(
            hand.landmarks_norm, hand.handedness
        )
        return sum(extended) <= 1

    def _is_open_palm(self, hand: HandLandmarks) -> bool:
        """True when at least 4 fingers are extended."""
        extended = self._finger_analyzer.fingers_extended(
            hand.landmarks_norm, hand.handedness
        )
        return sum(extended) >= 4

    def _deactivate(self) -> None:
        """Reset to INACTIVE and clear joystick state."""
        self._state = _State.INACTIVE
        self._neutral = None
        self._smooth_yaw = 0.0
        self._smooth_pitch = 0.0
        self._miss_counter = 0

    @staticmethod
    def _inactive_output() -> LookOutput:
        return LookOutput(yaw=0.0, pitch=0.0, active=False, state="INACTIVE")

    def _compute_output(
        self, palm: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Map palm displacement to smoothed analog output in [-1, 1]."""
        nx, ny = self._neutral  # type: ignore[misc]
        dx = palm[0] - nx
        dy = palm[1] - ny

        # Dead zone
        if abs(dx) < self._dead_zone:
            dx = 0.0
        if abs(dy) < self._dead_zone:
            dy = 0.0

        # Clamp to max_range, then normalize to [-1, 1]
        mr = self._max_range
        dx = max(-mr, min(mr, dx)) / mr
        dy = max(-mr, min(mr, dy)) / mr

        # Quadratic sensitivity curve
        raw_yaw = _sign(dx) * dx * dx * self._sensitivity
        raw_pitch = _sign(dy) * dy * dy * self._sensitivity

        # Clamp after sensitivity multiplier
        raw_yaw = max(-1.0, min(1.0, raw_yaw))
        raw_pitch = max(-1.0, min(1.0, raw_pitch))

        # EMA smoothing
        a = self._alpha
        self._smooth_yaw = a * raw_yaw + (1.0 - a) * self._smooth_yaw
        self._smooth_pitch = a * raw_pitch + (1.0 - a) * self._smooth_pitch

        return self._smooth_yaw, self._smooth_pitch

    # ---- Drawing helpers ------------------------------------------------

    def _draw_armed(
        self,
        frame: np.ndarray,
        left_hand: Optional[HandLandmarks],
        w: int,
        h: int,
    ) -> None:
        """Draw a yellow ARMED indicator near the left hand."""
        color = (0, 220, 220)  # BGR yellow
        if left_hand is not None:
            px, py = left_hand.palm_center
            label_x = px + 15
            label_y = py - 15
        elif self._last_palm_norm is not None:
            label_x = int(self._last_palm_norm[0] * w) + 15
            label_y = int(self._last_palm_norm[1] * h) - 15
        else:
            return

        cv2.putText(
            frame,
            "ARMED",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def _draw_active(
        self,
        frame: np.ndarray,
        left_hand: Optional[HandLandmarks],
        w: int,
        h: int,
    ) -> None:
        """Draw crosshair, position circle, connecting line, and values."""
        if self._neutral is None:
            return

        color_active = (0, 255, 0)  # BGR green
        color_crosshair = (200, 200, 200)  # light gray
        crosshair_size = 10

        # Neutral point in pixels
        nx_px = int(self._neutral[0] * w)
        ny_px = int(self._neutral[1] * h)

        # Draw crosshair at neutral
        cv2.line(
            frame,
            (nx_px - crosshair_size, ny_px),
            (nx_px + crosshair_size, ny_px),
            color_crosshair,
            1,
        )
        cv2.line(
            frame,
            (nx_px, ny_px - crosshair_size),
            (nx_px, ny_px + crosshair_size),
            color_crosshair,
            1,
        )

        # Current position in pixels
        if left_hand is not None:
            cx_px, cy_px = left_hand.palm_center
        elif self._last_palm_norm is not None:
            cx_px = int(self._last_palm_norm[0] * w)
            cy_px = int(self._last_palm_norm[1] * h)
        else:
            return

        # Line from neutral to current
        cv2.line(frame, (nx_px, ny_px), (cx_px, cy_px), color_active, 2)

        # Circle at current position
        cv2.circle(frame, (cx_px, cy_px), 8, color_active, 2)

        # Yaw/pitch text
        text = f"yaw:{self._smooth_yaw:+.2f}  pitch:{self._smooth_pitch:+.2f}"
        text_x = nx_px + crosshair_size + 5
        text_y = ny_px - crosshair_size - 5
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_active,
            1,
        )
