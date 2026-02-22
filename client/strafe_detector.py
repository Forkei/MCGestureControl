"""Body lean-based binary strafe detector.

Detects lateral upper-body lean using MediaPipe world landmarks
(3D coordinates in meters, hip-centered) and outputs left/right
strafe booleans suitable for driving A/D key presses in a
Minecraft-like game.

The detector compares shoulder-center X to hip-center X in world
coordinates, applies EMA smoothing and hysteresis thresholding to
produce stable, jitter-free strafe signals.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from pose_tracker import PoseLandmarks


# MediaPipe pose landmark indices
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24

# Indices we need to be visible for a confident reading
_REQUIRED_LANDMARKS = (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)

# Number of frames to hold previous strafe state when pose is lost
_DROPOUT_HOLD_FRAMES = 5


@dataclass
class StrafeOutput:
    """Result of a single strafe detection update."""
    strafe_left: bool = False
    strafe_right: bool = False
    lean_value: float = 0.0       # raw smoothed lean in meters (for debugging)
    confident: bool = False        # True if landmarks are visible enough to trust


class StrafeDetector:
    """Detects left/right strafe intention from upper-body lean.

    Uses MediaPipe world landmarks (meters, hip-centered) to compute
    the lateral offset between shoulder center and hip center.  A
    positive lean means the user is leaning right; negative means left.

    Hysteresis thresholding prevents rapid toggling near the threshold
    boundary, and EMA smoothing reduces frame-to-frame jitter.
    """

    def __init__(
        self,
        lean_threshold: float = 0.04,
        hysteresis: float = 0.01,
        smoothing_alpha: float = 0.5,
        min_visibility: float = 0.5,
    ):
        """
        Args:
            lean_threshold: Meters of lateral lean required to trigger strafe.
            hysteresis: Meters of hysteresis band for release.  The release
                threshold is ``lean_threshold - hysteresis``.
            smoothing_alpha: Weight of the new sample in the EMA filter
                (0 = ignore new, 1 = no smoothing).
            min_visibility: Minimum per-landmark visibility score (0-1) to
                consider the reading trustworthy.
        """
        self._threshold = lean_threshold
        self._hysteresis = hysteresis
        self._alpha = smoothing_alpha
        self._min_vis = min_visibility

        # Internal state
        self._smoothed_lean: float = 0.0
        self._strafe_left: bool = False
        self._strafe_right: bool = False
        self._frames_without_pose: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, pose: Optional[PoseLandmarks]) -> StrafeOutput:
        """Process one frame of pose data and return strafe state.

        Args:
            pose: PoseLandmarks from PoseTracker.process(), or None if
                no pose was detected this frame.

        Returns:
            StrafeOutput with left/right booleans and diagnostics.
        """

        # ----- Handle missing pose -----
        if pose is None:
            return self._handle_missing_pose()

        # ----- Visibility check -----
        if not self._landmarks_visible(pose):
            return self._handle_missing_pose()

        # Pose is present and confident -- reset dropout counter
        self._frames_without_pose = 0

        # ----- Compute raw lean -----
        raw_lean = self._compute_lean(pose)

        # ----- EMA smoothing -----
        self._smoothed_lean = (
            self._alpha * raw_lean
            + (1.0 - self._alpha) * self._smoothed_lean
        )

        # ----- Hysteresis thresholding -----
        release_threshold = self._threshold - self._hysteresis

        # Left strafe (lean is negative when leaning left)
        if self._strafe_left:
            if self._smoothed_lean > -release_threshold:
                self._strafe_left = False
        else:
            if self._smoothed_lean < -self._threshold:
                self._strafe_left = True

        # Right strafe (lean is positive when leaning right)
        if self._strafe_right:
            if self._smoothed_lean < release_threshold:
                self._strafe_right = False
        else:
            if self._smoothed_lean > self._threshold:
                self._strafe_right = True

        return StrafeOutput(
            strafe_left=self._strafe_left,
            strafe_right=self._strafe_right,
            lean_value=self._smoothed_lean,
            confident=True,
        )

    def draw(self, frame: np.ndarray, pose: Optional[PoseLandmarks]) -> np.ndarray:
        """Draw a strafe indicator overlay on the frame.

        Renders a horizontal bar near the bottom-center of the frame with
        a marker showing the current lean value.  The marker turns green
        when a strafe is active and shows directional text.

        Args:
            frame: BGR image to draw on (modified in-place and returned).
            pose: Current pose data (used only for context; the detector's
                internal state drives the visual).

        Returns:
            The frame with overlay drawn.
        """
        h, w = frame.shape[:2]

        # Bar geometry
        bar_width = min(300, w // 2)
        bar_height = 14
        bar_x = (w - bar_width) // 2
        bar_y = h - 50

        # Background bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (60, 60, 60),
            -1,
        )

        # Threshold markers (thin vertical lines at activation points)
        for sign in (-1, 1):
            tx = int(bar_x + bar_width / 2 + sign * self._threshold * self._lean_to_px(bar_width))
            tx = max(bar_x, min(bar_x + bar_width, tx))
            cv2.line(
                frame,
                (tx, bar_y),
                (tx, bar_y + bar_height),
                (120, 120, 120),
                1,
            )

        # Marker position (lean mapped onto bar)
        marker_x = int(
            bar_x + bar_width / 2
            + self._smoothed_lean * self._lean_to_px(bar_width)
        )
        marker_x = max(bar_x + 4, min(bar_x + bar_width - 4, marker_x))
        marker_cy = bar_y + bar_height // 2

        # Marker color
        if self._strafe_left or self._strafe_right:
            color = (0, 220, 0)  # green when active
        else:
            color = (160, 160, 160)  # gray when idle

        cv2.circle(frame, (marker_x, marker_cy), 6, color, -1)

        # Directional text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        if self._strafe_left:
            cv2.putText(
                frame,
                "< STRAFE",
                (bar_x - 100, bar_y + bar_height - 2),
                font,
                font_scale,
                (0, 220, 0),
                thickness,
            )
        if self._strafe_right:
            cv2.putText(
                frame,
                "STRAFE >",
                (bar_x + bar_width + 10, bar_y + bar_height - 2),
                font,
                font_scale,
                (0, 220, 0),
                thickness,
            )

        # Confidence indicator
        if not self._is_confident():
            cv2.putText(
                frame,
                "LOW CONF",
                (bar_x + bar_width // 2 - 35, bar_y - 6),
                font,
                0.4,
                (0, 0, 200),
                1,
            )

        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _landmarks_visible(self, pose: PoseLandmarks) -> bool:
        """Return True if all required landmarks meet the visibility threshold."""
        if not pose.world_landmarks:
            return False
        if len(pose.visibility) <= max(_REQUIRED_LANDMARKS):
            return False
        return all(
            pose.visibility[idx] > self._min_vis for idx in _REQUIRED_LANDMARKS
        )

    @staticmethod
    def _compute_lean(pose: PoseLandmarks) -> float:
        """Compute raw lateral lean (meters) from world landmarks.

        Returns:
            Positive value = leaning right, negative = leaning left.
        """
        lh = pose.world_landmarks[_LEFT_HIP]
        rh = pose.world_landmarks[_RIGHT_HIP]
        ls = pose.world_landmarks[_LEFT_SHOULDER]
        rs = pose.world_landmarks[_RIGHT_SHOULDER]

        hip_center_x = (lh[0] + rh[0]) / 2.0
        shoulder_center_x = (ls[0] + rs[0]) / 2.0

        return shoulder_center_x - hip_center_x

    def _handle_missing_pose(self) -> StrafeOutput:
        """Handle a frame where pose is absent or unreliable."""
        self._frames_without_pose += 1

        if self._frames_without_pose > _DROPOUT_HOLD_FRAMES:
            # Too many consecutive frames without a pose -- release strafe
            self._strafe_left = False
            self._strafe_right = False

        return StrafeOutput(
            strafe_left=self._strafe_left,
            strafe_right=self._strafe_right,
            lean_value=self._smoothed_lean,
            confident=False,
        )

    def _is_confident(self) -> bool:
        """Return True if the most recent reading was confident."""
        return self._frames_without_pose == 0

    def _lean_to_px(self, bar_width: int) -> float:
        """Scaling factor: lean (meters) -> pixels on the indicator bar.

        Maps a lean of ``threshold * 2`` to the full half-width of the bar
        so the marker stays within bounds for typical lean ranges.
        """
        if self._threshold == 0:
            return 0.0
        return (bar_width / 2.0) / (self._threshold * 2.0)
