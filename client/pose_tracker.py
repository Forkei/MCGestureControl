"""Body pose tracking via MediaPipe Pose Landmarker (Tasks API).

Detects a single person's body pose (33 landmarks) and draws
the skeleton overlay. Runs alongside the existing hand tracker.

Requires the pose_landmarker_lite.task model file in client/models/.
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions

from config import PoseVisualizationConfig


# Default model path: client/models/pose_landmarker_full.task
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "models", "pose_landmarker_full.task")


@dataclass
class PoseLandmarks:
    """Structured single-person pose data."""
    # 33 landmarks in normalized [0,1] coords
    landmarks_norm: List[Tuple[float, float, float]]
    # 33 landmarks in pixel coords
    landmarks_px: List[Tuple[int, int]]
    # Per-landmark visibility scores
    visibility: List[float]
    # 33 landmarks in world coordinates (meters, hip-centered)
    world_landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    # Bounding box computed from visible landmarks (x, y, w, h)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)


# MediaPipe body skeleton connections (33-landmark topology)
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),    # left eye
    (0, 4), (4, 5), (5, 6), (6, 8),    # right eye
    (9, 10),                             # mouth
    # Torso
    (11, 12),                            # shoulders
    (11, 23), (12, 24),                  # shoulders to hips
    (23, 24),                            # hips
    # Left arm
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27),
    (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28),
    (28, 30), (28, 32), (30, 32),
]

# Minimum visibility to consider a landmark "visible" for bounding box / drawing
_VIS_THRESHOLD = 0.5


class PoseTracker:
    """Wraps MediaPipe PoseLandmarker for single-person body pose detection."""

    def __init__(self, config: Optional[PoseVisualizationConfig] = None,
                 model_path: Optional[str] = None):
        self.config = config or PoseVisualizationConfig()

        model = model_path or _DEFAULT_MODEL_PATH
        if not os.path.isfile(model):
            raise FileNotFoundError(
                f"Pose model not found at {model}. "
                "Download pose_landmarker_lite.task into client/models/."
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_presence_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_ts = 0  # monotonic timestamp in ms for VIDEO mode

    def process(self, frame_bgr: np.ndarray) -> Optional[PoseLandmarks]:
        """Process a BGR frame and return pose landmarks if detected."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33  # ~30fps step; must be strictly increasing
        results = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not results.pose_landmarks:
            return None

        pose_lms = results.pose_landmarks[0]  # first (only) person

        landmarks_norm = []
        landmarks_px = []
        visibility = []

        for lm in pose_lms:
            landmarks_norm.append((lm.x, lm.y, lm.z))
            landmarks_px.append((int(round(lm.x * w)), int(round(lm.y * h))))
            vis = lm.visibility if lm.visibility is not None else 0.0
            visibility.append(vis)

        # Extract world landmarks (meters, hip-centered)
        world_landmarks = []
        if results.pose_world_landmarks:
            for lm in results.pose_world_landmarks[0]:
                world_landmarks.append((lm.x, lm.y, lm.z))

        # Compute bounding box from visible landmarks
        visible_pts = [
            landmarks_px[i]
            for i in range(len(landmarks_px))
            if visibility[i] > _VIS_THRESHOLD
        ]

        if visible_pts:
            xs = [p[0] for p in visible_pts]
            ys = [p[1] for p in visible_pts]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            margin = 10
            bx = max(0, x1 - margin)
            by = max(0, y1 - margin)
            bw = min(w, x2 + margin) - bx
            bh = min(h, y2 + margin) - by
            bbox = (bx, by, bw, bh)
        else:
            bbox = (0, 0, 0, 0)

        return PoseLandmarks(
            landmarks_norm=landmarks_norm,
            landmarks_px=landmarks_px,
            visibility=visibility,
            world_landmarks=world_landmarks,
            bounding_box=bbox,
        )

    def draw(self, frame: np.ndarray, pose: PoseLandmarks) -> np.ndarray:
        """Draw body skeleton and landmarks on frame."""
        # Draw connections
        for i, j in POSE_CONNECTIONS:
            if (pose.visibility[i] > _VIS_THRESHOLD
                    and pose.visibility[j] > _VIS_THRESHOLD):
                cv2.line(frame, pose.landmarks_px[i], pose.landmarks_px[j],
                         self.config.connection_color,
                         self.config.connection_thickness)

        # Draw landmark dots
        for idx, (px, py) in enumerate(pose.landmarks_px):
            if pose.visibility[idx] > _VIS_THRESHOLD:
                cv2.circle(frame, (px, py),
                           self.config.landmark_radius,
                           self.config.landmark_color, -1)

        return frame

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()
