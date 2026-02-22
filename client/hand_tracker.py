"""Hand tracking via WiLoR-mini (YOLO detector + MANO 3D reconstruction).

Outputs 21 keypoints in OpenPose/MediaPipe ordering — same indices as before,
so gesture_recognizer.py needs no changes.

Models auto-download from HuggingFace on first run.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import logging
import warnings

import cv2
import numpy as np
import torch

from config import HandTrackingConfig, VisualizationConfig


@dataclass
class HandLandmarks:
    """Structured per-hand landmark data."""
    # 21 landmarks in normalized [0,1] coords
    landmarks_norm: List[Tuple[float, float, float]]
    # 21 landmarks in pixel coords
    landmarks_px: List[Tuple[int, int]]
    # "Left" or "Right"
    handedness: str
    handedness_score: float
    # Derived
    palm_center: Tuple[int, int] = (0, 0)
    palm_center_norm: Tuple[float, float] = (0.0, 0.0)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x,y,w,h
    # 21 landmarks in 3D space (from WiLoR pred_keypoints_3d)
    landmarks_3d: List[Tuple[float, float, float]] = field(default_factory=list)


class HandTracker:
    """Wraps WiLoR-mini pipeline for hand detection + landmark estimation."""

    # Hand connections for drawing skeleton (OpenPose/MediaPipe ordering)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # index
        (0, 9), (9, 10), (10, 11), (11, 12),   # middle
        (0, 13), (13, 14), (14, 15), (15, 16), # ring
        (0, 17), (17, 18), (18, 19), (19, 20), # pinky
        (5, 9), (9, 13), (13, 17),              # palm
    ]

    def __init__(self, config: Optional[HandTrackingConfig] = None,
                 viz_config: Optional[VisualizationConfig] = None):
        self.config = config or HandTrackingConfig()
        self.viz = viz_config or VisualizationConfig()

        # Suppress WiLoR/ultralytics/MANO log spam
        logging.getLogger("wilor_mini").setLevel(logging.WARNING)
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*MANO model.*")

        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Suppress stray print() calls from smplx/MANO during model init
        import io, sys
        _real_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self._pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
        finally:
            sys.stdout = _real_stdout
        self._pipe.hand_detector.verbose = False  # suppress YOLO per-frame output

        self._process_fps = 0.0
        self._frame_times: List[float] = []

        # Landmark smoothing: per-hand EMA state (keyed by handedness)
        self._smoothed: Dict[str, List[Tuple[float, float, float]]] = {}

        # Hand persistence: last-seen HandLandmarks + miss counter (keyed by handedness)
        self._last_seen: Dict[str, Tuple[HandLandmarks, int]] = {}

    # If the wrist jumps more than this (normalized coords), reset EMA
    # to avoid cross-hand blending when handedness labels swap.
    _SMOOTH_RESET_THRESHOLD = 0.15

    def _smooth_landmarks(self, handedness: str,
                          raw_norm: List[Tuple[float, float, float]],
                          w: int, h: int) -> Tuple[List[Tuple[float, float, float]],
                                                    List[Tuple[int, int]]]:
        """Apply EMA smoothing to normalized landmarks and recompute pixel coords."""
        alpha = self.config.landmark_smoothing_alpha
        if handedness in self._smoothed:
            prev = self._smoothed[handedness]
            # Distance check: if wrist jumped too far, the label likely swapped
            # between physical hands — reset instead of blending.
            dx = raw_norm[0][0] - prev[0][0]
            dy = raw_norm[0][1] - prev[0][1]
            if (dx * dx + dy * dy) > self._SMOOTH_RESET_THRESHOLD ** 2:
                smoothed = list(raw_norm)
            else:
                smoothed = [
                    (alpha * r[0] + (1 - alpha) * p[0],
                     alpha * r[1] + (1 - alpha) * p[1],
                     alpha * r[2] + (1 - alpha) * p[2])
                    for r, p in zip(raw_norm, prev)
                ]
        else:
            smoothed = list(raw_norm)
        self._smoothed[handedness] = smoothed

        # Recompute pixel coords from smoothed norms
        px_list = [(int(round(nx * w)), int(round(ny * h))) for nx, ny, _ in smoothed]
        return smoothed, px_list

    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """Process a BGR frame and return detected hand landmarks."""
        t0 = time.time()
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        outputs = self._pipe.predict(rgb)

        detected_hands: Dict[str, HandLandmarks] = {}
        for out in outputs:
            kp2d = out["wilor_preds"]["pred_keypoints_2d"]  # (1, 21, 2)
            if hasattr(kp2d, 'cpu'):
                kp2d = kp2d.cpu().numpy()
            kp2d = kp2d[0]  # (21, 2) pixel coords

            is_right = bool(out["is_right"])
            handedness = "Right" if is_right else "Left"
            bbox = out["hand_bbox"]  # [x1, y1, x2, y2]

            # Raw normalized landmarks
            raw_norm = []
            for x_px, y_px in kp2d:
                raw_norm.append((float(x_px) / w, float(y_px) / h, 0.0))

            # Extract 3D keypoints
            kp3d_raw = out["wilor_preds"].get("pred_keypoints_3d")
            if kp3d_raw is not None:
                if hasattr(kp3d_raw, 'cpu'):
                    kp3d_raw = kp3d_raw.cpu().numpy()
                landmarks_3d = [(float(x), float(y), float(z))
                                for x, y, z in kp3d_raw[0]]
            else:
                landmarks_3d = []

            # Apply EMA smoothing
            landmarks_norm, landmarks_px = self._smooth_landmarks(handedness, raw_norm, w, h)

            # Palm center: average of wrist(0) and middle finger MCP(9)
            wrist = landmarks_px[0]
            mcp = landmarks_px[9]
            palm_px = ((wrist[0] + mcp[0]) // 2, (wrist[1] + mcp[1]) // 2)
            palm_norm = (
                (landmarks_norm[0][0] + landmarks_norm[9][0]) / 2,
                (landmarks_norm[0][1] + landmarks_norm[9][1]) / 2,
            )

            # Bounding box from detector
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            margin = 10
            bx = max(0, x1 - margin)
            by = max(0, y1 - margin)
            bw = min(w, x2 + margin) - bx
            bh = min(h, y2 + margin) - by

            hand = HandLandmarks(
                landmarks_norm=landmarks_norm,
                landmarks_px=landmarks_px,
                handedness=handedness,
                handedness_score=1.0,
                palm_center=palm_px,
                palm_center_norm=palm_norm,
                bounding_box=(bx, by, bw, bh),
                landmarks_3d=landmarks_3d,
            )
            detected_hands[handedness] = hand

        # --- Hand persistence ---
        detected_set = set(detected_hands.keys())

        # Update _last_seen for detected hands (reset miss counter)
        for hkey, hand in detected_hands.items():
            self._last_seen[hkey] = (hand, 0)

        # For hands not detected this frame, increment miss counter
        for hkey in list(self._last_seen.keys()):
            if hkey not in detected_set:
                prev_hand, miss_count = self._last_seen[hkey]
                miss_count += 1
                if miss_count >= self.config.hand_persist_frames:
                    del self._last_seen[hkey]
                    self._smoothed.pop(hkey, None)
                else:
                    self._last_seen[hkey] = (prev_hand, miss_count)

        # Build final output: all hands still in _last_seen
        hands_data = [hand for hand, _ in self._last_seen.values()]

        # Track processing FPS
        dt = time.time() - t0
        self._frame_times.append(dt)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg = sum(self._frame_times) / len(self._frame_times)
        self._process_fps = 1.0 / avg if avg > 0 else 0

        return hands_data

    def draw(self, frame: np.ndarray, hands: List[HandLandmarks]) -> np.ndarray:
        """Draw hand skeletons, landmarks, and bounding boxes on frame."""
        for hand in hands:
            # Draw connections
            for i, j in self.HAND_CONNECTIONS:
                pt1 = hand.landmarks_px[i]
                pt2 = hand.landmarks_px[j]
                cv2.line(frame, pt1, pt2,
                         self.viz.connection_color, self.viz.connection_thickness)

            # Draw landmark dots
            for px, py in hand.landmarks_px:
                cv2.circle(frame, (px, py),
                           self.viz.landmark_radius, self.viz.landmark_color, -1)

            # Draw bounding box
            bx, by, bw, bh = hand.bounding_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                          self.viz.bbox_color, self.viz.bbox_thickness)

            # Draw handedness label
            label = hand.handedness
            cv2.putText(frame, label, (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.viz.bbox_color, 1)

        return frame

    @property
    def processing_fps(self) -> float:
        return self._process_fps

    def close(self):
        pass  # WiLoR pipeline doesn't need explicit cleanup
