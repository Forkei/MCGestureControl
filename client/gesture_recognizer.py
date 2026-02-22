"""Core gesture recognition engine: static poses and dynamic gestures."""

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field as dataclass_field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from hand_tracker import HandLandmarks
from config import GestureConfig, StaticGestureDef


class StaticGesture(Enum):
    NONE = auto()
    OPEN_PALM = auto()
    FIST = auto()
    THUMBS_UP = auto()
    THUMBS_DOWN = auto()
    PEACE = auto()
    POINTING = auto()
    OK_SIGN = auto()
    ROCK = auto()
    THREE = auto()
    FOUR = auto()
    PINKY_UP = auto()


class DynamicGesture(Enum):
    NONE = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    PINCH = auto()
    SPREAD = auto()
    CIRCLE_CW = auto()
    CIRCLE_CCW = auto()
    WAVE = auto()


@dataclass
class GestureResult:
    hand_index: int
    handedness: str
    static_gesture: StaticGesture = StaticGesture.NONE
    static_confidence: float = 0.0
    dynamic_gesture: DynamicGesture = DynamicGesture.NONE
    dynamic_confidence: float = 0.0


# --- Landmark indices ---
# Thumb: 1=CMC, 2=MCP, 3=IP, 4=TIP
# Index: 5=MCP, 6=PIP, 7=DIP, 8=TIP
# Middle: 9=MCP, 10=PIP, 11=DIP, 12=TIP
# Ring: 13=MCP, 14=PIP, 15=DIP, 16=TIP
# Pinky: 17=MCP, 18=PIP, 19=DIP, 20=TIP
WRIST = 0
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]  # IP for thumb, PIP for others
FINGER_MCPS = [2, 5, 9, 13, 17]
FINGER_BASES = [1, 5, 9, 13, 17]  # CMC for thumb, MCP for others


def _angle_3pts(a: Tuple[float, ...], b: Tuple[float, ...],
                c: Tuple[float, ...]) -> float:
    """Angle at point b formed by segments a-b and b-c, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba * mag_bc < 1e-8:
        return 180.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


class FingerAnalyzer:
    """Geometric analysis of hand landmarks for finger state detection."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()

    def finger_angles(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """Return PIP joint angles for each finger (thumb uses IP)."""
        angles = []
        for i in range(5):
            base = landmarks[FINGER_BASES[i]]
            pip = landmarks[FINGER_PIPS[i]]
            tip = landmarks[FINGER_TIPS[i]]
            angles.append(_angle_3pts(base, pip, tip))
        return angles

    def fingers_extended(self, landmarks: List[Tuple[float, float, float]],
                         handedness: str) -> List[bool]:
        """Return which fingers are extended [thumb, index, middle, ring, pinky]."""
        extended = []

        # Thumb: check if tip is laterally away from palm
        # For right hand, thumb tip should be left of thumb IP (x decreases)
        # For left hand, thumb tip should be right of thumb IP
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]

        # Use lateral distance relative to palm width
        palm_width = _distance(landmarks[5], landmarks[17])
        if palm_width < 1e-6:
            palm_width = 0.1

        # Thumb extended: tip is further from palm center than IP
        palm_center_x = (landmarks[0][0] + landmarks[9][0]) / 2
        tip_dist = abs(thumb_tip[0] - palm_center_x)
        ip_dist = abs(thumb_ip[0] - palm_center_x)
        thumb_extended = tip_dist > ip_dist and (tip_dist - ip_dist) / palm_width > self.config.thumb_extend_min_ratio
        extended.append(thumb_extended)

        # Other fingers: tip above PIP (in image coords, y decreases upward)
        # Also check angle at PIP joint as secondary confirmation
        for i in range(1, 5):
            tip = landmarks[FINGER_TIPS[i]]
            pip = landmarks[FINGER_PIPS[i]]
            base = landmarks[FINGER_BASES[i]]

            # Primary check: tip is further from wrist than PIP
            wrist = landmarks[WRIST]
            tip_to_wrist = _distance(tip, wrist)
            pip_to_wrist = _distance(pip, wrist)
            distance_check = tip_to_wrist > pip_to_wrist

            # Secondary check: PIP joint angle (straight = extended)
            pip_angle = _angle_3pts(base, pip, tip)
            angle_check = pip_angle > self.config.finger_extend_threshold

            # Either passing = extended (a truly curled finger fails both)
            is_extended = distance_check or angle_check

            extended.append(is_extended)

        return extended

    def thumb_index_distance(self, landmarks: List[Tuple[float, float, float]]) -> float:
        """Normalized distance between thumb tip and index tip."""
        return _distance(landmarks[4], landmarks[8])


class StaticGestureClassifier:
    """Config-driven classification of static hand poses."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        self.finger_analyzer = FingerAnalyzer(config)
        # Pre-sort gesture defs by priority (descending) for first-match-wins
        self._gesture_defs: List[StaticGestureDef] = sorted(
            self.config.static_gestures, key=lambda g: g.priority, reverse=True
        )

    def classify(self, hand: HandLandmarks) -> Tuple[StaticGesture, float]:
        lm = hand.landmarks_norm
        extended = self.finger_analyzer.fingers_extended(lm, hand.handedness)
        thumb_index_dist = self.finger_analyzer.thumb_index_distance(lm)
        angles = self.finger_analyzer.finger_angles(lm)

        for gdef in self._gesture_defs:
            if not self._matches(gdef, extended, lm, thumb_index_dist):
                continue
            try:
                gesture = StaticGesture[gdef.name]
            except KeyError:
                continue
            confidence = self._compute_confidence(gdef, angles)
            return gesture, confidence

        return StaticGesture.NONE, 0.0

    def _compute_confidence(self, gdef: StaticGestureDef,
                            angles: List[float]) -> float:
        """Compute confidence based on how far finger angles are from thresholds.

        Returns a value in [0.5, 1.0]: borderline matches get ~0.5,
        strong matches get ~1.0.
        """
        curl_thresh = self.config.finger_curl_threshold
        extend_thresh = self.config.finger_extend_threshold
        midpoint = (curl_thresh + extend_thresh) / 2.0
        half_range = (extend_thresh - curl_thresh) / 2.0
        if half_range < 1e-6:
            return gdef.confidence

        margins = []
        for i, required in enumerate(gdef.fingers):
            if required is None:
                continue
            angle = angles[i]
            if required:  # finger should be extended
                margin = (angle - midpoint) / half_range
            else:  # finger should be curled
                margin = (midpoint - angle) / half_range
            margins.append(max(0.0, min(1.0, margin)))

        if not margins:
            return gdef.confidence

        avg_margin = sum(margins) / len(margins)
        # Map [0, 1] margin to [0.5, 1.0] confidence
        return 0.5 + 0.5 * avg_margin

    def _matches(self, gdef: StaticGestureDef, extended: List[bool],
                 lm: List[Tuple[float, float, float]],
                 thumb_index_dist: float) -> bool:
        # Check finger pattern
        for i, required in enumerate(gdef.fingers):
            if required is not None and extended[i] != required:
                return False

        # Check thumb-index closeness
        close_dist = self.config.thumb_index_close_dist
        if gdef.thumb_index_close is True and thumb_index_dist >= close_dist:
            return False
        if gdef.thumb_index_close is False and thumb_index_dist < close_dist:
            return False

        # Check thumb direction using vector angle (accounts for hand tilt)
        if gdef.thumb_direction is not None:
            thumb_tip = lm[4]
            thumb_mcp = lm[2]
            dx = thumb_tip[0] - thumb_mcp[0]
            dy = thumb_tip[1] - thumb_mcp[1]
            angle_deg = math.degrees(math.atan2(dy, dx))
            # In image coords: negative y = upward, positive y = downward
            # "up" requires angle < -30 (pointing upward with 30-degree tolerance)
            # "down" requires angle > 30 (pointing downward with 30-degree tolerance)
            if gdef.thumb_direction == "up" and angle_deg >= -30:
                return False
            if gdef.thumb_direction == "down" and angle_deg <= 30:
                return False

        return True


@dataclass
class _HandHistory:
    """Per-hand rolling buffers for dynamic gesture detection."""
    swipe_positions: deque   # (x_norm, y_norm, timestamp) — swipe only
    circle_positions: deque  # (x_norm, y_norm, timestamp) — circle only
    wave_positions: deque    # (x_norm, y_norm, timestamp) — wave only
    pinch_distances: deque   # (distance, timestamp)
    cooldowns: Dict[str, float] = dataclass_field(default_factory=dict)  # gesture_key -> expiry timestamp


class DynamicGestureDetector:
    """Detects dynamic gestures from landmark motion over time."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        self.finger_analyzer = FingerAnalyzer(config)
        self._histories: Dict[str, _HandHistory] = {}  # keyed by handedness

    def _get_history(self, handedness: str) -> _HandHistory:
        if handedness not in self._histories:
            max_frames = self.config.max_history_frames
            self._histories[handedness] = _HandHistory(
                swipe_positions=deque(maxlen=max_frames),
                circle_positions=deque(maxlen=max_frames),
                wave_positions=deque(maxlen=max_frames),
                pinch_distances=deque(maxlen=max_frames),
            )
        return self._histories[handedness]

    def _is_on_cooldown(self, hist: _HandHistory, gesture_key: str, now: float) -> bool:
        """Check if a gesture category is still in cooldown."""
        return now < hist.cooldowns.get(gesture_key, 0.0)

    def _clear_after_fire(self, gesture_key: str, hist: _HandHistory, now: float):
        """Clear the relevant history buffer and enter cooldown after a gesture fires."""
        cooldown = self.config.dynamic_gesture_cooldown_sec
        hist.cooldowns[gesture_key] = now + cooldown
        buffers = {
            "pinch_spread": hist.pinch_distances,
            "swipe": hist.swipe_positions,
            "circle": hist.circle_positions,
            "wave": hist.wave_positions,
        }
        buf = buffers.get(gesture_key)
        if buf is not None:
            buf.clear()

    def update(self, hand: HandLandmarks,
               static_gesture: 'StaticGesture' = None) -> Tuple[DynamicGesture, float]:
        """Update history and check for dynamic gestures."""
        now = time.time()
        hist = self._get_history(hand.handedness)
        lm = hand.landmarks_norm

        # Record palm position into each gesture's own buffer
        pos = (hand.palm_center_norm[0], hand.palm_center_norm[1], now)
        hist.swipe_positions.append(pos)
        hist.circle_positions.append(pos)
        hist.wave_positions.append(pos)

        # Record pinch distance
        pinch_dist = self.finger_analyzer.thumb_index_distance(lm)
        hist.pinch_distances.append((pinch_dist, now))

        # Check gestures in priority order, respecting cooldowns
        if not self._is_on_cooldown(hist, "pinch_spread", now):
            result = self._check_pinch_spread(hist, static_gesture)
            if result[0] != DynamicGesture.NONE:
                self._clear_after_fire("pinch_spread", hist, now)
                return result

        if not self._is_on_cooldown(hist, "swipe", now):
            result = self._check_swipe(hist)
            if result[0] != DynamicGesture.NONE:
                self._clear_after_fire("swipe", hist, now)
                return result

        if not self._is_on_cooldown(hist, "circle", now):
            result = self._check_circle(hist)
            if result[0] != DynamicGesture.NONE:
                self._clear_after_fire("circle", hist, now)
                return result

        if not self._is_on_cooldown(hist, "wave", now):
            result = self._check_wave(hist)
            if result[0] != DynamicGesture.NONE:
                self._clear_after_fire("wave", hist, now)
                return result

        return DynamicGesture.NONE, 0.0

    def _check_swipe(self, hist: _HandHistory) -> Tuple[DynamicGesture, float]:
        positions = hist.swipe_positions
        n = min(len(positions), self.config.swipe_history_frames)
        if n < 3:
            return DynamicGesture.NONE, 0.0

        recent = list(positions)[-n:]
        start = recent[0]
        end = recent[-1]
        dt = end[2] - start[2]
        if dt < self.config.min_dt:
            return DynamicGesture.NONE, 0.0

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.sqrt(dx**2 + dy**2)
        velocity = dist / dt

        if dist < self.config.swipe_min_distance:
            return DynamicGesture.NONE, 0.0
        if velocity < self.config.swipe_min_velocity:
            return DynamicGesture.NONE, 0.0

        # Reject diagonal movements: primary axis must dominate secondary
        ratio = self.config.swipe_direction_ratio
        if abs(dx) > abs(dy):
            if abs(dy) > 1e-6 and abs(dx) / abs(dy) < ratio:
                return DynamicGesture.NONE, 0.0
        else:
            if abs(dx) > 1e-6 and abs(dy) / abs(dx) < ratio:
                return DynamicGesture.NONE, 0.0

        confidence = min(1.0, velocity / (self.config.swipe_min_velocity * 2))

        # Determine direction (dominant axis)
        if abs(dx) > abs(dy):
            if dx < 0:
                return DynamicGesture.SWIPE_LEFT, confidence
            else:
                return DynamicGesture.SWIPE_RIGHT, confidence
        else:
            if dy < 0:
                return DynamicGesture.SWIPE_UP, confidence
            else:
                return DynamicGesture.SWIPE_DOWN, confidence

    def _check_pinch_spread(self, hist: _HandHistory,
                            static_gesture: 'StaticGesture' = None) -> Tuple[DynamicGesture, float]:
        # Gate on compatible static gestures to prevent false triggers during transitions
        if static_gesture is not None:
            if static_gesture.name not in self.config.pinch_compatible_gestures:
                return DynamicGesture.NONE, 0.0

        distances = hist.pinch_distances
        n = min(len(distances), self.config.pinch_history_frames)
        if n < 3:
            return DynamicGesture.NONE, 0.0

        recent = list(distances)[-n:]
        start_dist, start_t = recent[0]
        end_dist, end_t = recent[-1]
        dt = end_t - start_t
        if dt < self.config.min_dt:
            return DynamicGesture.NONE, 0.0

        rate = (end_dist - start_dist) / dt

        if abs(rate) < self.config.pinch_rate_threshold:
            return DynamicGesture.NONE, 0.0

        confidence = min(1.0, abs(rate) / (self.config.pinch_rate_threshold * 2))

        if rate < 0 and end_dist < self.config.pinch_close_threshold:
            return DynamicGesture.PINCH, confidence
        elif rate > 0 and end_dist > self.config.pinch_open_threshold:
            return DynamicGesture.SPREAD, confidence

        return DynamicGesture.NONE, 0.0

    def _check_circle(self, hist: _HandHistory) -> Tuple[DynamicGesture, float]:
        positions = hist.circle_positions
        n = min(len(positions), self.config.circle_history_frames)
        if n < self.config.circle_min_points:
            return DynamicGesture.NONE, 0.0

        recent = list(positions)[-n:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]

        cx = sum(xs) / n
        cy = sum(ys) / n

        # Check radius consistency
        radii = [math.sqrt((x - cx)**2 + (y - cy)**2) for x, y in zip(xs, ys)]
        mean_r = sum(radii) / len(radii)
        if mean_r < self.config.circle_min_radius:  # too small to be a circle
            return DynamicGesture.NONE, 0.0

        std_r = math.sqrt(sum((r - mean_r)**2 for r in radii) / len(radii))
        if std_r / mean_r > self.config.circle_radius_tolerance:
            return DynamicGesture.NONE, 0.0

        # Calculate total angular sweep
        total_angle = 0.0
        for i in range(1, n):
            a1 = math.atan2(ys[i-1] - cy, xs[i-1] - cx)
            a2 = math.atan2(ys[i] - cy, xs[i] - cx)
            da = a2 - a1
            # Normalize to [-pi, pi]
            while da > math.pi:
                da -= 2 * math.pi
            while da < -math.pi:
                da += 2 * math.pi
            total_angle += da

        if abs(total_angle) < self.config.circle_min_angle:
            return DynamicGesture.NONE, 0.0

        confidence = min(1.0, abs(total_angle) / (2 * math.pi))

        if total_angle > 0:
            return DynamicGesture.CIRCLE_CW, confidence
        else:
            return DynamicGesture.CIRCLE_CCW, confidence

    def _check_wave(self, hist: _HandHistory) -> Tuple[DynamicGesture, float]:
        positions = hist.wave_positions
        n = min(len(positions), self.config.wave_history_frames)
        if n < self.config.wave_min_frames:
            return DynamicGesture.NONE, 0.0

        recent = list(positions)[-n:]
        xs = [p[0] for p in recent]

        # Count direction reversals in x
        reversals = 0
        max_swing = 0.0
        last_dir = None
        swing_start = xs[0]

        for i in range(1, len(xs)):
            dx = xs[i] - xs[i-1]
            if abs(dx) < self.config.wave_deadzone:
                continue

            current_dir = 'R' if dx > 0 else 'L'
            if last_dir is not None and current_dir != last_dir:
                swing = abs(xs[i] - swing_start)
                max_swing = max(max_swing, swing)
                swing_start = xs[i]
                reversals += 1
            last_dir = current_dir

        if reversals >= self.config.wave_min_reversals and max_swing >= self.config.wave_min_amplitude:
            confidence = min(1.0, reversals / (self.config.wave_min_reversals * 2))
            return DynamicGesture.WAVE, confidence

        return DynamicGesture.NONE, 0.0

    def clear_history(self, handedness: Optional[str] = None):
        if handedness:
            self._histories.pop(handedness, None)
        else:
            self._histories.clear()


class GestureRecognizer:
    """Top-level class combining static and dynamic gesture recognition."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        self.static_classifier = StaticGestureClassifier(self.config)
        self.dynamic_detector = DynamicGestureDetector(self.config)
        # Hysteresis: per-hand history of raw static classifications
        self._gesture_history: Dict[str, deque] = {}
        # Per-hand confirmed (reported) gesture
        self._confirmed_gesture: Dict[str, Tuple[StaticGesture, float]] = {}

    def _apply_hysteresis(self, handedness: str,
                          raw_gesture: StaticGesture,
                          raw_conf: float) -> Tuple[StaticGesture, float]:
        """Switch reported gesture when at least min_agree of N frames agree."""
        n = self.config.gesture_hysteresis_frames
        min_agree = self.config.gesture_hysteresis_min_agree
        if n <= 1:
            return raw_gesture, raw_conf

        if handedness not in self._gesture_history:
            self._gesture_history[handedness] = deque(maxlen=n)
        history = self._gesture_history[handedness]
        history.append((raw_gesture, raw_conf))

        if len(history) >= min_agree:
            # Count most common gesture in the window
            gesture_counts = Counter(g for g, _ in history)
            most_common_gesture, count = gesture_counts.most_common(1)[0]
            if count >= min_agree:
                # Average confidence across matching frames
                matching_confs = [c for g, c in history if g == most_common_gesture]
                avg_conf = sum(matching_confs) / len(matching_confs)
                self._confirmed_gesture[handedness] = (most_common_gesture, avg_conf)

        return self._confirmed_gesture.get(handedness, (StaticGesture.NONE, 0.0))

    def recognize(self, hands: List[HandLandmarks]) -> List[GestureResult]:
        results = []
        for i, hand in enumerate(hands):
            raw_gesture, raw_conf = self.static_classifier.classify(hand)
            static_gesture, static_conf = self._apply_hysteresis(
                hand.handedness, raw_gesture, raw_conf
            )
            dynamic_gesture, dynamic_conf = self.dynamic_detector.update(
                hand, static_gesture=static_gesture
            )

            results.append(GestureResult(
                hand_index=i,
                handedness=hand.handedness,
                static_gesture=static_gesture,
                static_confidence=static_conf,
                dynamic_gesture=dynamic_gesture,
                dynamic_confidence=dynamic_conf,
            ))
        return results

    def clear(self):
        self.dynamic_detector.clear_history()
        self._gesture_history.clear()
        self._confirmed_gesture.clear()
