"""
CV Camera Interactive Stream Client

Connects to the Android camera stream and provides full keyboard control
over camera settings with a live HUD overlay.

On startup, runs a calibration phase:
  - Lets auto-exposure and auto-focus settle for a few seconds
  - Reads the values the camera chose
  - Locks exposure, ISO, focus, and white balance

Usage:
    python stream_client.py <phone_ip>
    python stream_client.py <phone_ip> --model path/to/gesture_transformer.pt

Controls:
    SPACE   Toggle auto/manual exposure (lock exposure)
    A       Toggle auto/manual focus
    E / D   Exposure compensation +/-  (auto mode)
    I / K   ISO up/down                (manual mode)
    T / G   Exposure time up/down      (manual mode)
    F / V   Focus far/near
    Z / X   Zoom in/out
    W       Cycle white balance
    R       Cycle resolution
    J / L   JPEG quality -/+
    N       Toggle hand+pose tracking
    B       Toggle game bridge on/off
    P       Toggle policy debug overlay
    0       Re-calibrate (unlock, settle, re-lock)
    H       Toggle help overlay
    S       Print full status to console
    Q/ESC   Quit
"""

import argparse
import os
import sys
import socket
import struct
import json
import time
import threading
from collections import deque

# Suppress Samsung's non-standard JPEG warnings from libjpeg
if sys.platform == "win32":
    _stderr_fd = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

# Restore stderr now that cv2 is loaded (suppression was only for JPEG warnings)
if sys.platform == "win32":
    os.dup2(_stderr_fd, 2)
    os.close(_stderr_fd)

from config import HandTrackingConfig, VisualizationConfig, PoseVisualizationConfig
from hand_tracker import HandTracker
from pose_tracker import PoseTracker
from look_joystick import LookJoystick
from strafe_detector import StrafeDetector

# --- Defaults ---
DEFAULT_RESOLUTION = (640, 480)
DEFAULT_JPEG_QUALITY = 70
DEFAULT_WB = "daylight"
CALIBRATION_SECONDS = 3

RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (2400, 1080),
    (3840, 2160),
]

WB_MODES = ["auto", "daylight", "cloudy", "shade", "fluorescent", "incandescent", "warm_fluorescent", "twilight"]

EXPOSURE_TIMES = [
    500_000,       # 1/2000s
    1_000_000,     # 1/1000s
    2_000_000,     # 1/500s
    4_000_000,     # 1/250s
    8_000_000,     # 1/125s
    16_666_666,    # 1/60s
    33_333_333,    # 1/30s
    66_666_666,    # 1/15s
    100_000_000,   # 1/10s
]

ISO_VALUES = [50, 100, 200, 400, 800, 1600, 3200]


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


class FrameGrabber:
    """Background thread that reads frames from the stream socket,
    always keeping only the latest frame. Eliminates buffering latency."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._latest: bytes = b""
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._running = True
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                length_bytes = recv_exact(self._sock, 4)
                length = struct.unpack(">I", length_bytes)[0]
                jpeg_data = recv_exact(self._sock, length)
                with self._lock:
                    if self._latest:
                        self._dropped += 1
                    self._latest = jpeg_data
                self._new_frame.set()
            except Exception:
                break

    def get(self, timeout: float = 1.0) -> bytes:
        """Block until a new frame is available, return JPEG bytes."""
        self._new_frame.wait(timeout)
        with self._lock:
            data = self._latest
            self._latest = b""
            self._new_frame.clear()
        return data

    @property
    def dropped(self) -> int:
        return self._dropped

    def stop(self):
        self._running = False


def format_exposure(ns: int) -> str:
    if ns <= 0:
        return "auto"
    sec = ns / 1_000_000_000
    if sec >= 0.1:
        return f"{sec:.1f}s"
    return f"1/{int(round(1/sec))}s"


def find_nearest_index(values, target):
    best = 0
    for i, v in enumerate(values):
        if abs(v - target) < abs(values[best] - target):
            best = i
    return best


class CameraControl:
    def __init__(self, host: str, port: int = 5001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect((host, port))
        self.reader = self.sock.makefile("r")

    def send(self, command: dict) -> dict:
        try:
            self.sock.sendall((json.dumps(command) + "\n").encode())
            line = self.reader.readline().strip()
            return json.loads(line) if line else {"status": "error", "message": "empty response"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def set_exposure(self, ns: int): return self.send({"command": "set_exposure_time", "value": ns})
    def set_iso(self, iso: int): return self.send({"command": "set_iso", "value": iso})
    def set_ev(self, ev: int): return self.send({"command": "set_exposure_compensation", "value": ev})
    def auto_exposure(self): return self.send({"command": "set_auto_exposure"})
    def set_focus(self, d: float): return self.send({"command": "set_focus", "value": d})
    def auto_focus(self): return self.send({"command": "set_auto_focus"})
    def set_zoom(self, r: float): return self.send({"command": "set_zoom", "value": r})
    def set_wb(self, mode: str): return self.send({"command": "set_white_balance", "value": mode})
    def set_resolution(self, w: int, h: int): return self.send({"command": "set_resolution", "width": w, "height": h})
    def set_jpeg_quality(self, q: int): return self.send({"command": "set_jpeg_quality", "value": q})
    def set_fps(self, mn: int, mx: int): return self.send({"command": "set_fps", "min": mn, "max": mx})
    def get_status(self): return self.send({"command": "get_status"})
    def get_capabilities(self): return self.send({"command": "get_capabilities"})
    def get_auto_values(self): return self.send({"command": "get_auto_values"})
    def lock_from_auto(self): return self.send({"command": "lock_from_auto"})

    def close(self):
        try: self.sock.close()
        except: pass


class GestureClassifier:
    """Wraps the trained GestureTransformer for live multi-label inference.

    Maintains a rolling buffer of feature frames and runs the model
    periodically to classify the current gesture sequence. Outputs
    independent sigmoid probabilities per gesture, thresholded to produce
    a set of active gestures (multiple can fire simultaneously).

    Uses different smoothing strategies for states (sustained gestures like
    walking/crouching) vs actions (one-shot gestures like jump/swing).
    """

    # Gestures that are sustained states -- use sticky hysteresis
    _STATE_GESTURES = {"walking", "sprinting", "crouching"}

    def __init__(self, model_path: str, config_path: str = None):
        import torch
        from gesture_model import GestureTransformer
        from gesture_dataset import append_velocity, _POSITION_DIMS

        self._torch = torch
        self._append_velocity = append_velocity
        self._position_dims = _POSITION_DIMS

        if config_path is None:
            config_path = os.path.join(os.path.dirname(model_path), "gesture_config.json")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.gesture_labels = cfg["gesture_labels"]
        self.num_gestures = cfg["num_gestures"]
        self.input_dim = cfg["input_dim"]
        self.max_seq_len = cfg["max_seq_len"]

        # Per-gesture thresholds (fall back to single threshold if not available)
        default_thresh = cfg.get("threshold", 0.5)
        per_thresh = cfg.get("per_gesture_thresholds", {})
        self.thresholds = np.array([
            per_thresh.get(name, default_thresh) for name in self.gesture_labels
        ], dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GestureTransformer(
            input_dim=cfg["input_dim"],
            num_gestures=cfg["num_gestures"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            max_seq_len=cfg["max_seq_len"],
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        self._buffer = deque(maxlen=self.max_seq_len)

        # Classify each gesture as state or action for smoothing
        self._is_state = np.array([
            name in self._STATE_GESTURES for name in self.gesture_labels
        ], dtype=bool)

        # Per-gesture smoothing counters
        self._consec_active = np.zeros(self.num_gestures, dtype=int)
        self._consec_inactive = np.zeros(self.num_gestures, dtype=int)
        self._display_active = np.zeros(self.num_gestures, dtype=bool)
        self._display_probs = np.zeros(self.num_gestures, dtype=np.float32)
        # Actions: hold display for N inference cycles after activation
        self._action_hold = np.zeros(self.num_gestures, dtype=int)
        self._ACTION_HOLD_FRAMES = 6  # keep action shown for ~12 real frames

    def push_frame(self, pose, hands):
        """Extract features from current pose+hand detections and add to buffer."""
        if pose is not None and pose.world_landmarks:
            pose_world = np.array(pose.world_landmarks, dtype=np.float32).reshape(-1)
        else:
            pose_world = np.zeros(99, dtype=np.float32)

        if pose is not None:
            pose_vis = np.array(pose.visibility, dtype=np.float32)
        else:
            pose_vis = np.zeros(33, dtype=np.float32)

        left_3d = np.zeros(63, dtype=np.float32)
        right_3d = np.zeros(63, dtype=np.float32)
        left_present = 0.0
        right_present = 0.0

        for hand in hands:
            kp3d = (np.array(hand.landmarks_3d, dtype=np.float32).reshape(-1)
                    if hand.landmarks_3d else np.zeros(63, dtype=np.float32))
            if hand.handedness == "Left":
                left_3d = kp3d
                left_present = 1.0
            else:
                right_3d = kp3d
                right_present = 1.0

        features = np.concatenate([
            pose_world,                              # 99
            left_3d,                                 # 63
            right_3d,                                # 63
            np.array([left_present], dtype=np.float32),   # 1
            np.array([right_present], dtype=np.float32),  # 1
            pose_vis,                                # 33
        ])  # total: 260

        self._buffer.append(features)

    def predict(self) -> tuple:
        """Run multi-label inference on the current buffer.

        Returns:
            (active_labels, active_probs): List of active gesture names and
            their sigmoid probabilities. If no gestures are active, returns
            (["idle"], [1.0]).
        """
        torch = self._torch
        if len(self._buffer) < 5:
            return self._get_display_output()

        with torch.no_grad():
            raw_frames = np.stack(list(self._buffer))  # (T, 260)
            T = raw_frames.shape[0]
            length = min(T, self.max_seq_len)
            raw_clip = raw_frames[-length:]  # (length, 260)

            # Compute velocity features from the sequence
            frames = self._append_velocity(raw_clip)  # (length, 485)

            padded = np.zeros((self.max_seq_len, self.input_dim), dtype=np.float32)
            padded[:length] = frames

            mask = np.zeros(self.max_seq_len, dtype=np.float32)
            mask[:length] = 1.0

            x = torch.from_numpy(padded).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).unsqueeze(0).to(self.device)

            logits = self.model(x, m)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (num_gestures,)

        raw_active = probs > self.thresholds

        # Per-gesture smoothing with different strategies for states vs actions
        for g in range(self.num_gestures):
            if raw_active[g]:
                self._consec_active[g] += 1
                self._consec_inactive[g] = 0
            else:
                self._consec_inactive[g] += 1
                self._consec_active[g] = 0

            if self._is_state[g]:
                # States: sticky hysteresis (2 to activate, 3 to deactivate)
                if not self._display_active[g] and self._consec_active[g] >= 2:
                    self._display_active[g] = True
                elif self._display_active[g] and self._consec_inactive[g] >= 3:
                    self._display_active[g] = False
            else:
                # Actions: fire on 1 strong prediction, hold for N cycles
                if raw_active[g]:
                    self._display_active[g] = True
                    self._action_hold[g] = self._ACTION_HOLD_FRAMES
                elif self._action_hold[g] > 0:
                    self._action_hold[g] -= 1
                else:
                    self._display_active[g] = False

            if self._display_active[g]:
                self._display_probs[g] = probs[g]

        return self._get_display_output()

    def _get_display_output(self) -> tuple:
        """Return current display state as (labels_list, probs_list)."""
        active_labels = []
        active_probs = []
        for g in range(self.num_gestures):
            if self._display_active[g]:
                active_labels.append(self.gesture_labels[g])
                active_probs.append(float(self._display_probs[g]))
        if not active_labels:
            active_labels = ["idle"]
            active_probs = [1.0]
        return active_labels, active_probs

    def clear(self):
        self._buffer.clear()
        self._consec_active[:] = 0
        self._consec_inactive[:] = 0
        self._display_active[:] = False
        self._display_probs[:] = 0.0
        self._action_hold[:] = 0


def calibrate(ctrl: CameraControl, stream_sock: socket.socket):
    """Run auto for a few seconds, then lock everything."""
    print(f"\n--- CALIBRATION ---")
    print(f"Setting resolution to {DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}, "
          f"JPEG Q{DEFAULT_JPEG_QUALITY}, WB {DEFAULT_WB}")

    ctrl.set_resolution(*DEFAULT_RESOLUTION)
    ctrl.set_jpeg_quality(DEFAULT_JPEG_QUALITY)
    ctrl.auto_exposure()
    ctrl.auto_focus()
    ctrl.set_wb(DEFAULT_WB)

    print(f"Auto-exposure/focus settling for {CALIBRATION_SECONDS}s...")
    time.sleep(CALIBRATION_SECONDS)

    # Read what auto settled on
    auto_vals = ctrl.get_auto_values()
    exp_ns = auto_vals.get("exposure_time_ns", 0)
    iso = auto_vals.get("iso", 0)
    focus = auto_vals.get("focus_distance", 0)

    print(f"  Auto chose: shutter={format_exposure(exp_ns)}, ISO={iso}, focus={focus:.2f} dpt")

    # Lock everything
    result = ctrl.lock_from_auto()
    print(f"  Locked! {result.get('message', '')}")
    print(f"--- READY ---\n")

    return {
        "exposure_time_ns": exp_ns,
        "iso": iso,
        "focus_distance": focus,
    }


def draw_control_hud(frame, control_output, show_debug):
    """Draw control policy state on the frame (bottom-right corner).

    Shows active controls as colored text. When show_debug is True,
    also shows raw probability bars for each control.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    gray = (128, 128, 128)

    if control_output is None:
        return frame

    # Active controls as colored text (bottom-right)
    active_labels = []
    if control_output.forward:   active_labels.append("FWD")
    if control_output.sprint:    active_labels.append("SPRINT")
    if control_output.sneak:     active_labels.append("SNEAK")
    if control_output.strafe_left:  active_labels.append("STR_L")
    if control_output.strafe_right: active_labels.append("STR_R")
    if control_output.jump:      active_labels.append("JUMP")
    if control_output.attack:    active_labels.append("ATK")
    if control_output.use_item:  active_labels.append("USE")

    if not active_labels:
        active_labels = ["IDLE"]

    controls_text = " ".join(active_labels)
    tw = cv2.getTextSize(controls_text, font, 0.7, 2)[0][0]

    # Look indicator text
    look_text = f"Look: ({control_output.look_yaw:+.2f}, {control_output.look_pitch:+.2f})"
    ltw = cv2.getTextSize(look_text, font, 0.5, 1)[0][0]
    max_text_w = max(tw, ltw)

    if show_debug:
        # Full debug overlay with probability bars
        bar_names = ["forward", "str_L", "str_R", "sprint",
                     "sneak", "jump", "attack", "use"]
        bar_w = 100
        bar_h = 14
        line_h = 20
        panel_w = 80 + bar_w + 60  # name + bar + value
        panel_h = len(bar_names) * line_h + 70  # bars + controls text + look
        panel_w = max(panel_w, max_text_w + 20)

        px = w - panel_w - 10
        py = h - panel_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 5, py - 5),
                      (w - 5, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Active controls text at top of panel
        is_idle = controls_text == "IDLE"
        ctrl_color = gray if is_idle else green
        cv2.putText(frame, controls_text, (px + 5, py + 18),
                    font, 0.6, ctrl_color, 2)

        # Look values
        look_color = cyan if (abs(control_output.look_yaw) > 0.01
                              or abs(control_output.look_pitch) > 0.01) else gray
        cv2.putText(frame, look_text, (px + 5, py + 40),
                    font, 0.45, look_color, 1)

        # Probability bars
        probs = control_output.raw_action_probs
        by = py + 55
        for i, name in enumerate(bar_names):
            prob = float(probs[i]) if i < len(probs) else 0.0
            label_x = px + 5
            bar_x = px + 70
            val_x = bar_x + bar_w + 5

            # Name
            cv2.putText(frame, name, (label_x, by + bar_h - 2),
                        font, 0.4, white, 1)

            # Background bar
            cv2.rectangle(frame, (bar_x, by),
                          (bar_x + bar_w, by + bar_h), (50, 50, 50), -1)

            # Fill bar
            fill_w = int(prob * bar_w)
            bar_color = green if prob > 0.5 else yellow if prob > 0.3 else red
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, by),
                              (bar_x + fill_w, by + bar_h), bar_color, -1)

            # Value text
            cv2.putText(frame, f"{prob:.2f}", (val_x, by + bar_h - 2),
                        font, 0.35, white, 1)

            by += line_h
    else:
        # Compact display: just active controls and look
        box_h = 50
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (w - max_text_w - 30, h - box_h - 10),
                      (w - 5, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        is_idle = controls_text == "IDLE"
        ctrl_color = gray if is_idle else green
        cv2.putText(frame, controls_text,
                    (w - max_text_w - 20, h - 35),
                    font, 0.6, ctrl_color, 2)

        look_color = cyan if (abs(control_output.look_yaw) > 0.01
                              or abs(control_output.look_pitch) > 0.01) else gray
        cv2.putText(frame, look_text,
                    (w - max_text_w - 20, h - 15),
                    font, 0.45, look_color, 1)

    return frame


def draw_hud(frame, state, show_help):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (5, 5), (340, 195), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 25
    line_h = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    white = (255, 255, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)

    ae = state.get("ae_mode", "auto")
    ae_color = green if ae == "auto" else yellow
    cv2.putText(frame, f"AE: {ae.upper()}", (15, y), font, fs, ae_color, 1); y += line_h

    if ae == "manual":
        exp = state.get("exposure_time_ns", -1)
        iso = state.get("iso", -1)
        cv2.putText(frame, f"Shutter: {format_exposure(exp)}", (15, y), font, fs, white, 1); y += line_h
        cv2.putText(frame, f"ISO: {iso}", (15, y), font, fs, white, 1); y += line_h
    else:
        ev = state.get("exposure_compensation", 0)
        cv2.putText(frame, f"EV: {ev:+d}", (15, y), font, fs, white, 1); y += line_h
        y += line_h

    af = state.get("af_mode", "auto")
    af_color = green if af == "auto" else yellow
    cv2.putText(frame, f"AF: {af.upper()}", (15, y), font, fs, af_color, 1); y += line_h

    if af == "manual":
        fd = state.get("focus_distance", -1)
        cv2.putText(frame, f"Focus: {fd:.1f} dpt", (15, y), font, fs, white, 1); y += line_h
    else:
        y += line_h

    zoom = state.get("zoom", 1.0)
    cv2.putText(frame, f"Zoom: {zoom:.1f}x", (15, y), font, fs, white, 1); y += line_h

    res = state.get("resolution", "?")
    quality = state.get("jpeg_quality", "?")
    cv2.putText(frame, f"Res: {res}  Q:{quality}", (15, y), font, fs, white, 1); y += line_h

    fps = state.get("_fps", 0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 25), font, 0.6, green, 2)

    # Hand tracking info (top-right, below FPS)
    ht_on = state.get("hand_tracking_enabled", False)
    if ht_on:
        hands_n = state.get("hands_detected", 0)
        ht_color = green if hands_n > 0 else yellow
        cv2.putText(frame, f"Hands: {hands_n}", (w - 130, 50), font, 0.5, ht_color, 1)

        pose_on = state.get("pose_detected", False)
        pose_color = (255, 0, 255) if pose_on else (128, 128, 128)
        pose_text = "Pose: YES" if pose_on else "Pose: NO"
        cv2.putText(frame, pose_text, (w - 130, 75), font, 0.5, pose_color, 1)

        # Model type indicator
        model_type = state.get("model_type", None)
        if model_type == "control":
            cv2.putText(frame, "Policy", (w - 130, 100), font, 0.5, (0, 200, 255), 1)
        elif model_type == "gesture":
            cv2.putText(frame, "Gesture", (w - 130, 100), font, 0.5, (255, 200, 0), 1)

        # Gesture model predictions (bottom-right corner, stacked)
        # Only show when using gesture classifier (not control policy)
        if model_type != "control":
            gesture_labels = state.get("predicted_gestures", [])
            gesture_probs = state.get("predicted_probs", [])
            if gesture_labels:
                line_height = 28
                g_lines = []
                for lbl, prob in zip(gesture_labels, gesture_probs):
                    g_lines.append((f"{lbl} ({prob:.0%})", prob))

                # Measure max width for background box
                max_tw = 0
                for g_text, _ in g_lines:
                    tw = cv2.getTextSize(g_text, font, 0.7, 2)[0][0]
                    max_tw = max(max_tw, tw)

                box_h = len(g_lines) * line_height + 12
                gy_base = h - 15
                gx = w - max_tw - 25
                overlay_g = frame.copy()
                cv2.rectangle(overlay_g,
                              (gx - 8, gy_base - box_h),
                              (w - 5, gy_base + 5),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay_g, 0.6, frame, 0.4, 0, frame)

                for i, (g_text, prob) in enumerate(reversed(g_lines)):
                    gy = gy_base - i * line_height
                    g_color = green if prob > 0.8 else yellow
                    cv2.putText(frame, g_text, (gx, gy), font, 0.7, g_color, 2)
    else:
        cv2.putText(frame, "Hands: OFF", (w - 130, 50), font, 0.5, (128, 128, 128), 1)
        cv2.putText(frame, "Pose: OFF", (w - 130, 75), font, 0.5, (128, 128, 128), 1)

    if show_help:
        help_lines = [
            "SPACE  Lock/unlock exposure",
            "A      Lock/unlock focus",
            "E/D    EV comp +/-",
            "I/K    ISO +/-",
            "T/G    Shutter +/-",
            "F/V    Focus far/near",
            "Z/X    Zoom +/-",
            "W      White balance",
            "R      Resolution",
            "J/L    JPEG quality -/+",
            "0      Re-calibrate",
            "N      Toggle hand+pose tracking",
            "B      Toggle game bridge",
            "P      Policy debug overlay",
            "H      Toggle help",
            "Q/ESC  Quit",
        ]
        bx, by = w - 290, 50
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (bx - 5, by - 20), (bx + 270, by + len(help_lines) * 20 + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        for i, line in enumerate(help_lines):
            cv2.putText(frame, line, (bx, by + i * 20), font, 0.45, white, 1)

    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="CV Camera Interactive Stream Client")
    parser.add_argument("host", help="Phone IP address")
    parser.add_argument("--stream-port", type=int, default=5000)
    parser.add_argument("--control-port", type=int, default=5001)
    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="Path to model .pt file (auto-detects gesture vs control policy)",
    )
    parser.add_argument(
        "--game",
        action="store_true",
        help="Enable game bridge: send gestures to Minecraft via MCCTP",
    )
    parser.add_argument(
        "--game-port",
        type=int,
        default=8765,
        help="MCCTP WebSocket port (default: 8765)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    host = args.host
    stream_port = args.stream_port
    control_port = args.control_port

    # Connect control
    print(f"Connecting control to {host}:{control_port}...")
    ctrl = CameraControl(host, control_port)

    caps = ctrl.get_capabilities()
    print(f"Camera: {caps.get('available_resolutions', '?')}")

    # Connect stream
    print(f"Connecting stream to {host}:{stream_port}...")
    stream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stream_sock.connect((host, stream_port))
    print("Connected!")

    # --- Calibration ---
    cal = calibrate(ctrl, stream_sock)

    # --- Optional model (auto-detect: control policy vs gesture classifier) ---
    gesture_clf = None
    control_policy = None
    if args.model:
        model_path = args.model
        if not os.path.isfile(model_path):
            print(f"WARNING: Model not found at {model_path}, running without classification")
        else:
            # Auto-detect model type by checking for control_config.json
            control_config_path = os.path.join(
                os.path.dirname(model_path), "control_config.json"
            )
            if os.path.isfile(control_config_path):
                from control_policy import ControlPolicy
                print(f"Loading control policy from {model_path}...")
                control_policy = ControlPolicy(model_path)
                print(f"  Input dim: {control_policy.input_dim}")
                print(f"  Device: {control_policy.device}")
            else:
                print(f"Loading gesture model from {model_path}...")
                gesture_clf = GestureClassifier(model_path)
                print(f"  Gestures: {gesture_clf.gesture_labels}")
                print(f"  Device: {gesture_clf.device}")

    # Inference throttle: run model every N frames to keep real-time
    # (only used for gesture classifier; control policy runs every frame)
    INFER_EVERY = 2
    infer_counter = 0

    # State tracking
    state = {
        "ae_mode": "manual",
        "exposure_time_ns": cal["exposure_time_ns"],
        "iso": cal["iso"],
        "exposure_compensation": 0,
        "af_mode": "manual",
        "focus_distance": cal["focus_distance"],
        "zoom": 1.0,
        "resolution": f"{DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}",
        "jpeg_quality": DEFAULT_JPEG_QUALITY,
        "_fps": 0,
        "hand_tracking_enabled": True,
        "hands_detected": 0,
        "pose_detected": False,
        "predicted_gestures": [],
        "predicted_probs": [],
        "model_type": "control" if control_policy else ("gesture" if gesture_clf else None),
        "control_output": None,
    }

    ev_val = 0
    iso_idx = find_nearest_index(ISO_VALUES, cal["iso"])
    exp_idx = find_nearest_index(EXPOSURE_TIMES, cal["exposure_time_ns"])
    focus_val = cal["focus_distance"]
    zoom_val = 1.0
    wb_idx = WB_MODES.index(DEFAULT_WB)
    res_idx = RESOLUTIONS.index(DEFAULT_RESOLUTION)
    quality_val = DEFAULT_JPEG_QUALITY
    show_help = False

    frame_count = 0
    fps_start = time.time()

    # --- Hand + pose tracking init ---
    tracker = HandTracker()
    pose_tracker = PoseTracker()
    look_joy = LookJoystick()
    strafe_det = StrafeDetector()

    # --- Optional game bridge ---
    game_bridge = None
    control_bridge = None
    if args.game:
        if control_policy is not None:
            from control_bridge import ControlBridge
            print(f"Connecting control bridge to MCCTP on port {args.game_port}...")
            control_bridge = ControlBridge(port=args.game_port)
            control_bridge.connect()
        else:
            from game_bridge import GameBridge
            print(f"Connecting game bridge to MCCTP on port {args.game_port}...")
            game_bridge = GameBridge(port=args.game_port)
            game_bridge.connect()
    game_bridge_active = True
    show_policy_debug = False

    cv2.namedWindow("CV Camera", cv2.WINDOW_NORMAL)
    grabber = FrameGrabber(stream_sock)

    try:
        while True:
            jpeg_data = grabber.get()
            if not jpeg_data:
                continue

            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 0.5:
                state["_fps"] = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # --- Hand + pose tracking ---
            if state["hand_tracking_enabled"]:
                hands = tracker.process(frame)
                state["hands_detected"] = len(hands)

                if hands:
                    tracker.draw(frame, hands)

                # Pose tracking
                pose = pose_tracker.process(frame)
                state["pose_detected"] = pose is not None
                if pose:
                    pose_tracker.draw(frame, pose)

                if control_policy is not None:
                    # --- Control policy path ---
                    # Get game state from MCCTP
                    game_state = control_bridge.get_game_state() if control_bridge else {}

                    # Push frame to policy
                    control_policy.push_frame(pose, hands, game_state)

                    # Run inference every frame (policy is lightweight)
                    output = control_policy.predict()
                    state["control_output"] = output

                    # Send to Minecraft
                    if control_bridge is not None and game_bridge_active:
                        control_bridge.update(output)

                elif gesture_clf is not None:
                    # --- Gesture classifier path (backward compat) ---
                    gesture_clf.push_frame(pose, hands)
                    infer_counter += 1
                    if infer_counter >= INFER_EVERY:
                        infer_counter = 0
                        labels, probs = gesture_clf.predict()
                        state["predicted_gestures"] = labels
                        state["predicted_probs"] = probs

                    # --- Look joystick (every frame) ---
                    look_output = look_joy.update(hands)
                    look_joy.draw(frame, hands)

                    # --- Strafe detector (every frame) ---
                    strafe_output = strafe_det.update(pose)
                    strafe_det.draw(frame, pose)

                    # --- Game bridge (send commands to Minecraft) ---
                    if game_bridge is not None and game_bridge_active:
                        game_bridge.update(
                            state.get("predicted_gestures", ["idle"]),
                            look_output,
                            strafe_output,
                        )
            else:
                state["hands_detected"] = 0
                state["pose_detected"] = False
                state["control_output"] = None
                if control_policy is not None:
                    control_policy.clear()
                    # Release control bridge actions when tracking is off
                    if control_bridge is not None:
                        control_bridge.release_all()
                if gesture_clf is not None:
                    gesture_clf.clear()
                    state["predicted_gestures"] = []
                    state["predicted_probs"] = []
                # Release game actions when tracking is off
                if game_bridge is not None:
                    game_bridge.update(["idle"], None, None)

            frame = draw_hud(frame, state, show_help)

            # Control policy HUD (drawn on top of main HUD)
            if control_policy is not None:
                frame = draw_control_hud(
                    frame, state.get("control_output"), show_policy_debug
                )

            cv2.imshow("CV Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):
                break

            elif key == ord('0'):  # Re-calibrate
                ctrl.auto_exposure()
                ctrl.auto_focus()
                state["ae_mode"] = "auto"
                state["af_mode"] = "auto"
                print("Re-calibrating...")
                cal = calibrate(ctrl, stream_sock)
                state["ae_mode"] = "manual"
                state["af_mode"] = "manual"
                state["exposure_time_ns"] = cal["exposure_time_ns"]
                state["iso"] = cal["iso"]
                state["focus_distance"] = cal["focus_distance"]
                iso_idx = find_nearest_index(ISO_VALUES, cal["iso"])
                exp_idx = find_nearest_index(EXPOSURE_TIMES, cal["exposure_time_ns"])
                focus_val = cal["focus_distance"]

            elif key == ord(' '):
                if state["ae_mode"] == "auto":
                    # Lock from current auto
                    auto = ctrl.get_auto_values()
                    ctrl.set_exposure(auto.get("exposure_time_ns", EXPOSURE_TIMES[exp_idx]))
                    ctrl.set_iso(auto.get("iso", ISO_VALUES[iso_idx]))
                    state["ae_mode"] = "manual"
                    state["exposure_time_ns"] = auto.get("exposure_time_ns", 0)
                    state["iso"] = auto.get("iso", 0)
                    iso_idx = find_nearest_index(ISO_VALUES, state["iso"])
                    exp_idx = find_nearest_index(EXPOSURE_TIMES, state["exposure_time_ns"])
                    print(f"[LOCKED] Exposure: {format_exposure(state['exposure_time_ns'])}, ISO: {state['iso']}")
                else:
                    ctrl.auto_exposure()
                    state["ae_mode"] = "auto"
                    ev_val = 0
                    state["exposure_compensation"] = 0
                    print("[UNLOCKED] Auto exposure")

            elif key == ord('a'):
                if state["af_mode"] == "auto":
                    auto = ctrl.get_auto_values()
                    focus_val = auto.get("focus_distance", focus_val)
                    ctrl.set_focus(focus_val)
                    state["af_mode"] = "manual"
                    state["focus_distance"] = focus_val
                    print(f"[LOCKED] Focus: {focus_val:.2f} diopters")
                else:
                    ctrl.auto_focus()
                    state["af_mode"] = "auto"
                    print("[UNLOCKED] Auto focus")

            elif key == ord('e'):
                if state["ae_mode"] == "auto":
                    ev_val = min(ev_val + 2, 20)
                    ctrl.set_ev(ev_val)
                    state["exposure_compensation"] = ev_val

            elif key == ord('d'):
                if state["ae_mode"] == "auto":
                    ev_val = max(ev_val - 2, -20)
                    ctrl.set_ev(ev_val)
                    state["exposure_compensation"] = ev_val

            elif key == ord('i'):
                iso_idx = min(iso_idx + 1, len(ISO_VALUES) - 1)
                ctrl.set_iso(ISO_VALUES[iso_idx])
                state["ae_mode"] = "manual"
                state["iso"] = ISO_VALUES[iso_idx]

            elif key == ord('k'):
                iso_idx = max(iso_idx - 1, 0)
                ctrl.set_iso(ISO_VALUES[iso_idx])
                state["ae_mode"] = "manual"
                state["iso"] = ISO_VALUES[iso_idx]

            elif key == ord('t'):
                exp_idx = max(exp_idx - 1, 0)
                ctrl.set_exposure(EXPOSURE_TIMES[exp_idx])
                state["ae_mode"] = "manual"
                state["exposure_time_ns"] = EXPOSURE_TIMES[exp_idx]

            elif key == ord('g'):
                exp_idx = min(exp_idx + 1, len(EXPOSURE_TIMES) - 1)
                ctrl.set_exposure(EXPOSURE_TIMES[exp_idx])
                state["ae_mode"] = "manual"
                state["exposure_time_ns"] = EXPOSURE_TIMES[exp_idx]

            elif key == ord('f'):
                focus_val = max(focus_val - 0.5, 0.0)
                ctrl.set_focus(focus_val)
                state["af_mode"] = "manual"
                state["focus_distance"] = focus_val

            elif key == ord('v'):
                focus_val = min(focus_val + 0.5, 10.0)
                ctrl.set_focus(focus_val)
                state["af_mode"] = "manual"
                state["focus_distance"] = focus_val

            elif key == ord('z'):
                zoom_val = min(zoom_val + 0.5, 8.0)
                ctrl.set_zoom(zoom_val)
                state["zoom"] = zoom_val

            elif key == ord('x'):
                zoom_val = max(zoom_val - 0.5, 1.0)
                ctrl.set_zoom(zoom_val)
                state["zoom"] = zoom_val

            elif key == ord('w'):
                wb_idx = (wb_idx + 1) % len(WB_MODES)
                ctrl.set_wb(WB_MODES[wb_idx])
                print(f"WB: {WB_MODES[wb_idx]}")

            elif key == ord('r'):
                res_idx = (res_idx + 1) % len(RESOLUTIONS)
                w, h = RESOLUTIONS[res_idx]
                ctrl.set_resolution(w, h)
                state["resolution"] = f"{w}x{h}"
                print(f"Resolution: {w}x{h}")

            elif key == ord('j'):
                quality_val = max(quality_val - 10, 10)
                ctrl.set_jpeg_quality(quality_val)
                state["jpeg_quality"] = quality_val

            elif key == ord('l'):
                quality_val = min(quality_val + 10, 100)
                ctrl.set_jpeg_quality(quality_val)
                state["jpeg_quality"] = quality_val

            elif key == ord('h'):
                show_help = not show_help

            elif key == ord('n'):
                state["hand_tracking_enabled"] = not state["hand_tracking_enabled"]
                status = "ON" if state["hand_tracking_enabled"] else "OFF"
                print(f"[Hand+Pose Tracking] {status}")

            elif key == ord('b'):
                if game_bridge is not None or control_bridge is not None:
                    game_bridge_active = not game_bridge_active
                    if not game_bridge_active:
                        if control_bridge is not None:
                            control_bridge.release_all()
                        elif game_bridge is not None:
                            game_bridge.update(["idle"], None, None)  # release all
                    print(f"[Game Bridge] {'ON' if game_bridge_active else 'OFF (paused)'}")

            elif key == ord('p'):
                show_policy_debug = not show_policy_debug
                print(f"[Policy Debug] {'ON' if show_policy_debug else 'OFF'}")

            elif key == ord('s'):
                s = ctrl.get_status()
                print(json.dumps(s, indent=2))

    except (ConnectionError, struct.error) as e:
        print(f"\nStream ended: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        grabber.stop()
        tracker.close()
        pose_tracker.close()
        if control_bridge is not None:
            control_bridge.disconnect()
        if game_bridge is not None:
            game_bridge.disconnect()
        stream_sock.close()
        ctrl.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
