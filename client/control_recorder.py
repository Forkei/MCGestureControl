"""V2 Control data recording tool for training a control policy model.

Records synchronized body pose + hand tracking + in-game control labels
from the MCCTP mod during Minecraft gameplay sessions. Saves session
recordings as .npz files for training a Transformer-based control policy.

V2 changes from V1:
  - Controls captured from MCCTP mod (resolved game inputs) instead of pynput
  - 28-dim control vector (was 10)
  - 46-dim game state (was 16/24)
  - No keyboard/mouse hooking — keybind and sensitivity independent
  - Inventory cursor + clicks supported

Usage:
    python control_recorder.py <phone_ip> --game
    python control_recorder.py <phone_ip> --game --game-port 8765

UI Controls:
    R           Start/stop recording session
    BACKSPACE   Discard last saved session
    H           Toggle help overlay
    Q / ESC     Quit (auto-saves if recording)
"""

import argparse
import os
import sys
import socket
import struct
import time
import math
import winsound

# Suppress Samsung's non-standard JPEG warnings from libjpeg
if sys.platform == "win32":
    _stderr_fd = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

if sys.platform == "win32":
    os.dup2(_stderr_fd, 2)
    os.close(_stderr_fd)

from stream_client import FrameGrabber, CameraControl
from stream_client import DEFAULT_RESOLUTION, DEFAULT_JPEG_QUALITY, DEFAULT_WB
from hand_tracker import HandTracker
from pose_tracker import PoseTracker
from control_policy import encode_game_state_v2
from control_dataset import (
    CTRL_MOVE_FORWARD, CTRL_MOVE_BACKWARD, CTRL_STRAFE_LEFT, CTRL_STRAFE_RIGHT,
    CTRL_SPRINT, CTRL_SNEAK, CTRL_JUMP, CTRL_ATTACK, CTRL_USE_ITEM,
    CTRL_LOOK_YAW, CTRL_LOOK_PITCH, CTRL_DROP_ITEM, CTRL_SWAP_OFFHAND,
    CTRL_OPEN_INVENTORY, CTRL_HOTBAR_START, CTRL_HOTBAR_END,
    CTRL_CURSOR_X, CTRL_CURSOR_Y, CTRL_INV_LEFT_CLICK,
    CTRL_INV_RIGHT_CLICK, CTRL_INV_SHIFT_HELD,
    NUM_CONTROLS, GAME_STATE_DIM,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(_SCRIPT_DIR, "recordings_control")

# Minimum frames for a valid session
MIN_SESSION_FRAMES = 30

# Countdown before recording starts (seconds)
COUNTDOWN_SECONDS = 3

# Look normalization: game yaw/pitch delta per tick → [-1, 1]
LOOK_NORMALIZE_FACTOR = 15.0

# Drawing
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# MCCTP Control Capture (replaces v1's pynput ControlInputCapture)
# ---------------------------------------------------------------------------


class MCCTPControlCapture:
    """Captures control inputs from MCCTP game state each frame.

    Instead of hooking keyboard/mouse with pynput, reads the resolved game
    input state from the MCCTP mod's state broadcast. This is keybind-
    independent and sensitivity-independent.
    """

    def __init__(self, mcctp_client=None):
        self._client = mcctp_client
        self._prev_yaw = None
        self._prev_pitch = None
        self._prev_slot = None
        self._active = False
        self._last_state_dict = {}

    def start_capture(self):
        """Enable control capture (call when recording starts)."""
        self._prev_yaw = None
        self._prev_pitch = None
        # Get initial hotbar slot
        if self._client is not None:
            try:
                d = self._client.state or {}
                self._prev_slot = int(d.get("selected_slot",
                                             d.get("selectedSlot", 0)))
            except Exception:
                self._prev_slot = 0
        else:
            self._prev_slot = 0
        self._active = True

    def stop_capture(self):
        """Disable control capture (call when recording stops)."""
        self._active = False

    def get_state(self) -> np.ndarray:
        """Read current MCCTP state and return 28-dim control vector.

        Also stores the raw state dict in self._last_state_dict for
        game state encoding.
        """
        controls = np.zeros(NUM_CONTROLS, dtype=np.float32)

        if not self._active or self._client is None:
            self._last_state_dict = {}
            return controls

        try:
            d = self._client.state or {}
        except Exception:
            self._last_state_dict = {}
            return controls

        self._last_state_dict = d

        # --- Movement ---
        fwd = float(d.get("movement_forward",
                          d.get("movementForward", 0)))
        controls[CTRL_MOVE_FORWARD] = float(fwd > 0)
        controls[CTRL_MOVE_BACKWARD] = float(fwd < 0)

        side = float(d.get("movement_sideways",
                           d.get("movementSideways", 0)))
        controls[CTRL_STRAFE_LEFT] = float(side > 0)
        controls[CTRL_STRAFE_RIGHT] = float(side < 0)

        controls[CTRL_SPRINT] = float(d.get("input_sprint",
                                             d.get("sprint",
                                             d.get("is_sprinting", False))))
        controls[CTRL_SNEAK] = float(d.get("input_sneak",
                                            d.get("sneak",
                                            d.get("is_sneaking", False))))
        controls[CTRL_JUMP] = float(d.get("input_jump",
                                           d.get("jump",
                                           d.get("jumping", False))))

        # --- Combat ---
        controls[CTRL_ATTACK] = float(d.get("input_attack",
                                             d.get("attack",
                                             d.get("attacking", False))))
        controls[CTRL_USE_ITEM] = float(d.get("input_use_item",
                                               d.get("use_item",
                                               d.get("using_item", False))))

        # --- Look (yaw/pitch deltas in game units) ---
        if "yaw_delta" in d:
            # Mod provides pre-computed deltas (preferred)
            controls[CTRL_LOOK_YAW] = np.clip(
                float(d["yaw_delta"]) / LOOK_NORMALIZE_FACTOR, -1.0, 1.0
            )
            controls[CTRL_LOOK_PITCH] = np.clip(
                float(d.get("pitch_delta", 0)) / LOOK_NORMALIZE_FACTOR, -1.0, 1.0
            )
        else:
            # Compute from yaw/pitch changes
            curr_yaw = float(d.get("yaw", d.get("currentYaw", 0.0)))
            curr_pitch = float(d.get("pitch", d.get("currentPitch", 0.0)))

            if self._prev_yaw is not None:
                yaw_delta = curr_yaw - self._prev_yaw
                # Handle angle wrapping
                if yaw_delta > 180:
                    yaw_delta -= 360
                elif yaw_delta < -180:
                    yaw_delta += 360
                pitch_delta = curr_pitch - self._prev_pitch

                controls[CTRL_LOOK_YAW] = np.clip(
                    yaw_delta / LOOK_NORMALIZE_FACTOR, -1.0, 1.0
                )
                controls[CTRL_LOOK_PITCH] = np.clip(
                    pitch_delta / LOOK_NORMALIZE_FACTOR, -1.0, 1.0
                )

            self._prev_yaw = curr_yaw
            self._prev_pitch = curr_pitch

        # --- Utility ---
        controls[CTRL_DROP_ITEM] = float(d.get("input_drop",
                                                d.get("drop",
                                                d.get("drop_item", False))))
        controls[CTRL_SWAP_OFFHAND] = float(d.get("input_swap_offhand",
                                                    d.get("swap_offhand", False)))
        controls[CTRL_OPEN_INVENTORY] = float(d.get("input_open_inventory",
                                                      d.get("open_inventory", False)))

        # --- Hotbar (one-hot on change) ---
        current_slot = int(d.get("selected_slot",
                                  d.get("selectedSlot", 0)))
        if self._prev_slot is not None and current_slot != self._prev_slot:
            if 0 <= current_slot <= 8:
                controls[CTRL_HOTBAR_START + current_slot] = 1.0
        self._prev_slot = current_slot

        # --- Inventory (only when screen open) ---
        screen_open = d.get("screen_open", d.get("screenOpen", False))
        if screen_open:
            controls[CTRL_CURSOR_X] = float(d.get("cursor_x",
                                                    d.get("cursorX", 0.0)))
            controls[CTRL_CURSOR_Y] = float(d.get("cursor_y",
                                                    d.get("cursorY", 0.0)))
            controls[CTRL_INV_LEFT_CLICK] = float(d.get("mouse_left",
                                                          d.get("mouseLeft", False)))
            controls[CTRL_INV_RIGHT_CLICK] = float(d.get("mouse_right",
                                                           d.get("mouseRight", False)))
            controls[CTRL_INV_SHIFT_HELD] = float(d.get("shift_held",
                                                          d.get("shiftHeld", False)))

        return controls

    @property
    def last_state_dict(self) -> dict:
        """Return the raw MCCTP state dict from the last get_state() call."""
        return self._last_state_dict

    def close(self):
        """No cleanup needed (no background threads)."""
        self._active = False


# encode_game_state_v2 is imported from control_policy.py (canonical implementation)


# ---------------------------------------------------------------------------
# Frame feature extraction (260-dim raw vector)
# ---------------------------------------------------------------------------


def extract_frame_features(pose, hands) -> np.ndarray:
    """Extract 260-dim raw feature vector from current frame's pose + hands.

    Layout matches gesture_dataset.extract_features:
        pose_world(99) + left_hand_3d(63) + right_hand_3d(63) +
        left_present(1) + right_present(1) + pose_visibility(33) = 260
    """
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

    return np.concatenate([
        pose_world,                                        # 99
        left_3d,                                           # 63
        right_3d,                                          # 63
        np.array([left_present], dtype=np.float32),        # 1
        np.array([right_present], dtype=np.float32),       # 1
        pose_vis,                                          # 33
    ])  # total: 260


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def get_next_session_path():
    """Return the next auto-incremented session file path."""
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".npz")]
    if not existing:
        return os.path.join(RECORDINGS_DIR, "session_000.npz")
    max_idx = -1
    for f in existing:
        try:
            idx = int(f.replace("session_", "").replace(".npz", ""))
            max_idx = max(max_idx, idx)
        except ValueError:
            pass
    return os.path.join(RECORDINGS_DIR, f"session_{max_idx + 1:03d}.npz")


def get_session_count():
    """Count existing .npz session files."""
    if not os.path.isdir(RECORDINGS_DIR):
        return 0
    return sum(1 for f in os.listdir(RECORDINGS_DIR) if f.endswith(".npz"))


def get_last_session_path():
    """Return path to the most recently saved session, or None."""
    if not os.path.isdir(RECORDINGS_DIR):
        return None
    files = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".npz"))
    if not files:
        return None
    return os.path.join(RECORDINGS_DIR, files[-1])


def save_session(frames, controls, game_states, timestamps):
    """Save a recording session as compressed .npz (V2 format). Returns the file path.

    Args:
        frames: list of (260,) feature vectors per frame.
        controls: list of (28,) control vectors per frame.
        game_states: list of (46,) game state vectors per frame.
        timestamps: list of float64 timestamps.
    """
    path = get_next_session_path()

    frames_arr = np.array(frames, dtype=np.float32)       # (T, 260)
    controls_arr = np.array(controls, dtype=np.float32)    # (T, 28)
    gs_arr = np.array(game_states, dtype=np.float32)       # (T, 46)
    ts_arr = np.array(timestamps, dtype=np.float64)        # (T,)

    # Compute FPS
    if len(timestamps) > 1:
        duration = timestamps[-1] - timestamps[0]
        fps = len(timestamps) / duration if duration > 0 else 30.0
    else:
        fps = 30.0

    np.savez_compressed(
        path,
        frames=frames_arr,
        controls=controls_arr,
        game_state=gs_arr,
        timestamps=ts_arr,
        fps=np.float32(fps),
        version=np.int32(2),
        session_type="mixed",
    )
    return path


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_control_overlay(frame, controls, session_info):
    """Draw V2 control state panel and session info at the bottom of the frame."""
    h, w = frame.shape[:2]

    # Enlarged panel for 28 controls
    panel_h = 130
    panel_y = h - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # --- Row 1: Movement (6 controls) ---
    move_names = ["FWD", "BWD", "STR_L", "STR_R", "SPR", "SNK"]
    move_indices = [CTRL_MOVE_FORWARD, CTRL_MOVE_BACKWARD, CTRL_STRAFE_LEFT,
                    CTRL_STRAFE_RIGHT, CTRL_SPRINT, CTRL_SNEAK]
    row1_y = panel_y + 20
    spacing = max(60, (w - 300) // max(len(move_names), 1))

    for i, (name, idx) in enumerate(zip(move_names, move_indices)):
        active = controls[idx] > 0.5
        color = (0, 255, 200) if active else (80, 80, 80)
        sq_x = 15 + i * spacing
        if active:
            cv2.rectangle(frame, (sq_x, row1_y - 8), (sq_x + 10, row1_y + 2), color, -1)
        else:
            cv2.rectangle(frame, (sq_x, row1_y - 8), (sq_x + 10, row1_y + 2), color, 1)
        cv2.putText(frame, name, (sq_x + 14, row1_y), _FONT, 0.33, color, 1)

    # --- Row 2: Actions (6 controls) ---
    act_names = ["JMP", "ATK", "USE", "DRP", "SWP", "INV"]
    act_indices = [CTRL_JUMP, CTRL_ATTACK, CTRL_USE_ITEM,
                   CTRL_DROP_ITEM, CTRL_SWAP_OFFHAND, CTRL_OPEN_INVENTORY]
    row2_y = panel_y + 42

    for i, (name, idx) in enumerate(zip(act_names, act_indices)):
        active = controls[idx] > 0.5
        color = (0, 200, 255) if active else (80, 80, 80)
        sq_x = 15 + i * spacing
        if active:
            cv2.rectangle(frame, (sq_x, row2_y - 8), (sq_x + 10, row2_y + 2), color, -1)
        else:
            cv2.rectangle(frame, (sq_x, row2_y - 8), (sq_x + 10, row2_y + 2), color, 1)
        cv2.putText(frame, name, (sq_x + 14, row2_y), _FONT, 0.33, color, 1)

    # --- Row 3: Look + Hotbar ---
    row3_y = panel_y + 64
    yaw_val = controls[CTRL_LOOK_YAW]
    pitch_val = controls[CTRL_LOOK_PITCH]
    look_active = abs(yaw_val) > 0.08 or abs(pitch_val) > 0.08
    look_color = (0, 200, 255) if look_active else (80, 80, 80)
    cv2.putText(frame, f"LOOK ({yaw_val:+.2f},{pitch_val:+.2f})",
                (15, row3_y), _FONT, 0.35, look_color, 1)

    # Hotbar indicator
    hotbar_active = np.any(controls[CTRL_HOTBAR_START:CTRL_HOTBAR_END + 1] > 0.5)
    if hotbar_active:
        slot = np.argmax(controls[CTRL_HOTBAR_START:CTRL_HOTBAR_END + 1])
        hb_text = f"HB:{slot + 1}"
        hb_color = (255, 200, 0)
    else:
        hb_text = "HB:--"
        hb_color = (80, 80, 80)
    cv2.putText(frame, hb_text, (250, row3_y), _FONT, 0.35, hb_color, 1)

    # --- Row 4: Inventory ---
    row4_y = panel_y + 86
    screen_open = session_info.get("screen_open", False)
    screen_color = (200, 100, 255) if screen_open else (80, 80, 80)
    cv2.putText(frame, "SCR" if screen_open else "scr",
                (15, row4_y), _FONT, 0.35, screen_color, 1)

    if screen_open:
        cx = controls[CTRL_CURSOR_X]
        cy = controls[CTRL_CURSOR_Y]
        cv2.putText(frame, f"cur({cx:.2f},{cy:.2f})",
                    (55, row4_y), _FONT, 0.33, screen_color, 1)

        inv_names = ["L", "R", "SH"]
        inv_indices = [CTRL_INV_LEFT_CLICK, CTRL_INV_RIGHT_CLICK, CTRL_INV_SHIFT_HELD]
        for i, (name, idx) in enumerate(zip(inv_names, inv_indices)):
            active = controls[idx] > 0.5
            color = (200, 100, 255) if active else (80, 80, 80)
            cv2.putText(frame, name, (220 + i * 35, row4_y), _FONT, 0.33, color, 1)

    # Game state indicator
    gs_on = session_info.get("game_state_live", False)
    gs_color = (0, 200, 0) if gs_on else (80, 80, 80)
    gs_label = "LIVE" if gs_on else "OFF"
    cv2.putText(frame, f"GAME[{gs_label}]", (350, row3_y), _FONT, 0.35, gs_color, 1)

    # --- Session info (right side) ---
    if session_info["recording"]:
        elapsed = time.time() - session_info["start_time"]
        n_frames = session_info["frame_count"]
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        rec_text = f"REC {mins:02d}:{secs:02d} | {n_frames} frames"
        pulse = int(128 + 127 * math.sin(time.time() * 6))
        cv2.circle(frame, (w - 230, panel_y + 20), 6, (0, 0, pulse), -1)
        cv2.putText(frame, rec_text, (w - 215, panel_y + 25),
                    _FONT, 0.42, (0, 0, 255), 1)
    else:
        count = session_info["session_count"]
        cv2.putText(frame, f"Sessions: {count}", (w - 180, panel_y + 25),
                    _FONT, 0.42, (150, 150, 150), 1)
        cv2.putText(frame, "R=Record", (w - 180, panel_y + 50),
                    _FONT, 0.38, (100, 100, 100), 1)

    # V2 badge
    cv2.putText(frame, "V2", (w - 40, panel_y + 80),
                _FONT, 0.35, (100, 100, 100), 1)

    # --- Top-right status badge ---
    now = time.time()
    if session_info.get("countdown_active", False):
        status_text = "GET READY"
        status_color = (0, 200, 255)
    elif session_info["recording"]:
        status_text = "RECORDING"
        status_color = (0, 0, 255)
        cv2.rectangle(frame, (2, 2), (w - 3, panel_y - 2), (0, 0, 255), 3)
    elif session_info.get("saved_flash", False):
        status_text = "SAVED"
        status_color = (255, 200, 0)
    else:
        status_text = "READY"
        status_color = (0, 200, 0)

    tw = cv2.getTextSize(status_text, _FONT, 0.7, 2)[0][0]
    cv2.putText(frame, status_text, (w - tw - 10, 28), _FONT, 0.7,
                status_color, 2)


def draw_countdown(frame, remaining, panel_h):
    """Draw countdown number overlay on the camera area."""
    h, w = frame.shape[:2]
    cam_h = h - panel_h
    digit = str(int(math.ceil(remaining)))

    scale = 4.0
    thickness = 8
    tw, th = cv2.getTextSize(digit, _FONT, scale, thickness)[0]
    cx = (w - tw) // 2
    cy = (cam_h + th) // 2

    ov = frame.copy()
    cv2.circle(ov, (w // 2, cam_h // 2), 80, (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, digit, (cx, cy), _FONT, scale, (0, 200, 255), thickness)
    cv2.putText(frame, "Get ready...", ((w - 150) // 2, cy + 50),
                _FONT, 0.6, (180, 180, 180), 2)


def draw_help_overlay(frame):
    """Draw translucent help text overlay."""
    h, w = frame.shape[:2]
    help_lines = [
        "R           Start/stop recording",
        "BACKSPACE   Discard last session",
        "H           Toggle this help",
        "Q / ESC     Quit",
        "",
        "-- V2 Controls --",
        "Controls captured from MCCTP mod",
        "  Movement, combat, look, hotbar,",
        "  inventory cursor + clicks",
        "",
        "Requires --game flag for capture",
    ]
    box_w = 340
    box_h = len(help_lines) * 22 + 40
    bx = (w - box_w) // 2
    by = (h - box_h) // 2

    ov = frame.copy()
    cv2.rectangle(ov, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "CONTROLS V2", (bx + box_w // 2 - 70, by + 25),
                _FONT, 0.6, (0, 255, 255), 2)
    for i, line in enumerate(help_lines):
        cv2.putText(frame, line, (bx + 15, by + 50 + i * 22),
                    _FONT, 0.38, (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="V2 Control Policy Recording Tool")
    parser.add_argument("host", help="Phone IP address")
    parser.add_argument("--stream-port", type=int, default=5000)
    parser.add_argument("--control-port", type=int, default=5001)
    parser.add_argument("--game", action="store_true",
                        help="Connect to MCCTP for game state + control capture")
    parser.add_argument("--game-host", default="localhost",
                        help="MCCTP host (default: localhost)")
    parser.add_argument("--game-port", type=int, default=8765,
                        help="MCCTP WebSocket port (default: 8765)")
    args = parser.parse_args()

    host = args.host

    # --- Connect to camera ---
    print(f"Connecting control to {host}:{args.control_port}...")
    ctrl = CameraControl(host, args.control_port)

    print(f"Connecting stream to {host}:{args.stream_port}...")
    stream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stream_sock.connect((host, args.stream_port))
    print("Connected!")

    ctrl.auto_exposure()
    ctrl.auto_focus()
    ctrl.set_resolution(*DEFAULT_RESOLUTION)
    ctrl.set_jpeg_quality(DEFAULT_JPEG_QUALITY)
    ctrl.set_wb(DEFAULT_WB)

    # --- Init trackers ---
    print("Initializing hand tracker (WiLoR-mini)...")
    hand_tracker = HandTracker()
    print("Initializing pose tracker (MediaPipe)...")
    pose_tracker = PoseTracker()

    # --- MCCTP connection ---
    mcctp_client = None
    if args.game:
        try:
            from mcctp import SyncMCCTPClient
            mcctp_client = SyncMCCTPClient(args.game_host, args.game_port)
            mcctp_client.connect()
            print(f"[MCCTP] Connected to {args.game_host}:{args.game_port}")
        except Exception as e:
            print(f"[MCCTP] Failed to connect: {e}")
            print("  Continuing without game state (controls will be zeros)")
            mcctp_client = None

    # --- Control capture (MCCTP-based) ---
    input_capture = MCCTPControlCapture(mcctp_client)

    print("Ready.\n")
    print("Press R to start recording, H for help, Q to quit.")
    if mcctp_client:
        print("Game state: LIVE from MCCTP (28-dim controls)")
    else:
        print("Game state: OFFLINE (controls will be zeros — use --game)")

    # --- Recording state ---
    recording = False
    frames_data = []          # list of (260,) feature vectors
    controls_data = []        # list of (28,) control vectors
    game_states_data = []     # list of (46,) game state vectors
    timestamps_data = []
    record_start = 0.0
    saved_flash_until = 0.0
    countdown_until = 0.0
    show_help = False

    # FPS tracking
    frame_count = 0
    fps_start = time.time()
    display_fps = 0.0

    cv2.namedWindow("Control Recorder V2", cv2.WINDOW_NORMAL)
    grabber = FrameGrabber(stream_sock)

    try:
        while True:
            jpeg_data = grabber.get()
            if not jpeg_data:
                continue

            frame = cv2.imdecode(
                np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 0.5:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # --- Process trackers ---
            hands = hand_tracker.process(frame)
            pose = pose_tracker.process(frame)

            # Draw live skeletons
            hand_tracker.draw(frame, hands)
            if pose:
                pose_tracker.draw(frame, pose)

            # --- Countdown -> start recording ---
            now = time.time()
            if countdown_until > 0 and now >= countdown_until:
                countdown_until = 0.0
                recording = True
                frames_data = []
                controls_data = []
                game_states_data = []
                timestamps_data = []
                record_start = now
                input_capture.start_capture()
                winsound.Beep(1000, 100)
                print("Recording started!")

            # --- Build current control vector from MCCTP ---
            current_controls = input_capture.get_state()

            # Game state from MCCTP (or defaults if not connected)
            game_state = encode_game_state_v2(input_capture.last_state_dict)

            # Screen open flag for overlay
            screen_open = bool(game_state[38] > 0.0)

            # --- Record frame ---
            if recording:
                features = extract_frame_features(pose, hands)  # (260,)
                frames_data.append(features)
                controls_data.append(current_controls.copy())
                game_states_data.append(game_state)
                timestamps_data.append(time.time())

            # --- Draw overlays ---
            session_info = {
                "recording": recording,
                "start_time": record_start,
                "frame_count": len(frames_data),
                "session_count": get_session_count(),
                "saved_flash": now < saved_flash_until,
                "countdown_active": countdown_until > 0,
                "game_state_live": mcctp_client is not None,
                "screen_open": screen_open,
            }
            draw_control_overlay(frame, current_controls, session_info)

            # Countdown overlay
            if countdown_until > 0:
                remaining = countdown_until - now
                if remaining > 0:
                    draw_countdown(frame, remaining, panel_h=130)

            # FPS display
            cv2.putText(frame, f"FPS: {display_fps:.1f}",
                        (frame.shape[1] - 120, 55), _FONT, 0.45,
                        (0, 200, 0), 1)

            if show_help:
                draw_help_overlay(frame)

            cv2.imshow("Control Recorder V2", frame)
            key = cv2.waitKeyEx(1)

            # --- Key handling ---
            if key == -1:
                continue

            # During countdown: only cancel
            if countdown_until > 0:
                if key == ord('r') or key == ord('R') or key == 27:
                    countdown_until = 0.0
                    print("Countdown cancelled.")
                continue

            # During recording: only R (stop) and ESC (quit+save)
            if recording:
                if key == ord('r') or key == ord('R'):
                    recording = False
                    input_capture.stop_capture()
                    n = len(frames_data)
                    if n < MIN_SESSION_FRAMES:
                        print(f"Too short ({n} frames, need "
                              f"{MIN_SESSION_FRAMES}). Discarded.")
                    else:
                        path = save_session(frames_data, controls_data,
                                            game_states_data, timestamps_data)
                        dur = timestamps_data[-1] - timestamps_data[0]
                        saved_flash_until = time.time() + 1.5
                        winsound.Beep(800, 100)
                        print(f"Saved: {path} ({n} frames, {dur:.1f}s)")
                    frames_data = []
                    controls_data = []
                    game_states_data = []
                    timestamps_data = []
                elif key == 27 or key == ord('q') or key == ord('Q'):
                    recording = False
                    input_capture.stop_capture()
                    n = len(frames_data)
                    if n >= MIN_SESSION_FRAMES:
                        path = save_session(frames_data, controls_data,
                                            game_states_data, timestamps_data)
                        print(f"Auto-saved on exit: {path} ({n} frames)")
                    break
                continue

            # --- Normal mode (not recording) ---
            if key == ord('r') or key == ord('R'):
                countdown_until = time.time() + COUNTDOWN_SECONDS
                print(f"Recording in {COUNTDOWN_SECONDS}...")

            elif key == 8:  # Backspace
                last = get_last_session_path()
                if last and os.path.exists(last):
                    os.remove(last)
                    print(f"Discarded: {os.path.basename(last)}")
                else:
                    print("No sessions to discard.")

            elif key == ord('h') or key == ord('H'):
                show_help = not show_help

            elif key == 27 or key == ord('q') or key == ord('Q'):
                break

    except (ConnectionError, struct.error) as e:
        print(f"\nStream ended: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        # Auto-save if still recording
        if recording:
            input_capture.stop_capture()
            n = len(frames_data)
            if n >= MIN_SESSION_FRAMES:
                path = save_session(frames_data, controls_data,
                                    game_states_data, timestamps_data)
                print(f"Auto-saved: {path} ({n} frames)")

        input_capture.close()
        grabber.stop()
        hand_tracker.close()
        pose_tracker.close()
        if mcctp_client is not None:
            try:
                mcctp_client.disconnect()
            except Exception:
                pass
        stream_sock.close()
        ctrl.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
