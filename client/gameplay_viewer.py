"""Gameplay Viewer — Plays back recorded session with control overlay.

Shows the recorded mp4 with a real-time overlay of what controls were
active at each frame. Designed for the playback phase: watch this while
performing gestures so you know what keys to mimic.

Usage:
    python gameplay_viewer.py recordings_gameplay/session_20260222_133843
    python gameplay_viewer.py recordings_gameplay/session_20260222_133843.mp4
    python gameplay_viewer.py recordings_gameplay/session_20260222_133843 --speed 0.5

Controls:
    SPACE       Pause/resume
    S           Cycle speed (1x → 0.75x → 0.5x → 0.25x → 1x)
    LEFT/RIGHT  Skip back/forward 2 seconds (while paused or playing)
    Q/ESC       Quit
"""

import argparse
import os
import sys
import time
import math
import numpy as np

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from control_dataset import (
    CTRL_MOVE_FORWARD, CTRL_MOVE_BACKWARD, CTRL_STRAFE_LEFT, CTRL_STRAFE_RIGHT,
    CTRL_SPRINT, CTRL_SNEAK, CTRL_JUMP, CTRL_ATTACK, CTRL_USE_ITEM,
    CTRL_LOOK_YAW, CTRL_LOOK_PITCH, CTRL_DROP_ITEM, CTRL_SWAP_OFFHAND,
    CTRL_OPEN_INVENTORY, CTRL_HOTBAR_START, CTRL_HOTBAR_END,
    CTRL_CURSOR_X, CTRL_CURSOR_Y, CTRL_INV_LEFT_CLICK,
    CTRL_INV_RIGHT_CLICK, CTRL_INV_SHIFT_HELD,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Colors (BGR)
COL_BG = (20, 20, 20)
COL_ACTIVE = (0, 255, 200)       # cyan-green for active keys
COL_INACTIVE = (60, 60, 60)      # dim gray
COL_MOVE = (0, 255, 200)         # movement
COL_COMBAT = (0, 120, 255)       # orange-red for combat
COL_LOOK = (255, 200, 0)         # blue-yellow for look
COL_UTIL = (200, 100, 255)       # purple for utility
COL_HOTBAR = (0, 200, 255)       # yellow for hotbar
COL_INV = (255, 100, 200)        # pink for inventory
COL_WHITE = (255, 255, 255)
COL_GRAY = (120, 120, 120)
COL_RED = (0, 0, 255)


def draw_key_box(frame, x, y, w, h, label, active, color_active):
    """Draw a keyboard-key style box with label."""
    if active:
        # Filled bright box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_active, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), COL_WHITE, 1)
        text_color = (0, 0, 0)
    else:
        # Dim outline
        cv2.rectangle(frame, (x, y), (x + w, y + h), COL_INACTIVE, 1)
        text_color = COL_INACTIVE

    # Center text
    ts = cv2.getTextSize(label, FONT, 0.45, 1)[0]
    tx = x + (w - ts[0]) // 2
    ty = y + (h + ts[1]) // 2
    cv2.putText(frame, label, (tx, ty), FONT, 0.45, text_color, 1, cv2.LINE_AA)


def draw_look_indicator(frame, x, y, radius, yaw, pitch):
    """Draw a circular look joystick indicator."""
    # Background circle
    cv2.circle(frame, (x, y), radius, COL_INACTIVE, 1, cv2.LINE_AA)
    # Crosshair
    cv2.line(frame, (x - radius, y), (x + radius, y), COL_INACTIVE, 1)
    cv2.line(frame, (x, y - radius), (x, y + radius), COL_INACTIVE, 1)

    # Dot showing look direction (yaw = horizontal, pitch = vertical)
    dx = int(yaw * radius * 0.9)
    dy = int(pitch * radius * 0.9)
    dot_x = x + dx
    dot_y = y + dy

    active = abs(yaw) > 0.05 or abs(pitch) > 0.05
    dot_color = COL_LOOK if active else COL_GRAY
    cv2.circle(frame, (dot_x, dot_y), 5, dot_color, -1, cv2.LINE_AA)
    if active:
        cv2.line(frame, (x, y), (dot_x, dot_y), dot_color, 2, cv2.LINE_AA)

    # Label
    cv2.putText(frame, "LOOK", (x - 20, y + radius + 16),
                FONT, 0.4, COL_LOOK if active else COL_GRAY, 1, cv2.LINE_AA)


def draw_control_overlay(frame, controls, panel_x, panel_y):
    """Draw the full control overlay panel at (panel_x, panel_y)."""

    kw, kh = 48, 32   # key box size
    gap = 4            # gap between keys
    section_gap = 16   # gap between sections

    cx = panel_x
    cy = panel_y

    # --- Section: Movement (WASD layout) ---
    cv2.putText(frame, "MOVEMENT", (cx, cy), FONT, 0.4, COL_MOVE, 1, cv2.LINE_AA)
    cy += 6

    # W (forward) — centered above A S D
    w_x = cx + kw + gap
    draw_key_box(frame, w_x, cy, kw, kh, "W",
                 controls[CTRL_MOVE_FORWARD] > 0.5, COL_MOVE)

    # A S D row
    row_y = cy + kh + gap
    draw_key_box(frame, cx, row_y, kw, kh, "A",
                 controls[CTRL_STRAFE_LEFT] > 0.5, COL_MOVE)
    draw_key_box(frame, cx + kw + gap, row_y, kw, kh, "S",
                 controls[CTRL_MOVE_BACKWARD] > 0.5, COL_MOVE)
    draw_key_box(frame, cx + 2 * (kw + gap), row_y, kw, kh, "D",
                 controls[CTRL_STRAFE_RIGHT] > 0.5, COL_MOVE)

    # Sprint + Sneak + Jump below
    mod_y = row_y + kh + gap
    draw_key_box(frame, cx, mod_y, kw + 12, kh, "SPRINT",
                 controls[CTRL_SPRINT] > 0.5, COL_MOVE)
    draw_key_box(frame, cx + kw + 12 + gap, mod_y, kw + 12, kh, "SNEAK",
                 controls[CTRL_SNEAK] > 0.5, COL_MOVE)
    draw_key_box(frame, cx + 2 * (kw + 12 + gap), mod_y, kw + 12, kh, "JUMP",
                 controls[CTRL_JUMP] > 0.5, COL_MOVE)

    # --- Section: Combat ---
    cy = mod_y + kh + section_gap
    cv2.putText(frame, "COMBAT", (cx, cy), FONT, 0.4, COL_COMBAT, 1, cv2.LINE_AA)
    cy += 6

    bw = kw + 20  # wider boxes for combat
    draw_key_box(frame, cx, cy, bw, kh, "ATTACK",
                 controls[CTRL_ATTACK] > 0.5, COL_COMBAT)
    draw_key_box(frame, cx + bw + gap, cy, bw, kh, "USE",
                 controls[CTRL_USE_ITEM] > 0.5, COL_COMBAT)

    # --- Section: Utility ---
    cy += kh + section_gap
    cv2.putText(frame, "UTILITY", (cx, cy), FONT, 0.4, COL_UTIL, 1, cv2.LINE_AA)
    cy += 6

    draw_key_box(frame, cx, cy, kw + 6, kh, "DROP",
                 controls[CTRL_DROP_ITEM] > 0.5, COL_UTIL)
    draw_key_box(frame, cx + kw + 6 + gap, cy, kw + 6, kh, "SWAP",
                 controls[CTRL_SWAP_OFFHAND] > 0.5, COL_UTIL)
    draw_key_box(frame, cx + 2 * (kw + 6 + gap), cy, kw + 6, kh, "INV",
                 controls[CTRL_OPEN_INVENTORY] > 0.5, COL_UTIL)

    # --- Section: Look (joystick) ---
    cy += kh + section_gap
    yaw = controls[CTRL_LOOK_YAW]
    pitch = controls[CTRL_LOOK_PITCH]
    look_radius = 35
    draw_look_indicator(frame, cx + look_radius + 10, cy + look_radius, look_radius,
                        yaw, pitch)

    # Numeric readout next to joystick
    look_active = abs(yaw) > 0.05 or abs(pitch) > 0.05
    lc = COL_LOOK if look_active else COL_GRAY
    cv2.putText(frame, f"Yaw:  {yaw:+.2f}", (cx + 2 * look_radius + 25, cy + 15),
                FONT, 0.38, lc, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Pitch:{pitch:+.2f}", (cx + 2 * look_radius + 25, cy + 35),
                FONT, 0.38, lc, 1, cv2.LINE_AA)

    # --- Section: Hotbar ---
    cy += 2 * look_radius + section_gap
    cv2.putText(frame, "HOTBAR", (cx, cy), FONT, 0.4, COL_HOTBAR, 1, cv2.LINE_AA)
    cy += 6

    slot_w = 24
    slot_gap = 2
    active_slot = -1
    for i in range(9):
        if controls[CTRL_HOTBAR_START + i] > 0.5:
            active_slot = i
            break

    for i in range(9):
        sx = cx + i * (slot_w + slot_gap)
        active = (i == active_slot)
        draw_key_box(frame, sx, cy, slot_w, kh - 4, str(i + 1), active, COL_HOTBAR)

    # --- Section: Inventory (only if screen open) ---
    has_inv = (controls[CTRL_CURSOR_X] != 0 or controls[CTRL_CURSOR_Y] != 0 or
               controls[CTRL_INV_LEFT_CLICK] > 0.5 or
               controls[CTRL_INV_RIGHT_CLICK] > 0.5 or
               controls[CTRL_INV_SHIFT_HELD] > 0.5)

    if has_inv:
        cy += kh + section_gap - 4
        cv2.putText(frame, "INVENTORY", (cx, cy), FONT, 0.4, COL_INV, 1, cv2.LINE_AA)
        cy += 6

        cur_x = controls[CTRL_CURSOR_X]
        cur_y = controls[CTRL_CURSOR_Y]
        cv2.putText(frame, f"Cursor: ({cur_x:.1f}, {cur_y:.1f})",
                    (cx, cy + 14), FONT, 0.38, COL_INV, 1, cv2.LINE_AA)
        cy += 20

        draw_key_box(frame, cx, cy, kw, kh, "L-CLK",
                     controls[CTRL_INV_LEFT_CLICK] > 0.5, COL_INV)
        draw_key_box(frame, cx + kw + gap, cy, kw, kh, "R-CLK",
                     controls[CTRL_INV_RIGHT_CLICK] > 0.5, COL_INV)
        draw_key_box(frame, cx + 2 * (kw + gap), cy, kw, kh, "SHIFT",
                     controls[CTRL_INV_SHIFT_HELD] > 0.5, COL_INV)

    return frame


def draw_status_bar(frame, current_frame, total_frames, fps, speed, paused,
                    elapsed_s, total_s):
    """Draw bottom status bar with progress, time, speed."""
    h, w = frame.shape[:2]
    bar_h = 36
    bar_y = h - bar_h

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), COL_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Progress bar
    prog_x = 10
    prog_w = w - 20
    prog_y = bar_y + 4
    prog_h = 6
    progress = current_frame / max(total_frames, 1)

    cv2.rectangle(frame, (prog_x, prog_y), (prog_x + prog_w, prog_y + prog_h),
                  COL_INACTIVE, -1)
    fill_w = int(prog_w * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (prog_x, prog_y), (prog_x + fill_w, prog_y + prog_h),
                      COL_ACTIVE, -1)

    # Text
    text_y = bar_y + 28

    # Time
    def fmt_time(s):
        m = int(s) // 60
        sec = int(s) % 60
        return f"{m}:{sec:02d}"

    time_text = f"{fmt_time(elapsed_s)} / {fmt_time(total_s)}"
    cv2.putText(frame, time_text, (10, text_y), FONT, 0.45, COL_WHITE, 1, cv2.LINE_AA)

    # Frame counter
    frame_text = f"Frame {current_frame}/{total_frames}"
    cv2.putText(frame, frame_text, (160, text_y), FONT, 0.4, COL_GRAY, 1, cv2.LINE_AA)

    # Speed
    speed_text = f"{speed}x"
    cv2.putText(frame, speed_text, (340, text_y), FONT, 0.45,
                COL_HOTBAR if speed != 1.0 else COL_WHITE, 1, cv2.LINE_AA)

    # Paused indicator
    if paused:
        cv2.putText(frame, "PAUSED", (w - 100, text_y), FONT, 0.5,
                    COL_RED, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "PLAYING", (w - 110, text_y), FONT, 0.45,
                    COL_ACTIVE, 1, cv2.LINE_AA)

    # Controls hint
    cv2.putText(frame, "SPC=Pause  S=Speed  Arrows=Skip  Q=Quit",
                (420, text_y), FONT, 0.33, COL_GRAY, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description="Gameplay Viewer — recorded session with control overlay")
    parser.add_argument("session",
                        help="Session path (with or without extension)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Initial playback speed (default: 1.0)")
    parser.add_argument("--overlay-side", choices=["left", "right"], default="right",
                        help="Which side for the control overlay (default: right)")
    args = parser.parse_args()

    # --- Resolve session files ---
    base = args.session
    # Strip extension if provided
    for ext in (".mp4", ".npz"):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    mp4_path = base + ".mp4"
    npz_path = base + ".npz"

    if not os.path.isfile(mp4_path):
        print(f"ERROR: Video not found: {mp4_path}")
        sys.exit(1)
    if not os.path.isfile(npz_path):
        print(f"ERROR: Data not found: {npz_path}")
        sys.exit(1)

    # --- Load data ---
    print(f"Loading: {os.path.basename(base)}.*")
    data = np.load(npz_path, allow_pickle=True)
    controls = data["controls"]          # (T, 28)
    game_state = data["game_state"]      # (T, 46)
    timestamps = data["timestamps"]      # (T,)
    frame_indices = data["frame_indices"]  # (T,)
    target_fps = float(data["fps"])
    version = int(data["version"])

    T = controls.shape[0]
    total_s = timestamps[-1] - timestamps[0] if T > 1 else 0

    print(f"  {T} frames, {total_s:.1f}s, {target_fps:.0f}fps target, v{version}")

    # --- Open video ---
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {mp4_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video: {video_w}x{video_h}, {video_total} frames")

    # --- Build frame index → data index mapping ---
    # frame_indices maps data_row → video_frame, we need video_frame → data_row
    fi_to_data = {}
    for data_idx, vid_frame in enumerate(frame_indices):
        fi_to_data[int(vid_frame)] = data_idx

    # --- Layout: video + overlay panel ---
    PANEL_W = 220
    OVERLAY_MARGIN = 12

    # Scale video to fit nicely (max 900px tall)
    MAX_H = 900
    if video_h > MAX_H:
        scale = MAX_H / video_h
    else:
        scale = 1.0

    disp_w = int(video_w * scale)
    disp_h = int(video_h * scale)

    if args.overlay_side == "right":
        canvas_w = disp_w + PANEL_W
        video_x = 0
        panel_x = disp_w + OVERLAY_MARGIN
    else:
        canvas_w = disp_w + PANEL_W
        video_x = PANEL_W
        panel_x = OVERLAY_MARGIN

    canvas_h = disp_h

    # --- Playback state ---
    SPEEDS = [1.0, 0.75, 0.5, 0.25]
    speed_idx = 0
    for i, s in enumerate(SPEEDS):
        if abs(s - args.speed) < 0.01:
            speed_idx = i
            break
    speed = SPEEDS[speed_idx]

    paused = False
    current_frame = 0

    cv2.namedWindow("Gameplay Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gameplay Viewer", min(canvas_w, 1600), min(canvas_h, 950))

    print(f"\n  SPACE=Pause  S=Speed  LEFT/RIGHT=Skip  Q=Quit\n")

    last_frame_time = time.perf_counter()

    # Read first frame
    ret, video_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first video frame")
        cap.release()
        sys.exit(1)

    try:
        while True:
            now = time.perf_counter()
            frame_delay = 1.0 / (video_fps * speed)

            # --- Advance frame ---
            should_advance = (not paused) and (now - last_frame_time >= frame_delay)

            if should_advance:
                ret, video_frame = cap.read()
                if not ret:
                    print("\n  Playback complete!")
                    # Hold on last frame
                    paused = True
                    current_frame = min(current_frame, video_total - 1)
                else:
                    current_frame += 1
                    last_frame_time = now

            # --- Build canvas ---
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:] = COL_BG

            # Draw video
            if video_frame is not None:
                if scale != 1.0:
                    resized = cv2.resize(video_frame, (disp_w, disp_h))
                else:
                    resized = video_frame
                canvas[0:disp_h, video_x:video_x + disp_w] = resized

            # --- Get control data for this frame ---
            data_idx = fi_to_data.get(current_frame, None)
            if data_idx is None:
                # Find nearest
                best = min(range(T), key=lambda i: abs(int(frame_indices[i]) - current_frame))
                data_idx = best

            ctrl = controls[data_idx]

            # --- Draw overlay ---
            draw_control_overlay(canvas, ctrl, panel_x, OVERLAY_MARGIN)

            # --- Draw status bar ---
            elapsed = current_frame / video_fps if video_fps > 0 else 0
            total_time = video_total / video_fps if video_fps > 0 else 0
            draw_status_bar(canvas, current_frame, video_total, video_fps, speed,
                            paused, elapsed, total_time)

            cv2.imshow("Gameplay Viewer", canvas)

            # --- Key handling ---
            wait_ms = max(1, int(frame_delay * 1000) - 2) if not paused else 30
            key = cv2.waitKeyEx(wait_ms)

            if key == -1:
                continue
            elif key == 27 or key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' '):
                paused = not paused
                if not paused:
                    last_frame_time = time.perf_counter()
            elif key == ord('s') or key == ord('S'):
                speed_idx = (speed_idx + 1) % len(SPEEDS)
                speed = SPEEDS[speed_idx]
                print(f"  Speed: {speed}x")
            elif key == 2424832:  # LEFT arrow
                skip_frames = int(2 * video_fps)
                target = max(0, current_frame - skip_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, video_frame = cap.read()
                if ret:
                    current_frame = target
                last_frame_time = time.perf_counter()
            elif key == 2555904:  # RIGHT arrow
                skip_frames = int(2 * video_fps)
                target = min(video_total - 1, current_frame + skip_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, video_frame = cap.read()
                if ret:
                    current_frame = target
                last_frame_time = time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
