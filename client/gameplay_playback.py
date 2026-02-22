"""Gameplay Playback + Pose Capture Tool.

Phase 2 of training data collection: watches recorded gameplay video while
capturing body pose from the phone camera. Creates paired data:
    (pose, hands) ↔ (controls, game_state) for each video frame.

Requires:
  - A recorded .mp4 from gameplay_recorder.py
  - Phone camera connected (same socket protocol as stream_client.py)
  - NO MCCTP needed (offline replay)

Output:
  playback_<source_recording>.npz containing:
    frames: (T, 260)  — pose+hand features
    video_frame_indices: (T,) int32
    timestamps: (T,) float64
    source_recording: string

Usage:
    python gameplay_playback.py <phone_ip> recording.mp4
    python gameplay_playback.py <phone_ip> recording.mp4 --speed 0.5

Controls:
    SPACE   Pause/resume
    S       Toggle speed (1x ↔ 0.5x)
    Q/ESC   Quit (saves data)
"""

import argparse
import os
import sys
import socket
import struct
import time
import threading
import numpy as np

# Suppress Samsung's non-standard JPEG warnings from libjpeg
if sys.platform == "win32":
    _stderr_fd = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2

if sys.platform == "win32":
    os.dup2(_stderr_fd, 2)
    os.close(_stderr_fd)

# Add client dir to path for sibling imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from stream_client import FrameGrabber
from hand_tracker import HandTracker
from pose_tracker import PoseTracker
from control_recorder import extract_frame_features


# ---------------------------------------------------------------------------
# Camera frame receiver (simplified — no CameraControl needed)
# ---------------------------------------------------------------------------

def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


# ---------------------------------------------------------------------------
# Save paired data
# ---------------------------------------------------------------------------

def save_playback_data(output_path, frames_list, video_indices_list,
                       timestamps_list, source_recording):
    """Save the paired pose data as npz."""
    frames_arr = np.array(frames_list, dtype=np.float32)           # (T, 260)
    vi_arr = np.array(video_indices_list, dtype=np.int32)          # (T,)
    ts_arr = np.array(timestamps_list, dtype=np.float64)           # (T,)

    np.savez_compressed(
        output_path,
        frames=frames_arr,
        video_frame_indices=vi_arr,
        timestamps=ts_arr,
        source_recording=source_recording,
    )
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gameplay Playback + Pose Capture Tool")
    parser.add_argument("host", help="Phone IP address")
    parser.add_argument("video", help="Path to recorded .mp4 file")
    parser.add_argument("--stream-port", type=int, default=5000,
                        help="Camera stream port (default: 5000)")
    parser.add_argument("--control-port", type=int, default=5001,
                        help="Camera control port (default: 5001)")
    parser.add_argument("--speed", type=float, default=1.0,
                        choices=[0.5, 1.0],
                        help="Playback speed (0.5 or 1.0, default: 1.0)")
    parser.add_argument("--output", default=None,
                        help="Output npz path (auto-generated if omitted)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Hide camera preview window")
    args = parser.parse_args()

    # --- Validate video ---
    if not os.path.isfile(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {args.video}")
    print(f"  {video_w}x{video_h} @ {video_fps:.1f}fps, {total_frames} frames")
    print(f"  Duration: {total_frames / video_fps:.1f}s")

    # Output path
    if args.output:
        output_path = args.output
    else:
        video_basename = os.path.splitext(os.path.basename(args.video))[0]
        output_dir = os.path.dirname(args.video) or "."
        output_path = os.path.join(output_dir, f"playback_{video_basename}.npz")

    source_recording = os.path.basename(args.video)

    # --- Connect to phone camera ---
    print(f"\nConnecting to camera at {args.host}:{args.stream_port}...")
    stream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        stream_sock.connect((args.host, args.stream_port))
    except Exception as e:
        print(f"ERROR: Cannot connect to camera: {e}")
        cap.release()
        sys.exit(1)
    print("Camera connected!")

    # --- Init trackers ---
    print("Initializing hand tracker (WiLoR-mini)...")
    hand_tracker = HandTracker()
    print("Initializing pose tracker (MediaPipe)...")
    pose_tracker = PoseTracker()

    # --- State ---
    speed = args.speed
    paused = False
    current_video_frame = 0
    frame_delay = 1.0 / (video_fps * speed)  # seconds between frames

    # Collected data
    frames_data = []          # (260,) feature vectors
    video_indices_data = []   # which video frame was showing
    timestamps_data = []

    # FPS tracking
    fps_counter = 0
    fps_timer = time.perf_counter()
    display_fps = 0.0
    pose_fps_counter = 0
    pose_fps_timer = time.perf_counter()
    pose_display_fps = 0.0

    # Scale video for display (cap at 960px wide)
    DISPLAY_MAX_W = 960
    if video_w > DISPLAY_MAX_W:
        scale = DISPLAY_MAX_W / video_w
        display_w = DISPLAY_MAX_W
        display_h = int(video_h * scale)
    else:
        display_w = video_w
        display_h = video_h

    print(f"\nPlayback speed: {speed}x")
    print(f"Frame delay:   {frame_delay * 1000:.1f}ms")
    print()
    print("=" * 50)
    print("  SPACE = Pause/Resume")
    print("  S     = Toggle speed (1x ↔ 0.5x)")
    print("  Q/ESC = Quit (saves data)")
    print("=" * 50)
    print()

    cv2.namedWindow("Gameplay Playback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gameplay Playback", display_w, display_h)

    if not args.no_preview:
        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Preview", 320, 240)

    grabber = FrameGrabber(stream_sock)
    last_frame_time = time.perf_counter()
    video_frame_bgr = None

    try:
        while True:
            loop_start = time.perf_counter()

            # --- Video frame advancement ---
            if not paused:
                now = time.perf_counter()
                if now - last_frame_time >= frame_delay:
                    ret, video_frame_bgr = cap.read()
                    if not ret:
                        print("\n\nVideo playback complete!")
                        break
                    current_video_frame += 1
                    last_frame_time = now

            # --- Display video frame ---
            if video_frame_bgr is not None:
                display_frame = video_frame_bgr.copy()

                # Resize for display
                if display_w != video_w:
                    display_frame = cv2.resize(display_frame, (display_w, display_h))

                # Overlay: frame counter
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"Frame: {current_video_frame}/{total_frames}"
                cv2.putText(display_frame, text, (10, 30), font, 0.7,
                            (0, 255, 0), 2)

                # Progress bar
                progress = current_video_frame / max(total_frames, 1)
                bar_w = display_w - 20
                bar_y = display_h - 20
                cv2.rectangle(display_frame, (10, bar_y), (10 + bar_w, bar_y + 10),
                              (50, 50, 50), -1)
                cv2.rectangle(display_frame, (10, bar_y),
                              (10 + int(bar_w * progress), bar_y + 10),
                              (0, 255, 0), -1)

                # Status
                status = "PAUSED" if paused else f"{speed}x"
                cv2.putText(display_frame, status, (display_w - 100, 30),
                            font, 0.7, (0, 255, 255) if paused else (0, 200, 0), 2)

                # Pose capture FPS
                cv2.putText(display_frame, f"Pose FPS: {pose_display_fps:.1f}",
                            (10, 60), font, 0.5, (255, 200, 0), 1)
                cv2.putText(display_frame, f"Captured: {len(frames_data)}",
                            (10, 80), font, 0.5, (255, 200, 0), 1)

                cv2.imshow("Gameplay Playback", display_frame)

            # --- Camera pose capture (runs regardless of pause state) ---
            cam_jpeg = grabber.get(timeout=0.005)  # short timeout, non-blocking-ish
            if cam_jpeg:
                cam_frame = cv2.imdecode(
                    np.frombuffer(cam_jpeg, np.uint8), cv2.IMREAD_COLOR)

                if cam_frame is not None:
                    # Run trackers
                    hands = hand_tracker.process(cam_frame)
                    pose = pose_tracker.process(cam_frame)

                    # Extract 260-dim features
                    features = extract_frame_features(pose, hands)

                    # Record — synced to current video frame
                    frames_data.append(features)
                    video_indices_data.append(current_video_frame)
                    timestamps_data.append(time.time())

                    # Pose FPS
                    pose_fps_counter += 1
                    pnow = time.perf_counter()
                    if pnow - pose_fps_timer >= 1.0:
                        pose_display_fps = pose_fps_counter / (pnow - pose_fps_timer)
                        pose_fps_counter = 0
                        pose_fps_timer = pnow

                    # Camera preview
                    if not args.no_preview:
                        hand_tracker.draw(cam_frame, hands)
                        if pose:
                            pose_tracker.draw(cam_frame, pose)
                        # Resize for preview
                        preview = cv2.resize(cam_frame, (320, 240))
                        cv2.imshow("Camera Preview", preview)

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                state_str = "PAUSED" if paused else "PLAYING"
                print(f"\n  [{state_str}]")
            elif key == ord('s'):
                speed = 0.5 if speed == 1.0 else 1.0
                frame_delay = 1.0 / (video_fps * speed)
                print(f"\n  Speed: {speed}x (delay: {frame_delay * 1000:.1f}ms)")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        # --- Save data ---
        n = len(frames_data)
        if n > 0:
            save_playback_data(
                output_path, frames_data, video_indices_data,
                timestamps_data, source_recording,
            )
            print(f"\nSaved {n} pose frames → {output_path}")
            print(f"  Video frames covered: {video_indices_data[0]}–{video_indices_data[-1]}")
            unique_vf = len(set(video_indices_data))
            print(f"  Unique video frames:  {unique_vf}/{total_frames}")
        else:
            print("\nNo pose data captured.")

        # Cleanup
        cap.release()
        grabber.stop()
        hand_tracker.close()
        pose_tracker.close()
        stream_sock.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
