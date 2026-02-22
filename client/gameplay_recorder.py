"""Gameplay Recorder — Screen + MCCTP Game State at 30fps.

Records simultaneously:
  - Screen capture → mp4 video (h264 via ffmpeg subprocess)
  - MCCTP game state + resolved inputs → npz data file
  - Frame-level sync: video frame N ↔ data row N

No phone camera needed — just screen + MCCTP connection.

Output per session:
  session_YYYYMMDD_HHMMSS.mp4  — screen recording
  session_YYYYMMDD_HHMMSS.npz  — controls, game_state, raw dicts, timestamps, etc.

Usage:
    python gameplay_recorder.py --host localhost --port 8765 --output ./recordings_gameplay/
    python gameplay_recorder.py  (uses defaults)

Global hotkeys (work while Minecraft has focus):
    F9      Start recording (with countdown) / Stop recording
    F10     Quit
"""

import argparse
import os
import sys
import time
import pickle
import platform
import subprocess
import threading
import numpy as np

# Add client dir to path for sibling imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from control_recorder import MCCTPControlCapture
from control_policy import encode_game_state_v2
from control_dataset import NUM_CONTROLS, GAME_STATE_DIM


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------

def get_screen_size():
    """Return (width, height) of the primary monitor via mss."""
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[1]  # primary monitor
        return mon["width"], mon["height"]


def grab_screen():
    """Capture primary monitor as a numpy BGR array."""
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[1]
        img = sct.grab(mon)
        # mss returns BGRA — drop alpha
        frame = np.array(img, dtype=np.uint8)[:, :, :3]
        return frame


class ScreenGrabber:
    """Persistent mss instance for faster repeated grabs."""

    def __init__(self, monitor_index=1):
        import mss
        self._sct = mss.mss()
        self._mon = self._sct.monitors[monitor_index]
        self.width = self._mon["width"]
        self.height = self._mon["height"]

    def grab(self):
        """Return BGR numpy array of the screen."""
        img = self._sct.grab(self._mon)
        return np.array(img, dtype=np.uint8)[:, :, :3]

    def close(self):
        self._sct.close()


# ---------------------------------------------------------------------------
# FFmpeg video writer (subprocess, h264)
# ---------------------------------------------------------------------------

class FFmpegWriter:
    """Writes raw BGR frames to an mp4 file via ffmpeg stdin pipe."""

    def __init__(self, output_path, width, height, fps=30):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self._frame_count = 0

        cmd = [
            "ffmpeg",
            "-y",                           # overwrite
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",                       # stdin
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_bgr):
        """Write one BGR frame. Frame must match (height, width, 3)."""
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
            self._frame_count += 1
        except BrokenPipeError:
            pass

    @property
    def frame_count(self):
        return self._frame_count

    def close(self):
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        self._proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Session data
# ---------------------------------------------------------------------------

def make_session_prefix(output_dir):
    """Generate session file prefix: output_dir/session_YYYYMMDD_HHMMSS"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"session_{ts}")


def get_recorder_id():
    """Machine identifier string."""
    return f"{platform.node()}_{platform.system()}"


def save_session_data(prefix, controls_list, game_states_list,
                      raw_dicts_list, timestamps_list, frame_indices_list,
                      target_fps, recorder_id):
    """Save the npz data file for a recording session."""
    npz_path = prefix + ".npz"

    controls_arr = np.array(controls_list, dtype=np.float32)       # (T, 28)
    gs_arr = np.array(game_states_list, dtype=np.float32)          # (T, 46)
    ts_arr = np.array(timestamps_list, dtype=np.float64)           # (T,)
    fi_arr = np.array(frame_indices_list, dtype=np.int32)          # (T,)

    # Pickle raw state dicts for debugging/reprocessing
    raw_pickled = pickle.dumps(raw_dicts_list)

    np.savez_compressed(
        npz_path,
        game_state=gs_arr,
        controls=controls_arr,
        raw_state_dicts=np.void(raw_pickled),
        timestamps=ts_arr,
        frame_indices=fi_arr,
        fps=np.float32(target_fps),
        version=np.int32(3),
        recorder_id=recorder_id,
    )
    return npz_path


# ---------------------------------------------------------------------------
# Global hotkey detection (works even when Minecraft has focus)
# ---------------------------------------------------------------------------

import ctypes

# Virtual key codes
VK_F9  = 0x78
VK_F10 = 0x79

def is_key_pressed(vk_code):
    """Check if a key was pressed since last check (global, any window).

    GetAsyncKeyState returns a short — if the most significant bit is set
    the key is currently down, if the least significant bit is set the key
    was pressed since the last call.
    """
    state = ctypes.windll.user32.GetAsyncKeyState(vk_code)
    return bool(state & 0x0001)  # "was pressed" bit


def countdown(seconds):
    """Blocking countdown with beeps. Returns True if completed, False if
    cancelled by F9 press."""
    import winsound
    deadline = time.perf_counter() + seconds
    last_printed = -1
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            winsound.Beep(1000, 200)  # GO beep
            print()
            return True

        # Check for cancel (F9 again)
        if is_key_pressed(VK_F9):
            print("\n  Cancelled.")
            return False

        secs = int(remaining) + 1
        if secs != last_printed:
            sys.stdout.write(f"\r  Starting in {secs}...")
            sys.stdout.flush()
            if secs <= 3:
                winsound.Beep(600, 80)
            last_printed = secs

        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Main recording loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gameplay Recorder — Screen + MCCTP at 30fps")
    parser.add_argument("--host", default="localhost",
                        help="MCCTP host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765,
                        help="MCCTP WebSocket port (default: 8765)")
    parser.add_argument("--output", default=os.path.join(_SCRIPT_DIR, "recordings_gameplay"),
                        help="Output directory for recordings")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS (default: 30)")
    parser.add_argument("--delay", type=int, default=5,
                        help="Countdown seconds before recording starts (default: 5)")
    parser.add_argument("--monitor", type=int, default=1,
                        help="Monitor index for mss (1=primary, default: 1)")
    args = parser.parse_args()

    TARGET_FPS = args.fps
    TICK_S = 1.0 / TARGET_FPS

    os.makedirs(args.output, exist_ok=True)

    # --- Connect to MCCTP ---
    print(f"Connecting to MCCTP at {args.host}:{args.port}...")
    mcctp_client = None
    try:
        from mcctp import SyncMCCTPClient
        mcctp_client = SyncMCCTPClient(args.host, args.port)
        mcctp_client.connect()
        print(f"[MCCTP] Connected!")
    except Exception as e:
        print(f"[MCCTP] Failed to connect: {e}")
        print("  Cannot record without MCCTP. Exiting.")
        sys.exit(1)

    # --- Init screen capture ---
    print("Initializing screen capture...")
    screen = ScreenGrabber(monitor_index=args.monitor)
    print(f"  Screen: {screen.width}x{screen.height}")

    # --- Init control capture ---
    control_capture = MCCTPControlCapture(mcctp_client)
    recorder_id = get_recorder_id()

    print(f"\nRecorder ID: {recorder_id}")
    print(f"Target FPS:  {TARGET_FPS}")
    print(f"Delay:       {args.delay}s countdown")
    print(f"Output dir:  {args.output}")
    print()
    print("=" * 50)
    print("  F9   Start recording (5s countdown) / Stop")
    print("  F10  Quit")
    print()
    print("  These work while Minecraft has focus!")
    print("=" * 50)
    print()

    # Drain any stale F9/F10 presses from before launch
    is_key_pressed(VK_F9)
    is_key_pressed(VK_F10)

    # --- State ---
    recording = False
    video_writer = None
    session_prefix = None

    controls_data = []
    game_states_data = []
    raw_dicts_data = []
    timestamps_data = []
    frame_indices_data = []
    frame_index = 0
    dropped_frames = 0

    # FPS tracking
    fps_counter = 0
    fps_timer = time.perf_counter()
    display_fps = 0.0

    def stop_and_save():
        """Stop current recording and save files."""
        nonlocal recording, video_writer
        recording = False
        control_capture.stop_capture()
        video_writer.close()

        n = len(controls_data)
        if n > 0:
            save_session_data(
                session_prefix, controls_data, game_states_data,
                raw_dicts_data, timestamps_data, frame_indices_data,
                TARGET_FPS, recorder_id,
            )
            duration = timestamps_data[-1] - timestamps_data[0] if n > 1 else 0
            print(f"\n*** RECORDING STOPPED ***")
            print(f"  Frames:  {n}")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Dropped: {dropped_frames}")
            print(f"  Files:   {os.path.basename(session_prefix)}.*")
        else:
            print("\n*** RECORDING STOPPED (no frames) ***")
        video_writer = None

    try:
        while True:
            tick_start = time.perf_counter()

            # --- Check global hotkeys ---
            if is_key_pressed(VK_F10):
                if recording:
                    stop_and_save()
                print("\nQuitting...")
                break

            if is_key_pressed(VK_F9):
                if not recording:
                    # --- Start with countdown ---
                    print(f"\n  F9 pressed — countdown {args.delay}s...")
                    if countdown(args.delay):
                        session_prefix = make_session_prefix(args.output)
                        mp4_path = session_prefix + ".mp4"

                        video_writer = FFmpegWriter(
                            mp4_path, screen.width, screen.height, TARGET_FPS
                        )

                        controls_data = []
                        game_states_data = []
                        raw_dicts_data = []
                        timestamps_data = []
                        frame_indices_data = []
                        frame_index = 0
                        dropped_frames = 0

                        control_capture.start_capture()
                        recording = True
                        print(f"*** RECORDING → {os.path.basename(session_prefix)}.* ***")

                        # Drain the key so it doesn't immediately stop
                        is_key_pressed(VK_F9)
                else:
                    # --- Stop recording ---
                    stop_and_save()

            # --- Capture tick ---
            if recording:
                # Grab screen
                screen_frame = screen.grab()

                # Poll MCCTP
                controls = control_capture.get_state()
                state_dict = dict(control_capture.last_state_dict)
                game_state = encode_game_state_v2(state_dict)
                timestamp = time.time()

                # Write video frame
                video_writer.write(screen_frame)

                # Store data
                controls_data.append(controls.copy())
                game_states_data.append(game_state)
                raw_dicts_data.append(state_dict)
                timestamps_data.append(timestamp)
                frame_indices_data.append(frame_index)
                frame_index += 1

            # --- FPS counter ---
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                display_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

                # Print status line
                if recording:
                    n = len(controls_data)
                    elapsed = timestamps_data[-1] - timestamps_data[0] if n > 1 else 0
                    sys.stdout.write(
                        f"\r  REC | FPS: {display_fps:5.1f} | "
                        f"Frames: {n:6d} | "
                        f"Time: {elapsed:6.1f}s | "
                        f"Dropped: {dropped_frames}    "
                    )
                    sys.stdout.flush()
                else:
                    sys.stdout.write(
                        f"\r  IDLE | FPS: {display_fps:5.1f} | "
                        f"F9=Record  F10=Quit             "
                    )
                    sys.stdout.flush()

            # --- Frame pacing ---
            tick_elapsed = time.perf_counter() - tick_start
            sleep_time = TICK_S - tick_elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)
            elif recording and tick_elapsed > TICK_S * 1.5:
                # Frame took too long
                dropped_frames += 1

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if recording and video_writer is not None:
            control_capture.stop_capture()
            video_writer.close()
            n = len(controls_data)
            if n > 0:
                save_session_data(
                    session_prefix, controls_data, game_states_data,
                    raw_dicts_data, timestamps_data, frame_indices_data,
                    TARGET_FPS, recorder_id,
                )
                print(f"\n  Auto-saved {n} frames on exit.")

        screen.close()
        if mcctp_client is not None:
            try:
                mcctp_client.disconnect()
            except Exception:
                pass
        print("\nDone.")


if __name__ == "__main__":
    main()
