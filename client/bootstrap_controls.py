"""Convert existing gesture recordings to control-label format.

Reads .npz clips from client/recordings/<gesture_name>/ and writes new .npz
files to client/recordings_control/ in the session-like format expected by
control_dataset.py.

Each output file contains:
    pose_world          (T, 33, 3)   float32
    pose_visibility     (T, 33)      float32
    left_hand_3d        (T, 21, 3)   float32
    right_hand_3d       (T, 21, 3)   float32
    left_hand_present   (T,)         float32
    right_hand_present  (T,)         float32
    game_state          (T, 16)      float32   (default values)
    controls            (T, 10)      float32
    timestamps          (T,)         float64
    session_type        str
    control_method      str  ("bootstrap")

Conversion table (from plans/reference.md):
    idle              -> all zeros
    walking           -> forward=1
    sprinting         -> forward=1, sprint=1
    crouching         -> sneak=1
    swing             -> attack=1  (pulse at peak hand velocity)
    draw_bow          -> use_item=1 (held entire clip)
    jump              -> jump=1    (pulse at peak upward body velocity)
    throw             -> attack=1  (pulse at peak hand velocity)
    place_block       -> use_item=1 (pulse at peak hand velocity)
    walking+crouching -> forward=1, sneak=1
    sword_draw        -> skipped (context-dependent, needs game state)
    invalid           -> all zeros (treated as idle)

Usage:
    python bootstrap_controls.py [--recordings_dir DIR] [--output_dir DIR]
"""

import os
import sys
import argparse
from typing import Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RECORDINGS_DIR = os.path.join(_SCRIPT_DIR, "recordings")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "recordings_control")

# Control vector indices (10-dim)
CTRL_FORWARD = 0
CTRL_STRAFE_LEFT = 1
CTRL_STRAFE_RIGHT = 2
CTRL_SPRINT = 3
CTRL_SNEAK = 4
CTRL_JUMP = 5
CTRL_ATTACK = 6
CTRL_USE_ITEM = 7
CTRL_LOOK_YAW = 8
CTRL_LOOK_PITCH = 9
NUM_CONTROLS = 10

# Default game state (16-dim) for bootstrapped data:
# No item properties, full health/hunger, on ground
DEFAULT_GAME_STATE = np.zeros(16, dtype=np.float32)
DEFAULT_GAME_STATE[6] = 1.0   # is_empty = 1
DEFAULT_GAME_STATE[7] = 1.0   # health = 1.0 (max)
DEFAULT_GAME_STATE[8] = 1.0   # hunger = 1.0 (max)
DEFAULT_GAME_STATE[9] = 1.0   # on_ground = 1

# Combat default: holding a melee weapon
COMBAT_GAME_STATE = DEFAULT_GAME_STATE.copy()
COMBAT_GAME_STATE[0] = 1.0    # can_melee = 1
COMBAT_GAME_STATE[6] = 0.0    # is_empty = 0

# Bow default: holding a bow
BOW_GAME_STATE = DEFAULT_GAME_STATE.copy()
BOW_GAME_STATE[1] = 1.0       # can_shoot = 1
BOW_GAME_STATE[6] = 0.0       # is_empty = 0

# Block placement default: holding a block
BLOCK_GAME_STATE = DEFAULT_GAME_STATE.copy()
BLOCK_GAME_STATE[2] = 1.0     # can_place = 1
BLOCK_GAME_STATE[6] = 0.0     # is_empty = 0

# Directories to skip
SKIP_DIRS = {"sword_draw"}

# Pulse half-width in frames (pulse spans [peak-PULSE_HALF, peak+PULSE_HALF])
PULSE_HALF = 2


# ---------------------------------------------------------------------------
# Pulse labeling helpers
# ---------------------------------------------------------------------------

def _hand_velocity_magnitude(data: dict) -> np.ndarray:
    """Compute per-frame hand velocity magnitude (max of left/right).

    Returns (T,) array of velocity magnitudes.
    """
    T = data["right_hand_3d"].shape[0]
    vel = np.zeros(T, dtype=np.float32)

    for hand_key, present_key in [("right_hand_3d", "right_hand_present"),
                                   ("left_hand_3d", "left_hand_present")]:
        hand = data[hand_key].reshape(T, -1)  # (T, 63)
        present = data[present_key].astype(np.float32)

        if T > 1:
            deltas = np.zeros_like(hand)
            deltas[1:] = hand[1:] - hand[:-1]
            frame_vel = np.linalg.norm(deltas, axis=1)  # (T,)
            # Only count frames where hand is present
            frame_vel *= present
            vel = np.maximum(vel, frame_vel)

    return vel


def _body_vertical_velocity(data: dict) -> np.ndarray:
    """Compute per-frame upward body velocity (from hip landmarks).

    Uses the average Y displacement of hip landmarks (indices 23, 24).
    Positive = upward movement (Y axis points down in MediaPipe, so we negate).

    Returns (T,) array.
    """
    T = data["pose_world"].shape[0]
    # Hip landmarks: left hip = 23, right hip = 24
    hips = (data["pose_world"][:, 23, :] + data["pose_world"][:, 24, :]) / 2.0  # (T, 3)
    vel = np.zeros(T, dtype=np.float32)
    if T > 1:
        # Y axis: negative delta = moving up in world coords
        vel[1:] = -(hips[1:, 1] - hips[:-1, 1])
    return vel


def _pulse_at_peak(signal: np.ndarray, half_width: int = PULSE_HALF) -> np.ndarray:
    """Create a binary pulse centered on the peak of a signal.

    Returns (T,) float32 array with 1.0 at [peak-half_width, peak+half_width].
    """
    T = len(signal)
    peak_idx = int(np.argmax(signal))
    pulse = np.zeros(T, dtype=np.float32)
    start = max(0, peak_idx - half_width)
    end = min(T, peak_idx + half_width + 1)
    pulse[start:end] = 1.0
    return pulse


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

# Mapping: directory name -> (label_type, control_indices, game_state)
# label_type: "held" = active entire clip, "pulse_hand" = pulse at peak hand vel,
#             "pulse_body" = pulse at peak body vertical vel, "zeros" = all zeros
GESTURE_MAP = {
    "idle":               ("zeros",      [],                              DEFAULT_GAME_STATE),
    "invalid":            ("zeros",      [],                              DEFAULT_GAME_STATE),
    "walking":            ("held",       [CTRL_FORWARD],                  DEFAULT_GAME_STATE),
    "sprinting":          ("held",       [CTRL_FORWARD, CTRL_SPRINT],     DEFAULT_GAME_STATE),
    "crouching":          ("held",       [CTRL_SNEAK],                    DEFAULT_GAME_STATE),
    "swing":              ("pulse_hand", [CTRL_ATTACK],                   COMBAT_GAME_STATE),
    "draw_bow":           ("held",       [CTRL_USE_ITEM],                 BOW_GAME_STATE),
    "jump":               ("pulse_body", [CTRL_JUMP],                     DEFAULT_GAME_STATE),
    "throw":              ("pulse_hand", [CTRL_ATTACK],                   COMBAT_GAME_STATE),
    "place_block":        ("pulse_hand", [CTRL_USE_ITEM],                 BLOCK_GAME_STATE),
    "walking+crouching":  ("held",       [CTRL_FORWARD, CTRL_SNEAK],      DEFAULT_GAME_STATE),
}


def _parse_compound_dir(dirname: str):
    """Parse a compound directory name like 'draw_bowpluscrouching' or 'jumppluscrouch'.

    Returns a list of (label_type, control_indices, game_state) for each component,
    or None if any component is unknown or should be skipped.
    """
    import re
    parts = re.split(r'\+|plus', dirname)

    components = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in SKIP_DIRS:
            return None
        if part in GESTURE_MAP:
            components.append(GESTURE_MAP[part])
        # Try common aliases
        elif part + "ing" in GESTURE_MAP:
            components.append(GESTURE_MAP[part + "ing"])
        elif part == "crouch":
            components.append(GESTURE_MAP["crouching"])
        else:
            return None

    return components if components else None


def convert_gesture_recording(npz_path: str, gesture_name: str) -> Optional[dict]:
    """Convert a gesture recording to control-label format.

    Args:
        npz_path: Path to the .npz recording file.
        gesture_name: Directory name of the gesture class.

    Returns:
        Dict with all fields for the output .npz, or None to skip.
    """
    if gesture_name in SKIP_DIRS:
        return None

    data = np.load(npz_path, allow_pickle=True)
    T = data["pose_world"].shape[0]
    if T < 2:
        return None

    controls = np.zeros((T, NUM_CONTROLS), dtype=np.float32)

    # Try direct lookup first, then compound parsing
    if gesture_name in GESTURE_MAP:
        components = [GESTURE_MAP[gesture_name]]
    else:
        components = _parse_compound_dir(gesture_name)
        if components is None:
            return None

    # Merge components: union of control activations, pick most specific game state
    game_state = DEFAULT_GAME_STATE.copy()
    for label_type, ctrl_indices, gs in components:
        # Use the most specialized game state (non-default)
        if not np.array_equal(gs, DEFAULT_GAME_STATE):
            game_state = gs.copy()

        if label_type == "zeros":
            pass  # controls stay zero
        elif label_type == "held":
            for idx in ctrl_indices:
                controls[:, idx] = 1.0
        elif label_type == "pulse_hand":
            pulse = _pulse_at_peak(_hand_velocity_magnitude(data))
            for idx in ctrl_indices:
                controls[:, idx] = pulse
        elif label_type == "pulse_body":
            pulse = _pulse_at_peak(_body_vertical_velocity(data))
            for idx in ctrl_indices:
                controls[:, idx] = pulse

    # Build game state array (same every frame for bootstrapped data)
    game_state_arr = np.tile(game_state, (T, 1))  # (T, 16)

    # Ensure consistent dtypes
    return {
        "pose_world": data["pose_world"].astype(np.float32),
        "pose_visibility": data["pose_visibility"].astype(np.float32),
        "left_hand_3d": data["left_hand_3d"].astype(np.float32),
        "right_hand_3d": data["right_hand_3d"].astype(np.float32),
        "left_hand_present": data["left_hand_present"].astype(np.float32),
        "right_hand_present": data["right_hand_present"].astype(np.float32),
        "game_state": game_state_arr,
        "controls": controls,
        "timestamps": data["timestamps"].astype(np.float64),
        "session_type": gesture_name,
        "control_method": "bootstrap",
    }


def convert_all(recordings_dir: str = DEFAULT_RECORDINGS_DIR,
                output_dir: str = DEFAULT_OUTPUT_DIR) -> int:
    """Convert all gesture recordings to control-label format.

    Returns the number of files written.
    """
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    skipped_dirs = set()

    for dirname in sorted(os.listdir(recordings_dir)):
        dir_path = os.path.join(recordings_dir, dirname)
        if not os.path.isdir(dir_path):
            continue

        if dirname in SKIP_DIRS:
            skipped_dirs.add(dirname)
            continue

        npz_files = sorted(f for f in os.listdir(dir_path) if f.endswith(".npz"))
        if not npz_files:
            continue

        for fname in npz_files:
            npz_path = os.path.join(dir_path, fname)
            result = convert_gesture_recording(npz_path, dirname)
            if result is None:
                continue

            # Output filename: <gesture_name>_<sample>.npz
            out_name = f"{dirname}_{fname}"
            out_path = os.path.join(output_dir, out_name)
            np.savez_compressed(out_path, **result)
            written += 1

        print(f"  {dirname}: {len(npz_files)} files converted")

    if skipped_dirs:
        print(f"  Skipped directories: {', '.join(sorted(skipped_dirs))}")

    return written


def print_summary(output_dir: str = DEFAULT_OUTPUT_DIR):
    """Print statistics about the converted dataset."""
    if not os.path.isdir(output_dir):
        print("No output directory found.")
        return

    files = sorted(f for f in os.listdir(output_dir) if f.endswith(".npz"))
    total_frames = 0
    ctrl_counts = np.zeros(NUM_CONTROLS, dtype=np.int64)
    ctrl_names = ["forward", "strafe_L", "strafe_R", "sprint", "sneak",
                  "jump", "attack", "use_item", "look_yaw", "look_pitch"]

    for fname in files:
        data = np.load(os.path.join(output_dir, fname), allow_pickle=True)
        controls = data["controls"]
        T = controls.shape[0]
        total_frames += T
        for i in range(8):  # binary controls only
            ctrl_counts[i] += int(np.sum(controls[:, i] > 0.5))

    print(f"\nBootstrapped dataset: {len(files)} files, {total_frames} frames")
    idle_frames = total_frames - int(np.max(ctrl_counts[:8]))
    for i in range(8):
        pct = 100.0 * ctrl_counts[i] / total_frames if total_frames > 0 else 0
        print(f"  {ctrl_names[i]:>12s}: {ctrl_counts[i]:>6d} ({pct:5.1f}%)")
    print(f"  {'idle':>12s}: ~{idle_frames:>5d} ({100.0 * idle_frames / total_frames if total_frames else 0:5.1f}%)")
    print(f"  look_yaw/pitch: all zeros (no look data in bootstrapped recordings)")


def main():
    parser = argparse.ArgumentParser(description="Convert gesture recordings to control-label format")
    parser.add_argument("--recordings_dir", default=DEFAULT_RECORDINGS_DIR,
                        help="Input recordings directory")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for converted files")
    args = parser.parse_args()

    print(f"Converting gesture recordings from: {args.recordings_dir}")
    print(f"Output directory: {args.output_dir}")
    count = convert_all(args.recordings_dir, args.output_dir)
    print(f"\nConverted {count} recordings.")
    print_summary(args.output_dir)


if __name__ == "__main__":
    main()
