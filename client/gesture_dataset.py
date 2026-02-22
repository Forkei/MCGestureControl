"""PyTorch Dataset for multi-label gesture classification from recorded pose+hand landmarks.

Loads .npz recordings from client/recordings/, extracts a 485-dim feature vector
per frame (260 raw + 225 velocity), and provides padded/masked sequences for a
Transformer model.

Each sample has a multi-label target vector (num_gestures,) where 1.0 indicates
the gesture is active. Multiple gestures can be active simultaneously (e.g.,
walking + crouching). "idle" and "invalid" directories map to all-zeros vectors.

Feature vector (485 dims per frame):
    -- Raw features (260) --
    pose_world      (33x3 = 99)  -- MediaPipe world coords in meters, hip-centered
    left_hand_3d    (21x3 = 63)  -- WiLoR left hand 3D coords
    right_hand_3d   (21x3 = 63)  -- WiLoR right hand 3D coords
    left_hand_present   (1)      -- 0/1 flag
    right_hand_present  (1)      -- 0/1 flag
    pose_visibility     (33)     -- per-landmark confidence
    -- Velocity features (225) --
    pose_world_vel      (99)     -- frame-to-frame delta of pose_world
    left_hand_3d_vel    (63)     -- frame-to-frame delta of left_hand_3d
    right_hand_3d_vel   (63)     -- frame-to-frame delta of right_hand_3d
"""

import os
import re
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(_SCRIPT_DIR, "recordings")

RAW_FEATURE_DIM = 260
_POSITION_DIMS = 225   # indices 0..224: pose_world(99) + left_hand(63) + right_hand(63)
VELOCITY_DIM = _POSITION_DIMS
INPUT_DIM = RAW_FEATURE_DIM + VELOCITY_DIM   # 485
DEFAULT_MAX_SEQ_LEN = 30
WINDOW_SIZE = 30
WINDOW_STRIDE = 5

# The 9 independent gesture labels (sigmoid outputs). Sorted alphabetically.
# "idle" and "invalid" are NOT labels -- they map to all-zeros.
GESTURE_LABELS = [
    "crouching",
    "draw_bow",
    "jump",
    "place_block",
    "sprinting",
    "swing",
    "sword_draw",
    "throw",
    "walking",
]

NUM_GESTURES = len(GESTURE_LABELS)
_LABEL_TO_INDEX = {name: i for i, name in enumerate(GESTURE_LABELS)}

# Directories that map to all-zeros (no gesture active)
_INACTIVE_DIRS = {"idle", "invalid"}


# ---------------------------------------------------------------------------
# Data augmentation helpers
# ---------------------------------------------------------------------------

def _time_warp(features: np.ndarray, rate_range=(0.8, 1.2)) -> np.ndarray:
    """Resample sequence to simulate speed changes. Input shape: (T, D)."""
    T, D = features.shape
    rate = random.uniform(*rate_range)
    new_T = max(1, int(round(T * rate)))
    old_indices = np.linspace(0, T - 1, new_T)
    old_grid = np.arange(T)
    # Vectorized: find interpolation weights once, apply to all dims
    idx = np.searchsorted(old_grid, old_indices, side='right') - 1
    idx = np.clip(idx, 0, T - 2)
    frac = (old_indices - old_grid[idx])
    frac = np.clip(frac, 0.0, 1.0)
    warped = features[idx] * (1.0 - frac[:, None]) + features[idx + 1] * frac[:, None]
    return warped.astype(np.float32)


def _add_noise(features: np.ndarray, std: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to landmark coordinates (not flags/visibility)."""
    noisy = features.copy()
    # Landmark coordinates: indices 0..224 (99 + 63 + 63 = 225 values)
    # Flags at 225, 226; visibility at 227..259
    noisy[:, :225] += np.random.normal(0, std, noisy[:, :225].shape).astype(np.float32)
    return noisy


def _random_temporal_crop(features: np.ndarray, max_trim_frac: float = 0.10) -> np.ndarray:
    """Randomly trim 0-max_trim_frac from start and end."""
    T = features.shape[0]
    trim_start = random.randint(0, max(0, int(T * max_trim_frac)))
    trim_end = random.randint(0, max(0, int(T * max_trim_frac)))
    end = max(trim_start + 1, T - trim_end)
    return features[trim_start:end]


# ---------------------------------------------------------------------------
# Velocity features
# ---------------------------------------------------------------------------

def append_velocity(features: np.ndarray) -> np.ndarray:
    """Append frame-to-frame velocity of position coordinates.

    Input: (T, 260) raw features.
    Output: (T, 485) features with velocity appended.

    Velocity is computed for the position coordinates (indices 0:225).
    First frame's velocity is zero.
    """
    T = features.shape[0]
    positions = features[:, :_POSITION_DIMS]  # (T, 225)
    velocity = np.zeros_like(positions)        # (T, 225)
    if T > 1:
        velocity[1:] = positions[1:] - positions[:-1]
    return np.concatenate([features, velocity], axis=1).astype(np.float32)  # (T, 485)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(data: dict) -> np.ndarray:
    """Extract 260-dim feature vectors from an .npz data dict.

    Returns: (T, 260) float32 array.
    """
    T = data["pose_world"].shape[0]

    pose_world = data["pose_world"].reshape(T, -1)             # (T, 99)
    left_hand_3d = data["left_hand_3d"].reshape(T, -1)         # (T, 63)
    right_hand_3d = data["right_hand_3d"].reshape(T, -1)       # (T, 63)
    left_present = data["left_hand_present"].reshape(T, 1).astype(np.float32)   # (T, 1)
    right_present = data["right_hand_present"].reshape(T, 1).astype(np.float32) # (T, 1)
    pose_vis = data["pose_visibility"]                          # (T, 33)

    features = np.concatenate([
        pose_world,       # 99
        left_hand_3d,     # 63
        right_hand_3d,    # 63
        left_present,     # 1
        right_present,    # 1
        pose_vis,         # 33
    ], axis=1)  # (T, 260)

    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Directory name parsing
# ---------------------------------------------------------------------------

def _parse_dir_to_label_vector(dirname: str) -> Optional[np.ndarray]:
    """Parse a recording directory name into a multi-hot label vector.

    Handles compound directories with inconsistent separators:
      "walking+crouching"       -> walking=1, crouching=1
      "draw_bowpluscrouching"   -> draw_bow=1, crouching=1
      "jumppluscrouch"          -> jump=1, crouching=1  (via crouch -> crouching alias)

    "idle" and "invalid" -> all-zeros vector.

    Returns:
        Float32 array of shape (NUM_GESTURES,), or None if the directory
        contains unknown gesture names (a warning is printed).
    """
    label_vec = np.zeros(NUM_GESTURES, dtype=np.float32)

    if dirname in _INACTIVE_DIRS:
        return label_vec

    # Split on "+" or "plus"
    parts = re.split(r'\+|plus', dirname)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Direct match
        if part in _LABEL_TO_INDEX:
            label_vec[_LABEL_TO_INDEX[part]] = 1.0
        # Handle common aliases/truncations (e.g., "crouch" -> "crouching")
        elif part + "ing" in _LABEL_TO_INDEX:
            label_vec[_LABEL_TO_INDEX[part + "ing"]] = 1.0
        else:
            print(f"WARNING: Unknown gesture '{part}' in directory '{dirname}' -- skipping directory")
            return None

    return label_vec


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GestureDataset(Dataset):
    """PyTorch Dataset for multi-label gesture sequences.

    Args:
        samples: List of (features_array, label_vector) tuples, where
                 label_vector is a float32 array of shape (NUM_GESTURES,).
        max_seq_len: Sequences are truncated/padded to this length.
        augment: Whether to apply data augmentation.
    """

    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray]],
                 max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
                 augment: bool = False):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, label_vec = self.samples[idx]
        features = features.copy()

        # Augmentation (on raw features before velocity computation)
        if self.augment:
            features = _random_temporal_crop(features)
            features = _time_warp(features)
            features = _add_noise(features)

        T = features.shape[0]

        # Truncate if too long
        if T > self.max_seq_len:
            features = features[:self.max_seq_len]
            T = self.max_seq_len

        # Compute velocity from the (possibly augmented) sequence
        features = append_velocity(features)  # (T, 260) -> (T, 485)

        # Pad to max_seq_len
        padded = np.zeros((self.max_seq_len, INPUT_DIM), dtype=np.float32)
        padded[:T] = features

        # Attention mask: 1 = real frame, 0 = padding
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:T] = 1.0

        return (
            torch.from_numpy(padded),       # (max_seq_len, 485)
            torch.from_numpy(mask),          # (max_seq_len,)
            torch.from_numpy(label_vec),     # (NUM_GESTURES,) float32
        )


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def scan_recordings(recordings_dir: str = RECORDINGS_DIR
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Scan recordings directory and load all samples with multi-label targets.

    Parses each subdirectory name to determine which gestures are active.
    Compound directories (e.g., "walking+crouching", "draw_bowpluscrouching")
    are split on '+' or 'plus' and mapped to multi-hot label vectors.

    Returns:
        samples: List of (features_array, label_vector) tuples, where
                 label_vector is a float32 array of shape (NUM_GESTURES,).
    """
    samples = []

    if not os.path.isdir(recordings_dir):
        print(f"WARNING: Recordings directory not found: {recordings_dir}")
        return samples

    for dirname in sorted(os.listdir(recordings_dir)):
        dir_path = os.path.join(recordings_dir, dirname)
        if not os.path.isdir(dir_path):
            continue

        npz_files = sorted(f for f in os.listdir(dir_path) if f.endswith(".npz"))
        if not npz_files:
            continue

        label_vec = _parse_dir_to_label_vector(dirname)
        if label_vec is None:
            # Unknown gesture name -- already warned in _parse_dir_to_label_vector
            continue

        for fname in npz_files:
            path = os.path.join(dir_path, fname)
            data = np.load(path, allow_pickle=True)
            features = extract_features(data)
            samples.append((features, label_vec.copy()))

    return samples


def _label_vector_to_key(label_vec: np.ndarray) -> str:
    """Convert a multi-hot label vector to a string key for stratification.

    Examples:
        [0,0,0,...] -> "idle"
        [1,0,1,...] -> "crouching+jump"  (uses sorted gesture names)
    """
    active = [GESTURE_LABELS[i] for i in range(NUM_GESTURES) if label_vec[i] > 0.5]
    return "+".join(active) if active else "idle"


def chunk_split_and_window(
    samples: List[Tuple[np.ndarray, np.ndarray]],
    val_fraction: float = 0.2,
    window_size: int = WINDOW_SIZE,
    window_stride: int = WINDOW_STRIDE,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]],
           List[Tuple[np.ndarray, np.ndarray]]]:
    """Split recordings into train/val by time chunks, then window each.

    For each recording, the first (1 - val_fraction) of frames go to train
    and the remaining frames go to val. This ensures:
      - Every gesture type appears in both train and val
      - No temporal overlap between train and val windows (no data leakage)
      - Long recordings contribute proportionally to both sets

    Short recordings (fewer than window_size / val_fraction frames) are
    assigned to train or val as a whole using stratified random assignment,
    since they can't be meaningfully split.

    Returns:
        (train_windows, val_windows) — each is a list of (window, label_vec).
    """
    rng = random.Random(seed)

    train_windows = []
    val_windows = []

    # Separate short recordings that can't be chunk-split
    short_samples: List[Tuple[np.ndarray, np.ndarray]] = []

    for features, label_vec in samples:
        T = features.shape[0]
        min_split_len = int(window_size / val_fraction) + 1

        if T < min_split_len:
            # Too short to split into meaningful train+val chunks
            short_samples.append((features, label_vec))
            continue

        # Split this recording's frames into contiguous train/val chunks
        split_idx = int(T * (1.0 - val_fraction))

        train_feat = features[:split_idx]
        val_feat = features[split_idx:]

        # Window each chunk
        train_windows.extend(_window_features(train_feat, label_vec, window_size, window_stride))
        val_windows.extend(_window_features(val_feat, label_vec, window_size, window_stride))

    # Handle short recordings: stratified random assignment
    by_key: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for sample in short_samples:
        key = _label_vector_to_key(sample[1])
        by_key.setdefault(key, []).append(sample)

    for key in sorted(by_key.keys()):
        group = by_key[key]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction)))
        for sample in group[:n_val]:
            val_windows.extend(_window_features(sample[0], sample[1], window_size, window_stride))
        for sample in group[n_val:]:
            train_windows.extend(_window_features(sample[0], sample[1], window_size, window_stride))

    rng.shuffle(train_windows)
    rng.shuffle(val_windows)
    return train_windows, val_windows


def _window_features(
    features: np.ndarray,
    label_vec: np.ndarray,
    window_size: int,
    window_stride: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Slice a single feature array into overlapping windows."""
    T = features.shape[0]
    windows = []
    if T <= window_size:
        windows.append((features, label_vec))
    else:
        for start in range(0, T - window_size + 1, window_stride):
            windows.append((features[start:start + window_size], label_vec))
        if (T - window_size) % window_stride != 0:
            windows.append((features[T - window_size:], label_vec))
    return windows
