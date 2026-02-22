"""PyTorch Dataset for control-policy training from recorded sessions (V2).

Loads .npz session recordings from client/recordings_control/, extracts a
671-dim feature vector per frame, and provides windowed sequences for a
Transformer model that predicts a 28-dim control vector via 5 output heads.

Feature vector (671 dims per frame):
    -- Raw features (260) --
    pose_world      (33x3 = 99)  -- MediaPipe world coords in meters, hip-centered
    left_hand_3d    (21x3 = 63)  -- WiLoR left hand 3D coords
    right_hand_3d   (21x3 = 63)  -- WiLoR right hand 3D coords
    left_hand_present   (1)      -- 0/1 flag
    right_hand_present  (1)      -- 0/1 flag
    pose_visibility     (33)     -- per-landmark confidence
    -- Velocity features (225) --
    Computed via append_velocity() from gesture_dataset.py
    -- Game state (46) --
    Item context, vitals, movement, combat, crosshair, threats, environment,
    status effects, extra
    -- Action history (140) --
    Flattened controls from previous K=5 frames (5 x 28)

Target: control vector at the LAST frame of each window (28 dims).
    0-8:   binary gameplay (forward, backward, strafe_l/r, sprint, sneak, jump, attack, use_item)
    9-10:  analog look (yaw, pitch) in [-1, 1]
    11-13: binary utility (drop, swap_offhand, open_inventory)
    14-22: one-hot hotbar (9 slots)
    23-24: analog cursor (x, y) in [0, 1]
    25-27: binary inv clicks (left, right, shift_held)
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from gesture_dataset import (
    append_velocity, extract_features, _POSITION_DIMS,
    RAW_FEATURE_DIM, VELOCITY_DIM, _time_warp, _add_noise,
    _random_temporal_crop,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RECORDINGS_DIR = os.path.join(_SCRIPT_DIR, "recordings_control")

# ---------------------------------------------------------------------------
# V2 Dimension Constants (Contract 1)
# ---------------------------------------------------------------------------

GAME_STATE_DIM = 46
ACTION_HISTORY_K = 5
NUM_CONTROLS = 28
ACTION_HISTORY_DIM = ACTION_HISTORY_K * NUM_CONTROLS  # 140
INPUT_DIM = RAW_FEATURE_DIM + VELOCITY_DIM + GAME_STATE_DIM + ACTION_HISTORY_DIM  # 671

# Windowing
WINDOW_SIZE = 30
WINDOW_STRIDE = 1  # v2 uses stride 1 (was 5 in v1)

# Action history dropout rate during training (prevents copying history)
ACTION_HISTORY_DROPOUT = 0.3

# ---------------------------------------------------------------------------
# V2 Control Vector Layout (Contract 2)
# ---------------------------------------------------------------------------

CTRL_MOVE_FORWARD    = 0
CTRL_MOVE_BACKWARD   = 1
CTRL_STRAFE_LEFT     = 2
CTRL_STRAFE_RIGHT    = 3
CTRL_SPRINT          = 4
CTRL_SNEAK           = 5
CTRL_JUMP            = 6
CTRL_ATTACK          = 7
CTRL_USE_ITEM        = 8
CTRL_LOOK_YAW        = 9
CTRL_LOOK_PITCH      = 10
CTRL_DROP_ITEM       = 11
CTRL_SWAP_OFFHAND    = 12
CTRL_OPEN_INVENTORY  = 13
CTRL_HOTBAR_START    = 14
CTRL_HOTBAR_END      = 22
CTRL_CURSOR_X        = 23
CTRL_CURSOR_Y        = 24
CTRL_INV_LEFT_CLICK  = 25
CTRL_INV_RIGHT_CLICK = 26
CTRL_INV_SHIFT_HELD  = 27

# Head groupings (Contract 2)
ACTION_INDICES = list(range(0, 9)) + [11, 12, 13]  # 12 binary actions
LOOK_INDICES = [9, 10]                               # 2 analog look
HOTBAR_INDICES = list(range(14, 23))                  # 9 one-hot
CURSOR_INDICES = [23, 24]                             # 2 analog cursor
INV_CLICK_INDICES = [25, 26, 27]                      # 3 binary inv

# Game state key index (Contract 3)
GS_SCREEN_OPEN_IDX = 38

# V1 compatibility
V1_NUM_CONTROLS = 10
V1_GAME_STATE_DIM = 24

# Control names for display
CONTROL_NAMES_V2 = [
    "move_fwd", "move_bwd", "strafe_L", "strafe_R", "sprint", "sneak",
    "jump", "attack", "use_item", "look_yaw", "look_pitch",
    "drop", "swap_off", "open_inv",
    "hb_1", "hb_2", "hb_3", "hb_4", "hb_5", "hb_6", "hb_7", "hb_8", "hb_9",
    "cursor_x", "cursor_y", "inv_Lclk", "inv_Rclk", "inv_shift",
]


# ---------------------------------------------------------------------------
# V1 → V2 mapping functions
# ---------------------------------------------------------------------------

def _map_v1_controls_to_v2(v1_controls: np.ndarray) -> np.ndarray:
    """Map v1 10-dim controls to v2 28-dim controls.

    V1 layout: [forward, strafe_left, strafe_right, sprint, sneak, jump,
                attack, use_item, look_yaw, look_pitch]
    """
    T = v1_controls.shape[0]
    v2 = np.zeros((T, NUM_CONTROLS), dtype=np.float32)
    v2[:, CTRL_MOVE_FORWARD]  = v1_controls[:, 0]
    # v1 has no backward — stays 0
    v2[:, CTRL_STRAFE_LEFT]   = v1_controls[:, 1]
    v2[:, CTRL_STRAFE_RIGHT]  = v1_controls[:, 2]
    v2[:, CTRL_SPRINT]        = v1_controls[:, 3]
    v2[:, CTRL_SNEAK]         = v1_controls[:, 4]
    v2[:, CTRL_JUMP]          = v1_controls[:, 5]
    v2[:, CTRL_ATTACK]        = v1_controls[:, 6]
    v2[:, CTRL_USE_ITEM]      = v1_controls[:, 7]
    v2[:, CTRL_LOOK_YAW]      = v1_controls[:, 8]
    v2[:, CTRL_LOOK_PITCH]    = v1_controls[:, 9]
    # Indices 1, 11-27 = 0 (no backward, drop, swap, inventory, hotbar, cursor data)
    return v2


def _remap_v1_game_state_to_v2(v1_gs: np.ndarray) -> np.ndarray:
    """Remap v1 game state (16 or 24 dims) to v2 46-dim layout.

    V1 layout (from control_policy.py encode_game_state):
        0-6:   item classification (can_melee, can_shoot, can_place, can_eat,
               can_block, is_tool, is_empty)
        7-8:   health, hunger
        9-15:  on_ground, in_water, is_sprinting, is_sneaking, is_blocking,
               is_using_item, fall_distance(/10)
        16-23: attack_cooldown, crosshair_entity, crosshair_block, velocity_y,
               offhand_shield, swimming, item_use_progress, on_fire
    """
    T, D = v1_gs.shape
    v2 = np.zeros((T, GAME_STATE_DIM), dtype=np.float32)

    if D == 0:
        return v2

    # Item context: v1[0:7] → v2[0:11]
    if D > 0: v2[:, 0] = v1_gs[:, 0]   # can_melee
    if D > 1: v2[:, 1] = v1_gs[:, 1]   # can_ranged
    if D > 2: v2[:, 2] = v1_gs[:, 2]   # can_place
    if D > 3: v2[:, 3] = v1_gs[:, 3]   # can_eat
    if D > 5: v2[:, 4] = v1_gs[:, 5]   # is_tool (v1 index 5)
    if D > 6: v2[:, 5] = v1_gs[:, 6]   # is_empty (v1 index 6)
    # v2[6]=is_other, v2[7:11]=offhand → zeros for v1

    # Vitals: v1[7:9] → v2[11:13]
    if D > 7: v2[:, 11] = v1_gs[:, 7]   # health
    if D > 8: v2[:, 12] = v1_gs[:, 8]   # hunger
    # v2[13]=armor → zero for v1

    # Movement: v1[9:16] → v2[14:24]
    if D > 9:  v2[:, 14] = v1_gs[:, 9]   # on_ground
    if D > 10: v2[:, 15] = v1_gs[:, 10]  # in_water
    if D > 11: v2[:, 20] = v1_gs[:, 11]  # sprinting
    if D > 12: v2[:, 21] = v1_gs[:, 12]  # sneaking
    if D > 15: v2[:, 22] = v1_gs[:, 15] * 0.5  # fall_distance: v1=/10, v2=/20

    # Combat: from various v1 indices → v2[24:29]
    if D > 13: v2[:, 26] = v1_gs[:, 13]  # is_blocking
    if D > 14: v2[:, 25] = v1_gs[:, 14]  # is_using_item

    # Extended v1 dims (16-23, only if v1 has 24 dims)
    if D > 16: v2[:, 24] = v1_gs[:, 16]  # attack_cooldown
    if D > 17: v2[:, 29] = v1_gs[:, 17]  # crosshair entity → target_is_entity
    if D > 18: v2[:, 30] = v1_gs[:, 18]  # crosshair block → target_is_block
    if D > 19: v2[:, 23] = v1_gs[:, 19]  # velocity_y
    if D > 20: v2[:, 7]  = v1_gs[:, 20]  # offhand_shield
    if D > 21: v2[:, 16] = v1_gs[:, 21]  # swimming
    if D > 22: v2[:, 27] = v1_gs[:, 22]  # item_use_progress
    if D > 23: v2[:, 19] = v1_gs[:, 23]  # on_fire

    return v2


def _default_game_state(T: int) -> np.ndarray:
    """Create default v2 game state for recordings without game state data."""
    gs = np.zeros(GAME_STATE_DIM, dtype=np.float32)
    gs[5] = 1.0    # is_empty
    gs[11] = 1.0   # health
    gs[12] = 1.0   # hunger
    gs[14] = 1.0   # on_ground
    return np.tile(gs, (T, 1))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_control_features(data: dict) -> np.ndarray:
    """Extract 260-dim raw feature vectors from a recording dict.

    Handles both v2 (pre-packed 'frames' key) and v1 (individual arrays).

    Returns: (T, 260) float32 array.
    """
    if "frames" in data and data["frames"].ndim == 2 and data["frames"].shape[1] == RAW_FEATURE_DIM:
        return data["frames"].astype(np.float32)
    return extract_features(data)


def build_action_history(controls: np.ndarray, K: int = ACTION_HISTORY_K) -> np.ndarray:
    """Build action history for each frame.

    For frame t, history = controls[t-K:t] flattened.
    For frames 0..K-1, missing frames are zero-padded.

    Args:
        controls: (T, C) array of control vectors (C=28 for v2).
        K: Number of history frames.

    Returns: (T, K*C) float32 array.
    """
    T = controls.shape[0]
    C = controls.shape[1]
    history = np.zeros((T, K * C), dtype=np.float32)
    for t in range(T):
        for k in range(K):
            src = t - K + k
            if src >= 0:
                history[t, k * C:(k + 1) * C] = controls[src]
    return history


# ---------------------------------------------------------------------------
# Session loading with v1/v2 auto-detection
# ---------------------------------------------------------------------------

def _load_session_data(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load and normalize a session to v2 format, auto-detecting version.

    Returns:
        raw: (T, 260) raw features
        controls: (T, 28) control vectors
        game_state: (T, 46) game state vectors
        version: detected version (1 or 2)
    """
    version = int(data.get("version", 1))

    # Raw features
    raw = extract_control_features(data)
    T = raw.shape[0]

    # Controls
    controls = data["controls"].astype(np.float32)[:T]
    if controls.shape[1] == V1_NUM_CONTROLS:
        controls = _map_v1_controls_to_v2(controls)
        version = max(version, 1)  # ensure marked as v1
    elif controls.shape[1] < NUM_CONTROLS:
        padding = np.zeros((controls.shape[0], NUM_CONTROLS - controls.shape[1]), dtype=np.float32)
        controls = np.concatenate([controls, padding], axis=1)
    elif controls.shape[1] > NUM_CONTROLS:
        controls = controls[:, :NUM_CONTROLS]

    # Game state
    if "game_state" in data and data["game_state"].shape[0] >= T:
        gs = data["game_state"][:T].astype(np.float32)
        if gs.shape[1] == GAME_STATE_DIM:
            game_state = gs
        elif gs.shape[1] < GAME_STATE_DIM:
            game_state = _remap_v1_game_state_to_v2(gs)
        else:
            game_state = gs[:, :GAME_STATE_DIM]
    else:
        game_state = _default_game_state(T)

    return raw, controls, game_state, version


# ---------------------------------------------------------------------------
# Mode mask generation
# ---------------------------------------------------------------------------

def compute_mode_masks(controls: np.ndarray, game_state: np.ndarray,
                       target_idx: int) -> dict:
    """Compute per-window mode masks for the target frame.

    Args:
        controls: (T, 28) control vectors for the session.
        game_state: (T, 46) game state vectors for the session.
        target_idx: Index of the target frame (last frame in window).

    Returns:
        Dict with bool values:
            gameplay: True if no screen open at target frame
            screen_open: True if a screen/GUI is open at target frame
            hotbar_changed: True if hotbar slot has a one-hot activation at target
            look_active: True if look magnitude > 0.08 at target frame
    """
    screen_open = bool(game_state[target_idx, GS_SCREEN_OPEN_IDX] > 0.0)

    hotbar_changed = bool(np.any(
        controls[target_idx, CTRL_HOTBAR_START:CTRL_HOTBAR_END + 1] > 0.5
    ))

    look_yaw = controls[target_idx, CTRL_LOOK_YAW]
    look_pitch = controls[target_idx, CTRL_LOOK_PITCH]
    look_active = bool(np.sqrt(look_yaw**2 + look_pitch**2) > 0.08)

    return {
        "gameplay": not screen_open,
        "screen_open": screen_open,
        "hotbar_changed": hotbar_changed,
        "look_active": look_active,
    }


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def _augment_raw_features(raw: np.ndarray, controls: np.ndarray,
                          game_state: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply augmentations to raw features + controls + game_state together.

    Augmentations that change sequence length (time warp, crop) must be
    applied identically to all arrays.

    Args:
        raw: (T, 260) raw features (before velocity).
        controls: (T, 28) control vectors.
        game_state: (T, 46) game state vectors.

    Returns:
        (augmented_raw, augmented_controls, augmented_game_state)
    """
    T = raw.shape[0]

    # 1. Random temporal crop
    trim_frac = 0.10
    trim_start = random.randint(0, max(0, int(T * trim_frac)))
    trim_end = random.randint(0, max(0, int(T * trim_frac)))
    end = max(trim_start + 1, T - trim_end)
    raw = raw[trim_start:end]
    controls = controls[trim_start:end]
    game_state = game_state[trim_start:end]
    T = raw.shape[0]

    # 2. Time warp (resample all identically)
    rate = random.uniform(0.8, 1.2)
    new_T = max(1, int(round(T * rate)))
    if new_T != T and T > 1:
        old_indices = np.linspace(0, T - 1, new_T)
        old_grid = np.arange(T)
        idx = np.searchsorted(old_grid, old_indices, side="right") - 1
        idx = np.clip(idx, 0, T - 2)
        frac = np.clip(old_indices - old_grid[idx], 0.0, 1.0)

        # Interpolate raw features
        raw = (raw[idx] * (1.0 - frac[:, None]) + raw[idx + 1] * frac[:, None]).astype(np.float32)

        # Nearest-neighbor for controls and game state (binary values shouldn't interpolate)
        nn_indices = np.clip(np.round(old_indices).astype(int), 0, T - 1)
        controls = controls[nn_indices]
        game_state = game_state[nn_indices]

    # 3. Gaussian noise on landmark coordinates only
    raw = _add_noise(raw, std=0.005)

    return raw, controls, game_state


def _apply_action_history_dropout(features: np.ndarray, dropout_rate: float,
                                  history_start: int) -> np.ndarray:
    """Randomly zero out action history entries to prevent history copying.

    Args:
        features: (T, 671) full feature matrix (mutated in place).
        dropout_rate: Probability of zeroing each history frame.
        history_start: Start index of action history in feature dim.
    """
    T = features.shape[0]
    for t in range(T):
        for k in range(ACTION_HISTORY_K):
            if random.random() < dropout_rate:
                start = history_start + k * NUM_CONTROLS
                end = start + NUM_CONTROLS
                features[t, start:end] = 0.0
    return features


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def window_session(features: np.ndarray, controls: np.ndarray,
                   game_state: np.ndarray,
                   window_size: int = WINDOW_SIZE,
                   stride: int = WINDOW_STRIDE
                   ) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """Slice a session into overlapping windows with mode masks.

    Each window: (features_window, target_control, mode_masks) where
    target_control is the control vector at the LAST frame of the window.

    Returns list of (window_features, target, mode_masks) tuples.
    """
    T = features.shape[0]
    windows = []

    if T <= window_size:
        target = controls[-1]
        masks = compute_mode_masks(controls, game_state, T - 1)
        windows.append((features, target, masks))
    else:
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window_feat = features[start:end]
            target_idx = end - 1
            target = controls[target_idx]
            masks = compute_mode_masks(controls, game_state, target_idx)
            windows.append((window_feat, target, masks))

        # Ensure last frames are included
        if (T - window_size) % stride != 0:
            window_feat = features[T - window_size:]
            target = controls[-1]
            masks = compute_mode_masks(controls, game_state, T - 1)
            windows.append((window_feat, target, masks))

    return windows


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ControlDataset(Dataset):
    """PyTorch Dataset for V2 control-policy training.

    Args:
        windows: List of (features_window, target_control, mode_masks) tuples.
        max_seq_len: Windows are padded to this length.
        augment: Whether to apply data augmentation (history dropout only,
                 since time warp/noise are applied before windowing).
    """

    def __init__(self, windows: List[Tuple[np.ndarray, np.ndarray, dict]],
                 max_seq_len: int = WINDOW_SIZE,
                 augment: bool = False):
        self.windows = windows
        self.max_seq_len = max_seq_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        features, target, mode_masks = self.windows[idx]
        features = features.copy()
        T = features.shape[0]

        # Truncate if needed
        if T > self.max_seq_len:
            features = features[:self.max_seq_len]
            T = self.max_seq_len

        # Action history dropout during training
        if self.augment:
            history_start = RAW_FEATURE_DIM + VELOCITY_DIM + GAME_STATE_DIM  # 531
            features = _apply_action_history_dropout(
                features, ACTION_HISTORY_DROPOUT, history_start
            )

        # Pad to max_seq_len
        padded = np.zeros((self.max_seq_len, INPUT_DIM), dtype=np.float32)
        padded[:T] = features

        # Attention mask: 1 = real frame, 0 = padding
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:T] = 1.0

        return (
            torch.from_numpy(padded),          # (max_seq_len, 671)
            torch.from_numpy(mask),             # (max_seq_len,)
            torch.from_numpy(target.copy()),    # (28,)
            mode_masks,                         # dict of bools
        )


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def load_sessions(recordings_dir: str = DEFAULT_RECORDINGS_DIR
                  ) -> List[dict]:
    """Load all session .npz files from the recordings directory.

    Returns list of numpy data dicts (lazy-loaded NpzFile objects).
    """
    if not os.path.isdir(recordings_dir):
        print(f"WARNING: Recordings directory not found: {recordings_dir}")
        return []

    sessions = []
    for fname in sorted(os.listdir(recordings_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(recordings_dir, fname)
        data = np.load(path, allow_pickle=True)
        sessions.append(data)

    return sessions


def sessions_to_windows(sessions: List, augment: bool = False,
                        window_size: int = WINDOW_SIZE,
                        stride: int = WINDOW_STRIDE
                        ) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """Convert loaded sessions into windowed training samples.

    If augment is True, applies time warp, noise, and temporal crop
    to raw features before velocity computation and windowing.
    """
    all_windows = []

    for data in sessions:
        raw, controls, game_state, version = _load_session_data(data)
        T = raw.shape[0]

        if T < 2:
            continue

        # Augmentation on raw features (before velocity)
        if augment:
            raw, controls, game_state = _augment_raw_features(raw, controls, game_state)
            T = raw.shape[0]
            if T < 2:
                continue

        # Velocity
        with_vel = append_velocity(raw)  # (T, 485)

        # Action history
        action_hist = build_action_history(controls)  # (T, 140)

        # Full features: (T, 485 + 46 + 140) = (T, 671)
        features = np.concatenate([with_vel, game_state[:T], action_hist], axis=1)

        # Window
        windows = window_session(features.astype(np.float32), controls[:T],
                                 game_state[:T], window_size, stride)
        all_windows.extend(windows)

    return all_windows


def split_sessions_by_type(sessions: List, val_fraction: float = 0.2,
                           seed: int = 42
                           ) -> Tuple[List, List]:
    """Split sessions into train/val BY SESSION, stratified by session_type.

    Entire sessions go into train or val — no temporal leakage.
    """
    rng = random.Random(seed)

    # Group by session_type
    by_type: Dict[str, List] = {}
    for session in sessions:
        stype = str(session.get("session_type", "unknown"))
        by_type.setdefault(stype, []).append(session)

    train_sessions = []
    val_sessions = []

    for stype in sorted(by_type.keys()):
        group = by_type[stype]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction)))
        val_sessions.extend(group[:n_val])
        train_sessions.extend(group[n_val:])

    return train_sessions, val_sessions


def prepare_datasets(recordings_dir: str = DEFAULT_RECORDINGS_DIR,
                     val_fraction: float = 0.2,
                     window_size: int = WINDOW_SIZE,
                     stride: int = WINDOW_STRIDE,
                     seed: int = 42
                     ) -> Tuple["ControlDataset", "ControlDataset"]:
    """Load recordings, split, window, and return train/val ControlDatasets.

    Returns:
        (train_dataset, val_dataset)
    """
    sessions = load_sessions(recordings_dir)
    if not sessions:
        print("No sessions found. Run control_recorder.py first.")
        return ControlDataset([]), ControlDataset([])

    train_sessions, val_sessions = split_sessions_by_type(
        sessions, val_fraction, seed
    )

    print(f"Sessions: {len(sessions)} total, {len(train_sessions)} train, "
          f"{len(val_sessions)} val")

    train_windows = sessions_to_windows(train_sessions, augment=True,
                                        window_size=window_size, stride=stride)
    val_windows = sessions_to_windows(val_sessions, augment=False,
                                      window_size=window_size, stride=stride)

    random.Random(seed).shuffle(train_windows)

    print(f"Windows: {len(train_windows)} train, {len(val_windows)} val")

    train_ds = ControlDataset(train_windows, max_seq_len=window_size, augment=True)
    val_ds = ControlDataset(val_windows, max_seq_len=window_size, augment=False)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_dataset_stats(recordings_dir: str = DEFAULT_RECORDINGS_DIR):
    """Print statistics about the control dataset."""
    sessions = load_sessions(recordings_dir)
    if not sessions:
        print("No sessions found.")
        return

    total_frames = 0
    v1_count = 0
    v2_count = 0
    ctrl_active = np.zeros(NUM_CONTROLS, dtype=np.int64)
    session_types: Dict[str, int] = {}
    durations = []

    for data in sessions:
        _, controls, game_state, version = _load_session_data(data)
        T = controls.shape[0]
        total_frames += T
        if version == 1:
            v1_count += 1
        else:
            v2_count += 1

        # Binary action activations
        for i in ACTION_INDICES:
            ctrl_active[i] += int(np.sum(controls[:, i] > 0.5))

        # Look activity
        look_mag = np.sqrt(controls[:, CTRL_LOOK_YAW]**2 + controls[:, CTRL_LOOK_PITCH]**2)
        look_frames = int(np.sum(look_mag > 0.08))
        ctrl_active[CTRL_LOOK_YAW] += look_frames
        ctrl_active[CTRL_LOOK_PITCH] += look_frames

        # Hotbar changes
        for i in HOTBAR_INDICES:
            ctrl_active[i] += int(np.sum(controls[:, i] > 0.5))

        # Inventory: screen open frames and clicks
        screen_open = game_state[:, GS_SCREEN_OPEN_IDX] > 0
        ctrl_active[CTRL_CURSOR_X] += int(np.sum(screen_open))
        ctrl_active[CTRL_CURSOR_Y] += int(np.sum(screen_open))
        for i in INV_CLICK_INDICES:
            ctrl_active[i] += int(np.sum(controls[:, i] > 0.5))

        # Session type
        stype = str(data.get("session_type", "unknown"))
        session_types[stype] = session_types.get(stype, 0) + 1

        # Duration
        ts = data["timestamps"]
        if len(ts) > 1:
            durations.append(float(ts[-1] - ts[0]))

    avg_dur = np.mean(durations) if durations else 0.0
    print(f"\nDataset: {total_frames} frames ({len(sessions)} sessions, "
          f"v1={v1_count}, v2={v2_count}, avg {avg_dur:.1f}s)")

    print(f"\n  Action controls (12):")
    for i in ACTION_INDICES:
        pct = 100.0 * ctrl_active[i] / total_frames if total_frames else 0
        print(f"    {CONTROL_NAMES_V2[i]:>12s}: {ctrl_active[i]:>6d} ({pct:5.1f}%)")

    print(f"\n  Look:")
    pct = 100.0 * ctrl_active[CTRL_LOOK_YAW] / total_frames if total_frames else 0
    print(f"    {'look_active':>12s}: {ctrl_active[CTRL_LOOK_YAW]:>6d} ({pct:5.1f}%)")

    print(f"\n  Hotbar:")
    total_hotbar = sum(ctrl_active[i] for i in HOTBAR_INDICES)
    pct = 100.0 * total_hotbar / total_frames if total_frames else 0
    print(f"    {'any_change':>12s}: {total_hotbar:>6d} ({pct:5.1f}%)")

    print(f"\n  Inventory:")
    pct = 100.0 * ctrl_active[CTRL_CURSOR_X] / total_frames if total_frames else 0
    print(f"    {'screen_open':>12s}: {ctrl_active[CTRL_CURSOR_X]:>6d} ({pct:5.1f}%)")
    for i in INV_CLICK_INDICES:
        pct = 100.0 * ctrl_active[i] / total_frames if total_frames else 0
        print(f"    {CONTROL_NAMES_V2[i]:>12s}: {ctrl_active[i]:>6d} ({pct:5.1f}%)")

    print(f"\n  Session types:")
    for stype in sorted(session_types.keys()):
        print(f"    {stype}: {session_types[stype]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Control dataset V2 utilities")
    parser.add_argument("--dir", default=DEFAULT_RECORDINGS_DIR,
                        help="Recordings directory")
    parser.add_argument("--stats", action="store_true",
                        help="Print dataset statistics")
    parser.add_argument("--test", action="store_true",
                        help="Test dataset loading and print shapes")
    args = parser.parse_args()

    if args.stats:
        print_dataset_stats(args.dir)
    elif args.test:
        train_ds, val_ds = prepare_datasets(args.dir)
        print(f"\nTrain: {len(train_ds)} samples")
        print(f"Val:   {len(val_ds)} samples")
        if len(train_ds) > 0:
            feat, mask, target, mode_masks = train_ds[0]
            print(f"\nSample shapes:")
            print(f"  features:    {feat.shape}  (expected: ({WINDOW_SIZE}, {INPUT_DIM}))")
            print(f"  mask:        {mask.shape}  (expected: ({WINDOW_SIZE},))")
            print(f"  target:      {target.shape}  (expected: ({NUM_CONTROLS},))")
            print(f"  mode_masks:  {mode_masks}")
    else:
        print_dataset_stats(args.dir)
