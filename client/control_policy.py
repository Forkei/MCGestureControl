"""Live inference wrapper for the ControlTransformer policy model.

Maintains a rolling buffer of pose+hand features, encodes game state,
feeds action history from its own previous outputs, runs the model,
and applies post-processing (thresholds with hysteresis, EMA smoothing,
deadzone, rate limiting) to produce a clean ControlOutput each frame.

Usage:
    policy = ControlPolicy("models/control_policy.pt")
    # In main loop:
    policy.push_frame(pose, hands, game_state_dict)
    output = policy.predict()
    # output.forward, output.sprint, output.look_yaw, etc.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import numpy as np
import torch

from control_model import ControlTransformer, ControlTransformerV2
from gesture_dataset import append_velocity, _POSITION_DIMS


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ControlOutput:
    """Post-processed control output from the policy model."""

    # Binary controls (after threshold + hysteresis)
    forward: bool = False
    strafe_left: bool = False
    strafe_right: bool = False
    sprint: bool = False
    sneak: bool = False
    jump: bool = False
    attack: bool = False
    use_item: bool = False

    # Analog controls (after smoothing + deadzone)
    look_yaw: float = 0.0    # [-1, 1]
    look_pitch: float = 0.0  # [-1, 1]

    # Raw model outputs (for debugging / HUD display)
    raw_action_probs: np.ndarray = field(default_factory=lambda: np.zeros(8))
    raw_look: np.ndarray = field(default_factory=lambda: np.zeros(2))


# Binary control names in output order (indices 0-7)
BINARY_CONTROL_NAMES = [
    "forward", "strafe_left", "strafe_right", "sprint",
    "sneak", "jump", "attack", "use_item",
]

# Game state vector layout (24 dims)
GAME_STATE_DIM = 24
ACTION_HISTORY_K = 5
NUM_CONTROLS = 10  # 8 binary + 2 analog

# Default post-processing parameters
DEFAULT_HYSTERESIS_MARGIN = 0.05
DEFAULT_LOOK_DEADZONE = 0.08
DEFAULT_LOOK_SMOOTHING_ALPHA = 0.35
DEFAULT_LOOK_RATE_LIMIT = 0.3


# ---------------------------------------------------------------------------
# Game state encoding
# ---------------------------------------------------------------------------

# Item category sets for classification (from mod's ItemCategorizer enum)
_CATEGORY_MELEE = {"SWORD", "AXE", "TRIDENT"}
_CATEGORY_RANGED = {"BOW", "CROSSBOW"}
_CATEGORY_TOOL = {"PICKAXE", "SHOVEL", "HOE"}

# Keyword fallback (used when held_item_category is not available)
_MELEE_KEYWORDS = {"sword", "axe", "mace", "trident"}
_BOW_KEYWORDS = {"bow", "crossbow"}
_TOOL_KEYWORDS = {"pickaxe", "shovel", "hoe", "shears", "flint_and_steel"}
_FOOD_KEYWORDS = {
    "apple", "bread", "beef", "porkchop", "chicken", "mutton", "cod", "salmon",
    "cookie", "melon_slice", "stew", "carrot", "potato", "beetroot", "berries",
    "golden_apple", "enchanted_golden_apple", "dried_kelp", "rabbit",
    "cooked_beef", "cooked_porkchop", "cooked_chicken", "cooked_mutton",
    "cooked_cod", "cooked_salmon", "cooked_rabbit", "baked_potato",
    "pumpkin_pie", "cake", "honey_bottle", "chorus_fruit", "suspicious_stew",
}


def _classify_item_by_keyword(held: str, gs: np.ndarray):
    """Fallback: classify held item by keyword matching on item name."""
    if not held or held in ("none", "air", "empty"):
        gs[6] = 1.0   # is_empty
    else:
        if any(k in held for k in _MELEE_KEYWORDS):
            gs[0] = 1.0
        if any(k in held for k in _BOW_KEYWORDS):
            gs[1] = 1.0
        if any(k in held for k in _FOOD_KEYWORDS):
            gs[3] = 1.0
        if any(k in held for k in _TOOL_KEYWORDS):
            gs[5] = 1.0
        if "shield" in held:
            gs[4] = 1.0
        if gs[:6].sum() == 0:
            gs[2] = 1.0   # can_place


def _classify_item_by_category(category: str, gs: np.ndarray):
    """Classify held item using the mod's ItemCategorizer category."""
    if category == "EMPTY":
        gs[6] = 1.0   # is_empty
    elif category in _CATEGORY_MELEE:
        gs[0] = 1.0   # can_melee
    elif category in _CATEGORY_RANGED:
        gs[1] = 1.0   # can_shoot
    elif category == "BLOCK":
        gs[2] = 1.0   # can_place
    elif category == "FOOD":
        gs[3] = 1.0   # can_eat
    elif category == "SHIELD":
        gs[4] = 1.0   # can_block
    elif category in _CATEGORY_TOOL:
        gs[5] = 1.0   # is_tool
    else:
        # FISHING_ROD, THROWABLE, OTHER -> can_place as fallback
        gs[2] = 1.0


def encode_game_state(raw_state: dict) -> np.ndarray:
    """Convert MCCTP game state dict to a 24-dim feature vector.

    Indices 0-15: same semantics as the original 16-dim vector.
    Indices 16-23: attack_cooldown, crosshair_entity, crosshair_block,
                   velocity_y, offhand_shield, is_swimming, item_use_progress,
                   on_fire.

    Args:
        raw_state: Dict from MCCTP client.state dict. Expected keys include
                   held_item_category, health, hunger, on_ground, etc.

    Returns:
        Float32 array of shape (24,).
    """
    gs = np.zeros(GAME_STATE_DIM, dtype=np.float32)

    if not raw_state:
        gs[6] = 1.0    # is_empty
        gs[7] = 1.0    # health
        gs[8] = 1.0    # hunger
        gs[9] = 1.0    # on_ground
        gs[16] = 1.0   # attack_cooldown (fully ready)
        return gs

    # --- Held item classification (indices 0-6) ---
    category = str(raw_state.get("held_item_category", "")).upper()
    if category and category != "EMPTY":
        _classify_item_by_category(category, gs)
    elif "held_item_category" in raw_state:
        # Category is explicitly EMPTY
        gs[6] = 1.0
    else:
        # No category field — fall back to keyword matching
        held = str(raw_state.get("held_item", "")).lower()
        _classify_item_by_keyword(held, gs)

    # --- Vitals (indices 7-8) ---
    gs[7] = min(1.0, max(0.0, raw_state.get("health", 20.0) / 20.0))
    gs[8] = min(1.0, max(0.0, raw_state.get("hunger", 20.0) / 20.0))

    # --- Player state flags (indices 9-15) ---
    gs[9] = 1.0 if raw_state.get("on_ground", True) else 0.0
    gs[10] = 1.0 if raw_state.get("in_water", False) else 0.0
    gs[11] = 1.0 if raw_state.get("is_sprinting", False) else 0.0
    gs[12] = 1.0 if raw_state.get("is_sneaking", False) else 0.0
    gs[13] = 1.0 if raw_state.get("is_blocking", False) else 0.0
    gs[14] = 1.0 if raw_state.get("is_using_item", False) else 0.0
    gs[15] = min(1.0, max(0.0, raw_state.get("fall_distance", 0.0) / 10.0))

    # --- New dimensions (indices 16-23) ---
    gs[16] = min(1.0, max(0.0, raw_state.get("attack_cooldown", 1.0)))
    gs[17] = 1.0 if raw_state.get("crosshair_target") == "ENTITY" else 0.0
    gs[18] = 1.0 if raw_state.get("crosshair_target") == "BLOCK" else 0.0
    velocity_y = float(raw_state.get("velocity_y", 0.0))
    gs[19] = max(-1.0, min(1.0, velocity_y))  # clamp to [-1, 1]
    offhand = str(raw_state.get("offhand_category", "")).upper()
    gs[20] = 1.0 if offhand == "SHIELD" else 0.0
    gs[21] = 1.0 if raw_state.get("swimming", False) else 0.0
    gs[22] = min(1.0, max(0.0, float(raw_state.get("item_use_progress", 0.0))))
    gs[23] = 1.0 if raw_state.get("on_fire", False) else 0.0

    return gs


# ---------------------------------------------------------------------------
# Control Policy
# ---------------------------------------------------------------------------

class ControlPolicy:
    """Wraps the trained ControlTransformer for live inference.

    Maintains a rolling buffer of raw features and a history of its own
    outputs. Runs the model and applies post-processing to produce clean
    control signals.

    Args:
        model_path: Path to the saved model weights (.pt file).
        config_path: Path to control_config.json. If None, auto-detected
                     from the same directory as model_path.
    """

    def __init__(self, model_path: str, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(model_path), "control_config.json"
            )

        with open(config_path, "r") as f:
            cfg = json.load(f)

        self._cfg = cfg

        # Architecture parameters
        self.input_dim = cfg["input_dim"]
        self.max_seq_len = cfg["max_seq_len"]
        self.num_binary = cfg["num_binary_controls"]
        self.num_analog = cfg["num_analog_controls"]
        self.action_history_k = cfg.get("action_history_length", ACTION_HISTORY_K)
        self.game_state_dim = cfg.get("game_state_dim", GAME_STATE_DIM)

        # Per-control thresholds from training
        thresh_dict = cfg.get("thresholds", {})
        self.thresholds = np.array([
            thresh_dict.get(name, 0.5) for name in BINARY_CONTROL_NAMES
        ], dtype=np.float32)

        # Post-processing parameters
        self.hysteresis_margin = cfg.get("hysteresis_margin", DEFAULT_HYSTERESIS_MARGIN)
        self.look_deadzone = cfg.get("look_deadzone", DEFAULT_LOOK_DEADZONE)
        self.look_alpha = cfg.get("look_smoothing_alpha", DEFAULT_LOOK_SMOOTHING_ALPHA)
        self.look_rate_limit = cfg.get("look_rate_limit", DEFAULT_LOOK_RATE_LIMIT)

        # Feature normalization (mean/std from training, if available)
        self._feat_mean = None
        self._feat_std = None
        if "feature_mean" in cfg:
            self._feat_mean = np.array(cfg["feature_mean"], dtype=np.float32)
        if "feature_std" in cfg:
            self._feat_std = np.array(cfg["feature_std"], dtype=np.float32)
            # Avoid division by zero
            self._feat_std[self._feat_std < 1e-6] = 1.0

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ControlTransformer(
            input_dim=self.input_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            max_seq_len=self.max_seq_len,
            num_binary=self.num_binary,
            num_analog=self.num_analog,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        # Rolling buffer of raw features (260 dims each)
        self._buffer = deque(maxlen=self.max_seq_len)

        # Game state for current frame
        self._game_state = np.zeros(GAME_STATE_DIM, dtype=np.float32)

        # Action history: ring buffer of last K outputs (10 dims each)
        self._action_history = deque(
            [np.zeros(NUM_CONTROLS, dtype=np.float32)] * self.action_history_k,
            maxlen=self.action_history_k,
        )

        # Post-processing state
        self._active = np.zeros(self.num_binary, dtype=bool)
        self._smooth_yaw = 0.0
        self._smooth_pitch = 0.0

    @property
    def control_names(self):
        return BINARY_CONTROL_NAMES + ["look_yaw", "look_pitch"]

    def push_frame(self, pose, hands, game_state: dict = None):
        """Extract features from current frame and add to buffer.

        Args:
            pose: PoseLandmarks or None.
            hands: List of HandLandmarks.
            game_state: Dict from MCCTP client.state, or None/empty.
        """
        # Extract 260 raw features (same as GestureClassifier.push_frame)
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
            pose_world,                                       # 99
            left_3d,                                          # 63
            right_3d,                                         # 63
            np.array([left_present], dtype=np.float32),       # 1
            np.array([right_present], dtype=np.float32),      # 1
            pose_vis,                                         # 33
        ])  # total: 260

        self._buffer.append(features)

        # Encode game state
        if game_state is not None:
            self._game_state = encode_game_state(game_state)

    def predict(self) -> ControlOutput:
        """Run model on current buffer, return post-processed controls.

        Returns:
            ControlOutput with binary actions + analog look values.
        """
        if len(self._buffer) < 3:
            return ControlOutput()

        with torch.no_grad():
            raw_frames = np.stack(list(self._buffer))  # (T, 260)
            T = raw_frames.shape[0]
            length = min(T, self.max_seq_len)
            raw_clip = raw_frames[-length:]  # (length, 260)

            # Compute velocity features
            with_vel = append_velocity(raw_clip)  # (length, 485)

            # Game state: tile for all frames in window
            game_state_tiled = np.tile(self._game_state, (length, 1))  # (length, 24)

            # Action history: tile for all frames in window
            # Use current history for all frames (approximation for live inference)
            action_hist = np.concatenate(
                list(self._action_history), axis=0
            )  # (K * 10,) = (50,)
            action_hist_tiled = np.tile(action_hist, (length, 1))  # (length, 50)

            # Concatenate to full feature vector
            full_features = np.concatenate(
                [with_vel, game_state_tiled, action_hist_tiled], axis=1
            )  # (length, 559)

            # Feature normalization
            if self._feat_mean is not None and self._feat_std is not None:
                full_features = (full_features - self._feat_mean) / self._feat_std

            # Pad to max_seq_len
            padded = np.zeros(
                (self.max_seq_len, self.input_dim), dtype=np.float32
            )
            padded[:length] = full_features

            mask = np.zeros(self.max_seq_len, dtype=np.float32)
            mask[:length] = 1.0

            x = torch.from_numpy(padded).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).unsqueeze(0).to(self.device)

            action_logits, look_pred = self.model(x, m)
            action_probs = torch.sigmoid(action_logits).squeeze(0).cpu().numpy()
            look_raw = look_pred.squeeze(0).cpu().numpy()

        # Post-process binary controls: threshold with hysteresis
        for i in range(self.num_binary):
            if not self._active[i]:
                # Activate at higher threshold
                if action_probs[i] > self.thresholds[i] + self.hysteresis_margin:
                    self._active[i] = True
            else:
                # Deactivate at lower threshold
                if action_probs[i] < self.thresholds[i] - self.hysteresis_margin:
                    self._active[i] = False

        # Post-process analog look: deadzone + EMA smoothing + rate limiting
        raw_yaw = float(look_raw[0])
        raw_pitch = float(look_raw[1])

        # Deadzone
        if abs(raw_yaw) < self.look_deadzone:
            raw_yaw = 0.0
        if abs(raw_pitch) < self.look_deadzone:
            raw_pitch = 0.0

        # EMA smoothing
        smooth_yaw = self.look_alpha * raw_yaw + (1 - self.look_alpha) * self._smooth_yaw
        smooth_pitch = self.look_alpha * raw_pitch + (1 - self.look_alpha) * self._smooth_pitch

        # Rate limiting
        delta_yaw = smooth_yaw - self._smooth_yaw
        delta_pitch = smooth_pitch - self._smooth_pitch
        if abs(delta_yaw) > self.look_rate_limit:
            smooth_yaw = self._smooth_yaw + self.look_rate_limit * np.sign(delta_yaw)
        if abs(delta_pitch) > self.look_rate_limit:
            smooth_pitch = self._smooth_pitch + self.look_rate_limit * np.sign(delta_pitch)

        self._smooth_yaw = smooth_yaw
        self._smooth_pitch = smooth_pitch

        # Build output
        output = ControlOutput(
            forward=bool(self._active[0]),
            strafe_left=bool(self._active[1]),
            strafe_right=bool(self._active[2]),
            sprint=bool(self._active[3]),
            sneak=bool(self._active[4]),
            jump=bool(self._active[5]),
            attack=bool(self._active[6]),
            use_item=bool(self._active[7]),
            look_yaw=smooth_yaw,
            look_pitch=smooth_pitch,
            raw_action_probs=action_probs.copy(),
            raw_look=look_raw.copy(),
        )

        # Update action history
        action_vec = np.zeros(NUM_CONTROLS, dtype=np.float32)
        action_vec[:self.num_binary] = self._active.astype(np.float32)
        action_vec[self.num_binary] = smooth_yaw
        action_vec[self.num_binary + 1] = smooth_pitch
        self._action_history.append(action_vec)

        return output

    def clear(self):
        """Reset all buffers and state."""
        self._buffer.clear()
        self._action_history = deque(
            [np.zeros(NUM_CONTROLS, dtype=np.float32)] * self.action_history_k,
            maxlen=self.action_history_k,
        )
        self._active[:] = False
        self._smooth_yaw = 0.0
        self._smooth_pitch = 0.0
        self._game_state = np.zeros(GAME_STATE_DIM, dtype=np.float32)


# ===========================================================================
# V2 Control Policy — 5 output heads, 28-dim control vector
# ===========================================================================

# V2 dimension constants (Contract 1)
V2_GAME_STATE_DIM = 46
V2_ACTION_HISTORY_K = 5
V2_NUM_CONTROLS = 28
V2_ACTION_HISTORY_DIM = V2_ACTION_HISTORY_K * V2_NUM_CONTROLS  # 140
V2_RAW_FEATURE_DIM = 260
V2_VELOCITY_DIM = 225
V2_INPUT_DIM = (V2_RAW_FEATURE_DIM + V2_VELOCITY_DIM
                + V2_GAME_STATE_DIM + V2_ACTION_HISTORY_DIM)  # 671
V2_WINDOW_SIZE = 30

# V2 control vector layout + head groupings (Contract 2) — imported from control_dataset
from control_dataset import (  # noqa: E402
    CTRL_MOVE_FORWARD, CTRL_MOVE_BACKWARD, CTRL_STRAFE_LEFT, CTRL_STRAFE_RIGHT,
    CTRL_SPRINT, CTRL_SNEAK, CTRL_JUMP, CTRL_ATTACK, CTRL_USE_ITEM,
    CTRL_LOOK_YAW, CTRL_LOOK_PITCH, CTRL_DROP_ITEM, CTRL_SWAP_OFFHAND,
    CTRL_OPEN_INVENTORY, CTRL_HOTBAR_START, CTRL_HOTBAR_END,
    CTRL_CURSOR_X, CTRL_CURSOR_Y, CTRL_INV_LEFT_CLICK,
    CTRL_INV_RIGHT_CLICK, CTRL_INV_SHIFT_HELD,
    ACTION_INDICES, LOOK_INDICES, HOTBAR_INDICES,
    CURSOR_INDICES, INV_CLICK_INDICES,
    GS_SCREEN_OPEN_IDX,
)

# V2 action head output names (maps action head index 0-11 to field names)
V2_ACTION_NAMES = [
    "move_forward", "move_backward", "strafe_left", "strafe_right",
    "sprint", "sneak", "jump", "attack", "use_item",
    "drop_item", "swap_offhand", "open_inventory",
]


# ---------------------------------------------------------------------------
# V2 Game State Encoding (Contract 8)
# ---------------------------------------------------------------------------

_V2_CATEGORY_MELEE = {"SWORD", "AXE", "TRIDENT"}
_V2_CATEGORY_RANGED = {"BOW", "CROSSBOW"}
_V2_CATEGORY_TOOL = {"PICKAXE", "SHOVEL", "HOE", "SHEARS"}


def _classify_mainhand_v2(raw_state: dict, gs: np.ndarray):
    """Classify mainhand item into v2 categories (indices 0-6)."""
    category = str(raw_state.get("held_item_category", "")).upper()
    if category and category not in ("EMPTY", "AIR", "NONE", ""):
        if category in _V2_CATEGORY_MELEE:
            gs[0] = 1.0   # can_melee
        elif category in _V2_CATEGORY_RANGED:
            gs[1] = 1.0   # can_ranged
        elif category == "BLOCK":
            gs[2] = 1.0   # can_place
        elif category == "FOOD":
            gs[3] = 1.0   # can_eat
        elif category in _V2_CATEGORY_TOOL:
            gs[4] = 1.0   # is_tool
        elif category == "EMPTY":
            gs[5] = 1.0   # is_empty
        else:
            gs[6] = 1.0   # is_other
        return

    # Keyword fallback
    held = str(raw_state.get("held_item", "")).lower()
    if not held or held in ("none", "air", "empty"):
        gs[5] = 1.0   # is_empty
    elif any(k in held for k in _MELEE_KEYWORDS):
        gs[0] = 1.0
    elif any(k in held for k in _BOW_KEYWORDS):
        gs[1] = 1.0
    elif any(k in held for k in _FOOD_KEYWORDS):
        gs[3] = 1.0
    elif any(k in held for k in _TOOL_KEYWORDS):
        gs[4] = 1.0
    else:
        gs[6] = 1.0   # is_other


def _classify_offhand_v2(raw_state: dict, gs: np.ndarray):
    """Classify offhand item (indices 7-10)."""
    offhand = str(raw_state.get("offhand_category", "")).upper()
    if not offhand or offhand in ("EMPTY", "AIR", "NONE", ""):
        gs[10] = 1.0   # offhand_is_empty
    elif offhand == "SHIELD":
        gs[7] = 1.0    # offhand_is_shield
    elif offhand == "FOOD":
        gs[8] = 1.0    # offhand_is_food
    elif offhand in ("TOTEM", "TOTEM_OF_UNDYING"):
        gs[9] = 1.0    # offhand_is_totem
    else:
        gs[10] = 1.0   # offhand_is_empty (unknown treated as empty)


def encode_game_state_v2(mcctp_state_dict: dict) -> np.ndarray:
    """Convert MCCTP MCCTP client.state dict → (46,) float32.

    This is THE canonical v2 game state encoder. Other files import from here.

    Args:
        mcctp_state_dict: Dict from MCCTP client.state dict.

    Returns:
        Float32 array of shape (46,) per Contract 3.
    """
    gs = np.zeros(V2_GAME_STATE_DIM, dtype=np.float32)

    if not mcctp_state_dict:
        gs[5] = 1.0    # is_empty
        gs[10] = 1.0   # offhand_is_empty
        gs[11] = 1.0   # health
        gs[12] = 1.0   # hunger
        gs[14] = 1.0   # on_ground
        gs[24] = 1.0   # attack_cooldown (fully ready)
        return gs

    d = mcctp_state_dict

    # --- Item Context (indices 0-10) ---
    _classify_mainhand_v2(d, gs)
    _classify_offhand_v2(d, gs)

    # --- Vitals (indices 11-13) ---
    gs[11] = min(1.0, max(0.0, d.get("health", 20.0) / 20.0))
    gs[12] = min(1.0, max(0.0, d.get("hunger", 20.0) / 20.0))
    gs[13] = min(1.0, max(0.0, d.get("armor", 0.0) / 20.0))

    # --- Movement State (indices 14-23) ---
    gs[14] = 1.0 if d.get("on_ground", True) else 0.0
    gs[15] = 1.0 if d.get("in_water", False) else 0.0
    gs[16] = 1.0 if d.get("swimming", d.get("is_swimming", False)) else 0.0
    gs[17] = 1.0 if d.get("is_flying", d.get("flying", False)) else 0.0
    gs[18] = 1.0 if d.get("is_climbing", d.get("climbing", False)) else 0.0
    gs[19] = 1.0 if d.get("on_fire", False) else 0.0
    gs[20] = 1.0 if d.get("is_sprinting", False) else 0.0
    gs[21] = 1.0 if d.get("is_sneaking", False) else 0.0
    fall = float(d.get("fall_distance", 0.0))
    gs[22] = min(1.0, max(0.0, fall / 20.0))
    vel_y = float(d.get("velocity_y", 0.0))
    gs[23] = max(-1.0, min(1.0, vel_y / 3.0))

    # --- Combat State (indices 24-28) ---
    gs[24] = min(1.0, max(0.0, float(d.get("attack_cooldown", 1.0))))
    gs[25] = 1.0 if d.get("is_using_item", False) else 0.0
    gs[26] = 1.0 if d.get("is_blocking", False) else 0.0
    gs[27] = min(1.0, max(0.0, float(d.get("item_use_progress", 0.0))))
    gs[28] = 1.0 if d.get("recently_hurt", False) else 0.0

    # --- Crosshair Target (indices 29-32) ---
    crosshair = d.get("crosshair_target", "")
    gs[29] = 1.0 if crosshair == "ENTITY" else 0.0
    gs[30] = 1.0 if crosshair == "BLOCK" else 0.0
    gs[31] = 1.0 if d.get("target_entity_hostile", False) else 0.0
    target_dist = float(d.get("target_distance", 6.0))
    gs[32] = min(1.0, max(0.0, target_dist / 6.0))

    # --- Nearby Threats (indices 33-35) ---
    hostile_dist = float(d.get("nearest_hostile_dist", 32.0))
    gs[33] = min(1.0, max(0.0, hostile_dist / 32.0))
    hostile_yaw = float(d.get("nearest_hostile_yaw", 0.0))
    gs[34] = max(-1.0, min(1.0, hostile_yaw / 180.0))
    hostile_count = float(d.get("hostile_count", 0.0))
    gs[35] = min(1.0, max(0.0, hostile_count / 10.0))

    # --- Environment (indices 36-38) ---
    gs[36] = min(1.0, max(0.0, float(d.get("time_of_day", 0.0)) / 24000.0))
    gm = d.get("game_mode", "")
    gs[37] = 1.0 if (
        d.get("game_mode_survival", False)
        or (isinstance(gm, str) and gm.lower() == "survival")
    ) else 0.0
    # screen_open_type: 0=none, 0.33=inventory, 0.66=chest, 1.0=other
    screen = d.get("screen_open_type", d.get("screen_open", ""))
    if isinstance(screen, (int, float)):
        gs[38] = min(1.0, max(0.0, float(screen)))
    elif isinstance(screen, str):
        sl = screen.lower()
        if not sl or sl == "none":
            gs[38] = 0.0
        elif sl == "inventory":
            gs[38] = 0.33
        elif sl in ("chest", "container"):
            gs[38] = 0.66
        else:
            gs[38] = 1.0

    # --- Status Effects (indices 39-43) ---
    gs[39] = 1.0 if d.get("has_speed", False) else 0.0
    gs[40] = 1.0 if d.get("has_slowness", False) else 0.0
    gs[41] = 1.0 if d.get("has_strength", False) else 0.0
    gs[42] = 1.0 if (
        d.get("taking_dot", False)
        or d.get("has_poison", False)
        or d.get("has_wither", False)
    ) else 0.0
    gs[43] = 1.0 if d.get("has_fire_resist", False) else 0.0

    # --- Extra (indices 44-45) ---
    slot = d.get("selected_slot", d.get("current_hotbar_slot", 0))
    gs[44] = min(1.0, max(0.0, float(slot) / 8.0))
    gs[45] = 1.0 if d.get("horizontal_collision", False) else 0.0

    return gs


# ---------------------------------------------------------------------------
# V2 Output Dataclass (Contract 5)
# ---------------------------------------------------------------------------

@dataclass
class ControlOutputV2:
    """Post-processed control output from the V2 policy model."""

    # Gameplay actions (binary, thresholded)
    move_forward: bool = False
    move_backward: bool = False
    strafe_left: bool = False
    strafe_right: bool = False
    sprint: bool = False
    sneak: bool = False
    jump: bool = False
    attack: bool = False
    use_item: bool = False

    # Look (analog)
    look_yaw: float = 0.0      # [-1, 1]
    look_pitch: float = 0.0    # [-1, 1]

    # Utility actions (binary, thresholded)
    drop_item: bool = False
    swap_offhand: bool = False
    open_inventory: bool = False

    # Hotbar (one-hot → int or None)
    hotbar_slot: Optional[int] = None   # 0-8, None = no change

    # Cursor (only valid when screen_open)
    cursor_x: float = 0.0     # [0, 1]
    cursor_y: float = 0.0     # [0, 1]

    # Inventory clicks (only valid when screen_open)
    inv_left_click: bool = False
    inv_right_click: bool = False
    inv_shift_held: bool = False

    # Raw model outputs (for debugging/logging)
    raw_action_probs: np.ndarray = field(
        default_factory=lambda: np.zeros(12))
    raw_look: np.ndarray = field(
        default_factory=lambda: np.zeros(2))
    raw_hotbar_probs: np.ndarray = field(
        default_factory=lambda: np.zeros(9))
    raw_cursor: np.ndarray = field(
        default_factory=lambda: np.zeros(2))
    raw_inv_probs: np.ndarray = field(
        default_factory=lambda: np.zeros(3))

    # Mode
    screen_open: bool = False

    def to_control_vector(self) -> np.ndarray:
        """Convert back to 28-dim vector (for action history)."""
        v = np.zeros(V2_NUM_CONTROLS, dtype=np.float32)
        v[CTRL_MOVE_FORWARD] = float(self.move_forward)
        v[CTRL_MOVE_BACKWARD] = float(self.move_backward)
        v[CTRL_STRAFE_LEFT] = float(self.strafe_left)
        v[CTRL_STRAFE_RIGHT] = float(self.strafe_right)
        v[CTRL_SPRINT] = float(self.sprint)
        v[CTRL_SNEAK] = float(self.sneak)
        v[CTRL_JUMP] = float(self.jump)
        v[CTRL_ATTACK] = float(self.attack)
        v[CTRL_USE_ITEM] = float(self.use_item)
        v[CTRL_LOOK_YAW] = self.look_yaw
        v[CTRL_LOOK_PITCH] = self.look_pitch
        v[CTRL_DROP_ITEM] = float(self.drop_item)
        v[CTRL_SWAP_OFFHAND] = float(self.swap_offhand)
        v[CTRL_OPEN_INVENTORY] = float(self.open_inventory)
        if self.hotbar_slot is not None:
            v[CTRL_HOTBAR_START + self.hotbar_slot] = 1.0
        v[CTRL_CURSOR_X] = self.cursor_x
        v[CTRL_CURSOR_Y] = self.cursor_y
        v[CTRL_INV_LEFT_CLICK] = float(self.inv_left_click)
        v[CTRL_INV_RIGHT_CLICK] = float(self.inv_right_click)
        v[CTRL_INV_SHIFT_HELD] = float(self.inv_shift_held)
        return v


# ---------------------------------------------------------------------------
# V2 Inference Wrapper
# ---------------------------------------------------------------------------

class ControlPolicyV2:
    """V2 inference wrapper: 5-head model with mode-aware post-processing.

    Maintains rolling buffers for raw features, game state, and action
    history. Runs ControlTransformerV2 and applies per-head post-processing
    (thresholds with hysteresis, EMA smoothing, deadzone, rate limiting).

    Args:
        model_path: Path to control_policy_v2.pt.
        config_path: Path to control_config_v2.json. Auto-detected if None.
    """

    def __init__(self, model_path: str, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(model_path), "control_config_v2.json"
            )

        with open(config_path, "r") as f:
            cfg = json.load(f)

        self._cfg = cfg

        # Architecture parameters
        model_cfg = cfg.get("model", cfg)
        self.input_dim = model_cfg.get("input_dim", V2_INPUT_DIM)
        self.max_seq_len = model_cfg.get("max_seq_len", V2_WINDOW_SIZE)

        # Per-action thresholds (Contract 7)
        thresh = cfg.get("thresholds", {})
        self.action_thresholds = np.array(
            thresh.get("action", [0.5] * 12), dtype=np.float32
        )
        self.hotbar_min_confidence = thresh.get("hotbar_min_confidence", 0.3)
        self.inv_click_thresholds = np.array(
            thresh.get("inv_click", [0.5, 0.5, 0.5]), dtype=np.float32
        )

        # Post-processing parameters (Contract 7)
        pp = cfg.get("post_processing", {})
        self.hysteresis_margin = pp.get("hysteresis_margin", 0.05)
        self.look_deadzone = pp.get("look_deadzone", 0.08)
        self.look_alpha = pp.get("look_ema_alpha", 0.35)
        self.look_rate_limit = pp.get("look_rate_limit", 0.3)
        self.cursor_alpha = pp.get("cursor_ema_alpha", 0.5)

        # Feature normalization (from training)
        self._feat_mean = None
        self._feat_std = None
        if "feature_mean" in cfg:
            self._feat_mean = np.array(cfg["feature_mean"], dtype=np.float32)
        if "feature_std" in cfg:
            self._feat_std = np.array(cfg["feature_std"], dtype=np.float32)
            self._feat_std[self._feat_std < 1e-6] = 1.0

        # Load model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = ControlTransformerV2(
            input_dim=model_cfg.get("input_dim", V2_INPUT_DIM),
            d_model=model_cfg.get("d_model", 256),
            nhead=model_cfg.get("nhead", 8),
            num_layers=model_cfg.get("num_layers", 6),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
            dropout=model_cfg.get("dropout", 0.3),
            max_seq_len=model_cfg.get("max_seq_len", V2_WINDOW_SIZE),
            num_action=model_cfg.get("num_action", 12),
            num_look=model_cfg.get("num_look", 2),
            num_hotbar=model_cfg.get("num_hotbar", 9),
            num_cursor=model_cfg.get("num_cursor", 2),
            num_inv_click=model_cfg.get("num_inv_click", 3),
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        # Rolling buffers
        self._buffer = deque(maxlen=self.max_seq_len)           # (260,) raw
        self._gs_buffer = deque(maxlen=self.max_seq_len)        # (46,) game state
        self._action_history = deque(
            [np.zeros(V2_NUM_CONTROLS, dtype=np.float32)] * V2_ACTION_HISTORY_K,
            maxlen=V2_ACTION_HISTORY_K,
        )

        # Post-processing state
        self._action_active = np.zeros(12, dtype=bool)
        self._inv_active = np.zeros(3, dtype=bool)
        self._smooth_yaw = 0.0
        self._smooth_pitch = 0.0
        self._smooth_cursor_x = 0.5
        self._smooth_cursor_y = 0.5
        self._current_hotbar_slot = 0

    def push_frame(self, pose, hands, game_state_dict: dict = None):
        """Extract features from current frame and add to buffers.

        Args:
            pose: PoseLandmarks or None.
            hands: List of HandLandmarks.
            game_state_dict: Dict from MCCTP client.state dict.
        """
        # Extract 260 raw features (same layout as v1)
        if pose is not None and pose.world_landmarks:
            pose_world = np.array(
                pose.world_landmarks, dtype=np.float32
            ).reshape(-1)
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
            kp3d = (
                np.array(hand.landmarks_3d, dtype=np.float32).reshape(-1)
                if hand.landmarks_3d
                else np.zeros(63, dtype=np.float32)
            )
            if hand.handedness == "Left":
                left_3d = kp3d
                left_present = 1.0
            else:
                right_3d = kp3d
                right_present = 1.0

        features = np.concatenate([
            pose_world,                                       # 99
            left_3d,                                          # 63
            right_3d,                                         # 63
            np.array([left_present], dtype=np.float32),       # 1
            np.array([right_present], dtype=np.float32),      # 1
            pose_vis,                                         # 33
        ])  # total: 260

        self._buffer.append(features)

        # Encode game state to 46-dim vector
        self._gs_buffer.append(
            encode_game_state_v2(game_state_dict or {})
        )

    def predict(self) -> ControlOutputV2:
        """Run model on current buffer, return post-processed controls."""
        if len(self._buffer) < 3:
            return ControlOutputV2()

        with torch.no_grad():
            raw_frames = np.stack(list(self._buffer))       # (T, 260)
            game_states = np.stack(list(self._gs_buffer))   # (T, 46)
            T = raw_frames.shape[0]
            length = min(T, self.max_seq_len)
            raw_clip = raw_frames[-length:]
            gs_clip = game_states[-length:]

            # Velocity features
            with_vel = append_velocity(raw_clip)  # (length, 485)

            # Action history: flatten and tile for all frames
            action_hist = np.concatenate(
                list(self._action_history), axis=0
            )  # (140,)
            action_hist_tiled = np.tile(action_hist, (length, 1))  # (L, 140)

            # Concatenate to 671-dim feature vector
            full_features = np.concatenate(
                [with_vel, gs_clip, action_hist_tiled], axis=1
            )  # (length, 671)

            # Feature normalization
            if self._feat_mean is not None and self._feat_std is not None:
                full_features = (
                    (full_features - self._feat_mean) / self._feat_std
                )

            # Pad to max_seq_len
            padded = np.zeros(
                (self.max_seq_len, self.input_dim), dtype=np.float32
            )
            padded[:length] = full_features

            mask = np.zeros(self.max_seq_len, dtype=np.float32)
            mask[:length] = 1.0

            x = torch.from_numpy(padded).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).unsqueeze(0).to(self.device)

            outputs = self.model(x, m)  # dict of 5 head outputs

        # Extract raw outputs from each head
        action_logits = outputs['action_logits'].squeeze(0).cpu().numpy()
        look_raw = outputs['look'].squeeze(0).cpu().numpy()
        hotbar_logits = outputs['hotbar_logits'].squeeze(0).cpu().numpy()
        cursor_raw = outputs['cursor'].squeeze(0).cpu().numpy()
        inv_logits = outputs['inv_click_logits'].squeeze(0).cpu().numpy()

        # Sigmoid for binary heads
        action_probs = 1.0 / (1.0 + np.exp(-action_logits))    # (12,)
        inv_probs = 1.0 / (1.0 + np.exp(-inv_logits))          # (3,)

        # Softmax for hotbar
        hotbar_shifted = hotbar_logits - hotbar_logits.max()
        hotbar_probs = np.exp(hotbar_shifted) / np.exp(hotbar_shifted).sum()

        # Determine mode from last frame's game state
        last_gs = gs_clip[-1]
        screen_open = last_gs[GS_SCREEN_OPEN_IDX] > 0

        # Build output with raw values
        output = ControlOutputV2(
            raw_action_probs=action_probs.copy(),
            raw_look=look_raw.copy(),
            raw_hotbar_probs=hotbar_probs.copy(),
            raw_cursor=cursor_raw.copy(),
            raw_inv_probs=inv_probs.copy(),
            screen_open=screen_open,
        )

        if not screen_open:
            # --- Gameplay Mode ---

            # Action thresholds with hysteresis (all 12 actions)
            for i in range(12):
                thresh = self.action_thresholds[i]
                if not self._action_active[i]:
                    if action_probs[i] > thresh + self.hysteresis_margin:
                        self._action_active[i] = True
                else:
                    if action_probs[i] < thresh - self.hysteresis_margin:
                        self._action_active[i] = False

            output.move_forward = bool(self._action_active[0])
            output.move_backward = bool(self._action_active[1])
            output.strafe_left = bool(self._action_active[2])
            output.strafe_right = bool(self._action_active[3])
            output.sprint = bool(self._action_active[4])
            output.sneak = bool(self._action_active[5])
            output.jump = bool(self._action_active[6])
            output.attack = bool(self._action_active[7])
            output.use_item = bool(self._action_active[8])
            output.drop_item = bool(self._action_active[9])
            output.swap_offhand = bool(self._action_active[10])

            # Look: deadzone + EMA smoothing + rate limiting
            raw_yaw = float(look_raw[0])
            raw_pitch = float(look_raw[1])

            if abs(raw_yaw) < self.look_deadzone:
                raw_yaw = 0.0
            if abs(raw_pitch) < self.look_deadzone:
                raw_pitch = 0.0

            smooth_yaw = (self.look_alpha * raw_yaw
                          + (1 - self.look_alpha) * self._smooth_yaw)
            smooth_pitch = (self.look_alpha * raw_pitch
                            + (1 - self.look_alpha) * self._smooth_pitch)

            delta_yaw = smooth_yaw - self._smooth_yaw
            delta_pitch = smooth_pitch - self._smooth_pitch
            if abs(delta_yaw) > self.look_rate_limit:
                smooth_yaw = (self._smooth_yaw
                              + self.look_rate_limit * np.sign(delta_yaw))
            if abs(delta_pitch) > self.look_rate_limit:
                smooth_pitch = (self._smooth_pitch
                                + self.look_rate_limit * np.sign(delta_pitch))

            self._smooth_yaw = smooth_yaw
            self._smooth_pitch = smooth_pitch
            output.look_yaw = smooth_yaw
            output.look_pitch = smooth_pitch

            # Hotbar: argmax with min confidence, only on slot change
            max_idx = int(hotbar_probs.argmax())
            max_conf = hotbar_probs[max_idx]
            if (max_conf >= self.hotbar_min_confidence
                    and max_idx != self._current_hotbar_slot):
                output.hotbar_slot = max_idx
                self._current_hotbar_slot = max_idx

        else:
            # --- Screen Open Mode ---

            # Cursor: EMA smoothing
            cx = float(cursor_raw[0])
            cy = float(cursor_raw[1])
            self._smooth_cursor_x = (
                self.cursor_alpha * cx
                + (1 - self.cursor_alpha) * self._smooth_cursor_x
            )
            self._smooth_cursor_y = (
                self.cursor_alpha * cy
                + (1 - self.cursor_alpha) * self._smooth_cursor_y
            )
            output.cursor_x = max(0.0, min(1.0, self._smooth_cursor_x))
            output.cursor_y = max(0.0, min(1.0, self._smooth_cursor_y))

            # Inv click thresholds with hysteresis
            for i in range(3):
                thresh = self.inv_click_thresholds[i]
                if not self._inv_active[i]:
                    if inv_probs[i] > thresh + self.hysteresis_margin:
                        self._inv_active[i] = True
                else:
                    if inv_probs[i] < thresh - self.hysteresis_margin:
                        self._inv_active[i] = False

            output.inv_left_click = bool(self._inv_active[0])
            output.inv_right_click = bool(self._inv_active[1])
            output.inv_shift_held = bool(self._inv_active[2])

        # open_inventory is always active regardless of mode (action index 11)
        oi_thresh = self.action_thresholds[11]
        if not self._action_active[11]:
            if action_probs[11] > oi_thresh + self.hysteresis_margin:
                self._action_active[11] = True
        else:
            if action_probs[11] < oi_thresh - self.hysteresis_margin:
                self._action_active[11] = False
        output.open_inventory = bool(self._action_active[11])

        # Update action history with this frame's output
        self._action_history.append(output.to_control_vector())

        # Sync current hotbar slot from game state
        if last_gs[44] > 0:
            self._current_hotbar_slot = int(round(last_gs[44] * 8))

        return output

    def clear(self):
        """Reset all buffers and state."""
        self._buffer.clear()
        self._gs_buffer.clear()
        self._action_history = deque(
            [np.zeros(V2_NUM_CONTROLS, dtype=np.float32)] * V2_ACTION_HISTORY_K,
            maxlen=V2_ACTION_HISTORY_K,
        )
        self._action_active[:] = False
        self._inv_active[:] = False
        self._smooth_yaw = 0.0
        self._smooth_pitch = 0.0
        self._smooth_cursor_x = 0.5
        self._smooth_cursor_y = 0.5
