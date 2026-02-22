"""Training script for the ControlTransformerV2 policy model.

Usage:
    cd client
    python train_controls.py
    python train_controls.py --recordings_dir recordings_control --epochs 200

Trains on recorded sessions, saves best model to models/control_policy_v2.pt
and config to models/control_config_v2.json.

V2: 5-head model (action, look, hotbar, cursor, inv_click) with mode-aware
masked losses. Each output head is only trained on relevant frames.

Requirements: torch, numpy
"""

import os
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from control_dataset import (
    ControlDataset, prepare_datasets,
    INPUT_DIM, WINDOW_SIZE, WINDOW_STRIDE, NUM_CONTROLS,
    GAME_STATE_DIM, ACTION_HISTORY_DIM,
    ACTION_INDICES, LOOK_INDICES, HOTBAR_INDICES,
    CURSOR_INDICES, INV_CLICK_INDICES,
)
from control_model import ControlTransformerV2

NUM_ACTION = 12
NUM_LOOK = 2
NUM_HOTBAR = 9
NUM_CURSOR = 2
NUM_INV_CLICK = 3

ACTION_CONTROL_NAMES = [
    "move_forward", "move_backward", "strafe_left", "strafe_right",
    "sprint", "sneak", "jump", "attack", "use_item",
    "drop_item", "swap_offhand", "open_inventory",
]
INV_CLICK_NAMES = ["inv_left_click", "inv_right_click", "inv_shift_held"]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = WINDOW_SIZE   # 30
BATCH_SIZE = 64
EPOCHS = 150
LR = 5e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
GRADIENT_CLIP = 1.0

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.3

EARLY_STOP_PATIENCE = 25
VAL_FRACTION = 0.2
SEED = 42
MAX_POS_WEIGHT = 50.0

# Loss weights
W_ACTION = 1.0
W_LOOK = 0.5
W_HOTBAR = 0.3
W_CURSOR = 0.5
W_INV = 0.3
W_IDLE = 0.3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")


# ---------------------------------------------------------------------------
# Custom collate
# ---------------------------------------------------------------------------

def v2_collate_fn(batch):
    """Collate that batches mode_masks dicts into tensors."""
    features = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])

    mode_mask_keys = batch[0][3].keys()
    mode_masks = {
        key: torch.tensor([b[3][key] for b in batch], dtype=torch.bool)
        for key in mode_mask_keys
    }
    return features, masks, targets, mode_masks


# ---------------------------------------------------------------------------
# Pos weights
# ---------------------------------------------------------------------------

def compute_pos_weights(all_targets: np.ndarray, indices: List[int]) -> torch.Tensor:
    """Compute per-control pos_weight for BCEWithLogitsLoss.

    pos_weight[i] = num_negative / num_positive, capped at MAX_POS_WEIGHT.
    """
    selected = all_targets[:, indices]
    n_samples = selected.shape[0]
    pos_counts = (selected > 0.5).sum(axis=0).astype(np.float64)
    neg_counts = n_samples - pos_counts
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = neg_counts / pos_counts
    weights = np.minimum(weights, MAX_POS_WEIGHT)
    return torch.from_numpy(weights.astype(np.float32))


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_v2_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    mode_masks: Dict[str, torch.Tensor],
    pos_weights_action: torch.Tensor,
    pos_weights_inv: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Mode-aware masked loss for all 5 heads + idle penalty.

    Args:
        outputs: dict from model forward (action_logits, look, hotbar_logits,
                 cursor, inv_click_logits).
        targets: (batch, 28) full control vector.
        mode_masks: dict of (batch,) bool tensors with keys:
            gameplay, screen_open, hotbar_changed, look_active.
        pos_weights_action: (12,) for action BCE.
        pos_weights_inv: (3,) for inv_click BCE.

    Returns:
        (total_loss, loss_dict) where loss_dict has per-head scalar losses.
    """
    device = targets.device
    gameplay = mode_masks['gameplay'].to(device)
    screen_open = mode_masks['screen_open'].to(device)
    hotbar_changed = mode_masks['hotbar_changed'].to(device)
    look_active = mode_masks['look_active'].to(device)

    loss_dict = {}

    # 1. Action loss: BCE on 12 binary actions — ONLY gameplay frames
    action_targets = targets[:, ACTION_INDICES]
    if gameplay.any():
        action_loss = F.binary_cross_entropy_with_logits(
            outputs['action_logits'][gameplay],
            action_targets[gameplay],
            pos_weight=pos_weights_action,
            reduction='mean',
        )
    else:
        action_loss = torch.tensor(0.0, device=device)
    loss_dict['action'] = action_loss

    # 2. Look loss: SmoothL1 — ONLY gameplay AND look_active frames
    look_targets = targets[:, LOOK_INDICES]
    look_mask = gameplay & look_active
    if look_mask.any():
        look_loss = F.smooth_l1_loss(
            outputs['look'][look_mask],
            look_targets[look_mask],
            reduction='mean',
        )
    else:
        look_loss = torch.tensor(0.0, device=device)
    loss_dict['look'] = look_loss

    # 3. Hotbar loss: CrossEntropy — ONLY hotbar_changed frames
    hotbar_targets_onehot = targets[:, HOTBAR_INDICES]
    if hotbar_changed.any():
        hotbar_class = hotbar_targets_onehot[hotbar_changed].argmax(dim=1)
        hotbar_loss = F.cross_entropy(
            outputs['hotbar_logits'][hotbar_changed],
            hotbar_class,
        )
    else:
        hotbar_loss = torch.tensor(0.0, device=device)
    loss_dict['hotbar'] = hotbar_loss

    # 4. Cursor loss: MSE — ONLY screen_open frames
    cursor_targets = targets[:, CURSOR_INDICES]
    if screen_open.any():
        cursor_loss = F.mse_loss(
            outputs['cursor'][screen_open],
            cursor_targets[screen_open],
            reduction='mean',
        )
    else:
        cursor_loss = torch.tensor(0.0, device=device)
    loss_dict['cursor'] = cursor_loss

    # 5. InvClick loss: BCE — ONLY screen_open frames
    inv_targets = targets[:, INV_CLICK_INDICES]
    if screen_open.any():
        inv_loss = F.binary_cross_entropy_with_logits(
            outputs['inv_click_logits'][screen_open],
            inv_targets[screen_open],
            pos_weight=pos_weights_inv,
            reduction='mean',
        )
    else:
        inv_loss = torch.tensor(0.0, device=device)
    loss_dict['inv_click'] = inv_loss

    # 6. Idle penalty: gameplay frames where ALL action targets are 0
    idle_mask = gameplay & (action_targets.sum(dim=1) < 0.5)
    if idle_mask.any():
        idle_probs = torch.sigmoid(outputs['action_logits'][idle_mask])
        idle_loss = idle_probs.mean()
    else:
        idle_loss = torch.tensor(0.0, device=device)
    loss_dict['idle'] = idle_loss

    total = (W_ACTION * action_loss
             + W_LOOK * look_loss
             + W_HOTBAR * hotbar_loss
             + W_CURSOR * cursor_loss
             + W_INV * inv_loss
             + W_IDLE * idle_loss)

    return total, loss_dict


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weights_action: torch.Tensor,
    pos_weights_inv: torch.Tensor,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> Dict[str, float]:
    """Train for one epoch. Returns dict of average losses."""
    model.train()
    accum = {}
    n = 0

    for features, mask, targets, mode_masks in loader:
        features = features.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        B = targets.size(0)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(features, mask)
            loss, loss_dict = compute_v2_loss(
                outputs, targets, mode_masks,
                pos_weights_action, pos_weights_inv,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()

        for key, val in loss_dict.items():
            v = val.item() if torch.is_tensor(val) else val
            accum[key] = accum.get(key, 0.0) + v * B
        accum['total'] = accum.get('total', 0.0) + loss.item() * B
        n += B

    return {k: v / n for k, v in accum.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    pos_weights_action: torch.Tensor,
    pos_weights_inv: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """Evaluate model on a loader.

    Returns:
        losses: dict of average losses
        all_outputs: dict of numpy arrays for each model head
        all_targets: (N, 28) numpy array
        all_masks: dict of bool numpy arrays for each mode mask
    """
    model.eval()
    accum = {}
    n = 0

    collect_outputs = {
        'action_logits': [], 'look': [], 'hotbar_logits': [],
        'cursor': [], 'inv_click_logits': [],
    }
    collect_targets = []
    collect_masks = {
        'gameplay': [], 'screen_open': [],
        'hotbar_changed': [], 'look_active': [],
    }

    for features, mask, targets, mode_masks in loader:
        features = features.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        B = targets.size(0)

        outputs = model(features, mask)
        loss, loss_dict = compute_v2_loss(
            outputs, targets, mode_masks,
            pos_weights_action, pos_weights_inv,
        )

        for key, val in loss_dict.items():
            v = val.item() if torch.is_tensor(val) else val
            accum[key] = accum.get(key, 0.0) + v * B
        accum['total'] = accum.get('total', 0.0) + loss.item() * B
        n += B

        for key in collect_outputs:
            collect_outputs[key].append(outputs[key].cpu().numpy())
        collect_targets.append(targets.cpu().numpy())
        for key in collect_masks:
            collect_masks[key].append(mode_masks[key].numpy())

    losses = {k: v / n for k, v in accum.items()}
    all_outputs = {k: np.concatenate(v) for k, v in collect_outputs.items()}
    all_targets = np.concatenate(collect_targets)
    all_masks = {k: np.concatenate(v) for k, v in collect_masks.items()}

    return losses, all_outputs, all_targets, all_masks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_action_metrics(
    action_logits: np.ndarray,
    action_targets: np.ndarray,
    gameplay_mask: np.ndarray,
    thresholds: np.ndarray,
) -> List[Dict[str, float]]:
    """Per-action precision, recall, F1 on gameplay frames only."""
    if not gameplay_mask.any():
        return [{"precision": 0, "recall": 0, "f1": 0, "support": 0}] * NUM_ACTION

    logits = action_logits[gameplay_mask]
    targets = action_targets[gameplay_mask]
    probs = 1.0 / (1.0 + np.exp(-logits))

    metrics = []
    for i in range(NUM_ACTION):
        preds = (probs[:, i] > thresholds[i]).astype(float)
        tgt = targets[:, i]

        tp = ((preds == 1) & (tgt > 0.5)).sum()
        fp = ((preds == 1) & (tgt < 0.5)).sum()
        fn = ((preds == 0) & (tgt > 0.5)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics.append({
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int((tgt > 0.5).sum()),
        })
    return metrics


def compute_look_metrics(
    look_preds: np.ndarray,
    look_targets: np.ndarray,
    gameplay_mask: np.ndarray,
    look_active_mask: np.ndarray,
) -> Dict[str, float]:
    """Look MSE and directional accuracy on gameplay + look_active frames."""
    active = gameplay_mask & look_active_mask
    if not active.any():
        return {"mse": 0.0, "directional_accuracy": 1.0}

    preds = look_preds[active]
    targets = look_targets[active]

    mse = float(np.mean((preds - targets) ** 2))
    dir_match = float((np.sign(preds) == np.sign(targets)).mean())

    return {"mse": mse, "directional_accuracy": dir_match}


def compute_hotbar_metrics(
    hotbar_logits: np.ndarray,
    hotbar_targets: np.ndarray,
    hotbar_mask: np.ndarray,
) -> Dict[str, float]:
    """Hotbar top-1 accuracy on hotbar_changed frames."""
    if not hotbar_mask.any():
        return {"accuracy": 1.0, "count": 0}

    logits = hotbar_logits[hotbar_mask]
    targets = hotbar_targets[hotbar_mask]

    pred_class = logits.argmax(axis=1)
    true_class = targets.argmax(axis=1)
    acc = float((pred_class == true_class).mean())

    return {"accuracy": acc, "count": int(hotbar_mask.sum())}


def compute_cursor_metrics(
    cursor_preds: np.ndarray,
    cursor_targets: np.ndarray,
    screen_mask: np.ndarray,
) -> Dict[str, float]:
    """Cursor MSE on screen_open frames."""
    if not screen_mask.any():
        return {"mse": 0.0, "count": 0}

    preds = cursor_preds[screen_mask]
    targets = cursor_targets[screen_mask]
    mse = float(np.mean((preds - targets) ** 2))

    return {"mse": mse, "count": int(screen_mask.sum())}


def compute_inv_metrics(
    inv_logits: np.ndarray,
    inv_targets: np.ndarray,
    screen_mask: np.ndarray,
    thresholds: np.ndarray,
) -> List[Dict[str, float]]:
    """Per-control F1 for inventory clicks on screen_open frames."""
    if not screen_mask.any():
        return [{"precision": 0, "recall": 0, "f1": 0, "support": 0}] * NUM_INV_CLICK

    logits = inv_logits[screen_mask]
    targets = inv_targets[screen_mask]
    probs = 1.0 / (1.0 + np.exp(-logits))

    metrics = []
    for i in range(NUM_INV_CLICK):
        preds = (probs[:, i] > thresholds[i]).astype(float)
        tgt = targets[:, i]

        tp = ((preds == 1) & (tgt > 0.5)).sum()
        fp = ((preds == 1) & (tgt < 0.5)).sum()
        fn = ((preds == 0) & (tgt > 0.5)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics.append({
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int((tgt > 0.5).sum()),
        })
    return metrics


def compute_idle_accuracy(
    action_logits: np.ndarray,
    action_targets: np.ndarray,
    gameplay_mask: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    """Fraction of idle gameplay frames where ALL actions are below threshold."""
    if not gameplay_mask.any():
        return 1.0

    logits = action_logits[gameplay_mask]
    targets = action_targets[gameplay_mask]

    idle = targets.sum(axis=1) < 0.5
    if not idle.any():
        return 1.0

    probs = 1.0 / (1.0 + np.exp(-logits[idle]))
    all_below = np.all(probs < thresholds[None, :], axis=1)
    return float(all_below.mean())


def compute_combined_score(
    action_metrics: List[Dict[str, float]],
    look_metrics: Dict[str, float],
    hotbar_metrics: Dict[str, float],
    cursor_metrics: Dict[str, float],
    inv_metrics: List[Dict[str, float]],
    idle_acc: float,
) -> float:
    """Combined score: 0.4*action_F1 + 0.2*look + 0.1*hotbar + 0.1*cursor + 0.1*inv + 0.1*idle."""
    avg_action_f1 = float(np.mean([m["f1"] for m in action_metrics]))
    look_score = max(0.0, 1.0 - look_metrics["mse"])
    hotbar_score = hotbar_metrics["accuracy"]
    cursor_score = max(0.0, 1.0 - cursor_metrics["mse"])
    avg_inv_f1 = float(np.mean([m["f1"] for m in inv_metrics]))

    return (0.4 * avg_action_f1
            + 0.2 * look_score
            + 0.1 * hotbar_score
            + 0.1 * cursor_score
            + 0.1 * avg_inv_f1
            + 0.1 * idle_acc)


def compute_all_metrics(
    all_outputs: Dict[str, np.ndarray],
    all_targets: np.ndarray,
    all_masks: Dict[str, np.ndarray],
    action_thresholds: np.ndarray,
    inv_thresholds: np.ndarray,
) -> Tuple:
    """Compute all per-head metrics. Returns tuple of metric dicts."""
    action_targets = all_targets[:, ACTION_INDICES]
    look_targets = all_targets[:, LOOK_INDICES]
    hotbar_targets = all_targets[:, HOTBAR_INDICES]
    cursor_targets = all_targets[:, CURSOR_INDICES]
    inv_targets = all_targets[:, INV_CLICK_INDICES]

    action_m = compute_action_metrics(
        all_outputs['action_logits'], action_targets,
        all_masks['gameplay'], action_thresholds,
    )
    look_m = compute_look_metrics(
        all_outputs['look'], look_targets,
        all_masks['gameplay'], all_masks['look_active'],
    )
    hotbar_m = compute_hotbar_metrics(
        all_outputs['hotbar_logits'], hotbar_targets,
        all_masks['hotbar_changed'],
    )
    cursor_m = compute_cursor_metrics(
        all_outputs['cursor'], cursor_targets,
        all_masks['screen_open'],
    )
    inv_m = compute_inv_metrics(
        all_outputs['inv_click_logits'], inv_targets,
        all_masks['screen_open'], inv_thresholds,
    )
    idle_acc = compute_idle_accuracy(
        all_outputs['action_logits'], action_targets,
        all_masks['gameplay'], action_thresholds,
    )

    return action_m, look_m, hotbar_m, cursor_m, inv_m, idle_acc


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------

def optimize_action_thresholds(
    action_logits: np.ndarray,
    action_targets: np.ndarray,
    gameplay_mask: np.ndarray,
) -> np.ndarray:
    """Sweep thresholds per action to maximize F1 (penalizing idle FPR)."""
    thresholds = np.full(NUM_ACTION, 0.5)
    if not gameplay_mask.any():
        return thresholds

    logits = action_logits[gameplay_mask]
    targets = action_targets[gameplay_mask]
    probs = 1.0 / (1.0 + np.exp(-logits))

    idle_mask = targets.sum(axis=1) < 0.5
    candidates = np.arange(0.10, 0.91, 0.05)

    for i in range(NUM_ACTION):
        best_score = -1.0
        best_t = 0.5
        for t in candidates:
            preds = (probs[:, i] > t).astype(float)
            tgt = targets[:, i]

            tp = ((preds == 1) & (tgt > 0.5)).sum()
            fp = ((preds == 1) & (tgt < 0.5)).sum()
            fn = ((preds == 0) & (tgt > 0.5)).sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if idle_mask.any():
                idle_preds = (probs[idle_mask, i] > t).astype(float)
                fpr = idle_preds.mean()
            else:
                fpr = 0.0

            score = f1 - 0.5 * fpr
            if score > best_score:
                best_score = score
                best_t = t
        thresholds[i] = best_t

    return thresholds


def optimize_hotbar_confidence(
    hotbar_logits: np.ndarray,
    hotbar_targets: np.ndarray,
    hotbar_mask: np.ndarray,
) -> float:
    """Find minimum confidence threshold for hotbar predictions."""
    if not hotbar_mask.any():
        return 0.3

    logits = hotbar_logits[hotbar_mask]
    targets = hotbar_targets[hotbar_mask]
    true_class = targets.argmax(axis=1)

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred_class = logits.argmax(axis=1)
    pred_conf = softmax[np.arange(len(pred_class)), pred_class]

    best_score = 0.0
    best_t = 0.3
    for t in np.arange(0.10, 0.91, 0.05):
        confident = pred_conf >= t
        if confident.any():
            acc = float((pred_class[confident] == true_class[confident]).mean())
            coverage = confident.mean()
            score = acc * coverage
        else:
            score = 0.0
        if score > best_score:
            best_score = score
            best_t = t

    return float(best_t)


def optimize_inv_thresholds(
    inv_logits: np.ndarray,
    inv_targets: np.ndarray,
    screen_mask: np.ndarray,
) -> np.ndarray:
    """Sweep thresholds per inv_click control to maximize F1."""
    thresholds = np.full(NUM_INV_CLICK, 0.5)
    if not screen_mask.any():
        return thresholds

    logits = inv_logits[screen_mask]
    targets = inv_targets[screen_mask]
    probs = 1.0 / (1.0 + np.exp(-logits))
    candidates = np.arange(0.10, 0.91, 0.05)

    for i in range(NUM_INV_CLICK):
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            preds = (probs[:, i] > t).astype(float)
            tgt = targets[:, i]

            tp = ((preds == 1) & (tgt > 0.5)).sum()
            fp = ((preds == 1) & (tgt < 0.5)).sum()
            fn = ((preds == 0) & (tgt > 0.5)).sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = best_t

    return thresholds


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_metrics(
    action_metrics: List[Dict[str, float]],
    look_metrics: Dict[str, float],
    hotbar_metrics: Dict[str, float],
    cursor_metrics: Dict[str, float],
    inv_metrics: List[Dict[str, float]],
    idle_acc: float,
    action_thresholds: np.ndarray,
    inv_thresholds: np.ndarray,
    hotbar_min_conf: float,
):
    """Print a clean metrics table for all heads."""
    print(f"\n{'='*70}")
    print("ACTION HEAD (12 binary controls, gameplay frames only)")
    print(f"{'Control':<18s} {'Thresh':>6s} {'Prec':>6s} {'Rec':>6s} "
          f"{'F1':>6s} {'Support':>8s}")
    print("-" * 52)
    for i, name in enumerate(ACTION_CONTROL_NAMES):
        m = action_metrics[i]
        print(f"{name:<18s} {action_thresholds[i]:>6.2f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>8d}")
    avg_f1 = np.mean([m["f1"] for m in action_metrics])
    print("-" * 52)
    print(f"{'Avg F1':<18s} {'':>6s} {'':>6s} {'':>6s} {avg_f1:>6.3f}")
    print(f"Idle accuracy: {idle_acc:.1%}")

    print(f"\n{'='*70}")
    print("LOOK HEAD (2 analog, gameplay + look_active frames)")
    print(f"  MSE: {look_metrics['mse']:.4f}")
    print(f"  Directional accuracy: {look_metrics['directional_accuracy']:.1%}")

    print(f"\n{'='*70}")
    print("HOTBAR HEAD (9-class, hotbar_changed frames)")
    print(f"  Accuracy: {hotbar_metrics['accuracy']:.1%} (n={hotbar_metrics['count']})")
    print(f"  Min confidence: {hotbar_min_conf:.2f}")

    print(f"\n{'='*70}")
    print("CURSOR HEAD (2 analog, screen_open frames)")
    print(f"  MSE: {cursor_metrics['mse']:.4f} (n={cursor_metrics['count']})")

    print(f"\n{'='*70}")
    print("INV CLICK HEAD (3 binary, screen_open frames)")
    for i, name in enumerate(INV_CLICK_NAMES):
        m = inv_metrics[i]
        print(f"  {name:<18s} thresh={inv_thresholds[i]:.2f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"sup={m['support']}")

    combined = compute_combined_score(
        action_metrics, look_metrics, hotbar_metrics,
        cursor_metrics, inv_metrics, idle_acc,
    )
    print(f"\n{'='*70}")
    print(f"COMBINED SCORE: {combined:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train ControlTransformerV2 policy")
    parser.add_argument("--recordings_dir", default=None,
                        help="Recordings directory (default: client/recordings_control)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("Loading recordings...")
    kwargs = {}
    if args.recordings_dir:
        kwargs["recordings_dir"] = args.recordings_dir

    train_ds, val_ds = prepare_datasets(
        val_fraction=VAL_FRACTION,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        seed=args.seed,
        **kwargs,
    )

    if len(train_ds) == 0:
        print("No training data found!")
        return

    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    # --- Gather all training targets for pos_weight + stats ---
    print("Computing dataset statistics...")
    all_targets = []
    all_mode_masks = {'gameplay': [], 'screen_open': [],
                      'hotbar_changed': [], 'look_active': []}
    for i in range(len(train_ds)):
        _, _, target, mm = train_ds[i]
        t = target.numpy() if torch.is_tensor(target) else np.asarray(target)
        all_targets.append(t)
        for key in all_mode_masks:
            all_mode_masks[key].append(mm[key])
    all_targets = np.stack(all_targets)
    all_mode_masks = {k: np.array(v) for k, v in all_mode_masks.items()}

    # Print activation rates
    print("\nAction activation rates (train):")
    for i, name in enumerate(ACTION_CONTROL_NAMES):
        idx = ACTION_INDICES[i]
        count = int((all_targets[:, idx] > 0.5).sum())
        pct = 100.0 * count / len(all_targets)
        print(f"  {name:>18s}: {count:>6d} ({pct:5.1f}%)")

    idle_count = int((all_targets[:, ACTION_INDICES].sum(axis=1) < 0.5).sum())
    print(f"  {'idle':>18s}: {idle_count:>6d} ({100.0 * idle_count / len(all_targets):5.1f}%)")

    hotbar_changes = int(all_mode_masks['hotbar_changed'].sum())
    screen_open_count = int(all_mode_masks['screen_open'].sum())
    look_active_count = int(all_mode_masks['look_active'].sum())
    gameplay_count = int(all_mode_masks['gameplay'].sum())
    print(f"\nMode distribution:")
    print(f"  {'gameplay':>18s}: {gameplay_count:>6d} ({100.0 * gameplay_count / len(all_targets):5.1f}%)")
    print(f"  {'screen_open':>18s}: {screen_open_count:>6d} ({100.0 * screen_open_count / len(all_targets):5.1f}%)")
    print(f"  {'hotbar_changed':>18s}: {hotbar_changes:>6d} ({100.0 * hotbar_changes / len(all_targets):5.1f}%)")
    print(f"  {'look_active':>18s}: {look_active_count:>6d} ({100.0 * look_active_count / len(all_targets):5.1f}%)")

    # --- Pos weights ---
    pos_weights_action = compute_pos_weights(all_targets, ACTION_INDICES).to(device)
    pos_weights_inv = compute_pos_weights(all_targets, INV_CLICK_INDICES).to(device)

    print("\nAction pos weights:")
    for i, name in enumerate(ACTION_CONTROL_NAMES):
        print(f"  {name:>18s}: {pos_weights_action[i].item():.1f}")
    print("InvClick pos weights:")
    for i, name in enumerate(INV_CLICK_NAMES):
        print(f"  {name:>18s}: {pos_weights_inv[i].item():.1f}")

    # --- Data loaders ---
    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=False, pin_memory=use_pin, collate_fn=v2_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin, collate_fn=v2_collate_fn,
    )

    # --- Model ---
    model = ControlTransformerV2(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
        num_action=NUM_ACTION,
        num_look=NUM_LOOK,
        num_hotbar=NUM_HOTBAR,
        num_cursor=NUM_CURSOR,
        num_inv_click=NUM_INV_CLICK,
    ).to(device)

    param_count = (model.param_count() if hasattr(model, 'param_count')
                   else sum(p.numel() for p in model.parameters()))
    print(f"\nModel parameters: {param_count:,}")

    # --- Optimizer, scheduler, AMP ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY,
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=LR_MIN / args.lr, total_iters=WARMUP_EPOCHS,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - WARMUP_EPOCHS, eta_min=LR_MIN,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # --- Training loop ---
    best_score = 0.0
    epochs_no_improve = 0
    best_state = None
    training_log = []

    default_action_thresh = np.full(NUM_ACTION, 0.5)
    default_inv_thresh = np.full(NUM_INV_CLICK, 0.5)

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(early stop patience: {args.patience})...\n")
    print(f"{'Ep':>4s}  {'TrLoss':>7s} {'Act':>6s} {'Look':>6s} {'Hbar':>6s} "
          f"{'Curs':>6s} {'Inv':>6s} {'Idle':>6s}  "
          f"{'VlLoss':>7s} {'VlF1':>6s} {'Score':>6s}  {'LR':>9s}")
    print("-" * 100)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer,
            pos_weights_action, pos_weights_inv, device, scaler,
        )

        # Validate
        val_losses, val_outputs, val_targets, val_masks = evaluate(
            model, val_loader, pos_weights_action, pos_weights_inv, device,
        )

        scheduler.step()

        # Metrics with default thresholds
        action_m, look_m, hotbar_m, cursor_m, inv_m, idle_acc = compute_all_metrics(
            val_outputs, val_targets, val_masks,
            default_action_thresh, default_inv_thresh,
        )
        avg_f1 = float(np.mean([m["f1"] for m in action_m]))
        combined = compute_combined_score(
            action_m, look_m, hotbar_m, cursor_m, inv_m, idle_acc,
        )

        lr = optimizer.param_groups[0]["lr"]
        marker = ""
        if combined > best_score:
            best_score = combined
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " *"
        else:
            epochs_no_improve += 1

        print(f"{epoch:>4d}  {train_losses.get('total', 0):>7.4f} "
              f"{train_losses.get('action', 0):>6.4f} "
              f"{train_losses.get('look', 0):>6.4f} "
              f"{train_losses.get('hotbar', 0):>6.4f} "
              f"{train_losses.get('cursor', 0):>6.4f} "
              f"{train_losses.get('inv_click', 0):>6.4f} "
              f"{train_losses.get('idle', 0):>6.4f}  "
              f"{val_losses.get('total', 0):>7.4f} {avg_f1:>6.3f} "
              f"{combined:>6.3f}  {lr:>9.2e}{marker}")

        training_log.append({
            "epoch": epoch,
            "lr": lr,
            "train": train_losses,
            "val": val_losses,
            "val_avg_action_f1": avg_f1,
            "val_look_mse": look_m["mse"],
            "val_look_dir_acc": look_m["directional_accuracy"],
            "val_hotbar_acc": hotbar_m["accuracy"],
            "val_cursor_mse": cursor_m["mse"],
            "val_idle_acc": float(idle_acc),
            "val_combined_score": float(combined),
        })

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best combined score: {best_score:.3f}")

    if best_state is None:
        print("No model saved (no improvement over baseline).")
        return

    # --- Final evaluation with best model ---
    model.load_state_dict(best_state)
    model.to(device)

    _, final_outputs, final_targets, final_masks = evaluate(
        model, val_loader, pos_weights_action, pos_weights_inv, device,
    )

    # --- Threshold optimization ---
    print("\nOptimizing thresholds on validation set...")

    action_targets_val = final_targets[:, ACTION_INDICES]
    hotbar_targets_val = final_targets[:, HOTBAR_INDICES]
    inv_targets_val = final_targets[:, INV_CLICK_INDICES]

    opt_action_thresh = optimize_action_thresholds(
        final_outputs['action_logits'], action_targets_val,
        final_masks['gameplay'],
    )
    opt_hotbar_conf = optimize_hotbar_confidence(
        final_outputs['hotbar_logits'], hotbar_targets_val,
        final_masks['hotbar_changed'],
    )
    opt_inv_thresh = optimize_inv_thresholds(
        final_outputs['inv_click_logits'], inv_targets_val,
        final_masks['screen_open'],
    )

    # --- Final metrics with optimized thresholds ---
    action_m, look_m, hotbar_m, cursor_m, inv_m, idle_acc = compute_all_metrics(
        final_outputs, final_targets, final_masks,
        opt_action_thresh, opt_inv_thresh,
    )

    print_metrics(
        action_m, look_m, hotbar_m, cursor_m, inv_m, idle_acc,
        opt_action_thresh, opt_inv_thresh, opt_hotbar_conf,
    )

    # --- Save model ---
    model_path = os.path.join(MODELS_DIR, "control_policy_v2.pt")
    torch.save(best_state, model_path)
    print(f"\nSaved model to {model_path}")

    # --- Save config (Contract 7 format) ---
    config = {
        "version": 2,
        "model": {
            "input_dim": INPUT_DIM,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "max_seq_len": MAX_SEQ_LEN,
            "num_action": NUM_ACTION,
            "num_look": NUM_LOOK,
            "num_hotbar": NUM_HOTBAR,
            "num_cursor": NUM_CURSOR,
            "num_inv_click": NUM_INV_CLICK,
        },
        "thresholds": {
            "action": [float(t) for t in opt_action_thresh],
            "hotbar_min_confidence": float(opt_hotbar_conf),
            "inv_click": [float(t) for t in opt_inv_thresh],
        },
        "post_processing": {
            "hysteresis_margin": 0.05,
            "look_deadzone": 0.08,
            "look_ema_alpha": 0.35,
            "look_rate_limit": 0.3,
            "cursor_ema_alpha": 0.5,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": WEIGHT_DECAY,
            "epochs_trained": len(training_log),
            "best_combined_score": float(best_score),
            "combined_score_weights": {
                "action_f1": 0.4,
                "look": 0.2,
                "hotbar": 0.1,
                "cursor": 0.1,
                "inv": 0.1,
                "idle": 0.1,
            },
            "per_head_metrics": {
                "action_avg_f1": float(np.mean([m["f1"] for m in action_m])),
                "look_mse": look_m["mse"],
                "look_dir_acc": look_m["directional_accuracy"],
                "hotbar_acc": hotbar_m["accuracy"],
                "cursor_mse": cursor_m["mse"],
                "inv_avg_f1": float(np.mean([m["f1"] for m in inv_m])),
                "idle_acc": float(idle_acc),
            },
            "pos_weights_action": {
                name: float(pos_weights_action[i].item())
                for i, name in enumerate(ACTION_CONTROL_NAMES)
            },
            "pos_weights_inv": {
                name: float(pos_weights_inv[i].item())
                for i, name in enumerate(INV_CLICK_NAMES)
            },
        },
    }

    config_path = os.path.join(MODELS_DIR, "control_config_v2.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # --- Save training log ---
    log_path = os.path.join(MODELS_DIR, "training_log_v2.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
