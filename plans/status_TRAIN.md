# Status: TRAIN

## Completed
- [x] Complete rewrite of `train_controls.py` for V2 5-head model
- [x] Custom `v2_collate_fn` to batch mode_masks dicts into tensors
- [x] `compute_v2_loss` — mode-aware masked loss for all 5 heads + idle penalty
  - Action: BCE on 12 binary actions, gameplay frames only
  - Look: SmoothL1 on yaw/pitch, gameplay AND look_active only
  - Hotbar: CrossEntropy on 9-class, hotbar_changed only
  - Cursor: MSE on 2D position, screen_open only
  - InvClick: BCE on 3 binary, screen_open only
  - Idle: penalize action activations on idle gameplay frames
  - Weights: action=1.0, look=0.5, hotbar=0.3, cursor=0.5, inv=0.3, idle=0.3
- [x] Per-head evaluation metrics
  - Action: per-control precision/recall/F1 + avg F1 (gameplay frames)
  - Look: MSE + directional accuracy (gameplay + look_active frames)
  - Hotbar: top-1 accuracy (hotbar_changed frames)
  - Cursor: MSE (screen_open frames)
  - InvClick: per-control F1 (screen_open frames)
  - Idle accuracy: % of idle gameplay frames correctly predicted as all-zero
- [x] Combined score: 0.4*action_F1 + 0.2*(1-look_mse) + 0.1*hotbar_acc + 0.1*(1-cursor_mse) + 0.1*inv_F1 + 0.1*idle_acc
- [x] Threshold optimization
  - Action: per-control sweep maximizing F1 - 0.5*idle_FPR
  - Hotbar: confidence sweep maximizing accuracy * coverage
  - InvClick: per-control sweep maximizing F1
- [x] Output files: control_policy_v2.pt, control_config_v2.json, training_log_v2.json
- [x] Config follows Contract 7 format exactly
- [x] Mixed precision, warmup + cosine LR, gradient clipping, early stopping

## Interface Changes
- None — follows all contracts exactly as specified

## Notes
- Imports `ControlTransformerV2` from `control_model` (Instance A)
- Imports `prepare_datasets`, constants from `control_dataset` (Instance B)
- Dataset `__getitem__` must return `(features, mask, target, mode_masks)` where mode_masks is dict with keys: gameplay, screen_open, hotbar_changed, look_active (all bool)
- `compute_pos_weights` now takes `indices` parameter to select which columns to compute weights for (supports both 12-action and 3-inv_click)
- All target slicing uses `ACTION_INDICES`, `LOOK_INDICES`, etc. from Contract 2 — applied to the 28-dim target vector
- Model hyperparams: D_MODEL=256, NHEAD=8, NUM_LAYERS=6, DIM_FEEDFORWARD=512, DROPOUT=0.3
- Training: BATCH_SIZE=64, EPOCHS=150, LR=5e-4, warmup 5 epochs, cosine to 1e-6, early stop patience 25
- The stats-gathering loop iterates dataset items individually to get mode_mask distributions — works fine for typical dataset sizes but could be slow for >1M windows
