# Risks & Mitigations v2

## Critical Risks

### 1. Mod Expansion Complexity

**Risk**: Phase 5 requires significant Fabric mod changes — resolved input capture,
entity scanning, status effects, screen state detection. This is Java/Fabric development
that blocks all subsequent phases.

**Impact**: High — everything depends on the mod broadcasting the right data.

**Mitigation**:
- Implement mod changes incrementally (inputs first, then game state categories)
- Test each new field with a simple Python print script
- Keep fallback: recorder can still work with zeros for missing game state dims
- Document exact Minecraft API calls needed (see reference.md)

### 2. Inventory Cursor Precision

**Risk**: Predicting exact cursor position [0,1] from body pose may not be precise
enough for inventory slot selection. Minecraft inventory slots are small targets.

**Impact**: Medium — inventory management may be too inaccurate to use.

**Mitigation**:
- Use EMA smoothing on cursor output for stability
- Consider snap-to-slot: map cursor to nearest inventory slot center
- Could add a "fine adjustment" gesture mode (slow cursor vs fast cursor)
- Train with extra cursor data (dedicated inventory practice sessions)
- Worst case: inventory control works for basic operations, precise crafting
  done manually

### 3. Mode Confusion

**Risk**: The model might output gameplay controls while screen is open (or vice versa),
especially during the open/close transition frames.

**Impact**: Medium — could cause unintended actions during inventory use.

**Mitigation**:
- Masked loss ensures heads only train on appropriate frames
- Bridge enforces mode: only sends gameplay commands when screen_open=False
- Add transition buffer: 3-5 frames of "no action" when screen state changes
- `screen_open` comes from game state (mod broadcast), not model prediction

### 4. Look Sensitivity Calibration

**Risk**: Even with in-game yaw/pitch deltas (keybind-independent), the mapping from
body head movement to look output magnitude needs calibration per user.

**Impact**: Medium — too sensitive = nausea, too sluggish = unplayable.

**Mitigation**:
- Configurable LOOK_SENSITIVITY multiplier in control bridge
- EMA smoothing (alpha=0.35) dampens noise
- Deadzone (0.08) prevents drift when "looking straight"
- User calibration step: "look left as far as comfortable" → scales range
- Can be adjusted at runtime via keyboard shortcut

### 5. Tick Synchronization

**Risk**: Camera runs at ~30fps, Minecraft at 20 ticks/sec. MCCTP WebSocket adds
latency. Timing mismatch between pose frame and game state could cause misaligned
training data.

**Impact**: Medium — model learns slightly wrong associations.

**Mitigation**:
- Record timestamps for both pose frames and game state updates
- Align by nearest timestamp during dataset creation
- MCCTP latency is typically <5ms (localhost WebSocket)
- Camera-to-pose latency is the bigger issue (~30-50ms for WiLoR), but this
  is consistent and the model can learn to compensate

### 6. Action History Dependence

**Risk**: During inference, the model uses its own previous outputs as action history.
If it makes an error, that error propagates through the history buffer, potentially
causing a cascade of wrong predictions (exposure bias).

**Impact**: Medium — could cause "stuck" behaviors.

**Mitigation**:
- Action history dropout (30%) during training reduces reliance on history
- History buffer is only 5 frames (~170ms) — errors wash out quickly
- Release-all safety: if tracking is lost, clear history buffer
- Can add scheduled sampling: during training, gradually replace teacher-forced
  history with model predictions

---

## Moderate Risks

### 7. Data Imbalance

**Risk**: Most frames are idle/walking. Rare actions (jump, drop, swap_offhand,
inventory clicks) have very few positive samples.

**Impact**: Model may never learn to fire rare actions, or fire them randomly.

**Mitigation**:
- Per-control pos_weight (capped at 50.0) in BCE loss
- Targeted recording sessions for rare actions
- Bootstrapping from v1 gesture data provides some combat samples
- Threshold optimization finds best threshold even for rare controls
- Can oversample rare-action windows during training

### 8. Insufficient Training Data

**Risk**: 28 output controls with 46-dim game state and 260-dim pose may need
significantly more training data than v1's 10 outputs.

**Impact**: Model underfits or overfits to limited scenarios.

**Mitigation**:
- Start with movement-only data (natural, easy to collect in bulk)
- Gradually add combat, then inventory sessions
- Data augmentation: mirror, temporal jitter, noise injection
- ~3.5M params is reasonable for behavioral cloning with this input size
- Curriculum learning can help each head learn with focused data

### 9. WebSocket Reliability

**Risk**: MCCTP WebSocket connection drops during gameplay or recording, causing
missing game state or failed command delivery.

**Impact**: Low-medium — recording gaps or temporary loss of control.

**Mitigation**:
- Auto-reconnect with exponential backoff in both recorder and bridge
- Missing game state → use last known good state (don't crash)
- Bridge releases all held actions on disconnect (safety)
- Recorder marks disconnected frames in metadata

### 10. Real-Time Performance Budget

**Risk**: Full pipeline (pose + hands + game state encoding + transformer inference +
post-processing + MCCTP send) might exceed the ~33ms budget for 30fps.

**Impact**: Low — causes input lag if inference takes too long.

**Budget breakdown** (estimated):
- MediaPipe pose: ~5ms (GPU)
- WiLoR hands: ~10ms (GPU)
- Feature assembly: ~1ms (CPU)
- Transformer inference: ~5-8ms (GPU, 3.5M params)
- Post-processing: ~0.5ms (CPU)
- MCCTP send: ~1ms (localhost WebSocket)
- **Total: ~23-26ms** — within budget

**Mitigation**:
- Run inference every N frames (e.g., every 2nd frame for ~15fps control rate)
- Hand tracking is already the bottleneck, not the policy model
- Can reduce model size if needed (d_model=192, 4 layers → ~1.5M params)

---

## Low Risks

### 11. Hotbar One-Hot Sparsity

**Risk**: Hotbar changes are extremely rare (maybe once per 10 seconds). The hotbar
head has very few positive training samples.

**Mitigation**: CrossEntropy loss handles class imbalance naturally. High threshold
for hotbar activation. No hotbar change = no loss (masked).

### 12. Backward Movement Learning

**Risk**: v1 had no backward movement data. Even v2 recordings may have little
backward movement (players rarely walk backward).

**Mitigation**: Dedicate a recording session to walking backward. Small amount of
data is enough since the gesture (leaning back) is very distinct.

### 13. Screen Type Ambiguity

**Risk**: Different screen types (inventory, crafting table, chest, furnace) have
different cursor target layouts. Model may not distinguish them.

**Mitigation**: `screen_open_type` in game state could encode different screen types.
Start with basic inventory only. Add other screens as separate training batches later.

---

## Monitoring Checklist

During live use, watch for:

- [ ] False positive rate during idle (should be <5%)
- [ ] Look drift when holding head still (deadzone working?)
- [ ] Action sticking (held actions not releasing on tracking loss)
- [ ] Inventory cursor precision (can hit inventory slots?)
- [ ] Hotbar switching responsiveness
- [ ] Pipeline latency (frame-to-action time)
- [ ] WebSocket connection stability
- [ ] GPU memory usage (should be stable, no leaks)
