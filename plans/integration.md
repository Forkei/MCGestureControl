# Integration v2

## Overview

How the trained ControlTransformer v2 connects to the live camera pipeline
and controls Minecraft through MCCTP.

---

## Pipeline Flow

```
Camera (Android) → stream_client.py → MediaPipe Pose + WiLoR Hands
                                              │
                                              ▼
                                    ControlPolicy (inference wrapper)
                                    - Assembles 671-dim input
                                    - Runs transformer forward pass
                                    - Parses 5 head outputs
                                    - Post-processes (threshold, smooth, edge detect)
                                    - Mode switch (gameplay vs inventory)
                                              │
                                              ▼
                                    ControlBridge (MCCTP sender)
                                    - Converts ControlOutput → MCCTP commands
                                    - Edge detection (held start/stop, pulse fire)
                                    - Sends to Minecraft via WebSocket
                                              │
                                              ▼
                                    Minecraft (MCCTP Fabric mod)
```

---

## ControlOutput Data Structure

```python
@dataclass
class ControlOutput:
    """28-dim control vector parsed from model output."""

    # Gameplay (Action Head)
    move_forward: bool = False
    move_backward: bool = False
    strafe_left: bool = False
    strafe_right: bool = False
    sprint: bool = False
    sneak: bool = False
    jump: bool = False          # pulse
    attack: bool = False        # pulse
    use_item: bool = False

    # Look (Look Head)
    look_yaw: float = 0.0      # [-1, 1]
    look_pitch: float = 0.0    # [-1, 1]

    # Utility (Action Head)
    drop_item: bool = False     # pulse
    swap_offhand: bool = False  # pulse
    open_inventory: bool = False  # pulse

    # Hotbar (Hotbar Head)
    hotbar_slot: Optional[int] = None  # 0-8 or None (no change)

    # Inventory (Cursor + InvClick Heads)
    cursor_x: float = 0.0      # [0, 1]
    cursor_y: float = 0.0      # [0, 1]
    inv_left_click: bool = False  # pulse
    inv_right_click: bool = False  # pulse
    inv_shift_held: bool = False

    # Mode
    screen_open: bool = False   # from game state, determines active heads
```

---

## ControlPolicy (Inference Wrapper)

### `control_policy.py` — Key Methods

```python
class ControlPolicy:
    def __init__(self, model_path, config_path, device='cuda'):
        self.model = ControlTransformer(config).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.frame_buffer = deque(maxlen=30)   # sliding window
        self.action_history = deque(maxlen=5)   # last 5 outputs
        self.prev_output = np.zeros(28)         # for edge detection
        self.look_ema = np.zeros(2)             # smoothed look
        self.cursor_ema = np.zeros(2)           # smoothed cursor
        self.thresholds = config['thresholds']
        self.feature_mean = config['feature_mean']
        self.feature_std = config['feature_std']

    def push_frame(self, pose, hands, game_state):
        """Add new frame to buffer. Called every frame."""
        raw = assemble_raw_features(pose, hands)  # (260,)
        self.frame_buffer.append((raw, game_state))

    def predict(self) -> ControlOutput:
        """Run inference and return post-processed controls."""
        # 1. Build input tensor
        features = self.build_input()  # (1, 30, 671)

        # 2. Forward pass
        with torch.no_grad():
            outputs = self.model(features, mask)

        # 3. Parse outputs from 5 heads
        raw = self.parse_outputs(outputs)

        # 4. Post-process
        result = self.post_process(raw)

        # 5. Update action history
        self.action_history.append(raw)

        return result

    def build_input(self):
        """Assemble 671-dim input from frame buffer."""
        frames = list(self.frame_buffer)
        raw_features = [f[0] for f in frames]        # (N, 260)
        game_states = [f[1] for f in frames]          # (N, 46)

        # Velocity
        velocity = compute_velocity(raw_features)      # (N, 225)

        # Action history (140 dims) — same for all frames in window
        history = self.get_action_history()             # (140,)

        # Concatenate per frame
        per_frame = np.concatenate([
            raw_features,    # (N, 260)
            velocity,        # (N, 225)
            game_states,     # (N, 46)
            np.tile(history, (N, 1)),  # (N, 140)
        ], axis=1)  # (N, 671)

        # Normalize
        normalized = (per_frame - self.feature_mean) / self.feature_std

        return torch.tensor(normalized).unsqueeze(0).float()

    def get_action_history(self):
        """Get flattened action history (140 dims)."""
        history = list(self.action_history)
        # Pad with zeros if less than 5 entries
        while len(history) < 5:
            history.insert(0, np.zeros(28))
        return np.concatenate(history[-5:])  # (140,)
```

### Post-Processing Pipeline

```python
def post_process(self, raw_output) -> ControlOutput:
    """Apply thresholds, smoothing, edge detection."""
    out = ControlOutput()
    game_state = self.frame_buffer[-1][1]
    screen_open = game_state[38] > 0.5

    if not screen_open:
        # --- Gameplay Mode ---

        # Binary actions: apply per-control thresholds
        for i, name in enumerate(self.action_names):
            prob = sigmoid(raw_output['action_logits'][i])
            threshold = self.thresholds[name]
            setattr(out, name, prob > threshold)

        # Look: EMA smoothing + deadzone
        yaw = raw_output['look'][0]
        pitch = raw_output['look'][1]
        self.look_ema = 0.35 * np.array([yaw, pitch]) + 0.65 * self.look_ema
        if abs(self.look_ema[0]) < 0.08:
            self.look_ema[0] = 0.0
        if abs(self.look_ema[1]) < 0.08:
            self.look_ema[1] = 0.0
        out.look_yaw = self.look_ema[0]
        out.look_pitch = self.look_ema[1]

        # Hotbar: argmax if any slot > threshold
        hotbar_probs = softmax(raw_output['hotbar_logits'])
        max_prob = hotbar_probs.max()
        if max_prob > 0.5:  # confident enough to switch
            out.hotbar_slot = hotbar_probs.argmax()

    else:
        # --- Inventory Mode ---

        # Cursor: EMA smoothing
        cx = raw_output['cursor'][0]
        cy = raw_output['cursor'][1]
        self.cursor_ema = 0.35 * np.array([cx, cy]) + 0.65 * self.cursor_ema
        out.cursor_x = np.clip(self.cursor_ema[0], 0, 1)
        out.cursor_y = np.clip(self.cursor_ema[1], 0, 1)

        # Inventory clicks: threshold
        out.inv_left_click = sigmoid(raw_output['inv_logits'][0]) > 0.5
        out.inv_right_click = sigmoid(raw_output['inv_logits'][1]) > 0.5
        out.inv_shift_held = sigmoid(raw_output['inv_logits'][2]) > 0.5

    # open_inventory is always active (can open/close in any mode)
    out.open_inventory = sigmoid(raw_output['action_logits'][13]) > self.thresholds['open_inventory']
    out.screen_open = screen_open

    return out
```

---

## ControlBridge (MCCTP Sender)

### `control_bridge.py` — Key Logic

```python
class ControlBridge:
    def __init__(self, host='localhost', port=8080):
        self.client = SyncMCCTPClient(host=host, port=port)
        self.prev = ControlOutput()  # previous state for edge detection
        self.pulse_cooldown = {}     # prevent rapid re-fire

    def update(self, output: ControlOutput):
        """Send control changes to Minecraft."""

        if not output.screen_open:
            self._update_gameplay(output)
        else:
            self._update_inventory(output)

        self.prev = output

    def _update_gameplay(self, out: ControlOutput):
        """Handle gameplay controls."""

        # Movement (held actions — send on edge)
        for attr in ['move_forward', 'move_backward', 'strafe_left',
                      'strafe_right', 'sprint', 'sneak', 'use_item']:
            curr = getattr(out, attr)
            prev = getattr(self.prev, attr)
            if curr != prev:
                self._send_held(attr, curr)

        # Pulse actions (send once on rising edge)
        for attr in ['jump', 'attack', 'drop_item', 'swap_offhand', 'open_inventory']:
            curr = getattr(out, attr)
            prev = getattr(self.prev, attr)
            if curr and not prev:
                if self._check_cooldown(attr, 0.15):  # 150ms cooldown
                    self._send_pulse(attr)

        # Look (send every frame if non-zero)
        if abs(out.look_yaw) > 0.01 or abs(out.look_pitch) > 0.01:
            self.client.perform(Actions.look(
                yaw=out.look_yaw * LOOK_SENSITIVITY,
                pitch=out.look_pitch * LOOK_SENSITIVITY,
            ))

        # Hotbar (send on change)
        if out.hotbar_slot is not None and out.hotbar_slot != self.prev.hotbar_slot:
            self.client.perform(Actions.hotbar(slot=out.hotbar_slot))

    def _update_inventory(self, out: ControlOutput):
        """Handle inventory controls."""

        # Cursor movement (send every frame)
        self.client.perform(Actions.cursor_move(
            x=out.cursor_x,
            y=out.cursor_y,
        ))

        # Clicks (pulse on rising edge)
        if out.inv_left_click and not self.prev.inv_left_click:
            self.client.perform(Actions.inventory_click(
                button=0,
                shift=out.inv_shift_held,
            ))
        if out.inv_right_click and not self.prev.inv_right_click:
            self.client.perform(Actions.inventory_click(
                button=1,
                shift=out.inv_shift_held,
            ))

    def release_all(self):
        """Release all held actions. Call on disconnect or tracking loss."""
        self.client.perform(Actions.move(forward=False))
        self.client.perform(Actions.move(backward=False))
        self.client.perform(Actions.move(left=False))
        self.client.perform(Actions.move(right=False))
        self.client.perform(Actions.sprint(False))
        self.client.perform(Actions.sneak(False))
        self.client.perform(Actions.stop_use_item())
```

---

## stream_client.py Integration

### Main Loop Addition

```python
# After pose/hand tracking and gesture classification:

# Control policy inference (every frame)
if control_policy is not None:
    control_policy.push_frame(pose, hands, game_state)
    infer_counter += 1
    if infer_counter >= INFER_EVERY:
        infer_counter = 0
        control_output = control_policy.predict()

        # Send to Minecraft
        if control_bridge is not None and bridge_active:
            control_bridge.update(control_output)

        # Display overlay
        draw_control_overlay(frame, control_output)
```

### Tracking Loss Safety

```python
# When tracking is disabled or lost:
if not state["hand_tracking_enabled"] or not state["pose_detected"]:
    if control_bridge is not None:
        control_bridge.release_all()  # release all held actions
    if control_policy is not None:
        control_policy.clear()  # reset frame buffer + history
```

### Cleanup

```python
finally:
    if control_bridge is not None:
        control_bridge.release_all()
        control_bridge.disconnect()
```

---

## CLI Arguments

```bash
python client/stream_client.py <phone_ip> \
    --control-model client/models/control_policy.pt \
    --game --game-port 8080
```

| Arg | Default | Description |
|---|---|---|
| `--control-model` | None | Path to control policy model (.pt) |
| `--game` | False | Enable MCCTP connection |
| `--game-host` | localhost | MCCTP host |
| `--game-port` | 8080 | MCCTP port |

---

## Debug Overlay

When control policy is active, show overlay on camera feed:

```
┌──────────────────────────────┐
│ [W] FWD  [S] BWD  [Sprint]  │
│ [A] LEFT [D] RIGHT          │
│ [Jump] [Atk] [Use]          │
│ Look: ←0.3  ↑0.1            │
│ Hotbar: [1] 2  3  4  5      │
│ Mode: GAMEPLAY               │
│ Game: health=0.8 hunger=1.0  │
└──────────────────────────────┘
```

Active controls highlighted in green, inactive in gray.
Inventory mode shows cursor crosshair overlay instead.
