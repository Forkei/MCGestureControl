# MCGestureControl

Control Minecraft with body gestures captured via phone camera. Uses a Transformer-based control policy that maps body pose + hand landmarks to 28-dimensional game inputs in real-time.

## How It Works

1. **Phone camera** captures body pose (MediaPipe) and hand landmarks (WiLoR-mini)
2. **Control policy** (ControlTransformerV2) predicts game inputs from 671-dim feature vectors
3. **MCCTP mod** receives controls via WebSocket and executes them in Minecraft

## Training Data Pipeline

### Phase 1: Record Gameplay
Record yourself playing Minecraft normally with keyboard/mouse. Captures screen video + all game state/inputs from the MCCTP mod at 30fps.

```bash
python client/gameplay_recorder.py --port 8765
# F9 = start/stop (5s countdown), F10 = quit
# Works while Minecraft has focus
```

### Phase 2: Record Gestures
Watch the recorded gameplay video while performing the actions with your body. The phone camera captures your pose synced to each video frame.

```bash
python client/gameplay_playback.py <phone_ip> recordings_gameplay/session_xxx.mp4
```

### Phase 3: Train
Train the control policy on paired (pose, controls) data.

```bash
python client/train_controls.py
```

## Viewing Recordings

Review recorded sessions with a control overlay showing which keys were pressed:

```bash
python client/gameplay_viewer.py recordings_gameplay/session_xxx
# SPACE=pause, S=speed, arrows=skip, Q=quit
```

## Architecture

- **Input**: 671 dims (260 raw pose/hand + 225 velocity + 46 game state + 140 action history)
- **Output**: 5 heads — Action(12), Look(2), Hotbar(9), Cursor(2), InvClick(3)
- **Model**: ControlTransformerV2, d_model=256, 6 layers, 8 heads (~3.5M params)
- **Controls**: 28-dim vector covering movement, combat, look, hotbar, inventory

## Requirements

- Python 3.10+
- [MCCTP](https://github.com/lucasoyen/mcctp) Fabric mod installed in Minecraft
- Phone running the [GestureCam](https://github.com/Forkei/GestureCam) Android app
- `mss`, `opencv-python`, `numpy`, `torch`, `mediapipe`, `mcctp`
- `ffmpeg` on PATH (for gameplay recording)

## Key Files

| File | Purpose |
|------|---------|
| `control_model.py` | ControlTransformerV2 model definition |
| `control_dataset.py` | Constants, data loading, head groupings |
| `control_policy.py` | Game state encoding, inference wrapper |
| `control_recorder.py` | MCCTP control capture (28-dim) |
| `control_bridge.py` | Send controls to Minecraft via MCCTP |
| `gameplay_recorder.py` | Screen + MCCTP recorder (F9/F10 hotkeys) |
| `gameplay_playback.py` | Video playback + pose capture |
| `gameplay_viewer.py` | Session viewer with control overlay |
| `train_controls.py` | Training loop with mode-aware masked loss |
