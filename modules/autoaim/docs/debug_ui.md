# Aim Debug UI

This debug UI is the first diagnostic gate for integrated auto-aim. Use it before
changing yaw/pitch offsets, FireControl, PnP object points, or detector corner
order.

## Start

From WSL:

```bash
cd /mnt/d/仿真/aim_sim_bridge
python3 scripts/serve_debug_ui.py --host 0.0.0.0 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

The integrated bridge writes telemetry here by default:

```text
build/debug/aim_bridge.json
build/debug/aim_pipeline.json
```

## Normal Integrated Run

1. Start the debug UI.
2. Start `Daedalus Integrated AutoAim.lnk`.
3. Press `F5` in the simulator.
4. Keep a visible armor target in the first-person view.
5. Hold `Space`.
6. Read the UI before changing code.

## First Checks

Check these in order:

1. Gimbal feedback: Talos global yaw, chassis yaw, local yaw, pitch.
2. Detector: detection count, first target center, vertex order.
3. PnP: `rVec`, `tVec`, position, `ypd`, selected armor type.
4. Tracker: detected flag, tracker state, update state, match ids.
5. FireControl: command yaw/pitch, shot mode, fire advice.
6. Transport: UDP command and Talos command must be interpreted separately.

The active integrated path uses UDP command output. Talos command-slot output is
kept for compatibility and uses a different pitch sign convention, so do not
mix conclusions across those two command paths.
