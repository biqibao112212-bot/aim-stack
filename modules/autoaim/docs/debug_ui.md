# Aim Debug UI

This debug UI is the first diagnostic gate for integrated auto-aim. Use it before
changing yaw/pitch offsets, FireControl, PnP object points, or detector corner
order.

## Start

From the repository root on Linux:

```bash
python3 ./modules/autoaim/scripts/serve_debug_ui.py \
  --bridge-json <evidence-root>/bridge.json \
  --pipeline-json <evidence-root>/pipeline.json
```

On Windows:

```powershell
python .\modules\autoaim\scripts\serve_debug_ui.py `
  --bridge-json <evidence-root>\bridge.json `
  --pipeline-json <evidence-root>\pipeline.json
```

Then open [http://127.0.0.1:8765/](http://127.0.0.1:8765/). This address is
available only while the local debug server is running.

The normal project launchers write telemetry to the selected evidence root:

```text
<evidence-root>/bridge.json
<evidence-root>/pipeline.json
```

On Linux, the server can inspect retained or copied telemetry. Live telemetry
still comes from the currently locked Windows runtime until a matching Linux
Release and SDK are published.

## Normal Integrated Run

1. Start the debug UI.
2. For live telemetry, start the current auto-aim launcher on the Windows host.
3. Keep a visible armor target in the first-person view.
4. Read the UI before changing code.

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
