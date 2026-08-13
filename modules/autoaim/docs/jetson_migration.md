# Jetson Migration Notes

Keep reusable code in:

- [`src/aim_core_from_vivsionn`](../src/aim_core_from_vivsionn)
- [`vivsionn_pipeline.cpp`](../src/aim_core_bridge/vivsionn_pipeline.cpp)

Keep simulator-only code out of Jetson builds:

- [`src/sim_adapter`](../src/sim_adapter)
- ROS2 topic names
- `sim_pitch_neutral_deg`
- Daedalus-specific launch scripts

Recommended Jetson build:

```bash
cmake -S . -B build/jetson \
  -DAIM_SIM_WITH_ROS2=OFF \
  -DAIM_SIM_WITH_VIVSIONN_TRT=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/jetson --parallel "$(nproc)"
```

The Jetson adapter should provide the same `SimFrame` information from real camera, gimbal feedback, bullet speed, and task mode, then forward `AimCommand` to the original CAN/serial layer.

Do not move real robot CAN, referee, camera SDK, or startup scripts into `aim_core_bridge`; those belong in a separate robot adapter.

