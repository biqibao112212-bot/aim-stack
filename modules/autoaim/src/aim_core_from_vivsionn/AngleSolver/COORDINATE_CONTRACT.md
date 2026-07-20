# Armor coordinate contract

Read this before changing `AngleSolver`, PnP, camera/gimbal calibration,
tracker inputs, or inverse projection.

## Frames and rigid transform

- OpenCV camera `C`: `+x right`, `+y down`, `+z forward`; PnP `tvec` is mm.
- calibrated gimbal `G`: `+x forward`, `+y left`, `+z up`.
- tracker/chassis `T`: `+x forward`, `+y left`, `+z up`.
- YAML stores `R_CAMERA2GIMBAL = ^G R_C` and
  `T_CAMERA2GIMBAL = ^G t_C` in metres. Both fields are mandatory when the
  extrinsic is enabled.
- `FrameMeta.poseEuler` is the exposure-matched optical pose. Its rotation
  `^T R_C` retains the explicit OpenCV-to-tracker mapping and optical pitch/yaw.
  It is not a local joint pose.

At one exposure, derive `^T R_G = ^T R_C (^G R_C)^T`, then use:

```text
p_T = ^T R_G (^G R_C p_C + ^G t_C)
p_C = (^G R_C)^T ((^T R_G)^T p_T - ^G t_C)
```

There is no empirical `H` or additional height bias. Point forward conversion,
single/batch inverse projection and hit-point rays call this same SE(3) pair.
Changing coordinate contract requires clearing tracker history; old/new points
must never coexist in one estimator state.

For the current simulator exact-exposure contract:

```text
R_CAMERA2GIMBAL = [[ 0,  0,  1],
                   [-1,  0,  0],
                   [ 0, -1,  0]]
T_CAMERA2GIMBAL = [0.25631080, 0.00183094, 0.09543117] m
```

The previous 25-degree asset-derived R/T is archived and invalid for the
current simulator. `calibrate_daedalus_calib_board.py` is a synthetic
self-consistency experiment and is forbidden from updating production YAML.

## Invariants and validation

1. Camera depth must remain tracker forward distance, never tracker height.
2. Candidate-0 camera-frame rvec/tvec and reprojection error are unchanged by
   downstream camera-to-tracker conversion.
3. R must be finite, orthonormal and `det(R)=+1`; T is finite and in metres.
   Missing/partial/invalid enabled calibration fails closed.
4. Armor yaw/tilt still uses the exposure optical rotation; translation never
   acts on normals or directions.
5. An image without its exact exposure optical pose is rejected before PnP.

Required checks:

```powershell
D:\Anaconda\envs\yolov8\python.exe -B -m training.stage3.validate_extrinsic `
  --manifest <formal-manifest> --evidence-root <formal-evidence> `
  --raw-root <raw-root> --extrinsic-yaml modules/autoaim/config/param.sim.yaml

wsl.exe -d Ubuntu-OSTEP -- bash -lc `
  "cd /mnt/d/仿真/repos/aim-stack/modules/autoaim && `
   cmake --build build/ros2_trt --target aim_angle_solver_pnp_candidates_test -j2 && `
   ctest --test-dir build/ros2_trt --output-on-failure -R aim_angle_solver_pnp_candidates_test"
```
