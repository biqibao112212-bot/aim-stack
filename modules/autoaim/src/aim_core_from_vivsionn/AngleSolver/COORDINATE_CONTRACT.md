# Armor coordinate contract

This file is mandatory reading before changing `AngleSolver.cpp`, `AngleSolver.h`,
camera/gimbal extrinsics, armor PnP, tracker inputs, or `calculateImagePoint()`.

## Frames

- OpenCV camera: `+x right`, `+y down`, `+z forward`; PnP `tvec` is millimetres.
- YPD tracker/planner: `+x forward`, `+y left`, `+z up`; armor positions are metres after division by 1000.
- At zero gimbal pose the required axis mapping is exactly:

  ```text
  tracker = [camera.z, -camera.x, -camera.y]
  ```

- Yaw and the inverse optical-pitch stabilization are applied before that explicit
  axis permutation by `cameraPointToTrackerConvention()`.
  `trackerPointToCameraConvention()` is its inverse. Do not change the pitch sign:
  applying optical pitch in the same direction creates a positive feedback loop
  where estimated height and commanded pitch grow together.
- The legacy `H` value is a camera/aiming vertical offset used when converting an
  armor point. It is not a claim that the vehicle centre is one fixed 3-D point.

## Non-negotiable invariants

1. A target approximately 3 m in front of the camera must have tracker `x≈3 m`;
   its depth must never appear as tracker `z≈-3 m`.
2. `calculateGimblePoint*()` and `calculateImagePoint()` must remain inverse-frame
   operations with the same pose and height-offset semantics.
3. Do not replace the explicit permutation with `R_camera2gimbal` merely because
   a matrix is available. Calibration matrices and tracker-axis conventions are
   different contracts. A replacement requires exact exposure truth plus live
   static-target A/B evidence.
4. Do not tune tracker, planner, pitch signs, offsets, or ballistic parameters to
   compensate for a failed coordinate test.
5. PnP candidate enumeration may change pose-candidate metadata, but candidate 0
   must preserve the legacy selected `tvec`, and no candidate change may alter the
   camera-to-tracker axis contract.
6. The solver needs the exposure-time optical camera/gimbal pose. It is not the
   same as `RuntimeState.gimbal_pitch_rad`: the simulated camera has a fixed
   25-degree mount rotation. For example, optical `+3.78 deg` corresponds to local
   joint about `-28.78 deg`. Do not substitute the joint angle for the optical pose.
7. Command conversion is the inverse contract. With the current 25-degree mount,
   zero optical pitch is encoded as UDP/Talos neutral `65 deg`; positive optical
   pitch subtracts from that encoded value. Do not restore `90 + optical_pitch`.
8. Armor outward normal polarity is unique in this project. Do not add a synthetic
   reversed-normal hypothesis to production telemetry or tracking.
9. If an image cannot be paired with its exposure-time optical gimbal pose, drop
   that observation before PnP/tracker update. Never substitute the current local
   joint pose for an older image; hold the previous valid command/state until a
   contract-complete exposure arrives.
10. The ordinary-armor `+15 degree` pitch tilt is fixed in tracker/chassis
    coordinates. Build the armor yaw/tilt rotation there, then project it through
    the exposure-matched gimbal pose into the OpenCV camera frame. Production
    `Armor::yaw` and `Armor::yaw_absolute` are tracker/chassis yaw; a camera-fixed
    value may exist only as explicitly labelled diagnostic telemetry.

## Required validation

Run the focused test after every relevant change:

```powershell
wsl ./aim_sim_bridge/build/ros2_trt/aim_angle_solver_pnp_candidates_test
```

The test includes a captured failure sample:

```text
camera tvec after H = [111.802147, 175.895243, 3640.185614] mm
gimbal yaw/pitch    = [-21.48591232, -0.9220832586] deg
expected tracker    = [3430.365912, 1230.131067, -117.292072] mm
```

The broken 2026-07-10 path produced approximately
`[584, -301, -3511] mm`, causing a FireControl pitch near `-79.21 deg`.
That output is forbidden even if reprojection error is small.

After the unit test, run a fresh static 3 m simulator row. Acceptance requires:

- solved armor position dominated by positive tracker `x`;
- `abs(z)` small compared with `x`;
- commanded pitch near the image/ballistic expectation, never tens of degrees
  solely because camera depth entered tracker `z`;
- target remains in view long enough to prove closed-loop centring.
