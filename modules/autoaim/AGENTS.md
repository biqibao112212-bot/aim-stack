# Agent instructions for aim_sim_bridge

Before modifying armor PnP, camera/gimbal transforms, `Armor::armorPosition`,
tracker state coordinates, projection, FireControl aim points, or simulator pose
adapters, read:

`src/aim_core_from_vivsionn/AngleSolver/COORDINATE_CONTRACT.md`

The coordinate contract and its focused regression test are mandatory. Do not
compensate for coordinate failures by tuning tracker, planner, pitch signs,
ballistics, or empirical offsets. Preserve candidate-0 legacy PnP behavior and
validate camera-depth-to-tracker-forward mapping before live testing.

For Talos input, the solver uses the exposure-time optical pose. The local joint
from `RuntimeState` is a fallback/diagnostic and differs because of the fixed
camera mount. Read the coordinate contract before changing pose priority or the
65-degree optical-neutral command conversion.
