# Energy-buff branch scope

This branch builds the small/big energy-buff feature only. Its implementation
owns `BuffDetector`, the buff tracker/solver/aimer, and its feature-specific
Talos/ROS entrypoints.

The second-order MPC is carried under `src/common/control` as a neutral control
utility required by the buff aimer. This branch intentionally contains no
armor detector, AngleSolver, robot estimator, FireControl implementation or
armor/outpost selector.

The generated targets are `aim_sim_energy_buff_core`,
`aim_sim_talos_energy_buff_bridge`, and `aim_sim_energy_buff_node`.
