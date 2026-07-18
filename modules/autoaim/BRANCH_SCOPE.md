# Auto-aim armor branch scope

This branch builds the armor/outpost feature only. Its implementation owns the
armor TensorRT detector, PnP/AngleSolver, robot estimator, FireControl and the
armor-specific Talos/ROS entrypoints.

It intentionally contains no `BuffDetector`, buff configuration, buff tests or
buff runtime selector. The generated targets are `aim_sim_auto_aim_armor_core`,
`aim_sim_talos_auto_aim_bridge`, and `aim_sim_auto_aim_node`.

The frame, command, provenance and transport fields are retained as a simulator
contract; feature behavior must not be reintroduced through a runtime mode that
links both implementations.
