#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_MODE="${1:-armor}"
WITH_TRT="${AIM_SIM_WITH_VIVSIONN_TRT:-OFF}"
ARMOR_ENGINE="${AIM_SIM_ARMOR_ENGINE:-${ROOT_DIR}/models/armor.engine}"
case "${WITH_TRT^^}" in
  ON|1|TRUE|YES)
    BUILD_DIR="${ROOT_DIR}/build/ros2_trt"
    ;;
  *)
    BUILD_DIR="${ROOT_DIR}/build/ros2"
    ;;
esac

"${ROOT_DIR}/scripts/build_wsl.sh"

set +u
source /opt/ros/humble/setup.bash
if [ -f /home/potato/daedalus_ros2_ws/install/setup.bash ]; then
  source /home/potato/daedalus_ros2_ws/install/setup.bash
fi
set -u

export AIM_SIM_PARAM_YAML="${ROOT_DIR}/config/param.sim.yaml"

exec "${BUILD_DIR}/aim_sim_energy_buff_node" --ros-args \
  -p target_mode:="${TARGET_MODE}" \
  -p param_yaml:="${ROOT_DIR}/config/param.sim.yaml" \
  -p buff_config_path:="${ROOT_DIR}/config/buff_config.sim.yaml" \
  -p bullet_speed_mps:=22.0 \
  -p sim_pitch_neutral_deg:=65.0
