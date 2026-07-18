#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_TRT="${AIM_SIM_WITH_VIVSIONN_TRT:-OFF}"
WITH_ROS2="${AIM_SIM_WITH_ROS2:-OFF}"
case "${WITH_TRT^^}" in
  ON|1|TRUE|YES)
    WITH_TRT=ON
    BUILD_DIR="${ROOT_DIR}/build/ros2_trt"
    ;;
  *)
    WITH_TRT=OFF
    BUILD_DIR="${ROOT_DIR}/build/ros2"
    ;;
esac

set +u
source /opt/ros/humble/setup.bash
if [ -f /home/potato/daedalus_ros2_ws/install/setup.bash ]; then
  source /home/potato/daedalus_ros2_ws/install/setup.bash
fi
set -u

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DAIM_SIM_WITH_ROS2="${WITH_ROS2}" \
  -DAIM_SIM_WITH_VIVSIONN_TRT="${WITH_TRT}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
