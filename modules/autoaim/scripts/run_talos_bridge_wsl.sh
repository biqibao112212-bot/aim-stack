#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_MODE="${1:-armor}"
WITH_TRT="${AIM_SIM_WITH_VIVSIONN_TRT:-ON}"
FORCE_REBUILD="${AIM_SIM_FORCE_REBUILD:-0}"
ENABLE_UDP="${AIM_SIM_ENABLE_UDP:-ON}"
ENABLE_FIRE="${AIM_SIM_ENABLE_FIRE:-true}"
DUAL_FOCAL="${AIM_SIM_DUAL_FOCAL:-true}"
WIDE_FOCAL_MM="${AIM_SIM_WIDE_FOCAL_MM:-6.0}"
PRECISION_FOCAL_MM="${AIM_SIM_PRECISION_FOCAL_MM:-16.0}"
ARMOR_ENGINE="${AIM_SIM_ARMOR_ENGINE:-${ROOT_DIR}/models/armor.engine}"
IPC_DIR="${TALOS_IPC_DIR:-${ROOT_DIR}/../talos-ipc}"
BULLET_SPEED_MPS="${AIM_SIM_BULLET_SPEED_MPS:-22.0}"
SCENE_CONTROL_MODE="${AIM_SIM_SCENE_CONTROL_MODE:-off}"
SCENE_CONTROL_HOST="${AIM_SIM_SCENE_CONTROL_HOST:-}"

if [[ -f "${ROOT_DIR}/build/debug/force_fire_off" ]]; then
  ENABLE_FIRE=false
fi

case "${WITH_TRT^^}" in
  ON|1|TRUE|YES)
    BUILD_DIR="${ROOT_DIR}/build/ros2_trt"
    ;;
  *)
    BUILD_DIR="${ROOT_DIR}/build/ros2"
    ;;
esac

BRIDGE_BIN="${BUILD_DIR}/aim_sim_talos_auto_aim_bridge"
case "${FORCE_REBUILD^^}" in
  ON|1|TRUE|YES)
    SHOULD_BUILD=1
    ;;
  *)
    SHOULD_BUILD=0
    ;;
esac

if [[ "${SCENE_CONTROL_MODE}" == shooting_range_g2* &&
      ! -x "${BUILD_DIR}/aim_sim_scene_control_cli" ]]; then
  SHOULD_BUILD=1
fi

if [[ "${SHOULD_BUILD}" == "1" || ! -x "${BRIDGE_BIN}" ]]; then
  AIM_SIM_WITH_VIVSIONN_TRT="${WITH_TRT}" "${ROOT_DIR}/scripts/build_wsl.sh"
else
  echo "Using existing bridge binary: ${BRIDGE_BIN}"
fi

if [[ "${SCENE_CONTROL_MODE}" == shooting_range_g2* ]]; then
  SCENE_CONTROL_BIN="${BUILD_DIR}/aim_sim_scene_control_cli"
  if [[ ! -x "${SCENE_CONTROL_BIN}" ]]; then
    echo "Missing SDK scene-control CLI: ${SCENE_CONTROL_BIN}" >&2
    exit 2
  fi
  if [[ -z "${SCENE_CONTROL_HOST}" ]]; then
    echo "AIM_SIM_SCENE_CONTROL_HOST is required for WSL scene control" >&2
    exit 2
  fi
  if [[ "${SCENE_CONTROL_MODE}" == "shooting_range_g2_stationary" ]]; then
    scene_control_args=(
      --stage3 --host "${SCENE_CONTROL_HOST}" --session aim-b-g2-stationary
      --target 3 --mode stationary --direction-deg 90
      --linear-speed-mps 0 --linear-span-m 0 --spin-deg-s 0
    )
  else
    scene_control_args=(--host "${SCENE_CONTROL_HOST}")
  fi
  if ! "${SCENE_CONTROL_BIN}" "${scene_control_args[@]}"; then
    echo "shooting_range Scene Control ACK did not arrive" >&2
    exit 2
  fi
fi

mkdir -p "${IPC_DIR}"
mkdir -p "${ROOT_DIR}/build/debug"
export TALOS_IPC_DIR="${IPC_DIR}"
export AIM_SIM_PARAM_YAML="${ROOT_DIR}/config/param.sim.yaml"
export AIM_SIM_DEBUG_BRIDGE_JSON="${AIM_SIM_DEBUG_BRIDGE_JSON:-${ROOT_DIR}/build/debug/aim_bridge.json}"
case "${AIM_SIM_DEBUG_PIPELINE_JSON:-OFF}" in
  OFF|'') unset AIM_SIM_DEBUG_PIPELINE_JSON ;;
  *) export AIM_SIM_DEBUG_PIPELINE_JSON ;;
esac

BRIDGE_ARGV0="${BRIDGE_BIN}"
if [[ -n "${DAEDALUS_BRIDGE_TOKEN:-}" ]]; then
  BRIDGE_ARGV0="aim_sim_talos_auto_aim_bridge_${DAEDALUS_BRIDGE_TOKEN}"
fi

exec -a "${BRIDGE_ARGV0}" "${BRIDGE_BIN}" \
  --mode "${TARGET_MODE}" \
  --ipc-dir "${IPC_DIR}" \
  --param-yaml "${ROOT_DIR}/config/param.sim.yaml" \
  --armor-engine "${ARMOR_ENGINE}" \
  --bullet-speed "${BULLET_SPEED_MPS}" \
  --talos-pitch-neutral 65.0 \
  --enable-udp "${ENABLE_UDP}" \
  --enable-fire "${ENABLE_FIRE}" \
  --dual-focal "${DUAL_FOCAL}" \
  --wide-focal-mm "${WIDE_FOCAL_MM}" \
  --precision-focal-mm "${PRECISION_FOCAL_MM}"
