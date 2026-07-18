#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRTEXEC="${AIM_SIM_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX_PATH="${AIM_SIM_ARMOR_ONNX:-${ROOT_DIR}/third_party/RobotDetectionModel/Model/0526.onnx}"
ENGINE_PATH="${AIM_SIM_ARMOR_ENGINE:-${ROOT_DIR}/models/armor.engine}"
LOG_PATH="${AIM_SIM_TRT_LOG:-${ROOT_DIR}/build/armor_0526_fp16_trtexec.log}"
EXTRA_ARGS=()

if [ -n "${AIM_SIM_TRT_SHAPES:-}" ]; then
  EXTRA_ARGS+=(--shapes="${AIM_SIM_TRT_SHAPES}")
fi

if [ -n "${AIM_SIM_TRT_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(${AIM_SIM_TRT_EXTRA_ARGS})
fi

if [ ! -x "${TRTEXEC}" ]; then
  echo "trtexec not found or not executable: ${TRTEXEC}" >&2
  exit 1
fi

if [ ! -f "${ONNX_PATH}" ]; then
  echo "ONNX model not found: ${ONNX_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${ENGINE_PATH}")" "$(dirname "${LOG_PATH}")"

echo "ONNX:   ${ONNX_PATH}"
echo "ENGINE: ${ENGINE_PATH}"
echo "LOG:    ${LOG_PATH}"

"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --fp16 \
  --saveEngine="${ENGINE_PATH}" \
  --skipInference \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"

echo "Wrote ${ENGINE_PATH}"
