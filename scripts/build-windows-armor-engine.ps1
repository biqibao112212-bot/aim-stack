[CmdletBinding()]
param(
  [string]$OnnxPath,
  [string]$TensorRtRoot,
  [string]$CudnnBin,
  [string]$OutputPath,
  [ValidateSet('fp16', 'fp32')]
  [string]$Precision = 'fp16',
  [int]$WorkspaceMiB = 2048
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if (-not $OnnxPath) {
  $OnnxPath = Join-Path $workspace 'aim_sim_bridge\third_party\RobotDetectionModel\Model\0708.onnx'
}
if (-not $TensorRtRoot) {
  $TensorRtRoot = Join-Path $workspace 'runtime\tool-cache\tensorrt-8.6.1.6-windows-cuda11.8\package\TensorRT-8.6.1.6'
}
if (-not $CudnnBin) {
  $CudnnBin = Join-Path $workspace 'runtime\tool-cache\cudnn-8.9.6.50-windows-cuda11\package\cudnn-windows-x86_64-8.9.6.50_cuda11-archive\bin'
}
if (-not $OutputPath) {
  $OutputPath = Join-Path $workspace "models\engines\windows\armor-0708-trt861-win-rtx4060-$Precision.engine"
}
$python = 'D:\Anaconda\envs\yolov8\python.exe'
$cudaBin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin'
$trtLib = Join-Path $TensorRtRoot 'lib'
$torchLib = 'D:\Anaconda\envs\yolov8\Lib\site-packages\torch\lib'

foreach ($path in @($python, $OnnxPath, $trtLib, (Join-Path $trtLib 'nvinfer.dll'), $CudnnBin, (Join-Path $CudnnBin 'cudnn64_8.dll'))) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Required build input is absent: $path"
  }
}
if (Test-Path -LiteralPath $OutputPath) {
  throw "Refusing to overwrite protected engine: $OutputPath"
}
New-Item -ItemType Directory -Path (Split-Path -Parent $OutputPath) -Force | Out-Null

$env:PATH = "$trtLib;$CudnnBin;$cudaBin;$torchLib;$env:PATH"
$builder = @'
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--onnx", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--workspace-mib", type=int, required=True)
parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
args = parser.parse_args()

import tensorrt as trt

onnx_path = Path(args.onnx)
output_path = Path(args.output)
partial_path = output_path.with_suffix(output_path.suffix + ".partial")
meta_path = output_path.with_suffix(output_path.suffix + ".json")
if output_path.exists() or partial_path.exists() or meta_path.exists():
    raise RuntimeError(f"Refusing to overwrite existing protected artifact: {output_path}")

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
onnx_parser = trt.OnnxParser(network, logger)
staging_dir = Path(tempfile.mkdtemp(prefix="aim-engine-"))
staged_onnx = staging_dir / "model.onnx"
try:
    shutil.copy2(onnx_path, staged_onnx)
    parsed = onnx_parser.parse_from_file(str(staged_onnx))
    if not parsed:
        details = "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{details}")
finally:
    shutil.rmtree(staging_dir, ignore_errors=True)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024)
if args.precision == "fp16":
    if not builder.platform_has_fast_fp16:
        raise RuntimeError("The active Windows GPU does not report fast FP16 support")
    config.set_flag(trt.BuilderFlag.FP16)
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError("TensorRT returned no serialized engine")
with partial_path.open("wb") as f:
    f.write(bytes(serialized))
partial_path.replace(output_path)

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(output_path.read_bytes())
if engine is None:
    raise RuntimeError("Post-build deserialization failed")

def dims(tensor_name):
    shape = engine.get_tensor_shape(tensor_name)
    return [int(v) for v in shape]

io = []
for index in range(engine.num_io_tensors):
    name = engine.get_tensor_name(index)
    io.append({
        "name": name,
        "mode": str(engine.get_tensor_mode(name)),
        "shape": dims(name),
        "dtype": str(engine.get_tensor_dtype(name)),
    })

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
    text=True, capture_output=True, check=False
)
metadata = {
    "schema_version": 1,
    "kind": "windows_tensorrt_engine",
    "engine": str(output_path),
    "engine_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    "engine_bytes": output_path.stat().st_size,
    "onnx": str(onnx_path),
    "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
    "onnx_bytes": onnx_path.stat().st_size,
    "tensorrt_version": trt.__version__,
    "precision": args.precision,
    "workspace_mib": args.workspace_mib,
    "io_tensors": io,
    "gpu": gpu.stdout.strip(),
    "validation": "deserialize_cuda_engine succeeded on the generation host",
}
meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metadata, ensure_ascii=False, indent=2))
'@

$builder | & $python - --onnx $OnnxPath --output $OutputPath --workspace-mib $WorkspaceMiB --precision $Precision
if ($LASTEXITCODE -ne 0) {
  throw "TensorRT engine generation failed with exit code $LASTEXITCODE"
}
