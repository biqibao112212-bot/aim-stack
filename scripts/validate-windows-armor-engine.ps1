[CmdletBinding()]
param(
  [string]$EnginePath,
  [string]$TensorRtRoot,
  [string]$CudnnBin,
  [int]$Warmup = 30,
  [int]$Samples = 300
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if (-not $EnginePath) {
  $EnginePath = Join-Path $workspace 'models\engines\windows\armor-0708-trt861-win-rtx4060-fp16.engine'
}
if (-not $TensorRtRoot) {
  $TensorRtRoot = Join-Path $workspace 'runtime\tool-cache\tensorrt-8.6.1.6-windows-cuda11.8\package\TensorRT-8.6.1.6'
}
if (-not $CudnnBin) {
  $CudnnBin = Join-Path $workspace 'runtime\tool-cache\cudnn-8.9.6.50-windows-cuda11\package\cudnn-windows-x86_64-8.9.6.50_cuda11-archive\bin'
}

$python = 'D:\Anaconda\envs\yolov8\python.exe'
$cudaBin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin'
$trtLib = Join-Path $TensorRtRoot 'lib'
$torchLib = 'D:\Anaconda\envs\yolov8\Lib\site-packages\torch\lib'
foreach ($path in @($python, $EnginePath, (Join-Path $trtLib 'nvinfer.dll'), (Join-Path $CudnnBin 'cudnn64_8.dll'))) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Required validation input is absent: $path"
  }
}

$env:PATH = "$torchLib;$trtLib;$CudnnBin;$cudaBin;$env:PATH"
$validator = @'
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

parser = argparse.ArgumentParser()
parser.add_argument("--engine", required=True)
parser.add_argument("--warmup", type=int, required=True)
parser.add_argument("--samples", type=int, required=True)
args = parser.parse_args()

import tensorrt as trt
import torch

engine_path = Path(args.engine)
report_path = engine_path.with_suffix(engine_path.suffix + ".benchmark.json")
if report_path.exists():
    raise RuntimeError(f"Refusing to overwrite existing report: {report_path}")
logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise RuntimeError("deserialize_cuda_engine returned None")
context = engine.create_execution_context()
if context is None:
    raise RuntimeError("create_execution_context returned None")

inputs = []
outputs = []
for index in range(engine.num_io_tensors):
    name = engine.get_tensor_name(index)
    record = (name, tuple(int(v) for v in engine.get_tensor_shape(name)))
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        inputs.append(record)
    else:
        outputs.append(record)
if inputs != [("images", (1, 3, 640, 640))] or outputs != [("output", (1, 25200, 22))]:
    raise RuntimeError(f"Unexpected engine I/O: inputs={inputs}, outputs={outputs}")

input_tensor = torch.zeros(inputs[0][1], device="cuda", dtype=torch.float32)
output_tensor = torch.empty(outputs[0][1], device="cuda", dtype=torch.float32)
if not context.set_tensor_address(inputs[0][0], input_tensor.data_ptr()):
    raise RuntimeError("Failed to bind input tensor")
if not context.set_tensor_address(outputs[0][0], output_tensor.data_ptr()):
    raise RuntimeError("Failed to bind output tensor")
stream = torch.cuda.current_stream().cuda_stream
for _ in range(args.warmup):
    if not context.execute_async_v3(stream):
        raise RuntimeError("Warmup enqueue failed")
torch.cuda.synchronize()

durations_ms = []
for _ in range(args.samples):
    start = time.perf_counter_ns()
    if not context.execute_async_v3(stream):
        raise RuntimeError("Measured enqueue failed")
    torch.cuda.synchronize()
    durations_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
if not bool(torch.isfinite(output_tensor).all().item()):
    raise RuntimeError("Engine produced non-finite output for an all-zero input")

ordered = sorted(durations_ms)
def percentile(p):
    return ordered[round((len(ordered) - 1) * p)]
report = {
    "schema_version": 1,
    "kind": "windows_tensorrt_engine_execution_benchmark",
    "engine": str(engine_path),
    "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
    "tensorrt_version": trt.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "warmup_iterations": args.warmup,
    "measured_iterations": args.samples,
    "input": {"name": inputs[0][0], "shape": list(inputs[0][1]), "dtype": "float32"},
    "output": {"name": outputs[0][0], "shape": list(outputs[0][1]), "dtype": "float32"},
    "synchronization": "torch.cuda.synchronize after every enqueue; latency includes enqueue and device completion",
    "latency_ms": {
        "min": min(durations_ms),
        "median": statistics.median(durations_ms),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(durations_ms),
    },
    "throughput_fps_from_median": 1000.0 / statistics.median(durations_ms),
    "validation": "fresh-process deserialization, 30 warmups, finite-output check, and measured CUDA execution succeeded",
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
'@

$validator | & $python - --engine $EnginePath --warmup $Warmup --samples $Samples
if ($LASTEXITCODE -ne 0) {
  throw "TensorRT engine validation failed with exit code $LASTEXITCODE"
}
