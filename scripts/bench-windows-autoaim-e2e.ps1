[CmdletBinding()]
param(
  [int]$WarmupSeconds = 5,
  [int]$DurationSeconds = 30,
  [switch]$EnableStage3,
  [string]$EvidenceRoot
)

$ErrorActionPreference = 'Stop'
if ($WarmupSeconds -lt 0 -or $DurationSeconds -lt 5) {
  throw 'WarmupSeconds must be nonnegative and DurationSeconds must be at least 5.'
}

function Stop-ProcessTree([int]$RootPid) {
  $children = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ParentProcessId -eq $RootPid })
  foreach ($child in $children) { Stop-ProcessTree ([int]$child.ProcessId) }
  Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
}

function Wait-ForTcpPort([int]$Port, [int]$TimeoutSeconds) {
  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
  while ([DateTime]::UtcNow -lt $deadline) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
      $task = $client.ConnectAsync('127.0.0.1', $Port)
      if ($task.Wait(250) -and $client.Connected) { return }
    } catch { } finally { $client.Dispose() }
    Start-Sleep -Milliseconds 100
  }
  throw "Timed out waiting for TCP port $Port."
}

$repo = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $repo)
$lock = Get-Content -LiteralPath (Join-Path $repo 'simulator.lock.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$release = Join-Path $workspace $lock.simulator.release_relative_to_workspace
$binary = Join-Path $release 'bin\daedalus.exe'
$bridge = 'C:\codex-autoaim-build-ninja-cuda118d\aim_sim_windows_auto_aim_bridge.exe'
$sceneControl = 'C:\codex-autoaim-build-ninja-cuda118d\aim_sim_scene_control_cli.exe'
$engine = Join-Path $workspace 'models\engines\windows\armor-0708-trt861-win-rtx4060-fp16.engine'
$params = Join-Path $repo 'modules\autoaim\config\param.sim.yaml'
$python = 'D:\Anaconda\envs\yolov8\python.exe'
foreach ($path in @($binary, $bridge, $sceneControl, $engine, $params, $python)) {
  if (-not (Test-Path -LiteralPath $path)) { throw "Required end-to-end input is absent: $path" }
}
if (@(Get-NetTCPConnection -LocalPort $lock.simulator.tcp_image_port -ErrorAction SilentlyContinue).Count -gt 0) {
  throw "TCP port $($lock.simulator.tcp_image_port) is already occupied. Refusing to disturb an existing run."
}
if (-not $EvidenceRoot) {
  $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
  $suffix = if ($EnableStage3) { 'stage3' } else { 'runtime' }
  $EvidenceRoot = Join-Path $workspace "runtime\windows-autoaim-e2e-$suffix-$stamp"
}
if (Test-Path -LiteralPath $EvidenceRoot) { throw "Evidence root already exists: $EvidenceRoot" }
$ipc = Join-Path $EvidenceRoot 'ipc'
New-Item -ItemType Directory -Force -Path $ipc | Out-Null

$trt = Join-Path $workspace 'runtime\tool-cache\tensorrt-8.6.1.6-windows-cuda11.8\package\TensorRT-8.6.1.6\lib'
$cudnn = Join-Path $workspace 'runtime\tool-cache\cudnn-8.9.6.50-windows-cuda11\package\cudnn-windows-x86_64-8.9.6.50_cuda11-archive\bin'
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin'
$opencv = 'D:\OpenCV\build\x64\vc15\bin'
$env:BEVY_ASSET_ROOT = $release
$env:TALOS_IPC_DIR = $ipc
$env:WGPU_BACKEND = 'dx12'
$env:WGPU_POWER_PREF = 'high'
$env:DAEDALUS_PERF_DISABLE_UI = '1'
$env:DAEDALUS_CONFIG = 'config.performance.toml'
$env:DAEDALUS_TALOS_RGB_ONLY = '1'
$env:DAEDALUS_TALOS_CAPTURE_MAX_HZ = '200'
$env:DAEDALUS_TALOS_IMAGE_TRANSPORT = 'tcp'
$env:DAEDALUS_STATS_JSON = Join-Path $EvidenceRoot 'simulator.stats.json'
$env:AIM_SIM_RANGE_TARGET_NUMBER = '3'
$env:AIM_SIM_RANGE_SPIN_DEG_S = '114.59156'
$env:PATH = "$trt;$cudnn;$cuda;$opencv;$(Join-Path $release 'bin');$env:PATH"
$stage3 = Join-Path $EvidenceRoot 'stage3_observations.jsonl'
if ($EnableStage3) {
  $env:AIM_SIM_STAGE3_OBSERVATIONS = $stage3
  $env:AIM_SIM_STAGE3_SESSION_ID = "windows-e2e-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))"
} else {
  Remove-Item Env:AIM_SIM_STAGE3_OBSERVATIONS -ErrorAction SilentlyContinue
  Remove-Item Env:AIM_SIM_STAGE3_SESSION_ID -ErrorAction SilentlyContinue
}

$simulator = $null
try {
  $simulator = Start-Process -FilePath $binary -WorkingDirectory $release -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $EvidenceRoot 'simulator.stdout.log') `
    -RedirectStandardError (Join-Path $EvidenceRoot 'simulator.stderr.log')
  Wait-ForTcpPort -Port $lock.simulator.tcp_image_port -TimeoutSeconds 20
  $deadline = [DateTime]::UtcNow.AddSeconds(10)
  while (-not (Test-Path -LiteralPath (Join-Path $ipc 'talos_ipc_meta'))) {
    if ([DateTime]::UtcNow -gt $deadline) { throw 'Timed out waiting for Talos metadata mapping.' }
    Start-Sleep -Milliseconds 100
  }
  $sceneProcess = Start-Process -FilePath $sceneControl -WorkingDirectory $repo -ArgumentList @('--session', "windows-e2e-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))") -PassThru -Wait -NoNewWindow `
    -RedirectStandardOutput (Join-Path $EvidenceRoot 'scene_control.stdout.log') `
    -RedirectStandardError (Join-Path $EvidenceRoot 'scene_control.stderr.log')
  if ($sceneProcess.ExitCode -ne 0) { throw "Scene control failed with exit code $($sceneProcess.ExitCode)." }
  Start-Sleep -Seconds 1
  $arguments = @('--ipc-dir', $ipc, '--armor-engine', $engine, '--param-yaml', $params,
                 '--duration-seconds', ($WarmupSeconds + $DurationSeconds),
                 '--frame-log', (Join-Path $EvidenceRoot 'frame_events.jsonl'))
  $bridgeProcess = Start-Process -FilePath $bridge -WorkingDirectory $repo -ArgumentList $arguments -PassThru -Wait -NoNewWindow `
    -RedirectStandardOutput (Join-Path $EvidenceRoot 'bridge.stdout.log') `
    -RedirectStandardError (Join-Path $EvidenceRoot 'bridge.stderr.log')
  if ($bridgeProcess.ExitCode -ne 0) { throw "Native Windows bridge failed with exit code $($bridgeProcess.ExitCode)." }
} finally {
  if ($null -ne $simulator) { Stop-ProcessTree $simulator.Id }
}

$analyzer = @'
import argparse, json, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--warmup', type=float, required=True)
p.add_argument('--duration', type=float, required=True)
p.add_argument('--stage3', action='store_true')
args = p.parse_args()
root = Path(args.root)
records = [json.loads(line) for line in (root / 'frame_events.jsonl').read_text(encoding='utf-8').splitlines() if line]
records = [r for r in records if r['elapsed_ms'] >= args.warmup * 1000]
def dist(values):
    if not values: return {'count': 0, 'min': None, 'median': None, 'mean': None, 'p95': None, 'p99': None, 'max': None}
    vals = sorted(values)
    q = lambda p: vals[round((len(vals)-1)*p)]
    return {'count':len(vals),'min':min(vals),'median':statistics.median(vals),'mean':statistics.fmean(vals),'p95':q(.95),'p99':q(.99),'max':max(vals)}
seq_gaps = sum(max(0, b['source_sequence'] - a['source_sequence'] - 1) for a,b in zip(records,records[1:]))
receive_ms = [b['elapsed_ms'] - a['elapsed_ms'] for a,b in zip(records,records[1:])]
capture_ms = [(b['capture_timestamp_ns'] - a['capture_timestamp_ns']) / 1e6 for a,b in zip(records,records[1:])]
stage3_path = root / 'stage3_observations.jsonl'
summary = {
  'schema_version': 1, 'kind': 'native_windows_autoaim_b_end_to_end',
  'warmup_seconds': args.warmup, 'measurement_seconds': args.duration,
  'frames_processed': len(records), 'processed_fps': len(records)/args.duration,
  'target_frames': sum(r['has_target'] for r in records), 'udp_commands_sent': sum(r['udp_sent'] for r in records),
  'source_sequence_gaps': seq_gaps, 'process_ms': dist([r['process_ms'] for r in records]),
  'bridge_completion_interval_ms': dist(receive_ms), 'source_capture_interval_ms': dist(capture_ms),
  'stage3_enabled': args.stage3, 'stage3_observation_lines': len(stage3_path.read_text(encoding='utf-8').splitlines()) if stage3_path.exists() else 0,
  'raw_logs': ['frame_events.jsonl','bridge.stdout.log','bridge.stderr.log','scene_control.stdout.log','scene_control.stderr.log','simulator.stdout.log','simulator.stderr.log'],
  'scope': 'Windows Release simulator -> localhost TCP RGBA32 -> TensorRT vision/PnP/fire-control -> UDP command; Stage3 is enabled only when stated.'
}
try:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(1, 2, figsize=(10, 4))
  ax[0].hist(receive_ms, bins=50, color='#2463a6', edgecolor='white'); ax[0].set_title('Bridge completion interval'); ax[0].set_xlabel('ms')
  ax[1].hist([r['process_ms'] for r in records], bins=50, color='#d05a32', edgecolor='white'); ax[1].set_title('Per-frame visual pipeline'); ax[1].set_xlabel('ms')
  fig.tight_layout(); fig.savefig(root / 'interval_distributions.png', dpi=160); plt.close(fig)
  summary['distribution_plot'] = 'interval_distributions.png'
except Exception as e: summary['distribution_plot_error'] = repr(e)
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
d = summary
(root / 'PERFORMANCE_REPORT.md').write_text(f'''# Windows Native Autoaim B End-to-End Report

- Measurement: {d['measurement_seconds']} s after {d['warmup_seconds']} s warmup
- Processed: {d['frames_processed']} frames, {d['processed_fps']:.3f} FPS
- Stage3 enabled: {d['stage3_enabled']}; observations: {d['stage3_observation_lines']}
- Source-sequence gaps while consumer was active: {d['source_sequence_gaps']}
- Per-frame visual pipeline ms — median {d['process_ms']['median']}, p95 {d['process_ms']['p95']}, p99 {d['process_ms']['p99']}, max {d['process_ms']['max']}
- Completion interval ms — median {d['bridge_completion_interval_ms']['median']}, p95 {d['bridge_completion_interval_ms']['p95']}, p99 {d['bridge_completion_interval_ms']['p99']}, max {d['bridge_completion_interval_ms']['max']}
- Capture interval ms — median {d['source_capture_interval_ms']['median']}, p95 {d['source_capture_interval_ms']['p95']}, p99 {d['source_capture_interval_ms']['p99']}, max {d['source_capture_interval_ms']['max']}

Scope: {d['scope']}

Raw per-frame evidence: `frame_events.jsonl`; launch/bridge logs are indexed by `summary.json`.
''', encoding='utf-8')
print(json.dumps(summary, indent=2))
'@
$args = @('--root', $EvidenceRoot, '--warmup', $WarmupSeconds, '--duration', $DurationSeconds)
if ($EnableStage3) { $args += '--stage3' }
$analyzer | & $python - @args | Tee-Object -FilePath (Join-Path $EvidenceRoot 'analyzer.stdout.log')
if ($LASTEXITCODE -ne 0) { throw "Analyzer failed with exit code $LASTEXITCODE." }
Copy-Item -LiteralPath (Join-Path $workspace 'models\engines\windows\armor-0708-trt861-win-rtx4060-fp16.engine.benchmark.json') -Destination (Join-Path $EvidenceRoot 'engine_benchmark_reference.json')
$index = [ordered]@{
  simulator_release = $lock.simulator.version
  simulator_binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
  consumer_commit = (git -C $repo rev-parse HEAD).Trim()
  engine_sha256 = (Get-FileHash -LiteralPath $engine -Algorithm SHA256).Hash
  mode = if ($EnableStage3) { 'stage3_capture' } else { 'runtime_only' }
  reproduce = ".\scripts\bench-windows-autoaim-e2e.ps1 -WarmupSeconds $WarmupSeconds -DurationSeconds $DurationSeconds" + $(if ($EnableStage3) { ' -EnableStage3' } else { '' })
  raw_logs = @('frame_events.jsonl','bridge.stdout.log','bridge.stderr.log','scene_control.stdout.log','scene_control.stderr.log','simulator.stdout.log','simulator.stderr.log','analyzer.stdout.log')
  summaries = @('summary.json','PERFORMANCE_REPORT.md')
} | ConvertTo-Json -Depth 3
Set-Content -LiteralPath (Join-Path $EvidenceRoot 'run_index.json') -Value $index -Encoding UTF8
Write-Output "evidence_root=$EvidenceRoot"
