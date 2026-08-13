[CmdletBinding()]
param(
  [string]$DataRoot = '',
  [double[]]$RadialScales = @(0.75, 1.0, 1.25),
  [double[]]$DistancesM = @(1.5, 2.2, 3.5, 5.0),
  [int]$Repeats = 3,
  [int]$RepeatOffset = 0,
  [int]$WarmupSeconds = 3,
  [int]$DurationSeconds = 8,
  [ValidateSet('stationary', 'linear', 'spin', 'linear_and_spin')]
  [string]$MotionMode = 'spin',
  [double]$SpinDegS = 114.59156,
  [double]$LinearDirectionDeg = 90.0,
  [double]$LinearSpeedMps = 0.0,
  [double]$LinearSpanM = 0.0,
  [switch]$EnablePipelineDiagnostics,
  [string]$BridgeBinary = 'C:\codex-autoaim-build-current-20260808\aim_sim_windows_auto_aim_bridge.exe',
  [string]$SceneControlBinary = 'C:\codex-autoaim-build-current-20260808\aim_sim_scene_control_cli.exe',
  [string]$EngineBinary = '',
  [double]$DetectorScoreThreshold = 0.5
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $repo)
$benchPath = Join-Path $repo 'scripts\bench-windows-autoaim-e2e.ps1'
$truthAuditPath = Join-Path $repo 'scripts\audit-stage3-truth-motion.py'
$simulatorRoot = Join-Path $workspace 'repos\daedalus-simulator'
$simulatorBinary = Join-Path $simulatorRoot 'target\release\daedalus.exe'
if (-not $EngineBinary) {
  $EngineBinary = Join-Path $workspace 'models\engines\windows\armor-0708-trt861-win-rtx4060-fp16.engine'
}

foreach ($path in @($benchPath, $truthAuditPath, $simulatorBinary, $BridgeBinary, $SceneControlBinary, $EngineBinary)) {
  if (-not (Test-Path -LiteralPath $path)) { throw "Required collection input is absent: $path" }
}
if ($Repeats -lt 1) { throw 'Repeats must be positive.' }
if ($RepeatOffset -lt 0) { throw 'RepeatOffset must be nonnegative.' }
if ($WarmupSeconds -lt 0 -or $DurationSeconds -lt 5) {
  throw 'WarmupSeconds must be nonnegative and DurationSeconds must be at least 5.'
}
if ($SpinDegS -lt 0.0 -or $SpinDegS -gt 720.0) {
  throw "SpinDegS must be in [0, 720], got: $SpinDegS"
}
if ($LinearDirectionDeg -lt -180.0 -or $LinearDirectionDeg -gt 180.0) {
  throw "LinearDirectionDeg must be in [-180, 180], got: $LinearDirectionDeg"
}
if ($LinearSpeedMps -lt 0.0 -or $LinearSpeedMps -gt 3.0) {
  throw "LinearSpeedMps must be in [0, 3], got: $LinearSpeedMps"
}
if ($LinearSpanM -lt 0.0 -or $LinearSpanM -gt 8.0) {
  throw "LinearSpanM must be in [0, 8], got: $LinearSpanM"
}
$hasLinear = $LinearSpeedMps -gt 0.0 -and $LinearSpanM -gt 0.0
$hasSpin = [math]::Abs($SpinDegS) -gt 0.0
$motionConsistent =
  ($MotionMode -eq 'stationary' -and -not $hasLinear -and -not $hasSpin) -or
  ($MotionMode -eq 'linear' -and $hasLinear -and -not $hasSpin) -or
  ($MotionMode -eq 'spin' -and -not $hasLinear -and $hasSpin) -or
  ($MotionMode -eq 'linear_and_spin' -and $hasLinear -and $hasSpin)
if (-not $motionConsistent) {
  throw "Motion parameters are inconsistent with mode $MotionMode."
}
if ($DetectorScoreThreshold -lt 0.01 -or $DetectorScoreThreshold -gt 0.99) {
  throw "DetectorScoreThreshold must be in [0.01, 0.99], got: $DetectorScoreThreshold"
}
foreach ($scale in $RadialScales) {
  if ($scale -lt 0.75 -or $scale -gt 1.25) { throw "Radial scale out of simulator range: $scale" }
}
foreach ($distance in $DistancesM) {
  if ($distance -lt 0.5 -or $distance -gt 12.0) { throw "Distance out of simulator range: $distance" }
}

if (-not $DataRoot) {
  $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
  $DataRoot = Join-Path $workspace "runtime\autoaim-b-trajectory-grid-dev-$stamp"
}
if (Test-Path -LiteralPath $DataRoot) { throw "Data root already exists: $DataRoot" }
New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null

# Adapt the formal benchmark only at runtime. Its defaults remain pinned to the
# current Windows Release; this collector points the same bridge at a local
# truth-capable development simulator without changing the formal launcher.
$bench = Get-Content -LiteralPath $benchPath -Raw
$bench = $bench.Replace('$repo = Split-Path -Parent $PSScriptRoot', ('$repo = ''' + $repo + ''''))
$bench = $bench.Replace('$release = Join-Path $workspace $lock.simulator.release_relative_to_workspace', ('$release = ''' + $simulatorRoot + ''''))
$bench = $bench.Replace('$binary = Join-Path $release ''bin\daedalus.exe''', ('$binary = ''' + $simulatorBinary + ''''))
$bench = $bench.Replace('$bridge = ''C:\codex-autoaim-build-ninja-cuda118d\aim_sim_windows_auto_aim_bridge.exe''', ('$bridge = ''' + $BridgeBinary + ''''))
$bench = $bench.Replace('$sceneControl = ''C:\codex-autoaim-build-ninja-cuda118d\aim_sim_scene_control_cli.exe''', ('$sceneControl = ''' + $SceneControlBinary + ''''))
$bench = $bench.Replace('$engine = Join-Path $workspace ''models\engines\windows\armor-0708-trt861-win-rtx4060-fp16.engine''', ('$engine = ''' + $EngineBinary + ''''))
$bench = $bench.Replace('$env:DAEDALUS_CONFIG = ''config.performance.toml''', ('$env:DAEDALUS_CONFIG = ''' + (Join-Path $simulatorRoot 'config.performance.toml') + ''''))
$benchScript = [scriptblock]::Create($bench)

$commonManifest = [ordered]@{
  schema_version = 1
  kind = 'stage3_truth_grid_collection'
  simulator_mode = 'development_truth_capable'
  simulator_source_root = $simulatorRoot
  simulator_binary = $simulatorBinary
  simulator_binary_sha256 = (Get-FileHash -LiteralPath $simulatorBinary -Algorithm SHA256).Hash
  bridge_binary = $BridgeBinary
  bridge_binary_sha256 = (Get-FileHash -LiteralPath $BridgeBinary -Algorithm SHA256).Hash
  scene_control_binary = $SceneControlBinary
  consumer_repo = $repo
  consumer_commit = (git -C $repo rev-parse HEAD).Trim()
  collector_script = $PSCommandPath
  collector_script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
  benchmark_script = $benchPath
  benchmark_script_sha256 = (Get-FileHash -LiteralPath $benchPath -Algorithm SHA256).Hash
  truth_audit_script = $truthAuditPath
  truth_audit_script_sha256 = (Get-FileHash -LiteralPath $truthAuditPath -Algorithm SHA256).Hash
  engine = $EngineBinary
  engine_sha256 = (Get-FileHash -LiteralPath $EngineBinary -Algorithm SHA256).Hash
  scene = 'shooting_range'
  target_number = 3
  motion_mode = $MotionMode
  spin_deg_s = $SpinDegS
  linear_direction_deg = $LinearDirectionDeg
  linear_speed_mps = $LinearSpeedMps
  linear_span_m = $LinearSpanM
  pipeline_diagnostics = [bool]$EnablePipelineDiagnostics
  detector_score_threshold = $DetectorScoreThreshold
  warmup_seconds = $WarmupSeconds
  duration_seconds = $DurationSeconds
  allowed_source_frame_loss = $true
  identity_contract = 'active target semantics + physical relative_slot 0..3 within target; target_id is run-local only'
  grid = [ordered]@{
    radial_scales = @($RadialScales)
    distances_m = @($DistancesM)
    repeats = $Repeats
    repeat_offset = $RepeatOffset
  }
} | ConvertTo-Json -Depth 5
Set-Content -LiteralPath (Join-Path $DataRoot 'collection_manifest.json') -Value $commonManifest -Encoding UTF8

$total = $RadialScales.Count * $DistancesM.Count * $Repeats
$index = 0
foreach ($scale in $RadialScales) {
  foreach ($distance in $DistancesM) {
    for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
      $index++
      $scaleTag = ('{0:0.00}' -f $scale).Replace('.', 'p')
      $distanceTag = ('{0:0.0}' -f $distance).Replace('.', 'p')
      $actualRepeat = $RepeatOffset + $repeat
      $runName = "r${scaleTag}-d${distanceTag}-rep$('{0:00}' -f $actualRepeat)"
      $root = Join-Path $DataRoot $runName
      if (Test-Path -LiteralPath $root) {
        throw "Refusing to overwrite protected run root: $root"
      }

      $env:DAEDALUS_RANGE_ACTIVE_TARGET_NUMBER = '3'
      $env:DAEDALUS_RANGE_TARGET_DISTANCE_M = "$distance"
      $env:DAEDALUS_RANGE_TARGET3_RADIAL_SCALE = "$scale"
      $env:DAEDALUS_RANGE_TARGET3_MOTION_MODE = $MotionMode
      $env:DAEDALUS_RANGE_TARGET3_SPIN_DEG_S = "$SpinDegS"
      $env:DAEDALUS_RANGE_TARGET3_LINEAR_DIRECTION_DEG = "$LinearDirectionDeg"
      $env:DAEDALUS_RANGE_TARGET3_LINEAR_SPEED_MPS = "$LinearSpeedMps"
      $env:DAEDALUS_RANGE_TARGET3_LINEAR_SPAN_M = "$LinearSpanM"
      $env:DAEDALUS_TALOS_TCP_BIND = '127.0.0.1:5602'
      $env:AIM_SIM_TCP_IMAGE_HOST = '127.0.0.1'
      $env:AIM_SIM_TCP_IMAGE_PORT = '5602'
      $env:AIM_SIM_STAGE3_TRUTH = Join-Path $root 'truth.jsonl'
      $env:AIM_SIM_STAGE3_DISTANCE_M = "$distance"
      $env:AIM_SIM_DETECTOR_SCORE_THRESHOLD = "$DetectorScoreThreshold"
      if ($EnablePipelineDiagnostics) {
        $env:AIM_SIM_DEBUG_PIPELINE_JSONL = Join-Path $root 'pipeline.jsonl'
      } else {
        Remove-Item Env:AIM_SIM_DEBUG_PIPELINE_JSONL -ErrorAction SilentlyContinue
      }

      Write-Host "[$index/$total] scale=$scale distance_m=$distance mode=$MotionMode spin_deg_s=$SpinDegS linear_mps=$LinearSpeedMps span_m=$LinearSpanM repeat=$repeat"
      & $benchScript -WarmupSeconds $WarmupSeconds -DurationSeconds $DurationSeconds `
        -EnableStage3 -InitialScene shooting_range -TruthGimbalTarget 3 -EvidenceRoot $root `
        -RangeMotionMode $MotionMode -RangeDirectionDeg $LinearDirectionDeg `
        -RangeLinearSpeedMps $LinearSpeedMps -RangeLinearSpanM $LinearSpanM `
        -RangeSpinDegS $SpinDegS -RangeRadialScale $scale

      $truthAuditOutput = Join-Path $root 'truth_motion_audit.json'
      & python $truthAuditPath --truth (Join-Path $root 'truth.jsonl') `
        --output $truthAuditOutput --radial-scale $scale --spin-deg-s $SpinDegS `
        --linear-speed-mps $LinearSpeedMps --linear-span-m $LinearSpanM
      if ($LASTEXITCODE -ne 0) {
        throw "Truth motion audit failed for protected run root: $root"
      }

      $manifest = [ordered]@{}
      $commonManifest | ConvertFrom-Json | Get-Member -MemberType NoteProperty | ForEach-Object {
        $manifest[$_.Name] = ($commonManifest | ConvertFrom-Json).($_.Name)
      }
      $manifest['run_name'] = $runName
      $manifest['radial_scale'] = $scale
      $manifest['requested_distance_m'] = $distance
      $manifest['repeat'] = $actualRepeat
      $manifest['captured_at_utc'] = (Get-Date).ToUniversalTime().ToString('o')
      $manifest['evidence_root'] = $root
      $manifest['truth_motion_audit'] = Get-Content -LiteralPath $truthAuditOutput -Raw | ConvertFrom-Json
      Set-Content -LiteralPath (Join-Path $root 'collection_run_manifest.json') `
        -Value ($manifest | ConvertTo-Json -Depth 8) -Encoding UTF8
    }
  }
}

Write-Output "collection_root=$DataRoot"
