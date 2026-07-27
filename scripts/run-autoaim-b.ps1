[CmdletBinding()]
param([switch]$Visible, [switch]$RebuildBridge, [switch]$DynamicRange,
      [switch]$StationaryRange,
      [int]$DurationSeconds = 0,
      [double]$RangeTargetDistanceMeters = 0,
      [double]$RangeSpinDegPerSec = 30,
      [switch]$PipelineOnly,
      [switch]$FullTelemetry,
      [switch]$PnpJointDiagnostics)

$ErrorActionPreference = 'Stop'

if ($DynamicRange -and $StationaryRange) {
    throw 'DynamicRange and StationaryRange are mutually exclusive.'
}
$rangeEnabled = $DynamicRange -or $StationaryRange

function Stop-ProcessTree([int]$RootPid) {
    $children = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ParentProcessId -eq $RootPid })
    foreach ($child in $children) {
        Stop-ProcessTree ([int]$child.ProcessId)
    }
    $process = Get-Process -Id $RootPid -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
    }
}

function Assert-PortFree([ValidateSet('TCP','UDP')][string]$Protocol, [int]$Port) {
    if ($Protocol -eq 'TCP') {
        $endpoints = @(Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
            Where-Object { $_.State -ne 'TimeWait' })
    } else {
        $endpoints = @(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue)
    }
    if ($endpoints.Count -eq 0) { return }
    $owners = foreach ($endpoint in $endpoints) {
        $owner = Get-Process -Id $endpoint.OwningProcess -ErrorAction SilentlyContinue
        $name = if ($null -eq $owner) { '<exited>' } else { $owner.ProcessName }
        "PID=$($endpoint.OwningProcess) process=$name state=$($endpoint.State)"
    }
    throw "$Protocol port $Port is already occupied: $($owners -join '; ')"
}

function Stop-LinuxBridgeByToken([string]$Token) {
    $target = "aim_sim_talos_auto_aim_bridge_$Token"
    $cleanup = @'
set -u
target='__TARGET__'
pids=$(ps -eo pid=,args= | awk -v target="$target" '$2 == target {print $1}')
for pid in $pids; do
  kill -TERM "$pid" 2>/dev/null || true
done
for tick in 1 2 3 4 5; do
  remaining=$(ps -eo pid=,args= | awk -v target="$target" '$2 == target {print $1}')
  [[ -z "$remaining" ]] && exit 0
  sleep 0.2
done
for pid in $remaining; do
  kill -KILL "$pid" 2>/dev/null || true
done
'@.Replace('__TARGET__',$target).Replace("`r",'')
    $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cleanup))
    & wsl.exe -d Ubuntu-OSTEP -- bash -lc "echo $payload | base64 -d | bash" | Out-Null
}

$repo = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $repo)
$lock = Get-Content -LiteralPath (Join-Path $repo 'simulator.lock.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$releaseRoot = Join-Path $workspace $lock.simulator.release_relative_to_workspace
$sdkRoot = Join-Path $releaseRoot 'sdk'
$binary = Join-Path $releaseRoot 'bin\daedalus.exe'
$sdkConfig = Join-Path $sdkRoot 'lib\cmake\DaedalusSimSdk\DaedalusSimSdkConfig.cmake'
if (-not (Test-Path -LiteralPath $binary)) { throw "Missing simulator Release: $binary" }
if (-not (Test-Path -LiteralPath $sdkConfig)) { throw "Missing DaedalusSimSdk: $sdkConfig" }

& (Join-Path $PSScriptRoot 'check-architecture.ps1')

$ipcDir = Join-Path $workspace 'runtime\talos-ipc-autoaim-b'
$model = Join-Path $workspace 'models\engines\armor.engine'
$bridgeToken = "autoaimb_$PID"
if ($bridgeToken -notmatch '^[A-Za-z0-9_]+$') { throw "Invalid bridge token: $bridgeToken" }
$evidenceRoot = $env:AIM_SIM_EVIDENCE_ROOT
if ([string]::IsNullOrWhiteSpace($evidenceRoot)) {
    $evidenceRoot = Join-Path $workspace 'runtime\aim-autoaim-b-evidence'
}
New-Item -ItemType Directory -Force -Path $evidenceRoot | Out-Null
$bridgeDebugJson = Join-Path $evidenceRoot 'bridge.json'
$bridgeDebugJsonl = Join-Path $evidenceRoot 'bridge.jsonl'
$pipelineDebugJson = Join-Path $evidenceRoot 'pipeline.json'
$pipelineDebugJsonl = Join-Path $evidenceRoot 'pipeline.jsonl'
$bridgeLog = Join-Path $evidenceRoot 'bridge.log'
$simulatorStdoutLog = Join-Path $evidenceRoot 'simulator.stdout.log'
$simulatorStderrLog = Join-Path $evidenceRoot 'simulator.stderr.log'
$simulatorStatsJson = Join-Path $evidenceRoot 'simulator.stats.json'
$projectileEventsJsonl = Join-Path $evidenceRoot 'projectiles.jsonl'
if ($rangeEnabled) {
    if ($RangeTargetDistanceMeters -gt 0 -and
        ($RangeTargetDistanceMeters -lt 0.5 -or $RangeTargetDistanceMeters -gt 12.0)) {
        throw 'RangeTargetDistanceMeters must be within the simulator-supported 0.5..12.0 m range.'
    }
    if ($RangeSpinDegPerSec -le 0) { throw 'RangeSpinDegPerSec must be positive.' }
}
New-Item -ItemType Directory -Force -Path $ipcDir | Out-Null
$bridgeRoot = Join-Path $repo 'modules\autoaim'
function Convert-ToWslPath([string]$WindowsPath) {
    $savedEncoding = [Console]::OutputEncoding
    $wslInput = $WindowsPath.Replace('\','/')
    try {
        # Windows PowerShell 5 decodes native stdout using the console code page.
        # wslpath emits UTF-8, so preserve Unicode paths while reading its output.
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
        $result = & wsl.exe -d Ubuntu-OSTEP -- wslpath -a -u $wslInput
        $exitCode = $LASTEXITCODE
    } finally {
        [Console]::OutputEncoding = $savedEncoding
    }
    $wslPath = ($result | Out-String).Trim()
    if ($exitCode -ne 0 -or [string]::IsNullOrWhiteSpace($wslPath)) {
        throw "wslpath failed for '$WindowsPath' (exit=$exitCode, output='$wslPath')"
    }
    return $wslPath
}

$bridgeRootWsl = Convert-ToWslPath $bridgeRoot
$ipcDirWsl = Convert-ToWslPath $ipcDir
$sdkRootWsl = Convert-ToWslPath $sdkRoot
$modelWsl = Convert-ToWslPath $model
$bridgeDebugJsonWsl = Convert-ToWslPath $bridgeDebugJson
$bridgeDebugJsonlWsl = if ($FullTelemetry -and -not $PipelineOnly) { Convert-ToWslPath $bridgeDebugJsonl } else { '' }
$pipelineDebugJsonWsl = if ($FullTelemetry -or $PipelineOnly) { Convert-ToWslPath $pipelineDebugJson } else { 'OFF' }
$pipelineDebugJsonlWsl = if ($FullTelemetry -or $PipelineOnly) { Convert-ToWslPath $pipelineDebugJsonl } else { '' }
$bridgeLogWsl = Convert-ToWslPath $bridgeLog
$forceRebuild = if ($RebuildBridge) { '1' } else { '0' }
$pnpJointDiagnosticsValue = if ($PnpJointDiagnostics) { 'ON' } else { 'OFF' }
$sceneMode = if ($StationaryRange) { 'shooting_range_g2_stationary' } elseif ($DynamicRange) { 'shooting_range_g2' } else { 'off' }
$spinDegString = $RangeSpinDegPerSec.ToString('0.###',[Globalization.CultureInfo]::InvariantCulture)
Write-Output "bridge_token=$bridgeToken"
if ($StationaryRange) {
    Write-Output "range_target=3 motion=stationary distance_m=$RangeTargetDistanceMeters"
} elseif ($DynamicRange) {
    Write-Output "range_target=3 motion=spin spin_deg_s=$RangeSpinDegPerSec distance_m=$RangeTargetDistanceMeters"
}
Assert-PortFree TCP $lock.simulator.tcp_image_port
Assert-PortFree UDP $lock.simulator.scene_control_port

$bridgeCommand = @'
set -euo pipefail
cd '__BRIDGE__'
export TALOS_IPC_DIR='__IPC__'
export DAEDALUS_SIM_SDK_ROOT='__SDK__'
export AIM_SIM_ARMOR_ENGINE='__MODEL__'
export AIM_SIM_WITH_VIVSIONN_TRT=ON
export AIM_SIM_ENABLE_UDP=ON
export AIM_SIM_IMAGE_TRANSPORT=tcp
export AIM_SIM_SCENE_CONTROL_HOST="$(ip route show default | awk '{print $3; exit}')"
export AIM_SIM_SCENE_CONTROL_MODE='__SCENE_MODE__'
export AIM_SIM_DEBUG_BRIDGE_JSON='__BRIDGE_DEBUG_JSON__'
export AIM_SIM_DEBUG_BRIDGE_JSONL='__BRIDGE_DEBUG_JSONL__'
export AIM_SIM_DEBUG_PIPELINE_JSON='__PIPELINE_DEBUG_JSON__'
export AIM_SIM_DEBUG_PIPELINE_JSONL='__PIPELINE_DEBUG_JSONL__'
export DAEDALUS_BRIDGE_TOKEN='__TOKEN__'
export AIM_SIM_RANGE_TARGET_NUMBER='3'
export AIM_SIM_RANGE_SPIN_DEG_S='__SPIN_DEG_S__'
export AIM_SIM_FORCE_REBUILD=__REBUILD__
export AIM_SIM_PNP_PARALLEL_JOINT_DIAGNOSTICS=__PNP_JOINT_DIAGNOSTICS__
# Predictor-research capture is intentionally excluded from the fire-control
# runtime. Use the dedicated stage3 launcher when those artifacts are needed.
unset AIM_SIM_STAGE3_OBSERVATIONS AIM_SIM_STAGE3_TRUTH AIM_SIM_STAGE3_SESSION_ID
unset AIM_SIM_STAGE3_DISTANCE_M AIM_SIM_STAGE3_TRUTH_GIMBAL
unset AIM_SIM_DEBUG_FOV_JSONL AIM_SIM_PROFILE_DETECTOR_STAGES
exec bash scripts/run_talos_bridge_wsl.sh armor >'__BRIDGE_LOG__' 2>&1
'@.Replace('__BRIDGE__',$bridgeRootWsl).Replace('__IPC__',$ipcDirWsl).Replace('__SDK__',$sdkRootWsl).Replace('__MODEL__',$modelWsl).Replace('__SCENE_MODE__',$sceneMode).Replace('__BRIDGE_DEBUG_JSON__',$bridgeDebugJsonWsl).Replace('__BRIDGE_DEBUG_JSONL__',$bridgeDebugJsonlWsl).Replace('__PIPELINE_DEBUG_JSON__',$pipelineDebugJsonWsl).Replace('__PIPELINE_DEBUG_JSONL__',$pipelineDebugJsonlWsl).Replace('__BRIDGE_LOG__',$bridgeLogWsl).Replace('__TOKEN__',$bridgeToken).Replace('__SPIN_DEG_S__',$spinDegString).Replace('__REBUILD__',$forceRebuild).Replace('__PNP_JOINT_DIAGNOSTICS__',$pnpJointDiagnosticsValue).Replace("`r",'')

$env:BEVY_ASSET_ROOT = $releaseRoot
$env:TALOS_IPC_DIR = $ipcDir
$env:DAEDALUS_SCENE_CONTROL_BIND = '0.0.0.0:5603'
if ($rangeEnabled) {
    $env:DAEDALUS_RANGE_ACTIVE_TARGET_NUMBER = '3'
    $env:DAEDALUS_RANGE_TARGET3_INITIAL_YAW_DEG = '0'
    if ($RangeTargetDistanceMeters -gt 0) {
        $env:DAEDALUS_RANGE_TARGET_DISTANCE_M = $RangeTargetDistanceMeters.ToString('0.###',[Globalization.CultureInfo]::InvariantCulture)
    } else {
        Remove-Item Env:DAEDALUS_RANGE_TARGET_DISTANCE_M -ErrorAction SilentlyContinue
    }
} else {
    Remove-Item Env:DAEDALUS_RANGE_ACTIVE_TARGET_NUMBER -ErrorAction SilentlyContinue
    Remove-Item Env:DAEDALUS_RANGE_TARGET3_INITIAL_YAW_DEG -ErrorAction SilentlyContinue
    Remove-Item Env:DAEDALUS_RANGE_TARGET_DISTANCE_M -ErrorAction SilentlyContinue
}
$env:WGPU_POWER_PREF = 'high'
$env:DAEDALUS_CONFIG = 'config.performance.toml'
$env:DAEDALUS_TALOS_RGB_ONLY = '1'
$env:DAEDALUS_TALOS_CAPTURE_MAX_HZ = '200'
$env:DAEDALUS_TALOS_IMAGE_TRANSPORT = 'tcp'
$env:DAEDALUS_AUTO_AIM_ON_START = '1'
$env:DAEDALUS_STATS_JSON = $simulatorStatsJson
$env:DAEDALUS_PROJECTILE_EVENTS_JSONL = $projectileEventsJsonl
$env:RUST_LOG = 'warn'
$env:PATH = "$(Join-Path $releaseRoot 'bin');$env:PATH"
if ($Visible) {
    $env:WGPU_BACKEND = 'vulkan'
    Remove-Item Env:DAEDALUS_PERF_DISABLE_UI -ErrorAction SilentlyContinue
    $env:DAEDALUS_PREVIEW_ENABLED = '1'
    $env:DAEDALUS_PREVIEW_MAX_HZ = '60'
} else {
    $env:WGPU_BACKEND = 'dx12'
    $env:DAEDALUS_PERF_DISABLE_UI = '1'
    Remove-Item Env:DAEDALUS_PREVIEW_ENABLED -ErrorAction SilentlyContinue
    Remove-Item Env:DAEDALUS_PREVIEW_MAX_HZ -ErrorAction SilentlyContinue
}

$simulator = $null
$bridge = $null
$runStartedAt = $null
$startupDeadline = [DateTime]::UtcNow.AddSeconds(120)
try {
    $simulatorStart = @{
        FilePath = $binary
        WorkingDirectory = $releaseRoot
        PassThru = $true
        RedirectStandardOutput = $simulatorStdoutLog
        RedirectStandardError = $simulatorStderrLog
    }
    if ($Visible) { $simulatorStart.WindowStyle = 'Normal' }
    # Performance mode follows the Release launcher exactly: UI is disabled by
    # DAEDALUS_PERF_DISABLE_UI, and no Win32 forced-hidden startup state is set.
    $simulator = Start-Process @simulatorStart
    Start-Sleep -Seconds 10
    $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bridgeCommand))
    $bridgeInfo = [Diagnostics.ProcessStartInfo]::new()
    $bridgeInfo.FileName = 'wsl.exe'
    $bridgeInfo.UseShellExecute = $false
    $bridgeInfo.Arguments = "-d Ubuntu-OSTEP -- bash -c `"echo $payload | base64 -d | bash`""
    $bridge = [Diagnostics.Process]::Start($bridgeInfo)
    while ($true) {
        $simulator.Refresh()
        $bridge.Refresh()
        if ($bridge.HasExited) {
            throw "Aim bridge exited before the simulator (exit=$($bridge.ExitCode))."
        }
        if ($simulator.HasExited) {
            if ($simulator.ExitCode -ne 0) {
                throw "Simulator exited before the bridge (exit=$($simulator.ExitCode))."
            }
            break
        }
        if ($DurationSeconds -gt 0) {
            $ready = $true
            if ($rangeEnabled) {
                $activeConnections = @(Get-NetTCPConnection -LocalPort $lock.simulator.tcp_image_port -State Established -ErrorAction SilentlyContinue)
                $ready = $activeConnections.Count -gt 0
            }
            if ($ready -and $null -eq $runStartedAt) {
                $runStartedAt = [DateTime]::UtcNow
            }
            if ($null -ne $runStartedAt -and
                (([DateTime]::UtcNow - $runStartedAt).TotalSeconds -ge $DurationSeconds)) {
                break
            }
            if ($null -eq $runStartedAt -and [DateTime]::UtcNow -gt $startupDeadline) {
                throw 'Timed out waiting for the dynamic bridge TCP connection.'
            }
        }
        Start-Sleep -Milliseconds 250
    }
    $fatalSimulatorLog = Select-String -LiteralPath $simulatorStderrLog `
        -Pattern 'tcp_bind_failed|ResizeBuffers failed|Invalid surface|Validation RenderError' `
        -ErrorAction SilentlyContinue
    if ($null -ne $fatalSimulatorLog) {
        throw "Simulator reported a fatal runtime error; see $simulatorStderrLog"
    }
    $simulatorStats = $null
    foreach ($attempt in 1..10) {
        try {
            $simulatorStats = Get-Content -LiteralPath $simulatorStatsJson -Raw `
                -Encoding UTF8 | ConvertFrom-Json
            break
        } catch {
            Start-Sleep -Milliseconds 50
        }
    }
    if ($null -eq $simulatorStats) {
        throw "Simulator stats were not readable: $simulatorStatsJson"
    }
    if ([int64]$simulatorStats.talos_tcp_image_bind_fail_total -ne 0 -or
        [int64]$simulatorStats.capture_fast_map_error_total -ne 0 -or
        [int64]$simulatorStats.talos_tcp_image_connect_total -le 0 -or
        [int64]$simulatorStats.talos_tcp_image_sent_total -le 0 -or
        [int64]$simulatorStats.capture_processing_complete_total -le 0) {
        throw "Simulator stats failed capture/TCP health checks: $simulatorStatsJson"
    }
} finally {
    Stop-LinuxBridgeByToken $bridgeToken
    if ($null -ne $bridge) { Stop-ProcessTree $bridge.Id }
    if ($null -ne $simulator) { Stop-ProcessTree $simulator.Id }
}
