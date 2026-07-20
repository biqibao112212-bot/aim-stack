[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$EvidenceRoot,
    [int]$DurationSeconds = 30,
    [switch]$Visible,
    [switch]$RebuildBridge,
    [switch]$DebugTelemetry
)

$ErrorActionPreference = 'Stop'
# wslpath emits UTF-8 paths; force the PowerShell subprocess encoding so the
# Chinese workspace path is not mojibaked when the runner is backgrounded.
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
if ($DurationSeconds -le 0) { throw 'DurationSeconds must be positive.' }
$repo = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $repo)
$manifestObject = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$required = 'session_id','mode','distance_m','initial_yaw_rad','direction_deg','linear_speed_mps','linear_span_m','spin_rad_s'
foreach ($field in $required) {
    if ($null -eq $manifestObject.$field) { throw "Manifest missing $field" }
}
if ($manifestObject.mode -notin @('stationary','linear','spin','linear_and_spin')) { throw 'Invalid Stage3 mode.' }
if ([double]$manifestObject.distance_m -lt 1 -or [double]$manifestObject.distance_m -gt 8) { throw 'distance_m must be in [1,8].' }
if ([double]$manifestObject.linear_speed_mps -lt 0 -or [double]$manifestObject.linear_speed_mps -gt 3) { throw 'linear_speed_mps must be in [0,3].' }
if ([math]::Abs([double]$manifestObject.spin_rad_s) -gt 15) { throw 'spin_rad_s must have abs <=15.' }

function Convert-ToWslPath([string]$path) {
    $savedEncoding = [Console]::OutputEncoding
    try {
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
        $result = & wsl.exe -d Ubuntu-OSTEP -- wslpath -a -u ($path.Replace('\','/'))
        $exitCode = $LASTEXITCODE
    } finally {
        [Console]::OutputEncoding = $savedEncoding
    }
    $wslPath = ($result | Out-String).Trim()
    if ($exitCode -ne 0 -or [string]::IsNullOrWhiteSpace($wslPath)) {
        throw "wslpath failed for '$path' (exit=$exitCode, output='$wslPath')"
    }
    return $wslPath
}
function Stop-ProcessTree([int]$rootPid) {
    $children = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ParentProcessId -eq $rootPid })
    foreach ($child in $children) { Stop-ProcessTree ([int]$child.ProcessId) }
    if (Get-Process -Id $rootPid -ErrorAction SilentlyContinue) { Stop-Process -Id $rootPid -Force -ErrorAction SilentlyContinue }
}
function Assert-PortFree([ValidateSet('TCP','UDP')][string]$Protocol, [int]$Port) {
    if ($Protocol -eq 'TCP') {
        $busy = @(Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
            Where-Object { $_.State -ne 'TimeWait' })
    } else {
        $busy = @(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue)
    }
    if ($busy.Count -eq 0) { return }
    $owners = foreach ($endpoint in $busy) {
        $owner = Get-Process -Id $endpoint.OwningProcess -ErrorAction SilentlyContinue
        $name = if ($null -eq $owner) { '<exited>' } else { $owner.ProcessName }
        "PID=$($endpoint.OwningProcess) process=$name state=$($endpoint.State)"
    }
    throw "$Protocol port $Port is already occupied: $($owners -join '; ')"
}
function Invoke-WslCleanup([string]$script) {
    $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($script))
    & wsl.exe -d Ubuntu-OSTEP -- bash -lc "echo $payload | base64 -d | bash" | Out-Null
}
function Stop-LinuxBridgeByToken([string]$Token) {
    $target = "aim_sim_talos_auto_aim_bridge_$Token"
    $cleanup = @'
set -u
target='__TARGET__'
pids=$(ps -eo pid=,args= | awk -v target="$target" '$2 == target {print $1}')
for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
for tick in 1 2 3 4 5; do
  remaining=$(ps -eo pid=,args= | awk -v target="$target" '$2 == target {print $1}')
  [[ -z "$remaining" ]] && exit 0
  sleep 0.2
done
for pid in $remaining; do kill -KILL "$pid" 2>/dev/null || true; done
'@.Replace('__TARGET__',$target)
    Invoke-WslCleanup $cleanup
}
function Stop-StaleStage3LinuxBridges {
    $cleanup = @'
set -u
pids=$(ps -eo pid=,args= | awk '$2 ~ /^aim_sim_talos_auto_aim_bridge_stage3_/ {print $1}')
for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
for tick in 1 2 3 4 5; do
  remaining=$(ps -eo pid=,args= | awk '$2 ~ /^aim_sim_talos_auto_aim_bridge_stage3_/ {print $1}')
  [[ -z "$remaining" ]] && exit 0
  sleep 0.2
done
for pid in $remaining; do kill -KILL "$pid" 2>/dev/null || true; done
'@
    Invoke-WslCleanup $cleanup
}

$datasetRoot = Join-Path $workspace 'dataset\autoaim-stage3-v1'
$sessionBase = Join-Path $datasetRoot ([string]$manifestObject.session_id)
# Keep each invocation isolated. A retry or repeated smoke run must never
# append records to the same JSONL files, otherwise the full-key join cannot
# distinguish independent simulator epochs.
$runNonce = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$sessionRoot = Join-Path $sessionBase ('run-' + $runNonce)
$controlSessionId = ([string]$manifestObject.session_id + '-run-' + $runNonce)
New-Item -ItemType Directory -Force -Path $sessionRoot,$EvidenceRoot | Out-Null
$obsPath = Join-Path $sessionRoot 'observations.jsonl'
$truthPath = Join-Path $sessionRoot 'truth.jsonl'
$bridgeLog = Join-Path $EvidenceRoot 'bridge.log'
$simOut = Join-Path $EvidenceRoot 'simulator.stdout.log'
$simErr = Join-Path $EvidenceRoot 'simulator.stderr.log'
$stats = Join-Path $EvidenceRoot 'simulator.stats.json'
$lock = Get-Content -LiteralPath (Join-Path $repo 'simulator.lock.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$releaseRoot = Join-Path $workspace $lock.simulator.release_relative_to_workspace
$binary = Join-Path $releaseRoot 'bin\daedalus.exe'
$sdkRoot = Join-Path $releaseRoot 'sdk'
$sdkConfig = Join-Path $sdkRoot 'lib\cmake\DaedalusSimSdk\DaedalusSimSdkConfig.cmake'
if (-not (Test-Path -LiteralPath $binary)) { throw "Missing simulator Release: $binary" }
if (-not (Test-Path -LiteralPath $sdkConfig)) { throw "Missing DaedalusSimSdk: $sdkConfig" }
# Keep simulator/Talos IPC isolated per invocation as well as raw JSONL. A
# retry must never reuse an old producer epoch or shared-memory metadata set.
$ipcDir = Join-Path $workspace ('runtime\stage3-' + [string]$manifestObject.session_id + '\' + ('run-' + $runNonce))
$bridgeToken = 'stage3_' + (([string]$manifestObject.session_id + '_' + $runNonce) -replace '[^A-Za-z0-9_]','_')
$bridgeRootWsl = Convert-ToWslPath (Join-Path $repo 'modules\autoaim')
$ipcWsl = Convert-ToWslPath $ipcDir
$sdkWsl = Convert-ToWslPath $sdkRoot
$obsWsl = Convert-ToWslPath $obsPath
$truthWsl = Convert-ToWslPath $truthPath
$evidenceWsl = Convert-ToWslPath $EvidenceRoot
$sceneCliWsl = Convert-ToWslPath (Join-Path $repo 'modules\autoaim\build\ros2_trt\aim_sim_scene_control_cli')
$spinDeg = ([double]$manifestObject.spin_rad_s * 180.0 / [math]::PI).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
$linearSpeed = ([double]$manifestObject.linear_speed_mps).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
$distance = ([double]$manifestObject.distance_m).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
$direction = ([double]$manifestObject.direction_deg).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
$yaw = ([double]$manifestObject.initial_yaw_rad).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
$span = ([double]$manifestObject.linear_span_m).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
$debugBridgeJsonWsl = Convert-ToWslPath (Join-Path $EvidenceRoot 'bridge.json')
$debugBridgeJsonlWsl = Convert-ToWslPath (Join-Path $EvidenceRoot 'bridge.jsonl')
$debugPipelineJsonWsl = Convert-ToWslPath (Join-Path $EvidenceRoot 'pipeline.json')
$debugPipelineJsonlWsl = Convert-ToWslPath (Join-Path $EvidenceRoot 'pipeline.jsonl')
$debugExports = if ($DebugTelemetry) {
@"
export AIM_SIM_DEBUG_BRIDGE_JSON='$debugBridgeJsonWsl'
export AIM_SIM_DEBUG_BRIDGE_JSONL='$debugBridgeJsonlWsl'
export AIM_SIM_DEBUG_PIPELINE_JSON='$debugPipelineJsonWsl'
export AIM_SIM_DEBUG_PIPELINE_JSONL='$debugPipelineJsonlWsl'
"@
} else {
@"
unset AIM_SIM_DEBUG_BRIDGE_JSON AIM_SIM_DEBUG_BRIDGE_JSONL AIM_SIM_DEBUG_PIPELINE_JSON AIM_SIM_DEBUG_PIPELINE_JSONL
"@
}

$sessionMutex = [Threading.Mutex]::new($false, 'Local\AimStackStage3Capture')
if (-not $sessionMutex.WaitOne(0)) { throw 'Another Stage-3 session runner is already active.' }
Stop-StaleStage3LinuxBridges
Assert-PortFree TCP ([int]$lock.simulator.tcp_image_port)
Assert-PortFree UDP ([int]$lock.simulator.scene_control_port)
$oldEnv = @{}
foreach ($name in 'BEVY_ASSET_ROOT','TALOS_IPC_DIR','DAEDALUS_SCENE_CONTROL_BIND','DAEDALUS_SCENE_MODE','DAEDALUS_RANGE_ACTIVE_TARGET_NUMBER','DAEDALUS_RANGE_TARGET_DISTANCE_M','DAEDALUS_RANGE_TARGET3_INITIAL_YAW_RAD','DAEDALUS_RANGE_TARGET3_INITIAL_YAW_DEG','DAEDALUS_STATS_JSON','AIM_SIM_STAGE3_SESSION_ID','AIM_SIM_STAGE3_OBSERVATIONS','AIM_SIM_STAGE3_TRUTH','AIM_SIM_STAGE3_DISTANCE_M','AIM_SIM_STAGE3_TRUTH_GIMBAL','AIM_SIM_SCENE_CONTROL_MODE','AIM_SIM_SCENE_CONTROL_HOST','AIM_SIM_IMAGE_TRANSPORT','AIM_SIM_DEBUG_BRIDGE_JSON','AIM_SIM_DEBUG_BRIDGE_JSONL','AIM_SIM_DEBUG_PIPELINE_JSON','AIM_SIM_DEBUG_PIPELINE_JSONL','DAEDALUS_BRIDGE_TOKEN','AIM_SIM_ARMOR_ENGINE','DAEDALUS_PERF_DISABLE_UI','WGPU_BACKEND','DAEDALUS_PREVIEW_ENABLED','DAEDALUS_PREVIEW_MAX_HZ') {
    $oldEnv[$name] = [Environment]::GetEnvironmentVariable($name)
}
$simulator = $null; $bridge = $null; $started = $false
try {
    $env:BEVY_ASSET_ROOT = $releaseRoot; $env:TALOS_IPC_DIR = $ipcDir; $env:DAEDALUS_PERF_DISABLE_UI = '1'; $env:WGPU_BACKEND = 'dx12'
    Remove-Item Env:DAEDALUS_PREVIEW_ENABLED -ErrorAction SilentlyContinue; Remove-Item Env:DAEDALUS_PREVIEW_MAX_HZ -ErrorAction SilentlyContinue
    $env:DAEDALUS_SCENE_CONTROL_BIND = '0.0.0.0:5603'; $env:DAEDALUS_SCENE_MODE = 'shooting_range'
    $env:DAEDALUS_RANGE_ACTIVE_TARGET_NUMBER = '3'; $env:DAEDALUS_RANGE_TARGET_DISTANCE_M = $distance
    $env:DAEDALUS_RANGE_TARGET3_INITIAL_YAW_RAD = $yaw; Remove-Item Env:DAEDALUS_RANGE_TARGET3_INITIAL_YAW_DEG -ErrorAction SilentlyContinue
    $env:DAEDALUS_STATS_JSON = $stats; $env:WGPU_POWER_PREF = 'high'; $env:DAEDALUS_CONFIG = 'config.performance.toml'
    $env:DAEDALUS_TALOS_RGB_ONLY = '1'; $env:DAEDALUS_TALOS_CAPTURE_MAX_HZ = '200'; $env:DAEDALUS_TALOS_IMAGE_TRANSPORT = 'tcp'; $env:DAEDALUS_AUTO_AIM_ON_START = '1'; $env:RUST_LOG = 'warn'; $env:PATH = "$(Join-Path $releaseRoot 'bin');$env:PATH"
    $env:AIM_SIM_STAGE3_SESSION_ID = [string]$manifestObject.session_id; $env:AIM_SIM_STAGE3_OBSERVATIONS = $obsPath; $env:AIM_SIM_STAGE3_TRUTH = $truthPath; $env:AIM_SIM_STAGE3_DISTANCE_M = $distance
    $env:AIM_SIM_IMAGE_TRANSPORT = 'tcp'; $env:AIM_SIM_SCENE_CONTROL_MODE = 'off'; $env:AIM_SIM_SCENE_CONTROL_HOST = ''
    if ($DebugTelemetry) {
        $env:AIM_SIM_DEBUG_BRIDGE_JSON = (Join-Path $EvidenceRoot 'bridge.json'); $env:AIM_SIM_DEBUG_BRIDGE_JSONL = (Join-Path $EvidenceRoot 'bridge.jsonl'); $env:AIM_SIM_DEBUG_PIPELINE_JSON = (Join-Path $EvidenceRoot 'pipeline.json'); $env:AIM_SIM_DEBUG_PIPELINE_JSONL = (Join-Path $EvidenceRoot 'pipeline.jsonl')
    } else {
        foreach ($name in 'AIM_SIM_DEBUG_BRIDGE_JSON','AIM_SIM_DEBUG_BRIDGE_JSONL','AIM_SIM_DEBUG_PIPELINE_JSON','AIM_SIM_DEBUG_PIPELINE_JSONL') { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
    }
    $env:DAEDALUS_BRIDGE_TOKEN = $bridgeToken
    $env:AIM_SIM_ARMOR_ENGINE = Join-Path $workspace 'models\engines\armor.engine'
    $windowStyle = if ($Visible) { 'Normal' } else { 'Hidden' }
    $simulator = Start-Process -FilePath $binary -WorkingDirectory $releaseRoot -PassThru -RedirectStandardOutput $simOut -RedirectStandardError $simErr -WindowStyle $windowStyle
    Start-Sleep -Seconds 8
    $bridgePayload = @"
set -euo pipefail
cd '$bridgeRootWsl'
export TALOS_IPC_DIR='$ipcWsl'
export DAEDALUS_SIM_SDK_ROOT='$sdkWsl'
export AIM_SIM_WITH_VIVSIONN_TRT=ON
export AIM_SIM_ENABLE_UDP=ON
export AIM_SIM_IMAGE_TRANSPORT=tcp
export AIM_SIM_STAGE3_SESSION_ID='$([string]$manifestObject.session_id)'
export AIM_SIM_STAGE3_OBSERVATIONS='$obsWsl'
export AIM_SIM_STAGE3_TRUTH='$truthWsl'
export AIM_SIM_STAGE3_DISTANCE_M='$distance'
export AIM_SIM_STAGE3_TRUTH_GIMBAL=1
export AIM_SIM_ARMOR_ENGINE='$(Convert-ToWslPath (Join-Path $workspace 'models\engines\armor.engine'))'
export DAEDALUS_BRIDGE_TOKEN='$bridgeToken'
$debugExports
export AIM_SIM_FORCE_REBUILD=$([int]$RebuildBridge.IsPresent)
exec bash scripts/run_talos_bridge_wsl.sh armor >'$(Convert-ToWslPath $bridgeLog)' 2>&1
"@
    $bridgePayload | Set-Content -LiteralPath (Join-Path $EvidenceRoot 'bridge-command.sh') -Encoding UTF8
    $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bridgePayload))
    $bridgeInfo = [Diagnostics.ProcessStartInfo]::new()
    $bridgeInfo.FileName = 'wsl.exe'
    $bridgeInfo.UseShellExecute = $false
    $bridgeInfo.Arguments = "-d Ubuntu-OSTEP -- bash -c `"echo $payload | base64 -d | bash`""
    $bridge = [Diagnostics.Process]::Start($bridgeInfo)
    $sceneReadyDeadline = [DateTime]::UtcNow.AddSeconds(60)
    do {
        $sceneEndpoint = @(Get-NetUDPEndpoint -LocalPort ([int]$lock.simulator.scene_control_port) -ErrorAction SilentlyContinue)
        if ($sceneEndpoint.Count -gt 0) { break }
        if ($simulator.HasExited) { throw "Simulator exited with $($simulator.ExitCode) before Scene Control became ready." }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $sceneReadyDeadline)
    if ($sceneEndpoint.Count -eq 0) { throw 'Timed out waiting for simulator Scene Control endpoint.' }
    # UDP bind only proves that the socket exists; allow the Scene Control
    # service and shooting-range scene to finish initialization before the
    # first command.
    Start-Sleep -Seconds 5
    $route = (& wsl.exe -d Ubuntu-OSTEP -- bash -lc 'ip route show default').Trim().Split()
    if ($route.Count -lt 3 -or $route[0] -ne 'default' -or $route[1] -ne 'via') { throw "Could not resolve WSL default host route: $($route -join ' ')" }
    $sceneHost = $route[2]
    $sceneCli = "'$sceneCliWsl' --stage3 --host '$sceneHost' --session '$controlSessionId' --target 3 --mode '$([string]$manifestObject.mode)' --direction-deg '$direction' --linear-speed-mps '$linearSpeed' --linear-span-m '$span' --spin-deg-s '$spinDeg'"
    $ack = $null
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        $ack = & wsl.exe -d Ubuntu-OSTEP -- bash -lc $sceneCli
        if ($LASTEXITCODE -eq 0) { break }
        if ($attempt -eq 5) { throw 'Scene Control Stage3 ACK failed after five bounded attempts.' }
        Start-Sleep -Seconds 2
    }
    $dataReadyDeadline = [DateTime]::UtcNow.AddSeconds(60)
    while ([DateTime]::UtcNow -lt $dataReadyDeadline) {
        if ($simulator.HasExited) { throw "Simulator exited with $($simulator.ExitCode) before data became ready." }
        if ($bridge.HasExited) { throw "Bridge exited with $($bridge.ExitCode) before data became ready." }
        $obsReady = (Test-Path -LiteralPath $obsPath) -and (Get-Item -LiteralPath $obsPath).Length -gt 0
        $truthReady = (Test-Path -LiteralPath $truthPath) -and (Get-Item -LiteralPath $truthPath).Length -gt 0
        if ($obsReady -and $truthReady) { break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $obsReady -or -not $truthReady) { throw 'Timed out waiting for Stage3 observation/truth streams.' }
    $started = $true
    $deadline = [DateTime]::UtcNow.AddSeconds($DurationSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($simulator.HasExited) { throw "Simulator exited with $($simulator.ExitCode)." }
        if ($bridge.HasExited) { throw "Bridge exited with $($bridge.ExitCode)." }
        Start-Sleep -Milliseconds 250
    }
    if (-not (Test-Path -LiteralPath $obsPath) -or (Get-Item -LiteralPath $obsPath).Length -eq 0) { throw 'Stage3 observation JSONL is missing or empty.' }
    if (-not (Test-Path -LiteralPath $truthPath) -or (Get-Item -LiteralPath $truthPath).Length -eq 0) { throw 'Stage3 truth JSONL is missing or empty.' }
    [ordered]@{ session_id=[string]$manifestObject.session_id; control_session_id=$controlSessionId; duration_s=$DurationSeconds; observations=$obsPath; truth=$truthPath; debug_telemetry=[bool]$DebugTelemetry; scene_ack=($ack -join "`n") } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $EvidenceRoot 'session_result.json') -Encoding UTF8
} finally {
    Stop-LinuxBridgeByToken $bridgeToken
    if ($null -ne $bridge) { Stop-ProcessTree ([int]$bridge.Id) }
    if ($null -ne $simulator) { Stop-ProcessTree ([int]$simulator.Id) }
    foreach ($entry in $oldEnv.GetEnumerator()) {
        if ($null -eq $entry.Value) { Remove-Item "Env:$($entry.Key)" -ErrorAction SilentlyContinue } else { Set-Item "Env:$($entry.Key)" $entry.Value }
    }
    if ($null -ne $sessionMutex) {
        $sessionMutex.ReleaseMutex()
        $sessionMutex.Dispose()
    }
}
