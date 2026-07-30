[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$EvidenceRoot,
    [int]$DurationSeconds = 30,
    [switch]$Visible,
    [switch]$RebuildBridge,
    [switch]$DebugTelemetry,
    [switch]$ValidateManifestOnly
)

$ErrorActionPreference = 'Stop'
# wslpath emits UTF-8 paths; force the PowerShell subprocess encoding so the
# Chinese workspace path is not mojibaked when the runner is backgrounded.
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$repo = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $repo)
$manifestObject = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$manifestSha256 = (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
$schemaVersion = [string]$manifestObject.schema_version
if ($schemaVersion -notin @('stage3-manifest-v1','stage3-multistate-manifest-v2')) {
    throw "Unsupported Stage3 manifest schema: $schemaVersion"
}
if ([string]$manifestObject.session_id -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,89}$') {
    throw 'session_id must be 1..90 safe ASCII characters so the run-scoped control ID remains bounded.'
}

function Convert-Stage3FiniteDouble($value, [string]$name) {
    if ($null -eq $value) { throw "Manifest missing $name" }
    $number = [double]$value
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) {
        throw "$name must be finite."
    }
    return $number
}

function Convert-Stage3Int64Scalar($value, [string]$name) {
    $items = @($value)
    if ($items.Count -ne 1) { throw "$name must contain exactly one integer value; got $($items.Count)." }
    try { return [Convert]::ToInt64($items[0], [Globalization.CultureInfo]::InvariantCulture) }
    catch { throw "$name is not a valid Int64: $($items[0])" }
}

$distanceM = Convert-Stage3FiniteDouble $manifestObject.distance_m 'distance_m'
$initialYawRad = Convert-Stage3FiniteDouble $manifestObject.initial_yaw_rad 'initial_yaw_rad'
if ($distanceM -lt 1 -or $distanceM -gt 6.5) {
    throw 'distance_m must be in [1,6.5] so the camera-to-target range stays below 7 m.'
}
if ($null -ne $manifestObject.camera_profile -and [string]$manifestObject.camera_profile -ne 'wide_6mm') {
    throw 'Stage3 collection is fixed to the wide_6mm camera profile.'
}
if ($null -ne $manifestObject.dual_focal -and [bool]$manifestObject.dual_focal) {
    throw 'Stage3 collection forbids dual focal capture.'
}

function Assert-Stage3Motion($motion, [string]$prefix) {
    foreach ($field in 'mode','direction_deg','linear_speed_mps','linear_span_m','spin_rad_s') {
        if ($null -eq $motion.$field) { throw "$prefix missing $field" }
    }
    $mode = [string]$motion.mode
    if ($mode -notin @('stationary','linear','spin','linear_and_spin')) {
        throw "$prefix has invalid mode."
    }
    $direction = Convert-Stage3FiniteDouble $motion.direction_deg "$prefix.direction_deg"
    $speed = Convert-Stage3FiniteDouble $motion.linear_speed_mps "$prefix.linear_speed_mps"
    $span = Convert-Stage3FiniteDouble $motion.linear_span_m "$prefix.linear_span_m"
    $spin = Convert-Stage3FiniteDouble $motion.spin_rad_s "$prefix.spin_rad_s"
    if ($direction -lt -360 -or $direction -gt 360) { throw "$prefix.direction_deg must be in [-360,360]." }
    if ($speed -lt 0 -or $speed -gt 3) { throw "$prefix.linear_speed_mps must be in [0,3]." }
    if ($span -lt 0 -or $span -gt 8) { throw "$prefix.linear_span_m must be in [0,8]." }
    if ([math]::Abs($spin) -gt 15) { throw "$prefix.spin_rad_s must have abs <=15." }
    $hasLinear = $speed -gt 0 -or $span -gt 0
    $hasSpin = [math]::Abs($spin) -gt 0
    $consistent =
        ($mode -eq 'stationary' -and -not $hasLinear -and -not $hasSpin) -or
        ($mode -eq 'linear' -and $speed -gt 0 -and $span -gt 0 -and -not $hasSpin) -or
        ($mode -eq 'spin' -and -not $hasLinear -and $hasSpin) -or
        ($mode -eq 'linear_and_spin' -and $speed -gt 0 -and $span -gt 0 -and $hasSpin)
    if (-not $consistent) { throw "$prefix motion parameters are inconsistent with mode=$mode." }
}

function Assert-Stage3CaptureEnvelope($motion, [double]$distance) {
    if ($motion.mode -notin @('linear','linear_and_spin')) { return }

    # Manifest distance is chassis-referenced while collection quality is
    # camera-referenced. Keep a conservative 0.5 m reserve below the requested
    # 7 m camera limit, and include the existing 0.10 s truth-velocity gimbal
    # lead when checking both ends of the reciprocal path.
    $halfExtent = 0.5 * [double]$motion.linear_span_m
    $commandLeadExtent = 0.10 * [double]$motion.linear_speed_mps
    $extent = $halfExtent + $commandLeadExtent
    $heading = [double]$motion.direction_deg * [math]::PI / 180.0
    $axisLateral = [math]::Sin($heading)
    $axisForward = [math]::Cos($heading)
    $maxNominalRangeM = 0.0
    $minForwardM = [double]::PositiveInfinity
    $maxAbsYawDeg = 0.0

    foreach ($sign in @(-1.0, 1.0)) {
        $lateral = $sign * $extent * $axisLateral
        $forward = $distance + $sign * $extent * $axisForward
        $range = [math]::Sqrt($lateral * $lateral + $forward * $forward)
        $yawDeg = if ($forward -gt 0.0) {
            [math]::Abs([math]::Atan2($lateral, $forward) * 180.0 / [math]::PI)
        } else {
            180.0
        }
        $maxNominalRangeM = [math]::Max($maxNominalRangeM, $range)
        $minForwardM = [math]::Min($minForwardM, $forward)
        $maxAbsYawDeg = [math]::Max($maxAbsYawDeg, $yawDeg)
    }

    if ($maxNominalRangeM -gt 6.5 -or $minForwardM -lt 0.75 -or $maxAbsYawDeg -gt 75.0) {
        $message = 'Stage3 trajectory leaves the safe 6 mm capture envelope: ' +
            'max_nominal_range_m={0:F3} (limit 6.5), min_forward_m={1:F3} (limit 0.75), ' +
            'max_abs_yaw_deg={2:F3} (limit 75). Resample direction/distance/span.'
        throw ($message -f $maxNominalRangeM, $minForwardM, $maxAbsYawDeg)
    }
}

$segments = @()
if ($schemaVersion -eq 'stage3-manifest-v1') {
    if ($DurationSeconds -le 0) { throw 'DurationSeconds must be positive.' }
    Assert-Stage3Motion $manifestObject 'manifest'
    Assert-Stage3CaptureEnvelope $manifestObject $distanceM
    $segments = @([pscustomobject][ordered]@{
        segment_index = 0
        mode = [string]$manifestObject.mode
        direction_deg = [double]$manifestObject.direction_deg
        linear_speed_mps = [double]$manifestObject.linear_speed_mps
        linear_span_m = [double]$manifestObject.linear_span_m
        spin_rad_s = [double]$manifestObject.spin_rad_s
        duration_s = [double]$DurationSeconds
    })
    $captureDurationSeconds = [double]$DurationSeconds
} else {
    if ($PSBoundParameters.ContainsKey('DurationSeconds')) {
        throw 'DurationSeconds must be omitted for v2; every segment owns duration_s.'
    }
    if ([string]$manifestObject.mode -notin @('spin','linear_and_spin')) {
        throw 'v2 mode must name the rotation or combined motion family.'
    }
    $segments = @($manifestObject.segments)
    if ($segments.Count -lt 3) { throw 'v2 requires at least three motion segments.' }
    $captureDurationSeconds = 0.0
    for ($index = 0; $index -lt $segments.Count; $index++) {
        $segment = $segments[$index]
        if ([int]$segment.segment_index -ne $index) { throw "segment[$index] has a non-contiguous segment_index." }
        Assert-Stage3Motion $segment "segment[$index]"
        if ($segment.mode -ne 'stationary' -and [string]$segment.mode -ne [string]$manifestObject.mode) {
            throw "segment[$index] does not belong to the declared motion family."
        }
        Assert-Stage3CaptureEnvelope $segment $distanceM
        $duration = Convert-Stage3FiniteDouble $segment.duration_s "segment[$index].duration_s"
        if ($duration -le 0) { throw "segment[$index].duration_s must be positive." }
        $captureDurationSeconds += $duration
    }
    $manifestDuration = Convert-Stage3FiniteDouble $manifestObject.duration_s 'duration_s'
    if ([math]::Abs($manifestDuration - $captureDurationSeconds) -gt 1e-6) {
        throw 'duration_s must exactly equal the sum of segment durations.'
    }
}
if ($ValidateManifestOnly) {
    Write-Output ('stage3_manifest_ok session={0} schema={1} segments={2} duration_s={3:F3} camera=wide_6mm dual_focal=false' -f [string]$manifestObject.session_id,$schemaVersion,$segments.Count,$captureDurationSeconds)
    return
}

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
$distance = $distanceM.ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
$yaw = $initialYawRad.ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
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

function Get-Stage3StreamSize([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return [long]0 }
    return [long](Get-Item -LiteralPath $path).Length
}

function Invoke-Stage3MotionCommand(
    $segment,
    [ValidateSet('initialize','update')][string]$commandKind,
    [string]$sceneHost,
    [string]$sceneCliWsl,
    [string]$controlSessionId
) {
    $isInitialize = $commandKind -eq 'initialize'
    $mode = [string]$segment.mode
    $direction = ([double]$segment.direction_deg).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
    $linearSpeed = ([double]$segment.linear_speed_mps).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
    $span = ([double]$segment.linear_span_m).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
    $spinDeg = ([double]$segment.spin_rad_s * 180.0 / [math]::PI).ToString('0.############',[Globalization.CultureInfo]::InvariantCulture)
    $verb = if ($isInitialize) { '--stage3' } else { '--stage3-update' }
    $command = "'$sceneCliWsl' $verb --host '$sceneHost' --session '$controlSessionId' --target 3 --mode '$mode' --direction-deg '$direction' --linear-speed-mps '$linearSpeed' --linear-span-m '$span' --spin-deg-s '$spinDeg'"
    # Motion updates are not retried inside a simulator session. Scene Control
    # v1 has no idempotency token, so a lost ACK after a successful apply could
    # otherwise create an unrecorded motion boundary. The outer manifest runner
    # may restart the whole session in a fresh run directory/control session.
    $stdout = @(& wsl.exe -d Ubuntu-OSTEP -- bash -lc $command)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Scene Control motion command failed without in-session retry (kind=$commandKind exit=$exitCode); discard and restart the whole session."
    }
    $acks = @()
    foreach ($line in $stdout) {
        if ([string]$line -notmatch '^\{') { continue }
        try { $acks += ([string]$line | ConvertFrom-Json) }
        catch { throw "Malformed Scene Control ACK JSON: $line" }
    }
    [string[]]$expected = if ($isInitialize) {
        @('create_session','set_scene','set_target_3_motion')
    } else {
        @('set_target_3_motion')
    }
    if ($acks.Count -ne $expected.Count) {
        throw "Scene Control returned $($acks.Count) ACKs; expected $($expected.Count)."
    }
    for ($index = 0; $index -lt $expected.Count; $index++) {
        if ([string]($acks[$index].operation) -ne $expected[$index] -or [string]($acks[$index].status) -ne 'ok') {
            throw "Unexpected Scene Control ACK at index ${index}: $($stdout -join '; ')"
        }
    }
    $marker = if ($isInitialize) { 'scene_control_stage3_ready' } else { 'scene_control_stage3_updated' }
    if (-not ($stdout | Where-Object { [string]$_ -like "$marker *" })) {
        throw "Scene Control success marker is missing: $marker"
    }
    $motionAck = $acks[-1]
    foreach ($field in 'command_id','applied_frame_seq','timestamp_ns') {
        if ($null -eq $motionAck.$field) { throw "Motion ACK missing $field." }
    }
    return [pscustomobject][ordered]@{
        command_id = Convert-Stage3Int64Scalar $motionAck.command_id 'motion_ack.command_id'
        applied_frame_seq = Convert-Stage3Int64Scalar $motionAck.applied_frame_seq 'motion_ack.applied_frame_seq'
        applied_timestamp_ns = Convert-Stage3Int64Scalar $motionAck.timestamp_ns 'motion_ack.timestamp_ns'
        raw_stdout = ($stdout -join "`n")
    }
}

$sessionMutex = [Threading.Mutex]::new($false, 'Local\AimStackStage3Capture')
if (-not $sessionMutex.WaitOne(0)) { throw 'Another Stage-3 session runner is already active.' }
Stop-StaleStage3LinuxBridges
Assert-PortFree TCP ([int]$lock.simulator.tcp_image_port)
Assert-PortFree UDP ([int]$lock.simulator.scene_control_port)
$oldEnv = @{}
foreach ($name in 'BEVY_ASSET_ROOT','TALOS_IPC_DIR','DAEDALUS_SCENE_CONTROL_BIND','DAEDALUS_SCENE_MODE','DAEDALUS_RANGE_ACTIVE_TARGET_NUMBER','DAEDALUS_RANGE_TARGET_DISTANCE_M','DAEDALUS_RANGE_TARGET3_INITIAL_YAW_RAD','DAEDALUS_RANGE_TARGET3_INITIAL_YAW_DEG','DAEDALUS_STATS_JSON','AIM_SIM_STAGE3_SESSION_ID','AIM_SIM_STAGE3_OBSERVATIONS','AIM_SIM_STAGE3_TRUTH','AIM_SIM_STAGE3_DISTANCE_M','AIM_SIM_STAGE3_TRUTH_GIMBAL','AIM_SIM_DUAL_FOCAL','AIM_SIM_SCENE_CONTROL_MODE','AIM_SIM_SCENE_CONTROL_HOST','AIM_SIM_IMAGE_TRANSPORT','AIM_SIM_DEBUG_BRIDGE_JSON','AIM_SIM_DEBUG_BRIDGE_JSONL','AIM_SIM_DEBUG_PIPELINE_JSON','AIM_SIM_DEBUG_PIPELINE_JSONL','DAEDALUS_BRIDGE_TOKEN','AIM_SIM_ARMOR_ENGINE','DAEDALUS_PERF_DISABLE_UI','WGPU_BACKEND','DAEDALUS_PREVIEW_ENABLED','DAEDALUS_PREVIEW_MAX_HZ') {
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
export AIM_SIM_DUAL_FOCAL=false
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
    $segmentResults = @()
    $previousAck = $null
    for ($segmentIndex = 0; $segmentIndex -lt $segments.Count; $segmentIndex++) {
        $segment = $segments[$segmentIndex]
        if ($simulator.HasExited) { throw "Simulator exited with $($simulator.ExitCode) before segment $segmentIndex." }
        if ($bridge.HasExited) { throw "Bridge exited with $($bridge.ExitCode) before segment $segmentIndex." }
        $commandParameters = @{
            segment = $segment
            commandKind = $(if ($segmentIndex -eq 0) { 'initialize' } else { 'update' })
            sceneHost = $sceneHost
            sceneCliWsl = $sceneCliWsl
            controlSessionId = $controlSessionId
        }
        $ack = Invoke-Stage3MotionCommand @commandParameters
        if ($null -ne $previousAck -and (
            $ack.applied_frame_seq -le $previousAck.applied_frame_seq -or
            $ack.applied_timestamp_ns -le $previousAck.applied_timestamp_ns)) {
            throw "Segment $segmentIndex Scene Control ACK is not strictly newer than the prior ACK."
        }
        $previousAck = $ack
        if ($segmentIndex -eq 0) {
            $dataReadyDeadline = [DateTime]::UtcNow.AddSeconds(60)
            $obsReady = $false; $truthReady = $false
            while ([DateTime]::UtcNow -lt $dataReadyDeadline) {
                if ($simulator.HasExited) { throw "Simulator exited with $($simulator.ExitCode) before data became ready." }
                if ($bridge.HasExited) { throw "Bridge exited with $($bridge.ExitCode) before data became ready." }
                $obsReady = (Get-Stage3StreamSize $obsPath) -gt 0
                $truthReady = (Get-Stage3StreamSize $truthPath) -gt 0
                if ($obsReady -and $truthReady) { break }
                Start-Sleep -Milliseconds 250
            }
            if (-not $obsReady -or -not $truthReady) { throw 'Timed out waiting for Stage3 observation/truth streams.' }
            $started = $true
        }
        $obsStartBytes = Get-Stage3StreamSize $obsPath
        $truthStartBytes = Get-Stage3StreamSize $truthPath
        $dwellStartedUtc = [DateTime]::UtcNow
        $deadline = $dwellStartedUtc.AddSeconds([double]$segment.duration_s)
        while ([DateTime]::UtcNow -lt $deadline) {
            if ($simulator.HasExited) { throw "Simulator exited with $($simulator.ExitCode) during segment $segmentIndex." }
            if ($bridge.HasExited) { throw "Bridge exited with $($bridge.ExitCode) during segment $segmentIndex." }
            Start-Sleep -Milliseconds 250
        }
        $obsEndBytes = Get-Stage3StreamSize $obsPath
        $truthEndBytes = Get-Stage3StreamSize $truthPath
        if ($obsEndBytes -le $obsStartBytes -or $truthEndBytes -le $truthStartBytes) {
            throw "Observation/truth streams did not both grow during segment $segmentIndex."
        }
        $segmentResults += [pscustomobject][ordered]@{
            motion_command_epoch = $segmentIndex
            segment_index = $segmentIndex
            mode = [string]$segment.mode
            direction_deg = [double]$segment.direction_deg
            linear_speed_mps = [double]$segment.linear_speed_mps
            linear_span_m = [double]$segment.linear_span_m
            spin_rad_s = [double]$segment.spin_rad_s
            requested_duration_s = [double]$segment.duration_s
            command_id = $ack.command_id
            applied_frame_seq = $ack.applied_frame_seq
            applied_timestamp_ns = $ack.applied_timestamp_ns
            dwell_started_utc = $dwellStartedUtc.ToString('o')
            dwell_finished_utc = [DateTime]::UtcNow.ToString('o')
            observation_start_bytes = $obsStartBytes
            observation_end_bytes = $obsEndBytes
            truth_start_bytes = $truthStartBytes
            truth_end_bytes = $truthEndBytes
            scene_ack = $ack.raw_stdout
        }
    }
    if (-not (Test-Path -LiteralPath $obsPath) -or (Get-Item -LiteralPath $obsPath).Length -eq 0) { throw 'Stage3 observation JSONL is missing or empty.' }
    if (-not (Test-Path -LiteralPath $truthPath) -or (Get-Item -LiteralPath $truthPath).Length -eq 0) { throw 'Stage3 truth JSONL is missing or empty.' }
    $lastTruthLine = Get-Content -LiteralPath $truthPath -Tail 1 -Encoding UTF8
    $lastTruth = $lastTruthLine | ConvertFrom-Json
    $captureEndTimestampNs = Convert-Stage3Int64Scalar $lastTruth.timestamp_ns 'truth.timestamp_ns'
    $lastAckTimestampNs = Convert-Stage3Int64Scalar $segmentResults[-1].applied_timestamp_ns 'last_motion_ack.applied_timestamp_ns'
    if ($captureEndTimestampNs -le $lastAckTimestampNs) {
        throw 'Final truth timestamp does not extend beyond the final motion ACK.'
    }
    [ordered]@{
        schema_version='stage3-session-result-v2'
        complete=$true
        session_id=[string]$manifestObject.session_id
        control_session_id=$controlSessionId
        manifest_schema_version=$schemaVersion
        captured_manifest_sha256=$manifestSha256
        segment_plan_sha256=$manifestSha256
        segment_count=$segmentResults.Count
        duration_s=$captureDurationSeconds
        capture_end_timestamp_ns=$captureEndTimestampNs
        observations=$obsPath
        truth=$truthPath
        camera_profile='wide_6mm'
        dual_focal=$false
        debug_telemetry=[bool]$DebugTelemetry
        motion_segments=$segmentResults
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $EvidenceRoot 'session_result.json') -Encoding UTF8
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
