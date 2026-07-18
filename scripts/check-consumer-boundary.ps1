[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Require-File([string]$RelativePath) {
    $path = Join-Path $root $RelativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required consumer-boundary file is missing: $RelativePath"
    }
    return $path
}

$required = @(
    'AGENTS.md',
    'SIMULATOR_CONSUMER_GUIDE.md',
    'simulator.lock.json',
    'agent-team\PROJECT.md',
    'agent-team\BOARD.md',
    'agent-team\DECISIONS.md'
)
foreach ($relative in $required) { Require-File $relative | Out-Null }

$lock = Get-Content -LiteralPath (Join-Path $root 'simulator.lock.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$sim = $lock.simulator
if ($sim.consumer_guide.path -ne 'SIMULATOR_CONSUMER_GUIDE.md' -or
    $sim.consumer_guide.version -ne 1) {
    throw 'simulator.lock.json does not pin consumer guide v1.'
}

$expectedMaps = @{
    normal_map = @('armor', 'F7')
    shooting_range = @('shooting_range', 'F8')
    energy_mechanism = @('energy', 'F9')
}
if (@($sim.native_operator_maps).Count -ne $expectedMaps.Count) {
    throw 'The simulator lock must expose exactly three native operator maps.'
}
foreach ($map in $sim.native_operator_maps) {
    if (-not $expectedMaps.ContainsKey($map.id)) { throw "Unexpected native map: $($map.id)" }
    $expected = $expectedMaps[$map.id]
    if ($map.sdk_scene -ne $expected[0] -or $map.shortcut -ne $expected[1]) {
        throw "Native map pin mismatch: $($map.id)"
    }
}
if ($sim.runtime_profiles.default -ne 'high_performance' -or
    $sim.runtime_profiles.high_performance.visible -ne $false -or
    $sim.runtime_profiles.visible_acceptance.visible -ne $true -or
    $sim.runtime_profiles.visible_acceptance.preview_max_hz -ne 60) {
    throw 'Simulator runtime profile pins are incomplete or incompatible.'
}

$guide = Get-Content -LiteralPath (Join-Path $root 'SIMULATOR_CONSUMER_GUIDE.md') -Raw -Encoding UTF8
$agents = Get-Content -LiteralPath (Join-Path $root 'AGENTS.md') -Raw -Encoding UTF8
foreach ($token in @('AIM_SIMULATOR_CONSUMER_GUIDE_V1', 'SIMULATOR_CHANGE_APPROVAL_REQUIRED')) {
    if (-not $guide.Contains($token)) { throw "Consumer guide is missing policy token: $token" }
}
if (-not $agents.Contains('SIMULATOR_CHANGE_APPROVAL_REQUIRED')) {
    throw 'AGENTS.md is missing the simulator change approval gate.'
}

$tracked = @(git -C $root ls-files)
if ($LASTEXITCODE -ne 0) { throw 'Unable to enumerate tracked files.' }
$forbidden = @($tracked | Where-Object {
    $_ -match '(^|/)(target|build|dist|out)(/|$)' -or
    $_ -match '\.(engine|plan|trt|onnx|pt|pth|ckpt|exe|dll|pdb|obj|o|a|lib)$' -or
    $_ -match '(^|/)(talos_v1|tcp_image_v1|endpoints_v1)\.hpp$'
})
if ($forbidden.Count -ne 0) {
    throw "Forbidden simulator copy, protected model, or build artifact is tracked: $($forbidden -join ', ')"
}
if ($tracked -contains 'Cargo.toml') {
    throw 'A simulator Cargo root must not exist in the consumer repository.'
}
if (-not (Select-String -LiteralPath (Join-Path $root 'modules\autoaim\CMakeLists.txt') -Pattern 'find_package\(DaedalusSimSdk 1 REQUIRED CONFIG\)' -Quiet)) {
    throw 'Auto-aim must consume the published DaedalusSimSdk 1.x package.'
}

"consumer_boundary_ok guide=$($sim.consumer_guide.version) native_maps=$(@($sim.native_operator_maps).Count)"
