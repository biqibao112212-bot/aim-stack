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
    shooting_range = @('shooting_range', 'F8')
    energy_mechanism = @('energy', 'F9')
}
if (@($sim.native_operator_maps).Count -ne $expectedMaps.Count) {
    throw 'The learning simulator lock must expose exactly Shooting Range and Energy Mechanism.'
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
$researchCmake = Join-Path $root 'modules\autoaim-research\CMakeLists.txt'
if (-not (Test-Path -LiteralPath $researchCmake) -or
    -not (Select-String -LiteralPath $researchCmake -Pattern 'find_package\(DaedalusSimSdk 1 REQUIRED CONFIG\)' -Quiet)) {
    throw 'The sole internal auto-aim research implementation must consume DaedalusSimSdk 1.x.'
}
$implementationLockPath = Join-Path $root 'modules\autoaim-research\implementation.lock.json'
if (-not (Test-Path -LiteralPath $implementationLockPath -PathType Leaf)) {
    throw 'The internal auto-aim research implementation lock is missing.'
}
$implementationLock = Get-Content -LiteralPath $implementationLockPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($implementationLock.status -ne 'sole_internal_implementation' -or
    $implementationLock.autoaim.module -ne 'modules/autoaim-research' -or
    $implementationLock.simulator.exact_release -ne $sim.version -or
    $implementationLock.autoaim.upstream_commit -ne 'bd9f5e798fa3c6dd3b483ae6627796afb41c608d') {
    throw 'The internal auto-aim research lock is incomplete or conflicts with simulator.lock.json.'
}
$researchCmakeText = Get-Content -LiteralPath $researchCmake -Raw -Encoding UTF8
if ($researchCmakeText.Contains('../autoaim/') -or
    $researchCmakeText.Contains('aim_core_from_vivsionn') -or
    $researchCmakeText.Contains('YpdAngleTracker')) {
    throw 'The research baseline must not link the legacy auto-aim implementation.'
}

"consumer_boundary_ok guide=$($sim.consumer_guide.version) native_maps=$(@($sim.native_operator_maps).Count) implementation=$($implementationLock.autoaim.module)"
