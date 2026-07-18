[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $root)
$lock = Get-Content -LiteralPath (Join-Path $root 'simulator.lock.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$releaseRoot = Join-Path $workspace $lock.simulator.release_relative_to_workspace
$releaseContract = Join-Path $releaseRoot 'release.json'
$sdkConfig = Join-Path $releaseRoot 'sdk\lib\cmake\DaedalusSimSdk\DaedalusSimSdkConfig.cmake'

if (-not (Test-Path -LiteralPath $releaseContract)) { throw "Missing simulator release: $releaseContract" }
if (-not (Test-Path -LiteralPath $sdkConfig)) { throw "Missing simulator SDK: $sdkConfig" }
$release = Get-Content -LiteralPath $releaseContract -Raw -Encoding UTF8 | ConvertFrom-Json
if ($release.version -ne $lock.simulator.version -or
    $release.sdk_version -ne $lock.simulator.sdk_version -or
    $release.shm_version -ne $lock.simulator.shm_version -or
    $release.sdk_abi_revision -ne $lock.simulator.sdk_abi_revision -or
    $release.image.width -ne $lock.simulator.image.width -or
    $release.image.height -ne $lock.simulator.image.height -or
    $release.scene_control_protocol -ne $lock.simulator.scene_control_protocol -or
    $release.scene_control_port -ne $lock.simulator.scene_control_port) {
    throw 'simulator.lock.json is incompatible with the installed release.'
}

$manifest = Get-Content -LiteralPath (Join-Path $root 'models\manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$assetRoot = Join-Path $workspace $manifest.asset_root_relative_to_workspace
foreach ($asset in $manifest.assets) {
    $path = Join-Path $assetRoot $asset.name
    if (-not (Test-Path -LiteralPath $path)) { throw "Protected model is missing: $path" }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($hash -ne $asset.sha256) { throw "Protected model hash mismatch: $path" }
}

if (Test-Path -LiteralPath (Join-Path $root 'Cargo.toml')) {
    throw 'Simulator Cargo.toml must not exist in the consumer repository root.'
}
$contextFiles = @(Get-ChildItem -LiteralPath (Join-Path $root 'agent-team') -Force -File)
$contextDirs = @(Get-ChildItem -LiteralPath (Join-Path $root 'agent-team') -Force -Directory)
if ($contextFiles.Count -ne 3 -or $contextDirs.Count -ne 0) {
    throw 'Active Agent Team context must contain exactly three files and no directories.'
}
if (-not (Select-String -LiteralPath (Join-Path $root 'modules\autoaim\CMakeLists.txt') -Pattern 'find_package\(DaedalusSimSdk 1 REQUIRED CONFIG\)' -Quiet)) {
    throw 'Auto-aim module does not consume DaedalusSimSdk 1.x.'
}

"architecture_ok simulator=$($release.version) sdk=$($release.sdk_version) protected_models=$($manifest.assets.Count)"
