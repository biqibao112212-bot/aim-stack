[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $root)
& (Join-Path $PSScriptRoot 'check-consumer-boundary.ps1')
$lock = Get-Content -LiteralPath (Join-Path $root 'simulator.lock.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$releaseRoot = Join-Path $workspace $lock.simulator.release_relative_to_workspace
$releaseContract = Join-Path $releaseRoot 'release.json'
$releaseManifest = Join-Path $releaseRoot 'release-manifest.json'
$sdkContract = Join-Path $releaseRoot 'docs\sdk-contract.json'
$sdkConfig = Join-Path $releaseRoot 'sdk\lib\cmake\DaedalusSimSdk\DaedalusSimSdkConfig.cmake'

if (-not (Test-Path -LiteralPath $releaseContract)) { throw "Missing simulator release: $releaseContract" }
if (-not (Test-Path -LiteralPath $releaseManifest)) { throw "Missing simulator release manifest: $releaseManifest" }
if (-not (Test-Path -LiteralPath $sdkContract)) { throw "Missing simulator SDK contract: $sdkContract" }
if (-not (Test-Path -LiteralPath $sdkConfig)) { throw "Missing simulator SDK: $sdkConfig" }
$release = Get-Content -LiteralPath $releaseContract -Raw -Encoding UTF8 | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $releaseManifest -Raw -Encoding UTF8 | ConvertFrom-Json
$sdk = Get-Content -LiteralPath $sdkContract -Raw -Encoding UTF8 | ConvertFrom-Json
if ($release.version -ne $lock.simulator.version -or
    $manifest.version -ne $lock.simulator.version -or
    $manifest.source_commit -ne $lock.simulator.source_commit -or
    $release.sdk_version -ne $lock.simulator.sdk_version -or
    $release.shm_version -ne $lock.simulator.shm_version -or
    $release.sdk_abi_revision -ne $lock.simulator.sdk_abi_revision -or
    $release.image.format -ne $lock.simulator.image.format -or
    $release.image.width -ne $lock.simulator.image.width -or
    $release.image.height -ne $lock.simulator.image.height -or
    $release.physics_hz -ne $lock.simulator.physics_hz -or
    $release.default_image_transport -ne $lock.simulator.image_transport -or
    $release.tcp_image_port -ne $lock.simulator.tcp_image_port -or
    $release.udp_command_port -ne $lock.simulator.udp_command_port -or
    $release.scene_control_protocol -ne $lock.simulator.scene_control_protocol -or
    $release.scene_control_port -ne $lock.simulator.scene_control_port -or
    $sdk.sdk_version -ne $lock.simulator.sdk_version -or
    $sdk.shm_version -ne $lock.simulator.shm_version -or
    $sdk.sdk_abi_revision -ne $lock.simulator.sdk_abi_revision -or
    $sdk.meta_size -ne $lock.simulator.metadata_size -or
    $sdk.image_width -ne $lock.simulator.image.width -or
    $sdk.image_height -ne $lock.simulator.image.height -or
    $sdk.image_channels -ne 3 -or
    $sdk.tcp_image_protocol -ne $lock.simulator.tcp_image_protocol -or
    $sdk.tcp_image_port -ne $lock.simulator.tcp_image_port -or
    $sdk.udp_command_port -ne $lock.simulator.udp_command_port -or
    $sdk.scene_control_protocol -ne $lock.simulator.scene_control_protocol -or
    $sdk.scene_control_port -ne $lock.simulator.scene_control_port) {
    throw 'simulator.lock.json is incompatible with the installed release.'
}

$modelManifest = Get-Content -LiteralPath (Join-Path $root 'models\manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$assetRoot = Join-Path $workspace $modelManifest.asset_root_relative_to_workspace
foreach ($asset in $modelManifest.assets) {
    $path = Join-Path $assetRoot $asset.name
    if (-not (Test-Path -LiteralPath $path)) { throw "Protected model is missing: $path" }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($hash -ne $asset.sha256) { throw "Protected model hash mismatch: $path" }
}

$contextFiles = @(Get-ChildItem -LiteralPath (Join-Path $root 'agent-team') -Force -File)
$contextDirs = @(Get-ChildItem -LiteralPath (Join-Path $root 'agent-team') -Force -Directory)
if ($contextFiles.Count -ne 3 -or $contextDirs.Count -ne 0) {
    throw 'Active Agent Team context must contain exactly three files and no directories.'
}
"architecture_ok simulator=$($release.version) source=$($manifest.source_commit) sdk=$($release.sdk_version) protected_models=$($modelManifest.assets.Count)"
