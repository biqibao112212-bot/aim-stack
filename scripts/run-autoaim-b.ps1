[CmdletBinding()]
param([switch]$Visible, [switch]$RebuildBridge)

$ErrorActionPreference = 'Stop'
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
New-Item -ItemType Directory -Force -Path $ipcDir | Out-Null
$bridgeRoot = Join-Path $repo 'modules\autoaim'
$bridgeRootForWsl = $bridgeRoot.Replace('\','/')
$ipcDirForWsl = $ipcDir.Replace('\','/')
$sdkRootForWsl = $sdkRoot.Replace('\','/')
$modelForWsl = $model.Replace('\','/')
$bridgeRootWsl = (& wsl.exe -d Ubuntu-OSTEP -- wslpath -a -u $bridgeRootForWsl).Trim()
$ipcDirWsl = (& wsl.exe -d Ubuntu-OSTEP -- wslpath -a -u $ipcDirForWsl).Trim()
$sdkRootWsl = (& wsl.exe -d Ubuntu-OSTEP -- wslpath -a -u $sdkRootForWsl).Trim()
$modelWsl = (& wsl.exe -d Ubuntu-OSTEP -- wslpath -a -u $modelForWsl).Trim()
$forceRebuild = if ($RebuildBridge) { '1' } else { '0' }

$bridgeCommand = @'
set -euo pipefail
cd '__BRIDGE__'
export TALOS_IPC_DIR='__IPC__'
export DAEDALUS_SIM_SDK_ROOT='__SDK__'
export AIM_SIM_ARMOR_ENGINE='__MODEL__'
export AIM_SIM_WITH_VIVSIONN_TRT=ON
export AIM_SIM_ENABLE_UDP=ON
export AIM_SIM_IMAGE_TRANSPORT=tcp
export AIM_SIM_FORCE_REBUILD=__REBUILD__
exec bash scripts/run_talos_bridge_wsl.sh armor
'@.Replace('__BRIDGE__',$bridgeRootWsl).Replace('__IPC__',$ipcDirWsl).Replace('__SDK__',$sdkRootWsl).Replace('__MODEL__',$modelWsl).Replace('__REBUILD__',$forceRebuild)

$env:BEVY_ASSET_ROOT = $releaseRoot
$env:TALOS_IPC_DIR = $ipcDir
$env:WGPU_BACKEND = 'dx12'
$env:WGPU_POWER_PREF = 'high'
$env:DAEDALUS_CONFIG = 'config.performance.toml'
$env:DAEDALUS_TALOS_RGB_ONLY = '1'
$env:DAEDALUS_TALOS_CAPTURE_MAX_HZ = '200'
$env:DAEDALUS_TALOS_IMAGE_TRANSPORT = 'tcp'
$env:DAEDALUS_AUTO_AIM_ON_START = '1'
$env:PATH = "$(Join-Path $releaseRoot 'bin');$env:PATH"
if ($Visible) {
    Remove-Item Env:DAEDALUS_PERF_DISABLE_UI -ErrorAction SilentlyContinue
    $env:DAEDALUS_PREVIEW_ENABLED = '1'
    $env:DAEDALUS_PREVIEW_MAX_HZ = '60'
} else {
    $env:DAEDALUS_PERF_DISABLE_UI = '1'
    Remove-Item Env:DAEDALUS_PREVIEW_ENABLED -ErrorAction SilentlyContinue
    Remove-Item Env:DAEDALUS_PREVIEW_MAX_HZ -ErrorAction SilentlyContinue
}

$simulator = $null
$bridge = $null
try {
    $windowStyle = if ($Visible) { 'Normal' } else { 'Hidden' }
    $simulator = Start-Process -FilePath $binary -WorkingDirectory $releaseRoot -WindowStyle $windowStyle -PassThru
    Start-Sleep -Seconds 2
    $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bridgeCommand))
    $bridgeInfo = [Diagnostics.ProcessStartInfo]::new()
    $bridgeInfo.FileName = 'wsl.exe'
    $bridgeInfo.UseShellExecute = $false
    $bridgeInfo.Arguments = "-d Ubuntu-OSTEP -- bash -c `"echo $payload | base64 -d | bash`""
    $bridge = [Diagnostics.Process]::Start($bridgeInfo)
    Wait-Process -Id $simulator.Id
} finally {
    if ($null -ne $bridge -and -not $bridge.HasExited) { Stop-Process -Id $bridge.Id -Force -ErrorAction SilentlyContinue }
    if ($null -ne $simulator -and -not $simulator.HasExited) { Stop-Process -Id $simulator.Id -Force -ErrorAction SilentlyContinue }
}
