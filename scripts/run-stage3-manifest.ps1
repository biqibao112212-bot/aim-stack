[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$EvidenceRoot,
    [int]$DurationSeconds = 30
)
$ErrorActionPreference = 'Continue'
$runner = Join-Path $PSScriptRoot 'run-stage3-session.ps1'
New-Item -ItemType Directory -Force -Path $EvidenceRoot | Out-Null
$lockPath = Join-Path $EvidenceRoot '.stage3-manifest.lock'
try {
    $lockStream = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None)
} catch {
    throw "Another Stage-3 manifest runner already owns $lockPath"
}
$index = 0
try {
foreach ($line in Get-Content -LiteralPath $Manifest -Encoding UTF8) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $manifestObject = $line | ConvertFrom-Json
    $id = [string]$manifestObject.session_id
    $evidence = Join-Path $EvidenceRoot $id
    $result = Join-Path $evidence 'session_result.json'
    if (Test-Path -LiteralPath $result) {
        Write-Host "SKIP $index $id"
        $index++
        continue
    }
    Write-Host "RUN $index $id"
    $oneManifest = Join-Path $EvidenceRoot ('.manifest-' + $id + '.json')
    $manifestObject | ConvertTo-Json -Depth 8 -Compress | Set-Content -LiteralPath $oneManifest -Encoding UTF8
    $attempt = 0
    do {
        $attempt++
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner -Manifest $oneManifest -EvidenceRoot $evidence -DurationSeconds $DurationSeconds
        $exitCode = $LASTEXITCODE
        if (Test-Path -LiteralPath $result) { break }
        if ($attempt -lt 3) { Start-Sleep -Seconds 3 }
    } while ($attempt -lt 3)
    Write-Host "EXIT $exitCode $id attempt=$attempt"
    if (-not (Test-Path -LiteralPath $result)) {
        throw "Stage-3 session failed after $attempt attempts: $id"
    }
    $index++
}
} finally {
    if ($null -ne $lockStream) { $lockStream.Dispose() }
}
