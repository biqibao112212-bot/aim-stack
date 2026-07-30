[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$EvidenceRoot,
    [int]$DurationSeconds = 30,
    [switch]$ValidateManifestOnly
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
$records = @()
$seenSessionIds = @{}
foreach ($line in Get-Content -LiteralPath $Manifest -Encoding UTF8) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $record = $line | ConvertFrom-Json
    $recordId = [string]$record.session_id
    if ([string]::IsNullOrWhiteSpace($recordId)) { throw 'Manifest contains an empty session_id.' }
    if ($seenSessionIds.ContainsKey($recordId)) { throw "Duplicate session_id in manifest: $recordId" }
    $seenSessionIds[$recordId] = $true
    $records += $record
}
if ($records.Count -eq 0) { throw 'Stage-3 manifest is empty.' }

foreach ($manifestObject in $records) {
    $id = [string]$manifestObject.session_id
    $evidence = Join-Path $EvidenceRoot $id
    $result = Join-Path $evidence 'session_result.json'
    $oneManifest = Join-Path $EvidenceRoot ('.manifest-' + $id + '.json')
    $manifestObject | ConvertTo-Json -Depth 12 -Compress | Set-Content -LiteralPath $oneManifest -Encoding UTF8
    $capturedManifestSha256 = (Get-FileHash -LiteralPath $oneManifest -Algorithm SHA256).Hash.ToLowerInvariant()
    if (Test-Path -LiteralPath $result) {
        $resultObject = Get-Content -LiteralPath $result -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not [bool]$resultObject.complete -or
            [string]$resultObject.captured_manifest_sha256 -ne $capturedManifestSha256) {
            throw "Stale or incomplete Stage-3 result for $id; use a new evidence root."
        }
        $expectedSegments = if ([string]$manifestObject.schema_version -eq 'stage3-multistate-manifest-v2') {
            @($manifestObject.segments).Count
        } else { 1 }
        if ([int]$resultObject.segment_count -ne $expectedSegments -or @($resultObject.motion_segments).Count -ne $expectedSegments) {
            throw "Stage-3 result segment plan mismatch for $id."
        }
        Write-Host "SKIP $index $id verified_plan=$capturedManifestSha256"
        $index++
        continue
    }
    Write-Host "RUN $index $id"
    $isMultistate = [string]$manifestObject.schema_version -eq 'stage3-multistate-manifest-v2'
    if ($ValidateManifestOnly) {
        if ($isMultistate) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner -Manifest $oneManifest -EvidenceRoot $evidence -ValidateManifestOnly
        } else {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner -Manifest $oneManifest -EvidenceRoot $evidence -DurationSeconds $DurationSeconds -ValidateManifestOnly
        }
        if ($LASTEXITCODE -ne 0) { throw "Stage-3 manifest validation failed: $id" }
        $index++
        continue
    }
    $attempt = 0
    do {
        $attempt++
        if ($isMultistate) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner -Manifest $oneManifest -EvidenceRoot $evidence
        } else {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner -Manifest $oneManifest -EvidenceRoot $evidence -DurationSeconds $DurationSeconds
        }
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
