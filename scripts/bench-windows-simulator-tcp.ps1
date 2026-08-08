[CmdletBinding()]
param(
  [int]$WarmupSeconds = 5,
  [int]$DurationSeconds = 30,
  [string]$EvidenceRoot
)

$ErrorActionPreference = 'Stop'
if ($WarmupSeconds -lt 0 -or $DurationSeconds -lt 5) {
  throw 'WarmupSeconds must be nonnegative and DurationSeconds must be at least 5.'
}

function Stop-ProcessTree([int]$RootPid) {
  $children = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ParentProcessId -eq $RootPid })
  foreach ($child in $children) { Stop-ProcessTree ([int]$child.ProcessId) }
  Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
}

$repo = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $repo)
$lock = Get-Content -LiteralPath (Join-Path $repo 'simulator.lock.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$release = Join-Path $workspace $lock.simulator.release_relative_to_workspace
$binary = Join-Path $release 'bin\daedalus.exe'
$python = 'D:\Anaconda\envs\yolov8\python.exe'
foreach ($path in @($binary, $python)) {
  if (-not (Test-Path -LiteralPath $path)) { throw "Required benchmark input is absent: $path" }
}
if (@(Get-NetTCPConnection -LocalPort $lock.simulator.tcp_image_port -ErrorAction SilentlyContinue).Count -gt 0) {
  throw "TCP port $($lock.simulator.tcp_image_port) is already occupied. Refusing to disturb an existing run."
}
if (-not $EvidenceRoot) {
  $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
  $EvidenceRoot = Join-Path $workspace "runtime\windows-autoaim-chain-benchmark-$stamp"
}
if (Test-Path -LiteralPath $EvidenceRoot) { throw "Evidence root already exists: $EvidenceRoot" }
$ipc = Join-Path $EvidenceRoot 'ipc'
New-Item -ItemType Directory -Force -Path $ipc | Out-Null

$env:BEVY_ASSET_ROOT = $release
$env:TALOS_IPC_DIR = $ipc
$env:WGPU_BACKEND = 'dx12'
$env:WGPU_POWER_PREF = 'high'
$env:DAEDALUS_PERF_DISABLE_UI = '1'
$env:DAEDALUS_CONFIG = 'config.performance.toml'
$env:DAEDALUS_TALOS_RGB_ONLY = '1'
$env:DAEDALUS_TALOS_CAPTURE_MAX_HZ = '200'
$env:DAEDALUS_TALOS_IMAGE_TRANSPORT = 'tcp'
$env:DAEDALUS_STATS_JSON = Join-Path $EvidenceRoot 'simulator.stats.json'
$env:PATH = "$(Join-Path $release 'bin');$env:PATH"

$collector = @'
import argparse, csv, json, math, socket, statistics, struct, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--host", required=True)
p.add_argument("--port", type=int, required=True)
p.add_argument("--warmup", type=float, required=True)
p.add_argument("--duration", type=float, required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
root = Path(args.out)
headers_path = root / "tcp_frame_headers.jsonl"
summary_path = root / "tcp_summary.json"
csv_path = root / "tcp_intervals.csv"
plot_path = root / "tcp_interval_histogram.png"

def recv_exact(sock, n):
    data = bytearray(n); view = memoryview(data); offset = 0
    while offset < n:
        got = sock.recv_into(view[offset:])
        if got == 0: raise ConnectionError("peer closed during frame")
        offset += got
    return data

deadline = time.monotonic() + 20.0
sock = None
last_error = None
while time.monotonic() < deadline:
    try:
        sock = socket.create_connection((args.host, args.port), timeout=2.0)
        sock.settimeout(10.0)
        break
    except OSError as exc:
        last_error = repr(exc); time.sleep(0.1)
if sock is None: raise RuntimeError(f"connect failed: {last_error}")

fmt = "!IHHHHIIIQQQQQ"
start = time.monotonic(); sample_start = start + args.warmup; sample_end = sample_start + args.duration
records = []; invalid_headers = 0; payload_bytes = 0; prior_seq = None; gaps = 0
with headers_path.open("w", encoding="utf-8") as log:
    while time.monotonic() < sample_end:
        wire = recv_exact(sock, 64)
        fields = struct.unpack(fmt, wire)
        magic, version, header_bytes, pixel_format, flags, width, height, nbytes, epoch, seq, capture_ns, r0, r1 = fields
        payload = recv_exact(sock, nbytes)
        received_ns = time.perf_counter_ns()
        valid = (magic == 0x54494D47 and version == 1 and header_bytes == 64 and pixel_format == 2 and flags == 0 and width == 1440 and height == 1080 and nbytes == 6220800 and epoch > 0 and seq > 0 and capture_ns > 0 and r0 == 0 and r1 == 0)
        if not valid: invalid_headers += 1
        if prior_seq is not None and seq > prior_seq + 1: gaps += seq - prior_seq - 1
        prior_seq = seq
        rec = {"received_perf_counter_ns": received_ns, "epoch": epoch, "source_sequence": seq, "capture_timestamp_ns": capture_ns, "payload_bytes": nbytes, "header_valid": valid}
        log.write(json.dumps(rec, separators=(",", ":")) + "\n")
        if time.monotonic() >= sample_start:
            records.append(rec); payload_bytes += nbytes
sock.close()

def quantile(sorted_values, p):
    if not sorted_values: return None
    return sorted_values[round((len(sorted_values) - 1) * p)]
receive_intervals = [(b["received_perf_counter_ns"] - a["received_perf_counter_ns"]) / 1e6 for a, b in zip(records, records[1:])]
capture_intervals = [(b["capture_timestamp_ns"] - a["capture_timestamp_ns"]) / 1e6 for a, b in zip(records, records[1:])]
def distribution(values):
    ordered = sorted(values)
    return {"count": len(values), "min": min(values) if values else None, "median": statistics.median(values) if values else None, "mean": statistics.fmean(values) if values else None, "p95": quantile(ordered, .95), "p99": quantile(ordered, .99), "max": max(values) if values else None}
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f); writer.writerow(["receive_interval_ms", "capture_interval_ms"])
    for index in range(max(len(receive_intervals), len(capture_intervals))):
        writer.writerow([receive_intervals[index] if index < len(receive_intervals) else "", capture_intervals[index] if index < len(capture_intervals) else ""])
plot_error = None
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5)); plt.hist(receive_intervals, bins=50, color="#2463a6", edgecolor="white")
    plt.xlabel("TCP receive interval (ms)"); plt.ylabel("frames"); plt.title("Windows TCP capture interval distribution")
    plt.tight_layout(); plt.savefig(plot_path, dpi=160); plt.close()
except Exception as exc:
    plot_error = repr(exc)
summary = {
  "schema_version": 1, "kind": "windows_simulator_tcp_capture_benchmark", "host": args.host, "port": args.port,
  "warmup_seconds": args.warmup, "duration_seconds": args.duration, "frames": len(records),
  "wall_fps": len(records) / args.duration, "payload_bytes": payload_bytes, "payload_mib_per_s": payload_bytes / args.duration / 1048576.0,
  "invalid_headers": invalid_headers, "source_sequence_gaps": gaps,
  "receive_interval_ms": distribution(receive_intervals), "capture_timestamp_interval_ms": distribution(capture_intervals),
  "raw_header_log": headers_path.name, "interval_csv": csv_path.name, "histogram": plot_path.name if plot_error is None else None, "histogram_error": plot_error,
  "scope": "native Windows simulator TCP acquisition only; it does not constitute a native Windows auto-aim B result"
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
'@

$simStdout = Join-Path $EvidenceRoot 'simulator.stdout.log'
$simStderr = Join-Path $EvidenceRoot 'simulator.stderr.log'
$simulator = $null
try {
  $simulator = Start-Process -FilePath $binary -WorkingDirectory $release -PassThru `
    -RedirectStandardOutput $simStdout -RedirectStandardError $simStderr
  $collector | & $python - --host 127.0.0.1 --port $lock.simulator.tcp_image_port `
    --warmup $WarmupSeconds --duration $DurationSeconds --out $EvidenceRoot | Tee-Object -FilePath (Join-Path $EvidenceRoot 'collector.stdout.log')
  if ($LASTEXITCODE -ne 0) { throw "TCP collector failed with exit code $LASTEXITCODE" }
} finally {
  if ($null -ne $simulator) { Stop-ProcessTree $simulator.Id }
}

$engineReport = Join-Path $workspace 'models\engines\windows\armor-0708-trt861-win-rtx4060-fp16.engine.benchmark.json'
if (Test-Path -LiteralPath $engineReport) { Copy-Item -LiteralPath $engineReport -Destination (Join-Path $EvidenceRoot 'engine_benchmark_reference.json') }
$nativeStatus = [ordered]@{
  status = 'blocked'
  scope = 'native Windows auto-aim B executable'
  reason = 'run-autoaim-b.ps1 invokes wsl.exe and modules/autoaim/CMakeLists.txt gates the TCP receiver/bridge under if(UNIX).'
  consequence = 'A complete Windows-only simulator -> TCP -> vivsionn/TRT -> PnP/fire-control -> Stage3 JSONL run cannot be launched from the current committed consumer.'
  evidence = @('scripts/run-autoaim-b.ps1', 'modules/autoaim/CMakeLists.txt')
} | ConvertTo-Json -Depth 4
Set-Content -LiteralPath (Join-Path $EvidenceRoot 'native_autoaim_status.json') -Value $nativeStatus -Encoding UTF8
$binaryHash = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
$runIndex = [ordered]@{
  simulator_release = $lock.simulator.version
  simulator_binary = $binary
  simulator_binary_sha256 = $binaryHash
  backend = 'dx12 performance mode'
  transport = 'localhost TCP RGBA32 1440x1080'
  engine_reference = if (Test-Path -LiteralPath $engineReport) { 'engine_benchmark_reference.json' } else { $null }
  raw_logs = @('simulator.stdout.log', 'simulator.stderr.log', 'collector.stdout.log', 'tcp_frame_headers.jsonl', 'tcp_intervals.csv')
  summaries = @('tcp_summary.json', 'native_autoaim_status.json')
} | ConvertTo-Json -Depth 4
Set-Content -LiteralPath (Join-Path $EvidenceRoot 'run_index.json') -Value $runIndex -Encoding UTF8
Write-Output "evidence_root=$EvidenceRoot"
