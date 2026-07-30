from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-stage3-session.ps1"


@unittest.skipUnless(shutil.which("powershell.exe"), "Windows PowerShell is required")
class Stage3SessionRunnerTest(unittest.TestCase):
    def _invoke_tail_reader(self, truth_path: Path) -> subprocess.CompletedProcess[str]:
        escaped_runner = str(RUNNER).replace("'", "''")
        escaped_truth = str(truth_path).replace("'", "''")
        command = rf"""
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    '{escaped_runner}', [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) {{ throw ($parseErrors | Out-String) }}
$wanted = @('Convert-Stage3Int64Scalar', 'Get-Stage3LatestCompleteTruthTimestamp')
$definitions = $ast.FindAll({{
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $wanted -contains $node.Name
}}, $true)
if ($definitions.Count -ne 2) {{ throw 'Stage3 tail-reader functions are missing.' }}
$definitions | ForEach-Object {{ Invoke-Expression $_.Extent.Text }}
Get-Stage3LatestCompleteTruthTimestamp '{escaped_truth}'
"""
        return subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_latest_complete_timestamp_ignores_incomplete_append_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            truth_path = Path(temporary_directory) / "truth.jsonl"
            truth_path.write_text(
                json.dumps({"timestamp_ns": 101})
                + "\n"
                + json.dumps({"timestamp_ns": 205})
                + "\n"
                + '{"timestamp_ns":',
                encoding="utf-8",
            )
            result = self._invoke_tail_reader(truth_path)

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("205", result.stdout.strip())

    def test_latest_timestamp_uses_file_order_not_numeric_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            truth_path = Path(temporary_directory) / "truth.jsonl"
            truth_path.write_bytes(b'{"timestamp_ns":200}\r\n{"timestamp_ns":100}\r\n')
            result = self._invoke_tail_reader(truth_path)

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("100", result.stdout.strip())

    def test_valid_but_unterminated_tail_is_not_committed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            truth_path = Path(temporary_directory) / "truth.jsonl"
            truth_path.write_bytes(
                b'{"timestamp_ns":101}\n{"timestamp_ns":205}'
            )
            result = self._invoke_tail_reader(truth_path)

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("101", result.stdout.strip())

    def test_malformed_committed_tail_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            truth_path = Path(temporary_directory) / "truth.jsonl"
            truth_path.write_bytes(b'{"timestamp_ns":101}\n{bad}\n')
            result = self._invoke_tail_reader(truth_path)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("not valid JSON", result.stderr)

    def test_complete_record_with_non_scalar_timestamp_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            truth_path = Path(temporary_directory) / "truth.jsonl"
            truth_path.write_text(
                json.dumps({"timestamp_ns": 101})
                + "\n"
                + json.dumps({"timestamp_ns": [205, 206]})
                + "\n",
                encoding="utf-8",
            )
            result = self._invoke_tail_reader(truth_path)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("integer scalar JSON value", result.stderr)

    def test_string_float_null_missing_and_top_level_array_are_rejected(self) -> None:
        invalid_records = (
            {"timestamp_ns": "205"},
            {"timestamp_ns": 205.5},
            {"timestamp_ns": None},
            {"frame_seq": 7},
            [205],
        )
        for record in invalid_records:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as temporary_directory:
                truth_path = Path(temporary_directory) / "truth.jsonl"
                truth_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                result = self._invoke_tail_reader(truth_path)
            self.assertNotEqual(0, result.returncode)

    def test_only_unterminated_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            truth_path = Path(temporary_directory) / "truth.jsonl"
            truth_path.write_text('{"timestamp_ns":205}', encoding="utf-8")
            result = self._invoke_tail_reader(truth_path)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("no LF-committed record", result.stderr)

    def test_bridge_is_stopped_before_truth_tail_is_read(self) -> None:
        runner_text = RUNNER.read_text(encoding="utf-8")
        reader_call = runner_text.index(
            "$captureEndTimestampNs = Get-Stage3LatestCompleteTruthTimestamp $truthPath"
        )
        stop_call = runner_text.rfind(
            "Stop-LinuxBridgeByToken $bridgeToken", 0, reader_call
        )
        self.assertGreater(stop_call, 0)


if __name__ == "__main__":
    unittest.main()
