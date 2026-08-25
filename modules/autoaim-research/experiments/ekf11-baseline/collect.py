#!/usr/bin/env python3
"""Collect the three locked 20 s EKF11 baseline scenarios.

Every scenario starts a fresh Daedalus 1.4.0 learning Release process in
headless high-performance mode. Target truth steers only the simulator gimbal;
the estimator receives images and same-exposure camera pose, never target truth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path


SCENARIOS = (
    {
        "id": "spin_8",
        "label": "stationary spin",
        "mode": "spin",
        "linear_speed_mps": 0.0,
        "spin_rad_s": 8.0,
    },
    {
        "id": "translate_1p5",
        "label": "linear translation",
        "mode": "linear",
        "linear_speed_mps": 1.5,
        "spin_rad_s": 0.0,
    },
    {
        "id": "translate_1_spin_6",
        "label": "translation and spin",
        "mode": "linear_and_spin",
        "linear_speed_mps": 1.0,
        "spin_rad_s": 6.0,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def set_target_motion(scenario: dict[str, object], target: int) -> dict:
    request = {
        "protocol": "daedalus.scene-control/2",
        "command_id": 9000,
        # Reuse the launcher-created SDK session. The simulator deliberately
        # rejects commands from a second undeclared session.
        "session_id": "contest-cpp-client",
        "op": "set_range_target_motion",
        "args": {
            "target": target,
            "mode": scenario["mode"],
            "direction_deg": 90.0,
            "linear_speed_mps": scenario["linear_speed_mps"],
            "linear_span_m": 8.0,
            "spin_deg_s": math.degrees(float(scenario["spin_rad_s"])),
        },
    }
    payload = json.dumps(request, separators=(",", ":")).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.settimeout(2.0)
        client.sendto(payload, ("127.0.0.1", 5603))
        response = json.loads(client.recv(65536).decode())
    if response.get("status") != "ok":
        raise RuntimeError(f"set_range_target_motion failed: {response}")
    return response


def validate_scenario(path: Path, scenario: dict[str, object], duration_s: float) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 2:
        raise RuntimeError(f"too few records in {path}")
    identities = {
        (row["producer_epoch"], row["frame_seq"], row["timestamp_ns"])
        for row in rows
    }
    if len(identities) != len(rows):
        raise RuntimeError(f"duplicate exposure identity in {path}")
    actual_duration_s = (rows[-1]["timestamp_ns"] - rows[0]["timestamp_ns"]) * 1e-9
    if actual_duration_s < duration_s:
        raise RuntimeError(
            f"timestamp window is short: {actual_duration_s:.6f} < {duration_s:.6f}"
        )
    truths = [row["truth"] for row in rows if isinstance(row.get("truth"), dict)]
    if len(truths) != len(rows):
        raise RuntimeError(f"missing same-exposure truth in {path}")
    speed = statistics.median(
        math.sqrt(sum(component * component for component in truth["velocity_mps"]))
        for truth in truths
    )
    omega = statistics.median(abs(truth["omega_rad_s"]) for truth in truths)
    expected_speed = float(scenario["linear_speed_mps"])
    expected_omega = abs(float(scenario["spin_rad_s"]))
    if abs(speed - expected_speed) > 0.02 or abs(omega - expected_omega) > 0.02:
        raise RuntimeError(
            f"motion truth mismatch for {scenario['id']}: "
            f"speed={speed}, omega={omega}"
        )
    return {
        "records": len(rows),
        "unique_exposure_identities": len(identities),
        "timestamp_duration_s": actual_duration_s,
        "truth_speed_mps_median": speed,
        "truth_abs_omega_rad_s_median": omega,
        "detected_records": sum(row.get("detection_count", 0) > 0 for row in rows),
        "ekf_records": sum(row.get("ekf_estimate") is not None for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--target", type=int, choices=(1, 3), default=3)
    args = parser.parse_args()
    if args.duration_s <= 0.0 or not math.isfinite(args.duration_s):
        parser.error("--duration-s must be finite and positive")
    if args.output_root.exists():
        parser.error(f"refusing to overwrite existing output root: {args.output_root}")

    module_root = Path(__file__).resolve().parents[2]
    repository = module_root.parents[1]
    workspace = repository.parents[1]
    release = workspace / "releases/daedalus-simulator/1.4.0-learning-r1/linux-x86_64"
    launcher = release / "daedalus-learning.sh"
    runner = repository / "build/autoaim-research/autoaim_research_runner"
    truth_gimbal = repository / "build/autoaim-research/autoaim_research_truth_gimbal"
    config = module_root / "config/research.yaml"
    lock = module_root / "implementation.lock.json"
    for required in (launcher, runner, truth_gimbal, config, lock, release / "release.json"):
        if not required.is_file():
            parser.error(f"required locked artifact is missing: {required}")

    raw_dir = args.output_root / "raw"
    runtime_root = args.output_root / "runtime"
    raw_dir.mkdir(parents=True)
    runtime_root.mkdir()
    started_at = dt.datetime.now(dt.timezone.utc)
    manifest: dict[str, object] = {
        "schema": "aim-stack.ekf11-baseline-collection/v1",
        "started_at_utc": started_at.isoformat(),
        "duration_requested_s": args.duration_s,
        "settle_s": args.settle_s,
        "platform": platform.platform(),
        "python": sys.version,
        "simulator": {
            "release": "1.4.0-learning-r1",
            "release_root": str(release),
            "release_json_sha256": sha256(release / "release.json"),
            "mode": "release_headless_high_performance",
            "visible_frontend": False,
            "scene": "shooting-range",
            "truth_gimbal_target": args.target,
        },
        "estimator": {
            "module": "modules/autoaim-research",
            "implementation_lock_sha256": sha256(lock),
            "runner": str(runner),
        },
        "identity": ["producer_epoch", "frame_seq", "timestamp_ns"],
        "scenarios": [],
    }

    for scenario in SCENARIOS:
        scenario_id = str(scenario["id"])
        runtime_dir = runtime_root / scenario_id
        output = raw_dir / f"{scenario_id}.jsonl"
        start = dt.datetime.now(dt.timezone.utc)
        controller = None
        controller_log_path = runtime_dir / "truth-gimbal.log"
        try:
            run(
                [
                    str(launcher),
                    "--runtime-dir",
                    str(runtime_dir),
                    "start",
                    "--performance",
                    "--scene",
                    "shooting-range",
                ],
            )
            motion_ack = set_target_motion(scenario, args.target)
            controller_log = controller_log_path.open("w", encoding="utf-8")
            controller = subprocess.Popen(
                [
                    str(truth_gimbal),
                    "--ipc-dir",
                    str(runtime_dir),
                    "--lookahead-s",
                    "0.03",
                    "--target",
                    str(args.target),
                ],
                stdout=controller_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            time.sleep(args.settle_s)
            if controller.poll() is not None:
                controller_log.close()
                raise RuntimeError(controller_log_path.read_text(encoding="utf-8"))
            run(
                [
                    str(runner),
                    "--config",
                    str(config),
                    "--ipc-dir",
                    str(runtime_dir),
                    "--output",
                    str(output),
                    "--duration-s",
                    str(args.duration_s),
                ]
            )
        finally:
            if controller is not None:
                controller.terminate()
                try:
                    controller.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    controller.kill()
                    controller.wait()
                controller_log.close()
            subprocess.run(
                [str(launcher), "--runtime-dir", str(runtime_dir), "stop"],
                check=False,
            )
        end = dt.datetime.now(dt.timezone.utc)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"scenario did not produce data: {scenario_id}")
        validation = validate_scenario(output, scenario, args.duration_s)
        simulator_log_path = runtime_dir / "daedalus-learning.log"
        if "Daedalus launch mode=performance" not in simulator_log_path.read_text(
            encoding="utf-8"
        ):
            raise RuntimeError(f"scenario was not headless performance mode: {scenario_id}")
        if "command_count=0" in controller_log_path.read_text(encoding="utf-8"):
            raise RuntimeError(f"truth gimbal sent no commands: {scenario_id}")
        manifest["scenarios"].append(
            {
                **scenario,
                "target": args.target,
                "linear_direction_deg": 90.0,
                "linear_span_m": 8.0,
                "started_at_utc": start.isoformat(),
                "finished_at_utc": end.isoformat(),
                "jsonl": str(output),
                "jsonl_bytes": output.stat().st_size,
                "jsonl_sha256": sha256(output),
                "simulator_log": str(simulator_log_path),
                "truth_gimbal_log": str(controller_log_path),
                "motion_ack": motion_ack,
                "validation": validation,
            }
        )

    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest_path = args.output_root / "collection-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"collection complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
