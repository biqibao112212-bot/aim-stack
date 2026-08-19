#!/usr/bin/env python3
"""Collect a predeclared corner-repair matrix through Release-owned tools.

This orchestrator never parses the TCP or Scene Control wire.  It launches the
immutable simulator package, uses the package collector for full-frame capture,
uses the public SDK CLI for geometry/motion, and uses the package validator.
Every output is create-once and resumable at complete-session boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


SCHEMA = "aim-stack.corner-repair-collection/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--scene-cli", required=True, type=Path)
    parser.add_argument("--pose-capture", type=Path,
                        help="SDK-only single-client RGBA + exposure collector")
    parser.add_argument("--pose-qualifier", type=Path,
                        help="qualifier used with --pose-capture")
    parser.add_argument("--split", choices=("train", "validation", "test"), action="append")
    parser.add_argument("--session-id", action="append", help="collect only this declared session id")
    parser.add_argument("--timeout-s", type=float, default=90.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def write_new_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_manifest(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def wait_for_tcp_listener(process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"simulator exited before TCP listen: {process.returncode}")
        result = subprocess.run(["ss", "-ltn"], check=True, capture_output=True, text=True)
        if re.search(r":5602\b", result.stdout):
            return
        time.sleep(0.1)
    raise TimeoutError("simulator did not expose TCP 5602")


def wait_for_path(path: Path, processes: list[subprocess.Popen[bytes]], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"process exited before {path.name}: {process.returncode}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path}")


def parse_applied_frame(stdout: str) -> int:
    applied: list[int] = []
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        item = json.loads(line)
        if item.get("operation") in {"set_target_3_geometry", "set_target_3_motion"}:
            applied.append(int(item["applied_frame_seq"]))
    if len(applied) != 2:
        raise ValueError("scene CLI did not report geometry and motion applied_frame_seq")
    return max(applied) + 1


def stop_owned_process(process: subprocess.Popen[bytes], timeout_s: float) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def session_result(session_dir: Path, planned: dict[str, object], first_eligible: int,
                   validator_stdout: str, scene_stdout: str) -> dict[str, object]:
    capture = json.loads((session_dir / "capture-manifest.json").read_text(encoding="utf-8"))
    artifact_names = ["capture-manifest.json", "tcp-identities.jsonl", "exact-corners.jsonl"]
    artifact_names.extend(
        name for name in ("exposure-states.jsonl", "exposure-manifest.json")
        if (session_dir / name).exists()
    )
    return {
        "schema_version": "aim-stack.corner-repair-session-result/1",
        "planned": planned,
        "first_eligible_frame_seq": first_eligible,
        "capture": capture,
        "artifacts": {
            name: {"sha256": sha256(session_dir / name), "bytes": (session_dir / name).stat().st_size}
            for name in artifact_names
        },
        "scene_control_stdout": scene_stdout,
        "validator_stdout": validator_stdout,
        "test_accessed_for_model_selection": False,
    }


def collect_one(session_dir: Path, planned: dict[str, object], target_frames: int,
                release: Path, scene_cli: Path, timeout_s: float,
                pose_capture: Path | None, pose_qualifier: Path | None) -> dict[str, object]:
    session_dir.mkdir(parents=False)
    (session_dir / "talos-ipc").mkdir()
    simulator_log = (session_dir / "simulator.log").open("xb")
    collector_log = (session_dir / "collector.log").open("xb")
    simulator: subprocess.Popen[bytes] | None = None
    collector: subprocess.Popen[bytes] | None = None
    try:
        simulator = subprocess.Popen(
            [str(release / "start-simulator.sh"), "--ipc-dir", str(session_dir / "talos-ipc"),
             "--corner-labels-jsonl", str(session_dir / "exact-corners.jsonl")],
            stdout=simulator_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        wait_for_tcp_listener(simulator, timeout_s)
        if pose_capture is None:
            collector_command = [
                sys.executable, str(release / "docs/capture-corner-label-experiment.py"),
                "--output-dir", str(session_dir), "--until-eof", "--save-rgba-frames",
                "--motion-mode", str(planned["mode"]),
                "--direction-deg", str(planned["direction_deg"]),
                "--linear-speed-mps", str(planned["linear_speed_mps"]),
                "--linear-span-m", str(planned["linear_span_m"]),
                "--spin-deg-s", str(planned["spin_deg_s"]),
            ]
        else:
            collector_command = [
                str(pose_capture), "--ipc-dir", str(session_dir / "talos-ipc"),
                "--output-dir", str(session_dir), "--tcp-host", "127.0.0.1",
                "--tcp-port", "5602", "--until-eof",
            ]
        collector = subprocess.Popen(
            collector_command, stdout=collector_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        wait_for_path(session_dir / "collector-ready", [simulator, collector], timeout_s)
        scene_command = [
            str(scene_cli), "--stage3", "--host", "127.0.0.1",
            "--session", f"corner-repair-{planned['id']}", "--target", "3",
            "--mode", str(planned["mode"]), "--direction-deg", str(planned["direction_deg"]),
            "--linear-speed-mps", str(planned["linear_speed_mps"]),
            "--linear-span-m", str(planned["linear_span_m"]),
            "--spin-deg-s", str(planned["spin_deg_s"]),
            "--radial-scale", str(planned["radial_scale"]),
        ]
        scene = subprocess.run(scene_command, check=False, capture_output=True, text=True, timeout=timeout_s)
        scene_text = scene.stdout + scene.stderr
        write_new_text(session_dir / "scene-control.log", scene_text)
        if scene.returncode != 0:
            raise RuntimeError(f"scene control failed: {scene.returncode}")
        first_eligible = parse_applied_frame(scene.stdout)

        deadline = time.monotonic() + timeout_s
        frames_dir = session_dir / "frames"
        while time.monotonic() < deadline:
            if simulator.poll() is not None or collector.poll() is not None:
                raise RuntimeError("capture process exited before target frame count")
            count = sum(1 for _ in frames_dir.iterdir()) if frames_dir.exists() else 0
            if count >= target_frames:
                break
            time.sleep(0.1)
        else:
            raise TimeoutError(f"capture did not reach {target_frames} complete frames")

        stop_owned_process(simulator, 10.0)
        collector.wait(timeout=30.0)
        if collector.returncode != 0:
            raise RuntimeError(f"collector failed: {collector.returncode}")
        if pose_capture is not None:
            assert pose_qualifier is not None
            qualified = subprocess.run(
                [sys.executable, str(pose_qualifier), "--session-dir", str(session_dir)],
                check=False, capture_output=True, text=True, timeout=timeout_s,
            )
            write_new_text(session_dir / "pose-qualifier.log", qualified.stdout + qualified.stderr)
            if qualified.returncode != 0:
                raise RuntimeError(f"pose capture qualification failed: {qualified.returncode}")
        verifier = [
            sys.executable, str(release / "docs/verify-corner-label-export.py"),
            str(session_dir / "exact-corners.jsonl"), "--tcp-identities",
            str(session_dir / "tcp-identities.jsonl"), "--require-raw-frames", "--require-complete-z4",
        ]
        if planned["mode"] != "stationary":
            verifier.append("--require-uniform-and-excluded")
        verified = subprocess.run(verifier, check=False, capture_output=True, text=True, timeout=timeout_s)
        validator_text = verified.stdout + verified.stderr
        write_new_text(session_dir / "validator.log", validator_text)
        if verified.returncode != 0:
            raise RuntimeError(f"validator failed: {verified.returncode}")
        result = session_result(session_dir, planned, first_eligible, verified.stdout, scene.stdout)
        write_new_json(session_dir / "session-result.json", result)
        return result
    finally:
        if simulator is not None:
            stop_owned_process(simulator, 5.0)
        if collector is not None:
            stop_owned_process(collector, 5.0)
        simulator_log.close()
        collector_log.close()


def recover_validator_dependency_failure(session_dir: Path, planned: dict[str, object],
                                         release: Path, timeout_s: float) -> dict[str, object] | None:
    """Recover only a complete capture rejected because its Python lacked OpenCV."""
    failure_log = session_dir / "validator.log"
    required = (
        session_dir / "capture-manifest.json", session_dir / "tcp-identities.jsonl",
        session_dir / "exact-corners.jsonl", session_dir / "scene-control.log",
    )
    if not failure_log.exists() or not all(path.exists() for path in required):
        return None
    failure = failure_log.read_text(encoding="utf-8")
    if "OpenCV and NumPy are required" not in failure or "No module named 'cv2'" not in failure:
        return None
    scene_stdout = (session_dir / "scene-control.log").read_text(encoding="utf-8")
    first_eligible = parse_applied_frame(scene_stdout)
    verifier = [
        sys.executable, str(release / "docs/verify-corner-label-export.py"),
        str(session_dir / "exact-corners.jsonl"), "--tcp-identities",
        str(session_dir / "tcp-identities.jsonl"), "--require-raw-frames", "--require-complete-z4",
    ]
    if planned["mode"] != "stationary":
        verifier.append("--require-uniform-and-excluded")
    verified = subprocess.run(verifier, check=False, capture_output=True, text=True, timeout=timeout_s)
    validator_text = verified.stdout + verified.stderr
    write_new_text(session_dir / "validator-recovery.log", validator_text)
    if verified.returncode != 0:
        raise RuntimeError(f"recovered validator failed: {verified.returncode}")
    result = session_result(session_dir, planned, first_eligible, verified.stdout, scene_stdout)
    result["qualification_recovery"] = {
        "reason": "original orchestrator Python lacked OpenCV",
        "original_failure_log": str(failure_log),
        "successful_log": str(session_dir / "validator-recovery.log"),
        "python": sys.executable,
    }
    write_new_json(session_dir / "session-result.json", result)
    return result


def main() -> None:
    args = parse_args()
    plan_path = args.plan.resolve(strict=True)
    release = args.release.resolve(strict=True)
    scene_cli = args.scene_cli.resolve(strict=True)
    if (args.pose_capture is None) != (args.pose_qualifier is None):
        raise ValueError("--pose-capture and --pose-qualifier must be provided together")
    pose_capture = args.pose_capture.resolve(strict=True) if args.pose_capture else None
    pose_qualifier = args.pose_qualifier.resolve(strict=True) if args.pose_qualifier else None
    output_root = args.output_root.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "aim-stack.corner-repair-formal-plan/1":
        raise ValueError("unsupported plan schema")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    selected = set(args.split or ("train", "validation", "test"))
    if output_root.exists():
        bound = json.loads((output_root / "collection-plan.json").read_text(encoding="utf-8"))
        if bound["plan_sha256"] != sha256(plan_path):
            raise ValueError("existing output root is bound to a different plan")
    else:
        output_root.mkdir(parents=True)
        write_new_json(output_root / "collection-plan.json", {
            "schema_version": SCHEMA, "plan": str(plan_path), "plan_sha256": sha256(plan_path),
            "release": str(release), "release_json_sha256": sha256(release / "release.json"),
            "test_accessed_for_model_selection": False,
        })
    manifest_path = output_root / "collection-manifest.json"
    results: dict[str, object] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != sha256(plan_path):
            raise ValueError("existing collection manifest is bound to a different plan")
        results.update(existing.get("sessions", {}))
    requested_ids = set(args.session_id or ())
    declared_ids = {str(item["id"]) for item in plan["sessions"]}
    unknown_ids = requested_ids - declared_ids
    if unknown_ids:
        raise ValueError(f"unknown --session-id values: {sorted(unknown_ids)}")
    for planned in plan["sessions"]:
        if planned["split"] not in selected:
            continue
        if requested_ids and planned["id"] not in requested_ids:
            continue
        session_dir = output_root / planned["id"]
        result_path = session_dir / "session-result.json"
        if result_path.exists():
            results[planned["id"]] = json.loads(result_path.read_text(encoding="utf-8"))
            continue
        if session_dir.exists():
            recovered = recover_validator_dependency_failure(session_dir, planned, release, args.timeout_s)
            if recovered is None:
                raise FileExistsError(f"incomplete protected session requires manual audit: {session_dir}")
            results[planned["id"]] = recovered
            write_manifest(manifest_path, {
                "schema_version": SCHEMA, "plan_sha256": sha256(plan_path), "sessions": results,
                "test_accessed_for_model_selection": False,
            })
            continue
        print(f"collecting {planned['id']} split={planned['split']} mode={planned['mode']}", flush=True)
        results[planned["id"]] = collect_one(
            session_dir, planned, int(plan["target_complete_frames_per_session"]),
            release, scene_cli, args.timeout_s, pose_capture, pose_qualifier,
        )
        write_manifest(manifest_path, {
            "schema_version": SCHEMA, "plan_sha256": sha256(plan_path), "sessions": results,
            "test_accessed_for_model_selection": False,
        })
    print(json.dumps({"sessions_complete": len(results), "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
