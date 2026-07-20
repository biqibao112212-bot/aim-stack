"""Create deterministic, non-overwriting Stage-3 overfit/pilot selections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .build_dataset import _omega_band, _speed_band


MODE_ORDER = ("stationary", "linear", "spin", "linear_and_spin")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_tags(report: Mapping[str, Any]) -> set[str]:
    manifest = report["manifest"]
    omega = float(manifest.get("spin_rad_s", 0.0))
    mode = str(manifest.get("mode", ""))
    tags = {f"distance={int(manifest.get('distance_bin', 0))}"}
    if mode in {"linear", "linear_and_spin"}:
        tags.update({
            f"direction={int(manifest.get('direction_sector', 0))}",
            f"speed={_speed_band(float(manifest.get('linear_speed_mps', 0.0)))}",
        })
    if mode in {"spin", "linear_and_spin"}:
        tags.update({
            f"omega={_omega_band(omega)}",
            f"sign={'positive' if omega > 0 else 'negative'}",
        })
    return tags


def _stable_hash(seed: int, session_id: str) -> str:
    return hashlib.sha256(f"{seed}:{session_id}".encode("utf-8")).hexdigest()


def _choose_diverse(
    candidates: list[Mapping[str, Any]], count: int, seed: int, covered: set[str]
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < count:
        def key(report: Mapping[str, Any]):
            tags = _record_tags(report)
            new_count = len(tags - covered)
            return (-new_count, _stable_hash(seed, str(report["session_id"])))
        chosen = min(remaining, key=key)
        remaining.remove(chosen)
        selected.append(chosen)
        covered.update(_record_tags(chosen))
    if len(selected) != count:
        raise ValueError(f"could select only {len(selected)} of {count} requested sessions")
    return selected


def build_selection(dataset_dir: Path, purpose: str, seed: int) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("schema_version") != "stage3-dataset-v3":
        raise ValueError("selection requires stage3-dataset-v3")
    qualification_path = dataset_dir / str(dataset_manifest["qualification_report"])
    if _sha256(qualification_path) != dataset_manifest.get("artifact_sha256", {}).get("qualification_report"):
        raise ValueError("qualification report hash mismatch")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if not qualification.get("qualification_passed", False):
        raise ValueError("cannot select sessions from an unqualified dataset")
    sessions = [
        report for report in qualification["sessions"]
        if int(report["tensorization"].get("sample_count", 0)) > 0
    ]
    by_split_mode: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for split in ("train", "validation"):
        for mode in MODE_ORDER:
            by_split_mode[(split, mode)] = [
                report for report in sessions
                if report["split"] == split and report["manifest"].get("mode") == mode
            ]

    mode_quotas: dict[str, dict[str, int]] | None = None
    if purpose == "overfit1":
        candidates = by_split_mode[("train", "linear_and_spin")]
        def score(report: Mapping[str, Any]):
            manifest = report["manifest"]
            return (
                abs(float(manifest["distance_m"]) - 4.5) +
                abs(float(manifest["linear_speed_mps"]) - 1.5) +
                abs(abs(float(manifest["spin_rad_s"])) - 7.5) / 5.0,
                _stable_hash(seed, str(report["session_id"])),
            )
        selected = [min(candidates, key=score)]
        train = validation = [str(selected[0]["session_id"])]
        validation_source_split = "train"
    elif purpose == "overfit4":
        covered: set[str] = set()
        selected = []
        for mode in MODE_ORDER:
            selected.extend(_choose_diverse(by_split_mode[("train", mode)], 1, seed, covered))
        train = validation = sorted(str(report["session_id"]) for report in selected)
        validation_source_split = "train"
    elif purpose == "pilot24":
        quotas = {
            "train": {"stationary": 2, "linear": 4, "spin": 4, "linear_and_spin": 6},
            "validation": {"stationary": 1, "linear": 2, "spin": 2, "linear_and_spin": 3},
        }
        mode_quotas = quotas
        chosen_by_split: dict[str, list[Mapping[str, Any]]] = {"train": [], "validation": []}
        for split in ("train", "validation"):
            covered = set()
            for mode in MODE_ORDER:
                chosen_by_split[split].extend(_choose_diverse(
                    by_split_mode[(split, mode)], quotas[split][mode], seed, covered
                ))
        train = sorted(str(report["session_id"]) for report in chosen_by_split["train"])
        validation = sorted(str(report["session_id"]) for report in chosen_by_split["validation"])
        validation_source_split = "validation"
    else:
        raise ValueError(f"unsupported purpose: {purpose}")

    return {
        "schema_version": "stage3-session-selection-v1",
        "purpose": purpose,
        "seed": seed,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "train": train,
        "validation": validation,
        "validation_source_split": validation_source_split,
        "mode_quotas": mode_quotas,
        "selector_source_sha256": _sha256(Path(__file__).resolve()),
        "test": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--purpose", required=True, choices=["overfit1", "overfit4", "pilot24"])
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selection = build_selection(dataset, args.purpose, args.seed)
    output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
