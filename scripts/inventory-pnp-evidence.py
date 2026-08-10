#!/usr/bin/env python3
"""Create a hash-bound inventory of retained PnP and observation-trajectory evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


EVIDENCE_NAME = re.compile(
    r"pnp|pose|trajectory|observation|arcflip|method-selection|autoaim-b-5m|"
    r"motion-comparison|combined-baseline|spin30|radius|rotation-observation|temporal-control",
    re.IGNORECASE,
)
ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
REPO_RELATIVE_PREFIXES = {"agent-team", "modules", "scripts", "training"}
REFERENCED_SUFFIXES = {
    ".bin",
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".npy",
    ".npz",
    ".onnx",
    ".pdf",
    ".png",
    ".ps1",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
}


HASH_CACHE: dict[Path, str] = {}


def sha256(path: Path) -> str:
    path = path.resolve()
    cached = HASH_CACHE.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    HASH_CACHE[path] = value
    return value


def walk_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_strings(child)


def referenced_paths(json_path: Path, repo_root: Path) -> Iterable[Path]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    found: set[Path] = set()
    for value in walk_strings(payload):
        if " --" in value or re.search(r"\.exe\s", value, re.IGNORECASE):
            continue
        value = value.replace("D:\\浠跨湡\\", "D:\\仿真\\")
        value = value.replace("D:/浠跨湡/", "D:/仿真/")
        candidate = Path(value)
        if candidate.suffix.lower() not in REFERENCED_SUFFIXES:
            continue
        if ABSOLUTE_WINDOWS_PATH.match(value):
            found.add(candidate.resolve())
        elif value.startswith(("./", "../", ".\\", "..\\")):
            found.add((repo_root / candidate).resolve())
        elif "/" in value or "\\" in value:
            first_part = candidate.parts[0].lower() if candidate.parts else ""
            base = repo_root if first_part in REPO_RELATIVE_PREFIXES else json_path.parent
            found.add((base / candidate).resolve())
    return sorted(found, key=lambda path: str(path).lower())


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    selected_roots = [
        child.resolve()
        for child in runtime_root.iterdir()
        if child.resolve() != output and EVIDENCE_NAME.search(child.name)
    ]
    selected_files: set[Path] = set()
    for root in selected_roots:
        if root.is_file():
            selected_files.add(root)
        elif root.is_dir():
            selected_files.update(path.resolve() for path in root.rglob("*") if path.is_file())

    reference_sources: dict[Path, set[Path]] = defaultdict(set)
    for source in sorted(selected_files, key=lambda path: str(path).lower()):
        if source.suffix.lower() != ".json":
            continue
        for reference in referenced_paths(source, repo_root):
            reference_sources[reference].add(source)

    file_rows: list[dict[str, object]] = []
    for path in sorted(selected_files, key=lambda value: str(value).lower()):
        file_rows.append(
            {
                "scope": "selected_runtime_evidence",
                "absolute_path": str(path),
                "relative_path": str(path.relative_to(runtime_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    reference_rows: list[dict[str, object]] = []
    linked_files: set[Path] = set()
    for path, sources in sorted(reference_sources.items(), key=lambda item: str(item[0]).lower()):
        inside_selected = path in selected_files
        exists = path.is_file()
        if exists and not inside_selected:
            linked_files.add(path)
        reference_rows.append(
            {
                "absolute_path": str(path),
                "exists": exists,
                "already_in_selected_inventory": inside_selected,
                "bytes": path.stat().st_size if exists else None,
                "sha256": sha256(path) if exists else None,
                "referenced_by_count": len(sources),
                "referenced_by": " | ".join(str(source) for source in sorted(sources)),
            }
        )

    for path in sorted(linked_files, key=lambda value: str(value).lower()):
        file_rows.append(
            {
                "scope": "linked_source_file",
                "absolute_path": str(path),
                "relative_path": "",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    directory_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for path in selected_files:
        relative = path.relative_to(runtime_root)
        top_level = relative.parts[0]
        directory_totals[top_level]["files"] += 1
        directory_totals[top_level]["bytes"] += path.stat().st_size
    directory_rows = [
        {"top_level_asset": name, **counts}
        for name, counts in sorted(directory_totals.items())
    ]

    inventory_path = output / "file_inventory.csv"
    references_path = output / "path_references.csv"
    directories_path = output / "directory_summary.csv"
    summary_path = output / "summary.json"
    report_path = output / "PNP_EVIDENCE_CATALOG.md"
    write_csv(
        inventory_path,
        ["scope", "absolute_path", "relative_path", "bytes", "sha256"],
        file_rows,
    )
    write_csv(
        references_path,
        [
            "absolute_path",
            "exists",
            "already_in_selected_inventory",
            "bytes",
            "sha256",
            "referenced_by_count",
            "referenced_by",
        ],
        reference_rows,
    )
    write_csv(directories_path, ["top_level_asset", "files", "bytes"], directory_rows)

    missing = [row for row in reference_rows if not row["exists"]]
    summary = {
        "schema_version": "pnp-evidence-catalog-v1",
        "runtime_root": str(runtime_root),
        "selection_regex": EVIDENCE_NAME.pattern,
        "selection_policy": (
            "top-level runtime assets related to PnP, pose, observation/trajectory, arc-flip, "
            "method selection, 5 m diagnostics, radius, rotation coverage or temporal controls"
        ),
        "path_normalization_policy": (
            "known historical D:\\\\浠跨湡 workspace mojibake is normalized to "
            "D:\\\\仿真 before existence checks; command-line strings are not treated as paths"
        ),
        "selected_top_level_assets": len(selected_roots),
        "selected_runtime_files": len(selected_files),
        "selected_runtime_bytes": sum(path.stat().st_size for path in selected_files),
        "linked_existing_source_files": len(linked_files),
        "linked_existing_source_bytes": sum(path.stat().st_size for path in linked_files),
        "path_references": len(reference_rows),
        "missing_path_references": len(missing),
        "missing_references": [str(row["absolute_path"]) for row in missing],
        "retention_policy": (
            "protected PnP evidence catalog; accepted, rejected, superseded, invalid and failed "
            "experiments remain indexed"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        "\n".join(
            [
                "# PnP 与整体观测轨迹历史证据目录",
                "",
                "本目录保留已采用、未采用、被取代、无效和失败的 PnP/轨迹相关资产。",
                "结论摘要不能替代逐文件哈希、原始行或完整分布。",
                "",
                f"- 顶层资产：{summary['selected_top_level_assets']}",
                f"- runtime 内文件：{summary['selected_runtime_files']}",
                f"- runtime 内字节：{summary['selected_runtime_bytes']}",
                f"- 当前存在的外部源文件：{summary['linked_existing_source_files']}",
                f"- 缺失路径引用：{summary['missing_path_references']}",
                "",
                "`file_inventory.csv` 保存所有现存文件的大小和 SHA-256；"
                "`path_references.csv` 同时保存存在与缺失的引用。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()
    artifacts = (inventory_path, references_path, directories_path, summary_path, report_path)
    manifest = {
        "schema_version": "pnp-evidence-catalog-retention-v1",
        "protected": True,
        "analysis_source": {str(script_path): sha256(script_path)},
        "outputs": {
            artifact.name: {
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            }
            for artifact in artifacts
        },
        "retention_class": "protected PnP and observation-trajectory evidence inventory",
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
