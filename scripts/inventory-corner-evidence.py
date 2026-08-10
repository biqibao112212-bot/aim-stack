#!/usr/bin/env python3
"""Create a hash-bound inventory of retained corner-observation evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


EVIDENCE_NAME = re.compile(r"corner|fixed[-_]tilt|pnp[-_]joint", re.IGNORECASE)
ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
REPO_RELATIVE_PREFIXES = {"agent-team", "modules", "scripts", "training"}
REFERENCED_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".npy",
    ".npz",
    ".pdf",
    ".png",
    ".ps1",
    ".py",
    ".sh",
}
EXPLICIT_ROOTS = (
    "autoaim-b-arcflip-diag-20260809T013519Z",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


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

    selected_roots: list[Path] = []
    for child in runtime_root.iterdir():
        if child.resolve() == output:
            continue
        if EVIDENCE_NAME.search(child.name) or child.name in EXPLICIT_ROOTS:
            selected_roots.append(child.resolve())

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
    references_path = output / "absolute_path_references.csv"
    directories_path = output / "directory_summary.csv"
    summary_path = output / "summary.json"
    report_path = output / "CORNER_EVIDENCE_CATALOG.md"
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
        "schema_version": "corner-evidence-catalog-v1",
        "runtime_root": str(runtime_root),
        "selection_policy": (
            "top-level runtime assets whose names match corner, fixed-tilt or pnp-joint, "
            "plus explicit bounded full-pipeline arc-flip evidence"
        ),
        "selected_top_level_assets": len(selected_roots),
        "selected_runtime_files": len(selected_files),
        "selected_runtime_bytes": sum(path.stat().st_size for path in selected_files),
        "linked_existing_source_files": len(linked_files),
        "linked_existing_source_bytes": sum(path.stat().st_size for path in linked_files),
        "absolute_path_references": len(reference_rows),
        "missing_absolute_path_references": len(missing),
        "missing_references": [str(row["absolute_path"]) for row in missing],
        "retention_policy": "protected evidence catalog; rejected and superseded attempts are retained",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        "\n".join(
            [
                "# 四角点历史证据文件目录",
                "",
                "本目录保留已采用、未采用、被取代和失败的角点相关资产。结论索引不能替代文件级哈希。",
                "",
                f"- 顶层资产：{summary['selected_top_level_assets']}",
                f"- runtime 内已索引文件：{summary['selected_runtime_files']}",
                f"- runtime 内已索引字节：{summary['selected_runtime_bytes']}",
                f"- JSON/manifest 引用且当前存在的外部源文件：{summary['linked_existing_source_files']}",
                f"- 缺失的绝对路径引用：{summary['missing_absolute_path_references']}",
                "",
                "`file_inventory.csv` 保存每个现存文件的大小和 SHA-256；"
                "`absolute_path_references.csv` 同时保存现存与缺失引用。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()
    artifacts = (inventory_path, references_path, directories_path, summary_path, report_path)
    manifest = {
        "schema_version": "corner-evidence-catalog-retention-v1",
        "protected": True,
        "analysis_source": {str(script_path): sha256(script_path)},
        "outputs": {
            artifact.name: {
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            }
            for artifact in artifacts
        },
        "retention_class": "protected historical evidence inventory; no automatic deletion",
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
