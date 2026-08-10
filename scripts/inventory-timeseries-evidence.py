#!/usr/bin/env python3
"""Inventory retained Stage3 time-series, exact-join and association evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path


RUNTIME_ASSET = re.compile(
    r"^stage3-(formal|observation|temporal|future-query|rotation-observation|trajectory)|"
    r"^autoaim-b-(method-selection|trajectory|timeseries)|^autoaim-observation-rate",
    re.IGNORECASE,
)
FORMAL_SESSION = re.compile(r"^stage3-20260719-v1(?:-\d{4})?$")
DERIVED_EVENT_DATASET = re.compile(
    r"^stage3-dataset-v[23]-|^stage3-dataset-v4-observation-", re.IGNORECASE
)
INDEPENDENT_OBSERVATION = re.compile(r"^stage3-observation-v3-independent-", re.IGNORECASE)


def load_helpers(repo_root: Path):
    path = repo_root / "scripts" / "inventory-pnp-evidence.py"
    spec = importlib.util.spec_from_file_location("inventory_pnp_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def preload_hash_cache(helper, catalog: Path) -> int:
    inventory = catalog / "file_inventory.csv"
    if not inventory.is_file():
        return 0
    loaded = 0
    with inventory.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["absolute_path"])
            digest = row.get("sha256")
            if path.is_file() and digest:
                helper.HASH_CACHE[path.resolve()] = digest
                loaded += 1
    return loaded


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hash-cache-catalog", type=Path)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    dataset_root = args.dataset_root.resolve()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    helper, helper_path = load_helpers(repo_root)
    cache_rows = (
        preload_hash_cache(helper, args.hash_cache_catalog.resolve())
        if args.hash_cache_catalog
        else 0
    )

    selected_assets: list[tuple[str, Path, Path]] = []
    for child in runtime_root.iterdir():
        if child.resolve() != output and RUNTIME_ASSET.search(child.name):
            selected_assets.append(("runtime_history", child.resolve(), runtime_root))
    for child in dataset_root.iterdir():
        if FORMAL_SESSION.match(child.name):
            selected_assets.append(("formal_360_raw", child.resolve(), dataset_root))
        elif INDEPENDENT_OBSERVATION.match(child.name):
            selected_assets.append(("independent_observation_v3", child.resolve(), dataset_root))
    derived = dataset_root / "derived"
    for child in derived.iterdir():
        if DERIVED_EVENT_DATASET.search(child.name):
            selected_assets.append(("derived_event_contract", child.resolve(), dataset_root))

    selected_files: dict[Path, tuple[str, str, Path]] = {}
    for scope, asset, base in selected_assets:
        files = [asset] if asset.is_file() else [path for path in asset.rglob("*") if path.is_file()]
        for path in files:
            resolved = path.resolve()
            current = selected_files.get(resolved)
            if current is None:
                selected_files[resolved] = (scope, asset.name, base)
            elif scope not in current[0].split("|"):
                selected_files[resolved] = (f"{current[0]}|{scope}", current[1], current[2])

    reference_sources: dict[Path, set[Path]] = defaultdict(set)
    for source in sorted(selected_files, key=lambda path: str(path).lower()):
        if source.suffix.lower() != ".json":
            continue
        for reference in helper.referenced_paths(source, repo_root):
            reference_sources[reference].add(source)

    file_rows: list[dict] = []
    for path, (scope, asset, base) in sorted(
        selected_files.items(), key=lambda item: str(item[0]).lower()
    ):
        file_rows.append(
            {
                "scope": scope,
                "asset": asset,
                "absolute_path": str(path),
                "relative_path": str(path.relative_to(base)),
                "bytes": path.stat().st_size,
                "sha256": helper.sha256(path),
            }
        )

    reference_rows: list[dict] = []
    linked_files: set[Path] = set()
    for path, sources in sorted(reference_sources.items(), key=lambda item: str(item[0]).lower()):
        inside = path in selected_files
        exists = path.is_file()
        if exists and not inside:
            linked_files.add(path)
        reference_rows.append(
            {
                "absolute_path": str(path),
                "exists": exists,
                "already_in_selected_inventory": inside,
                "bytes": path.stat().st_size if exists else None,
                "sha256": helper.sha256(path) if exists else None,
                "referenced_by_count": len(sources),
                "referenced_by": " | ".join(str(source) for source in sorted(sources)),
            }
        )
    for path in sorted(linked_files, key=lambda value: str(value).lower()):
        file_rows.append(
            {
                "scope": "linked_source_file",
                "asset": "",
                "absolute_path": str(path),
                "relative_path": "",
                "bytes": path.stat().st_size,
                "sha256": helper.sha256(path),
            }
        )

    asset_totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"files": 0, "bytes": 0}
    )
    scope_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"assets": 0, "files": 0, "bytes": 0})
    for scope, asset, _ in selected_assets:
        scope_totals[scope]["assets"] += 1
    for path, (scope, asset, _) in selected_files.items():
        size = path.stat().st_size
        asset_totals[(scope, asset)]["files"] += 1
        asset_totals[(scope, asset)]["bytes"] += size
        scope_totals[scope]["files"] += 1
        scope_totals[scope]["bytes"] += size
    asset_rows = [
        {"scope": scope, "asset": asset, **counts}
        for (scope, asset), counts in sorted(asset_totals.items())
    ]

    inventory_path = output / "file_inventory.csv"
    references_path = output / "path_references.csv"
    assets_path = output / "asset_summary.csv"
    summary_path = output / "summary.json"
    report_path = output / "TIMESERIES_EVIDENCE_CATALOG.md"
    write_csv(
        inventory_path,
        ["scope", "asset", "absolute_path", "relative_path", "bytes", "sha256"],
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
    write_csv(assets_path, ["scope", "asset", "files", "bytes"], asset_rows)

    missing = [row for row in reference_rows if not row["exists"]]
    summary = {
        "schema_version": "stage3-timeseries-evidence-catalog-v1",
        "runtime_root": str(runtime_root),
        "dataset_root": str(dataset_root),
        "selection_policy": {
            "runtime": RUNTIME_ASSET.pattern,
            "formal_raw": FORMAL_SESSION.pattern,
            "derived_event_datasets": DERIVED_EVENT_DATASET.pattern,
            "independent_observation": INDEPENDENT_OBSERVATION.pattern,
        },
        "selected_assets": len(selected_assets),
        "selected_files": len(selected_files),
        "selected_bytes": sum(path.stat().st_size for path in selected_files),
        "linked_existing_source_files": len(linked_files),
        "linked_existing_source_bytes": sum(path.stat().st_size for path in linked_files),
        "path_references": len(reference_rows),
        "missing_path_references": len(missing),
        "missing_references": [str(row["absolute_path"]) for row in missing],
        "scope_totals": scope_totals,
        "preloaded_hash_cache_rows": cache_rows,
        "retention_policy": (
            "accepted, failed, invalid and superseded collection/event-contract evidence is retained; "
            "formal raw streams, point samples and shards are protected assets"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        "\n".join(
            [
                "# Stage3 因果时间序列历史证据目录",
                "",
                "目录覆盖 360-session 原始流、v2/v3/v4 事件数据契约、独立观测采集、",
                "120-run truth-gated 矩阵以及关联/处理历史。失败、无效和被取代资产不删除。",
                "",
                f"- selected assets: {summary['selected_assets']}",
                f"- selected files: {summary['selected_files']}",
                f"- selected bytes: {summary['selected_bytes']}",
                f"- linked existing sources: {summary['linked_existing_source_files']}",
                f"- missing references: {summary['missing_path_references']}",
                "",
                "`file_inventory.csv` 与 `path_references.csv` 是文件级权威；汇总不替代完整分布。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve()
    artifacts = [inventory_path, references_path, assets_path, summary_path, report_path]
    manifest = {
        "schema_version": "stage3-timeseries-evidence-catalog-retention-v1",
        "protected": True,
        "analysis_source": {
            str(script_path): helper.sha256(script_path),
            str(helper_path): helper.sha256(helper_path),
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": helper.sha256(path)}
            for path in artifacts
        },
        "retention_class": "protected Stage3 time-series evidence inventory",
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
