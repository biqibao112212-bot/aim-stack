#!/usr/bin/env python3
"""Export lossless long-form corner evidence and exact empirical distributions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CORNERS = ("bl", "tl", "tr", "br")
SOURCES = ("raw", "refined")
ROW_KEY_FIELDS = ("session_id", "producer_epoch", "frame_seq", "timestamp_ns", "armor_index")
CONTEXT_FIELDS = (
    "session",
    "session_id",
    "producer_epoch",
    "frame_seq",
    "timestamp_ns",
    "armor_index",
    "truth_slot",
    "segment_index",
    "motion_mode",
    "target_detection_count",
    "detector_number",
    "detector_type",
    "detector_valid",
    "association_nearest_distance_px",
    "association_margin_px",
    "view_incidence_cos",
    "range_m",
    "projected_sqrt_area_px",
    "projected_width_px",
    "projected_height_px",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: str | float | int | None) -> float:
    number = float(value) if value not in (None, "") else math.nan
    return number if math.isfinite(number) else math.nan


def percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    location = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return sorted_values[lower]
    weight = location - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return {"count": 0}
    result: dict[str, float | int | None] = {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }
    for label, fraction in (
        ("p01", 0.01),
        ("p05", 0.05),
        ("p10", 0.10),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p95", 0.95),
        ("p99", 0.99),
    ):
        result[label] = percentile(ordered, fraction)
    return result


def sample_key(row: dict[str, str]) -> str:
    return "/".join(row[field] for field in ROW_KEY_FIELDS)


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = set(CONTEXT_FIELDS)
    for corner in CORNERS:
        required.update(
            {
                f"truth_{corner}_x_px",
                f"truth_{corner}_y_px",
                f"raw_{corner}_x_px",
                f"raw_{corner}_y_px",
                f"raw_{corner}_dx_px",
                f"raw_{corner}_dy_px",
                f"raw_{corner}_norm_px",
                f"refined_{corner}_x_px",
                f"refined_{corner}_y_px",
                f"refined_{corner}_dx_px",
                f"refined_{corner}_dy_px",
                f"refined_{corner}_norm_px",
                f"refinement_{corner}_dx_px",
                f"refinement_{corner}_dy_px",
                f"refinement_{corner}_norm_px",
            }
        )
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise RuntimeError(f"atlas rows are missing required fields: {missing}")
    keys = [sample_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("atlas row key is not unique")
    return rows, fieldnames


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def context(row: dict[str, str]) -> dict[str, object]:
    result: dict[str, object] = {field: row[field] for field in CONTEXT_FIELDS}
    result["sample_key"] = sample_key(row)
    return result


def build_long_rows(
    atlas_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    residual_rows: list[dict[str, object]] = []
    refinement_rows: list[dict[str, object]] = []
    for row in atlas_rows:
        base = context(row)
        for corner_index, corner in enumerate(CORNERS):
            truth_x = finite(row[f"truth_{corner}_x_px"])
            truth_y = finite(row[f"truth_{corner}_y_px"])
            raw_x = finite(row[f"raw_{corner}_x_px"])
            raw_y = finite(row[f"raw_{corner}_y_px"])
            refined_x = finite(row[f"refined_{corner}_x_px"])
            refined_y = finite(row[f"refined_{corner}_y_px"])
            refinement = {
                **base,
                "corner_index": corner_index,
                "corner": corner,
                "truth_x_px": truth_x,
                "truth_y_px": truth_y,
                "raw_x_px": raw_x,
                "raw_y_px": raw_y,
                "refined_x_px": refined_x,
                "refined_y_px": refined_y,
                "dx_px": finite(row[f"refinement_{corner}_dx_px"]),
                "dy_px": finite(row[f"refinement_{corner}_dy_px"]),
                "norm_px": finite(row[f"refinement_{corner}_norm_px"]),
            }
            refinement_rows.append(refinement)
            for source, observed_x, observed_y in (
                ("raw", raw_x, raw_y),
                ("refined", refined_x, refined_y),
            ):
                residual_rows.append(
                    {
                        **base,
                        "source": source,
                        "corner_index": corner_index,
                        "corner": corner,
                        "truth_x_px": truth_x,
                        "truth_y_px": truth_y,
                        "observed_x_px": observed_x,
                        "observed_y_px": observed_y,
                        "dx_px": finite(row[f"{source}_{corner}_dx_px"]),
                        "dy_px": finite(row[f"{source}_{corner}_dy_px"]),
                        "norm_px": finite(row[f"{source}_{corner}_norm_px"]),
                    }
                )
    return residual_rows, refinement_rows


def exact_empirical_rows(
    rows: list[dict[str, object]], group_fields: tuple[str, ...]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric in ("dx_px", "dy_px", "norm_px"):
        groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            value = float(row[metric])
            if math.isfinite(value):
                groups[tuple(row[field] for field in group_fields)].append(row)
        for group, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
            ordered = sorted(members, key=lambda item: (float(item[metric]), str(item["sample_key"])))
            count = len(ordered)
            for rank, member in enumerate(ordered, start=1):
                output.append(
                    {
                        **{field: value for field, value in zip(group_fields, group)},
                        "metric": metric,
                        "rank": rank,
                        "count": count,
                        "empirical_cdf": rank / count,
                        "empirical_survival": (count - rank + 1) / count,
                        "value_px": member[metric],
                        "sample_key": member["sample_key"],
                    }
                )
    return output


def summary_for(
    residual_rows: list[dict[str, object]], refinement_rows: list[dict[str, object]]
) -> dict[str, object]:
    per_corner: dict[str, object] = {}
    for corner in CORNERS:
        corner_refinement = [row for row in refinement_rows if row["corner"] == corner]
        corner_summary: dict[str, object] = {
            "refinement_shift": {
                metric: describe(float(row[metric]) for row in corner_refinement)
                for metric in ("dx_px", "dy_px", "norm_px")
            }
        }
        for source in SOURCES:
            selected = [
                row for row in residual_rows
                if row["corner"] == corner and row["source"] == source
            ]
            corner_summary[f"{source}_truth_residual"] = {
                metric: describe(float(row[metric]) for row in selected)
                for metric in ("dx_px", "dy_px", "norm_px")
            }
        per_corner[corner] = corner_summary
    return {
        "schema_version": "corner-evidence-complete-distribution-v1",
        "atlas_rows": len(refinement_rows) // len(CORNERS),
        "corner_refinement_samples": len(refinement_rows),
        "corner_truth_residual_samples": len(residual_rows),
        "corner_order": list(CORNERS),
        "residual_convention": "observed_px - exact_projected_px_after_screen_canonicalization",
        "refinement_convention": "refined_px - raw_px",
        "per_corner": per_corner,
        "distribution_policy": (
            "CSV sample files retain every finite observation; empirical-distribution files retain "
            "every sorted value, sample key, rank, CDF and survival value. Summary statistics are indexes only."
        ),
    }


def report(summary: dict[str, object]) -> str:
    lines = [
        "# 四角点完整经验分布证据",
        "",
        "本资产不以均值或分位数替代原始分布。完整逐样本记录与排序后的经验分布均保留在 CSV 中。",
        "",
        f"- 原始观测行：{summary['atlas_rows']:,}",
        f"- 逐角点 raw->refined 位移记录：{summary['corner_refinement_samples']:,}",
        f"- 逐角点 raw/refined->truth 残差记录：{summary['corner_truth_residual_samples']:,}",
        "- 角点顺序：`bl, tl, tr, br`。",
        "- truth 只用于离线关联、精确投影和评分，不进入 detector、精修或 PnP 输入。",
        "",
        "## 完整分布文件",
        "",
        "- `corner_refinement_samples.csv`：每个观测、每个角点的 raw/refined 坐标及 dx/dy/模长。",
        "- `corner_refinement_empirical_distribution.csv`：四角点分别按 dx/dy/模长排序的全部值、样本键、秩、经验 CDF 和生存函数。",
        "- `corner_truth_residual_samples.csv`：raw 与 refined 相对 truth 投影的全部逐角点残差。",
        "- `corner_truth_residual_empirical_distribution.csv`：raw/refined、四角点、dx/dy/模长的完整经验分布。",
        "",
        "## 快速索引（不是证据替代）",
        "",
        "| corner | shift mean | shift P50 | shift P95 | shift max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    per_corner = summary["per_corner"]
    assert isinstance(per_corner, dict)
    for corner in CORNERS:
        values = per_corner[corner]["refinement_shift"]["norm_px"]
        lines.append(
            f"| {corner} | {values['mean']:.4f} | {values['p50']:.4f} | "
            f"{values['p95']:.4f} | {values['max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "上表只用于导航。任何后续判断都必须读取完整 sample/empirical-distribution 文件，不能只使用该表。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-rows", required=True, type=Path)
    parser.add_argument("--atlas-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    atlas_rows_path = args.atlas_rows.resolve()
    atlas_manifest_path = args.atlas_manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    atlas_rows, _ = read_rows(atlas_rows_path)
    residual_rows, refinement_rows = build_long_rows(atlas_rows)
    refinement_empirical = exact_empirical_rows(refinement_rows, ("corner",))
    residual_empirical = exact_empirical_rows(residual_rows, ("source", "corner"))

    context_names = ["sample_key", *CONTEXT_FIELDS]
    refinement_fields = [
        *context_names,
        "corner_index",
        "corner",
        "truth_x_px",
        "truth_y_px",
        "raw_x_px",
        "raw_y_px",
        "refined_x_px",
        "refined_y_px",
        "dx_px",
        "dy_px",
        "norm_px",
    ]
    residual_fields = [
        *context_names,
        "source",
        "corner_index",
        "corner",
        "truth_x_px",
        "truth_y_px",
        "observed_x_px",
        "observed_y_px",
        "dx_px",
        "dy_px",
        "norm_px",
    ]
    empirical_fields = [
        "source",
        "corner",
        "metric",
        "rank",
        "count",
        "empirical_cdf",
        "empirical_survival",
        "value_px",
        "sample_key",
    ]

    refinement_path = output / "corner_refinement_samples.csv"
    refinement_empirical_path = output / "corner_refinement_empirical_distribution.csv"
    residual_path = output / "corner_truth_residual_samples.csv"
    residual_empirical_path = output / "corner_truth_residual_empirical_distribution.csv"
    summary_path = output / "summary.json"
    report_path = output / "CORNER_EVIDENCE_REPORT.md"

    write_csv(refinement_path, refinement_fields, refinement_rows)
    write_csv(refinement_empirical_path, empirical_fields[1:], refinement_empirical)
    write_csv(residual_path, residual_fields, residual_rows)
    write_csv(residual_empirical_path, empirical_fields, residual_empirical)

    summary = summary_for(residual_rows, refinement_rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(report(summary), encoding="utf-8")

    script_path = Path(__file__).resolve()
    artifacts = (
        refinement_path,
        refinement_empirical_path,
        residual_path,
        residual_empirical_path,
        summary_path,
        report_path,
    )
    manifest = {
        "schema_version": "corner-evidence-complete-distribution-retention-v1",
        "protected": True,
        "inputs": {
            str(atlas_rows_path): sha256(atlas_rows_path),
            str(atlas_manifest_path): sha256(atlas_manifest_path),
        },
        "analysis_source": {str(script_path): sha256(script_path)},
        "outputs": {
            artifact.name: {
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            }
            for artifact in artifacts
        },
        "retention_class": (
            "protected lossless derived corner distributions; no automatic deletion"
        ),
    }
    (output / "retention_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
