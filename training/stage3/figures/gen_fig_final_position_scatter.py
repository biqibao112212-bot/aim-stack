"""Draw one simple future-time versus final-position-error scatter plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate(query_file: str | Path, output_file: str | Path) -> Path:
    source = Path(query_file).resolve()
    output = Path(output_file).resolve()
    with np.load(source) as data:
        tau_s = np.asarray(data["tau_s"], dtype=np.float64)
        error_mm = 1000.0 * np.asarray(
            data["final_error_m"], dtype=np.float64,
        )
    if tau_s.shape != error_mm.shape or tau_s.ndim != 1:
        raise ValueError("scatter inputs must be aligned one-dimensional arrays")
    if tau_s.size == 0 or not (
        np.isfinite(tau_s).all() and np.isfinite(error_mm).all()
    ):
        raise ValueError("scatter inputs must be nonempty and finite")

    # Keep the dense body readable. Rare larger errors remain visible on the
    # top boundary rather than stretching the vertical axis for every point.
    top_mm = max(600.0, float(np.percentile(error_mm, 99.5)))
    shown_error_mm = np.minimum(error_mm, top_mm)
    clipped = int(np.sum(error_mm > top_mm))

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
    })
    figure, axis = plt.subplots(figsize=(9.2, 5.4), dpi=160)
    axis.scatter(
        tau_s, shown_error_mm, s=7, alpha=0.16,
        color="#1769aa", edgecolors="none", rasterized=True,
    )
    axis.set_xlabel("预测未来时间（秒）")
    axis.set_ylabel("最终位置误差（毫米）")
    title = "同一未来时刻的最终位置预测误差分布"
    if clipped:
        title += f"（{clipped} 个超范围点显示在顶部）"
    axis.set_title(title)
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, top_mm * 1.02)
    axis.grid(True, alpha=0.18, linewidth=0.7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(generate(args.queries, args.output))


if __name__ == "__main__":
    main()
