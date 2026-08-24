#!/usr/bin/env python3
"""CLI wrapper for CUDA-only sparse/dense armor-pose validation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.armor_pose.evaluate import main


if __name__ == "__main__":
    main()
