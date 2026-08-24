#!/usr/bin/env python3
"""CLI wrapper for CUDA-only observable sparse-pose risk-gate training."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.armor_pose.risk_gate import main


if __name__ == "__main__":
    main()
