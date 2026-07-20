"""Evaluate a checkpoint and the two deterministic baselines by session."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .baselines import constant_twist, static_hold
from .model import Stage3TCN

PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _set_error(pred: np.ndarray, target: np.ndarray) -> float:
    values = []
    for permutation in PERMUTATIONS:
        aligned = pred[:, list(permutation), :]
        values.append(float(np.linalg.norm(aligned - target, axis=-1).mean()))
    return min(values)


def _metrics(errors: list[float]) -> dict[str, float]:
    if not errors:
        return {"count": 0, "median_m": float("nan"), "p95_m": float("nan"), "p99_m": float("nan")}
    values = np.asarray(errors)
    return {"count": int(len(values)), "median_m": float(np.quantile(values, 0.50)), "p95_m": float(np.quantile(values, 0.95)), "p99_m": float(np.quantile(values, 0.99))}


def evaluate(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset)
    arrays_file = np.load(dataset / "samples.npz", allow_pickle=False)
    arrays = {key: arrays_file[key] for key in arrays_file.files}
    splits = json.loads((dataset / "splits.json").read_text(encoding="utf-8"))
    wanted = set(str(value) for value in splits[args.split])
    indices = [index for index, value in enumerate(arrays["session_id"].astype(str)) if value in wanted]
    model = Stage3TCN()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    learned: list[float] = []
    static: list[float] = []
    twist: list[float] = []
    motion_correct = 0
    with torch.no_grad():
        for index in indices:
            obs = torch.from_numpy(arrays["obs"][index:index + 1])
            obs_mask = torch.from_numpy(arrays["obs_mask"][index:index + 1])
            event_mask = torch.from_numpy(arrays["event_mask"][index:index + 1])
            event_time_s = torch.from_numpy(arrays["event_time_s"][index:index + 1])
            tau = torch.from_numpy(arrays["tau"][index:index + 1].astype(np.float32))
            output = model(obs, obs_mask, event_mask, event_time_s, tau)
            pred = output["position_mean"][0].numpy()
            target = arrays["future_position"][index]
            learned.append(_set_error(pred, target))
            static_pred, _ = static_hold(arrays["obs"][index], arrays["obs_mask"][index], arrays["event_mask"][index], arrays["tau"][index])
            twist_pred, _ = constant_twist(arrays["obs"][index], arrays["obs_mask"][index], arrays["event_mask"][index], arrays["event_time_s"][index], arrays["tau"][index])
            static.append(_set_error(static_pred, target))
            twist.append(_set_error(twist_pred, target))
            motion_correct += int(int(output["motion_logits"].argmax(dim=-1)[0]) == int(arrays["motion_class"][index]))
    report = {
        "split": args.split,
        "learned": _metrics(learned),
        "static_hold": _metrics(static),
        "constant_twist": _metrics(twist),
        "motion_accuracy": motion_correct / max(len(indices), 1),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
