"""Export a Stage-3 checkpoint and verify PyTorch/ONNX Runtime parity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import Stage3TCN


class _ExportWrapper(torch.nn.Module):
    def __init__(self, model: Stage3TCN) -> None:
        super().__init__()
        self.model = model

    def forward(self, obs, obs_mask, event_mask, event_time_s, tau):
        output = self.model(obs, obs_mask, event_mask, event_time_s, tau)
        return output["position_mean"], output["position_logvar"], output["normal"], output["motion_logits"]


def export(args: argparse.Namespace) -> Path:
    torch.manual_seed(0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Stage3TCN()
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    wrapper = _ExportWrapper(model)
    wrapper.eval()
    obs = torch.randn(2, 200, 4, 5)
    obs_mask = torch.ones(2, 200, 4, dtype=torch.bool)
    event_mask = torch.ones(2, 200, dtype=torch.bool)
    event_time_s = torch.linspace(-1.0, 0.0, 200).expand(2, -1)
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.5, 0.03, 0.17, 0.31, 0.44], [0.0, 0.1, 0.2, 0.5, 0.07, 0.22, 0.36, 0.48]])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (obs, obs_mask, event_mask, event_time_s, tau),
        output,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs", "obs_mask", "event_mask", "event_time_s", "tau"],
        output_names=["position_mean", "position_logvar", "normal", "motion_logits"],
        dynamic_axes={
            "obs": {0: "batch", 1: "time"},
            "obs_mask": {0: "batch", 1: "time"},
            "event_mask": {0: "batch", 1: "time"},
            "event_time_s": {0: "batch", 1: "time"},
            "tau": {0: "batch", 1: "query"},
            "position_mean": {0: "batch", 1: "query"},
            "position_logvar": {0: "batch", 1: "query"},
            "normal": {0: "batch", 1: "query"},
            "motion_logits": {0: "batch"},
        },
    )
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = wrapper(obs, obs_mask, event_mask, event_time_s, tau)
    actual = session.run(None, {"obs": obs.numpy(), "obs_mask": obs_mask.numpy(), "event_mask": event_mask.numpy(), "event_time_s": event_time_s.numpy(), "tau": tau.numpy()})
    errors = [float(np.max(np.abs(expected.numpy() - received))) for expected, received in zip(reference, actual)]
    max_error = max(errors)
    print(f"onnx_parity_errors={errors}")
    if max_error >= 1e-4:
        raise RuntimeError(f"ONNX parity failed: max_abs_error={max_error}")
    for batch_size, time_steps, query_count in ((1, 64, 3), (3, 200, 5)):
        dynamic_obs = torch.randn(batch_size, time_steps, 4, 5)
        dynamic_obs_mask = torch.ones(batch_size, time_steps, 4, dtype=torch.bool)
        dynamic_event_mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
        dynamic_event_time_s = torch.linspace(-1.0, 0.0, time_steps).expand(batch_size, -1)
        dynamic_tau = torch.rand(batch_size, query_count) * 0.5
        with torch.no_grad():
            dynamic_reference = wrapper(dynamic_obs, dynamic_obs_mask, dynamic_event_mask, dynamic_event_time_s, dynamic_tau)
        dynamic_actual = session.run(None, {
            "obs": dynamic_obs.numpy(), "obs_mask": dynamic_obs_mask.numpy(),
            "event_mask": dynamic_event_mask.numpy(), "event_time_s": dynamic_event_time_s.numpy(), "tau": dynamic_tau.numpy()
        })
        dynamic_error = max(float(np.max(np.abs(expected.numpy() - received))) for expected, received in zip(dynamic_reference, dynamic_actual))
        if dynamic_error >= 1e-4:
            raise RuntimeError(f"ONNX dynamic-shape parity failed for {(batch_size, time_steps, query_count)}: {dynamic_error}")
    output.with_suffix(output.suffix + ".parity.txt").write_text(f"max_abs_error={max_error}\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args))


if __name__ == "__main__":
    main()
