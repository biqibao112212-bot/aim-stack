# Third-party notices

## TongjiSuperPower/sp_vision_25

- Upstream: <https://github.com/TongjiSuperPower/sp_vision_25>
- Commit: `bd9f5e798fa3c6dd3b483ae6627796afb41c608d`
- License: MIT; the unmodified license is stored at
  [`third_party/tongji_sp_vision_25/LICENSE`](third_party/tongji_sp_vision_25/LICENSE).
- Vendored scope: `Armor`, `Target`, `ExtendedKalmanFilter` and their required math helpers.

Vendored source content matches the pinned upstream commit; final newlines and trailing whitespace were
normalized without semantic changes.
The ONNX Runtime loader, Daedalus exposure-pose adapter, coordinate transform, research-only
tracker wrapper and JSONL logger are local integration code.

## ONNX Runtime

- Project: <https://github.com/microsoft/onnxruntime>
- Release: `v1.22.1`
- License: MIT; the installed dependency retains its upstream `LICENSE`.
- Distribution source: official `Microsoft.ML.OnnxRuntime.1.22.1.nupkg` GitHub release asset.

The runtime is installed outside the Git repository under the workspace `deps` directory. Its
package and shared-library hashes are pinned in
[`implementation.lock.json`](implementation.lock.json).
