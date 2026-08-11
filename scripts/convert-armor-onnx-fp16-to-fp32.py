#!/usr/bin/env python3
"""Convert a mixed-FP16 armor ONNX contract to explicit FP32 without overwriting assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FLOAT16 graph inputs, initializers, tensor attributes, and "
            "value-info contracts to FLOAT before TensorRT engine generation."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def convert_tensor(tensor: TensorProto) -> bool:
    if tensor.data_type != TensorProto.FLOAT16:
        return False
    name = tensor.name
    values = numpy_helper.to_array(tensor).astype(np.float32)
    tensor.CopyFrom(numpy_helper.from_array(values, name=name))
    return True


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    manifest = output.with_suffix(output.suffix + ".json")
    partial = output.with_suffix(output.suffix + ".partial")
    manifest_partial = manifest.with_suffix(manifest.suffix + ".partial")

    if not source.is_file():
        raise FileNotFoundError(source)
    if output == source:
        raise RuntimeError("Input and output must be different protected assets")
    if any(path.exists() for path in (output, manifest, partial, manifest_partial)):
        raise RuntimeError(f"Refusing to overwrite protected or partial output for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(source))
    initializer_count = sum(convert_tensor(tensor) for tensor in model.graph.initializer)
    attribute_count = 0
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                attribute_count += int(convert_tensor(attribute.t))
            elif attribute.type == onnx.AttributeProto.TENSORS:
                attribute_count += sum(convert_tensor(tensor) for tensor in attribute.tensors)

    value_info_count = 0
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value_info in collection:
            tensor_type = value_info.type.tensor_type
            if tensor_type.elem_type == TensorProto.FLOAT16:
                tensor_type.elem_type = TensorProto.FLOAT
                value_info_count += 1

    onnx.checker.check_model(model)
    onnx.save(model, str(partial))

    input_info = model.graph.input[0]
    dims = [dimension.dim_value for dimension in input_info.type.tensor_type.shape.dim]
    if not dims or any(dimension <= 0 for dimension in dims):
        raise RuntimeError(f"Static validation input is required, got {dims}")
    rng = np.random.default_rng(args.seed)
    input_fp32 = rng.random(dims, dtype=np.float32)
    source_session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    output_session = ort.InferenceSession(str(partial), providers=["CPUExecutionProvider"])
    source_output = source_session.run(
        None, {source_session.get_inputs()[0].name: input_fp32.astype(np.float16)}
    )[0]
    converted_output = output_session.run(
        None, {output_session.get_inputs()[0].name: input_fp32}
    )[0]
    delta = np.abs(source_output.astype(np.float32) - converted_output.astype(np.float32))

    report = {
        "schema_version": 1,
        "kind": "armor_onnx_explicit_fp32_conversion",
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(partial),
        "converted_float16_initializers": initializer_count,
        "converted_float16_tensor_attributes": attribute_count,
        "converted_float16_value_info": value_info_count,
        "validation": {
            "seed": args.seed,
            "input_shape": dims,
            "max_abs_output_delta": float(delta.max()),
            "mean_abs_output_delta": float(delta.mean()),
        },
    }
    manifest_partial.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(output)
    manifest_partial.replace(manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
