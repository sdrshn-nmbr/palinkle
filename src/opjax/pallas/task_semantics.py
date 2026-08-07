"""Exact visible semantics for generated Pallas tasks."""

from __future__ import annotations

import json
from typing import Any


class TaskSemanticsError(RuntimeError):
    """A task does not have a complete, representable public specification."""


def operation_specification(task: dict[str, Any]) -> dict[str, Any]:
    operation = task["operation"]
    shapes = task["input_shapes"]
    dtypes = task["input_dtypes"]
    if len(shapes) != len(dtypes) or not shapes:
        raise TaskSemanticsError("TASK_INPUT_CONTRACT_INVALID")
    output_shape = list(shapes[0])
    equations = {
        "add": "output = x0 + x1",
        "multiply": "output = x0 * x1",
        "subtract": "output = x0 - x1",
        "maximum": "output = maximum(x0, x1)",
        "minimum": "output = minimum(x0, x1)",
        "safe_divide": "output = x0 / (abs(x1) + 0.25)",
        "relu": "output = maximum(x0, 0)",
        "tanh": "output = tanh(x0)",
        "sigmoid": "output = sigmoid(x0)",
        "square": "output = square(x0)",
        "absolute": "output = abs(x0)",
        "exp": "output = exp(x0)",
        "rmsnorm": "output = x0 * rsqrt(mean(square(x0), axis -1, keepdims true) + epsilon 1e-05)",
        "layernorm": "mean0 = mean(x0, axis -1, keepdims true); output = (x0 - mean0) * rsqrt(mean(square(x0 - mean0), axis -1, keepdims true) + epsilon 1e-05)",
        "l2norm": "output = x0 * rsqrt(sum(square(x0), axis -1, keepdims true) + epsilon 1e-05)",
        "softmax": "output = softmax(x0, axis -1)",
        "log_softmax": "output = log_softmax(x0, axis -1)",
        "softmin": "output = softmax(-x0, axis -1)",
        "transpose": "output = transpose(x0)",
        "transpose_square": "output = transpose(square(x0))",
        "transpose_abs": "output = transpose(abs(x0))",
        "matmul": "output = matmul(x0, x1) with float32 accumulation",
        "matmul_relu": "output = maximum(matmul(x0, x1) with float32 accumulation, 0)",
        "matmul_square": "output = square(matmul(x0, x1) with float32 accumulation)",
        "silu_gate": "output = silu(x0) * x1",
        "gelu_gate": "output = jax.nn.gelu(x0, approximate true) * x1",
        "relu_gate": "output = maximum(x0, 0) * x1",
        "tanh_gate": "output = tanh(x0) * x1",
        "sum": "reduce x0 along axis -1 with sum, keepdims true, then broadcast to the original input shape",
        "row_sum": "reduce x0 along axis -1 with sum, keepdims true, then broadcast to the original input shape",
        "max": "reduce x0 along axis -1 with max, keepdims true, then broadcast to the original input shape",
        "mean": "reduce x0 along axis -1 with mean, keepdims true, then broadcast to the original input shape",
        "min": "reduce x0 along axis -1 with min, keepdims true, then broadcast to the original input shape",
    }
    try:
        equation = equations[operation]
    except KeyError as exc:
        raise TaskSemanticsError(f"TASK_OPERATION_UNSUPPORTED:{operation}") from exc
    if operation.startswith("transpose") or operation == "transpose":
        if len(output_shape) != 2:
            raise TaskSemanticsError("TRANSPOSE_RANK_INVALID")
        output_shape = [output_shape[1], output_shape[0]]
    elif operation.startswith("matmul") or operation == "matmul":
        output_shape = [shapes[0][0], shapes[1][1]]
    tolerance = task.get("correctness_tolerance", {"rtol": 0.001, "atol": 0.001})
    input_ranges = list(task.get("input_ranges", [None] * len(shapes)))
    if len(input_ranges) != len(shapes):
        raise TaskSemanticsError("TASK_INPUT_RANGE_CONTRACT_INVALID")
    return {
        "operation": operation,
        "inputs": [
            {
                "name": f"x{index}",
                "shape": list(shape),
                "dtype": dtype,
                "range": input_ranges[index],
            }
            for index, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True))
        ],
        "output_shape": output_shape,
        "output_dtype": "float32",
        "equation": equation,
        "tolerance": tolerance,
    }


def render_task_instruction(task: dict[str, Any], *, repair: str | None) -> str:
    specification = operation_specification(task)
    repair_text = (
        f" The starter contains one known `{repair}` defect."
        if repair is not None
        else ""
    )
    return (
        "Implement an authentic, normally lowered Pallas kernel in kernel.py."
        f"{repair_text}\n\n"
        "The complete public contract is:\n"
        f"- Inputs: {json.dumps(specification['inputs'], sort_keys=True)}\n"
        f"- Equation: {specification['equation']}\n"
        f"- Output: shape {json.dumps(specification['output_shape'])}, "
        f"dtype {specification['output_dtype']}\n"
        f"- Correctness tolerance: {json.dumps(specification['tolerance'], sort_keys=True)}\n\n"
        "Preserve the workload interface and run public checks. Do not use interpret "
        "mode or a plain-JAX fallback. The hidden verifier uses three fixed input seeds.\n"
    )
