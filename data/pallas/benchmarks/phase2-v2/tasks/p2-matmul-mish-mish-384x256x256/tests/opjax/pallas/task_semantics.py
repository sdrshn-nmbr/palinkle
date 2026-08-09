"""Exact visible semantics for generated Pallas tasks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp


class TaskSemanticsError(RuntimeError):
    """A task does not have a complete, representable public specification."""


def operation_specification(task: dict[str, Any]) -> dict[str, Any]:
    operation = task["operation"]
    shapes = task["input_shapes"]
    dtypes = task["input_dtypes"]
    if len(shapes) != len(dtypes) or not shapes:
        raise TaskSemanticsError("TASK_INPUT_CONTRACT_INVALID")
    if any(
        not isinstance(shape, list)
        or not shape
        or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape)
        for shape in shapes
    ):
        raise TaskSemanticsError("TASK_INPUT_SHAPE_INVALID")
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
        "matmul_bias_gelu": "output = gelu(matmul(x0, x1) with float32 accumulation + x2, approximate true)",
        "matmul_divide_gelu": "output = gelu((matmul(x0, x1) + x2) / 10)",
        "matmul_scale_residual_clamp_logsumexp_mish": "values = matmul(x0, transpose(x1)) + x2; values = clip(4 * values, -10, 10); reduced = logsumexp(values, axis 1, keepdims true); mish = reduced * tanh(logaddexp(reduced, 0)); output = reduced * mish",
        "swiglu_mlp": "gate = silu(dot(x0, x1)); up = dot(x0, x2); output = dot(gate * up, x3)",
        "matmul_gelu_softmax": "values = gelu(matmul(x0, x1) + x2); output = softmax(values, axis 1)",
        "causal_attention_bhsd": "scores[b,h,q,k] = dot(x0[b,h,q,:], x1[b,h,k,:]) / sqrt(head_dim); scores where k > q are -1e9; output = softmax(scores, axis -1) @ x2",
        "rmsnorm_scale": "x_f32 = float32(x0); output = cast(x_f32 * rsqrt(mean(square(x_f32), axis -1, keepdims true) + 1e-5), dtype(x0)) * x1",
        "matmul_mish_mish": "values = matmul(x0, x1) + x2; first = values * tanh(logaddexp(values, 0)); output = first * tanh(logaddexp(first, 0))",
        "gemm_add_relu": "output = maximum(matmul(x0, x1) + x2, 0)",
        "grouped_matmul": "x2 contains contiguous group sizes summing to rows(x0); each output group slice is matmul(the matching x0 rows, x1[group])",
        "matmul_residual_silu": "output = silu(matmul(x0, x1) with float32 accumulation) + x2",
        "swiglu_projection": "output = silu(matmul(x0, x1) with float32 accumulation) * matmul(x0, x2) with float32 accumulation",
        "matmul_softmax": "output = softmax(matmul(x0, x1) with float32 accumulation, axis -1)",
        "causal_attention": "scores[h,q,k] = dot(x0[h,q,:], x1[h,k,:]) / sqrt(head_dim); scores where k > q are negative infinity; output = softmax(scores, axis -1) @ x2",
        "soft_target_cross_entropy": "target = softmax(x1, axis -1); output = -target * log_softmax(x0, axis -1)",
        "residual_rmsnorm": "values = x0 + x1; output = values * rsqrt(mean(square(values), axis -1, keepdims true) + epsilon 1e-05) * x2",
        "residual_layernorm": "values = x0 + x1; mean0 = mean(values, axis -1, keepdims true); normalized = (values - mean0) * rsqrt(mean(square(values - mean0), axis -1, keepdims true) + epsilon 1e-05); output = normalized * x2 + x3",
    }
    try:
        equation = equations[operation]
    except KeyError as exc:
        raise TaskSemanticsError(f"TASK_OPERATION_UNSUPPORTED:{operation}") from exc
    unary = {
        "relu",
        "tanh",
        "sigmoid",
        "square",
        "absolute",
        "exp",
        "rmsnorm",
        "layernorm",
        "l2norm",
        "softmax",
        "log_softmax",
        "softmin",
        "transpose",
        "transpose_square",
        "transpose_abs",
        "sum",
        "row_sum",
        "max",
        "mean",
        "min",
    }
    binary_elementwise = {
        "add",
        "multiply",
        "subtract",
        "maximum",
        "minimum",
        "safe_divide",
        "silu_gate",
        "gelu_gate",
        "relu_gate",
        "tanh_gate",
    }
    if operation in unary and len(shapes) != 1:
        raise TaskSemanticsError("TASK_INPUT_ARITY_INVALID")
    if operation in binary_elementwise and (
        len(shapes) != 2 or shapes[0] != shapes[1]
    ):
        raise TaskSemanticsError("TASK_ELEMENTWISE_CONTRACT_INVALID")
    if operation in {"matmul", "matmul_relu", "matmul_square", "matmul_softmax"}:
        if (
            len(shapes) != 2
            or len(shapes[0]) != 2
            or len(shapes[1]) != 2
            or shapes[0][1] != shapes[1][0]
        ):
            raise TaskSemanticsError("TASK_MATMUL_CONTRACT_INVALID")
    if operation == "matmul_bias_gelu" and (
        len(shapes) != 3
        or len(shapes[0]) != 2
        or len(shapes[1]) != 2
        or shapes[0][1] != shapes[1][0]
        or shapes[2] != [shapes[1][1]]
    ):
        raise TaskSemanticsError("TASK_MATMUL_BIAS_CONTRACT_INVALID")
    if operation in {"matmul_divide_gelu", "matmul_gelu_softmax", "gemm_add_relu"} and (
        len(shapes) != 3
        or len(shapes[0]) != 2
        or len(shapes[1]) != 2
        or shapes[0][1] != shapes[1][0]
        or shapes[2] != [shapes[1][1]]
    ):
        raise TaskSemanticsError("TASK_GEMM_BIAS_CONTRACT_INVALID")
    if operation == "matmul_scale_residual_clamp_logsumexp_mish" and (
        len(shapes) != 3
        or len(shapes[0]) != 2
        or len(shapes[1]) != 2
        or shapes[0][1] != shapes[1][1]
        or shapes[2] != [shapes[1][0]]
    ):
        raise TaskSemanticsError("TASK_MATMUL_SCALE_MISH_CONTRACT_INVALID")
    if operation == "swiglu_mlp" and (
        len(shapes) != 4
        or len(shapes[0]) != 3
        or any(len(shape) != 2 for shape in shapes[1:])
        or shapes[0][-1] != shapes[1][0]
        or shapes[1] != shapes[2]
        or shapes[1][1] != shapes[3][0]
    ):
        raise TaskSemanticsError("TASK_SWIGLU_MLP_CONTRACT_INVALID")
    if operation == "causal_attention_bhsd" and (
        len(shapes) != 3
        or any(len(shape) != 4 for shape in shapes)
        or shapes[0] != shapes[1]
        or shapes[0] != shapes[2]
    ):
        raise TaskSemanticsError("TASK_ATTENTION_BHSD_CONTRACT_INVALID")
    if operation == "rmsnorm_scale" and (
        len(shapes) != 2 or shapes[1] != [shapes[0][-1]]
    ):
        raise TaskSemanticsError("TASK_RMSNORM_SCALE_CONTRACT_INVALID")
    if operation == "matmul_mish_mish" and (
        len(shapes) != 3
        or len(shapes[0]) != 2
        or len(shapes[1]) != 2
        or shapes[0][1] != shapes[1][0]
        or shapes[2] != [shapes[1][1]]
    ):
        raise TaskSemanticsError("TASK_MATMUL_MISH_CONTRACT_INVALID")
    if operation == "grouped_matmul" and (
        len(shapes) != 3
        or len(shapes[0]) != 2
        or len(shapes[1]) != 3
        or shapes[2] != [shapes[1][0]]
        or shapes[0][1] != shapes[1][1]
    ):
        raise TaskSemanticsError("TASK_GROUPED_MATMUL_CONTRACT_INVALID")
    if operation == "matmul_residual_silu" and (
        len(shapes) != 3
        or len(shapes[0]) != 2
        or len(shapes[1]) != 2
        or shapes[0][1] != shapes[1][0]
        or shapes[2] != [shapes[0][0], shapes[1][1]]
    ):
        raise TaskSemanticsError("TASK_MATMUL_RESIDUAL_CONTRACT_INVALID")
    if operation == "swiglu_projection" and (
        len(shapes) != 3
        or len(shapes[0]) != 2
        or len(shapes[1]) != 2
        or shapes[0][1] != shapes[1][0]
        or shapes[1] != shapes[2]
    ):
        raise TaskSemanticsError("TASK_SWIGLU_CONTRACT_INVALID")
    if operation == "causal_attention" and (
        len(shapes) != 3
        or any(len(shape) != 3 for shape in shapes)
        or shapes[0] != shapes[1]
        or shapes[0] != shapes[2]
    ):
        raise TaskSemanticsError("TASK_ATTENTION_CONTRACT_INVALID")
    if operation == "soft_target_cross_entropy" and (
        len(shapes) != 2
        or len(shapes[0]) != 2
        or shapes[1] != shapes[0]
    ):
        raise TaskSemanticsError("TASK_CROSS_ENTROPY_CONTRACT_INVALID")
    if operation == "residual_rmsnorm" and (
        len(shapes) != 3
        or shapes[0] != shapes[1]
        or shapes[2] != [shapes[0][-1]]
    ):
        raise TaskSemanticsError("TASK_RESIDUAL_RMSNORM_CONTRACT_INVALID")
    if operation == "residual_layernorm" and (
        len(shapes) != 4
        or shapes[0] != shapes[1]
        or shapes[2] != [shapes[0][-1]]
        or shapes[3] != shapes[2]
    ):
        raise TaskSemanticsError("TASK_RESIDUAL_LAYERNORM_CONTRACT_INVALID")
    if operation.startswith("transpose") or operation == "transpose":
        if len(output_shape) != 2:
            raise TaskSemanticsError("TRANSPOSE_RANK_INVALID")
        output_shape = [output_shape[1], output_shape[0]]
    elif operation in {
        "matmul",
        "matmul_relu",
        "matmul_square",
        "matmul_bias_gelu",
        "matmul_divide_gelu",
        "matmul_residual_silu",
        "swiglu_projection",
        "matmul_softmax",
        "matmul_gelu_softmax",
        "gemm_add_relu",
        "matmul_mish_mish",
    }:
        output_shape = [shapes[0][0], shapes[1][1]]
    elif operation == "matmul_scale_residual_clamp_logsumexp_mish":
        output_shape = [shapes[0][0], 1]
    elif operation == "swiglu_mlp":
        output_shape = [shapes[0][0], shapes[0][1], shapes[3][1]]
    elif operation == "grouped_matmul":
        output_shape = [shapes[0][0], shapes[1][2]]
    tolerance = task.get("correctness_tolerance", {"rtol": 0.001, "atol": 0.001})
    input_ranges = list(
        task.get(
            "correctness_input_ranges",
            task.get("input_ranges", [None] * len(shapes)),
        )
    )
    timing_ranges = list(task.get("timing_input_ranges", task.get("input_ranges", input_ranges)))
    if len(input_ranges) != len(shapes) or len(timing_ranges) != len(shapes):
        raise TaskSemanticsError("TASK_INPUT_RANGE_CONTRACT_INVALID")
    return {
        "operation": operation,
        "inputs": [
            {
                "name": f"x{index}",
                "shape": list(shape),
                "dtype": dtype,
                "range": input_ranges[index],
                "timing_range": timing_ranges[index],
            }
            for index, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True))
        ],
        "output_shape": output_shape,
        "output_dtype": task.get("output_dtype", "float32"),
        "equation": equation,
        "tolerance": tolerance,
        "timing_tolerance": task.get("timing_correctness_tolerance", tolerance),
    }


def semantic_oracle(task: dict[str, Any], *inputs: jax.Array) -> jax.Array:
    operation = task["operation"]
    x = inputs[0]
    if operation == "matmul_divide_gelu":
        values = jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
        return jax.nn.gelu((values + inputs[2]) / 10.0)
    if operation == "matmul_scale_residual_clamp_logsumexp_mish":
        values = jnp.matmul(x, inputs[1].T, preferred_element_type=jnp.float32) + inputs[2]
        values = jnp.clip(values * 4.0, -10.0, 10.0)
        reduced = jax.scipy.special.logsumexp(values, axis=1, keepdims=True)
        mish = reduced * jnp.tanh(jnp.logaddexp(reduced, 0.0))
        return reduced * mish
    if operation == "swiglu_mlp":
        gate = jax.nn.silu(jnp.dot(x, inputs[1]))
        up = jnp.dot(x, inputs[2])
        return jnp.dot(gate * up, inputs[3])
    if operation == "matmul_gelu_softmax":
        values = jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32) + inputs[2]
        return jax.nn.softmax(jax.nn.gelu(values), axis=1)
    if operation == "causal_attention_bhsd":
        head_dim = x.shape[-1]
        scores = jnp.einsum("bhqd,bhkd->bhqk", x, inputs[1]) / jnp.sqrt(
            jnp.asarray(head_dim, dtype=jnp.float32)
        )
        sequence = x.shape[-2]
        scores = jnp.where(jnp.tril(jnp.ones((sequence, sequence), dtype=bool)), scores, -1e9)
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, axis=-1), inputs[2])
    if operation == "rmsnorm_scale":
        values = x.astype(jnp.float32)
        normalized = values * jax.lax.rsqrt(
            jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-5
        )
        return normalized.astype(x.dtype) * inputs[1]
    if operation == "matmul_mish_mish":
        values = (
            jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
            + inputs[2]
        )
        first = values * jnp.tanh(jnp.logaddexp(values, 0.0))
        return first * jnp.tanh(jnp.logaddexp(first, 0.0))
    if operation == "gemm_add_relu":
        values = jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
        return jnp.maximum(values + inputs[2], 0.0)
    if operation == "grouped_matmul":
        group_sizes = inputs[2]
        group_count, input_width, output_width = inputs[1].shape
        rows_per_group = x.shape[0] // group_count
        group_ends = jnp.cumsum(group_sizes)
        group_starts = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
        )
        result = jnp.zeros(
            (x.shape[0] + rows_per_group, output_width), dtype=x.dtype
        )

        def body(carry: jax.Array, index: jax.Array) -> tuple[jax.Array, None]:
            start = group_starts[index]
            count = group_sizes[index]
            expert_lhs = jax.lax.dynamic_slice(
                x, (start, 0), (rows_per_group, input_width)
            )
            values = jax.lax.dot(
                expert_lhs,
                inputs[1][index],
                preferred_element_type=jnp.float32,
            )
            row_ids = jax.lax.broadcasted_iota(
                jnp.int32, (rows_per_group, output_width), 0
            )
            values = jnp.where(row_ids < count, values, 0.0)
            current = jax.lax.dynamic_slice(
                carry, (start, 0), (rows_per_group, output_width)
            )
            return (
                jax.lax.dynamic_update_slice(
                    carry, current + values.astype(carry.dtype), (start, 0)
                ),
                None,
            )

        result, _ = jax.lax.scan(body, result, jnp.arange(group_count))
        return result[: x.shape[0]]
    if operation == "matmul_bias_gelu":
        values = jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
        return jax.nn.gelu(values + inputs[2], approximate=True)
    if operation == "matmul_residual_silu":
        values = jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
        return jax.nn.silu(values) + inputs[2]
    if operation == "swiglu_projection":
        gate = jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
        up = jnp.matmul(x, inputs[2], preferred_element_type=jnp.float32)
        return jax.nn.silu(gate) * up
    if operation == "matmul_softmax":
        values = jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
        return jax.nn.softmax(values, axis=-1)
    if operation == "causal_attention":
        head_dim = x.shape[-1]
        scores = jnp.matmul(x, jnp.swapaxes(inputs[1], -1, -2)) / jnp.sqrt(
            jnp.asarray(head_dim, dtype=jnp.float32)
        )
        positions = jnp.arange(x.shape[-2])
        scores = jnp.where(positions[None, :, None] >= positions[None, None, :], scores, -jnp.inf)
        return jnp.matmul(jax.nn.softmax(scores, axis=-1), inputs[2])
    if operation == "soft_target_cross_entropy":
        logits = x.astype(jnp.float32)
        targets = jax.nn.softmax(inputs[1].astype(jnp.float32), axis=-1)
        return -targets * jax.nn.log_softmax(logits, axis=-1)
    if operation == "residual_rmsnorm":
        values = x.astype(jnp.float32) + inputs[1].astype(jnp.float32)
        normalized = values * jax.lax.rsqrt(
            jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-5
        )
        return normalized * inputs[2]
    if operation == "residual_layernorm":
        values = x.astype(jnp.float32) + inputs[1].astype(jnp.float32)
        mean = jnp.mean(values, axis=-1, keepdims=True)
        normalized = (values - mean) * jax.lax.rsqrt(
            jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True) + 1e-5
        )
        return normalized * inputs[2] + inputs[3]
    if operation == "add":
        return x + inputs[1]
    if operation == "matmul":
        value = jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
        return value.astype(jnp.dtype(task.get("output_dtype", "float32")))
    if operation == "multiply":
        return x * inputs[1]
    if operation == "subtract":
        return x - inputs[1]
    if operation == "maximum":
        return jnp.maximum(x, inputs[1])
    if operation == "minimum":
        return jnp.minimum(x, inputs[1])
    if operation == "safe_divide":
        return x / (jnp.abs(inputs[1]) + 0.25)
    if operation == "relu":
        return jnp.maximum(x, 0.0)
    if operation == "tanh":
        return jnp.tanh(x)
    if operation == "sigmoid":
        return jax.nn.sigmoid(x)
    if operation == "square":
        return jnp.square(x)
    if operation == "absolute":
        return jnp.abs(x)
    if operation == "exp":
        return jnp.exp(x)
    if operation == "rmsnorm":
        values = x.astype(jnp.float32)
        return values * jax.lax.rsqrt(
            jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-5
        )
    if operation == "layernorm":
        values = x.astype(jnp.float32)
        mean = jnp.mean(values, axis=-1, keepdims=True)
        return (values - mean) * jax.lax.rsqrt(
            jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True) + 1e-5
        )
    if operation == "l2norm":
        values = x.astype(jnp.float32)
        return values * jax.lax.rsqrt(
            jnp.sum(jnp.square(values), axis=-1, keepdims=True) + 1e-5
        )
    if operation == "softmax":
        return jax.nn.softmax(x.astype(jnp.float32), axis=-1)
    if operation == "log_softmax":
        return jax.nn.log_softmax(x.astype(jnp.float32), axis=-1)
    if operation == "softmin":
        return jax.nn.softmax(-x.astype(jnp.float32), axis=-1)
    if operation == "transpose":
        return jnp.transpose(x)
    if operation == "transpose_square":
        return jnp.transpose(jnp.square(x))
    if operation == "transpose_abs":
        return jnp.transpose(jnp.abs(x))
    if operation == "matmul_relu":
        return jnp.maximum(
            jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32), 0.0
        )
    if operation == "matmul_square":
        return jnp.square(
            jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
        )
    if operation == "silu_gate":
        return jax.nn.silu(x) * inputs[1]
    if operation == "gelu_gate":
        return jax.nn.gelu(x) * inputs[1]
    if operation == "relu_gate":
        return jnp.maximum(x, 0.0) * inputs[1]
    if operation == "tanh_gate":
        return jnp.tanh(x) * inputs[1]
    reduction = "sum" if operation == "row_sum" else operation
    if reduction in {"sum", "max", "mean", "min"}:
        reducer = {
            "sum": jnp.sum,
            "max": jnp.max,
            "mean": jnp.mean,
            "min": jnp.min,
        }[reduction]
        return jnp.broadcast_to(reducer(x, axis=-1, keepdims=True), x.shape)
    raise TaskSemanticsError(f"TASK_ORACLE_UNSUPPORTED:{operation}")


def generate_inputs(
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[str] | None,
    ranges: Sequence[Sequence[float] | None] | None,
    *,
    seed: int,
) -> tuple[jax.Array, ...]:
    dtype_values = list(dtypes or ["float32"] * len(shapes))
    range_values = list(ranges or [None] * len(shapes))
    if len(dtype_values) != len(shapes) or len(range_values) != len(shapes):
        raise TaskSemanticsError("INPUT_SPEC_LENGTH_MISMATCH")
    key = jax.random.PRNGKey(seed)
    values = []
    for shape, dtype_name, bounds in zip(
        shapes, dtype_values, range_values, strict=True
    ):
        key, subkey = jax.random.split(key)
        dtype = jnp.dtype(dtype_name)
        shape_tuple = tuple(shape)
        if jnp.issubdtype(dtype, jnp.integer):
            low, high = bounds or (0, max(shape_tuple[-1] if shape_tuple else 1, 2))
            value = jax.random.randint(
                subkey,
                shape_tuple,
                int(low),
                max(int(high), int(low) + 1),
                dtype=dtype,
            )
        elif jnp.issubdtype(dtype, jnp.bool_):
            value = jax.random.bernoulli(subkey, 0.5, shape_tuple)
        elif bounds:
            value = jax.random.uniform(
                subkey,
                shape_tuple,
                dtype=dtype,
                minval=bounds[0],
                maxval=bounds[1],
            )
        else:
            value = jax.random.normal(subkey, shape_tuple, dtype=dtype)
        values.append(value)
    return tuple(values)


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
