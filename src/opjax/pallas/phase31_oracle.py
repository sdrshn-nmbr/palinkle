"""Phase 3.1 hidden-input and scale-aware correctness contracts."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp


SKIP_MUTATION_FRAGMENTS = (
    "index",
    "indices",
    "length",
    "mask",
    "offset",
    "page",
    "segment",
    "size",
    "token",
)


def oracle_contract(input_argument_names: list[str], output_dtype: str) -> dict[str, Any]:
    mutable = [
        index
        for index, name in enumerate(input_argument_names)
        if not any(fragment in name.lower() for fragment in SKIP_MUTATION_FRAGMENTS)
    ]
    low_precision = output_dtype in {"bfloat16", "float16"}
    return {
        "schema_version": 1,
        "input_cases": ["jaxbench-original", "derived-seed-1", "derived-seed-2"],
        "mutable_input_indices": mutable,
        "correctness": {
            "rtol": 0.05 if low_precision else 0.01,
            "relative_atol": 0.02 if low_precision else 0.001,
            "max_normalized_error": 0.10 if low_precision else 0.025,
        },
        "performance_case": "jaxbench-original",
        "performance_requires_nonzero_signal": True,
    }


def derive_input_case(
    inputs: tuple[Any, ...], *, contract: dict[str, Any], seed: int
) -> tuple[Any, ...]:
    if seed == 0:
        return inputs
    mutable = set(contract["mutable_input_indices"])
    derived = []
    for index, value in enumerate(inputs):
        dtype = getattr(value, "dtype", None)
        if (
            index not in mutable
            or dtype is None
            or not jnp.issubdtype(dtype, jnp.inexact)
            or getattr(value, "size", 0) <= 1
        ):
            derived.append(value)
            continue
        key = jax.random.fold_in(jax.random.key(seed), index)
        reduction = value.shape[-2] if value.ndim >= 2 else value.shape[-1]
        scale = 1.0 / math.sqrt(max(int(reduction), 1))
        noise = jax.random.normal(key, value.shape, dtype=dtype) * scale
        finite = jnp.isfinite(value)
        derived.append(jnp.where(finite, value * (0.75 + 0.1 * seed) + noise, value))
    return tuple(derived)


def compare_output(
    expected: Any, actual: Any, *, contract: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(expected, jax.Array) or not isinstance(actual, jax.Array):
        return {"correct": False, "reason": "single array output required"}
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {
            "correct": False,
            "reason": "shape or dtype mismatch",
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "expected_dtype": str(expected.dtype),
            "actual_dtype": str(actual.dtype),
        }
    policy = contract["correctness"]
    expected_f32 = expected.astype(jnp.float32)
    actual_f32 = actual.astype(jnp.float32)
    difference = jnp.abs(expected_f32 - actual_f32)
    signal = float(jax.device_get(jnp.max(jnp.abs(expected_f32))))
    max_difference = float(jax.device_get(jnp.max(difference)))
    denominator = max(signal, float(jnp.finfo(jnp.float32).tiny))
    normalized_error = max_difference / denominator
    atol = max(denominator * policy["relative_atol"], float(jnp.finfo(jnp.float32).tiny))
    allclose = bool(
        jax.device_get(
            jnp.allclose(
                expected_f32,
                actual_f32,
                rtol=policy["rtol"],
                atol=atol,
            )
        )
    )
    correct = allclose and normalized_error <= policy["max_normalized_error"]
    return {
        "correct": correct,
        "reason": "ok" if correct else "values differ",
        "signal_max_abs": signal,
        "max_difference": max_difference,
        "normalized_max_error": normalized_error,
        "rtol": policy["rtol"],
        "atol": atol,
        "max_normalized_error": policy["max_normalized_error"],
        "zero_would_pass": signal == 0.0,
    }
