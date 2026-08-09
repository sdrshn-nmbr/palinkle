"""Probe public JAXBench tensor schemas without materializing full inputs."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from JAXBench.harness.loader import load_module


def _tensor_schema(value: Any) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def probe_schema(path: Path) -> dict[str, Any]:
    module = load_module(str(path), f"{path.parent.name}.public_schema")
    create_parameters = inspect.signature(module.create_inputs).parameters
    inputs = (
        jax.eval_shape(lambda: module.create_inputs(dtype=jnp.bfloat16))
        if "dtype" in create_parameters
        else jax.eval_shape(module.create_inputs)
    )
    input_leaves = jax.tree.leaves(inputs)
    argument_names = module.workload.__code__.co_varnames[
        : module.workload.__code__.co_argcount
    ]
    if len(argument_names) != len(input_leaves):
        raise ValueError(
            f"JAXBENCH_SCHEMA_ARITY_MISMATCH:{path.parent.name}:"
            f"{len(argument_names)}:{len(input_leaves)}"
        )
    input_schema = [
        {"name": name, **_tensor_schema(value)}
        for name, value in zip(argument_names, input_leaves, strict=True)
    ]
    try:
        outputs = jax.eval_shape(module.workload, *inputs)
        output_schema = [_tensor_schema(value) for value in jax.tree.leaves(outputs)]
    except TypeError:
        if path.parent.name != "11p_Megablox_GMM":
            raise
        output_schema = [
            {
                "shape": [input_leaves[0].shape[0], input_leaves[1].shape[-1]],
                "dtype": str(input_leaves[0].dtype),
            }
        ]
    return {"inputs": input_schema, "outputs": output_schema}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-jaxbench-schema")
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(probe_schema(args.baseline), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
