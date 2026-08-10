"""Pinned public development surface for the Phase 3.1 benchmark."""

from __future__ import annotations

import json
from typing import Any


PALLAS_API = """# Pinned JAX Pallas TPU API

The development environment contains JAX and JAXlib 0.10.1. The hidden TPU
runtime uses the same versions. Implement `workload(*inputs)` with normal Pallas
lowering. Do not use `interpret=True` or a plain-JAX result path.

## Core interfaces

```python
pl.pallas_call(
    kernel,
    out_shape,
    *,
    grid=(),
    grid_spec=None,
    in_specs=pl.no_block_spec,
    out_specs=pl.no_block_spec,
    scratch_shapes=(),
    compiler_params=None,
    interpret=False,
)(*inputs)

pl.BlockSpec(block_shape, index_map, *, memory_space=None)
pl.program_id(axis)
pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch,
    in_specs,
    out_specs,
    grid,
    scratch_shapes=(),
)
pltpu.CompilerParams(dimension_semantics=(...))
pltpu.VMEM(shape, dtype)
```

Kernel arguments are references. Read and write them with indexing:

```python
def add_kernel(x_ref, y_ref, out_ref):
    out_ref[...] = x_ref[...] + y_ref[...]

def workload(x, y):
    block = 128
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.size // block,),
        in_specs=(
            pl.BlockSpec((block,), lambda i: (i,)),
            pl.BlockSpec((block,), lambda i: (i,)),
        ),
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, y)
```

For TPU matmul, use `jnp.dot` inside the Pallas kernel and accumulate in
float32 VMEM. TPU block dimensions generally need multiples of 8 for bf16 and
128-element tiling along vectorized dimensions. Use `pl.when` for conditional
initialization and prefer static grids and block shapes.

Run `python dev_check.py` after each edit. It parses and imports `kernel.py`,
traces `workload` against the public tensor schema, and requires a reachable
Pallas primitive. This does not replace hidden TPU compilation or correctness.
"""


def render_dev_check(tensor_schema: dict[str, Any]) -> str:
    schema = json.dumps(tensor_schema, sort_keys=True)
    return f'''import ast
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

TENSOR_SCHEMA = {schema}

source_path = Path("kernel.py")
source = source_path.read_text(encoding="utf-8")
tree = ast.parse(source)
workloads = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "workload"
]
if len(workloads) != 1:
    raise SystemExit("WORKLOAD_FUNCTION_REQUIRED")
if any(
    isinstance(node, ast.keyword)
    and node.arg == "interpret"
    and isinstance(node.value, ast.Constant)
    and node.value.value is True
    for node in ast.walk(tree)
):
    raise SystemExit("INTERPRET_MODE_FORBIDDEN")

spec = importlib.util.spec_from_file_location("candidate_kernel", source_path)
if spec is None or spec.loader is None:
    raise SystemExit("KERNEL_IMPORT_SPEC_INVALID")
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except Exception as exc:
    raise SystemExit(f"KERNEL_IMPORT_FAILED:{{type(exc).__name__}}:{{exc}}") from exc
workload = getattr(module, "workload", None)
if not callable(workload):
    raise SystemExit("WORKLOAD_CALLABLE_REQUIRED")
inputs = tuple(
    jax.ShapeDtypeStruct(tuple(item["shape"]), jnp.dtype(item["dtype"]))
    for item in TENSOR_SCHEMA["inputs"]
)
try:
    traced = str(jax.make_jaxpr(workload)(*inputs))
except Exception as exc:
    raise SystemExit(f"WORKLOAD_TRACE_FAILED:{{type(exc).__name__}}:{{exc}}") from exc
if "pallas_call[" not in traced:
    raise SystemExit("PALLAS_PRIMITIVE_REQUIRED")
print("PUBLIC_TRACE_OK")
'''
