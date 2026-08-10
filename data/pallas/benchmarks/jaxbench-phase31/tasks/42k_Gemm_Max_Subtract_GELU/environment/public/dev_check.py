import ast
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

TENSOR_SCHEMA = {"inputs": [{"dtype": "bfloat16", "name": "x", "shape": [4096, 8192]}, {"dtype": "bfloat16", "name": "weight", "shape": [8192, 8192]}, {"dtype": "bfloat16", "name": "bias", "shape": [8192]}], "outputs": [{"dtype": "bfloat16", "shape": [4096, 1]}]}

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
    raise SystemExit(f"KERNEL_IMPORT_FAILED:{type(exc).__name__}:{exc}") from exc
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
    raise SystemExit(f"WORKLOAD_TRACE_FAILED:{type(exc).__name__}:{exc}") from exc
if "pallas_call[" not in traced:
    raise SystemExit("PALLAS_PRIMITIVE_REQUIRED")
print("PUBLIC_TRACE_OK")
