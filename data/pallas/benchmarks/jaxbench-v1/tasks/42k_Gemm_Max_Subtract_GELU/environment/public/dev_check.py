import ast
from pathlib import Path

source = Path("kernel.py").read_text(encoding="utf-8")
tree = ast.parse(source)
workloads = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "workload"
]
if len(workloads) != 1:
    raise SystemExit("WORKLOAD_FUNCTION_REQUIRED")
if "interpret=True" in source or "interpret = True" in source:
    raise SystemExit("INTERPRET_MODE_FORBIDDEN")
if not any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "pallas_call"
    for node in ast.walk(tree)
) and "jax.experimental.pallas.ops" not in source:
    raise SystemExit("PALLAS_ENTRYPOINT_REQUIRED")
print("PUBLIC_CONTRACT_OK")
