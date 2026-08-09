import ast
import pathlib
import sys

from candidate_policy import candidate_module_policy_error

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "kernel.py")
try:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
except Exception as exc:
    print(f"DEV_CHECK output_contract {type(exc).__name__}: {exc}")
    raise SystemExit(2)
policy_error = candidate_module_policy_error(
    source, allowed_entrypoints=()
)
if policy_error is not None:
    print(f"DEV_CHECK candidate_policy {policy_error}")
    raise SystemExit(2)
workloads = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "workload"]
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
pallas = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "pallas_call"]
blocks = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "BlockSpec"]
interpreted = any(any(keyword.arg == "interpret" for keyword in call.keywords) for call in pallas)
reversed_blocks = any(len(call.args) >= 2 and isinstance(call.args[0], ast.Lambda) for call in blocks)
if len(workloads) != 1 or interpreted or reversed_blocks:
    print("DEV_CHECK pallas_api workload=1, normal lowering, and BlockSpec(block_shape, index_map) are required")
    raise SystemExit(2)
print("DEV_CHECK static_complete")
