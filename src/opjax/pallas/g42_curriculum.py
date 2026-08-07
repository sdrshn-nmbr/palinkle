"""Build and validate the balanced G4.2 Pallas repair-task release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import tomli

from opjax.pallas.g42_harness import (
    TASK_SCHEMA_VERSION,
    canonical_sha256,
    file_sha256,
    load_task_package,
    validate_task_release,
)
from opjax.pallas.task_semantics import operation_specification, render_task_instruction

MUTATIONS = (
    "reversed_blockspec",
    "illegal_block_geometry",
    "unsafe_grid_index",
    "incomplete_compute",
)


class G42CurriculumError(RuntimeError):
    """The repair curriculum cannot be derived without weakening its contract."""


class _SingleMutation(ast.NodeTransformer):
    def __init__(self, mutation: str) -> None:
        self.mutation = mutation
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.changed or not isinstance(node.func, ast.Attribute) or node.func.attr != "BlockSpec" or len(node.args) < 2:
            return node
        if self.mutation == "reversed_blockspec":
            node.args[0], node.args[1] = node.args[1], node.args[0]
            self.changed = True
        elif self.mutation == "illegal_block_geometry":
            original = node.args[0]
            dimensions = len(original.elts) if isinstance(original, (ast.Tuple, ast.List)) else 2
            node.args[0] = ast.Tuple(elts=[ast.Constant(7) for _ in range(dimensions)], ctx=ast.Load())
            self.changed = True
        elif self.mutation == "unsafe_grid_index" and isinstance(node.args[1], ast.Lambda):
            body = node.args[1].body
            if isinstance(body, ast.Tuple):
                body.elts = [ast.BinOp(left=item, op=ast.Add(), right=ast.Constant(1)) for item in body.elts]
            else:
                node.args[1].body = ast.BinOp(left=body, op=ast.Add(), right=ast.Constant(1))
            self.changed = True
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if self.mutation != "incomplete_compute" or self.changed:
            return node
        if any(isinstance(target, ast.Subscript) for target in node.targets):
            node.value = ast.BinOp(left=ast.Constant(0.0), op=ast.Mult(), right=node.value)
            self.changed = True
        return node


def mutate_kernel(source: str, mutation: str) -> str:
    if mutation not in MUTATIONS:
        raise G42CurriculumError(f"MUTATION_INVALID: {mutation}")
    tree = ast.parse(source)
    transformer = _SingleMutation(mutation)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    if not transformer.changed:
        raise G42CurriculumError(f"MUTATION_NOT_APPLICABLE: {mutation}")
    return ast.unparse(tree) + "\n"


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:72]


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _task_toml(*, task_id: str, metadata: dict[str, Any], task_sha256: str | None) -> str:
    shapes = json.dumps(metadata["input_shapes"], separators=(",", ":"))
    dtypes = json.dumps(metadata["input_dtypes"], separators=(",", ":"))
    return f'''schema_version = "{TASK_SCHEMA_VERSION}"
artifacts = ["/logs/artifacts/model.patch"]

[task]
name = "opjax/{task_id}"
description = "Repair one authentic Pallas kernel"

[metadata]
task_id = "{task_id}"
split = "{metadata['split']}"
mode = "{metadata['mode']}"
family = "{metadata['family']}"
source_row_id = "{metadata['source_row_id']}"
mutation = "{metadata['mutation']}"
task_sha256 = {json.dumps(task_sha256 or "")}
input_shapes_json = {json.dumps(shapes)}
input_dtypes_json = {json.dumps(dtypes)}

[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 900.0

[verifier.environment]
cpus = 2
memory_mb = 8192
storage_mb = 20480

[agent]
network_mode = "no-network"
timeout_sec = 900.0

[environment]
docker_image = "python:3.12-slim"
os = "linux"
cpus = 2
memory_mb = 4096
storage_mb = 4096
gpus = 0
'''


PUBLIC_API = """# Pallas task API

Edit `kernel.py`. It must define one complete `workload(*inputs)` implementation.
Use `pl.BlockSpec(block_shape, index_map)` in that order. The kernel must use a
reachable `pl.pallas_call`, must not use `interpret=True`, and must not include a
plain-JAX fallback. Run `python dev_check.py kernel.py` for public static feedback.
The final TPU verifier is separate and hidden.
"""


DEV_CHECK = '''import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "kernel.py")
try:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
except Exception as exc:
    print(f"DEV_CHECK output_contract {type(exc).__name__}: {exc}")
    raise SystemExit(2)
workloads = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "workload"]
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
pallas = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "pallas_call"]
blocks = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "BlockSpec"]
interpreted = any(any(keyword.arg == "interpret" for keyword in call.keywords) for call in pallas)
reversed_blocks = any(len(call.args) >= 2 and isinstance(call.args[0], ast.Lambda) for call in blocks)
if len(workloads) != 1 or not pallas or not blocks or interpreted or reversed_blocks:
    print("DEV_CHECK pallas_api workload=1, reachable pallas_call, normal lowering, and BlockSpec(block_shape, index_map) are required")
    raise SystemExit(2)
print("DEV_CHECK static_complete")
'''


def _write_task(*, root: Path, row: dict[str, Any], task: dict[str, Any], mutation: str, ordinal: int) -> dict[str, Any]:
    family = row["family_category"]
    task_id = f"g42-{_slug(family)}-{ordinal:02d}-{_slug(mutation)}"
    solution = row["messages"][-1]["content"]
    starter = mutate_kernel(solution, mutation)
    expected_stage = {
        "reversed_blockspec": "pallas_api",
        "illegal_block_geometry": "tpu_compile",
        "unsafe_grid_index": (
            "runtime_safety"
            if family in {"matmul", "normalization", "row_reduction", "softmax"}
            else "full_shape_correctness"
        ),
        "incomplete_compute": "full_shape_correctness",
    }[mutation]
    metadata = {
        "split": "train",
        "mode": "curriculum",
        "family": family,
        "source_row_id": row["row_id"],
        "mutation": mutation,
        "input_shapes": task["input_shapes"],
        "input_dtypes": task["input_dtypes"],
    }
    instruction = render_task_instruction(task, repair=mutation)
    verifier_task = {
        **task,
        "task_id": task_id,
        "source_row_id": row["row_id"],
        "family": family,
        "mutation": mutation,
        "expected_initial_failure_stage": expected_stage,
        "correctness_seeds": [0, 1, 2],
        "reference_kernel_sha256": row["verification"]["kernel_sha256"],
    }
    return _write_package(
        root=root,
        task_id=task_id,
        metadata=metadata,
        instruction=instruction,
        starter=starter,
        solution=solution,
        verifier_task=verifier_task,
        expected_stage=expected_stage,
    )


def _write_package(
    *,
    root: Path,
    task_id: str,
    metadata: dict[str, Any],
    instruction: str,
    starter: str,
    solution: str,
    verifier_task: dict[str, Any],
    expected_stage: str,
) -> dict[str, Any]:
    task_root = root / "tasks" / task_id
    specification = operation_specification(verifier_task)
    verifier_task = {
        **verifier_task,
        "public_specification": specification,
        "public_specification_sha256": canonical_sha256(specification),
    }
    _write(task_root / "instruction.md", instruction)
    _write(task_root / "environment" / "Dockerfile", "FROM python:3.12-slim\nWORKDIR /workspace\n")
    _write(task_root / "environment" / "starter" / "kernel.py", starter)
    _write(task_root / "environment" / "public" / "dev_check.py", DEV_CHECK)
    _write(task_root / "environment" / "public" / "PALLAS_API.md", PUBLIC_API)
    _write(task_root / "solution" / "kernel.py", solution)
    _write(task_root / "tests" / "task.json", json.dumps(verifier_task, indent=2, sort_keys=True) + "\n")
    _write(
        task_root / "pre_artifacts.sh",
        "#!/bin/sh\nset -eu\nmkdir -p /logs/artifacts\ngit -C /workspace diff --binary "
        "$(git -C /workspace rev-list --max-parents=0 HEAD) HEAD > /logs/artifacts/model.patch\n",
        executable=True,
    )
    _write(
        task_root / "tests" / "test.sh",
        "#!/bin/sh\nset -eu\nopjax-pallas-environment-runner --task /tests/task.json "
        "--kernel /app/kernel.py --evidence-dir /logs/verifier/evidence\n",
        executable=True,
    )
    _write(task_root / "task.toml", _task_toml(task_id=task_id, metadata=metadata, task_sha256=None))
    package = load_task_package_without_hash(task_root)
    _write(task_root / "task.toml", _task_toml(task_id=task_id, metadata=metadata, task_sha256=package["task_sha256"]))
    validated = load_task_package(task_root)
    return {
        "task_id": validated.task_id,
        "task_sha256": validated.task_sha256,
        "family": metadata["family"],
        "source_row_id": metadata["source_row_id"],
        "mutation": metadata["mutation"],
        "expected_initial_failure_stage": expected_stage,
    }


def load_task_package_without_hash(root: Path) -> dict[str, Any]:
    manifest = tomli.loads((root / "task.toml").read_text(encoding="utf-8"))
    manifest["metadata"]["task_sha256"] = None
    required = (
        "instruction.md",
        "pre_artifacts.sh",
        "environment/Dockerfile",
        "environment/starter/kernel.py",
        "environment/public/dev_check.py",
        "environment/public/PALLAS_API.md",
        "tests/task.json",
        "tests/test.sh",
        "solution/kernel.py",
    )
    hashes = {relative: file_sha256(root / relative) for relative in required}
    return {"task_sha256": canonical_sha256({"manifest": manifest, "files": hashes})}


def build_repair_release(*, source_root: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise G42CurriculumError(f"OUTPUT_EXISTS: {out_dir}")
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (source_root / "datasets" / "sft.jsonl").read_text(encoding="utf-8").splitlines() if line]
    tasks = [json.loads(line) for line in (source_root / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_task_id = {task["task_id"]: task for task in tasks}
    if len(rows) != 32 or len(by_task_id) != 32:
        raise G42CurriculumError(f"SOURCE_COUNT_INVALID: rows={len(rows)} tasks={len(by_task_id)}")
    out_dir.mkdir(parents=True)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    family_ordinals: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows):
        family = row["family_category"]
        mutation = MUTATIONS[index % len(MUTATIONS)]
        family_ordinals[family] += 1
        record = _write_task(
            root=out_dir,
            row=row,
            task=by_task_id[row["row_id"]],
            mutation=mutation,
            ordinal=family_ordinals[family],
        )
        records.append(record)
        by_family[family].append(record)
    for family, family_rows in sorted(by_family.items()):
        if len(family_rows) >= 4:
            continue
        source_record = family_rows[0]
        row = next(item for item in rows if item["row_id"] == source_record["source_row_id"])
        alternate = next(mutation for mutation in MUTATIONS if mutation != source_record["mutation"])
        family_ordinals[family] += 1
        record = _write_task(
            root=out_dir,
            row=row,
            task=by_task_id[row["row_id"]],
            mutation=alternate,
            ordinal=family_ordinals[family],
        )
        records.append(record)
        by_family[family].append(record)
    selection = [record["task_id"] for family in sorted(by_family) for record in by_family[family][:4]]
    manifest = {
        "schema_version": 1,
        "kind": "pallas_g42_task_release",
        "source_release_sha256": source_manifest["release_sha256"],
        "source_dataset_sha256": source_manifest["artifacts"]["datasets/sft.jsonl"],
        "counts": {"source_rows": 32, "pool": len(records), "training_selection": len(selection)},
        "tasks": [f"tasks/{record['task_id']}" for record in records],
        "task_records": records,
        "training_selection": selection,
        "family_policy": {"families": 8, "selected_per_family": 4},
        "release_sha256": canonical_sha256(
            {
                "tasks": [{"task_id": record["task_id"], "task_sha256": record["task_sha256"]} for record in records],
                "training_selection": selection,
                "source_release_sha256": source_manifest["release_sha256"],
            }
        ),
    }
    _write(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validation = validate_task_release(out_dir)
    return {**manifest, "validation": validation}


def _benchmark_reference(task: dict[str, Any]) -> str:
    operation = task["operation"]
    shapes = task["input_shapes"]
    if operation == "add":
        rows, columns = shapes[0]
        expression = "x_ref[...] + y_ref[...]"
        return f'''import jax
from jax.experimental import pallas as pl

SHAPE = ({rows}, {columns})

def _kernel(x_ref, y_ref, o_ref):
    o_ref[...] = {expression}

def workload(x, y):
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct(SHAPE, x.dtype), grid=(SHAPE[0] // 128, SHAPE[1] // 128), in_specs=(spec, spec), out_specs=spec)(x, y)
'''
    if operation == "matmul":
        rows, inner = shapes[0]
        _, columns = shapes[1]
        return f'''import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

M, K, N = {rows}, {inner}, {columns}

def _kernel(x_ref, y_ref, o_ref):
    o_ref[...] = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

def workload(x, y):
    x_spec = pl.BlockSpec((128, K), lambda i, j: (i, 0))
    y_spec = pl.BlockSpec((K, 128), lambda i, j: (0, j))
    out_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32), grid=(M // 128, N // 128), in_specs=(x_spec, y_spec), out_specs=out_spec)(x, y)
'''
    if operation in {"rmsnorm", "row_sum"}:
        rows, columns = shapes[0]
        body = (
            "values = x_ref[...].astype(jnp.float32)\n    mean_square = jnp.mean(jnp.square(values), axis=-1, keepdims=True)\n    o_ref[...] = values * jax.lax.rsqrt(mean_square + 1e-5)"
            if operation == "rmsnorm"
            else "reduced = jnp.sum(x_ref[...], axis=-1, keepdims=True)\n    o_ref[...] = jnp.broadcast_to(reduced, x_ref.shape)"
        )
        return f'''import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

SHAPE = ({rows}, {columns})

def _kernel(x_ref, o_ref):
    {body}

def workload(x):
    spec = pl.BlockSpec((8, SHAPE[1]), lambda i: (i, 0))
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32), grid=(SHAPE[0] // 8,), in_specs=(spec,), out_specs=spec)(x)
'''
    raise G42CurriculumError(f"BENCHMARK_OPERATION_UNSUPPORTED: {operation}")


def validate_benchmark_release(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "pallas_g42_benchmark_release":
        raise G42CurriculumError(f"BENCHMARK_RELEASE_KIND_INVALID: {root}")
    packages = [load_task_package(root / relative) for relative in manifest.get("tasks", [])]
    if len(packages) != 4 or any(package.mode != "benchmark" for package in packages):
        raise G42CurriculumError(f"BENCHMARK_TASKS_INVALID: {len(packages)}")
    observed = canonical_sha256(
        [{"task_id": package.task_id, "task_sha256": package.task_sha256} for package in packages]
    )
    if manifest.get("release_sha256") != observed:
        raise G42CurriculumError(f"BENCHMARK_RELEASE_HASH_MISMATCH: {root}")
    return {"task_count": len(packages), "release_sha256": observed}


def build_benchmark_release(*, diagnostic_config: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise G42CurriculumError(f"OUTPUT_EXISTS: {out_dir}")
    config = json.loads(diagnostic_config.read_text(encoding="utf-8"))
    heldout = config.get("heldout_tasks", [])
    if len(heldout) != 4:
        raise G42CurriculumError(f"HELDOUT_TASK_COUNT_INVALID: {len(heldout)}")
    out_dir.mkdir(parents=True)
    records = []
    for task in heldout:
        task_id = f"g42-benchmark-{_slug(task['task_id'])}"
        metadata = {
            "split": "near_heldout",
            "mode": "benchmark",
            "family": task["operation"],
            "source_row_id": f"near-heldout:{task['task_id']}",
            "mutation": "missing_implementation",
            "input_shapes": task["input_shapes"],
            "input_dtypes": task["input_dtypes"],
        }
        solution = _benchmark_reference(task)
        verifier_task = {
            **task,
            "task_id": task_id,
            "source_task_id": task["task_id"],
            "family": task["operation"],
            "mutation": "missing_implementation",
            "expected_initial_failure_stage": "artifact_contract",
            "correctness_seeds": [0, 1, 2],
            "correctness_tolerance": {"rtol": 0.001, "atol": 0.001},
            "reference_kernel_sha256": hashlib.sha256(solution.encode()).hexdigest(),
        }
        record = _write_package(
            root=out_dir,
            task_id=task_id,
            metadata=metadata,
            instruction=render_task_instruction(task, repair=None),
            starter="def workload(*inputs):\n    ...\n",
            solution=solution,
            verifier_task=verifier_task,
            expected_stage="artifact_contract",
        )
        records.append(record)
    manifest = {
        "schema_version": 1,
        "kind": "pallas_g42_benchmark_release",
        "source_config_sha256": file_sha256(diagnostic_config),
        "counts": {"tasks": 4},
        "tasks": [f"tasks/{record['task_id']}" for record in records],
        "task_records": records,
        "release_sha256": canonical_sha256(
            [{"task_id": record["task_id"], "task_sha256": record["task_sha256"]} for record in records]
        ),
    }
    _write(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "validation": validate_benchmark_release(out_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g42-curriculum")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    build_benchmark = commands.add_parser("build-benchmark")
    build_benchmark.add_argument("--diagnostic-config", type=Path, required=True)
    build_benchmark.add_argument("--out-dir", type=Path, required=True)
    validate_benchmark = commands.add_parser("validate-benchmark")
    validate_benchmark.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_repair_release(source_root=args.source_root, out_dir=args.out_dir)
        elif args.command == "validate":
            result = validate_task_release(args.root)
        elif args.command == "build-benchmark":
            result = build_benchmark_release(
                diagnostic_config=args.diagnostic_config,
                out_dir=args.out_dir,
            )
        else:
            result = validate_benchmark_release(args.root)
    except (G42CurriculumError, ValueError, OSError) as exc:
        print(f"G42_CURRICULUM_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
