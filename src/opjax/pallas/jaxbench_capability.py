"""Build the full, original-shape JAXBench capability task release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


PINNED_JAXBENCH_REVISION = "6b6c44293c43976032ba12d2f72d6bebeaf2394f"
EXPECTED_WORKLOAD_COUNT = 50
HARBOR_SCHEMA_VERSION = "1.3"
PUBLIC_WORKSPACE_FILES = (
    "instruction.md",
    "kernel.py",
    "PALLAS_API.md",
    "dev_check.py",
)
PYTHON_IMAGE = (
    "python:3.12.11-slim-bookworm@sha256:"
    "519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
)
WORKER_SOURCE_FILES = (
    "jaxbench_executable.py",
    "jaxbench_verifier.py",
    "jaxbench_worker.py",
)


class JaxBenchCapabilityError(RuntimeError):
    """The full JAXBench capability release violates its frozen contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _shingles(value: str, width: int = 7) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", value.lower())
    if not tokens:
        return []
    if len(tokens) < width:
        return [" ".join(tokens)]
    return sorted(
        {
            " ".join(tokens[index : index + width])
            for index in range(len(tokens) - width + 1)
        }
    )


def _write(path: Path, value: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _git(path: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise JaxBenchCapabilityError(
            f"JAXBENCH_GIT_FAILED:{args[0]}:{process.stderr.strip()[-500:]}"
        )
    return process.stdout.strip()


def validate_source_checkout(source_root: Path) -> list[Path]:
    revision = _git(source_root, "rev-parse", "HEAD")
    if revision != PINNED_JAXBENCH_REVISION:
        raise JaxBenchCapabilityError(f"JAXBENCH_REVISION_INVALID:{revision}")
    if _git(source_root, "status", "--porcelain"):
        raise JaxBenchCapabilityError("JAXBENCH_CHECKOUT_DIRTY")
    benchmark_root = source_root / "JAXBench/benchmark"
    baselines = sorted(benchmark_root.glob("*/baseline.py"))
    if len(baselines) != EXPECTED_WORKLOAD_COUNT:
        raise JaxBenchCapabilityError(
            f"JAXBENCH_WORKLOAD_COUNT_INVALID:{len(baselines)}"
        )
    return baselines


def _source_contract(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    config_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "CONFIG"
                for target in node.targets
            )
        ),
        None,
    )
    workload = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "workload"
        ),
        None,
    )
    create_inputs = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_inputs"
        ),
        None,
    )
    if config_node is None or workload is None or create_inputs is None:
        raise JaxBenchCapabilityError("JAXBENCH_BASELINE_CONTRACT_INCOMPLETE")
    try:
        config = ast.literal_eval(config_node.value)
    except (TypeError, ValueError) as exc:
        raise JaxBenchCapabilityError("JAXBENCH_CONFIG_NOT_STATIC") from exc
    if not isinstance(config, dict):
        raise JaxBenchCapabilityError("JAXBENCH_CONFIG_INVALID")
    module_doc = ast.get_docstring(tree)
    workload_doc = ast.get_docstring(workload)
    if not module_doc or not workload_doc:
        raise JaxBenchCapabilityError("JAXBENCH_PUBLIC_DOCUMENTATION_MISSING")
    return {
        "module_documentation": module_doc,
        "configuration": config,
        "workload_signature": f"def workload({ast.unparse(workload.args)})",
        "workload_documentation": workload_doc,
        "input_argument_names": [argument.arg for argument in workload.args.args],
        "semantic_contract": _semantic_contract(tree, workload),
    }


def _semantic_node(value: Any) -> Any:
    if isinstance(value, ast.AST):
        result = {"kind": type(value).__name__}
        for field, child in ast.iter_fields(value):
            if field in {"ctx", "type_comment"}:
                continue
            result[field] = _semantic_node(child)
        return result
    if isinstance(value, list):
        return [_semantic_node(item) for item in value]
    if value is Ellipsis:
        return {"kind": "Ellipsis"}
    return value


def _function_body(function: ast.FunctionDef) -> list[ast.stmt]:
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return body


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    bindings = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{module}.{alias.name}"
    return bindings


def _loaded_names(value: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(value)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _local_names(function: ast.FunctionDef) -> set[str]:
    names = {
        argument.arg
        for node in ast.walk(function)
        if isinstance(node, (ast.FunctionDef, ast.Lambda))
        for argument in node.args.args
    }
    names.update(
        node.name
        for node in ast.walk(function)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    names.update(
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    return names


def _semantic_contract(tree: ast.Module, workload: ast.FunctionDef) -> dict[str, Any]:
    imports = _import_bindings(tree)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name not in {"create_inputs", "workload", "benchmark", "get_flops"}
    }
    values: dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id != "CONFIG":
                values[target.id] = node.value
    builtins = {
        "CONFIG",
        "False",
        "None",
        "True",
        "abs",
        "float",
        "int",
        "len",
        "max",
        "min",
        "range",
        "tuple",
    }
    required_functions: dict[str, ast.FunctionDef] = {}
    required_values: dict[str, ast.AST] = {}
    required_imports: dict[str, str] = {}
    unresolved: set[str] = set()
    pending = _loaded_names(workload) - _local_names(workload)
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited or name in builtins:
            continue
        visited.add(name)
        if name in imports:
            required_imports[name] = imports[name]
        elif name in functions:
            function = functions[name]
            required_functions[name] = function
            pending.update(_loaded_names(function) - _local_names(function))
        elif name in values:
            value = values[name]
            required_values[name] = value
            pending.update(_loaded_names(value))
        else:
            unresolved.add(name)
    if unresolved:
        raise JaxBenchCapabilityError(
            "JAXBENCH_SEMANTIC_CONTRACT_UNRESOLVED:" + ",".join(sorted(unresolved))
        )
    return {
        "format": "canonical_python_ast_semantics_v1",
        "imports": dict(sorted(required_imports.items())),
        "module_values": {
            name: _semantic_node(value)
            for name, value in sorted(required_values.items())
        },
        "helper_functions": {
            name: {
                "arguments": [argument.arg for argument in function.args.args],
                "body": [
                    _semantic_node(statement) for statement in _function_body(function)
                ],
            }
            for name, function in sorted(required_functions.items())
        },
        "arguments": [argument.arg for argument in workload.args.args],
        "body": [_semantic_node(statement) for statement in _function_body(workload)],
        "unresolved_names": [],
    }


def _probe_tensor_schema(*, baseline_path: Path, source_root: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    python_path = [str(source_root)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "opjax.pallas.jaxbench_schema",
            "--baseline",
            str(baseline_path),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if process.returncode != 0:
        raise JaxBenchCapabilityError(
            f"JAXBENCH_SCHEMA_PROBE_FAILED:{baseline_path.parent.name}:"
            f"{process.stderr.strip()[-500:]}"
        )
    return json.loads(process.stdout)


def render_public_specification(
    *,
    workload_id: str,
    baseline_source: str,
    tensor_schema: dict[str, Any] | None = None,
) -> str:
    contract = _source_contract(baseline_source)
    configuration = json.dumps(contract["configuration"], indent=2, sort_keys=True)
    schema_section = ""
    if tensor_schema is not None:
        schema_section = (
            "## Tensor contract\n\n"
            f"```json\n{json.dumps(tensor_schema, indent=2, sort_keys=True)}\n```\n\n"
        )
    semantic_section = (
        "## Exact semantic contract\n\n"
        "The following non-executable canonical AST defines operation order, "
        "constants, axes, layouts, padding, precision, and all other observable "
        "semantics. `Name` and `Attribute` nodes name mathematical/JAX operations; "
        "the hidden source implementation is not included.\n\n"
        f"```json\n{json.dumps(contract['semantic_contract'], indent=2, sort_keys=True)}\n```\n\n"
    )
    return (
        f"# {workload_id}\n\n"
        "Implement this pinned JAXBench workload as a normally lowered TPU Pallas "
        "kernel. Preserve the original workload semantics and original deployment "
        "shapes. The hidden verifier creates the exact inputs, computes the JAX "
        "baseline, checks normal Pallas lowering, captures a TPU profile, and "
        "compares performance against XLA.\n\n"
        f"{contract['module_documentation']}\n\n"
        "## Configuration\n\n"
        f"```json\n{configuration}\n```\n\n"
        "## Required interface\n\n"
        f"```python\n{contract['workload_signature']}:\n    ...\n```\n\n"
        f"{schema_section}"
        f"{semantic_section}"
        f"{contract['workload_documentation']}\n\n"
        "The input generator, concrete test values, baseline implementation, "
        "optimized reference, correctness tests, and grading logic are hidden. "
        "Do not use `interpret=True` or a plain-JAX fallback.\n"
    )


def _render_dev_check() -> str:
    return """import ast
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
"""


def _task_toml(
    *,
    workload_id: str,
    task_sha256: str,
    baseline_sha256: str,
    optimized_sha256: str | None,
) -> str:
    optimized = optimized_sha256 or ""
    return f'''schema_version = "{HARBOR_SCHEMA_VERSION}"
artifacts = ["/logs/artifacts/model.patch"]

[task]
name = "opjax/jaxbench/{workload_id}"
description = "Implement the original-shape pinned JAXBench workload in TPU Pallas"
authors = []
keywords = ["jaxbench", "jax", "pallas", "tpu", "kernel"]

[metadata]
task_id = "{workload_id}"
display_title = {json.dumps(workload_id.replace("_", " "))}
category = "kernel_optimization"
language = "python"
split = "sealed_public_evaluation"
mode = "benchmark"
suite = "jaxbench"
shape_policy = "original_unmodified"
jaxbench_revision = "{PINNED_JAXBENCH_REVISION}"
jaxbench_baseline_sha256 = "{baseline_sha256}"
jaxbench_optimized_sha256 = "{optimized}"
task_sha256 = "{task_sha256}"
authoritative_verifier = "hidden-jaxbench-disposable-tpu-worker"

[verifier]
network_mode = "no-network"
environment_mode = "external"
timeout_sec = 5400.0

[agent]
network_mode = "no-network"
timeout_sec = 5400.0

[environment]
build_timeout_sec = 1800.0
os = "linux"
cpus = 2
memory_mb = 8192
storage_mb = 20480
gpus = 0
mcp_servers = []

[environment.env]
[solution.env]
'''


def _task_hash(task_root: Path) -> str:
    manifest = tomllib.loads((task_root / "task.toml").read_text())
    manifest["metadata"]["task_sha256"] = ""
    files = {
        path.relative_to(task_root).as_posix(): file_sha256(path)
        for path in sorted(task_root.rglob("*"))
        if path.is_file() and path.name != "task.toml"
    }
    return canonical_sha256({"manifest": manifest, "files": files})


def _contamination_signatures(release_root: Path) -> dict[str, Any]:
    documents = []
    identifiers = []
    for task_root in sorted((release_root / "tasks").iterdir()):
        task = json.loads((task_root / "tests/task.json").read_text(encoding="utf-8"))
        identifiers.extend(
            (
                task_root.name,
                task["baseline_sha256"],
                task.get("optimized_sha256") or "",
            )
        )
        forbidden_paths = [task_root / "instruction.md"]
        forbidden_paths.extend(sorted((task_root / "tests/jaxbench").glob("*.py")))
        for path in forbidden_paths:
            source = path.read_text(encoding="utf-8")
            documents.append(
                {
                    "task_id": task_root.name,
                    "path": path.relative_to(task_root).as_posix(),
                    "sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "shingles": _shingles(source),
                }
            )
    return {
        "schema_version": 1,
        "policy": "forbidden_from_all_training_splits",
        "documents": documents,
        "identifiers": sorted(
            identifier for identifier in set(identifiers) if identifier
        ),
    }


def _build_task(
    *, baseline_path: Path, release_root: Path, source_root: Path
) -> dict[str, Any]:
    workload_id = baseline_path.parent.name
    task_root = release_root / "tasks" / workload_id
    baseline_source = baseline_path.read_text(encoding="utf-8")
    optimized_path = baseline_path.with_name("optimized.py")
    baseline_sha = file_sha256(baseline_path)
    optimized_sha = file_sha256(optimized_path) if optimized_path.is_file() else None
    tensor_schema = _probe_tensor_schema(
        baseline_path=baseline_path, source_root=source_root
    )
    public_spec = render_public_specification(
        workload_id=workload_id,
        baseline_source=baseline_source,
        tensor_schema=tensor_schema,
    )
    contract = _source_contract(baseline_source)

    _write(task_root / "instruction.md", public_spec)
    _write(
        task_root / "environment/starter/kernel.py", "def workload(*inputs):\n    ...\n"
    )
    _write(
        task_root / "environment/public/PALLAS_API.md",
        "Use JAX Pallas with normal TPU lowering. `interpret=True` and plain-JAX "
        "fallbacks are forbidden. Run `python dev_check.py` before submission.\n",
    )
    _write(task_root / "environment/public/dev_check.py", _render_dev_check())
    _write(
        task_root / "environment/Dockerfile",
        f"FROM {PYTHON_IMAGE}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "WORKDIR /app\n"
        "COPY starter/kernel.py /app/kernel.py\n"
        "COPY public/PALLAS_API.md /app/PALLAS_API.md\n"
        "COPY public/dev_check.py /app/dev_check.py\n"
        "RUN git init -q && git config user.name opjax-harness "
        "&& git config user.email harness@opjax.invalid && git add . "
        "&& git commit -q -m 'task base'\n",
    )
    _write(
        task_root / "pre_artifacts.sh",
        "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/artifacts\n"
        "git add -A\n"
        "git -c user.name=opjax-submit -c user.email=submit@opjax.invalid "
        "commit --allow-empty -q -m submission\n"
        "git diff --binary $(git rev-list --max-parents=0 HEAD) HEAD "
        "> /logs/artifacts/model.patch\n",
        executable=True,
    )
    (task_root / "tests/jaxbench").mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline_path, task_root / "tests/jaxbench/baseline.py")
    if optimized_path.is_file():
        shutil.copy2(optimized_path, task_root / "tests/jaxbench/optimized.py")
    verifier_contract = {
        "schema_version": 1,
        "task_id": workload_id,
        "jaxbench_revision": PINNED_JAXBENCH_REVISION,
        "baseline_sha256": baseline_sha,
        "optimized_sha256": optimized_sha,
        "public_specification_sha256": hashlib.sha256(public_spec.encode()).hexdigest(),
        "input_argument_names": contract["input_argument_names"],
        "tensor_schema": tensor_schema,
        "shape_policy": "original_unmodified",
    }
    _write(
        task_root / "tests/task.json",
        json.dumps(verifier_contract, indent=2, sort_keys=True) + "\n",
    )
    _write(
        task_root / "tests/test.sh",
        "#!/bin/bash\nset -euo pipefail\n"
        "test -f /app/kernel.py\n"
        "mkdir -p /logs/artifacts\n"
        "git -C /app add -A\n"
        "git -C /app -c user.name=opjax-submit "
        "-c user.email=submit@opjax.invalid commit --allow-empty -q -m submission\n"
        "git -C /app diff --binary $(git -C /app rev-list --max-parents=0 HEAD) "
        "HEAD > /logs/artifacts/model.patch\n"
        "printf '%s\\n' TPU_WORKER_REQUIRED >&2\n"
        "exit 2\n",
        executable=True,
    )
    _write(
        task_root / "task.toml",
        _task_toml(
            workload_id=workload_id,
            task_sha256="",
            baseline_sha256=baseline_sha,
            optimized_sha256=optimized_sha,
        ),
    )
    task_sha = _task_hash(task_root)
    _write(
        task_root / "task.toml",
        _task_toml(
            workload_id=workload_id,
            task_sha256=task_sha,
            baseline_sha256=baseline_sha,
            optimized_sha256=optimized_sha,
        ),
    )
    return {
        "task_id": workload_id,
        "path": f"tasks/{workload_id}",
        "task_sha256": task_sha,
        "baseline_sha256": baseline_sha,
        "optimized_sha256": optimized_sha,
        "public_specification_sha256": verifier_contract["public_specification_sha256"],
        "shape_policy": "original_unmodified",
    }


def build_release(*, source_root: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise JaxBenchCapabilityError(f"OUTPUT_EXISTS:{out_dir}")
    baselines = validate_source_checkout(source_root)
    out_dir.mkdir(parents=True)
    upstream_license = source_root / "LICENSE"
    if not upstream_license.is_file():
        raise JaxBenchCapabilityError("JAXBENCH_LICENSE_MISSING")
    shutil.copy2(upstream_license, out_dir / "UPSTREAM_LICENSE")
    tasks = [
        _build_task(
            baseline_path=baseline,
            release_root=out_dir,
            source_root=source_root,
        )
        for baseline in baselines
    ]
    signatures_path = out_dir / "contamination-signatures.json"
    _write(
        signatures_path,
        json.dumps(_contamination_signatures(out_dir), indent=2, sort_keys=True) + "\n",
    )
    worker_lock = (
        Path(__file__).parents[3]
        / "config/pallas/phase2-worker-requirements.lock"
    )
    manifest = {
        "schema_version": 1,
        "kind": "opjax_jaxbench_capability_benchmark",
        "benchmark_id": "opjax-jaxbench-full-v1",
        "status": "frozen",
        "execution_status": "worker_adapter_ready",
        "scoreability_status": "original_shape_canary_only",
        "jaxbench_revision": PINNED_JAXBENCH_REVISION,
        "shape_policy": "original_unmodified",
        "upstream_license_sha256": file_sha256(out_dir / "UPSTREAM_LICENSE"),
        "builder_source_sha256": file_sha256(Path(__file__)),
        "schema_probe_source_sha256": file_sha256(
            Path(__file__).with_name("jaxbench_schema.py")
        ),
        "worker_requirements_lock_sha256": file_sha256(worker_lock),
        "worker_source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in WORKER_SOURCE_FILES
        },
        "runtime": {
            "python": "3.12.11",
            "jax": "0.10.1",
            "jaxlib": "0.10.1",
            "libtpu": "0.0.41",
            "accelerator_type": "v5litepod-1",
        },
        "task_count": len(tasks),
        "optimized_reference_count": sum(
            task["optimized_sha256"] is not None for task in tasks
        ),
        "contamination_signatures_sha256": file_sha256(signatures_path),
        "tasks": tasks,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    _write(
        out_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    validate_release(root=out_dir, source_root=source_root)
    return manifest


def validate_release(*, root: Path, source_root: Path) -> dict[str, Any]:
    baselines = validate_source_checkout(source_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    signatures_path = root / "contamination-signatures.json"
    worker_lock = (
        Path(__file__).parents[3]
        / "config/pallas/phase2-worker-requirements.lock"
    )
    payload = dict(manifest)
    expected_release = payload.pop("release_sha256", None)
    if (
        manifest.get("kind") != "opjax_jaxbench_capability_benchmark"
        or canonical_sha256(payload) != expected_release
        or manifest.get("task_count") != EXPECTED_WORKLOAD_COUNT
        or manifest.get("shape_policy") != "original_unmodified"
        or manifest.get("execution_status") != "worker_adapter_ready"
        or manifest.get("scoreability_status") != "original_shape_canary_only"
        or manifest.get("upstream_license_sha256")
        != file_sha256(root / "UPSTREAM_LICENSE")
        or manifest.get("builder_source_sha256") != file_sha256(Path(__file__))
        or manifest.get("schema_probe_source_sha256")
        != file_sha256(Path(__file__).with_name("jaxbench_schema.py"))
        or manifest.get("worker_requirements_lock_sha256")
        != file_sha256(worker_lock)
        or manifest.get("worker_source_sha256")
        != {
            name: file_sha256(Path(__file__).with_name(name))
            for name in WORKER_SOURCE_FILES
        }
        or not signatures_path.is_file()
        or file_sha256(signatures_path)
        != manifest.get("contamination_signatures_sha256")
    ):
        raise JaxBenchCapabilityError("JAXBENCH_RELEASE_MANIFEST_INVALID")
    by_id = {task["task_id"]: task for task in manifest["tasks"]}
    expected_ids = {baseline.parent.name for baseline in baselines}
    if set(by_id) != expected_ids:
        raise JaxBenchCapabilityError("JAXBENCH_RELEASE_TASK_SET_INVALID")
    optimized_count = 0
    for baseline in baselines:
        task = by_id[baseline.parent.name]
        task_root = root / task["path"]
        task_manifest = tomllib.loads((task_root / "task.toml").read_text())
        hidden_baseline = task_root / "tests/jaxbench/baseline.py"
        optimized = baseline.with_name("optimized.py")
        hidden_optimized = task_root / "tests/jaxbench/optimized.py"
        if (
            task["task_sha256"] != _task_hash(task_root)
            or task_manifest["metadata"]["shape_policy"] != "original_unmodified"
            or file_sha256(hidden_baseline) != file_sha256(baseline)
            or hidden_optimized.is_file() != optimized.is_file()
        ):
            raise JaxBenchCapabilityError(
                f"JAXBENCH_RELEASE_TASK_INVALID:{baseline.parent.name}"
            )
        if optimized.is_file():
            optimized_count += 1
            if file_sha256(hidden_optimized) != file_sha256(optimized):
                raise JaxBenchCapabilityError(
                    f"JAXBENCH_OPTIMIZED_REFERENCE_DRIFT:{baseline.parent.name}"
                )
        public_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                task_root / "instruction.md",
                task_root / "environment/starter/kernel.py",
                task_root / "environment/public/PALLAS_API.md",
                task_root / "environment/public/dev_check.py",
            )
        )
        if "def create_inputs" in public_text or "jax.random" in public_text:
            raise JaxBenchCapabilityError(
                f"JAXBENCH_INPUT_GENERATOR_PUBLIC:{baseline.parent.name}"
            )
    if optimized_count != manifest.get("optimized_reference_count"):
        raise JaxBenchCapabilityError("JAXBENCH_OPTIMIZED_COUNT_INVALID")
    return {
        "release_sha256": expected_release,
        "task_count": len(by_id),
        "optimized_reference_count": optimized_count,
    }


def materialize_agent_workspace(
    *, task_root: Path, destination: Path
) -> dict[str, Any]:
    if destination.exists():
        raise JaxBenchCapabilityError(f"OUTPUT_EXISTS:{destination}")
    destination.mkdir(parents=True)
    shutil.copy2(task_root / "instruction.md", destination / "instruction.md")
    shutil.copy2(task_root / "environment/starter/kernel.py", destination / "kernel.py")
    shutil.copy2(
        task_root / "environment/public/PALLAS_API.md",
        destination / "PALLAS_API.md",
    )
    shutil.copy2(
        task_root / "environment/public/dev_check.py",
        destination / "dev_check.py",
    )
    observed = tuple(sorted(path.name for path in destination.iterdir()))
    if observed != tuple(sorted(PUBLIC_WORKSPACE_FILES)):
        raise JaxBenchCapabilityError("JAXBENCH_AGENT_WORKSPACE_INVALID")
    if any(path.name in {"tests", "solution"} for path in destination.rglob("*")):
        raise JaxBenchCapabilityError("JAXBENCH_HIDDEN_MATERIAL_PUBLIC")
    return {
        "task_id": task_root.name,
        "files": {
            path.name: file_sha256(path)
            for path in sorted(destination.iterdir())
            if path.is_file()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-jaxbench-capability")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--source", type=Path, required=True)
    validate.add_argument("--root", type=Path, required=True)
    workspace = commands.add_parser("workspace")
    workspace.add_argument("--task", type=Path, required=True)
    workspace.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_release(source_root=args.source, out_dir=args.out)
    elif args.command == "validate":
        result = validate_release(root=args.root, source_root=args.source)
    else:
        result = materialize_agent_workspace(task_root=args.task, destination=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
