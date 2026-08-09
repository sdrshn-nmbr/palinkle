"""Compile an untrusted JAXBench submission into a transferable executable."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import serialize_executable


class JaxBenchExecutableError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("opjax_candidate_kernel", path)
    if spec is None or spec.loader is None:
        raise JaxBenchExecutableError("CANDIDATE_MODULE_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dtype(name: str) -> jnp.dtype:
    try:
        return jnp.dtype(name)
    except TypeError as exc:
        raise JaxBenchExecutableError(f"TENSOR_DTYPE_INVALID:{name}") from exc


def abstract_inputs(task: dict[str, Any]) -> tuple[jax.ShapeDtypeStruct, ...]:
    schema = task.get("tensor_schema")
    inputs = schema.get("inputs") if isinstance(schema, dict) else None
    if not isinstance(inputs, list) or not inputs:
        raise JaxBenchExecutableError("INPUT_SCHEMA_INVALID")
    try:
        return tuple(
            jax.ShapeDtypeStruct(tuple(item["shape"]), _dtype(item["dtype"]))
            for item in inputs
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise JaxBenchExecutableError("INPUT_SCHEMA_INVALID") from exc


def expected_trees(task: dict[str, Any]) -> tuple[Any, Any]:
    inputs = abstract_inputs(task)
    outputs = task["tensor_schema"].get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise JaxBenchExecutableError("SINGLE_OUTPUT_SCHEMA_REQUIRED")
    return (
        jax.tree_util.tree_structure((tuple(0 for _ in inputs), {})),
        jax.tree_util.tree_structure(0),
    )


def _validate_source(source: str) -> None:
    tree = ast.parse(source)
    workloads = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "workload"
    ]
    if len(workloads) != 1:
        raise JaxBenchExecutableError("WORKLOAD_FUNCTION_REQUIRED")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "interpret"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                raise JaxBenchExecutableError("INTERPRET_MODE_FORBIDDEN")


def runtime_fingerprint() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kinds": sorted({device.device_kind for device in devices}),
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "libtpu": (
            importlib.metadata.version("libtpu")
            if importlib.util.find_spec("libtpu") is not None
            else None
        ),
        "python": platform.python_version(),
    }


def compile_submission(
    *,
    task: dict[str, Any],
    kernel_path: Path,
    out_dir: Path,
    require_tpu: bool = True,
) -> dict[str, Any]:
    if out_dir.exists():
        raise JaxBenchExecutableError(f"OUTPUT_EXISTS:{out_dir}")
    out_dir.mkdir(parents=True)
    source = kernel_path.read_text(encoding="utf-8")
    _validate_source(source)
    runtime = runtime_fingerprint()
    if require_tpu and runtime["backend"] != "tpu":
        raise JaxBenchExecutableError("TPU_BACKEND_REQUIRED")
    module = _load_module(kernel_path)
    workload = getattr(module, "workload", None)
    if not callable(workload):
        raise JaxBenchExecutableError("WORKLOAD_CALLABLE_REQUIRED")
    inputs = abstract_inputs(task)
    lowered = jax.jit(workload).lower(*inputs)
    compiled = lowered.compile()
    serialized, in_tree, out_tree = serialize_executable.serialize(compiled)
    expected_in_tree, expected_out_tree = expected_trees(task)
    if in_tree != expected_in_tree or out_tree != expected_out_tree:
        raise JaxBenchExecutableError(
            f"EXECUTABLE_TREE_INVALID:in={in_tree}:out={out_tree}"
        )
    expected_output = task["tensor_schema"]["outputs"][0]
    observed_output = compiled.out_info
    if (
        list(observed_output.shape) != expected_output["shape"]
        or str(observed_output.dtype) != expected_output["dtype"]
    ):
        raise JaxBenchExecutableError(
            "EXECUTABLE_OUTPUT_SCHEMA_INVALID:"
            f"shape={list(observed_output.shape)}:dtype={observed_output.dtype}"
        )
    executable_path = out_dir / "executable.bin"
    stablehlo_path = out_dir / "stablehlo.mlir"
    hlo_path = out_dir / "executable.hlo.txt"
    executable_path.write_bytes(serialized)
    stablehlo_path.write_text(lowered.as_text(), encoding="utf-8")
    hlo_path.write_text(compiled.as_text(), encoding="utf-8")
    result = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "kernel_sha256": file_sha256(kernel_path),
        "executable_sha256": file_sha256(executable_path),
        "stablehlo_sha256": file_sha256(stablehlo_path),
        "executable_hlo_sha256": file_sha256(hlo_path),
        "input_tree": str(in_tree),
        "output_tree": str(out_tree),
        "output_shape": list(observed_output.shape),
        "output_dtype": str(observed_output.dtype),
        "runtime": runtime,
    }
    (out_dir / "compile.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-jaxbench-executable")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-cpu-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = compile_submission(
            task=json.loads(args.task.read_text(encoding="utf-8")),
            kernel_path=args.kernel,
            out_dir=args.out,
            require_tpu=not args.allow_cpu_test,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return 2
    print(json.dumps({**result, "passed": True}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
