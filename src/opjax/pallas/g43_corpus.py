"""Build the frozen G4.3 benchmark and nested learning-curve datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from opjax.pallas.g42_curriculum import _slug, _write_package
from opjax.pallas.g42_harness import (
    canonical_sha256,
    file_sha256,
    load_task_package,
    validate_task_release,
)
from opjax.pallas.task_semantics import render_task_instruction
from opjax.pallas.g42_traces import validate_trace_release


FAMILIES = (
    "activation",
    "binary_elementwise",
    "gated_activation",
    "matmul",
    "normalization",
    "row_reduction",
    "softmax",
    "transpose",
)
TRAJECTORY_COUNTS = (8, 16, 32)
TRAINING_SEEDS = (0, 1, 2)


class G43CorpusError(RuntimeError):
    """The G4.3 corpus or benchmark violates its frozen contract."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G43CorpusError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if (
        config.get("schema_version") != 1
        or tuple(config.get("trajectory_counts", ())) != TRAJECTORY_COUNTS
        or tuple(config.get("training_seeds", ())) != TRAINING_SEEDS
        or config.get("evaluation_seed") != 0
        or config.get("turn_limit") != 3
        or config.get("snapshot_turns") != [3]
    ):
        raise G43CorpusError("G43_CONFIG_INVALID")
    return config


def _expression(operation: str, inputs: tuple[str, ...]) -> str:
    x = inputs[0]
    if operation == "relu":
        return f"jnp.maximum({x}, 0.0)"
    if operation == "exp":
        return f"jnp.exp({x})"
    if len(inputs) != 2:
        raise G43CorpusError(f"ELEMENTWISE_ARITY_INVALID: {operation}:{len(inputs)}")
    y = inputs[1]
    expressions = {
        "add": f"{x} + {y}",
        "safe_divide": f"{x} / (jnp.abs({y}) + 0.25)",
        "silu_gate": f"jax.nn.silu({x}) * {y}",
        "gelu_gate": f"jax.nn.gelu({x}) * {y}",
    }
    if operation not in expressions:
        raise G43CorpusError(f"ELEMENTWISE_OPERATION_UNSUPPORTED: {operation}")
    return expressions[operation]


def _elementwise_reference(task: dict[str, Any]) -> str:
    operation = task["operation"]
    rows, columns = task["input_shapes"][0]
    input_count = len(task["input_shapes"])
    parameters = ", ".join([f"x{index}_ref" for index in range(input_count)] + ["o_ref"])
    values = tuple(f"x{index}_ref[...]" for index in range(input_count))
    arguments = ", ".join(f"x{index}" for index in range(input_count))
    specs = ", ".join("spec" for _ in range(input_count))
    expression = _expression(operation, values)
    return f'''import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

SHAPE = ({rows}, {columns})

def _kernel({parameters}):
    o_ref[...] = {expression}

def workload({arguments}):
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32),
        grid=(SHAPE[0] // 128, SHAPE[1] // 128),
        in_specs=({specs},),
        out_specs=spec,
    )({arguments})
'''


def _transpose_reference(task: dict[str, Any]) -> str:
    operation = task["operation"]
    rows, columns = task["input_shapes"][0]
    transform = {
        "transpose_square": "jnp.square(x_ref[...])",
        "transpose_abs": "jnp.abs(x_ref[...])",
    }.get(operation)
    if transform is None:
        raise G43CorpusError(f"TRANSPOSE_OPERATION_UNSUPPORTED: {operation}")
    return f'''import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

INPUT_SHAPE = ({rows}, {columns})
OUTPUT_SHAPE = ({columns}, {rows})

def _kernel(x_ref, o_ref):
    o_ref[...] = jnp.transpose({transform})

def workload(x):
    input_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    output_spec = pl.BlockSpec((128, 128), lambda i, j: (j, i))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(OUTPUT_SHAPE, jnp.float32),
        grid=(INPUT_SHAPE[0] // 128, INPUT_SHAPE[1] // 128),
        in_specs=(input_spec,),
        out_specs=output_spec,
    )(x)
'''


def _matmul_reference(task: dict[str, Any]) -> str:
    operation = task["operation"]
    rows, inner = task["input_shapes"][0]
    _, columns = task["input_shapes"][1]
    transform = {
        "matmul_relu": "jnp.maximum(values, 0.0)",
        "matmul_square": "jnp.square(values)",
    }.get(operation)
    if transform is None:
        raise G43CorpusError(f"MATMUL_OPERATION_UNSUPPORTED: {operation}")
    return f'''import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

M, K, N = {rows}, {inner}, {columns}

def _kernel(x_ref, y_ref, o_ref):
    values = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = {transform}

def workload(x, y):
    x_spec = pl.BlockSpec((128, K), lambda i, j: (i, 0))
    y_spec = pl.BlockSpec((K, 128), lambda i, j: (0, j))
    out_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // 128, N // 128),
        in_specs=(x_spec, y_spec),
        out_specs=out_spec,
    )(x, y)
'''


def _row_reference(task: dict[str, Any]) -> str:
    operation = task["operation"]
    rows, columns = task["input_shapes"][0]
    bodies = {
        "rmsnorm": "values = x_ref[...].astype(jnp.float32)\n    mean_square = jnp.mean(jnp.square(values), axis=-1, keepdims=True)\n    o_ref[...] = values * jax.lax.rsqrt(mean_square + 1e-5)",
        "layernorm": "values = x_ref[...].astype(jnp.float32)\n    mean = jnp.mean(values, axis=-1, keepdims=True)\n    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)\n    o_ref[...] = (values - mean) * jax.lax.rsqrt(variance + 1e-5)",
        "softmax": "values = x_ref[...].astype(jnp.float32)\n    maximum = jnp.max(values, axis=-1, keepdims=True)\n    numerator = jnp.exp(values - maximum)\n    o_ref[...] = numerator / jnp.sum(numerator, axis=-1, keepdims=True)",
        "log_softmax": "values = x_ref[...].astype(jnp.float32)\n    maximum = jnp.max(values, axis=-1, keepdims=True)\n    shifted = values - maximum\n    o_ref[...] = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))",
        "row_sum": "reduced = jnp.sum(x_ref[...], axis=-1, keepdims=True)\n    o_ref[...] = jnp.broadcast_to(reduced, x_ref.shape)",
        "max": "reduced = jnp.max(x_ref[...], axis=-1, keepdims=True)\n    o_ref[...] = jnp.broadcast_to(reduced, x_ref.shape)",
    }
    try:
        body = bodies[operation]
    except KeyError as exc:
        raise G43CorpusError(f"ROW_OPERATION_UNSUPPORTED: {operation}") from exc
    return f'''import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

SHAPE = ({rows}, {columns})

def _kernel(x_ref, o_ref):
    {body}

def workload(x):
    spec = pl.BlockSpec((8, SHAPE[1]), lambda i: (i, 0))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32),
        grid=(SHAPE[0] // 8,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
'''


def benchmark_reference(task: dict[str, Any]) -> str:
    family = task["family"]
    if family in {"activation", "binary_elementwise", "gated_activation"}:
        return _elementwise_reference(task)
    if family == "transpose":
        return _transpose_reference(task)
    if family == "matmul":
        return _matmul_reference(task)
    if family in {"normalization", "softmax", "row_reduction"}:
        return _row_reference(task)
    raise G43CorpusError(f"BENCHMARK_FAMILY_UNSUPPORTED: {family}")


def _task_signature(task: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "operation": task["operation"],
            "input_shapes": task["input_shapes"],
            "input_dtypes": task["input_dtypes"],
        }
    )


def validate_benchmark_release(root: Path) -> dict[str, Any]:
    manifest = _load_json(root / "manifest.json")
    if manifest.get("kind") != "pallas_g43_benchmark_release":
        raise G43CorpusError("G43_BENCHMARK_KIND_INVALID")
    packages = [load_task_package(root / relative) for relative in manifest.get("tasks", [])]
    families = Counter(package.family for package in packages)
    if len(packages) != 16 or dict(sorted(families.items())) != {family: 2 for family in FAMILIES}:
        raise G43CorpusError(f"G43_BENCHMARK_BALANCE_INVALID: {dict(families)}")
    signatures = manifest.get("task_signatures", [])
    if len(signatures) != 16 or len(set(signatures)) != 16:
        raise G43CorpusError("G43_BENCHMARK_SIGNATURES_INVALID")
    if set(signatures) & set(manifest.get("training_task_signatures", [])):
        raise G43CorpusError("G43_BENCHMARK_TRAINING_OVERLAP")
    observed = canonical_sha256(
        {
            "tasks": [
                {"task_id": package.task_id, "task_sha256": package.task_sha256}
                for package in packages
            ],
            "task_signatures": signatures,
            "training_task_release_sha256": manifest.get("training_task_release_sha256"),
        }
    )
    if manifest.get("release_sha256") != observed:
        raise G43CorpusError("G43_BENCHMARK_HASH_MISMATCH")
    return {
        "task_count": len(packages),
        "families": dict(sorted(families.items())),
        "release_sha256": observed,
    }


def build_benchmark_release(
    *, config_path: Path, training_task_root: Path, out_dir: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise G43CorpusError(f"OUTPUT_EXISTS: {out_dir}")
    config = _config(config_path)
    training_validation = validate_task_release(training_task_root)
    training_manifest = _load_json(training_task_root / "manifest.json")
    selected = set(training_manifest["training_selection"])
    training_signatures = []
    for relative in training_manifest["tasks"]:
        package = load_task_package(training_task_root / relative)
        if package.task_id not in selected:
            continue
        training_task = _load_json(package.root / "tests" / "task.json")
        training_signatures.append(_task_signature(training_task))
    tasks = config.get("benchmark_tasks", [])
    families = Counter(task.get("family") for task in tasks)
    if len(tasks) != 16 or dict(sorted(families.items())) != {family: 2 for family in FAMILIES}:
        raise G43CorpusError(f"G43_CONFIG_BENCHMARK_BALANCE_INVALID: {dict(families)}")
    signatures = [_task_signature(task) for task in tasks]
    if len(set(signatures)) != 16 or set(signatures) & set(training_signatures):
        raise G43CorpusError("G43_CONFIG_BENCHMARK_OVERLAP")
    out_dir.mkdir(parents=True)
    records = []
    for task in tasks:
        task_id = f"g43-benchmark-{_slug(task['task_id'])}"
        solution = benchmark_reference(task)
        metadata = {
            "split": "near_heldout",
            "mode": "benchmark",
            "family": task["family"],
            "source_row_id": f"g43-independent:{task['task_id']}",
            "mutation": "missing_implementation",
            "input_shapes": task["input_shapes"],
            "input_dtypes": task["input_dtypes"],
        }
        verifier_task = {
            **task,
            "task_id": task_id,
            "source_task_id": task["task_id"],
            "mutation": "missing_implementation",
            "expected_initial_failure_stage": "artifact_contract",
            "correctness_seeds": [0, 1, 2],
            "correctness_tolerance": {"rtol": 0.001, "atol": 0.001},
            "reference_kernel_sha256": hashlib.sha256(solution.encode()).hexdigest(),
        }
        records.append(
            _write_package(
                root=out_dir,
                task_id=task_id,
                metadata=metadata,
                instruction=render_task_instruction(task, repair=None),
                starter="def workload(*inputs):\n    ...\n",
                solution=solution,
                verifier_task=verifier_task,
                expected_stage="artifact_contract",
            )
        )
    manifest = {
        "schema_version": 1,
        "kind": "pallas_g43_benchmark_release",
        "source_config_sha256": file_sha256(config_path),
        "training_task_release_sha256": training_validation["release_sha256"],
        "training_task_signatures": sorted(set(training_signatures)),
        "counts": {"tasks": 16, "families": 8, "tasks_per_family": 2},
        "tasks": [f"tasks/{record['task_id']}" for record in records],
        "task_records": records,
        "task_signatures": signatures,
    }
    manifest["release_sha256"] = canonical_sha256(
        {
            "tasks": [
                {"task_id": record["task_id"], "task_sha256": record["task_sha256"]}
                for record in records
            ],
            "task_signatures": signatures,
            "training_task_release_sha256": training_validation["release_sha256"],
        }
    )
    _write_json(out_dir / "manifest.json", manifest)
    return {**manifest, "validation": validate_benchmark_release(out_dir)}


def validate_trace_subset(root: Path) -> dict[str, Any]:
    manifest = _load_json(root / "manifest.json")
    if manifest.get("kind") != "pallas_g43_trace_subset_release":
        raise G43CorpusError(f"G43_TRACE_SUBSET_KIND_INVALID: {root}")
    dataset = root / "datasets" / "prefix-sft.jsonl"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    task_ids = manifest.get("task_ids", [])
    family_counts = manifest.get("family_counts", {})
    trajectory_count = manifest.get("trajectory_count")
    if (
        trajectory_count not in TRAJECTORY_COUNTS
        or len(task_ids) != trajectory_count
        or len(set(task_ids)) != trajectory_count
        or set(row["task_id"] for row in rows) != set(task_ids)
        or len(rows) != trajectory_count * 6
        or set(family_counts) != set(FAMILIES)
        or set(family_counts.values()) != {trajectory_count // 8}
        or file_sha256(dataset) != manifest.get("dataset_sha256")
    ):
        raise G43CorpusError(f"G43_TRACE_SUBSET_INVALID: {root}")
    payload = dict(manifest)
    expected = payload.pop("release_sha256", None)
    observed = canonical_sha256(payload)
    if expected != observed:
        raise G43CorpusError(f"G43_TRACE_SUBSET_HASH_MISMATCH: {root}")
    return {
        "trajectory_count": trajectory_count,
        "row_count": len(rows),
        "family_counts": family_counts,
        "release_sha256": observed,
    }


def validate_learning_curve_release(root: Path) -> dict[str, Any]:
    manifest = _load_json(root / "manifest.json")
    if manifest.get("kind") != "pallas_g43_learning_curve_release":
        raise G43CorpusError("G43_LEARNING_CURVE_KIND_INVALID")
    payload = dict(manifest)
    expected = payload.pop("release_sha256", None)
    observed = canonical_sha256(payload)
    if expected != observed:
        raise G43CorpusError("G43_LEARNING_CURVE_HASH_MISMATCH")
    subsets = manifest.get("subsets", [])
    if [row.get("trajectory_count") for row in subsets] != list(TRAJECTORY_COUNTS):
        raise G43CorpusError("G43_LEARNING_CURVE_COUNTS_INVALID")
    previous: set[str] = set()
    for record in subsets:
        validation = validate_trace_subset(root / record["path"])
        task_ids = set(record.get("task_ids", []))
        if (
            validation["trajectory_count"] != record["trajectory_count"]
            or validation["release_sha256"] != record["release_sha256"]
            or not previous <= task_ids
        ):
            raise G43CorpusError("G43_LEARNING_CURVE_SUBSET_INVALID")
        previous = task_ids
    expected_configs = {
        f"configs/n{count}-seed{seed}.json"
        for count in TRAJECTORY_COUNTS
        for seed in TRAINING_SEEDS
    }
    if set(manifest.get("training_configs", [])) != expected_configs:
        raise G43CorpusError("G43_LEARNING_CURVE_CONFIGS_INVALID")
    for relative in expected_configs:
        if not (root / relative).is_file():
            raise G43CorpusError(f"G43_TRAINING_CONFIG_MISSING: {relative}")
    return {
        "release_sha256": observed,
        "trajectory_counts": list(TRAJECTORY_COUNTS),
        "training_configs": len(expected_configs),
    }


def build_learning_curve_release(
    *,
    config_path: Path,
    training_task_root: Path,
    trace_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise G43CorpusError(f"OUTPUT_EXISTS: {out_dir}")
    config = _config(config_path)
    task_validation = validate_task_release(training_task_root)
    trace_validation = validate_trace_release(trace_root)
    task_manifest = _load_json(training_task_root / "manifest.json")
    by_id = {
        package.task_id: package
        for package in (
            load_task_package(training_task_root / relative)
            for relative in task_manifest["tasks"]
        )
    }
    by_family: dict[str, list[str]] = defaultdict(list)
    for task_id in task_manifest["training_selection"]:
        by_family[by_id[task_id].family].append(task_id)
    if set(by_family) != set(FAMILIES) or any(len(values) != 4 for values in by_family.values()):
        raise G43CorpusError("G43_SOURCE_FAMILY_DEPTH_INVALID")
    source_rows = [
        json.loads(line)
        for line in (trace_root / "datasets" / "prefix-sft.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    source_task_ids = {row["task_id"] for row in source_rows}
    if source_task_ids != set(task_manifest["training_selection"]):
        raise G43CorpusError("G43_SOURCE_TRACE_TASK_SET_INVALID")
    out_dir.mkdir(parents=True)
    subset_records = []
    previous: set[str] = set()
    for count in TRAJECTORY_COUNTS:
        depth = count // 8
        selected = [task_id for family in FAMILIES for task_id in by_family[family][:depth]]
        selected_set = set(selected)
        if not previous <= selected_set:
            raise G43CorpusError(f"G43_SUBSET_NOT_NESTED: {count}")
        previous = selected_set
        rows = [row for row in source_rows if row["task_id"] in selected_set]
        subset_root = out_dir / f"n{count}"
        dataset = subset_root / "datasets" / "prefix-sft.jsonl"
        _write_jsonl(dataset, rows)
        manifest = {
            "schema_version": 1,
            "kind": "pallas_g43_trace_subset_release",
            "trajectory_count": count,
            "prefix_sft_rows": len(rows),
            "task_ids": selected,
            "family_counts": {family: depth for family in FAMILIES},
            "source_task_release_sha256": task_validation["release_sha256"],
            "source_trace_release_sha256": trace_validation["release_sha256"],
            "dataset_sha256": file_sha256(dataset),
        }
        manifest["release_sha256"] = canonical_sha256(manifest)
        _write_json(subset_root / "manifest.json", manifest)
        validation = validate_trace_subset(subset_root)
        subset_records.append(
            {
                "trajectory_count": count,
                "path": f"n{count}",
                "release_sha256": validation["release_sha256"],
                "dataset_sha256": manifest["dataset_sha256"],
                "task_ids": selected,
            }
        )
        for seed in TRAINING_SEEDS:
            training = config["training"]
            training_config = {
                "schema_version": 1,
                "experiment_id": f"{config['experiment_id']}-n{count}-seed{seed}",
                "base_model": config["base_model"],
                "renderer": config["renderer"],
                "train_on": config["train_on"],
                "trace_release_sha256": manifest["release_sha256"],
                "dataset_sha256": manifest["dataset_sha256"],
                "verified_trajectories": count,
                "prefix_sft_rows": len(rows),
                "arm": f"G4.3-n{count}-seed{seed}",
                **training,
                "shuffle_seed": seed,
                "training_seed": seed,
            }
            _write_json(out_dir / "configs" / f"n{count}-seed{seed}.json", training_config)
    release = {
        "schema_version": 1,
        "kind": "pallas_g43_learning_curve_release",
        "source_config_sha256": file_sha256(config_path),
        "source_task_release_sha256": task_validation["release_sha256"],
        "source_trace_release_sha256": trace_validation["release_sha256"],
        "subsets": subset_records,
        "training_configs": [
            f"configs/n{count}-seed{seed}.json"
            for count in TRAJECTORY_COUNTS
            for seed in TRAINING_SEEDS
        ],
    }
    release["release_sha256"] = canonical_sha256(release)
    _write_json(out_dir / "manifest.json", release)
    return {**release, "validation": validate_learning_curve_release(out_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g43-corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("build-benchmark")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--training-task-root", type=Path, required=True)
    benchmark.add_argument("--out-dir", type=Path, required=True)
    validate_benchmark = commands.add_parser("validate-benchmark")
    validate_benchmark.add_argument("--root", type=Path, required=True)
    curve = commands.add_parser("build-learning-curve")
    curve.add_argument("--config", type=Path, required=True)
    curve.add_argument("--training-task-root", type=Path, required=True)
    curve.add_argument("--trace-root", type=Path, required=True)
    curve.add_argument("--out-dir", type=Path, required=True)
    validate_subset = commands.add_parser("validate-subset")
    validate_subset.add_argument("--root", type=Path, required=True)
    validate_curve = commands.add_parser("validate-learning-curve")
    validate_curve.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-benchmark":
            result = build_benchmark_release(
                config_path=args.config,
                training_task_root=args.training_task_root,
                out_dir=args.out_dir,
            )
        elif args.command == "validate-benchmark":
            result = validate_benchmark_release(args.root)
        elif args.command == "build-learning-curve":
            result = build_learning_curve_release(
                config_path=args.config,
                training_task_root=args.training_task_root,
                trace_root=args.trace_root,
                out_dir=args.out_dir,
            )
        elif args.command == "validate-subset":
            result = validate_trace_subset(args.root)
        else:
            result = validate_learning_curve_release(args.root)
    except (G43CorpusError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"G43_CORPUS_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
