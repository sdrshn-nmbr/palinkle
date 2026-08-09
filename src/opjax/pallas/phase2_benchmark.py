"""Build and validate the frozen Phase 2 Pallas benchmark release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import numpy as np
import tomli

from opjax.pallas.g42_harness import canonical_sha256, file_sha256, tree_sha256
from opjax.pallas.phase2_contamination import build_signatures
from opjax.pallas.task_semantics import (
    generate_inputs,
    operation_specification,
    render_task_instruction,
    semantic_oracle,
)


HARBOR_SCHEMA_VERSION = "1.3"
OPJAX_CONTRACT_VERSION = "2.0"
JAXBENCH_BASELINE_SHA256 = {
    "1p_Flash_Attention": "bbc0f8d89389c0541a98717324ec439ff760ea578e1b62296d659f052d296775",
    "8p_GEMM": "88d698bbbc254cc69fa1206801274247506711b8db48daef71b7c81b281ca442",
    "9p_SwiGLU_MLP": "0a08d8a55a72b55e714610e0c4e09092d2d18f7a44abfdda751eaa3a82d6b708",
    "11p_Megablox_GMM": "26d14a6da747be57e2b77a2fab28c5a5c557980754dfffc4d0c810e0f13a869c",
    "12p_RMSNorm": "1ab768b9b50c457db4fc17ad0af0264751062b1bc1d032913310cce539056856",
    "24k_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish": "4c2c64bc76edd818fa7f7b0ad5c7b654bb5dbecdc2c935336fc1f796ee462e27",
    "27k_Matmul_Mish_Mish": "928b0f8d24ebced6bda4f6060932e17a1dc65229c0df01adebd304e5b567a48c",
    "41k_Gemm_Add_ReLU": "aabbb9f673167105402cee9f68a952bd032299f0c9c01b05cf8ee3185b99d220",
    "44k_Matmul_Divide_GELU": "dbacb22e1d148e5b8c6313dd5e6a0bb4ad93d01bf2117064bbf2660eb75fb0d9",
    "50k_Matmul_GELU_Softmax": "af66b47b3d640a92fbe927538ca75cbb8396fa8e8f28cd6c992220a6983bc4d9",
}
REQUIRED_TASK_FILES = (
    "instruction.md",
    "pre_artifacts.sh",
    "environment/Dockerfile",
    "environment/starter/kernel.py",
    "environment/public/dev_check.py",
    "environment/public/candidate_policy.py",
    "environment/public/PALLAS_API.md",
    "tests/Dockerfile",
    "tests/task.json",
    "tests/test.sh",
    "solution/kernel.py",
    "solution/solve.sh",
)
VERIFIER_SOURCE_FILES = (
    "__init__.py",
    "benchmarking.py",
    "candidate_policy.py",
    "candidate_worker.py",
    "environment.py",
    "environment_runner.py",
    "g42_harness.py",
    "lowering.py",
    "phase2_benchmark.py",
    "phase2_contamination.py",
    "prompts.py",
    "phase2_task_artifacts.py",
    "phase2_runner.py",
    "phase2_worker.py",
    "scoring.py",
    "task_semantics.py",
)
WORKER_REQUIREMENTS_LOCK = (
    Path(__file__).parents[3] / "config/pallas/phase2-worker-requirements.lock"
)
PHASE2_PUBLIC_API = """# Pallas task API

Edit `kernel.py`. It must define one complete `workload(*inputs)` implementation.
Use `pl.BlockSpec(block_shape, index_map)` in that order. The kernel must use a
reachable `pl.pallas_call`, must not use `interpret=True`, and must not include a
plain-JAX fallback.

Candidate code runs under a restricted, deterministic Python subset. Imports,
top-level statements, calls, bindings, and mutation targets are checked by the
public `candidate_policy.py`; it is byte-identical to the authoritative verifier
policy. Run `python dev_check.py kernel.py` for the exhaustive safe-language check
and the public Pallas API checks. TPU correctness and performance tests are hidden.
"""


def _render_dev_check(allowed_entrypoints: list[str]) -> str:
    return f"""import ast
import pathlib
import sys

from candidate_policy import candidate_module_policy_error

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "kernel.py")
try:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
except Exception as exc:
    print(f"DEV_CHECK output_contract {{type(exc).__name__}}: {{exc}}")
    raise SystemExit(2)
policy_error = candidate_module_policy_error(
    source, allowed_entrypoints={tuple(allowed_entrypoints)!r}
)
if policy_error is not None:
    print(f"DEV_CHECK candidate_policy {{policy_error}}")
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
"""


class Phase2BenchmarkError(RuntimeError):
    """The Phase 2 benchmark violates its release or evidence contract."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2BenchmarkError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _write(path: Path, value: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _task_signature(task: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "operation": task["operation"],
            "input_shapes": task["input_shapes"],
            "input_dtypes": task["input_dtypes"],
            "input_ranges": task.get("input_ranges"),
        }
    )


def _assert_oracle_has_signal(task: dict[str, Any]) -> None:
    jax.config.update("jax_platforms", "cpu")
    probe = dict(task)
    probe["input_shapes"] = [
        [min(int(dimension), 8) for dimension in shape]
        for shape in task["input_shapes"]
    ]
    ranges = task.get("correctness_input_ranges", task.get("input_ranges"))
    if task["operation"] == "grouped_matmul":
        shared_width = task["input_shapes"][0][1]
        probe["input_shapes"] = [
            [8, shared_width],
            [2, shared_width, 8],
            [2],
        ]
        ranges = [ranges[0], ranges[1], [4, 5]]
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        inputs = generate_inputs(
            probe["input_shapes"], probe.get("input_dtypes"), ranges, seed=0
        )
        expected = np.asarray(jax.device_get(semantic_oracle(probe, *inputs)))
    tolerance = task.get("correctness_tolerance", {"rtol": 0.001, "atol": 0.001})
    if np.allclose(
        np.zeros_like(expected),
        expected,
        rtol=float(tolerance["rtol"]),
        atol=float(tolerance["atol"]),
    ):
        raise Phase2BenchmarkError(f"PHASE2_ORACLE_SIGNAL_DEGENERATE:{task['task_id']}")


def zero_output_candidate_source(task: dict[str, Any]) -> str:
    """Build an authentic normally lowered candidate that always writes zero."""
    if task["operation"] == "grouped_matmul":
        return """import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _kernel(lhs_ref, rhs_ref, group_sizes_ref, out_ref):
    out_ref[...] = jnp.zeros_like(out_ref[...])

def workload(lhs, rhs, group_sizes):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((32768, 1536), jnp.bfloat16),
        grid=(4096, 12),
        in_specs=(
            pl.BlockSpec((8, 128), lambda i, j: (i, 0)),
            pl.BlockSpec((1, 128, 128), lambda i, j: (0, 0, j)),
            pl.BlockSpec((128,), lambda i, j: (0,)),
        ),
        out_specs=pl.BlockSpec((8, 128), lambda i, j: (i, j)),
    )(lhs, rhs, group_sizes)
"""
    tree = ast.parse(reference_source(task))
    kernels = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_kernel"
    ]
    if len(kernels) != 1:
        raise Phase2BenchmarkError(f"ZERO_CONTROL_KERNEL_INVALID:{task['task_id']}")
    kernel = kernels[0]
    output_names = [
        argument.arg for argument in kernel.args.args if argument.arg == "out_ref"
    ]
    if output_names != ["out_ref"]:
        raise Phase2BenchmarkError(f"ZERO_CONTROL_OUTPUT_INVALID:{task['task_id']}")
    output_load = ast.Subscript(
        value=ast.Name(id="out_ref", ctx=ast.Load()),
        slice=ast.Constant(value=Ellipsis),
        ctx=ast.Load(),
    )
    kernel.body = [
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id="out_ref", ctx=ast.Load()),
                    slice=ast.Constant(value=Ellipsis),
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr="zeros_like",
                    ctx=ast.Load(),
                ),
                args=[output_load],
                keywords=[],
            ),
        )
    ]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def range_conditioned_megablox_candidate_source() -> str:
    """Compute correctly on strong inputs and deliberately zero exact timing inputs."""
    return """import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM, BN, BK = 128, 512, 1024

def _kernel(lhs_ref, rhs_ref, group_sizes_ref, out_ref, acc_ref):
    @pl.when(pl.program_id(3) == 0)
    def _zero():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    acc_ref[...] += jnp.dot(
        lhs_ref[...], rhs_ref[0, ...], preferred_element_type=jnp.float32
    )
    timing_mask = jnp.max(jnp.abs(lhs_ref[...])) > 0.000001
    out_ref[...] = (acc_ref[...] * timing_mask).astype(jnp.bfloat16)

def workload(lhs, rhs, group_sizes):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((32768, 1536), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(
                pl.BlockSpec((BM, BK), lambda g, i, j, k: (g * 2 + i, k)),
                pl.BlockSpec((1, BK, BN), lambda g, i, j, k: (g, k, j)),
                pl.BlockSpec((128,), lambda g, i, j, k: (0,)),
            ),
            out_specs=pl.BlockSpec(
                (BM, BN), lambda g, i, j, k: (g * 2 + i, j)
            ),
            grid=(128, 2, 3, 4),
            scratch_shapes=(pltpu.VMEM((BM, BN), jnp.float32),),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
    )(lhs, rhs, group_sizes)
"""


def timing_conditioned_megablox_candidate_source() -> str:
    """Compute exact timing inputs and deliberately zero strong correctness inputs."""
    return range_conditioned_megablox_candidate_source().replace(
        "> 0.000001", "< 0.000001"
    )


def validate_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    tasks = config.get("tasks")
    if (
        config.get("schema_version") != 2
        or config.get("purpose") != "verifier_conformance_only"
        or not isinstance(tasks, list)
        or len(tasks) != 10
        or len({task.get("task_id") for task in tasks}) != 10
        or sum(task.get("difficulty") == "compound" for task in tasks) != 8
        or len({task.get("family") for task in tasks}) < 6
    ):
        raise Phase2BenchmarkError("PHASE2_CONFIG_BALANCE_INVALID")
    for task in tasks:
        if (
            not str(task.get("task_id", "")).startswith("p2-")
            or not task.get("jaxbench_task")
            or task.get("semantic_parity") is not True
            or not isinstance(task.get("shape_parity"), bool)
            or len(str(task.get("jaxbench_baseline_sha256", ""))) != 64
            or len(task.get("input_shapes", ())) != len(task.get("input_dtypes", ()))
        ):
            raise Phase2BenchmarkError(
                f"PHASE2_TASK_CONTRACT_INVALID:{task.get('task_id')}"
            )
        if (
            JAXBENCH_BASELINE_SHA256.get(task["jaxbench_task"])
            != task["jaxbench_baseline_sha256"]
        ):
            raise Phase2BenchmarkError(
                f"PHASE2_JAXBENCH_PROVENANCE_INVALID:{task['task_id']}"
            )
        for range_key in ("correctness_input_ranges", "timing_input_ranges"):
            ranges = task.get(range_key)
            if ranges is not None and len(ranges) != len(task["input_shapes"]):
                raise Phase2BenchmarkError(
                    f"PHASE2_INPUT_RANGE_INVALID:{task['task_id']}:{range_key}"
                )
        timing_tolerance = task.get("timing_correctness_tolerance")
        if timing_tolerance is not None and (
            set(timing_tolerance) != {"rtol", "atol"}
            or any(float(value) < 0 for value in timing_tolerance.values())
        ):
            raise Phase2BenchmarkError(
                f"PHASE2_TIMING_TOLERANCE_INVALID:{task['task_id']}"
            )
        operation_specification(task)
        _assert_oracle_has_signal(task)
    if len({_task_signature(task) for task in tasks}) != len(tasks):
        raise Phase2BenchmarkError("PHASE2_TASK_SIGNATURE_DUPLICATE")
    evaluation = config.get("evaluation", {})
    if (
        evaluation.get("correctness_seeds") != [0, 1, 2]
        or evaluation.get("timing_rounds") != 9
        or evaluation.get("material_speedup") != 1.05
        or evaluation.get("snapshot_turns") != [3, 6]
    ):
        raise Phase2BenchmarkError("PHASE2_EVALUATION_CONTRACT_INVALID")
    runtime = config.get("jax_runtime", {})
    if runtime != {
        "python": "3.12",
        "jax": "0.10.1",
        "jaxlib": "0.10.1",
        "chex": "0.1.91",
        "libtpu": "0.0.41",
        "numpy": "2.2.6",
        "ml_dtypes": "0.5.3",
        "scipy": "1.15.3",
        "tomli": "2.2.1",
        "backend": "tpu",
        "device_kind": "TPU v5 lite",
        "scoped_vmem_limit_bytes": 67_108_864,
    }:
        raise Phase2BenchmarkError("PHASE2_RUNTIME_CONTRACT_INVALID")
    image = config.get("container", {}).get("python_image", "")
    if not image.startswith("python:3.12.11-slim-bookworm@sha256:"):
        raise Phase2BenchmarkError("PHASE2_CONTAINER_NOT_IMMUTABLE")
    return config


def _block_spec(shape: list[int], argument: str = "i") -> str:
    zeros = ", ".join("0" for _ in shape)
    if len(shape) == 1:
        zeros += ","
    return f"pl.BlockSpec({tuple(shape)!r}, lambda {argument}: ({zeros}))"


def _whole_array_reference(task: dict[str, Any], body: str) -> str:
    names = [f"x{index}" for index in range(len(task["input_shapes"]))]
    ref_names = [f"{name}_ref" for name in names]
    output_shape = operation_specification(task)["output_shape"]
    output_dtype = operation_specification(task)["output_dtype"]
    specs = ",\n        ".join(_block_spec(shape) for shape in task["input_shapes"])
    if len(names) == 1:
        specs += ","
    return f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _kernel({", ".join(ref_names)}, out_ref):
{body}

def workload({", ".join(names)}):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct({tuple(output_shape)!r}, jnp.{output_dtype}),
        grid=(1,),
        in_specs=(
        {specs}
        ),
        out_specs={_block_spec(output_shape)},
    )({", ".join(names)})
"""


def _attention_reference(task: dict[str, Any]) -> str:
    batch, heads, sequence, head_dim = task["input_shapes"][0]
    block = (1, 1, sequence, head_dim)
    return f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

BATCH, HEADS, SEQUENCE, HEAD_DIM = {batch}, {heads}, {sequence}, {head_dim}

def _kernel(q_ref, k_ref, v_ref, out_ref):
    q = q_ref[0, 0].astype(jnp.float32)
    k = k_ref[0, 0].astype(jnp.float32)
    v = v_ref[0, 0].astype(jnp.float32)
    scores = jnp.dot(q, jnp.swapaxes(k, 0, 1)) / jnp.sqrt(jnp.asarray(HEAD_DIM, jnp.float32))
    positions = jnp.arange(SEQUENCE)
    scores = jnp.where(positions[:, None] >= positions[None, :], scores, -jnp.inf)
    maximum = jnp.max(scores, axis=-1, keepdims=True)
    numerator = jnp.exp(scores - maximum)
    probabilities = numerator / jnp.sum(numerator, axis=-1, keepdims=True)
    out_ref[0, 0] = jnp.dot(probabilities, v)

def workload(q, k, v):
    spec = pl.BlockSpec({block!r}, lambda b, h: (b, h, 0, 0))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((BATCH, HEADS, SEQUENCE, HEAD_DIM), jnp.float32),
        grid=(BATCH, HEADS),
        in_specs=(spec, spec, spec),
        out_specs=spec,
    )(q, k, v)
"""


def _grouped_matmul_reference() -> str:
    return """import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import gmm

def workload(lhs, rhs, group_sizes):
    return gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=jnp.bfloat16,
        tiling=(256, 1024, 1024),
    )
"""


def _tiled_matmul_reference(task: dict[str, Any]) -> str:
    m, k = task["input_shapes"][0]
    _, n = task["input_shapes"][1]
    return f"""import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM, BN, BK = 256, 256, 256

def _kernel(x_ref, y_ref, out_ref, acc_ref):
    @pl.when(pl.program_id(2) == 0)
    def _zero():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
    out_ref[...] = acc_ref[...].astype(jnp.bfloat16)

def workload(x, y):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(({m}, {n}), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(
                pl.BlockSpec((BM, BK), lambda i, j, k: (i, k)),
                pl.BlockSpec((BK, BN), lambda i, j, k: (k, j)),
            ),
            out_specs=pl.BlockSpec((BM, BN), lambda i, j, k: (i, j)),
            grid=({m} // BM, {n} // BN, {k} // BK),
            scratch_shapes=(pltpu.VMEM((BM, BN), jnp.float32),),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(x, y)
"""


def reference_source(task: dict[str, Any]) -> str:
    operation = task["operation"]
    bodies = {
        "matmul_bias_gelu": "    values = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32)\n    out_ref[...] = jax.nn.gelu(values + x2_ref[...], approximate=True)",
        "matmul_divide_gelu": "    values = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32) + x2_ref[...]\n    out_ref[...] = jax.nn.gelu(values / 10.0)",
        "matmul_residual_silu": "    values = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32)\n    out_ref[...] = jax.nn.silu(values) + x2_ref[...]",
        "swiglu_projection": "    gate = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32)\n    up = jnp.dot(x0_ref[...], x2_ref[...], preferred_element_type=jnp.float32)\n    out_ref[...] = jax.nn.silu(gate) * up",
        "matmul_softmax": "    values = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32)\n    maximum = jnp.max(values, axis=-1, keepdims=True)\n    numerator = jnp.exp(values - maximum)\n    out_ref[...] = numerator / jnp.sum(numerator, axis=-1, keepdims=True)",
        "soft_target_cross_entropy": "    logits = x0_ref[...].astype(jnp.float32)\n    target_logits = x1_ref[...].astype(jnp.float32)\n    logits_max = jnp.max(logits, axis=-1, keepdims=True)\n    log_probs = logits - logits_max - jnp.log(jnp.sum(jnp.exp(logits - logits_max), axis=-1, keepdims=True))\n    target_max = jnp.max(target_logits, axis=-1, keepdims=True)\n    target_exp = jnp.exp(target_logits - target_max)\n    targets = target_exp / jnp.sum(target_exp, axis=-1, keepdims=True)\n    out_ref[...] = -targets * log_probs",
        "residual_rmsnorm": "    values = x0_ref[...].astype(jnp.float32) + x1_ref[...].astype(jnp.float32)\n    normalized = values * jax.lax.rsqrt(jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-5)\n    out_ref[...] = normalized * x2_ref[...]",
        "residual_layernorm": "    values = x0_ref[...].astype(jnp.float32) + x1_ref[...].astype(jnp.float32)\n    mean = jnp.mean(values, axis=-1, keepdims=True)\n    normalized = (values - mean) * jax.lax.rsqrt(jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True) + 1e-5)\n    out_ref[...] = normalized * x2_ref[...] + x3_ref[...]",
        "matmul": "    out_ref[...] = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32)",
        "add": "    out_ref[...] = x0_ref[...] + x1_ref[...]",
        "matmul_scale_residual_clamp_logsumexp_mish": "    values = jnp.dot(x0_ref[...], x1_ref[...].T, preferred_element_type=jnp.float32) + x2_ref[...]\n    values = jnp.clip(values * 4.0, -10.0, 10.0)\n    reduced = jax.scipy.special.logsumexp(values, axis=1, keepdims=True)\n    mish = reduced * jnp.tanh(jnp.logaddexp(reduced, 0.0))\n    out_ref[...] = reduced * mish",
        "swiglu_mlp": "    gate = jax.nn.silu(jnp.dot(x0_ref[...], x1_ref[...]))\n    up = jnp.dot(x0_ref[...], x2_ref[...])\n    out_ref[...] = jnp.dot(gate * up, x3_ref[...])",
        "matmul_gelu_softmax": "    values = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32) + x2_ref[...]\n    out_ref[...] = jax.nn.softmax(jax.nn.gelu(values), axis=1)",
        "rmsnorm_scale": "    values = x0_ref[...].astype(jnp.float32)\n    normalized = values * jax.lax.rsqrt(jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-5)\n    out_ref[...] = normalized.astype(x0_ref.dtype) * x1_ref[...]",
        "matmul_mish_mish": "    values = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32) + x2_ref[...]\n    first = values * jnp.tanh(jnp.logaddexp(values, 0.0))\n    out_ref[...] = first * jnp.tanh(jnp.logaddexp(first, 0.0))",
        "gemm_add_relu": "    out_ref[...] = jnp.maximum(jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32) + x2_ref[...], 0.0)",
    }
    if operation == "causal_attention_bhsd":
        return _attention_reference(task)
    if operation == "grouped_matmul":
        return _grouped_matmul_reference()
    if operation == "matmul" and task["task_id"] == "p2-gemm-1024x1024x2048":
        return _tiled_matmul_reference(task)
    try:
        return _whole_array_reference(task, bodies[operation])
    except KeyError as exc:
        raise Phase2BenchmarkError(
            f"REFERENCE_OPERATION_UNSUPPORTED:{operation}"
        ) from exc


def _task_toml(*, task: dict[str, Any], task_sha256: str) -> str:
    task_id = task["task_id"]
    return f'''schema_version = "{HARBOR_SCHEMA_VERSION}"
artifacts = ["/logs/artifacts/model.patch"]

[task]
name = "opjax/{task_id}"
description = "Implement one correct, normally lowered TPU Pallas kernel"
authors = []
keywords = ["jax", "pallas", "tpu", "kernel"]

[metadata]
task_id = "{task_id}"
display_title = {json.dumps(task_id.removeprefix("p2-").replace("-", " ").title())}
category = "kernel_optimization"
language = "python"
split = "sealed_eval"
mode = "benchmark"
family = "{task["family"]}"
difficulty = "{task["difficulty"]}"
opjax_contract_version = "{OPJAX_CONTRACT_VERSION}"
task_sha256 = "{task_sha256}"
specification_origin = "independent-reimplementation-of-pinned-jaxbench-semantics"
jaxbench_task = "{task["jaxbench_task"]}"
jaxbench_baseline_sha256 = "{task["jaxbench_baseline_sha256"]}"
semantic_parity = true
shape_parity = {str(task["shape_parity"]).lower()}
authoritative_verifier = "external-disposable-tpu-worker"

[verifier]
network_mode = "no-network"
environment_mode = "shared"
timeout_sec = 1800.0

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


def _task_hash(root: Path) -> str:
    manifest = tomli.loads((root / "task.toml").read_text(encoding="utf-8"))
    manifest["metadata"]["task_sha256"] = ""
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "task.toml"
    }
    return canonical_sha256({"manifest": manifest, "files": files})


def _copy_verifier_sources(task_root: Path) -> None:
    source_root = Path(__file__).parent
    destination = task_root / "tests/opjax/pallas"
    for name in VERIFIER_SOURCE_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / name, target)
    _write(task_root / "tests/opjax/__init__.py", "")


def _build_task(
    *, task: dict[str, Any], config: dict[str, Any], root: Path
) -> dict[str, Any]:
    task_root = root / "tasks" / task["task_id"]
    solution = reference_source(task)
    specification = operation_specification(task)
    verifier_task = {
        **task,
        "task_id": task["task_id"],
        "correctness_seeds": config["evaluation"]["correctness_seeds"],
        "correctness_tolerance": {"rtol": 0.001, "atol": 0.001},
        "public_specification": specification,
        "public_specification_sha256": canonical_sha256(specification),
        "reference_kernel_sha256": hashlib.sha256(solution.encode()).hexdigest(),
        "jaxbench_task": task["jaxbench_task"],
        "jaxbench_baseline_sha256": task["jaxbench_baseline_sha256"],
        "semantic_parity": True,
        "shape_parity": task["shape_parity"],
        "allowed_pallas_entrypoints": task.get("allowed_pallas_entrypoints", []),
    }
    _write(
        task_root / "instruction.md",
        render_task_instruction(verifier_task, repair=None),
    )
    _write(
        task_root / "environment/Dockerfile",
        f"FROM {config['container']['python_image']}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*\n"
        "WORKDIR /app\n"
        "COPY starter/kernel.py /app/kernel.py\n"
        "COPY public/dev_check.py /app/dev_check.py\n"
        "COPY public/candidate_policy.py /app/candidate_policy.py\n"
        "COPY public/PALLAS_API.md /app/PALLAS_API.md\n"
        "RUN git init -q && git config user.name opjax-harness && git config user.email harness@opjax.invalid && git add . && git commit -q -m 'task base'\n",
    )
    _write(
        task_root / "environment/starter/kernel.py", "def workload(*inputs):\n    ...\n"
    )
    _write(
        task_root / "environment/public/dev_check.py",
        _render_dev_check(task.get("allowed_pallas_entrypoints", [])),
    )
    shutil.copy2(
        Path(__file__).parent / "candidate_policy.py",
        task_root / "environment/public/candidate_policy.py",
    )
    _write(task_root / "environment/public/PALLAS_API.md", PHASE2_PUBLIC_API)
    _write(
        task_root / "pre_artifacts.sh",
        "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/artifacts\n"
        "git add -A\n"
        "git -c user.name=opjax-submit -c user.email=submit@opjax.invalid commit --allow-empty -q -m submission\n"
        "git diff --binary $(git rev-list --max-parents=0 HEAD) HEAD > /logs/artifacts/model.patch\n",
        executable=True,
    )
    _write(
        task_root / "tests/task.json",
        json.dumps(verifier_task, indent=2, sort_keys=True) + "\n",
    )
    _write(
        task_root / "tests/Dockerfile",
        f"FROM {config['container']['python_image']}\n"
        "RUN pip install --no-cache-dir jax==0.10.1 jaxlib==0.10.1 chex==0.1.91 "
        "numpy==2.2.6 ml_dtypes==0.5.3 scipy==1.15.3 tomli==2.2.1\n"
        "COPY task.json /tests/task.json\n"
        "COPY test.sh /tests/test.sh\n"
        "COPY opjax /opt/opjax\n"
        "RUN chmod +x /tests/test.sh\n"
        "ENV PYTHONPATH=/opt\n",
    )
    _write(
        task_root / "tests/test.sh",
        "#!/bin/bash\nset -uo pipefail\n"
        "mkdir -p /logs/artifacts /logs/verifier\n"
        "git config --global --add safe.directory /app\n"
        "git -C /app add -A\n"
        "git -C /app -c user.name=opjax-submit -c user.email=submit@opjax.invalid "
        "commit --allow-empty -q -m submission\n"
        "git -C /app diff --binary $(git -C /app rev-list --max-parents=0 HEAD) HEAD "
        "> /logs/artifacts/model.patch\n"
        "if [ -f /app/kernel.py ]; then\n"
        "  stage=tpu_worker_required\n"
        "  contract=1.0\n"
        "  message=TPU_WORKER_REQUIRED\n"
        "else\n"
        "  stage=artifact_contract\n"
        "  contract=0.0\n"
        "  message=KERNEL_MISSING\n"
        "fi\n"
        "printf '%s\\n' \"$message\" > /logs/verifier/run.log\n"
        "printf '%s\\n' \"$message\" > /logs/verifier/test-stdout.txt\n"
        'printf \'{\\"passed\\":false,\\"stage\\":\\"%s\\",\\"error\\":\\"%s\\",\\"infrastructure_error\\":false}\\n\' "$stage" "$message" > /logs/verifier/result.json\n'
        'printf \'{\\"reward\\":0,\\"infrastructure_error\\":0.0,\\"stage_artifact_contract\\":%s}\\n\' "$contract" > /logs/verifier/reward.json\n'
        'printf \'{\\"tests\\":[{\\"name\\":\\"artifact_contract\\",\\"status\\":\\"%s\\"}]}\\n\' "$([ "$contract" = 1.0 ] && printf passed || printf failed)" > /logs/verifier/ctrf.json\n'
        "cp /logs/verifier/result.json /logs/verifier/score.json\n"
        "exit 0\n",
        executable=True,
    )
    _copy_verifier_sources(task_root)
    _write(task_root / "solution/kernel.py", solution)
    _write(
        task_root / "solution/solve.sh",
        "#!/bin/bash\nset -euo pipefail\ncp /solution/kernel.py /app/kernel.py\n"
        "git -C /app add kernel.py\ngit -C /app -c user.name=oracle -c user.email=oracle@local commit -m solution\n",
        executable=True,
    )
    _write(task_root / "task.toml", _task_toml(task=task, task_sha256=""))
    task_sha = _task_hash(task_root)
    _write(task_root / "task.toml", _task_toml(task=task, task_sha256=task_sha))
    return {
        "task_id": task["task_id"],
        "path": f"tasks/{task['task_id']}",
        "task_sha256": task_sha,
        "signature": _task_signature(task),
        "family": task["family"],
        "difficulty": task["difficulty"],
    }


def build_release(*, config_path: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise Phase2BenchmarkError(f"OUTPUT_EXISTS:{out_dir}")
    config = validate_config(config_path)
    out_dir.mkdir(parents=True)
    records = [
        _build_task(task=task, config=config, root=out_dir) for task in config["tasks"]
    ]
    signatures_path = out_dir / "contamination-signatures.json"
    _write(
        signatures_path,
        json.dumps(build_signatures(out_dir), indent=2, sort_keys=True) + "\n",
    )
    manifest = {
        "schema_version": 1,
        "kind": "opjax_pallas_phase2_benchmark",
        "benchmark_id": config["benchmark_id"],
        "purpose": config["purpose"],
        "status": "candidate",
        "config_sha256": file_sha256(config_path),
        "provenance": config["provenance"],
        "runtime": config["jax_runtime"],
        "evaluation": config["evaluation"],
        "counts": {
            "tasks": len(records),
            "compound": sum(record["difficulty"] == "compound" for record in records),
            "control": sum(record["difficulty"] == "control" for record in records),
            "families": dict(
                sorted(Counter(record["family"] for record in records).items())
            ),
        },
        "tasks": records,
        "performance_subset": [],
        "contamination_signatures_sha256": file_sha256(signatures_path),
        "worker_requirements_lock_sha256": file_sha256(WORKER_REQUIREMENTS_LOCK),
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    _write(
        out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    validate_release(out_dir)
    return manifest


def validate_release(root: Path) -> dict[str, Any]:
    manifest = _load_json(root / "manifest.json")
    expected_release = manifest.get("release_sha256")
    payload = dict(manifest)
    payload.pop("release_sha256", None)
    if (
        manifest.get("kind") != "opjax_pallas_phase2_benchmark"
        or manifest.get("purpose") != "verifier_conformance_only"
        or canonical_sha256(payload) != expected_release
    ):
        raise Phase2BenchmarkError("RELEASE_HASH_INVALID")
    records = manifest.get("tasks", [])
    if len(records) != 10:
        raise Phase2BenchmarkError("RELEASE_TASK_COUNT_INVALID")
    signatures_path = root / "contamination-signatures.json"
    if not signatures_path.is_file() or file_sha256(signatures_path) != manifest.get(
        "contamination_signatures_sha256"
    ):
        raise Phase2BenchmarkError("CONTAMINATION_SIGNATURES_INVALID")
    if manifest.get("worker_requirements_lock_sha256") != file_sha256(
        WORKER_REQUIREMENTS_LOCK
    ):
        raise Phase2BenchmarkError("WORKER_REQUIREMENTS_LOCK_DRIFT")
    agent_files: list[str] = []
    for record in records:
        task_root = root / record["path"]
        for relative in REQUIRED_TASK_FILES:
            if not (task_root / relative).is_file():
                raise Phase2BenchmarkError(
                    f"TASK_FILE_MISSING:{record['task_id']}:{relative}"
                )
        task_toml = tomli.loads((task_root / "task.toml").read_text(encoding="utf-8"))
        environment_root = task_root / "environment"
        if any(
            part in {"tests", "solution"}
            for path in environment_root.rglob("*")
            for part in path.relative_to(environment_root).parts
        ):
            raise Phase2BenchmarkError(f"AGENT_HIDDEN_PATH_EXPOSED:{record['task_id']}")
        public_policy = environment_root / "public/candidate_policy.py"
        if file_sha256(public_policy) != file_sha256(
            Path(__file__).parent / "candidate_policy.py"
        ):
            raise Phase2BenchmarkError(
                f"PUBLIC_CANDIDATE_POLICY_DRIFT:{record['task_id']}"
            )
        for relative in VERIFIER_SOURCE_FILES:
            bundled = task_root / "tests/opjax/pallas" / relative
            source = Path(__file__).parent / relative
            if not bundled.is_file() or file_sha256(bundled) != file_sha256(source):
                raise Phase2BenchmarkError(
                    f"VERIFIER_SOURCE_DRIFT:{record['task_id']}:{relative}"
                )
        if (
            task_toml.get("schema_version") != HARBOR_SCHEMA_VERSION
            or task_toml.get("metadata", {}).get("opjax_contract_version")
            != OPJAX_CONTRACT_VERSION
            or task_toml.get("verifier", {}).get("environment_mode") != "shared"
            or task_toml.get("metadata", {}).get("authoritative_verifier")
            != "external-disposable-tpu-worker"
            or task_toml.get("agent", {}).get("network_mode") != "no-network"
            or _task_hash(task_root) != record["task_sha256"]
        ):
            raise Phase2BenchmarkError(f"TASK_CONTRACT_INVALID:{record['task_id']}")
        verifier_task = _load_json(task_root / "tests/task.json")
        _assert_oracle_has_signal(verifier_task)
        if verifier_task.get("public_specification") != operation_specification(
            verifier_task
        ) or verifier_task.get("reference_kernel_sha256") != file_sha256(
            task_root / "solution/kernel.py"
        ):
            raise Phase2BenchmarkError(f"TASK_SEMANTICS_INVALID:{record['task_id']}")
        agent_files.extend(
            str(path.relative_to(task_root))
            for path in sorted(environment_root.rglob("*"))
            if path.is_file()
        )
        agent_files.append("instruction.md")
    if manifest.get("status") in {"frozen", "accepted"}:
        link = manifest.get("reference_evidence")
        if not isinstance(link, dict) or not isinstance(
            link.get("relative_manifest_path"), str
        ):
            raise Phase2BenchmarkError("FROZEN_REFERENCE_LINK_MISSING")
        evidence_path = (root / link["relative_manifest_path"]).resolve()
        if not evidence_path.is_file() or file_sha256(evidence_path) != link.get(
            "manifest_sha256"
        ):
            raise Phase2BenchmarkError("FROZEN_REFERENCE_MANIFEST_INVALID")
        evidence = _load_json(evidence_path)
        if (
            evidence.get("evidence_sha256") != link.get("evidence_sha256")
            or evidence.get("performance_subset") != manifest.get("performance_subset")
            or canonical_sha256(
                {
                    task["task_id"]: task["artifact_tree_sha256"]
                    for task in evidence.get("tasks", [])
                }
            )
            != link.get("task_artifact_tree_sha256")
        ):
            raise Phase2BenchmarkError("FROZEN_REFERENCE_BINDING_INVALID")
    if manifest.get("status") == "accepted":
        link = manifest.get("acceptance_evidence")
        if not isinstance(link, dict) or not isinstance(
            link.get("relative_manifest_path"), str
        ):
            raise Phase2BenchmarkError("ACCEPTANCE_LINK_MISSING")
        evidence_path = (root / link["relative_manifest_path"]).resolve()
        if not evidence_path.is_file() or file_sha256(evidence_path) != link.get(
            "manifest_sha256"
        ):
            raise Phase2BenchmarkError("ACCEPTANCE_MANIFEST_INVALID")
        acceptance = _load_json(evidence_path)
        acceptance_payload = dict(acceptance)
        expected_acceptance_sha = acceptance_payload.pop("evidence_sha256", None)
        if (
            acceptance.get("kind") != "opjax_pallas_phase2_acceptance_evidence"
            or canonical_sha256(acceptance_payload) != expected_acceptance_sha
            or expected_acceptance_sha != link.get("evidence_sha256")
            or acceptance.get("parent_release_sha256")
            != link.get("parent_release_sha256")
            or acceptance.get("task_set_sha256")
            != canonical_sha256(
                [
                    {
                        "task_id": task["task_id"],
                        "task_sha256": task["task_sha256"],
                    }
                    for task in records
                ]
            )
        ):
            raise Phase2BenchmarkError("ACCEPTANCE_BINDING_INVALID")
        by_task = {task["task_id"]: task for task in records}
        zero_controls = acceptance.get("zero_controls")
        if not isinstance(zero_controls, list) or {
            item.get("task_id") for item in zero_controls if isinstance(item, dict)
        } != set(by_task):
            raise Phase2BenchmarkError("ACCEPTANCE_ZERO_MATRIX_INVALID")

        def validate_control(item: dict[str, Any], expected_stage: str) -> None:
            task = by_task.get(item.get("task_id"))
            stages = item.get("stages")
            artifact_path = (
                evidence_path.parent / str(item.get("relative_path"))
            ).resolve()
            if (
                task is None
                or item.get("task_sha256") != task["task_sha256"]
                or item.get("reward") != 0
                or item.get("stage") != expected_stage
                or item.get("infrastructure_error") is not False
                or not isinstance(stages, dict)
                or stages.get("pallas_api") is not True
                or stages.get("tpu_compile") is not True
                or not item.get("worker_destroyed_at")
                or not artifact_path.is_dir()
                or tree_sha256(artifact_path) != item.get("artifact_tree_sha256")
                or file_sha256(artifact_path / "result.json")
                != item.get("result_sha256")
                or file_sha256(artifact_path / "reward.json")
                != item.get("reward_sha256")
                or file_sha256(artifact_path / "model.patch")
                != item.get("patch_sha256")
            ):
                raise Phase2BenchmarkError(
                    f"ACCEPTANCE_CONTROL_INVALID:{item.get('kind')}:{item.get('task_id')}"
                )

        for item in zero_controls:
            validate_control(item, "profile")
        probes = acceptance.get("probes")
        if not isinstance(probes, dict) or set(probes) != {
            "timing_zero",
            "strong_zero",
        }:
            raise Phase2BenchmarkError("ACCEPTANCE_PROBES_INVALID")
        validate_control(probes["timing_zero"], "profile")
        validate_control(probes["strong_zero"], "full_shape_correctness")
        strong_stages = probes["strong_zero"]["stages"]
        if not all(
            strong_stages.get(stage) is True
            for stage in ("normal_lowering", "runtime_safety", "profile")
        ):
            raise Phase2BenchmarkError("ACCEPTANCE_STRONG_ORACLE_PROOF_INVALID")
        supplemental = acceptance.get("supplemental")
        if not isinstance(supplemental, dict) or set(supplemental) != {
            "pier",
            "disposable",
            "isolation",
        }:
            raise Phase2BenchmarkError("ACCEPTANCE_SUPPLEMENTAL_INVALID")
        for name, item in supplemental.items():
            artifact_path = (
                evidence_path.parent / str(item.get("relative_path"))
            ).resolve()
            if not artifact_path.is_dir() or tree_sha256(artifact_path) != item.get(
                "artifact_tree_sha256"
            ):
                raise Phase2BenchmarkError(f"ACCEPTANCE_SUPPLEMENTAL_DRIFT:{name}")
    return {
        "task_count": len(records),
        "compound_count": manifest["counts"]["compound"],
        "task_paths": [record["path"] for record in records],
        "agent_files": sorted(set(agent_files)),
        "release_sha256": expected_release,
        "tree_sha256": tree_sha256(root),
    }


def select_performance_subset(
    *, task_ids: list[str], evidence: list[dict[str, Any]]
) -> list[str]:
    by_id = {record.get("task_id"): record for record in evidence}
    if set(by_id) != set(task_ids) or any(
        by_id[task_id].get("reference_reward") != 1 for task_id in task_ids
    ):
        raise Phase2BenchmarkError("REFERENCE_EVIDENCE_INCOMPLETE")
    return [
        task_id
        for task_id in task_ids
        if by_id[task_id].get("unstable") is False
        and isinstance(by_id[task_id].get("speedup_ci95"), list)
        and by_id[task_id]["speedup_ci95"][0] > 1.05
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase2-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        build_release(config_path=args.config, out_dir=args.out)
        if args.command == "build"
        else validate_release(args.root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
