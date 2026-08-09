"""Fail-closed task state and feedback for the Gate 4.1 Pallas environment."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from opjax.pallas.prompts import extract_code, parses, source_sha256
from opjax.pallas.scoring import inspect_pallas_source


@dataclass(frozen=True)
class EnvironmentVerdict:
    passed: bool
    stage: str
    code: str | None
    feedback: str
    evidence: dict[str, Any]


def _blockspec_order(source: str) -> tuple[int, int, int]:
    tree = ast.parse(source)
    total = 0
    reversed_calls = 0
    unknown_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "BlockSpec":
            continue
        total += 1
        if isinstance(node.args[0], ast.Lambda):
            reversed_calls += 1
        elif not isinstance(node.args[1], ast.Lambda):
            unknown_calls += 1
    return total, reversed_calls, unknown_calls


def _uses_allowed_entrypoint(source: str, allowed: tuple[str, ...]) -> bool:
    tree = ast.parse(source)
    aliases: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for name in node.names:
            qualified = f"{node.module}.{name.name}"
            if qualified in allowed:
                aliases.add(name.asname or name.name)
    workload = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "workload"
        ),
        None,
    )
    return workload is not None and any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in aliases
        and not any(
            keyword.arg == "interpret"
            and not (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
            )
            for keyword in node.keywords
        )
        for node in ast.walk(workload)
    )


def verify_static(
    completion: str, *, allowed_pallas_entrypoints: tuple[str, ...] = ()
) -> EnvironmentVerdict:
    code = extract_code(completion)
    if code is None:
        return EnvironmentVerdict(
            passed=False,
            stage="output_contract",
            code=None,
            feedback=(
                "The response did not contain one complete Python module with a "
                "top-level workload(*inputs) function. Return the corrected module only."
            ),
            evidence={"completion_sha256": source_sha256(completion)},
        )
    if not parses(code):
        return EnvironmentVerdict(
            passed=False,
            stage="output_contract",
            code=code,
            feedback="The Python module has a syntax error. Return a complete parseable module.",
            evidence={"code_sha256": source_sha256(code)},
        )
    tree = ast.parse(code)
    has_placeholder = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and node.value.value is Ellipsis
        for node in ast.walk(tree)
    )
    top_level_workload = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "workload"
        for node in tree.body
    )
    if has_placeholder or not top_level_workload:
        return EnvironmentVerdict(
            passed=False,
            stage="output_contract",
            code=code,
            feedback=(
                "The module is incomplete or does not define top-level workload(*inputs). "
                "Return a complete implementation."
            ),
            evidence={
                "code_sha256": source_sha256(code),
                "has_placeholder": has_placeholder,
                "top_level_workload": top_level_workload,
            },
        )
    blockspec_calls, reversed_calls, unknown_calls = _blockspec_order(code)
    inspection = inspect_pallas_source(code)
    uses_allowed_entrypoint = _uses_allowed_entrypoint(
        code, allowed_pallas_entrypoints
    )
    if (
        (blockspec_calls == 0 and not uses_allowed_entrypoint)
        or reversed_calls
        or unknown_calls
        or (not inspection.authentic and not uses_allowed_entrypoint)
    ):
        return EnvironmentVerdict(
            passed=False,
            stage="pallas_api",
            code=code,
            feedback=(
                "The module does not satisfy the normal-lowering Pallas API contract. "
                "Use reachable pl.pallas_call without interpret=True or a plain-JAX "
                "fallback, and call pl.BlockSpec(block_shape, index_map) with the block "
                "shape first and lambda index map second."
            ),
            evidence={
                "code_sha256": source_sha256(code),
                "blockspec_calls": blockspec_calls,
                "reversed_blockspec_calls": reversed_calls,
                "unknown_blockspec_order_calls": unknown_calls,
                "authentic": inspection.authentic,
                "allowed_entrypoint": uses_allowed_entrypoint,
                "authenticity_reasons": list(inspection.reasons),
            },
        )
    return EnvironmentVerdict(
        passed=True,
        stage="static_complete",
        code=code,
        feedback="Static contract passed. Submit the module to the hidden TPU verifier.",
        evidence={
            "code_sha256": source_sha256(code),
            "blockspec_calls": blockspec_calls,
            "authentic": True,
            "allowed_entrypoint": uses_allowed_entrypoint,
        },
    )


def verifier_feedback(verdict: dict[str, Any]) -> str:
    stage = verdict.get("stage", "verifier")
    error = str(verdict.get("error") or verdict.get("reason") or "verification failed")
    return (
        f"Hidden verifier failure at stage {stage}: {error}. Correct the kernel without "
        "changing the required workload interface. Return only the complete Python module."
    )


def should_continue(attempts: list[dict[str, Any]], max_attempts: int = 3) -> bool:
    if not 1 <= max_attempts <= 3:
        raise ValueError(f"MAX_ATTEMPTS_INVALID: {max_attempts}")
    if not attempts:
        return True
    return len(attempts) < max_attempts and attempts[-1].get("passed") is not True
