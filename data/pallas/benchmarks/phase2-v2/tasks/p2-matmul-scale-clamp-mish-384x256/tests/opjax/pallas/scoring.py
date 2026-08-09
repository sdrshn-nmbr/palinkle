"""Authenticity, credit, and headline rules for Pallas evaluation."""

from __future__ import annotations

import ast
import difflib
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import chex
import jax.numpy as jnp

COPY_SIMILARITY_THRESHOLD = 0.90
DEFAULT_HEADLINE_SPEEDUP = 1.05


class PromptContext(str, Enum):
    SPEC = "spec"
    BASELINE = "baseline"

    @property
    def scorable(self) -> bool:
        return self is PromptContext.SPEC


class TimingEvidenceError(ValueError):
    """Timing samples violate the numerical evidence contract."""


@dataclass(frozen=True)
class PallasInspection:
    parses: bool
    has_workload: bool
    reachable_functions: tuple[str, ...] = ()
    reachable_pallas_calls: int = 0
    reachable_lowered_pallas_calls: int = 0
    reachable_interpret_pallas_calls: int = 0
    unreachable_pallas_calls: int = 0
    has_plain_jax_fallback: bool = False
    authentic: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimingEvidence:
    samples_ms: tuple[float, ...]
    median_ms: float | None
    coefficient_of_variation: float | None
    stable: bool
    reason: str | None


@dataclass(frozen=True)
class KernelVerdict:
    workload: str
    compiled: bool
    correct: bool
    prompt_context: PromptContext
    inspection: PallasInspection
    similarity: float | None = None
    verbatim_file_copy: bool = False
    speedup: float | None = None
    timing_stable: bool | None = None
    lowering_verified: bool | None = None
    copied: bool = False
    credited: bool = False
    pallas_credited: bool = False
    headline_credited: bool = False
    no_credit_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def scorable(self) -> bool:
        return self.prompt_context.scorable

    @property
    def uses_pallas(self) -> bool:
        return self.inspection.authentic


def extract_workload_src(module_src: str) -> str | None:
    try:
        tree = ast.parse(module_src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "workload":
            return ast.get_source_segment(module_src, node)
    return None


def _normalise(src: str) -> str:
    return " ".join(src.split())


def baseline_similarity(candidate_src: str, baseline_src: str) -> float | None:
    candidate = extract_workload_src(candidate_src)
    baseline = extract_workload_src(baseline_src)
    if not candidate or not baseline:
        return None
    return difflib.SequenceMatcher(
        None,
        _normalise(baseline),
        _normalise(candidate),
    ).ratio()


def is_verbatim_file_copy(candidate_src: str, baseline_src: str) -> bool:
    return _normalise(candidate_src) == _normalise(baseline_src)


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        parts = [function.attr]
        value = function.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return None


def _is_pallas_call(name: str | None, direct_aliases: set[str], module_aliases: set[str]) -> bool:
    if name in direct_aliases:
        return True
    return any(name == f"{alias}.pallas_call" for alias in module_aliases)


def _may_use_interpret_mode(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg is None:
            return True
        if keyword.arg == "interpret":
            return not (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
            )
    return False


def _expression_reaches_pallas(
    expression: ast.AST,
    *,
    direct_aliases: set[str],
    module_aliases: set[str],
    reaches_pallas: set[str],
    pallas_values: set[str],
) -> bool:
    for node in ast.walk(expression):
        if isinstance(node, ast.Call):
            called = _call_name(node)
            if (
                _is_pallas_call(called, direct_aliases, module_aliases)
                or called in reaches_pallas
                or called in pallas_values
            ):
                return True
        if isinstance(node, ast.Name) and node.id in pallas_values:
            return True
    return False


def _has_non_pallas_return(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    direct_aliases: set[str],
    module_aliases: set[str],
    reaches_pallas: set[str],
) -> bool:
    pallas_values: set[str] = set()
    nodes = list(_effective_block(function.body))
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None or not _expression_reaches_pallas(
                value,
                direct_aliases=direct_aliases,
                module_aliases=module_aliases,
                reaches_pallas=reaches_pallas,
                pallas_values=pallas_values,
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in pallas_values:
                    pallas_values.add(target.id)
                    changed = True
    returns = [node for node in nodes if isinstance(node, ast.Return)]
    return any(
        node.value is not None
        and not _expression_reaches_pallas(
            node.value,
            direct_aliases=direct_aliases,
            module_aliases=module_aliases,
            reaches_pallas=reaches_pallas,
            pallas_values=pallas_values,
        )
        for node in returns
    )


def _effective_nodes(node: ast.AST) -> Iterable[ast.AST]:
    yield node
    if isinstance(node, ast.If):
        yield from _effective_nodes(node.test)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            branch = node.body if node.test.value else node.orelse
            yield from _effective_block(branch)
        else:
            yield from _effective_block(node.body)
            yield from _effective_block(node.orelse)
        return
    if isinstance(node, ast.While):
        yield from _effective_nodes(node.test)
        if not (
            isinstance(node.test, ast.Constant)
            and node.test.value is False
        ):
            yield from _effective_block(node.body)
        yield from _effective_block(node.orelse)
        return
    for field_name, value in ast.iter_fields(node):
        if field_name in {"body", "orelse", "finalbody"} and isinstance(value, list):
            yield from _effective_block(value)
        elif isinstance(value, ast.AST):
            yield from _effective_nodes(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    yield from _effective_nodes(item)


def _effective_block(statements: list[ast.stmt]) -> Iterable[ast.AST]:
    for statement in statements:
        yield from _effective_nodes(statement)
        if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            break


def inspect_pallas_source(source: str) -> PallasInspection:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return PallasInspection(
            parses=False,
            has_workload=False,
            reasons=("SYNTAX_INVALID",),
        )

    module_aliases: set[str] = set()
    direct_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"jax.experimental", "jax.experimental.pallas"}:
                for alias in node.names:
                    if module == "jax.experimental" and alias.name == "pallas":
                        module_aliases.add(alias.asname or alias.name)
                    if module == "jax.experimental.pallas" and alias.name == "pallas_call":
                        direct_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jax.experimental.pallas":
                    module_aliases.add(alias.asname or alias.name)

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "workload" not in functions:
        return PallasInspection(
            parses=True,
            has_workload=False,
            reasons=("WORKLOAD_MISSING",),
        )

    graph: dict[str, set[str]] = {name: set() for name in functions}
    pallas_calls: dict[str, int] = {name: 0 for name in functions}
    interpret_pallas_calls: dict[str, int] = {name: 0 for name in functions}
    all_pallas_calls: dict[str, int] = {
        name: sum(
            1
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _is_pallas_call(_call_name(node), direct_aliases, module_aliases)
        )
        for name, function in functions.items()
    }
    for name, function in functions.items():
        for node in _effective_block(function.body):
            if isinstance(node, ast.Call):
                called = _call_name(node)
                if called in functions:
                    graph[name].add(called)
                if _is_pallas_call(called, direct_aliases, module_aliases):
                    pallas_calls[name] += 1
                    if _may_use_interpret_mode(node):
                        interpret_pallas_calls[name] += 1
                    if node.args and isinstance(node.args[0], ast.Name):
                        kernel_name = node.args[0].id
                        if kernel_name in functions:
                            graph[name].add(kernel_name)

    reachable = {"workload"}
    pending = ["workload"]
    while pending:
        current = pending.pop()
        for called in graph[current]:
            if called not in reachable:
                reachable.add(called)
                pending.append(called)

    reaches_pallas = {
        name for name, count in pallas_calls.items() if count > 0
    }
    changed = True
    while changed:
        changed = False
        for name, called_functions in graph.items():
            if name not in reaches_pallas and called_functions & reaches_pallas:
                reaches_pallas.add(name)
                changed = True

    reachable_count = sum(pallas_calls[name] for name in reachable)
    reachable_interpret_count = sum(
        interpret_pallas_calls[name] for name in reachable
    )
    reachable_lowered_count = reachable_count - reachable_interpret_count
    total_count = sum(all_pallas_calls.values())
    has_fallback = reachable_count > 0 and any(
        _has_non_pallas_return(
            functions[name],
            direct_aliases=direct_aliases,
            module_aliases=module_aliases,
            reaches_pallas=reaches_pallas,
        )
        for name in reachable & reaches_pallas
    )
    reasons: list[str] = []
    if reachable_count == 0:
        reasons.append("PALLAS_PATH_UNREACHABLE")
    if has_fallback:
        reasons.append("PLAIN_JAX_FALLBACK")
    if reachable_interpret_count:
        reasons.append("PALLAS_INTERPRET_MODE")
    return PallasInspection(
        parses=True,
        has_workload=True,
        reachable_functions=tuple(sorted(reachable)),
        reachable_pallas_calls=reachable_count,
        reachable_lowered_pallas_calls=reachable_lowered_count,
        reachable_interpret_pallas_calls=reachable_interpret_count,
        unreachable_pallas_calls=total_count - reachable_count,
        has_plain_jax_fallback=has_fallback,
        authentic=(
            reachable_lowered_count > 0
            and reachable_interpret_count == 0
            and not has_fallback
        ),
        reasons=tuple(reasons),
    )


def timing_evidence(
    samples_ms: Iterable[float],
    *,
    min_runs: int,
    max_coefficient_of_variation: float,
) -> TimingEvidence:
    try:
        raw_samples = tuple(samples_ms)
        for sample in raw_samples:
            chex.assert_rank(sample, 0)
            try:
                chex.assert_type(sample, float)
            except AssertionError:
                chex.assert_type(sample, int)
        samples = tuple(float(sample) for sample in raw_samples)
        values = jnp.asarray(samples)
        chex.assert_rank(values, 1)
        chex.assert_tree_all_finite(values)
        if samples:
            chex.assert_scalar_positive(min(samples))
    except (AssertionError, TypeError, ValueError) as exc:
        raise TimingEvidenceError(f"TIMING_SAMPLES_INVALID: {exc}") from exc
    if len(samples) < min_runs:
        return TimingEvidence(
            samples_ms=samples,
            median_ms=statistics.median(samples) if samples else None,
            coefficient_of_variation=None,
            stable=False,
            reason=f"TIMING_REPEATS_INSUFFICIENT: {len(samples)} < {min_runs}",
        )
    mean = statistics.fmean(samples)
    coefficient = statistics.pstdev(samples) / mean if mean > 0 else None
    stable = coefficient is not None and coefficient <= max_coefficient_of_variation
    return TimingEvidence(
        samples_ms=samples,
        median_ms=statistics.median(samples),
        coefficient_of_variation=coefficient,
        stable=stable,
        reason=None if stable else "TIMING_UNSTABLE",
    )


def judge(
    *,
    workload: str,
    candidate_src: str,
    baseline_src: str,
    compiled: bool,
    correct: bool,
    prompt_context: PromptContext | str,
    speedup: float | None = None,
    timing_stable: bool | None = None,
    copy_threshold: float = COPY_SIMILARITY_THRESHOLD,
    headline_speedup_threshold: float = DEFAULT_HEADLINE_SPEEDUP,
    lowering_verified: bool | None = None,
    require_lowering_evidence: bool = False,
) -> KernelVerdict:
    context = PromptContext(prompt_context)
    inspection = inspect_pallas_source(candidate_src)
    similarity = baseline_similarity(candidate_src, baseline_src)
    verbatim = is_verbatim_file_copy(candidate_src, baseline_src)
    near_copy = verbatim or (similarity is not None and similarity >= copy_threshold)
    copied = context is PromptContext.BASELINE and near_copy

    reasons: list[str] = []
    if not context.scorable:
        reasons.append("DIAGNOSTIC_PROMPT_CONTEXT")
    if not compiled:
        reasons.append("TPU_COMPILE_FAILED")
    if not correct:
        reasons.append("CORRECTNESS_FAILED")
    if copied:
        reasons.append("REFERENCE_COPY")
    reasons.extend(inspection.reasons)
    if timing_stable is False:
        reasons.append("TIMING_UNSTABLE")
    if (
        require_lowering_evidence
        and compiled
        and correct
        and inspection.authentic
        and lowering_verified is not True
    ):
        reasons.append(
            "LOWERING_EVIDENCE_FAILED"
            if lowering_verified is False
            else "LOWERING_EVIDENCE_MISSING"
        )

    credited = context.scorable and correct and not copied
    pallas_credited = (
        credited
        and compiled
        and inspection.authentic
        and (not require_lowering_evidence or lowering_verified is True)
    )
    headline = (
        pallas_credited
        and timing_stable is True
        and speedup is not None
        and speedup >= headline_speedup_threshold
    )
    return KernelVerdict(
        workload=workload,
        compiled=compiled,
        correct=correct,
        prompt_context=context,
        inspection=inspection,
        similarity=similarity,
        verbatim_file_copy=verbatim,
        speedup=speedup,
        timing_stable=timing_stable,
        lowering_verified=lowering_verified,
        copied=copied,
        credited=credited,
        pallas_credited=pallas_credited,
        headline_credited=headline,
        no_credit_reasons=tuple(dict.fromkeys(reasons)),
    )


REWARD_CORRECT = 0.3
REWARD_PALLAS = 0.3
REWARD_SPEED_MAX = 0.4
REWARD_SPEED_SATURATION = 2.0


def diagnostic_reward(verdict: KernelVerdict) -> float:
    if not verdict.credited:
        return 0.0
    total = REWARD_CORRECT
    if not verdict.pallas_credited:
        return total
    total += REWARD_PALLAS
    if verdict.timing_stable and verdict.speedup and verdict.speedup > 1:
        fraction = min(
            (verdict.speedup - 1) / (REWARD_SPEED_SATURATION - 1),
            1,
        )
        total += REWARD_SPEED_MAX * fraction
    return round(total, 4)


def summarise(verdicts: list[KernelVerdict]) -> dict[str, object]:
    scorable = [verdict for verdict in verdicts if verdict.scorable]
    speedups = [
        verdict.speedup
        for verdict in verdicts
        if verdict.pallas_credited and verdict.timing_stable and verdict.speedup is not None
    ]
    return {
        "n": len(verdicts),
        "n_scorable": len(scorable),
        "n_correct_raw": sum(verdict.correct for verdict in verdicts),
        "n_copied": sum(verdict.copied for verdict in verdicts),
        "n_credited": sum(verdict.credited for verdict in verdicts),
        "n_authentic_pallas": sum(verdict.inspection.authentic for verdict in verdicts),
        "n_pallas_credited": sum(verdict.pallas_credited for verdict in verdicts),
        "n_headline_credited": sum(verdict.headline_credited for verdict in verdicts),
        "best_stable_pallas_speedup": max(speedups) if speedups else None,
        "mean_diagnostic_reward": (
            round(sum(diagnostic_reward(verdict) for verdict in scorable) / len(scorable), 4)
            if scorable
            else None
        ),
        "generalization_claim_ready": False,
    }
