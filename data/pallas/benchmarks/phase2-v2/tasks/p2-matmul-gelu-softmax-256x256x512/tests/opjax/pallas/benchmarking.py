"""Order-controlled timing statistics for TPU kernel comparisons."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable
from typing import Any


class BenchmarkingError(RuntimeError):
    """A timing comparison cannot support a performance claim."""


def _measurement_orders(*, rounds: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    order_codes = [index % 2 for index in range(rounds)]
    rng.shuffle(order_codes)
    return [
        ["candidate", "baseline"]
        if code == 0
        else ["baseline", "candidate"]
        for code in order_codes
    ]


def _relative_spread(values: list[float]) -> float:
    median = statistics.median(values)
    if median <= 0:
        raise BenchmarkingError("TIMING_SAMPLE_NONPOSITIVE")
    deviations = statistics.median(abs(value - median) for value in values) / median
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return max(deviations, (quartiles[2] - quartiles[0]) / median)


def _bootstrap_speedup_ci(
    candidate: list[float], baseline: list[float], *, seed: int, draws: int = 4_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    ratios = []
    count = len(candidate)
    for _ in range(draws):
        indices = [rng.randrange(count) for _ in range(count)]
        candidate_median = statistics.median(candidate[index] for index in indices)
        baseline_median = statistics.median(baseline[index] for index in indices)
        ratios.append(baseline_median / candidate_median)
    ratios.sort()
    return ratios[math.floor(0.025 * draws)], ratios[math.ceil(0.975 * draws) - 1]


def measure_interleaved(
    *,
    candidate: Callable[[], float],
    baseline: Callable[[], float],
    rounds: int,
    seed: int,
    material_speedup: float = 1.05,
    maximum_relative_mad: float = 0.10,
) -> dict[str, Any]:
    """Measure paired candidate/XLA samples in a reproducibly shuffled order."""
    if rounds < 5 or material_speedup <= 1 or maximum_relative_mad <= 0:
        raise BenchmarkingError("TIMING_CONTRACT_INVALID")
    candidate_samples: list[float] = []
    baseline_samples: list[float] = []
    orders = _measurement_orders(rounds=rounds, seed=seed)
    for order in orders:
        measurements: dict[str, float] = {}
        for label in order:
            value = candidate() if label == "candidate" else baseline()
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise BenchmarkingError(f"TIMING_SAMPLE_INVALID:{label}:{value!r}")
            measurements[label] = float(value)
        candidate_samples.append(measurements["candidate"])
        baseline_samples.append(measurements["baseline"])
    candidate_median = statistics.median(candidate_samples)
    baseline_median = statistics.median(baseline_samples)
    speedup = baseline_median / candidate_median
    ci95 = _bootstrap_speedup_ci(candidate_samples, baseline_samples, seed=seed)
    relative_spread = {
        "candidate": _relative_spread(candidate_samples),
        "baseline": _relative_spread(baseline_samples),
    }
    unstable = max(relative_spread.values()) > maximum_relative_mad
    return {
        "candidate_ms": candidate_samples,
        "baseline_ms": baseline_samples,
        "candidate_median_ms": candidate_median,
        "baseline_median_ms": baseline_median,
        "speedup": speedup,
        "speedup_ci95": list(ci95),
        "relative_spread": relative_spread,
        "maximum_relative_spread": maximum_relative_mad,
        "unstable": unstable,
        "material_speedup_threshold": material_speedup,
        "materially_beats_xla": not unstable and ci95[0] > material_speedup,
        "measurement_orders": orders,
    }


def validate_timing_result(result: dict[str, Any], *, seed: int) -> dict[str, Any]:
    candidate = result.get("candidate_ms")
    baseline = result.get("baseline_ms")
    orders = result.get("measurement_orders")
    if (
        not isinstance(candidate, list)
        or not isinstance(baseline, list)
        or len(candidate) != len(baseline)
        or len(candidate) < 5
        or not isinstance(orders, list)
        or len(orders) != len(candidate)
    ):
        raise BenchmarkingError("TIMING_RESULT_SCHEMA_INVALID")
    if orders != _measurement_orders(rounds=len(candidate), seed=seed):
        raise BenchmarkingError("TIMING_ORDER_SCHEDULE_INVALID")
    try:
        candidate_values = [float(value) for value in candidate]
        baseline_values = [float(value) for value in baseline]
        if any(
            not math.isfinite(value) or value <= 0
            for value in candidate_values + baseline_values
        ):
            raise BenchmarkingError("TIMING_SAMPLE_INVALID")
        candidate_median = statistics.median(candidate_values)
        baseline_median = statistics.median(baseline_values)
        speedup = baseline_median / candidate_median
        ci95 = _bootstrap_speedup_ci(candidate_values, baseline_values, seed=seed)
        relative_spread = {
            "candidate": _relative_spread(candidate_values),
            "baseline": _relative_spread(baseline_values),
        }
        maximum_spread = float(result["maximum_relative_spread"])
        material_speedup = float(result["material_speedup_threshold"])
        if maximum_spread <= 0 or material_speedup <= 1:
            raise BenchmarkingError("TIMING_CONTRACT_INVALID")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise BenchmarkingError("TIMING_RESULT_SCHEMA_INVALID") from exc
    unstable = max(relative_spread.values()) > maximum_spread
    materially_beats = not unstable and ci95[0] > material_speedup
    numeric_checks = (
        (result.get("candidate_median_ms"), candidate_median),
        (result.get("baseline_median_ms"), baseline_median),
        (result.get("speedup"), speedup),
    )
    valid = all(
        isinstance(observed, (int, float))
        and math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
        for observed, expected in numeric_checks
    )
    observed_ci = result.get("speedup_ci95")
    valid = valid and isinstance(observed_ci, list) and len(observed_ci) == 2 and all(
        math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
        for observed, expected in zip(observed_ci, ci95, strict=True)
    )
    valid = (
        valid
        and result.get("relative_spread") == relative_spread
        and result.get("unstable") is unstable
        and result.get("materially_beats_xla") is materially_beats
    )
    if not valid:
        raise BenchmarkingError("TIMING_RESULT_MISMATCH")
    return {"verified": True, "rounds": len(candidate_values)}
