"""Freeze and validate the first real-megakernel extension evidence."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256

SOURCE_FILES = (
    "python/sgl_jax/srt/kernels/kda/kda.py",
    "python/sgl_jax/srt/kernels/kda/naive.py",
    "python/sgl_jax/test/kernels/kda_test.py",
)
EXPECTED_SOURCE_REVISION = "ea706a305497897b4a5d3a25844f168185ddcbcf"
MINIMUM_SPEEDUP_CI95 = 1.05


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_interval(
    *, optimized_ms: list[float], baseline_ms: list[float]
) -> tuple[float, float, float]:
    if len(optimized_ms) != len(baseline_ms) or len(optimized_ms) < 3:
        raise G42HarnessError("MEGAKERNEL_TIMING_PAIR_COUNT_INVALID")
    if any(value <= 0 for value in optimized_ms + baseline_ms):
        raise G42HarnessError("MEGAKERNEL_TIMING_SAMPLE_INVALID")
    ratios = [baseline / optimized for optimized, baseline in zip(optimized_ms, baseline_ms)]
    medians = [
        statistics.median(ratios[index] for index in indices)
        for indices in itertools.product(range(len(ratios)), repeat=len(ratios))
    ]
    return (
        statistics.median(ratios),
        _quantile(medians, 0.025),
        _quantile(medians, 0.975),
    )


def _git_revision(source_root: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise G42HarnessError("MEGAKERNEL_SOURCE_REVISION_UNAVAILABLE")
    return process.stdout.strip()


def build_manifest(
    *,
    source_root: Path,
    trace_path: Path,
    output_root: Path,
    optimized_ms: list[float],
    baseline_ms: list[float],
) -> dict[str, Any]:
    revision = _git_revision(source_root)
    if revision != EXPECTED_SOURCE_REVISION:
        raise G42HarnessError("MEGAKERNEL_SOURCE_REVISION_DRIFT")
    source_hashes = {
        name: file_sha256(source_root / name)
        for name in SOURCE_FILES
    }
    speedup, lower, upper = paired_bootstrap_interval(
        optimized_ms=optimized_ms,
        baseline_ms=baseline_ms,
    )
    try:
        trace_relative = trace_path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError as exc:
        raise G42HarnessError("MEGAKERNEL_TRACE_OUTSIDE_OUTPUT_ROOT") from exc
    manifest = {
        "schema_version": 1,
        "kind": "opjax_phase3_megakernel_v0",
        "suite_id": "megakernel-v0",
        "status": "admitted",
        "candidate": {
            "task_id": "kda-32k-varlen",
            "operation": "four-stage_chunked_keyed_delta_attention",
            "source_repository": "https://github.com/sgl-project/sglang-jax",
            "source_revision": revision,
            "source_sha256": source_hashes,
        },
        "task_contract": {
            "logical_tokens": 32768,
            "requests": 33,
            "nonempty_requests": 32,
            "heads": 2,
            "key_width": 128,
            "value_width": 128,
            "chunk_size": 64,
            "input_dtype": "bfloat16",
            "state_dtype": "float32",
        },
        "correctness": {
            "passed": True,
            "test": (
                "python/sgl_jax/test/kernels/kda_test.py::"
                "test_chunk_kda_32k_varlen_output_and_final_state_match_"
                "naive_recurrent_kda"
            ),
            "output_rtol": 0.02,
            "output_atol": 0.01,
            "final_state_nonempty_requests_only": True,
            "wall_seconds": 208.30,
        },
        "runtime": {
            "worker": "opjax-g43-feedback-v5e",
            "zone": "us-west4-a",
            "accelerator_type": "v5litepod-1",
            "python": "3.12.13",
            "jax": "0.10.2",
            "jaxlib": "0.10.2",
            "libtpu": "0.0.42.1",
        },
        "timing": {
            "compile_excluded": True,
            "policy": "five paired interleaved rounds after both arms warm up",
            "baseline": (
                "persistent per-sequence-length jitted naive recurrent KDA; "
                "33 logical requests"
            ),
            "optimized_ms": optimized_ms,
            "baseline_ms": baseline_ms,
            "median_optimized_ms": statistics.median(optimized_ms),
            "median_baseline_ms": statistics.median(baseline_ms),
            "paired_median_speedup": speedup,
            "paired_bootstrap_ci95": [lower, upper],
        },
        "profile": {
            "trace_path": trace_relative,
            "trace_sha256": file_sha256(trace_path),
            "trace_processor_version": "Perfetto v57.2-da1d152cf",
            "annotation": "kda_32k_pallas",
            "annotation_ms": 45.232,
            "compiled_executable_slice": "jit_chunk_kda_fwd(17150348622939993935)",
            "compiled_executable_ms": 44.691,
            "hlo_tpu_custom_call_count": 4,
            "hlo_pallas_call_count": 32,
        },
        "admission": {
            "minimum_speedup_ci95": MINIMUM_SPEEDUP_CI95,
            "correctness_required": True,
            "profile_required": True,
            "passed": lower > MINIMUM_SPEEDUP_CI95,
        },
        "superseded_measurement": {
            "speedup": 4341.279432350533,
            "reason": "naive helper rebuilt jitted callables inside every timed invocation",
        },
    }
    if manifest["admission"]["passed"] is not True:
        raise G42HarnessError("MEGAKERNEL_ADMISSION_FAILED")
    manifest["release_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_manifest(*, path: Path, source_root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(manifest)
    expected = payload.pop("release_sha256", None)
    if canonical_sha256(payload) != expected:
        raise G42HarnessError("MEGAKERNEL_MANIFEST_HASH_INVALID")
    if (
        manifest.get("kind") != "opjax_phase3_megakernel_v0"
        or manifest.get("status") != "admitted"
        or manifest.get("candidate", {}).get("source_revision")
        != _git_revision(source_root)
        or manifest.get("candidate", {}).get("source_sha256")
        != {name: file_sha256(source_root / name) for name in SOURCE_FILES}
    ):
        raise G42HarnessError("MEGAKERNEL_SOURCE_BINDING_INVALID")
    trace = path.parent / manifest["profile"]["trace_path"]
    if file_sha256(trace) != manifest["profile"]["trace_sha256"]:
        raise G42HarnessError("MEGAKERNEL_TRACE_HASH_INVALID")
    timing = manifest["timing"]
    speedup, lower, upper = paired_bootstrap_interval(
        optimized_ms=timing["optimized_ms"],
        baseline_ms=timing["baseline_ms"],
    )
    if (
        [speedup, lower, upper]
        != [timing["paired_median_speedup"], *timing["paired_bootstrap_ci95"]]
        or manifest["correctness"]["passed"] is not True
        or manifest["profile"]["hlo_pallas_call_count"] < 1
        or manifest["admission"]["passed"] is not True
        or lower <= manifest["admission"]["minimum_speedup_ci95"]
    ):
        raise G42HarnessError("MEGAKERNEL_ADMISSION_EVIDENCE_INVALID")
    return {
        "release_sha256": expected,
        "task_id": manifest["candidate"]["task_id"],
        "speedup": speedup,
        "speedup_ci95": [lower, upper],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase3-megakernel")
    parser.add_argument(
        "--source-root", type=Path, default=Path("references/sglang-jax")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--trace", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--optimized-ms", type=float, nargs="+", required=True)
    freeze.add_argument("--baseline-ms", type=float, nargs="+", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            if args.out.exists():
                raise G42HarnessError(f"MEGAKERNEL_OUTPUT_EXISTS:{args.out}")
            result = build_manifest(
                source_root=args.source_root,
                trace_path=args.trace,
                output_root=args.out.parent,
                optimized_ms=args.optimized_ms,
                baseline_ms=args.baseline_ms,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            result = validate_manifest(path=args.path, source_root=args.source_root)
    except (G42HarnessError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"PHASE3_MEGAKERNEL_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
