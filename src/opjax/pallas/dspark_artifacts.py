from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(run_root: Path) -> dict[str, object]:
    manifest_path = run_root / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifact manifest must contain an artifacts object")

    missing: list[str] = []
    mismatched: list[dict[str, object]] = []
    for relative, expected in sorted(artifacts.items()):
        path = run_root / relative
        if expected.get("kind") == "symlink":
            if not path.is_symlink():
                missing.append(relative)
                continue
            actual_target = path.readlink().as_posix()
            if actual_target != expected.get("target"):
                mismatched.append(
                    {
                        "path": relative,
                        "expected_target": expected.get("target"),
                        "actual_target": actual_target,
                    }
                )
            continue
        if not path.is_file() or path.is_symlink():
            missing.append(relative)
            continue
        actual_size = path.stat().st_size
        actual_hash = file_sha256(path)
        if actual_size != expected.get("size") or actual_hash != expected.get("sha256"):
            mismatched.append(
                {
                    "path": relative,
                    "expected_size": expected.get("size"),
                    "actual_size": actual_size,
                    "expected_sha256": expected.get("sha256"),
                    "actual_sha256": actual_hash,
                }
            )
    return {
        "valid": not missing and not mismatched,
        "artifact_count": len(artifacts),
        "missing": missing,
        "mismatched": mismatched,
    }


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_telemetry(path: Path) -> dict[str, object]:
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid telemetry JSON at line {line_number}") from exc
        if row.get("event") == "opjax_dspark_heartbeat":
            samples.append(row)
    if not samples:
        raise ValueError("telemetry contains no heartbeat samples")

    elapsed = [_number(sample.get("elapsed_s")) for sample in samples]
    elapsed_values = [value for value in elapsed if value is not None]
    gaps = [right - left for left, right in zip(elapsed_values, elapsed_values[1:])]
    stages: dict[str, float] = {}
    checkpoint_first_seen_s: float | None = None
    gpu_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for sample in samples:
        sample_elapsed = _number(sample.get("elapsed_s"))
        for name, state in sample.get("files", {}).items():
            if (
                state.get("exists")
                and name not in stages
                and sample_elapsed is not None
            ):
                stages[name] = sample_elapsed
        if (
            sample.get("checkpoint_states")
            and checkpoint_first_seen_s is None
            and sample_elapsed is not None
        ):
            checkpoint_first_seen_s = sample_elapsed
        for gpu in sample.get("gpus", []):
            index = str(gpu.get("index", "unknown"))
            for field in (
                "memory.used",
                "utilization.gpu",
                "utilization.memory",
                "power.draw",
                "temperature.gpu",
                "clocks.sm",
                "clocks.mem",
            ):
                value = _number(gpu.get(field))
                if value is not None:
                    gpu_values[index][field].append(value)

    gpu_summary: dict[str, object] = {}
    for index, fields in sorted(gpu_values.items()):
        gpu_summary[index] = {
            field: {
                "mean": round(fmean(values), 3),
                "p95": _percentile(values, 0.95),
                "max": max(values),
            }
            for field, values in sorted(fields.items())
        }

    timestamps = [
        datetime.fromisoformat(sample["timestamp"])
        for sample in samples
        if sample.get("timestamp")
    ]
    return {
        "sample_count": len(samples),
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
        "first_elapsed_s": min(elapsed_values) if elapsed_values else None,
        "last_elapsed_s": max(elapsed_values) if elapsed_values else None,
        "max_sample_gap_s": max(gaps) if gaps else None,
        "rank0_terminal_status": samples[-1].get("rank0_status"),
        "rank1_terminal_status": samples[-1].get("rank1_status"),
        "stage_first_seen_s": dict(sorted(stages.items())),
        "checkpoint_first_seen_s": checkpoint_first_seen_s,
        "gpus": gpu_summary,
    }


def _read_trace(path: Path) -> dict[str, Any] | list[Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def summarize_trace(path: Path, *, top_n: int = 25) -> dict[str, object]:
    document = _read_trace(path)
    events = document.get("traceEvents", []) if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise ValueError(f"traceEvents must be a list: {path}")

    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "duration_us": 0.0}
    )
    complete_events = 0
    total_duration_us = 0.0
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        duration = _number(event.get("dur"))
        if duration is None:
            continue
        complete_events += 1
        total_duration_us += duration
        key = (str(event.get("cat", "")), str(event.get("name", "<unnamed>")))
        totals[key]["count"] += 1
        totals[key]["duration_us"] += duration

    ranked = sorted(
        (
            {
                "category": category,
                "name": name,
                "count": int(values["count"]),
                "total_duration_us": round(values["duration_us"], 3),
                "mean_duration_us": round(values["duration_us"] / values["count"], 3),
            }
            for (category, name), values in totals.items()
        ),
        key=lambda item: item["total_duration_us"],
        reverse=True,
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size": path.stat().st_size,
        "event_count": len(events),
        "complete_event_count": complete_events,
        "summed_complete_event_duration_us": round(total_duration_us, 3),
        "top_complete_events": ranked[:top_n],
    }


def analyze_run(run_root: Path, *, top_n: int = 25) -> dict[str, object]:
    telemetry_path = run_root / "runtime-telemetry.jsonl"
    trace_paths = sorted(run_root.glob("**/profile_rank*.trace.json.gz"))
    return {
        "run_root": str(run_root.resolve()),
        "manifest": validate_manifest(run_root),
        "telemetry": summarize_telemetry(telemetry_path),
        "traces": [summarize_trace(path, top_n=top_n) for path in trace_paths],
    }
