"""Post-facto analysis for a Laguna DSpark production profile."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from opjax.pallas.laguna_dspark_conformance import canonical_sha256


_METRIC = re.compile(
    r'^(?P<name>vllm:[^{ ]+)(?:\{(?P<labels>[^}]*)\})? (?P<value>[-+0-9.eE]+)$'
)
_SELECTED = {
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:request_success_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_drafts_total",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prometheus_values(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _METRIC.match(line)
        if match is None or match.group("name") not in _SELECTED:
            continue
        name = match.group("name")
        labels = match.group("labels") or ""
        position = re.search(r'position="(?P<value>\d+)"', labels)
        reason = re.search(r'finished_reason="(?P<value>[^"]+)"', labels)
        if position:
            name = f"{name}.position_{position.group('value')}"
        if reason:
            name = f"{name}.finished_{reason.group('value')}"
        values[name] = float(match.group("value"))
    return values


def _trace_summary(path: Path) -> dict[str, Any]:
    handle = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("rt", encoding="utf-8")
    )
    with handle:
        trace = json.load(handle)
    counts: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    selected = (
        "cudaGraphLaunch",
        "vllm::unified_attention_with_output",
        "_C::rotary_embedding",
        "_combine_sampled_and_draft_tokens_kernel",
    )
    for event in trace.get("traceEvents", []):
        name = event.get("name")
        if name not in selected:
            continue
        counts[name] += 1
        durations[name] += float(event.get("dur", 0.0) or 0.0)
    return {
        name: {
            "calls": counts[name],
            "total_duration_us": durations[name],
        }
        for name in selected
    }


def build_profile_analysis(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["trace_files"]:
        path = root / item["path"]
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"PROFILE_TRACE_HASH_MISMATCH:{path}")
    before = prometheus_values((root / "metrics-before.prom").read_text())
    after = prometheus_values((root / "metrics-after.prom").read_text())
    delta = {
        key: value - before.get(key, 0.0)
        for key, value in sorted(after.items())
    }
    drafts = delta.get("vllm:spec_decode_num_drafts_total", 0.0)
    drafted = delta.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    worker_trace = next(
        root / item["path"]
        for item in manifest["trace_files"]
        if "rank0" in item["path"]
    )
    analysis: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_dspark_production_profile_analysis",
        "manifest_sha256": manifest["manifest_sha256"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "response_metrics": manifest["response"].get("metrics"),
        "prometheus_delta": delta,
        "speculation": {
            "draft_rounds": drafts,
            "drafted_tokens": drafted,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / drafted if drafted else None,
            "accepted_tokens_per_round": accepted / drafts if drafts else None,
        },
        "trace": _trace_summary(worker_trace),
    }
    analysis["analysis_sha256"] = canonical_sha256(analysis)
    return analysis
