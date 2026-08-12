from __future__ import annotations

import gzip
import json
from hashlib import sha256
from pathlib import Path

from opjax.pallas.dspark_artifacts import (
    summarize_gpu_sampler,
    summarize_host_sampler,
    summarize_telemetry,
    summarize_trace,
    validate_manifest,
)


def test_validate_manifest_detects_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"valid")
    (tmp_path / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "checkpoint.pt": {
                        "size": 5,
                        "sha256": sha256(b"valid").hexdigest(),
                    }
                }
            }
        )
    )
    assert validate_manifest(tmp_path)["valid"] is True

    artifact.write_bytes(b"changed")
    result = validate_manifest(tmp_path)
    assert result["valid"] is False
    assert result["mismatched"][0]["path"] == "checkpoint.pt"


def test_validate_manifest_checks_symlink_target(tmp_path: Path) -> None:
    (tmp_path / "actual").write_text("artifact")
    (tmp_path / "latest").symlink_to("actual")
    (tmp_path / "artifact-manifest.json").write_text(
        json.dumps({"artifacts": {"latest": {"kind": "symlink", "target": "actual"}}})
    )
    assert validate_manifest(tmp_path)["valid"] is True

    (tmp_path / "latest").unlink()
    (tmp_path / "latest").symlink_to("missing")
    assert validate_manifest(tmp_path)["valid"] is False


def test_summarize_telemetry_reports_stage_and_gpu_metrics(tmp_path: Path) -> None:
    telemetry = tmp_path / "runtime-telemetry.jsonl"
    rows = [
        {
            "event": "opjax_dspark_heartbeat",
            "timestamp": "2026-08-12T01:00:00+00:00",
            "elapsed_s": 0,
            "rank0_status": None,
            "rank1_status": None,
            "files": {"inference.ready": {"exists": False}},
            "checkpoint_states": [],
            "gpus": [{"index": "0", "memory.used": "100", "utilization.gpu": "0"}],
        },
        {
            "event": "opjax_dspark_heartbeat",
            "timestamp": "2026-08-12T01:00:30+00:00",
            "elapsed_s": 30,
            "rank0_status": 0,
            "rank1_status": 0,
            "files": {"inference.ready": {"exists": True}},
            "checkpoint_states": ["output/run-step1/training_state.pt"],
            "gpus": [{"index": "0", "memory.used": "300", "utilization.gpu": "80"}],
        },
    ]
    telemetry.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = summarize_telemetry(telemetry)
    assert result["sample_count"] == 2
    assert result["stage_first_seen_s"]["inference.ready"] == 30
    assert result["checkpoint_first_seen_s"] == 30
    assert result["gpus"]["0"]["memory.used"]["max"] == 300
    assert result["gpus"]["0"]["utilization.gpu"]["mean"] == 40


def test_summarize_trace_aggregates_complete_events(tmp_path: Path) -> None:
    trace = tmp_path / "profile_rank0_1.trace.json.gz"
    document = {
        "traceEvents": [
            {"ph": "X", "cat": "cpu_op", "name": "forward", "dur": 10},
            {"ph": "X", "cat": "cpu_op", "name": "forward", "dur": 20},
            {"ph": "i", "cat": "meta", "name": "ignored"},
        ]
    }
    with gzip.open(trace, "wt") as handle:
        json.dump(document, handle)

    result = summarize_trace(trace)
    assert result["complete_event_count"] == 2
    assert result["summed_complete_event_duration_us"] == 30
    assert result["top_complete_events"][0] == {
        "category": "cpu_op",
        "name": "forward",
        "count": 2,
        "total_duration_us": 30,
        "mean_duration_us": 15,
    }


def test_summarize_gpu_sampler_reports_utilization_and_failures(tmp_path: Path) -> None:
    sampler = tmp_path / "gpu-sampler.log"
    sampler.write_text(
        "2026-08-12T01:00:00+00:00\n"
        "0, GPU-1, NVIDIA H200, P0, 100, 140000, 20, 10, 200, 700, 50, 1000, 3000\n"
        "2026-08-12T01:00:05+00:00\n"
        "0, GPU-1, NVIDIA H200, P0, 300, 140000, 80, 20, 400, 700, 60, 1800, 3000\n"
        "nvidia-smi-status=124\n"
    )

    result = summarize_gpu_sampler(sampler)
    assert result["sample_count"] == 2
    assert result["failure_count"] == 1
    assert result["gpus"]["0"]["utilization.gpu"]["mean"] == 50
    assert result["gpus"]["0"]["memory.used"]["max"] == 300


def test_summarize_host_sampler_reports_memory_and_process_peaks(
    tmp_path: Path,
) -> None:
    sampler = tmp_path / "host-sampler.log"
    sampler.write_text(
        "2026-08-12T01:00:00+00:00\n"
        "cgroup-memory-current=1000\n"
        "cgroup-memory-max=2000\n"
        "MemAvailable: 900 kB\n"
        "10 1 S 4 100 200 5.0 1.0 python python server.py\n"
        "2026-08-12T01:00:10+00:00\n"
        "cgroup-memory-current=1800\n"
        "cgroup-memory-max=2000\n"
        "MemAvailable: 100 kB\n"
        "10 1 S 4 150 200 5.0 1.0 python python server.py\n"
    )

    result = summarize_host_sampler(sampler)
    assert result["sample_count"] == 2
    assert result["cgroup_memory_current_bytes"] == {"max": 1800, "last": 1800}
    assert result["mem_available_kib"] == {"min": 100, "last": 100}
    assert result["process_max_rss_kib"]["python"] == 150
