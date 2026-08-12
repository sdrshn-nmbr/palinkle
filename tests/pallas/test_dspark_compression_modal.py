from __future__ import annotations

import json
import importlib.util
import time
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
SPECIFICATION = importlib.util.spec_from_file_location(
    "train_dspark_compression_modal",
    REPO_ROOT / "scripts/pallas/train_dspark_compression_modal.py",
)
assert SPECIFICATION is not None and SPECIFICATION.loader is not None
training = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(training)


class FakeProcess:
    def __init__(self, status: int | None) -> None:
        self.status = status

    def poll(self) -> int | None:
        return self.status


def test_config_enables_one_bounded_profile_step(tmp_path: Path) -> None:
    canary = training.write_config("dspark-500m", tmp_path / "canary", 1)
    full = training.write_config("dspark-500m", tmp_path / "full", 156)

    canary_text = canary.read_text()
    full_text = full.read_text()
    assert "profiling:\n  enabled: true\n  start_step: 0\n  num_steps: 1" in canary_text
    assert "profiling:\n  enabled: true\n  start_step: 5\n  num_steps: 1" in full_text


def test_heartbeat_is_structured_and_durable(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    sampler = tmp_path.parent / f"{tmp_path.name}-gpu-sampler.log"
    sampler.write_text("2026-08-12T01:00:00+00:00\n0, GPU-1, NVIDIA H200\n")
    monkeypatch.setattr(
        training,
        "run_text_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("heartbeat must not launch external commands")
        ),
    )

    training.emit_runtime_status(
        run_root=tmp_path,
        rank0=FakeProcess(None),
        rank1=FakeProcess(0),
        started_at=time.monotonic(),
    )

    rows = (tmp_path / "runtime-telemetry.jsonl").read_text().splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["event"] == "opjax_dspark_heartbeat"
    assert payload["rank0_status"] is None
    assert payload["rank1_status"] == 0
    assert payload["gpu_sampler_tail"][-1] == "0, GPU-1, NVIDIA H200"
    logged = json.loads(capsys.readouterr().out)
    assert logged["event"] == "opjax_dspark_heartbeat"
    assert logged["gpu_sampler_tail"][-1] == "0, GPU-1, NVIDIA H200"


def test_manifest_hashes_all_frozen_files(tmp_path: Path) -> None:
    (tmp_path / "trace.json.gz").write_bytes(b"trace")
    output = tmp_path / "output"
    output.mkdir()
    (output / "training_state.pt").write_bytes(b"checkpoint")

    training.write_artifact_manifest(tmp_path)

    manifest = json.loads((tmp_path / "artifact-manifest.json").read_text())
    artifacts = manifest["artifacts"]
    assert artifacts["trace.json.gz"]["sha256"] == training.file_sha256(
        tmp_path / "trace.json.gz"
    )
    assert artifacts["output/training_state.pt"]["size"] == len(b"checkpoint")
    assert "artifact-manifest.json" not in artifacts


def test_volume_commits_stop_before_training_writes_begin(tmp_path: Path) -> None:
    assert training.should_commit_while_running(tmp_path) is True

    (tmp_path / "inference.ready").touch()
    assert training.should_commit_while_running(tmp_path) is False


def test_persist_run_snapshot_replaces_prior_state(tmp_path: Path) -> None:
    active = tmp_path / "active" / "run"
    durable = tmp_path / "durable" / "run"
    active.mkdir(parents=True)
    (active / "telemetry.jsonl").write_text("first")
    (active.parent / "run-rank0.log").write_text("rank zero")

    training.persist_run_snapshot(active, durable, replace=True)
    assert (durable / "telemetry.jsonl").read_text() == "first"
    assert (durable / "rank0.log").read_text() == "rank zero"

    (durable / "stale").write_text("remove me")
    (active / "telemetry.jsonl").write_text("final")
    training.persist_run_snapshot(active, durable, replace=True)
    assert (durable / "telemetry.jsonl").read_text() == "final"
    assert not (durable / "stale").exists()
