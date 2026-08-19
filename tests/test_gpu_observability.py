from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from opjax.pallas.laguna_speculative import canonical_sha256
from opjax.remote.gpu_observability import (
    GpuStage,
    NsysProfile,
    cuda_capture,
    gpu_runtime_identity,
    model_stage,
    nvtx_range,
    publish_hf,
    publish_r2,
    run_nsys_profile,
    validate_artifact_manifest,
    warm_then_capture,
    write_artifact_manifest,
)


def test_gpu_runtime_identity_is_observed_and_hash_bound() -> None:
    def run(command: tuple[str, ...], **_: object):
        assert command[0] == "nvidia-smi"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="NVIDIA H200, GPU-abc, 570.10, 143771, 9.0\n",
            stderr="",
        )

    identity = gpu_runtime_identity(run=run)
    assert identity["devices"] == [
        {
            "name": "NVIDIA H200",
            "uuid": "GPU-abc",
            "driver_version": "570.10",
            "memory_total_mib": 143771,
            "compute_capability": "9.0",
        }
    ]
    assert identity["sha256"] == canonical_sha256(
        {key: value for key, value in identity.items() if key != "sha256"}
    )


def test_gpu_runtime_identity_supports_bound_multi_gpu_topology() -> None:
    def run(command: tuple[str, ...], **_: object):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "NVIDIA A100, GPU-a, 580.1, 40960, 8.0\n"
                "NVIDIA A100, GPU-b, 580.1, 40960, 8.0\n"
            ),
            stderr="",
        )

    identity = gpu_runtime_identity(run=run, expected_count=2)
    assert [device["uuid"] for device in identity["devices"]] == ["GPU-a", "GPU-b"]


class FakeNvtx:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None]] = []

    def range_push(self, name: str) -> None:
        self.events.append(("push", name))

    def range_pop(self) -> None:
        self.events.append(("pop", None))


class FakeCudaRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def cudaProfilerStart(self) -> None:
        self.events.append("profile-start")

    def cudaProfilerStop(self) -> None:
        self.events.append("profile-stop")


def test_profile_ranges_close_on_failure() -> None:
    nvtx = FakeNvtx()
    runtime_events: list[str] = []

    with pytest.raises(RuntimeError, match="boom"):
        with cuda_capture(runtime=FakeCudaRuntime(runtime_events)):
            with nvtx_range("opjax.attention", nvtx=nvtx):
                raise RuntimeError("boom")

    assert runtime_events == ["profile-start", "profile-stop"]
    assert nvtx.events == [("push", "opjax.attention"), ("pop", None)]


def test_model_stage_names_are_stable() -> None:
    nvtx = FakeNvtx()
    with model_stage(GpuStage.TARGET_FEATURES, nvtx=nvtx):
        pass
    with model_stage(GpuStage.DRAFT_LAYER, layer=3, nvtx=nvtx):
        pass
    assert nvtx.events == [
        ("push", "opjax.target_features"),
        ("pop", None),
        ("push", "opjax.draft_layer.3"),
        ("pop", None),
    ]

    with pytest.raises(ValueError, match="GPU_DRAFT_LAYER_INVALID"):
        with model_stage(GpuStage.DRAFT_LAYER, nvtx=nvtx):
            pass


def test_warm_then_capture_excludes_warmup() -> None:
    events: list[str] = []
    runtime = FakeCudaRuntime(events)
    nvtx = FakeNvtx()

    result = warm_then_capture(
        warmup=lambda: events.append("warmup"),
        workload=lambda: events.append("workload") or 7,
        warmup_steps=2,
        synchronize=lambda: events.append("sync"),
        runtime=runtime,
        nvtx=nvtx,
        range_name="opjax.verification",
    )

    assert result == 7
    assert events == [
        "warmup",
        "warmup",
        "sync",
        "profile-start",
        "workload",
        "sync",
        "profile-stop",
    ]
    assert nvtx.events == [
        ("push", "opjax.verification"),
        ("pop", None),
    ]


def test_nsys_command_is_bounded_and_gpu_only(tmp_path: Path) -> None:
    profile = NsysProfile(
        run_root=tmp_path,
        target=("python", "workload.py"),
    )

    assert profile.command() == (
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--export=sqlite",
        "--force-overwrite=false",
        f"--output={tmp_path / 'profile'}",
        "python",
        "workload.py",
    )


def test_nsys_run_writes_and_validates_all_evidence(tmp_path: Path) -> None:
    def run(command: list[str] | tuple[str, ...], **_: object):
        if command[:2] == ("nsys", "profile"):
            (tmp_path / "profile.nsys-rep").write_bytes(b"report")
            (tmp_path / "profile.sqlite").write_bytes(b"sqlite")
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="ok\n",
            stderr="",
        )

    manifest_path = run_nsys_profile(
        NsysProfile(run_root=tmp_path, target=("python", "workload.py")),
        run=run,
        runtime={"image": "sha256:image", "model": "sha256:model"},
    )

    manifest = validate_artifact_manifest(manifest_path)
    assert {item["path"] for item in manifest["files"]} == {
        "nsys-environment.txt",
        "nsys-version.txt",
        "nvidia-smi.txt",
        "profile.nsys-rep",
        "profile.sqlite",
        "profiler.stderr.txt",
        "profiler.stdout.txt",
        "runtime.json",
    }

    (tmp_path / "profile.sqlite").write_bytes(b"sqlitX")
    with pytest.raises(RuntimeError, match="GPU_ARTIFACT_HASH_MISMATCH"):
        validate_artifact_manifest(manifest_path)


def test_artifact_manifest_rejects_unbound_files(tmp_path: Path) -> None:
    (tmp_path / "trace").write_bytes(b"trace")
    manifest_path = write_artifact_manifest(tmp_path, runtime={"run": "one"})
    (tmp_path / "unbound").write_bytes(b"not in manifest")

    with pytest.raises(RuntimeError, match="GPU_ARTIFACT_UNBOUND_FILES"):
        validate_artifact_manifest(manifest_path)


class FakeCommit:
    oid = "0123456789abcdef"


class FakeHfApi:
    def __init__(self) -> None:
        self.uploaded: dict[str, object] | None = None

    def create_repo(self, **_: object) -> None:
        return None

    def upload_folder(self, **kwargs: object) -> FakeCommit:
        self.uploaded = kwargs
        return FakeCommit()


def test_hf_publish_returns_immutable_revision(tmp_path: Path) -> None:
    (tmp_path / "trace").write_bytes(b"trace")
    write_artifact_manifest(tmp_path, runtime={"run": "one"})
    api = FakeHfApi()
    result = publish_hf(
        tmp_path,
        repo_id="owner/traces",
        path_in_repo="runs/one",
        api=api,
    )

    assert result["revision"] == "0123456789abcdef"
    assert result["locator"] == "hf://datasets/owner/traces@0123456789abcdef/runs/one"
    assert api.uploaded is not None


def test_r2_publish_is_content_addressed_and_manifest_last(tmp_path: Path) -> None:
    (tmp_path / "trace").write_bytes(b"trace")
    manifest_path = write_artifact_manifest(tmp_path, runtime={"run": "one"})
    calls: list[tuple[str, ...]] = []

    def run(command: list[str] | tuple[str, ...], **_: object):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    result = publish_r2(
        tmp_path,
        bucket="traces",
        prefix="opjax",
        run=run,
    )

    manifest = json.loads(manifest_path.read_text())
    assert result["locator"] == (
        "r2://traces/opjax/manifests/"
        f"{manifest['manifest_sha256']}.json"
    )
    assert calls[-1][4] == result["locator"].removeprefix("r2://")
    assert all("/blobs/" in call[4] for call in calls[:-1])


def test_frozen_laguna_attention_cache_evidence_is_bound() -> None:
    root = (
        Path(__file__).parents[1]
        / "data"
        / "pallas"
        / "runs"
        / "laguna-speculator-training-v1"
        / "conformance"
        / "attention-cache-step120-v1"
    )
    binding = json.loads((root / "binding.json").read_text(encoding="utf-8"))
    artifact_root = root / "artifact"
    manifest = validate_artifact_manifest(
        artifact_root / "artifact-manifest.json"
    )
    report = json.loads((artifact_root / "report.json").read_text(encoding="utf-8"))
    inputs = json.loads((artifact_root / "inputs.json").read_text(encoding="utf-8"))

    assert manifest["manifest_sha256"] == binding["immutable_report"][
        "artifact_manifest_sha256"
    ]
    assert binding["immutable_report"]["revision"] == (
        "a8e5b6c7ce2e10d2e97f61f04b171a7fb5f24dbf"
    )
    assert report["report_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    assert report["report_sha256"] == binding["report"]["report_sha256"]
    assert inputs["report_sha256"] == report["report_sha256"]
    assert report["result"] == binding["result"]
