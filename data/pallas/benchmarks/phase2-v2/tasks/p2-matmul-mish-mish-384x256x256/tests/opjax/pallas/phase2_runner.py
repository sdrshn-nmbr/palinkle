"""Run pristine Phase 2 reference grading and freeze its evidence boundary."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from opjax.pallas.benchmarking import BenchmarkingError, validate_timing_result
from opjax.pallas.candidate_policy import candidate_module_policy_error
from opjax.pallas.g42_harness import canonical_sha256, file_sha256, tree_sha256
from opjax.pallas.environment import verify_static
from opjax.pallas.environment_runner import has_host_compute_in_workload
from opjax.pallas.phase2_benchmark import (
    Phase2BenchmarkError,
    range_conditioned_megablox_candidate_source,
    select_performance_subset,
    timing_conditioned_megablox_candidate_source,
    validate_release,
    zero_output_candidate_source,
)
from opjax.pallas.phase2_task_artifacts import render_artifacts
from opjax.pallas.phase2_worker import build_submission_request, write_request
from opjax.pallas.phase2_worker import Phase2WorkerError
from opjax.pallas.scoring import baseline_similarity, is_verbatim_file_copy


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2BenchmarkError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _task_set_sha256(manifest: dict[str, Any]) -> str:
    return canonical_sha256(
        [
            {"task_id": task["task_id"], "task_sha256": task["task_sha256"]}
            for task in manifest["tasks"]
        ]
    )


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "opjax-harness",
            "GIT_AUTHOR_EMAIL": "harness@opjax.invalid",
            "GIT_COMMITTER_NAME": "opjax-harness",
            "GIT_COMMITTER_EMAIL": "harness@opjax.invalid",
        }
    )
    return environment


def materialize_submission(
    *, task_root: Path, patch_path: Path, destination: Path
) -> dict[str, Any]:
    """Apply one captured patch to a fresh copy of only the public task base."""
    if destination.exists():
        raise Phase2BenchmarkError(f"SUBMISSION_DESTINATION_EXISTS:{destination}")
    if not patch_path.is_file():
        raise Phase2BenchmarkError(f"SUBMISSION_PATCH_MISSING:{patch_path}")
    destination.mkdir(parents=True)
    shutil.copy2(task_root / "instruction.md", destination / "instruction.md")
    shutil.copy2(task_root / "environment/starter/kernel.py", destination / "kernel.py")
    for name in ("dev_check.py", "candidate_policy.py", "PALLAS_API.md"):
        shutil.copy2(task_root / "environment/public" / name, destination / name)
    environment = _git_environment()
    subprocess.run(["git", "init", "-q", str(destination)], check=True, env=environment)
    subprocess.run(
        ["git", "-C", str(destination), "add", "."], check=True, env=environment
    )
    subprocess.run(
        ["git", "-C", str(destination), "commit", "-q", "-m", "task base"],
        check=True,
        env=environment,
    )
    patch = patch_path.read_bytes()
    if patch:
        applied = subprocess.run(
            ["git", "-C", str(destination), "apply", "--whitespace=error-all", "-"],
            input=patch,
            capture_output=True,
            check=False,
        )
        if applied.returncode != 0:
            error = applied.stderr.decode(errors="replace").strip()
            raise Phase2BenchmarkError(f"SUBMISSION_PATCH_INVALID:{error}")
    kernel = destination / "kernel.py"
    if not kernel.is_file() or kernel.is_symlink():
        raise Phase2BenchmarkError("SUBMISSION_KERNEL_INVALID")
    for hidden in ("tests", "solution"):
        if (destination / hidden).exists():
            raise Phase2BenchmarkError(f"SUBMISSION_HIDDEN_PATH_CREATED:{hidden}")
    return {
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "kernel_sha256": file_sha256(kernel),
        "workspace_sha256": tree_sha256(destination, excluded={".git"}),
        "kernel_path": str(kernel),
    }


def _validate_runtime(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    exact = (
        "jax",
        "jaxlib",
        "chex",
        "libtpu",
        "numpy",
        "ml_dtypes",
        "scipy",
        "tomli",
        "backend",
    )
    if any(observed.get(key) != expected.get(key) for key in exact):
        raise Phase2BenchmarkError("REFERENCE_RUNTIME_MISMATCH")
    python = observed.get("python")
    device_kinds = observed.get("device_kinds")
    if (
        not isinstance(python, str)
        or not python.startswith(f"{expected['python']}.")
        or not isinstance(device_kinds, list)
        or expected["device_kind"] not in device_kinds
    ):
        raise Phase2BenchmarkError("REFERENCE_RUNTIME_MISMATCH")


def _run_verifier(
    *, task_root: Path, kernel_path: Path, task_out: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_dir = task_out / "evidence"
    task_out.mkdir(parents=True)
    environment = os.environ.copy()
    environment["LIBTPU_INIT_ARGS"] = "--xla_tpu_scoped_vmem_limit_kib=65536"
    package_root = str(Path(__file__).parents[2].resolve())
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{package_root}{os.pathsep}{python_path}" if python_path else package_root
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "opjax.pallas.environment_runner",
            "--task",
            str(task_root / "tests/task.json"),
            "--kernel",
            str(kernel_path),
            "--evidence-dir",
            str(evidence_dir),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
        check=False,
    )
    (task_out / "run.log").write_text(
        process.stdout + process.stderr, encoding="utf-8"
    )
    result_path = evidence_dir / "result.json"
    if not result_path.is_file():
        raise Phase2BenchmarkError(
            f"VERIFIER_RESULT_MISSING:{task_root.name}:returncode={process.returncode}"
        )
    result = _load_json(result_path)
    (task_out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result, render_artifacts(task_out)


def _probe_tpu_health() -> dict[str, Any]:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,jax,jax.numpy as jnp; x=jax.jit(lambda y:y+1)(jnp.ones((128,128))); x.block_until_ready(); print(json.dumps({'backend':jax.default_backend(),'device_kinds':sorted({d.device_kind for d in jax.devices()})}))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        timeout=120,
        check=False,
    )
    if process.returncode != 0:
        raise Phase2BenchmarkError(
            f"TPU_WORKER_HEALTH_FAILED:returncode={process.returncode}:{process.stderr[-500:]}"
        )
    payload = json.loads(process.stdout.splitlines()[-1])
    if payload.get("backend") != "tpu":
        raise Phase2BenchmarkError("TPU_WORKER_HEALTH_BACKEND_INVALID")
    time.sleep(15)
    return payload


def grade_submission(
    *, release_root: Path, task_id: str, patch_path: Path, out_dir: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise Phase2BenchmarkError(f"OUTPUT_EXISTS:{out_dir}")
    validate_release(release_root)
    release = _load_json(release_root / "manifest.json")
    matches = [task for task in release["tasks"] if task["task_id"] == task_id]
    if len(matches) != 1:
        raise Phase2BenchmarkError(f"SUBMISSION_TASK_UNKNOWN:{task_id}")
    task = matches[0]
    task_root = release_root / task["path"]
    with tempfile.TemporaryDirectory(prefix=f"opjax-{task_id}-") as temporary:
        materialized = materialize_submission(
            task_root=task_root,
            patch_path=patch_path,
            destination=Path(temporary) / "workspace",
        )
        kernel_path = Path(materialized["kernel_path"])
        source = kernel_path.read_text(encoding="utf-8")
        reference_source = (task_root / "solution/kernel.py").read_text(
            encoding="utf-8"
        )
        verifier_task = _load_json(task_root / "tests/task.json")
        static = verify_static(
            f"```python\n{source}\n```",
            allowed_pallas_entrypoints=tuple(
                verifier_task.get("allowed_pallas_entrypoints", ())
            ),
        )
        similarity = baseline_similarity(source, reference_source)
        reference_mimic = is_verbatim_file_copy(
            source, reference_source
        ) or (similarity is not None and similarity >= 0.98)
        if reference_mimic:
            out_dir.mkdir(parents=True)
            result = {
                "passed": False,
                "stage": "pallas_api",
                "error": "REFERENCE_MIMIC",
                "infrastructure_error": False,
                "worker_recovery_required": False,
                "stages": {
                    "artifact_contract": True,
                    "pallas_api": False,
                },
                "hardware": {"target": "tpu", "execution": "not_started"},
                "kernel_sha256": materialized["kernel_sha256"],
            }
            (out_dir / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (out_dir / "run.log").write_text("REFERENCE_MIMIC\n", encoding="utf-8")
            reward = render_artifacts(out_dir)
        elif (
            static.passed
            and not has_host_compute_in_workload(
                source,
                allowed_entrypoints=tuple(
                    verifier_task.get("allowed_pallas_entrypoints", ())
                ),
            )
            and candidate_module_policy_error(
                source,
                allowed_entrypoints=tuple(
                    verifier_task.get("allowed_pallas_entrypoints", ())
                ),
            )
            is None
        ):
            out_dir.mkdir(parents=True)
            shutil.copy2(patch_path, out_dir / "model.patch")
            request = build_submission_request(
                release_manifest=release,
                task=task,
                patch_path=out_dir / "model.patch",
            )
            write_request(out_dir / "request.json", request)
            pending = {
                **request,
                **materialized,
                "kernel_path": "kernel.py",
                "status": "tpu_worker_required",
                "reward": None,
            }
            (out_dir / "submission.json").write_text(
                json.dumps(pending, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return pending
        else:
            result, reward = _run_verifier(
                task_root=task_root,
                kernel_path=kernel_path,
                task_out=out_dir,
            )
    shutil.copy2(patch_path, out_dir / "model.patch")
    submission = {
        "task_id": task_id,
        "task_sha256": task["task_sha256"],
        "release_sha256": release["release_sha256"],
        **materialized,
        "kernel_path": "kernel.py",
        "result_sha256": file_sha256(out_dir / "result.json"),
        "reward_sha256": file_sha256(out_dir / "reward.json"),
        "reward": reward["reward"],
        "stage": result.get("stage"),
    }
    (out_dir / "submission.json").write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return submission


def grade_worker_submission(
    *,
    release_root: Path,
    request_path: Path,
    patch_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    identity = os.environ.get("OPJAX_DISPOSABLE_WORKER_IDENTITY")
    if not identity:
        raise Phase2WorkerError("DISPOSABLE_WORKER_IDENTITY_MISSING")
    request = _load_json(request_path)
    request_payload = dict(request)
    expected_request_sha = request_payload.pop("request_sha256", None)
    if canonical_sha256(request_payload) != expected_request_sha:
        raise Phase2WorkerError("TPU_REQUEST_HASH_INVALID")
    release = _load_json(release_root / "manifest.json")
    if release.get("kind") != "opjax_phase2_sanitized_worker_bundle":
        raise Phase2WorkerError("TPU_WORKER_BUNDLE_KIND_INVALID")
    worker_manifest = _load_json(release_root / "worker-manifest.json")
    expected_bundle_sha = worker_manifest.pop("bundle_sha256", None)
    if canonical_sha256(worker_manifest) != expected_bundle_sha:
        raise Phase2WorkerError("TPU_WORKER_BUNDLE_HASH_INVALID")
    observed_files = {
        str(path.relative_to(release_root)): file_sha256(path)
        for path in sorted(release_root.rglob("*"))
        if path.is_file()
        and path.name != "worker-manifest.json"
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    }
    if observed_files != worker_manifest.get("files"):
        raise Phase2WorkerError("TPU_WORKER_BUNDLE_CONTENT_INVALID")
    if any(release_root.rglob("solution")):
        raise Phase2WorkerError("HIDDEN_SOLUTION_IN_WORKER_BUNDLE")
    tasks = {
        task["task_id"]: task for task in release["tasks"]
    }
    task = tasks.get(request.get("task_id"))
    if (
        task is None
        or request.get("release_sha256") != release["release_sha256"]
        or request.get("task_sha256") != task["task_sha256"]
        or request.get("patch_sha256") != file_sha256(patch_path)
    ):
        raise Phase2WorkerError("TPU_REQUEST_BINDING_INVALID")
    task_root = release_root / task["path"]
    with tempfile.TemporaryDirectory(prefix=f"opjax-worker-{task['task_id']}-") as temporary:
        materialized = materialize_submission(
            task_root=task_root,
            patch_path=patch_path,
            destination=Path(temporary) / "workspace",
        )
        result, reward = _run_verifier(
            task_root=task_root,
            kernel_path=Path(materialized["kernel_path"]),
            task_out=out_dir,
        )
    shutil.copy2(patch_path, out_dir / "model.patch")
    response = {
        **request,
        **materialized,
        "kernel_path": "kernel.py",
        "result_sha256": file_sha256(out_dir / "result.json"),
        "reward_sha256": file_sha256(out_dir / "reward.json"),
        "reward": reward["reward"],
        "stage": result.get("stage"),
        "worker": {
            "identity": identity,
            "disposable": True,
            "destroyed_at": None,
        },
    }
    (out_dir / "submission.json").write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return response


def validate_reference_evidence(
    *, release_root: Path, evidence_root: Path
) -> dict[str, Any]:
    validate_release(release_root)
    release = _load_json(release_root / "manifest.json")
    evidence = _load_json(evidence_root / "manifest.json")
    payload = dict(evidence)
    expected_sha = payload.pop("evidence_sha256", None)
    if (
        evidence.get("kind") != "opjax_pallas_phase2_reference_evidence"
        or canonical_sha256(payload) != expected_sha
        or evidence.get("task_set_sha256") != _task_set_sha256(release)
        or evidence.get("runtime") != release.get("runtime")
    ):
        raise Phase2BenchmarkError("REFERENCE_EVIDENCE_HASH_INVALID")
    worker = evidence.get("worker_fingerprint")
    if (
        not isinstance(worker, dict)
        or worker.get("acceleratorType") != "v5litepod-1"
        or worker.get("runtimeVersion") != "tpu-ubuntu2204-base"
        or worker.get("zone") != "us-west4-a"
        or not worker.get("name")
        or not (evidence_root / "worker-fingerprint.json").is_file()
        or evidence.get("worker_fingerprint_sha256")
        != file_sha256(evidence_root / "worker-fingerprint.json")
    ):
        raise Phase2BenchmarkError("REFERENCE_WORKER_FINGERPRINT_INVALID")
    by_id = {record.get("task_id"): record for record in evidence.get("tasks", [])}
    expected_tasks = {task["task_id"]: task for task in release["tasks"]}
    if set(by_id) != set(expected_tasks):
        raise Phase2BenchmarkError("REFERENCE_EVIDENCE_TASK_SET_INVALID")
    for task_id, task in expected_tasks.items():
        record = by_id[task_id]
        root = evidence_root / "tasks" / task_id
        result_path = root / "result.json"
        reward_path = root / "reward.json"
        if (
            record.get("task_sha256") != task["task_sha256"]
            or not result_path.is_file()
            or not reward_path.is_file()
            or record.get("result_sha256") != file_sha256(result_path)
            or record.get("reward_sha256") != file_sha256(reward_path)
            or record.get("artifact_tree_sha256") != tree_sha256(root)
        ):
            raise Phase2BenchmarkError(
                f"REFERENCE_ARTIFACT_HASH_INVALID:{task_id}"
            )
        result = _load_json(result_path)
        reward = _load_json(reward_path)
        runtime = result.get("profile", {}).get("runtime")
        if not isinstance(runtime, dict):
            raise Phase2BenchmarkError(f"REFERENCE_RUNTIME_MISSING:{task_id}")
        _validate_runtime(runtime, release["runtime"])
        if record.get("runtime") != runtime:
            raise Phase2BenchmarkError(f"REFERENCE_RUNTIME_RECORD_MISMATCH:{task_id}")
        timing = result.get("profile", {}).get("timing")
        try:
            validate_timing_result(timing, seed=0)
        except (BenchmarkingError, TypeError) as exc:
            raise Phase2BenchmarkError(
                f"REFERENCE_TIMING_INVALID:{task_id}:{exc}"
            ) from exc
        if (
            result.get("passed") is not True
            or result.get("stage") != "verified"
            or reward.get("reward") != 1
            or record.get("reference_reward") != 1
            or record.get("speedup") != timing.get("speedup")
            or record.get("speedup_ci95") != timing.get("speedup_ci95")
            or record.get("unstable") is not timing.get("unstable")
        ):
            raise Phase2BenchmarkError(f"REFERENCE_RESULT_INVALID:{task_id}")
    subset = select_performance_subset(
        task_ids=list(expected_tasks), evidence=list(by_id.values())
    )
    if evidence.get("performance_subset") != subset:
        raise Phase2BenchmarkError("REFERENCE_PERFORMANCE_SUBSET_INVALID")
    return {
        "task_count": len(expected_tasks),
        "performance_subset": subset,
        "evidence_sha256": expected_sha,
    }


def grade_references(
    *, release_root: Path, out_dir: Path, worker_fingerprint_path: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise Phase2BenchmarkError(f"OUTPUT_EXISTS:{out_dir}")
    validate_release(release_root)
    release = _load_json(release_root / "manifest.json")
    worker_fingerprint = _load_json(worker_fingerprint_path)
    if (
        worker_fingerprint.get("acceleratorType") != "v5litepod-1"
        or worker_fingerprint.get("runtimeVersion") != "tpu-ubuntu2204-base"
        or worker_fingerprint.get("zone") != "us-west4-a"
    ):
        raise Phase2BenchmarkError("REFERENCE_WORKER_FINGERPRINT_INVALID")
    out_dir.mkdir(parents=True)
    shutil.copy2(worker_fingerprint_path, out_dir / "worker-fingerprint.json")
    worker_health = {"before": _probe_tpu_health(), "after": None}
    records = []
    for task in release["tasks"]:
        task_root = release_root / task["path"]
        task_out = out_dir / "tasks" / task["task_id"]
        result, reward = _run_verifier(
            task_root=task_root,
            kernel_path=task_root / "solution/kernel.py",
            task_out=task_out,
        )
        if result.get("worker_recovery_required") is True:
            raise Phase2BenchmarkError(
                f"TPU_WORKER_QUARANTINED:{task['task_id']}"
            )
        timing = result.get("profile", {}).get("timing", {})
        records.append(
            {
                "task_id": task["task_id"],
                "task_sha256": task["task_sha256"],
                "reference_reward": reward["reward"],
                "failure_stage": reward["failure_stage"],
                "speedup": timing.get("speedup"),
                "speedup_ci95": timing.get("speedup_ci95"),
                "unstable": timing.get("unstable"),
                "runtime": result.get("profile", {}).get("runtime"),
                "result_sha256": file_sha256(task_out / "result.json"),
                "reward_sha256": file_sha256(task_out / "reward.json"),
                "artifact_tree_sha256": tree_sha256(task_out),
            }
        )
        time.sleep(15)
    worker_health["after"] = _probe_tpu_health()
    manifest = {
        "schema_version": 1,
        "kind": "opjax_pallas_phase2_reference_evidence",
        "task_set_sha256": _task_set_sha256(release),
        "runtime": release["runtime"],
        "worker_health": worker_health,
        "worker_fingerprint": worker_fingerprint,
        "worker_fingerprint_sha256": file_sha256(
            out_dir / "worker-fingerprint.json"
        ),
        "tasks": records,
    }
    manifest["performance_subset"] = select_performance_subset(
        task_ids=[task["task_id"] for task in release["tasks"]], evidence=records
    )
    manifest["evidence_sha256"] = canonical_sha256(manifest)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def freeze_release(*, release_root: Path, evidence_root: Path) -> dict[str, Any]:
    validation = validate_reference_evidence(
        release_root=release_root, evidence_root=evidence_root
    )
    release = _load_json(release_root / "manifest.json")
    evidence = _load_json(evidence_root / "manifest.json")
    expected_evidence_sha = validation["evidence_sha256"]
    performance_subset = validation["performance_subset"]
    release["status"] = "frozen"
    release["performance_subset"] = performance_subset
    release["reference_evidence"] = {
        "relative_manifest_path": os.path.relpath(
            evidence_root / "manifest.json", release_root
        ),
        "manifest_sha256": file_sha256(evidence_root / "manifest.json"),
        "evidence_sha256": expected_evidence_sha,
        "task_artifact_tree_sha256": canonical_sha256(
            {
                task["task_id"]: task["artifact_tree_sha256"]
                for task in evidence["tasks"]
            }
        ),
    }
    release.pop("release_sha256", None)
    release["release_sha256"] = canonical_sha256(release)
    (release_root / "manifest.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_release(release_root)
    return release


def build_acceptance_patches(*, release_root: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise Phase2BenchmarkError(f"OUTPUT_EXISTS:{out_dir}")
    release = _load_json(release_root / "manifest.json")
    out_dir.mkdir(parents=True)
    records = []
    for task in release["tasks"]:
        task_root = release_root / task["path"]
        task_config = _load_json(task_root / "tests/task.json")
        source = zero_output_candidate_source(task_config)
        patch_path = out_dir / "zero" / f"{task['task_id']}.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        starter = (task_root / "environment/starter/kernel.py").read_text()
        patch_path.write_text(
            "".join(
                difflib.unified_diff(
                    starter.splitlines(keepends=True),
                    source.splitlines(keepends=True),
                    fromfile="a/kernel.py",
                    tofile="b/kernel.py",
                )
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "task_id": task["task_id"],
                "task_sha256": task["task_sha256"],
                "path": str(patch_path.relative_to(out_dir)),
                "sha256": file_sha256(patch_path),
            }
        )
    megablox = next(
        task for task in release["tasks"] if task["family"] == "mixture_of_experts"
    )
    task_root = release_root / megablox["path"]
    starter = (task_root / "environment/starter/kernel.py").read_text()
    probes = {}
    for name, source in (
        ("timing_zero", range_conditioned_megablox_candidate_source()),
        ("strong_zero", timing_conditioned_megablox_candidate_source()),
    ):
        patch_path = out_dir / "probes" / f"{name}.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(
            "".join(
                difflib.unified_diff(
                    starter.splitlines(keepends=True),
                    source.splitlines(keepends=True),
                    fromfile="a/kernel.py",
                    tofile="b/kernel.py",
                )
            ),
            encoding="utf-8",
        )
        probes[name] = {
            "task_id": megablox["task_id"],
            "task_sha256": megablox["task_sha256"],
            "path": str(patch_path.relative_to(out_dir)),
            "sha256": file_sha256(patch_path),
        }
    manifest = {
        "schema_version": 1,
        "kind": "opjax_pallas_phase2_acceptance_patches",
        "release_sha256": release["release_sha256"],
        "zero_controls": records,
        "probes": probes,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _acceptance_run_record(
    *, root: Path, relative_to: Path, kind: str
) -> dict[str, Any]:
    submission = _load_json(root / "submission.json")
    result = _load_json(root / "result.json")
    reward = _load_json(root / "reward.json")
    worker = submission.get("worker")
    if not isinstance(worker, dict):
        raise Phase2BenchmarkError(f"ACCEPTANCE_WORKER_MISSING:{kind}")
    return {
        "kind": kind,
        "task_id": submission["task_id"],
        "task_sha256": submission["task_sha256"],
        "reward": reward["reward"],
        "stage": result["stage"],
        "stages": result["stages"],
        "infrastructure_error": result["infrastructure_error"],
        "patch_sha256": submission["patch_sha256"],
        "result_sha256": file_sha256(root / "result.json"),
        "reward_sha256": file_sha256(root / "reward.json"),
        "worker_identity": worker["identity"],
        "worker_destroyed_at": worker["destroyed_at"],
        "relative_path": os.path.relpath(root, relative_to),
        "artifact_tree_sha256": tree_sha256(root),
    }


def render_acceptance_evidence(
    *,
    release_root: Path,
    zero_root: Path,
    timing_zero_root: Path,
    strong_zero_root: Path,
    pier_root: Path,
    disposable_root: Path,
    isolation_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise Phase2BenchmarkError(f"OUTPUT_EXISTS:{out_dir}")
    release = _load_json(release_root / "manifest.json")
    if release.get("status") != "frozen":
        raise Phase2BenchmarkError("ACCEPTANCE_PARENT_NOT_FROZEN")
    out_dir.mkdir(parents=True)
    zero_records = [
        _acceptance_run_record(
            root=zero_root / task["task_id"], relative_to=out_dir, kind="zero"
        )
        for task in release["tasks"]
    ]
    supplemental = {}
    for name, root in (
        ("pier", pier_root),
        ("disposable", disposable_root),
        ("isolation", isolation_root),
    ):
        supplemental[name] = {
            "relative_path": os.path.relpath(root, out_dir),
            "artifact_tree_sha256": tree_sha256(root),
        }
    manifest = {
        "schema_version": 1,
        "kind": "opjax_pallas_phase2_acceptance_evidence",
        "parent_release_sha256": release["release_sha256"],
        "task_set_sha256": _task_set_sha256(release),
        "zero_controls": zero_records,
        "probes": {
            "timing_zero": _acceptance_run_record(
                root=timing_zero_root,
                relative_to=out_dir,
                kind="timing_zero",
            ),
            "strong_zero": _acceptance_run_record(
                root=strong_zero_root,
                relative_to=out_dir,
                kind="strong_zero",
            ),
        },
        "supplemental": supplemental,
    }
    manifest["evidence_sha256"] = canonical_sha256(manifest)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def bind_acceptance_evidence(
    *, release_root: Path, evidence_root: Path
) -> dict[str, Any]:
    release = _load_json(release_root / "manifest.json")
    evidence = _load_json(evidence_root / "manifest.json")
    if release.get("status") != "frozen" or evidence.get(
        "parent_release_sha256"
    ) != release.get("release_sha256"):
        raise Phase2BenchmarkError("ACCEPTANCE_PARENT_RELEASE_MISMATCH")
    release["status"] = "accepted"
    release["acceptance_evidence"] = {
        "parent_release_sha256": evidence["parent_release_sha256"],
        "relative_manifest_path": os.path.relpath(
            evidence_root / "manifest.json", release_root
        ),
        "manifest_sha256": file_sha256(evidence_root / "manifest.json"),
        "evidence_sha256": evidence["evidence_sha256"],
    }
    release.pop("release_sha256", None)
    release["release_sha256"] = canonical_sha256(release)
    (release_root / "manifest.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_release(release_root)
    return release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase2-runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    grade = subparsers.add_parser("grade-references")
    grade.add_argument("--release", type=Path, required=True)
    grade.add_argument("--out", type=Path, required=True)
    grade.add_argument("--worker-fingerprint", type=Path, required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--release", type=Path, required=True)
    freeze.add_argument("--evidence", type=Path, required=True)
    submission = subparsers.add_parser("grade-submission")
    submission.add_argument("--release", type=Path, required=True)
    submission.add_argument("--task-id", required=True)
    submission.add_argument("--patch", type=Path, required=True)
    submission.add_argument("--out", type=Path, required=True)
    worker = subparsers.add_parser("grade-worker-submission")
    worker.add_argument("--release", type=Path, required=True)
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--patch", type=Path, required=True)
    worker.add_argument("--out", type=Path, required=True)
    patches = subparsers.add_parser("build-acceptance-patches")
    patches.add_argument("--release", type=Path, required=True)
    patches.add_argument("--out", type=Path, required=True)
    acceptance = subparsers.add_parser("render-acceptance")
    acceptance.add_argument("--release", type=Path, required=True)
    acceptance.add_argument("--zero-root", type=Path, required=True)
    acceptance.add_argument("--timing-zero", type=Path, required=True)
    acceptance.add_argument("--strong-zero", type=Path, required=True)
    acceptance.add_argument("--pier", type=Path, required=True)
    acceptance.add_argument("--disposable", type=Path, required=True)
    acceptance.add_argument("--isolation", type=Path, required=True)
    acceptance.add_argument("--out", type=Path, required=True)
    bind = subparsers.add_parser("bind-acceptance")
    bind.add_argument("--release", type=Path, required=True)
    bind.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "grade-references":
        result = grade_references(
            release_root=args.release,
            out_dir=args.out,
            worker_fingerprint_path=args.worker_fingerprint,
        )
    elif args.command == "grade-submission":
        result = grade_submission(
            release_root=args.release,
            task_id=args.task_id,
            patch_path=args.patch,
            out_dir=args.out,
        )
    elif args.command == "grade-worker-submission":
        result = grade_worker_submission(
            release_root=args.release,
            request_path=args.request,
            patch_path=args.patch,
            out_dir=args.out,
        )
    elif args.command == "build-acceptance-patches":
        result = build_acceptance_patches(
            release_root=args.release, out_dir=args.out
        )
    elif args.command == "render-acceptance":
        result = render_acceptance_evidence(
            release_root=args.release,
            zero_root=args.zero_root,
            timing_zero_root=args.timing_zero,
            strong_zero_root=args.strong_zero,
            pier_root=args.pier,
            disposable_root=args.disposable,
            isolation_root=args.isolation,
            out_dir=args.out,
        )
    elif args.command == "bind-acceptance":
        result = bind_acceptance_evidence(
            release_root=args.release, evidence_root=args.evidence
        )
    else:
        result = freeze_release(release_root=args.release, evidence_root=args.evidence)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
